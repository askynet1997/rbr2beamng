from __future__ import annotations

import math
import struct

import numpy as np

from .binary import (
    BinaryReader,
    FormatError,
    Source,
    checked_count,
    load_source,
    require_finite,
)
from .models import (
    Bounds,
    BrakeWall,
    ColDescriptor,
    ColFile,
    CollisionSubtree,
    CollisionTreeNode,
    CollisionTriangleBatch,
    WaterSurface,
)


COL_MAGIC = b"OC7R"
DESCRIPTOR_SIZE = 20
NODE_HEADER_SIZE = 32
_NODE_HEADER = struct.Struct("<6f2I")
MAX_SUBTREES = 1_000_000
MAX_TREE_NODES = 10_000_000
MAX_TRIANGLES = 10_000_000
MAX_VERTICES = 10_000_000
MAX_TREE_DEPTH = 512
MAX_BRAKE_WALL_POINTS = 16_384
BRAKE_WALL_NODE_SIZE = 24

COLLISION_TRIANGLE_DTYPE = np.dtype(
    [
        ("a_index", "<u2"),
        ("b_index", "<u2"),
        ("c_index", "<u2"),
        ("blending", "<u2"),
        ("shading", "<u2"),
        ("material_1_id", "u1"),
        ("material_2_id", "u1"),
        ("a_material_1_uv", "u1"),
        ("b_material_1_uv", "u1"),
        ("c_material_1_uv", "u1"),
        ("a_material_2_uv", "u1"),
        ("b_material_2_uv", "u1"),
        ("c_material_2_uv", "u1"),
    ]
)


def _descriptor(data: memoryview, offset: int, context: str) -> ColDescriptor:
    if offset < 0 or offset > len(data) - DESCRIPTOR_SIZE:
        raise FormatError("COL descriptor is out of bounds", offset=offset, context=context)
    reader = BinaryReader(
        data,
        start=offset,
        end=offset + DESCRIPTOR_SIZE,
        context=context,
    )
    data_type, tree_type, padding = reader.unpack("<BBH")
    if padding != 0:
        raise FormatError(
            f"COL descriptor padding is 0x{padding:x}, expected zero",
            offset=offset + 2,
            context=context,
        )
    traversal_bytes = reader.unpack("<3B")
    level = reader.u8()
    if level > 24:
        raise FormatError(
            f"COL branch traversal level {level} exceeds 24",
            offset=offset + 7,
            context=context,
        )
    traversal = tuple(
        bool(traversal_bytes[index // 8] & (1 << (index % 8)))
        for index in range(level)
    )
    vertex_count, vertices_offset, tree_offset = reader.unpack("<III")
    checked_count(
        vertex_count,
        limit=MAX_VERTICES,
        offset=offset + 8,
        context=f"{context} vertex count",
    )
    return ColDescriptor(
        offset=offset,
        data_type=data_type,
        tree_type=tree_type,
        traversal=traversal,
        vertex_count=vertex_count,
        vertices_offset=vertices_offset,
        tree_offset=tree_offset,
    )


class _TreeParser:
    def __init__(
        self,
        data: memoryview,
        root_offset: int,
        *,
        subtree: bool,
        vertex_count: int,
    ):
        self.data = data
        self.root_offset = root_offset
        self.subtree = subtree
        self.vertex_count = vertex_count
        self.node_count = 0
        self.active_headers: set[int] = set()

    def parse(self) -> CollisionTreeNode:
        if self.root_offset < 0 or self.root_offset > len(self.data) - NODE_HEADER_SIZE:
            raise FormatError(
                "COL collision-tree root is out of bounds",
                offset=self.root_offset,
                context="COL collision tree",
            )
        return self._node(self.root_offset, depth=0, subtree_root=self.subtree)

    def _header(self, offset: int):
        if offset < 0 or offset > len(self.data) - NODE_HEADER_SIZE:
            raise FormatError(
                "COL tree node header is out of bounds",
                offset=offset,
                context="COL collision tree",
            )
        values = _NODE_HEADER.unpack_from(self.data, offset)
        require_finite(values[:6], offset=offset, context="COL tree bounds")
        if values[3] < 0 or values[4] < 0 or values[5] < 0:
            raise FormatError(
                "Negative COL tree half extent",
                offset=offset,
                context="COL tree bounds",
            )
        bounds = Bounds(values[0:3], values[3:6])
        packed, data_offset = values[6:]
        triangle_count = packed & 0x1FFFFF
        link = bool(packed & (1 << 21))
        checked_count(
            triangle_count,
            limit=MAX_TRIANGLES,
            offset=offset + 24,
            context="COL tree triangle count",
        )
        return bounds, triangle_count, link, data_offset

    def _node(
        self,
        header_offset: int,
        *,
        depth: int,
        subtree_root: bool = False,
    ) -> CollisionTreeNode:
        if depth > MAX_TREE_DEPTH:
            raise FormatError(
                f"COL tree depth exceeds {MAX_TREE_DEPTH}",
                offset=header_offset,
                context="COL collision tree",
            )
        self.node_count += 1
        if self.node_count > MAX_TREE_NODES:
            raise FormatError(
                f"COL tree exceeds {MAX_TREE_NODES} nodes",
                offset=header_offset,
                context="COL collision tree",
            )
        if header_offset in self.active_headers:
            raise FormatError(
                "Cycle in COL tree offsets",
                offset=header_offset,
                context="COL collision tree",
            )
        self.active_headers.add(header_offset)
        try:
            bounds, triangle_count, link, data_offset = self._header(header_offset)
            effective_link = link and not subtree_root
            if effective_link:
                return CollisionTreeNode(
                    bounds=bounds,
                    triangle_count=triangle_count,
                    link=True,
                    data_offset=data_offset,
                )

            value_offset = self.root_offset + data_offset
            if triangle_count:
                byte_count = triangle_count * COLLISION_TRIANGLE_DTYPE.itemsize
                if value_offset < 0 or value_offset > len(self.data) - byte_count:
                    raise FormatError(
                        "COL triangle leaf is out of bounds",
                        offset=value_offset,
                        context="COL collision tree",
                    )
                records = np.frombuffer(
                    self.data,
                    dtype=COLLISION_TRIANGLE_DTYPE,
                    count=triangle_count,
                    offset=value_offset,
                )
                if self.vertex_count:
                    # The vertex indices are the first three u2 words of a record.
                    largest = int(
                        records.view("<u2").reshape((triangle_count, -1))[:, :3].max()
                    )
                    if largest >= self.vertex_count:
                        raise FormatError(
                            f"COL triangle index {largest} exceeds "
                            f"{self.vertex_count} vertices",
                            offset=value_offset,
                            context="COL collision tree",
                        )
                return CollisionTreeNode(
                    bounds=bounds,
                    triangle_count=triangle_count,
                    link=link,
                    data_offset=data_offset,
                    triangles=CollisionTriangleBatch(records),
                )

            if value_offset < 0 or value_offset > len(self.data) - 2 * NODE_HEADER_SIZE:
                raise FormatError(
                    "COL internal node children are out of bounds",
                    offset=value_offset,
                    context="COL collision tree",
                )
            left = self._node(value_offset, depth=depth + 1)
            right = self._node(value_offset + NODE_HEADER_SIZE, depth=depth + 1)
            return CollisionTreeNode(
                bounds=bounds,
                triangle_count=0,
                link=link,
                data_offset=data_offset,
                left=left,
                right=right,
            )
        finally:
            self.active_headers.remove(header_offset)


def _water_surfaces(
    data: memoryview,
    offset: int,
    count: int,
    context: str,
) -> tuple[WaterSurface, ...]:
    size = count * 64
    if offset < 0 or offset > len(data) - size:
        raise FormatError(f"{context} array is out of bounds", offset=offset, context="COL")
    reader = BinaryReader(data, start=offset, end=offset + size, context=context)
    result: list[WaterSurface] = []
    for index in range(count):
        item_offset = reader.offset
        vertices = []
        padding = []
        for _ in range(4):
            vertices.append(reader.vec3())
            padding.append(reader.f32())
        flat = tuple(value for vertex in vertices for value in vertex) + tuple(padding)
        require_finite(flat, offset=item_offset, context=f"{context} {index}")
        result.append(WaterSurface(tuple(vertices), tuple(padding)))  # type: ignore[arg-type]
    return tuple(result)


def parse_brake_wall(data: bytes | bytearray | memoryview) -> BrakeWall:
    raw = data if isinstance(data, memoryview) else memoryview(data)
    raw = raw.cast("B")
    reader = BinaryReader(raw, context="COL brake wall")
    if len(raw) < 16:
        raise FormatError("Brake wall is shorter than its header", context="COL brake wall")
    declared_size, point_count, points_offset, tree_offset = reader.unpack("<4I")
    checked_count(
        point_count,
        limit=MAX_BRAKE_WALL_POINTS,
        offset=4,
        context="brake-wall point count",
    )
    if point_count < 2 or point_count % 2:
        raise FormatError(
            f"Brake wall has invalid point count {point_count}; expected inner/outer pairs",
            offset=4,
            context="COL brake wall",
        )
    if declared_size < 16 or declared_size > len(raw):
        raise FormatError(
            f"Brake wall declares invalid size {declared_size}",
            offset=0,
            context="COL brake wall",
        )
    points_size = point_count * 8
    points_end = points_offset + points_size
    if (
        points_offset < 16
        or points_offset % 4
        or points_end > declared_size
        or tree_offset < points_end
        or tree_offset > declared_size - BRAKE_WALL_NODE_SIZE
    ):
        raise FormatError(
            "Brake-wall point/tree offsets are invalid",
            offset=8,
            context="COL brake wall",
        )
    points = np.frombuffer(
        raw,
        dtype="<f4",
        count=point_count * 2,
        offset=points_offset,
    ).reshape((point_count, 2))
    if not np.isfinite(points).all():
        raise FormatError(
            "Brake wall contains non-finite points",
            offset=points_offset,
            context="COL brake wall",
        )

    active: set[int] = set()
    seen_nodes: set[int] = set()
    segments: set[int] = set()

    def parse_node(relative_offset: int, depth: int) -> None:
        if depth > MAX_TREE_DEPTH:
            raise FormatError(
                f"Brake-wall tree depth exceeds {MAX_TREE_DEPTH}",
                offset=tree_offset + relative_offset,
                context="COL brake wall",
            )
        absolute_offset = tree_offset + relative_offset
        if (
            relative_offset < 0
            or absolute_offset > declared_size - BRAKE_WALL_NODE_SIZE
        ):
            raise FormatError(
                "Brake-wall tree node is out of bounds",
                offset=absolute_offset,
                context="COL brake wall",
            )
        if relative_offset in active:
            raise FormatError(
                "Cycle in brake-wall tree",
                offset=absolute_offset,
                context="COL brake wall",
            )
        if relative_offset in seen_nodes:
            raise FormatError(
                "Duplicate brake-wall tree node",
                offset=absolute_offset,
                context="COL brake wall",
            )
        active.add(relative_offset)
        seen_nodes.add(relative_offset)
        node = BinaryReader(
            raw,
            start=absolute_offset,
            end=absolute_offset + BRAKE_WALL_NODE_SIZE,
            context="COL brake-wall node",
        )
        bounds = node.unpack("<4f")
        count, data_offset = node.unpack("<2I")
        if not all(math.isfinite(value) for value in bounds):
            raise FormatError(
                "Brake-wall tree has non-finite bounds",
                offset=absolute_offset,
                context="COL brake wall",
            )
        if bounds[2] < 0 or bounds[3] < 0:
            raise FormatError(
                "Brake-wall tree has negative half extents",
                offset=absolute_offset,
                context="COL brake wall",
            )
        checked_count(
            count,
            limit=point_count // 2,
            offset=absolute_offset + 16,
            context="brake-wall leaf count",
        )
        if count:
            payload_offset = tree_offset + data_offset
            payload_size = count * 2
            if payload_offset > declared_size - payload_size:
                raise FormatError(
                    "Brake-wall leaf indices are out of bounds",
                    offset=payload_offset,
                    context="COL brake wall",
                )
            references = np.frombuffer(
                raw,
                dtype="<u2",
                count=count,
                offset=payload_offset,
            )
            for reference in references:
                point_index = int(reference) & 0x3FFF
                if point_index % 2 or point_index + 3 >= point_count:
                    raise FormatError(
                        f"Invalid brake-wall segment point {point_index}",
                        offset=payload_offset,
                        context="COL brake wall",
                    )
                segment_index = point_index // 2
                if segment_index in segments:
                    raise FormatError(
                        f"Duplicate brake-wall segment {segment_index}",
                        offset=payload_offset,
                        context="COL brake wall",
                    )
                segments.add(segment_index)
        else:
            parse_node(data_offset, depth + 1)
            parse_node(data_offset + BRAKE_WALL_NODE_SIZE, depth + 1)
        active.remove(relative_offset)

    parse_node(0, 0)
    return BrakeWall(
        inner_points=points[0::2],
        outer_points=points[1::2],
        segments=tuple(sorted(segments)),
    )


def parse_col(source: Source) -> ColFile:
    loaded = load_source(source)
    reader = BinaryReader(loaded.view, context=f"COL {loaded.name}")
    if reader.read(4) != COL_MAGIC:
        raise FormatError("Invalid COL magic; expected OC7R", offset=0, context="COL")
    root_offset, subtree_count, subtree_table_offset = reader.unpack("<III")
    checked_count(
        subtree_count,
        limit=MAX_SUBTREES,
        offset=8,
        context="COL subtree count",
    )
    root = _descriptor(loaded.view, root_offset, "COL root descriptor")
    if root.data_type != 4 or root.tree_type != 3:
        raise FormatError(
            f"Unexpected COL root descriptor types {root.data_type}/{root.tree_type}",
            offset=root_offset,
            context="COL root descriptor",
        )

    root_header_offset = root_offset + DESCRIPTOR_SIZE
    if root_header_offset > len(loaded.view) - 20:
        raise FormatError(
            "COL root metadata is out of bounds",
            offset=root_header_offset,
            context="COL root",
        )
    metadata = BinaryReader(
        loaded.view,
        start=root_header_offset,
        end=root_header_offset + 20,
        context="COL root metadata",
    )
    brake_relative = metadata.u32()
    wet_count, wet_relative, water_count, water_relative = metadata.unpack("<4I")
    checked_count(wet_count, limit=512, offset=root_header_offset + 4, context="wet count")
    checked_count(
        water_count,
        limit=512,
        offset=root_header_offset + 12,
        context="water count",
    )
    if wet_count + water_count > 512:
        raise FormatError(
            f"COL has {wet_count + water_count} wet/water surfaces; maximum is 512",
            offset=root_header_offset + 4,
            context="COL root metadata",
        )
    wet_offset = root_offset + wet_relative
    water_offset = root_offset + water_relative
    wet_surfaces = _water_surfaces(loaded.view, wet_offset, wet_count, "COL wet surfaces")
    water_surfaces = _water_surfaces(
        loaded.view,
        water_offset,
        water_count,
        "COL water surfaces",
    )

    root_tree_offset = root_offset + root.tree_offset
    root_tree = _TreeParser(
        loaded.view,
        root_tree_offset,
        subtree=False,
        vertex_count=0,
    ).parse()

    table_size = subtree_count * 4
    if (
        subtree_table_offset < 0
        or subtree_table_offset > len(loaded.view) - table_size
    ):
        raise FormatError(
            "COL subtree descriptor table is out of bounds",
            offset=subtree_table_offset,
            context="COL header",
        )
    table_reader = BinaryReader(
        loaded.view,
        start=subtree_table_offset,
        end=subtree_table_offset + table_size,
        context="COL subtree table",
    )
    descriptor_offsets = table_reader.unpack(f"<{subtree_count}I") if subtree_count else ()
    if len(set(descriptor_offsets)) != len(descriptor_offsets):
        raise FormatError(
            "COL subtree table contains duplicate descriptor offsets",
            offset=subtree_table_offset,
            context="COL subtree table",
        )

    subtrees: list[CollisionSubtree] = []
    for index, descriptor_offset in enumerate(descriptor_offsets):
        descriptor = _descriptor(
            loaded.view,
            descriptor_offset,
            f"COL subtree descriptor {index}",
        )
        if descriptor.data_type != 4 or descriptor.tree_type != 1:
            raise FormatError(
                f"Unexpected COL subtree types "
                f"{descriptor.data_type}/{descriptor.tree_type}",
                offset=descriptor_offset,
                context=f"COL subtree {index}",
            )
        vertices_offset = descriptor_offset + descriptor.vertices_offset
        vertices_size = descriptor.vertex_count * 12
        if (
            vertices_offset < 0
            or vertices_offset > len(loaded.view) - vertices_size
        ):
            raise FormatError(
                "COL subtree vertices are out of bounds",
                offset=vertices_offset,
                context=f"COL subtree {index}",
            )
        vertices = np.frombuffer(
            loaded.view,
            dtype="<f4",
            count=descriptor.vertex_count * 3,
            offset=vertices_offset,
        ).reshape((descriptor.vertex_count, 3))
        if len(vertices) and not np.isfinite(vertices).all():
            raise FormatError(
                "COL subtree has non-finite vertices",
                offset=vertices_offset,
                context=f"COL subtree {index}",
            )
        tree_offset = descriptor_offset + descriptor.tree_offset
        subtree_root = _TreeParser(
            loaded.view,
            tree_offset,
            subtree=True,
            vertex_count=descriptor.vertex_count,
        ).parse()
        subtrees.append(CollisionSubtree(descriptor, vertices, subtree_root))

    brake_wall_raw = None
    brake_wall = None
    if brake_relative:
        brake_offset = root_offset + brake_relative
        candidates = [
            value
            for value in (wet_offset, water_offset, root_tree_offset)
            if value > brake_offset
        ]
        brake_end = min(candidates, default=len(loaded.view))
        if brake_offset < root_header_offset + 20 or brake_offset > brake_end:
            raise FormatError(
                "COL brake-wall offset is invalid",
                offset=brake_offset,
                context="COL root metadata",
            )
        brake_wall_raw = loaded.view[brake_offset:brake_end]
        brake_wall = parse_brake_wall(brake_wall_raw)

    return ColFile(
        root_descriptor=root,
        root_tree=root_tree,
        subtrees=tuple(subtrees),
        wet_surfaces=wet_surfaces,
        water_surfaces=water_surfaces,
        brake_wall_raw=brake_wall_raw,
        raw=loaded.view,
        brake_wall=brake_wall,
    )
