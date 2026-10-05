from __future__ import annotations

import os
import warnings
from pathlib import Path
from typing import Any

import numpy as np

from ..filesystem import current_filesystem
from .binary import (
    BinaryReader,
    BinarySource,
    FormatError,
    Source,
    checked_count,
    load_source,
    optional_index,
    require_finite,
    scan_segments,
)
from .models import (
    Bounds,
    CarLocation,
    GeomBlock,
    GeomBlocks,
    LbsFile,
    MeshBuffer,
    ObjectBlock,
    ObjectBlockGroup,
    ObjectData3D,
    ObjectData3DGroup,
    ObjectData3DItem,
    RenderChunk,
)


GEOM_BLOCKS = 0x0
OBJECT_BLOCKS = 0x1
SUPER_BOWL = 0x3
INTERACTIVE_OBJECTS = 0x6
REFLECTION_OBJECTS = 0x7
WATER_OBJECTS = 0x8
VISIBLE_OBJECTS = 0xA
CAR_LOCATION = 0xB

HAS_SINGLE_TEXTURE = 0x10
HAS_DOUBLE_TEXTURE = 0x20
HAS_SPECULAR_TEXTURE = 0x42
HAS_SHADER_DATA = 0x80

MAX_BLOCKS = 1_000_000
MAX_BUFFERS = 10_000_000
MAX_OBJECTS = 2_000_000
_IBS_BYTE_SHIFT = 0x19
_IBS_DECODE_TABLE = bytes(
    (value - _IBS_BYTE_SHIFT) & 0xFF for value in range(256)
)
_IBS_ENCODED_HEADER = bytes(
    (value + _IBS_BYTE_SHIFT) & 0xFF for value in (8, 0, 0, 0)
)
_LBS_KNOWN_SECTIONS = frozenset(
    (
        GEOM_BLOCKS,
        OBJECT_BLOCKS,
        SUPER_BOWL,
        INTERACTIVE_OBJECTS,
        REFLECTION_OBJECTS,
        WATER_OBJECTS,
        VISIBLE_OBJECTS,
        CAR_LOCATION,
    )
)
_IBS_REQUIRED_SECTIONS = frozenset(
    (GEOM_BLOCKS, OBJECT_BLOCKS, VISIBLE_OBJECTS, CAR_LOCATION)
)


UV = np.dtype([("u", "<f4"), ("v", "<f4")])
VEC3_LH = np.dtype([("x", "<f4"), ("z", "<f4"), ("y", "<f4")])
COLOR = np.dtype([("b", "u1"), ("g", "u1"), ("r", "u1"), ("a", "u1")])
NORMAL_VERTEX_DTYPES = (
    np.dtype([("position", VEC3_LH), ("color", COLOR)]),
    np.dtype([("position", VEC3_LH), ("color", COLOR), ("diffuse_1_uv", UV)]),
    np.dtype(
        [
            ("position", VEC3_LH),
            ("normal", VEC3_LH),
            ("color", COLOR),
            ("diffuse_1_uv", UV),
            ("specular_uv", UV),
            ("specular_strength", "<f4"),
        ]
    ),
    np.dtype(
        [
            ("position", VEC3_LH),
            ("color", COLOR),
            ("diffuse_1_uv", UV),
            ("shadow_uv", UV),
            ("shadow_strength", "<f4"),
        ]
    ),
    np.dtype(
        [
            ("position", VEC3_LH),
            ("normal", VEC3_LH),
            ("color", COLOR),
            ("diffuse_1_uv", UV),
            ("specular_uv", UV),
            ("specular_strength", "<f4"),
            ("shadow_uv", UV),
            ("shadow_strength", "<f4"),
        ]
    ),
    np.dtype(
        [
            ("position", VEC3_LH),
            ("color", COLOR),
            ("diffuse_1_uv", UV),
            ("diffuse_2_uv", UV),
        ]
    ),
    np.dtype(
        [
            ("position", VEC3_LH),
            ("normal", VEC3_LH),
            ("color", COLOR),
            ("diffuse_1_uv", UV),
            ("diffuse_2_uv", UV),
            ("specular_uv", UV),
            ("specular_strength", "<f4"),
        ]
    ),
    np.dtype(
        [
            ("position", VEC3_LH),
            ("color", COLOR),
            ("diffuse_1_uv", UV),
            ("diffuse_2_uv", UV),
            ("shadow_uv", UV),
            ("shadow_strength", "<f4"),
        ]
    ),
    np.dtype(
        [
            ("position", VEC3_LH),
            ("normal", VEC3_LH),
            ("color", COLOR),
            ("diffuse_1_uv", UV),
            ("diffuse_2_uv", UV),
            ("specular_uv", UV),
            ("specular_strength", "<f4"),
            ("shadow_uv", UV),
            ("shadow_strength", "<f4"),
        ]
    ),
)
SWAY = np.dtype(
    [
        ("amplitude", "<f4"),
        ("angular_frequency", "<f4"),
        ("phase_offset", "<f4"),
    ]
)
SWAY_VERTEX_DTYPES = {
    0: np.dtype([("position", VEC3_LH), ("color", COLOR), ("sway", SWAY)]),
    1: np.dtype(
        [
            ("position", VEC3_LH),
            ("color", COLOR),
            ("diffuse_1_uv", UV),
            ("sway", SWAY),
        ]
    ),
    2: np.dtype(
        [
            ("position", VEC3_LH),
            ("color", COLOR),
            ("diffuse_1_uv", UV),
            ("diffuse_2_uv", UV),
            ("sway", SWAY),
        ]
    ),
}

BUFFER_NAMES = (
    "vertex_color",
    "single_texture",
    "single_texture_specular",
    "single_texture_shadow",
    "single_texture_specular_shadow",
    "double_texture",
    "double_texture_specular",
    "double_texture_shadow",
    "double_texture_specular_shadow",
)
PIXEL_SHADERS = tuple(range(9))
VERTEX_SHADERS = (
    (9,),
    (10, 21),
    (14, 25),
    (11, 22),
    (15, 26),
    (12, 23),
    (16, 27),
    (13, 24),
    (17, 28),
)


def _bounds(reader: BinaryReader, context: str) -> Bounds:
    offset = reader.offset
    result = reader.bounds()
    require_finite(
        result.center + result.half_extents,
        offset=offset,
        context=context,
    )
    if any(value < 0 for value in result.half_extents):
        raise FormatError("Negative bounding-box half extent", offset=offset, context=context)
    return result


def _triangles(reader: BinaryReader, context: str) -> Any:
    count_offset = reader.offset
    index_count = checked_count(
        reader.u32(),
        limit=MAX_BUFFERS * 3,
        offset=count_offset,
        context=f"{context} triangle index count",
    )
    triangle_count, remainder = divmod(index_count, 3)
    if remainder:
        raise FormatError(
            f"Triangle index count {index_count} is not divisible by 3",
            offset=count_offset,
            context=context,
        )
    return reader.array("<u2", triangle_count * 3, limit=MAX_BUFFERS * 3).reshape(
        (triangle_count, 3)
    )


def _vertices(
    reader: BinaryReader,
    dtype: Any,
    context: str,
) -> tuple[Any, memoryview]:
    count_offset = reader.offset
    count = checked_count(
        reader.u32(),
        limit=MAX_BUFFERS,
        offset=count_offset,
        context=f"{context} vertex count",
    )
    parsed_dtype = np.dtype(dtype)
    raw = reader.view(parsed_dtype.itemsize * count)
    return np.frombuffer(raw, dtype=parsed_dtype, count=count), raw


def _validate_triangle_indices(triangles: Any, vertex_count: int, context: str) -> None:
    if len(triangles) and int(np.max(triangles)) >= vertex_count:
        raise FormatError(
            f"Triangle index exceeds {vertex_count} vertices",
            context=context,
        )


def _mesh_buffer(reader: BinaryReader, index: int, context: str) -> MeshBuffer:
    name = BUFFER_NAMES[index]
    triangles = _triangles(reader, f"{context} {name}")
    vertices, raw = _vertices(reader, NORMAL_VERTEX_DTYPES[index], f"{context} {name}")
    _validate_triangle_indices(triangles, len(vertices), f"{context} {name}")
    return MeshBuffer(name, triangles, vertices, vertices.dtype.itemsize, raw)


def _render_chunk(
    reader: BinaryReader,
    buffers: tuple[MeshBuffer, ...],
    context: str,
) -> RenderChunk:
    start = reader.offset
    render_type, vertex_shader, pixel_shader, first_raw = reader.unpack("<4I")
    if render_type >= len(buffers):
        raise FormatError(
            f"Unknown geom render type {render_type}",
            offset=start,
            context=context,
        )
    if pixel_shader != PIXEL_SHADERS[render_type]:
        raise FormatError(
            f"Pixel shader {pixel_shader} is invalid for render type {render_type}",
            offset=start + 8,
            context=context,
        )
    first_triangle, remainder = divmod(first_raw, 3)
    if remainder:
        first_triangle = first_raw
        warnings.warn(
            f"Recovered nonstandard geom first triangle offset {first_raw} "
            f"at 0x{start:x} ({context})",
            RuntimeWarning,
            stacklevel=2,
        )
    triangle_count, vertex_count, first_vertex = reader.unpack("<3I")
    bounds = _bounds(reader, context)

    shadow_flags = reader.unpack("<4B")
    shadow_raw = reader.u32()
    specular_flags = reader.unpack("<4B")
    specular_raw = reader.u32()
    texture_count, texture_1_raw, texture_2_raw = reader.unpack("<3I")
    shader_flags = reader.unpack("<4B")
    velocities = reader.unpack("<6f")
    distance_flags = reader.unpack("<4B")

    for value, name in (
        (shadow_flags[0], "render marker"),
        (shadow_flags[1], "shadow flag"),
        (specular_flags[0], "specular flag"),
        (shader_flags[0], "UV animation flag"),
    ):
        if value not in (0, 1):
            raise FormatError(
                f"Invalid {name} {value}",
                offset=start,
                context=context,
            )
    if shadow_flags[0] != 1:
        raise FormatError("Geom render marker is not 1", offset=start, context=context)

    texture_1 = optional_index(texture_1_raw)
    texture_2 = optional_index(texture_2_raw)
    actual_texture_count = int(texture_1 is not None) + int(texture_2 is not None)
    if texture_count != actual_texture_count:
        raise FormatError(
            f"Geom chunk declares {texture_count} textures but stores "
            f"{actual_texture_count}",
            offset=start,
            context=context,
        )
    shadow = optional_index(shadow_raw)
    specular = optional_index(specular_raw)
    if bool(shadow_flags[1]) != (shadow is not None):
        raise FormatError("Inconsistent shadow texture flag", offset=start, context=context)
    if bool(specular_flags[0]) != (specular is not None):
        raise FormatError("Inconsistent specular texture flag", offset=start, context=context)
    require_finite(velocities, offset=start, context=context)
    uv_velocity = velocities if shader_flags[0] else None
    expected_vertex_shader = VERTEX_SHADERS[render_type]
    if vertex_shader not in expected_vertex_shader:
        raise FormatError(
            f"Vertex shader {vertex_shader} is invalid for render type {render_type}",
            offset=start + 4,
            context=context,
        )
    if bool(shader_flags[0]) != (len(expected_vertex_shader) == 2 and vertex_shader == expected_vertex_shader[1]):
        raise FormatError(
            "Geom UV animation flag does not match its vertex shader",
            offset=start + 80,
            context=context,
        )
    if distance_flags[0] not in (1, 2, 3):
        raise FormatError(
            f"Invalid geom distance class {distance_flags[0]}",
            offset=start + 108,
            context=context,
        )

    buffer = buffers[render_type]
    if first_triangle + triangle_count > len(buffer.triangles):
        raise FormatError(
            f"Geom triangle slice [{first_triangle}, "
            f"{first_triangle + triangle_count}) exceeds "
            f"{len(buffer.triangles)} {buffer.name} triangles",
            offset=start,
            context=context,
        )
    if first_vertex + vertex_count > len(buffer.vertices):
        selected = buffer.triangles[first_triangle : first_triangle + triangle_count]
        if (
            first_vertex == 0
            and len(selected)
            and int(np.max(selected)) < len(buffer.vertices)
        ):
            warnings.warn(
                f"Recovered malformed geom vertex slice [{first_vertex}, "
                f"{first_vertex + vertex_count}) for {buffer.name}; "
                f"using {len(buffer.vertices)} referenced vertices at "
                f"0x{start:x} ({context})",
                RuntimeWarning,
                stacklevel=2,
            )
            vertex_count = len(buffer.vertices)
        else:
            raise FormatError(
                f"Geom vertex slice [{first_vertex}, "
                f"{first_vertex + vertex_count}) exceeds "
                f"{len(buffer.vertices)} {buffer.name} vertices",
                offset=start,
                context=context,
            )
    selected = buffer.triangles[first_triangle : first_triangle + triangle_count]
    if len(selected) and (
        int(np.min(selected)) < first_vertex
        or int(np.max(selected)) >= first_vertex + vertex_count
    ):
        selected_min = int(np.min(selected))
        selected_max = int(np.max(selected))
        if selected_max < len(buffer.vertices):
            original_first_vertex = first_vertex
            original_vertex_count = vertex_count
            first_vertex = selected_min
            vertex_count = selected_max - selected_min + 1
            warnings.warn(
                f"Recovered malformed geom vertex slice "
                f"[{original_first_vertex}, "
                f"{original_first_vertex + original_vertex_count}) as "
                f"[{first_vertex}, {first_vertex + vertex_count}) for "
                f"{buffer.name} at 0x{start:x} ({context})",
                RuntimeWarning,
                stacklevel=2,
            )
        else:
            raise FormatError(
                "Geom chunk triangle references vertices outside its declared slice",
                offset=start,
                context=context,
            )
    return RenderChunk(
        render_type=render_type,
        vertex_shader=vertex_shader,
        pixel_shader=pixel_shader,
        first_triangle=first_triangle,
        triangle_count=triangle_count,
        first_vertex=first_vertex,
        vertex_count=vertex_count,
        bounds=bounds,
        shadow_texture=shadow,
        specular_texture=specular,
        texture_1=texture_1,
        texture_2=texture_2,
        uv_velocity=uv_velocity,
        distance_class=distance_flags[0],
        raw_flags=bytes(
            shadow_flags[2:]
            + specular_flags[1:]
            + shader_flags[1:]
            + distance_flags[1:]
        ),
    )


def parse_geom_blocks(
    payload: bytes | bytearray | memoryview,
) -> GeomBlocks:
    reader = BinaryReader(payload, context="LBS geom blocks")
    count_offset = reader.offset
    block_count = checked_count(
        reader.u32(),
        limit=MAX_BLOCKS,
        offset=count_offset,
        context="geom block count",
    )
    glossiness = reader.f32()
    require_finite(
        (glossiness,),
        offset=count_offset + 4,
        context="geom glossiness",
    )
    blocks: list[GeomBlock] = []
    for index in range(block_count):
        context = f"geom block {index}"
        buffers = tuple(_mesh_buffer(reader, buffer_index, context) for buffer_index in range(9))
        chunk_count_offset = reader.offset
        chunk_count = checked_count(
            reader.u32(),
            limit=MAX_BUFFERS,
            offset=chunk_count_offset,
            context=f"{context} render chunk count",
        )
        chunks = tuple(_render_chunk(reader, buffers, context) for _ in range(chunk_count))
        blocks.append(GeomBlock(buffers, chunks, _bounds(reader, context)))
    return GeomBlocks(glossiness, tuple(blocks), reader.remaining_view())


def _flags_texture_count(flags: int) -> int:
    if flags & HAS_DOUBLE_TEXTURE:
        return 2
    if flags & HAS_SINGLE_TEXTURE:
        return 1
    return 0


def _sway_dtype(flags: int, stride: int) -> Any | None:
    count = _flags_texture_count(flags)
    dtype = SWAY_VERTEX_DTYPES.get(count)
    return dtype if dtype is not None and dtype.itemsize == stride else None


def _normal_dtype(flags: int, stride: int) -> Any | None:
    texture_count = _flags_texture_count(flags)
    specular = (flags & HAS_SPECULAR_TEXTURE) == HAS_SPECULAR_TEXTURE
    index = 0
    if texture_count == 1:
        index = 2 if specular else 1
    elif texture_count == 2:
        index = 6 if specular else 5
    elif specular:
        return None
    dtype = NORMAL_VERTEX_DTYPES[index]
    return dtype if dtype.itemsize == stride else None


def _raw_vertices(
    reader: BinaryReader,
    stride: int,
    context: str,
    *,
    dtype: Any | None,
) -> tuple[Any, memoryview]:
    if stride <= 0 or stride > 1024:
        raise FormatError(
            f"Invalid vertex stride {stride}",
            offset=reader.offset,
            context=context,
        )
    count_offset = reader.offset
    count = checked_count(
        reader.u32(),
        limit=MAX_BUFFERS,
        offset=count_offset,
        context=f"{context} vertex count",
    )
    raw = reader.view(count * stride)
    if dtype is None:
        vertices = np.frombuffer(raw, dtype=np.uint8).reshape((count, stride))
    else:
        vertices = np.frombuffer(raw, dtype=dtype, count=count)
    return vertices, raw


def _object_block(reader: BinaryReader, context: str) -> ObjectBlock:
    start = reader.offset
    track_flags, render_flags = reader.unpack("<II")
    texture_count = _flags_texture_count(track_flags)
    texture_1 = reader.u32() if texture_count else None
    texture_2 = reader.u32() if texture_count == 2 else None
    unused, vertex_stride, fvf = reader.unpack("<III")
    main = _triangles(reader, context)
    lod = reader.u8()
    if lod not in (0, 1, 2):
        raise FormatError(f"Unknown object-block LOD {lod}", offset=reader.offset - 1)
    far = _triangles(reader, f"{context} far buffer") if lod == 2 else None
    dtype = _sway_dtype(track_flags, vertex_stride)
    vertices, raw_vertices = _raw_vertices(
        reader,
        vertex_stride,
        context,
        dtype=dtype,
    )
    _validate_triangle_indices(main, len(vertices), context)
    if far is not None:
        _validate_triangle_indices(far, len(vertices), f"{context} far buffer")
    return ObjectBlock(
        track_flags=track_flags,
        render_flags=render_flags,
        texture_1=texture_1,
        texture_2=texture_2,
        unused=unused,
        vertex_stride=vertex_stride,
        fvf=fvf,
        main_triangles=main,
        lod=lod,
        far_triangles=far,
        vertices=vertices,
        raw_vertices=raw_vertices,
        bounds=_bounds(reader, context),
    )


def parse_object_blocks(
    payload: bytes | bytearray | memoryview,
) -> tuple[tuple[ObjectBlockGroup | None, ...], memoryview]:
    reader = BinaryReader(payload, context="LBS object blocks")
    group_count = checked_count(
        reader.u32(),
        limit=MAX_BLOCKS,
        offset=reader.offset - 4,
        context="object-block group count",
    )
    groups: list[ObjectBlockGroup | None] = []
    for group_index in range(group_count):
        present_offset = reader.offset
        present = reader.u32()
        if present not in (0, 1):
            raise FormatError(
                f"Invalid object-block presence marker {present}",
                offset=present_offset,
                context=f"object-block group {group_index}",
            )
        if not present:
            groups.append(None)
            continue

        def block_list(label: str) -> tuple[ObjectBlock, ...]:
            count_offset = reader.offset
            count = checked_count(
                reader.u32(),
                limit=MAX_OBJECTS,
                offset=count_offset,
                context=f"object-block {label} count",
            )
            return tuple(
                _object_block(reader, f"object group {group_index} {label} {index}")
                for index in range(count)
            )

        groups.append(ObjectBlockGroup(block_list("primary"), block_list("secondary")))
    return tuple(groups), reader.remaining_view()


def parse_car_location(
    payload: bytes | bytearray | memoryview,
) -> tuple[CarLocation, memoryview]:
    reader = BinaryReader(payload, context="LBS car location")
    offset = reader.offset
    matrix = reader.unpack("<16f")
    position = reader.vec3()
    euler = reader.vec3()
    require_finite(
        matrix + position + euler,
        offset=offset,
        context="car location",
    )
    return CarLocation(matrix, position, euler), reader.remaining_view()


def parse_object_data_3d(reader: BinaryReader, context: str) -> ObjectData3D:
    start = reader.offset
    track_flags, render_flags = reader.unpack("<II")
    single = bool(track_flags & HAS_SINGLE_TEXTURE)
    double = bool(track_flags & HAS_DOUBLE_TEXTURE)
    specular = (track_flags & HAS_SPECULAR_TEXTURE) == HAS_SPECULAR_TEXTURE
    animated = bool(track_flags & HAS_SHADER_DATA)
    if double and not single:
        raise FormatError(
            "Double-texture ObjectData3D has no first texture",
            offset=start,
            context=context,
        )
    texture_1: int | None = None
    texture_2: int | None = None
    specular_texture: int | None = None
    uv_1 = uv_2 = uv_specular = None
    if single:
        texture_1 = reader.u32()
        if animated:
            uv_1 = reader.vec2()
    if double:
        texture_2 = reader.u32()
        if animated:
            uv_2 = reader.vec2()
    if specular:
        specular_texture = reader.u32()
        if animated:
            uv_specular = reader.vec2()
    vertex_stride, fvf = reader.unpack("<II")
    triangles = _triangles(reader, context)
    dtype = _normal_dtype(track_flags, vertex_stride)
    if dtype is None:
        raise FormatError(
            f"ObjectData3D flags 0x{track_flags:x} do not match stride {vertex_stride}",
            offset=start,
            context=context,
        )
    vertices, raw_vertices = _raw_vertices(
        reader,
        vertex_stride,
        context,
        dtype=dtype,
    )
    _validate_triangle_indices(triangles, len(vertices), context)
    raw = reader.view_at(start, reader.offset - start)
    return ObjectData3D(
        track_flags=track_flags,
        render_flags=render_flags,
        texture_1=texture_1,
        texture_2=texture_2,
        specular_texture=specular_texture,
        uv_velocity=(uv_1, uv_2, uv_specular) if animated else None,
        vertex_stride=vertex_stride,
        fvf=fvf,
        triangles=triangles,
        vertices=vertices,
        raw_vertices=raw_vertices,
        raw=raw,
    )


def _parse_super_bowl(reader: BinaryReader) -> tuple[ObjectData3DGroup, ...]:
    name = reader.cstring()
    count = checked_count(
        reader.u32(),
        limit=MAX_OBJECTS,
        offset=reader.offset - 4,
        context="super-bowl object count",
    )
    items = []
    for index in range(count):
        position = reader.vec3()
        require_finite(
            position,
            offset=reader.offset - 12,
            context=f"super-bowl object {index}",
        )
        items.append(
            ObjectData3DItem(
                position,
                parse_object_data_3d(reader, f"super-bowl object {index}"),
            )
        )
    return (ObjectData3DGroup(SUPER_BOWL, name, None, tuple(items)),)


def _parse_data_3d_groups(
    reader: BinaryReader,
    segment_kind: int,
) -> tuple[ObjectData3DGroup, ...]:
    count = checked_count(
        reader.u32(),
        limit=MAX_OBJECTS,
        offset=reader.offset - 4,
        context="ObjectData3D group count",
    )
    groups: list[ObjectData3DGroup] = []
    for group_index in range(count):
        name = reader.cstring()
        object_kind_raw = reader.u8()
        if segment_kind in (REFLECTION_OBJECTS, WATER_OBJECTS) and object_kind_raw != 0:
            raise FormatError(
                f"ObjectData3D group marker is {object_kind_raw}, expected 0",
                offset=reader.offset - 1,
                context=name,
            )
        data_count = checked_count(
            reader.u32(),
            limit=MAX_OBJECTS,
            offset=reader.offset - 4,
            context=f"{name} ObjectData3D count",
        )
        items = tuple(
            ObjectData3DItem(
                None,
                parse_object_data_3d(reader, f"{name} ObjectData3D {index}"),
            )
            for index in range(data_count)
        )
        instances: list[tuple[int, tuple[float, ...]]] = []
        if segment_kind == INTERACTIVE_OBJECTS:
            instance_count = checked_count(
                reader.u32(),
                limit=MAX_OBJECTS,
                offset=reader.offset - 4,
                context=f"{name} instance count",
            )
            for instance_index in range(instance_count):
                offset = reader.offset
                key = reader.u32()
                matrix = reader.unpack("<16f")
                require_finite(
                    matrix,
                    offset=offset,
                    context=f"{name} instance {instance_index}",
                )
                instances.append((key, matrix))
        groups.append(
            ObjectData3DGroup(
                segment_kind=segment_kind,
                name=name,
                object_kind=object_kind_raw if segment_kind == INTERACTIVE_OBJECTS else None,
                items=items,
                instances=tuple(instances),
            )
        )
    return tuple(groups)


def _parse_world_bounds(
    payload: memoryview,
) -> tuple[tuple[Bounds, ...], memoryview]:
    reader = BinaryReader(payload, context="LBS visible objects")
    count = checked_count(
        reader.u32(),
        limit=MAX_BLOCKS,
        offset=reader.offset - 4,
        context="visible-object bound count",
    )
    bounds = tuple(_bounds(reader, f"world bound {index}") for index in range(count))
    return bounds, reader.remaining_view()


def _load_lbs_source(source: Source) -> tuple[BinarySource, bool]:
    loaded = load_source(source)
    if bytes(loaded.view[:4]) != _IBS_ENCODED_HEADER:
        return loaded, False
    decoded = loaded.view.tobytes().translate(_IBS_DECODE_TABLE)
    if not decoded:
        raise FormatError(f"Encoded Ibs source {loaded.name} is empty")
    return (
        BinarySource(memoryview(decoded), decoded, f"{loaded.name} (decoded Ibs)"),
        True,
    )


def _validate_ibs_sections(segments: tuple[Any, ...], source_name: str) -> None:
    seen: set[int] = set()
    for segment in segments:
        if segment.kind not in _LBS_KNOWN_SECTIONS:
            continue
        if segment.kind in seen:
            raise FormatError(
                f"Encoded Ibs source {source_name} has duplicate LBS section "
                f"0x{segment.kind:x}"
            )
        seen.add(segment.kind)
    missing = sorted(_IBS_REQUIRED_SECTIONS - seen)
    if missing:
        raise FormatError(
            f"Encoded Ibs source {source_name} is missing required LBS sections: "
            + ", ".join(f"0x{kind:x}" for kind in missing)
        )


def _validate_ibs_geometry(lbs: LbsFile, source_name: str) -> None:
    if lbs.geom_blocks is None or not any(
        chunk.triangle_count
        for block in lbs.geom_blocks.blocks
        for chunk in block.render_chunks
    ):
        raise FormatError(f"Encoded Ibs source {source_name} has no visual geometry")
    if lbs.section_trailing:
        sections = ", ".join(f"0x{kind:x}" for kind in sorted(lbs.section_trailing))
        raise FormatError(
            f"Encoded Ibs source {source_name} has trailing data in LBS sections: "
            f"{sections}"
        )


def _lbs_segments(source: Source) -> tuple[BinarySource, bool, tuple[Any, ...]]:
    loaded, encoded = _load_lbs_source(source)
    segments = scan_segments(loaded.view, context=f"LBS {loaded.name}")
    if isinstance(source, (str, os.PathLike)):
        companion = Path(loaded.name).with_suffix(".lb2")
        if current_filesystem().is_file(companion):
            extra, _ = _load_lbs_source(companion)
            segments += scan_segments(extra.view, context=f"LBS {extra.name}")
    if encoded:
        _validate_ibs_sections(segments, loaded.name)
    return loaded, encoded, segments


def parse_lbs(source: Source) -> LbsFile:
    loaded, encoded, segments = _lbs_segments(source)
    world_bounds = None
    geom_blocks = None
    object_blocks = None
    car_location = None
    groups: dict[int, tuple[ObjectData3DGroup, ...]] = {}
    trailing: dict[int, memoryview] = {}
    parsed_kinds = {
        GEOM_BLOCKS,
        OBJECT_BLOCKS,
        SUPER_BOWL,
        INTERACTIVE_OBJECTS,
        REFLECTION_OBJECTS,
        WATER_OBJECTS,
        VISIBLE_OBJECTS,
        CAR_LOCATION,
    }
    seen: set[int] = set()
    for segment in segments:
        if segment.kind not in parsed_kinds:
            continue
        if segment.kind in seen:
            raise FormatError(
                f"Duplicate LBS section 0x{segment.kind:x}",
                offset=segment.offset,
                context="LBS segment",
            )
        seen.add(segment.kind)
        remainder: memoryview
        if segment.kind == GEOM_BLOCKS:
            geom_blocks = parse_geom_blocks(segment.payload)
            remainder = geom_blocks.trailing
        elif segment.kind == OBJECT_BLOCKS:
            object_blocks, remainder = parse_object_blocks(segment.payload)
        elif segment.kind == CAR_LOCATION:
            car_location, remainder = parse_car_location(segment.payload)
        elif segment.kind == VISIBLE_OBJECTS:
            world_bounds, remainder = _parse_world_bounds(segment.payload)
        else:
            reader = BinaryReader(
                segment.payload,
                context=f"LBS ObjectData3D section 0x{segment.kind:x}",
            )
            if segment.kind == SUPER_BOWL:
                groups[segment.kind] = _parse_super_bowl(reader)
            else:
                groups[segment.kind] = _parse_data_3d_groups(reader, segment.kind)
            remainder = reader.remaining_view()
        if remainder:
            trailing[segment.kind] = remainder
    result = LbsFile(
        segments=segments,
        world_bounds=world_bounds,
        geom_blocks=geom_blocks,
        object_blocks=object_blocks,
        car_location=car_location,
        object_data_groups=groups,
        section_trailing=trailing,
        raw=loaded.view,
    )
    if encoded:
        _validate_ibs_geometry(result, loaded.name)
    return result


def parse_lbs_pacenote_inputs(source: Source) -> LbsFile:
    loaded, _, segments = _lbs_segments(source)
    car_location = None
    trailing: dict[int, memoryview] = {}
    for segment in segments:
        if segment.kind != CAR_LOCATION:
            continue
        if car_location is not None:
            raise FormatError(
                "Duplicate LBS section 0xb",
                offset=segment.offset,
                context="LBS segment",
            )
        car_location, remainder = parse_car_location(segment.payload)
        if remainder:
            trailing[segment.kind] = remainder
    return LbsFile(
        segments=segments,
        world_bounds=None,
        geom_blocks=None,
        object_blocks=None,
        car_location=car_location,
        object_data_groups={},
        section_trailing=trailing,
        raw=loaded.view,
    )
