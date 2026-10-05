from __future__ import annotations

from .binary import (
    BinaryReader,
    FormatError,
    Source,
    checked_count,
    load_source,
    require_finite,
    scan_segments,
)
from .models import (
    DrivelinePoint,
    ShapeCollisionMesh,
    ShapeFace,
    ShapeInstance,
    SoftVolume,
    TrkFile,
)


DRIVELINE_SEGMENT = 0x14
SHAPE_COLLISION_SEGMENT = 0x16
PHYSICS_CATEGORY = 0x3

MAX_DRIVELINE_POINTS = 2_000_000
MAX_SHAPE_MESHES = 0x2000
MAX_SHAPE_VERTICES = 30
MAX_SHAPE_EDGES = 84
MAX_SHAPE_FACES = 56
MAX_SHAPE_INSTANCES = 2_000_000


def parse_driveline(
    payload: bytes | bytearray | memoryview,
) -> tuple[tuple[DrivelinePoint, ...], memoryview]:
    reader = BinaryReader(payload, context="TRK driveline")
    count_offset = reader.offset
    count = checked_count(
        reader.u32(),
        limit=MAX_DRIVELINE_POINTS,
        offset=count_offset,
        context="driveline point count",
    )
    points: list[DrivelinePoint] = []
    for index in range(count):
        point_offset = reader.offset
        position = reader.vec3()
        tangent = reader.vec3()
        distance = reader.f32()
        unused = reader.unpack("<HH")
        require_finite(
            position + tangent + (distance,),
            offset=point_offset,
            context=f"driveline point {index}",
        )
        points.append(DrivelinePoint(position, tangent, distance, unused))
    return tuple(points), reader.remaining_view()


def _edge_count(faces: list[ShapeFace]) -> int:
    edges: set[tuple[int, int]] = set()
    for face in faces:
        a, b, c = face.indices
        edges.add(tuple(sorted((a, b))))
        edges.add(tuple(sorted((b, c))))
        edges.add(tuple(sorted((c, a))))
    return len(edges)


def parse_shape_collision_meshes(
    payload: bytes | bytearray | memoryview,
) -> tuple[tuple[ShapeCollisionMesh, ...], memoryview]:
    reader = BinaryReader(payload, context="TRK shape collision meshes")
    count_offset = reader.offset
    mesh_count = checked_count(
        reader.u32(),
        limit=MAX_SHAPE_MESHES,
        offset=count_offset,
        context="shape mesh count",
    )
    meshes: list[ShapeCollisionMesh] = []
    for mesh_index in range(mesh_count):
        name = reader.cstring()
        object_kind = reader.u32()
        material_id = reader.u8()
        use_local_raw = reader.u8()
        if use_local_raw not in (0, 1):
            raise FormatError(
                f"Invalid local-rotation flag {use_local_raw}",
                offset=reader.offset - 1,
                context=name,
            )
        volume_offset = reader.offset
        volume_kind = reader.u32()
        soft_volume: SoftVolume | None
        if volume_kind == 0:
            soft_volume = None
        elif volume_kind == 1:
            center = reader.vec3()
            stored_half_extents = reader.vec3()
            require_finite(
                center + stored_half_extents,
                offset=volume_offset,
                context=f"shape soft box {name}",
            )
            # Half-extents are stored in (x, z, y) order, unlike the (x, y, z) center.
            half_extents = (
                stored_half_extents[0],
                stored_half_extents[2],
                stored_half_extents[1],
            )
            soft_volume = SoftVolume(volume_kind, center, half_extents)
        elif volume_kind == 2:
            center = reader.vec3()
            radius = reader.f32()
            require_finite(
                center + (radius,),
                offset=volume_offset,
                context=f"shape soft sphere {name}",
            )
            if radius < 0:
                raise FormatError(
                    f"Negative soft-volume radius {radius}",
                    offset=volume_offset,
                    context=name,
                )
            soft_volume = SoftVolume(volume_kind, center, radius)
        else:
            raise FormatError(
                f"Unknown soft-volume type {volume_kind}",
                offset=volume_offset,
                context=name,
            )

        vertex_count_offset = reader.offset
        vertex_count = checked_count(
            reader.u32(),
            limit=MAX_SHAPE_VERTICES,
            offset=vertex_count_offset,
            context=f"{name} vertex count",
        )
        vertices: list[tuple[float, float, float]] = []
        for vertex_index in range(vertex_count):
            vertex_offset = reader.offset
            vertex = reader.vec3()
            require_finite(
                vertex,
                offset=vertex_offset,
                context=f"{name} vertex {vertex_index}",
            )
            vertices.append(vertex)

        face_count_offset = reader.offset
        face_count = checked_count(
            reader.u32(),
            limit=MAX_SHAPE_FACES,
            offset=face_count_offset,
            context=f"{name} face count",
        )
        faces: list[ShapeFace] = []
        for face_index in range(face_count):
            face_offset = reader.offset
            selector = reader.u8()
            indices = reader.unpack("<III")
            if any(index >= vertex_count for index in indices):
                raise FormatError(
                    f"Shape face {indices} references {vertex_count} vertices",
                    offset=face_offset,
                    context=f"{name} face {face_index}",
                )
            faces.append(ShapeFace(selector, indices))
        edges = _edge_count(faces)
        if edges > MAX_SHAPE_EDGES:
            raise FormatError(
                f"Shape mesh has {edges} edges; maximum is {MAX_SHAPE_EDGES}",
                offset=face_count_offset,
                context=name,
            )

        instance_count_offset = reader.offset
        instance_count = checked_count(
            reader.u32(),
            limit=MAX_SHAPE_INSTANCES,
            offset=instance_count_offset,
            context=f"{name} instance count",
        )
        instances: list[ShapeInstance] = []
        for instance_index in range(instance_count):
            instance_offset = reader.offset
            key = reader.u32()
            position = reader.vec3()
            scale = reader.vec3()
            rotation = reader.unpack("<4f")
            require_finite(
                position + scale + rotation,
                offset=instance_offset,
                context=f"{name} instance {instance_index}",
            )
            instances.append(ShapeInstance(key, position, scale, rotation))

        meshes.append(
            ShapeCollisionMesh(
                name=name,
                object_kind=object_kind,
                material_id=material_id,
                use_local_rotation=bool(use_local_raw),
                soft_volume=soft_volume,
                vertices=tuple(vertices),
                faces=tuple(faces),
                instances=tuple(instances),
            )
        )
    return tuple(meshes), reader.remaining_view()


def parse_trk(source: Source) -> TrkFile:
    loaded = load_source(source)
    segments = scan_segments(loaded.view, context=f"TRK {loaded.name}")
    driveline: tuple[DrivelinePoint, ...] | None = None
    shapes: tuple[ShapeCollisionMesh, ...] | None = None
    trailing: dict[int, memoryview] = {}
    seen: set[int] = set()
    for segment in segments:
        if segment.kind not in (DRIVELINE_SEGMENT, SHAPE_COLLISION_SEGMENT):
            continue
        if segment.category != PHYSICS_CATEGORY:
            raise FormatError(
                f"TRK section 0x{segment.kind:x} has category {segment.category}, expected 3",
                offset=segment.offset,
                context="TRK segment",
            )
        if segment.kind in seen:
            raise FormatError(
                f"Duplicate TRK section 0x{segment.kind:x}",
                offset=segment.offset,
                context="TRK segment",
            )
        seen.add(segment.kind)
        if segment.kind == DRIVELINE_SEGMENT:
            driveline, remainder = parse_driveline(segment.payload)
        else:
            shapes, remainder = parse_shape_collision_meshes(segment.payload)
        if remainder:
            trailing[segment.kind] = remainder
    return TrkFile(segments, driveline, shapes, trailing, loaded.view)
