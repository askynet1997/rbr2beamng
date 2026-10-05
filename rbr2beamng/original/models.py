from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping


Vec2 = tuple[float, float]
Vec3 = tuple[float, float, float]
Quat = tuple[float, float, float, float]
Matrix4 = tuple[float, ...]


@dataclass(frozen=True)
class Bounds:
    center: Vec3
    half_extents: Vec3


@dataclass(frozen=True)
class FencePost:
    position: Vec3
    bounds: Bounds
    color: tuple[int, int, int, int]


@dataclass(frozen=True)
class Fence:
    tile_type: int
    pole_type: int
    tile_texture_index: int
    pole_texture_index: int
    bounds: Bounds
    posts: tuple[FencePost, ...]


@dataclass(frozen=True)
class FncFile:
    version: int
    fences: tuple[Fence, ...]
    textures: tuple[str, ...]
    raw: memoryview = field(repr=False, compare=False)


@dataclass(frozen=True)
class OriginalSegment:
    offset: int
    header_size: int
    category: int
    kind: int
    payload: memoryview


@dataclass(frozen=True)
class MaterialMap:
    width: int
    height: int
    values: Any
    raw: memoryview


@dataclass(frozen=True)
class MaterialCondition:
    name: str
    surface: str
    age: str
    identifier: str
    maps: tuple[MaterialMap, ...]


@dataclass(frozen=True)
class MatFile:
    conditions: tuple[MaterialCondition, ...]
    trailing: memoryview
    raw: memoryview = field(repr=False, compare=False)


@dataclass(frozen=True)
class DrivelinePoint:
    position: Vec3
    tangent: Vec3
    distance: float
    unused: tuple[int, int] = (0, 0)


@dataclass(frozen=True)
class SoftVolume:
    kind: int
    center: Vec3
    size_or_radius: Vec3 | float


@dataclass(frozen=True)
class ShapeFace:
    selector: int
    indices: tuple[int, int, int]

@dataclass(frozen=True)
class ShapeInstance:
    key: int
    position: Vec3
    scale: Vec3
    rotation: Quat


@dataclass(frozen=True)
class ShapeCollisionMesh:
    name: str
    object_kind: int
    material_id: int
    use_local_rotation: bool
    soft_volume: SoftVolume | None
    vertices: tuple[Vec3, ...]
    faces: tuple[ShapeFace, ...]
    instances: tuple[ShapeInstance, ...]


@dataclass(frozen=True)
class TrkFile:
    segments: tuple[OriginalSegment, ...]
    driveline: tuple[DrivelinePoint, ...] | None
    shape_collision_meshes: tuple[ShapeCollisionMesh, ...] | None
    section_trailing: Mapping[int, memoryview]
    raw: memoryview = field(repr=False, compare=False)

@dataclass(frozen=True)
class DlsAnimationSetDescriptor:
    name: str
    name_offset: int
    descriptor_offset: int
    descriptor_size: int
    counts: tuple[int, ...]
    offsets: tuple[int, ...]
    raw_sections: Mapping[int, memoryview]


@dataclass(frozen=True)
class OriginalPacenote:
    note_id: int
    flags: int
    distance: float
    animation_set: str

@dataclass(frozen=True)
class DlsFile:
    header: bytes
    directory: tuple[int | None, ...]
    sections: Mapping[int, memoryview]
    names: Mapping[int, str]
    animation_sets: tuple[DlsAnimationSetDescriptor, ...]
    all_pacenotes: tuple[OriginalPacenote, ...]
    pacenotes: tuple[OriginalPacenote, ...]
    raw: memoryview = field(repr=False, compare=False)


@dataclass(frozen=True)
class MeshBuffer:
    name: str
    triangles: Any
    vertices: Any
    vertex_stride: int
    raw_vertices: memoryview


@dataclass(frozen=True)
class RenderChunk:
    render_type: int
    vertex_shader: int
    pixel_shader: int
    first_triangle: int
    triangle_count: int
    first_vertex: int
    vertex_count: int
    bounds: Bounds
    shadow_texture: int | None
    specular_texture: int | None
    texture_1: int | None
    texture_2: int | None
    uv_velocity: tuple[float, ...] | None
    distance_class: int
    raw_flags: bytes


@dataclass(frozen=True)
class GeomBlock:
    buffers: tuple[MeshBuffer, ...]
    render_chunks: tuple[RenderChunk, ...]
    bounds: Bounds


@dataclass(frozen=True)
class GeomBlocks:
    glossiness: float
    blocks: tuple[GeomBlock, ...]
    trailing: memoryview


@dataclass(frozen=True)
class ObjectBlock:
    track_flags: int
    render_flags: int
    texture_1: int | None
    texture_2: int | None
    unused: int
    vertex_stride: int
    fvf: int
    main_triangles: Any
    lod: int
    far_triangles: Any | None
    vertices: Any
    raw_vertices: memoryview
    bounds: Bounds


@dataclass(frozen=True)
class ObjectBlockGroup:
    primary: tuple[ObjectBlock, ...]
    secondary: tuple[ObjectBlock, ...]


@dataclass(frozen=True)
class CarLocation:
    matrix: Matrix4
    position: Vec3
    euler_vector: Vec3


@dataclass(frozen=True)
class ObjectData3D:
    track_flags: int
    render_flags: int
    texture_1: int | None
    texture_2: int | None
    specular_texture: int | None
    uv_velocity: tuple[Vec2 | None, Vec2 | None, Vec2 | None] | None
    vertex_stride: int
    fvf: int
    triangles: Any
    vertices: Any
    raw_vertices: memoryview
    raw: memoryview


@dataclass(frozen=True)
class ObjectData3DItem:
    position: Vec3 | None
    data: ObjectData3D


@dataclass(frozen=True)
class ObjectData3DGroup:
    segment_kind: int
    name: str
    object_kind: int | None
    items: tuple[ObjectData3DItem, ...]
    instances: tuple[tuple[int, Matrix4], ...] = ()


@dataclass(frozen=True)
class LbsFile:
    segments: tuple[OriginalSegment, ...]
    world_bounds: tuple[Bounds, ...] | None
    geom_blocks: GeomBlocks | None
    object_blocks: tuple[ObjectBlockGroup | None, ...] | None
    car_location: CarLocation | None
    object_data_groups: Mapping[int, tuple[ObjectData3DGroup, ...]]
    section_trailing: Mapping[int, memoryview]
    raw: memoryview = field(repr=False, compare=False)

@dataclass(frozen=True)
class WaterSurface:
    vertices: tuple[Vec3, Vec3, Vec3, Vec3]
    padding: tuple[float, float, float, float]


@dataclass(frozen=True)
class ColDescriptor:
    offset: int
    data_type: int
    tree_type: int
    traversal: tuple[bool, ...]
    vertex_count: int
    vertices_offset: int
    tree_offset: int


@dataclass(frozen=True)
class CollisionTriangleBatch:
    records: Any

    @property
    def count(self) -> int:
        return len(self.records)


@dataclass(frozen=True)
class CollisionTreeNode:
    bounds: Bounds
    triangle_count: int
    link: bool
    data_offset: int
    triangles: CollisionTriangleBatch | None = None
    left: CollisionTreeNode | None = None
    right: CollisionTreeNode | None = None


@dataclass(frozen=True)
class CollisionSubtree:
    descriptor: ColDescriptor
    vertices: Any
    root: CollisionTreeNode


@dataclass(frozen=True)
class BrakeWall:
    inner_points: Any
    outer_points: Any
    segments: tuple[int, ...]


@dataclass(frozen=True)
class ColFile:
    root_descriptor: ColDescriptor
    root_tree: CollisionTreeNode
    subtrees: tuple[CollisionSubtree, ...]
    wet_surfaces: tuple[WaterSurface, ...]
    water_surfaces: tuple[WaterSurface, ...]
    brake_wall_raw: memoryview | None
    raw: memoryview = field(repr=False, compare=False)
    brake_wall: BrakeWall | None = None

    @property
    def triangle_count(self) -> int:
        def count(node: CollisionTreeNode) -> int:
            if node.triangles is not None:
                return node.triangles.count
            return (count(node.left) if node.left else 0) + (
                count(node.right) if node.right else 0
            )

        return sum(count(subtree.root) for subtree in self.subtrees)
