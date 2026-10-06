from __future__ import annotations

import hashlib
import math
import re
import shutil
import warnings
from collections import OrderedDict
from dataclasses import dataclass, field, replace
from pathlib import Path, PurePosixPath, PureWindowsPath
from types import MappingProxyType
from typing import Callable, Hashable, Iterable, Mapping, Sequence

import numpy as np

from ..filesystem import current_filesystem
from ..models import (
    DEFAULT_WATER_NAME_MATCHES,
    DrivelinePoint,
    MaterialVariant,
    MeshPart,
    Pacenote,
    PbrMaterialOverride,
    RbrMaterial,
    RbrStage,
    RbrSurface,
    Spawn,
    StageLocation,
    StageMetadata,
)
from ..surface_profiles import current_surface_rules
from ..profiling import profile_span
from ..rbr import (
    is_water_material_reference,
    surface_profile_warnings,
    texture_has_alpha,
)
from ..surface_clip import (
    AREA_EPSILON,
    SELECTOR_MAP_CELLS,
    Region,
    Vertex,
    clip_polygon,
    clip_polygon_to_selector_regions,
    decompose_regions,
    merge_fragments,
    polygon_area,
    triangle_cost,
    triangulate,
)
from .fnc import fence_render_definition
from .models import (
    BrakeWall,
    ColFile,
    CollisionTreeNode,
    DlsFile,
    FncFile,
    LbsFile,
    MatFile,
    MaterialCondition,
    ObjectBlock,
    ObjectData3D,
    TrkFile,
)
from .textures import (
    DEFAULT_ROAD_CONDITION,
    MissingTextureError,
    OriginalTexture,
    OriginalTextureIni,
    OriginalTextureLookup,
)


SUPER_BOWL = 0x3
INTERACTIVE_OBJECTS = 0x6
REFLECTION_OBJECTS = 0x7
WATER_OBJECTS = 0x8
NO_CULLING = 0x1

# Alpha-test references as BeamNG ``alpha >= ref``. HedgeHog3D object
# materials use ALPHAFUNC GREATER / ALPHAREF 200 (RenderQuality "high");
# CFenceRenderer uses GREATER / 10. Blended passes that also wrote depth in
# RBR keep only fully opaque texels in the depth-writing pass.
_OBJECT_CUTOUT_ALPHA_REF = 201
_FENCE_BLEND_ALPHA_REF = 11
_OPAQUE_TEXEL_ALPHA_REF = 255

# Visual objects and fences are written without collision, so BeamNG never
# reads their ground type.
_RENDER_ONLY_GROUND_TYPE = "ROCK"


_LH_TO_Z_UP = np.array(
    (
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0),
    ),
    dtype=np.float64,
)
_LH_TO_Z_UP_4 = np.identity(4, dtype=np.float64)
_LH_TO_Z_UP_4[:3, :3] = _LH_TO_Z_UP


class OriginalAdapterError(ValueError):
    pass


@dataclass(frozen=True)
class TextureReference:
    kind: str
    index: int


@dataclass(frozen=True)
class VisualMaterialSpec:
    name: str
    material_index: int
    source_kind: str
    diffuse_1: int | None
    diffuse_2: int | None
    specular: int | None
    shadow: int | None
    render_flags: int
    water: bool
    glossiness: float | None = None

    @property
    def texture_references(self) -> tuple[TextureReference, ...]:
        result: list[TextureReference] = []
        for kind, index in (
            ("diffuse", self.diffuse_1),
            ("diffuse", self.diffuse_2),
            ("specular", self.specular),
            ("shadow", self.shadow),
        ):
            if index is not None:
                reference = TextureReference(kind, index)
                if reference not in result:
                    result.append(reference)
        return tuple(result)


@dataclass(frozen=True)
class AdaptedVisuals:
    parts: tuple[MeshPart, ...]
    material_specs: tuple[VisualMaterialSpec, ...]
    warnings: tuple[str, ...]


@dataclass(frozen=True)
class CollisionSubdivisionStats:
    source_triangles: int = 0
    generated_triangles: int = 0
    subdivided_triangles: int = 0
    fallbacks: int = 0
    merged_away_triangles: int = 0


@dataclass(frozen=True)
class AdaptedCollision:
    parts: tuple[MeshPart, ...]
    surface_ids: tuple[int, ...]
    warnings: tuple[str, ...]
    subdivision: CollisionSubdivisionStats = CollisionSubdivisionStats()


ConditionKey = tuple[str, str]


@dataclass(frozen=True)
class CollisionProfileAssignment:
    profile_name: str
    surface_id: int


@dataclass(frozen=True)
class AllConditionsCollision:
    """COL geometry and per-condition physical-material remaps.

    Each part uses a generated profile name. ``condition_remaps`` resolves that
    name to a canonical surface ID for a ``(surface, age)`` MAT condition.
    """

    parts: tuple[MeshPart, ...]
    condition_remaps: Mapping[
        ConditionKey, tuple[CollisionProfileAssignment, ...]
    ]
    warnings: tuple[str, ...]
    subdivision: CollisionSubdivisionStats = CollisionSubdivisionStats()


@dataclass(frozen=True)
class PreparedOriginalVariant:
    stage: RbrStage
    render_parts: tuple[MeshPart, ...]
    collision_parts: tuple[MeshPart, ...]
    materials: Mapping[str, MaterialVariant]
    bounds_min: tuple[float, float, float]
    bounds_max: tuple[float, float, float]
    condition_remaps: Mapping[
        ConditionKey, tuple[CollisionProfileAssignment, ...]
    ] = field(default_factory=dict)
    condition_ground_types: Mapping[ConditionKey, Mapping[str, str]] = (
        field(default_factory=dict)
    )
    collision_subdivision: CollisionSubdivisionStats = (
        CollisionSubdivisionStats()
    )
    road_condition: ConditionKey = DEFAULT_ROAD_CONDITION


class _VisualMaterialRegistry:
    def __init__(self) -> None:
        self._by_signature: dict[tuple[object, ...], VisualMaterialSpec] = {}
        self.specs: list[VisualMaterialSpec] = []

    def register(
        self,
        *,
        source_kind: str,
        diffuse_1: int | None,
        diffuse_2: int | None,
        specular: int | None,
        shadow: int | None,
        render_flags: int,
        water: bool,
        glossiness: float | None = None,
    ) -> VisualMaterialSpec:
        signature = (
            source_kind,
            diffuse_1,
            diffuse_2,
            specular,
            shadow,
            render_flags,
            water,
            glossiness,
        )
        existing = self._by_signature.get(signature)
        if existing is not None:
            return existing
        index = len(self.specs)
        association = "_".join(
            (
                f"d{diffuse_1}" if diffuse_1 is not None else "dn",
                f"d2{diffuse_2}" if diffuse_2 is not None else "d2n",
                f"s{specular}" if specular is not None else "sn",
                f"h{shadow}" if shadow is not None else "hn",
            )
        )
        name = f"original_visual_{index:04d}_{source_kind}_{association}"
        spec = VisualMaterialSpec(
            name=name,
            material_index=index,
            source_kind=source_kind,
            diffuse_1=diffuse_1,
            diffuse_2=diffuse_2,
            specular=specular,
            shadow=shadow,
            render_flags=render_flags,
            water=water,
            glossiness=glossiness,
        )
        self._by_signature[signature] = spec
        self.specs.append(spec)
        return spec


def _require_finite(values: np.ndarray, context: str) -> None:
    if not np.isfinite(values).all():
        raise OriginalAdapterError(f"{context} contains non-finite values")


def _structured_vector(values: np.ndarray, field: str) -> np.ndarray:
    names = values.dtype.names
    if names is None or field not in names:
        raise OriginalAdapterError(f"Original vertices have no {field!r} field")
    vectors = values[field]
    vector_names = vectors.dtype.names
    if vector_names is None or not {"x", "y", "z"}.issubset(vector_names):
        raise OriginalAdapterError(f"Original {field!r} field is not a 3D vector")
    result = np.column_stack((vectors["x"], vectors["y"], vectors["z"])).astype(
        np.float32,
        copy=False,
    )
    _require_finite(result, field)
    return result


_BAKED_SHADE_REFERENCE = 0.33
_BAKED_SHADE_CONTRAST = 0.5
_LUMINANCE_WEIGHTS = np.asarray((0.2126, 0.7152, 0.0722), dtype=np.float32)


def _soften_baked_shade(colors: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Scale baked colours, then compress shade darker than the reference.

    Baked colours hold canopy and terrain shade that BeamNG neither applies to
    the car nor needs on top of its own shadows, so below the reference their
    luminance keeps only part of its contrast (in log terms). Hue and alpha
    are kept.
    """
    result = colors.copy()
    result[:, :3] *= scale
    luminance = result[:, :3] @ _LUMINANCE_WEIGHTS
    shaded = (luminance > 0.0) & (luminance < _BAKED_SHADE_REFERENCE)
    lifted = _BAKED_SHADE_REFERENCE * (luminance[shaded] / _BAKED_SHADE_REFERENCE) ** _BAKED_SHADE_CONTRAST
    result[shaded, :3] = np.minimum(result[shaded, :3] * (lifted / luminance[shaded])[:, None], 1.0)
    return result


def _baked_color_scale(colors: np.ndarray) -> float:
    """Scale moving one kind of baked colour's median toward the reference.

    Stage tools bake at very different levels, from about 0.2 to pure white
    for open ground. Under BeamNG's auto exposure a bright world makes the
    car look dim, so a kind brighter than the reference keeps only part of
    its ratio to it, like shade below the reference does.
    """
    if not len(colors):
        return 1.0
    median = float(np.median(colors[:, :3] @ _LUMINANCE_WEIGHTS))
    if median <= _BAKED_SHADE_REFERENCE:
        return 1.0
    return (_BAKED_SHADE_REFERENCE / median) ** (1.0 - _BAKED_SHADE_CONTRAST)


def _parts_color_scale(parts: Sequence[MeshPart]) -> float:
    shown = [part.colors for part in parts if part.lod_kind != "far"]
    return _baked_color_scale(np.concatenate(shown)) if shown else 1.0


def _bake_colors(parts: Sequence[MeshPart], scale: float) -> list[MeshPart]:
    return [replace(part, colors=_soften_baked_shade(part.colors, scale)) for part in parts]


def _colors(values: np.ndarray) -> np.ndarray:
    names = values.dtype.names
    if names is None or "color" not in names:
        return np.ones((len(values), 4), dtype=np.float32)
    color = values["color"]
    color_names = color.dtype.names
    if color_names is None or not {"r", "g", "b", "a"}.issubset(color_names):
        raise OriginalAdapterError("Original color field is malformed")
    return (
        np.column_stack((color["r"], color["g"], color["b"], color["a"]))
        .astype(np.float32)
        / 255.0
    )


def _texcoord_sets(
    values: np.ndarray,
) -> tuple[tuple[np.ndarray, ...], dict[str, int]]:
    names = values.dtype.names or ()
    result: list[np.ndarray] = []
    semantic_indices: dict[str, int] = {}
    for semantic, field in (
        ("diffuse_1", "diffuse_1_uv"),
        ("diffuse_2", "diffuse_2_uv"),
        ("specular", "specular_uv"),
        ("shadow", "shadow_uv"),
    ):
        if field not in names:
            continue
        uv = values[field]
        uv_names = uv.dtype.names
        if uv_names is None or not {"u", "v"}.issubset(uv_names):
            raise OriginalAdapterError(f"Original UV field {field!r} is malformed")
        channel = np.column_stack((uv["u"], uv["v"])).astype(np.float32, copy=False)
        channel[:, 1] = 1.0 - channel[:, 1]
        invalid = ~np.isfinite(channel).all(axis=1)
        if invalid.any():
            warnings.warn(
                f"Recovered {int(invalid.sum())} vertices with non-finite "
                f"{field} as 0",
                RuntimeWarning,
                stacklevel=2,
            )
            channel[invalid] = 0.0
        semantic_indices[semantic] = len(result)
        result.append(channel)
    if not result:
        result.append(np.zeros((len(values), 2), dtype=np.float32))
    return tuple(result), semantic_indices


def _specular_strengths(values: np.ndarray) -> np.ndarray | None:
    if "specular_strength" not in (values.dtype.names or ()):
        return None
    result = np.asarray(values["specular_strength"], dtype=np.float32).copy()
    _require_finite(result, "specular_strength")
    return result


def _generated_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros(vertices.shape, dtype=np.float64)
    if len(faces):
        triangles = vertices[faces].astype(np.float64)
        face_normals = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        for corner in range(3):
            np.add.at(normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-12
    normals[valid] /= lengths[valid, None]
    normals[~valid, 2] = 1.0
    return normals.astype(np.float32)


def _triangle_soup_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """``_generated_normals`` for faces that use every vertex exactly once."""
    triangles = vertices[faces].astype(np.float64)
    face_normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    # Accumulating onto zeros turns -0.0 into 0.0.
    face_normals += 0.0
    lengths = np.linalg.norm(face_normals, axis=1)
    valid = lengths > 1e-12
    face_normals[valid] /= lengths[valid, None]
    face_normals[~valid, 2] = 1.0
    normals = np.empty(vertices.shape, dtype=np.float32)
    normals[faces] = face_normals.astype(np.float32)[:, None, :]
    return normals


def _position_smoothed_normals(parts: Sequence[MeshPart]) -> list[np.ndarray]:
    """Area-weighted normals over every face touching a position, per part.

    A vertex keeps its own normal where the faces around it cancel out or
    point away from it, as back-to-back faces do.
    """
    vertices = np.concatenate([part.vertices for part in parts]).astype(np.float64)
    starts = np.cumsum([0, *(len(part.vertices) for part in parts)])
    faces = np.concatenate(
        [
            np.asarray(part.faces, dtype=np.int64) + start
            for part, start in zip(parts, starts)
        ]
    )
    positions, welded = np.unique(
        np.round(vertices * 1000.0).astype(np.int64),
        axis=0,
        return_inverse=True,
    )
    welded = welded.reshape(-1)
    triangles = vertices[faces]
    face_normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    sums = np.zeros((len(positions), 3), dtype=np.float64)
    for corner in range(3):
        np.add.at(sums, welded[faces[:, corner]], face_normals)
    normals = sums[welded]
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-12
    normals[valid] /= lengths[valid, None]
    own = np.concatenate([part.normals for part in parts]).astype(np.float64)
    keep = ~valid | (np.einsum("ij,ij->i", normals, own) <= 0.0)
    normals[keep] = own[keep]
    normals = normals.astype(np.float32)
    return [normals[start:end] for start, end in zip(starts[:-1], starts[1:])]


def _smooth_generated_normals(
    parts: Sequence[MeshPart],
    generated: Sequence[bool],
) -> list[MeshPart]:
    """Smooth generated geom normals across vertex splits, chunks and blocks.

    Geom buffers without normals give each quad its own vertices for its UVs;
    their baked vertex lighting, and the normals of buffers that have them,
    are continuous across those splits. Near and far detail are smoothed
    separately.
    """
    result = list(parts)
    for excluded_kind, target_kinds in (("far", ("near", "any")), ("near", ("far",))):
        members = [
            index for index, part in enumerate(parts) if part.lod_kind != excluded_kind
        ]
        targets = {
            index
            for index in members
            if generated[index] and parts[index].lod_kind in target_kinds
        }
        if not targets:
            continue
        smoothed = _position_smoothed_normals([parts[index] for index in members])
        for index, normals in zip(members, smoothed):
            if index in targets:
                result[index] = replace(parts[index], normals=normals)
    return result


def adapt_brake_wall_collision(
    brake_wall: BrakeWall,
    *,
    material_index: int,
    collision_parts: Sequence[MeshPart],
    vertical_bounds: tuple[float, float],
    origin: np.ndarray | None = None,
    max_triangle_diagonal: float = 5.0,
    ground_depth: float = 2.0,
    ground_clearance: float = 10.0,
) -> MeshPart | None:
    """Build the inward-facing wall along the outer brake-wall boundary.

    Each segment spans the Z range of the collision triangles whose XY bounds
    overlap its inner/outer brake zone, extended by ``ground_depth`` below and
    ``ground_clearance`` above. Segments without such triangles span
    ``vertical_bounds``. ``collision_parts`` must already be in the
    origin-relative output frame.
    """
    if not brake_wall.segments:
        return None
    inner = np.asarray(brake_wall.inner_points, dtype=np.float64).copy()
    outer = np.asarray(brake_wall.outer_points, dtype=np.float64).copy()
    stage_origin = (
        np.zeros(3, dtype=np.float64)
        if origin is None
        else np.asarray(origin, dtype=np.float64)
    )
    if stage_origin.shape != (3,) or not np.isfinite(stage_origin).all():
        raise OriginalAdapterError("Original brake-wall origin is invalid")
    vertical_range = np.asarray(vertical_bounds, dtype=np.float64)
    if (
        vertical_range.shape != (2,)
        or not np.isfinite(vertical_range).all()
        or vertical_range[1] <= vertical_range[0]
        or not math.isfinite(max_triangle_diagonal)
        or max_triangle_diagonal <= 0.0
        or not math.isfinite(ground_depth)
        or ground_depth < 0.0
        or not math.isfinite(ground_clearance)
        or ground_clearance <= 0.0
    ):
        raise OriginalAdapterError("Original brake-wall dimensions are invalid")
    if not np.isfinite(inner).all() or not np.isfinite(outer).all():
        raise OriginalAdapterError("Original brake-wall points are invalid")
    inner -= stage_origin[:2]
    outer -= stage_origin[:2]

    triangle_lows: list[np.ndarray] = []
    triangle_highs: list[np.ndarray] = []
    for part in collision_parts:
        if not part.collision_eligible or not len(part.faces):
            continue
        triangles = np.asarray(part.vertices, dtype=np.float32)[
            np.asarray(part.faces, dtype=np.int64)
        ]
        triangle_lows.append(triangles.min(axis=1))
        triangle_highs.append(triangles.max(axis=1))
    triangle_low = (
        np.concatenate(triangle_lows)
        if triangle_lows
        else np.empty((0, 3), dtype=np.float32)
    )
    triangle_high = (
        np.concatenate(triangle_highs)
        if triangle_highs
        else np.empty((0, 3), dtype=np.float32)
    )
    order = np.argsort(triangle_low[:, 0], kind="stable")
    triangle_low = triangle_low[order]
    triangle_high = triangle_high[order]
    # searchsorted converts the whole array to the float64 key dtype on every call.
    sorted_low_x = triangle_low[:, 0].astype(np.float64)
    widest_triangle = (
        float(np.max(triangle_high[:, 0] - triangle_low[:, 0]))
        if len(order)
        else 0.0
    )

    def segment_band(segment_index: int) -> tuple[float, float]:
        corners = np.concatenate(
            (
                inner[segment_index : segment_index + 2],
                outer[segment_index : segment_index + 2],
            )
        )
        zone_low = corners.min(axis=0)
        zone_high = corners.max(axis=0)
        start = np.searchsorted(
            sorted_low_x,
            zone_low[0] - widest_triangle,
            side="left",
        )
        end = np.searchsorted(sorted_low_x, zone_high[0], side="right")
        low = triangle_low[start:end]
        high = triangle_high[start:end]
        overlapping = (
            (high[:, 0] >= zone_low[0])
            & (low[:, 1] <= zone_high[1])
            & (high[:, 1] >= zone_low[1])
        )
        if not np.any(overlapping):
            return float(vertical_range[0]), float(vertical_range[1])
        return (
            float(np.min(low[overlapping, 2])) - ground_depth,
            float(np.max(high[overlapping, 2])) + ground_clearance,
        )

    def horizontal_piece_count(length: float, height: float, rows: int) -> int:
        maximum_horizontal_length = math.sqrt(
            max_triangle_diagonal**2 - (height / rows) ** 2
        )
        return max(1, math.ceil(length / maximum_horizontal_length))

    segment_indices = np.asarray(brake_wall.segments, dtype=np.int64)
    segment_lengths = np.linalg.norm(
        outer[segment_indices + 1] - outer[segment_indices],
        axis=1,
    )
    valid_segments = segment_lengths > 1e-6
    segment_indices = segment_indices[valid_segments]
    segment_lengths = segment_lengths[valid_segments]

    vertices: list[tuple[float, float, float]] = []
    faces: list[tuple[int, int, int]] = []
    previous_segment_index: int | None = None
    previous_heights: list[float] = []
    for segment_index, segment_length in zip(
        segment_indices.tolist(),
        segment_lengths.tolist(),
    ):
        bottom, top = segment_band(segment_index)
        wall_height = top - bottom
        # Fewer rows leave less horizontal room per tile; beyond square tiles
        # more rows only add faces.
        vertical_piece_count = min(
            range(
                math.floor(wall_height / max_triangle_diagonal) + 1,
                math.ceil(wall_height * math.sqrt(2.0) / max_triangle_diagonal)
                + 1,
            ),
            key=lambda rows: rows
            * horizontal_piece_count(segment_length, wall_height, rows),
        )
        horizontal_pieces = horizontal_piece_count(
            segment_length,
            wall_height,
            vertical_piece_count,
        )
        heights = [
            bottom + wall_height * (index / vertical_piece_count)
            for index in range(vertical_piece_count + 1)
        ]
        column_size = len(heights)
        segment_start = outer[segment_index]
        segment_vector = outer[segment_index + 1] - segment_start
        inner_start = inner[segment_index]
        inner_vector = inner[segment_index + 1] - inner_start
        segment_normal = np.asarray((-segment_vector[1], segment_vector[0]))
        if (
            previous_segment_index != segment_index - 1
            or heights != previous_heights
        ):
            vertices.extend(
                (segment_start[0], segment_start[1], height) for height in heights
            )
        for horizontal_index in range(horizontal_pieces):
            end_fraction = (horizontal_index + 1) / horizontal_pieces
            middle_fraction = (horizontal_index + 0.5) / horizontal_pieces
            first_column = len(vertices) - column_size
            second_column = len(vertices)
            second_outer = segment_start + segment_vector * end_fraction
            vertices.extend(
                (second_outer[0], second_outer[1], height) for height in heights
            )
            inner_direction = (inner_start + inner_vector * middle_fraction) - (
                segment_start + segment_vector * middle_fraction
            )
            faces_forward = float(np.dot(segment_normal, inner_direction)) < 0.0
            for vertical_index in range(vertical_piece_count):
                first_low = first_column + vertical_index
                second_low = second_column + vertical_index
                if faces_forward:
                    faces.extend(
                        (
                            (first_low, second_low, second_low + 1),
                            (first_low, second_low + 1, first_low + 1),
                        )
                    )
                else:
                    faces.extend(
                        (
                            (first_low, second_low + 1, second_low),
                            (first_low, first_low + 1, second_low + 1),
                        )
                    )
        previous_segment_index = segment_index
        previous_heights = heights

    if not faces:
        return None
    vertex_values = np.asarray(vertices, dtype=np.float32)
    face_values = np.asarray(faces, dtype=np.uint32)
    texcoords = np.zeros((len(vertex_values), 2), dtype=np.float32)
    return MeshPart(
        name="original_brake_wall_collision",
        vertices=vertex_values,
        faces=face_values,
        normals=_generated_normals(vertex_values, face_values),
        texcoords=texcoords,
        colors=np.ones((len(vertex_values), 4), dtype=np.float32),
        material_index=material_index,
        material_name="original_brake_wall",
        collision_eligible=True,
        water=False,
        texcoord_sets=(texcoords,),
    )


def brake_wall_material_variant(material_index: int) -> MaterialVariant:
    material = RbrMaterial(
        index=material_index,
        name="original_brake_wall",
        effect="RBR_Original_Physics",
        technique="Default",
        diffuse_texture=None,
        second_diffuse_texture=None,
        normal_texture=None,
        specular_texture=None,
        properties={
            "originalSourceKind": "brakeWall",
            "targetAdaptation": "finiteBrakeWallFacade",
        },
    )
    return MaterialVariant(
        material=material,
        ground_type="ROCK",
        hard=True,
        water=False,
        bendable=False,
        source_surface_ids=(),
    )


def adapt_fences(
    fnc: FncFile,
    texture_paths: Mapping[int, Path],
    *,
    material_index_start: int,
    warnings: list[str] | None = None,
) -> tuple[tuple[MeshPart, ...], dict[str, MaterialVariant]]:
    variants: dict[str, MaterialVariant] = {}
    material_names: dict[int, str] = {}
    for texture_index, texture_path in sorted(texture_paths.items()):
        material_name = f"original_fence_texture_{texture_index:03d}"
        material_names[texture_index] = material_name
        transparent = texture_has_alpha(texture_path)
        material = RbrMaterial(
            index=material_index_start + texture_index,
            name=material_name,
            effect="RBR_Fence",
            technique="Transparent" if transparent else "Default",
            diffuse_texture=texture_path,
            second_diffuse_texture=None,
            normal_texture=None,
            specular_texture=None,
            properties={"originalSourceKind": "fence"},
            double_sided=True,
            cutout_alpha_ref=_OPAQUE_TEXEL_ALPHA_REF if transparent else None,
            blend_alpha_ref=_FENCE_BLEND_ALPHA_REF if transparent else None,
        )
        variants[material_name] = MaterialVariant(
            material=material,
            ground_type=_RENDER_ONLY_GROUND_TYPE,
            hard=False,
            water=False,
            bendable=False,
            source_surface_ids=(),
            pbr_override=PbrMaterialOverride(
                base_color_texture=texture_path,
                base_vertex_color=True,
            ),
        )

    parts: list[MeshPart] = []
    up = np.asarray((0.0, 0.0, 1.0), dtype=np.float32)
    color_scale = _baked_color_scale(
        np.asarray(
            [post.color for fence in fnc.fences for post in fence.posts],
            dtype=np.float32,
        ).reshape((-1, 4))
        / 255.0
    )
    for fence_index, fence in enumerate(fnc.fences):
        tile_definition = fence_render_definition(fence.tile_type)
        pole_definition = fence_render_definition(fence.pole_type)
        if tile_definition is None or pole_definition is None:
            if warnings is not None:
                warnings.append(
                    f"Original FNC fence {fence_index} uses unsupported "
                    f"tile/pole selectors {fence.tile_type}/{fence.pole_type}"
                )
            continue
        positions = np.asarray(
            [
                _LH_TO_Z_UP
                @ np.asarray(post.position, dtype=np.float64)
                for post in fence.posts
            ],
            dtype=np.float32,
        )
        colors = _soften_baked_shade(
            np.asarray(
                [
                    tuple(channel / 255.0 for channel in post.color)
                    for post in fence.posts
                ],
                dtype=np.float32,
            ),
            color_scale,
        )

        tile_vertices: list[np.ndarray] = []
        tile_faces: list[tuple[int, int, int]] = []
        tile_uvs: list[tuple[float, float]] = []
        tile_colors: list[np.ndarray] = []
        for post_index in range(len(positions) - 1):
            first = positions[post_index]
            second = positions[post_index + 1]
            if float(np.linalg.norm(second - first)) <= 1e-5:
                continue
            tile_top = 1.0
            for _strip in range(tile_definition.strip_count):
                tile_bottom = tile_top - tile_definition.strip_height
                base = len(tile_vertices)
                tile_vertices.extend(
                    (
                        first + up * tile_bottom,
                        second + up * tile_bottom,
                        first + up * tile_top,
                        second + up * tile_top,
                    )
                )
                tile_faces.extend(
                    ((base, base + 1, base + 2), (base + 2, base + 1, base + 3))
                )
                tile_uvs.extend(
                    (
                        (tile_definition.u_minimum, tile_definition.v_minimum),
                        (tile_definition.u_maximum, tile_definition.v_minimum),
                        (tile_definition.u_minimum, tile_definition.v_maximum),
                        (tile_definition.u_maximum, tile_definition.v_maximum),
                    )
                )
                tile_colors.extend(
                    (
                        colors[post_index],
                        colors[post_index + 1],
                        colors[post_index],
                        colors[post_index + 1],
                    )
                )
                tile_top = tile_bottom - tile_definition.strip_gap
        if tile_faces:
            vertices = np.asarray(tile_vertices, dtype=np.float32)
            faces = np.asarray(tile_faces, dtype=np.uint32)
            texcoords = np.asarray(tile_uvs, dtype=np.float32)
            parts.append(
                MeshPart(
                    name=f"original_fence_{fence_index:03d}_tape",
                    vertices=vertices,
                    faces=faces,
                    normals=_generated_normals(vertices, faces),
                    texcoords=texcoords,
                    colors=np.asarray(tile_colors, dtype=np.float32),
                    material_index=material_index_start + fence.tile_texture_index,
                    material_name=material_names[fence.tile_texture_index],
                    texcoord_sets=(texcoords,),
                    force_color_stream=True,
                )
            )

        pole_vertices: list[np.ndarray] = []
        pole_faces: list[tuple[int, int, int]] = []
        pole_uvs: list[tuple[float, float]] = []
        pole_colors: list[np.ndarray] = []
        for post_index, position in enumerate(positions):
            bottom = position
            top = position + up * 1.1
            corners = (
                np.asarray((-0.015, -0.015, 0.0), dtype=np.float32),
                np.asarray((0.015, -0.015, 0.0), dtype=np.float32),
                np.asarray((0.015, 0.015, 0.0), dtype=np.float32),
                np.asarray((-0.015, 0.015, 0.0), dtype=np.float32),
            )
            faces = (
                (0, 1, 5, 4),
                (1, 2, 6, 5),
                (2, 3, 7, 6),
                (3, 0, 4, 7),
                (4, 5, 6, 7),
            )
            for first_index, second_index, third_index, fourth_index in faces:
                base = len(pole_vertices)
                pole_vertices.extend(
                    (
                        (bottom if first_index < 4 else top) + corners[first_index % 4],
                        (bottom if second_index < 4 else top) + corners[second_index % 4],
                        (bottom if third_index < 4 else top) + corners[third_index % 4],
                        (bottom if fourth_index < 4 else top) + corners[fourth_index % 4],
                    )
                )
                pole_faces.extend(
                    ((base, base + 1, base + 2), (base + 2, base + 1, base + 3))
                )
                pole_uvs.extend(
                    (
                        (pole_definition.u_minimum, pole_definition.v_minimum),
                        (pole_definition.u_maximum, pole_definition.v_minimum),
                        (pole_definition.u_minimum, pole_definition.v_maximum),
                        (pole_definition.u_maximum, pole_definition.v_maximum),
                    )
                )
                pole_colors.extend((colors[post_index],) * 4)
        if pole_faces:
            vertices = np.asarray(pole_vertices, dtype=np.float32)
            faces = np.asarray(pole_faces, dtype=np.uint32)
            texcoords = np.asarray(pole_uvs, dtype=np.float32)
            parts.append(
                MeshPart(
                    name=f"original_fence_{fence_index:03d}_poles",
                    vertices=vertices,
                    faces=faces,
                    normals=_generated_normals(vertices, faces),
                    texcoords=texcoords,
                    colors=np.asarray(pole_colors, dtype=np.float32),
                    material_index=material_index_start + fence.pole_texture_index,
                    material_name=material_names[fence.pole_texture_index],
                    texcoord_sets=(texcoords,),
                    force_color_stream=True,
                )
            )
    return tuple(parts), variants


def _original_mesh_arrays(
    values: np.ndarray,
    faces: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[np.ndarray, ...],
    dict[str, int],
    np.ndarray | None,
]:
    vertices = _structured_vector(values, "position")
    faces = np.asarray(faces, dtype=np.int64).reshape((-1, 3))
    if len(faces):
        if int(np.min(faces)) < 0 or int(np.max(faces)) >= len(vertices):
            raise OriginalAdapterError("Original mesh face index is out of bounds")
        # Original visual buffers are left-handed. Swapping Y/Z while reading
        # them changes handedness, so reverse winding for right-handed Z-up.
        faces = faces[:, (0, 2, 1)]
    faces = faces.astype(np.uint32, copy=False)
    names = values.dtype.names or ()
    if "normal" in names:
        normals = _structured_vector(values, "normal")
        lengths = np.linalg.norm(normals, axis=1)
        valid = lengths > 1e-12
        normals = normals.copy()
        normals[valid] /= lengths[valid, None]
        normals[~valid] = _generated_normals(vertices, faces)[~valid]
    else:
        normals = _generated_normals(vertices, faces)
    colors = _colors(values)
    texcoords, semantic_uv_indices = _texcoord_sets(values)
    return (
        vertices,
        faces,
        normals,
        colors,
        texcoords,
        semantic_uv_indices,
        _specular_strengths(values),
    )


def _matrix_to_z_up(raw: tuple[float, ...]) -> np.ndarray:
    if len(raw) != 16:
        raise OriginalAdapterError(f"Expected 16 matrix values, got {len(raw)}")
    row_major_lh = np.asarray(raw, dtype=np.float64).reshape((4, 4))
    _require_finite(row_major_lh, "interactive instance matrix")
    column_lh = row_major_lh.T
    matrix = _LH_TO_Z_UP_4 @ column_lh @ _LH_TO_Z_UP_4
    if not np.allclose(matrix[3], (0.0, 0.0, 0.0, 1.0), atol=1e-5):
        raise OriginalAdapterError("Interactive instance matrix is not affine")
    return matrix


def _basis_is_singular(basis: np.ndarray) -> bool:
    return bool(np.linalg.matrix_rank(basis) < 3)


def _flatten_instances(
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    colors: np.ndarray,
    texcoords: tuple[np.ndarray, ...],
    specular_strengths: np.ndarray | None,
    matrices: Iterable[tuple[float, ...]],
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[np.ndarray, ...],
    np.ndarray | None,
]:
    out_vertices: list[np.ndarray] = []
    out_faces: list[np.ndarray] = []
    out_normals: list[np.ndarray] = []
    out_colors: list[np.ndarray] = []
    out_texcoords: list[list[np.ndarray]] = [[] for _ in texcoords]
    out_specular_strengths: list[np.ndarray] = []
    vertex_offset = 0
    for raw_matrix in matrices:
        matrix = _matrix_to_z_up(raw_matrix)
        basis = matrix[:3, :3]
        transformed_vertices = (
            np.asarray(vertices, dtype=np.float64) @ basis.T + matrix[:3, 3]
        ).astype(np.float32)
        determinant = np.linalg.det(basis)
        instance_faces = faces
        if determinant < 0:
            instance_faces = instance_faces[:, (0, 2, 1)]
        if _basis_is_singular(basis):
            transformed_normals = _generated_normals(
                transformed_vertices,
                instance_faces,
            )
        else:
            normal_matrix = np.linalg.inv(basis).T
            transformed_normals = (
                np.asarray(normals, dtype=np.float64) @ normal_matrix.T
            )
            lengths = np.linalg.norm(transformed_normals, axis=1)
            valid = lengths > 1e-12
            transformed_normals[valid] /= lengths[valid, None]
            transformed_normals[~valid] = (0.0, 0.0, 1.0)
        out_vertices.append(transformed_vertices)
        out_faces.append(instance_faces + vertex_offset)
        out_normals.append(transformed_normals.astype(np.float32))
        out_colors.append(colors)
        if specular_strengths is not None:
            out_specular_strengths.append(specular_strengths)
        for index, channel in enumerate(texcoords):
            out_texcoords[index].append(channel)
        vertex_offset += len(vertices)
    if not out_vertices:
        return (
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.uint32),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 4), dtype=np.float32),
            tuple(np.empty((0, 2), dtype=np.float32) for _ in texcoords),
            (
                np.empty(0, dtype=np.float32)
                if specular_strengths is not None
                else None
            ),
        )
    return (
        np.concatenate(out_vertices),
        np.concatenate(out_faces).astype(np.uint32, copy=False),
        np.concatenate(out_normals),
        np.concatenate(out_colors),
        tuple(np.concatenate(channels) for channels in out_texcoords),
        (
            np.concatenate(out_specular_strengths)
            if out_specular_strengths
            else None
        ),
    )


def _make_part(
    *,
    name: str,
    values: np.ndarray,
    faces: np.ndarray,
    material: VisualMaterialSpec,
    translation: tuple[float, float, float] | None = None,
    matrices: Iterable[tuple[float, ...]] | None = None,
    lod_group: str | None = None,
    lod_kind: str = "any",
) -> MeshPart | None:
    if not len(values) or not len(faces):
        return None
    (
        vertices,
        faces,
        normals,
        colors,
        texcoord_sets,
        semantic_uv_indices,
        specular_strengths,
    ) = _original_mesh_arrays(values, faces)
    if translation is not None:
        offset = np.asarray(translation, dtype=np.float32)
        _require_finite(offset, f"{name} translation")
        vertices = vertices + offset
    if matrices is not None:
        (
            vertices,
            faces,
            normals,
            colors,
            texcoord_sets,
            specular_strengths,
        ) = _flatten_instances(
            vertices,
            faces,
            normals,
            colors,
            texcoord_sets,
            specular_strengths,
            matrices,
        )
        if not len(faces):
            return None
    _require_finite(vertices, f"{name} vertices")
    _require_finite(normals, f"{name} normals")
    texcoords = texcoord_sets[0]
    return MeshPart(
        name=name,
        vertices=vertices,
        faces=faces,
        normals=normals,
        texcoords=texcoords,
        colors=colors,
        material_index=material.material_index,
        material_name=material.name,
        collision_eligible=False,
        water=material.water,
        texcoord_sets=texcoord_sets,
        semantic_uv_indices=semantic_uv_indices,
        specular_strengths=specular_strengths,
        lod_group=lod_group,
        lod_kind=lod_kind,
    )


def _object_material(
    registry: _VisualMaterialRegistry,
    source_kind: str,
    data: ObjectBlock | ObjectData3D,
    *,
    water: bool = False,
) -> VisualMaterialSpec:
    if isinstance(data, ObjectBlock):
        specular = None
        shadow = None
    else:
        specular = data.specular_texture
        shadow = None
    return registry.register(
        source_kind=source_kind,
        diffuse_1=data.texture_1,
        diffuse_2=data.texture_2,
        specular=specular,
        shadow=shadow,
        render_flags=data.render_flags,
        water=water,
    )


def adapt_lbs_meshes(lbs: LbsFile) -> AdaptedVisuals:
    registry = _VisualMaterialRegistry()
    parts: list[MeshPart] = []
    warnings: list[str] = []
    ground_scale = 1.0

    if lbs.geom_blocks is not None:
        geom_parts: list[MeshPart] = []
        generated_normals: list[bool] = []
        for block_index, block in enumerate(lbs.geom_blocks.blocks):
            for chunk_index, chunk in enumerate(block.render_chunks):
                buffer = block.buffers[chunk.render_type]
                values = buffer.vertices[
                    chunk.first_vertex : chunk.first_vertex + chunk.vertex_count
                ]
                faces = (
                    np.asarray(
                        buffer.triangles[
                            chunk.first_triangle : chunk.first_triangle
                            + chunk.triangle_count
                        ],
                        dtype=np.int64,
                    )
                    - chunk.first_vertex
                )
                material = registry.register(
                    source_kind="geom",
                    diffuse_1=chunk.texture_1,
                    diffuse_2=chunk.texture_2,
                    specular=chunk.specular_texture,
                    shadow=chunk.shadow_texture,
                    render_flags=0,
                    water=False,
                    glossiness=lbs.geom_blocks.glossiness,
                )
                part = _make_part(
                    name=f"original_geom_{block_index}_{chunk_index}",
                    values=values,
                    faces=faces,
                    material=material,
                    lod_group=f"original_block_{block_index}",
                    lod_kind={
                        1: "near",
                        2: "any",
                        3: "far",
                    }[chunk.distance_class],
                )
                if part is not None:
                    geom_parts.append(part)
                    generated_normals.append("normal" not in (values.dtype.names or ()))
        ground_scale = _parts_color_scale(geom_parts)
        parts.extend(_bake_colors(_smooth_generated_normals(geom_parts, generated_normals), ground_scale))

    if lbs.object_blocks is not None:
        object_parts: list[MeshPart] = []
        for group_index, group in enumerate(lbs.object_blocks):
            if group is None:
                continue
            for label, blocks in (("primary", group.primary), ("secondary", group.secondary)):
                for block_index, block in enumerate(blocks):
                    name = f"original_object_{group_index}_{label}_{block_index}"
                    material = _object_material(registry, "object", block)
                    has_far_lod = bool(
                        block.far_triangles is not None
                        and len(block.far_triangles)
                    )
                    part = _make_part(
                        name=name,
                        values=block.vertices,
                        faces=block.main_triangles,
                        material=material,
                        lod_group=f"original_block_{group_index}",
                        lod_kind="near" if has_far_lod else "any",
                    )
                    if part is not None:
                        object_parts.append(part)
                    if has_far_lod:
                        far_part = _make_part(
                            name=name,
                            values=block.vertices,
                            faces=block.far_triangles,
                            material=material,
                            lod_group=f"original_block_{group_index}",
                            lod_kind="far",
                        )
                        if far_part is not None:
                            object_parts.append(far_part)
        parts.extend(_bake_colors(object_parts, _parts_color_scale(object_parts)))

    backdrop_parts: list[MeshPart] = []
    for segment_kind in (SUPER_BOWL, REFLECTION_OBJECTS, WATER_OBJECTS):
        for group_index, group in enumerate(lbs.object_data_groups.get(segment_kind, ())):
            source_kind = {
                SUPER_BOWL: "superbowl",
                REFLECTION_OBJECTS: "reflection",
                WATER_OBJECTS: "water",
            }[segment_kind]
            for item_index, item in enumerate(group.items):
                part = _make_part(
                    name=f"original_{source_kind}_{group_index}_{item_index}",
                    values=item.data.vertices,
                    faces=item.data.triangles,
                    material=_object_material(
                        registry,
                        source_kind,
                        item.data,
                        water=segment_kind == WATER_OBJECTS,
                    ),
                )
                if part is not None:
                    backdrop_parts.append(part)
    parts.extend(_bake_colors(backdrop_parts, ground_scale))

    interactive_parts: list[MeshPart] = []
    for group_index, group in enumerate(
        lbs.object_data_groups.get(INTERACTIVE_OBJECTS, ())
    ):
        matrices = tuple(matrix for _key, matrix in group.instances)
        if not matrices and group.items:
            warnings.append(f"Interactive visual {group.name!r} has no instances")
            continue
        singular_instances = sum(
            _basis_is_singular(_matrix_to_z_up(matrix)[:3, :3])
            for matrix in matrices
        )
        if singular_instances:
            warnings.append(
                f"Interactive visual {group.name!r} has {singular_instances} "
                "singular instance transform(s); regenerated mesh normals"
            )
        for instance_index, matrix in enumerate(matrices):
            for item_index, item in enumerate(group.items):
                part = _make_part(
                    name=(
                        f"original_interactive_{group_index}_"
                        f"{instance_index}_{item_index}"
                    ),
                    values=item.data.vertices,
                    faces=item.data.triangles,
                    material=_object_material(
                        registry,
                        "interactive",
                        item.data,
                    ),
                    matrices=(matrix,),
                    lod_group=(
                        f"original_interactive_{group_index}_{instance_index}"
                    ),
                )
                if part is not None:
                    interactive_parts.append(part)
    parts.extend(_bake_colors(interactive_parts, _parts_color_scale(interactive_parts)))
    for part in parts:
        part.force_color_stream = True
    return AdaptedVisuals(tuple(parts), tuple(registry.specs), tuple(warnings))


def select_collision_condition(
    mat: MatFile,
) -> tuple[MaterialCondition | None, str | None]:
    preferences = (
        ("dry", "normal"),
        ("dry", "new"),
        ("dry", "worn"),
    )
    for index, (surface, age) in enumerate(preferences):
        matches = tuple(
            condition
            for condition in mat.conditions
            if condition.surface == surface and condition.age == age
        )
        if matches:
            warning = None
            if index:
                warning = (
                    f"MAT has no dry/normal condition; using "
                    f"{matches[0].identifier!r}"
                )
            return matches[0], warning
    if mat.conditions:
        return (
            mat.conditions[0],
            f"MAT has no dry condition; using {mat.conditions[0].identifier!r}",
        )
    return None, "MAT contains no conditions; using raw collision selectors"


def _leaf_batches(node: CollisionTreeNode):
    if node.triangles is not None:
        yield node.triangles.records
    if node.left is not None:
        yield from _leaf_batches(node.left)
    if node.right is not None:
        yield from _leaf_batches(node.right)


def _resolved_side(
    records: np.ndarray,
    condition: MaterialCondition | None,
    side: int,
    warnings: list[str],
    warned_selectors: set[tuple[int, int]],
) -> np.ndarray:
    map_indices = records[f"material_{side}_id"].astype(np.int64)
    result = np.empty((len(records), 3), dtype=np.int64)
    uv_fields = (
        f"a_material_{side}_uv",
        f"b_material_{side}_uv",
        f"c_material_{side}_uv",
    )
    for map_index in np.unique(map_indices):
        selected = map_indices == map_index
        if condition is None or map_index >= len(condition.maps):
            result[selected, :] = map_index
            warning_key = side, int(map_index)
            if warning_key not in warned_selectors:
                warned_selectors.add(warning_key)
                warnings.append(
                    f"Collision MAT selector {map_index} on side {side} is "
                    "unresolved; using its raw byte as the physical material ID"
                )
            continue
        values = condition.maps[int(map_index)].values
        for corner, field in enumerate(uv_fields):
            # An integer selector lands in the MAT cell of the same index.
            selector = records[field][selected]
            result[selected, corner] = values[
                (selector >> 4).astype(np.intp),
                (selector & 0xF).astype(np.intp),
            ]
    return result


def _dominant_material_ids(
    records: np.ndarray,
    condition: MaterialCondition | None,
    warnings: list[str],
    warned_selectors: set[tuple[int, int]],
) -> np.ndarray:
    first = _resolved_side(records, condition, 1, warnings, warned_selectors)
    second = _resolved_side(records, condition, 2, warnings, warned_selectors)
    second = np.where(second == _PASSTHROUGH_SURFACE_ID, first, second)
    packed = records["blending"].astype(np.uint16)
    blend = np.column_stack(
        (
            packed & 0x1F,
            (packed >> 5) & 0x1F,
            (packed >> 10) & 0x1F,
        )
    )
    chosen = np.where(blend >= 16, second, first)
    a, b, c = chosen[:, 0], chosen[:, 1], chosen[:, 2]
    dominant = np.where((a == b) | (a == c), a, np.where(b == c, b, -1))
    unresolved = dominant < 0
    if np.any(unresolved):
        confidence = np.abs(blend.astype(np.int16) * 2 - 31)
        corner = np.argmax(confidence, axis=1)
        dominant[unresolved] = chosen[
            np.arange(len(chosen))[unresolved],
            corner[unresolved],
        ]
    return dominant.astype(np.uint16)


@dataclass(frozen=True)
class _CollisionTriangleTemplate:
    surface_id: int
    weights: np.ndarray


@dataclass(frozen=True)
class _CachedCollisionTemplate:
    weights: np.ndarray
    surface_groups: tuple[tuple[int, np.ndarray], ...]
    used_fallback: bool
    merged_away: int


_BARYCENTRIC_TRIANGLE = np.identity(3, dtype=np.float64)
_COLLISION_TEMPLATE_CACHE_LIMIT = 262_144
_COLLISION_BATCH_RECORDS = 4096


def _cache_collision_template(
    value: tuple[tuple[_CollisionTriangleTemplate, ...], bool, int],
) -> _CachedCollisionTemplate:
    templates, used_fallback, merged_away = value
    grouped: dict[int, list[int]] = {}
    for index, template in enumerate(templates):
        grouped.setdefault(template.surface_id, []).append(index)
    return _CachedCollisionTemplate(
        weights=np.stack([template.weights for template in templates]),
        surface_groups=tuple(
            (
                surface_id,
                np.asarray(indices, dtype=np.int64),
            )
            for surface_id, indices in grouped.items()
        ),
        used_fallback=used_fallback,
        merged_away=merged_away,
    )

# Clip vertices carry the barycentric weights, the interpolated blend value,
# and the selectors u and v of each blend side. Every field is affine across
# the triangle, so all of them survive clipping.
_BLEND_FIELD = 3
_SIDE_FIELDS = {1: (4, 5), 2: (6, 7)}
_MAP_CELLS = SELECTOR_MAP_CELLS
# NGP 7.5 gives a contact whose second blend side reads this undefined ID the
# first side's ID ("Passthrough"). On the first side it stays undefined.
_PASSTHROUGH_SURFACE_ID = 254


def _condition_surface_ids(condition: MaterialCondition | None) -> set[int]:
    if condition is None:
        return set()
    return {
        int(value)
        for material_map in condition.maps
        for value in np.unique(material_map.values)
    }


def _original_material_regions(
    values: np.ndarray,
    canonical: Mapping[int, int],
) -> tuple[Region, ...]:
    if values.shape != (_MAP_CELLS, _MAP_CELLS):
        raise OriginalAdapterError("Original material maps must contain a 16x16 grid")
    return decompose_regions(
        tuple(
            tuple(canonical.get(int(value), int(value)) for value in row)
            for row in values
        )
    )


def _clip_polygon_to_regions(
    polygon: Sequence[Vertex],
    regions: Sequence[Region],
    side: int,
) -> list[tuple[Hashable, list[Vertex]]]:
    return clip_polygon_to_selector_regions(polygon, regions, *_SIDE_FIELDS[side])


def _collision_template_key(record: np.void) -> tuple[int, ...]:
    return (
        int(record["material_1_id"]),
        int(record["material_2_id"]),
        *(
            int(record[f"{name}_material_{side}_uv"])
            for side in (1, 2)
            for name in ("a", "b", "c")
        ),
        int(record["blending"]) & 0x7FFF,
    )


def _collision_template_keys(records: np.ndarray) -> np.ndarray:
    """``_collision_template_key`` of every record, one row each."""
    return np.column_stack(
        (
            records["material_1_id"],
            records["material_2_id"],
            *(
                records[f"{name}_material_{side}_uv"]
                for side in (1, 2)
                for name in ("a", "b", "c")
            ),
            records["blending"] & 0x7FFF,
        )
    ).astype(np.int64)


def _collision_clip_triangle(key: Sequence[int]) -> tuple[Vertex, ...]:
    """Clip vertices of a ``_collision_template_key``.

    Each corner's (u, v) selectors are the low and high nibble of its byte.
    """
    blending = key[8]
    return tuple(
        (
            1.0 if corner == 0 else 0.0,
            1.0 if corner == 1 else 0.0,
            1.0 if corner == 2 else 0.0,
            float((blending >> 5 * corner) & 0x1F),
            float(key[2 + corner] & 0xF),
            float(key[2 + corner] >> 4),
            float(key[5 + corner] & 0xF),
            float(key[5 + corner] >> 4),
        )
        for corner in range(3)
    )


def _triangulate_collision_polygon(
    surface_id: int,
    polygon: Sequence[Vertex],
) -> list[_CollisionTriangleTemplate]:
    return [
        _CollisionTriangleTemplate(
            surface_id,
            np.asarray(
                [corner[:3] for corner in corners],
                dtype=np.float64,
            ),
        )
        for corners in triangulate(polygon)
    ]


def _build_all_conditions_collision_template(
    key: Sequence[int],
    profile_regions: Mapping[int, tuple[Region, ...]],
    missing_map_regions: tuple[Region, ...],
) -> tuple[np.ndarray, tuple[tuple[int, ...], ...]]:
    """Subdivide COL geometry without using any MAT condition's values.

    Returns the barycentric weights of each generated triangle, shaped
    ``(n, 3, 3)``, and each triangle's surface vector.
    """
    triangle = _collision_clip_triangle(key)
    weights: list[list[Vertex]] = []
    vectors: list[tuple[int, ...]] = []
    for side, keep_above in ((1, False), (2, True)):
        polygon = clip_polygon(triangle, _BLEND_FIELD, 15.5, keep_above)
        if polygon_area(polygon) <= AREA_EPSILON:
            continue
        regions = profile_regions.get(key[side - 1], missing_map_regions)
        fragments = _clip_polygon_to_regions(polygon, regions, side)
        if side == 2:
            fragments = _all_conditions_passthrough(
                fragments,
                profile_regions.get(key[0], missing_map_regions),
            )
        for vector, clipped in fragments:
            for corners in triangulate(clipped):
                vectors.append(vector)
                weights.append([corner[:3] for corner in corners])
    return (
        np.asarray(weights, dtype=np.float64).reshape((-1, 3, 3)),
        tuple(vectors),
    )


def _all_conditions_passthrough(
    fragments: list[tuple[Hashable, list[Vertex]]],
    first_regions: tuple[Region, ...],
) -> list[tuple[Hashable, list[Vertex]]]:
    result: list[tuple[Hashable, list[Vertex]]] = []
    for vector, polygon in fragments:
        if _PASSTHROUGH_SURFACE_ID not in vector:
            result.append((vector, polygon))
            continue
        result.extend(
            (
                tuple(
                    first if second == _PASSTHROUGH_SURFACE_ID else second
                    for first, second in zip(first_vector, vector)
                ),
                clipped,
            )
            for first_vector, clipped in _clip_polygon_to_regions(
                polygon,
                first_regions,
                1,
            )
        )
    return result


def _warn_unresolved_collision_selector(
    side: int,
    map_index: int,
    warnings: list[str],
    warned_selectors: set[tuple[int, int]],
) -> None:
    warning_key = side, map_index
    if warning_key in warned_selectors:
        return
    warned_selectors.add(warning_key)
    warnings.append(
        f"Collision MAT selector {map_index} on side {side} is "
        "unresolved; using its raw byte as the physical material ID"
    )


def _build_collision_subdivision_template(
    record: np.void,
    condition: MaterialCondition | None,
    map_regions: dict[int, tuple[Region, ...]],
    canonical: Mapping[int, int],
    warnings: list[str],
    warned_selectors: set[tuple[int, int]],
) -> tuple[tuple[_CollisionTriangleTemplate, ...], bool, int]:
    triangle = _collision_clip_triangle(_collision_template_key(record))

    def side_fragments(
        side: int,
        polygon: Sequence[Vertex],
    ) -> list[tuple[int, Sequence[Vertex]]]:
        map_index = int(record[f"material_{side}_id"])
        if condition is None or map_index >= len(condition.maps):
            _warn_unresolved_collision_selector(
                side,
                map_index,
                warnings,
                warned_selectors,
            )
            return [(canonical.get(map_index, map_index), polygon)]
        regions = map_regions.get(map_index)
        if regions is None:
            regions = _original_material_regions(
                condition.maps[map_index].values,
                canonical,
            )
            map_regions[map_index] = regions
        return _clip_polygon_to_regions(polygon, regions, side)

    fragments: list[tuple[int, Sequence[Vertex]]] = []
    for side, keep_above in ((1, False), (2, True)):
        polygon = clip_polygon(triangle, _BLEND_FIELD, 15.5, keep_above)
        if polygon_area(polygon) <= AREA_EPSILON:
            continue
        for surface_id, fragment in side_fragments(side, polygon):
            if side == 2 and surface_id == _PASSTHROUGH_SURFACE_ID:
                fragments.extend(side_fragments(1, fragment))
            else:
                fragments.append((surface_id, fragment))

    if fragments:
        surface_ids = {surface_id for surface_id, _polygon in fragments}
        if len(surface_ids) == 1:
            return (
                (
                    _CollisionTriangleTemplate(
                        surface_ids.pop(),
                        _BARYCENTRIC_TRIANGLE,
                    ),
                ),
                False,
                0,
            )
        merged = merge_fragments(fragments)
        saved = triangle_cost(
            polygon for _surface_id, polygon in fragments
        ) - triangle_cost(outline for _surface_id, outline in merged)
        templates: list[_CollisionTriangleTemplate] = []
        for surface_id, outline in merged:
            templates.extend(
                _triangulate_collision_polygon(surface_id, outline)
            )
        if templates:
            return tuple(templates), False, saved

    fallback_record = np.empty(1, dtype=record.dtype)
    fallback_record[0] = record
    dominant = int(
        _dominant_material_ids(
            fallback_record,
            condition,
            warnings,
            warned_selectors,
        )[0]
    )
    surface_id = canonical.get(dominant, dominant)
    return (
        (_CollisionTriangleTemplate(surface_id, _BARYCENTRIC_TRIANGLE),),
        True,
        0,
    )


def _append_collision_triangle_batch(
    triangles_by_material: Mapping[int, list[np.ndarray]],
    grouped_vertices: dict[int, list[np.ndarray]],
    grouped_faces: dict[int, list[np.ndarray]],
    grouped_counts: dict[int, int],
) -> None:
    for material_id, triangles in triangles_by_material.items():
        vertices = np.concatenate(triangles).reshape((-1, 3)).astype(
            np.float32,
            copy=False,
        )
        faces = np.arange(len(vertices), dtype=np.uint32).reshape((-1, 3))
        faces = faces[:, (0, 2, 1)]
        base = grouped_counts.get(material_id, 0)
        grouped_vertices.setdefault(material_id, []).append(vertices)
        grouped_faces.setdefault(material_id, []).append(faces + base)
        grouped_counts[material_id] = base + len(vertices)


def _finalize_collision_parts(
    grouped_vertices: Mapping[int, list[np.ndarray]],
    grouped_faces: Mapping[int, list[np.ndarray]],
    surfaces: Mapping[int, RbrSurface] | None,
    name_suffix: str,
) -> tuple[tuple[MeshPart, ...], tuple[int, ...]]:
    surface_ids = tuple(sorted(grouped_faces))
    parts: list[MeshPart] = []
    for material_id in surface_ids:
        vertices = np.concatenate(grouped_vertices[material_id])
        faces = np.concatenate(grouped_faces[material_id]).astype(
            np.uint32,
            copy=False,
        )
        texcoords = np.zeros((len(vertices), 2), dtype=np.float32)
        surface = surfaces.get(material_id) if surfaces is not None else None
        profile = (
            surface.profile
            if surface and surface.profile
            else current_surface_rules().unknown_profile
        )
        water = profile.water
        name = f"original_surface_{material_id:03d}"
        parts.append(
            MeshPart(
                name=f"{name}_{name_suffix}",
                vertices=vertices,
                faces=faces,
                normals=_generated_normals(vertices, faces),
                texcoords=texcoords,
                colors=np.ones((len(vertices), 4), dtype=np.float32),
                material_index=0,
                material_name=name,
                collision_eligible=profile.collision_eligible and not water,
                water=water,
                texcoord_sets=(texcoords,),
            )
        )
    return tuple(parts), surface_ids


def adapt_col_collision(
    col: ColFile,
    mat: MatFile,
    *,
    surfaces: Mapping[int, RbrSurface] | None = None,
    progress: Callable[[str], None] | None = None,
) -> AdaptedCollision:
    condition, condition_warning = select_collision_condition(mat)
    warnings = [condition_warning] if condition_warning else []
    warned_selectors: set[tuple[int, int]] = set()
    grouped_vertices: dict[int, list[np.ndarray]] = {}
    grouped_faces: dict[int, list[np.ndarray]] = {}
    grouped_counts: dict[int, int] = {}
    canonical = canonical_surface_ids(_condition_surface_ids(condition), surfaces)
    map_regions: dict[int, tuple[Region, ...]] = {}
    template_cache: OrderedDict[
        tuple[int, ...],
        _CachedCollisionTemplate,
    ] = OrderedDict()
    source_triangle_count = 0
    generated_triangle_count = 0
    subdivided_triangle_count = 0
    fallback_count = 0
    merged_triangle_count = 0

    for subtree_index, subtree in enumerate(col.subtrees):
        vertices = np.asarray(subtree.vertices, dtype=np.float32)
        _require_finite(vertices, f"COL subtree {subtree_index} vertices")
        for records in _leaf_batches(subtree.root):
            source_faces = np.column_stack(
                (
                    records["a_index"],
                    records["b_index"],
                    records["c_index"],
                )
            ).astype(np.uint32)
            if (
                len(source_faces)
                and int(np.max(source_faces)) >= len(vertices)
            ):
                raise OriginalAdapterError(
                    f"COL subtree {subtree_index} contains an out-of-range face"
                )
            triangles_by_material: dict[int, list[np.ndarray]] = {}
            for record_index, record in enumerate(records):
                source_triangle_count += 1
                if (
                    progress is not None
                    and source_triangle_count % 50_000 == 0
                ):
                    progress(
                        "Converting Original RBR collision geometry: "
                        f"{source_triangle_count:,} triangles"
                    )
                key = _collision_template_key(record)
                cached = template_cache.get(key)
                if cached is None:
                    cached = _cache_collision_template(
                        _build_collision_subdivision_template(
                            record,
                            condition,
                            map_regions,
                            canonical,
                            warnings,
                            warned_selectors,
                        )
                    )
                    template_cache[key] = cached
                    if (
                        len(template_cache)
                        > _COLLISION_TEMPLATE_CACHE_LIMIT
                    ):
                        template_cache.popitem(last=False)
                else:
                    template_cache.move_to_end(key)
                fallback_count += int(cached.used_fallback)
                merged_triangle_count += cached.merged_away
                triangle = np.asarray(
                    vertices[source_faces[record_index]],
                    dtype=np.float64,
                )
                generated = cached.weights @ triangle
                first = generated[:, 1] - generated[:, 0]
                second = generated[:, 2] - generated[:, 0]
                cross_x = first[:, 1] * second[:, 2] - first[:, 2] * second[:, 1]
                cross_y = first[:, 2] * second[:, 0] - first[:, 0] * second[:, 2]
                cross_z = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
                valid = (
                    cross_x * cross_x
                    + cross_y * cross_y
                    + cross_z * cross_z
                    > 1e-20
                )
                generated_for_source = int(np.count_nonzero(valid))
                for surface_id, indices in cached.surface_groups:
                    selected = indices[valid[indices]]
                    if len(selected):
                        triangles_by_material.setdefault(
                            surface_id,
                            [],
                        ).append(generated[selected])
                generated_triangle_count += generated_for_source
                subdivided_triangle_count += int(generated_for_source > 1)
            _append_collision_triangle_batch(
                triangles_by_material,
                grouped_vertices,
                grouped_faces,
                grouped_counts,
            )

    parts, surface_ids = _finalize_collision_parts(
        grouped_vertices,
        grouped_faces,
        surfaces,
        "collision",
    )
    if fallback_count:
        warnings.append(
            f"Used dominant-material fallback for {fallback_count} original "
            "COL triangles whose MAT subdivision produced no valid area"
        )
    result = AdaptedCollision(
        parts,
        surface_ids,
        tuple(warnings),
        CollisionSubdivisionStats(
            source_triangle_count,
            generated_triangle_count,
            subdivided_triangle_count,
            fallback_count,
            merged_triangle_count,
        ),
    )
    return result


def _all_condition_keys(
    mat: MatFile,
) -> tuple[tuple[ConditionKey, MaterialCondition], ...]:
    keyed = sorted(
        (((condition.surface, condition.age), condition) for condition in mat.conditions),
        key=lambda item: item[0],
    )
    if not keyed:
        raise OriginalAdapterError(
            "MAT contains no conditions for all-conditions collision adaptation"
        )
    keys = [key for key, _condition in keyed]
    if len(set(keys)) != len(keys):
        raise OriginalAdapterError(
            "MAT contains duplicate surface/age condition keys"
        )
    for _key, condition in keyed:
        for material_map in condition.maps:
            if np.asarray(material_map.values).shape != (16, 16):
                raise OriginalAdapterError(
                    "Original material maps must contain a 16x16 grid"
                )
    return tuple(keyed)


def _all_condition_profile_regions(
    conditions: Sequence[MaterialCondition],
    canonical: Mapping[int, int],
) -> dict[int, tuple[Region, ...]]:
    result = {}
    for map_index in range(min(len(condition.maps) for condition in conditions)):
        maps = [condition.maps[map_index].values for condition in conditions]
        result[map_index] = decompose_regions(
            tuple(
                tuple(
                    tuple(
                        canonical.get(int(values[row, column]), int(values[row, column]))
                        for values in maps
                    )
                    for column in range(_MAP_CELLS)
                )
                for row in range(_MAP_CELLS)
            )
        )
    return result


def _missing_map_regions(
    condition_count: int,
    canonical: Mapping[int, int],
) -> tuple[Region, ...]:
    # RBR's map lookup (FUN_00520240) returns surface 0 for a map index at or
    # beyond the MAT's map count; NGP keeps that bounds check.
    vector = (canonical.get(0, 0),) * condition_count
    return decompose_regions(((vector,) * _MAP_CELLS,) * _MAP_CELLS)


def _all_collision_surface_ids(
    col: ColFile,
    conditions: Iterable[MaterialCondition],
) -> set[int]:
    result = {
        int(value)
        for condition in conditions
        for material_map in condition.maps
        for value in np.unique(material_map.values)
    }
    for subtree in col.subtrees:
        leaves = tuple(_leaf_batches(subtree.root))
        if not leaves:
            continue
        records = np.concatenate(leaves, dtype=leaves[0].dtype)
        for side in (1, 2):
            result.update(np.unique(records[f"material_{side}_id"]).tolist())
    return result


def adapt_col_collision_all_conditions(
    col: ColFile,
    mat: MatFile,
    *,
    surfaces: Mapping[int, RbrSurface] | None = None,
    progress: Callable[[str], None] | None = None,
) -> AllConditionsCollision:
    """Preserve all MAT assignments on one selector-cell-subdivided COL mesh."""
    keyed_conditions = _all_condition_keys(mat)
    condition_keys = tuple(key for key, _condition in keyed_conditions)
    conditions = tuple(condition for _key, condition in keyed_conditions)
    canonical = canonical_surface_ids(
        _all_collision_surface_ids(col, conditions),
        surfaces,
    )
    warnings: list[str] = []
    profile_regions = _all_condition_profile_regions(conditions, canonical)
    missing_map_regions = _missing_map_regions(len(conditions), canonical)
    # Per template key: stacked barycentric weights and profile vector ids.
    template_cache: OrderedDict[
        tuple[int, ...], tuple[np.ndarray, np.ndarray]
    ] = OrderedDict()
    vector_ids: dict[tuple[int, ...], int] = {}
    triangles_by_vector: dict[int, list[np.ndarray]] = {}
    source_triangle_count = 0
    generated_triangle_count = 0
    subdivided_triangle_count = 0

    for subtree_index, subtree in enumerate(col.subtrees):
        vertices = np.asarray(subtree.vertices, dtype=np.float32)
        _require_finite(vertices, f"COL subtree {subtree_index} vertices")
        leaves = tuple(_leaf_batches(subtree.root))
        if not leaves:
            continue
        subtree_records = np.concatenate(leaves, dtype=leaves[0].dtype)
        for start in range(0, len(subtree_records), _COLLISION_BATCH_RECORDS):
            records = subtree_records[start : start + _COLLISION_BATCH_RECORDS]
            source_faces = np.column_stack(
                (
                    records["a_index"],
                    records["b_index"],
                    records["c_index"],
                )
            ).astype(np.uint32)
            if int(np.max(source_faces)) >= len(vertices):
                raise OriginalAdapterError(
                    f"COL subtree {subtree_index} contains an out-of-range face"
                )
            previous_count = source_triangle_count
            source_triangle_count += len(records)
            if (
                progress is not None
                and source_triangle_count // 50_000 > previous_count // 50_000
            ):
                progress(
                    "Converting Original RBR all-condition collision geometry: "
                    f"{source_triangle_count // 50_000 * 50_000:,} triangles"
                )
            unique_keys, key_indices = np.unique(
                _collision_template_keys(records),
                axis=0,
                return_inverse=True,
            )
            key_weights: list[np.ndarray] = []
            key_vectors: list[np.ndarray] = []
            for key_values in unique_keys.tolist():
                key = tuple(key_values)
                cached = template_cache.get(key)
                if cached is None:
                    weights, vectors = _build_all_conditions_collision_template(
                        key,
                        profile_regions,
                        missing_map_regions,
                    )
                    cached = (
                        weights,
                        np.asarray(
                            [
                                vector_ids.setdefault(vector, len(vector_ids))
                                for vector in vectors
                            ],
                            dtype=np.int64,
                        ),
                    )
                    template_cache[key] = cached
                    if len(template_cache) > _COLLISION_TEMPLATE_CACHE_LIMIT:
                        template_cache.popitem(last=False)
                else:
                    template_cache.move_to_end(key)
                key_weights.append(cached[0])
                key_vectors.append(cached[1])
            # One row per (source triangle, template), in source order.
            key_indices = key_indices.reshape(-1)
            template_counts = np.asarray(
                [len(vectors) for vectors in key_vectors],
                dtype=np.int64,
            )
            record_counts = template_counts[key_indices]
            record_indices = np.repeat(np.arange(len(records)), record_counts)
            template_indices = np.repeat(
                (np.cumsum(template_counts) - template_counts)[key_indices]
                - (np.cumsum(record_counts) - record_counts),
                record_counts,
            ) + np.arange(len(record_indices))
            triangles = vertices[source_faces].astype(np.float64)
            generated = np.matmul(
                np.concatenate(key_weights)[template_indices],
                triangles[record_indices],
            )
            cross = np.cross(
                generated[:, 1] - generated[:, 0],
                generated[:, 2] - generated[:, 0],
            )
            valid = ~(
                np.matmul(cross[:, None, :], cross[:, :, None])[:, 0, 0]
                <= 1e-20
            )
            generated_per_record = np.bincount(
                record_indices[valid],
                minlength=len(records),
            )
            generated_triangle_count += int(generated_per_record.sum())
            subdivided_triangle_count += int(
                np.count_nonzero(generated_per_record > 1)
            )
            kept = generated[valid].astype(np.float32)
            kept_vectors = np.concatenate(key_vectors)[template_indices[valid]]
            order = np.argsort(kept_vectors, kind="stable")
            sorted_vectors = kept_vectors[order]
            group_starts = np.flatnonzero(np.diff(sorted_vectors, prepend=-1))
            for vector_id, group in zip(
                sorted_vectors[group_starts].tolist(),
                np.split(order, group_starts[1:]),
            ):
                triangles_by_vector.setdefault(vector_id, []).append(kept[group])

    vectors_by_id = {vector_id: vector for vector, vector_id in vector_ids.items()}
    triangles_by_profile = {
        vectors_by_id[vector_id]: triangles
        for vector_id, triangles in triangles_by_vector.items()
    }
    profile_names = {
        vector: f"original_collision_profile_{index:04d}"
        for index, vector in enumerate(sorted(triangles_by_profile))
    }
    parts: list[MeshPart] = []
    for vector in sorted(triangles_by_profile):
        vertices = np.concatenate(triangles_by_profile[vector]).reshape((-1, 3))
        vertices = vertices.astype(np.float32, copy=False)
        faces = np.arange(len(vertices), dtype=np.uint32).reshape((-1, 3))
        faces = faces[:, (0, 2, 1)]
        texcoords = np.zeros((len(vertices), 2), dtype=np.float32)
        parts.append(
            MeshPart(
                name=f"{profile_names[vector]}_collision",
                vertices=vertices,
                faces=faces,
                normals=_triangle_soup_normals(vertices, faces),
                texcoords=texcoords,
                colors=np.ones((len(vertices), 4), dtype=np.float32),
                material_index=0,
                material_name=profile_names[vector],
                texcoord_sets=(texcoords,),
            )
        )
    remaps = {
        condition_key: tuple(
            CollisionProfileAssignment(profile_names[vector], vector[condition_index])
            for vector in sorted(profile_names)
        )
        for condition_index, condition_key in enumerate(condition_keys)
    }
    return AllConditionsCollision(
        tuple(parts),
        MappingProxyType(remaps),
        tuple(warnings),
        CollisionSubdivisionStats(
            source_triangle_count,
            generated_triangle_count,
            subdivided_triangle_count,
        ),
    )


def _quaternion_matrix(
    rotation: tuple[float, float, float, float],
) -> np.ndarray:
    x, y, z, w = (float(value) for value in rotation)
    length = math.sqrt(x * x + y * y + z * z + w * w)
    if length < 1e-12:
        return np.identity(3, dtype=np.float64)
    x, y, z, w = x / length, y / length, z / length, w / length
    return np.asarray(
        (
            (
                1 - 2 * (y * y + z * z),
                2 * (x * y - z * w),
                2 * (x * z + y * w),
            ),
            (
                2 * (x * y + z * w),
                1 - 2 * (x * x + z * z),
                2 * (y * z - x * w),
            ),
            (
                2 * (x * z - y * w),
                2 * (y * z + x * w),
                1 - 2 * (x * x + y * y),
            ),
        ),
        dtype=np.float64,
    )


def adapt_trk_shape_collision(
    trk: TrkFile,
    *,
    surfaces: Mapping[int, RbrSurface] | None = None,
) -> AdaptedCollision:
    grouped_vertices: dict[int, list[np.ndarray]] = {}
    grouped_faces: dict[int, list[np.ndarray]] = {}
    grouped_counts: dict[int, int] = {}
    warnings: list[str] = []
    for mesh in trk.shape_collision_meshes or ():
        if mesh.faces and mesh.vertices:
            local_vertices = np.asarray(mesh.vertices, dtype=np.float32)
            local_faces = np.asarray(
                [face.indices for face in mesh.faces],
                dtype=np.uint32,
            )
        else:
            continue
        if not mesh.instances:
            warnings.append(
                f"Original shape collision {mesh.name!r} has no instances"
            )
            continue
        material_id = int(mesh.material_id)
        for instance in mesh.instances:
            scale = np.asarray(instance.scale, dtype=np.float64)
            rotation = _quaternion_matrix(instance.rotation)
            vertices = (
                np.asarray(local_vertices, dtype=np.float64)
                * scale
            ) @ rotation.T + np.asarray(instance.position, dtype=np.float64)
            vertices = vertices.astype(np.float32)
            faces = local_faces
            if np.linalg.det(rotation) * float(np.prod(scale)) < 0:
                faces = faces[:, (0, 2, 1)]
            base = grouped_counts.get(material_id, 0)
            grouped_vertices.setdefault(material_id, []).append(vertices)
            grouped_faces.setdefault(material_id, []).append(faces + base)
            grouped_counts[material_id] = base + len(vertices)

    parts, surface_ids = _finalize_collision_parts(
        grouped_vertices,
        grouped_faces,
        surfaces,
        "shape_collision",
    )
    return AdaptedCollision(parts, surface_ids, tuple(warnings))


def _texture_entries(
    texture_ini: OriginalTextureIni,
    kind: str,
) -> tuple[OriginalTexture, ...]:
    entries = {
        "diffuse": texture_ini.textures,
        "specular": texture_ini.specular_textures,
        "shadow": texture_ini.shadow_textures,
    }.get(kind)
    if entries is None:
        raise OriginalAdapterError(f"Unknown texture reference kind {kind!r}")
    return entries


def _texture_entry(
    texture_ini: OriginalTextureIni,
    reference: TextureReference,
) -> OriginalTexture:
    entries = _texture_entries(texture_ini, reference.kind)
    if reference.index < 0 or reference.index >= len(entries):
        raise OriginalAdapterError(
            f"Referenced {reference.kind} texture index {reference.index} "
            f"is outside 0..{len(entries) - 1}"
        )
    return entries[reference.index]


def _sanitize_visual_material_specs(
    specs: Iterable[VisualMaterialSpec],
    texture_ini: OriginalTextureIni,
    water_name_matches: tuple[str, ...] | None = DEFAULT_WATER_NAME_MATCHES,
) -> tuple[tuple[VisualMaterialSpec, ...], tuple[str, ...]]:
    result: list[VisualMaterialSpec] = []
    warnings: list[str] = []
    for spec in specs:
        replacements: dict[str, None] = {}
        for field, kind in (
            ("diffuse_1", "diffuse"),
            ("diffuse_2", "diffuse"),
            ("specular", "specular"),
            ("shadow", "shadow"),
        ):
            index = getattr(spec, field)
            if index is None:
                continue
            entries = _texture_entries(texture_ini, kind)
            if 0 <= index < len(entries):
                continue
            replacements[field] = None
            warnings.append(
                f"Original {spec.source_kind} material {spec.name} references "
                f"missing {kind} texture {index}; using no texture"
            )
        sanitized = replace(spec, **replacements) if replacements else spec
        water = sanitized.water or (
            water_name_matches is not None
            and sanitized.source_kind == "object"
            and any(
                is_water_material_reference(
                    sanitized.name,
                    _texture_entry(
                        texture_ini,
                        TextureReference("diffuse", texture_index),
                    ).filename,
                    water_name_matches,
                )
                for texture_index in (
                    sanitized.diffuse_1,
                    sanitized.diffuse_2,
                )
                if texture_index is not None
            )
        )
        result.append(replace(sanitized, water=True) if water else sanitized)
    return tuple(result), tuple(warnings)


def _safe_texture_name(
    reference: TextureReference,
    entry: OriginalTexture,
    member: str,
) -> str:
    basename = PureWindowsPath(entry.filename.replace("/", "\\")).name
    suffix = PurePosixPath(member).suffix.lower()
    if not suffix or not re.fullmatch(r"\.[a-z0-9]{1,10}", suffix):
        suffix = ".bin"
    stem = basename[: -len(Path(basename).suffix)] if Path(basename).suffix else basename
    stem = re.sub(r"[^A-Za-z0-9_.-]+", "_", stem).strip("._-") or "texture"
    digest = hashlib.sha1(entry.filename.casefold().encode("latin-1")).hexdigest()[:8]
    return f"{reference.kind}_{reference.index:04d}_{stem[:64]}_{digest}{suffix}"


def extract_referenced_textures(
    references: Iterable[TextureReference],
    texture_ini: OriginalTextureIni,
    textures: OriginalTextureLookup,
    target_dir: str | Path,
    *,
    max_texture_size: int = 512 * 1024**2,
    missing_warnings: list[str] | None = None,
) -> dict[TextureReference, Path]:
    filesystem = current_filesystem()
    target = filesystem.write_path(target_dir)
    filesystem.mkdir(target, parents=True, exist_ok=True)
    result: dict[TextureReference, Path] = {}
    for reference in sorted(set(references), key=lambda item: (item.kind, item.index)):
        entry = _texture_entry(texture_ini, reference)
        member = textures.path(entry)
        if member is None:
            wetness, wear = textures.condition
            message = (
                f"Original {reference.kind} texture {entry.filename!r} is "
                f"missing from the stage payload for {wetness}/{wear}; "
                "using no texture"
            )
            if missing_warnings is None:
                raise MissingTextureError(message)
            missing_warnings.append(message)
            continue
        output = filesystem.write_path(
            target / _safe_texture_name(reference, entry, member)
        )
        try:
            output.relative_to(target)
        except ValueError as exc:
            raise OriginalAdapterError("Generated texture path escapes target directory") from exc
        temporary = output.with_name(output.name + ".part")
        try:
            with textures.open(entry, max_size=max_texture_size) as source:
                with filesystem.open(temporary, "wb") as destination:
                    shutil.copyfileobj(source, destination, length=1024 * 1024)
            if filesystem.stat(temporary).st_size > max_texture_size:
                raise OriginalAdapterError(
                    f"Texture {entry.filename!r} exceeds {max_texture_size} bytes"
                )
            filesystem.replace(temporary, output)
        finally:
            if filesystem.exists(temporary):
                filesystem.unlink(temporary)
        result[reference] = output
    return result


_SurfacePhysics = tuple[str, bool, bool, bool, float, bool]


def _surface_physics(
    surface_id: int,
    surfaces: Mapping[int, RbrSurface],
) -> _SurfacePhysics:
    surface = surfaces.get(surface_id)
    profile = (
        surface.profile
        if surface and surface.profile
        else current_surface_rules().unknown_profile
    )
    ground_depth = profile.ground_depth
    return (
        profile.ground_type,
        profile.hard,
        profile.water,
        profile.bendable,
        ground_depth,
        profile.snowbank,
    )


def canonical_surface_ids(
    surface_ids: Iterable[int],
    surfaces: Mapping[int, RbrSurface] | None,
) -> dict[int, int]:
    """Map each physical material ID onto the lowest ID BeamNG treats alike.

    Collision fragments can only merge when they share a ground model, so the
    subdivision works on these representatives instead of the raw IDs. An ID
    without surface data maps to itself: two unknown surfaces resolve to the
    same placeholder variant, but that is missing data rather than evidence
    that they behave alike.
    """
    resolved = surfaces or {}
    representatives: dict[_SurfacePhysics, int] = {}
    canonical: dict[int, int] = {}
    for surface_id in sorted(set(surface_ids)):
        if surface_id not in resolved:
            canonical[surface_id] = surface_id
            continue
        physics = _surface_physics(surface_id, resolved)
        canonical[surface_id] = representatives.setdefault(physics, surface_id)
    return canonical


def _surface_variant(
    material: RbrMaterial,
    surface_id: int,
    surfaces: Mapping[int, RbrSurface],
    merged_surface_ids: tuple[int, ...] = (),
) -> MaterialVariant:
    surface = surfaces.get(surface_id)
    (
        ground_type,
        hard,
        water,
        bendable,
        ground_depth,
        snowbank,
    ) = _surface_physics(surface_id, surfaces)
    return MaterialVariant(
        material=material,
        ground_type=ground_type,
        hard=hard,
        water=water,
        bendable=bendable,
        source_surface_ids=merged_surface_ids or (surface_id,),
        ground_depth=ground_depth,
        snowbank=snowbank,
        classification_fallback=(
            "original-unresolved-surface"
            if surface is None or surface.profile_status == "unmapped"
            else None
        ),
    )


def _original_alpha_refs(
    spec: VisualMaterialSpec,
    texture: OriginalTexture | None,
    diffuse_texture: Path | None,
    second_diffuse_texture: Path | None,
) -> tuple[int | None, int | None]:
    """HedgeHog3D alpha passes as ``(cutout_alpha_ref, blend_alpha_ref)``.

    Geom chunks select render states from their render type only; single-
    texture objects read the texture INI flags; reflection and water always
    blend, and their double-texture alpha also depends on the second texture
    and vertex alpha, which the atlas bake resolves.
    """
    if diffuse_texture is None:
        return None, None
    if spec.source_kind in {"reflection", "water"}:
        if second_diffuse_texture is not None or texture_has_alpha(diffuse_texture):
            return _OPAQUE_TEXEL_ALPHA_REF, 1
        return None, None
    if not texture_has_alpha(diffuse_texture):
        return None, None
    single_texture = spec.diffuse_2 is None
    if spec.source_kind == "geom":
        return (1, None) if single_texture and spec.specular is None else (None, None)
    if not single_texture or texture is None or texture.opacity_map is not True:
        return None, None
    if texture.one_bit_opacity is True:
        return _OBJECT_CUTOUT_ALPHA_REF, None
    return _OBJECT_CUTOUT_ALPHA_REF, 1


def build_original_materials(
    visual_specs: Iterable[VisualMaterialSpec],
    collision_surface_ids: Iterable[int],
    texture_ini: OriginalTextureIni,
    extracted: Mapping[TextureReference, Path],
    surfaces: Mapping[int, RbrSurface],
) -> tuple[list[RbrMaterial], dict[str, MaterialVariant]]:
    specs = sorted(visual_specs, key=lambda item: item.material_index)
    materials: list[RbrMaterial] = []
    variants: dict[str, MaterialVariant] = {}
    for spec in specs:
        first_entry = (
            _texture_entry(texture_ini, TextureReference("diffuse", spec.diffuse_1))
            if spec.diffuse_1 is not None
            else None
        )
        properties = dict(first_entry.properties) if first_entry else {}
        properties.update(
            {
                "originalSourceKind": spec.source_kind,
                "originalDiffuse1": str(spec.diffuse_1),
                "originalDiffuse2": str(spec.diffuse_2),
                "originalSpecular": str(spec.specular),
                "originalShadow": str(spec.shadow),
                "originalRenderFlags": hex(spec.render_flags),
            }
        )
        if spec.glossiness is not None:
            properties["originalGlossiness"] = str(spec.glossiness)
        shadow_path = (
            extracted.get(TextureReference("shadow", spec.shadow))
            if spec.shadow is not None
            else None
        )
        if shadow_path is not None:
            properties["originalShadowPath"] = str(shadow_path)
        diffuse_texture = (
            extracted.get(TextureReference("diffuse", spec.diffuse_1))
            if spec.diffuse_1 is not None
            else None
        )
        second_diffuse_texture = (
            extracted.get(TextureReference("diffuse", spec.diffuse_2))
            if spec.diffuse_2 is not None
            else None
        )
        cutout_alpha_ref, blend_alpha_ref = _original_alpha_refs(
            spec,
            first_entry,
            diffuse_texture,
            second_diffuse_texture,
        )
        transparent = cutout_alpha_ref is not None or blend_alpha_ref is not None
        material = RbrMaterial(
            index=spec.material_index,
            name=spec.name,
            effect="RBR_Original",
            technique="Transparent" if transparent else "Default",
            diffuse_texture=diffuse_texture,
            second_diffuse_texture=second_diffuse_texture,
            normal_texture=None,
            specular_texture=(
                extracted.get(TextureReference("specular", spec.specular))
                if spec.specular is not None
                else None
            ),
            properties=properties,
            double_sided=bool(spec.render_flags & NO_CULLING),
            cutout_alpha_ref=cutout_alpha_ref,
            blend_alpha_ref=blend_alpha_ref,
        )
        materials.append(material)
        if (
            material.diffuse_texture is not None
            and material.second_diffuse_texture is not None
        ):
            pbr_override = PbrMaterialOverride.vertex_color_lerp(
                material.diffuse_texture,
                material.second_diffuse_texture,
            )
        elif material.diffuse_texture is not None:
            pbr_override = PbrMaterialOverride(
                base_color_texture=material.diffuse_texture,
                base_vertex_color=True,
            )
        else:
            pbr_override = None
        variants[spec.name] = MaterialVariant(
            material=material,
            ground_type="WATER" if spec.water else _RENDER_ONLY_GROUND_TYPE,
            hard=not transparent and not spec.water,
            water=spec.water,
            bendable=False,
            source_surface_ids=(),
            pbr_override=pbr_override,
            base_vertex_color=True,
        )

    for ordinal, surface_id in enumerate(sorted(set(collision_surface_ids))):
        index = len(specs) + ordinal
        name = f"original_surface_{surface_id:03d}"
        surface = surfaces.get(surface_id)
        material = RbrMaterial(
            index=index,
            name=name,
            effect="RBR_Original_Physics",
            technique="Default",
            diffuse_texture=None,
            second_diffuse_texture=None,
            normal_texture=None,
            specular_texture=None,
            properties={
                "originalPhysicalMaterialID": str(surface_id),
                "originalPhysicalMaterialName": surface.name if surface else "unknown",
                "originalPhysicalMaterialStatus": (
                    "resolved" if surface else "unresolved"
                ),
            },
        )
        materials.append(material)
        variants[name] = _surface_variant(
            material,
            surface_id,
            surfaces,
        )
    return materials, variants


def _stage_source_vector(vector: tuple[float, float, float]) -> tuple[float, float, float]:
    # Existing route writers swap source Y/Z. Store the inverse arrangement so
    # their output is the already-right-handed Original RBR Z-up coordinate.
    return vector[0], vector[2], vector[1]


def _provenance_strings(values: Mapping[str, object] | None) -> dict[str, str]:
    return {key: str(value) for key, value in (values or {}).items()}


def build_original_stage(
    *,
    root: str | Path,
    metadata: StageMetadata,
    trk: TrkFile,
    dls: DlsFile,
    lbs: LbsFile,
    materials: list[RbrMaterial],
    surfaces: Mapping[int, RbrSurface],
    location: StageLocation | None = None,
    source_variant: str = "",
    provenance: Mapping[str, object] | None = None,
    warnings: Iterable[str] = (),
    unresolved_surface_ids: Iterable[int] = (),
    used_surface_ids: Iterable[int] = (),
) -> RbrStage:
    if trk.driveline is None or len(trk.driveline) < 2:
        raise OriginalAdapterError("Original TRK must contain at least two driveline points")
    driveline: list[DrivelinePoint] = []
    previous_distance = -math.inf
    for index, point in enumerate(trk.driveline):
        if point.distance < previous_distance:
            raise OriginalAdapterError(
                f"Original driveline distance decreases at point {index}"
            )
        previous_distance = point.distance
        driveline.append(
            DrivelinePoint(
                position=_stage_source_vector(point.position),
                direction=_stage_source_vector(point.tangent),
                distance=point.distance,
                flags=0,
            )
        )

    if lbs.car_location is None:
        raise OriginalAdapterError(
            "Original LBS has no CAR_LOCATION section; "
            "an authoritative spawn cannot be determined"
        )
    stage_warnings = list(warnings)
    spawn_position = lbs.car_location.position
    spawn_angles = lbs.car_location.euler_vector
    spawn_matrix = (
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        1.0,
        0.0,
        spawn_position[0],
        spawn_position[1],
        spawn_position[2],
        1.0,
    )
    pacenotes = [
        Pacenote(note.note_id, note.distance, note.flags)
        for note in dls.pacenotes
    ]
    if not any(note.note_type == 22 for note in pacenotes):
        stage_warnings.append(
            "Original DLS has no finish event (22); converting it as a freeroam-only level"
        )
    surface_dict = dict(surfaces)
    source_provenance = _provenance_strings(provenance)
    if surface_dict:
        first_surface = next(iter(surface_dict.values()))
        source_provenance.setdefault(
            "physicsLspFingerprint",
            first_surface.physics_fingerprint,
        )
        source_provenance.setdefault(
            "surfaceProfileCatalogVersion",
            str(first_surface.catalog_version),
        )
    return RbrStage(
        root=Path(root),
        metadata=metadata,
        objects=[],
        materials=materials,
        surface_maps={},
        surface_types={
            surface_id: surface.ground_type
            for surface_id, surface in surface_dict.items()
        },
        surfaces=surface_dict,
        spawn=Spawn(spawn_matrix, spawn_angles),
        driveline=driveline,
        pacenotes=pacenotes,
        location=location,
        warnings=stage_warnings,
        source_format="original",
        source_variant=source_variant,
        source_provenance=source_provenance,
        unresolved_surface_ids=tuple(sorted(set(unresolved_surface_ids))),
        used_surface_ids=tuple(sorted(set(used_surface_ids))),
    )


def _combined_bounds(
    render_parts: Iterable[MeshPart],
    collision_parts: Iterable[MeshPart],
    lbs: LbsFile,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    minima: list[np.ndarray] = []
    maxima: list[np.ndarray] = []
    for part in tuple(render_parts) + tuple(collision_parts):
        vertices = np.asarray(part.vertices, dtype=np.float64)
        if len(vertices):
            _require_finite(vertices, f"{part.name} bounds")
            minima.append(np.min(vertices, axis=0))
            maxima.append(np.max(vertices, axis=0))
    if lbs.world_bounds:
        for bounds in lbs.world_bounds:
            center = np.asarray(bounds.center, dtype=np.float64)[[0, 2, 1]]
            extents = np.asarray(bounds.half_extents, dtype=np.float64)[[0, 2, 1]]
            minima.append(center - extents)
            maxima.append(center + extents)
    if not minima:
        zero = (0.0, 0.0, 0.0)
        return zero, zero
    lower = np.min(np.stack(minima), axis=0)
    upper = np.max(np.stack(maxima), axis=0)
    return tuple(float(value) for value in lower), tuple(float(value) for value in upper)


def adapt_visual_snowbank_collision(
    visuals: AdaptedVisuals,
    trk: TrkFile,
    surfaces: Mapping[int, RbrSurface],
    *,
    material_index: int,
) -> tuple[MeshPart, ...]:
    boxes: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for mesh in trk.shape_collision_meshes or ():
        volume = mesh.soft_volume
        surface = surfaces.get(int(mesh.material_id))
        if (
            surface is None
            or surface.profile is None
            or surface.profile.ground_type != "SNOWBANK"
            or volume is None
            or volume.kind != 1
            or not isinstance(volume.size_or_radius, tuple)
        ):
            continue
        local_center = np.asarray(volume.center, dtype=np.float64)
        local_half = np.abs(np.asarray(volume.size_or_radius, dtype=np.float64))
        for instance in mesh.instances:
            scale = np.asarray(instance.scale, dtype=np.float64)
            rotation = (
                np.identity(3, dtype=np.float64)
                if mesh.use_local_rotation
                else _quaternion_matrix(instance.rotation)
            )
            world_center = (
                local_center * scale
            ) @ rotation.T + np.asarray(instance.position, dtype=np.float64)
            scaled_half = np.abs(local_half * scale)
            boxes.append(
                (
                    world_center,
                    scaled_half,
                    rotation,
                    np.abs(rotation) @ scaled_half,
                )
            )
    if not boxes:
        return ()

    grid_size = 8.0
    box_grid: dict[tuple[int, int, int], list[int]] = {}
    for box_index, (center, _half, _rotation, aabb_half) in enumerate(boxes):
        grid_min = np.floor((center - aabb_half) / grid_size).astype(np.int64)
        grid_max = np.floor((center + aabb_half) / grid_size).astype(np.int64)
        for grid_x in range(int(grid_min[0]), int(grid_max[0]) + 1):
            for grid_y in range(int(grid_min[1]), int(grid_max[1]) + 1):
                for grid_z in range(int(grid_min[2]), int(grid_max[2]) + 1):
                    box_grid.setdefault((grid_x, grid_y, grid_z), []).append(
                        box_index
                    )

    geom_materials = {
        spec.name
        for spec in visuals.material_specs
        if spec.source_kind == "geom"
    }
    selected_chunks: list[np.ndarray] = []
    for part in visuals.parts:
        if (
            part.lod_kind == "far"
            or part.material_name not in geom_materials
            or not len(part.faces)
        ):
            continue
        triangles = np.asarray(part.vertices[part.faces], dtype=np.float64)
        centroids = np.mean(triangles, axis=1)
        cells = np.floor(centroids / grid_size).astype(np.int64)
        unique_cells, cell_indices = np.unique(
            cells,
            axis=0,
            return_inverse=True,
        )
        selected = np.zeros(len(triangles), dtype=bool)
        for cell_index, cell in enumerate(unique_cells):
            candidate_boxes = box_grid.get(tuple(int(value) for value in cell))
            if not candidate_boxes:
                continue
            triangle_indices = np.flatnonzero(cell_indices == cell_index)
            points = np.concatenate(
                (
                    triangles[triangle_indices],
                    centroids[triangle_indices, None, :],
                ),
                axis=1,
            )
            for box_index in candidate_boxes:
                center, half, rotation, _aabb_half = boxes[box_index]
                local_points = (points - center) @ rotation
                inside = np.any(
                    np.all(np.abs(local_points) <= half + 0.1, axis=2),
                    axis=1,
                )
                selected[triangle_indices[inside]] = True
        if np.any(selected):
            selected_chunks.append(triangles[selected])
    if not selected_chunks:
        return ()

    source_vertices = np.concatenate(selected_chunks).reshape((-1, 3)).astype(
        np.float32,
        copy=False,
    )
    vertices, inverse = np.unique(source_vertices, axis=0, return_inverse=True)
    faces = inverse.reshape((-1, 3)).astype(np.uint32, copy=False)
    texcoords = np.zeros((len(vertices), 2), dtype=np.float32)
    return (
        MeshPart(
            name="original_visual_snowwall_collision",
            vertices=vertices,
            faces=faces,
            normals=_generated_normals(vertices, faces),
            texcoords=texcoords,
            colors=np.ones((len(vertices), 4), dtype=np.float32),
            material_index=material_index,
            material_name="original_visual_snowwall",
            collision_eligible=True,
            water=False,
            texcoord_sets=(texcoords,),
        ),
    )


def prepare_original_variant(
    *,
    root: str | Path,
    metadata: StageMetadata,
    trk: TrkFile,
    dls: DlsFile,
    lbs: LbsFile,
    col: ColFile,
    mat: MatFile,
    texture_ini: OriginalTextureIni,
    textures: OriginalTextureLookup,
    texture_target: str | Path,
    surfaces: Mapping[int, RbrSurface],
    fnc: FncFile | None = None,
    fence_texture_paths: Mapping[int, Path] | None = None,
    use_snowwall_collision_override: bool = True,
    water_name_matches: tuple[str, ...] | None = DEFAULT_WATER_NAME_MATCHES,
    location: StageLocation | None = None,
    source_variant: str = "",
    provenance: Mapping[str, object] | None = None,
    collision_cache: dict[object, AllConditionsCollision] | None = None,
    collision_cache_key: object | None = None,
    progress: Callable[[str], None] | None = None,
) -> PreparedOriginalVariant:
    if progress is not None:
        progress("Converting Original RBR visual geometry")
    with profile_span("adapt_lbs_meshes", category="original") as visual_span:
        visuals = adapt_lbs_meshes(lbs)
        visual_material_specs, texture_reference_warnings = (
            _sanitize_visual_material_specs(
                visuals.material_specs,
                texture_ini,
                water_name_matches,
            )
        )
        water_material_names = {
            spec.name
            for spec in visual_material_specs
            if spec.water
        }
        if water_material_names:
            visuals = replace(
                visuals,
                parts=tuple(
                    replace(
                        part,
                        water=part.water or part.material_name in water_material_names,
                    )
                    for part in visuals.parts
                ),
            )
        visual_span.update(
            parts=len(visuals.parts),
            vertices=sum(len(part.vertices) for part in visuals.parts),
            faces=sum(len(part.faces) for part in visuals.parts),
            materials=len(visuals.material_specs),
            warnings=len(visuals.warnings) + len(texture_reference_warnings),
        )
    with profile_span(
        "adapt_snowbank_collision",
        category="original",
        enabled=use_snowwall_collision_override,
    ) as snowbank_span:
        snowbank_visual_collision = (
            adapt_visual_snowbank_collision(
                visuals,
                trk,
                surfaces,
                material_index=len(visuals.material_specs),
            )
            if use_snowwall_collision_override
            else ()
        )
        snowbank_span.update(
            parts=len(snowbank_visual_collision),
            faces=sum(
                len(part.faces)
                for part in snowbank_visual_collision
            ),
        )
    with profile_span("adapt_col_collision", category="original") as col_span:
        if progress is not None:
            progress("Building Original RBR collision geometry")
        collision_cache_hit = (
            collision_cache is not None
            and collision_cache_key in collision_cache
        )
        if collision_cache_hit:
            source_collision = collision_cache[collision_cache_key]
        else:
            source_collision = adapt_col_collision_all_conditions(
                col,
                mat,
                surfaces=surfaces,
                progress=progress,
            )
        if (
            collision_cache is not None
            and collision_cache_key is not None
            and not collision_cache_hit
        ):
            collision_cache[collision_cache_key] = source_collision
        collision = replace(
            source_collision,
            parts=tuple(replace(part) for part in source_collision.parts),
        )
        col_span.update(
            cacheHit=collision_cache_hit,
            parts=len(collision.parts),
            faces=sum(len(part.faces) for part in collision.parts),
            surfaceIds=len(
                {
                    assignment.surface_id
                    for assignments in collision.condition_remaps.values()
                    for assignment in assignments
                }
            ),
            warnings=len(collision.warnings),
            sourceTriangles=collision.subdivision.source_triangles,
            generatedTriangles=collision.subdivision.generated_triangles,
            subdividedTriangles=collision.subdivision.subdivided_triangles,
            subdivisionFallbacks=collision.subdivision.fallbacks,
        )
    with profile_span("adapt_trk_collision", category="original") as trk_span:
        if progress is not None:
            progress("Converting Original RBR shape collision")
        shape_collision = adapt_trk_shape_collision(
            trk,
            surfaces=surfaces,
        )
        trk_span.update(
            parts=len(shape_collision.parts),
            faces=sum(
                len(part.faces)
                for part in shape_collision.parts
            ),
            surfaceIds=len(shape_collision.surface_ids),
            warnings=len(shape_collision.warnings),
        )
    default_assignments = collision.condition_remaps.get(textures.condition)
    if default_assignments is None:
        default_assignments = next(iter(collision.condition_remaps.values()), ())
    profile_default_surfaces = {
        assignment.profile_name: assignment.surface_id
        for assignment in default_assignments
    }
    collision_surface_ids = tuple(
        sorted(
            set(profile_default_surfaces.values())
            | set(shape_collision.surface_ids)
        )
    )
    used_surface_ids = {
        assignment.surface_id
        for assignments in collision.condition_remaps.values()
        for assignment in assignments
    } | set(shape_collision.surface_ids)
    unresolved_surface_ids = tuple(
        sorted(
            surface_id
            for surface_id in used_surface_ids
            if surface_id not in surfaces
            or surfaces[surface_id].profile_status == "unmapped"
        )
    )
    unresolved_surface_warnings = surface_profile_warnings(
        used_surface_ids,
        surfaces,
    ) + (
        [
            "Original RBR physical material IDs have no physics.lsp definition: "
            + ", ".join(str(surface_id) for surface_id in unresolved_surface_ids)
        ]
        if unresolved_surface_ids
        else []
    )
    collision_parts = [*collision.parts, *shape_collision.parts]
    if snowbank_visual_collision:
        visual_snowwall_material_index = (
            len(visuals.material_specs) + len(collision_surface_ids)
        )
        for part in snowbank_visual_collision:
            part.material_index = visual_snowwall_material_index
        collision_parts.extend(snowbank_visual_collision)
    references = (
        reference
        for spec in visual_material_specs
        for reference in spec.texture_references
    )
    with profile_span(
        "extract_original_textures",
        category="original",
    ) as texture_span:
        if progress is not None:
            progress("Extracting Original RBR textures")
        missing_texture_warnings: list[str] = []
        extracted = extract_referenced_textures(
            references,
            texture_ini,
            textures,
            texture_target,
            missing_warnings=missing_texture_warnings,
        )
        texture_span.update(
            textures=len(extracted),
            warnings=len(missing_texture_warnings),
            outputBytes=sum(
                current_filesystem().stat(path).st_size
                for path in extracted.values()
            ),
        )
    if progress is not None:
        progress("Building Original RBR materials")
    with profile_span("build_original_materials", category="original"):
        rbr_materials, variants = build_original_materials(
            visual_material_specs,
            collision_surface_ids,
            texture_ini,
            extracted,
            surfaces,
        )
    for profile_name, surface_id in profile_default_surfaces.items():
        source_name = f"original_surface_{surface_id:03d}"
        source = variants[source_name]
        material = replace(
            source.material,
            index=len(rbr_materials),
            name=profile_name,
        )
        rbr_materials.append(material)
        variants[profile_name] = replace(source, material=material)
    for part in collision.parts:
        part.material_index = variants[part.material_name].material.index
    for part, surface_id in zip(shape_collision.parts, shape_collision.surface_ids):
        part.material_index = variants[f"original_surface_{surface_id:03d}"].material.index
    if snowbank_visual_collision:
        variants["original_visual_snowwall"] = MaterialVariant(
            material=None,
            ground_type="SNOWBANK",
            hard=False,
            water=False,
            bendable=True,
            source_surface_ids=tuple(
                sorted(
                    surface_id
                    for surface_id, surface in surfaces.items()
                    if surface.profile is not None
                    and surface.profile.ground_type == "SNOWBANK"
                )
            ),
            ground_depth=0.3,
            snowbank=True,
        )
    fence_parts: tuple[MeshPart, ...] = ()
    fence_warnings: list[str] = []
    if fnc is not None:
        if progress is not None:
            progress("Converting Original RBR fences")
        with profile_span(
            "adapt_fences",
            category="original",
            fences=len(fnc.fences),
            posts=sum(len(fence.posts) for fence in fnc.fences),
        ) as fence_span:
            fence_parts, fence_variants = adapt_fences(
                fnc,
                fence_texture_paths or {},
                material_index_start=(
                    len(visuals.material_specs) + len(collision_surface_ids)
                    + int(bool(snowbank_visual_collision))
                ),
                warnings=fence_warnings,
            )
            fence_span.update(
                parts=len(fence_parts),
                faces=sum(len(part.faces) for part in fence_parts),
            )
        variants.update(fence_variants)
        rbr_materials.extend(
            variant.material
            for variant in fence_variants.values()
            if variant.material is not None
        )
    provenance_strings = _provenance_strings(provenance)
    for reference, output in extracted.items():
        entry = _texture_entry(texture_ini, reference)
        provenance_strings[
            f"texture:{reference.kind}:{reference.index}"
        ] = f"{entry.filename} -> {output}"
    combined_warnings = (
        list(visuals.warnings)
        + list(texture_reference_warnings)
        + missing_texture_warnings
        + list(collision.warnings)
        + list(shape_collision.warnings)
        + fence_warnings
        + unresolved_surface_warnings
    )
    if progress is not None:
        progress("Building Original RBR level data")
    with profile_span("build_original_stage", category="original"):
        stage = build_original_stage(
            root=root,
            metadata=metadata,
            trk=trk,
            dls=dls,
            lbs=lbs,
            materials=rbr_materials,
            surfaces=surfaces,
            location=location,
            source_variant=source_variant,
            provenance=provenance_strings,
            warnings=combined_warnings,
            unresolved_surface_ids=unresolved_surface_ids,
            used_surface_ids=used_surface_ids,
        )
    if progress is not None:
        progress("Computing Original RBR geometry bounds")
    with profile_span("compute_original_bounds", category="original"):
        bounds_min, bounds_max = _combined_bounds(
            (*visuals.parts, *fence_parts),
            collision_parts,
            lbs,
        )
    return PreparedOriginalVariant(
        stage=stage,
        render_parts=(*visuals.parts, *fence_parts),
        collision_parts=tuple(collision_parts),
        materials=variants,
        bounds_min=bounds_min,
        bounds_max=bounds_max,
        condition_remaps=collision.condition_remaps,
        condition_ground_types={
            condition: {
                assignment.profile_name: _surface_variant(
                    None,
                    assignment.surface_id,
                    surfaces,
                ).ground_type
                for assignment in assignments
            }
            for condition, assignments in collision.condition_remaps.items()
        },
        collision_subdivision=collision.subdivision,
        road_condition=textures.condition,
    )
