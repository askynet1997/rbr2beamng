from __future__ import annotations

import hashlib
import itertools
import math
import zipfile
from collections.abc import Mapping
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from . import __version__
from .beamng import (
    ForestType,
    StaticInstance,
    WaterSpec,
    add_cutout_passes,
    validate_package,
    write_collada_to_archive,
    write_level,
)
from .conversion_common import (
    ConversionOptions,
    conversion_parameters,
    install_zip,
    option_stats_for_summary,
    output_path_for,
    pacenote_log_writers,
    pacenote_visualizer_writer,
    report_warnings,
    resolved_map_altitude_meters,
    temporary_workspace,
)
from .core import ConversionError, ProgressReporter, slugify
from .environment import (
    EnvironmentSettings,
    location_override,
    resolve_environment_settings,
)
from .filesystem import current_filesystem, use_filesystem
from .geometry import (
    AssimpMeshLoader,
    ThinWallTemplate,
    convert_transform_matrices,
    decompose_matrices,
    detect_thin_wall_shells,
    extract_thin_wall_templates,
    inflate_thin_wall_templates,
    lod_detail_size,
    remap_mesh_part,
    rbr_col_box_part,
    rbr_stuff_source_matrices,
    remove_overdrawn_faces,
    rotate_z,
    signed_azimuth_delta,
    source_position_to_beamng,
    source_sun_azimuth,
    split_collision_parts_by_surface,
    split_parts_by_surface,
    thicken_rbr_col_box_dimensions,
    write_collada,
)
from .material_baker import (
    bake_vertex_lerp,
    clear_temporary_material_state,
    is_vertex_lerp_material,
)
from .models import (
    ConversionResult,
    DEFAULT_SNOWBANK_NAME_MATCHES,
    DEFAULT_SNOWBANK_NAME_MESH_PATTERNS,
    MaterialVariant,
    MeshAsset,
    MeshPart,
    RbrMaterial,
    RbrStage,
    StageObject,
    WaterAppearance,
    WaterRegion,
)
from .original.source import original_stage_id
from .pacenotes import format_pacenote_summary
from . import plugins
from .plugins import PluginLevel
from .profiling import profile_span
from .rbr import (
    SNOWWALL_SURFACE_ID,
    find_stage,
    is_btb_terrain_mesh,
    is_snowwall_surface_map,
    load_stage,
    load_transforms,
    surface_map_key,
)
from .surface_profiles import current_surface_rules
from .water_appearance import water_appearance_for_source
from .zip_deflate import ZIP_COMPRESSLEVEL


_MIN_WATER_RECTANGLE_COVERAGE = 0.7
_SKY_MIN_HEIGHT = 100.0
_SKY_MIN_ROUTE_SHARE = 0.1
_SKY_ROUTE_SAMPLES = 256

AssetKey = tuple[Path, str, tuple[float, ...]]
AssetLodKey = tuple[
    Path,
    str,
    tuple[float, ...],
    tuple[float | None, float | None],
]


def _rx_map_yaw(
    stage: RbrStage,
    environment: EnvironmentSettings,
    warnings: list[str],
) -> tuple[float, float | None]:
    if stage.sun_direction is None:
        warnings.append(
            "RX stage has no sunDirection; keeping its source map orientation"
        )
        return 0.0, None
    source_azimuth = source_sun_azimuth(stage.sun_direction)
    if source_azimuth is None:
        warnings.append(
            "RX sunDirection has no usable horizontal bearing; "
            "keeping its source map orientation"
        )
        return 0.0, None
    target_azimuth = environment.sun_azimuth
    if target_azimuth is None or not math.isfinite(target_azimuth):
        warnings.append(
            "RX sunDirection could not be aligned without usable GPS/timezone; "
            "keeping its source map orientation"
        )
        return 0.0, source_azimuth
    return signed_azimuth_delta(source_azimuth, target_azimuth), source_azimuth


def _asset_mode(collision_model: int, visible: bool) -> str:
    if collision_model in {0, 2}:
        return "collidable" if visible else "collision"
    return "noncollidable" if visible else "normal"


def _asset_key(stage_object) -> AssetKey:
    collision_box = (
        tuple(getattr(stage_object, "collision_box", ()))
        if stage_object.collision_model == 2
        else ()
    )
    return (
        stage_object.mesh_path,
        _asset_mode(stage_object.collision_model, stage_object.visible),
        collision_box,
    )


def _asset_name(
    source: Path,
    mode: str,
    lod_range: tuple[float | None, float | None] = (None, None),
    collision_box: tuple[float, ...] = (),
) -> str:
    suffix = "" if mode == "normal" else f"_{mode}"
    digest = hashlib.sha1(source.name.casefold().encode("utf-8")).hexdigest()[:8]
    if collision_box:
        box_digest = hashlib.sha1(
            repr(collision_box).encode("ascii")
        ).hexdigest()[:6]
        suffix += f"_box_{box_digest}"
    if lod_range != (None, None):
        lod_digest = hashlib.sha1(
            repr(lod_range).encode("ascii")
        ).hexdigest()[:6]
        suffix += f"_lod_{lod_digest}"
    return f"{slugify(source.stem)[:80]}_{digest}{suffix}.dae"


def _object_transform_radius(
    stage_object,
    transforms: np.ndarray | None = None,
) -> float:
    transforms = (
        load_transforms(stage_object)
        if transforms is None
        else transforms
    )
    translations = np.asarray(transforms[:, 3, :3], dtype=np.float64)
    if len(translations) <= 1:
        return 0.0
    center = (translations.min(axis=0) + translations.max(axis=0)) * 0.5
    return float(np.linalg.norm(translations - center, axis=1).max())


def _stage_object_lod_range(
    stage_object,
    transforms: np.ndarray | None = None,
) -> tuple[float | None, float | None]:
    lod_in = stage_object.lod_in if stage_object.lod_in > 2.0 else None
    lod_out = (
        stage_object.lod_out
        - _object_transform_radius(stage_object, transforms)
    )
    if (
        not math.isfinite(lod_out)
        or lod_out <= 0.0
        or lod_in is not None
        and lod_out <= lod_in
    ):
        return None, None
    return (
        round(lod_in, 3) if lod_in is not None else None,
        round(lod_out, 3),
    )


def _asset_lod_bands(
    stage: RbrStage,
    use_visual_lods: bool = True,
) -> tuple[
    dict[
        AssetKey,
        set[tuple[float | None, float | None]],
    ],
    int,
]:
    values: dict[
        AssetKey,
        set[tuple[float | None, float | None]],
    ] = {}
    transform_ranges: dict[
        tuple[Path, int, float, float],
        tuple[float | None, float | None],
    ] = {}
    invalid_count = 0
    for stage_object in stage.objects:
        asset_key = _asset_key(stage_object)
        mode = asset_key[1]
        if (
            not use_visual_lods
            or mode not in {"collidable", "noncollidable"}
        ):
            values.setdefault(asset_key, set()).add(
                (None, None)
            )
            continue
        transform_key = (
            stage_object.transform_path,
            stage_object.clone_count,
            stage_object.lod_in,
            stage_object.lod_out,
        )
        lod_range = transform_ranges.get(transform_key)
        if lod_range is None:
            lod_range = _stage_object_lod_range(stage_object)
            transform_ranges[transform_key] = lod_range
        if lod_range == (None, None):
            invalid_count += 1
        values.setdefault(asset_key, set()).add(lod_range)
    return values, invalid_count


def _lod_detail_config(
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    lod_range: tuple[float | None, float | None],
) -> tuple[int | None, tuple[int, ...]]:
    lod_in, lod_out = lod_range
    if lod_out is None:
        return None, ()
    radius = float(np.linalg.norm(bounds_max - bounds_min) * 0.5)
    if radius <= 1e-4:
        return None, ()

    far_null = lod_detail_size(radius, lod_out)
    render_detail = far_null + 1
    null_details = [far_null]
    if lod_in is not None:
        near_null = lod_detail_size(radius, lod_in)
        if near_null > render_detail:
            null_details.insert(0, near_null)
    return render_detail, tuple(null_details)


@dataclass(frozen=True)
class SnowbankRules:
    use_surfaces: bool = True
    use_names: bool = True
    name_mesh_patterns: tuple[str, ...] = DEFAULT_SNOWBANK_NAME_MESH_PATTERNS
    names: tuple[str, ...] = DEFAULT_SNOWBANK_NAME_MATCHES

    @classmethod
    def from_options(cls, options: ConversionOptions) -> SnowbankRules:
        return cls(
            options.use_snowwall_collision_override,
            options.use_snowbank_name_fallback,
            options.snowbank_name_mesh_patterns,
            options.snowbank_name_matches,
        )


def _parts_for_asset_mode(
    parts: list[MeshPart],
    variants: dict[str, MaterialVariant],
    mode: str,
    source: Path | None = None,
    snowbank_rules: SnowbankRules = SnowbankRules(),
    stage: RbrStage | None = None,
) -> list[MeshPart]:
    if mode == "noncollidable":
        result = []
        for part in parts:
            variant = variants[part.material_name]
            collision_eligible = bool(
                source
                and _snowwall_collision_override_applies(
                    source,
                    variant,
                    snowbank_rules,
                    stage,
                )
            )
            if collision_eligible:
                surface = stage.surfaces.get(SNOWWALL_SURFACE_ID) if stage else None
                profile = (
                    surface.profile
                    if surface and surface.profile
                    else current_surface_rules().unknown_profile
                )
                material_name = f"{part.material_name}_recovered_snowbank"
                variants[material_name] = replace(
                    variant,
                    ground_type=profile.ground_type,
                    hard=profile.hard,
                    water=False,
                    bendable=profile.bendable,
                    ground_depth=profile.ground_depth,
                    snowbank=True,
                )
                part = replace(part, material_name=material_name)
            result.append(
                replace(part, collision_eligible=collision_eligible)
            )
        return result
    if mode not in {"collidable", "collision"}:
        return parts
    return [
        replace(
            part,
            collision_eligible=(
                not variants[part.material_name].water
                and variants[part.material_name].ground_type != "VOID"
            ),
        )
        for part in parts
    ]


def _show_retained_water_materials(
    parts: list[MeshPart],
    variants: dict[str, MaterialVariant],
) -> None:
    for material_name in {
        part.material_name
        for part in parts
        if part.water
    }:
        variant = variants[material_name]
        if variant.water:
            variants[material_name] = replace(
                variant,
                ground_type="VOID",
                water=False,
            )


def _snowwall_name_candidate(
    source: Path,
    material: RbrMaterial | None,
    rules: SnowbankRules,
) -> bool:
    return is_btb_terrain_mesh(source, rules.name_mesh_patterns) and _has_snowbank_name(
        material,
        rules,
    )


def _is_snowbank_texture(
    material: RbrMaterial | None,
    stage: RbrStage | None,
) -> bool:
    if stage is None or material is None or not material.diffuse_texture:
        return False
    surface_map = stage.surface_maps.get(surface_map_key(material.diffuse_texture))
    return surface_map is not None and any(
        (surface := stage.surfaces.get(surface_id)) is not None
        and surface.profile is not None
        and surface.profile.snowbank
        for row in surface_map.cells
        for surface_id in row
    )


def _is_snowbank_snow(
    variant: MaterialVariant,
    stage: RbrStage | None,
) -> bool:
    return variant.ground_type in {"SNOW", "SNOWBANK"} and _is_snowbank_texture(
        variant.material,
        stage,
    )


def _is_snowwall_collision_override(
    source: Path,
    variant: MaterialVariant,
    rules: SnowbankRules = SnowbankRules(),
    stage: RbrStage | None = None,
) -> bool:
    return (
        variant.snowbank
        or _is_snowbank_snow(variant, stage)
        or _snowwall_name_candidate(source, variant.material, rules)
    )


def _snowwall_collision_override_applies(
    source: Path,
    variant: MaterialVariant,
    rules: SnowbankRules,
    stage: RbrStage | None = None,
) -> bool:
    return bool(
        rules.use_surfaces
        and _is_snowbank_snow(variant, stage)
        or rules.use_names
        and _snowwall_name_candidate(source, variant.material, rules)
    )


def _alphanumeric(value: str) -> str:
    return "".join(char for char in value.casefold() if char.isalnum())


def _has_snowbank_name(
    material: RbrMaterial | None,
    rules: SnowbankRules = SnowbankRules(),
) -> bool:
    if material is None:
        return False
    names = [_alphanumeric(material.name)]
    if material.diffuse_texture:
        names.append(_alphanumeric(material.diffuse_texture.stem))
    return any(
        (normalized := _alphanumeric(name_match))
        and any(normalized in name for name in names)
        for name_match in rules.names
    )


def _asset_may_have_snowwall_collision(
    asset: MeshAsset,
    stage: RbrStage,
    materials_by_name: dict[str, RbrMaterial],
) -> bool:
    return any(
        _is_snowbank_texture(
            materials_by_name.get(part.material_name.casefold()),
            stage,
        )
        for part in asset.parts
    )


# Collidable ground painted entirely Snowwall is the firm floor of a snowbank,
# not part of it, where collision added to a non-colliding bank lies at most
# this far above it.
_SNOWBANK_FLOOR_DEPTH = 2.0


def _world_triangles(part: MeshPart, matrices: np.ndarray) -> np.ndarray:
    faces = np.asarray(part.faces, dtype=np.int64).reshape((-1, 3))
    corners = np.asarray(part.vertices, dtype=np.float64)[faces]
    homogeneous = np.concatenate(
        (corners, np.ones((*corners.shape[:2], 1), dtype=np.float64)),
        axis=2,
    )
    return np.concatenate(
        [(homogeneous @ np.asarray(matrix, dtype=np.float64).T)[:, :, :3] for matrix in matrices]
    )


def _expand_ranges(counts: np.ndarray) -> np.ndarray:
    """Position of every element within its run, for runs of the given lengths."""
    return np.arange(int(counts.sum())) - np.repeat(np.cumsum(counts) - counts, counts)


class _BankIndex:
    """Added bank triangles bucketed by the 2 m ground cells they overlap."""

    cell_size = 2.0

    def __init__(self, banks: np.ndarray) -> None:
        low = np.floor(banks[:, :, :2].min(axis=1) / self.cell_size).astype(np.int64)
        high = np.floor(banks[:, :, :2].max(axis=1) / self.cell_size).astype(np.int64)
        spans = high - low + 1
        counts = spans[:, 0] * spans[:, 1]
        offsets = _expand_ranges(counts)
        width = np.repeat(spans[:, 0], counts)
        cells = np.stack(
            (
                np.repeat(low[:, 0], counts) + offsets % width,
                np.repeat(low[:, 1], counts) + offsets // width,
            ),
            axis=1,
        )
        self.origin = cells.min(axis=0)
        self.shape = cells.max(axis=0) - self.origin + 1
        keys = self._keys(cells)
        order = np.argsort(keys, kind="stable")
        self.keys = keys[order]
        self.bank_index = np.repeat(np.arange(len(banks)), counts)[order]
        self.banks = banks

    def _keys(self, cells: np.ndarray) -> np.ndarray:
        relative = cells - self.origin
        return relative[:, 0] * self.shape[1] + relative[:, 1]

    def covered(self, floors: np.ndarray) -> np.ndarray:
        centroids = floors.mean(axis=1)
        cells = np.floor(centroids[:, :2] / self.cell_size).astype(np.int64)
        relative = cells - self.origin
        candidates = np.flatnonzero(
            np.all((relative >= 0) & (relative < self.shape), axis=1)
        )
        keys = self._keys(cells[candidates])
        start = np.searchsorted(self.keys, keys, side="left")
        matches = np.searchsorted(self.keys, keys, side="right") - start
        pair_floor = np.repeat(candidates, matches)
        pair_bank = self.bank_index[np.repeat(start, matches) + _expand_ranges(matches)]
        a, b, c = (self.banks[pair_bank, corner] for corner in range(3))
        point = centroids[pair_floor]
        first, second, offset = b[:, :2] - a[:, :2], c[:, :2] - a[:, :2], point[:, :2] - a[:, :2]
        determinant = first[:, 0] * second[:, 1] - second[:, 0] * first[:, 1]
        valid = np.abs(determinant) >= 1e-12
        determinant = np.where(valid, determinant, 1.0)
        u = (offset[:, 0] * second[:, 1] - second[:, 0] * offset[:, 1]) / determinant
        v = (first[:, 0] * offset[:, 1] - offset[:, 0] * first[:, 1]) / determinant
        height = a[:, 2] + u * (b[:, 2] - a[:, 2]) + v * (c[:, 2] - a[:, 2]) - point[:, 2]
        hit = (
            valid
            & (u >= 0.0)
            & (v >= 0.0)
            & (u + v <= 1.0)
            & (height >= 0.0)
            & (height <= _SNOWBANK_FLOOR_DEPTH)
        )
        covered = np.zeros(len(floors), dtype=bool)
        covered[pair_floor[hit]] = True
        return covered


def _gets_added_bank_collision(
    source: Path,
    material: RbrMaterial | None,
    stage: RbrStage,
    rules: SnowbankRules,
) -> bool:
    return bool(
        rules.use_surfaces
        and _is_snowbank_texture(material, stage)
        or rules.use_names
        and _snowwall_name_candidate(source, material, rules)
    )


def _added_bank_index(
    stage: RbrStage,
    loader: AssimpMeshLoader,
    rules: SnowbankRules,
) -> _BankIndex | None:
    if not any(is_snowwall_surface_map(surface_map) for surface_map in stage.surface_maps.values()):
        return None
    materials_by_name = {
        material.name.casefold(): material for material in stage.materials
    }
    selections: dict[Path, list[MeshPart]] = {}
    banks: list[np.ndarray] = []
    for stage_object in stage.objects:
        source, mode, _collision_box = _asset_key(stage_object)
        if mode != "noncollidable":
            continue
        selected = selections.get(source)
        if selected is None:
            selected = [
                part
                for part in loader.load(source, geometry_only=True).parts
                if _gets_added_bank_collision(
                    source,
                    materials_by_name.get(part.material_name.casefold()),
                    stage,
                    rules,
                )
            ]
            selections[source] = selected
        if not selected:
            continue
        matrices = convert_transform_matrices(
            load_transforms(stage_object),
            np.zeros(3, dtype=np.float64),
            0.0,
        )
        banks.extend(_world_triangles(part, matrices) for part in selected)
    return _BankIndex(np.concatenate(banks)) if banks else None


def _mark_snowbank_floors(
    asset: MeshAsset,
    matrices: np.ndarray,
    banks: _BankIndex,
    stage: RbrStage,
    materials_by_name: dict[str, RbrMaterial],
) -> tuple[MeshAsset, int]:
    parts: list[MeshPart] = []
    floor_faces = 0
    for part in asset.parts:
        material = materials_by_name.get(part.material_name.casefold())
        surface_map = (
            stage.surface_maps.get(surface_map_key(material.diffuse_texture))
            if material and material.diffuse_texture
            else None
        )
        faces = np.asarray(part.faces, dtype=np.int64).reshape((-1, 3))
        if surface_map is None or not is_snowwall_surface_map(surface_map) or not len(faces):
            parts.append(part)
            continue
        covered = banks.covered(
            _world_triangles(part, matrices)
        ).reshape((len(matrices), len(faces))).any(axis=0)
        if not covered.any():
            parts.append(part)
            continue
        for mask, floor in ((~covered, False), (covered, True)):
            if mask.any():
                used, new_faces = np.unique(faces[mask], return_inverse=True)
                parts.append(
                    remap_mesh_part(part, used, new_faces, snowbank_floor=floor)
                )
        floor_faces += int(covered.sum())
    return replace(asset, parts=parts), floor_faces


def _split_water_components(
    part: MeshPart,
    appearance: WaterAppearance | None = None,
) -> list[tuple[MeshPart, WaterRegion]]:
    faces = np.asarray(part.faces, dtype=np.uint32).reshape((-1, 3))
    if len(faces) == 0:
        return []

    faces_by_vertex: dict[int, list[int]] = {}
    for face_index, face in enumerate(faces):
        for vertex_index in face:
            faces_by_vertex.setdefault(int(vertex_index), []).append(face_index)

    visited = np.zeros(len(faces), dtype=bool)
    result: list[tuple[MeshPart, WaterRegion]] = []
    for first_face in range(len(faces)):
        if visited[first_face]:
            continue
        visited[first_face] = True
        pending = [first_face]
        component: list[int] = []
        while pending:
            face_index = pending.pop()
            component.append(face_index)
            for vertex_index in faces[face_index]:
                for neighbour in faces_by_vertex[int(vertex_index)]:
                    if not visited[neighbour]:
                        visited[neighbour] = True
                        pending.append(neighbour)

        component_faces = faces[np.asarray(component, dtype=np.uint32)]
        used_vertices, compact_faces = np.unique(component_faces, return_inverse=True)
        compact_faces = compact_faces.reshape((-1, 3)).astype(np.uint32)
        component_part = remap_mesh_part(
            part,
            used_vertices,
            compact_faces,
            name=f"{part.name}_water_{len(result)}",
            water=True,
        )
        result.append(
            (
                component_part,
                WaterRegion(
                    material_name=part.material_name,
                    vertices=component_part.vertices,
                    faces=component_part.faces,
                    source_ids=(),
                    appearance=appearance,
                ),
            )
        )
    return result


def _merge_flat_water_patches(patches: list[WaterRegion]) -> list[WaterRegion]:
    result = list(patches)
    while True:
        merged = False
        for first_index, first in enumerate(result):
            first_vertices = np.asarray(first.vertices, dtype=np.float64)
            first_min = first_vertices.min(axis=0)
            first_max = first_vertices.max(axis=0)
            for second_index in range(first_index + 1, len(result)):
                second = result[second_index]
                if (
                    first.material_name != second.material_name
                    or first.source_ids != second.source_ids
                    or first.appearance != second.appearance
                ):
                    continue
                second_vertices = np.asarray(second.vertices, dtype=np.float64)
                second_min = second_vertices.min(axis=0)
                second_max = second_vertices.max(axis=0)
                if abs(first_min[2] - second_min[2]) > 0.01:
                    continue
                same_y = np.allclose(
                    first_min[1:2],
                    second_min[1:2],
                    atol=1e-6,
                ) and np.allclose(
                    first_max[1:2],
                    second_max[1:2],
                    atol=1e-6,
                )
                same_x = abs(first_min[0] - second_min[0]) <= 1e-6 and abs(
                    first_max[0] - second_max[0]
                ) <= 1e-6
                connects_x = max(first_min[0], second_min[0]) <= min(
                    first_max[0], second_max[0]
                ) + 1e-6
                connects_y = max(first_min[1], second_min[1]) <= min(
                    first_max[1], second_max[1]
                ) + 1e-6
                if not (same_y and connects_x or same_x and connects_y):
                    continue
                bounds_min = np.minimum(first_min, second_min)
                bounds_max = np.maximum(first_max, second_max)
                z = float((first_min[2] + second_min[2]) * 0.5)
                vertices = np.asarray(
                    (
                        (bounds_min[0], bounds_min[1], z),
                        (bounds_max[0], bounds_min[1], z),
                        (bounds_max[0], bounds_max[1], z),
                        (bounds_min[0], bounds_max[1], z),
                    ),
                    dtype=np.float64,
                )
                result[first_index] = replace(first, vertices=vertices)
                del result[second_index]
                merged = True
                break
            if merged:
                break
        if not merged:
            return result


def _clip_water_triangle(
    triangle: np.ndarray,
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
) -> np.ndarray:
    polygon = [vertex.astype(np.float64) for vertex in triangle]
    for axis, boundary, keep_above in (
        (0, bounds_min[0], True),
        (0, bounds_max[0], False),
        (1, bounds_min[1], True),
        (1, bounds_max[1], False),
    ):
        if not polygon:
            break
        clipped: list[np.ndarray] = []
        previous = polygon[-1]
        previous_inside = previous[axis] >= boundary if keep_above else previous[axis] <= boundary
        for current in polygon:
            current_inside = current[axis] >= boundary if keep_above else current[axis] <= boundary
            if current_inside != previous_inside:
                distance = current[axis] - previous[axis]
                if abs(distance) > 1e-9:
                    factor = (boundary - previous[axis]) / distance
                    clipped.append(previous + (current - previous) * factor)
            if current_inside:
                clipped.append(current)
            previous = current
            previous_inside = current_inside
        polygon = clipped
    return np.asarray(polygon, dtype=np.float64)


def _water_block_patches(
    region: WaterRegion,
    minimum_cell_size: float = 8.0,
) -> list[WaterRegion]:
    vertices = np.asarray(region.vertices, dtype=np.float64)
    faces = np.asarray(region.faces, dtype=np.uint32).reshape((-1, 3))
    triangles = vertices[faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-8
    if not np.any(valid) or np.any(np.abs(normals[valid, 2] / lengths[valid]) < 0.9):
        return []

    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    width, height = bounds_max[:2] - bounds_min[:2]
    if width < 0.25 or height < 0.25:
        return []
    cell_size = max(
        minimum_cell_size,
        float(np.sqrt(width * height / 32.0)),
        float(width / 16.0),
        float(height / 16.0),
    )
    columns = max(1, int(np.ceil(width / cell_size)))
    rows = max(1, int(np.ceil(height / cell_size)))
    result: list[WaterRegion] = []
    for row in range(rows):
        for column in range(columns):
            cell_min = np.array(
                (
                    bounds_min[0] + column * cell_size,
                    bounds_min[1] + row * cell_size,
                )
            )
            cell_max = np.minimum(cell_min + cell_size, bounds_max[:2])
            clipped_points: list[np.ndarray] = []
            for triangle in triangles:
                if (
                    triangle[:, 0].max() < cell_min[0]
                    or triangle[:, 0].min() > cell_max[0]
                    or triangle[:, 1].max() < cell_min[1]
                    or triangle[:, 1].min() > cell_max[1]
                ):
                    continue
                clipped = _clip_water_triangle(triangle, cell_min, cell_max)
                if len(clipped) < 3:
                    continue
                area = 0.5 * abs(
                    np.dot(clipped[:, 0], np.roll(clipped[:, 1], 1))
                    - np.dot(clipped[:, 1], np.roll(clipped[:, 0], 1))
                )
                if area > 0.01:
                    clipped_points.extend(clipped)
            if not clipped_points:
                continue
            points = np.asarray(clipped_points)
            patch_min = points[:, :2].min(axis=0)
            patch_max = points[:, :2].max(axis=0)
            for axis in range(2):
                if (
                    cell_min[axis] > bounds_min[axis] + 1e-6
                    and abs(patch_min[axis] - cell_min[axis]) <= 1e-6
                ):
                    patch_min[axis] -= 0.05
                if (
                    cell_max[axis] < bounds_max[axis] - 1e-6
                    and abs(patch_max[axis] - cell_max[axis]) <= 1e-6
                ):
                    patch_max[axis] += 0.05
            patch_min = np.maximum(patch_min, bounds_min[:2])
            patch_max = np.minimum(patch_max, bounds_max[:2])
            if np.any(patch_max - patch_min < 0.25):
                continue
            z = float(np.median(points[:, 2]))
            patch_vertices = np.array(
                (
                    (patch_min[0], patch_min[1], z),
                    (patch_max[0], patch_min[1], z),
                    (patch_max[0], patch_max[1], z),
                    (patch_min[0], patch_max[1], z),
                ),
                dtype=np.float64,
            )
            result.append(
                WaterRegion(
                    material_name=region.material_name,
                    vertices=patch_vertices,
                    faces=np.array(((0, 1, 2), (0, 2, 3)), dtype=np.uint32),
                    tiled=True,
                    source_ids=region.source_ids,
                    appearance=region.appearance,
                )
            )
    return _merge_flat_water_patches(result)


def _transform_water_region(
    region: WaterRegion,
    matrix: np.ndarray,
) -> WaterRegion:
    vertices = np.asarray(region.vertices, dtype=np.float64)
    homogeneous = np.column_stack(
        (vertices, np.ones(len(vertices), dtype=np.float64))
    )
    return replace(
        region,
        vertices=(homogeneous @ np.asarray(matrix, dtype=np.float64).T)[:, :3],
    )


def _combine_water_regions(regions: list[WaterRegion]) -> WaterRegion:
    vertices: list[np.ndarray] = []
    faces: list[np.ndarray] = []
    vertex_offset = 0
    for region in regions:
        region_vertices = np.asarray(region.vertices, dtype=np.float64)
        region_faces = np.asarray(region.faces, dtype=np.uint32).reshape((-1, 3))
        vertices.append(region_vertices)
        faces.append(region_faces + vertex_offset)
        vertex_offset += len(region_vertices)
    return WaterRegion(
        material_name=regions[0].material_name,
        vertices=np.concatenate(vertices),
        faces=np.concatenate(faces),
        source_ids=tuple(
            sorted(
                {
                    source_id
                    for region in regions
                    for source_id in region.source_ids
                }
            )
        ),
        appearance=regions[0].appearance,
    )


def _water_regions_touch(
    first: WaterRegion,
    second: WaterRegion,
    *,
    horizontal_tolerance: float = 1.0,
    vertical_tolerance: float = 2.5,
) -> bool:
    if first.appearance != second.appearance:
        return False
    first_vertices = np.asarray(first.vertices, dtype=np.float64)
    second_vertices = np.asarray(second.vertices, dtype=np.float64)
    first_min = first_vertices.min(axis=0)
    first_max = first_vertices.max(axis=0)
    second_min = second_vertices.min(axis=0)
    second_max = second_vertices.max(axis=0)
    horizontal_gap = np.maximum(
        np.maximum(first_min[:2] - second_max[:2], second_min[:2] - first_max[:2]),
        0.0,
    )
    if float(np.linalg.norm(horizontal_gap)) > horizontal_tolerance:
        return False
    delta = first_vertices[:, None, :] - second_vertices[None, :, :]
    close_xy = np.linalg.norm(delta[:, :, :2], axis=2) <= horizontal_tolerance
    return bool(np.any(close_xy & (np.abs(delta[:, :, 2]) <= vertical_tolerance)))


def _group_water_regions(
    regions: list[tuple[str, WaterRegion]],
) -> list[list[tuple[str, WaterRegion]]]:
    parent = list(range(len(regions)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = root(first)
        second_root = root(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for first_index, (_first_name, first_region) in enumerate(regions):
        for second_index in range(first_index + 1, len(regions)):
            if _water_regions_touch(first_region, regions[second_index][1]):
                union(first_index, second_index)

    grouped: dict[int, list[tuple[str, WaterRegion]]] = {}
    for index, region in enumerate(regions):
        grouped.setdefault(root(index), []).append(region)
    return list(grouped.values())


def _parts_bounds(parts: list[MeshPart]) -> tuple[np.ndarray, np.ndarray]:
    bounds_min = np.full(3, math.inf, dtype=np.float64)
    bounds_max = np.full(3, -math.inf, dtype=np.float64)
    for part in parts:
        vertices = np.asarray(part.vertices, dtype=np.float64)
        vertices = vertices[np.all(np.isfinite(vertices), axis=1)]
        if len(vertices) == 0:
            continue
        bounds_min = np.minimum(bounds_min, vertices.min(axis=0))
        bounds_max = np.maximum(bounds_max, vertices.max(axis=0))
    return bounds_min, bounds_max


def _aabb_distance_to_origin(
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    matrix: np.ndarray,
) -> float:
    item_min, item_max = _transform_bounds(bounds_min, bounds_max, matrix[None])
    closest = np.clip(np.zeros(3, dtype=np.float64), item_min[0], item_max[0])
    return float(np.linalg.norm(closest))


def _filter_objects_near_spawn(
    stage: RbrStage,
    origin: np.ndarray,
    radius: float,
    map_yaw_degrees: float = 0.0,
) -> list:
    """Keep objects that overlap the preview sphere around spawn.

    BTB terrain uses world-space meshes with a shared far translation, so translation
    distance alone drops all ground. Terrain is kept when its AABB overlaps the sphere.
    """
    keep_distance = radius + 100.0
    keep_distance_sq = keep_distance * keep_distance
    loader = AssimpMeshLoader()
    bounds_cache: dict[Path, tuple[np.ndarray, np.ndarray]] = {}
    kept = []
    for stage_object in stage.objects:
        matrices = convert_transform_matrices(
            load_transforms(stage_object),
            origin,
            map_yaw_degrees,
        )
        if is_btb_terrain_mesh(stage_object.mesh_path):
            bounds = bounds_cache.get(stage_object.mesh_path)
            if bounds is None:
                bounds = _parts_bounds(loader.load(stage_object.mesh_path).parts)
                bounds_cache[stage_object.mesh_path] = bounds
            if any(
                _aabb_distance_to_origin(bounds[0], bounds[1], matrix) <= radius
                for matrix in matrices
            ):
                kept.append(stage_object)
            continue
        if any(
            float(np.sum(np.asarray(matrix[:3, 3], dtype=np.float64) ** 2))
            <= keep_distance_sq
            for matrix in matrices
        ):
            kept.append(stage_object)
    return kept


def _clip_part_to_radius(
    part: MeshPart,
    matrices: np.ndarray,
    radius: float,
) -> MeshPart | None:
    vertices = np.asarray(part.vertices, dtype=np.float64)
    faces = np.asarray(part.faces, dtype=np.int64).reshape((-1, 3))
    if len(vertices) == 0 or len(faces) == 0 or len(matrices) == 0:
        return None
    radius_sq = radius * radius
    homogeneous = np.column_stack(
        (vertices, np.ones(len(vertices), dtype=np.float64))
    )
    near_face = np.zeros(len(faces), dtype=bool)
    for matrix in matrices:
        world = (homogeneous @ np.asarray(matrix, dtype=np.float64).T)[:, :3]
        vertex_near = np.sum(world * world, axis=1) <= radius_sq
        near_face |= vertex_near[faces].any(axis=1)
        if bool(near_face.all()):
            break
    kept_faces = faces[near_face]
    if len(kept_faces) == 0:
        return None
    used, new_faces = np.unique(kept_faces, return_inverse=True)
    return remap_mesh_part(
        part,
        used,
        new_faces,
    )


def _clip_parts_to_radius(
    parts: list[MeshPart],
    matrices: np.ndarray,
    radius: float,
) -> list[MeshPart]:
    return [
        kept
        for part in parts
        if (kept := _clip_part_to_radius(part, matrices, radius)) is not None
    ]


def _source_matrices(
    stage_objects: list[StageObject],
    origin: np.ndarray,
    map_yaw_degrees: float = 0.0,
) -> np.ndarray:
    matrices: list[np.ndarray] = []
    for stage_object in stage_objects:
        matrices.extend(
            convert_transform_matrices(
                load_transforms(stage_object),
                origin,
                map_yaw_degrees,
            )
        )
    if not matrices:
        return np.zeros((0, 4, 4), dtype=np.float64)
    return np.asarray(matrices, dtype=np.float64)


def _transform_bounds(bounds_min: np.ndarray, bounds_max: np.ndarray, matrices: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World AABB minimum and maximum of each ``(n, 4, 4)`` matrix's placed bounds."""
    corners = np.array(
        list(
            itertools.product(
                (bounds_min[0], bounds_max[0]),
                (bounds_min[1], bounds_max[1]),
                (bounds_min[2], bounds_max[2]),
            )
        ),
        dtype=np.float64,
    )
    homogeneous = np.column_stack((corners, np.ones(corners.shape[0], dtype=np.float64)))
    transformed = homogeneous @ matrices.transpose(0, 2, 1)
    return transformed[..., :3].min(axis=1), transformed[..., :3].max(axis=1)


def _water_surface_height(
    triangles: np.ndarray,
    point: np.ndarray,
) -> float | None:
    heights = _surface_heights(triangles, point)
    return float(np.median(heights)) if len(heights) else None


def _surface_heights(triangles: np.ndarray, point: np.ndarray) -> np.ndarray:
    """Heights of the triangles straight above or below the XY point."""
    first = triangles[:, 0, :2]
    first_edge = triangles[:, 1, :2] - first
    second_edge = triangles[:, 2, :2] - first
    relative = point - first
    denominator = (
        first_edge[:, 0] * second_edge[:, 1]
        - first_edge[:, 1] * second_edge[:, 0]
    )
    valid = np.abs(denominator) > 1e-9
    first_weight = np.zeros(len(triangles), dtype=np.float64)
    second_weight = np.zeros(len(triangles), dtype=np.float64)
    first_weight[valid] = (
        relative[valid, 0] * second_edge[valid, 1]
        - relative[valid, 1] * second_edge[valid, 0]
    ) / denominator[valid]
    second_weight[valid] = (
        first_edge[valid, 0] * relative[valid, 1]
        - first_edge[valid, 1] * relative[valid, 0]
    ) / denominator[valid]
    contains = (
        valid
        & (first_weight >= -1e-6)
        & (second_weight >= -1e-6)
        & (first_weight + second_weight <= 1.0 + 1e-6)
    )
    return (
        triangles[contains, 0, 2] * (1.0 - first_weight[contains] - second_weight[contains])
        + triangles[contains, 1, 2] * first_weight[contains]
        + triangles[contains, 2, 2] * second_weight[contains]
    )


def is_source_sky(part: MeshPart, matrices: np.ndarray, route_positions: np.ndarray) -> bool:
    """Whether the part, placed at each of the matrices, passes far over a
    good share of the route, as old sky domes and cloud layers do. Trees,
    bridges and tunnels only ever cover a short stretch of it."""
    vertices = np.asarray(part.vertices, dtype=np.float64)
    faces = np.asarray(part.faces, dtype=np.int64).reshape((-1, 3))
    route = route_positions[:: max(1, len(route_positions) // _SKY_ROUTE_SAMPLES)]
    if not len(vertices) or not len(faces) or not len(route) or not len(matrices):
        return False
    needed = _SKY_MIN_ROUTE_SHARE * len(route)
    instance_min, instance_max = _transform_bounds(vertices.min(axis=0), vertices.max(axis=0), matrices)
    under = (
        np.all(route[None, :, :2] >= instance_min[:, None, :2], axis=2)
        & np.all(route[None, :, :2] <= instance_max[:, None, :2], axis=2)
        & (route[None, :, 2] + _SKY_MIN_HEIGHT <= instance_max[:, None, 2])
    )
    if np.count_nonzero(under.any(axis=0)) < needed:
        return False
    local_triangles = vertices[faces]
    covered = np.zeros(len(route), dtype=bool)
    for matrix, candidates in zip(matrices, under):
        candidates &= ~covered
        if not candidates.any():
            continue
        triangles = local_triangles @ matrix[:3, :3].T + matrix[:3, 3]
        low = triangles[:, :, :2].min(axis=1)
        high = triangles[:, :, :2].max(axis=1)
        top = triangles[:, :, 2].max(axis=1)
        for index in np.flatnonzero(candidates):
            point = route[index]
            clearance = point[2] + _SKY_MIN_HEIGHT
            nearby = np.all(low <= point[:2], axis=1) & np.all(high >= point[:2], axis=1) & (top >= clearance)
            covered[index] = bool(np.any(_surface_heights(triangles[nearby], point[:2]) >= clearance))
    return np.count_nonzero(covered) >= needed


def _river_water(
    region: WaterRegion,
    name: str,
    route_positions: np.ndarray | None = None,
) -> WaterSpec | None:
    vertices = np.asarray(region.vertices, dtype=np.float64)
    faces = np.asarray(region.faces, dtype=np.uint32).reshape((-1, 3))
    if len(vertices) < 4 or not len(faces):
        return None
    triangles = vertices[faces]
    normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-8
    if not np.any(valid):
        return None
    horizontal_ratio = float(
        np.mean(np.abs(normals[valid, 2] / lengths[valid]) > 0.75)
    )
    if horizontal_ratio < 0.75:
        return None

    xy = vertices[:, :2]
    source_center = xy.mean(axis=0)
    centered = xy - source_center
    covariance = np.cov(centered.T)
    if not np.all(np.isfinite(covariance)):
        return None
    _values, vectors = np.linalg.eigh(covariance)
    axis = vectors[:, -1]
    side = np.asarray((-axis[1], axis[0]))
    along = centered @ axis
    across = centered @ side
    minimum = float(along.min())
    maximum = float(along.max())
    river_length = maximum - minimum
    river_width = float(across.max() - across.min())
    aspect = river_length / max(river_width, 1e-6)
    if river_length < 20.0 or aspect < 4.0:
        return None

    route_xy = np.asarray(
        route_positions if route_positions is not None else (),
        dtype=np.float64,
    ).reshape((-1, 3))[:, :2]

    def distance_to_route(point: np.ndarray) -> float:
        if not len(route_xy):
            return math.inf
        if len(route_xy) == 1:
            return float(np.linalg.norm(point - route_xy[0]))
        starts = route_xy[:-1]
        segments = route_xy[1:] - starts
        lengths_squared = np.sum(segments * segments, axis=1)
        offsets = point - starts
        factors = np.divide(
            np.sum(offsets * segments, axis=1),
            lengths_squared,
            out=np.zeros(len(segments), dtype=np.float64),
            where=lengths_squared > 1e-12,
        )
        factors = np.clip(factors, 0.0, 1.0)
        nearest = starts + segments * factors[:, None]
        return float(np.min(np.linalg.norm(nearest - point, axis=1)))

    def sample_spacing(position: float) -> float:
        distances = np.abs(along - position)
        selected = distances <= 5.0
        if np.count_nonzero(selected) < 2:
            selected = np.zeros(len(vertices), dtype=bool)
            selected[np.argsort(distances)[:2]] = True
        lateral = float(np.median(across[selected]))
        center = source_center + axis * position + side * lateral
        route_distance = distance_to_route(center)
        if route_distance <= 5.0:
            return 0.5
        if route_distance <= 15.0:
            return 1.0
        return 10.0

    candidates = np.arange(minimum, maximum, 0.5, dtype=np.float64)
    if not len(candidates) or candidates[-1] < maximum - 1e-8:
        candidates = np.append(candidates, maximum)
    positions = [float(candidates[0])]
    last_position = positions[0]
    spacings = [sample_spacing(last_position)]
    for position in candidates[1:]:
        spacing = sample_spacing(float(position))
        if position - last_position + 1e-8 < spacing:
            continue
        positions.append(float(position))
        spacings.append(spacing)
        last_position = float(position)
    if positions[-1] < maximum - 1e-8:
        positions.append(maximum)
        spacings.append(sample_spacing(maximum))

    nodes: list[list[float]] = []
    for position, spacing in zip(positions, spacings):
        window = max(0.5, spacing * 0.75)
        distances = np.abs(along - position)
        height_selected = distances <= window
        if np.count_nonzero(height_selected) < 2:
            nearest = np.argsort(distances)[:2]
            height_selected = np.zeros(len(vertices), dtype=bool)
            height_selected[nearest] = True
        shape_selected = distances <= max(5.0, window)
        if np.count_nonzero(shape_selected) < 2:
            shape_selected = height_selected
        lateral = float(np.median(across[shape_selected]))
        center = source_center + axis * position + side * lateral
        node_z = _water_surface_height(triangles, center)
        if node_z is None:
            node_z = float(np.median(vertices[height_selected, 2]))
        node_width = max(
            2.0,
            float(
                across[shape_selected].max()
                - across[shape_selected].min()
            ),
        )
        nodes.append(
            [
                float(center[0]),
                float(center[1]),
                node_z,
                node_width,
                2.0,
                0.0,
                0.0,
                1.0,
            ]
        )
    if nodes[0][2] < nodes[-1][2]:
        nodes.reverse()
    return WaterSpec(
        name=name,
        kind="River",
        position=nodes[0][:3],
        scale=[1, 1, 1],
        nodes=nodes,
        source_ids=region.source_ids,
        appearance=region.appearance,
    )


def _minimum_water_rectangle(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[float], float] | None:
    xy = np.unique(np.asarray(vertices, dtype=np.float64)[:, :2], axis=0)
    if len(xy) < 3:
        return None
    ordered = xy[np.lexsort((xy[:, 1], xy[:, 0]))]

    def cross(origin: np.ndarray, first: np.ndarray, second: np.ndarray) -> float:
        return float(
            (first[0] - origin[0]) * (second[1] - origin[1])
            - (first[1] - origin[1]) * (second[0] - origin[0])
        )

    lower: list[np.ndarray] = []
    for point in ordered:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], point) <= 1e-9:
            lower.pop()
        lower.append(point)
    upper: list[np.ndarray] = []
    for point in reversed(ordered):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], point) <= 1e-9:
            upper.pop()
        upper.append(point)
    hull = np.asarray(lower[:-1] + upper[:-1], dtype=np.float64)
    if len(hull) < 3:
        return None

    best: tuple[float, np.ndarray, np.ndarray, np.ndarray] | None = None
    for index, point in enumerate(hull):
        edge = hull[(index + 1) % len(hull)] - point
        edge_length = float(np.linalg.norm(edge))
        if edge_length < 1e-9:
            continue
        axis = edge / edge_length
        side = np.asarray((-axis[1], axis[0]), dtype=np.float64)
        projected = xy @ np.column_stack((axis, side))
        projected_min = projected.min(axis=0)
        projected_max = projected.max(axis=0)
        size = projected_max - projected_min
        area = float(size[0] * size[1])
        if best is None or area < best[0]:
            best = (area, axis, side, (projected_min + projected_max) * 0.5)
    if best is None or best[0] < 1e-8:
        return None

    rectangle_area, axis, side, projected_center = best
    projected = xy @ np.column_stack((axis, side))
    size = projected.max(axis=0) - projected.min(axis=0)
    center = axis * projected_center[0] + side * projected_center[1]
    triangles = np.asarray(vertices, dtype=np.float64)[
        np.asarray(faces, dtype=np.uint32).reshape((-1, 3))
    ]
    first = triangles[:, 1, :2] - triangles[:, 0, :2]
    second = triangles[:, 2, :2] - triangles[:, 0, :2]
    projected_area = float(
        np.sum(np.abs(first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]))
        * 0.5
    )
    coverage = min(1.0, projected_area / rectangle_area)
    rotation = [
        float(axis[0]),
        float(axis[1]),
        0.0,
        float(side[0]),
        float(side[1]),
        0.0,
        0.0,
        0.0,
        1.0,
    ]
    return center, size, rotation, coverage


def _classify_water(
    region: WaterRegion,
    name: str,
    warnings: list[str],
    route_positions: np.ndarray | None = None,
) -> WaterSpec | None:
    vertices = np.asarray(region.vertices, dtype=np.float64)
    bounds_min = vertices.min(axis=0)
    bounds_max = vertices.max(axis=0)
    width, height = bounds_max[:2] - bounds_min[:2]
    if width < 0.25 or height < 0.25:
        return None
    z_range = float(bounds_max[2] - bounds_min[2])
    triangles = vertices[np.asarray(region.faces, dtype=np.uint32)]
    normals = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])
    lengths = np.linalg.norm(normals, axis=1)
    valid = lengths > 1e-8
    horizontal_ratio = float(np.mean(np.abs(normals[valid, 2] / lengths[valid]) > 0.75)) if np.any(valid) else 0
    river = _river_water(region, name, route_positions)
    if river is not None:
        return river
    rectangle = _minimum_water_rectangle(vertices, region.faces)
    if rectangle is None:
        return None
    rectangle_center, rectangle_size, rectangle_rotation, footprint_coverage = rectangle
    width, height = rectangle_size
    max_extent = float(max(width, height))
    z_tolerance = max(0.4, min(1.5, max_extent * 0.01))
    if z_range > z_tolerance or horizontal_ratio < 0.75:
        warnings.append(f"Water geometry {name} is not reliably planar and was left visual-only")
        return None
    surface_z = float(np.median(vertices[:, 2]))
    bounds_area = float(
        (bounds_max[0] - bounds_min[0])
        * (bounds_max[1] - bounds_min[1])
    )
    area = float(width * height)
    if bounds_area >= 1000000 and z_range <= 1.5:
        return WaterSpec(
            name=name,
            kind="WaterPlane",
            position=[0.0, 0.0, surface_z],
            scale=[1, 1, 1],
            source_ids=region.source_ids,
            appearance=region.appearance,
        )

    aspect = max(width, height) / max(1e-6, min(width, height))
    if aspect <= 6:
        if footprint_coverage < _MIN_WATER_RECTANGLE_COVERAGE:
            warnings.append(
                f"Water geometry {name} covers only "
                f"{footprint_coverage:.0%} of its fitted rectangle"
            )
            return None
        depth = 0.5 if area <= 100 else max(2.0, z_range + 2.0)
        return WaterSpec(
            name=name,
            kind="WaterBlock",
            position=[
                float(rectangle_center[0]),
                float(rectangle_center[1]),
                surface_z,
            ],
            scale=[float(width), float(height), depth],
            rotation=rectangle_rotation,
            source_ids=region.source_ids,
            grid_element_size=(
                max(0.1, min(0.5, float(min(width, height)) / 8.0))
                if region.tiled
                else None
            ),
            appearance=region.appearance,
        )
    warnings.append(f"Water geometry {name} is too irregular for a safe WaterBlock conversion")
    return None


def _classify_water_or_patches(
    region: WaterRegion,
    name: str,
    warnings: list[str] | None = None,
    route_positions: np.ndarray | None = None,
) -> list[WaterSpec]:
    classification_warnings: list[str] = []
    water = _classify_water(
        region,
        name,
        classification_warnings,
        route_positions,
    )
    if water is not None:
        return [water]
    rectangle = _minimum_water_rectangle(region.vertices, region.faces)
    minimum_cell_size = (
        4.0
        if (
            rectangle is not None
            and rectangle[3] < _MIN_WATER_RECTANGLE_COVERAGE
        )
        else 8.0
    )
    patches = _water_block_patches(region, minimum_cell_size)
    patch_waters = [
        _classify_water(patch, f"{name}_{index}", [])
        for index, patch in enumerate(patches)
    ]
    if patches and all(patch_waters):
        if warnings is not None:
            warnings.append(
                f"Approximated water {name} with "
                f"{len(patches)} WaterBlocks"
            )
        return [water for water in patch_waters if water is not None]
    if warnings is not None:
        warnings.extend(classification_warnings)
    return []


def _classify_water_or_fallback(
    region: WaterRegion,
    name: str,
    warnings: list[str] | None = None,
    route_positions: np.ndarray | None = None,
) -> list[WaterSpec]:
    waters = _classify_water_or_patches(
        region,
        name,
        warnings,
        route_positions,
    )
    if waters:
        return waters
    vertices = np.asarray(region.vertices, dtype=np.float64).copy()
    if len(vertices) < 3:
        return []
    vertices[:, 2] = float(np.median(vertices[:, 2]))
    flattened = replace(region, vertices=vertices, tiled=True)
    waters = _classify_water_or_patches(
        flattened,
        name,
        [],
        route_positions,
    )
    if waters and warnings is not None:
        warnings.append(
            f"Flattened transformed water {name} into "
            f"{len(waters)} procedural object{'s' if len(waters) != 1 else ''}"
        )
    return waters


def _convert_water_regions_detailed(
    regions: list[tuple[str, WaterRegion]],
    warnings: list[str],
    route_positions: np.ndarray | None = None,
) -> tuple[list[WaterSpec], tuple[str, ...]]:
    waters: list[WaterSpec] = []
    for group_index, group in enumerate(_group_water_regions(regions)):
        if len(group) > 1:
            combined = _combine_water_regions(
                [region for _name, region in group]
            )
            river = _river_water(
                combined,
                f"water_group_{group_index}",
                route_positions,
            )
            if river is not None:
                waters.append(river)
                warnings.append(
                    f"Merged {len(group)} connected water pieces into one River"
                )
                continue

        for name, region in group:
            waters.extend(
                _classify_water_or_fallback(
                    region,
                    name,
                    warnings,
                    route_positions,
                )
            )
    covered_source_ids = tuple(
        sorted(
            {
                source_id
                for water in waters
                for source_id in water.source_ids
            }
        )
    )
    return waters, covered_source_ids


def _convert_water_regions(
    regions: list[tuple[str, WaterRegion]],
    warnings: list[str],
    route_positions: np.ndarray | None = None,
) -> list[WaterSpec]:
    return _convert_water_regions_detailed(
        regions,
        warnings,
        route_positions,
    )[0]


def _is_forest_candidate(
    collision_model: int,
    visible: bool,
    instanced: bool,
    clone_count: int,
) -> bool:
    return (
        instanced
        and clone_count > 1
        and (collision_model == -1 or (collision_model == 2 and not visible))
    )


def _build_assets(
    payload: Path,
    level_id: str,
    stage: RbrStage,
    reporter: ProgressReporter,
    options: ConversionOptions,
    warnings: list[str],
    shape_archive: zipfile.ZipFile,
    origin: np.ndarray,
    route_positions: np.ndarray,
    map_yaw_degrees: float = 0.0,
) -> tuple[
    dict[
        AssetLodKey,
        tuple[str | None, np.ndarray, np.ndarray, bool, int] | None,
    ],
    dict[str, MaterialVariant],
    dict[AssetKey, list[WaterRegion]],
    dict[AssetKey, list[ThinWallTemplate]],
    int,
    int,
    dict[str, int],
]:
    filesystem = current_filesystem()
    water_name_matches = (
        options.water_name_matches if options.use_water_name_fallback else None
    )
    snowbank_rules = SnowbankRules.from_options(options)
    usage: dict[AssetKey, list[StageObject]] = {}
    for stage_object in stage.objects:
        usage.setdefault(_asset_key(stage_object), []).append(stage_object)
    asset_lod_bands, invalid_lod_ranges = _asset_lod_bands(
        stage,
        options.use_visual_lods,
    )
    if invalid_lod_ranges:
        warnings.append(
            f"Ignored {invalid_lod_ranges} invalid RX LOD ranges"
        )
    loader = AssimpMeshLoader()
    with profile_span("added_bank_index", category="conversion"):
        added_banks = _added_bank_index(stage, loader, snowbank_rules)
    output_dir = payload / "levels" / level_id / "art" / "shapes"
    filesystem.mkdir(output_dir, parents=True, exist_ok=True)
    materials_by_name = {
        material.name.casefold(): material for material in stage.materials
    }
    assets: dict[
        AssetLodKey,
        tuple[str | None, np.ndarray, np.ndarray, bool, int] | None,
    ] = {}
    material_variants: dict[str, MaterialVariant] = {}
    water_regions: dict[AssetKey, list[WaterRegion]] = {}
    thin_wall_templates: dict[AssetKey, list[ThinWallTemplate]] = {}
    collision_flag_conflicts = 0
    snowwall_collision_overrides = 0
    warned_name_classifications: set[tuple[str, str, str]] = set()
    asset_stats = {
        "hybridBakes": 0,
        "materials": 0,
        "vertexColors": 0,
        "vertexPbrBakes": 0,
        "fallbacks": 0,
        "waterNameFallbacks": 0,
        "lodShapes": 0,
        "collisionMergedAwayTriangles": 0,
        "sourceSkyboxPartsRemoved": 0,
        "snowwallCollisionOverrideParts": 0,
        "snowwallCollisionOverrideFaces": 0,
        "snowbankFloorFaces": 0,
        "overdrawnFacesRemoved": 0,
    }
    sorted_usage = sorted(
        usage,
        key=lambda item: (item[0].name.casefold(), item[1], item[2]),
    )
    for index, asset_key in enumerate(sorted_usage, 1):
        source, mode, collision_box = asset_key
        reporter.emit(
            "geometry",
            "Converting meshes",
            current=index,
            total=len(sorted_usage),
            detail=source.name,
        )
        with profile_span(
            "load_mesh",
            category="asset",
            source=str(source),
            mode=mode,
        ) as load_span:
            asset = loader.load(source)
            load_span.update(
                parts=len(asset.parts),
                vertices=sum(len(part.vertices) for part in asset.parts),
                faces=sum(len(part.faces) for part in asset.parts),
                sourceBytes=filesystem.stat(source).st_size,
            )
        if options.remove_source_skybox:
            matrices = _source_matrices(usage[asset_key], origin, map_yaw_degrees)
            kept_parts = [
                part
                for part in asset.parts
                if not is_source_sky(part, matrices, route_positions)
            ]
            asset_stats["sourceSkyboxPartsRemoved"] += len(asset.parts) - len(kept_parts)
            asset = replace(asset, parts=kept_parts)
        if added_banks is not None and mode in {"collidable", "collision"}:
            asset, floor_faces = _mark_snowbank_floors(
                asset,
                _source_matrices(usage[asset_key], np.zeros(3, dtype=np.float64)),
                added_banks,
                stage,
                materials_by_name,
            )
            asset_stats["snowbankFloorFaces"] += floor_faces
        with profile_span(
            "split_render_surfaces",
            category="asset",
            source=str(source),
            mode=mode,
        ) as split_span:
            render_asset = asset
            if mode != "collision":
                render_asset, overdrawn_faces = remove_overdrawn_faces(
                    asset,
                    materials_by_name,
                )
                asset_stats["overdrawnFacesRemoved"] += overdrawn_faces
            parts, variants = split_parts_by_surface(
                render_asset,
                stage,
                water_name_matches,
            )
            split_span.update(
                parts=len(parts),
                faces=sum(len(part.faces) for part in parts),
                variants=len(variants),
            )
        for material_name, variant in variants.items():
            if variant.classification_fallback is None:
                continue
            warning_key = (
                str(source),
                material_name,
                variant.classification_fallback,
            )
            if warning_key in warned_name_classifications:
                continue
            warned_name_classifications.add(warning_key)
            asset_stats["waterNameFallbacks"] += 1
            warnings.append(
                f"Material {material_name!r} on {source.name} has no MAT surface map; "
                f"using {variant.classification_fallback} classification"
            )
        if (
            mode == "noncollidable"
            and not _asset_may_have_snowwall_collision(
                asset,
                stage,
                materials_by_name,
            )
        ):
            collision_parts = parts
        else:
            with profile_span(
                "split_collision_surfaces",
                category="asset",
                source=str(source),
                mode=mode,
            ) as collision_span:
                (
                    collision_parts,
                    collision_variants,
                    asset_merged_triangles,
                ) = split_collision_parts_by_surface(
                    asset,
                    stage,
                    water_name_matches,
                )
                collision_span.update(
                    parts=len(collision_parts),
                    faces=sum(
                        len(part.faces)
                        for part in collision_parts
                    ),
                    variants=len(collision_variants),
                    mergedAwayTriangles=asset_merged_triangles,
                )
            variants.update(collision_variants)
            asset_stats["collisionMergedAwayTriangles"] += asset_merged_triangles
        if mode == "noncollidable" and any(
            _snowwall_collision_override_applies(
                source,
                variants[part.material_name],
                snowbank_rules,
                stage,
            )
            for part in collision_parts
        ):
            snowwall_collision_overrides += 1
        if mode == "noncollidable" and any(
            variants[part.material_name].source_surface_ids
            and (
                variants[part.material_name].hard
                or variants[part.material_name].bendable
                or variants[part.material_name].snowbank
            )
            and not _is_snowwall_collision_override(
                source,
                variants[part.material_name],
                snowbank_rules,
                stage,
            )
            for part in collision_parts
        ):
            collision_flag_conflicts += 1
        if options.preview_radius_m is not None:
            matrices = _source_matrices(
                usage[asset_key],
                origin,
                map_yaw_degrees,
            )
            parts = _clip_parts_to_radius(
                parts,
                matrices,
                options.preview_radius_m,
            )
            collision_parts = _clip_parts_to_radius(
                collision_parts,
                matrices,
                options.preview_radius_m,
            )
        if not parts:
            for lod_range in asset_lod_bands[asset_key]:
                assets[(*asset_key, lod_range)] = None
            water_regions[asset_key] = []
            continue

        bounds_min, bounds_max = _parts_bounds(parts)
        if not np.all(np.isfinite(bounds_min)) or not np.all(np.isfinite(bounds_max)):
            raise ConversionError(f"Mesh {source} has no finite vertices")

        procedural_regions: list[WaterRegion] = []
        output_parts: list[MeshPart] = []
        for part in parts:
            if not part.water:
                output_parts.append(part)
                continue
            with profile_span(
                "split_water_components",
                category="water",
                source=str(source),
                part=part.name,
                faces=len(part.faces),
            ) as water_part_span:
                material = variants[part.material_name].material
                appearance = water_appearance_for_source(
                    (
                        f"rx:{stage.metadata.folder_name}:material:{material.index}"
                        if material is not None
                        else None
                    ),
                    options.water_profiles,
                )
                components = _split_water_components(part, appearance)
                water_part_span.update(components=len(components))
            accepted_regions: list[WaterRegion] = []
            rejected_parts: list[MeshPart] = []
            for region_index, (component_part, region) in enumerate(components):
                name = f"{source.stem}_{part.name}_{region_index}"
                if _classify_water_or_fallback(region, name):
                    accepted_regions.append(region)
                else:
                    rejected_parts.append(component_part)
            if accepted_regions:
                procedural_regions.extend(accepted_regions)
                output_parts.extend(rejected_parts)
                if rejected_parts:
                    warnings.append(
                        f"Some water geometry in {source.name}:{part.name} could not be converted reliably; "
                        "keeping those source mesh sections"
                    )
            else:
                output_parts.append(part)
                warnings.append(
                    f"Water mesh {source.name}:{part.name} could not be converted reliably; keeping the original mesh"
                )

        collision_parts = _parts_for_asset_mode(
            collision_parts,
            variants,
            mode,
            source,
            snowbank_rules,
            stage,
        )
        if mode == "noncollidable":
            snowwall_parts = [
                part
                for part in collision_parts
                if part.collision_eligible
                and _is_snowwall_collision_override(
                    source,
                    variants[part.material_name],
                    snowbank_rules,
                    stage,
                )
            ]
            asset_stats["snowwallCollisionOverrideParts"] += len(
                snowwall_parts
            )
            asset_stats["snowwallCollisionOverrideFaces"] += sum(
                len(part.faces)
                for part in snowwall_parts
            )
        if collision_box:
            if not collision_parts:
                raise ConversionError(
                    f"Collision-box object {source.name} has no collision material"
                )
            box_template = collision_parts[0]
            effective_collision_box = (
                thicken_rbr_col_box_dimensions(collision_box)
                if options.inflate_thin_walls
                else collision_box
            )
            collision_parts = [
                rbr_col_box_part(box_template, effective_collision_box)
            ]
            if mode == "collision":
                output_parts = collision_parts
        if options.inflate_thin_walls and not collision_box:
            with profile_span(
                "detect_thin_walls",
                category="asset",
                source=str(source),
                mode=mode,
                collisionParts=len(collision_parts),
                collisionFaces=sum(
                    len(part.faces)
                    for part in collision_parts
                ),
            ) as thin_wall_span:
                collision_parts, asset_thin_wall_templates = (
                    extract_thin_wall_templates(collision_parts)
                )
                thin_wall_span.update(
                    templates=len(asset_thin_wall_templates),
                    retainedParts=len(collision_parts),
                )
        else:
            asset_thin_wall_templates = []
        thin_wall_templates[asset_key] = asset_thin_wall_templates
        if mode != "collision":
            baked_parts: list[MeshPart] = []
            for part_index, part in enumerate(output_parts):
                variant = variants[part.material_name]
                if not is_vertex_lerp_material(variant.material):
                    baked_parts.append(part)
                    continue
                try:
                    with profile_span(
                        "bake_vertex_lerp",
                        category="material",
                        source=str(source),
                        part=part.name,
                        material=part.material_name,
                        vertices=len(part.vertices),
                        faces=len(part.faces),
                    ) as bake_span:
                        baked = bake_vertex_lerp(
                            part,
                            variant.material,
                            payload.parent / "baked-textures",
                            source,
                        )
                        bake_span.update(method=baked.method)
                except ConversionError as exc:
                    asset_stats["fallbacks"] += 1
                    warnings.append(
                        f"Vertex-lerp fallback for {source.name}:{part.name}: {exc}"
                    )
                    baked_parts.append(part)
                    continue
                digest = hashlib.sha1(
                    f"{source}|{mode}|{part_index}|{part.material_name}".casefold().encode(
                        "utf-8"
                    )
                ).hexdigest()[:8]
                material_name = f"{part.material_name}_blend_{digest}"
                variants[material_name] = replace(
                    variant,
                    pbr_override=baked.material,
                )
                baked_parts.append(
                    replace(
                        baked.part,
                        material_name=material_name,
                    )
                )
                asset_stats["materials"] += 1
                if baked.method == "vertex":
                    asset_stats["vertexColors"] += 1
                elif baked.method == "vertex-pbr":
                    asset_stats["vertexPbrBakes"] += 1
                elif baked.method == "hybrid":
                    asset_stats["hybridBakes"] += 1
            output_parts = baked_parts
        if collision_box:
            bounds_min, bounds_max = _parts_bounds(
                [*output_parts, *collision_parts]
            )
        _show_retained_water_materials(output_parts, variants)
        if mode != "collision":
            output_parts = add_cutout_passes(
                output_parts,
                variants,
                foliage_name_matches=(
                    options.foliage_name_matches
                    if options.use_foliage_name_fallback
                    else ()
                ),
                foliage_ground_types=(
                    options.foliage_ground_types
                    if options.use_foliage_ground_types
                    else ()
                ),
            )
        used_materials = {
            part.material_name
            for part in [*output_parts, *collision_parts]
        } | {
            template.part.material_name
            for template in asset_thin_wall_templates
        }
        material_variants.update(
            {key: value for key, value in variants.items() if key in used_materials}
        )
        water_regions[asset_key] = procedural_regions
        has_collision = any(part.collision_eligible for part in collision_parts)
        for lod_range in sorted(
            asset_lod_bands[asset_key],
            key=repr,
        ):
            render_detail_size, null_detail_sizes = (
                _lod_detail_config(
                    bounds_min,
                    bounds_max,
                    lod_range,
                )
                if mode in {"collidable", "noncollidable"}
                else (None, ())
            )
            asset_stats["lodShapes"] += int(
                render_detail_size is not None
            )
            shape_vfs: str | None = None
            collision_triangles = 0
            if output_parts:
                target_name = _asset_name(
                    source,
                    mode,
                    lod_range,
                    collision_box,
                )
                with profile_span(
                    "write_collada",
                    category="collada",
                    source=str(source),
                    mode=mode,
                    lodRange=lod_range,
                    renderParts=len(output_parts),
                    renderFaces=sum(
                        len(part.faces)
                        for part in output_parts
                    ),
                    collisionParts=len(collision_parts),
                    collisionFaces=sum(
                        len(part.faces)
                        for part in collision_parts
                    ),
                ) as collada_span:
                    archive_name = f"levels/{level_id}/art/shapes/{target_name}"
                    _, _, collision_triangles = write_collada_to_archive(
                        shape_archive,
                        archive_name,
                        output_parts,
                        collision_only=mode == "collision",
                        collision_parts=collision_parts,
                        render_detail_size=render_detail_size,
                        null_detail_sizes=null_detail_sizes,
                    )
                    collada_span.update(
                        target=target_name,
                        collisionTriangles=collision_triangles,
                    )
                shape_vfs = f"/levels/{level_id}/art/shapes/{target_name}"
            assets[(*asset_key, lod_range)] = (
                shape_vfs,
                bounds_min,
                bounds_max,
                has_collision,
                collision_triangles,
            )
    if snowwall_collision_overrides:
        reporter.emit(
            "geometry",
            f"Enabled proper snowbank physics on {snowwall_collision_overrides} "
            "non-collidable terrain meshes with recognized snowbank surfaces or visuals",
        )
    if asset_stats["snowbankFloorFaces"]:
        reporter.emit(
            "geometry",
            f"Kept {asset_stats['snowbankFloorFaces']} snowbank floor triangles firm "
            "under added snowbank collision",
        )
    if collision_flag_conflicts:
        warnings.append(
            f"{collision_flag_conflicts} non-collidable source meshes reference "
            "hard or bendable physics surfaces; object-level CollisionModel was respected"
        )
    return (
        assets,
        material_variants,
        water_regions,
        thin_wall_templates,
        snowwall_collision_overrides,
        asset_stats,
    )


def _build_instances(
    stage: RbrStage,
    level_id: str,
    origin: np.ndarray,
    assets: dict[
        AssetLodKey,
        tuple[str | None, np.ndarray, np.ndarray, bool, int] | None,
    ],
    water_regions: dict[AssetKey, list[WaterRegion]],
    thin_wall_templates: dict[AssetKey, list[ThinWallTemplate]],
    route_positions: np.ndarray,
    reporter: ProgressReporter,
    warnings: list[str],
    preview_radius_m: float | None = None,
    use_visual_lods: bool = True,
    map_yaw_degrees: float = 0.0,
) -> tuple[
    list[StaticInstance],
    dict[str, ForestType],
    list[WaterSpec],
    list[MeshPart],
    dict[str, int],
    np.ndarray,
    np.ndarray,
    int,
]:
    statics: list[StaticInstance] = []
    forest_types: dict[str, ForestType] = {}
    waters: list[WaterSpec] = []
    world_water_regions: list[tuple[str, WaterRegion]] = []
    world_min = np.full(3, math.inf, dtype=np.float64)
    world_max = np.full(3, -math.inf, dtype=np.float64)
    warned_collision_models: set[int] = set()
    warned_shear = False
    skipped_transforms: list[str] = []
    static_collision_triangles = 0
    thin_wall_repairs: list[MeshPart] = []
    thin_wall_stats = {
        "detected": 0,
        "repaired": 0,
        "restored": 0,
        "removedFaces": 0,
        "generatedFaces": 0,
    }
    total_instances = sum(stage_object.clone_count for stage_object in stage.objects)
    moveable_objects = [
        stage_object
        for stage_object in stage.objects
        if getattr(stage_object, "moveable", False)
    ]
    if moveable_objects:
        warnings.append(
            f"{len(moveable_objects)} source Moveable object definitions were emitted as static objects; "
            "the RX format provides no target motion behavior"
        )
    shadow_disabled_objects = [
        stage_object
        for stage_object in stage.objects
        if getattr(stage_object, "shadow_caster", None) is False
    ]
    if shadow_disabled_objects:
        warnings.append(
            f"{len(shadow_disabled_objects)} source ShadowCaster=0 object definitions could not be preserved; "
            "BeamNG has no per-object shadow switch for TSStatic or forest instances"
        )
    processed = 0
    for object_index, stage_object in enumerate(stage.objects, 1):
        asset_key = _asset_key(stage_object)
        _source, mode, collision_box = asset_key
        raw = load_transforms(stage_object)
        lod_range = (
            _stage_object_lod_range(stage_object, raw)
            if use_visual_lods
            and mode in {"collidable", "noncollidable"}
            else (None, None)
        )
        asset_data = assets[(*asset_key, lod_range)]
        if asset_data is None:
            processed += stage_object.clone_count
            continue
        matrices = convert_transform_matrices(
            rbr_stuff_source_matrices(raw) if collision_box else raw,
            origin,
            map_yaw_degrees,
        )
        (
            shape_vfs,
            bounds_min,
            bounds_max,
            has_collision,
            collision_triangles,
        ) = asset_data
        object_water_regions = water_regions.get(asset_key, ())
        object_thin_walls = thin_wall_templates.get(asset_key, ())
        forest_candidate = _is_forest_candidate(
            stage_object.collision_model,
            stage_object.visible,
            stage_object.draw_instanced,
            stage_object.clone_count,
        )
        mesh_digest = hashlib.sha1(stage_object.mesh_path.name.casefold().encode("utf-8")).hexdigest()[:8]
        forest_key = f"{slugify(level_id)}_{slugify(stage_object.mesh_name)[:60]}_{mesh_digest}"
        if mode != "normal":
            forest_key += f"_{mode}"
        forest_key += "_" + hashlib.sha1(
            repr(lod_range).encode("ascii")
        ).hexdigest()[:6]
        if collision_box:
            forest_key += "_box_" + hashlib.sha1(
                repr(collision_box).encode("ascii")
            ).hexdigest()[:6]
        decomposed = decompose_matrices(matrices)
        item_mins, item_maxs = _transform_bounds(bounds_min, bounds_max, matrices)
        for transform_index, (matrix, transform) in enumerate(zip(matrices, decomposed)):
            processed += 1
            if processed == 1 or processed % 500 == 0 or processed == total_instances:
                reporter.emit(
                    "objects",
                    "Converting object instances",
                    current=processed,
                    total=total_instances,
                    detail=f"{object_index}/{len(stage.objects)} {stage_object.mesh_name}",
                )
            if transform is None:
                skipped_transforms.append(
                    f"{stage_object.index}:{transform_index} {stage_object.mesh_name}"
                )
                continue
            position, rotation, scale, forest_compatible = transform
            if (
                preview_radius_m is not None
                and _aabb_distance_to_origin(bounds_min, bounds_max, matrix)
                > preview_radius_m + 5.0
            ):
                continue
            if object_thin_walls:
                instance_repairs, instance_thin_wall_stats = (
                    inflate_thin_wall_templates(
                        object_thin_walls,
                        route_positions,
                        matrix=matrix,
                        name_prefix=(
                            f"thin_wall_{stage_object.index}_{transform_index}"
                        ),
                    )
                )
                thin_wall_repairs.extend(instance_repairs)
                for key in thin_wall_stats:
                    thin_wall_stats[key] += instance_thin_wall_stats[key]
            world_min = np.minimum(world_min, item_mins[transform_index])
            world_max = np.maximum(world_max, item_maxs[transform_index])
            for water_index, region in enumerate(object_water_regions):
                water_name = (
                    f"water_{stage_object.index}_{transform_index}_{water_index}"
                )
                world_water_regions.append(
                    (
                        water_name,
                        replace(
                            _transform_water_region(region, matrix),
                            source_ids=(water_name,),
                        ),
                    )
                )
            if shape_vfs is None:
                continue
            if forest_candidate and forest_compatible:
                collidable = stage_object.collision_model == 2 or has_collision
                forest_type = forest_types.setdefault(
                    forest_key,
                    ForestType(
                        key=forest_key,
                        shape_vfs=shape_vfs,
                        collidable=collidable,
                    ),
                )
                forest_type.items.append(
                    {
                        "ctxid": 0,
                        "pos": position,
                        "rotationMatrix": rotation,
                        "scale": abs(float(scale[0])),
                        "type": forest_key,
                    }
                )
                if collidable:
                    static_collision_triangles += collision_triangles
                continue
            if forest_candidate and not forest_compatible and not warned_shear:
                warnings.append("Some instanced transforms use non-uniform scale or shear and were emitted as TSStatic objects")
                warned_shear = True
            if stage_object.collision_model == -1:
                collision_type = "Collision Mesh" if has_collision else "None"
            elif stage_object.collision_model == 2:
                collision_type = "Collision Mesh"
            elif stage_object.collision_model == 0:
                collision_type = "Collision Mesh" if has_collision else "None"
            else:
                collision_type = "None"
                if stage_object.collision_model not in warned_collision_models:
                    warnings.append(
                        f"Unknown RBR CollisionModel={stage_object.collision_model}; using no collision"
                    )
                    warned_collision_models.add(stage_object.collision_model)
            if collision_type != "None":
                static_collision_triangles += collision_triangles
            statics.append(
                StaticInstance(
                    key=f"{stage_object.index}-{transform_index}",
                    shape_vfs=shape_vfs,
                    position=position,
                    rotation=rotation,
                    scale=scale,
                    visible=stage_object.visible,
                    collision_type=collision_type,
                )
            )
    with profile_span(
        "convert_world_water",
        category="water",
        regions=len(world_water_regions),
        vertices=sum(
            len(region.vertices)
            for _name, region in world_water_regions
        ),
        faces=sum(
            len(region.faces)
            for _name, region in world_water_regions
        ),
    ) as water_span:
        converted_waters = _convert_water_regions(
            world_water_regions,
            warnings,
            route_positions,
        )
        waters.extend(converted_waters)
        water_span.update(
            outputObjects=len(converted_waters),
            rivers=sum(
                water.kind == "River"
                for water in converted_waters
            ),
            waterBlocks=sum(
                water.kind == "WaterBlock"
                for water in converted_waters
            ),
            waterPlanes=sum(
                water.kind == "WaterPlane"
                for water in converted_waters
            ),
        )
    if skipped_transforms:
        examples = ", ".join(skipped_transforms[:5])
        suffix = "" if len(skipped_transforms) <= 5 else ", ..."
        warnings.append(
            f"Skipped {len(skipped_transforms)} object instances with malformed transforms: "
            f"{examples}{suffix}"
        )
    return (
        statics,
        forest_types,
        waters,
        thin_wall_repairs,
        thin_wall_stats,
        world_min,
        world_max,
        static_collision_triangles,
    )


def _convert(
    options: ConversionOptions,
    reporter: ProgressReporter | None = None,
) -> ConversionResult:
    reporter = reporter or ProgressReporter()
    filesystem = options.filesystem
    if original_stage_id(options.stage) is not None:
        from .original_converter import convert_original

        with profile_span(
            "original_conversion",
            category="conversion",
            stage=options.stage,
        ):
            return convert_original(options, reporter, _convert_water_regions)
    with profile_span(
        "resolve_stage",
        category="parse",
        selector=options.stage,
    ):
        stage_root = find_stage(
            filesystem.read_path(options.rbr_root),
            options.stage,
        )
    with profile_span(
        "load_stage",
        category="parse",
        path=str(stage_root),
    ) as stage_span:
        stage = load_stage(stage_root, reporter)
        stage_span.update(
            objects=len(stage.objects),
            materials=len(stage.materials),
            drivelinePoints=len(stage.driveline),
        )
    stage.location = location_override(
        stage.location,
        options.latitude,
        options.longitude,
    )
    environment_settings = resolve_environment_settings(
        stage.metadata.physics,
        options.temperature_night,
        options.temperature_day,
        location=stage.location,
        environment_variant=options.environment,
        environment_date=options.environment_date,
    )
    level_id = f"rbr_{slugify(stage.metadata.folder_name)}"
    destination = output_path_for(options, level_id)
    if filesystem.exists(destination) and not options.overwrite:
        raise ConversionError(f"Output already exists: {destination}. Use --overwrite to replace it.")
    pacenote_log, pacenote_notes = pacenote_log_writers(destination, filesystem)
    pacenote_visualizer_log = pacenote_visualizer_writer(
        destination,
        filesystem,
    )

    reporter.emit("prepare", "Preparing conversion", detail=stage.metadata.name)
    origin = source_position_to_beamng(stage.spawn.position)
    map_altitude_meters = resolved_map_altitude_meters(options, stage.location)
    origin[2] -= map_altitude_meters
    warnings = list(stage.warnings)
    warnings.extend(environment_settings.warnings)
    map_yaw_degrees, source_sun_bearing = _rx_map_yaw(
        stage,
        environment_settings,
        warnings,
    )
    if options.preview_radius_m is not None:
        before = len(stage.objects)
        stage.objects = _filter_objects_near_spawn(
            stage,
            origin,
            options.preview_radius_m,
            map_yaw_degrees,
        )
        terrain_kept = sum(
            1 for stage_object in stage.objects if is_btb_terrain_mesh(stage_object.mesh_path)
        )
        warnings.append(
            f"Preview radius {options.preview_radius_m:g}m around start: "
            f"kept {len(stage.objects)}/{before} objects "
            f"({terrain_kept} terrain)"
        )
    route_positions = rotate_z(
        np.asarray(
            [
                source_position_to_beamng(point.position) - origin
                for point in stage.driveline
            ],
            dtype=np.float64,
        ),
        map_yaw_degrees,
    )
    if not np.all(np.isfinite(route_positions)):
        raise ConversionError("Stage driveline contains non-finite positions")

    with temporary_workspace(level_id, filesystem) as temporary:
        temporary_root = filesystem.write_path(Path(temporary))
        payload = filesystem.write_path(temporary_root / "payload")
        filesystem.mkdir(payload)
        prepared_zip = filesystem.write_path(temporary_root / f"{level_id}.zip")
        with zipfile.ZipFile(
            prepared_zip,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=ZIP_COMPRESSLEVEL,
            allowZip64=True,
        ) as shape_archive:
            with profile_span(
                "build_assets",
                category="conversion",
                sourceObjects=len(stage.objects),
            ) as asset_span:
                (
                    assets,
                    materials,
                    water_regions,
                    thin_wall_templates,
                    snowwall_collision_overrides,
                    asset_stats,
                ) = _build_assets(
                    payload,
                    level_id,
                    stage,
                    reporter,
                    options,
                    warnings,
                    shape_archive,
                    origin,
                    route_positions,
                    map_yaw_degrees,
                )
                asset_span.update(
                    convertedShapes=sum(
                        asset is not None
                        for asset in assets.values()
                    ),
                    materialVariants=len(materials),
                    **asset_stats,
                )
        archived_paths = {
            asset[0].lstrip("/")
            for asset in assets.values()
            if asset is not None and asset[0] is not None
        }
        with profile_span(
            "build_instances",
            category="conversion",
            sourceObjects=len(stage.objects),
        ) as instance_span:
            (
                statics,
                forest_types,
                waters,
                thin_wall_repairs,
                thin_wall_stats,
                bounds_min,
                bounds_max,
                static_collision_triangles,
            ) = _build_instances(
                stage,
                level_id,
                origin,
                assets,
                water_regions,
                thin_wall_templates,
                route_positions,
                reporter,
                warnings,
                options.preview_radius_m,
                options.use_visual_lods,
                map_yaw_degrees,
            )
            instance_span.update(
                statics=len(statics),
                forestTypes=len(forest_types),
                waterObjects=len(waters),
                thinWallRepairs=len(thin_wall_repairs),
                collisionTriangles=static_collision_triangles,
            )
        if thin_wall_repairs:
            thin_wall_archive_name = (
                f"levels/{level_id}/art/shapes/thin_wall_collision.dae"
            )
            with profile_span(
                "write_thin_wall_collision",
                category="collada",
                parts=len(thin_wall_repairs),
            ):
                with zipfile.ZipFile(
                    filesystem.write_path(prepared_zip),
                    "a",
                    compression=zipfile.ZIP_DEFLATED,
                    compresslevel=ZIP_COMPRESSLEVEL,
                    allowZip64=True,
                ) as shape_archive:
                    _, _, repair_triangles = write_collada_to_archive(
                        shape_archive,
                        thin_wall_archive_name,
                        thin_wall_repairs,
                        collision_only=True,
                    )
            archived_paths.add(thin_wall_archive_name)
            statics.append(
                StaticInstance(
                    key="thin_wall_collision",
                    shape_vfs=f"/{thin_wall_archive_name}",
                    position=[0.0, 0.0, 0.0],
                    rotation=[
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                        0.0,
                        0.0,
                        0.0,
                        1.0,
                    ],
                    scale=[1.0, 1.0, 1.0],
                    visible=True,
                    collision_type="Collision Mesh",
                )
            )
            static_collision_triangles += repair_triangles
            if thin_wall_stats["repaired"]:
                warnings.append(
                    "Inflated "
                    f"{thin_wall_stats['repaired']} thin collision walls "
                    "outward to 0.5 m"
                )
            if thin_wall_stats["restored"]:
                warnings.append(
                    "Kept original collision for "
                    f"{thin_wall_stats['restored']} thin walls that could not "
                    "be inflated"
                )
        plugin_found = plugins.find(
            "find_rx",
            stage,
            origin,
            map_yaw_degrees,
            warnings,
        )
        with zipfile.ZipFile(
            filesystem.write_path(prepared_zip),
            "a",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=ZIP_COMPRESSLEVEL,
            allowZip64=True,
        ) as shape_archive:
            plugin_stats = plugins.write_level(
                PluginLevel(
                    archive=shape_archive,
                    archived_paths=archived_paths,
                    level_dir=payload / "levels" / level_id,
                    level_id=level_id,
                    origin=np.zeros(3, dtype=np.float64),
                    version_suffix=__version__.replace(".", "_"),
                    waters=waters,
                    prepared=None,
                    progress=lambda message: reporter.emit("geometry", message),
                ),
                plugin_found,
            )
        bounds_min = np.minimum(bounds_min, route_positions.min(axis=0))
        bounds_max = np.maximum(bounds_max, route_positions.max(axis=0))
        if not np.all(np.isfinite(bounds_min)) or not np.all(np.isfinite(bounds_max)):
            raise ConversionError("Converted stage has no finite world bounds")
        stats: dict[str, bool | int | float | str] = {
            "sourceFormat": "rx",
            "sourceObjects": len(stage.objects),
            "sourceInstances": sum(stage_object.clone_count for stage_object in stage.objects),
            "sourceMoveableObjects": sum(
                getattr(stage_object, "moveable", False)
                for stage_object in stage.objects
            ),
            "sourceShadowCasterDisabledObjects": sum(
                getattr(stage_object, "shadow_caster", None) is False
                for stage_object in stage.objects
            ),
            "uniqueMeshes": len({stage_object.mesh_path for stage_object in stage.objects}),
            "convertedShapes": sum(asset is not None for asset in assets.values()),
            "removedSkyboxShapes": sum(asset is None for asset in assets.values()),
            "staticInstances": len(statics),
            "forestInstances": sum(len(forest_type.items) for forest_type in forest_types.values()),
            "collidableStaticInstances": sum(
                static.collision_type != "None"
                for static in statics
            ),
            "collidableForestInstances": sum(
                len(forest_type.items)
                for forest_type in forest_types.values()
                if forest_type.collidable
            ),
            "staticCollisionTriangles": static_collision_triangles,
            "collisionMergedAwayTriangles": asset_stats[
                "collisionMergedAwayTriangles"
            ],
            "thinWallComponentsDetected": thin_wall_stats["detected"],
            "thinWallComponentsRepaired": thin_wall_stats["repaired"],
            "thinWallComponentsRestored": thin_wall_stats["restored"],
            "thinWallFacesRemoved": thin_wall_stats["removedFaces"],
            "thinWallFacesGenerated": thin_wall_stats["generatedFaces"],
            "materialVariants": len(materials),
            "sourceSkyboxRemovalEnabled": options.remove_source_skybox,
            "sourceSkyboxPartsRemoved": asset_stats[
                "sourceSkyboxPartsRemoved"
            ],
            "overdrawnFacesRemoved": asset_stats["overdrawnFacesRemoved"],
            "waterObjects": len(waters),
            "waterNameFallbacks": asset_stats["waterNameFallbacks"],
            "snowwallCollisionOverrideEnabled": options.use_snowwall_collision_override,
            "snowwallCollisionOverrides": snowwall_collision_overrides,
            "snowwallCollisionOverrideParts": asset_stats[
                "snowwallCollisionOverrideParts"
            ],
            "snowwallCollisionOverrideFaces": asset_stats[
                "snowwallCollisionOverrideFaces"
            ],
            "thinWallInflationEnabled": options.inflate_thin_walls,
            "vertexLerpMaterials": asset_stats["materials"],
            "vertexLerpVertexColors": asset_stats["vertexColors"],
            "vertexLerpVertexPbrBakes": asset_stats["vertexPbrBakes"],
            "vertexLerpHybridBakes": asset_stats["hybridBakes"],
            "vertexLerpFallbacks": asset_stats["fallbacks"],
            "visualLodsEnabled": options.use_visual_lods,
            "lodCulledShapes": asset_stats["lodShapes"],
            "visualLodGroups": asset_stats["lodShapes"],
            "mapBorderBrakeWallsSupported": False,
            "mapBorderBrakeWallsEnabled": options.use_map_border_brake_walls,
            "brakeWallSegments": 0,
            "brakeWallFaces": 0,
            **plugin_stats,
            "drivelinePoints": len(stage.driveline),
            "routeLengthMeters": round(stage.driveline[-1].distance, 2),
            "sectorCount": (
                sum(note.note_type == 23 for note in stage.pacenotes) + 1
            ),
            "worldOriginPolicy": "sourceSpawnRelativeAboveMeanSeaLevel",
            "mapAltitudeMeters": round(map_altitude_meters, 3),
            "physicsFloorZMeters": 0.0,
            "environmentPreset": environment_settings.preset,
            "temperatureNightC": environment_settings.temperature_night,
            "temperatureDayC": environment_settings.temperature_day,
            "environmentDate": environment_settings.calendar_date.isoformat(),
            "timeOfDay": round(environment_settings.time_of_day, 8),
            "mapYawDegrees": round(map_yaw_degrees, 6),
        }
        if environment_settings.utc_offset is not None:
            stats["environmentUtcOffset"] = environment_settings.utc_offset
        if environment_settings.sun_azimuth is not None:
            stats["targetSunAzimuth"] = round(
                environment_settings.sun_azimuth,
                6,
            )
        if environment_settings.sun_elevation is not None:
            stats["targetSunElevation"] = round(
                environment_settings.sun_elevation,
                6,
            )
        if source_sun_bearing is not None:
            stats["sourceSunAzimuth"] = round(source_sun_bearing, 6)
        reporter.emit("level", "Writing BeamNG level and mission")
        with profile_span(
            "write_level",
            category="conversion",
            materials=len(materials),
            statics=len(statics),
            forestTypes=len(forest_types),
        ):
            write_level(
                payload,
                level_id,
                stage,
                origin,
                statics,
                forest_types,
                waters,
                materials,
                bounds_min,
                bounds_max,
                warnings,
                stats,
                environment_settings,
                conversion_parameters(options),
                lambda current, total, path: reporter.emit(
                    "textures",
                    "Cooking texture",
                    current=current,
                    total=total,
                    detail=path.name,
                ),
                use_foliage_name_fallback=options.use_foliage_name_fallback,
                foliage_name_matches=options.foliage_name_matches,
                foliage_ground_types=(
                    options.foliage_ground_types
                    if options.use_foliage_ground_types
                    else ()
                ),
                pacenote_log=pacenote_log,
                pacenote_notes=pacenote_notes,
                pacenote_visualizer_log=pacenote_visualizer_log,
                map_yaw_degrees=map_yaw_degrees,
            )
        if "rallyPacenotes" in stats:
            reporter.emit(
                "rally",
                "Pacenote summary",
                detail=format_pacenote_summary(stats),
            )
        reporter.emit("validate", "Validating generated package")
        with profile_span(
            "validate_package",
            category="package",
        ):
            validate_package(
                payload,
                level_id,
                archived_paths,
                rally=stage.has_finish,
            )
        report_warnings(reporter, warnings)
        reporter.emit("install", "Installing generated mod", detail=str(destination))
        with profile_span(
            "install_zip",
            category="package",
            destination=str(destination),
        ):
            install_zip(
                payload,
                destination,
                options.overwrite,
                filesystem,
                prepared_zip,
            )
    zip_size = filesystem.stat(destination).st_size
    stats["zipSizeBytes"] = zip_size
    reporter.emit(
        "complete",
        "Conversion complete",
        detail=str(destination),
        zip_size_bytes=zip_size,
        collision_triangles=static_collision_triangles,
        merged_away_triangles=asset_stats["collisionMergedAwayTriangles"],
        route_length_meters=stats["routeLengthMeters"],
        sector_count=stats["sectorCount"],
        option_stats=option_stats_for_summary(stats),
    )
    return ConversionResult(destination, level_id, tuple(warnings), stats)


def convert(
    options: ConversionOptions,
    reporter: ProgressReporter | None = None,
) -> ConversionResult:
    with use_filesystem(options.filesystem):
        clear_temporary_material_state()
        try:
            return _convert(options, reporter)
        finally:
            clear_temporary_material_state()
