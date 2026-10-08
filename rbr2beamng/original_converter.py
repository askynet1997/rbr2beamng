from __future__ import annotations

import hashlib
import zipfile
from collections.abc import Callable, Mapping
from dataclasses import replace
from pathlib import Path

import numpy as np

from . import __version__
from .beamng import (
    StaticInstance,
    WaterSpec,
    add_cutout_passes,
    has_blend_pass,
    validate_package,
    write_collada_to_archive,
    write_level,
    write_mod_info,
)
from .core import (
    ConversionError,
    ProgressReporter,
    slugify,
    stage_zip_name,
)
from .converter import _minimum_water_rectangle, is_source_sky
from .conversion_common import (
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
from .environment import (
    location_override,
    resolve_environment_settings,
)
from .filesystem import current_filesystem, use_filesystem
from .geometry import (
    collision_shape_digest,
    extract_thin_wall_templates,
    inflate_thin_wall_templates,
    lod_detail_size,
    remap_mesh_part,
    source_position_to_beamng,
)
from .material_baker import (
    bake_original_material,
    is_original_bake_material,
    prepare_original_material_part,
)
from .models import (
    ConversionResult,
    MaterialVariant,
    MeshPart,
    StageMetadata,
    WaterRegion,
)
from .original.col import parse_col
from .original.dls import parse_dls
from .original.fnc import read_fence_texture_archive
from .original.fnc import parse_fnc
from .original.lbs import parse_lbs
from .original.mat import parse_mat
from .original.models import FncFile
from .original.textures import (
    DEFAULT_ROAD_CONDITION,
    TEXTURE_PAYLOAD_STEM,
    OriginalTextureLookup,
    TextureResolver,
    first_complete_condition,
    parse_texture_filename_map_file,
    parse_texture_ini,
    parse_texture_remap,
)
from .pacenotes import format_pacenote_summary
from .original.trk import parse_trk
from .original.adapter import (
    adapt_brake_wall_collision,
    brake_wall_material_variant,
    prepare_original_variant,
)
from .original.catalog import OriginalCatalog, OriginalStageFiles
from .original.source import (
    ORIGINAL_TINT_NAMES,
    original_catalog,
    original_country_code,
    original_level_id,
    resolve_original_variants,
    select_original_variants,
)
from . import plugins
from .plugins import PluginLevel
from .profiling import profile_span
from .rbr import location_warning, read_rbr_surfaces
from .texture_cooker import CookedTextures
from .water_appearance import (
    DEFAULT_WATER_PROFILE_MANIFEST,
    WaterProfileManifest,
    water_appearance_for_source,
)
from .zip_deflate import ZIP_COMPRESSLEVEL


_ORIGINAL_BAKE_BATCH_FACES = 8_000
_BRAKE_WALL_MAX_TRIANGLE_DIAGONAL = 5.0
_BRAKE_WALL_GROUND_DEPTH = 2.0
_BRAKE_WALL_GROUND_CLEARANCE = 10.0
_IMPLEMENTED_ORIGINAL_RENDER_FLAGS = 0x1
_OPTION_COUNTER_KEYS = (
    "sourceSkyboxPartsRemoved",
    "waterObjects",
    "snowwallCollisionOverrideParts",
    "snowwallCollisionOverrideFaces",
    "thinWallComponentsDetected",
    "thinWallComponentsRepaired",
    "thinWallComponentsRestored",
    "thinWallFacesGenerated",
    "visualLodGroups",
    "brakeWallSegments",
    "brakeWallFaces",
)


def _original_render_metadata(
    materials: Mapping[str, MaterialVariant],
) -> tuple[dict[str, int | str], list[str]]:
    values: dict[int, int] = {}
    shadow_materials = 0
    for variant in materials.values():
        material = variant.material
        if material is None:
            continue
        raw_flags = material.properties.get("originalRenderFlags")
        if raw_flags is not None:
            try:
                flags = int(raw_flags, 0)
            except ValueError:
                continue
            values[flags] = values.get(flags, 0) + 1
        if material.properties.get("originalShadow") not in {None, "None"}:
            shadow_materials += 1
    unimplemented = sum(
        count
        for flags, count in values.items()
        if flags & ~_IMPLEMENTED_ORIGINAL_RENDER_FLAGS
    )
    raw_values = ", ".join(
        f"0x{flags:x}:{count}"
        for flags, count in sorted(values.items())
    )
    warnings: list[str] = []
    if unimplemented:
        warnings.append(
            "Original LBS render flags contain unimplemented bits; "
            "raw values were retained in material provenance"
        )
    if shadow_materials:
        warnings.append(
            f"Original LBS shadow maps are not rendered: {shadow_materials} material references"
        )
    return (
        {
            "originalLbsRenderFlagMaterials": sum(values.values()),
            "originalLbsRenderFlagValues": raw_values,
            "originalLbsUnimplementedRenderFlagMaterials": unimplemented,
            "originalLbsShadowMaterialReferences": shadow_materials,
            "originalLbsShadowMapsBound": 0,
        },
        warnings,
    )


def _case_insensitive_path(path: Path) -> Path | None:
    filesystem = current_filesystem()
    if not filesystem.is_dir(path.parent):
        return None
    matches = [
        candidate
        for candidate in filesystem.iterdir(path.parent)
        if candidate.name.casefold() == path.name.casefold()
    ]
    if len(matches) > 1:
        raise ConversionError(
            f"Multiple Original RBR texture companions match {path.name}: "
            + ", ".join(candidate.name for candidate in matches)
        )
    return matches[0] if matches else None


def _payload_resolver(
    payload: Path,
    cache: dict[Path, TextureResolver],
) -> TextureResolver | None:
    filesystem = current_filesystem()
    archive = _case_insensitive_path(payload.with_name(f"{payload.name}.rbz"))
    if archive and filesystem.is_file(archive):
        source = filesystem.read_path(archive)
        if source not in cache:
            cache[source] = TextureResolver.from_rbz(source)
        return cache[source]
    directory = _case_insensitive_path(payload)
    if directory and filesystem.is_dir(directory):
        source = filesystem.read_path(directory)
        if source not in cache:
            cache[source] = TextureResolver.from_directory(source)
        return cache[source]
    return None


def _texture_lookup(
    files: OriginalStageFiles,
    catalog: OriginalCatalog,
    cache: dict[Path, TextureResolver],
) -> OriginalTextureLookup:
    texture_ini = files.files["texture_ini"]
    stage_id = files.entry.stage_id

    def payload(name: str) -> TextureResolver | None:
        match = TEXTURE_PAYLOAD_STEM.fullmatch(name)
        if match is None:
            return None
        payload_id = int(match.group(1))
        entry = catalog.entries.get(payload_id)
        if payload_id == stage_id:
            folder = texture_ini.parent
        elif entry is not None:
            folder = entry.track_base.parent
        else:
            # RSF keeps track-00, its replacement stock-stage banners, in Maps.
            folder = catalog.maps_root
        return _payload_resolver(folder / name, cache)

    remap_path = _case_insensitive_path(
        files.entry.track_base.with_name(f"TextureFilenameMap{stage_id}.ini")
    )
    if remap_path is None and 10 <= stage_id <= 99:
        # RSF gives the stock stages without a map of their own its banners.
        remap_path = _case_insensitive_path(
            catalog.maps_root / "TextureFilenameMap00.ini"
        )
    stock_path = _case_insensitive_path(
        catalog.maps_root / "TextureFilenameMap.dat"
    )
    return OriginalTextureLookup(
        texture_ini.stem,
        payload,
        rsf_remap=parse_texture_remap(remap_path) if remap_path else None,
        stock_map=(
            parse_texture_filename_map_file(stock_path) if stock_path else None
        ),
    )


def _extract_fence_textures(
    rbr_root: Path,
    fnc: FncFile,
    target: Path,
    archive_cache: dict[Path, dict[str, bytes]],
) -> dict[int, Path]:
    filesystem = current_filesystem()
    texture_indices = {
        index
        for fence in fnc.fences
        if len(fence.posts) >= 2
        for index in (
            fence.tile_texture_index,
            fence.pole_texture_index,
        )
    }
    if not texture_indices:
        return {}
    archive = _case_insensitive_path(rbr_root / "Textures" / "fence.dat")
    if archive is None or not filesystem.is_file(archive):
        raise ConversionError("RBR fence texture archive Textures/fence.dat not found")
    archive = filesystem.read_path(archive)
    if archive not in archive_cache:
        archive_cache[archive] = read_fence_texture_archive(archive)
    payloads = archive_cache[archive]
    filesystem.mkdir(target, parents=True, exist_ok=True)
    result: dict[int, Path] = {}
    for texture_index in sorted(texture_indices):
        texture_name = f"{fnc.textures[texture_index]}.dds"
        payload = payloads.get(texture_name.casefold())
        if payload is None:
            raise ConversionError(
                f"FNC texture {texture_name!r} is missing from {archive}"
            )
        output = filesystem.write_path(target / f"fence_{texture_index:03d}.dds")
        temporary = output.with_name(output.name + ".part")
        filesystem.write_bytes(temporary, payload)
        filesystem.replace(temporary, output)
        result[texture_index] = output
    return result


def _shift_part(part: MeshPart, origin: np.ndarray) -> MeshPart:
    return replace(
        part,
        vertices=(
            np.asarray(part.vertices, dtype=np.float32)
            - np.asarray(origin, dtype=np.float32)
        ),
    )


def _collision_vertical_bounds(
    parts: list[MeshPart],
) -> tuple[float, float] | None:
    minimum = float("inf")
    maximum = float("-inf")
    for part in parts:
        vertices = np.asarray(part.vertices, dtype=np.float64)
        if (
            vertices.ndim != 2
            or vertices.shape[1:] != (3,)
            or not len(vertices)
            or not np.isfinite(vertices).all()
        ):
            continue
        minimum = min(minimum, float(np.min(vertices[:, 2])))
        maximum = max(maximum, float(np.max(vertices[:, 2])))
    if not minimum < maximum:
        return None
    return minimum, maximum


def _is_original_reflection_material(
    material_name: str,
    materials: Mapping[str, MaterialVariant],
) -> bool:
    variant = materials.get(material_name)
    material = variant.material if variant else None
    if material is None:
        return False
    return any(
        key.casefold() == "originalsourcekind"
        and value.casefold() == "reflection"
        for key, value in material.properties.items()
    )


def _filter_original_reflection_parts(
    parts: list[MeshPart],
    materials: Mapping[str, MaterialVariant],
    *,
    replace_reflections: bool,
) -> tuple[list[MeshPart], int]:
    if not replace_reflections:
        return parts, 0
    retained = [
        part
        for part in parts
        if not _is_original_reflection_material(part.material_name, materials)
    ]
    return retained, len(parts) - len(retained)


def _combine_original_bake_parts(parts: list[MeshPart]) -> MeshPart:
    first = parts[0]
    vertex_offsets = np.cumsum(
        [0, *(len(part.vertices) for part in parts[:-1])],
        dtype=np.uint32,
    )
    texcoord_sets = tuple(
        np.concatenate(
            [np.asarray(part.texcoord_sets[index]) for part in parts],
            axis=0,
        )
        for index in range(len(first.texcoord_sets))
    )
    specular_strengths = (
        np.concatenate(
            [np.asarray(part.specular_strengths) for part in parts],
            axis=0,
        )
        if first.specular_strengths is not None
        else None
    )
    return replace(
        first,
        name=f"{first.name}_batch",
        vertices=np.concatenate([np.asarray(part.vertices) for part in parts]),
        faces=np.concatenate(
            [
                np.asarray(part.faces, dtype=np.uint32) + offset
                for part, offset in zip(parts, vertex_offsets)
            ]
        ),
        normals=np.concatenate([np.asarray(part.normals) for part in parts]),
        texcoords=texcoord_sets[0],
        colors=np.concatenate([np.asarray(part.colors) for part in parts]),
        texcoord_sets=texcoord_sets,
        specular_strengths=specular_strengths,
        blend_weights=None,
        lod_group=None,
        lod_kind="any",
    )


def _split_original_baked_part(
    baked: MeshPart,
    sources: list[tuple[int, MeshPart]],
    material_name: str,
) -> list[tuple[int, MeshPart]]:
    result: list[tuple[int, MeshPart]] = []
    face_offset = 0
    for output_index, source in sources:
        face_count = len(source.faces)
        faces = np.asarray(
            baked.faces[face_offset : face_offset + face_count],
            dtype=np.uint32,
        )
        referenced = np.unique(faces)
        inverse = np.full(len(baked.vertices), -1, dtype=np.int64)
        inverse[referenced] = np.arange(len(referenced), dtype=np.int64)
        compact = remap_mesh_part(
            baked,
            referenced,
            inverse[faces],
            name=source.name,
            material_name=material_name,
            lod_group=source.lod_group,
            lod_kind=source.lod_kind,
        )
        result.append((output_index, compact))
        face_offset += face_count
    if face_offset != len(baked.faces):
        raise ConversionError("Original material bake changed the source face count")
    return result


def _bake_original_render_parts(
    parts: list[MeshPart],
    materials: dict[str, MaterialVariant],
    output_dir: Path,
    level_id: str,
    warnings: list[str],
) -> list[MeshPart]:
    output: list[MeshPart | None] = [None] * len(parts)
    buckets: dict[tuple[object, ...], list[tuple[int, MeshPart]]] = {}
    for part_index, part in enumerate(parts):
        variant = materials.get(part.material_name)
        material = variant.material if variant is not None else None
        if not is_original_bake_material(part, material):
            output[part_index] = (
                prepare_original_material_part(part, material)
                if material is not None
                else part
            )
            continue
        key = (
            part.material_name,
            tuple(sorted(part.semantic_uv_indices.items())),
            len(part.texcoord_sets),
            part.specular_strengths is not None,
        )
        buckets.setdefault(key, []).append((part_index, part))

    batch_index = 0
    for bucket in buckets.values():
        batch: list[tuple[int, MeshPart]] = []
        batch_faces = 0
        for indexed_part in [*bucket, None]:
            if indexed_part is not None:
                part_faces = len(indexed_part[1].faces)
                if batch and batch_faces + part_faces <= _ORIGINAL_BAKE_BATCH_FACES:
                    batch.append(indexed_part)
                    batch_faces += part_faces
                    continue
                if not batch:
                    batch.append(indexed_part)
                    batch_faces = part_faces
                    continue
            combined = _combine_original_bake_parts(
                [part for _index, part in batch]
            )
            variant = materials[combined.material_name]
            if variant.material is None:
                raise ConversionError(
                    f"Original material {combined.material_name!r} is missing"
                )
            try:
                baked = bake_original_material(
                    combined,
                    variant.material,
                    output_dir,
                    Path(f"{level_id}_{batch_index}"),
                )
            except ConversionError as exc:
                warnings.append(
                    f"Original material bake fallback for "
                    f"{combined.material_name!r}: {exc}"
                )
                for output_index, part in batch:
                    output[output_index] = prepare_original_material_part(
                        part,
                        variant.material,
                    )
            else:
                material_name = (
                    f"{combined.material_name}_baked_{batch_index:04d}"
                )
                materials[material_name] = replace(
                    variant,
                    pbr_override=baked.material,
                )
                for output_index, part in _split_original_baked_part(
                    baked.part,
                    batch,
                    material_name,
                ):
                    output[output_index] = part
            batch_index += 1
            batch = [] if indexed_part is None else [indexed_part]
            batch_faces = (
                0 if indexed_part is None else len(indexed_part[1].faces)
            )

    if any(part is None for part in output):
        raise ConversionError("Original material baking left an output part unresolved")
    return [part for part in output if part is not None]


def _original_render_groups(
    parts: list[MeshPart],
    use_visual_lods: bool = True,
) -> list[
    tuple[
        str,
        list[float],
        list[MeshPart],
        tuple[int, ...],
        bool,
    ]
]:
    if not use_visual_lods:
        parts = [
            replace(part, lod_kind="any")
            for part in parts
            if part.lod_kind != "far"
        ]
    grouped: dict[str, list[MeshPart]] = {}
    for part in parts:
        grouped.setdefault(part.lod_group or "original_misc", []).append(part)

    result = []
    for group_name, group_parts in sorted(grouped.items()):
        vertices = np.concatenate(
            [np.asarray(part.vertices, dtype=np.float64) for part in group_parts]
        )
        bounds_min = vertices.min(axis=0)
        bounds_max = vertices.max(axis=0)
        center = (bounds_min + bounds_max) * 0.5
        local_parts = [
            replace(
                part,
                vertices=(
                    np.asarray(part.vertices, dtype=np.float32)
                    - center.astype(np.float32)
                ),
            )
            for part in group_parts
        ]
        has_lod = any(part.lod_kind != "any" for part in local_parts)
        if not has_lod:
            result.append(
                (
                    group_name,
                    center.tolist(),
                    [
                        replace(part, name=f"{part.name}_L2")
                        for part in local_parts
                    ],
                    (),
                    False,
                )
            )
            continue

        radius = float(np.linalg.norm(bounds_max - bounds_min) * 0.5)
        near_detail = max(3, lod_detail_size(radius, 575.0))
        far_detail = 2
        expanded: list[MeshPart] = []
        has_near = False
        has_far = False
        for part in local_parts:
            if part.lod_kind in {"near", "any"}:
                expanded.append(
                    replace(part, name=f"{part.name}_L{near_detail}")
                )
                has_near = True
            if part.lod_kind in {"far", "any"}:
                expanded.append(
                    replace(part, name=f"{part.name}_L{far_detail}")
                )
                has_far = True
        null_details = tuple(
            detail
            for detail, present in (
                (near_detail, has_near),
                (far_detail, has_far),
            )
            if not present
        )
        result.append(
            (
                group_name,
                center.tolist(),
                expanded,
                null_details,
                True,
            )
        )
    return result


def _original_water_regions(
    col,
    origin: np.ndarray,
    water_profiles: WaterProfileManifest = DEFAULT_WATER_PROFILE_MANIFEST,
) -> list[tuple[str, WaterRegion]]:
    result: list[tuple[str, WaterRegion]] = []
    for index, surface in enumerate(col.water_surfaces):
        vertices = (
            np.asarray(surface.vertices, dtype=np.float64)
            - np.asarray(origin, dtype=np.float64)
        )
        size = np.ptp(vertices, axis=0)
        if size[0] <= 0.05 or size[1] <= 0.05:
            continue
        result.append(
            (
                f"original_water_{index}",
                WaterRegion(
                    material_name="original_water",
                    vertices=vertices,
                    faces=np.asarray(
                        ((0, 1, 2), (0, 2, 3)),
                        dtype=np.uint32,
                    ),
                    source_ids=(f"original_col:{index}",),
                    appearance=water_appearance_for_source(None, water_profiles),
                ),
            )
        )
    return result


def _original_puddles(
    col,
    origin: np.ndarray,
    water_profiles: WaterProfileManifest = DEFAULT_WATER_PROFILE_MANIFEST,
) -> list[WaterSpec]:
    """Water for the COL wet quads, which RBR adds only on a wet road.

    RBR's water surface is horizontal at the first corner's height.
    """
    result: list[WaterSpec] = []
    for index, surface in enumerate(col.wet_surfaces):
        vertices = (
            np.asarray(surface.vertices, dtype=np.float64)
            - np.asarray(origin, dtype=np.float64)
        )
        rectangle = _minimum_water_rectangle(vertices, np.asarray(((0, 1, 2), (0, 2, 3))))
        if rectangle is None or min(rectangle[1]) <= 0.05:
            continue
        center, size, rotation, _coverage = rectangle
        result.append(
            WaterSpec(
                name=f"original_puddle_{index}",
                kind="WaterBlock",
                position=[float(center[0]), float(center[1]), float(vertices[0, 2])],
                scale=[float(size[0]), float(size[1]), 0.5],
                rotation=rotation,
                source_ids=(f"original_col_wet:{index}",),
                appearance=water_appearance_for_source(None, water_profiles),
                wet_only=True,
            )
        )
    return result


def _original_lbs_water_regions(
    parts: list[MeshPart],
    materials: Mapping[str, MaterialVariant] | None = None,
    weld_tolerance: float = 0.05,
    water_profiles: WaterProfileManifest = DEFAULT_WATER_PROFILE_MANIFEST,
    source_prefix: str = "original",
) -> tuple[list[tuple[str, WaterRegion]], set[str]]:
    regions: list[tuple[str, WaterRegion]] = []
    source_ids: set[str] = set()
    for part in parts:
        if not part.water:
            continue
        vertices = np.asarray(part.vertices, dtype=np.float64)
        faces = np.asarray(part.faces, dtype=np.uint32).reshape((-1, 3))
        if not len(vertices) or not len(faces):
            continue
        variant = materials.get(part.material_name) if materials else None
        material = variant.material if variant else None
        appearance = water_appearance_for_source(
            (
                f"{source_prefix}:material:{material.index}"
                if material is not None
                else None
            ),
            water_profiles,
        )
        quantized = np.round(vertices / weld_tolerance).astype(np.int64)
        faces_by_position: dict[tuple[int, int, int], list[int]] = {}
        for face_index, face in enumerate(faces):
            for vertex_index in face:
                key = tuple(int(value) for value in quantized[int(vertex_index)])
                faces_by_position.setdefault(key, []).append(face_index)
        visited = np.zeros(len(faces), dtype=bool)
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
                    key = tuple(
                        int(value)
                        for value in quantized[int(vertex_index)]
                    )
                    for neighbour in faces_by_position[key]:
                        if not visited[neighbour]:
                            visited[neighbour] = True
                            pending.append(neighbour)
            component_faces = faces[np.asarray(component, dtype=np.uint32)]
            used_vertices, compact_faces = np.unique(
                component_faces,
                return_inverse=True,
            )
            source_id = f"original_lbs:{part.name}:{len(regions)}"
            source_ids.add(source_id)
            regions.append(
                (
                    source_id,
                    WaterRegion(
                        material_name=part.material_name,
                        vertices=vertices[used_vertices],
                        faces=compact_faces.reshape((-1, 3)).astype(np.uint32),
                        source_ids=(source_id,),
                        appearance=appearance,
                    ),
                )
            )
    return regions, source_ids


def _point_in_triangle_xy(
    point: np.ndarray,
    triangle: np.ndarray,
) -> bool:
    first = triangle[1] - triangle[0]
    second = triangle[2] - triangle[0]
    relative = point - triangle[0]
    denominator = first[0] * second[1] - first[1] * second[0]
    if abs(denominator) < 1e-9:
        return False
    first_weight = (
        relative[0] * second[1] - relative[1] * second[0]
    ) / denominator
    second_weight = (
        first[0] * relative[1] - first[1] * relative[0]
    ) / denominator
    return (
        first_weight >= -1e-6
        and second_weight >= -1e-6
        and first_weight + second_weight <= 1.0 + 1e-6
    )


def _segments_intersect_xy(
    first_start: np.ndarray,
    first_end: np.ndarray,
    second_start: np.ndarray,
    second_end: np.ndarray,
) -> bool:
    def orientation(
        start: np.ndarray,
        end: np.ndarray,
        point: np.ndarray,
    ) -> float:
        return float(
            (end[0] - start[0]) * (point[1] - start[1])
            - (end[1] - start[1]) * (point[0] - start[0])
        )

    first_side = orientation(first_start, first_end, second_start)
    second_side = orientation(first_start, first_end, second_end)
    third_side = orientation(second_start, second_end, first_start)
    fourth_side = orientation(second_start, second_end, first_end)
    if first_side * second_side < -1e-9 and third_side * fourth_side < -1e-9:
        return True

    def on_segment(
        start: np.ndarray,
        end: np.ndarray,
        point: np.ndarray,
    ) -> bool:
        return bool(
            np.all(point >= np.minimum(start, end) - 1e-9)
            and np.all(point <= np.maximum(start, end) + 1e-9)
        )

    return bool(
        abs(first_side) <= 1e-9
        and on_segment(first_start, first_end, second_start)
        or abs(second_side) <= 1e-9
        and on_segment(first_start, first_end, second_end)
        or abs(third_side) <= 1e-9
        and on_segment(second_start, second_end, first_start)
        or abs(fourth_side) <= 1e-9
        and on_segment(second_start, second_end, first_end)
    )


def _water_regions_overlap(
    first: WaterRegion,
    second: WaterRegion,
    vertical_tolerance: float = 2.5,
) -> bool:
    first_vertices = np.asarray(first.vertices, dtype=np.float64)
    second_vertices = np.asarray(second.vertices, dtype=np.float64)
    if abs(
        float(np.median(first_vertices[:, 2]))
        - float(np.median(second_vertices[:, 2]))
    ) > vertical_tolerance:
        return False
    first_min = first_vertices[:, :2].min(axis=0)
    first_max = first_vertices[:, :2].max(axis=0)
    second_min = second_vertices[:, :2].min(axis=0)
    second_max = second_vertices[:, :2].max(axis=0)
    if np.any(first_max < second_min) or np.any(second_max < first_min):
        return False
    first_triangles = first_vertices[
        np.asarray(first.faces, dtype=np.uint32).reshape((-1, 3)),
        :2,
    ]
    second_triangles = second_vertices[
        np.asarray(second.faces, dtype=np.uint32).reshape((-1, 3)),
        :2,
    ]
    for first_triangle in first_triangles:
        for second_triangle in second_triangles:
            if (
                np.any(first_triangle.max(axis=0) < second_triangle.min(axis=0))
                or np.any(second_triangle.max(axis=0) < first_triangle.min(axis=0))
            ):
                continue
            if any(
                _point_in_triangle_xy(point, second_triangle)
                for point in first_triangle
            ) or any(
                _point_in_triangle_xy(point, first_triangle)
                for point in second_triangle
            ):
                return True
            for first_index in range(3):
                for second_index in range(3):
                    if _segments_intersect_xy(
                        first_triangle[first_index],
                        first_triangle[(first_index + 1) % 3],
                        second_triangle[second_index],
                        second_triangle[(second_index + 1) % 3],
                    ):
                        return True
    return False


def _correlate_original_water_regions(
    col_regions: list[tuple[str, WaterRegion]],
    lbs_regions: list[tuple[str, WaterRegion]],
) -> list[tuple[str, WaterRegion]]:
    matched_col_ids: set[str] = set()
    result: list[tuple[str, WaterRegion]] = []
    for lbs_name, lbs_region in lbs_regions:
        overlapping_col_ids = {
            col_region.source_ids[0]
            for _col_name, col_region in col_regions
            if _water_regions_overlap(lbs_region, col_region)
        }
        if overlapping_col_ids:
            matched_col_ids.update(overlapping_col_ids)
            lbs_region = replace(
                lbs_region,
                source_ids=tuple(
                    sorted((*lbs_region.source_ids, *overlapping_col_ids))
                ),
            )
        result.append((lbs_name, lbs_region))
    result.extend(
        (name, region)
        for name, region in col_regions
        if not matched_col_ids.intersection(region.source_ids)
    )
    return result


def _write_static_shape(
    archive: zipfile.ZipFile,
    archived_paths: set[str],
    statics: list[StaticInstance],
    archive_name: str,
    key: str,
    parts: list[MeshPart],
    position: list[float],
    rotation: list[float],
    *,
    dynamic: bool = False,
    collision_only: bool = False,
    null_detail_sizes: tuple[int, ...] = (),
    progress: Callable[[str], None] | None = None,
    collision_compaction_cache: dict[bytes, MeshPart | None] | None = None,
    shared_collision_shapes: dict[bytes, tuple[str, int]] | None = None,
) -> int:
    """Write ``parts`` as a static shape; with ``shared_collision_shapes``, a
    collision-only shape identical to one already written is referenced
    instead of written again."""
    shape_digest = (
        collision_shape_digest(parts)
        if shared_collision_shapes is not None
        else None
    )
    if shape_digest is not None and shape_digest in shared_collision_shapes:
        archive_name, collision_triangles = shared_collision_shapes[shape_digest]
    else:
        _, _, collision_triangles = write_collada_to_archive(
            archive,
            archive_name,
            parts,
            collision_only=collision_only,
            collision_parts=None if collision_only else [],
            null_detail_sizes=null_detail_sizes,
            progress=progress,
            collision_compaction_cache=collision_compaction_cache,
        )
        archived_paths.add(archive_name)
        if shape_digest is not None:
            shared_collision_shapes[shape_digest] = (
                archive_name,
                collision_triangles,
            )
    statics.append(
        StaticInstance(
            key=key,
            shape_vfs=f"/{archive_name}",
            position=position,
            rotation=rotation,
            scale=[1.0, 1.0, 1.0],
            visible=True,
            collision_type="Collision Mesh" if collision_only else "None",
            dynamic=dynamic or collision_only,
        )
    )
    return collision_triangles


def _variant_metadata(
    base: StageMetadata,
    stage_id: int,
    tint: str,
) -> StageMetadata:
    return replace(
        base,
        folder_name=f"original_{stage_id}_{tint.casefold()}",
    )


def _file_digest(path: Path) -> bytes:
    filesystem = current_filesystem()
    digest = hashlib.sha256()
    with filesystem.open(path, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.digest()


def _mesh_geometry_digest(parts: list[MeshPart]) -> str:
    digest = hashlib.sha256()
    for part in parts:
        for values in (part.vertices, part.faces):
            array = np.asarray(values)
            digest.update(array.dtype.str.encode("ascii"))
            digest.update(
                np.asarray(array.shape, dtype=np.int64).tobytes()
            )
            digest.update(array.tobytes())
    return digest.hexdigest()[:16]


def _prepare(
    rbr_root: Path,
    files: OriginalStageFiles,
    metadata: StageMetadata,
    location,
    texture_target: Path,
    use_snowwall_collision_override: bool,
    inflate_thin_walls: bool,
    water_name_matches,
    parsed_cache: dict[tuple[object, Path], object],
    resolver_cache: dict[Path, TextureResolver],
    fence_archive_cache: dict[Path, dict[str, bytes]],
    collision_cache: dict[object, object],
    surfaces,
    reporter: ProgressReporter,
    progress_detail: str,
):
    reporter_detail = (
        "environment: "
        f"{ORIGINAL_TINT_NAMES.get(files.tint.value, files.tint.value).casefold()}"
    )
    active_parser: tuple[str, Path] | None = None
    try:
        parsed: dict[str, object] = {}
        parsers = (
            ("lbs", "track", parse_lbs),
            ("col", "collision", parse_col),
            ("mat", "material", parse_mat),
            ("trk", "physics", parse_trk),
            ("dls", "driveline", parse_dls),
            ("fnc", "fence", parse_fnc),
            ("texture_ini", "texture_ini", parse_texture_ini),
        )
        for label, key, parser in parsers:
            path = files.files[key]
            cache_key = (parser, path)
            if cache_key not in parsed_cache:
                active_parser = (label.upper(), path)
                reporter.emit(
                    "original",
                    "Parsing Original RBR source",
                    detail=f"{progress_detail}: {label.upper()}",
                )
                with profile_span(
                    f"parse_{label}",
                    category="original",
                    path=str(path),
                    sourceBytes=current_filesystem().stat(path).st_size,
                ):
                    parsed_cache[cache_key] = parser(path)
                active_parser = None
            parsed[label] = parsed_cache[cache_key]
        lbs = parsed["lbs"]
        col = parsed["col"]
        mat = parsed["mat"]
        trk = parsed["trk"]
        dls = parsed["dls"]
        fnc = parsed["fnc"]
        texture_ini = parsed["texture_ini"]
        collision_cache_key = (
            _file_digest(files.files["collision"]),
            _file_digest(files.files["material"]),
        )
        with profile_span(
            "resolve_original_textures",
            category="original",
        ):
            textures = _texture_lookup(
                files,
                original_catalog(rbr_root),
                resolver_cache,
            )
            road_condition = first_complete_condition(textures, texture_ini)
            textures = textures.with_condition(
                road_condition or DEFAULT_ROAD_CONDITION
            )
            fence_texture_paths = _extract_fence_textures(
                rbr_root,
                fnc,
                texture_target,
                fence_archive_cache,
            )
        with profile_span(
            "adapt_original_variant",
            category="original",
        ):
            reporter.emit(
                "original",
                "Converting Original RBR geometry",
                detail=reporter_detail,
            )
            prepared = prepare_original_variant(
                root=rbr_root,
                metadata=metadata,
                trk=trk,
                dls=dls,
                lbs=lbs,
                col=col,
                mat=mat,
                texture_ini=texture_ini,
                textures=textures,
                texture_target=texture_target,
                surfaces=surfaces,
                fnc=fnc,
                fence_texture_paths=fence_texture_paths,
                use_snowwall_collision_override=use_snowwall_collision_override,
                inflate_thin_walls=inflate_thin_walls,
                water_name_matches=water_name_matches,
                location=location,
                source_variant=files.tint.value,
                provenance=files.files,
                collision_cache=collision_cache,
                collision_cache_key=collision_cache_key,
                progress=lambda message: reporter.emit(
                    "original",
                    message,
                    detail=reporter_detail,
                ),
            )
        if road_condition is None:
            wetness, wear = DEFAULT_ROAD_CONDITION
            prepared.stage.warnings.append(
                "RBR has no road condition with every Original texture of this "
                f"stage; using {wetness}/{wear}"
            )
        plugin_found = plugins.find(
            "find_original",
            lbs,
            texture_ini,
            textures,
            prepared.stage.warnings,
        )
        return prepared, col, plugin_found
    except (ValueError, OSError) as exc:
        source = (
            f" while parsing {active_parser[0]} file {active_parser[1]}"
            if active_parser
            else ""
        )
        raise ConversionError(
            f"Unable to prepare Original RBR stage {reporter_detail}{source}: {exc}"
        ) from exc


def _convert_original(
    options,
    reporter: ProgressReporter,
    convert_water_regions,
) -> ConversionResult:
    filesystem = options.filesystem
    rbr_root = filesystem.read_path(options.rbr_root)
    with profile_span(
        "resolve_original_variants",
        category="parse",
        selector=options.stage,
    ) as resolve_span:
        inspection, variants = resolve_original_variants(
            rbr_root,
            options.stage,
            options.original_variants,
        )
        variants, primary_tint = select_original_variants(
            variants,
            options.original_variants,
        )
        entry = variants[0].entry
        resolve_span.update(
            stageId=entry.stage_id,
            variants=[variant.tint.value for variant in variants],
        )
    archive_name = stage_zip_name(
        inspection.metadata.folder_name,
        display_name=inspection.metadata.name,
        source_format="original",
    )
    package_id = Path(archive_name).stem
    destination = output_path_for(options, package_id)
    if filesystem.exists(destination) and not options.overwrite:
        raise ConversionError(
            f"Output already exists: {destination}. Use --overwrite to replace it."
        )
    pacenote_log, pacenote_notes = pacenote_log_writers(destination, filesystem)

    reporter.emit(
        "prepare",
        "Preparing Original RBR conversion",
        detail=f"{inspection.metadata.name} ({len(variants)} variants)",
    )
    stage_location = location_override(
        inspection.location,
        options.latitude,
        options.longitude,
    )
    map_altitude_meters = resolved_map_altitude_meters(
        options,
        stage_location,
    )
    warnings = [location_warning(original_country_code(rbr_root, entry))]
    total_collision_triangles = 0
    total_merged_away_triangles = 0
    aggregate_stats: dict[str, bool | int | float | str] = {
        "sourceFormat": "original",
        "sourceStageId": entry.stage_id,
        "originalVariants": len(variants),
        "sourceSkyboxRemovalEnabled": options.remove_source_skybox,
        "sourceSkyboxPartsRemoved": 0,
        "waterObjects": 0,
        "snowwallCollisionOverrideEnabled": (
            options.use_snowwall_collision_override
        ),
        "snowwallCollisionOverrideParts": 0,
        "snowwallCollisionOverrideFaces": 0,
        "thinWallInflationEnabled": options.inflate_thin_walls,
        "thinWallComponentsDetected": 0,
        "thinWallComponentsRepaired": 0,
        "thinWallComponentsRestored": 0,
        "thinWallFacesGenerated": 0,
        "visualLodsEnabled": options.use_visual_lods,
        "visualLodGroups": 0,
        "mapBorderBrakeWallsSupported": True,
        "mapBorderBrakeWallsEnabled": (
            options.use_map_border_brake_walls
        ),
        "brakeWallVerticalPolicy": "localCollisionBand",
        "brakeWallSourceVerticalSemantics": "unverified",
        "brakeWallMaxTriangleDiagonalMeters": (
            _BRAKE_WALL_MAX_TRIANGLE_DIAGONAL
        ),
        "brakeWallGroundDepthMeters": _BRAKE_WALL_GROUND_DEPTH,
        "brakeWallGroundClearanceMeters": _BRAKE_WALL_GROUND_CLEARANCE,
        "brakeWallSegments": 0,
        "brakeWallFaces": 0,
        "originalLbsRenderFlagMaterials": 0,
        "originalLbsUnimplementedRenderFlagMaterials": 0,
        "originalLbsShadowMaterialReferences": 0,
        "originalLbsShadowMapsBound": 0,
    }
    level_ids: list[str] = []
    rally_level_ids: set[str] = set()
    render_flag_value_reports: set[str] = set()
    packaged_environments = tuple(
        ORIGINAL_TINT_NAMES.get(files.tint.value, files.tint.value).casefold()
        for files in variants
    )
    parsed_cache: dict[tuple[object, Path], object] = {}
    resolver_cache: dict[Path, TextureResolver] = {}
    fence_archive_cache: dict[Path, dict[str, bytes]] = {}
    collision_cache: dict[object, object] = {}
    collision_compaction_cache: dict[bytes, MeshPart | None] = {}
    shared_collision_shapes: dict[bytes, tuple[str, int]] = {}
    thin_wall_shell_cache: dict[bytes, tuple] = {}
    thin_wall_inflate_cache: dict[
        tuple[str, str],
        tuple[tuple[MeshPart, ...], dict[str, int]],
    ] = {}
    cooked_textures: CookedTextures = {}
    surfaces = read_rbr_surfaces(rbr_root)

    with temporary_workspace(package_id, filesystem) as temporary:
        temporary_root = filesystem.write_path(Path(temporary))
        payload = filesystem.write_path(temporary_root / "payload")
        filesystem.mkdir(payload)
        prepared_zip = filesystem.write_path(temporary_root / f"{package_id}.zip")
        archived_paths: set[str] = set()

        with zipfile.ZipFile(
            filesystem.write_path(prepared_zip),
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=ZIP_COMPRESSLEVEL,
            allowZip64=True,
        ) as shape_archive:
            for variant_index, files in enumerate(variants, 1):
                tint = files.tint.value
                is_primary = tint == primary_tint
                level_id = original_level_id(package_id, tint, primary_tint)
                level_ids.append(level_id)
                pacenote_visualizer_log = pacenote_visualizer_writer(
                    destination,
                    filesystem,
                    level_id,
                )
                reporter.emit(
                    "original",
                    "Parsing Original RBR variant",
                    current=variant_index,
                    total=len(variants),
                    detail=ORIGINAL_TINT_NAMES.get(tint, tint),
                )
                metadata = _variant_metadata(
                    inspection.metadata,
                    entry.stage_id,
                    tint,
                )

                def original_progress(message: str) -> None:
                    reporter.emit(
                        "original",
                        message,
                        current=variant_index,
                        total=len(variants),
                        detail=(
                            "environment: "
                            f"{ORIGINAL_TINT_NAMES.get(tint, tint).casefold()}"
                        ),
                    )

                def original_texture_progress(
                    current: int,
                    total: int,
                    path: Path,
                ) -> None:
                    if current != 1 and current % 25 != 0 and current != total:
                        return
                    reporter.emit(
                        "textures",
                        "Cooking Original RBR texture",
                        current=current,
                        total=total,
                        detail=path.name,
                    )

                with profile_span(
                    "prepare_original_variant",
                    category="original",
                    stageId=entry.stage_id,
                    tint=tint,
                ) as prepare_span:
                    prepared, col, plugin_found = _prepare(
                        rbr_root,
                        files,
                        metadata,
                        stage_location,
                        temporary_root / "original_textures" / tint,
                        options.use_snowwall_collision_override,
                        options.inflate_thin_walls,
                        (
                            options.water_name_matches
                            if options.use_water_name_fallback
                            else None
                        ),
                        parsed_cache,
                        resolver_cache,
                        fence_archive_cache,
                        collision_cache,
                        surfaces,
                        reporter,
                        metadata.name,
                    )
                    prepare_span.update(
                        renderParts=len(prepared.render_parts),
                        collisionParts=len(prepared.collision_parts),
                        materials=len(prepared.materials),
                        colSourceTriangles=(
                            prepared.collision_subdivision.source_triangles
                        ),
                        colGeneratedTriangles=(
                            prepared.collision_subdivision.generated_triangles
                        ),
                        colSubdividedTriangles=(
                            prepared.collision_subdivision.subdivided_triangles
                        ),
                    colSubdivisionFallbacks=(
                        prepared.collision_subdivision.fallbacks
                    ),
                    colMergedAwayTriangles=(
                        prepared.collision_subdivision.merged_away_triangles
                    ),
                    )
                prepared.stage.documents = inspection.documents
                variant_warnings = list(prepared.stage.warnings)
                original_render_stats, original_render_warnings = (
                    _original_render_metadata(prepared.materials)
                )
                variant_warnings.extend(original_render_warnings)
                if original_render_stats["originalLbsRenderFlagValues"]:
                    render_flag_value_reports.add(
                        str(original_render_stats["originalLbsRenderFlagValues"])
                    )
                if variant_index == 1:
                    aggregate_stats["routeLengthMeters"] = round(
                        prepared.stage.driveline[-1].distance,
                        2,
                    )
                    aggregate_stats["sectorCount"] = (
                        sum(
                            note.note_type == 23
                            for note in prepared.stage.pacenotes
                        )
                        + 1
                    )
                if options.preview_radius_m is not None:
                    variant_warnings.append(
                        "Preview radius is ignored for Original RBR stages"
                    )

                original_progress("Preparing Original RBR output geometry")
                origin = source_position_to_beamng(
                    prepared.stage.spawn.position
                )
                origin[2] -= map_altitude_meters
                route_positions = np.asarray(
                    [
                        source_position_to_beamng(point.position) - origin
                        for point in prepared.stage.driveline
                    ],
                    dtype=np.float64,
                )
                render_parts = [
                    part
                    for part in (
                        _shift_part(part, origin)
                        for part in prepared.render_parts
                    )
                    if not (
                        options.remove_source_skybox
                        and is_source_sky(part, np.eye(4)[None], route_positions)
                    )
                ]
                source_skybox_parts_removed = len(prepared.render_parts) - len(render_parts)
                snowwall_collision_parts = [
                    part
                    for part in prepared.collision_parts
                    if part.material_name == "original_visual_snowwall"
                ]
                collision_parts = [
                    _shift_part(part, origin)
                    for part in prepared.collision_parts
                ]
                brake_wall_vertical_bounds = _collision_vertical_bounds(
                    collision_parts
                )
                route_digest = hashlib.sha256(
                    route_positions.tobytes()
                ).hexdigest()[:16]
                original_progress("Hashing Original RBR collision geometry")
                collision_geometry_digest = _mesh_geometry_digest(
                    collision_parts
                )
                if options.inflate_thin_walls:
                    original_progress("Extracting Original RBR thin wall templates")
                    thin_wall_cache_stats: dict[str, int] = {}
                    thin_wall_inflate_key = (
                        collision_geometry_digest,
                        route_digest,
                    )
                    thin_wall_inflate_hit = thin_wall_inflate_key in (
                        thin_wall_inflate_cache
                    )
                    with profile_span(
                        "repair_thin_walls",
                        category="original",
                        tint=tint,
                        collisionParts=len(collision_parts),
                        collisionFaces=sum(
                            len(part.faces)
                            for part in collision_parts
                        )
                    ) as thin_wall_span:
                        collision_parts, thin_wall_templates = (
                            extract_thin_wall_templates(
                                collision_parts,
                                shell_cache=thin_wall_shell_cache,
                                cache_stats=thin_wall_cache_stats,
                                progress=original_progress,
                            )
                        )
                        original_progress("Inflating Original RBR thin wall templates")
                        if thin_wall_inflate_hit:
                            cached_repairs, cached_stats = (
                                thin_wall_inflate_cache[
                                    thin_wall_inflate_key
                                ]
                            )
                            thin_wall_repairs = list(cached_repairs)
                            thin_wall_stats = dict(cached_stats)
                        else:
                            thin_wall_repairs, thin_wall_stats = (
                                inflate_thin_wall_templates(
                                    thin_wall_templates,
                                    route_positions,
                                    name_prefix="original_thin_wall",
                                )
                            )
                            thin_wall_inflate_cache[thin_wall_inflate_key] = (
                                tuple(thin_wall_repairs),
                                dict(thin_wall_stats),
                            )
                        thin_wall_span.update(
                            cacheHits=thin_wall_cache_stats.get("hits", 0),
                            cacheMisses=thin_wall_cache_stats.get("misses", 0),
                            inflateCacheHit=thin_wall_inflate_hit,
                            **thin_wall_stats,
                        )
                    collision_parts.extend(thin_wall_repairs)
                else:
                    thin_wall_stats = {
                        "detected": 0,
                        "repaired": 0,
                        "restored": 0,
                        "removedFaces": 0,
                        "generatedFaces": 0,
                    }
                    thin_wall_cache_stats = {}
                if thin_wall_stats["repaired"]:
                    variant_warnings.append(
                        "Inflated "
                        f"{thin_wall_stats['repaired']} thin collision walls "
                        "outward to 0.5 m"
                    )
                if thin_wall_stats["restored"]:
                    variant_warnings.append(
                        "Kept Original RBR collision for "
                        f"{thin_wall_stats['restored']} thin walls that could "
                        "not be inflated"
                    )
                materials: dict[str, MaterialVariant] = dict(
                    prepared.materials
                )
                brake_wall_faces = 0
                brake_wall_segments = 0
                brake_wall_points = 0
                brake_wall_part: MeshPart | None = None
                if (
                    options.use_map_border_brake_walls
                    and col.brake_wall is not None
                ):
                    if brake_wall_vertical_bounds is None:
                        variant_warnings.append(
                            "Omitted Original RBR map-border brake wall because "
                            "source collision has no finite vertical extent"
                        )
                    else:
                        original_progress(
                            "Generating Original RBR map-border brake wall"
                        )
                        material_index = max(
                            (
                                variant.material.index
                                for variant in materials.values()
                                if variant.material is not None
                            ),
                            default=-1,
                        ) + 1
                        brake_variant = brake_wall_material_variant(
                            material_index
                        )
                        brake_part = adapt_brake_wall_collision(
                            col.brake_wall,
                            material_index=material_index,
                            collision_parts=collision_parts,
                            vertical_bounds=brake_wall_vertical_bounds,
                            origin=origin,
                            max_triangle_diagonal=(
                                _BRAKE_WALL_MAX_TRIANGLE_DIAGONAL
                            ),
                            ground_depth=_BRAKE_WALL_GROUND_DEPTH,
                            ground_clearance=_BRAKE_WALL_GROUND_CLEARANCE,
                        )
                        if brake_part is not None:
                            brake_wall_part = brake_part
                            materials["original_brake_wall"] = brake_variant
                            brake_wall_faces = len(brake_part.faces)
                            brake_wall_segments = len(col.brake_wall.segments)
                            brake_wall_points = len(
                                col.brake_wall.inner_points
                            )
                            variant_warnings.append(
                                "Generated synthetic Original RBR map-border "
                                "brake wall from local source collision "
                                "bounds; source vertical semantics are "
                                "unverified"
                            )
                if not render_parts:
                    raise ConversionError(
                        f"Original variant {entry.stage_id}:{tint} has no render geometry"
                    )
                identity_rotation = [
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0,
                ]
                retained_water_materials: set[str] = set()
                original_progress("Generating Original RBR procedural water")
                col_water_regions = _original_water_regions(
                    col,
                    origin,
                    options.water_profiles,
                )
                lbs_water_regions, lbs_water_source_ids = (
                    _original_lbs_water_regions(
                        render_parts,
                        materials,
                        water_profiles=options.water_profiles,
                        source_prefix=f"original:{entry.stage_id}",
                    )
                )
                source_water_regions = _correlate_original_water_regions(
                    col_water_regions,
                    lbs_water_regions,
                )
                waters = convert_water_regions(
                    source_water_regions,
                    variant_warnings,
                    route_positions,
                )
                if any(
                    wetness == "wet"
                    for wetness, _wear in prepared.condition_ground_types
                ):
                    waters.extend(
                        _original_puddles(col, origin, options.water_profiles)
                    )
                covered_source_ids = {
                    source_id
                    for water in waters
                    for source_id in water.source_ids
                }
                filtered_render_parts: list[MeshPart] = []
                for part in render_parts:
                    if not part.water:
                        filtered_render_parts.append(part)
                        continue
                    prefix = f"original_lbs:{part.name}:"
                    part_source_ids = {
                        source_id
                        for source_id in lbs_water_source_ids
                        if source_id.startswith(prefix)
                    }
                    if (
                        part_source_ids
                        and part_source_ids.issubset(covered_source_ids)
                    ):
                        continue
                    filtered_render_parts.append(part)
                    retained_water_materials.add(part.material_name)
                render_parts = filtered_render_parts
                uncovered_count = len(
                    lbs_water_source_ids - covered_source_ids
                )
                if uncovered_count:
                    variant_warnings.append(
                        f"Retained {uncovered_count} Original RBR water visual "
                        "components without a valid procedural footprint"
                    )
                render_parts, reflection_proxy_parts_removed = (
                    _filter_original_reflection_parts(
                        render_parts,
                        materials,
                        replace_reflections=bool(
                            lbs_water_source_ids
                            and lbs_water_source_ids.issubset(
                                covered_source_ids
                            )
                        ),
                    )
                )
                if retained_water_materials:
                    materials = {
                        key: (
                            replace(value, water=False, ground_type="VOID")
                            if key in retained_water_materials
                            else value
                        )
                        for key, value in materials.items()
                    }
                original_progress("Baking Original RBR render textures")
                render_parts = _bake_original_render_parts(
                    render_parts,
                    materials,
                    temporary_root / "baked_textures" / tint,
                    level_id,
                    warnings,
                )
                render_parts = add_cutout_passes(
                    render_parts,
                    materials,
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
                version_suffix = __version__.replace(".", "_")
                statics: list[StaticInstance] = []
                original_lod_groups = 0
                original_progress("Grouping Original RBR render geometry")
                render_groups = _original_render_groups(
                    render_parts,
                    options.use_visual_lods,
                )
                for group_index, (
                    group_name,
                    group_position,
                    group_parts,
                    null_detail_sizes,
                    has_lod,
                ) in enumerate(render_groups, 1):
                    if (
                        group_index == 1
                        or group_index % 100 == 0
                        or group_index == len(render_groups)
                    ):
                        original_progress(
                            "Writing Original RBR render shapes: "
                            f"{group_index:,}/{len(render_groups):,}"
                        )
                    original_lod_groups += int(has_lod)
                    shape_name = (
                        f"{slugify(group_name)}_{version_suffix}.dae"
                    )
                    archive_name = (
                        f"levels/{level_id}/art/shapes/{shape_name}"
                    )
                    with profile_span(
                        "write_render_group",
                        category="collada",
                        tint=tint,
                        group=group_name,
                        parts=len(group_parts),
                        faces=sum(
                            len(part.faces)
                            for part in group_parts
                        ),
                        hasLod=has_lod,
                    ):
                        _write_static_shape(
                            shape_archive,
                            archived_paths,
                            statics,
                            archive_name,
                            f"original_render_{group_name}",
                            group_parts,
                            group_position,
                            identity_rotation,
                            dynamic=has_blend_pass(
                                group_parts,
                                materials,
                            ),
                            null_detail_sizes=null_detail_sizes,
                        )

                collision_triangles = 0
                if collision_parts:
                    original_progress("Writing Original RBR collision shape")
                    collision_shape_name = (
                        f"original_collision_{version_suffix}.dae"
                    )
                    collision_archive_name = (
                        f"levels/{level_id}/art/shapes/"
                        f"{collision_shape_name}"
                    )
                    with profile_span(
                        "write_collision",
                        category="collada",
                        tint=tint,
                        parts=len(collision_parts),
                        faces=sum(
                            len(part.faces)
                            for part in collision_parts
                        ),
                    ) as collision_span:
                        collision_triangles = _write_static_shape(
                            shape_archive,
                            archived_paths,
                            statics,
                            collision_archive_name,
                            "original_collision",
                            collision_parts,
                            [0.0, 0.0, 0.0],
                            identity_rotation,
                            collision_only=True,
                            progress=original_progress,
                            collision_compaction_cache=(
                                collision_compaction_cache
                            ),
                            shared_collision_shapes=shared_collision_shapes,
                        )
                        collision_span.update(
                            collisionTriangles=collision_triangles,
                        )
                if brake_wall_part is not None:
                    original_progress("Writing Original RBR map-border brake wall")
                    brake_shape_name = (
                        f"original_map_border_brake_wall_"
                        f"{version_suffix}.dae"
                    )
                    brake_archive_name = (
                        f"levels/{level_id}/art/shapes/"
                        f"{brake_shape_name}"
                    )
                    with profile_span(
                        "write_map_border_brake_wall",
                        category="collada",
                        tint=tint,
                        faces=len(brake_wall_part.faces),
                    ):
                        brake_collision_triangles = _write_static_shape(
                            shape_archive,
                            archived_paths,
                            statics,
                            brake_archive_name,
                            "original_map_border_brake_wall",
                            [brake_wall_part],
                            [0.0, 0.0, 0.0],
                            identity_rotation,
                            collision_only=True,
                        )
                    collision_triangles += brake_collision_triangles
                plugin_stats = plugins.write_level(
                    PluginLevel(
                        archive=shape_archive,
                        archived_paths=archived_paths,
                        level_dir=payload / "levels" / level_id,
                        level_id=level_id,
                        origin=origin,
                        version_suffix=version_suffix,
                        waters=waters,
                        prepared=prepared,
                        progress=original_progress,
                    ),
                    plugin_found,
                )
                total_collision_triangles += collision_triangles
                total_merged_away_triangles += (
                    prepared.collision_subdivision.merged_away_triangles
                )

                bounds_min = (
                    np.asarray(prepared.bounds_min, dtype=np.float64) - origin
                )
                bounds_max = (
                    np.asarray(prepared.bounds_max, dtype=np.float64) - origin
                )
                if collision_parts:
                    bounds_min[2] = min(
                        float(np.min(part.vertices[:, 2]))
                        for part in collision_parts
                        if len(part.vertices)
                    )
                original_progress("Finalizing Original RBR level bounds")
                bounds_min = np.minimum(
                    bounds_min,
                    route_positions.min(axis=0),
                )
                bounds_max = np.maximum(
                    bounds_max,
                    route_positions.max(axis=0),
                )
                stats: dict[str, bool | int | float | str] = {
                    "sourceFormat": "original",
                    "sourceStageId": entry.stage_id,
                    "sourceVariant": tint,
                    "renderParts": len(render_parts),
                    "renderShapes": len(render_groups),
                    "sourceSkyboxRemovalEnabled": (
                        options.remove_source_skybox
                    ),
                    "sourceSkyboxPartsRemoved": (
                        source_skybox_parts_removed
                    ),
                    "visualLodsEnabled": options.use_visual_lods,
                    "originalLodGroups": original_lod_groups,
                    "visualLodGroups": original_lod_groups,
                    "collisionParts": (
                        len(collision_parts)
                        + int(brake_wall_part is not None)
                    ),
                    "staticCollisionTriangles": collision_triangles,
                    "sourceColCollisionTriangles": (
                        prepared.collision_subdivision.source_triangles
                    ),
                    "generatedColCollisionTriangles": (
                        prepared.collision_subdivision.generated_triangles
                    ),
                    "subdividedColCollisionTriangles": (
                        prepared.collision_subdivision.subdivided_triangles
                    ),
                    "colMaterialSubdivisionFallbacks": (
                        prepared.collision_subdivision.fallbacks
                    ),
                    "collisionMergedAwayTriangles": (
                        prepared.collision_subdivision.merged_away_triangles
                    ),
                    "thinWallComponentsDetected": thin_wall_stats["detected"],
                    "thinWallComponentsRepaired": thin_wall_stats["repaired"],
                    "thinWallComponentsRestored": thin_wall_stats["restored"],
                    "thinWallFacesRemoved": thin_wall_stats["removedFaces"],
                    "thinWallFacesGenerated": thin_wall_stats["generatedFaces"],
                    "thinWallInflationEnabled": options.inflate_thin_walls,
                    "materialVariants": len(materials),
                    "waterObjects": len(waters),
                    "reflectionProxyPartsRemoved": (
                        reflection_proxy_parts_removed
                    ),
                    "mapBorderBrakeWallsEnabled": (
                        options.use_map_border_brake_walls
                    ),
                    "brakeWallPoints": brake_wall_points,
                    "brakeWallSegments": brake_wall_segments,
                    "brakeWallFaces": brake_wall_faces,
                    "brakeWallVerticalPolicy": "localCollisionBand",
                    "brakeWallSourceVerticalSemantics": "unverified",
                    "brakeWallSynthetic": brake_wall_part is not None,
                    **plugin_stats,
                    "brakeWallMaxTriangleDiagonalMeters": (
                        _BRAKE_WALL_MAX_TRIANGLE_DIAGONAL
                    ),
                    "brakeWallGroundDepthMeters": _BRAKE_WALL_GROUND_DEPTH,
                    "brakeWallGroundClearanceMeters": (
                        _BRAKE_WALL_GROUND_CLEARANCE
                    ),
                    "snowwallCollisionOverrideEnabled": (
                        options.use_snowwall_collision_override
                    ),
                    "snowwallCollisionOverrideParts": len(
                        snowwall_collision_parts
                    ),
                    "snowwallCollisionOverrideFaces": sum(
                        len(part.faces)
                        for part in snowwall_collision_parts
                    ),
                    "mapBorderBrakeWallsSupported": True,
                    "drivelinePoints": len(prepared.stage.driveline),
                    "routeLengthMeters": round(
                        prepared.stage.driveline[-1].distance,
                        2,
                    ),
                    "worldOriginPolicy": "sourceSpawnRelativeAboveMeanSeaLevel",
                    "mapAltitudeMeters": round(map_altitude_meters, 3),
                    "physicsFloorZMeters": 0.0,
                    **original_render_stats,
                }
                if brake_wall_part is not None:
                    stats["brakeWallMinZ"] = float(
                        np.min(brake_wall_part.vertices[:, 2])
                    )
                    stats["brakeWallMaxZ"] = float(
                        np.max(brake_wall_part.vertices[:, 2])
                    )
                for key in (
                    "originalLbsRenderFlagMaterials",
                    "originalLbsUnimplementedRenderFlagMaterials",
                    "originalLbsShadowMaterialReferences",
                    "originalLbsShadowMapsBound",
                ):
                    aggregate_stats[key] += int(stats[key])
                for key in _OPTION_COUNTER_KEYS:
                    aggregate_stats[key] += int(stats[key])
                environment = resolve_environment_settings(
                    metadata.physics,
                    options.temperature_night,
                    options.temperature_day,
                    location=prepared.stage.location,
                    environment_variant=(
                        options.environment if is_primary else tint
                    ),
                    environment_date=options.environment_date,
                )
                variant_warnings.extend(environment.warnings)
                stats.update(
                    environmentDate=environment.calendar_date.isoformat(),
                    timeOfDay=round(environment.time_of_day, 8),
                    mapYawDegrees=0.0,
                )
                if environment.utc_offset is not None:
                    stats["environmentUtcOffset"] = environment.utc_offset
                if environment.sun_azimuth is not None:
                    stats["targetSunAzimuth"] = round(
                        environment.sun_azimuth,
                        6,
                    )
                if environment.sun_elevation is not None:
                    stats["targetSunElevation"] = round(
                        environment.sun_elevation,
                        6,
                    )
                reporter.emit(
                    "level",
                    "Writing Original RBR BeamNG level",
                    current=variant_index,
                    total=len(variants),
                    detail=metadata.name,
                )
                with profile_span(
                    "write_original_level",
                    category="original",
                    tint=tint,
                    statics=len(statics),
                    materials=len(materials),
                    waterObjects=len(waters),
                ):
                    write_level(
                        payload,
                        level_id,
                        prepared.stage,
                        origin,
                        statics,
                        {},
                        waters,
                        materials,
                        bounds_min,
                        bounds_max,
                        variant_warnings,
                        stats,
                        environment,
                        conversion_parameters(options),
                        original_texture_progress,
                        texture_copy_progress=lambda current, total, path: reporter.emit(
                            "textures",
                            "Preparing Original RBR texture",
                            current=current,
                            total=total,
                            detail=path.name,
                        ),
                        include_mod_info=False,
                        use_foliage_name_fallback=(
                            options.use_foliage_name_fallback
                        ),
                        foliage_name_matches=options.foliage_name_matches,
                        foliage_ground_types=(
                            options.foliage_ground_types
                            if options.use_foliage_ground_types
                            else ()
                        ),
                        progress=lambda message: reporter.emit(
                            "level",
                            message,
                            current=variant_index,
                            total=len(variants),
                            detail=metadata.name,
                        ),
                        display_environment=(
                            None
                            if is_primary
                            else ORIGINAL_TINT_NAMES.get(tint, tint).casefold()
                        ),
                        default_road_condition=prepared.road_condition,
                        pacenote_log=pacenote_log,
                        pacenote_notes=pacenote_notes,
                        pacenote_visualizer_log=pacenote_visualizer_log,
                        cooked_textures=cooked_textures,
                    )
                if prepared.stage.has_finish:
                    rally_level_ids.add(level_id)
                if "rallyPacenotes" in stats:
                    reporter.emit(
                        "rally",
                        "Pacenote summary",
                        current=variant_index,
                        total=len(variants),
                        detail=(
                            f"{ORIGINAL_TINT_NAMES.get(tint, tint).casefold()}: "
                            f"{format_pacenote_summary(stats)}"
                        ),
                    )
                warnings.extend(
                    f"{tint}: {warning}" for warning in variant_warnings
                )

        write_mod_info(
            payload,
            package_id,
            inspection.metadata,
            stage_location,
            available_environments=packaged_environments,
            rally=bool(rally_level_ids),
        )
        for level_id in level_ids:
            with profile_span(
                "validate_original_level",
                category="package",
                levelId=level_id,
            ):
                validate_package(
                    payload,
                    level_id,
                    archived_paths,
                    mod_info_level_id=package_id,
                    rally=level_id in rally_level_ids,
                )
        report_warnings(reporter, warnings)
        reporter.emit(
            "install",
            "Installing Original RBR stage mod",
            detail=str(destination),
        )
        with profile_span(
            "install_original_zip",
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
    aggregate_stats["staticCollisionTriangles"] = (
        total_collision_triangles
    )
    aggregate_stats["collisionMergedAwayTriangles"] = (
        total_merged_away_triangles
    )
    aggregate_stats["originalLbsRenderFlagValues"] = "; ".join(
        sorted(render_flag_value_reports)
    )
    aggregate_stats["zipSizeBytes"] = zip_size
    reporter.emit(
        "complete",
        "Original RBR conversion complete",
        detail=str(destination),
        zip_size_bytes=zip_size,
        collision_triangles=total_collision_triangles,
        merged_away_triangles=total_merged_away_triangles,
        route_length_meters=aggregate_stats["routeLengthMeters"],
        sector_count=aggregate_stats["sectorCount"],
        option_stats=option_stats_for_summary(aggregate_stats),
    )
    return ConversionResult(
        destination,
        package_id,
        tuple(warnings),
        aggregate_stats,
    )


def convert_original(
    options,
    reporter: ProgressReporter,
    convert_water_regions,
) -> ConversionResult:
    with use_filesystem(options.filesystem):
        return _convert_original(options, reporter, convert_water_regions)

