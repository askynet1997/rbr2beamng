from __future__ import annotations

import hashlib
import io
import json
import math
import struct
import time
import zipfile
from dataclasses import dataclass, field, replace
from itertools import pairwise
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np
from PIL import Image, ImageDraw, ImageOps
from zlib_ng import zlib_ng

from . import __version__
from .core import ConversionError, slugify, stable_uuid
from .filesystem import current_filesystem
from .environment import (
    EnvironmentSettings,
    resolve_environment_settings,
)
from .geometry import (
    quaternion_from_direction,
    rotate_z,
    source_position_to_beamng,
    spawn_rotation,
    write_collada,
)
from .models import (
    DEFAULT_FOLIAGE_GROUND_TYPES,
    DEFAULT_FOLIAGE_NAME_MATCHES,
    MaterialVariant,
    MeshPart,
    OpacityTexture,
    RbrMaterial,
    RbrStage,
    StageMetadata,
    WaterAppearance,
)
from .pacenotes import (
    NOTEBOOK_BASENAMES,
    RallyNotebookResult,
    SUPPORTED_STRUCTURED_TYPES,
    build_rally_notebook,
    project_route,
    route_positions,
)
from .pacenote_visualizer import render_pacenote_visualizer
from .profiling import profile_span
from .surface_profiles import current_surface_rules
from .texture_cooker import CookedTextures, cook_textures


_COOKABLE_TEXTURE_SUFFIXES = (".color.png", ".normal.png", ".data.png")
# BeamNG decodes a version 1.5 baseColorMap as sRGB only when the DDS format
# says so; legacy FourCC headers load as linear UNORM and render washed out.
_SRGB_DXGI_FORMATS = {b"DXT1": 72, b"DXT3": 75, b"DXT5": 78}
_ConvertedRoute = list[tuple[np.ndarray, np.ndarray, float]]
_TRANSLUCENT_FLAGS = {"translucent": True, "translucentZWrite": False}
# Metres. Only orders depth between a cutout copy and its blended copies;
# BeamNG's reversed-Z float depth resolves it to kilometres.
_BLEND_PASS_DEPTH_OFFSET = 0.001
# translucentZWrite switches BeamNG to deferred decal blending, which draws
# nothing over surfaces missing from the G-buffer; blended passes keep it off.
_BLEND_PASS_PROFILE = {
    "castShadows": False,
    "translucent": True,
    "translucentBlendOp": "LerpAlpha",
    "translucentZWrite": False,
}


def _surface_profile_provenance(stage: RbrStage) -> list[dict[str, object]]:
    surface_ids = sorted(
        set(stage.used_surface_ids) | set(stage.unresolved_surface_ids)
    )
    result: list[dict[str, object]] = []
    for surface_id in surface_ids:
        surface = stage.surfaces.get(surface_id)
        profile = (
            surface.profile
            if surface and surface.profile
            else current_surface_rules().unknown_profile
        )
        result.append(
            {
                "id": surface_id,
                "name": surface.name if surface else None,
                "flags": list(surface.flags) if surface else [],
                "coefficients": surface.coefficients if surface else {},
                "mappingStatus": surface.profile_status if surface else "unmapped",
                "profile": {
                    "id": profile.profile_id,
                    "groundType": profile.ground_type,
                    "hard": profile.hard,
                    "bendable": profile.bendable,
                    "water": profile.water,
                    "collisionEligible": profile.collision_eligible,
                    "groundDepth": profile.ground_depth,
                    "snowbank": profile.snowbank,
                },
            }
        )
    return result
_BLINN_PHONG_SPECULAR_EFFECTS = frozenset(
    {
        "btb_diffusespecular",
        "btb_diffusebumpspecular",
        "btb_diffusespecularvertlerp",
        "btb_diffusebumpspecularvertlerp",
    }
)
_NO_LEGACY_SPECULAR_ROUGHNESS = 1.0
_VFS_REFERENCE_KEYS = {
    "baseColorMap",
    "clearCoatMap",
    "normalMap",
    "opacityMap",
    "roughnessMap",
    "shapeFile",
    "shapeName",
    "specularMap",
    "terrainFile",
}


def write_json(path: Path, value: object) -> None:
    filesystem = current_filesystem()
    filesystem.mkdir(path.parent, parents=True, exist_ok=True)
    with filesystem.open(path, "w", encoding="utf-8", newline="\n") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def write_jsonl(path: Path, values: Iterable[dict[str, object]]) -> None:
    filesystem = current_filesystem()
    filesystem.mkdir(path.parent, parents=True, exist_ok=True)
    with filesystem.open(path, "w", encoding="utf-8", newline="\n") as stream:
        for value in values:
            stream.write(json.dumps(value, ensure_ascii=False, separators=(",", ":")) + "\n")


def write_collada_to_archive(
    archive: zipfile.ZipFile,
    archive_name: str,
    parts: list[MeshPart],
    *,
    collision_only: bool = False,
    collision_parts: list[MeshPart] | None = None,
    render_detail_size: int | None = None,
    null_detail_sizes: tuple[int, ...] = (),
    progress: Callable[[str], None] | None = None,
    collision_compaction_cache: dict[bytes, MeshPart | None] | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    # BeamNG keeps a cached .cdae whenever it is not older than the .dae, so a
    # streamed entry must not use zipfile's 1980 placeholder timestamp.
    entry = zipfile.ZipInfo(archive_name, date_time=time.localtime()[:6])
    entry.compress_type = archive.compression
    entry._compresslevel = archive.compresslevel
    with archive.open(entry, "w", force_zip64=True) as binary_stream:
        with io.TextIOWrapper(
            binary_stream,
            encoding="utf-8",
            newline="\n",
        ) as text_stream:
            return write_collada(
                text_stream,
                parts,
                collision_only=collision_only,
                collision_parts=collision_parts,
                render_detail_size=render_detail_size,
                null_detail_sizes=null_detail_sizes,
                progress=progress,
                collision_compaction_cache=collision_compaction_cache,
            )


def _mod_info_id(level_id: str) -> str:
    digest = hashlib.sha1(level_id.encode("utf-8")).hexdigest()[:8].upper()
    return f"RBR{digest}"


def _stage_title(
    metadata: StageMetadata,
    location,
    environment: str | None = None,
    *,
    rally: bool = False,
) -> str:
    title = metadata.name
    country = getattr(location, "country", "")
    if country and country != "Unknown":
        title = f"{title}, {country}"
    if environment:
        title = f"{title} - {environment}"
    return f"SS {title}" if rally else title


def _stage_length(metadata: StageMetadata) -> str:
    length = metadata.length_km or 0.0
    return f"{length:.2f}".rstrip("0").rstrip(".")


def _stage_description(
    metadata: StageMetadata,
    location,
    environment: str | None = None,
    available_environments: tuple[str, ...] = (),
    rally: bool = True,
) -> str:
    title = _stage_title(metadata, location)
    surface = (
        f"surface mix: {metadata.surface_composition_text}"
        if metadata.surface_composition
        else f"{metadata.physics} surface"
    )
    if environment:
        surface = f"{surface} and {environment} environment conditions"
    elif len(available_environments) > 1:
        surface = (
            f"{surface} and available environment conditions: "
            f"{', '.join(available_environments)}"
        )
    kind = (
        f"A {_stage_length(metadata)} km long rally stage"
        if rally
        else "A freeroam level"
    )
    description = "\n".join(
        (
            f"{title}. {kind} with {surface}.",
            f"Original author: {metadata.author or 'Unknown'}.",
            f"Converted with RBR2BeamNG v{__version__}.",
        )
    )
    return description


def write_mod_info(
    package_root: Path,
    package_id: str,
    metadata: StageMetadata,
    location=None,
    available_environments: tuple[str, ...] = (),
    rally: bool = True,
) -> None:
    filesystem = current_filesystem()
    timestamp = int(time.time())
    mod_id = _mod_info_id(package_id)
    mod_info_dir = package_root / "mod_info" / mod_id
    author = metadata.author or "Unknown"
    description = _stage_description(
        metadata,
        location,
        available_environments=available_environments,
        rally=rally,
    )
    preview = _preview_image(metadata, (512, 288))
    kind = "Rally Stage" if rally else "Freeroam"
    write_json(
        mod_info_dir / "info.json",
        {
            "title": f"{kind}: {_stage_title(metadata, location)}",
            "username": author,
            "author": author,
            "icon": "icon.jpg",
            "tagid": mod_id,
            "version_string": __version__,
            "converter_version": __version__,
            "source_version": metadata.version,
            "current_version_id": 0,
            "resource_version_id": 0,
            "last_update": timestamp,
            "resource_date": timestamp,
            "category_title": "Maps",
            "tag_line": description,
            "message": description,
            "via": "RBR2BeamNG",
        },
    )
    preview.save(filesystem.write_path(mod_info_dir / "icon.jpg"), quality=88)

@dataclass(frozen=True)
class StaticInstance:
    key: str
    shape_vfs: str
    position: list[float]
    rotation: list[float]
    scale: list[float]
    visible: bool
    collision_type: str
    dynamic: bool = False


@dataclass
class ForestType:
    key: str
    shape_vfs: str
    collidable: bool
    items: list[dict[str, object]] = field(default_factory=list)


@dataclass(frozen=True)
class SpawnSpec:
    name: str
    label: str
    description: str
    position: list[float]
    rotation: list[float]


@dataclass(frozen=True)
class _StageStart:
    position: np.ndarray
    direction: np.ndarray
    route_index: int
    route_distance: float


@dataclass(frozen=True)
class _RouteSplit:
    name: str
    label: str
    distance: float
    route_index: int
    position: np.ndarray
    direction: np.ndarray


@dataclass(frozen=True)
class _MissionPathPoint:
    distance: float
    position: np.ndarray
    direction: np.ndarray
    name: str | None = None
    use_as_split: bool = False
    order: int = 0


@dataclass(frozen=True)
class _MissionAuthoringProfile:
    name: str
    target_speed_mps: float
    minimum_baseline_seconds: float
    gold_multiplier: float
    silver_multiplier: float
    bronze_multiplier: float
    gold_penalty_seconds: float
    silver_penalty_seconds: float
    bronze_penalty_seconds: float
    just_finish_penalty_seconds: float

    def mission_type_data(self, route_length: float) -> dict[str, float | str]:
        baseline_time = round(
            max(
                self.minimum_baseline_seconds,
                route_length / self.target_speed_mps,
            ),
            1,
        )
        gold_time = round(baseline_time * self.gold_multiplier, 1)
        silver_time = round(baseline_time * self.silver_multiplier, 1)
        bronze_time = round(baseline_time * self.bronze_multiplier, 1)
        return {
            "baselineTime": baseline_time,
            "bronzeTime": bronze_time,
            "bronzeTimePenalty": self.bronze_penalty_seconds,
            "bronzeTimeTotal": bronze_time,
            "endScreenText": "",
            "goldTime": gold_time,
            "goldTimePenalty": self.gold_penalty_seconds,
            "goldTimeTotal": gold_time,
            "justFinishPenalty": self.just_finish_penalty_seconds,
            "outroTitleText": "missions.missions.rally.endScreen.stageTitle",
            "silverTime": silver_time,
            "silverTimePenalty": self.silver_penalty_seconds,
            "silverTimeTotal": silver_time,
            "startScreenText": "",
        }


_LEGACY_20MPS_MISSION_PROFILE = _MissionAuthoringProfile(
    name="legacy-20mps",
    target_speed_mps=20.0,
    minimum_baseline_seconds=30.0,
    gold_multiplier=1.05,
    silver_multiplier=1.3,
    bronze_multiplier=1.65,
    gold_penalty_seconds=0.0,
    silver_penalty_seconds=10.0,
    bronze_penalty_seconds=20.0,
    just_finish_penalty_seconds=30.0,
)


@dataclass(frozen=True)
class WaterSpec:
    name: str
    kind: str
    position: list[float]
    scale: list[float]
    nodes: list[list[float]] = field(default_factory=list)
    grid_element_size: float | None = None
    rotation: list[float] | None = None
    source_ids: tuple[str, ...] = ()
    appearance: WaterAppearance | None = None
    wet_only: bool = False


def _preview_image(metadata: StageMetadata, size: tuple[int, int]) -> Image.Image:
    if metadata.splashscreen:
        try:
            with Image.open(
                current_filesystem().read_path(metadata.splashscreen)
            ) as source:
                return ImageOps.fit(source.convert("RGB"), size, method=Image.Resampling.LANCZOS)
        except OSError:
            pass
    return Image.new("RGB", size, (28, 38, 52))


def _draw_route_preview(
    image: Image.Image,
    route: _ConvertedRoute,
    markers: dict[str, np.ndarray],
    active_spawn: str | None = None,
    sector_distances: Sequence[float] = (),
) -> None:
    points = np.asarray(
        [
            *(position[:2] for position, _direction, _distance in route),
            *(position[:2] for position in markers.values()),
        ],
        dtype=np.float64,
    )
    bounds_min = points.min(axis=0)
    extent = np.maximum(points.max(axis=0) - bounds_min, 1.0)
    padding = min(image.size) * 0.08
    scale = min(
        (image.width - padding * 2) / extent[0],
        (image.height - padding * 2) / extent[1],
    )
    offset = (
        (image.width - extent[0] * scale) * 0.5,
        (image.height - extent[1] * scale) * 0.5,
    )

    def project(position: np.ndarray | list[float]) -> tuple[float, float]:
        return (
            offset[0] + (float(position[0]) - bounds_min[0]) * scale,
            image.height - offset[1] - (float(position[1]) - bounds_min[1]) * scale,
        )

    draw = ImageDraw.Draw(image)
    route_points = [project(position) for position, _direction, _distance in route]
    line_width = math.ceil(max(2, round(min(image.size) / 180)) * 2.25)
    draw.line(
        route_points,
        fill=(0, 0, 0),
        width=line_width + max(2, line_width),
        joint="curve",
    )
    draw.line(route_points, fill=(255, 255, 255), width=line_width, joint="curve")
    sector_colors = ((150, 235, 130), (255, 240, 130))
    for index, (start, end) in enumerate(pairwise(sector_distances)):
        draw.line(
            [project(position) for position in route_positions(route, start, end)],
            fill=sector_colors[index % 2],
            width=line_width,
            joint="curve",
        )
    marker_outline_width = math.ceil(max(2, line_width) * 0.5)
    marker_radius = line_width + marker_outline_width
    marker_items = [
        (name, position)
        for name, position in markers.items()
        if name != active_spawn
    ]
    if active_spawn in markers:
        marker_items.append((active_spawn, markers[active_spawn]))
    for name, position in marker_items:
        x, y = project(position)
        if name == "spawn_start":
            color = (0, 255, 0)
        elif name == "spawn_time_control":
            color = (245, 196, 62)
        elif name == "spawn_stop":
            color = (255, 0, 0)
        elif name.startswith("spawn_split_"):
            color = (77, 163, 232)
        else:
            color = (255, 255, 255)
        marker_scale = 0.65 if name.startswith("spawn_split_") else 1.0
        radius = marker_radius * marker_scale
        outline_width = max(
            1,
            round(marker_outline_width * marker_scale),
        )
        bounds = (x - radius, y - radius, x + radius, y + radius)
        if name == "spawn_finish":
            draw.ellipse(bounds, fill=color)
            tile_size = max(1, math.ceil(radius / 2))
            checker = Image.new("RGB", image.size, (255, 255, 255))
            checker_draw = ImageDraw.Draw(checker)
            left = math.floor(x - radius)
            top = math.floor(y - radius)
            for row in range(4):
                for column in range(4):
                    if (row + column) % 2:
                        checker_draw.rectangle(
                            (
                                left + column * tile_size,
                                top + row * tile_size,
                                left + (column + 1) * tile_size,
                                top + (row + 1) * tile_size,
                            ),
                            fill=(0, 0, 0),
                        )
            mask = Image.new("L", image.size, 0)
            ImageDraw.Draw(mask).ellipse(bounds, fill=255)
            image.paste(checker, mask=mask)
            draw.ellipse(bounds, outline=(0, 0, 0), width=outline_width)
        else:
            draw.ellipse(
                bounds,
                fill=color,
                outline=(0, 0, 0),
                width=outline_width,
            )
        if name == active_spawn:
            draw.ellipse(
                (
                    x - radius - outline_width,
                    y - radius - outline_width,
                    x + radius + outline_width,
                    y + radius + outline_width,
                ),
                outline=(255, 128, 0),
                width=outline_width,
            )
            draw.ellipse(
                (
                    x - radius - outline_width * 2,
                    y - radius - outline_width * 2,
                    x + radius + outline_width * 2,
                    y + radius + outline_width * 2,
                ),
                outline=(0, 0, 0),
                width=outline_width,
            )


def _preview_marker_positions(
    spawns: Iterable[SpawnSpec],
) -> dict[str, np.ndarray]:
    markers: dict[str, np.ndarray] = {}
    for spawn in spawns:
        position = np.asarray(spawn.position, dtype=np.float64).copy()
        position[2] -= 0.5
        markers[spawn.name] = position
    return markers


def render_stage_pacenote_visualizer(
    stage: RbrStage,
    origin: np.ndarray,
    map_yaw_degrees: float = 0.0,
) -> str:
    route, route_range, rally = evaluate_stage_pacenotes(
        stage,
        origin,
        map_yaw_degrees,
    )
    stage_start = _resolve_stage_start(
        stage,
        origin,
        route,
        map_yaw_degrees,
    )
    spawns = _build_spawns(
        stage,
        route,
        route_range,
        stage_start,
        map_yaw_degrees,
    )
    return render_pacenote_visualizer(
        stage,
        route,
        _preview_marker_positions(spawns),
        rally,
    )


def evaluate_stage_pacenotes(
    stage: RbrStage,
    origin: np.ndarray,
    map_yaw_degrees: float = 0.0,
) -> tuple[_ConvertedRoute, tuple[int, int], RallyNotebookResult]:
    route = _converted_driveline(stage, origin, map_yaw_degrees)
    stage_start = _resolve_stage_start(
        stage,
        origin,
        route,
        map_yaw_degrees,
    )
    route_range = _route_range(stage, route, stage_start.route_distance)
    return route, route_range, build_rally_notebook(
        stage,
        route,
        route_range,
        start_distance=stage_start.route_distance,
    )


def _write_previews(
    level_dir: Path,
    mission_dir: Path | None,
    metadata: StageMetadata,
    route: _ConvertedRoute,
    markers: dict[str, np.ndarray],
    spawns: list[SpawnSpec],
    sector_distances: Sequence[float],
) -> None:
    filesystem = current_filesystem()
    large_preview = _preview_image(metadata, (1280, 720))
    small_preview = _preview_image(metadata, (500, 281))
    _draw_route_preview(
        large_preview, route, markers, sector_distances=sector_distances
    )
    _draw_route_preview(
        small_preview, route, markers, sector_distances=sector_distances
    )
    large_preview.save(
        filesystem.write_path(level_dir / "preview.jpg"),
        quality=90,
    )
    if mission_dir is not None:
        filesystem.mkdir(mission_dir, parents=True, exist_ok=True)
        large_preview.save(
            filesystem.write_path(mission_dir / "preview.jpg"),
            quality=90,
        )
        small_preview.save(
            filesystem.write_path(mission_dir / "thumbnail.jpg"),
            quality=88,
        )
    for spawn in spawns:
        preview = _preview_image(metadata, (500, 281))
        _draw_route_preview(
            preview, route, markers, spawn.name, sector_distances
        )
        preview.save(
            filesystem.write_path(level_dir / f"{spawn.name}_preview.jpg"),
            quality=88,
        )


def texture_target_name(source: Path, role: str, digest: str = "") -> str:
    suffix = f"_{digest}" if digest else ""
    return f"{slugify(source.stem)}{suffix}.{role}{source.suffix}"


def _needs_cutout_pass(variant: MaterialVariant | None) -> bool:
    material = variant.material if variant is not None else None
    return (
        material is not None
        and material.cutout_alpha_ref is not None
        and material.blend_alpha_ref is not None
        and not variant.water
    )


def add_cutout_passes(
    parts: list[MeshPart],
    materials: dict[str, MaterialVariant],
    foliage_name_matches: tuple[str, ...] = DEFAULT_FOLIAGE_NAME_MATCHES,
    foliage_ground_types: tuple[str, ...] = DEFAULT_FOLIAGE_GROUND_TYPES,
) -> list[MeshPart]:
    """Draw each blended part's depth-writing cutout pass as its own part.

    The source renders these materials twice: an alpha-tested pass that writes
    depth and a blended pass that does not. The cutout copy is emitted first.
    """
    foliage_names = _normalized_foliage_names(foliage_name_matches)
    result: list[MeshPart] = []
    cutout_names: dict[str, str] = {}
    next_material_index = max(
        (
            variant.material.index
            for variant in materials.values()
            if variant.material is not None
        ),
        default=-1,
    ) + 1
    for part in parts:
        variant = materials.get(part.material_name)
        if not _needs_cutout_pass(variant):
            result.append(part)
            continue
        cutout_name = cutout_names.get(part.material_name)
        if cutout_name is None:
            cutout_name = f"{part.material_name}_opacity_depth"
            cutout_names[part.material_name] = cutout_name
            materials[cutout_name] = replace(
                variant,
                material=replace(
                    variant.material,
                    index=next_material_index,
                    name=cutout_name,
                ),
                opacity_depth_prepass=True,
            )
            next_material_index += 1
        result.append(
            replace(
                part,
                name=f"{part.name}_opacity_depth",
                material_name=cutout_name,
            )
        )
        foliage = any(
            _foliage_sources(variant, foliage_names, foliage_ground_types, True)
        )
        result.extend(
            _blend_pass_copies(
                part,
                double_sided=variant.material.double_sided or foliage,
                invert_back_normals=foliage,
            )
        )
    return result


def _blend_pass_copies(
    part: MeshPart,
    *,
    double_sided: bool,
    invert_back_normals: bool,
) -> list[MeshPart]:
    """Single-sided blended copies just behind the cutout copy, one per side.

    The source's blended pass redraws the cutout's texels unchanged. BeamNG
    shades translucent passes differently, so each copy sits behind the
    visible face and the depth test rejects it wherever the cutout drew.
    "Behind" follows the face winding: source vertex normals need not point
    away from the surface. The back copy is shaded like the cutout's back
    faces, which keep the source normals unless vegetation shading inverts them.
    """
    vertices = np.asarray(part.vertices, dtype=np.float64)
    faces = np.asarray(part.faces, dtype=np.int64).reshape((-1, 3))
    corners = vertices[faces]
    face_normals = np.cross(
        corners[:, 1] - corners[:, 0],
        corners[:, 2] - corners[:, 0],
    )
    normals = np.zeros_like(vertices)
    for corner in range(3):
        np.add.at(normals, faces[:, corner], face_normals)
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    offset = np.divide(
        normals,
        lengths,
        out=np.zeros_like(normals),
        where=lengths > 0,
    ) * _BLEND_PASS_DEPTH_OFFSET
    copies = [
        replace(part, vertices=(vertices - offset).astype(np.float32)),
    ]
    if double_sided:
        copies.append(
            replace(
                part,
                name=f"{part.name}_back",
                vertices=(vertices + offset).astype(np.float32),
                faces=np.asarray(part.faces).reshape(-1, 3)[:, ::-1].copy(),
                normals=(
                    -np.asarray(part.normals)
                    if invert_back_normals
                    else part.normals
                ),
            )
        )
    return copies


def has_blend_pass(
    parts: Iterable[MeshPart],
    materials: dict[str, MaterialVariant],
) -> bool:
    return any(
        (variant := materials.get(part.material_name)) is not None
        and variant.material is not None
        and variant.material.blend_alpha_ref is not None
        for part in parts
    )


def _uses_foliage_opacity(
    material: RbrMaterial,
    name_matches: set[str],
) -> bool:
    texture_name = material.diffuse_texture.stem if material.diffuse_texture else ""
    value = slugify(f"{material.name} {texture_name}")
    return not name_matches.isdisjoint(value.split("_"))


def _normalized_foliage_names(name_matches: Iterable[str]) -> set[str]:
    return {slugify(name) for name in name_matches if name.strip()}


def _foliage_sources(
    variant: MaterialVariant,
    name_matches: set[str],
    ground_types: tuple[str, ...],
    has_opacity: bool,
) -> tuple[bool, bool]:
    """Whether BeamNG vegetation shading applies, by surface and by name."""
    material = variant.material
    return (
        variant.ground_type in ground_types,
        material is not None
        and has_opacity
        and _uses_foliage_opacity(material, name_matches),
    )


_TextureSlot = tuple[Path, str, int]


def _material_texture_slots(
    variant: MaterialVariant,
) -> dict[str, _TextureSlot]:
    material = variant.material
    if material is None:
        return {}
    override = variant.pbr_override
    candidates = (
        (
            ("base_color", override.base_color_texture, "color", override.base_color_uv),
            ("normal", override.normal_texture, "normal", override.normal_uv),
            ("roughness", override.roughness_texture, "data", override.roughness_uv),
            ("clear_coat", override.clear_coat_texture, "data", override.clear_coat_uv),
            ("layer_color", override.layer_color_texture, "color", override.layer_color_uv),
            ("layer_opacity", override.layer_opacity_texture, "data", override.layer_opacity_uv),
        )
        if override is not None
        else (
            ("base_color", material.diffuse_texture, "color", 0),
            ("source_layer_color", material.second_diffuse_texture, "color", 0),
            ("normal", material.normal_texture, "normal", 0),
            ("source_specular", material.specular_texture, "data", 0),
        )
    )
    return {
        name: (source, role, uv)
        for name, source, role, uv in candidates
        if source is not None
    }


def _texture_source_sets(
    materials: dict[str, MaterialVariant],
) -> tuple[set[tuple[Path, str]], dict[Path, list[RbrMaterial]]]:
    sources: set[tuple[Path, str]] = set()
    transparent_diffuse_sources: dict[Path, list[RbrMaterial]] = {}

    for variant in materials.values():
        material = variant.material
        if material is None:
            continue
        slots = _material_texture_slots(variant)
        diffuse_texture = slots.get("base_color")
        diffuse_source = diffuse_texture[0] if diffuse_texture else None
        if diffuse_source and material.uses_alpha and not variant.water:
            transparent_diffuse_sources.setdefault(diffuse_source, []).append(material)
        sources.update(
            (source, role)
            for source, role, _uv in slots.values()
        )
    return sources, transparent_diffuse_sources


def _srgb_dds(data: bytes) -> bytes | None:
    """The DDS relabelled as its sRGB BC1-BC3 format, with unchanged blocks."""
    if len(data) < 128 or data[:4] != b"DDS ":
        return None
    pixel_flags, fourcc = struct.unpack_from("<I4s", data, 80)
    caps2 = struct.unpack_from("<I", data, 112)[0]
    if not pixel_flags & 0x4 or fourcc not in _SRGB_DXGI_FORMATS or caps2:
        return None
    dx10 = struct.pack("<5I", _SRGB_DXGI_FORMATS[fourcc], 3, 0, 1, 0)
    return data[:84] + b"DX10" + data[88:128] + dx10 + data[128:]


_PNG_COLOR_TYPES = {"L": 0, "RGB": 2, "RGBA": 6}
_PNG_BAND_ROWS = 64


def _write_png_chunk(stream, kind: bytes, data: bytes) -> None:
    stream.write(struct.pack(">I", len(data)) + kind)
    stream.write(data)
    stream.write(struct.pack(">I", zlib_ng.crc32(data, zlib_ng.crc32(kind))))


def _save_cooker_png(image: Image.Image, target: Path) -> None:
    """Save the chunks Pillow writes at compress_level=0, with unfiltered rows.

    Pillow still picks a filter for every row without compression, and
    texconv decodes the same pixels either way.
    """
    filesystem = current_filesystem()
    color_type = _PNG_COLOR_TYPES.get(image.mode)
    if (
        color_type is None
        or image.info.get("icc_profile")
        or image.info.get("transparency") is not None
    ):
        image.save(filesystem.write_path(target), format="PNG", compress_level=0)
        return
    width, height = image.size
    rows = np.zeros(
        (_PNG_BAND_ROWS, width * len(image.getbands()) + 1),
        dtype=np.uint8,
    )
    compressor = zlib_ng.compressobj(0)
    with filesystem.open(target, "wb") as stream:
        stream.write(b"\x89PNG\r\n\x1a\n")
        _write_png_chunk(
            stream,
            b"IHDR",
            struct.pack(">2I5B", width, height, 8, color_type, 0, 0, 0),
        )
        for start in range(0, height, _PNG_BAND_ROWS):
            end = min(start + _PNG_BAND_ROWS, height)
            band = rows[: end - start]
            band[:, 1:] = np.frombuffer(
                image.crop((0, start, width, end)).tobytes(),
                dtype=np.uint8,
            ).reshape(len(band), -1)
            if data := compressor.compress(band):
                _write_png_chunk(stream, b"IDAT", data)
        _write_png_chunk(stream, b"IDAT", compressor.flush())
        _write_png_chunk(stream, b"IEND", b"")


def copy_textures(
    level_dir: Path,
    materials: dict[str, MaterialVariant],
    progress: Callable[[int, int, Path], None] | None = None,
    warnings: list[str] | None = None,
) -> tuple[dict[tuple[Path, str], str], dict[Path, OpacityTexture]]:
    filesystem = current_filesystem()
    target_dir = level_dir / "art" / "textures"
    filesystem.mkdir(target_dir, parents=True, exist_ok=True)
    (
        sources,
        transparent_diffuse_sources,
    ) = _texture_source_sets(materials)
    result: dict[tuple[Path, str], str] = {}
    opacity_result: dict[Path, OpacityTexture] = {}
    used_names: set[str] = set()
    sorted_sources = sorted(
        sources,
        key=lambda item: (str(item[0]).casefold(), item[1]),
    )
    for current, (source, role) in enumerate(sorted_sources, 1):
        if (
            progress is not None
            and (
                current == 1
                or current % 25 == 0
                or current == len(sorted_sources)
            )
        ):
            progress(current, len(sorted_sources), source)
        alpha_image = None
        transparent_coverage = 0.0
        image: Image.Image | None = None
        needs_conversion = source.suffix.casefold() != ".dds"
        source_stem = slugify(source.stem)
        if role == "color" or needs_conversion:
            try:
                with Image.open(filesystem.read_path(source)) as opened:
                    has_alpha = "A" in opened.getbands() or "transparency" in opened.info
                    if role == "color":
                        image = opened.convert("RGBA" if has_alpha else "RGB")
                    elif role == "normal":
                        image = opened.convert("RGB")
                    else:
                        image = opened.copy()
                        if image.mode not in {"1", "L", "RGB", "RGBA"}:
                            image = image.convert(
                                "RGBA" if has_alpha else "RGB"
                            )
            except OSError as exc:
                if needs_conversion:
                    raise ConversionError(
                        f"Unable to convert non-DDS texture {source}: {exc}"
                    ) from exc
        if image is not None:
            if (
                source in transparent_diffuse_sources
                and "A" in image.getbands()
                and image.getchannel("A").getextrema()[0] < 255
            ):
                alpha_image = image.getchannel("A").copy()
                histogram = alpha_image.histogram()
                transparent_coverage = 1.0 - histogram[255] / (
                    alpha_image.width * alpha_image.height
                )
                # Native-fp16 shaders round opacity, so a texel exactly on
                # alphaRef can fail its cutout test. Kept texels draw opaque
                # whatever their value.
                cutout_refs = {
                    material.cutout_alpha_ref
                    for material in transparent_diffuse_sources[source]
                    if material.cutout_alpha_ref is not None
                    and material.cutout_alpha_ref < 255
                }
                if cutout_refs:
                    alpha_image = alpha_image.point(
                        [value + (value in cutout_refs) for value in range(256)]
                    )
        if alpha_image is None and source in transparent_diffuse_sources:
            requesting_materials = [
                material.name
                for material in transparent_diffuse_sources[source]
            ]
            if warnings is not None:
                state = "could not be decoded" if image is None else "is fully opaque"
                warnings.append(
                    f"Opacity requested for {source.name!r} by "
                    f"{', '.join(sorted(requesting_materials))}, but the texture "
                    f"{state}; exporting it as opaque"
                )
        color_without_alpha = (
            image.convert("RGB")
            if image is not None and alpha_image is not None
            else None
        )
        srgb_dds = None
        if role == "color" and not needs_conversion and color_without_alpha is None:
            srgb_dds = _srgb_dds(filesystem.read_bytes(source))
            needs_conversion = srgb_dds is None and image is not None
        output_image = (
            color_without_alpha
            if color_without_alpha is not None
            else image if needs_conversion else None
        )
        output_source = (
            source.with_suffix(".png")
            if output_image is not None
            else source
        )
        name = texture_target_name(output_source, role)
        source_role = (source, role)
        digest = ""
        if name.casefold() in used_names:
            digest = hashlib.sha1(str(source).casefold().encode("utf-8")).hexdigest()[:8]
            name = texture_target_name(output_source, role, digest)
        used_names.add(name.casefold())
        target = target_dir / name
        if output_image is not None:
            _save_cooker_png(output_image, target)
        elif srgb_dds is not None:
            filesystem.write_bytes(target, srgb_dds)
        else:
            filesystem.copy2(source, target)
        result[source_role] = name
        if alpha_image is not None:
            opacity_name = f"{source_stem}.opacity.data.png"
            if opacity_name.casefold() in used_names:
                digest = digest or hashlib.sha1(str(source).casefold().encode("utf-8")).hexdigest()[:8]
                opacity_name = f"{source_stem}_{digest}.opacity.data.png"
            _save_cooker_png(alpha_image, target_dir / opacity_name)
            used_names.add(opacity_name.casefold())
            opacity_result[source] = OpacityTexture(
                opacity_name,
                "smooth",
                transparent_coverage,
            )
    return result, opacity_result


def _roughness(material: RbrMaterial | None) -> float:
    if (
        material is None
        or material.effect.casefold() not in _BLINN_PHONG_SPECULAR_EFFECTS
    ):
        return _NO_LEGACY_SPECULAR_ROUGHNESS
    raw = material.properties.get("SpecularPower") or material.properties.get(
        "specularPower"
    )
    if raw is None:
        return _NO_LEGACY_SPECULAR_ROUGHNESS
    try:
        power = float(raw.replace(",", "."))
    except ValueError:
        return _NO_LEGACY_SPECULAR_ROUGHNESS
    if not math.isfinite(power) or power < 0.0:
        return _NO_LEGACY_SPECULAR_ROUGHNESS
    return math.sqrt(2.0 / (power + 2.0))


def write_materials(
    level_dir: Path,
    level_id: str,
    materials: dict[str, MaterialVariant],
    texture_names: dict[tuple[Path, str], str],
    opacity_names: dict[Path, OpacityTexture | str],
    fallback_stats: dict[str, int] | None = None,
    foliage_name_matches: tuple[str, ...] = DEFAULT_FOLIAGE_NAME_MATCHES,
    foliage_ground_types: tuple[str, ...] = DEFAULT_FOLIAGE_GROUND_TYPES,
) -> None:
    texture_prefix = f"/levels/{level_id}/art/textures/"
    normalized_foliage_name_matches = _normalized_foliage_names(foliage_name_matches)

    def texture_path(source: Path, role: str) -> str:
        return f"{texture_prefix}{texture_names[source, role]}"

    output: dict[str, object] = {}
    for key, variant in sorted(materials.items()):
        material = variant.material
        override = variant.pbr_override
        slots = _material_texture_slots(variant)
        roughness = _roughness(material)
        transparent = material is not None and material.uses_alpha
        base_color = slots.get("base_color")
        base_color_texture = base_color[0] if base_color else None
        opacity = (
            opacity_names.get(base_color_texture)
            if base_color_texture is not None
            else None
        )
        has_opacity = opacity is not None
        foliage_from_surface, foliage_from_name = _foliage_sources(
            variant,
            normalized_foliage_name_matches,
            foliage_ground_types,
            has_opacity,
        )
        foliage = foliage_from_surface or foliage_from_name
        if (
            foliage_from_name
            and not foliage_from_surface
            and fallback_stats is not None
        ):
            fallback_stats["foliageNameFallbacks"] = (
                fallback_stats.get("foliageNameFallbacks", 0) + 1
            )
        stage: dict[str, object] = {
            "roughnessFactor": roughness,
            "vertColor": override.base_vertex_color if override else variant.base_vertex_color,
        }
        if base_color:
            stage["baseColorMap"] = texture_path(base_color[0], base_color[1])
        if override is not None:
            # Always emit UV index when an override is present. With two mesh UV
            # sets, omitting 0 can make the engine sample the second set.
            stage["diffuseMapUseUV"] = override.base_color_uv
        normal = slots.get("normal")
        if normal:
            stage["normalMap"] = texture_path(normal[0], normal[1])
            # RBR BTB bump pixel shader scales unpacked normals by 2.0.
            stage["normalMapStrength"] = 2.0
        if override is not None and normal:
            stage["normalMapUseUV"] = normal[2]
        roughness_texture = slots.get("roughness")
        if roughness_texture:
            stage["roughnessMap"] = texture_path(
                roughness_texture[0],
                roughness_texture[1],
            )
            stage["roughnessFactor"] = 1.0
            stage["roughnessMapUseUV"] = roughness_texture[2]
        clear_coat = slots.get("clear_coat")
        if clear_coat:
            stage["clearCoatMap"] = texture_path(clear_coat[0], clear_coat[1])
            stage["clearCoatFactor"] = 1.0
            stage["clearCoatRoughnessFactor"] = (
                override.clear_coat_roughness
                if override is not None
                and override.clear_coat_roughness is not None
                else roughness
            )
            stage["clearCoatMapUseUV"] = clear_coat[2]
        blend_pass = (
            transparent
            and material.blend_alpha_ref is not None
            and not variant.opacity_depth_prepass
        )
        if transparent and has_opacity:
            opacity_name = opacity.name if isinstance(opacity, OpacityTexture) else opacity
            stage["opacityMap"] = f"{texture_prefix}{opacity_name}"
        if variant.water:
            stage["opacityFactor"] = 0.0
        stages: list[dict[str, object]] = [stage, {}, {}, {}]
        active_layers = 1
        layer_color = slots.get("layer_color")
        layer_opacity = slots.get("layer_opacity")
        if override and layer_color and (layer_opacity or override.layer_vertex_color):
            layer: dict[str, object] = {
                "baseColorMap": texture_path(layer_color[0], layer_color[1]),
                "diffuseMapUseUV": layer_color[2],
                "roughnessFactor": roughness,
                "vertColor": override.layer_vertex_color,
            }
            if layer_opacity:
                layer["opacityMap"] = texture_path(
                    layer_opacity[0],
                    layer_opacity[1],
                )
                layer["opacityMapUseUV"] = layer_opacity[2]
            stages[1] = layer
            active_layers = 2
        entry: dict[str, object] = {
            "name": key,
            "mapTo": key,
            "class": "Material",
            "Stages": stages,
            "activeLayers": active_layers,
            "groundType": "VOID" if variant.water else variant.ground_type,
            "version": override.material_version if override else 1.5,
        }
        if variant.water:
            entry.update(_TRANSLUCENT_FLAGS)
        if variant.ground_depth > 0:
            entry["groundDepth"] = variant.ground_depth
        if material and material.double_sided:
            entry["doubleSided"] = True
        if foliage:
            entry.update(
                {
                    "doubleSided": True,
                    "invertBackFaceNormals": True,
                    "materialTag0": "beamng",
                    "materialTag1": "vegetation",
                    "subSurface": True,
                    "subSurfaceIntensity": 1,
                }
            )
            # Version 1.5 always adds a 4 % specular reflection, which turns
            # dark sunlit vegetation sprites grey; version 1.0 has none by default.
            if transparent:
                entry["version"] = 1.0
        if transparent and has_opacity:
            if blend_pass:
                entry.update(_BLEND_PASS_PROFILE)
                if material.cutout_alpha_ref is not None:
                    entry.pop("doubleSided", None)
                    entry.pop("invertBackFaceNormals", None)
                if material.blend_alpha_ref > 0:
                    entry.update(
                        {"alphaRef": material.blend_alpha_ref, "alphaTest": True}
                    )
            elif material.cutout_alpha_ref is not None:
                entry.update(
                    {
                        "alphaRef": material.cutout_alpha_ref,
                        "alphaTest": True,
                        "translucentBlendOp": "None",
                    }
                )
        output[key] = entry

    invisible_road = f"{level_id}_invisible_road"
    output[invisible_road] = {
        "name": invisible_road,
        "mapTo": invisible_road,
        "class": "Material",
        "Stages": [{"opacityFactor": 0.0}, {}, {}, {}],
        **_TRANSLUCENT_FLAGS,
        "version": 1.5,
    }
    write_json(level_dir / "art" / "shapes" / "main.materials.json", output)
    soft_ground_models = {
        variant.ground_type: _soft_ground_model(variant.soft_resistance)
        for variant in materials.values()
        if variant.soft_resistance > 0
    }
    if soft_ground_models:
        write_json(level_dir / "groundModels" / "rbr_soft.json", soft_ground_models)


# A Sunburst (1.4 t) coasting through a 4 m deep test volume met a mean resisting
# force of about 3.15 N per unit of flowConsistencyIndex at flowBehaviorIndex 0.2,
# nearly independent of speed between 7 and 23 m/s. With groundDepth 0 the same
# ground model is solid.
_SOFT_FORCE_PER_CONSISTENCY = 3.15


def _soft_ground_model(resistance: float) -> dict[str, object]:
    return {
        "staticFrictionCoefficient": 0.6,
        "slidingFrictionCoefficient": 0.6,
        "hydrodynamicFriction": 0,
        "stribeckVelocity": 6,
        "strength": 1,
        "roughnessCoefficient": 0,
        "fluidDensity": 0,
        "flowConsistencyIndex": round(resistance / _SOFT_FORCE_PER_CONSISTENCY, 1),
        "flowBehaviorIndex": 0.2,
        "dragAnisotropy": 1,
        "shearStrength": 0,
        "defaultDepth": 0,
        "collisiontype": "FOLIAGE",
        "skidMarks": False,
    }


def _sim_group(name: str, parent: str, level_id: str) -> dict[str, object]:
    return {
        "name": name,
        "class": "SimGroup",
        "persistentId": stable_uuid(level_id, "group", name),
        "__parent": parent,
    }


def _write_scene_roots(level_dir: Path, level_id: str) -> None:
    mission_group_dir = level_dir / "main" / "MissionGroup"
    write_jsonl(
        level_dir / "main" / "items.level.json",
        (
            {
                "name": "MissionGroup",
                "class": "SimGroup",
                "persistentId": stable_uuid(level_id, "MissionGroup"),
                "enabled": "1",
            },
            {
                "name": "rbr_sea_level_physics_floor",
                "class": "GroundPlane",
                "persistentId": stable_uuid(level_id, "seaLevelPhysicsFloor"),
                "__parent": "MissionGroup",
                "position": [0.0, 0.0, 0.0],
                "material": f"{level_id}_invisible_road",
            },
        ),
    )
    write_jsonl(
        mission_group_dir / "items.level.json",
        (
            _sim_group(name, "MissionGroup", level_id)
            for name in ("PlayerDropPoints", "level_objects", "StaticObjects", "route", "vegetation")
        ),
    )
    write_jsonl(
        mission_group_dir / "level_objects" / "items.level.json",
        (_sim_group(name, "level_objects", level_id) for name in ("sky_and_sun", "water")),
    )


def _write_environment(
    level_dir: Path,
    level_id: str,
    extent: float,
    stage: RbrStage,
    settings: EnvironmentSettings | None = None,
) -> None:
    location = stage.location
    visible_distance = max(5000.0, min(30000.0, extent * 2.0))
    settings = settings or resolve_environment_settings(
        stage.metadata.physics,
        location=stage.location,
    )
    day_temperature = settings.temperature_day
    night_temperature = settings.temperature_night
    azimuth = settings.sun_azimuth if settings.sun_azimuth is not None else 235.0
    elevation = settings.sun_elevation if settings.sun_elevation is not None else 35.0
    objects: list[dict[str, object]] = [
        {
            "name": "sunsky",
            "class": "ScatterSky",
            "persistentId": stable_uuid(level_id, "sunsky"),
            "__parent": "sky_and_sun",
            "ambientScale": [1, 0.92, 0.85, 1],
            "azimuth": azimuth,
            "elevation": elevation,
            "flareScale": 3,
            "flareType": "BNG_Sunflare_3",
            "skyBrightness": 35,
            "sunScale": [1, 0.9, 0.8, 1],
            "useNightCubemap": True,
            "nightCubemap": "nightCubemap",
        },
        {
            "name": "clouds",
            "class": "CloudLayer",
            "persistentId": stable_uuid(level_id, "cloudLayer"),
            "__parent": "sky_and_sun",
            "Textures": [
                {"texSpeed": 0.002},
                {"texDirection": [0.8, 0.2], "texScale": 2, "texSpeed": 0.025},
                {"texDirection": [0.2, 0.5], "texScale": 0.5, "texSpeed": 0.035},
            ],
            "baseColor": [1, 1, 1, 1],
            "coverage": settings.cloud_cover,
            "exposure": 1.5,
            "height": 7,
            "texture": "art/skies/clouds/clouds_normal_displacement.png",
            "windSpeed": settings.cloud_wind_speed,
        },
        {
            "name": "theLevelInfo",
            "class": "LevelInfo",
            "persistentId": stable_uuid(level_id, "levelInfo"),
            "__parent": "sky_and_sun",
            "canvasClearColor": [190, 210, 235, 255],
            "enabled": "1",
            "fogAtmosphereHeight": settings.fog_height,
            "fogColor": [0.74, 0.82, 0.93, 1],
            "fogDensity": settings.fog_density,
            "globalEnviromentMap": "BNG_Sky_02_cubemap",
            "gravity": -9.81,
            "temperatureCurveC": [0, day_temperature, 0.2, day_temperature, 0.5, night_temperature, 0.8, night_temperature, 1, day_temperature],
            "visibleDistance": visible_distance,
        },
    ]
    tod: dict[str, object] = {
        "name": "tod",
        "class": "TimeOfDay",
        "version": 2,
        "persistentId": stable_uuid(level_id, "timeOfDay"),
        "__parent": "sky_and_sun",
        "animate": "0",
        "play": False,
        "startTime": settings.time_of_day,
        "time": settings.time_of_day,
        "year": settings.calendar_date.year,
        "month": settings.calendar_date.month,
        "day": settings.calendar_date.day,
    }
    if location:
        tod.update(latitude=location.latitude, longitude=location.longitude)
        if settings.utc_offset is not None:
            tod["utcOffset"] = f"{settings.utc_offset:g}"
        if location.dst_rule:
            tod["dstRule"] = location.dst_rule
    objects.append(tod)
    write_jsonl(
        level_dir / "main" / "MissionGroup" / "level_objects" / "sky_and_sun" / "items.level.json",
        objects,
    )


def _write_water(
    level_dir: Path,
    level_id: str,
    waters: list[WaterSpec],
    show_wet_only: bool = False,
) -> None:
    values: list[dict[str, object]] = []
    for water in waters:
        is_plane = water.kind == "WaterPlane"
        wave_scale = 1.0 if is_plane else 0.55
        entry: dict[str, object] = {
            "name": water.name,
            "class": water.kind,
            "persistentId": stable_uuid(level_id, "water", water.name),
            "__parent": "water",
            "position": water.position,
            "Foam": [
                {"foamDir": [0, 1], "foamOpacity": 0.4, "foamSpeed": 0.01, "foamTexScale": [8, 8]},
                {"foamDir": [0, -1], "foamOpacity": 0.25, "foamSpeed": 0.006, "foamTexScale": [4, 4]},
            ],
            "Ripples (texture animation)": [
                {"rippleDir": [0, -1], "rippleMagnitude": 0.5, "rippleSpeed": 0.03, "rippleTexScale": [8, 8]},
                {"rippleDir": [0.707, 0.707], "rippleMagnitude": 0.35, "rippleSpeed": 0.015, "rippleTexScale": [16, 16]},
                {"rippleDir": [-0.5, 0.86], "rippleMagnitude": 0.2, "rippleSpeed": 0.008, "rippleTexScale": [40, 40]},
            ],
            "Waves (vertex undulation)": [
                {"waveDir": [0, -1], "waveMagnitude": 0.5 * wave_scale, "waveSpeed": 1.2},
                {"waveDir": [0.25, 0.2], "waveMagnitude": 0.35 * wave_scale, "waveSpeed": 1.5},
                {"waveDir": [0.1, -0.7], "waveMagnitude": 0.2 * wave_scale, "waveSpeed": 1.9},
            ],
            "baseColor": [55, 110, 150, 255],
            "underwaterColor": [35, 80, 105, 255],
            "clarity": 0.15,
            "cubemap": "DefaultSkyCubemap",
            "depthGradientMax": 20,
            "depthGradientTex": "core/art/water/depthcolor_ramp.png",
            "foamAmbientLerp": 1,
            "foamMaxDepth": 0.5,
            "foamRippleInfluence": 0.1,
            "foamTex": "core/art/water/foam.dds",
            "fresnelBias": 0.1,
            "fresnelPower": 4,
            "fullReflect": True,
            "overallFoamOpacity": 0.5,
            "overallRippleMagnitude": 0.45,
            "overallWaveMagnitude": 0.12 if is_plane else 0.08,
            "reflectivity": 0.7,
            "rippleTex": "core/art/water/ripple.dds",
            "specularPower": 800,
            "viscosity": 0.001,
            "waterFogDensity": 1,
            "wetDarkening": 0.35,
            "wetDepth": 0.4,
        }
        appearance = water.appearance
        if appearance is not None:
            entry.update(
                (field_name, value)
                for field_name, value in vars(appearance).items()
                if value is not None
            )
        if water.kind == "WaterBlock":
            entry["scale"] = water.scale
            entry["gridElementSize"] = water.grid_element_size if water.grid_element_size is not None else min(5.0, max(0.5, min(water.scale[0], water.scale[1]) / 8))
            if water.rotation is not None:
                entry["rotationMatrix"] = water.rotation
        elif is_plane:
            entry["gridElementSize"] = 1
            entry["gridSize"] = 200
        elif water.kind == "River":
            entry["nodes"] = water.nodes
            entry["flowMagnitudePhysics"] = 0.5
            entry["subdivideLength"] = 2
        if water.wet_only and not show_wet_only:
            entry["hidden"] = True
        values.append(entry)
    write_jsonl(
        level_dir / "main" / "MissionGroup" / "level_objects" / "water" / "items.level.json",
        values,
    )


def _write_spawns(level_dir: Path, level_id: str, spawns: list[SpawnSpec]) -> None:
    write_jsonl(
        level_dir / "main" / "MissionGroup" / "PlayerDropPoints" / "items.level.json",
        (
            {
                "name": spawn.name,
                "class": "SpawnSphere",
                "persistentId": stable_uuid(level_id, spawn.name),
                "__parent": "PlayerDropPoints",
                "position": spawn.position,
                "rotationMatrix": spawn.rotation,
                "autoplaceOnSpawn": "0",
                "dataBlock": "SpawnSphereMarker",
                "enabled": "1",
                "radius": 5,
            }
            for spawn in spawns
        ),
    )


def _write_statics(level_dir: Path, level_id: str, statics: list[StaticInstance]) -> None:
    write_jsonl(
        level_dir / "main" / "MissionGroup" / "StaticObjects" / "items.level.json",
        (
            {
                "class": "TSStatic",
                "persistentId": stable_uuid(level_id, "static", instance.key, index),
                "__parent": "StaticObjects",
                "position": instance.position,
                "rotationMatrix": instance.rotation,
                "scale": instance.scale,
                "shapeName": instance.shape_vfs,
                "collisionType": instance.collision_type,
                "decalType": instance.collision_type,
                "isRenderEnabled": instance.visible,
                "useInstanceRenderData": True,
                "originSort": True,
                "dynamic": instance.dynamic,
            }
            for index, instance in enumerate(statics)
        ),
    )


def _write_forest(level_dir: Path, level_id: str, forest_types: dict[str, ForestType]) -> None:
    vegetation_path = level_dir / "main" / "MissionGroup" / "vegetation" / "items.level.json"
    if not forest_types:
        write_jsonl(vegetation_path, ())
        return
    write_jsonl(
        vegetation_path,
        (
            {
                "name": "theForest",
                "class": "Forest",
                "persistentId": stable_uuid(level_id, "theForest"),
                "__parent": "vegetation",
                "lodReflectScalar": 0.15,
            },
        ),
    )
    managed: dict[str, object] = {}
    for key, forest_type in sorted(forest_types.items()):
        managed[key] = {
            "name": key,
            "internalName": key,
            "class": "TSForestItemData",
            "persistentId": stable_uuid(level_id, "forestData", key),
            "shapeFile": forest_type.shape_vfs,
            "collidable": forest_type.collidable,
            "radius": 0.5,
        }
        write_jsonl(level_dir / "forest" / f"{key}.forest4.json", forest_type.items)
    write_json(level_dir / "art" / "forest" / "managedItemData.json", managed)


def _converted_driveline(
    stage: RbrStage,
    origin: np.ndarray,
    map_yaw_degrees: float = 0.0,
) -> _ConvertedRoute:
    result: _ConvertedRoute = []
    for point in stage.driveline:
        position = rotate_z(
            source_position_to_beamng(point.position) - origin,
            map_yaw_degrees,
        )
        direction = rotate_z(
            source_position_to_beamng(point.direction),
            map_yaw_degrees,
        )
        length = float(np.linalg.norm(direction))
        if length > 1e-8:
            direction /= length
        result.append((position, direction, point.distance))
    return result


def _resolve_stage_start(
    stage: RbrStage,
    origin: np.ndarray,
    route: _ConvertedRoute,
    map_yaw_degrees: float = 0.0,
) -> _StageStart:
    source_position = rotate_z(
        source_position_to_beamng(stage.spawn.position) - origin,
        map_yaw_degrees,
    )
    position, direction, route_index, distance = _project_position_to_route(
        route,
        source_position,
    )
    return _StageStart(
        position,
        direction,
        route_index,
        distance,
    )


def _project_position_to_route(
    route: _ConvertedRoute,
    position: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, int, float]:
    best_distance_squared = math.inf
    result: tuple[np.ndarray, np.ndarray, int, float] | None = None
    for index, (start_position, start_direction, start_distance) in enumerate(
        route[:-1]
    ):
        end_position, end_direction, end_distance = route[index + 1]
        segment = end_position - start_position
        segment_length_squared = float(np.dot(segment, segment))
        factor = (
            0.0
            if segment_length_squared <= 1e-8
            else float(
                np.clip(
                    np.dot(position - start_position, segment)
                    / segment_length_squared,
                    0.0,
                    1.0,
                )
            )
        )
        projected_position = start_position + segment * factor
        offset = projected_position - position
        distance_squared = float(np.dot(offset, offset))
        if distance_squared >= best_distance_squared:
            continue
        direction = start_direction + (end_direction - start_direction) * factor
        direction_length = float(np.linalg.norm(direction))
        if direction_length > 1e-8:
            direction /= direction_length
        best_distance_squared = distance_squared
        result = (
            projected_position,
            direction,
            index,
            start_distance + (end_distance - start_distance) * factor,
        )
    if result is not None:
        return result
    route_index = min(
        range(len(route)),
        key=lambda index: np.linalg.norm(route[index][0] - position),
    )
    route_position, direction, distance = route[route_index]
    return route_position.copy(), direction.copy(), route_index, distance


def _route_index_at_distance(
    route: _ConvertedRoute,
    distance: float,
) -> int:
    return max(
        (
            index
            for index, (_position, _direction, route_distance) in enumerate(route)
            if route_distance <= distance
        ),
        default=0,
    )


def _route_range(
    stage: RbrStage,
    route: _ConvertedRoute,
    start_distance: float,
) -> tuple[int, int]:
    start_index = _route_index_at_distance(route, start_distance)
    finish_distance = next(
        (
            note.distance
            for note in sorted(stage.pacenotes, key=lambda item: item.distance)
            if note.note_type == 22 and note.distance >= start_distance
        ),
        route[-1][2],
    )
    finish_index = min(
        range(len(route)),
        key=lambda index: abs(route[index][2] - finish_distance),
    )
    if finish_index <= start_index:
        return 0, len(route) - 1
    return start_index, finish_index


def _timing_range(
    stage: RbrStage,
    route: _ConvertedRoute,
    route_range: tuple[int, int],
    start_distance: float | None = None,
) -> tuple[float, float]:
    start_index, finish_index = route_range
    if start_distance is None:
        start_distance = route[start_index][2]
    finish_distance = next(
        (
            note.distance
            for note in sorted(stage.pacenotes, key=lambda item: item.distance)
            if note.note_type == 22 and note.distance >= start_distance
        ),
        route[finish_index][2],
    )
    start_distance = min(max(start_distance, route[0][2]), route[-1][2])
    finish_distance = min(
        max(finish_distance, start_distance),
        route[-1][2],
    )
    return start_distance, finish_distance


def _route_splits(
    stage: RbrStage,
    route: _ConvertedRoute,
    route_range: tuple[int, int],
    start_distance: float,
) -> list[_RouteSplit]:
    start_distance, finish_distance = _timing_range(
        stage,
        route,
        route_range,
        start_distance,
    )
    result: list[_RouteSplit] = []
    for distance in sorted(
        note.distance
        for note in stage.pacenotes
        if (
            note.note_type == 23
            and start_distance < note.distance < finish_distance
        )
    ):
        route_index = _route_index_at_distance(route, distance)
        position, direction = project_route(route, distance)
        number = len(result) + 1
        result.append(
            _RouteSplit(
                name=f"spawn_split_{number}",
                label=f"Split {number}",
                distance=distance,
                route_index=route_index,
                position=position,
                direction=direction,
            )
        )
    return result


def _mission_path_points(
    route: _ConvertedRoute,
    start: _StageStart,
    finish_distance: float,
    splits: list[_RouteSplit],
) -> list[_MissionPathPoint]:
    remaining_splits = list(splits)
    points = [
        _MissionPathPoint(
            distance=start.route_distance,
            position=start.position,
            direction=start.direction,
            order=0,
        ),
    ]
    for position, direction, distance in route:
        if not start.route_distance < distance < finish_distance:
            continue
        split_index = next(
            (
                index
                for index, split in enumerate(remaining_splits)
                if math.isclose(split.distance, distance, abs_tol=1e-9)
            ),
            None,
        )
        if split_index is None:
            points.append(
                _MissionPathPoint(
                    distance=distance,
                    position=position,
                    direction=direction,
                    order=1,
                )
            )
            continue
        split = remaining_splits.pop(split_index)
        points.append(
            _MissionPathPoint(
                distance=split.distance,
                position=split.position,
                direction=split.direction,
                name=split.label,
                use_as_split=True,
                order=1,
            )
        )
    points.extend(
        _MissionPathPoint(
            distance=split.distance,
            position=split.position,
            direction=split.direction,
            name=split.label,
            use_as_split=True,
            order=2,
        )
        for split in remaining_splits
    )
    finish_position, finish_direction = project_route(route, finish_distance)
    points.append(
        _MissionPathPoint(
            distance=finish_distance,
            position=finish_position,
            direction=finish_direction,
            order=3,
        )
    )
    return sorted(points, key=lambda point: (point.distance, point.order))


def _build_spawns(
    stage: RbrStage,
    route: _ConvertedRoute,
    route_range: tuple[int, int],
    stage_start: _StageStart,
    map_yaw_degrees: float = 0.0,
    rally: bool = True,
) -> list[SpawnSpec]:
    start_position = stage_start.position.copy()
    start_position[2] += 0.5
    spawns = [
        SpawnSpec(
            name="spawn_start",
            label="Start Line" if rally else "Start",
            description=(
                "Start line of the converted rally stage."
                if rally
                else "Start of the converted level."
            ),
            position=[float(value) for value in start_position],
            rotation=spawn_rotation(
                stage.driveline[stage_start.route_index].direction,
                map_yaw_degrees,
            ),
        )
    ]
    time_control = max(
        (
            note
            for note in stage.pacenotes
            if (
                note.note_type == 21
                and note.distance < stage_start.route_distance
            )
        ),
        key=lambda note: note.distance,
        default=None,
    )
    if time_control is not None:
        position, _direction = project_route(route, time_control.distance)
        route_index = _route_index_at_distance(route, time_control.distance)
        position[2] += 0.5
        spawns.append(
            SpawnSpec(
                name="spawn_time_control",
                label="Time Control",
                description="Time control before the Start Line.",
                position=[float(value) for value in position],
                rotation=spawn_rotation(
                    stage.driveline[route_index].direction,
                    map_yaw_degrees,
                ),
            )
        )
    for split in _route_splits(
        stage,
        route,
        route_range,
        stage_start.route_distance,
    ):
        position = split.position.copy()
        position[2] += 0.5
        spawns.append(
            SpawnSpec(
                name=split.name,
                label=split.label,
                description=(
                    f"Timing split {split.label.removeprefix('Split ')} "
                    "of the converted rally stage."
                ),
                position=[float(value) for value in position],
                rotation=spawn_rotation(
                    stage.driveline[split.route_index].direction,
                    map_yaw_degrees,
                ),
            )
        )
    _start_distance, finish_distance = _timing_range(
        stage,
        route,
        route_range,
        stage_start.route_distance,
    )
    finish_control = next(
        (
            note
            for note in sorted(stage.pacenotes, key=lambda item: item.distance)
            if note.note_type == 22 and note.distance >= stage_start.route_distance
        ),
        None,
    )
    stop_control = next(
        (
            note
            for note in sorted(stage.pacenotes, key=lambda item: item.distance)
            if note.note_type == 24 and note.distance >= finish_distance
        ),
        None,
    )
    controls: list[tuple[float, str, str, str]] = []
    if finish_control is not None:
        controls.append(
            (
                finish_distance,
                "spawn_finish",
                "Stage Finish",
                "Flying finish of the converted rally stage.",
            )
        )
    if stop_control is not None:
        controls.append(
            (
                stop_control.distance,
                "spawn_stop",
                "Stop Control",
                "Stop control after the flying finish.",
            )
        )
    for distance, name, label, description in controls:
        index = _route_index_at_distance(route, distance)
        position, _direction = project_route(route, distance)
        position[2] += 0.5
        spawns.append(
            SpawnSpec(
                name=name,
                label=label,
                description=description,
                position=[float(value) for value in position],
                rotation=spawn_rotation(
                    stage.driveline[index].direction,
                    map_yaw_degrees,
                ),
            )
        )
    return spawns


def _write_route(
    level_dir: Path,
    level_id: str,
    stage: RbrStage,
    route: _ConvertedRoute,
    route_range: tuple[int, int],
    stage_start: _StageStart,
) -> None:
    _start_distance, finish_distance = _timing_range(
        stage,
        route,
        route_range,
        stage_start.route_distance,
    )
    nodes = [
        [float(value) for value in point.position] + [8.0]
        for point in _mission_path_points(
            route,
            stage_start,
            finish_distance,
            _route_splits(
                stage,
                route,
                route_range,
                stage_start.route_distance,
            ),
        )
    ]
    road = {
        "name": "rbr_driveline",
        "class": "DecalRoad",
        "persistentId": stable_uuid(level_id, "rbr_driveline"),
        "__parent": "route",
        "nodes": nodes,
        "material": f"{level_id}_invisible_road",
        "drivability": 1,
        "improvedSpline": True,
        "isRenderEnabled": False,
        "lanesLeft": 0,
        "lanesRight": 1,
        "oneWay": True,
        "overObjects": True,
        "textureLength": 32,
    }
    write_jsonl(level_dir / "main" / "MissionGroup" / "route" / "items.level.json", (road,))


def _write_rally_stage(
    mission_dir: Path,
    level_id: str,
    stage: RbrStage,
    route: _ConvertedRoute,
    route_range: tuple[int, int],
    stage_start: _StageStart,
    markers: dict[str, np.ndarray],
    warnings: list[str],
    stats: dict[str, bool | int | float | str],
    title: str,
    description: str,
    pacenote_log: Callable[[str], None] | None = None,
    pacenote_notes: Callable[[str], None] | None = None,
    pacenote_visualizer_log: Callable[[str], None] | None = None,
) -> None:
    metadata = stage.metadata
    start_distance, finish_distance = _timing_range(
        stage,
        route,
        route_range,
        stage_start.route_distance,
    )
    splits = _route_splits(
        stage,
        route,
        route_range,
        stage_start.route_distance,
    )
    path_points = _mission_path_points(
        route,
        stage_start,
        finish_distance,
        splits,
    )
    node_ids = list(range(100, 100 + len(path_points)))
    pathnodes = [
        {
            "customFields": {"values": {}, "names": {}, "tags": {}, "types": {}},
            "mode": "manual",
            "name": point.name or f"Pathnode {index + 1}",
            "navRadiusScale": 1,
            "normal": [float(value) for value in point.direction],
            "oldId": node_id,
            "pos": [float(value) for value in point.position],
            "radius": 8,
            "sidePadding": [1, 3],
            "useAsSplit": point.use_as_split,
            "visible": True,
        }
        for index, (node_id, point) in enumerate(zip(node_ids, path_points))
    ]
    segments = [
        {
            "capsules": {},
            "from": from_node,
            "mode": "waypoint",
            "name": f"Segment {index + 1}",
            "oldId": 1000 + index,
            "to": to_node,
        }
        for index, (from_node, to_node) in enumerate(zip(node_ids, node_ids[1:]))
    ]
    start_position = stage_start.position.copy()
    start_direction = stage_start.direction.copy()
    start_position[2] += 0.35
    start_quaternion = quaternion_from_direction(start_direction)
    stop_distance = next(
        (
            note.distance
            for note in sorted(stage.pacenotes, key=lambda item: item.distance)
            if note.note_type == 24 and note.distance >= finish_distance
        ),
        finish_distance,
    )
    stop_distance = min(
        max(stop_distance, finish_distance),
        route[-1][2],
    )
    stop_index = min(
        range(len(route)),
        key=lambda index: abs(route[index][2] - stop_distance),
    )
    stop_position, stop_direction = project_route(route, stop_distance)
    stop_position[2] += 0.35
    stop_quaternion = quaternion_from_direction(stop_direction)
    timestamp = int(time.time())
    race = {
        "authors": metadata.author,
        "classification": {
            "allowRollingStart": False,
            "branching": False,
            "closed": False,
            "reversible": False,
        },
        "date": timestamp,
        "defaultLaps": 1,
        "defaultStartPosition": 1,
        "description": description,
        "difficulty": 50,
        "endNode": node_ids[-1],
        "forwardPrefabs": {},
        "hideMission": False,
        "name": title,
        "pacenotes": {},
        "pathnodes": pathnodes,
        "prefabs": {},
        "reversePrefabs": {},
        "reverseStartPosition": -1,
        "rollingReverseStartPosition": -1,
        "rollingStartPosition": -1,
        "simplifyAiPath": False,
        "segments": segments,
        "startNode": node_ids[0],
        "startPositions": [
            {
                "group": "start",
                "name": "SS_start_line",
                "oldId": 1,
                "pos": [float(value) for value in start_position],
                "rot": start_quaternion,
            },
            {
                "group": "start",
                "name": "SS_stop_control",
                "oldId": 2,
                "pos": [float(value) for value in stop_position],
                "rot": stop_quaternion,
            }
        ],
    }
    write_json(mission_dir / "race.race.json", race)
    rally = build_rally_notebook(
        stage,
        route,
        route_range,
        start_distance=stage_start.route_distance,
    )
    warnings.extend(rally.warnings)
    stats.update(rally.stats)
    if pacenote_log is not None:
        for line in rally.log_lines:
            pacenote_log(line)
    if pacenote_notes is not None:
        for line in rally.notes_lines:
            pacenote_notes(line)
    visualizer_html = render_pacenote_visualizer(
        stage,
        route,
        markers,
        rally,
    )
    filesystem = current_filesystem()
    filesystem.mkdir(
        mission_dir / "rally",
        parents=True,
        exist_ok=True,
    )
    filesystem.write_text(
        mission_dir / "rally" / "pacenote_visualizer.html",
        visualizer_html,
        encoding="utf-8",
    )
    if pacenote_visualizer_log is not None:
        pacenote_visualizer_log(visualizer_html)
    write_json(mission_dir / "rally" / "drivelineSpline.json", rally.spline)
    for basename in NOTEBOOK_BASENAMES:
        write_json(
            mission_dir
            / "rally"
            / "notebooks"
            / f"{basename}.notebook.json",
            rally.notebook,
        )
    route_length = max(1.0, finish_distance - start_distance)
    mission_type_data = _LEGACY_20MPS_MISSION_PROFILE.mission_type_data(
        route_length
    )
    stats.update(
        {
            "missionAuthoringProfile": _LEGACY_20MPS_MISSION_PROFILE.name,
            "missionTimingSource": "targetProfile",
            "missionMedalSource": "targetProfile",
            "missionTargetSpeedMps": (
                _LEGACY_20MPS_MISSION_PROFILE.target_speed_mps
            ),
            "missionMinimumBaselineSeconds": (
                _LEGACY_20MPS_MISSION_PROFILE.minimum_baseline_seconds
            ),
            "missionGoldMultiplier": _LEGACY_20MPS_MISSION_PROFILE.gold_multiplier,
            "missionSilverMultiplier": _LEGACY_20MPS_MISSION_PROFILE.silver_multiplier,
            "missionBronzeMultiplier": _LEGACY_20MPS_MISSION_PROFILE.bronze_multiplier,
            "missionGoldPenaltySeconds": (
                _LEGACY_20MPS_MISSION_PROFILE.gold_penalty_seconds
            ),
            "missionSilverPenaltySeconds": (
                _LEGACY_20MPS_MISSION_PROFILE.silver_penalty_seconds
            ),
            "missionBronzePenaltySeconds": (
                _LEGACY_20MPS_MISSION_PROFILE.bronze_penalty_seconds
            ),
            "missionJustFinishPenaltySeconds": (
                _LEGACY_20MPS_MISSION_PROFILE.just_finish_penalty_seconds
            ),
        }
    )
    mission_info = {
        "author": metadata.author,
        "date": timestamp,
        "description": description,
        "devMission": False,
        "missionType": "rallyStage",
        "missionTypeData": mission_type_data,
        "name": title,
        "retryBehaviour": "infiniteRetries",
        "setupModules": {
            "environment": {"enabled": False},
            "traffic": {"enabled": False},
            "vehicles": {"enabled": True, "includePlayerVehicle": True, "vehicles": {}},
        },
        "startCondition": {"type": "automatic"},
        "startTrigger": {
            "level": level_id,
            "pos": [float(value) for value in start_position],
            "radius": 5,
            "rot": start_quaternion,
            "type": "coordinates",
        },
        "visibleCondition": {"type": "always"},
    }
    write_json(mission_dir / "info.json", mission_info)


def write_level(
    package_root: Path,
    level_id: str,
    stage: RbrStage,
    origin: np.ndarray,
    statics: list[StaticInstance],
    forest_types: dict[str, ForestType],
    waters: list[WaterSpec],
    materials: dict[str, MaterialVariant],
    bounds_min: np.ndarray,
    bounds_max: np.ndarray,
    warnings: list[str],
    stats: dict[str, bool | int | float | str],
    environment_settings: EnvironmentSettings,
    parameters: dict[str, object] | None,
    texture_progress: Callable[[int, int, Path], None] | None = None,
    texture_copy_progress: Callable[[int, int, Path], None] | None = None,
    include_mod_info: bool = True,
    use_foliage_name_fallback: bool = True,
    foliage_name_matches: tuple[str, ...] = DEFAULT_FOLIAGE_NAME_MATCHES,
    foliage_ground_types: tuple[str, ...] = DEFAULT_FOLIAGE_GROUND_TYPES,
    progress: Callable[[str], None] | None = None,
    display_environment: str | None = None,
    default_road_condition: tuple[str, str] = ("dry", "new"),
    pacenote_log: Callable[[str], None] | None = None,
    pacenote_notes: Callable[[str], None] | None = None,
    pacenote_visualizer_log: Callable[[str], None] | None = None,
    map_yaw_degrees: float = 0.0,
    cooked_textures: CookedTextures | None = None,
) -> None:
    filesystem = current_filesystem()
    level_dir = package_root / "levels" / level_id
    filesystem.mkdir(level_dir, parents=True, exist_ok=True)
    mission_dir = package_root / "gameplay" / "missions" / level_id / "rallyStage" / "stage"
    rally = stage.has_finish
    metadata = stage.metadata
    location = stage.location
    title = _stage_title(metadata, location, display_environment)
    description = _stage_description(
        metadata,
        location,
        environment=display_environment,
        rally=rally,
    )
    rally_title = _stage_title(
        metadata,
        location,
        display_environment,
        rally=True,
    )
    if progress is not None:
        progress("Writing level metadata and previews")
    route = _converted_driveline(stage, origin, map_yaw_degrees)
    stage_start = _resolve_stage_start(
        stage,
        origin,
        route,
        map_yaw_degrees,
    )
    route_range = _route_range(stage, route, stage_start.route_distance)
    spawns = _build_spawns(
        stage,
        route,
        route_range,
        stage_start,
        map_yaw_degrees,
        rally=rally,
    )
    markers = _preview_marker_positions(spawns)
    start_distance, finish_distance = _timing_range(
        stage,
        route,
        route_range,
        stage_start.route_distance,
    )
    sector_distances = [
        start_distance,
        *(
            split.distance
            for split in _route_splits(
                stage,
                route,
                route_range,
                stage_start.route_distance,
            )
        ),
        finish_distance,
    ]
    _write_previews(
        level_dir,
        mission_dir if rally else None,
        metadata,
        route,
        markers,
        spawns,
        sector_distances,
    )
    extent_values = np.maximum(abs(bounds_min), abs(bounds_max))
    extent = float(max(extent_values[0], extent_values[1], 1000.0))
    stage_length_meters = (metadata.length_km or 0.0) * 1000.0
    info = {
        "title": title,
        "description": description,
        "authors": metadata.author,
        "converterVersion": __version__,
        "sourceVersion": metadata.version,
        "previews": ["preview.jpg"],
        "size": [stage_length_meters, stage_length_meters],
        "stageLengthMeters": stage_length_meters,
        "defaultSpawnPointName": "spawn_start",
        "spawnPoints": [
            {
                "translationId": spawn.label,
                "description": spawn.description,
                "objectname": spawn.name,
                "preview": f"{spawn.name}_preview.jpg",
            }
            for spawn in spawns
        ],
        "country": location.country if location else "Unknown",
        "region": location.region if location else "",
        "roads": metadata.surface_composition_text,
        "suitablefor": "Rally and freeroam" if rally else "Freeroam",
        "supportsTraffic": False,
        "supportsTimeOfDay": True,
    }
    write_json(level_dir / "info.json", info)
    if include_mod_info:
        write_mod_info(
            package_root,
            level_id,
            metadata,
            location,
            rally=rally,
        )

    if progress is not None:
        progress("Writing level scene data")
    _write_scene_roots(level_dir, level_id)
    _write_environment(level_dir, level_id, extent, stage, environment_settings)
    _write_spawns(level_dir, level_id, spawns)
    _write_statics(level_dir, level_id, statics)
    _write_forest(level_dir, level_id, forest_types)
    _write_water(
        level_dir,
        level_id,
        waters,
        show_wet_only=default_road_condition[0] == "wet",
    )
    _write_route(
        level_dir,
        level_id,
        stage,
        route,
        route_range,
        stage_start,
    )

    with profile_span(
        "copy_textures",
        category="texture",
        materials=len(materials),
    ) as copy_span:
        if progress is not None:
            progress("Preparing level textures")
        fallback_stats = {
            "foliageNameFallbacks": 0,
        }
        (
            texture_names,
            opacity_names,
        ) = copy_textures(
            level_dir,
            materials,
            texture_copy_progress,
            warnings,
        )
        copy_span.update(
            copiedTextures=len(texture_names),
            opacityTextures=len(opacity_names),
        )
    with profile_span(
        "write_materials",
        category="material",
        materials=len(materials),
    ):
        if progress is not None:
            progress("Writing level materials")
        write_materials(
            level_dir,
            level_id,
            materials,
            texture_names,
            opacity_names,
            fallback_stats,
            foliage_name_matches=(
                foliage_name_matches if use_foliage_name_fallback else ()
            ),
            foliage_ground_types=foliage_ground_types,
        )
    stats.update(fallback_stats)
    with profile_span(
        "cook_textures",
        category="texture",
        sourceTextures=len(texture_names) + len(opacity_names),
    ) as cook_span:
        stats["cookedTextures"] = cook_textures(
            level_dir / "art" / "textures",
            texture_progress,
            cooked_textures,
        )
        cook_span.update(cookedTextures=stats["cookedTextures"])
    if rally:
        if progress is not None:
            progress("Writing rally mission data")
        _write_rally_stage(
            mission_dir,
            level_id,
            stage,
            route,
            route_range,
            stage_start,
            markers,
            warnings,
            stats,
            rally_title,
            description,
            pacenote_log=pacenote_log,
            pacenote_notes=pacenote_notes,
            pacenote_visualizer_log=pacenote_visualizer_log,
        )
    document_paths: list[str] = []
    if stage.documents:
        if progress is not None:
            progress("Copying source documents")
    for document in stage.documents:
        target = level_dir / f"rbr_source__{document.path.name}"
        filesystem.copy2(document.path, target)
        document_paths.append(target.relative_to(package_root).as_posix())

    source = {
        "format": "Original RBR" if stage.source_format == "original" else "RBR RX",
        "folder": metadata.folder_name,
        "name": metadata.name,
        "author": metadata.author,
        "physics": metadata.physics,
        "version": metadata.version,
        "date": metadata.date,
        "country": location.country if location else None,
        "variant": stage.source_variant or None,
        "provenance": stage.source_provenance,
    }
    if metadata.surface_composition:
        source["surfaceComposition"] = dict(metadata.surface_composition)
    if stage.unresolved_surface_ids:
        source["unresolvedPhysicalMaterialIds"] = list(
            stage.unresolved_surface_ids
        )
    surface_profiles = _surface_profile_provenance(stage)
    if surface_profiles:
        source["surfaceProfiles"] = surface_profiles
    manifest = {
        "converter": {"name": "RBR2BeamNG", "version": __version__},
        "parameters": parameters or {},
        "source": source,
        "sourceDocuments": document_paths,
        "stats": stats,
        "warnings": warnings,
    }
    if progress is not None:
        progress("Writing conversion manifest")
    write_json(level_dir / "rbr_conversion.json", manifest)


def validate_package(
    package_root: Path,
    level_id: str,
    archived_paths: set[str] | None = None,
    mod_info_level_id: str | None = None,
    rally: bool = True,
) -> None:
    filesystem = current_filesystem()
    package_root = filesystem.read_path(package_root)
    archived_paths = archived_paths or set()
    level_dir = package_root / "levels" / level_id
    mission_dir = package_root / "gameplay" / "missions" / level_id / "rallyStage" / "stage"
    mod_info_dir = package_root / "mod_info" / _mod_info_id(mod_info_level_id or level_id)
    mission_files = (
        (
            mission_dir / "info.json",
            mission_dir / "race.race.json",
            mission_dir / "preview.jpg",
            mission_dir / "thumbnail.jpg",
            mission_dir / "rally" / "drivelineSpline.json",
            mission_dir / "rally" / "pacenote_visualizer.html",
            *(
                mission_dir
                / "rally"
                / "notebooks"
                / f"{basename}.notebook.json"
                for basename in NOTEBOOK_BASENAMES
            ),
        )
        if rally
        else ()
    )
    required = (
        level_dir / "info.json",
        level_dir / "preview.jpg",
        level_dir / "main" / "items.level.json",
        level_dir / "main" / "MissionGroup" / "items.level.json",
        *mission_files,
        mod_info_dir / "info.json",
        mod_info_dir / "icon.jpg",
    )
    missing = [
        str(path.relative_to(package_root))
        for path in required
        if not filesystem.is_file(path)
    ]
    if missing:
        raise ConversionError(f"Generated package is missing required files: {', '.join(missing)}")
    uncooked_textures = [
        str(path.relative_to(package_root))
        for path in filesystem.rglob(package_root, "*.png")
        if path.name.casefold().endswith(_COOKABLE_TEXTURE_SUFFIXES)
    ]
    if uncooked_textures:
        raise ConversionError(
            "Generated package contains uncooked textures: "
            + ", ".join(uncooked_textures)
        )
    level_prefix = f"/levels/{level_id}/"
    references: set[str] = set()
    spawn_names: set[str] = set()
    documents: dict[Path, object] = {}

    def inspect(value: object) -> None:
        if isinstance(value, dict):
            if value.get("class") == "SpawnSphere" and isinstance(value.get("name"), str):
                spawn_names.add(value["name"])
            for key, child in value.items():
                if key in _VFS_REFERENCE_KEYS and isinstance(child, str) and child.startswith(level_prefix):
                    references.add(child)
                inspect(child)
        elif isinstance(value, list):
            for child in value:
                inspect(child)

    for path in filesystem.rglob(package_root, "*.json"):
        if path.name == "items.level.json" or path.name.endswith(".forest4.json"):
            with filesystem.open(path, "r", encoding="utf-8") as stream:
                for line_number, line in enumerate(stream, 1):
                    if line.strip():
                        try:
                            inspect(json.loads(line))
                        except json.JSONDecodeError as exc:
                            raise ConversionError(f"Invalid JSONL in {path}:{line_number}: {exc}") from exc
        else:
            try:
                value = json.loads(
                    filesystem.read_text(path, encoding="utf-8")
                )
            except json.JSONDecodeError as exc:
                raise ConversionError(f"Invalid JSON in {path}: {exc}") from exc
            documents[path] = value
            inspect(value)

    def reference_exists(reference: str) -> bool:
        relative = reference.removeprefix("/")
        path = package_root / relative
        return (
            relative in archived_paths
            or filesystem.is_file(path)
            or (
                path.suffix.casefold() == ".png"
                and path.name.casefold().endswith(_COOKABLE_TEXTURE_SUFFIXES)
                and filesystem.is_file(path.with_suffix(".dds"))
            )
        )

    missing_references = [
        reference
        for reference in sorted(references)
        if not reference_exists(reference)
    ]
    if missing_references:
        raise ConversionError(f"Generated package has missing VFS references: {', '.join(missing_references)}")

    level_info = documents[level_dir / "info.json"]
    if not isinstance(level_info, dict):
        raise ConversionError("Generated level metadata is not a JSON object")
    if level_info.get("converterVersion") != __version__:
        raise ConversionError("Generated level metadata has no valid converter version")
    mod_info = documents[mod_info_dir / "info.json"]
    if not isinstance(mod_info, dict):
        raise ConversionError("Generated mod metadata is not a JSON object")
    if mod_info.get("converter_version") != __version__:
        raise ConversionError("Generated mod metadata has no valid converter version")
    default_spawn = level_info.get("defaultSpawnPointName")
    if not isinstance(default_spawn, str) or default_spawn not in spawn_names:
        raise ConversionError(f"Default spawn point {default_spawn!r} is not present in the generated scene")
    for spawn in level_info.get("spawnPoints", []):
        preview = spawn.get("preview")
        if not isinstance(preview, str) or not filesystem.is_file(
            level_dir / preview
        ):
            raise ConversionError(f"Spawn point {spawn.get('objectname')!r} has no valid preview")
    if not rally:
        return

    race = documents[mission_dir / "race.race.json"]
    if not isinstance(race, dict):
        raise ConversionError("Generated race is not a JSON object")
    mission_info = documents[mission_dir / "info.json"]
    if (
        not isinstance(mission_info, dict)
        or mission_info.get("missionType") != "rallyStage"
    ):
        raise ConversionError("Generated mission is not a rallyStage")
    mission_type_data = mission_info.get("missionTypeData")
    mission_type_fields = {
        "baselineTime",
        "bronzeTime",
        "bronzeTimePenalty",
        "bronzeTimeTotal",
        "endScreenText",
        "goldTime",
        "goldTimePenalty",
        "goldTimeTotal",
        "justFinishPenalty",
        "outroTitleText",
        "silverTime",
        "silverTimePenalty",
        "silverTimeTotal",
        "startScreenText",
    }
    if (
        not isinstance(mission_type_data, dict)
        or set(mission_type_data) != mission_type_fields
        or mission_info.get("startCondition") != {"type": "automatic"}
    ):
        raise ConversionError("Generated rally mission data is invalid")
    start_positions = race.get("startPositions")
    if (
        not isinstance(start_positions, list)
        or any(not isinstance(value, dict) for value in start_positions)
    ):
        raise ConversionError("Generated rally start positions are invalid")
    positions_by_name = {
        value.get("name"): value
        for value in start_positions
    }
    start_position_names = {
        value.get("name")
        for value in start_positions
    }
    if not {"SS_start_line", "SS_stop_control"} <= start_position_names:
        raise ConversionError(
            "Generated rally has no SS_start_line or SS_stop_control"
        )
    if race.get("defaultStartPosition") not in {
        value.get("oldId") for value in start_positions
    }:
        raise ConversionError(
            "Generated rally has an invalid default start position"
        )
    pathnodes = race.get("pathnodes")
    if (
        not isinstance(pathnodes, list)
        or len(pathnodes) < 2
        or any(not isinstance(node, dict) for node in pathnodes)
    ):
        raise ConversionError("Generated rally pathnodes are invalid")
    node_ids = {node.get("oldId") for node in pathnodes}
    if race.get("startNode") not in node_ids or race.get("endNode") not in node_ids:
        raise ConversionError("Generated race has invalid start or end node references")
    if race.get("endNode") != pathnodes[-1].get("oldId"):
        raise ConversionError("Generated rally finish is not the last pathnode")
    for segment in race.get("segments", []):
        if segment.get("from") not in node_ids or segment.get("to") not in node_ids:
            raise ConversionError(f"Generated race segment {segment.get('name')!r} references an unknown node")

    spline = documents[mission_dir / "rally" / "drivelineSpline.json"]
    spline_nodes = spline.get("nodes") if isinstance(spline, dict) else None
    if (
        not isinstance(spline, dict)
        or spline.get("version") != 1
        or not isinstance(spline_nodes, list)
        or len(spline_nodes) < 2
        or any(
            not isinstance(node, list)
            or len(node) != 3
            or not all(
                isinstance(value, (int, float)) and math.isfinite(value)
                for value in node
            )
            for node in spline_nodes
        )
    ):
        raise ConversionError("Generated rally driveline spline is invalid")
    for key in ("nmls", "widths"):
        values = spline.get(key)
        if values is not None and (
            not isinstance(values, list)
            or len(values) != len(spline_nodes)
        ):
            raise ConversionError(
                f"Generated rally driveline spline has invalid {key}"
            )
    for name in ("SS_start_line", "SS_stop_control"):
        control_position = positions_by_name[name].get("pos")
        if (
            not isinstance(control_position, list)
            or len(control_position) != 3
            or not all(
                isinstance(value, (int, float)) and math.isfinite(value)
                for value in control_position
            )
        ):
            raise ConversionError(
                f"Generated rally control {name} has an invalid position"
            )
    if min(
        math.dist(
            positions_by_name["SS_start_line"]["pos"],
            node,
        )
        for node in spline_nodes
    ) > 25:
        raise ConversionError(
            "Generated rally driveline does not reach SS_start_line"
        )
    if math.dist(
        positions_by_name["SS_stop_control"]["pos"],
        spline_nodes[-1],
    ) > 25:
        raise ConversionError(
            "Generated rally driveline does not reach SS_stop_control"
        )

    for basename in NOTEBOOK_BASENAMES:
        path = (
            mission_dir
            / "rally"
            / "notebooks"
            / f"{basename}.notebook.json"
        )
        notebook = documents[path]
        if (
            not isinstance(notebook, dict)
            or str(notebook.get("version")) != "4"
            or not isinstance(notebook.get("pacenotes"), list)
        ):
            raise ConversionError(f"Generated rally notebook {basename} is invalid")
        identifiers: set[int] = set()
        primary_keys: set[str] = set()
        for pacenote in notebook["pacenotes"]:
            if not isinstance(pacenote, dict):
                raise ConversionError(
                    f"Generated rally notebook {basename} contains an invalid pacenote"
                )
            primary_key = pacenote.get("pk")
            if (
                not isinstance(primary_key, str)
                or not primary_key
                or primary_key in primary_keys
            ):
                raise ConversionError(
                    f"Generated rally notebook {basename} has duplicate pacenote keys"
                )
            primary_keys.add(primary_key)
            waypoints = pacenote.get("pacenoteWaypoints")
            if (
                not isinstance(waypoints, list)
                or any(not isinstance(waypoint, dict) for waypoint in waypoints)
                or not {"cornerStart", "cornerEnd"}
                <= {waypoint.get("waypointType") for waypoint in waypoints}
            ):
                raise ConversionError(
                    f"Generated rally pacenote {pacenote.get('name')!r} "
                    "has no cornerStart or cornerEnd"
                )
            structured = pacenote.get("structured")
            items = structured.get("items") if isinstance(structured, dict) else None
            if (
                not isinstance(structured, dict)
                or structured.get("schemaVersion") != 3
                or not isinstance(items, dict)
                or set(items) != {str(index) for index in range(1, 7)}
                or any(
                    not isinstance(items.get(str(index)), dict)
                    for index in range(1, 7)
                )
                or any(
                    item
                    and item.get("type") not in SUPPORTED_STRUCTURED_TYPES
                    for item in items.values()
                    if isinstance(item, dict)
                )
            ):
                raise ConversionError(
                    f"Generated rally pacenote {pacenote.get('name')!r} "
                    "has invalid structured slots"
                )
            for value in [pacenote, *waypoints]:
                identifier = value.get("oldId")
                if not isinstance(identifier, int) or identifier in identifiers:
                    raise ConversionError(
                        f"Generated rally notebook {basename} has duplicate IDs"
                    )
                identifiers.add(identifier)
