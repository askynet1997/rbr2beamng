from __future__ import annotations

import configparser
import json
import re
import shutil
import stat as stat_module
import unicodedata
import zipfile
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path, PurePosixPath
from warnings import warn

import numpy as np
from PIL import Image

from . import dds_rgb  # noqa: F401  (registers the DDS RGB decoder with Pillow)
from .core import ConversionError, ProgressReporter, slugify, temporary_root
from .filesystem import SandboxViolationError, current_filesystem
from .ini import config_parser
from .original.textures import _normalise_member
from .models import (
    DEFAULT_WATER_NAME_MATCHES,
    DrivelinePoint,
    Pacenote,
    RbrMaterial,
    RbrSurface,
    RbrStage,
    Spawn,
    StageInspection,
    StageDocument,
    StageLocation,
    StageMetadata,
    StageObject,
    SurfaceMap,
)
from .profiling import profile_span
from .surface_profiles import (
    SURFACE_COEFFICIENTS,
    current_surface_rules,
    physics_lsp_fingerprint,
    surface_profile_catalog,
)


REQUIRED_STAGE_FILES = (
    "track.ini",
    "objectlist.ini",
    "materials.ini",
    "mat.ini",
    "spawn.ini",
    "driveline.ini",
    "pacenotes.ini",
    "settings.ini",
)

# BTB_*.fx "Transparent*" techniques: pass 0 blends without depth writes and
# no alpha test; pass 1 writes depth with AlphaRef 200 / GREATEREQUAL.
BTB_TRANSPARENT_CUTOUT_ALPHA_REF = 200

NUMBER_PATTERN = re.compile(r"[-+]?(?:\d+(?:[.,]\d+)?|\.\d+)(?:[eE][-+]?\d+)?")
_SURFACE_NUMBER_PATTERN = re.compile(
    rf"^\s*({'|'.join(SURFACE_COEFFICIENTS)})[ \t]+([^\s;)]+)",
    re.MULTILINE,
)
_SURFACE_VALUE_PATTERN = re.compile(r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?")


def _read_ini_text(path: Path) -> str:
    filesystem = current_filesystem()
    for encoding in ("utf-8-sig", "cp1252", "latin-1"):
        try:
            return filesystem.read_text(path, encoding=encoding)
        except UnicodeDecodeError:
            continue
        except OSError as exc:
            raise ConversionError(f"Unable to read {path}: {exc}") from exc
    raise AssertionError("latin-1 decodes every byte")


def read_ini(path: Path) -> configparser.ConfigParser:
    parser = config_parser()
    try:
        parser.read_string(_read_ini_text(path))
    except configparser.Error as exc:
        raise ConversionError(f"Invalid INI file {path}: {exc}") from exc
    return parser


def _read_profile_ini(path: Path) -> dict[str, dict[str, str]]:
    # Win32 GetPrivateProfileStringA lookup: headers only start a line, the
    # first section and key of a name win, and values are cut to 254 chars.
    sections: dict[str, dict[str, str]] = {}
    current: dict[str, str] | None = None
    for line in _read_ini_text(path).splitlines():
        stripped = line.strip()
        if stripped.startswith("[") and "]" in stripped:
            name = stripped[1 : stripped.index("]")].strip().casefold()
            current = None if name in sections else sections.setdefault(name, {})
            continue
        key, separator, value = line.partition("=")
        if current is None or not separator:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "'\"":
            value = value[1:-1]
        current.setdefault(key.strip().casefold(), value[:254])
    return sections


def _casefold_index(
    owner: configparser.ConfigParser | configparser.SectionProxy,
    pairs: Callable[[], Iterable[tuple[str, str]]],
) -> dict[str, str]:
    # Parsers from read_ini are never modified, so the index stays valid.
    index = getattr(owner, "_rbr_casefold_index", None)
    if index is None:
        index = {}
        for key, value in pairs():
            index.setdefault(key.casefold(), value)
        owner._rbr_casefold_index = index
    return index


def get_section(parser: configparser.ConfigParser, name: str) -> configparser.SectionProxy:
    names = _casefold_index(
        parser,
        lambda: ((section_name, section_name) for section_name in parser.sections()),
    )
    section_name = names.get(name.casefold())
    if section_name is None:
        raise ConversionError(f"Missing [{name}] section")
    return parser[section_name]


def get_value(
    section: configparser.SectionProxy,
    name: str,
    default: str | None = None,
) -> str:
    value = _casefold_index(section, section.items).get(name.casefold())
    if value is not None:
        return value.strip()
    if default is not None:
        return default
    raise ConversionError(f"Missing {name} in [{section.name}]")


@lru_cache(maxsize=4096)
def texture_has_alpha(path: Path) -> bool:
    """Whether a source texture has any decoded alpha below 255."""
    try:
        with Image.open(current_filesystem().read_path(path)) as image:
            if "A" not in image.getbands() and "transparency" not in image.info:
                return False
            return image.convert("RGBA").getchannel("A").getextrema()[0] < 255
    except OSError:
        return False


def parse_float(value: str, *, field: str = "number") -> float:
    text = value.replace(",", ".") if "," in value and "." not in value else value
    try:
        return float(text)
    except ValueError as exc:
        raise ConversionError(f"Invalid {field}: {value!r}") from exc


def parse_float_list(value: str, *, expected: int | None = None, field: str = "numbers") -> tuple[float, ...]:
    simple_parts = tuple(part.strip() for part in value.split(","))
    if expected is not None and len(simple_parts) == expected:
        return tuple(parse_float(part, field=field) for part in simple_parts)

    matches = NUMBER_PATTERN.findall(value)
    if expected is not None and len(matches) != expected:
        if len(simple_parts) == expected * 2 and "." not in value:
            matches = [f"{simple_parts[index]}.{simple_parts[index + 1]}" for index in range(0, len(simple_parts), 2)]
        else:
            raise ConversionError(f"Expected {expected} {field}, got {value!r}")
    try:
        return tuple(float(item.replace(",", ".")) for item in matches)
    except ValueError as exc:
        raise ConversionError(f"Invalid {field}: {value!r}") from exc


_SCANF_FLOAT_PATTERN = re.compile(r"\s*([-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?)")


def parse_col_box(value: str) -> tuple[float, float, float] | None:
    """Read an RX colBox exactly as the RX plugin's C-locale sscanf("%f, %f, %f").

    A decimal comma ends a number, so "0,1, 1, 0,1" reads as (0, 1, 1).
    """
    values: list[float] = []
    position = 0
    for index in range(3):
        match = _SCANF_FLOAT_PATTERN.match(value, position)
        if match is None:
            return None
        values.append(float(match.group(1)))
        position = match.end()
        if index < 2:
            if not value.startswith(",", position):
                return None
            position += 1
    return values[0], values[1], values[2]


def parse_bool(value: str) -> bool:
    return value.strip().casefold() in {"1", "true", "yes", "on"}


def _filename(value: str) -> str:
    return value.replace("\\", "/").rsplit("/", 1)[-1].strip()


@dataclass(frozen=True)
class _FileIndex:
    paths: dict[str, Path]
    basenames: dict[str, tuple[Path, ...]]


@dataclass(frozen=True)
class _IndexedResolution:
    path: Path | None
    ambiguous: tuple[Path, ...] = ()


def _normalise_file_reference(value: str) -> str:
    return str(PurePosixPath(value.replace("\\", "/").strip()))


def surface_map_key(texture: Path) -> str:
    return _normalise_file_reference(str(texture)).casefold()


# BTB writes Tarmac_Fine_Covered_Dry to every cell of textures whose surface
# the author never set, including trees, fences and skies.
BTB_DEFAULT_SURFACE_ID = 53


def is_default_surface_map(surface_map: SurfaceMap) -> bool:
    return all(value == BTB_DEFAULT_SURFACE_ID for row in surface_map.cells for value in row)


SNOWWALL_SURFACE_ID = 62
SNOWWALL_BOTTOM_SURFACE_ID = 88


def is_snowwall_surface_map(surface_map: SurfaceMap) -> bool:
    return all(value == SNOWWALL_SURFACE_ID for row in surface_map.cells for value in row)


_ARCHIVE_MAX_MEMBERS = 100_000
_ARCHIVE_MAX_TOTAL_SIZE = 8 * 1024**3


def _has_asset_folder(stage_root: Path, name: str) -> bool:
    filesystem = current_filesystem()
    return filesystem.is_file(stage_root / f"{name}.zip") or filesystem.is_dir(stage_root / name)


def _asset_folder(stage_root: Path, name: str) -> Path:
    filesystem = current_filesystem()
    archive = stage_root / f"{name}.zip"
    if filesystem.is_file(archive):
        return _extracted_asset_archive(filesystem.read_path(archive), name)
    if filesystem.is_dir(stage_root / name):
        return filesystem.read_path(stage_root / name)
    raise ConversionError(f"Missing {archive} or {name} folder")


def _extracted_asset_archive(archive: Path, name: str) -> Path:
    filesystem = current_filesystem()
    stat = filesystem.stat(archive)
    cache = temporary_root() / "rx_archives" / slugify(archive.parent.name)
    target = cache / f"{name}-{stat.st_size}-{stat.st_mtime_ns}"
    if not filesystem.is_dir(target):
        with filesystem.temporary_directory(cache, prefix=f"{name}-extracting-") as temporary:
            staging = Path(temporary) / "payload"
            try:
                _extract_asset_archive(archive, staging / name, name)
            except (OSError, ValueError, zipfile.BadZipFile) as exc:
                raise ConversionError(f"Unable to extract {archive}: {exc}") from exc
            try:
                filesystem.replace(staging, target)
            except OSError:
                if not filesystem.is_dir(target):
                    raise
    return target / name


def _extract_asset_archive(archive: Path, destination: Path, name: str) -> None:
    filesystem = current_filesystem()
    with zipfile.ZipFile(filesystem.read_path(archive)) as source:
        members = source.infolist()
        if len(members) > _ARCHIVE_MAX_MEMBERS:
            raise ValueError(f"{len(members)} entries; maximum is {_ARCHIVE_MAX_MEMBERS}")
        if sum(member.file_size for member in members) > _ARCHIVE_MAX_TOTAL_SIZE:
            raise ValueError(f"uncompressed size exceeds {_ARCHIVE_MAX_TOTAL_SIZE} bytes")
        written: set[str] = set()
        for member in members:
            if (member.external_attr >> 16) & 0o170000 == stat_module.S_IFLNK:
                raise ValueError(f"symbolic link {member.filename!r}")
            parts = PurePosixPath(_normalise_member(member.filename)).parts
            if parts[0].casefold() != name.casefold():
                raise ValueError(f"entry {member.filename!r} is outside {name}/")
            relative = PurePosixPath(*parts[1:]) if len(parts) > 1 else None
            if relative is None or member.is_dir():
                continue
            if str(relative).casefold() in written:
                raise ValueError(f"duplicate entry {member.filename!r}")
            written.add(str(relative).casefold())
            target = destination.joinpath(*relative.parts)
            filesystem.mkdir(target.parent, parents=True, exist_ok=True)
            with source.open(member) as reader, filesystem.open(target, "wb") as writer:
                shutil.copyfileobj(reader, writer)


def _file_index(
    folder: Path,
    suffix: str | None = None,
    *,
    recursive: bool = False,
) -> _FileIndex:
    filesystem = current_filesystem()
    resolved_folder = filesystem.read_path(folder)
    if not resolved_folder.is_dir():
        return _FileIndex({}, {})
    paths: dict[str, Path] = {}
    basenames: dict[str, list[Path]] = {}
    entries = (
        filesystem.rglob(folder, "*")
        if recursive
        else filesystem.iterdir(folder)
    )
    for path in sorted(entries, key=lambda item: str(item).casefold()):
        resolved = filesystem.read_path(path)
        if resolved.is_file() and (
            suffix is None or path.suffix.casefold() == suffix.casefold()
        ):
            relative = _normalise_file_reference(
                str(resolved.relative_to(resolved_folder))
            )
            paths[relative.casefold()] = resolved
            basenames.setdefault(path.name.casefold(), []).append(resolved)
    return _FileIndex(
        paths,
        {
            name: tuple(sorted(candidates, key=lambda item: str(item).casefold()))
            for name, candidates in basenames.items()
        },
    )


def _resolve_indexed(
    index: _FileIndex,
    value: str,
    *,
    suffix: str = "",
) -> _IndexedResolution:
    reference = _normalise_file_reference(value)
    if suffix and not reference.casefold().endswith(suffix.casefold()):
        reference += suffix
    if "/" in reference:
        path = index.paths.get(reference.casefold())
        if path is not None:
            return _IndexedResolution(path)
    basename = PurePosixPath(reference).name.casefold()
    candidates = index.basenames.get(basename, ())
    if len(candidates) == 1:
        return _IndexedResolution(candidates[0])
    return _IndexedResolution(None, candidates if len(candidates) > 1 else ())


def read_metadata(stage_root: Path) -> StageMetadata:
    filesystem = current_filesystem()
    section = get_section(read_ini(stage_root / "track.ini"), "INFO")
    splash_name = get_value(section, "splashscreen", "")
    splash = stage_root / _filename(splash_name) if splash_name else None
    if splash and not filesystem.is_file(splash):
        splash = None
    raw_length = get_value(section, "length", "")
    return StageMetadata(
        folder_name=stage_root.name,
        name=get_value(section, "name", stage_root.name),
        author=get_value(section, "author", "Unknown"),
        physics=get_value(section, "physics", "unknown").lower(),
        version=get_value(section, "version", ""),
        date=get_value(section, "date", ""),
        comment=get_value(section, "comment", ""),
        length_km=parse_float(raw_length, field="stage length") if raw_length else None,
        splashscreen=splash,
    )


def inspect_stage(stage_root: Path) -> StageInspection:
    filesystem = current_filesystem()
    issues: list[str] = []
    for filename in REQUIRED_STAGE_FILES:
        if not filesystem.is_file(stage_root / filename):
            issues.append(f"Missing {filename}")
    for name in ("Objects", "Textures"):
        if not _has_asset_folder(stage_root, name):
            issues.append(f"Missing {name}.zip or {name} folder")
    try:
        metadata = read_metadata(stage_root)
    except ConversionError as exc:
        metadata = StageMetadata(stage_root.name, stage_root.name, "Unknown", "unknown", "", "", "", None, None)
        issues.append(str(exc))
    location = _read_location(stage_root, metadata)[0]
    return StageInspection(
        root=stage_root,
        metadata=metadata,
        valid=not issues,
        issues=tuple(issues),
        location=location,
        source_key=f"rx:{metadata.folder_name}",
        documents=discover_stage_documents(stage_root),
    )


def discover_stages(rbr_root: Path, reporter: ProgressReporter | None = None) -> list[StageInspection]:
    filesystem = current_filesystem()
    tracks_root = filesystem.read_path(
        filesystem.read_path(rbr_root) / "RX_CONTENT" / "TRACKS"
    )
    stage_roots = sorted(
        (
            filesystem.read_path(entry)
            for entry in filesystem.iterdir(tracks_root)
            if filesystem.is_dir(entry)
            and filesystem.is_file(entry / "track.ini")
        ),
        key=lambda item: item.name.casefold(),
    )
    result: list[StageInspection] = []
    for index, stage_root in enumerate(stage_roots, 1):
        if reporter:
            reporter.emit("scan", "Inspecting stages", current=index, total=len(stage_roots), detail=stage_root.name)
        result.append(inspect_stage(stage_root))
    return result


BTB_TERRAIN_MESH_PATTERNS = (r"t_\d+", r"ta\d+", r"wall\d+")


def is_btb_terrain_mesh(
    path: Path | str,
    patterns: tuple[str, ...] = BTB_TERRAIN_MESH_PATTERNS,
) -> bool:
    stem = Path(path).stem
    return any(re.match(pattern, stem, flags=re.IGNORECASE) for pattern in patterns)


def _read_objects(stage_root: Path) -> tuple[list[StageObject], list[str]]:
    filesystem = current_filesystem()
    parser = read_ini(stage_root / "objectlist.ini")
    count = int(get_value(get_section(parser, "OBJECTS"), "count"))
    objects_folder = _asset_folder(stage_root, "Objects")
    mesh_index = _file_index(objects_folder, ".x")
    transform_index = _file_index(objects_folder, ".xform")
    objects: list[StageObject] = []
    warnings: list[str] = []
    for index in range(count):
        section = get_section(parser, f"Object_{index}")
        mesh_name = get_value(section, "file")
        transform_name = get_value(section, "xform")
        mesh = _resolve_indexed(mesh_index, mesh_name, suffix=".x")
        transform = _resolve_indexed(transform_index, transform_name, suffix=".xform")
        if mesh.path is None:
            detail = (
                f"ambiguous mesh {mesh_name!r}: "
                + ", ".join(str(path) for path in mesh.ambiguous)
                if mesh.ambiguous
                else f"missing mesh {mesh_name}.x"
            )
            warnings.append(f"Object {index} references {detail} and was skipped")
            continue
        if transform.path is None:
            detail = (
                f"ambiguous transform {transform_name!r}: "
                + ", ".join(str(path) for path in transform.ambiguous)
                if transform.ambiguous
                else f"missing transform {transform_name}.xform"
            )
            warnings.append(f"Object {index} references {detail} and was skipped")
            continue
        mesh_path = mesh.path
        transform_path = transform.path
        declared_clone_count = int(get_value(section, "cloneCount", "1"))
        actual_size = filesystem.stat(transform_path).st_size
        if actual_size == 0 or actual_size % 64 != 0:
            warnings.append(
                f"Object {index} transform {transform_path.name} has invalid size {actual_size} and was skipped"
            )
            continue
        stored_clone_count = actual_size // 64
        clone_count = min(declared_clone_count, stored_clone_count)
        if stored_clone_count != declared_clone_count:
            warnings.append(
                f"{transform_path.name} declares {declared_clone_count} transforms but contains {stored_clone_count}; "
                f"using {clone_count}"
            )
        col_box_raw = get_value(section, "colBox", "")
        collision_box = parse_col_box(col_box_raw) if col_box_raw else None
        if col_box_raw and collision_box is None:
            warnings.append(
                f"Object {index} colBox {col_box_raw!r} is not three numbers as RBR reads it; "
                "no collision box was emitted"
            )
        shadow_caster_raw = get_value(section, "ShadowCaster", "")
        objects.append(
            StageObject(
                index=index,
                mesh_name=mesh_name,
                mesh_path=mesh_path,
                transform_path=transform_path,
                clone_count=clone_count,
                lod_in=parse_float(get_value(section, "LODIn", "0"), field="LODIn"),
                lod_out=parse_float(get_value(section, "LODOut", "0"), field="LODOut"),
                visible=parse_bool(get_value(section, "Visible", "1")),
                draw_instanced=parse_bool(get_value(section, "DrawInstanced", "0")),
                moveable=parse_bool(get_value(section, "Moveable", "0")),
                shadow_caster=(
                    parse_bool(shadow_caster_raw)
                    if shadow_caster_raw
                    else None
                ),
                collision_model=int(get_value(section, "CollisionModel", "0")),
                collision_box=collision_box or (),
            )
        )
    return objects, warnings


def load_transforms(stage_object: StageObject) -> np.ndarray:
    values = np.fromfile(
        current_filesystem().read_path(stage_object.transform_path),
        dtype="<f4",
    )
    required = stage_object.clone_count * 16
    if values.size < required:
        raise ConversionError(f"Invalid transform count in {stage_object.transform_path}")
    return values[:required].reshape((-1, 4, 4))


def _read_material_manifest(
    stage_root: Path,
) -> tuple[configparser.ConfigParser, configparser.SectionProxy, int]:
    parser = read_ini(stage_root / "materials.ini")
    root = get_section(parser, "MATERIALS")
    return parser, root, int(get_value(root, "count"))


def _material_matches(
    material: RbrMaterial | None,
    predicate: Callable[[str, str], object],
) -> bool:
    if material is None:
        return False
    texture = material.diffuse_texture
    return bool(predicate(material.name, texture.name if texture else ""))


def _reference_words(name: str, diffuse_texture: str = "") -> set[str]:
    texture_name = Path(_filename(diffuse_texture)).stem
    normalized = unicodedata.normalize("NFKD", f"{name} {texture_name}")
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii")
    return set(re.findall(r"[a-z0-9]+", ascii_value.casefold()))


def is_water_material_reference(
    name: str,
    diffuse_texture: str = "",
    name_matches: tuple[str, ...] = DEFAULT_WATER_NAME_MATCHES,
) -> bool:
    words = _reference_words(name, diffuse_texture)
    blocked = [match[1:] for match in name_matches if match.startswith("!")]
    if any(term in word for word in words for term in blocked):
        return False
    return any(
        word.startswith(term)
        for word in words
        for term in name_matches
        if not term.startswith("!")
    )


def is_water_material(
    material: RbrMaterial | None,
    name_matches: tuple[str, ...] = DEFAULT_WATER_NAME_MATCHES,
) -> bool:
    return _material_matches(
        material,
        lambda name, diffuse_texture: is_water_material_reference(
            name,
            diffuse_texture,
            name_matches,
        ),
    )


def _effect_is_double_sided(effect_index: _FileIndex, effect: str) -> bool:
    effect_path = _resolve_indexed(effect_index, effect, suffix=".fx").path
    if effect_path is None:
        return False
    text = current_filesystem().read_text(
        effect_path,
        encoding="cp1252",
        errors="replace",
    )
    text = re.sub(r"/\*.*?\*/|//[^\n]*", "", text, flags=re.DOTALL)
    cull_modes = re.findall(r"\bCullMode\s*=\s*([A-Za-z_]+)", text, flags=re.IGNORECASE)
    return bool(cull_modes) and all(mode.casefold() == "none" for mode in cull_modes)


def _read_materials(stage_root: Path) -> tuple[list[RbrMaterial], list[str]]:
    parser, root, count = _read_material_manifest(stage_root)
    texture_index = _file_index(_asset_folder(stage_root, "Textures"), recursive=True)
    effect_index = _file_index(stage_root / "FX", suffix=".fx")
    double_sided_effects: dict[str, bool] = {}
    materials: list[RbrMaterial] = []
    warnings: list[str] = []

    def resolve_texture(value: str, material_name: str) -> Path | None:
        if not value:
            return None
        resolution = _resolve_indexed(texture_index, value)
        if resolution.path is not None:
            return resolution.path
        if resolution.ambiguous:
            warnings.append(
                f"Material {material_name!r} references ambiguous texture {value!r}: "
                + ", ".join(str(path) for path in resolution.ambiguous)
                + "; using material factors only"
            )
        else:
            warnings.append(
                f"Material {material_name!r} references unavailable texture {_filename(value)!r}; using material factors only"
            )
        return None

    for index in range(count):
        name = get_value(root, f"MAT_{index}")
        section = get_section(parser, name)
        effect = get_value(section, "effectName", "BTB_Diffuse")
        effect_key = effect.casefold()
        if effect_key not in double_sided_effects:
            double_sided_effects[effect_key] = _effect_is_double_sided(effect_index, effect)
        technique = get_value(section, "techniqueName", "Default")
        diffuse_texture = resolve_texture(get_value(section, "diffuseTexture", ""), name)
        transparent = (
            technique.casefold().startswith("transparent")
            and diffuse_texture is not None
            and texture_has_alpha(diffuse_texture)
        )
        materials.append(
            RbrMaterial(
                index=index,
                name=name,
                effect=effect,
                technique=technique,
                diffuse_texture=diffuse_texture,
                second_diffuse_texture=resolve_texture(get_value(section, "secondDiffuseTexture", ""), name),
                normal_texture=resolve_texture(
                    get_value(section, "normalTexture", get_value(section, "bumpTexture", "")),
                    name,
                ),
                specular_texture=resolve_texture(get_value(section, "specularTexture", ""), name),
                properties={key: value.strip() for key, value in section.items()},
                double_sided=double_sided_effects[effect_key],
                multiplier_texture=resolve_texture(get_value(section, "multiplierTexture", ""), name),
                additive_texture=resolve_texture(get_value(section, "additiveTexture", ""), name),
                cutout_alpha_ref=BTB_TRANSPARENT_CUTOUT_ALPHA_REF if transparent else None,
                blend_alpha_ref=0 if transparent else None,
            )
        )
    return materials, warnings


_SCANF_INT = re.compile(r"\s*([-+]?\d+)")


def _surface_map_rows(
    section: dict[str, str],
    label: str,
    warnings: list[str],
) -> tuple[tuple[int, ...], ...] | None:
    ids: list[int | None] = [None] * 16
    rows: list[tuple[int, ...]] = []
    for row_index in range(16):
        key = f"data{row_index}"
        text = section.get(key)
        if text is None:
            warnings.append(f"{label} has no {key}; RSF reads surface 0 for that whole row")
            text = " ".join(["0"] * 16)
        parsed: list[int] = []
        position = 0
        while len(parsed) < 16 and (match := _SCANF_INT.match(text, position)):
            parsed.append(int(match.group(1)))
            position = match.end()
        ids[: len(parsed)] = parsed
        if len(parsed) < 16:
            if None in ids:
                warnings.append(
                    f"{label} has {len(parsed)} surface IDs in {key}; RSF leaves the "
                    "remaining cells undefined, so the map is ignored"
                )
                return None
            warnings.append(
                f"{label} has {len(parsed)} surface IDs in {key}; RSF keeps the "
                "previous row's IDs in the remaining columns"
            )
        elif _SCANF_INT.match(text, position):
            warnings.append(f"{label} has more than 16 surface IDs in {key}; RSF reads the first 16")
        rows.append(tuple(value & 0xFF for value in ids))
    return tuple(rows)


def _read_surface_maps(
    stage_root: Path,
) -> tuple[dict[str, SurfaceMap], SurfaceMap | None, list[str]]:
    """Read ``mat.ini`` maps keyed by texture, and the map for unmatched textures.

    The RX plugin writes map index 0 for a mesh whose first texture matches no
    map, so RBR reads MAP0 for it as the plugin compiled it: 16 rows of
    surface 0 when the stage has no MAP0.
    """
    sections = _read_profile_ini(stage_root / "mat.ini")
    texture_index = _file_index(_asset_folder(stage_root, "Textures"), recursive=True)
    result: dict[str, SurfaceMap] = {}
    unmatched: SurfaceMap | None = SurfaceMap(Path("none"), ((0,) * 16,) * 16)
    warnings: list[str] = []
    count_match = _SCANF_INT.match(sections.get("maps", {}).get("count", "0"))
    count = int(count_match.group(1)) if count_match else 0
    for map_index in range(count):
        section_name = f"MAP{map_index}"
        section = sections.get(section_name.casefold())
        if section is None:
            warnings.append(
                f"Surface map [{section_name}] is missing although [MAPS] count is "
                f"{count}; RSF reads no texture for it"
            )
            continue
        raw_texture = section.get("texture", "none")
        resolution = _resolve_indexed(texture_index, raw_texture)
        texture = resolution.path
        if texture is None:
            if resolution.ambiguous:
                warnings.append(
                    f"Surface map references ambiguous texture {raw_texture!r}: "
                    + ", ".join(str(path) for path in resolution.ambiguous)
                    + "; ignoring it"
                )
            else:
                warnings.append(f"Surface map references unavailable texture {_filename(raw_texture)!r}; ignoring it")
            if map_index:
                continue
        rows = _surface_map_rows(
            section,
            f"Surface map [{section_name}] for texture {_filename(raw_texture)!r}",
            warnings,
        )
        if map_index == 0:
            unmatched = (
                SurfaceMap(texture or Path(_filename(raw_texture)), rows)
                if rows is not None
                else None
            )
        if rows is not None and texture is not None:
            result[surface_map_key(texture)] = SurfaceMap(texture, rows)
    return result, unmatched, warnings


def read_physics_lsp(rbr_root: Path) -> bytes | None:
    """RBR reads a loose Physics/physics.lsp first, then the one in physics.rbz."""
    filesystem = current_filesystem()
    loose = rbr_root / "Physics" / "physics.lsp"
    if filesystem.is_file(loose):
        return filesystem.read_bytes(loose)
    archive = rbr_root / "physics.rbz"
    if not filesystem.is_file(archive):
        return None
    try:
        with filesystem.open(archive, "rb") as stream, zipfile.ZipFile(stream) as source:
            for member in source.infolist():
                if member.filename.replace("\\", "/").casefold() == "physics/physics.lsp":
                    return source.read(member)
    except (OSError, zipfile.BadZipFile):
        return None
    return None


def read_rbr_surfaces(rbr_root: Path) -> dict[int, RbrSurface]:
    data = read_physics_lsp(rbr_root)
    if data is None:
        return {}
    text = data.decode("cp1252", errors="replace")
    fingerprint = physics_lsp_fingerprint(data)
    catalog = surface_profile_catalog()
    result: dict[int, RbrSurface] = {}
    pattern = re.compile(
        r"^([A-Za-z0-9_]+)\s+\([^;\r\n]*;\s*id\s+(\d+)[^\r\n]*\r?\n(.*?)^\)",
        re.MULTILINE | re.DOTALL,
    )
    for name, raw_id, body in pattern.findall(text):
        raw_flags = match.group(1) if (match := re.search(r"^\s*Flags\s+([^\r\n]+)", body, re.MULTILINE)) else ""
        flags = tuple(re.findall(r"[+-][A-Z0-9_]+", raw_flags))
        numbers: dict[str, float] = {}
        for field, value in _SURFACE_NUMBER_PATTERN.findall(body):
            if _SURFACE_VALUE_PATTERN.fullmatch(value):
                numbers[field] = float(value)
            else:
                warn(f"Ignoring malformed {field} value {value!r} of surface {name} in physics.lsp")
        surface_id = int(raw_id)
        profile, profile_status = catalog.resolve(name, numbers, flags)
        result[surface_id] = RbrSurface(
            surface_id=surface_id,
            name=name,
            ground_type=profile.ground_type,
            wetness="",
            depth_class="",
            hard=profile.hard,
            bendable=profile.bendable,
            water_factor=numbers.get("WaterFactor", 0.0),
            soil_thickness=numbers.get("SoilThickness", 0.0),
            coefficients=numbers,
            flags=flags,
            profile=profile,
            profile_status=profile_status,
            physics_fingerprint=fingerprint,
            catalog_version=catalog.version,
        )
    return result


def surface_profile_warnings(
    surface_ids: Iterable[int],
    surfaces: Mapping[int, RbrSurface],
) -> list[str]:
    if not surfaces:
        return [
            "No surface definitions found in Physics/physics.lsp or physics.rbz; "
            f"every surface uses the {current_surface_rules().unknown_ground_type} fallback"
        ]
    derived = [
        f"{surface_id} {surfaces[surface_id].name} as {surfaces[surface_id].ground_type}"
        for surface_id in sorted(set(surface_ids))
        if surface_id in surfaces and surfaces[surface_id].profile_status == "derived"
    ]
    if not derived:
        return []
    return [
        "Surfaces missing from the surface catalog were classified from their "
        "physics.lsp Type, WaterFactor and Flags: " + ", ".join(derived)
    ]


DOCUMENT_TOKENS = (
    "readme",
    "license",
    "licence",
    "copyright",
    "terms",
    "permission",
    "attribution",
    "credits",
)
PLAIN_TEXT_DOCUMENT_SUFFIXES = {".txt", ".md", ".nfo", ".ini"}
DOCUMENT_SUFFIXES = PLAIN_TEXT_DOCUMENT_SUFFIXES | {".rtf", ".html", ".htm", ".pdf"}


def discover_stage_documents(stage_root: Path) -> tuple[StageDocument, ...]:
    filesystem = current_filesystem()
    documents = (
        StageDocument(path=path, title=path.name)
        for path in filesystem.iterdir(stage_root)
        if filesystem.is_file(path)
        and (suffix := path.suffix.casefold()) in DOCUMENT_SUFFIXES
        and (
            (name := path.name.casefold()) == "track.ini"
            or suffix == ".nfo"
            or any(token in name for token in DOCUMENT_TOKENS)
        )
    )
    return tuple(sorted(
        documents,
        key=lambda document: (
            document.path.suffix.casefold() not in PLAIN_TEXT_DOCUMENT_SUFFIXES,
            document.title.casefold(),
        ),
    ))


@lru_cache(maxsize=1)
def _location_data() -> dict[str, dict[str, object]]:
    path = current_filesystem().read_path(
        Path(__file__).parent / "data" / "locations.json"
    )
    return json.loads(current_filesystem().read_text(path, encoding="utf-8"))


@lru_cache(maxsize=4)
def rsf_catalog_entries(catalog_path: Path) -> list[dict[str, object]]:
    try:
        data = json.loads(_read_ini_text(catalog_path))
    except (ConversionError, ValueError) as exc:
        warn(f"Ignoring RSF stage list {catalog_path}: {exc}")
        return []
    if not isinstance(data, list):
        warn(f"Ignoring RSF stage list {catalog_path}: it is not a list of stages")
        return []
    return [entry for entry in data if isinstance(entry, dict)]


def _catalog_country_code(stage_root: Path, metadata: StageMetadata) -> tuple[str, str]:
    filesystem = current_filesystem()
    catalog_path = stage_root.parents[2] / "rsfdata" / "cache" / "stages_data.json"
    if not filesystem.is_file(catalog_path):
        return "", ""
    entries = rsf_catalog_entries(filesystem.read_path(catalog_path))
    target = metadata.name.casefold()
    matches = [entry for entry in entries if str(entry.get("name", "")).casefold() == target]
    if not matches:
        # stages missing from the list, like Te Parae Road 2021 R, are where their author's other stages are
        author = metadata.author.casefold()
        codes = {
            str(entry.get("short_country") or "").upper()
            for entry in entries
            if str(entry.get("author", "")).casefold() == author
        }
        return (next(iter(codes)) if len(codes) == 1 else ""), ""
    codes = {str(entry.get("short_country") or "").upper() for entry in matches}
    if len(codes) > 1:
        candidates = ", ".join(
            f"stage {entry.get('stage_id', entry.get('id'))} in {entry.get('short_country')}"
            for entry in matches
        )
        return "", f"Stage country is ambiguous: RSF's stage list has {metadata.name!r} as {candidates}"
    return next(iter(codes), ""), ""


def location_for_country_code(country_code: str) -> StageLocation:
    code = country_code.strip().upper()
    locations = _location_data()
    if code not in locations:
        code = "EU"
    country = locations[code]
    return StageLocation(
        country_code=code,
        country=str(country["country"]),
        region=str(country["region"]),
        latitude=float(country["latitude"]),
        longitude=float(country["longitude"]),
        utc_offset=str(country["utcOffset"]),
        dst_rule=str(country["dstRule"]),
        precision="country",
        altitude_meters=float(country["altitudeMeters"]),
        summer_temperature_night=float(country["summerNight"]),
        summer_temperature_day=float(country["summerDay"]),
        autumn_temperature_night=float(country["autumnNight"]),
        autumn_temperature_day=float(country["autumnDay"]),
        winter_temperature_night=float(country["winterNight"]),
        winter_temperature_day=float(country["winterDay"]),
    )


@lru_cache(maxsize=1)
def country_locations() -> tuple[StageLocation, ...]:
    return tuple(
        sorted(
            (
                location_for_country_code(country_code)
                for country_code in _location_data()
            ),
            key=lambda location: location.country.casefold(),
        )
    )


def location_warning(country_code: str, ambiguity: str = "") -> str:
    code = country_code.strip().upper()
    if code in _location_data():
        country = _location_data()[code]["country"]
        return f"Location uses the approximate centroid of {country} ({code}); map altitude uses its country average"
    if code:
        return f"Stage country {code} has no location data; using Europe defaults"
    return f"{ambiguity or 'Stage country is unknown'}; using Europe defaults"


def _read_location(stage_root: Path, metadata: StageMetadata) -> tuple[StageLocation | None, list[str]]:
    code, ambiguity = _catalog_country_code(stage_root, metadata)
    return location_for_country_code(code), [location_warning(code, ambiguity)]


def _read_spawn(stage_root: Path) -> Spawn:
    section = get_section(read_ini(stage_root / "spawn.ini"), "SPAWN")
    matrix = tuple(parse_float(get_value(section, f"m{index:02d}", "0"), field=f"spawn m{index:02d}") for index in range(16))
    angles = tuple(parse_float(get_value(section, f"angle{axis}", "0"), field=f"spawn angle{axis}") for axis in "XYZ")
    return Spawn(matrix, angles)


def _read_driveline(stage_root: Path) -> list[DrivelinePoint]:
    section = get_section(read_ini(stage_root / "driveline.ini"), "DRIVELINE")
    count = int(get_value(section, "count"))
    result: list[DrivelinePoint] = []
    for index in range(count):
        values = parse_float_list(get_value(section, f"K{index}"), expected=8, field=f"driveline K{index}")
        result.append(
            DrivelinePoint(
                position=(values[0], values[2], values[1]),
                direction=(values[3], values[5], values[4]),
                distance=values[6],
                flags=int(values[7]),
            )
        )
    return result


def read_pacenotes(stage_root: Path) -> list[Pacenote]:
    parser = read_ini(stage_root / "pacenotes.ini")
    root = get_section(parser, "PACENOTES")
    count = int(get_value(root, "count"))
    result: list[Pacenote] = []
    for index in range(count):
        section = get_section(parser, f"P{index}")
        result.append(
            Pacenote(
                note_type=int(get_value(section, "type")),
                distance=parse_float(get_value(section, "distance"), field=f"pacenote {index} distance"),
                flag=int(get_value(section, "flag", "0")),
            )
        )
    return result


def load_pacenote_visualizer_stage(stage_root: Path) -> RbrStage:
    metadata = read_metadata(stage_root)
    driveline = _read_driveline(stage_root)
    if len(driveline) < 2:
        raise ConversionError(
            f"RX stage {stage_root} needs at least two driveline points"
        )
    return RbrStage(
        root=stage_root,
        metadata=metadata,
        objects=[],
        materials=[],
        surface_maps={},
        surface_types={},
        surfaces={},
        spawn=_read_spawn(stage_root),
        driveline=driveline,
        pacenotes=read_pacenotes(stage_root),
    )


def load_stage(
    stage_root: Path,
    reporter: ProgressReporter | None = None,
) -> RbrStage:
    inspection = inspect_stage(stage_root)
    if not inspection.valid:
        raise ConversionError(f"Invalid RX stage {stage_root}: {'; '.join(inspection.issues)}")

    def progress(message: str) -> None:
        if reporter:
            reporter.emit("parse", message, detail=stage_root.name)

    progress("Reading metadata")
    metadata = inspection.metadata
    progress("Reading object manifest")
    with profile_span("read_objects", category="parse") as object_span:
        objects, object_warnings = _read_objects(stage_root)
        object_span.update(
            objects=len(objects),
            instances=sum(item.clone_count for item in objects),
        )
    progress("Reading materials")
    with profile_span("read_materials", category="parse") as material_span:
        materials, material_warnings = _read_materials(stage_root)
        material_span.update(materials=len(materials))
    progress("Reading physical surfaces")
    with profile_span("read_surfaces", category="parse") as surface_span:
        surface_maps, unmatched_surface_map, surface_warnings = _read_surface_maps(
            stage_root
        )
        surfaces = read_rbr_surfaces(stage_root.parents[2])
        surface_types = {surface_id: surface.ground_type for surface_id, surface in surfaces.items()}
        used_surface_ids = tuple(
            sorted(
                {
                    surface_id
                    for surface_map in (
                        *surface_maps.values(),
                        *((unmatched_surface_map,) if unmatched_surface_map else ()),
                    )
                    for row in surface_map.cells
                    for surface_id in row
                }
            )
        )
        unresolved_surface_ids = tuple(
            surface_id
            for surface_id in used_surface_ids
            if surface_id not in surfaces
            or surfaces[surface_id].profile_status == "unmapped"
        )
        location, location_warnings = _read_location(stage_root, metadata)
        documents = inspection.documents
        surface_span.update(
            surfaceMaps=len(surface_maps),
            surfaces=len(surfaces),
            documents=len(documents),
        )
    progress("Reading route data")
    with profile_span("read_route", category="parse") as route_span:
        spawn = _read_spawn(stage_root)
        driveline = _read_driveline(stage_root)
        pacenotes = read_pacenotes(stage_root)
        settings = get_section(read_ini(stage_root / "settings.ini"), "SETTINGS")
        sun_direction_raw = get_value(settings, "sunDirection", "")
        sun_direction = parse_float_list(sun_direction_raw, expected=3, field="sun direction") if sun_direction_raw else None
        route_span.update(
            drivelinePoints=len(driveline),
            pacenotes=len(pacenotes),
        )

    warnings = object_warnings + material_warnings + surface_warnings + location_warnings
    warnings.extend(surface_profile_warnings(used_surface_ids, surfaces))
    if unresolved_surface_ids:
        warnings.append(
            "MAT maps use physical material IDs that physics.lsp does not define: "
            + ", ".join(str(surface_id) for surface_id in unresolved_surface_ids)
        )
    if not driveline:
        raise ConversionError("Stage has no driveline points")
    if not any(note.note_type == 22 for note in pacenotes):
        warnings.append(
            "No RBR finish marker (pacenote type 22); converting it as a freeroam-only level"
        )
    return RbrStage(
        root=stage_root,
        metadata=metadata,
        objects=objects,
        materials=materials,
        surface_maps=surface_maps,
        surface_types=surface_types,
        surfaces=surfaces,
        spawn=spawn,
        driveline=driveline,
        pacenotes=pacenotes,
        location=location,
        documents=documents,
        sun_direction=sun_direction,
        warnings=warnings,
        source_provenance={
            "root": str(stage_root),
            "physicsLspFingerprint": next(
                (surface.physics_fingerprint for surface in surfaces.values()),
                "",
            ),
            "surfaceProfileCatalogVersion": str(
                next((surface.catalog_version for surface in surfaces.values()), 0)
            ),
        },
        unresolved_surface_ids=unresolved_surface_ids,
        used_surface_ids=used_surface_ids,
        unmatched_surface_map=unmatched_surface_map,
    )


def find_stage(rbr_root: Path, stage_id: str) -> Path:
    filesystem = current_filesystem()
    tracks_root = filesystem.read_path(
        filesystem.read_path(rbr_root) / "RX_CONTENT" / "TRACKS"
    )
    try:
        direct = filesystem.read_path(tracks_root / stage_id)
        direct.relative_to(tracks_root)
    except (SandboxViolationError, ValueError):
        direct = None
    if direct is not None and filesystem.is_dir(direct):
        return direct
    target = stage_id.casefold()
    matches = [
        filesystem.read_path(entry)
        for entry in filesystem.iterdir(tracks_root)
        if filesystem.is_dir(entry)
        and (
            entry.name.casefold() == target
            or (
                filesystem.is_file(entry / "track.ini")
                and read_metadata(entry).name.casefold() == target
            )
        )
    ]
    if not matches:
        raise ConversionError(f"RX stage not found: {stage_id}")
    if len(matches) > 1:
        raise ConversionError(f"Stage name is ambiguous: {stage_id}")
    return matches[0]
