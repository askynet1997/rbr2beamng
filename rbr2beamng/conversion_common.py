from __future__ import annotations

import errno
import json
import math
import os
import uuid
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Mapping

from .core import (
    ConversionError,
    ProgressReporter,
    settings_path,
    temporary_root,
)
from .filesystem import FileSandbox, SandboxViolationError
from .models import (
    DEFAULT_FOLIAGE_GROUND_TYPES,
    DEFAULT_FOLIAGE_NAME_MATCHES,
    DEFAULT_SNOWBANK_NAME_MATCHES,
    DEFAULT_SNOWBANK_NAME_MESH_PATTERNS,
    DEFAULT_WATER_NAME_MATCHES,
)
from .profiling import profile_span
from .water_appearance import (
    DEFAULT_WATER_PROFILE_MANIFEST,
    WaterProfileManifest,
    water_profile_manifest_data,
)
from .zip_deflate import ZIP_COMPRESSLEVEL


@dataclass(frozen=True)
class ConversionOptions:
    rbr_root: Path
    stage: str
    filesystem: FileSandbox
    beamng_mods_dir: Path | None = None
    output: Path | None = None
    overwrite: bool = False
    rights_confirmed: bool = False
    remove_source_skybox: bool = False
    use_water_name_fallback: bool = True
    water_name_matches: tuple[str, ...] = DEFAULT_WATER_NAME_MATCHES
    water_profiles: WaterProfileManifest = DEFAULT_WATER_PROFILE_MANIFEST
    use_snowwall_collision_override: bool = True
    use_snowbank_name_fallback: bool = True
    snowbank_name_matches: tuple[str, ...] = DEFAULT_SNOWBANK_NAME_MATCHES
    snowbank_name_mesh_patterns: tuple[str, ...] = DEFAULT_SNOWBANK_NAME_MESH_PATTERNS
    inflate_thin_walls: bool = True
    use_visual_lods: bool = True
    use_map_border_brake_walls: bool = True
    temperature_night: float | None = None
    temperature_day: float | None = None
    latitude: float | None = None
    longitude: float | None = None
    map_altitude_meters: float | None = None
    environment_date: str | None = None
    preview_radius_m: float | None = None
    original_variants: tuple[str, ...] = ()
    environment: str = "N"
    use_foliage_name_fallback: bool = True
    foliage_name_matches: tuple[str, ...] = DEFAULT_FOLIAGE_NAME_MATCHES
    use_foliage_ground_types: bool = True
    foliage_ground_types: tuple[str, ...] = DEFAULT_FOLIAGE_GROUND_TYPES


def resolved_map_altitude_meters(
    options: ConversionOptions,
    location: object | None,
) -> float:
    altitude = (
        options.map_altitude_meters
        if options.map_altitude_meters is not None
        else getattr(location, "altitude_meters", 0.0)
    )
    altitude = float(altitude)
    if not math.isfinite(altitude):
        raise ConversionError("Map altitude must be a finite number of meters")
    return altitude


_OPTION_STAT_KEYS = (
    "sourceFormat",
    "sourceSkyboxRemovalEnabled",
    "sourceSkyboxPartsRemoved",
    "waterObjects",
    "snowwallCollisionOverrideEnabled",
    "snowwallCollisionOverrideParts",
    "snowwallCollisionOverrideFaces",
    "thinWallInflationEnabled",
    "thinWallComponentsDetected",
    "thinWallComponentsRepaired",
    "thinWallComponentsRestored",
    "thinWallFacesGenerated",
    "visualLodsEnabled",
    "visualLodGroups",
    "mapBorderBrakeWallsSupported",
    "mapBorderBrakeWallsEnabled",
    "brakeWallSegments",
    "brakeWallFaces",
)


def option_stats_for_summary(
    stats: Mapping[str, object],
) -> dict[str, bool | int | str]:
    return {
        key: value
        for key in _OPTION_STAT_KEYS
        if isinstance(
            value := stats.get(key),
            (bool, int, str),
        )
    }


def conversion_parameters(options: ConversionOptions) -> dict[str, object]:
    return {
        "rbrDir": str(options.rbr_root),
        "stage": options.stage,
        "beamngModsDir": (
            str(options.beamng_mods_dir)
            if options.beamng_mods_dir is not None
            else None
        ),
        "output": str(options.output) if options.output is not None else None,
        "overwrite": options.overwrite,
        "rightsConfirmed": options.rights_confirmed,
        "removeSourceSkybox": options.remove_source_skybox,
        "useWaterNameFallback": options.use_water_name_fallback,
        "waterNameMatches": list(options.water_name_matches),
        "waterProfilePolicy": water_profile_manifest_data(options.water_profiles),
        "useSnowwallCollisionOverride": options.use_snowwall_collision_override,
        "useSnowbankNameFallback": options.use_snowbank_name_fallback,
        "snowbankNameMatches": list(options.snowbank_name_matches),
        "snowbankNameMeshPatterns": list(options.snowbank_name_mesh_patterns),
        "inflateThinWalls": options.inflate_thin_walls,
        "useVisualLods": options.use_visual_lods,
        "useMapBorderBrakeWalls": options.use_map_border_brake_walls,
        "mapBorderBrakeWallTargetProfile": "localCollisionBand",
        "mapBorderBrakeWallSourceVerticalSemantics": "unverified",
        "mapBorderBrakeWallMaxTriangleDiagonalMeters": 5.0,
        "mapBorderBrakeWallGroundDepthMeters": 2.0,
        "mapBorderBrakeWallGroundClearanceMeters": 10.0,
        "temperatureNightC": options.temperature_night,
        "temperatureDayC": options.temperature_day,
        "latitude": options.latitude,
        "longitude": options.longitude,
        "mapAltitudeMeters": options.map_altitude_meters,
        "environmentDate": options.environment_date,
        "previewRadiusMeters": options.preview_radius_m,
        "originalVariants": list(options.original_variants),
        "environment": options.environment,
        "useFoliageNameFallback": options.use_foliage_name_fallback,
        "foliageNameMatches": list(options.foliage_name_matches),
        "useFoliageGroundTypes": options.use_foliage_ground_types,
        "foliageGroundTypes": list(options.foliage_ground_types),
    }


def output_path_for(options: ConversionOptions, level_id: str) -> Path:
    if options.output:
        return options.filesystem.write_path(options.output)
    if options.beamng_mods_dir is None:
        raise ConversionError(
            "A BeamNG mods folder or explicit output path is required"
        )
    mods_dir = options.filesystem.write_path(options.beamng_mods_dir)
    return options.filesystem.write_path(mods_dir / f"{level_id}.zip")


def temporary_workspace(level_id: str, filesystem: FileSandbox):
    return filesystem.temporary_directory(
        temporary_root(),
        prefix=f"{level_id}-",
    )


def report_warnings(
    reporter: ProgressReporter,
    warnings: list[str],
) -> None:
    for warning in warnings:
        reporter.warning("warning", warning)


def pacenote_log_writers(
    destination: Path,
    filesystem: FileSandbox,
) -> tuple[Callable[[str], None], Callable[[str], None]]:
    gui_log = os.environ.get("RBR2BEAMNG_GUI_LOG_ACTIVE") == "1"
    try:
        log_dir = filesystem.write_path(settings_path().parent / "logs")
        detail_path = filesystem.write_path(log_dir / f"{destination.stem}.log")
        notes_path = filesystem.write_path(log_dir / f"{destination.stem}.notes")
        filesystem.mkdir(log_dir, parents=True, exist_ok=True)
        if not gui_log:
            filesystem.write_text(detail_path, "", encoding="utf-8")
        filesystem.write_text(notes_path, "", encoding="utf-8")
    except (OSError, SandboxViolationError):
        return (lambda _line: None, lambda _line: None)

    def append(path: Path, line: str) -> None:
        try:
            with filesystem.open(path, "a", encoding="utf-8") as stream:
                stream.write(line.rstrip("\r\n") + "\n")
        except (OSError, SandboxViolationError):
            pass

    # The GUI owns the stage log while it runs the conversion; two processes
    # appending to one file on Windows overwrite each other's lines.
    def send_to_gui(line: str) -> None:
        print(
            json.dumps({"log": line.rstrip("\r\n")}, ensure_ascii=False),
            flush=True,
        )

    return (
        send_to_gui if gui_log else lambda line: append(detail_path, line),
        lambda line: append(notes_path, line),
    )


def pacenote_visualizer_writer(
    destination: Path,
    filesystem: FileSandbox,
    log_stem: str | None = None,
) -> Callable[[str], None]:
    try:
        log_dir = filesystem.write_path(settings_path().parent / "logs")
        visualizer_path = filesystem.write_path(
            log_dir / f"{log_stem or destination.stem}.html"
        )
        filesystem.mkdir(log_dir, parents=True, exist_ok=True)
    except (OSError, SandboxViolationError):
        return lambda _html: None

    def write(html: str) -> None:
        try:
            filesystem.write_text(visualizer_path, html, encoding="utf-8")
        except (OSError, SandboxViolationError):
            pass

    return write


def _invalid_zip_entry(
    archive: zipfile.ZipFile,
    expected_sizes: Mapping[str, int],
) -> str | None:
    """First entry whose local header is unreadable or whose size differs
    from ``expected_sizes``, found without decompressing anything."""
    sizes: dict[str, int] = {}
    for info in archive.infolist():
        try:
            archive.open(info).close()
        except zipfile.BadZipFile:
            return info.filename
        sizes[info.filename] = info.file_size
    return next(
        (name for name, size in expected_sizes.items() if sizes.get(name) != size),
        None,
    )


def install_zip(
    payload: Path,
    destination: Path,
    overwrite: bool,
    filesystem: FileSandbox,
    prepared_zip: Path | None = None,
) -> None:
    payload = filesystem.read_path(payload)
    destination = filesystem.write_path(destination)
    filesystem.mkdir(destination.parent, parents=True, exist_ok=True)
    if filesystem.exists(destination) and not overwrite:
        raise ConversionError(
            f"Output already exists: {destination}. Use --overwrite to replace it."
        )
    temporary_zip = filesystem.write_path(
        prepared_zip
        or payload.parent / f"{destination.stem}-{uuid.uuid4().hex}.zip"
    )
    install_temporary: Path | None = None
    try:
        with profile_span(
            "write_zip_payload",
            category="package",
            preparedZip=prepared_zip is not None,
        ) as zip_span:
            files = sorted(
                path
                for path in filesystem.rglob(payload, "*")
                if filesystem.is_file(path)
            )
            with zipfile.ZipFile(
                temporary_zip,
                "a" if prepared_zip else "w",
                compression=zipfile.ZIP_DEFLATED,
                compresslevel=ZIP_COMPRESSLEVEL,
                allowZip64=True,
            ) as archive:
                for path in files:
                    archive.write(
                        filesystem.read_path(path),
                        path.relative_to(payload).as_posix(),
                    )
            payload_sizes = {
                path.relative_to(payload).as_posix(): filesystem.stat(path).st_size
                for path in files
            }
            zip_span.update(
                files=len(files),
                sourceBytes=sum(payload_sizes.values()),
                zipBytes=filesystem.stat(temporary_zip).st_size,
            )
        with profile_span(
            "validate_zip",
            category="package",
            zipBytes=filesystem.stat(temporary_zip).st_size,
        ):
            with zipfile.ZipFile(filesystem.read_path(temporary_zip), "r") as archive:
                bad = _invalid_zip_entry(archive, payload_sizes)
                if bad:
                    raise ConversionError(
                        f"Generated ZIP failed validation at {bad}"
                    )
                if not any(
                    name.partition("/")[0] == "levels"
                    for name in archive.namelist()
                ):
                    raise ConversionError(
                        "Generated ZIP does not contain levels/ at its root"
                    )
        with profile_span(
            "move_zip",
            category="package",
            destination=str(destination),
        ):
            try:
                filesystem.replace(temporary_zip, destination)
            except OSError as exc:
                if (
                    exc.errno != errno.EXDEV
                    and getattr(exc, "winerror", None) != 17
                ):
                    raise
                install_temporary = filesystem.write_path(
                    destination.with_name(
                        f".{destination.name}.install-{uuid.uuid4().hex}.tmp"
                    )
                )
                filesystem.copy2(temporary_zip, install_temporary)
                with zipfile.ZipFile(
                    filesystem.read_path(install_temporary),
                    "r",
                ) as archive:
                    bad = _invalid_zip_entry(archive, payload_sizes)
                    if bad:
                        raise ConversionError(
                            f"Installed ZIP copy failed validation at {bad}"
                        )
                filesystem.replace(install_temporary, destination)
    finally:
        filesystem.unlink(temporary_zip, missing_ok=True)
        if install_temporary:
            filesystem.unlink(install_temporary, missing_ok=True)


def install_companion_mod(
    destination: Path,
    files: Mapping[str, str],
    filesystem: FileSandbox,
) -> None:
    """Build and atomically replace a generated companion mod archive."""
    destination = filesystem.write_path(destination)
    filesystem.mkdir(destination.parent, parents=True, exist_ok=True)
    temporary = filesystem.write_path(
        destination.with_name(f".{destination.name}-{uuid.uuid4().hex}.tmp")
    )
    try:
        with zipfile.ZipFile(
            temporary,
            "w",
            compression=zipfile.ZIP_DEFLATED,
            compresslevel=ZIP_COMPRESSLEVEL,
        ) as archive:
            for name, content in files.items():
                archive.writestr(name, content)
        with zipfile.ZipFile(filesystem.read_path(temporary)) as archive:
            invalid_entry = archive.testzip()
            if invalid_entry:
                raise RuntimeError(
                    f"{destination.name} failed CRC validation at {invalid_entry}"
                )
        filesystem.replace(temporary, destination)
    finally:
        filesystem.unlink(temporary, missing_ok=True)
