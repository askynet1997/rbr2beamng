from __future__ import annotations

import json
import os
import re
import sys
import time
import unicodedata
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, TextIO

from .filesystem import FileSandbox, SandboxViolationError
from .profiling import record_progress
from .rsf_files import RsfDeobfuscatingSandbox


LEVEL_NAMESPACE = uuid.UUID("0e1adf09-24d8-4d47-a322-6d2d57ea4f4e")
RIGHTS_CONFIRMATION_TEXT = (
    "I confirm that I am legally permitted to convert this stage."
)
RBR_FOLDER_NAMES = {"rbr", "richard burns rally"}
SETTINGS_RBR_PATH = "rbrInstallFolder"
SETTINGS_BEAMNG_MODS_PATH = "beamngModsFolder"
SETTINGS_SNOWBANK_NAME_FALLBACK = "useSnowbankNameFallback"
SETTINGS_SNOWBANK_NAME_MATCHES = "snowbankNameMatches"
SETTINGS_SNOWBANK_NAME_MESH_PATTERNS = "snowbankNameMeshPatterns"
SETTINGS_FOLIAGE_NAME_FALLBACK = "useFoliageNameFallback"
SETTINGS_FOLIAGE_NAME_MATCHES = "foliageNameMatches"
SETTINGS_FOLIAGE_GROUND_TYPES_ENABLED = "useFoliageGroundTypes"
SETTINGS_FOLIAGE_GROUND_TYPES = "foliageGroundTypes"
SETTINGS_WATER_NAME_FALLBACK = "useWaterNameFallback"
SETTINGS_WATER_NAME_MATCHES = "waterNameMatches"


class ConversionError(RuntimeError):
    pass


class ConversionCancelled(ConversionError):
    pass


def format_duration(seconds: float) -> str:
    total = max(0, int(seconds))
    hours, remainder = divmod(total, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}:{minutes:02d}:{seconds:02d}"
    return f"{minutes:02d}:{seconds:02d}"


def format_file_size(size_bytes: int) -> str:
    value = float(max(0, size_bytes))
    for unit in ("B", "KiB", "MiB", "GiB"):
        if value < 1024.0 or unit == "GiB":
            return f"{int(value)} B" if unit == "B" else f"{value:.1f} {unit}"
        value /= 1024.0


def slugify(value: str) -> str:
    normalized = unicodedata.normalize("NFKD", value)
    ascii_value = normalized.encode("ascii", "ignore").decode("ascii").lower()
    slug = re.sub(r"[^a-z0-9]+", "_", ascii_value).strip("_")
    return slug or "stage"


def stage_zip_name(
    folder_name: str,
    *,
    display_name: str = "",
    source_format: str = "rx",
) -> str:
    if source_format == "original":
        readable = slugify(display_name or folder_name)[:100]
        return f"rbr_{readable}.zip"
    return f"rbr_{slugify(folder_name)}.zip"


def stable_uuid(*parts: object) -> str:
    key = "/".join(str(part) for part in parts)
    return str(uuid.uuid5(LEVEL_NAMESPACE, key))


def settings_path() -> Path:
    override = os.environ.get("RBR2BEAMNG_CONFIG_DIR")
    if override:
        root = Path(override)
    elif getattr(sys, "frozen", False):
        root = Path(sys.executable).resolve().parent
    else:
        root = Path(__file__).resolve().parents[1]
    return root / "settings.json"


def temporary_root() -> Path:
    return settings_path().parent / "temp"


def create_runtime_filesystem(
    *,
    rbr_root: Path | None = None,
    beamng_mods_dir: Path | None = None,
    output: Path | None = None,
    include_runtime_write_roots: bool = True,
) -> FileSandbox:
    filesystem = RsfDeobfuscatingSandbox()
    filesystem.add_read_only(Path(__file__).resolve().parent)
    executable = Path(sys.executable).resolve()
    filesystem.add_read_only(executable)
    if getattr(sys, "frozen", False):
        filesystem.add_read_only(
            executable.with_name("rbr2beamng-cli.exe")
        )
    if include_runtime_write_roots:
        filesystem.add_read_write(settings_path().parent)
    else:
        filesystem.add_read_only(settings_path().parent)
    if rbr_root is not None:
        filesystem.add_read_only(rbr_root)
    if beamng_mods_dir is not None:
        filesystem.add_read_write(beamng_mods_dir)
    if output is not None:
        filesystem.add_read_write(output.parent)
    if sys.platform != "win32":
        filesystem.add_read_only(Path("/proc/self"))
    return filesystem


def load_settings(filesystem: FileSandbox | None = None) -> dict[str, str]:
    try:
        path = settings_path()
        text = (
            filesystem.read_text(path, encoding="utf-8")
            if filesystem is not None
            else path.read_text(encoding="utf-8")
        )
        return json.loads(text)
    except SandboxViolationError:
        raise
    except OSError:
        return {}


def save_settings(values: dict[str, str | None], filesystem: FileSandbox) -> None:
    path = filesystem.write_path(settings_path())
    settings = load_settings(filesystem)
    for key, value in values.items():
        if value is None:
            settings.pop(key, None)
        else:
            settings[key] = value
    temporary = path.with_suffix(".tmp")
    try:
        filesystem.mkdir(path.parent, parents=True, exist_ok=True)
        filesystem.write_text(
            temporary,
            json.dumps(settings, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        filesystem.replace(temporary, path)
    except SandboxViolationError:
        raise
    except OSError:
        filesystem.unlink(temporary, missing_ok=True)


def _has_rbr_executable(
    path: Path,
    filesystem: FileSandbox | None = None,
) -> bool:
    try:
        return any(
            (
                filesystem.is_file(entry)
                if filesystem is not None
                else entry.is_file()
            )
            and entry.name.casefold().startswith("richardburnsrally")
            and entry.name.casefold().endswith(".exe")
            for entry in (
                filesystem.iterdir(path)
                if filesystem is not None
                else path.iterdir()
            )
        )
    except OSError:
        return False


def is_rbr_install(path: Path, filesystem: FileSandbox | None = None) -> bool:
    is_dir = (
        filesystem.is_dir
        if filesystem is not None
        else lambda candidate: candidate.is_dir()
    )
    is_file = (
        filesystem.is_file
        if filesystem is not None
        else lambda candidate: candidate.is_file()
    )
    return (
        is_dir(path)
        and (
            is_dir(path / "RX_CONTENT" / "TRACKS")
            or is_file(path / "Maps" / "Tracks.ini")
        )
        and _has_rbr_executable(path, filesystem)
    )


def _child_directories(path: Path) -> list[Path]:
    try:
        return sorted(
            (entry for entry in path.iterdir() if entry.is_dir()),
            key=lambda entry: entry.name.casefold(),
        )
    except OSError:
        return []


def _rbr_named_children(path: Path) -> list[Path]:
    # Compares names only: a stat per child makes folders with many thousands of entries slow.
    try:
        with os.scandir(path) as entries:
            return sorted(
                (
                    Path(entry.path)
                    for entry in entries
                    if entry.name.casefold() in RBR_FOLDER_NAMES
                ),
                key=lambda entry: entry.name.casefold(),
            )
    except OSError:
        return []


def _search_rbr_roots(roots: list[Path]) -> Path | None:
    for root in roots:
        first_level = _child_directories(root)
        for candidate in first_level:
            if candidate.name.casefold() in RBR_FOLDER_NAMES and is_rbr_install(candidate):
                return candidate.resolve()
        for parent in first_level:
            for candidate in _rbr_named_children(parent):
                if is_rbr_install(candidate):
                    return candidate.resolve()
    return None


def _windows_drives() -> list[Path]:
    if sys.platform != "win32":
        return []
    listdrives = getattr(os, "listdrives", None)
    try:
        drives = (
            listdrives()
            if listdrives is not None
            else [f"{chr(ord('A') + index)}:/" for index in range(26)]
        )
    except OSError:
        drives = [f"{chr(ord('A') + index)}:/" for index in range(26)]
    result = []
    for drive in drives:
        root = Path(drive)
        try:
            if root.is_dir():
                result.append(root)
        except OSError:
            continue
    return result


def _first_valid_path(
    candidates: list[Path],
    predicate: Callable[[Path], bool],
) -> Path | None:
    seen: set[str] = set()
    for candidate in candidates:
        try:
            resolved = candidate.expanduser().resolve()
        except OSError:
            continue
        key = os.path.normcase(str(resolved))
        if key not in seen and predicate(resolved):
            return resolved
        seen.add(key)
    return None


def find_rbr_install() -> Path | None:
    candidates: list[Path] = []
    stored = load_settings().get(SETTINGS_RBR_PATH)
    if stored:
        candidates.append(Path(stored))
    configured = os.environ.get("RBR_HOME")
    if configured:
        candidates.append(Path(configured))

    if sys.platform != "win32":
        home = Path.home()
        candidates.extend(
            (
                home / "RBR",
                home / "Games" / "RBR",
                home / ".steam" / "steam" / "steamapps" / "compatdata",
            )
        )

    return (
        _first_valid_path(candidates, is_rbr_install)
        or _search_rbr_roots(_windows_drives())
    )


def is_beamng_mods_dir(
    path: Path,
    filesystem: FileSandbox | None = None,
) -> bool:
    return (
        filesystem.is_dir(path)
        if filesystem is not None
        else path.is_dir()
    )


def _read_ini_value(path: Path, key: str) -> str | None:
    try:
        lines = path.read_text(encoding="utf-8-sig", errors="replace").splitlines()
    except OSError:
        return None
    expected = key.casefold()
    for line in lines:
        name, separator, value = line.partition("=")
        if separator and name.strip().casefold() == expected:
            return value.strip().strip('"').strip()
    return None


def _beamng_mods_dir_from_ini(local_app_data: Path) -> Path | None:
    ini_path = local_app_data / "BeamNG" / "BeamNG.drive.ini"
    value = _read_ini_value(ini_path, "userFolder")
    if not value:
        return None
    user_root = Path(os.path.expandvars(value)).expanduser()
    if not user_root.is_absolute():
        user_root = ini_path.parent / user_root
    return user_root / "current" / "mods"


def find_beamng_mods_dir() -> Path | None:
    candidates: list[Path] = []
    stored = load_settings().get(SETTINGS_BEAMNG_MODS_PATH)
    if stored:
        candidates.append(Path(stored))

    if sys.platform == "win32":
        local_app_data = os.environ.get("LOCALAPPDATA")
        if local_app_data:
            local_app_data_path = Path(local_app_data)
            ini_mods_dir = _beamng_mods_dir_from_ini(local_app_data_path)
            if ini_mods_dir:
                candidates.append(ini_mods_dir)
            candidates.append(
                local_app_data_path / "BeamNG" / "BeamNG.drive" / "current" / "mods"
            )
    elif sys.platform.startswith("linux"):
        candidates.append(
            Path.home() / ".local" / "share" / "BeamNG" / "BeamNG.drive" / "current" / "mods"
        )

    return _first_valid_path(candidates, is_beamng_mods_dir)


@dataclass(frozen=True)
class ProgressEvent:
    phase: str
    message: str
    current: int | None = None
    total: int | None = None
    detail: str | None = None
    severity: str = "info"
    zip_size_bytes: int | None = None
    collision_triangles: int | None = None
    merged_away_triangles: int | None = None
    route_length_meters: float | None = None
    sector_count: int | None = None
    option_stats: dict[str, bool | int | str] | None = None

    def as_dict(self) -> dict[str, object]:
        data: dict[str, object] = {
            "phase": self.phase,
            "message": self.message,
            "severity": self.severity,
        }
        if self.current is not None:
            data["current"] = self.current
        if self.total is not None:
            data["total"] = self.total
        if self.detail:
            data["detail"] = self.detail
        if self.zip_size_bytes is not None:
            data["zipSizeBytes"] = self.zip_size_bytes
        if self.collision_triangles is not None:
            data["collisionTriangles"] = self.collision_triangles
        if self.merged_away_triangles is not None:
            data["mergedAwayTriangles"] = self.merged_away_triangles
        if self.route_length_meters is not None:
            data["routeLengthMeters"] = self.route_length_meters
        if self.sector_count is not None:
            data["sectorCount"] = self.sector_count
        if self.option_stats is not None:
            data["optionStats"] = self.option_stats
        return data


class ProgressReporter:
    def __init__(
        self,
        *,
        json_lines: bool = False,
        stream: TextIO | None = None,
        callback: Callable[[ProgressEvent], None] | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> None:
        self.json_lines = json_lines
        self.stream = stream or sys.stdout
        self.callback = callback
        self.cancelled = cancelled
        self.started_at = time.monotonic()

    def emit(
        self,
        phase: str,
        message: str,
        *,
        current: int | None = None,
        total: int | None = None,
        detail: str | None = None,
        severity: str = "info",
        zip_size_bytes: int | None = None,
        collision_triangles: int | None = None,
        merged_away_triangles: int | None = None,
        route_length_meters: float | None = None,
        sector_count: int | None = None,
        option_stats: dict[str, bool | int | str] | None = None,
    ) -> None:
        if self.cancelled and self.cancelled():
            raise ConversionCancelled("Conversion cancelled")

        event = ProgressEvent(
            phase,
            message,
            current,
            total,
            detail,
            severity,
            zip_size_bytes,
            collision_triangles,
            merged_away_triangles,
            route_length_meters,
            sector_count,
            option_stats,
        )
        record_progress(
            phase,
            message,
            current=current,
            total=total,
            detail=detail,
            severity=severity,
            zipSizeBytes=zip_size_bytes,
            collisionTriangles=collision_triangles,
            mergedAwayTriangles=merged_away_triangles,
            routeLengthMeters=route_length_meters,
            sectorCount=sector_count,
            optionStats=option_stats,
        )
        if self.callback:
            self.callback(event)

        elapsed = time.monotonic() - self.started_at
        if self.json_lines:
            payload = event.as_dict()
            payload["elapsed"] = round(elapsed, 3)
            print(json.dumps(payload, ensure_ascii=False, separators=(",", ":")), file=self.stream, flush=True)
            return

        progress = ""
        if current is not None and total:
            progress = f" [{current}/{total}]"
        suffix = f": {detail}" if detail else ""
        size = (
            f"; ZIP {format_file_size(zip_size_bytes)}"
            if zip_size_bytes is not None
            else ""
        )
        coltris = (
            f"; {collision_triangles:,} coltris"
            if collision_triangles is not None
            else ""
        )
        print(
            f"{phase}{progress} [{format_duration(elapsed)}]: "
            f"{message}{suffix}{size}{coltris}",
            file=self.stream,
            flush=True,
        )

    def elapsed_seconds(self) -> float:
        return time.monotonic() - self.started_at

    def warning(self, phase: str, message: str, *, detail: str | None = None) -> None:
        self.emit(phase, message, detail=detail, severity="warning")
