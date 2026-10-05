from __future__ import annotations

import configparser
import enum
import re
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path, PureWindowsPath
from typing import Mapping
from warnings import warn

from ..filesystem import current_filesystem
from ..ini import config_parser


class CatalogError(ValueError):
    pass


class MissingOriginalAssetError(CatalogError):
    pass


class AmbiguousOriginalAssetError(CatalogError):
    pass


class Tint(str, enum.Enum):
    MORNING = "M"
    NOON = "N"
    EVENING = "E"
    OVERCAST = "O"
    SERVICE = "S"


TINT_ORDER = tuple(Tint)
CORE_STAGE_EXTENSIONS = (
    "track",
    "physics",
    "driveline",
    "collision",
    "material",
)


def _as_tint(value: Tint | str) -> Tint:
    if isinstance(value, Tint):
        return value
    return Tint(value.upper())


@dataclass(frozen=True)
class OriginalCatalogEntry:
    stage_id: int
    section: str
    track_name: str
    track_base: Path
    stage_name: str
    properties: Mapping[str, str]


@dataclass(frozen=True)
class OriginalStageFiles:
    entry: OriginalCatalogEntry
    tint: Tint
    files: Mapping[str, Path]


@dataclass(frozen=True)
class OriginalCatalog:
    path: Path
    install_root: Path
    maps_root: Path
    entries: Mapping[int, OriginalCatalogEntry]
    extensions: Mapping[str, str]
    general: Mapping[str, str] = field(default_factory=dict)
    file_index: dict[str, tuple[Path, ...]] = field(default_factory=dict)
    indexed_directories: set[str] = field(
        default_factory=set,
        repr=False,
        compare=False,
    )

    def entry(self, stage_id: int) -> OriginalCatalogEntry:
        try:
            return self.entries[stage_id]
        except KeyError as exc:
            raise CatalogError(f"Original stage ID {stage_id} is not in {self.path}") from exc

    def resolve_file(
        self,
        stage: int | OriginalCatalogEntry,
        tint: Tint | str,
        extension: str,
    ) -> Path:
        entry = self.entry(stage) if isinstance(stage, int) else stage
        tint_value = _as_tint(tint)
        requested_extension = extension.casefold().lstrip(".")
        ext = self.extensions.get(requested_extension, requested_extension)
        if requested_extension in {"textures", "texture_archive", "textures.rbz", "rbz"}:
            ext = "textures.rbz"
        elif requested_extension in {"texture_ini", "ini"}:
            ext = "ini"
        path = self._match(_candidate_for(entry.track_base, tint_value, ext))
        if path is None and requested_extension == "track":
            path = self._match(_candidate_for(entry.track_base, tint_value, _IBS_EXTENSION))
        if path is None:
            raise MissingOriginalAssetError(
                f"No {tint_value.value}.{ext} asset for stage {entry.stage_id}"
            )
        return path

    def resolve_stage(
        self,
        stage: int | OriginalCatalogEntry,
        tint: Tint | str,
        *,
        extensions: tuple[str, ...] | None = None,
    ) -> OriginalStageFiles:
        entry = self.entry(stage) if isinstance(stage, int) else stage
        tint_value = _as_tint(tint)
        requested = extensions or CORE_STAGE_EXTENSIONS
        files = {name: self.resolve_file(entry, tint_value, name) for name in requested}
        return OriginalStageFiles(entry, tint_value, files)

    def complete_tints(
        self,
        stage: int | OriginalCatalogEntry,
        *,
        extensions: tuple[str, ...] | None = None,
    ) -> tuple[Tint, ...]:
        result: list[Tint] = []
        for tint in self.available_tints(stage):
            try:
                self.resolve_stage(stage, tint, extensions=extensions)
            except (AmbiguousOriginalAssetError, MissingOriginalAssetError):
                continue
            result.append(tint)
        return tuple(result)

    def available_tints(
        self,
        stage: int | OriginalCatalogEntry,
    ) -> tuple[Tint, ...]:
        entry = self.entry(stage) if isinstance(stage, int) else stage
        extensions = {*self.extensions.values(), _IBS_EXTENSION}
        return tuple(
            tint
            for tint in TINT_ORDER
            if any(
                self._match(_candidate_for(entry.track_base, tint, extension)) is not None
                for extension in extensions
            )
        )

    def _index_directory(self, directory: Path) -> None:
        filesystem = current_filesystem()
        directory = filesystem.read_path(directory)
        directory_key = _resolved_path_key(directory)
        if directory_key in self.indexed_directories:
            return
        self.indexed_directories.add(directory_key)
        try:
            paths = filesystem.iterdir(directory)
        except OSError:
            return
        indexed: dict[str, list[Path]] = {}
        for path in paths:
            path = filesystem.read_path(path)
            if path.is_file():
                indexed.setdefault(_resolved_path_key(path), []).append(path)
        for key, paths in indexed.items():
            existing = self.file_index.get(key, ())
            self.file_index[key] = tuple(
                sorted((*existing, *paths), key=lambda path: path.name)
            )

    def _match(self, candidate: Path) -> Path | None:
        key = _path_key(candidate)
        matches = self.file_index.get(key)
        if matches is None:
            self._index_directory(candidate.parent)
            matches = self.file_index.get(key, ())
        if len(matches) > 1:
            raise AmbiguousOriginalAssetError(
                f"Multiple case-insensitive matches for {candidate}: "
                + ", ".join(path.name for path in matches)
            )
        if matches and matches[0].suffix.casefold() == ".ini":
            return matches[0] if _may_be_texture_ini(matches[0]) else None
        return matches[0] if matches else None


@lru_cache(maxsize=4096)
def _may_be_texture_ini(path: Path) -> bool:
    raw = current_filesystem().read_bytes(path)
    if b"[textureinfo]" in raw.lower():
        return True
    parser = config_parser()
    try:
        parser.read_string(raw.decode("latin-1"))
    except configparser.Error:
        return True
    sections = parser.sections()
    return not sections or any(name.casefold() == "textureinfo" for name in sections)


_MAP_SECTION = re.compile(r"map0*(\d+)\Z", re.IGNORECASE)
# older installs ship some track files as .Ibs: an LBS with every byte increased by 0x19
_IBS_EXTENSION = "ibs"
_TRACK_FILE = re.compile(r"track-(\d+)_[mneos]\.([^.]+)\Z", re.IGNORECASE)
_FOLDER_ID_PREFIX = re.compile(r"\A\d+\s*-\s*")
_DEFAULT_EXTENSIONS = {
    "track": "lbs",
    "physics": "trk",
    "driveline": "dls",
    "collision": "col",
    "material": "mat",
    "fence": "fnc",
    "texture_ini": "ini",
}
_GENERAL_KEYS = {
    "track": "TrackExt",
    "physics": "PhysicsExt",
    "driveline": "DriveLineExt",
    "collision": "CollisionExt",
    "material": "MaterialExt",
    "fence": "FenceDataExt",
}


def _unquote(value: str) -> str:
    stripped = value.strip()
    if len(stripped) >= 2 and stripped[0] == stripped[-1] and stripped[0] in "'\"":
        return stripped[1:-1]
    return stripped


def _case_value(values: Mapping[str, str], name: str, default: str = "") -> str:
    wanted = name.casefold()
    for key, value in values.items():
        if key.casefold() == wanted:
            return _unquote(value)
    return default


def _safe_track_base(track_name: str, install_root: Path) -> Path:
    win_path = PureWindowsPath(track_name.replace("/", "\\"))
    if win_path.is_absolute() or win_path.drive or ".." in win_path.parts:
        raise CatalogError(f"Unsafe TrackName path: {track_name!r}")
    parts = tuple(part for part in win_path.parts if part not in ("", "."))
    if not parts:
        raise CatalogError("TrackName is empty")
    filesystem = current_filesystem()
    root = filesystem.read_path(install_root)
    candidate = filesystem.read_path(root.joinpath(*parts))
    resolved_parent = candidate.parent
    try:
        resolved_parent.relative_to(root)
    except ValueError as exc:
        raise CatalogError(f"TrackName escapes the install root: {track_name!r}") from exc
    return candidate


def _candidate_for(track_base: Path, tint: Tint, extension: str) -> Path:
    if extension == "textures.rbz":
        return track_base.with_name(f"{track_base.name}_{tint.value}_textures.rbz")
    return track_base.with_name(f"{track_base.name}_{tint.value}.{extension}")


def _resolved_path_key(path: Path) -> str:
    return str(path).replace("\\", "/").casefold()


def _path_key(path: Path) -> str:
    return _resolved_path_key(current_filesystem().read_path(path))


def _folder_properties(folder: Path, names: list[str], stage_id: int) -> dict[str, str]:
    """The [Map<id>] section of the Tracks<id>.ini that RSF installs in a stage folder."""
    name = next((name for name in names if name.casefold() == f"tracks{stage_id}.ini"), None)
    if name is None:
        return {}
    parser = config_parser()
    try:
        parser.read_string(current_filesystem().read_text(folder / name, encoding="latin-1"))
    except (OSError, configparser.Error) as exc:
        warn(f"Ignoring {folder / name}: {exc}")
        return {}
    for section_name in parser.sections():
        match = _MAP_SECTION.fullmatch(section_name)
        if match and int(match.group(1)) == stage_id:
            return dict(parser[section_name])
    return {}


def _installed_entries(
    maps_root: Path,
    track_extension: str,
    known_ids: Mapping[int, OriginalCatalogEntry],
) -> dict[int, OriginalCatalogEntry]:
    """Stages installed in Maps/ or Maps/<folder>/ that Tracks.ini does not list."""
    filesystem = current_filesystem()
    try:
        folders = [maps_root, *sorted(path for path in filesystem.iterdir(maps_root) if filesystem.is_dir(path))]
    except OSError:
        return {}
    entries: dict[int, OriginalCatalogEntry] = {}
    for folder in folders:
        try:
            names = sorted(path.name for path in filesystem.iterdir(folder))
        except OSError:
            continue
        for name in names:
            match = _TRACK_FILE.fullmatch(name)
            if not match or match.group(2).casefold() not in (track_extension, _IBS_EXTENSION):
                continue
            stage_id = int(match.group(1))
            if stage_id in known_ids or stage_id in entries:
                continue
            track_base = folder / f"track-{match.group(1)}"
            folder_name = "" if folder == maps_root else _FOLDER_ID_PREFIX.sub("", folder.name)
            properties = {} if folder == maps_root else _folder_properties(folder, names, stage_id)
            entries[stage_id] = OriginalCatalogEntry(
                stage_id=stage_id,
                section="",
                track_name=str(PureWindowsPath(maps_root.name, track_base.relative_to(maps_root))),
                track_base=track_base,
                stage_name=_case_value(properties, "StageName", folder_name or f"Map {stage_id}"),
                properties=properties,
            )
    return entries


def parse_catalog_text(
    text: str,
    *,
    path: Path,
    install_root: Path,
) -> OriginalCatalog:
    filesystem = current_filesystem()
    path = filesystem.read_path(path)
    install_root = filesystem.read_path(install_root)
    parser = config_parser()
    try:
        parser.read_string(text)
    except configparser.Error as exc:
        warn(f"Ignoring unreadable parts of {path}: {exc}")

    general_name = next(
        (name for name in parser.sections() if name.casefold() == "general"),
        None,
    )
    general = dict(parser[general_name]) if general_name else {}
    extensions = {
        name: _case_value(general, key).lstrip(".").casefold() or _DEFAULT_EXTENSIONS[name]
        for name, key in _GENERAL_KEYS.items()
    }
    extensions["texture_ini"] = "ini"

    entries: dict[int, OriginalCatalogEntry] = {}
    for section_name in parser.sections():
        match = _MAP_SECTION.fullmatch(section_name)
        if not match:
            continue
        stage_id = int(match.group(1))
        properties = dict(parser[section_name])
        track_name = _case_value(properties, "TrackName")
        try:
            if stage_id in entries:
                raise CatalogError(f"stage ID {stage_id} is already defined by [{entries[stage_id].section}]")
            if not track_name:
                raise CatalogError("it has no TrackName")
            track_base = _safe_track_base(track_name, install_root)
        except CatalogError as exc:
            warn(f"Ignoring [{section_name}] in {path}: {exc}")
            continue
        entries[stage_id] = OriginalCatalogEntry(
            stage_id=stage_id,
            section=section_name,
            track_name=track_name,
            track_base=track_base,
            stage_name=_case_value(properties, "StageName", f"Map {stage_id}"),
            properties=properties,
        )
    entries.update(_installed_entries(path.parent, extensions["track"], entries))

    return OriginalCatalog(
        path=path,
        install_root=install_root,
        maps_root=path.parent,
        entries=entries,
        extensions=extensions,
        general=general,
    )


def parse_tracks_ini(path: str | Path, *, install_root: str | Path | None = None) -> OriginalCatalog:
    filesystem = current_filesystem()
    source = filesystem.read_path(path)
    root = filesystem.read_path(
        install_root if install_root is not None else source.parent.parent
    )
    try:
        text = filesystem.read_text(source, encoding="latin-1")
    except OSError as exc:
        raise CatalogError(f"Unable to read {source}: {exc}") from exc
    return parse_catalog_text(text, path=source, install_root=root)
