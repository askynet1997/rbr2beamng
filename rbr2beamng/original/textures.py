from __future__ import annotations

import configparser
import contextlib
import copy
import re
import stat
import warnings
import zipfile
from dataclasses import dataclass, replace
from pathlib import Path, PurePosixPath
from typing import BinaryIO, Callable, Iterable, Iterator, Mapping

from ..filesystem import current_filesystem
from ..ini import config_parser


class TextureError(ValueError):
    pass


class UnsafeTexturePathError(TextureError):
    pass


class AmbiguousTextureError(TextureError):
    pass


class MissingTextureError(TextureError):
    pass


@dataclass(frozen=True)
class OriginalTexture:
    index: int
    kind: str
    filename: str
    mip_levels: int | None
    dynamic: bool | None
    opacity_map: bool | None
    one_bit_opacity: bool | None
    ground_texture: bool | None
    texture_format: str | None
    properties: Mapping[str, str]


@dataclass(frozen=True)
class OriginalTextureIni:
    textures: tuple[OriginalTexture, ...]
    shadow_textures: tuple[OriginalTexture, ...]
    specular_textures: tuple[OriginalTexture, ...]
    sections: Mapping[str, Mapping[str, str]]

@dataclass(frozen=True)
class ResolvedTexture:
    requested_name: str
    source: str
    location: str
    size: int
    archive: Path | None = None


def _case_value(values: Mapping[str, str], key: str) -> str | None:
    wanted = key.casefold()
    for actual, value in values.items():
        if actual.casefold() == wanted:
            return value.strip()
    return None


def _parse_count(
    values: Mapping[str, str],
    key: str,
    *,
    limit: int,
    default: int | None = None,
) -> int:
    raw = _case_value(values, key)
    if raw is None:
        if default is not None:
            return default
        raise TextureError(f"[TextureInfo] is missing {key}")
    try:
        value = int(raw)
    except ValueError as exc:
        raise TextureError(f"[TextureInfo] {key} is not an integer: {raw!r}") from exc
    if value < 0 or value > limit:
        raise TextureError(f"[TextureInfo] {key} is outside 0..{limit}: {value}")
    return value


def _parse_optional_int(values: Mapping[str, str], key: str) -> int | None:
    raw = _case_value(values, key)
    if raw is None:
        return None
    try:
        return int(raw)
    except ValueError as exc:
        raise TextureError(f"Invalid integer {key}={raw!r}") from exc


def _parse_optional_bool(values: Mapping[str, str], key: str) -> bool | None:
    raw = _case_value(values, key)
    if raw is None:
        return None
    normal = raw.strip().casefold()
    values = {
        "1": True,
        "true": True,
        "yes": True,
        "on": True,
        "0": False,
        "false": False,
        "no": False,
        "off": False,
    }
    if normal in values:
        return values[normal]
    if normal.startswith("t"):
        warnings.warn(
            f"Recovered malformed boolean {key}={raw!r} as true",
            RuntimeWarning,
            stacklevel=2,
        )
        return True
    if normal.startswith("f"):
        warnings.warn(
            f"Recovered malformed boolean {key}={raw!r} as false",
            RuntimeWarning,
            stacklevel=2,
        )
        return False
    warnings.warn(
        f"Unrecognized boolean {key}={raw!r}; using true",
        RuntimeWarning,
        stacklevel=2,
    )
    return True


def parse_texture_ini_text(text: str, *, max_textures: int = 100_000) -> OriginalTextureIni:
    parser = config_parser()
    try:
        parser.read_string(text)
    except configparser.Error as exc:
        raise TextureError(f"Invalid Original RBR texture INI: {exc}") from exc

    info_names = [name for name in parser.sections() if name.casefold() == "textureinfo"]
    if len(info_names) != 1:
        raise TextureError("Original texture INI must contain exactly one [TextureInfo]")
    info = dict(parser[info_names[0]])

    section_index: dict[str, list[str]] = {}
    sections: dict[str, Mapping[str, str]] = {}
    for name in parser.sections():
        if name.casefold() == "textureinfo":
            continue
        section_index.setdefault(name.casefold(), []).append(name)
        sections[name] = dict(parser[name])

    def parse_group(
        kind: str,
        count_key: str,
        item_prefix: str,
        *,
        default_count: int | None = None,
    ) -> tuple[OriginalTexture, ...]:
        count = _parse_count(
            info,
            count_key,
            limit=max_textures,
            default=default_count,
        )
        result: list[OriginalTexture] = []
        for index in range(count):
            item_key = f"{item_prefix}{index}"
            filename = _case_value(info, item_key)
            if filename is None:
                raise TextureError(f"[TextureInfo] is missing {item_key}")
            matching = section_index.get(filename.casefold(), [])
            if filename in matching:
                matching = [filename]
            if len(matching) > 1:
                raise AmbiguousTextureError(
                    f"Multiple sections match texture {filename!r}: {matching}"
                )
            properties = dict(parser[matching[0]]) if matching else {}
            result.append(
                OriginalTexture(
                    index=index,
                    kind=kind,
                    filename=filename,
                    mip_levels=_parse_optional_int(properties, "MipLevels"),
                    dynamic=_parse_optional_bool(properties, "Dynamic"),
                    opacity_map=_parse_optional_bool(properties, "OpacityMap"),
                    one_bit_opacity=_parse_optional_bool(properties, "OneBitOpacity"),
                    ground_texture=_parse_optional_bool(properties, "IsGroundTexture"),
                    texture_format=_case_value(properties, "TextureFormat"),
                    properties=properties,
                )
            )
        return tuple(result)

    return OriginalTextureIni(
        textures=parse_group("diffuse", "NumTextures", "Texture"),
        shadow_textures=parse_group(
            "shadow",
            "NumShadowTextures",
            "ShadowTexture",
            default_count=0,
        ),
        specular_textures=parse_group(
            "specular",
            "NumSpecularTextures",
            "SpecularTexture",
            default_count=0,
        ),
        sections=sections,
    )


def parse_texture_ini(path: str | Path, *, max_textures: int = 100_000) -> OriginalTextureIni:
    source = current_filesystem().read_path(path)
    try:
        text = current_filesystem().read_text(source, encoding="latin-1")
    except OSError as exc:
        raise TextureError(f"Unable to read {source}: {exc}") from exc
    return parse_texture_ini_text(text, max_textures=max_textures)


TEXTURE_PAYLOAD_STEM = re.compile(r"track-(\d+)_([MNEOS])_textures", re.IGNORECASE)


def parse_texture_remap_text(text: str) -> dict[str, str]:
    remap: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip() or line.lstrip().startswith(";"):
            continue
        source, separator, target = line.partition("\t")
        if not separator:
            raise TextureError(f"Texture remap line has no TAB separator: {line!r}")
        source = _normalise_member(source.strip())
        target = _normalise_member(target.strip())
        if not TEXTURE_PAYLOAD_STEM.fullmatch(target.partition("/")[0]):
            target = f"{source.partition('/')[0]}/{target}"
        remap[source.casefold()] = target
    return remap


def parse_texture_remap(path: str | Path) -> dict[str, str]:
    source = current_filesystem().read_path(path)
    try:
        text = current_filesystem().read_text(source, encoding="latin-1")
    except OSError as exc:
        raise TextureError(f"Unable to read {source}: {exc}") from exc
    return parse_texture_remap_text(text)


def _normalise_member(name: str) -> str:
    if "\0" in name:
        raise UnsafeTexturePathError("Texture path contains a NUL")
    path = PurePosixPath(name.replace("\\", "/"))
    if path.is_absolute() or not path.parts or any(part in ("", ".", "..") for part in path.parts):
        raise UnsafeTexturePathError(f"Unsafe texture path: {name!r}")
    if path.parts[0].endswith(":"):
        raise UnsafeTexturePathError(f"Unsafe drive-qualified texture path: {name!r}")
    return path.as_posix()


def _member_name(info: zipfile.ZipInfo) -> str:
    """The member name as RSF compares it: its raw bytes read as Latin-1, like the INIs."""
    encoding = "utf-8" if info.flag_bits & 0x800 else "cp437"
    return info.filename.encode(encoding).decode("latin-1")


class TextureResolver:
    def __init__(
        self,
        *,
        archive: str | Path | None = None,
        directory: str | Path | None = None,
        max_members: int = 100_000,
        max_total_size: int = 8 * 1024**3,
        max_ratio: int = 1_032,
    ):
        if (archive is None) == (directory is None):
            raise ValueError("Specify exactly one RBZ archive or extracted directory")
        self.archive = (
            current_filesystem().read_path(archive)
            if archive is not None
            else None
        )
        self.directory = (
            current_filesystem().read_path(directory)
            if directory is not None
            else None
        )
        self.max_ratio = max_ratio
        self._full: dict[str, ResolvedTexture] = {}

        if self.archive is not None:
            self._index_archive(max_members, max_total_size)
        else:
            self._index_directory(max_members, max_total_size)

    @classmethod
    def from_rbz(cls, path: str | Path, **limits: int) -> TextureResolver:
        return cls(archive=path, **limits)

    @classmethod
    def from_directory(cls, path: str | Path, **limits: int) -> TextureResolver:
        return cls(directory=path, **limits)

    def _add(self, normalised: str, result: ResolvedTexture) -> None:
        folded = normalised.casefold()
        if folded in self._full:
            raise AmbiguousTextureError(
                f"Duplicate case-insensitive texture path {normalised!r}"
            )
        self._full[folded] = result

    def _index_archive(self, max_members: int, max_total_size: int) -> None:
        assert self.archive is not None
        try:
            with current_filesystem().open(
                self.archive, "rb"
            ) as stream, zipfile.ZipFile(stream) as archive:
                infos = archive.infolist()
                if len(infos) > max_members:
                    raise TextureError(
                        f"RBZ has {len(infos)} entries; maximum is {max_members}"
                    )
                total = 0
                for info in infos:
                    if info.is_dir():
                        continue
                    mode = (info.external_attr >> 16) & 0o170000
                    if mode == stat.S_IFLNK:
                        raise UnsafeTexturePathError(
                            f"RBZ contains a symbolic link: {info.filename!r}"
                        )
                    normalised = _normalise_member(_member_name(info))
                    total += info.file_size
                    if total > max_total_size:
                        raise TextureError(
                            f"RBZ uncompressed size exceeds {max_total_size} bytes"
                        )
                    if (
                        info.file_size
                        and info.compress_size == 0
                        or info.compress_size
                        and info.file_size / info.compress_size > self.max_ratio
                    ):
                        raise TextureError(
                            f"RBZ entry has unsafe compression ratio: {info.filename!r}"
                        )
                    self._add(
                        normalised,
                        ResolvedTexture(
                            requested_name=normalised,
                            source="rbz",
                            location=info.filename,
                            size=info.file_size,
                        ),
                    )
        except (OSError, zipfile.BadZipFile) as exc:
            raise TextureError(f"Invalid RBZ archive {self.archive}: {exc}") from exc

    def _index_directory(self, max_members: int, max_total_size: int) -> None:
        assert self.directory is not None
        filesystem = current_filesystem()
        if not filesystem.is_dir(self.directory):
            raise TextureError(f"Extracted texture directory does not exist: {self.directory}")
        count = 0
        total = 0
        for path in filesystem.rglob(self.directory, "*"):
            if not filesystem.is_file(path):
                continue
            resolved = filesystem.read_path(path)
            try:
                relative = resolved.relative_to(self.directory)
            except ValueError as exc:
                raise UnsafeTexturePathError(
                    f"Extracted texture escapes its root: {path}"
                ) from exc
            normalised = _normalise_member(relative.as_posix())
            count += 1
            if count > max_members:
                raise TextureError(
                    f"Extracted texture directory exceeds {max_members} files"
                )
            size = filesystem.stat(resolved).st_size
            total += size
            if total > max_total_size:
                raise TextureError(
                    f"Extracted texture directory exceeds {max_total_size} bytes"
                )
            self._add(
                normalised,
                ResolvedTexture(
                    requested_name=normalised,
                    source="directory",
                    location=str(resolved),
                    size=size,
                ),
            )

    @property
    def payload_name(self) -> str:
        if self.archive is not None:
            return self.archive.stem
        assert self.directory is not None
        return self.directory.name

    def _local_name(self, path: str) -> str | None:
        normalised = _normalise_member(path)
        root, _, rest = normalised.partition("/")
        if not rest or root.casefold() != self.payload_name.casefold():
            return None
        return normalised if self.archive is not None else rest

    def entry(self, path: str) -> ResolvedTexture | None:
        name = self._local_name(path)
        result = self._full.get(name.casefold()) if name else None
        if result is None or self.archive is None:
            return result
        return replace(result, archive=self.archive)

    @contextlib.contextmanager
    def open(self, path: str, *, max_size: int = 512 * 1024**2) -> Iterator[BinaryIO]:
        resolved = self.entry(path)
        if resolved is None:
            raise MissingTextureError(f"Texture not found: {path!r}")
        if resolved.size > max_size:
            raise TextureError(
                f"Texture {path!r} is {resolved.size} bytes; maximum is {max_size}"
            )
        if resolved.source == "directory":
            with current_filesystem().open(Path(resolved.location), "rb") as stream:
                yield stream
            return
        source = resolved.archive or self.archive
        assert source is not None
        with current_filesystem().open(
            source, "rb"
        ) as file, zipfile.ZipFile(file) as archive:
            with archive.open(resolved.location, "r") as stream:
                yield stream


def parse_texture_filename_map(data: bytes) -> dict[str, str]:
    if len(data) < 4:
        raise TextureError("TextureFilenameMap.dat is truncated")
    count = int.from_bytes(data[:4], "little")
    strings = data[4:].split(b"\0")
    if len(strings) <= 2 * count:
        raise TextureError(
            f"TextureFilenameMap.dat declares {count} entries but holds fewer"
        )
    result: dict[str, str] = {}
    for index in range(count):
        source = _normalise_member(strings[2 * index].decode("latin-1"))
        target = _normalise_member(strings[2 * index + 1].decode("latin-1"))
        result.setdefault(source.casefold(), target)
    return result


def parse_texture_filename_map_file(path: str | Path) -> dict[str, str]:
    source = current_filesystem().read_path(path)
    try:
        data = current_filesystem().read_bytes(source)
    except OSError as exc:
        raise TextureError(f"Unable to read {source}: {exc}") from exc
    return parse_texture_filename_map(data)


RoadCondition = tuple[str, str]
DEFAULT_ROAD_CONDITION: RoadCondition = ("dry", "new")
ROAD_CONDITIONS: tuple[RoadCondition, ...] = tuple(
    (wetness, wear)
    for wetness in ("dry", "damp", "wet")
    for wear in ("new", "normal", "worn")
)

_RSF_TINT_ORDER = {"M": "MNEO", "N": "NMEO", "E": "ENMO", "O": "OENM"}
_RSF_WETNESS_ORDER: dict[str | None, tuple[str, ...]] = {
    None: ("", "dry", "damp", "wet"),
    "dry": ("dry", "damp", "wet", ""),
    "damp": ("damp", "wet", "dry", ""),
    "wet": ("wet", "damp", "dry", ""),
}
_RSF_WEAR_ORDER: dict[str | None, tuple[str, ...]] = {
    None: ("", "new", "normal", "worn"),
    "new": ("new", "normal", "worn", ""),
    "normal": ("normal", "new", "worn", ""),
    "worn": ("worn", "normal", "new", ""),
}


def _request_stem(filename: str) -> str:
    name = re.split(r"[\\/]", filename)[-1]
    head, dot, _ = name.rpartition(".")
    return head if dot else name


class OriginalTextureLookup:
    """The payload member RBR with RSF loads for each texture of one stage variant.

    HedgeHog3D requests ``<base>_textures/<wetness>[/<wear>]/<stem>.dds``
    (shadows: ``<base>_textures/<name>``). When that path is neither a file nor a
    key of the stage's RSF map, Rallysimfans.hu.dll substitutes the first
    existing tint, wetness and wear sibling. HedgeHog3D then maps the result
    through the RSF map merged with Maps/TextureFilenameMap.dat; shadows are not
    mapped.
    """

    def __init__(
        self,
        base: str,
        payloads: Callable[[str], TextureResolver | None],
        *,
        rsf_remap: Mapping[str, str] | None = None,
        stock_map: Mapping[str, str] | None = None,
        condition: RoadCondition = DEFAULT_ROAD_CONDITION,
    ):
        self.base = base
        self.condition = condition
        self._payloads = payloads
        self._payload_cache: dict[str, TextureResolver | None] = {}
        self._rsf_remap = dict(rsf_remap or {})
        self._map = dict(self._rsf_remap)
        for source, target in (stock_map or {}).items():
            self._map.setdefault(
                source, self._rsf_remap.get(target.casefold(), target)
            )

    def with_condition(self, condition: RoadCondition) -> OriginalTextureLookup:
        lookup = copy.copy(self)
        lookup.condition = condition
        return lookup

    def _payload(self, path: str) -> TextureResolver | None:
        name = path.partition("/")[0]
        key = name.casefold()
        if key not in self._payload_cache:
            self._payload_cache[key] = self._payloads(name)
        return self._payload_cache[key]

    def _exists(self, path: str) -> bool:
        resolver = self._payload(path)
        return resolver is not None and resolver.entry(path) is not None

    def _rsf(
        self,
        requested: str,
        wetness: str | None,
        wear: str | None,
        name: str,
        extension: str,
    ) -> str:
        if (
            not requested.startswith("track-")
            or requested.casefold() in self._rsf_remap
            or self._exists(requested)
        ):
            return requested
        prefix = self.base[:-1]
        for tint in _RSF_TINT_ORDER.get(self.base[-1:].upper(), ""):
            for candidate_wetness in _RSF_WETNESS_ORDER[wetness]:
                for candidate_wear in _RSF_WEAR_ORDER[wear]:
                    if candidate_wear and not candidate_wetness:
                        continue
                    candidate = "/".join(
                        (
                            f"{prefix}{tint}_textures",
                            *filter(None, (candidate_wetness, candidate_wear)),
                            name + extension,
                        )
                    )
                    # RSF checks loose files, then its map, then the .rbz.
                    resolver = self._payload(candidate)
                    member = resolver.entry(candidate) if resolver else None
                    if member is not None and resolver.archive is None:
                        return candidate
                    target = self._rsf_remap.get(candidate.casefold())
                    if target is not None:
                        return target
                    if member is not None:
                        return candidate
        return requested

    def _mapped(
        self,
        requested: str,
        wetness: str,
        wear: str | None,
        stem: str,
    ) -> str:
        path = self._rsf(requested, wetness, wear, stem, ".dds").lower()
        return self._map.get(path.casefold(), path)

    def path(self, texture: OriginalTexture) -> str | None:
        wetness, wear = self.condition
        if texture.kind == "shadow":
            name = _normalise_member(texture.filename)
            extension = "" if name.casefold().endswith(".dat") else ".dds"
            path = self._rsf(
                f"{self.base}_textures/{name}", None, None, name, extension
            )
            return path if self._exists(path) else None
        stem = _request_stem(texture.filename)
        with_wear = f"{self.base}_textures/{wetness}/{wear}/{stem}.dds"
        without_wear = f"{self.base}_textures/{wetness}/{stem}.dds"
        if texture.kind == "specular":
            path = self._mapped(with_wear, wetness, wear, stem)
            if not self._exists(path):
                path = self._mapped(without_wear, wetness, None, stem)
        elif texture.ground_texture:
            path = self._mapped(with_wear, wetness, wear, stem)
        else:
            path = self._mapped(without_wear, wetness, None, stem)
        return path if self._exists(path) else None

    def complete(self, textures: Iterable[OriginalTexture]) -> bool:
        return all(self.path(texture) is not None for texture in textures)

    @contextlib.contextmanager
    def open(
        self,
        texture: OriginalTexture,
        *,
        max_size: int = 512 * 1024**2,
    ) -> Iterator[BinaryIO]:
        path = self.path(texture)
        if path is None:
            wetness, wear = self.condition
            raise MissingTextureError(
                f"Texture {texture.filename!r} has no {wetness}/{wear} file"
            )
        resolver = self._payload(path)
        assert resolver is not None
        with resolver.open(path, max_size=max_size) as stream:
            yield stream


def first_complete_condition(
    lookup: OriginalTextureLookup,
    texture_ini: OriginalTextureIni,
) -> RoadCondition | None:
    textures = (
        *texture_ini.textures,
        *texture_ini.specular_textures,
        *texture_ini.shadow_textures,
    )
    for condition in ROAD_CONDITIONS:
        if lookup.with_condition(condition).complete(textures):
            return condition
    return None
