from __future__ import annotations

import io
import re
import struct
import threading
from collections.abc import Iterable, Iterator
from pathlib import Path, PureWindowsPath
from typing import TYPE_CHECKING

import numpy as np
from PIL import Image

from ...models import MeshPart
from ...original.adapter import INTERACTIVE_OBJECTS, VisualMaterialSpec, _make_part
from ...original.binary import FormatError
from ...original.lbs import OBJECT_BLOCKS, parse_lbs_sections
from ...original.models import LbsFile
from ...original.textures import (
    MissingTextureError,
    OriginalTexture,
    OriginalTextureIni,
    OriginalTextureLookup,
    TextureResolver,
    parse_texture_filename_map_file,
    parse_texture_ini,
    parse_texture_remap,
)
from .mod import ALGE_BANNER, CERTINA, TAG_HEUER, WARTHOG
from .start_clocks import (
    StartClock,
    StartClocks,
    clock_background,
    is_tag_heuer_texture,
    region_overlay,
    warthog_clock,
    with_board_insets,
)

if TYPE_CHECKING:
    from ...models import StageInspection


# HedgeHog3D drives the interactive group named exactly START_CLOCK as the
# start clock: it draws HH:MM:SS glyphs over the digits placeholder and a red
# or green flare over the lamps. Blender copies of it keep a ".001" style
# suffix on the group and texture names, and RBR leaves those static. Some
# exporters flatten the texture folder into a dash prefix ("assets-objects-").
_START_CLOCK_GROUP = re.compile(r"START_CLOCK(\.\d+)?")
_ITEM_ROLE = re.compile(r"(?:.*-)?start_clock_(digits_placeholder|redlight_on|greenlight_on)(\.\d+)?(_[a-z])?")
_ROLES = {"digits_placeholder": "digits", "redlight_on": "red", "greenlight_on": "green"}
# Textures used by more triangles than this are scenery, not a TAG Heuer board
# (one board is 24 to 106 triangles).
_TAG_HEUER_MAX_TRIANGLES = 400
# The Karlstad start banners, with an ALGE-TIMING time and countdown display.
_ALGE_BANNER_TEXTURE = re.compile("start-z")
# The Certina board above the start clock of PS Valstagna and Saalahti, in an
# atlas also holding a tripod and spare digits.
_CERTINA_TEXTURE = re.compile(r"o-clockan(\.\d+)?(_[a-z])?")
# Metres; narrower board overlays are texture edges on the sides of other objects.
_MIN_BOARD_SIZE = 0.05
_MATERIAL = VisualMaterialSpec(
    name="rbr_start_clock",
    material_index=0,
    source_kind="interactive",
    diffuse_1=None,
    diffuse_2=None,
    specular=None,
    shadow=None,
    render_flags=0,
    water=False,
)


def _redness(texture_resolver: TextureResolver, filename: str) -> float | None:
    """Mean red minus green of a lamp texture, None when it is missing or does not decode."""
    try:
        with texture_resolver.open(filename, max_size=1024**2) as stream:
            image = Image.open(io.BytesIO(stream.read())).convert("RGB")
    except (MissingTextureError, OSError, ValueError):
        return None
    pixels = np.asarray(image, dtype=np.float64)
    return float(np.mean(pixels[..., 0] - pixels[..., 1]))


def _warthog_clocks(
    lbs: LbsFile,
    texture_ini: OriginalTextureIni,
    texture_resolver: OriginalTextureLookup,
    warnings: list[str],
) -> tuple[list[StartClock], dict]:
    groups = []
    textures: dict[str, OriginalTexture] = {}
    for group in lbs.object_data_groups.get(INTERACTIVE_OBJECTS, ()):
        # unplaced copies are never drawn (Osli-Stryckovy keeps one without its green lamp)
        if not _START_CLOCK_GROUP.fullmatch(group.name) or not group.instances:
            continue
        items = {}
        for item in group.items:
            index = item.data.texture_1
            if index is None or not 0 <= index < len(texture_ini.textures):
                continue
            texture = texture_ini.textures[index]
            match = _ITEM_ROLE.fullmatch(
                PureWindowsPath(texture.filename.replace("/", "\\")).stem.casefold()
            )
            if match:
                role = _ROLES[match.group(1)]
                items[role] = item
                textures[role] = texture
        # stages hide unwanted clocks by retexturing them, or reuse the group name for signs
        if "digits" not in items:
            continue
        if set(items) != set(_ROLES.values()):
            warnings.append(
                f"RBR start clock is missing {sorted(set(_ROLES.values()) - set(items))}; "
                "it stays static"
            )
            continue
        groups.append((group, items))
    if not groups:
        return [], {}
    # some stages give the red lamp item the green texture and the other way round;
    # the lamp lit before the start is the one that looks red
    red, green = _redness(texture_resolver, textures["red"]), _redness(texture_resolver, textures["green"])
    swapped = red is not None and green is not None and red < green
    clocks: list[StartClock] = []
    for group, items in groups:
        if swapped:
            items = {**items, "red": items["green"], "green": items["red"]}
        roles = {id(item) for item in items.values()}
        for _key, matrix in group.instances:
            parts = {
                role: _make_part(
                    name=f"rbr_start_clock_{len(clocks)}_{role}",
                    values=item.data.vertices,
                    faces=item.data.triangles,
                    material=_MATERIAL,
                    matrices=(matrix,),
                )
                for role, item in items.items()
            }
            if all(part is not None for part in parts.values()):
                housing = [
                    _make_part(
                        name="housing",
                        values=item.data.vertices,
                        faces=item.data.triangles,
                        material=_MATERIAL,
                        matrices=(matrix,),
                    )
                    for item in group.items
                    if id(item) not in roles
                ]
                clocks.append(warthog_clock(parts, [part for part in housing if part is not None]))
    return clocks, {}


def _tag_heuer_textures(
    lbs: LbsFile,
    texture_ini: OriginalTextureIni,
    texture_resolver: OriginalTextureLookup,
) -> dict[int, Image.Image]:
    """Textures with the TAG Heuer board layout, among those of small objects.

    Stages name the board texture freely (CLOCK.dds, texture01_O.dds, ...),
    so it is recognised by content.
    """
    triangles: dict[int, int] = {}
    for group in lbs.object_blocks or ():
        for block in (*group.primary, *group.secondary) if group is not None else ():
            triangles[block.texture_1] = triangles.get(block.texture_1, 0) + len(block.main_triangles)
    for group in lbs.object_data_groups.get(INTERACTIVE_OBJECTS, ()):
        for item in group.items:
            triangles[item.data.texture_1] = (
                triangles.get(item.data.texture_1, 0) + len(item.data.triangles) * len(group.instances)
            )
    textures = {}
    for index, count in triangles.items():
        if index is None or not 0 <= index < len(texture_ini.textures) or count > _TAG_HEUER_MAX_TRIANGLES:
            continue
        try:
            with texture_resolver.open(texture_ini.textures[index], max_size=16 * 1024**2) as stream:
                image = Image.open(io.BytesIO(stream.read()))
                image.load()
        except (MissingTextureError, OSError, ValueError):
            continue
        if is_tag_heuer_texture(image):
            textures[index] = image
    return textures


def _readable(overlay: MeshPart | None) -> bool:
    """Whether the overlay spans _MIN_BOARD_SIZE both across and up its board."""
    if overlay is None:
        return False
    vertices = np.asarray(overlay.vertices, dtype=np.float64)
    triangles = vertices[np.asarray(overlay.faces).reshape(-1, 3)]
    normal = np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0]).sum(axis=0)
    across = np.cross(normal, (0.0, 0.0, 1.0))
    if np.linalg.norm(across) < 1e-9 * np.linalg.norm(normal):
        across = np.cross(normal, (1.0, 0.0, 0.0))
    up = np.cross(across, normal)
    return all(
        np.ptp(vertices @ (axis / np.linalg.norm(axis))) >= _MIN_BOARD_SIZE
        for axis in (across, up)
    )


def _board_clocks(
    lbs: LbsFile,
    textures: dict[int, Image.Image],
    design: str,
) -> tuple[list[StartClock], dict[str, Image.Image]]:
    """Overlays of the object blocks and interactive groups showing the design's region of one of textures."""
    if not textures:
        return [], {}
    clocks: list[StartClock] = []
    used = []
    for group in lbs.object_blocks or ():
        for block in (*group.primary, *group.secondary) if group is not None else ():
            if block.texture_1 not in textures:
                continue
            part = _make_part(
                name=f"rbr_start_clock_{len(clocks)}",
                values=block.vertices,
                faces=block.main_triangles,
                material=_MATERIAL,
            )
            overlay = part and region_overlay(part, design)
            if _readable(overlay):
                clocks.append(StartClock(design, (overlay,)))
                used.append(block.texture_1)
    for group in lbs.object_data_groups.get(INTERACTIVE_OBJECTS, ()):
        boards = [item for item in group.items if item.data.texture_1 in textures]
        for _key, matrix in group.instances if boards else ():
            parts = {
                id(item): _make_part(
                    name=f"rbr_start_clock_{len(clocks)}",
                    values=item.data.vertices,
                    faces=item.data.triangles,
                    material=_MATERIAL,
                    matrices=(matrix,),
                )
                for item in group.items
            }
            for board in boards:
                overlay = parts[id(board)] and region_overlay(parts[id(board)], design)
                if not _readable(overlay):
                    continue
                # lamp lenses and other items set into the board
                for item in group.items:
                    if item is not board and parts[id(item)] is not None:
                        overlay = with_board_insets(overlay, parts[id(item)])
                clocks.append(StartClock(design, (overlay,)))
                used.append(board.data.texture_1)
    if not clocks:
        return [], {}
    return clocks, {"background": clock_background(design, textures[used[0]])}


def _tag_heuer_clocks(
    lbs: LbsFile,
    texture_ini: OriginalTextureIni,
    texture_resolver: OriginalTextureLookup,
    warnings: list[str],
) -> tuple[list[StartClock], dict[str, Image.Image]]:
    return _board_clocks(lbs, _tag_heuer_textures(lbs, texture_ini, texture_resolver), TAG_HEUER)


def _named_textures(
    texture_ini: OriginalTextureIni,
    texture_resolver: OriginalTextureLookup,
    stem: re.Pattern[str],
    board: str,
    warnings: list[str],
) -> dict[int, Image.Image]:
    """The textures whose lowercase file stem matches stem, by texture index."""
    textures = {}
    for index, texture in enumerate(texture_ini.textures):
        if not stem.fullmatch(PureWindowsPath(texture.filename.replace("/", "\\")).stem.casefold()):
            continue
        try:
            with texture_resolver.open(texture, max_size=16 * 1024**2) as stream:
                textures[index] = Image.open(io.BytesIO(stream.read()))
                textures[index].load()
        except (MissingTextureError, OSError, ValueError) as exc:
            warnings.append(f"{board} texture is unreadable ({exc}); it stays static")
    return textures


def _alge_banner_clocks(
    lbs: LbsFile,
    texture_ini: OriginalTextureIni,
    texture_resolver: OriginalTextureLookup,
    warnings: list[str],
) -> tuple[list[StartClock], dict[str, Image.Image]]:
    textures = _named_textures(texture_ini, texture_resolver, _ALGE_BANNER_TEXTURE, "ALGE-TIMING start banner", warnings)
    return _board_clocks(lbs, textures, ALGE_BANNER)


def _certina_clocks(
    lbs: LbsFile,
    texture_ini: OriginalTextureIni,
    texture_resolver: OriginalTextureLookup,
    warnings: list[str],
) -> tuple[list[StartClock], dict[str, Image.Image]]:
    textures = _named_textures(texture_ini, texture_resolver, _CERTINA_TEXTURE, "Certina start clock", warnings)
    return _board_clocks(lbs, textures, CERTINA)


def adapt_start_clocks(
    lbs: LbsFile,
    texture_ini: OriginalTextureIni,
    texture_resolver: OriginalTextureLookup,
    warnings: list[str],
) -> StartClocks | None:
    """Overlays of the start clocks the RBR Countdown Clocks mod drives, in Z-up source space."""
    clocks = []
    textures = {}
    for design, find in (
        (WARTHOG, _warthog_clocks),
        (TAG_HEUER, _tag_heuer_clocks),
        (ALGE_BANNER, _alge_banner_clocks),
        (CERTINA, _certina_clocks),
    ):
        design_clocks, design_textures = find(lbs, texture_ini, texture_resolver, warnings)
        if design_clocks:
            clocks.extend(design_clocks)
            textures[design] = design_textures
    if not clocks:
        return None
    return StartClocks(clocks=tuple(clocks), textures=textures)


def _has_warthog_clock(lbs: LbsFile, texture_ini: OriginalTextureIni) -> bool:
    for group in lbs.object_data_groups.get(INTERACTIVE_OBJECTS, ()):
        if not _START_CLOCK_GROUP.fullmatch(group.name) or not group.instances:
            continue
        items = {}
        for item in group.items:
            index = item.data.texture_1
            if index is None or not 0 <= index < len(texture_ini.textures):
                continue
            texture = texture_ini.textures[index]
            match = _ITEM_ROLE.fullmatch(
                PureWindowsPath(texture.filename.replace("/", "\\")).stem.casefold()
            )
            if match:
                items[_ROLES[match.group(1)]] = item
        if set(items) != set(_ROLES.values()):
            continue
        for _key, matrix in group.instances:
            if all(
                _make_part(
                    name=role,
                    values=item.data.vertices,
                    faces=item.data.triangles,
                    material=_MATERIAL,
                    matrices=(matrix,),
                )
                is not None
                for role, item in items.items()
            ):
                return True
    return False


def _has_readable_board(
    lbs: LbsFile,
    texture_indices: set[int],
    design: str,
) -> bool:
    if not texture_indices:
        return False
    for group in lbs.object_blocks or ():
        for block in (*group.primary, *group.secondary) if group is not None else ():
            if block.texture_1 not in texture_indices:
                continue
            part = _make_part(
                name="rbr_start_clock",
                values=block.vertices,
                faces=block.main_triangles,
                material=_MATERIAL,
            )
            if part is not None and _readable(region_overlay(part, design)):
                return True
    for group in lbs.object_data_groups.get(INTERACTIVE_OBJECTS, ()):
        boards = [item for item in group.items if item.data.texture_1 in texture_indices]
        if not boards or not group.instances:
            continue
        for _key, matrix in group.instances:
            for board in boards:
                part = _make_part(
                    name="rbr_start_clock",
                    values=board.data.vertices,
                    faces=board.data.triangles,
                    material=_MATERIAL,
                    matrices=(matrix,),
                )
                if part is not None and _readable(region_overlay(part, design)):
                    return True
    return False


def _texture_is_readable(
    texture_resolver: OriginalTextureLookup,
    texture: OriginalTexture,
) -> bool:
    try:
        with texture_resolver.open(texture, max_size=16 * 1024**2) as stream:
            Image.open(io.BytesIO(stream.read())).load()
            return True
    except (MissingTextureError, OSError, ValueError):
        return False


def _is_tag_heuer_bytes(raw: bytes) -> bool:
    if len(raw) >= 20 and raw[:4] == b"DDS ":
        height, width = struct.unpack_from("<2I", raw, 12)
        if width != height:
            return False
    try:
        image = Image.open(io.BytesIO(raw))
        if image.width != image.height:
            return False
        image.load()
    except (OSError, ValueError):
        return False
    return is_tag_heuer_texture(image)


def _has_tag_heuer_clock(
    lbs: LbsFile,
    texture_ini: OriginalTextureIni,
    texture_resolver: OriginalTextureLookup,
    cancelled: threading.Event | None = None,
) -> bool:
    triangles: dict[int, int] = {}
    for group in lbs.object_blocks or ():
        for block in (*group.primary, *group.secondary) if group is not None else ():
            triangles[block.texture_1] = (
                triangles.get(block.texture_1, 0) + len(block.main_triangles)
            )
    for group in lbs.object_data_groups.get(INTERACTIVE_OBJECTS, ()):
        for item in group.items:
            triangles[item.data.texture_1] = (
                triangles.get(item.data.texture_1, 0)
                + len(item.data.triangles) * len(group.instances)
            )
    for index, count in triangles.items():
        if cancelled is not None and cancelled.is_set():
            return False
        if (
            index is None
            or not 0 <= index < len(texture_ini.textures)
            or count > _TAG_HEUER_MAX_TRIANGLES
        ):
            continue
        try:
            with texture_resolver.open(
                texture_ini.textures[index], max_size=16 * 1024**2
            ) as stream:
                raw = stream.read()
        except (MissingTextureError, OSError, ValueError):
            continue
        if _is_tag_heuer_bytes(raw) and _has_readable_board(lbs, {index}, TAG_HEUER):
            return True
    return False


def scan_original_start_clocks(
    rbr_root: Path,
    stages: Iterable[StageInspection],
    cancelled: threading.Event,
) -> Iterator[str]:
    """Yield source keys of Original RBR stages with at least one supported start clock."""
    from ...original.catalog import (
        AmbiguousOriginalAssetError,
        CatalogError,
        MissingOriginalAssetError,
        OriginalStageFiles,
    )
    from ...original.source import (
        ORIGINAL_REQUIRED_EXTENSIONS,
        PRIMARY_TINT_ORDER,
        original_catalog,
        original_stage_id,
    )
    from ...original_converter import _case_insensitive_path, _texture_lookup

    original_stages: list[tuple[StageInspection, int]] = []
    for stage in stages:
        if stage.source_format != "original" or not stage.valid:
            continue
        stage_id = original_stage_id(stage.source_key)
        if stage_id is not None:
            original_stages.append((stage, stage_id))
    if not original_stages or cancelled.is_set():
        return
    try:
        catalog = original_catalog(rbr_root)
    except (CatalogError, OSError):
        return

    stock_path = _case_insensitive_path(catalog.maps_root / "TextureFilenameMap.dat")
    stock_map = (
        parse_texture_filename_map_file(stock_path) if stock_path is not None else None
    )
    remap00_path = _case_insensitive_path(
        catalog.maps_root / "TextureFilenameMap00.ini"
    )
    remap00 = parse_texture_remap(remap00_path) if remap00_path is not None else None
    resolver_cache: dict[Path, TextureResolver] = {}

    pass2: list[tuple[str, OriginalStageFiles, OriginalTextureIni, LbsFile | None]] = []

    # Pass 1: stages with named clock textures (Warthog, ALGE Banner, Certina).
    for stage, stage_id in original_stages:
        if cancelled.is_set():
            return
        tints = stage.variants
        if not tints:
            try:
                tints = tuple(
                    tint.value
                    for tint in catalog.complete_tints(
                        stage_id, extensions=ORIGINAL_REQUIRED_EXTENSIONS
                    )
                )
            except (CatalogError, OSError):
                continue
        if not tints:
            continue
        primary_tint = next(
            (tint for tint in PRIMARY_TINT_ORDER if tint in tints), tints[0]
        )
        try:
            files = catalog.resolve_stage(
                stage_id, primary_tint, extensions=("track", "texture_ini")
            )
            texture_ini = parse_texture_ini(files.files["texture_ini"])
        except (
            KeyError,
            CatalogError,
            MissingOriginalAssetError,
            AmbiguousOriginalAssetError,
            FormatError,
            OSError,
        ):
            continue

        warthog_roles: set[str] = set()
        alge_indices: set[int] = set()
        certina_indices: set[int] = set()
        for index, texture in enumerate(texture_ini.textures):
            stem = PureWindowsPath(texture.filename.replace("/", "\\")).stem.casefold()
            match = _ITEM_ROLE.fullmatch(stem)
            if match:
                warthog_roles.add(_ROLES[match.group(1)])
            if _ALGE_BANNER_TEXTURE.fullmatch(stem):
                alge_indices.add(index)
            if _CERTINA_TEXTURE.fullmatch(stem):
                certina_indices.add(index)

        has_warthog_stems = len(warthog_roles) == len(_ROLES)
        if not has_warthog_stems and not alge_indices and not certina_indices:
            pass2.append((stage.source_key, files, texture_ini, None))
            continue

        sections = (
            frozenset((INTERACTIVE_OBJECTS, OBJECT_BLOCKS))
            if (alge_indices or certina_indices)
            else frozenset((INTERACTIVE_OBJECTS,))
        )
        try:
            lbs = parse_lbs_sections(files.files["track"], sections)
        except (FormatError, OSError):
            continue

        if has_warthog_stems and _has_warthog_clock(lbs, texture_ini):
            yield stage.source_key
            continue

        if alge_indices or certina_indices:
            lookup = _texture_lookup(
                files,
                catalog,
                resolver_cache,
                stock_map=stock_map,
                remap00=remap00,
            )
            if alge_indices:
                readable_alge = {
                    index
                    for index in alge_indices
                    if _texture_is_readable(lookup, texture_ini.textures[index])
                }
                if _has_readable_board(lbs, readable_alge, ALGE_BANNER):
                    yield stage.source_key
                    continue
            if certina_indices:
                readable_certina = {
                    index
                    for index in certina_indices
                    if _texture_is_readable(lookup, texture_ini.textures[index])
                }
                if _has_readable_board(lbs, readable_certina, CERTINA):
                    yield stage.source_key
                    continue
            pass2.append((stage.source_key, files, texture_ini, lbs))
        else:
            pass2.append((stage.source_key, files, texture_ini, None))

    # Pass 2: remaining stages checked for content-detected TAG Heuer boards.
    for source_key, files, texture_ini, cached_lbs in pass2:
        if cancelled.is_set():
            return
        try:
            lbs = (
                cached_lbs
                if cached_lbs is not None
                else parse_lbs_sections(
                    files.files["track"],
                    frozenset((INTERACTIVE_OBJECTS, OBJECT_BLOCKS)),
                )
            )
        except (FormatError, OSError):
            continue
        lookup = _texture_lookup(
            files,
            catalog,
            resolver_cache,
            stock_map=stock_map,
            remap00=remap00,
        )
        if _has_tag_heuer_clock(lbs, texture_ini, lookup, cancelled):
            yield source_key
