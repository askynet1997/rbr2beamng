from __future__ import annotations

import re
import struct
from dataclasses import dataclass

from .binary import BinaryReader, FormatError, Source, checked_count, load_source
from .models import Fence, FencePost, FncFile


_DAT_MAGIC = b"CHELALIC"
_DDS_NAME = re.compile(rb"([A-Za-z0-9_ .-]+\.dds)\x00", re.IGNORECASE)


@dataclass(frozen=True)
class FenceRenderDefinition:
    strip_height: float
    u_minimum: float
    v_minimum: float
    u_maximum: float
    v_maximum: float
    strip_count: int
    strip_gap: float


# Source: RichardBurnsRally_SSE.exe SHA-256
# 2c1f28035d1120502db47a9dbfeec0c48dd2f1704202c1b2299a4a763e72937a,
# definition table at VA 0x007D81B0.
FENCE_RENDER_DEFINITIONS = (
    FenceRenderDefinition(0.074, 0.0, 0.0, 14.0, 1.0, 1, 0.0),
    FenceRenderDefinition(0.074, 0.0, 0.0, 28.0, 1.0, 1, 0.0),
    FenceRenderDefinition(0.9, 0.0, 0.0, 32.0, 27.125, 1, 0.0),
    FenceRenderDefinition(0.9, 0.0, 0.0, 64.0, 27.125, 1, 0.0),
    FenceRenderDefinition(0.074, 0.961, 0.031, 0.992, 0.73, 1, 0.0),
    FenceRenderDefinition(0.9, 0.961, 0.031, 0.992, 0.73, 1, 0.0),
    FenceRenderDefinition(0.074, 0.816, 0.031, 0.848, 0.73, 1, 0.0),
    FenceRenderDefinition(0.9, 0.816, 0.031, 0.848, 0.73, 1, 0.0),
    FenceRenderDefinition(0.074, 0.863, 0.027, 0.895, 0.73, 1, 0.0),
    FenceRenderDefinition(0.9, 0.863, 0.027, 0.895, 0.73, 1, 0.0),
    FenceRenderDefinition(0.074, 0.91, 0.027, 0.941, 0.73, 1, 0.0),
    FenceRenderDefinition(0.9, 0.91, 0.027, 0.941, 0.73, 1, 0.0),
    FenceRenderDefinition(0.074, 0.0, 0.0, 14.0, 1.0, 2, 0.4),
    FenceRenderDefinition(0.074, 0.0, 0.0, 28.0, 1.0, 2, 0.4),
    FenceRenderDefinition(0.074, 0.0, 0.0, 1.0, 1.0, 2, 0.4),
)


def fence_render_definition(selector: int) -> FenceRenderDefinition | None:
    if 0 <= selector < len(FENCE_RENDER_DEFINITIONS):
        return FENCE_RENDER_DEFINITIONS[selector]
    return None


def parse_fnc(
    source: Source,
    *,
    max_fences: int = 100_000,
    max_posts: int = 1_000_000,
    max_textures: int = 1_000,
) -> FncFile:
    loaded = load_source(source)
    reader = BinaryReader(loaded.view, context=f"FNC {loaded.name}")
    version = reader.u32()
    if version != 2:
        raise FormatError(
            f"Unsupported FNC version {version}; expected 2",
            offset=0,
            context="FNC header",
        )
    fence_count = checked_count(
        reader.u32(),
        limit=max_fences,
        offset=4,
        context="FNC fence count",
    )
    fences: list[Fence] = []
    total_posts = 0
    for fence_index in range(fence_count):
        count_offset = reader.offset
        post_count = checked_count(
            reader.u32(),
            limit=max_posts - total_posts,
            offset=count_offset,
            context=f"FNC fence {fence_index} post count",
        )
        total_posts += post_count
        tile_type, pole_type, tile_texture, pole_texture = reader.unpack("<4I")
        bounds = reader.bounds()
        posts: list[FencePost] = []
        for _ in range(post_count):
            position = reader.vec3()
            post_bounds = reader.bounds()
            blue, green, red, alpha = reader.unpack("<4B")
            posts.append(
                FencePost(
                    position=position,
                    bounds=post_bounds,
                    color=(red, green, blue, alpha),
                )
            )
        fences.append(
            Fence(
                tile_type=tile_type,
                pole_type=pole_type,
                tile_texture_index=tile_texture,
                pole_texture_index=pole_texture,
                bounds=bounds,
                posts=tuple(posts),
            )
        )

    if reader.remaining == 0:
        return FncFile(version, tuple(fences), (), loaded.view)

    texture_count = checked_count(
        reader.u32(),
        limit=max_textures,
        offset=reader.offset - 4,
        context="FNC texture count",
    )
    record_size = reader.u32()
    if record_size != 32:
        raise FormatError(
            f"Unsupported FNC texture record size {record_size}; expected 32",
            offset=reader.offset - 4,
            context="FNC texture table",
        )
    textures = tuple(
        reader.fixed_string(record_size)
        for _ in range(texture_count)
    )
    reader.require_end()
    for fence_index, fence in enumerate(fences):
        if len(fence.posts) < 2:
            continue
        for label, index in (
            ("tile", fence.tile_texture_index),
            ("pole", fence.pole_texture_index),
        ):
            if index >= len(textures):
                raise FormatError(
                    f"FNC fence {fence_index} {label} texture index {index} "
                    f"exceeds texture count {len(textures)}",
                    context="FNC fence texture",
                )
    return FncFile(version, tuple(fences), textures, loaded.view)


def read_fence_texture_archive(source: Source) -> dict[str, bytes]:
    loaded = load_source(source)
    raw = bytes(loaded.view)
    if len(raw) < 16 or raw[:8] != _DAT_MAGIC:
        raise FormatError("Invalid RBR fence DAT header", context=loaded.name)
    data_offset = struct.unpack_from("<I", raw, 12)[0]
    if data_offset < 16 or data_offset >= len(raw):
        raise FormatError(
            f"Invalid RBR fence DAT data offset {data_offset}",
            offset=12,
            context=loaded.name,
        )
    names = [
        match.group(1).decode("latin-1")
        for match in _DDS_NAME.finditer(raw, 16, data_offset)
    ]
    starts = [
        match.start()
        for match in re.finditer(b"DDS ", raw[data_offset:])
    ]
    starts = [data_offset + offset for offset in starts]
    if not names or len(names) != len(starts):
        raise FormatError(
            f"RBR fence DAT has {len(names)} names and {len(starts)} DDS payloads",
            context=loaded.name,
        )
    result: dict[str, bytes] = {}
    for index, name in enumerate(names):
        end = starts[index + 1] if index + 1 < len(starts) else len(raw)
        result[name.casefold()] = raw[starts[index]:end]
    return result
