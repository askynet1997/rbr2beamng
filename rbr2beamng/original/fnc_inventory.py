from __future__ import annotations

import hashlib
import json
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from ..core import ConversionCancelled, ConversionError, ProgressReporter
from ..filesystem import current_filesystem
from .catalog import CatalogError, parse_tracks_ini
from .fnc import parse_fnc
from .models import Bounds, Fence, FncFile


@dataclass(frozen=True)
class FncInventoryResult:
    output_path: Path
    source_count: int
    selector_count: int
    error_count: int


def _bounds_values(bounds: Bounds) -> tuple[float, ...]:
    return (*bounds.center, *bounds.half_extents)


def _bounds_range(values: list[Bounds]) -> dict[str, list[float]] | None:
    if not values:
        return None
    raw = [_bounds_values(value) for value in values]
    return {
        "minimum": [min(value[index] for value in raw) for index in range(6)],
        "maximum": [max(value[index] for value in raw) for index in range(6)],
    }


def _fence_summary(fence: Fence) -> dict[str, object]:
    posts = fence.posts
    return {
        "tileType": fence.tile_type,
        "poleType": fence.pole_type,
        "tileTextureIndex": fence.tile_texture_index,
        "poleTextureIndex": fence.pole_texture_index,
        "postCount": len(posts),
        "fenceBounds": list(_bounds_values(fence.bounds)),
        "postBounds": _bounds_range([post.bounds for post in posts]),
    }


def _source_summary(path: Path, raw: bytes, parsed: FncFile) -> dict[str, object]:
    fences_by_selector: dict[str, list[Fence]] = defaultdict(list)
    for fence in parsed.fences:
        fences_by_selector[_selector_key(fence)].append(fence)
    return {
        "path": str(path),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "textures": list(parsed.textures),
        "fenceCount": len(parsed.fences),
        "postCount": sum(len(fence.posts) for fence in parsed.fences),
        "selectors": [
            {
                "tileType": fences[0].tile_type,
                "poleType": fences[0].pole_type,
                "fenceCount": len(fences),
                "postCount": sum(len(fence.posts) for fence in fences),
            }
            for _key, fences in sorted(
                fences_by_selector.items(),
                key=lambda item: tuple(int(value) for value in item[0].split(":")),
            )
        ],
    }


def _selector_key(fence: Fence) -> str:
    return f"{fence.tile_type}:{fence.pole_type}"


def _selector_summary(
    fences: list[Fence],
    occurrence_count: int,
) -> dict[str, object]:
    tile_type = fences[0].tile_type
    pole_type = fences[0].pole_type
    fence_bounds = [fence.bounds for fence in fences]
    posts = [post for fence in fences for post in fence.posts]
    textures = sorted(
        {
            (fence.tile_texture_index, fence.pole_texture_index)
            for fence in fences
        }
    )
    return {
        "tileType": tile_type,
        "poleType": pole_type,
        "sourceOccurrences": occurrence_count,
        "fenceCount": len(fences),
        "postCount": sum(len(fence.posts) for fence in fences),
        "textureIndexPairs": [list(value) for value in textures],
        "fenceBounds": _bounds_range(fence_bounds),
        "postBounds": _bounds_range([post.bounds for post in posts]),
        "postColors": {
            "minimum": [
                min(post.color[index] for post in posts)
                for index in range(4)
            ] if posts else [],
            "maximum": [
                max(post.color[index] for post in posts)
                for index in range(4)
            ] if posts else [],
        },
        "samples": [_fence_summary(fence) for fence in fences[:3]],
    }


def build_fnc_inventory(
    rbr_root: Path,
    output: Path,
    *,
    reporter: ProgressReporter | None = None,
) -> FncInventoryResult:
    filesystem = current_filesystem()
    root = filesystem.read_path(rbr_root)
    output = filesystem.write_path(output)
    if filesystem.exists(output):
        raise ConversionError(f"FNC inventory output already exists: {output}")
    filesystem.mkdir(output.parent, parents=True, exist_ok=True)

    catalog = parse_tracks_ini(root / "Maps" / "Tracks.ini", install_root=root)
    sources: dict[Path, dict[str, object]] = {}
    parsed_sources: dict[Path, FncFile] = {}
    occurrences: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    entries = sorted(catalog.entries.values(), key=lambda entry: entry.stage_id)
    for stage_index, entry in enumerate(entries, 1):
        if reporter:
            reporter.emit(
                "fnc-inventory",
                "Reading Original RBR fence definitions",
                current=stage_index,
                total=len(entries),
                detail=f"{entry.stage_id}: {entry.stage_name}",
            )
        for tint in catalog.available_tints(entry):
            try:
                path = catalog.resolve_file(entry, tint, "fence")
            except CatalogError as exc:
                failures.append(
                    {
                        "stageId": entry.stage_id,
                        "stageName": entry.stage_name,
                        "tint": tint.value,
                        "error": str(exc),
                    }
                )
                continue
            source = sources.get(path)
            if source is None:
                try:
                    raw = filesystem.read_bytes(path)
                    parsed = parse_fnc(raw)
                except (OSError, ValueError) as exc:
                    failures.append(
                        {
                            "stageId": entry.stage_id,
                            "stageName": entry.stage_name,
                            "tint": tint.value,
                            "error": f"{path.name}: {exc}",
                        }
                    )
                    continue
                source = _source_summary(path, raw, parsed)
                sources[path] = source
                parsed_sources[path] = parsed
            occurrences.append(
                {
                    "stageId": entry.stage_id,
                    "stageName": entry.stage_name,
                    "tint": tint.value,
                    "path": str(path),
                }
            )

    fences_by_selector: dict[str, list[Fence]] = defaultdict(list)
    occurrences_by_selector: dict[str, int] = defaultdict(int)
    for occurrence in occurrences:
        source = parsed_sources[Path(str(occurrence["path"]))]
        for fence in source.fences:
            key = _selector_key(fence)
            occurrences_by_selector[key] += 1
    for source in parsed_sources.values():
        for fence in source.fences:
            fences_by_selector[_selector_key(fence)].append(fence)

    selectors = [
        _selector_summary(fences, occurrences_by_selector[key])
        for key, fences in sorted(
            fences_by_selector.items(),
            key=lambda item: tuple(int(value) for value in item[0].split(":")),
        )
    ]
    payload = {
        "schemaVersion": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "inputRoot": str(root),
        "summary": {
            "sourceCount": len(sources),
            "stageTintOccurrences": len(occurrences),
            "selectorCount": len(selectors),
            "errorCount": len(failures),
        },
        "selectors": selectors,
        "sources": sorted(sources.values(), key=lambda value: str(value["path"])),
        "occurrences": occurrences,
        "sourceFailures": failures,
    }
    temporary = output.parent / f".{output.name}.partial-{uuid.uuid4().hex}"
    try:
        filesystem.write_text(
            temporary,
            json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        filesystem.replace(temporary, output)
    except BaseException:
        filesystem.unlink(temporary, missing_ok=True)
        raise
    return FncInventoryResult(
        output_path=output,
        source_count=len(sources),
        selector_count=len(selectors),
        error_count=len(failures),
    )
