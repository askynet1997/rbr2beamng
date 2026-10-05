from __future__ import annotations

import re
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path
from typing import Mapping

from ..core import ConversionError, ProgressReporter, slugify
from ..filesystem import current_filesystem
from ..models import RbrStage, StageDocument, StageInspection, StageMetadata
from ..rbr import location_for_country_code, rsf_catalog_entries
from .catalog import (
    CatalogError,
    OriginalCatalog,
    OriginalCatalogEntry,
    OriginalStageFiles,
    Tint,
    parse_tracks_ini,
)


ORIGINAL_REQUIRED_EXTENSIONS = (
    "track",
    "physics",
    "driveline",
    "collision",
    "material",
    "fence",
    "texture_ini",
)
ORIGINAL_PACENOTE_VISUALIZER_EXTENSIONS = (
    "track",
    "physics",
    "driveline",
)
ORIGINAL_TINT_NAMES = {
    "M": "Morning",
    "N": "Noon",
    "E": "Evening",
    "O": "Overcast",
    "S": "Custom",
}
PRIMARY_TINT_ORDER = ("O", "N", "M", "E", "S")
_DOCUMENT_SUFFIXES = {".txt", ".md", ".nfo", ".rtf", ".html", ".htm", ".pdf"}


@dataclass(frozen=True)
class OriginalPacenoteSource:
    stage_id: int
    stage_name: str
    tint: str
    dls_path: Path


@dataclass(frozen=True)
class OriginalPacenoteSourceError:
    stage_id: int
    stage_name: str
    tint: str
    message: str


@lru_cache(maxsize=4)
def _catalog(path: str) -> OriginalCatalog:
    root = Path(path)
    return parse_tracks_ini(root / "Maps" / "Tracks.ini", install_root=root)


def original_catalog(rbr_root: Path) -> OriginalCatalog:
    return _catalog(str(current_filesystem().read_path(rbr_root)))


@lru_cache(maxsize=4)
def _rsf_entries(path: str) -> dict[int, dict[str, object]]:
    catalog_path = Path(path) / "rsfdata" / "cache" / "stages_data.json"
    filesystem = current_filesystem()
    if not filesystem.is_file(catalog_path):
        return {}
    entries: dict[int, dict[str, object]] = {}
    for value in rsf_catalog_entries(catalog_path):
        try:
            entries[int(value["stage_id"] if "stage_id" in value else value["id"])] = value
        except (KeyError, TypeError, ValueError):
            continue
    return entries


def _number(value: object) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


_RSF_SURFACE_IDS = {"1": "tarmac", "2": "gravel", "3": "snow"}
_TRACKS_INI_SURFACES = {"0": "tarmac", "1": "gravel", "2": "snow"}


def _surface_composition(entry: dict[str, object]) -> tuple[tuple[str, float], ...]:
    composition = tuple(
        (name, _number(entry.get(name)) or 0.0)
        for name in ("tarmac", "gravel", "snow")
    )
    return composition if any(amount > 0 for _name, amount in composition) else ()


def _surface(rsf: dict[str, object], properties: Mapping[str, str]) -> str:
    values = dict(_surface_composition(rsf))
    if values:
        return max(values.items(), key=lambda item: item[1])[0]
    return _RSF_SURFACE_IDS.get(str(rsf.get("surface_id", "")).strip()) or _TRACKS_INI_SURFACES.get(
        str(properties.get("Surface", "")).strip(), "unknown"
    )


def _is_document(path: Path) -> bool:
    return (
        path.suffix.casefold() in _DOCUMENT_SUFFIXES
        and current_filesystem().is_file(path)
    )


def _stage_folders(catalog: OriginalCatalog) -> tuple[tuple[int, Path], ...]:
    return tuple(
        sorted(
            (entry.stage_id, entry.track_base.parent)
            for entry in catalog.entries.values()
            if entry.track_base.parent != catalog.maps_root
        )
    )


def _catalog_documents(
    catalog: OriginalCatalog,
) -> dict[int, tuple[StageDocument, ...]]:
    return _document_index(_stage_folders(catalog))


@lru_cache(maxsize=4)
def _document_index(
    stage_folders: tuple[tuple[int, Path], ...],
) -> dict[int, tuple[StageDocument, ...]]:
    filesystem = current_filesystem()
    grouped: dict[int, list[StageDocument]] = {}
    for stage_id, folder in stage_folders:
        if not filesystem.is_dir(folder):
            continue
        for path in filesystem.iterdir(folder):
            if _is_document(path):
                grouped.setdefault(stage_id, []).append(
                    StageDocument(path, path.name)
                )
    return {
        stage_id: tuple(sorted(documents, key=lambda item: item.title.casefold()))
        for stage_id, documents in grouped.items()
    }


def _country_code(entry: OriginalCatalogEntry, rsf: dict[str, object]) -> str:
    return str(rsf.get("short_country") or entry.properties.get("CountryCode") or "")


def original_country_code(rbr_root: Path, entry: OriginalCatalogEntry) -> str:
    rsf_entries = _rsf_entries(str(current_filesystem().read_path(rbr_root)))
    return _country_code(entry, rsf_entries.get(entry.stage_id, {}))


def _splashscreen(root: Path, entry: OriginalCatalogEntry) -> Path | None:
    image = root / "rsfdata" / "images" / "stage_images" / f"{entry.stage_id}.png"
    return image if current_filesystem().is_file(image) else None


def _metadata(
    entry: OriginalCatalogEntry,
    rsf: dict[str, object],
    variants: tuple[Tint, ...],
    splashscreen: Path | None,
) -> StageMetadata:
    length = _number(rsf.get("length")) or _number(entry.properties.get("Length"))
    author = str(rsf.get("author") or entry.properties.get("Author") or "Unknown")
    name = str(rsf.get("name") or entry.stage_name or f"Original RBR stage {entry.stage_id}")
    return StageMetadata(
        folder_name=f"original:{entry.stage_id}",
        name=name,
        author=author,
        physics=_surface(rsf, entry.properties),
        version="original",
        date="",
        comment="",
        length_km=(length / 1000.0 if length and length > 100 else length),
        splashscreen=splashscreen,
        surface_composition=_surface_composition(rsf),
        author_website=str(rsf.get("author_web") or "").strip(),
        author_note=str(rsf.get("author_note") or "").strip(),
    )


def _inspection_for_entry(
    catalog: OriginalCatalog,
    entry: OriginalCatalogEntry,
    variants: tuple[Tint, ...],
    rsf_entries: dict[int, dict[str, object]],
    documents: dict[int, tuple[StageDocument, ...]],
    issues: tuple[str, ...] = (),
    include_splashscreen: bool = True,
) -> StageInspection:
    rsf = rsf_entries.get(entry.stage_id, {})
    return StageInspection(
        root=entry.track_base.parent,
        metadata=_metadata(
            entry,
            rsf,
            variants,
            _splashscreen(catalog.install_root, entry) if include_splashscreen else None,
        ),
        valid=not issues,
        issues=issues,
        location=location_for_country_code(_country_code(entry, rsf)),
        source_format="original",
        source_key=f"original:{entry.stage_id}",
        variants=tuple(tint.value for tint in variants),
        documents=documents.get(entry.stage_id, ()),
    )


def _complete_variants(
    catalog: OriginalCatalog,
    entry: OriginalCatalogEntry,
) -> tuple[tuple[Tint, ...], tuple[str, ...]]:
    try:
        variants = catalog.complete_tints(
            entry,
            extensions=ORIGINAL_REQUIRED_EXTENSIONS,
        )
    except (CatalogError, OSError) as exc:
        return (), (str(exc),)
    if not variants:
        return (), (
            "No environment of this stage has a complete set of stage files, "
            "so it cannot be converted",
        )
    return variants, ()


def discover_original_stages(
    rbr_root: Path,
    reporter: ProgressReporter | None = None,
    *,
    inspect_variants: bool = True,
) -> list[StageInspection]:
    filesystem = current_filesystem()
    root = filesystem.read_path(rbr_root)
    tracks_ini = root / "Maps" / "Tracks.ini"
    if not filesystem.is_file(tracks_ini):
        return []
    try:
        catalog = _catalog(str(root))
    except CatalogError as exc:
        raise ConversionError(str(exc)) from exc
    rsf_entries = _rsf_entries(str(root))
    documents = _catalog_documents(catalog) if inspect_variants else {}
    entries = sorted(catalog.entries.values(), key=lambda item: item.stage_id)
    result: list[StageInspection] = []
    for index, entry in enumerate(entries, 1):
        if reporter:
            reporter.emit(
                "scan",
                "Inspecting Original RBR stages",
                current=index,
                total=len(entries),
                detail=str(entry.stage_id),
            )
        variants, issues = (
            _complete_variants(catalog, entry)
            if inspect_variants
            else ((), ())
        )
        result.append(
            _inspection_for_entry(
                catalog,
                entry,
                variants,
                rsf_entries,
                documents,
                issues,
                include_splashscreen=inspect_variants,
            )
        )
    return result


def inspect_original_stage(rbr_root: Path, selector: str) -> StageInspection:
    stage_id = original_stage_id(selector)
    if stage_id is None:
        raise ConversionError(f"Not an Original RBR stage selector: {selector}")
    filesystem = current_filesystem()
    root = filesystem.read_path(rbr_root)
    try:
        catalog = _catalog(str(root))
        entry = catalog.entry(stage_id)
    except CatalogError as exc:
        raise ConversionError(str(exc)) from exc
    variants, issues = _complete_variants(catalog, entry)
    return _inspection_for_entry(
        catalog,
        entry,
        variants,
        _rsf_entries(str(root)),
        _catalog_documents(catalog),
        issues,
    )


def original_stage_id(selector: str) -> int | None:
    prefix, separator, value = selector.partition(":")
    if separator and prefix.casefold() == "original":
        try:
            return int(value)
        except ValueError as exc:
            raise ConversionError(f"Invalid Original RBR stage selector: {selector}") from exc
    return None


def discover_original_pacenote_sources(
    rbr_root: Path,
) -> tuple[
    tuple[OriginalPacenoteSource, ...],
    tuple[OriginalPacenoteSourceError, ...],
]:
    root = current_filesystem().read_path(rbr_root)
    if not current_filesystem().is_file(root / "Maps" / "Tracks.ini"):
        return (), ()
    try:
        catalog = _catalog(str(root))
    except CatalogError as exc:
        raise ConversionError(str(exc)) from exc
    sources: list[OriginalPacenoteSource] = []
    errors: list[OriginalPacenoteSourceError] = []
    for entry in sorted(catalog.entries.values(), key=lambda item: item.stage_id):
        try:
            tints = catalog.available_tints(entry)
        except (CatalogError, OSError) as exc:
            errors.append(
                OriginalPacenoteSourceError(
                    entry.stage_id,
                    entry.stage_name,
                    "",
                    str(exc),
                )
            )
            continue
        for tint in tints:
            try:
                dls_path = catalog.resolve_file(entry, tint, "driveline")
            except (CatalogError, OSError) as exc:
                errors.append(
                    OriginalPacenoteSourceError(
                        entry.stage_id,
                        entry.stage_name,
                        tint.value,
                        str(exc),
                    )
                )
                continue
            sources.append(
                OriginalPacenoteSource(
                    entry.stage_id,
                    entry.stage_name,
                    tint.value,
                    dls_path,
                )
            )
    return tuple(sources), tuple(errors)


def _resolve_original_variants(
    rbr_root: Path,
    selector: str,
    requested: tuple[str, ...],
    extensions: tuple[str, ...],
    include_inspection_assets: bool,
) -> tuple[StageInspection, tuple[OriginalStageFiles, ...]]:
    stage_id = original_stage_id(selector)
    if stage_id is None:
        raise ConversionError(f"Not an Original RBR stage selector: {selector}")
    root = current_filesystem().read_path(rbr_root)
    try:
        catalog = _catalog(str(root))
        entry = catalog.entry(stage_id)
        complete = catalog.complete_tints(entry, extensions=extensions)
        inspection = _inspection_for_entry(
            catalog,
            entry,
            complete,
            _rsf_entries(str(root)),
            _catalog_documents(catalog) if include_inspection_assets else {},
            include_splashscreen=include_inspection_assets,
        )
        available = complete
        if requested:
            wanted = {Tint(value.upper()) for value in requested}
            available = tuple(tint for tint in available if tint in wanted)
        if not available:
            raise ConversionError(
                f"Original stage {stage_id} has no selected complete variants"
            )
        return (
            inspection,
            tuple(
                catalog.resolve_stage(
                    entry,
                    tint,
                    extensions=extensions,
                )
                for tint in available
            ),
        )
    except ValueError as exc:
        raise ConversionError(str(exc)) from exc


def resolve_original_variants(
    rbr_root: Path,
    selector: str,
    requested: tuple[str, ...] = (),
) -> tuple[StageInspection, tuple[OriginalStageFiles, ...]]:
    return _resolve_original_variants(
        rbr_root,
        selector,
        requested,
        ORIGINAL_REQUIRED_EXTENSIONS,
        True,
    )


def resolve_original_pacenote_visualizer_variants(
    rbr_root: Path,
    selector: str,
    requested: tuple[str, ...] = (),
) -> tuple[StageInspection, tuple[OriginalStageFiles, ...]]:
    return _resolve_original_variants(
        rbr_root,
        selector,
        requested,
        ORIGINAL_PACENOTE_VISUALIZER_EXTENSIONS,
        False,
    )


def primary_original_tint(tints) -> str:
    available = {getattr(tint, "value", tint) for tint in tints}
    return next(tint for tint in PRIMARY_TINT_ORDER if tint in available)


def original_level_id(package_id: str, tint: str, primary_tint: str) -> str:
    if tint == primary_tint:
        return package_id
    return f"{package_id}_{slugify(ORIGINAL_TINT_NAMES.get(tint, tint))}"


def select_original_variants(
    variants: tuple[OriginalStageFiles, ...],
    requested: tuple[str, ...],
) -> tuple[tuple[OriginalStageFiles, ...], str]:
    primary = primary_original_tint(files.tint for files in variants)
    if not requested:
        variants = tuple(
            files for files in variants if files.tint.value == primary
        )
    return (
        tuple(sorted(variants, key=lambda files: files.tint.value != primary)),
        primary,
    )


def build_original_pacenote_stage(
    rbr_root: Path,
    inspection: StageInspection,
    files: OriginalStageFiles,
) -> RbrStage:
    from .adapter import build_original_stage
    from .dls import parse_dls
    from .lbs import parse_lbs_pacenote_inputs
    from .trk import parse_trk

    try:
        return build_original_stage(
            root=rbr_root,
            metadata=replace(
                inspection.metadata,
                folder_name=(
                    f"original_{files.entry.stage_id}_{files.tint.value.casefold()}"
                ),
            ),
            trk=parse_trk(files.files["physics"]),
            dls=parse_dls(files.files["driveline"]),
            lbs=parse_lbs_pacenote_inputs(files.files["track"]),
            materials=[],
            surfaces={},
            location=inspection.location,
            source_variant=files.tint.value,
            provenance=files.files,
        )
    except (OSError, ValueError) as exc:
        raise ConversionError(
            "Unable to read Original RBR pacenote inputs: " + str(exc)
        ) from exc

