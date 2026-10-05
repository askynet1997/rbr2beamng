from __future__ import annotations

import hashlib
import json
import uuid
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

from .core import ConversionCancelled, ConversionError, ProgressReporter
from .filesystem import current_filesystem
from .original.dls import parse_dls
from .original.source import (
    OriginalPacenoteSource,
    OriginalPacenoteSourceError,
    discover_original_pacenote_sources,
)
from .pacenotes import (
    KNOWN_FLAG_MASK,
    KNOWN_LOW_FLAG_BITS,
    LEGACY_DEFINITIONS,
    PLUGIN_CONTROL_CONVERSIONS,
    PLUGIN_CONTROL_SPECS,
    PLUGIN_FLAG_REGISTRY,
    UNSUPPORTED_FLAG_MODIFIERS,
    decode_plugin_control,
    known_flag_mask_for_note,
    load_pacenote_catalog_from_root,
    pacenote_config_header_indexes,
)
from .rbr import read_metadata, read_pacenotes


SCHEMA_VERSION = 2


@dataclass(frozen=True)
class PacenoteInventoryResult:
    output_path: Path
    source_count: int
    record_count: int
    error_count: int


class _ReportWriter:
    def __init__(self, directory: Path) -> None:
        filesystem = current_filesystem()
        self.directory = directory
        self._records = filesystem.open(
            directory / "records.jsonl",
            "w",
            encoding="utf-8",
        )
        self._sources = filesystem.open(
            directory / "sources.jsonl",
            "w",
            encoding="utf-8",
        )
        self._errors = filesystem.open(
            directory / "errors.jsonl",
            "w",
            encoding="utf-8",
        )

    @staticmethod
    def _write(stream, value: dict[str, object]) -> None:
        stream.write(
            json.dumps(
                value,
                ensure_ascii=False,
                separators=(",", ":"),
                sort_keys=True,
            )
            + "\n"
        )

    def record(self, value: dict[str, object]) -> None:
        self._write(self._records, value)

    def source(self, value: dict[str, object]) -> None:
        self._write(self._sources, value)

    def error(self, value: dict[str, object]) -> None:
        self._write(self._errors, value)

    def close(self) -> None:
        self._records.close()
        self._sources.close()
        self._errors.close()


class _Aggregator:
    def __init__(self) -> None:
        self.classifications: Counter[str] = Counter()
        self.control_codes: Counter[str] = Counter()
        self.flag_masks: Counter[str] = Counter()
        self.formats: Counter[str] = Counter()
        self.type_flags: Counter[tuple[int, int]] = Counter()
        self.type_flag_stages: dict[tuple[int, int], set[str]] = defaultdict(set)

    def add(self, record: dict[str, object]) -> None:
        raw = record["raw"]
        evidence = record["evidence"]
        assert isinstance(raw, dict)
        assert isinstance(evidence, dict)
        note_type = int(raw["type"])
        flag = int(raw["flag"])
        signature = (note_type, flag)
        self.type_flags[signature] += 1
        bindings = record["bindings"]
        assert isinstance(bindings, list)
        stages: set[str] = set()
        for binding in bindings:
            assert isinstance(binding, dict)
            if record["sourceFormat"] == "original":
                stages.add(
                    f"original:{binding['stageId']}:{binding['tint']}"
                )
            else:
                stages.add(f"rx:{binding['folder']}")
        if not stages:
            stages.add(str(record["stageKey"]))
        self.type_flag_stages[signature].update(stages)
        self.classifications[str(evidence["classification"])] += 1
        self.formats[str(record["sourceFormat"])] += 1
        control = evidence.get("control")
        if isinstance(control, dict):
            self.control_codes[str(control["codeHex"])] += 1
        unknown_mask = int(evidence["unknownFlagMask"])
        if unknown_mask:
            self.flag_masks[hex(unknown_mask)] += 1

    def summary(self) -> dict[str, object]:
        return {
            "classifications": dict(sorted(self.classifications.items())),
            "controlCodes": dict(sorted(self.control_codes.items())),
            "sourceFormats": dict(sorted(self.formats.items())),
            "typeFlags": [
                {
                    "count": count,
                    "flag": flag,
                    "flagHex": hex(flag),
                    "stageCount": len(self.type_flag_stages[(note_type, flag)]),
                    "stages": sorted(
                        self.type_flag_stages[(note_type, flag)]
                    ),
                    "type": note_type,
                    "typeHex": hex(note_type),
                }
                for (note_type, flag), count in sorted(self.type_flags.items())
            ],
            "unknownFlagMasks": dict(sorted(self.flag_masks.items())),
        }

def _relative(root: Path, path: Path) -> str:
    try:
        return path.relative_to(root).as_posix()
    except ValueError:
        return str(path)


def _fingerprint(path: Path, *, hash_inputs: bool) -> dict[str, object]:
    filesystem = current_filesystem()
    stat = filesystem.stat(path)
    result: dict[str, object] = {
        "mtimeNs": stat.st_mtime_ns,
        "size": stat.st_size,
    }
    if hash_inputs:
        digest = hashlib.sha256()
        with filesystem.open(path, "rb") as stream:
            while chunk := stream.read(1024 * 1024):
                digest.update(chunk)
        result["sha256"] = digest.hexdigest()
    return result


def _definition_evidence(definition) -> dict[str, object] | None:
    if definition is None:
        return None
    return {
        "configFile": definition.config_file,
        "cornerModifier": asdict(definition.corner_modifier)
        if definition.corner_modifier is not None
        else None,
        "custom": definition.custom,
        "modifier": definition.modifier,
        "name": definition.name,
        "spoken": definition.spoken,
    }


def _evidence(
    note_type: int,
    flag: int,
    index: int,
    config_header_indexes: frozenset[int],
    catalog,
) -> dict[str, object]:
    definition = catalog.definitions.get(note_type)
    control = decode_plugin_control(note_type, flag)
    if index in config_header_indexes:
        classification = "prestart-sentinel"
    elif control is not None:
        classification = (
            "confirmed-plugin-control"
            if control.kind != "UNKNOWN_PLUGIN_CONTROL"
            else "unconfirmed-plugin-control"
        )
    elif note_type in LEGACY_DEFINITIONS:
        classification = "legacy"
    elif definition is not None and definition.corner_modifier is not None:
        classification = "plugin-corner-modifier"
    elif definition is not None and definition.spoken:
        classification = "plugin-spoken"
    elif definition is not None:
        classification = "plugin-unspoken"
    else:
        classification = "opaque"
    known_low_bits = [
        hex(bit)
        for bit in sorted(KNOWN_LOW_FLAG_BITS)
        if flag & bit
    ]
    control_data = None
    if control is not None:
        control_data = {
            "code": control.code,
            "codeHex": hex(control.code),
            "kind": control.kind,
            "rawPayload": control.raw_payload,
            "scope": control.scope,
            "sticky": control.sticky,
            "unit": control.unit,
            "value": control.value,
        }
    return {
        "classification": classification,
        "configDefinition": _definition_evidence(definition),
        "control": control_data,
        "knownLowFlagBits": known_low_bits,
        "unknownFlagMask": flag & ~known_flag_mask_for_note(note_type, flag),
    }


def _record(
    *,
    source_format: str,
    stage_key: str,
    source_path: Path,
    source_path_relative: str,
    source_index: int,
    note_type: int,
    distance: float,
    flag: int,
    animation_set: str | None,
    bindings: list[dict[str, object]],
    catalog,
    config_header_indexes: frozenset[int],
) -> dict[str, object]:
    return {
        "animationSet": animation_set,
        "bindings": bindings,
        "evidence": _evidence(
            note_type,
            flag,
            source_index,
            config_header_indexes,
            catalog,
        ),
        "raw": {
            "distance": distance,
            "flag": flag,
            "flagHex": hex(flag),
            "type": note_type,
            "typeHex": hex(note_type),
        },
        "schemaVersion": SCHEMA_VERSION,
        "sourceFormat": source_format,
        "sourceIndex": source_index,
        "sourcePath": str(source_path),
        "sourcePathRelative": source_path_relative,
        "stageKey": stage_key,
    }


def _rx_sources(rbr_root: Path) -> list[Path]:
    filesystem = current_filesystem()
    tracks_root = rbr_root / "RX_CONTENT" / "TRACKS"
    if not filesystem.is_dir(tracks_root):
        return []
    return sorted(
        (
            path
            for path in filesystem.iterdir(tracks_root)
            if filesystem.is_dir(path)
            and filesystem.is_file(path / "pacenotes.ini")
        ),
        key=lambda path: path.name.casefold(),
    )


def _original_dls_paths(
    rbr_root: Path,
    bindings: tuple[OriginalPacenoteSource, ...],
) -> list[Path]:
    filesystem = current_filesystem()
    maps_root = rbr_root / "Maps"
    paths = {
        filesystem.read_path(binding.dls_path)
        for binding in bindings
    }
    if filesystem.is_dir(maps_root):
        paths.update(
            filesystem.read_path(path)
            for path in filesystem.iterdir(maps_root)
            if filesystem.is_file(path)
            and path.suffix.casefold() == ".dls"
        )
    return sorted(paths, key=lambda path: str(path).casefold())


def _write_json(path: Path, value: object) -> None:
    current_filesystem().write_text(
        path,
        json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_summary_markdown(path: Path, summary: dict[str, object]) -> None:
    classifications = summary["classifications"]
    controls = summary["controlCodes"]
    formats = summary["sourceFormats"]
    type_flags = summary["typeFlags"]
    assert isinstance(classifications, dict)
    assert isinstance(controls, dict)
    assert isinstance(formats, dict)
    assert isinstance(type_flags, list)
    lines = [
        "# Pacenote Corpus Inventory",
        "",
        "This is an evidence-only inventory. See `records.jsonl` for every raw occurrence.",
        "",
        "## Source formats",
        "",
    ]
    lines.extend(f"- `{name}`: {count}" for name, count in formats.items())
    lines.extend(("", "## Classifications", ""))
    lines.extend(
        f"- `{name}`: {count}" for name, count in classifications.items()
    )
    lines.extend(("", "## Control codes", ""))
    if controls:
        lines.extend(f"- `{name}`: {count}" for name, count in controls.items())
    else:
        lines.append("- None")
    lines.extend(("", "## Exact type and flag signatures", ""))
    for item in type_flags:
        assert isinstance(item, dict)
        lines.append(
            f"- `{item['typeHex']}` / `{item['flagHex']}`: "
            f"{item['count']} records in {item['stageCount']} stages"
        )
    current_filesystem().write_text(
        path,
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )


def _catalog_report(catalog) -> dict[str, object]:
    return {
        "configPath": str(catalog.config_path) if catalog.config_path else None,
        "definitions": [
            {
                "definition": asdict(definition),
                "id": note_id,
            }
            for note_id, definition in sorted(catalog.definitions.items())
        ],
        "issues": list(catalog.issues),
        "knownFlagMask": KNOWN_FLAG_MASK,
        "knownLowFlagBits": [hex(bit) for bit in sorted(KNOWN_LOW_FLAG_BITS)],
        "pluginControlSpecs": {
            hex(code): {
                "kind": kind,
                "scope": scope,
                "unit": unit,
                "usesSignedFixedPointPayload": uses_fixed_point_payload,
                "usesSignedIntegerPayload": uses_integer_payload,
                "conversion": (
                    {
                        "kind": PLUGIN_CONTROL_CONVERSIONS[code][0],
                        "value": PLUGIN_CONTROL_CONVERSIONS[code][1],
                    }
                    if code in PLUGIN_CONTROL_CONVERSIONS
                    else None
                ),
            }
            for (
                code,
                (
                    kind,
                    unit,
                    scope,
                    uses_fixed_point_payload,
                    uses_integer_payload,
                ),
            ) in sorted(
                PLUGIN_CONTROL_SPECS.items()
            )
        },
        "pluginFlagRegistry": {
            hex(bit): name
            for bit, name in sorted(PLUGIN_FLAG_REGISTRY.items())
        },
        "schemaVersion": SCHEMA_VERSION,
        "unsupportedFlagModifiers": {
            hex(bit): modifier
            for bit, modifier in sorted(UNSUPPORTED_FLAG_MODIFIERS.items())
        },
    }


def _binding_record(binding: OriginalPacenoteSource) -> dict[str, object]:
    return {
        "stageId": binding.stage_id,
        "stageName": binding.stage_name,
        "tint": binding.tint,
    }


def _error_record(
    *,
    source_format: str,
    stage_key: str,
    source_path: Path | None,
    message: str,
) -> dict[str, object]:
    return {
        "message": message,
        "schemaVersion": SCHEMA_VERSION,
        "sourceFormat": source_format,
        "sourcePath": str(source_path) if source_path else None,
        "stageKey": stage_key,
    }


def build_pacenote_inventory(
    rbr_root: Path,
    output: Path,
    *,
    reporter: ProgressReporter | None = None,
    hash_inputs: bool = False,
) -> PacenoteInventoryResult:
    filesystem = current_filesystem()
    root = filesystem.read_path(rbr_root)
    output = filesystem.write_path(output)
    if filesystem.exists(output):
        raise ConversionError(f"Inventory output already exists: {output}")
    filesystem.mkdir(output.parent, parents=True, exist_ok=True)
    staging = filesystem.write_path(
        output.parent / f".{output.name}.partial-{uuid.uuid4().hex}"
    )
    filesystem.mkdir(staging)
    writer = _ReportWriter(staging)
    aggregate = _Aggregator()
    record_count = 0
    source_count = 0
    error_count = 0
    complete = False
    try:
        catalog = load_pacenote_catalog_from_root(root, apply_overrides=False)
        original_bindings, original_binding_errors = (
            discover_original_pacenote_sources(root)
        )
        bindings_by_path: dict[Path, list[dict[str, object]]] = defaultdict(list)
        for binding in original_bindings:
            bindings_by_path[filesystem.read_path(binding.dls_path)].append(
                _binding_record(binding)
            )
        for error in original_binding_errors:
            writer.error(
                _error_record(
                    source_format="original",
                    stage_key=f"original:{error.stage_id}:{error.tint}",
                    source_path=None,
                    message=error.message,
                )
            )
            error_count += 1
        rx_sources = _rx_sources(root)
        dls_sources = _original_dls_paths(root, original_bindings)
        total_sources = len(rx_sources) + len(dls_sources)
        current = 0
        for stage_root in rx_sources:
            current += 1
            source_count += 1
            if reporter:
                reporter.emit(
                    "pacenote-inventory",
                    "Scanning RX pacenotes",
                    current=current,
                    total=total_sources,
                    detail=stage_root.name,
                )
            source_path = filesystem.read_path(stage_root / "pacenotes.ini")
            stage_key = f"rx:{stage_root.name}"
            try:
                notes = read_pacenotes(stage_root)
                try:
                    metadata = read_metadata(stage_root)
                    stage_name = metadata.name
                except (ConversionError, OSError, ValueError):
                    stage_name = stage_root.name
                config_header_indexes = pacenote_config_header_indexes(
                    [
                        (note.note_type, note.distance, note.flag)
                        for note in notes
                    ]
                )
                bindings = [
                    {
                        "folder": stage_root.name,
                        "name": stage_name,
                        "sourceFormat": "rx",
                    }
                ]
                for source_index, note in enumerate(notes):
                    record = _record(
                        source_format="rx",
                        stage_key=stage_key,
                        source_path=source_path,
                        source_path_relative=_relative(root, source_path),
                        source_index=source_index,
                        note_type=note.note_type,
                        distance=note.distance,
                        flag=note.flag,
                        animation_set=None,
                        bindings=bindings,
                        catalog=catalog,
                        config_header_indexes=config_header_indexes,
                    )
                    writer.record(record)
                    aggregate.add(record)
                    record_count += 1
                writer.source(
                    {
                        "bindings": bindings,
                        "fingerprint": _fingerprint(
                            source_path,
                            hash_inputs=hash_inputs,
                        ),
                        "recordCount": len(notes),
                        "schemaVersion": SCHEMA_VERSION,
                        "sourceFormat": "rx",
                        "sourcePath": str(source_path),
                        "sourcePathRelative": _relative(root, source_path),
                        "stageKey": stage_key,
                    }
                )
            except ConversionCancelled:
                raise
            except (ConversionError, OSError, ValueError) as exc:
                writer.source(
                    {
                        "bindings": [
                            {
                                "folder": stage_root.name,
                                "sourceFormat": "rx",
                            }
                        ],
                        "error": str(exc),
                        "fingerprint": _fingerprint(
                            source_path,
                            hash_inputs=hash_inputs,
                        ),
                        "recordCount": 0,
                        "schemaVersion": SCHEMA_VERSION,
                        "sourceFormat": "rx",
                        "sourcePath": str(source_path),
                        "sourcePathRelative": _relative(root, source_path),
                        "stageKey": stage_key,
                    }
                )
                writer.error(
                    _error_record(
                        source_format="rx",
                        stage_key=stage_key,
                        source_path=source_path,
                        message=str(exc),
                    )
                )
                error_count += 1
        for dls_path in dls_sources:
            current += 1
            source_count += 1
            if reporter:
                reporter.emit(
                    "pacenote-inventory",
                    "Scanning Original RBR pacenotes",
                    current=current,
                    total=total_sources,
                    detail=dls_path.name,
                )
            bindings = sorted(
                bindings_by_path.get(dls_path, []),
                key=lambda item: (
                    int(item["stageId"]),
                    str(item["tint"]),
                ),
            )
            stage_key = (
                f"original:{bindings[0]['stageId']}"
                if bindings
                else f"original-file:{dls_path.stem}"
            )
            try:
                dls = parse_dls(dls_path)
                notes = dls.all_pacenotes
                notes_by_animation_set: dict[
                    str,
                    list[tuple[int, OriginalPacenote]],
                ] = defaultdict(list)
                for source_index, note in enumerate(notes):
                    notes_by_animation_set[note.animation_set].append(
                        (source_index, note)
                    )
                config_header_indexes = frozenset(
                    source_index
                    for animation_notes in notes_by_animation_set.values()
                    for local_index in pacenote_config_header_indexes(
                        [
                            (note.note_id, note.distance, note.flags)
                            for _source_index, note in animation_notes
                        ]
                    )
                    for source_index, _note in [animation_notes[local_index]]
                )
                for source_index, note in enumerate(notes):
                    record = _record(
                        source_format="original",
                        stage_key=stage_key,
                        source_path=dls_path,
                        source_path_relative=_relative(root, dls_path),
                        source_index=source_index,
                        note_type=note.note_id,
                        distance=note.distance,
                        flag=note.flags,
                        animation_set=note.animation_set,
                        bindings=bindings,
                        catalog=catalog,
                        config_header_indexes=config_header_indexes,
                    )
                    writer.record(record)
                    aggregate.add(record)
                    record_count += 1
                writer.source(
                    {
                        "bindings": bindings,
                        "fingerprint": _fingerprint(
                            dls_path,
                            hash_inputs=hash_inputs,
                        ),
                        "recordCount": len(notes),
                        "schemaVersion": SCHEMA_VERSION,
                        "sourceFormat": "original",
                        "sourcePath": str(dls_path),
                        "sourcePathRelative": _relative(root, dls_path),
                        "stageKey": stage_key,
                    }
                )
            except ConversionCancelled:
                raise
            except (ConversionError, OSError, ValueError) as exc:
                writer.source(
                    {
                        "bindings": bindings,
                        "error": str(exc),
                        "fingerprint": _fingerprint(
                            dls_path,
                            hash_inputs=hash_inputs,
                        ),
                        "recordCount": 0,
                        "schemaVersion": SCHEMA_VERSION,
                        "sourceFormat": "original",
                        "sourcePath": str(dls_path),
                        "sourcePathRelative": _relative(root, dls_path),
                        "stageKey": stage_key,
                    }
                )
                writer.error(
                    _error_record(
                        source_format="original",
                        stage_key=stage_key,
                        source_path=dls_path,
                        message=str(exc),
                    )
                )
                error_count += 1
        summary = {
            "errorCount": error_count,
            "recordCount": record_count,
            "schemaVersion": SCHEMA_VERSION,
            "sourceCount": source_count,
            **aggregate.summary(),
        }
        _write_json(staging / "catalog.json", _catalog_report(catalog))
        _write_json(staging / "summary.json", summary)
        _write_summary_markdown(staging / "summary.md", summary)
        complete = True
        _write_json(
            staging / "manifest.json",
            {
                "catalogIssues": list(catalog.issues),
                "complete": True,
                "generatedAt": datetime.now(timezone.utc).isoformat(),
                "hashInputs": hash_inputs,
                "inputRoot": str(root),
                "reports": [
                    "catalog.json",
                    "errors.jsonl",
                    "records.jsonl",
                    "sources.jsonl",
                    "summary.json",
                    "summary.md",
                ],
                "schemaVersion": SCHEMA_VERSION,
                "summary": {
                    "errorCount": error_count,
                    "recordCount": record_count,
                    "sourceCount": source_count,
                },
            },
        )
    except BaseException:
        _write_json(
            staging / "manifest.json",
            {
                "complete": False,
                "generatedAt": datetime.now(timezone.utc).isoformat(),
                "hashInputs": hash_inputs,
                "inputRoot": str(root),
                "schemaVersion": SCHEMA_VERSION,
                "summary": {
                    "errorCount": error_count,
                    "recordCount": record_count,
                    "sourceCount": source_count,
                },
            },
        )
        raise
    finally:
        writer.close()
    if not complete:
        raise ConversionError(f"Pacenote inventory did not complete: {staging}")
    filesystem.replace(staging, output)
    return PacenoteInventoryResult(
        output,
        source_count,
        record_count,
        error_count,
    )
