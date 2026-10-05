from __future__ import annotations

import json
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from . import __version__
from .beamng import evaluate_stage_pacenotes
from .core import ConversionCancelled, ConversionError, ProgressReporter
from .filesystem import current_filesystem
from .geometry import source_position_to_beamng
from .models import RbrStage, StageInspection
from .original.catalog import OriginalStageFiles
from .original.dls import parse_dls
from .original.source import (
    build_original_pacenote_stage,
    resolve_original_pacenote_visualizer_variants,
)
from .pacenotes import (
    MODIFIER_SLOTS_FULL_PREFIX,
    MODIFIER_SLOTS_FULL_SEPARATOR,
    NO_MODIFIER_CORNER_REASON,
    NO_MODIFIER_PACENOTE_REASON,
    UNSUPPORTED_BEAMNG_MODIFIER_PREFIX,
    RallyNotebookResult,
)
from .rbr import load_pacenote_visualizer_stage, read_pacenotes
from .stage_sources import discover_stages


_EXPECTED_DISPOSITIONS = {
    "config-header",
    "distance-marker-recorded",
    "distance-marker-used",
    "outside-rally-range",
    "split-marker",
    "start-marker",
    "stop-marker",
}
_ATTACHED_DISPOSITIONS = {
    "attached-corner-modifier",
    "attached-fixed-corner-modifier",
    "attached-plugin-control-modifier",
    "preserved-plugin-control",
    "preserved-start-distance-marker",
}


@dataclass(frozen=True)
class PacenoteAuditResult:
    output_path: Path
    stage_count: int
    completed_stage_count: int
    error_count: int
    problem_count: int


@dataclass(frozen=True)
class _AuditTarget:
    inspection: StageInspection
    files: OriginalStageFiles | None = None

    @property
    def source_format(self) -> str:
        return self.inspection.source_format

    @property
    def variant(self) -> str:
        return self.files.tint.value if self.files is not None else "base"

    @property
    def detail(self) -> str:
        if self.files is None:
            return self.inspection.metadata.folder_name
        return f"{self.inspection.metadata.folder_name} ({self.variant})"

    @property
    def selector(self) -> str:
        if self.files is None:
            return self.inspection.metadata.folder_name
        return self.inspection.source_key


def _json_for_script(value: object) -> str:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def _string_values(value: object) -> list[str]:
    if not isinstance(value, (list, tuple)):
        return []
    return [str(item) for item in value]


def _deduplicated(values: list[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))


def _record_outcome(record: dict[str, object]) -> str:
    if _string_values(record.get("reasons")) or _string_values(record.get("issues")):
        return "needs-attention"
    disposition = str(record.get("disposition", "recorded"))
    if disposition == "generated":
        return "generated"
    if disposition.startswith("generated-with"):
        return "generated-with-qualification"
    if disposition in _ATTACHED_DISPOSITIONS:
        return "attached-or-preserved"
    if disposition in _EXPECTED_DISPOSITIONS:
        return "expected-metadata"
    return "recorded"


def _source_call_name(record: dict[str, object]) -> str:
    return str(record.get("label", "source call")).split(" — ", 1)[0].strip()


def _quoted_term(value: str) -> str:
    return json.dumps(value, ensure_ascii=False)


def _loss_actions(record: dict[str, object]) -> list[dict[str, object]]:
    actions: list[dict[str, object]] = []
    seen: set[str] = set()

    def add(
        key: str,
        title: str,
        detail: str,
        *,
        beamng_actionable: bool = True,
    ) -> None:
        if key in seen:
            return
        seen.add(key)
        actions.append(
            {
                "beamngActionable": beamng_actionable,
                "detail": detail,
                "key": key,
                "title": title,
            }
        )

    def add_no_modifier_target() -> None:
        add(
            "modifier:no-target",
            "RBR modifier: no call to modify",
            "No converted call sits between the surrounding distance calls and "
            "link words; attaching it elsewhere would be a guess.",
            beamng_actionable=False,
        )

    quoted_source_call = _quoted_term(_source_call_name(record))
    for issue in _string_values(record.get("issues")):
        if issue in (NO_MODIFIER_CORNER_REASON, NO_MODIFIER_PACENOTE_REASON):
            add_no_modifier_target()
            continue
        if issue.startswith(MODIFIER_SLOTS_FULL_PREFIX):
            slot_kind, separator, _modifier = issue.removeprefix(
                MODIFIER_SLOTS_FULL_PREFIX
            ).partition(MODIFIER_SLOTS_FULL_SEPARATOR)
            if separator:
                add(
                    f"modifier-slots-full:{slot_kind}",
                    f"BeamNG {slot_kind} modifier slots are full",
                    f"The generated BeamNG pacenote already uses both {slot_kind} modifier slots. "
                    "Adding a mapping will not fix this; use a selection or metadata-preservation policy.",
                )
                continue
        if issue.startswith(UNSUPPORTED_BEAMNG_MODIFIER_PREFIX):
            modifier = issue.removeprefix(UNSUPPORTED_BEAMNG_MODIFIER_PREFIX)
            add(
                f"semantic:modifier:{modifier}",
                f"Define BeamNG conversion for modifier: {_quoted_term(modifier.upper())}",
                "Requires a BeamNG Rally Mode semantic, not only a mapping.",
            )
            continue
        unknown_flag_prefix = "unknown RBR flag bits "
        if issue.startswith(unknown_flag_prefix):
            bits = issue.removeprefix(unknown_flag_prefix)
            add(
                f"flag:{bits}",
                f"Unknown RBR flag: {bits}",
                "Requires reverse-engineering evidence before conversion logic changes.",
                beamng_actionable=False,
            )
            continue
        add(
            f"issue:{issue}",
            f"Converter issue: {issue}",
            "Needs a converter or source-format decision.",
        )
    for reason in _string_values(record.get("reasons")):
        if reason == "no BeamNG equivalent":
            add(
                f"semantic:call:{record['type']}",
                f"Define BeamNG conversion for call: {quoted_source_call}",
                "Requires a BeamNG Rally Mode semantic, not only a mapping.",
            )
            continue
        if reason == "no matching definition":
            add(
                f"definition:{record['type']}",
                f"Unknown RBR call: {record['typeHex']}",
                "Needs catalog or source-format evidence before mapping.",
                beamng_actionable=False,
            )
            continue
        if reason == "no BeamNG mapping":
            add(
                f"mapping:{record['type']}",
                f"Define BeamNG conversion for call: {quoted_source_call}",
                "The source call is known but has no verified BeamNG mapping.",
            )
            continue
        unknown_prefix = "unknown RBR pacenote flag bits "
        if reason.startswith(unknown_prefix):
            bits = reason.removeprefix(unknown_prefix)
            add(
                f"flag:{bits}",
                f"Unknown RBR flag: {bits}",
                "Requires reverse-engineering evidence before conversion logic changes.",
                beamng_actionable=False,
            )
            continue
        if reason == "preceding content pacenote was not converted":
            continue
        if reason == "no preceding content pacenote":
            add(
                "distance-marker:no-preceding-content",
                "RBR distance marker: no preceding content",
                "This may be source placement rather than a converter defect.",
                beamng_actionable=False,
            )
            continue
        if reason == "outside the preceding pacenote range":
            add(
                "distance-marker:outside-preceding-range",
                "RBR distance marker: outside preceding range",
                "This may be source placement rather than a converter defect.",
            )
            continue
        if reason == NO_MODIFIER_CORNER_REASON:
            add_no_modifier_target()
            continue
        if reason.startswith("no following BeamNG pacenote"):
            add(
                "plugin-control:no-next-target",
                "RBR control: no following target",
                "A metadata fallback can retain the source control without inventing timing.",
            )
            continue
        add(
            f"reason:{reason}",
            f"Conversion reason: {reason}",
            "Needs a converter or source-format decision.",
        )
    return actions


def _stage_failure(
    inspection: StageInspection,
    *,
    message: str,
    variant: str = "",
) -> dict[str, object]:
    return {
        "message": message,
        "sourceFormat": inspection.source_format,
        "stageKey": inspection.source_key,
        "stageName": inspection.metadata.name,
        "stageSelector": (
            inspection.source_key
            if inspection.source_format == "original"
            else inspection.metadata.folder_name
        ),
        "variant": variant,
    }


def _audit_targets(rbr_root: Path) -> tuple[list[_AuditTarget], list[dict[str, object]]]:
    targets: list[_AuditTarget] = []
    failures: list[dict[str, object]] = []
    for inspection in discover_stages(rbr_root, inspect_original_variants=False):
        if inspection.source_format != "original":
            targets.append(_AuditTarget(inspection))
            continue
        try:
            resolved, variants = resolve_original_pacenote_visualizer_variants(
                rbr_root,
                inspection.source_key,
            )
        except ConversionCancelled:
            raise
        except (ConversionError, OSError, ValueError) as exc:
            failures.append(_stage_failure(inspection, message=str(exc)))
            continue
        targets.extend(_AuditTarget(resolved, files) for files in variants)
    return targets, failures


def _stage_source_records(
    rally: RallyNotebookResult,
) -> tuple[list[dict[str, object]], Counter[str]]:
    beamng_by_entry_index = {
        int(record["entryIndex"]): record
        for record in rally.visualizer_records
        if record.get("mode") == "beamng"
    }
    outcome_counts: Counter[str] = Counter()
    records: list[dict[str, object]] = []
    for record in rally.visualizer_records:
        if record.get("mode") != "rbr":
            continue
        entry_indexes = [
            int(index) for index in _string_values(record.get("beamngEntryIndexes"))
        ]
        outputs = [
            {
                "entryIndex": entry_index,
                "label": str(beamng_by_entry_index[entry_index].get("label", "")),
            }
            for entry_index in entry_indexes
            if entry_index in beamng_by_entry_index
        ]
        item = {
            "beamngEntryIndexes": entry_indexes,
            "distance": float(record["distance"]),
            "disposition": str(record["disposition"]),
            "flag": int(record["flag"]),
            "ignored": bool(record["ignored"]),
            "issues": _string_values(record.get("issues")),
            "label": str(record["label"]),
            "outputs": outputs,
            "reasons": _string_values(record.get("reasons")),
            "sourceIndex": int(record["sourceIndex"]),
            "targetSourceIndexes": [
                int(index)
                for index in _string_values(record.get("targetSourceIndexes"))
            ],
            "type": int(record["type"]),
            "typeHex": str(record["typeHex"]),
        }
        item["outcome"] = _record_outcome(item)
        item["lossActions"] = (
            _loss_actions(item)
            if item["outcome"] == "needs-attention"
            else []
        )
        outcome_counts[str(item["outcome"])] += 1
        records.append(item)
    return records, outcome_counts


def _stage_record(
    target: _AuditTarget,
    stage: RbrStage,
    rally: RallyNotebookResult,
    route: list[tuple[object, object, float]],
) -> dict[str, object]:
    records, outcome_counts = _stage_source_records(rally)
    problems = [
        record
        for record in records
        if record["outcome"] == "needs-attention"
    ]
    warnings = _deduplicated(
        [
            *[str(warning) for warning in target.inspection.issues],
            *[str(warning) for warning in stage.warnings],
            *[str(warning) for warning in rally.warnings],
        ]
    )
    route_length = (
        max(0.0, float(route[-1][2]) - float(route[0][2]))
        if len(route) >= 2
        else 0.0
    )
    return {
        "id": f"{target.inspection.source_key}:{target.variant}",
        "outcomes": dict(sorted(outcome_counts.items())),
        "problemCount": len(problems),
        "records": records,
        "routeLengthMeters": route_length,
        "sourceFormat": target.source_format,
        "stageKey": target.inspection.source_key,
        "stageName": stage.metadata.name,
        "stageSelector": target.selector,
        "stats": dict(rally.stats),
        "status": "loss" if problems else "warning" if warnings else "clean",
        "variant": target.variant,
        "warnings": warnings,
    }


def _evaluate_target(target: _AuditTarget, rbr_root: Path) -> dict[str, object]:
    if target.files is None:
        stage = load_pacenote_visualizer_stage(target.inspection.root)
    else:
        stage = build_original_pacenote_stage(
            rbr_root,
            target.inspection,
            target.files,
        )
    route, _route_range, rally = evaluate_stage_pacenotes(
        stage,
        source_position_to_beamng(stage.spawn.position),
    )
    return _stage_record(target, stage, rally, route)


def _raw_source_failure(
    source_format: str,
    source_path: Path,
    message: str,
) -> dict[str, object]:
    return {
        "message": message,
        "sourceFormat": source_format,
        "sourcePath": str(source_path),
        "stageKey": "",
        "stageName": "",
        "stageSelector": "",
        "variant": "",
    }


def _raw_source_evidence(
    rbr_root: Path,
    targets: list[_AuditTarget],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    filesystem = current_filesystem()
    rx_bindings: dict[Path, list[str]] = defaultdict(list)
    original_bindings: dict[Path, list[str]] = defaultdict(list)
    for target in targets:
        if target.files is None:
            rx_bindings[target.inspection.root / "pacenotes.ini"].append(
                target.inspection.source_key
            )
            continue
        original_bindings[target.files.files["driveline"]].append(
            f"{target.inspection.source_key}:{target.variant}"
        )

    sources: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    tracks_root = rbr_root / "RX_CONTENT" / "TRACKS"
    if filesystem.is_dir(tracks_root):
        for stage_root in sorted(
            (
                entry
                for entry in filesystem.iterdir(tracks_root)
                if filesystem.is_dir(entry)
                and filesystem.is_file(entry / "pacenotes.ini")
            ),
            key=lambda path: path.name.casefold(),
        ):
            source_path = stage_root / "pacenotes.ini"
            try:
                notes = read_pacenotes(stage_root)
                sources.append(
                    {
                        "bindings": sorted(rx_bindings.get(source_path, [])),
                        "path": str(source_path),
                        "playableRecordCount": len(notes),
                        "rawOnlyRecordCount": 0,
                        "recordCount": len(notes),
                        "sourceFormat": "rx",
                        "unbound": source_path not in rx_bindings,
                    }
                )
            except ConversionCancelled:
                raise
            except (ConversionError, OSError, ValueError) as exc:
                failures.append(_raw_source_failure("rx", source_path, str(exc)))

    dls_paths = set(original_bindings)
    maps_root = rbr_root / "Maps"
    if filesystem.is_dir(maps_root):
        dls_paths.update(
            path
            for path in filesystem.iterdir(maps_root)
            if filesystem.is_file(path) and path.suffix.casefold() == ".dls"
        )
    for dls_path in sorted(dls_paths, key=lambda path: str(path).casefold()):
        try:
            dls = parse_dls(dls_path)
            raw_notes = tuple(getattr(dls, "all_pacenotes", dls.pacenotes))
            playable_notes = tuple(dls.pacenotes)
            animation_sets = Counter(
                str(getattr(note, "animation_set", "unknown"))
                for note in raw_notes
            )
            sources.append(
                {
                    "animationSets": dict(sorted(animation_sets.items())),
                    "bindings": sorted(original_bindings.get(dls_path, [])),
                    "path": str(dls_path),
                    "playableRecordCount": len(playable_notes),
                    "rawOnlyRecordCount": max(0, len(raw_notes) - len(playable_notes)),
                    "recordCount": len(raw_notes),
                    "sourceFormat": "original",
                    "unbound": dls_path not in original_bindings,
                }
            )
        except ConversionCancelled:
            raise
        except (ConversionError, OSError, ValueError) as exc:
            failures.append(_raw_source_failure("original", dls_path, str(exc)))
    return sources, failures


def _fix_opportunities(
    stages: list[dict[str, object]],
) -> list[dict[str, object]]:
    opportunities: dict[str, dict[str, object]] = {}
    for stage in stages:
        records = stage["records"]
        assert isinstance(records, list)
        for record in records:
            assert isinstance(record, dict)
            for action in record.get("lossActions", []):
                if not isinstance(action, dict):
                    continue
                key = str(action.get("key", ""))
                if not key:
                    continue
                opportunity = opportunities.setdefault(
                    key,
                    {
                        "count": 0,
                        "detail": str(action.get("detail", "")),
                        "examples": [],
                        "key": key,
                        "stages": set(),
                        "title": str(action.get("title", key)),
                    },
                )
                opportunity["count"] = int(opportunity["count"]) + 1
                opportunity["stages"].add(str(stage["id"]))
                examples = opportunity["examples"]
                assert isinstance(examples, list)
                if len(examples) < 5:
                    examples.append(
                        {
                            "label": record["label"],
                            "sourceIndex": record["sourceIndex"],
                            "stage": stage["stageName"],
                            "stageId": stage["id"],
                            "variant": stage["variant"],
                        }
                    )
    result = []
    for opportunity in opportunities.values():
        stages_for_opportunity = opportunity.pop("stages")
        assert isinstance(stages_for_opportunity, set)
        opportunity["stageCount"] = len(stages_for_opportunity)
        result.append(opportunity)
    return sorted(
        result,
        key=lambda item: (
            -int(item["count"]),
            -int(item["stageCount"]),
            str(item["title"]),
        ),
    )


def _summary(
    stages: list[dict[str, object]],
    failures: list[dict[str, object]],
    raw_sources: list[dict[str, object]],
) -> dict[str, object]:
    totals: Counter[str] = Counter()
    outcomes: Counter[str] = Counter()
    status_counts: Counter[str] = Counter()
    for stage in stages:
        status_counts[str(stage["status"])] += 1
        outcomes.update(stage["outcomes"])
        stats = stage["stats"]
        assert isinstance(stats, dict)
        for key, value in stats.items():
            if isinstance(value, int):
                totals[str(key)] += value
    raw_totals = Counter(
        {
            "playableRecordCount": 0,
            "rawOnlyRecordCount": 0,
            "recordCount": 0,
        }
    )
    raw_formats = Counter()
    unbound_count = 0
    for source in raw_sources:
        raw_totals["recordCount"] += int(source["recordCount"])
        raw_totals["playableRecordCount"] += int(source["playableRecordCount"])
        raw_totals["rawOnlyRecordCount"] += int(source["rawOnlyRecordCount"])
        raw_formats[str(source["sourceFormat"])] += 1
        unbound_count += int(bool(source["unbound"]))
    return {
        "completedStageCount": len(stages),
        "conversionStats": dict(sorted(totals.items())),
        "errorCount": len(failures),
        "outcomes": dict(sorted(outcomes.items())),
        "problemCount": outcomes["needs-attention"],
        "raw": {
            **dict(sorted(raw_totals.items())),
            "sourceCount": len(raw_sources),
            "sourceFormats": dict(sorted(raw_formats.items())),
            "unboundSourceCount": unbound_count,
        },
        "stageStatuses": dict(sorted(status_counts.items())),
        "warningCount": sum(
            len(_string_values(stage["warnings"])) for stage in stages
        ),
    }


def _report_payload(
    rbr_root: Path,
    stages: list[dict[str, object]],
    failures: list[dict[str, object]],
    raw_sources: list[dict[str, object]],
) -> dict[str, object]:
    return {
        "fixOpportunities": _fix_opportunities(stages),
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "inputRoot": str(rbr_root),
        "schemaVersion": 2,
        "sourceFailures": failures,
        "stages": stages,
        "summary": _summary(stages, failures, raw_sources),
    }


def render_pacenote_audit_report(data: dict[str, object]) -> str:
    return _HTML_TEMPLATE.replace("__RBR2BEAMNG_VERSION__", __version__).replace(
        "__PACENOTE_AUDIT_DATA__",
        _json_for_script(data),
    )


def build_pacenote_audit(
    rbr_root: Path,
    output: Path,
    *,
    reporter: ProgressReporter | None = None,
) -> PacenoteAuditResult:
    filesystem = current_filesystem()
    root = filesystem.read_path(rbr_root)
    output = filesystem.write_path(output)
    if filesystem.exists(output):
        raise ConversionError(f"Pacenote audit output already exists: {output}")
    filesystem.mkdir(output.parent, parents=True, exist_ok=True)
    targets, failures = _audit_targets(root)
    stages: list[dict[str, object]] = []
    for index, target in enumerate(targets, 1):
        if reporter:
            reporter.emit(
                "pacenote-audit",
                "Converting pacenotes",
                current=index,
                total=len(targets),
                detail=target.detail,
            )
        try:
            stages.append(_evaluate_target(target, root))
        except ConversionCancelled:
            raise
        except (ConversionError, OSError, ValueError) as exc:
            failures.append(
                _stage_failure(
                    target.inspection,
                    message=str(exc),
                    variant=target.variant,
                )
            )
    raw_sources, raw_failures = _raw_source_evidence(root, targets)
    failures.extend(raw_failures)
    payload = _report_payload(root, stages, failures, raw_sources)
    html = render_pacenote_audit_report(payload)
    temporary = output.parent / f".{output.name}.partial-{uuid.uuid4().hex}"
    try:
        filesystem.write_text(temporary, html, encoding="utf-8")
        filesystem.replace(temporary, output)
    except BaseException:
        filesystem.unlink(temporary, missing_ok=True)
        raise
    summary = payload["summary"]
    assert isinstance(summary, dict)
    return PacenoteAuditResult(
        output_path=output,
        stage_count=len(targets),
        completed_stage_count=len(stages),
        error_count=len(failures),
        problem_count=int(summary["problemCount"]),
    )


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pacenote Conversion Audit - RBR2BeamNG v__RBR2BEAMNG_VERSION__</title>
<style>
:root {
  color-scheme: dark;
  font-family: "Segoe UI", system-ui, sans-serif;
  background: #101720;
  color: #e9f0f6;
}
* { box-sizing: border-box; }
body { margin: 0; min-width: 320px; }
button, input, select { font: inherit; }
button {
  border: 1px solid #4c6278;
  border-radius: 6px;
  background: #213449;
  color: inherit;
  cursor: pointer;
  padding: 0.38rem 0.55rem;
}
button:hover, button:focus-visible { background: #2e4a65; outline: 2px solid #8ac5ff; outline-offset: 2px; }
input, select {
  border: 1px solid #526a80;
  border-radius: 6px;
  background: #172536;
  color: inherit;
  min-height: 2.2rem;
  padding: 0.35rem 0.5rem;
}
header {
  align-items: baseline;
  border-bottom: 1px solid #30465d;
  display: flex;
  flex-wrap: wrap;
  gap: 0.6rem;
  padding: 1.25rem clamp(1rem, 3vw, 3rem);
  background: linear-gradient(120deg, #14283d, #101720);
}
.header-title { align-items: baseline; display: flex; flex-wrap: wrap; gap: 0.6rem; }
h1 { margin: 0; font-size: clamp(1.35rem, 3vw, 2rem); }
header .muted { margin: 0; }
h2 { margin: 0 0 0.65rem; font-size: 1.1rem; }
h3 { margin: 0.8rem 0 0.35rem; font-size: 0.96rem; }
p { margin: 0.35rem 0; }
.muted { color: #aab8c5; }
.issue-filters {
  align-items: center;
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  margin-left: auto;
}
.issue-filter {
  align-items: center;
  display: flex;
  flex: 0 0 auto;
  font-size: 0.8rem;
  gap: 0.35rem;
}
.issue-filter input { min-height: auto; }
.issue-filter.converter { color: #8ac5ff; }
.issue-filter.beamng { color: #efad70; }
.layout { max-width: 1800px; margin: 0 auto; padding: 1rem clamp(1rem, 3vw, 3rem) 2rem; }
.controls {
  align-items: end;
  display: flex;
  flex-wrap: wrap;
  gap: 0.75rem;
  margin: 1.25rem 0 0.75rem;
}
.control { display: grid; gap: 0.25rem; min-width: 180px; }
.control label { color: #aab8c5; font-size: 0.8rem; }
.panel {
  border: 1px solid #30465d;
  border-radius: 9px;
  background: #142131;
  min-width: 0;
  overflow: hidden;
  padding: 0.85rem;
}
.triage-grid {
  align-items: start;
  display: grid;
  gap: 1rem;
  grid-template-columns: minmax(0, 2fr) minmax(0, 1fr);
  margin: 1rem 0;
}
.ranking-list { margin-top: 0.45rem; }
.ranking-row {
  background-color: #172536;
  border: 0;
  border-bottom: 1px solid #2d4358;
  color: inherit;
  display: block;
  font: inherit;
  overflow: hidden;
  padding: 0;
  position: relative;
  text-align: left;
  -webkit-user-select: text;
  user-select: text;
  width: 100%;
}
.ranking-bar {
  display: flex;
  inset: 0 auto 0 0;
  pointer-events: none;
  position: absolute;
  transition: width 120ms ease-out;
  width: var(--ranking-width, 0%);
}
.ranking-bar-segment { height: 100%; transition: width 120ms ease-out; }
.ranking-bar-segment.beamng { background: #a96227; }
.ranking-bar-segment.converter { background: #315a77; }
.ranking-content {
  align-items: center;
  display: flex;
  gap: 0.6rem;
  justify-content: space-between;
  min-width: 0;
  padding: 0.38rem 0.1rem;
  position: relative;
  width: 100%;
  z-index: 1;
}
.ranking-actions {
  align-items: center;
  display: flex;
  flex: 0 0 auto;
  gap: 0.45rem;
}
.ranking-jump {
  background: rgba(10, 27, 42, 0.78);
  border: 1px solid #6f9dc4;
  border-radius: 4px;
  color: #f5f8fa;
  cursor: pointer;
  font: inherit;
  line-height: 1;
  padding: 0.16rem 0.3rem;
  user-select: none;
}
.ranking-jump:hover, .ranking-jump:focus-visible {
  background: #315a77;
  outline: 2px solid #8ac5ff;
  outline-offset: -2px;
}
.ranking-row:hover, .ranking-row.active {
  background-color: #1d3348;
}
.ranking-label { color: #f5f8fa; min-width: 0; overflow: hidden; text-overflow: ellipsis; white-space: nowrap; }
.ranking-summary { color: #e2eaf1; font-size: 0.78rem; white-space: nowrap; }
.empty-ranking { color: #aab8c5; font-size: 0.86rem; margin: 0.5rem 0; }
table { border-collapse: collapse; font-size: 0.84rem; width: 100%; }
th, td { border-bottom: 1px solid #2d4358; padding: 0.48rem 0.45rem; text-align: left; vertical-align: top; }
th { color: #aab8c5; font-size: 0.74rem; letter-spacing: 0.03em; position: sticky; top: 0; background: #172536; text-transform: uppercase; z-index: 1; }
tr:last-child td { border-bottom: 0; }
.record-reason { color: #f4dfae; font-size: 0.82rem; }
.record-reason.not-beamng-actionable { color: #9ca8b2; }
.record-technical { color: #aab8c5; font-size: 0.76rem; margin-top: 0.2rem; white-space: pre-wrap; }
.hidden { display: none; }
@media (max-width: 1000px) {
  .triage-grid { grid-template-columns: 1fr; }
}
@media (max-width: 560px) {
  .ranking-content { align-items: flex-start; flex-direction: column; gap: 0.1rem; }
}
</style>
</head>
<body>
<header>
  <div class="header-title">
    <h1>Pacenote Conversion Audit</h1>
    <p id="report-meta" class="muted"></p>
  </div>
  <div class="issue-filters" aria-label="Issue type filters">
    <label class="issue-filter converter">
      <input id="show-converter-issues" type="checkbox" checked>
      Converter issues (<span id="converter-issue-count"></span>)
    </label>
    <label class="issue-filter beamng">
      <input id="show-beamng-issues" type="checkbox" checked>
      BeamNG issues (<span id="beamng-issue-count"></span>)
    </label>
  </div>
</header>
<main class="layout">
  <section class="triage-grid" aria-label="Conversion issue rankings">
    <section class="panel">
      <h2>By Issue</h2>
      <div id="opportunity-list" class="ranking-list"></div>
    </section>
    <section class="panel">
      <h2>By Stage</h2>
      <div id="validation-stage-list" class="ranking-list"></div>
    </section>
  </section>
  <section class="controls" aria-label="Report filters">
    <div class="control">
      <label for="search">Search stages and problems</label>
      <input id="search" type="search" placeholder="Stage, call, reason, type…">
    </div>
    <div class="control">
      <label for="status-filter">Stage status</label>
      <select id="status-filter">
        <option value="all">All stages</option>
        <option value="attention">Needs attention</option>
        <option value="loss">Conversion issues</option>
        <option value="warning">Warnings only</option>
        <option value="clean">Clean</option>
      </select>
    </div>
    <button id="clear-ranking-filter" type="button" class="hidden">Clear list focus</button>
  </section>
  <section id="problem-panel" class="panel">
    <h2 id="problem-heading">Pacenotes needing attention</h2>
    <p id="problem-context" class="muted"></p>
    <div id="problem-table" class="table-wrap"></div>
  </section>
</main>
<script id="pacenote-audit-data" type="application/json">__PACENOTE_AUDIT_DATA__</script>
<script>
(() => {
  const data = JSON.parse(document.getElementById("pacenote-audit-data").textContent);
  const state = {
    showBeamNGIssues: true,
    showConverterIssues: true,
    query: "",
    selectedActionKey: null,
    problemStageId: null,
    status: "all",
    problemLimit: 50,
  };
  const number = new Intl.NumberFormat();
  const formatDistance = value => `${Number(value).toFixed(1)} m`;
  const stageById = new Map(data.stages.map(stage => [stage.id, stage]));

  function element(tag, text, className) {
    const node = document.createElement(tag);
    if (text !== undefined && text !== null) node.textContent = String(text);
    if (className) node.className = className;
    return node;
  }

  function table(headers) {
    const root = document.createElement("table");
    const head = document.createElement("thead");
    const row = document.createElement("tr");
    for (const header of headers) row.append(element("th", header));
    head.append(row);
    root.append(head, document.createElement("tbody"));
    return root;
  }

  function appendRow(body, values, className) {
    const row = document.createElement("tr");
    if (className) row.className = className;
    for (const value of values) {
      const cell = document.createElement("td");
      if (value instanceof Node) cell.append(value);
      else cell.textContent = String(value ?? "");
      row.append(cell);
    }
    body.append(row);
    return row;
  }

  function plural(count, singular, pluralForm = `${singular}s`) {
    return `${number.format(count)} ${count === 1 ? singular : pluralForm}`;
  }

  function stageLabel(stage) {
    return `${stage.stageName}${stage.variant !== "base" ? ` (${stage.variant})` : ""}`;
  }

  function technicalIssue(record) {
    return [...(record.reasons || []), ...(record.issues || [])].join("; ");
  }

  function conciseIssue(record) {
    const actionTitles = (record.lossActions || [])
      .map(action => action.title)
      .filter(Boolean);
    if (actionTitles.length) return actionTitles.join("; ");
    const issue = technicalIssue(record).toLocaleLowerCase();
    if (issue.includes("no matching definition")) return "Unknown pacenote definition";
    if (issue.includes("no adjacent beamng pacenote")) return "Cannot attach to a converted pacenote";
    if (issue.includes("beamng pre-corner modifier slots are full")) return "BeamNG pre-corner modifier slots are full";
    if (issue.includes("beamng post-corner modifier slots are full")) return "BeamNG post-corner modifier slots are full";
    if (issue.includes("after finish")) return "Call is after the finish";
    if (issue.includes("missing") && issue.includes("corner")) return "Missing adjacent corner";
    if (issue.includes("unknown")) return "Unknown source value";
    return record.label || "Conversion needs review";
  }

  function isNotBeamNGActionable(record) {
    const actions = record.lossActions || [];
    return actions.length > 0 && actions.every(
      action => action.beamngActionable === false,
    );
  }

  function issueKind(record) {
    return isNotBeamNGActionable(record) ? "converter" : "beamng";
  }

  function isActionVisible(action) {
    return action.beamngActionable === false
      ? state.showConverterIssues
      : state.showBeamNGIssues;
  }

  function isRecordVisible(record) {
    return issueKind(record) === "converter"
      ? state.showConverterIssues
      : state.showBeamNGIssues;
  }

  function failureLabel(failure) {
    return failure.stageName || failure.stageKey || failure.sourcePath || "Unknown source";
  }

  function searchText(stage) {
    return [
      stage.stageName,
      stage.stageKey,
      stage.variant,
      ...stage.warnings,
      ...stage.records.flatMap(record => [
        record.label,
        record.typeHex,
        ...record.reasons,
        ...record.issues,
      ]),
    ].join(" ").toLocaleLowerCase();
  }

  const stageSearchIndex = new Map(
    data.stages.map(stage => [stage.id, searchText(stage)])
  );
  const allProblems = data.stages.flatMap(stage => stage.records
    .filter(record => record.outcome === "needs-attention")
    .map(record => ({
      record,
      searchText: [
        stage.stageName,
        stage.stageKey,
        record.label,
        record.typeHex,
        ...record.reasons,
        ...record.issues,
      ].join(" ").toLocaleLowerCase(),
      stage,
    })));
  const issueTotals = allProblems.reduce(
    (totals, { record }) => {
      totals[issueKind(record)] += 1;
      return totals;
    },
    { beamng: 0, converter: 0 },
  );

  function activeQuery() {
    return state.query.trim().toLocaleLowerCase();
  }

  function matchesStatus(stage) {
    return state.status === "all"
      || (state.status === "attention" && stage.status !== "clean")
      || stage.status === state.status;
  }

  function visibleFailures() {
    if (state.status !== "all" && state.status !== "attention") return [];
    const query = activeQuery();
    return data.sourceFailures.filter(failure => {
      const text = [
        failure.sourceFormat,
        failureLabel(failure),
        failure.variant,
        failure.message,
      ].join(" ").toLocaleLowerCase();
      return !query || text.includes(query);
    });
  }

  function visibleStages() {
    const query = activeQuery();
    return data.stages.filter(stage => (
      matchesStatus(stage)
      && (!query || stageSearchIndex.get(stage.id).includes(query))
    ));
  }

  function matchingProblemRows() {
    const stageIds = new Set(visibleStages().map(stage => stage.id));
    const query = activeQuery();
    return allProblems.filter(item => (
      stageIds.has(item.stage.id)
      && (!query || item.searchText.includes(query))
    ));
  }

  function baseProblemRows() {
    return matchingProblemRows().filter(({ record }) => isRecordVisible(record));
  }

  function problemRows() {
    const rows = state.selectedActionKey
      ? matchingProblemRows()
      : baseProblemRows();
    return rows
      .filter(item => (
        !state.problemStageId || item.stage.id === state.problemStageId
      ))
      .filter(item => (
        !state.selectedActionKey
        || (item.record.lossActions || []).some(
          action => action.key === state.selectedActionKey,
        )
      ))
      .sort((left, right) => (
        right.stage.problemCount - left.stage.problemCount
        || left.stage.stageName.localeCompare(right.stage.stageName)
        || left.record.distance - right.record.distance
      ));
  }

  function fixOpportunities(rows) {
    const opportunities = new Map();
    for (const { stage, record } of rows) {
      for (const action of record.lossActions || []) {
        const opportunity = opportunities.get(action.key) || {
          action,
          count: 0,
          stageIds: new Set(),
        };
        opportunity.count += 1;
        opportunity.stageIds.add(stage.id);
        opportunities.set(action.key, opportunity);
      }
    }
    return [...opportunities.values()]
      .map(opportunity => ({
        ...opportunity,
        stageCount: opportunity.stageIds.size,
      }))
      .sort((left, right) => (
        right.count - left.count
        || right.stageCount - left.stageCount
        || left.action.title.localeCompare(right.action.title)
      ));
  }

  function stageRankings() {
    const rankings = new Map();
    for (const { stage, record } of baseProblemRows()) {
      const ranking = rankings.get(stage.id) || {
        beamng: 0,
        converter: 0,
        stage,
        total: 0,
      };
      ranking[issueKind(record)] += 1;
      ranking.total += 1;
      rankings.set(stage.id, ranking);
    }
    return [...rankings.values()].sort((left, right) => (
      right.total - left.total
      || left.stage.stageName.localeCompare(right.stage.stageName)
    ));
  }

  function rankingRow({ label, summary, value, max, active, segments, onClick }) {
    const root = document.createElement("div");
    root.className = [
      "ranking-row",
      active ? "active" : "",
    ].filter(Boolean).join(" ");
    root.style.setProperty(
      "--ranking-width",
      `${100 * Math.max(0, Math.min(1, value / Math.max(max, 1)))}%`,
    );
    const bar = document.createElement("span");
    bar.className = "ranking-bar";
    bar.setAttribute("aria-hidden", "true");
    for (const segment of segments) {
      if (!segment.value) continue;
      const part = document.createElement("span");
      part.className = `ranking-bar-segment ${segment.kind}`;
      part.style.width = `${100 * segment.value / Math.max(value, 1)}%`;
      bar.append(part);
    }
    const content = document.createElement("div");
    content.className = "ranking-content";
    const actions = document.createElement("span");
    actions.className = "ranking-actions";
    const jump = document.createElement("button");
    jump.className = "ranking-jump";
    jump.type = "button";
    jump.textContent = "🔍";
    jump.title = `View ${label}`;
    jump.setAttribute("aria-label", `View ${label}`);
    jump.setAttribute("aria-pressed", active ? "true" : "false");
    jump.addEventListener("click", onClick);
    actions.append(
      element("span", summary, "ranking-summary"),
      jump,
    );
    content.append(element("span", label, "ranking-label"), actions);
    root.append(bar, content);
    return root;
  }

  function selectAction(key) {
    state.selectedActionKey = key;
    state.problemStageId = null;
    state.status = "loss";
    state.problemLimit = 50;
    document.getElementById("status-filter").value = "loss";
    renderTriage();
    document.getElementById("problem-panel").scrollIntoView({
      behavior: "smooth",
      block: "start",
    });
  }

  function renderOpportunityList() {
    const root = document.getElementById("opportunity-list");
    const opportunities = fixOpportunities(matchingProblemRows()).filter(
      opportunity => isActionVisible(opportunity.action),
    );
    if (!opportunities.length) {
      root.replaceChildren(element(
        "p",
        "No one-change opportunities match the current filters.",
        "empty-ranking",
      ));
      return;
    }
    const max = Math.max(...opportunities.map(opportunity => opportunity.count), 1);
    root.replaceChildren(...opportunities.map(opportunity => rankingRow({
      active: state.selectedActionKey === opportunity.action.key,
      label: opportunity.action.title,
      max,
      onClick: () => selectAction(opportunity.action.key),
      segments: [{
        kind: opportunity.action.beamngActionable === false
          ? "converter"
          : "beamng",
        value: opportunity.count,
      }],
      summary: `${plural(opportunity.count, "hit")} (in ${plural(opportunity.stageCount, "stage")})`,
      value: opportunity.count,
    })));
  }

  function renderValidationStageList() {
    const root = document.getElementById("validation-stage-list");
    const rankings = stageRankings();
    if (!rankings.length) {
      root.replaceChildren(element(
        "p",
        "No stages with confirmed hits match the current filters.",
        "empty-ranking",
      ));
      return;
    }
    const max = Math.max(...rankings.map(ranking => ranking.total), 1);
    root.replaceChildren(...rankings.map(ranking => rankingRow({
      active: state.problemStageId === ranking.stage.id,
      label: stageLabel(ranking.stage),
      max,
      onClick: () => {
        state.problemStageId = ranking.stage.id;
        state.selectedActionKey = null;
        state.status = "loss";
        state.problemLimit = 50;
        document.getElementById("status-filter").value = "loss";
        renderTriage();
        document.getElementById("problem-panel").scrollIntoView({ behavior: "smooth", block: "start" });
      },
      segments: [
        { kind: "beamng", value: ranking.beamng },
        { kind: "converter", value: ranking.converter },
      ],
      summary: plural(ranking.total, "hit"),
      value: ranking.total,
    })));
  }

  function renderListFocus() {
    const button = document.getElementById("clear-ranking-filter");
    const hasFocus = Boolean(state.selectedActionKey || state.problemStageId);
    button.classList.toggle("hidden", !hasFocus);
    if (!hasFocus) return;
    if (state.problemStageId) {
      button.textContent = `Show all stages instead of ${stageLabel(stageById.get(state.problemStageId))}`;
      return;
    }
    const opportunity = fixOpportunities(matchingProblemRows()).find(
      item => item.action.key === state.selectedActionKey,
    );
    button.textContent = opportunity
      ? `Show all opportunities instead of ${opportunity.action.title}`
      : "Clear list focus";
  }

  function renderProblems() {
    const root = document.getElementById("problem-table");
    const rows = problemRows();
    const failures = visibleFailures();
    const total = rows.length + failures.length;
    const heading = document.getElementById("problem-heading");
    const context = document.getElementById("problem-context");
    let focus = "all matching stages";
    if (state.problemStageId) {
      focus = stageLabel(stageById.get(state.problemStageId));
    } else if (state.selectedActionKey) {
      const opportunity = fixOpportunities(matchingProblemRows()).find(
        item => item.action.key === state.selectedActionKey,
      );
      focus = opportunity
        ? opportunity.action.title
        : "the selected opportunity";
    }
    heading.textContent = `Pacenotes needing attention: ${focus}`;
    context.textContent = total
      ? `Showing ${plural(Math.min(rows.length, state.problemLimit) + failures.length, "record")} of ${plural(total, "matched record")}.`
      : "No pacenotes needing attention match the current filters.";
    const rootTable = table(["Stage", "At", "Source call", "What happened"]);
    const body = rootTable.tBodies[0];
    for (const { stage, record } of rows.slice(0, state.problemLimit)) {
      const issue = element(
        "div",
        undefined,
        `record-reason${isNotBeamNGActionable(record) ? " not-beamng-actionable" : ""}`,
      );
      issue.append(element("div", conciseIssue(record)));
      const detail = technicalIssue(record);
      if (detail) issue.append(element("div", detail, "record-technical"));
      const row = appendRow(body, [
        stageLabel(stage),
        formatDistance(record.distance),
        `${record.typeHex} · ${record.label}`,
        issue,
      ]);
    }
    for (const failure of failures) {
      appendRow(body, [
        failureLabel(failure),
        "—",
        `${failure.sourceFormat}${failure.variant ? ` · ${failure.variant}` : ""}`,
        element("div", failure.message, "record-technical"),
      ]);
    }
    if (!total) {
      appendRow(body, [element("span", "No problematic pacenotes match the current filters.")]);
    } else if (rows.length > state.problemLimit) {
      const more = element("button", `Show ${Math.min(50, rows.length - state.problemLimit)} more`);
      more.addEventListener("click", () => {
        state.problemLimit += 50;
        renderProblems();
      });
      appendRow(body, [more]);
    }
    root.replaceChildren(rootTable);
  }

  function renderTriage() {
    renderOpportunityList();
    renderValidationStageList();
    renderListFocus();
    renderProblems();
  }

  document.getElementById("report-meta").textContent = `Generated ${data.generatedAt} · ${data.inputRoot}`;
  document.getElementById("converter-issue-count").textContent = number.format(issueTotals.converter);
  document.getElementById("beamng-issue-count").textContent = number.format(issueTotals.beamng);
  document.getElementById("search").addEventListener("input", event => {
    state.query = event.target.value;
    state.problemLimit = 50;
    renderTriage();
  });
  document.getElementById("status-filter").addEventListener("change", event => {
    state.status = event.target.value;
    state.problemLimit = 50;
    renderTriage();
  });
  function applyIssueFilters() {
    const selected = fixOpportunities(matchingProblemRows()).find(
      opportunity => opportunity.action.key === state.selectedActionKey,
    );
    if (selected && !isActionVisible(selected.action)) {
      state.selectedActionKey = null;
    }
    if (
      state.problemStageId
      && !stageRankings().some(
        ranking => ranking.stage.id === state.problemStageId,
      )
    ) {
      state.problemStageId = null;
    }
    state.problemLimit = 50;
    renderTriage();
  }
  document.getElementById("show-converter-issues").addEventListener("change", event => {
    state.showConverterIssues = event.target.checked;
    applyIssueFilters();
  });
  document.getElementById("show-beamng-issues").addEventListener("change", event => {
    state.showBeamNGIssues = event.target.checked;
    applyIssueFilters();
  });
  document.getElementById("clear-ranking-filter").addEventListener("click", () => {
    state.problemStageId = null;
    state.selectedActionKey = null;
    state.problemLimit = 50;
    renderTriage();
  });
  renderTriage();
})();
</script>
</body>
</html>
"""
