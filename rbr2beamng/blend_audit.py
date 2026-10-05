from __future__ import annotations

import html
import json
import uuid
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import __version__
from .core import ConversionCancelled, ConversionError, ProgressReporter
from .filesystem import current_filesystem
from .geometry import AssimpMeshLoader
from .material_baker import (
    is_vertex_lerp_material,
    vertex_lerp_effect_needs_custom_material,
    vertex_lerp_uv_indices,
)
from .models import MeshPart, RbrMaterial, RbrStage
from .rbr import load_stage, load_transforms
from .stage_sources import discover_stages

_AREA_FACE_CHUNK = 65_536


@dataclass(frozen=True)
class BlendAuditResult:
    output_path: Path
    stage_count: int
    completed_stage_count: int
    error_count: int
    affected_stage_count: int


def _metrics() -> dict[str, float | int]:
    return {
        "meshParts": 0,
        "instances": 0,
        "triangles": 0,
        "invalidAreaTriangles": 0,
        "areaM2": 0.0,
    }


def _add_metrics(
    destination: dict[str, float | int],
    source: dict[str, float | int],
) -> None:
    for key, value in source.items():
        destination[key] = destination[key] + value


def _metric_copy(values: dict[str, float | int]) -> dict[str, float | int]:
    return {
        "meshParts": int(values["meshParts"]),
        "instances": int(values["instances"]),
        "triangles": int(values["triangles"]),
        "invalidAreaTriangles": int(values["invalidAreaTriangles"]),
        "areaM2": round(float(values["areaM2"]), 3),
    }


def _part_world_area(part: MeshPart, transforms: np.ndarray) -> tuple[float, int]:
    vertices = np.asarray(part.vertices, dtype=np.float64)
    faces = np.asarray(part.faces, dtype=np.uint32).reshape((-1, 3))
    if len(faces) == 0:
        return 0.0, 0
    result = 0.0
    invalid_triangles = 0
    for transform in np.asarray(transforms, dtype=np.float64):
        linear = transform[:3, :3]
        for start in range(0, len(faces), _AREA_FACE_CHUNK):
            triangles = vertices[faces[start : start + _AREA_FACE_CHUNK]]
            first_edges = (triangles[:, 1] - triangles[:, 0]) @ linear
            second_edges = (triangles[:, 2] - triangles[:, 0]) @ linear
            areas = np.linalg.norm(np.cross(first_edges, second_edges), axis=1) * 0.5
            finite = np.isfinite(areas)
            invalid_triangles += int(len(areas) - np.count_nonzero(finite))
            result += float(areas[finite].sum())
    return result, invalid_triangles


def _part_metrics(part: MeshPart, transforms: np.ndarray) -> dict[str, float | int]:
    instance_count = len(transforms)
    area, invalid_area_triangles = _part_world_area(part, transforms)
    return {
        "meshParts": 1,
        "instances": instance_count,
        "triangles": len(part.faces) * instance_count,
        "invalidAreaTriangles": invalid_area_triangles,
        "areaM2": area,
    }


def classify_vertex_lerp_part(
    part: MeshPart,
    material: RbrMaterial,
) -> dict[str, object]:
    """Describe whether a vertex-lerp part fits the standard PBR material."""
    required_indices = vertex_lerp_uv_indices(material.effect)
    missing_indices = [
        index for index in required_indices if index >= len(part.texcoord_sets)
    ]
    unique_uvs: list[np.ndarray] = []
    if not missing_indices:
        for index in required_indices:
            candidate = np.asarray(part.texcoord_sets[index])
            if not any(np.array_equal(candidate, known) for known in unique_uvs):
                unique_uvs.append(candidate)

    reasons: list[str] = []
    if part.blend_weights is None:
        reasons.append("missing-blend-weights")
    if missing_indices:
        reasons.append("missing-required-uv-stream")
    if len(unique_uvs) > 2:
        reasons.append("more-than-two-distinct-uv-streams")
    if vertex_lerp_effect_needs_custom_material(material.effect):
        reasons.append("unsupported-multiplier-additive-effect")

    custom_material_reasons = {
        "more-than-two-distinct-uv-streams",
        "unsupported-multiplier-additive-effect",
    }
    return {
        "effect": material.effect,
        "requiredUvIndices": list(required_indices),
        "distinctUvStreamCount": len(unique_uvs) if not missing_indices else None,
        "missingUvIndices": missing_indices,
        "reasons": reasons,
        "directVertexBlendPossible": not reasons,
        "requiresCustomMaterial": bool(custom_material_reasons.intersection(reasons)),
    }


def _add_part_record(
    stage_metrics: dict[str, dict[str, float | int]],
    effect_metrics: dict[str, dict[str, dict[str, float | int]]],
    reason_metrics: dict[str, dict[str, float | int]],
    part: MeshPart,
    material: RbrMaterial | None,
    transforms: np.ndarray,
) -> None:
    metrics = _part_metrics(part, transforms)
    _add_metrics(stage_metrics["visibleGeometry"], metrics)
    if not is_vertex_lerp_material(material):
        return

    assert material is not None
    _add_metrics(stage_metrics["vertexLerpGeometry"], metrics)
    classification = classify_vertex_lerp_part(part, material)
    effect = str(classification["effect"])
    effect_bucket = effect_metrics[effect]
    _add_metrics(effect_bucket["vertexLerpGeometry"], metrics)
    if bool(classification["directVertexBlendPossible"]):
        _add_metrics(stage_metrics["directVertexBlend"], metrics)
        _add_metrics(effect_bucket["directVertexBlend"], metrics)
    if bool(classification["requiresCustomMaterial"]):
        _add_metrics(stage_metrics["requiresCustomMaterial"], metrics)
        _add_metrics(effect_bucket["requiresCustomMaterial"], metrics)
    for reason in classification["reasons"]:
        _add_metrics(reason_metrics[str(reason)], metrics)
        _add_metrics(effect_bucket["reasons"][str(reason)], metrics)


def _empty_effect_metrics() -> dict[str, dict[str, float | int]]:
    return {
        "vertexLerpGeometry": _metrics(),
        "directVertexBlend": _metrics(),
        "requiresCustomMaterial": _metrics(),
        "reasons": defaultdict(_metrics),
    }


def _serialise_effect_metrics(
    values: dict[str, dict[str, dict[str, float | int]]],
) -> list[dict[str, object]]:
    result = []
    for effect, metrics in sorted(values.items(), key=lambda item: item[0].casefold()):
        reasons = metrics["reasons"]
        result.append(
            {
                "effect": effect,
                "vertexLerpGeometry": _metric_copy(metrics["vertexLerpGeometry"]),
                "directVertexBlend": _metric_copy(metrics["directVertexBlend"]),
                "requiresCustomMaterial": _metric_copy(
                    metrics["requiresCustomMaterial"]
                ),
                "reasons": {
                    key: _metric_copy(value)
                    for key, value in sorted(reasons.items())
                },
            }
        )
    return result


def _analyse_stage(stage: RbrStage, loader: AssimpMeshLoader) -> dict[str, object]:
    materials = {material.name.casefold(): material for material in stage.materials}
    mesh_cache = {}
    metrics = {
        "visibleGeometry": _metrics(),
        "vertexLerpGeometry": _metrics(),
        "directVertexBlend": _metrics(),
        "requiresCustomMaterial": _metrics(),
    }
    effect_metrics: dict[str, dict[str, dict[str, float | int]]] = defaultdict(
        _empty_effect_metrics
    )
    reason_metrics: dict[str, dict[str, float | int]] = defaultdict(_metrics)
    errors: list[str] = []

    for stage_object in stage.objects:
        if not stage_object.visible:
            continue
        try:
            asset = mesh_cache.get(stage_object.mesh_path)
            if asset is None:
                asset = loader.load(stage_object.mesh_path)
                mesh_cache[stage_object.mesh_path] = asset
            transforms = load_transforms(stage_object)
        except (ConversionError, OSError, ValueError) as exc:
            errors.append(f"{stage_object.mesh_name}: {exc}")
            continue
        for part in asset.parts:
            _add_part_record(
                metrics,
                effect_metrics,
                reason_metrics,
                part,
                materials.get(part.material_name.casefold()),
                transforms,
            )

    return {
        "folder": stage.metadata.folder_name,
        "name": stage.metadata.name,
        "metrics": {key: _metric_copy(value) for key, value in metrics.items()},
        "effects": _serialise_effect_metrics(effect_metrics),
        "reasons": {
            key: _metric_copy(value)
            for key, value in sorted(reason_metrics.items())
        },
        "errors": errors,
    }


def _summary(
    stages: list[dict[str, object]],
    failures: list[dict[str, str]],
    skipped_source_formats: dict[str, int],
) -> dict[str, object]:
    metrics = {
        "visibleGeometry": _metrics(),
        "vertexLerpGeometry": _metrics(),
        "directVertexBlend": _metrics(),
        "requiresCustomMaterial": _metrics(),
    }
    reason_metrics: dict[str, dict[str, float | int]] = defaultdict(_metrics)
    for stage in stages:
        stage_metrics = stage["metrics"]
        assert isinstance(stage_metrics, dict)
        for key, value in metrics.items():
            candidate = stage_metrics[key]
            assert isinstance(candidate, dict)
            _add_metrics(value, candidate)
        stage_reasons = stage["reasons"]
        assert isinstance(stage_reasons, dict)
        for reason, reason_values in stage_reasons.items():
            assert isinstance(reason_values, dict)
            _add_metrics(reason_metrics[str(reason)], reason_values)
    affected_stage_count = sum(
        int(
            (
                stage["metrics"]["requiresCustomMaterial"]["instances"]
            )
            > 0
        )
        for stage in stages
    )
    return {
        "stageCount": len(stages) + len(failures),
        "completedStageCount": len(stages),
        "errorCount": len(failures),
        "affectedStageCount": affected_stage_count,
        "skippedSourceFormats": dict(sorted(skipped_source_formats.items())),
        "reasons": {
            key: _metric_copy(value)
            for key, value in sorted(reason_metrics.items())
        },
        **{key: _metric_copy(value) for key, value in metrics.items()},
    }


def build_blend_audit(
    rbr_root: Path,
    output: Path,
    *,
    reporter: ProgressReporter | None = None,
) -> BlendAuditResult:
    filesystem = current_filesystem()
    root = filesystem.read_path(rbr_root)
    output = filesystem.write_path(output)
    if filesystem.exists(output):
        raise ConversionError(f"Blend audit output already exists: {output}")
    filesystem.mkdir(output.parent, parents=True, exist_ok=True)

    failures: list[dict[str, str]] = []
    skipped_source_formats: dict[str, int] = defaultdict(int)
    inspections = discover_stages(root)
    targets = []
    for inspection in inspections:
        if inspection.source_format != "rx":
            skipped_source_formats[inspection.source_format] += 1
        elif inspection.valid:
            targets.append(inspection)
        else:
            failures.append(
                {
                    "folder": inspection.metadata.folder_name,
                    "name": inspection.metadata.name,
                    "error": "; ".join(inspection.issues),
                }
            )

    stages: list[dict[str, object]] = []
    loader = AssimpMeshLoader()
    for index, inspection in enumerate(targets, 1):
        if reporter:
            reporter.emit(
                "blend-audit",
                "Analysing source blend capability",
                current=index,
                total=len(targets),
                detail=inspection.metadata.folder_name,
            )
        try:
            stages.append(_analyse_stage(load_stage(inspection.root), loader))
        except ConversionCancelled:
            raise
        except (ConversionError, OSError, ValueError) as exc:
            failures.append(
                {
                    "folder": inspection.metadata.folder_name,
                    "name": inspection.metadata.name,
                    "error": str(exc),
                }
            )
    payload = {
        "schemaVersion": 1,
        "generatedAt": datetime.now(timezone.utc).isoformat(),
        "inputRoot": str(root),
        "summary": _summary(stages, failures, skipped_source_formats),
        "stages": stages,
        "sourceFailures": failures,
    }
    temporary = output.parent / f".{output.name}.partial-{uuid.uuid4().hex}"
    try:
        filesystem.write_text(
            temporary,
            render_blend_audit_report(payload),
            encoding="utf-8",
        )
        filesystem.replace(temporary, output)
    except BaseException:
        filesystem.unlink(temporary, missing_ok=True)
        raise
    summary = payload["summary"]
    assert isinstance(summary, dict)
    return BlendAuditResult(
        output_path=output,
        stage_count=int(summary["stageCount"]),
        completed_stage_count=int(summary["completedStageCount"]),
        error_count=int(summary["errorCount"]),
        affected_stage_count=int(summary["affectedStageCount"]),
    )


def _format_metric(metrics: dict[str, object]) -> str:
    result = (
        f"{int(metrics['instances']):,} instances · "
        f"{int(metrics['triangles']):,} triangles · "
        f"{float(metrics['areaM2']):,.0f} m²"
    )
    invalid = int(metrics["invalidAreaTriangles"])
    if invalid:
        result += f" · {invalid:,} triangles with undefined area"
    return result


def _percentage(numerator: float | int, denominator: float | int) -> str:
    if not denominator:
        return "0%"
    return f"{float(numerator) / float(denominator) * 100:.1f}%"


def _effect_rows(stage: dict[str, object]) -> str:
    rows = []
    for effect in stage["effects"]:
        assert isinstance(effect, dict)
        reasons = effect["reasons"]
        assert isinstance(reasons, dict)
        reason_text = "<br>".join(
            f"{html.escape(str(reason))}: "
            f"{html.escape(_format_metric(reason_metrics))}"
            for reason, reason_metrics in reasons.items()
        ) or "None"
        rows.append(
            "<tr>"
            f"<td>{html.escape(str(effect['effect']))}</td>"
            f"<td>{html.escape(_format_metric(effect['vertexLerpGeometry']))}</td>"
            f"<td>{html.escape(_format_metric(effect['directVertexBlend']))}</td>"
            f"<td>{html.escape(_format_metric(effect['requiresCustomMaterial']))}</td>"
            f"<td>{reason_text}</td>"
            "</tr>"
        )
    return "".join(rows) or "<tr><td colspan=\"5\">No vertex-lerp materials</td></tr>"


def render_blend_audit_report(data: dict[str, object]) -> str:
    summary = data["summary"]
    assert isinstance(summary, dict)
    visible = summary["visibleGeometry"]
    vertex_lerp = summary["vertexLerpGeometry"]
    direct = summary["directVertexBlend"]
    custom = summary["requiresCustomMaterial"]
    assert all(isinstance(item, dict) for item in (visible, vertex_lerp, direct, custom))
    stage_rows = []
    for stage in data["stages"]:
        assert isinstance(stage, dict)
        metrics = stage["metrics"]
        assert isinstance(metrics, dict)
        custom_metrics = metrics["requiresCustomMaterial"]
        visible_metrics = metrics["visibleGeometry"]
        stage_rows.append(
            "<details><summary>"
            f"{html.escape(str(stage['name']))} — "
            f"{_percentage(custom_metrics['triangles'], visible_metrics['triangles'])} "
            "of visible triangles need custom material support"
            "</summary>"
            "<table><thead><tr><th>Effect</th><th>Vertex-lerp geometry</th>"
            "<th>Direct vertex blend</th><th>Needs custom material</th><th>Reasons</th>"
            "</tr></thead><tbody>"
            f"{_effect_rows(stage)}"
            "</tbody></table></details>"
        )
    failures = data["sourceFailures"]
    assert isinstance(failures, list)
    failure_rows = "".join(
        "<li>"
        f"{html.escape(str(failure['name']))}: {html.escape(str(failure['error']))}"
        "</li>"
        for failure in failures
    ) or "<li>None</li>"
    reasons = summary.get("reasons", {})
    assert isinstance(reasons, dict)
    reason_rows = "".join(
        "<tr>"
        f"<td>{html.escape(str(reason))}</td>"
        f"<td>{html.escape(_format_metric(reason_metrics))}</td>"
        "</tr>"
        for reason, reason_metrics in reasons.items()
    ) or "<tr><td colspan=\"2\">None</td></tr>"
    report_json = json.dumps(data, ensure_ascii=False, separators=(",", ":"))
    report_json = report_json.replace("<", "\\u003c").replace(">", "\\u003e")
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Blend Capability Audit - RBR2BeamNG v{__version__}</title>
<style>
body {{ font: 16px/1.5 system-ui, sans-serif; margin: 2rem auto; max-width: 1100px; padding: 0 1rem; color: #18212b; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0; }}
th, td {{ border: 1px solid #c6d0db; padding: .45rem .65rem; text-align: left; vertical-align: top; }}
th {{ background: #edf2f7; }}
details {{ margin: .8rem 0; padding: .4rem; border: 1px solid #c6d0db; }}
summary {{ cursor: pointer; font-weight: 600; }}
.metrics {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(230px, 1fr)); gap: 1rem; }}
.metric {{ border: 1px solid #c6d0db; padding: .75rem; border-radius: .3rem; }}
.label {{ color: #536273; font-size: .9rem; }}
</style>
</head>
<body>
<h1>Blend Capability Audit</h1>
<p>Source: {html.escape(str(data['inputRoot']))}</p>
<p>Scanned {summary['completedStageCount']} RX stages. {summary['affectedStageCount']} need material support beyond the standard BeamNG PBR material.</p>
<p>Area percentages exclude triangles whose source coordinates make their area undefined.</p>
<div class="metrics">
<div class="metric"><div class="label">All visible geometry</div>{html.escape(_format_metric(visible))}</div>
<div class="metric"><div class="label">Vertex-lerp geometry</div>{html.escape(_format_metric(vertex_lerp))}<br>{_percentage(vertex_lerp['areaM2'], visible['areaM2'])} of visible area</div>
<div class="metric"><div class="label">Direct vertex blend possible</div>{html.escape(_format_metric(direct))}<br>{_percentage(direct['triangles'], vertex_lerp['triangles'])} of vertex-lerp triangles · {_percentage(direct['areaM2'], vertex_lerp['areaM2'])} of vertex-lerp area</div>
<div class="metric"><div class="label">Needs custom material</div>{html.escape(_format_metric(custom))}<br>{_percentage(custom['triangles'], visible['triangles'])} of visible triangles · {_percentage(custom['triangles'], vertex_lerp['triangles'])} of vertex-lerp triangles · {_percentage(custom['areaM2'], vertex_lerp['areaM2'])} of vertex-lerp area</div>
</div>
<h2>Reasons custom material support is needed</h2>
<table><thead><tr><th>Reason</th><th>Affected geometry</th></tr></thead><tbody>{reason_rows}</tbody></table>
<h2>Stages</h2>
{''.join(stage_rows) or '<p>No valid RX stages were scanned.</p>'}
<h2>Stages not fully scanned</h2>
<ul>{failure_rows}</ul>
<script id="blend-audit-data" type="application/json">{report_json}</script>
</body>
</html>
"""
