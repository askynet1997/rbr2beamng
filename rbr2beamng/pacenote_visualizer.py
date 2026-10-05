from __future__ import annotations

from bisect import bisect_left
import json
import math
from collections.abc import Mapping

import numpy as np

from . import __version__
from .models import RbrStage
from .pacenotes import RallyNotebookResult, project_route


_MAX_ROUTE_POINTS = 5000
_CURVATURE_WINDOW = 4
_ROAD_WIDTH_METRES = 7.0
_MARKER_RADIUS_METRES = 1.5
_PACENOTE_LEADER_WIDTH_PIXELS = 1.0
_PACENOTE_LEADER_LENGTH_PIXELS = 32.0
_PACENOTE_LABEL_GAP_PIXELS = 4.0
_MAX_ZOOM_SPAN_METRES = 30.0
_TIMING_MARKER_KINDS = {
    21: "start",
    22: "finish",
    23: "split",
    24: "stop",
}


def _sample_route(
    route: list[tuple[np.ndarray, np.ndarray, float]],
) -> list[list[float]]:
    if not route:
        return []
    stride = max(1, math.ceil(len(route) / _MAX_ROUTE_POINTS))
    samples = [
        [float(position[0]), -float(position[1])]
        for position, _direction, _distance in route[::stride]
    ]
    final_position = route[-1][0]
    final_point = [float(final_position[0]), -float(final_position[1])]
    if samples[-1] != final_point:
        samples.append(final_point)
    return samples


def _marker_data(
    markers: Mapping[str, np.ndarray],
    route: list[tuple[np.ndarray, np.ndarray, float]],
) -> list[dict[str, object]]:
    result: list[dict[str, object]] = []
    for name, position in markers.items():
        if name == "spawn_start":
            kind = "start"
            label = "Start Line"
        elif name == "spawn_time_control":
            kind = "time-control"
            label = "Time Control"
        elif name == "spawn_finish":
            kind = "finish"
            label = "Finish"
        elif name == "spawn_stop":
            kind = "stop"
            label = "Stop"
        elif name.startswith("spawn_split_"):
            kind = "split"
            label = "Split " + name.removeprefix("spawn_split_")
        else:
            kind = "marker"
            label = name
        if route:
            _route_position, direction, distance = min(
                route,
                key=lambda point: float(
                    np.sum((point[0][:2] - position[:2]) ** 2)
                ),
            )
            direction_x = float(direction[0])
            direction_y = -float(direction[1])
        else:
            distance = 0.0
            direction_x = 0.0
            direction_y = 0.0
        result.append(
            {
                "kind": kind,
                "label": label,
                "distance": float(distance),
                "x": float(position[0]),
                "y": -float(position[1]),
                "position": {
                    "x": float(position[0]),
                    "y": -float(position[1]),
                    "directionX": direction_x,
                    "directionY": direction_y,
                    "clamped": False,
                },
            }
        )
    return result


def _entry_position(
    route: list[tuple[np.ndarray, np.ndarray, float]],
    distance: float,
) -> dict[str, float | bool]:
    first_distance = route[0][2]
    last_distance = route[-1][2]
    position, direction = project_route(route, distance)
    return {
        "x": float(position[0]),
        "y": -float(position[1]),
        "directionX": float(direction[0]),
        "directionY": -float(direction[1]),
        "clamped": distance < first_distance or distance > last_distance,
    }


def _route_outer_sides(
    route: list[tuple[np.ndarray, np.ndarray, float]],
) -> tuple[list[float], list[int | None]]:
    distances = [float(distance) for _position, _direction, distance in route]
    sides: list[int | None] = []
    for index in range(len(route)):
        turn = 0.0
        for middle in range(
            max(1, index - _CURVATURE_WINDOW),
            min(len(route) - 2, index + _CURVATURE_WINDOW) + 1,
        ):
            previous = route[middle - 1][0]
            current = route[middle][0]
            following = route[middle + 1][0]
            incoming_x = float(current[0] - previous[0])
            incoming_y = float(current[1] - previous[1])
            outgoing_x = float(following[0] - current[0])
            outgoing_y = float(following[1] - current[1])
            if not (incoming_x or incoming_y) or not (
                outgoing_x or outgoing_y
            ):
                continue
            turn += math.atan2(
                incoming_x * outgoing_y - incoming_y * outgoing_x,
                incoming_x * outgoing_x + incoming_y * outgoing_y,
            )
        sides.append(1 if turn > 0.02 else -1 if turn < -0.02 else None)
    return distances, sides


def _outer_side_at_distance(
    distance: float,
    route_distances: list[float],
    route_sides: list[int | None],
) -> int | None:
    if not route_distances:
        return None
    index = bisect_left(route_distances, distance)
    if index == len(route_distances):
        index -= 1
    elif index > 0 and (
        distance - route_distances[index - 1]
        < route_distances[index] - distance
    ):
        index -= 1
    return route_sides[index]


def _record_marker_kind(record: Mapping[str, object]) -> str | None:
    note_type = record.get("type", record.get("sourceType"))
    return _TIMING_MARKER_KINDS.get(note_type) if isinstance(note_type, int) else None


def _record_conversion_loss(record: Mapping[str, object]) -> list[str]:
    result: list[str] = []
    for key in ("reasons", "issues"):
        values = record.get(key)
        if isinstance(values, list):
            result.extend(value for value in values if isinstance(value, str))
    if record.get("ignored") and not result:
        result.append("not converted")
    return list(dict.fromkeys(result))


def _record_source_indexes(record: Mapping[str, object]) -> list[int]:
    result: list[int] = []
    source_index = record.get("sourceIndex")
    if isinstance(source_index, int):
        result.append(source_index)
    contributors = record.get("contributingSourceIndexes")
    if isinstance(contributors, list):
        result.extend(index for index in contributors if isinstance(index, int))
    return list(dict.fromkeys(result))


def _visualizer_records(
    route: list[tuple[np.ndarray, np.ndarray, float]],
    records: tuple[dict[str, object], ...],
    marker_kinds: frozenset[str] = frozenset(),
) -> dict[str, list[dict[str, object]]]:
    modes: dict[str, list[dict[str, object]]] = {"rbr": [], "beamng": []}
    route_distances, route_sides = _route_outer_sides(route)
    source_losses = {
        source_index: _record_conversion_loss(record)
        for record in records
        if record.get("mode") == "rbr"
        and isinstance(
            source_index := record.get("sourceIndex"),
            int,
        )
    }
    for record in records:
        mode = record.get("mode")
        if mode not in modes or _record_marker_kind(record) in marker_kinds:
            continue
        entry = dict(record)
        if mode == "rbr":
            conversion_loss = _record_conversion_loss(record)
        else:
            conversion_loss = list(
                dict.fromkeys(
                    loss
                    for source_index in _record_source_indexes(record)
                    for loss in source_losses.get(source_index, [])
                )
            )
        if conversion_loss:
            entry["conversionLoss"] = conversion_loss
        entry["position"] = (
            _entry_position(route, float(entry["distance"]))
            if entry.get("mapVisible")
            else None
        )
        if entry["position"] is not None:
            outer_side = _outer_side_at_distance(
                float(entry["distance"]),
                route_distances,
                route_sides,
            )
            if outer_side is not None:
                entry["outerSide"] = outer_side
        modes[mode].append(entry)
    for entries in modes.values():
        entries.sort(
            key=lambda entry: (
                float(entry["distance"]),
                int(entry.get("sourceIndex", entry.get("entryIndex", 0))),
            )
        )
    return modes


def _view_box(
    route: list[tuple[np.ndarray, np.ndarray, float]],
    markers: list[dict[str, object]],
) -> list[float]:
    points = [
        (float(position[0]), -float(position[1]))
        for position, _direction, _distance in route
    ]
    points.extend(
        (float(marker["x"]), float(marker["y"]))
        for marker in markers
    )
    if not points:
        return [-50.0, -50.0, 100.0, 100.0]
    minimum_x = min(point[0] for point in points)
    maximum_x = max(point[0] for point in points)
    minimum_y = min(point[1] for point in points)
    maximum_y = max(point[1] for point in points)
    width = max(1.0, maximum_x - minimum_x)
    height = max(1.0, maximum_y - minimum_y)
    padding = max(width, height) * 0.08
    return [
        minimum_x - padding,
        minimum_y - padding,
        width + padding * 2,
        height + padding * 2,
    ]


def _json_for_script(value: object) -> str:
    return (
        json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        .replace("<", "\\u003c")
        .replace(">", "\\u003e")
        .replace("&", "\\u0026")
    )


def render_pacenote_visualizer(
    stage: RbrStage,
    route: list[tuple[np.ndarray, np.ndarray, float]],
    markers: Mapping[str, np.ndarray],
    rally: RallyNotebookResult,
) -> str:
    marker_data = _marker_data(markers, route)
    view_box = _view_box(route, marker_data)
    data = {
        "schemaVersion": 1,
        "converterVersion": __version__,
        "stage": {
            "title": stage.metadata.name,
            "source": stage.metadata.folder_name,
            "sourceFormat": stage.source_format,
            "variant": stage.source_variant or "base",
            "coordinateSpace": "beamng-local-metres",
            "northUpAxis": "+Y",
        },
        "route": _sample_route(route),
        "markers": marker_data,
        "modes": _visualizer_records(
            route,
            rally.visualizer_records,
            frozenset(str(marker["kind"]) for marker in marker_data),
        ),
        "viewBox": view_box,
        "markerRadius": _MARKER_RADIUS_METRES,
        "pacenoteLeaderWidthPixels": _PACENOTE_LEADER_WIDTH_PIXELS,
        "pacenoteLeaderLengthPixels": _PACENOTE_LEADER_LENGTH_PIXELS,
        "pacenoteLabelGapPixels": _PACENOTE_LABEL_GAP_PIXELS,
        "maxZoomSpanMetres": _MAX_ZOOM_SPAN_METRES,
        "roadWidthMetres": _ROAD_WIDTH_METRES,
    }
    return _HTML_TEMPLATE.replace(
        "__PACENOTE_VISUALIZER_DATA__",
        _json_for_script(data),
    )


_HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pacenote Visualizer</title>
<style>
:root {
  color-scheme: dark;
  font-family: "Segoe UI", system-ui, sans-serif;
  background: #111923;
  color: #edf3fb;
}
* { box-sizing: border-box; }
html, body { height: 100%; margin: 0; overflow: hidden; }
body { display: grid; grid-template-columns: minmax(390px, 34vw) 1fr; }
#sidebar {
  display: flex;
  min-height: 0;
  min-width: 0;
  flex-direction: column;
  border-right: 1px solid #344557;
  background: #182330;
}
#note-list {
  flex: 1;
  min-height: 0;
  overflow-x: hidden;
  overflow-y: scroll;
  overscroll-behavior: contain;
  scrollbar-gutter: stable;
}
#note-table {
  width: 100%;
  border-collapse: separate;
  border-spacing: 0;
  table-layout: fixed;
}
#note-table th {
  position: sticky;
  top: 0;
  z-index: 2;
  padding: 6px 4px;
  border-bottom: 1px solid #41576d;
  background: #182330;
}
#note-table .distance-heading {
  width: 62px;
  color: #9caebe;
  font: 600 10px/1.2 ui-monospace, "Cascadia Code", monospace;
  text-align: right;
}
.mode-header button {
  width: 100%;
  border: 1px solid #5c7188;
  border-radius: 4px;
  background: #172432;
  padding: 6px 4px;
  font: 700 11px/1 system-ui, sans-serif;
  cursor: pointer;
}
.mode-header button[data-mode="rbr"] { color: #8cdcff; }
.mode-header button[data-mode="beamng"] { color: #ffc19c; }
.mode-header button[data-mode="rbr"].active {
  border-color: #58b9ea;
  background: #2c648c;
  color: #fff;
}
.mode-header button[data-mode="beamng"].active {
  border-color: #ff9b62;
  background: #b54c12;
  color: #fff;
}
#note-table td {
  vertical-align: top;
  border-bottom: 1px solid #283b4e;
}
.distance-cell {
  padding: 8px 5px 0 3px;
  color: #9caebe;
  font: 10px/1.3 ui-monospace, "Cascadia Code", monospace;
  text-align: right;
  white-space: nowrap;
}
.note-cell {
  padding: 3px 2px;
}
.note-cell.empty {
  background: #1b2938;
}
.note-entry {
  --note-color: #78d5ff;
  display: block;
  width: 100%;
  margin: 0 0 3px;
  padding: 7px 8px;
  border: 0;
  border-left: 3px solid var(--note-color);
  border-radius: 3px;
  background: #213043;
  color: inherit;
  font: 12px/1.35 ui-monospace, "Cascadia Code", monospace;
  text-align: left;
  cursor: pointer;
  user-select: text;
}
.note-entry:last-child { margin-bottom: 0; }
.note-entry:hover:not(.selected), .note-entry:focus-visible:not(.selected) {
  background: #30445d;
  outline: none;
}
.note-entry.beamng { --note-color: #ff7b22; }
.note-entry.ignored { --note-color: #ff955a; color: #ffd5c0; }
.note-entry.unmapped { opacity: 0.66; }
.note-entry.conversion-loss {
  --note-color: #ff5363;
  color: #ffc0c6;
}
#map-panel {
  position: relative;
  min-height: 0;
  min-width: 0;
  overflow: hidden;
  background: #111923;
}
#map {
  position: relative;
  width: 100%;
  height: 100%;
  overflow: hidden;
  touch-action: none;
  cursor: grab;
}
#map.panning { cursor: grabbing; }
#map.rotating { cursor: all-scroll; }
#map.panning, #map.panning *,
#map.rotating, #map.rotating * {
  user-select: none;
}
#road-canvas {
  position: absolute;
  inset: 0;
  width: 100%;
  height: 100%;
  pointer-events: none;
}
#world-layer {
  position: absolute;
  inset: 0;
  --map-hit-size: 10px;
  --map-note-leader-width: 1px;
  --map-note-tick-width: 2px;
  --map-marker-selected-size: 4px;
  --map-marker-selected-stroke: 0.2px;
  --map-selected-label-padding-y: 2px;
  --map-selected-label-padding-x: 4px;
  --map-selected-label-border: 1px;
  --map-selected-label-radius: 3px;
  --map-label-size: 13px;
  --map-label-rotation: 0rad;
  transform-origin: 0 0;
  pointer-events: none;
}
.marker, .note {
  position: absolute;
  width: 0;
  height: 0;
  pointer-events: auto;
}
.marker { --note-color: #fff; }
.marker.start { --note-color: #00ff00; }
.marker.time-control { --note-color: #f5c43e; }
.marker.stop { --note-color: #ff0000; }
.marker.split { --note-color: #4da3e8; }
.marker-dot {
  position: absolute;
  border: 0.8px solid #071018;
  border-radius: 50%;
  transform: translate(-50%, -50%);
}
.marker-dot { background: var(--note-color); }
.marker.finish .marker-dot {
  background: conic-gradient(#000 25%, #fff 0 50%, #000 0 75%, #fff 0);
}
.marker-label, .note-label {
  position: absolute;
  z-index: 1;
  white-space: nowrap;
  color: #f6fbff;
  font-family: system-ui, sans-serif;
  font-size: var(--map-label-size);
  font-weight: 600;
  line-height: 1.2;
  text-shadow: 0 0 1px #071018, 0 0 1px #071018, 0 0 1px #071018;
  user-select: none;
  pointer-events: auto;
  cursor: pointer;
}
.marker-label {
  transform: rotate(var(--map-label-rotation));
  transform-origin: left center;
}
.note { --note-color: #78d5ff; color: var(--note-color); }
.note.beamng { --note-color: #ff7b22; }
.note.ignored { --note-color: #ff955a; }
.note.conversion-loss { --note-color: #ff5363; }
.note.missing { --note-color: #94636a; }
.note-tick, .note-leader, .note-elbow {
  position: absolute;
  height: 0;
  transform-origin: left center;
}
.note-tick { border-top: var(--map-note-tick-width) solid currentColor; }
.note-leader, .note-elbow {
  border-top: var(--map-note-leader-width) solid currentColor;
}
.note-leader-hit, .note-elbow-hit {
  position: absolute;
  height: var(--map-hit-size);
  margin-top: calc(var(--map-hit-size) * -0.5);
  background: transparent;
  transform-origin: left center;
  cursor: pointer;
}
.note-label {
  --map-label-anchor: 0%;
  color: #b9ebff;
  font-weight: 400;
  transform: rotate(var(--map-label-rotation)) translate(var(--map-label-anchor), -50%);
  transform-origin: left center;
}
.note.beamng .note-label { color: #ffd0b2; }
.note.ignored .note-label { color: #ffd5c0; }
.note.conversion-loss .note-label { color: #ff9ea8; }
.note.missing .note-label { color: #b58f94; }
.marker.selected .marker-dot::after {
  content: "";
  position: absolute;
  left: 50%;
  top: 50%;
  width: var(--map-marker-selected-size);
  height: var(--map-marker-selected-size);
  box-sizing: border-box;
  border: var(--map-marker-selected-stroke) solid var(--note-color);
  border-radius: 50%;
  transform: translate(-50%, -50%);
  pointer-events: none;
}
.note.selected .note-tick, .note.selected .note-leader, .note.selected .note-elbow {
  border-top-color: var(--note-color);
}
.note.selected .note-label,
.marker.selected .marker-label {
  padding: var(--map-selected-label-padding-y) var(--map-selected-label-padding-x);
  border: var(--map-selected-label-border) solid #071018;
  border-radius: var(--map-selected-label-radius);
  background: var(--note-color);
  box-shadow: 0 0 0 var(--map-selected-label-border) var(--note-color);
  color: #071018;
  font-weight: 700;
  text-shadow: none;
}
.note.hovered:not(.selected) .note-tick,
.note.hovered:not(.selected) .note-leader,
.note.hovered:not(.selected) .note-elbow {
  border-top-color: #fff;
}
.note.hovered:not(.selected) .note-label {
  color: #fff;
  font-weight: 700;
}
.note.conversion-loss.hovered:not(.selected) .note-tick,
.note.conversion-loss.hovered:not(.selected) .note-leader,
.note.conversion-loss.hovered:not(.selected) .note-elbow {
  border-top-color: #ff9ea8;
}
.note.conversion-loss.hovered:not(.selected) .note-label { color: #ffbec4; }
.note-entry.selected {
  background: #263849;
  border-left-width: 5px;
  outline: 2px solid var(--note-color);
  outline-offset: -1px;
  color: var(--note-color);
  font-weight: 700;
}
.note-entry.associated:not(.selected) {
  background: #293c50;
  outline: 1px solid var(--note-color);
  outline-offset: -1px;
  color: var(--note-color);
}
.note-entry.hovered:not(.selected) {
  background: #31516c;
  outline: 1px solid var(--note-color);
}
.note-entry.marker-entry { --note-color: #fff; }
.note-entry.marker-entry.start { --note-color: #00ff00; }
.note-entry.marker-entry.time-control { --note-color: #f5c43e; }
.note-entry.marker-entry.stop { --note-color: #ff0000; }
.note-entry.marker-entry.split { --note-color: #4da3e8; }
#map-controls {
  position: absolute;
  z-index: 3;
  top: 12px;
  left: 12px;
  right: 12px;
  display: flex;
  min-width: 0;
  align-items: center;
  gap: 6px;
  pointer-events: auto;
}
#reset-view {
  border: 0;
  padding: 7px 10px;
  font: 600 12px system-ui, sans-serif;
  cursor: pointer;
}
#reset-view {
  flex: 0 0 auto;
  border: 1px solid #5c7188;
  border-radius: 5px;
  background: #172432;
  color: #c9d6e4;
  box-shadow: 0 2px 10px #0008;
}
#orientation-toggle {
  flex: 0 0 auto;
  min-width: 42px;
  border: 1px solid #5c7188;
  border-radius: 5px;
  background: #172432;
  color: #f0f5fa;
  font: 700 13px/1 system-ui, sans-serif;
  text-align: center;
  text-shadow: 0 1px 3px #000;
  cursor: pointer;
}
#orientation-arrow { display: block; font-size: 24px; }
#orientation-toggle[aria-pressed="true"] { background: #3d6f93; }
#stage-title {
  flex: 1 1 auto;
  min-width: 0;
  margin: 0 0 0 4px;
  overflow: hidden;
  color: #edf3fb;
  font-size: 15px;
  line-height: 1.2;
  text-overflow: ellipsis;
  white-space: nowrap;
}
@media (max-width: 700px) {
  body { grid-template-columns: minmax(300px, 48vw) 1fr; }
  #stage-title { font-size: 13px; }
  .note-entry { font-size: 11px; }
}
</style>
</head>
<body>
<aside id="sidebar">
  <div id="note-list" aria-label="Pacenote comparison">
    <table id="note-table">
      <thead>
        <tr>
          <th scope="col" class="distance-heading">Distance</th>
          <th scope="col" class="mode-header">
            <button type="button" data-mode="rbr" class="active">RBR</button>
          </th>
          <th scope="col" class="mode-header">
            <button type="button" data-mode="beamng">BEAMNG</button>
          </th>
        </tr>
      </thead>
      <tbody id="note-table-body"></tbody>
    </table>
  </div>
</aside>
<main id="map-panel">
  <div id="map-controls">
    <button type="button" id="reset-view">Fit map</button>
    <button type="button" id="orientation-toggle" aria-pressed="true">
      <span id="orientation-label">ROAD</span>
      <span id="orientation-arrow">↑</span>
    </button>
    <h1 id="stage-title">Pacenotes</h1>
  </div>
  <div id="map" tabindex="0" autofocus role="region" aria-label="Top-down road and pacenote map">
    <canvas id="road-canvas" aria-hidden="true"></canvas>
    <div id="world-layer"></div>
  </div>
</main>
<script id="pacenote-visualizer-data" type="application/json">__PACENOTE_VISUALIZER_DATA__</script>
<script>
(() => {
  const data = JSON.parse(document.getElementById("pacenote-visualizer-data").textContent);
  const map = document.getElementById("map");
  const canvas = document.getElementById("road-canvas");
  const worldLayer = document.getElementById("world-layer");
  const noteList = document.getElementById("note-list");
  const noteTableBody = document.getElementById("note-table-body");
  const modeButtons = document.querySelectorAll("#note-table button[data-mode]");
  const resetView = document.getElementById("reset-view");
  const orientationToggle = document.getElementById("orientation-toggle");
  const orientationLabel = document.getElementById("orientation-label");
  let mode = "rbr";
  let orientation = "road";
  let orientationPoint = null;
  let orientationEntryKey = null;
  let animatedRoadAngle = null;
  let rotationOffset = 0;
  let selectedKey = null;
  let hoveredKey = null;
  let entryElements = new Map();
  let activeMapEntries = [];
  let displayedEntries = [];
  let modeEntries = { rbr: [], beamng: [] };
  let missingEntries = [];
  let markerEntries = [];
  let tableRows = [];
  let relatedKeysByEntryKey = new Map();
  let tableRowKeysByEntryKey = new Map();
  let selectedRelatedKeys = new Set();
  let selectedTableRowKey = null;
  let labelAnchors = [];
  let labelLayoutAngle = null;
  let labelLayoutScale = null;
  let labelCollisionLayout = null;
  let labelLayoutDirty = true;
  let view = { x: data.viewBox[0], y: data.viewBox[1], width: data.viewBox[2], height: data.viewBox[3] };
  let drag = null;
  let focusAnimationFrame = null;
  const focusDuration = 600;
  const defaultViewSpan = 500;
  const labelCollisionMargin = 2;
  const labelVerticalOverlapAllowance = 2;
  const maxLabelLayoutPasses = 8;

  document.title = `${data.stage.title} Pacenote Visualizer - RBR2BeamNG v${data.converterVersion}`;
  document.getElementById("stage-title").textContent = data.stage.title;

  function element(name, className = "") {
    const node = document.createElement(name);
    node.className = className;
    return node;
  }

  function rotate(x, y, angle) {
    const cosine = Math.cos(angle);
    const sine = Math.sin(angle);
    return {
      x: x * cosine - y * sine,
      y: x * sine + y * cosine,
    };
  }

  function roadAngleAt(x, y) {
    let nearest = null;
    for (let index = 1; index < data.route.length; index += 1) {
      const start = data.route[index - 1];
      const end = data.route[index];
      const directionX = end[0] - start[0];
      const directionY = end[1] - start[1];
      const lengthSquared = directionX * directionX + directionY * directionY;
      if (!lengthSquared) {
        continue;
      }
      const fraction = Math.max(
        0,
        Math.min(
          1,
          ((x - start[0]) * directionX + (y - start[1]) * directionY) / lengthSquared
        )
      );
      const nearestX = start[0] + directionX * fraction;
      const nearestY = start[1] + directionY * fraction;
      const distanceSquared = (x - nearestX) ** 2 + (y - nearestY) ** 2;
      if (!nearest || distanceSquared < nearest.distanceSquared) {
        nearest = { directionX, directionY, distanceSquared };
      }
    }
    return nearest
      ? -Math.PI / 2 - Math.atan2(nearest.directionY, nearest.directionX)
      : 0;
  }

  function entryRoadAngle(entry) {
    if (!entry?.position) {
      return null;
    }
    const index = displayedEntries.findIndex(candidate =>
      entryKey(candidate) === entryKey(entry)
    );
    let averageX = 0;
    let averageY = 0;
    let count = 0;
    for (
      let previousIndex = index - 1;
      previousIndex >= 0 && count < 10;
      previousIndex -= 1
    ) {
      const previous = displayedEntries[previousIndex];
      if (!previous?.position) {
        continue;
      }
      const directionX = entry.position.x - previous.position.x;
      const directionY = entry.position.y - previous.position.y;
      const length = Math.hypot(directionX, directionY);
      if (length) {
        averageX += directionX / length;
        averageY += directionY / length;
        count += 1;
      }
    }
    return count && (averageX || averageY)
      ? -Math.PI / 2 - Math.atan2(averageY, averageX)
      : null;
  }

  function viewCenter() {
    return {
      x: view.x + view.width * 0.5,
      y: view.y + view.height * 0.5,
    };
  }

  function currentRoadAngle(ignoreAnimation = false) {
    if (!ignoreAnimation && animatedRoadAngle !== null) {
      return animatedRoadAngle;
    }
    const orientationEntry = orientationEntryKey === null
      ? null
      : displayedEntries.find(entry => entryKey(entry) === orientationEntryKey);
    const entryAngle = entryRoadAngle(orientationEntry);
    if (entryAngle !== null) {
      return entryAngle;
    }
    const center = viewCenter();
    return roadAngleAt(
      orientationPoint?.x ?? center.x,
      orientationPoint?.y ?? center.y
    );
  }

  function shortestAngleDelta(from, to) {
    return Math.atan2(Math.sin(to - from), Math.cos(to - from));
  }

  function defaultViewAt(position) {
    const rect = map.getBoundingClientRect();
    const aspect = rect.width > 0 && rect.height > 0
      ? rect.width / rect.height
      : 1;
    const width = aspect >= 1 ? defaultViewSpan : defaultViewSpan * aspect;
    const height = aspect >= 1 ? defaultViewSpan / aspect : defaultViewSpan;
    return {
      x: position.x - width * 0.5,
      y: position.y - height * 0.5,
      width,
      height,
    };
  }

  function viewport() {
    const rect = map.getBoundingClientRect();
    const scale = Math.min(rect.width / view.width, rect.height / view.height);
    const center = viewCenter();
    return {
      rect,
      scale,
      worldCenterX: center.x,
      worldCenterY: center.y,
      screenCenterX: rect.width * 0.5,
      screenCenterY: rect.height * 0.5,
      angle: (
        orientation === "road"
          ? currentRoadAngle()
          : 0
      ) + rotationOffset,
    };
  }

  function updateOrientationToggle() {
    const roadUp = orientation === "road";
    orientationLabel.textContent = roadUp ? "ROAD" : "N";
    orientationToggle.setAttribute("aria-pressed", String(roadUp));
    orientationToggle.title = roadUp
      ? "Selected road direction is up. Click for north up. Ctrl-drag to rotate."
      : "North is up. Click for selected road direction up. Ctrl-drag to rotate.";
  }

  function hasUsableViewport(viewportData) {
    return (
      viewportData.rect.width >= 1
      && viewportData.rect.height >= 1
      && Number.isFinite(viewportData.scale)
      && viewportData.scale > 0
    );
  }

  function updateWorldTransform(viewportData) {
    const {
      scale,
      worldCenterX,
      worldCenterY,
      screenCenterX,
      screenCenterY,
      angle,
    } = viewportData;
    const inverseScale = Math.min(1 / scale, 40);
    const pacenoteLeaderWidth =
      data.pacenoteLeaderWidthPixels * inverseScale;
    worldLayer.style.setProperty(
      "--map-label-size",
      `${13 * inverseScale}px`
    );
    worldLayer.style.setProperty(
      "--map-hit-size",
      `${10 * inverseScale}px`
    );
    worldLayer.style.setProperty(
      "--map-note-leader-width",
      `${pacenoteLeaderWidth}px`
    );
    worldLayer.style.setProperty(
      "--map-note-tick-width",
      `${pacenoteLeaderWidth * 2}px`
    );
    worldLayer.style.setProperty(
      "--map-marker-selected-size",
      `${data.markerRadius * 2.8}px`
    );
    worldLayer.style.setProperty(
      "--map-marker-selected-stroke",
      `${pacenoteLeaderWidth}px`
    );
    worldLayer.style.setProperty(
      "--map-selected-label-padding-y",
      `${2 * inverseScale}px`
    );
    worldLayer.style.setProperty(
      "--map-selected-label-padding-x",
      `${4 * inverseScale}px`
    );
    worldLayer.style.setProperty(
      "--map-selected-label-border",
      `${1 * inverseScale}px`
    );
    worldLayer.style.setProperty(
      "--map-selected-label-radius",
      `${3 * inverseScale}px`
    );
    worldLayer.style.setProperty(
      "--map-label-rotation",
      `${-angle}rad`
    );
    worldLayer.style.transform =
      `translate(${screenCenterX}px, ${screenCenterY}px) rotate(${angle}rad) scale(${scale}) translate(${-worldCenterX}px, ${-worldCenterY}px)`;
    updateLabelGeometry(viewportData, inverseScale);
    updateLabelAlignment(viewportData);
    updateLabelLayout(viewportData);
  }

  function updateLabelGeometry(viewportData, inverseScale) {
    if (labelLayoutScale === viewportData.scale) {
      return;
    }
    labelLayoutScale = viewportData.scale;
    const leaderLength = data.pacenoteLeaderLengthPixels * inverseScale;
    const labelGap = data.pacenoteLabelGapPixels * inverseScale;
    for (const anchor of labelAnchors) {
      const labelOffset = anchor.roadHalfWidth + leaderLength + labelGap;
      anchor.baseLabelX = anchor.outwardX * labelOffset;
      anchor.baseLabelY = anchor.outwardY * labelOffset;
      anchor.leader.style.width = `${leaderLength + labelGap}px`;
      anchor.leaderHit.style.width = anchor.leader.style.width;
      anchor.label.style.left = `${anchor.baseLabelX}px`;
      anchor.label.style.top = `${anchor.baseLabelY}px`;
    }
  }

  function updateLabelAlignment(viewportData) {
    if (labelLayoutAngle === viewportData.angle) {
      return;
    }
    labelLayoutAngle = viewportData.angle;
    for (const anchor of labelAnchors) {
      const direction = rotate(anchor.outwardX, anchor.outwardY, viewportData.angle);
      const anchorOffset = (1 - Math.max(-1, Math.min(1, direction.x))) * 0.5;
      anchor.label.style.setProperty(
        "--map-label-anchor",
        `${-anchorOffset * 100}%`
      );
    }
  }

  function invalidateLabelLayout() {
    labelLayoutDirty = true;
  }

  function screenRect(rect) {
    return {
      left: rect.left,
      top: rect.top,
      right: rect.right,
      bottom: rect.bottom,
      width: rect.width,
      height: rect.height,
    };
  }

  function offsetScreenRect(rect, offsetY) {
    return {
      left: rect.left,
      top: rect.top + offsetY,
      right: rect.right,
      bottom: rect.bottom + offsetY,
      width: rect.width,
      height: rect.height,
    };
  }

  function rectsOverlap(left, right) {
    return (
      left.left < right.right + labelCollisionMargin
      && left.right + labelCollisionMargin > right.left
      && left.top < right.bottom - labelVerticalOverlapAllowance
      && left.bottom > right.top + labelVerticalOverlapAllowance
    );
  }

  function verticalClearance(first, second) {
    return Math.max(
      0,
      Math.min(first.bottom, second.bottom)
        - Math.max(first.top, second.top)
        - labelVerticalOverlapAllowance
    );
  }

  function verticalDirection(first, second, tieBreak) {
    const firstCenter = (first.top + first.bottom) * 0.5;
    const secondCenter = (second.top + second.bottom) * 0.5;
    if (Math.abs(firstCenter - secondCenter) > 0.01) {
      return firstCenter < secondCenter ? -1 : 1;
    }
    return tieBreak % 2 ? 1 : -1;
  }

  function resolveLabelOffsets(baseRects, markerRects) {
    const offsets = baseRects.map(() => 0);
    for (let pass = 0; pass < maxLabelLayoutPasses; pass += 1) {
      const rects = baseRects.map((rect, index) =>
        offsetScreenRect(rect, offsets[index])
      );
      const adjustments = baseRects.map(() => 0);
      let foundCollision = false;

      for (let index = 0; index < rects.length; index += 1) {
        for (const markerRect of markerRects) {
          if (!rectsOverlap(rects[index], markerRect)) {
            continue;
          }
          foundCollision = true;
          const clearance = verticalClearance(rects[index], markerRect) + 0.25;
          adjustments[index] += (
            verticalDirection(rects[index], markerRect, index) * clearance
          );
        }
      }

      for (let first = 0; first < rects.length; first += 1) {
        for (let second = first + 1; second < rects.length; second += 1) {
          if (!rectsOverlap(rects[first], rects[second])) {
            continue;
          }
          foundCollision = true;
          const halfClearance = (
            verticalClearance(rects[first], rects[second]) + 0.25
          ) * 0.5;
          const direction = verticalDirection(rects[first], rects[second], first);
          adjustments[first] += direction * halfClearance;
          adjustments[second] -= direction * halfClearance;
        }
      }

      if (!foundCollision) {
        break;
      }
      let moved = false;
      for (let index = 0; index < offsets.length; index += 1) {
        const nextOffset = offsets[index] + adjustments[index];
        moved = moved || Math.abs(nextOffset - offsets[index]) > 0.01;
        offsets[index] = nextOffset;
      }
      if (!moved) {
        break;
      }
    }
    return offsets;
  }

  function positionLabelAnchor(anchor, offsetY, viewportData) {
    const worldOffset = rotate(
      0,
      offsetY / viewportData.scale,
      -viewportData.angle
    );
    const elbowLength = Math.hypot(worldOffset.x, worldOffset.y);
    const elbowRotation = Math.atan2(worldOffset.y, worldOffset.x) * 180 / Math.PI;
    anchor.label.style.left = `${anchor.baseLabelX + worldOffset.x}px`;
    anchor.label.style.top = `${anchor.baseLabelY + worldOffset.y}px`;
    anchor.elbow.style.left = `${anchor.baseLabelX}px`;
    anchor.elbow.style.top = `${anchor.baseLabelY}px`;
    anchor.elbow.style.width = `${elbowLength}px`;
    anchor.elbow.style.transform = `rotate(${elbowRotation}deg)`;
    anchor.elbowHit.style.left = anchor.elbow.style.left;
    anchor.elbowHit.style.top = anchor.elbow.style.top;
    anchor.elbowHit.style.width = anchor.elbow.style.width;
    anchor.elbowHit.style.transform = anchor.elbow.style.transform;
  }

  function updateLabelLayout(viewportData) {
    const { scale, angle } = viewportData;
    const unchanged = (
      !labelLayoutDirty
      && labelCollisionLayout?.scale === scale
      && labelCollisionLayout?.angle === angle
    );
    if (unchanged) {
      return;
    }
    labelLayoutDirty = false;
    labelCollisionLayout = {
      scale,
      angle,
    };

    for (const anchor of labelAnchors) {
      anchor.label.style.left = `${anchor.baseLabelX}px`;
      anchor.label.style.top = `${anchor.baseLabelY}px`;
      anchor.elbow.style.width = "0px";
      anchor.elbowHit.style.width = "0px";
    }

    const markerRects = Array.from(
      worldLayer.querySelectorAll(".marker-label"),
      label => screenRect(label.getBoundingClientRect())
    ).filter(rect => rect.width && rect.height);
    const baseRects = labelAnchors.map(anchor =>
      screenRect(anchor.label.getBoundingClientRect())
    );
    const offsets = resolveLabelOffsets(baseRects, markerRects);
    for (const [index, anchor] of labelAnchors.entries()) {
      positionLabelAnchor(anchor, offsets[index], viewportData);
    }
  }

  function drawRoad(viewportData) {
    const {
      rect,
      scale,
      worldCenterX,
      worldCenterY,
      screenCenterX,
      screenCenterY,
      angle,
    } = viewportData;
    const pixelRatio = window.devicePixelRatio || 1;
    const width = Math.max(1, Math.round(rect.width * pixelRatio));
    const height = Math.max(1, Math.round(rect.height * pixelRatio));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const context = canvas.getContext("2d");
    context.setTransform(pixelRatio, 0, 0, pixelRatio, 0, 0);
    context.clearRect(0, 0, rect.width, rect.height);
    if (!data.route.length) {
      return;
    }
    const project = point => {
      const rotated = rotate(
        (point[0] - worldCenterX) * scale,
        (point[1] - worldCenterY) * scale,
        angle
      );
      return {
        x: screenCenterX + rotated.x,
        y: screenCenterY + rotated.y,
      };
    };
    context.beginPath();
    data.route.forEach((point, index) => {
      const projected = project(point);
      if (index) {
        context.lineTo(projected.x, projected.y);
      } else {
        context.moveTo(projected.x, projected.y);
      }
    });
    const roadWidth = data.roadWidthMetres * scale;
    context.lineCap = "round";
    context.lineJoin = "round";
    context.strokeStyle = "#f4f7fb";
    context.lineWidth = roadWidth;
    context.stroke();
  }

  function updateMap() {
    updateOrientationToggle();
    const viewportData = viewport();
    if (!hasUsableViewport(viewportData)) {
      return;
    }
    updateWorldTransform(viewportData);
    drawRoad(viewportData);
  }

  function drawMarkers() {
    const fragment = document.createDocumentFragment();
    data.markers.forEach((marker, markerIndex) => {
      const entry = displayedEntries.find(candidate =>
        candidate.marker && candidate.markerIndex === markerIndex
      );
      if (!entry) {
        return;
      }
      const key = entryKey(entry);
      const radius = marker.kind === "split" ? data.markerRadius * 0.65 : data.markerRadius;
      const group = element(
        "div",
        `marker ${marker.kind}`
      );
      group.dataset.entryKey = key;
      registerEntryElement(key, group);
      activeMapEntries.push({ key, entry });
      group.style.left = `${marker.x}px`;
      group.style.top = `${marker.y}px`;
      group.addEventListener("click", event => {
        event.stopPropagation();
        activateEntry(key, entry);
      });
      const dot = element("div", "marker-dot");
      dot.style.width = `${radius * 2}px`;
      dot.style.height = `${radius * 2}px`;
      const label = element("span", "marker-label selectable");
      label.style.left = `${radius * 1.3}px`;
      label.style.top = `${-radius * 1.3}px`;
      label.textContent = marker.label;
      group.append(dot, label);
      fragment.append(group);
    });
    worldLayer.append(fragment);
    updateSelectionState();
  }

  function formatDistance(distance) {
    return `${Math.round(distance * 10) / 10} m`;
  }

  function cancelFocusAnimation(preserveRoadOrientation = true) {
    if (focusAnimationFrame !== null) {
      cancelAnimationFrame(focusAnimationFrame);
      focusAnimationFrame = null;
    }
    if (animatedRoadAngle !== null) {
      if (preserveRoadOrientation && orientation === "road") {
        const targetRoadAngle = currentRoadAngle(true);
        rotationOffset += animatedRoadAngle - targetRoadAngle;
      }
      animatedRoadAngle = null;
    }
  }

  function focusEase(progress) {
    return 1 - (1 - progress) ** 3;
  }

  function focus(entry, animate = true, startRoadAngle = null) {
    if (!entry.position) {
      return;
    }
    const initialRoadAngle = startRoadAngle ?? currentRoadAngle();
    cancelFocusAnimation(false);
    const targetX = entry.position.x - view.width * 0.5;
    const targetY = entry.position.y - view.height * 0.5;
    const entryAngle = entryRoadAngle(entry);
    const targetRoadAngle = entryAngle === null
      ? roadAngleAt(entry.position.x, entry.position.y)
      : entryAngle;
    const roadAngleDelta = shortestAngleDelta(
      initialRoadAngle,
      targetRoadAngle
    );
    const movesView = view.x !== targetX || view.y !== targetY;
    const rotatesView = orientation === "road" && Math.abs(roadAngleDelta) > 0.001;
    if (!animate || (!movesView && !rotatesView)) {
      view.x = targetX;
      view.y = targetY;
      animatedRoadAngle = null;
      updateMap();
      return;
    }
    const startX = view.x;
    const startY = view.y;
    const startTime = performance.now();
    if (orientation === "road") {
      animatedRoadAngle = initialRoadAngle;
    }
    const animateFocus = now => {
      const progress = Math.min(1, (now - startTime) / focusDuration);
      const eased = focusEase(progress);
      view.x = startX + (targetX - startX) * eased;
      view.y = startY + (targetY - startY) * eased;
      if (orientation === "road") {
        animatedRoadAngle = initialRoadAngle + roadAngleDelta * eased;
      }
      updateMap();
      if (progress < 1) {
        focusAnimationFrame = requestAnimationFrame(animateFocus);
      } else {
        focusAnimationFrame = null;
        animatedRoadAngle = null;
      }
    };
    focusAnimationFrame = requestAnimationFrame(animateFocus);
  }

  function activateEntry(
    key,
    entry,
    animate = true,
    startRoadAngle = null,
    tableRowKey = null
  ) {
    const initialRoadAngle = orientation === "road"
      ? (startRoadAngle ?? currentRoadAngle())
      : null;
    selectEntry(key, entry, tableRowKey);
    focus(entry, animate, initialRoadAngle);
  }

  function selectEntry(key, entry, tableRowKey = null) {
    selectedKey = key;
    selectedRelatedKeys = new Set(
      relatedKeysByEntryKey.get(key) ?? [key]
    );
    selectedTableRowKey = tableRowKey
      ?? tableRowKeysByEntryKey.get(key)?.[0]
      ?? null;
    orientationEntryKey = entry.position ? key : null;
    if (entry.position) {
      orientationPoint = {
        x: entry.position.x,
        y: entry.position.y,
      };
    }
    updateSelectionState();
    const row = selectedTableRowKey === null
      ? noteList.querySelector(`[data-entry-key="${key}"]`)
      : noteList.querySelector(
        `[data-row-key="${selectedTableRowKey}"] [data-entry-key="${key}"]`
      );
    if (row) {
      row.scrollIntoView({ block: "center", behavior: "smooth" });
    }
  }

  function updateSelectionState() {
    for (const [key, nodes] of entryElements) {
      const selected = key === selectedKey;
      const associated = !selected && selectedRelatedKeys.has(key);
      for (const node of nodes) {
        node.classList.toggle("selected", selected);
        node.classList.toggle("associated", associated);
      }
    }
    invalidateLabelLayout();
  }

  function registerEntryElement(key, node) {
    const nodes = entryElements.get(key);
    if (nodes) {
      nodes.push(node);
    } else {
      entryElements.set(key, [node]);
    }
  }

  function setHovered(key) {
    if (hoveredKey === key) {
      return;
    }
    if (hoveredKey !== null) {
      for (const node of entryElements.get(hoveredKey) ?? []) {
        node.classList.remove("hovered");
      }
    }
    hoveredKey = key;
    if (hoveredKey !== null) {
      for (const node of entryElements.get(hoveredKey) ?? []) {
        node.classList.add("hovered");
      }
    }
    invalidateLabelLayout();
    updateMap();
  }

  function highlightClosestEntry(event) {
    const point = pointAtEvent(event);
    let closest = null;
    for (const candidate of activeMapEntries) {
      const deltaX = point.x - candidate.entry.position.x;
      const deltaY = point.y - candidate.entry.position.y;
      const distanceSquared = deltaX * deltaX + deltaY * deltaY;
      if (!closest || distanceSquared < closest.distanceSquared) {
        closest = { key: candidate.key, distanceSquared };
      }
    }
    setHovered(closest?.key ?? null);
  }

  function entryKey(entry) {
    return entry.marker
      ? `marker:${entry.markerIndex}`
      : `${entry.mode}:${entry.pacenoteIndex}`;
  }

  function sourceIndexesForEntry(entry) {
    const sourceIndexes = [];
    if (Number.isInteger(entry.sourceIndex)) {
      sourceIndexes.push(entry.sourceIndex);
    }
    if (
      entry.mode === "beamng"
      && Array.isArray(entry.contributingSourceIndexes)
    ) {
      for (const sourceIndex of entry.contributingSourceIndexes) {
        if (Number.isInteger(sourceIndex)) {
          sourceIndexes.push(sourceIndex);
        }
      }
    }
    return [...new Set(sourceIndexes)];
  }

  function entrySort(left, right) {
    const distance = Number(left.distance) - Number(right.distance);
    if (distance) {
      return distance;
    }
    if (left.marker !== right.marker) {
      return left.marker ? -1 : 1;
    }
    return (
      (left.marker ? left.markerIndex : left.pacenoteIndex)
      - (right.marker ? right.markerIndex : right.pacenoteIndex)
    );
  }

  function buildTableRows(linkedSourceIndexesByBeamngKey) {
    const rows = [];
    const rowsBySourceIndex = new Map();
    for (const entry of modeEntries.rbr) {
      const sourceIndex = sourceIndexesForEntry(entry)[0];
      const key = Number.isInteger(sourceIndex)
        ? `source:${sourceIndex}`
        : `rbr:${entry.pacenoteIndex}`;
      const row = {
        key,
        distance: Number(entry.distance),
        rbr: [entry],
        beamng: [],
        marker: null,
      };
      rows.push(row);
      if (Number.isInteger(sourceIndex)) {
        rowsBySourceIndex.set(sourceIndex, row);
      }
    }
    for (const entry of modeEntries.beamng) {
      const sourceIndexes = linkedSourceIndexesByBeamngKey.get(entryKey(entry));
      if (sourceIndexes?.size) {
        for (const sourceIndex of sourceIndexes) {
          rowsBySourceIndex.get(sourceIndex)?.beamng.push(entry);
        }
      } else {
        rows.push({
          key: `beamng:${entry.pacenoteIndex}`,
          distance: Number(entry.distance),
          rbr: [],
          beamng: [entry],
          marker: null,
        });
      }
    }
    for (const marker of markerEntries) {
      rows.push({
        key: entryKey(marker),
        distance: Number(marker.distance),
        rbr: [],
        beamng: [],
        marker,
      });
    }
    return rows.sort((left, right) => {
      const distance = left.distance - right.distance;
      if (distance) {
        return distance;
      }
      if (Boolean(left.marker) !== Boolean(right.marker)) {
        return left.marker ? -1 : 1;
      }
      return left.key.localeCompare(right.key);
    });
  }

  function initializeEntries() {
    modeEntries = {
      rbr: data.modes.rbr.map((entry, pacenoteIndex) => Object.assign(
        {},
        entry,
        { mode: "rbr", marker: false, pacenoteIndex }
      )),
      beamng: data.modes.beamng.map((entry, pacenoteIndex) => Object.assign(
        {},
        entry,
        { mode: "beamng", marker: false, pacenoteIndex }
      )),
    };
    markerEntries = data.markers.map((marker, markerIndex) => Object.assign(
      {},
      marker,
      { mode: "marker", marker: true, markerIndex }
    ));
    const rbrEntriesBySourceIndex = new Map();
    for (const entry of modeEntries.rbr) {
      for (const sourceIndex of sourceIndexesForEntry(entry)) {
        rbrEntriesBySourceIndex.set(sourceIndex, entry);
      }
    }
    const beamngEntriesByIndex = new Map(
      modeEntries.beamng.map(entry => [entry.entryIndex, entry])
    );
    const linkedSourceIndexesByBeamngKey = new Map();
    const linkSourceToBeamng = (sourceIndex, entry) => {
      if (!rbrEntriesBySourceIndex.has(sourceIndex)) {
        return;
      }
      const key = entryKey(entry);
      let sourceIndexes = linkedSourceIndexesByBeamngKey.get(key);
      if (!sourceIndexes) {
        sourceIndexes = new Set();
        linkedSourceIndexesByBeamngKey.set(key, sourceIndexes);
      }
      sourceIndexes.add(sourceIndex);
    };
    for (const entry of modeEntries.beamng) {
      for (const sourceIndex of sourceIndexesForEntry(entry)) {
        linkSourceToBeamng(sourceIndex, entry);
      }
    }
    for (const entry of modeEntries.rbr) {
      const sourceIndex = sourceIndexesForEntry(entry)[0];
      if (!Number.isInteger(sourceIndex) || !Array.isArray(entry.beamngEntryIndexes)) {
        continue;
      }
      for (const entryIndex of entry.beamngEntryIndexes) {
        const beamngEntry = beamngEntriesByIndex.get(entryIndex);
        if (beamngEntry) {
          linkSourceToBeamng(sourceIndex, beamngEntry);
        }
      }
    }
    relatedKeysByEntryKey = new Map();
    for (const entry of [
      ...modeEntries.rbr,
      ...modeEntries.beamng,
      ...markerEntries,
    ]) {
      const key = entryKey(entry);
      relatedKeysByEntryKey.set(key, new Set([key]));
    }
    for (const entry of modeEntries.beamng) {
      const beamngKey = entryKey(entry);
      for (
        const sourceIndex of linkedSourceIndexesByBeamngKey.get(beamngKey) ?? []
      ) {
        const rbrEntry = rbrEntriesBySourceIndex.get(sourceIndex);
        if (!rbrEntry) {
          continue;
        }
        const rbrKey = entryKey(rbrEntry);
        relatedKeysByEntryKey.get(beamngKey)?.add(rbrKey);
        relatedKeysByEntryKey.get(rbrKey)?.add(beamngKey);
      }
    }
    missingEntries = modeEntries.rbr.filter(entry =>
      entryConversionLoss(entry).length
      && relatedKeysByEntryKey.get(entryKey(entry)).size === 1
    );
    tableRows = buildTableRows(linkedSourceIndexesByBeamngKey);
    tableRowKeysByEntryKey = new Map();
    for (const row of tableRows) {
      const entries = row.marker
        ? [row.marker]
        : [...row.rbr, ...row.beamng];
      for (const entry of entries) {
        const key = entryKey(entry);
        const rowKeys = tableRowKeysByEntryKey.get(key);
        if (rowKeys) {
          rowKeys.push(row.key);
        } else {
          tableRowKeysByEntryKey.set(key, [row.key]);
        }
      }
    }
  }

  function nearestPacenote(referenceDistance) {
    let nearest = null;
    for (const entry of displayedEntries) {
      if (entry.marker || !entry.position) {
        continue;
      }
      const distance = Math.abs(
        Number(entry.distance) - referenceDistance
      );
      if (!nearest || distance < nearest.distance) {
        nearest = { entry, distance };
      }
    }
    return nearest?.entry ?? null;
  }

  function entriesForMode() {
    const pacenotes = mode === "beamng"
      ? [...modeEntries.beamng, ...missingEntries]
      : modeEntries[mode];
    return [...markerEntries, ...pacenotes].sort(entrySort);
  }

  function pageStep(row) {
    const height = Math.max(1, row.getBoundingClientRect().height);
    return Math.max(1, Math.floor(noteList.clientHeight / height));
  }

  function listNavigationIndex(key, index, row, count) {
    switch (key) {
      case "ArrowUp":
        return Math.min(count - 1, index + 1);
      case "ArrowDown":
        return Math.max(0, index - 1);
      case "Home":
        return 0;
      case "End":
        return count - 1;
      case "PageUp":
        return Math.max(0, index - pageStep(row));
      case "PageDown":
        return Math.min(count - 1, index + pageStep(row));
      default:
        return null;
    }
  }

  function navigateList(event) {
    const entries = entriesForMode();
    const index = entries.findIndex(entry => entryKey(entry) === selectedKey);
    if (index < 0) {
      return;
    }
    const row = selectedTableRowKey === null
      ? noteList
      : noteList.querySelector(`[data-row-key="${selectedTableRowKey}"]`);
    const nextIndex = listNavigationIndex(
      event.key,
      index,
      row ?? noteList,
      entries.length
    );
    if (nextIndex !== null) {
      event.preventDefault();
      const nextEntry = entries[nextIndex];
      activateEntry(entryKey(nextEntry), nextEntry);
      focusListEntry(entryKey(nextEntry));
    }
  }

  function focusListEntry(key) {
    const selector = selectedTableRowKey === null
      ? `[data-entry-key="${key}"]`
      : `[data-row-key="${selectedTableRowKey}"] [data-entry-key="${key}"]`;
    noteList.querySelector(selector)?.focus({
      preventScroll: true,
    });
  }

  function focusMap() {
    map.focus({ preventScroll: true });
  }

  function entryConversionLoss(entry) {
    return Array.isArray(entry.conversionLoss) ? entry.conversionLoss : [];
  }

  function entryClassName(entry) {
    if (entry.marker) {
      return `note-entry marker-entry ${entry.kind}`;
    }
    const conversionLoss = entryConversionLoss(entry);
    return `note-entry ${entry.mode}${entry.ignored ? " ignored" : ""}${entry.position ? "" : " unmapped"}${conversionLoss.length ? " conversion-loss" : ""}`;
  }

  function entryTitle(entry) {
    const conversionLoss = entryConversionLoss(entry);
    return conversionLoss.length
      ? `RBR conversion loss: ${conversionLoss.join("; ")}`
      : "";
  }

  function activateTableEntry(entry, tableRowKey) {
    const key = entryKey(entry);
    if (!entry.marker && !displayedEntries.includes(entry)) {
      setMode(entry.mode, entry, tableRowKey);
      return;
    }
    activateEntry(key, entry, true, null, tableRowKey);
    focusListEntry(key);
  }

  function appendTableEntry(cell, entry, tableRowKey) {
    const key = entryKey(entry);
    const button = element("button", entryClassName(entry));
    button.type = "button";
    button.dataset.entryKey = key;
    button.title = entryTitle(entry);
    button.textContent = entry.label;
    registerEntryElement(key, button);
    button.addEventListener("click", () => {
      activateTableEntry(entry, tableRowKey);
    });
    button.addEventListener("pointerenter", () => setHovered(key));
    button.addEventListener("pointerleave", () => {
      if (hoveredKey === key) {
        setHovered(null);
      }
    });
    cell.append(button);
  }

  function drawTable() {
    const fragment = document.createDocumentFragment();
    for (const row of tableRows) {
      const tableRow = element(
        "tr",
        row.marker ? `marker-table-row ${row.marker.kind}` : "pacenote-table-row"
      );
      tableRow.dataset.rowKey = row.key;
      const distance = element("td", "distance-cell");
      distance.textContent = formatDistance(row.distance);
      tableRow.append(distance);
      if (row.marker) {
        const markerCell = element("td", "note-cell marker-table-cell");
        markerCell.colSpan = 2;
        appendTableEntry(markerCell, row.marker, row.key);
        tableRow.append(markerCell);
      } else {
        for (const entryMode of ["rbr", "beamng"]) {
          const entries = row[entryMode];
          const cell = element(
            "td",
            entries.length ? `note-cell ${entryMode}` : `note-cell ${entryMode} empty`
          );
          for (const entry of entries) {
            appendTableEntry(cell, entry, row.key);
          }
          tableRow.append(cell);
        }
      }
      fragment.append(tableRow);
    }
    noteTableBody.replaceChildren(fragment);
  }

  function drawEntries() {
    displayedEntries = entriesForMode();
    const entries = displayedEntries;
    const mapFragment = document.createDocumentFragment();
    entryElements = new Map();
    activeMapEntries = [];
    labelAnchors = [];
    labelLayoutAngle = null;
    labelLayoutScale = null;
    labelCollisionLayout = null;
    invalidateLabelLayout();
    hoveredKey = null;
    drawTable();
    entries.forEach(entry => {
      const key = entryKey(entry);
      const conversionLoss = entryConversionLoss(entry);
      const lossClass = conversionLoss.length ? " conversion-loss" : "";
      if (entry.marker || !entry.position) {
        return;
      }
      const position = entry.position;
      const directionLength = Math.hypot(position.directionX, position.directionY) || 1;
      const perpendicularX = -position.directionY / directionLength;
      const perpendicularY = position.directionX / directionLength;
      const side = entry.outerSide ?? (
        entry.pacenoteIndex % 2 === 0 ? 1 : -1
      );
      const roadHalfWidth = data.roadWidthMetres * 0.5;
      const group = element(
        "div",
        `note ${entry.mode}${entry.mode !== mode ? " missing" : ""}${entry.ignored ? " ignored" : ""}${lossClass}`
      );
      group.dataset.entryKey = key;
      group.title = entryTitle(entry);
      registerEntryElement(key, group);
      activeMapEntries.push({ key, entry });
      group.style.left = `${position.x}px`;
      group.style.top = `${position.y}px`;
      group.addEventListener("click", event => {
        event.stopPropagation();
        activateEntry(key, entry);
      });
      const tick = element("div", "note-tick");
      tick.style.left = `${-perpendicularX * roadHalfWidth}px`;
      tick.style.top = `${-perpendicularY * roadHalfWidth}px`;
      tick.style.width = `${data.roadWidthMetres}px`;
      tick.style.transform = `rotate(${Math.atan2(perpendicularY, perpendicularX) * 180 / Math.PI}deg)`;
      const leader = element("div", "note-leader");
      leader.style.left = `${perpendicularX * side * roadHalfWidth}px`;
      leader.style.top = `${perpendicularY * side * roadHalfWidth}px`;
      leader.style.transform =
        `rotate(${Math.atan2(perpendicularY * side, perpendicularX * side) * 180 / Math.PI}deg)`;
      const leaderHit = element("div", "note-leader-hit");
      leaderHit.style.left = leader.style.left;
      leaderHit.style.top = leader.style.top;
      leaderHit.style.width = leader.style.width;
      leaderHit.style.transform = leader.style.transform;
      const elbow = element("div", "note-elbow");
      const elbowHit = element("div", "note-elbow-hit");
      const label = element(
        "span",
        "note-label selectable"
      );
      label.textContent = entry.label;
      labelAnchors.push({
        label,
        leader,
        leaderHit,
        elbow,
        elbowHit,
        roadHalfWidth,
        outwardX: perpendicularX * side,
        outwardY: perpendicularY * side,
      });
      group.append(tick, leader, leaderHit, elbow, elbowHit, label);
      mapFragment.append(group);
    });
    worldLayer.replaceChildren(mapFragment);
    updateSelectionState();
  }

  function pointAtEvent(event) {
    const {
      rect,
      scale,
      worldCenterX,
      worldCenterY,
      screenCenterX,
      screenCenterY,
      angle,
    } = viewport();
    const unrotated = rotate(
      event.clientX - rect.left - screenCenterX,
      event.clientY - rect.top - screenCenterY,
      -angle
    );
    return {
      x: worldCenterX + unrotated.x / scale,
      y: worldCenterY + unrotated.y / scale,
    };
  }

  function updateModeButtons() {
    for (const button of modeButtons) {
      button.classList.toggle("active", button.dataset.mode === mode);
    }
  }

  function setMode(nextMode, targetEntry = null, tableRowKey = null) {
    if (nextMode === mode) {
      if (targetEntry) {
        activateEntry(
          entryKey(targetEntry),
          targetEntry,
          true,
          null,
          tableRowKey
        );
        focusListEntry(entryKey(targetEntry));
      }
      return;
    }
    const previousEntry = displayedEntries.find(
      entry => entryKey(entry) === selectedKey
    );
    const referenceDistance = previousEntry
      ? Number(previousEntry.distance)
      : null;
    cancelFocusAnimation();
    const previousRoadAngle = orientation === "road" ? currentRoadAngle() : null;
    mode = nextMode;
    selectedKey = null;
    selectedRelatedKeys = new Set();
    selectedTableRowKey = null;
    orientationEntryKey = null;
    updateModeButtons();
    drawEntries();
    drawMarkers();
    const nextEntry = targetEntry ?? (
      Number.isFinite(referenceDistance)
        ? nearestPacenote(referenceDistance)
        : null
    );
    if (nextEntry) {
      activateEntry(
        entryKey(nextEntry),
        nextEntry,
        true,
        previousRoadAngle,
        tableRowKey
      );
      focusListEntry(entryKey(nextEntry));
    } else {
      updateMap();
    }
  }

  function selectInitialEntry() {
    const start = displayedEntries.find(entry =>
      entry.marker && entry.kind === "start"
    );
    if (start) {
      view = defaultViewAt(start.position);
      const key = entryKey(start);
      activateEntry(key, start, false);
      focusMap();
      return;
    }
    const firstPacenote = displayedEntries.find(entry =>
      !entry.marker && entry.position
    );
    if (firstPacenote) {
      view = defaultViewAt(firstPacenote.position);
      const key = entryKey(firstPacenote);
      activateEntry(key, firstPacenote, false);
      focusMap();
    }
  }

  for (const button of modeButtons) {
    button.addEventListener("click", () => setMode(button.dataset.mode));
  }
  window.addEventListener("keydown", event => {
    if (
      event.defaultPrevented
      || event.ctrlKey
      || event.metaKey
      || event.altKey
    ) {
      return;
    }
    if (event.key === "ArrowLeft") {
      event.preventDefault();
      setMode("rbr");
    } else if (event.key === "ArrowRight") {
      event.preventDefault();
      setMode("beamng");
    } else {
      navigateList(event);
    }
  });

  function fitRoute() {
    cancelFocusAnimation();
    view = { x: data.viewBox[0], y: data.viewBox[1], width: data.viewBox[2], height: data.viewBox[3] };
    rotationOffset = 0;
    updateMap();
  }

  resetView.addEventListener("click", fitRoute);
  orientationToggle.addEventListener("click", () => {
    cancelFocusAnimation();
    if (orientation === "north") {
      orientation = "road";
      if (orientationPoint === null) {
        orientationPoint = viewCenter();
      }
    } else {
      orientation = "north";
    }
    updateMap();
  });
  map.addEventListener("contextmenu", event => {
    event.preventDefault();
    setMode(mode === "rbr" ? "beamng" : "rbr");
  });
  map.addEventListener("wheel", event => {
    event.preventDefault();
    cancelFocusAnimation();
    const pointer = pointAtEvent(event);
    const factor = Math.exp(event.deltaY * 0.001);
    const currentSpan = Math.max(view.width, view.height);
    const nextSpan = currentSpan * factor;
    const stageSpan = Math.max(data.viewBox[2], data.viewBox[3]);
    if (nextSpan >= stageSpan) {
      view = {
        x: data.viewBox[0],
        y: data.viewBox[1],
        width: data.viewBox[2],
        height: data.viewBox[3],
      };
      updateMap();
      return;
    }
    const clampedFactor = nextSpan <= data.maxZoomSpanMetres
      ? data.maxZoomSpanMetres / currentSpan
      : factor;
    const nextWidth = view.width * clampedFactor;
    const nextHeight = view.height * clampedFactor;
    view.x = pointer.x - (pointer.x - view.x) * (nextWidth / view.width);
    view.y = pointer.y - (pointer.y - view.y) * (nextHeight / view.height);
    view.width = nextWidth;
    view.height = nextHeight;
    updateMap();
  }, { passive: false });

  map.addEventListener("pointerdown", event => {
    if (
      event.button !== 0
      || event.target.closest(".selectable")
      || event.target.closest(".note")
      || event.target.closest(".marker")
    ) {
      return;
    }
    event.preventDefault();
    cancelFocusAnimation();
    const { angle, scale } = viewport();
    setHovered(null);
    drag = {
      kind: event.ctrlKey ? "rotate" : "pan",
      id: event.pointerId,
      clientX: event.clientX,
      clientY: event.clientY,
      view: { ...view },
      angle,
      scale,
      rotationOffset,
    };
    map.setPointerCapture(event.pointerId);
    map.classList.add(drag.kind === "rotate" ? "rotating" : "panning");
  });
  map.addEventListener("pointermove", event => {
    if (!drag || event.pointerId !== drag.id) {
      const directEntry = event.target.closest("[data-entry-key]");
      if (directEntry) {
        setHovered(directEntry.dataset.entryKey);
      } else {
        highlightClosestEntry(event);
      }
      return;
    }
    if (drag.kind === "rotate") {
      event.preventDefault();
      rotationOffset = drag.rotationOffset
        + (
          (event.clientX - drag.clientX)
          + (event.clientY - drag.clientY)
        ) * Math.PI / 360;
      updateMap();
      return;
    }
    event.preventDefault();
    const movement = rotate(
      event.clientX - drag.clientX,
      event.clientY - drag.clientY,
      -drag.angle
    );
    view.x = drag.view.x - movement.x / drag.scale;
    view.y = drag.view.y - movement.y / drag.scale;
    updateMap();
  });
  function stopDrag(event) {
    if (!drag || event.pointerId !== drag.id) {
      return;
    }
    drag = null;
    map.classList.remove("panning");
    map.classList.remove("rotating");
  }
  map.addEventListener("pointerup", stopDrag);
  map.addEventListener("pointercancel", stopDrag);
  map.addEventListener("pointerleave", () => setHovered(null));
  if (typeof ResizeObserver === "function") {
    new ResizeObserver(updateMap).observe(map);
  } else {
    window.addEventListener("resize", updateMap);
  }

  initializeEntries();
  updateModeButtons();
  drawEntries();
  drawMarkers();
  requestAnimationFrame(selectInitialEntry);
})();
</script>
</body>
</html>
"""
