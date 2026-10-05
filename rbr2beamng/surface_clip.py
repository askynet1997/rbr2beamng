"""Subdivision of collision triangles against RBR surface maps.

RBR's physics reads a collision point's surface from a 16x16 map at the
point's interpolated per-vertex selectors. The RX path in
`geometry.split_collision_parts_by_surface` and the Original RBR path in
`original.adapter` both cut source triangles along the map cell edges of that
selector range, so the primitives live here and both share one
implementation.

Polygons are lists of vertex tuples. The first three components are barycentric
weights of the source triangle; any further components are extra affine fields
such as surface-map coordinates or blend values. Every component interpolates
linearly, so a caller may carry as many fields as it needs to clip against.

Working in barycentric space keeps every fragment of a source triangle inside
that triangle's plane, so merging fragments back together never changes the
collision surface and needs no coplanarity test.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from functools import lru_cache
from typing import Hashable, Iterable, Sequence

Vertex = tuple[float, ...]
Polygon = list[Vertex]

BARYCENTRIC_TRIANGLE: tuple[Vertex, Vertex, Vertex] = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)

AREA_EPSILON = 1e-12

SELECTOR_MAP_CELLS = 16

# Joining fragments costs time quadratic in their vertex count, and a key split
# into this many pieces is dominated by real material detail rather than by
# redundant splits, so leave the extremes alone.
_MERGE_FRAGMENT_LIMIT = 256

_WELD_EPSILON = 1e-9
_COLLINEAR_EPSILON = 1e-12
_INTERSECTION_EPSILON = 1e-12


class SurfaceClipError(Exception):
    """Raised when a surface grid cannot be subdivided."""


@dataclass(frozen=True)
class Region:
    """A maximal same-value rectangle of a 16x16 grid, in cell units."""

    value: int
    row: int
    column: int
    height: int
    width: int


@lru_cache(maxsize=4096)
def decompose_regions(
    cells: tuple[tuple[int, ...], ...],
    row_offset: int = 0,
    column_offset: int = 0,
) -> tuple[Region, ...]:
    """Cover a rectangular grid with maximal same-value rectangles.

    The offsets are added to the emitted rows and columns, so a caller may
    decompose a sub-grid and still get coordinates in the full map's space.
    """
    height = len(cells)
    if not height or any(len(row) != len(cells[0]) for row in cells):
        raise SurfaceClipError("Surface grids must be rectangular and non-empty")
    width = len(cells[0])
    covered = [[False] * width for _ in range(height)]
    regions: list[Region] = []
    for row in range(height):
        for column in range(width):
            if covered[row][column]:
                continue
            value = cells[row][column]
            maximum_width = 0
            while (
                column + maximum_width < width
                and not covered[row][column + maximum_width]
                and cells[row][column + maximum_width] == value
            ):
                maximum_width += 1

            best_width = 1
            best_height = 1
            best_area = 1
            span = maximum_width
            for bottom in range(row, height):
                row_width = 0
                while (
                    row_width < span
                    and not covered[bottom][column + row_width]
                    and cells[bottom][column + row_width] == value
                ):
                    row_width += 1
                span = min(span, row_width)
                if span == 0:
                    break
                block_height = bottom - row + 1
                area = span * block_height
                if area > best_area or (
                    area == best_area and span > best_width
                ):
                    best_width = span
                    best_height = block_height
                    best_area = area

            for covered_row in range(row, row + best_height):
                for covered_column in range(column, column + best_width):
                    covered[covered_row][covered_column] = True
            regions.append(
                Region(
                    value,
                    row + row_offset,
                    column + column_offset,
                    best_height,
                    best_width,
                )
            )
    return tuple(regions)


def clip_polygon(
    polygon: Sequence[Vertex],
    field: int,
    boundary: float,
    keep_above: bool,
) -> Polygon:
    """Clip a polygon against a half-plane on one interpolated field."""
    if not polygon:
        return []
    clipped: Polygon = []
    previous = polygon[-1]
    previous_value = previous[field]
    previous_inside = (
        previous_value >= boundary
        if keep_above
        else previous_value <= boundary
    )
    for current in polygon:
        current_value = current[field]
        current_inside = (
            current_value >= boundary
            if keep_above
            else current_value <= boundary
        )
        if current_inside != previous_inside:
            distance = current_value - previous_value
            if abs(distance) > _INTERSECTION_EPSILON:
                factor = (boundary - previous_value) / distance
                clipped.append(
                    tuple(
                        [
                            start + (end - start) * factor
                            for start, end in zip(previous, current)
                        ]
                    )
                )
        if current_inside:
            clipped.append(tuple(current))
        previous = current
        previous_value = current_value
        previous_inside = current_inside
    return clipped


# RBR's collision raycasts interpolate the selector nibbles (0 to 15) of a
# triangle's corners and pass u / 15 and v / 15 to the map sampler. The
# PhysicsNG plugin that RSF installs replaces that sampler with
# map[min(trunc(16 * v / 15), 15)][min(trunc(16 * u / 15), 15)] (unmodified RBR
# reads row 0 only), so cell k of either axis covers selectors
# [15k / 16, 15(k + 1) / 16), and the outer cells extend to infinity because
# the sampler clamps.
_SELECTOR_BOUNDARIES = tuple(
    15 * cell / SELECTOR_MAP_CELLS for cell in range(1, SELECTOR_MAP_CELLS)
)


def _clip_selector_axis(
    polygon: Polygon,
    field: int,
    first_cell: int,
    cell_count: int,
    low: float,
    high: float,
) -> Polygon:
    """Clip to the cells' selector range; ``low`` and ``high`` bound ``field``."""
    if first_cell and low < 15 * first_cell / SELECTOR_MAP_CELLS:
        polygon = clip_polygon(
            polygon,
            field,
            15 * first_cell / SELECTOR_MAP_CELLS,
            True,
        )
        if polygon:
            high = max([vertex[field] for vertex in polygon])
    end_cell = first_cell + cell_count
    if (
        polygon
        and end_cell < SELECTOR_MAP_CELLS
        and high > 15 * end_cell / SELECTOR_MAP_CELLS
    ):
        polygon = clip_polygon(
            polygon,
            field,
            15 * end_cell / SELECTOR_MAP_CELLS,
            False,
        )
    return polygon


def clip_polygon_to_selector_regions(
    polygon: Sequence[Vertex],
    regions: Sequence[Region],
    u_field: int,
    v_field: int,
) -> list[tuple[Hashable, Polygon]]:
    """Split a polygon by the map regions RBR samples at its selectors.

    Region columns follow the selector ``u`` in ``u_field`` and rows follow
    ``v`` in ``v_field``. Fragments without area are dropped.
    """
    u_values = [vertex[u_field] for vertex in polygon]
    v_values = [vertex[v_field] for vertex in polygon]
    u_low = min(u_values)
    u_high = max(u_values)
    v_low = min(v_values)
    v_high = max(v_values)
    # A region overlaps the polygon's selector range exactly when its cells
    # reach from the cell holding the lowest selector to the one below the
    # highest, so whole regions are skipped with integer comparisons.
    first_column = bisect_right(_SELECTOR_BOUNDARIES, u_low)
    last_column = bisect_left(_SELECTOR_BOUNDARIES, u_high)
    first_row = bisect_right(_SELECTOR_BOUNDARIES, v_low)
    last_row = bisect_left(_SELECTOR_BOUNDARIES, v_high)
    result: list[tuple[Hashable, Polygon]] = []
    column_clips: dict[tuple[int, int], tuple[Polygon, float, float]] = {}
    for region in regions:
        column = region.column
        row = region.row
        if (
            column > last_column
            or column + region.width <= first_column
            or row > last_row
            or row + region.height <= first_row
        ):
            continue
        columns = (column, region.width)
        column_clip = column_clips.get(columns)
        if column_clip is None:
            clip = _clip_selector_axis(
                list(polygon),
                u_field,
                column,
                region.width,
                u_low,
                u_high,
            )
            if clip:
                clip_v_values = [vertex[v_field] for vertex in clip]
                column_clip = (clip, min(clip_v_values), max(clip_v_values))
            else:
                column_clip = (clip, 0.0, 0.0)
            column_clips[columns] = column_clip
        clipped = list(column_clip[0])
        if clipped:
            clipped = _clip_selector_axis(
                clipped,
                v_field,
                row,
                region.height,
                column_clip[1],
                column_clip[2],
            )
        if clipped and polygon_area(clipped) > AREA_EPSILON:
            result.append((region.value, clipped))
    return result


def polygon_area(polygon: Sequence[Vertex]) -> float:
    """Unsigned barycentric area, as a fraction of the source triangle."""
    return abs(signed_area(polygon))


def signed_area(polygon: Sequence[Vertex]) -> float:
    """Signed barycentric area; positive matches the source winding."""
    if len(polygon) < 3:
        return 0.0
    total = 0.0
    previous = polygon[-1]
    for current in polygon:
        total += previous[0] * current[1] - current[0] * previous[1]
        previous = current
    return 0.5 * total


class _VertexWelder:
    """Assigns stable ids to vertices that coincide within an epsilon.

    A vertex takes the id of the first welded cell found in row-major order
    among its 3x3 neighbouring cells. A new cell is only welded when all of its
    neighbours are empty, so welded cells are at least two cells apart and a
    vertex in a welded cell can only match that cell.
    """

    def __init__(self) -> None:
        self._rows: dict[int, dict[int, int]] = {}
        self.vertices: list[Vertex] = []

    def identify(self, vertex: Vertex) -> int:
        row = int(round(vertex[0] / _WELD_EPSILON))
        column = int(round(vertex[1] / _WELD_EPSILON))
        rows = self._rows
        own_row = rows.get(row)
        if own_row is not None:
            existing = own_row.get(column)
            if existing is not None:
                return existing
        for columns in (rows.get(row - 1), own_row, rows.get(row + 1)):
            if columns is None:
                continue
            for neighbour_column in (column - 1, column, column + 1):
                existing = columns.get(neighbour_column)
                if existing is not None:
                    return existing
        identifier = len(self.vertices)
        if own_row is None:
            rows[row] = {column: identifier}
        else:
            own_row[column] = identifier
        self.vertices.append(vertex)
        return identifier


def _split_edges_at_vertices(
    loops: Sequence[Sequence[int]],
    vertices: Sequence[Vertex],
) -> list[list[int]]:
    """Insert T-junction vertices that lie on an edge of another fragment.

    Vertices strictly inside an edge, within the weld epsilon, are inserted in
    order along it.
    """
    candidates = sorted(
        (vertices[index][0], vertices[index][1], index)
        for index in {index for loop in loops for index in loop}
    )
    candidate_xs = [candidate[0] for candidate in candidates]
    result: list[list[int]] = []
    for loop in loops:
        expanded: list[int] = []
        count = len(loop)
        for position, start in enumerate(loop):
            end = loop[(position + 1) % count]
            expanded.append(start)
            start_x, start_y = vertices[start][0], vertices[start][1]
            end_x, end_y = vertices[end][0], vertices[end][1]
            delta_x = end_x - start_x
            delta_y = end_y - start_y
            length = delta_x * delta_x + delta_y * delta_y
            if length <= 0.0:
                continue
            first = bisect_left(candidate_xs, min(start_x, end_x) - _WELD_EPSILON)
            last = bisect_right(candidate_xs, max(start_x, end_x) + _WELD_EPSILON)
            # The edge's own two vertices always fall inside its x range.
            if last - first <= 2:
                continue
            box_y_min = min(start_y, end_y) - _WELD_EPSILON
            box_y_max = max(start_y, end_y) + _WELD_EPSILON
            cross_limit = _WELD_EPSILON * _WELD_EPSILON * length
            found: list[tuple[float, int]] = []
            for slot in range(first, last):
                candidate_x, candidate_y, candidate = candidates[slot]
                if candidate == start or candidate == end:
                    continue
                if not box_y_min <= candidate_y <= box_y_max:
                    continue
                point_x = candidate_x - start_x
                point_y = candidate_y - start_y
                cross = delta_x * point_y - delta_y * point_x
                if cross * cross > cross_limit:
                    continue
                along = (delta_x * point_x + delta_y * point_y) / length
                if _WELD_EPSILON < along < 1.0 - _WELD_EPSILON:
                    found.append((along, candidate))
            if found:
                found.sort()
                expanded.extend(candidate for _along, candidate in found)
        result.append(expanded)
    return result


def _trace_boundary_loops(
    edges: Iterable[tuple[int, int]],
) -> list[list[int]] | None:
    """Chain directed edges into closed loops, or None if they branch."""
    successors: dict[int, int] = {}
    for start, end in edges:
        if start in successors:
            return None
        successors[start] = end
    loops: list[list[int]] = []
    while successors:
        start = next(iter(successors))
        loop: list[int] = []
        current = start
        while True:
            following = successors.pop(current, None)
            if following is None:
                return None
            loop.append(current)
            current = following
            if current == start:
                break
        loops.append(loop)
    return loops


def _drop_collinear(loop: Sequence[int], vertices: Sequence[Vertex]) -> list[int]:
    if len(loop) < 3:
        return list(loop)
    result: list[int] = []
    count = len(loop)
    for position in range(count):
        previous = vertices[loop[position - 1]]
        current = vertices[loop[position]]
        following = vertices[loop[(position + 1) % count]]
        first_x = current[0] - previous[0]
        first_y = current[1] - previous[1]
        second_x = following[0] - current[0]
        second_y = following[1] - current[1]
        cross = first_x * second_y - first_y * second_x
        scale = (first_x * first_x + first_y * first_y) * (
            second_x * second_x + second_y * second_y
        )
        if scale > 0.0 and cross * cross > _COLLINEAR_EPSILON * scale:
            result.append(loop[position])
    return result


def merge_fragments(
    fragments: Sequence[tuple[Hashable, Sequence[Vertex]]],
) -> list[tuple[Hashable, Polygon]]:
    """Join fragments of one source triangle that share a material key.

    Fragments must be non-overlapping and wound consistently. Same-key
    fragments that touch along an edge are replaced by their outline, with
    vertices that fall on a straight run removed. The merged outline covers
    exactly the same area, so the collision surface is unchanged.

    A key whose fragments enclose a hole, or whose outline pinches at a
    vertex, is left untouched rather than merged.

    Merging is only kept when it lowers the triangle count. Any triangulation
    of a simple polygon with N vertices uses N-2 triangles, so joining two
    fragments that share part of an edge yields a concave outline that costs
    more than the separate convex pieces. The fragments share the source
    triangle's plane, so leaving them apart is cheaper and covers the same
    surface. The Collada writer welds collision vertices only at identical
    positions, so T-junctions between unmerged fragments are left as they are.
    """
    by_key: dict[Hashable, list[Sequence[Vertex]]] = {}
    for key, polygon in fragments:
        if len(polygon) >= 3:
            by_key.setdefault(key, []).append(polygon)

    merged: list[tuple[Hashable, Polygon]] = []
    for key, polygons in by_key.items():
        unmerged = [
            (key, [tuple(vertex) for vertex in polygon])
            for polygon in polygons
        ]
        if len(polygons) == 1:
            merged.extend(unmerged)
            continue
        if len(polygons) > _MERGE_FRAGMENT_LIMIT:
            merged.extend(unmerged)
            continue
        outlines = _merge_same_key(polygons)
        if outlines is None:
            merged.extend(unmerged)
            continue
        merged_cost = triangle_cost(outlines)
        unmerged_cost = triangle_cost(
            polygon for _key, polygon in unmerged
        )
        if merged_cost >= unmerged_cost:
            merged.extend(unmerged)
            continue
        merged.extend((key, outline) for outline in outlines)
    return merged


def triangle_cost(polygons: Iterable[Sequence[Vertex]]) -> int:
    """Triangles needed to triangulate every polygon."""
    return sum(max(len(polygon) - 2, 0) for polygon in polygons)


def _merge_same_key(
    polygons: Sequence[Sequence[Vertex]],
) -> list[Polygon] | None:
    welder = _VertexWelder()
    loops = [
        [welder.identify(vertex) for vertex in polygon]
        for polygon in polygons
    ]
    loops = _split_edges_at_vertices(loops, welder.vertices)

    interior: set[tuple[int, int]] = set()
    boundary: dict[tuple[int, int], None] = {}
    for loop in loops:
        for position, start in enumerate(loop):
            end = loop[(position + 1) % len(loop)]
            if start == end:
                continue
            if (end, start) in boundary:
                del boundary[(end, start)]
                interior.add((end, start))
                continue
            if (start, end) in boundary or (start, end) in interior:
                return None
            boundary[(start, end)] = None

    if not boundary:
        return None
    traced = _trace_boundary_loops(boundary)
    if traced is None:
        return None

    outlines: list[Polygon] = []
    for loop in traced:
        simplified = _drop_collinear(loop, welder.vertices)
        if len(simplified) < 3:
            continue
        outline = [welder.vertices[index] for index in simplified]
        if signed_area(outline) <= AREA_EPSILON:
            return None
        outlines.append(outline)
    if not outlines:
        return None
    return outlines


def triangulate(polygon: Sequence[Vertex]) -> list[tuple[Vertex, Vertex, Vertex]]:
    """Split a simple polygon into triangles by ear clipping.

    Clipping through a vertex can repeat it within rounding, and a repeated
    vertex lies inside every ear it touches, so repeats are dropped first.
    """
    polygon = _without_repeated_vertices(polygon)
    if len(polygon) < 3:
        return []
    if len(polygon) == 3:
        if polygon_area(polygon) <= AREA_EPSILON:
            return []
        return [(polygon[0], polygon[1], polygon[2])]

    remaining = list(polygon)
    if signed_area(remaining) < 0.0:
        remaining.reverse()

    result: list[tuple[Vertex, Vertex, Vertex]] = []
    guard = 0
    while len(remaining) > 3:
        guard += 1
        if guard > len(polygon) * len(polygon) + 8:
            break
        for position in range(len(remaining)):
            previous = remaining[position - 1]
            current = remaining[position]
            following = remaining[(position + 1) % len(remaining)]
            if not _is_ear(previous, current, following, remaining):
                continue
            if _triangle_area(previous, current, following) > AREA_EPSILON:
                result.append((previous, current, following))
            remaining.pop(position)
            break
        else:
            break
    if len(remaining) == 3 and _triangle_area(*remaining) > AREA_EPSILON:
        result.append((remaining[0], remaining[1], remaining[2]))
    return result


def _without_repeated_vertices(polygon: Sequence[Vertex]) -> list[Vertex]:
    result: list[Vertex] = []
    for vertex in polygon:
        if not result or not _coincide(result[-1], vertex):
            result.append(vertex)
    while len(result) > 1 and _coincide(result[-1], result[0]):
        result.pop()
    return result


def _coincide(first: Vertex, second: Vertex) -> bool:
    return (
        abs(first[0] - second[0]) <= _WELD_EPSILON
        and abs(first[1] - second[1]) <= _WELD_EPSILON
    )


def _triangle_area(first: Vertex, second: Vertex, third: Vertex) -> float:
    return 0.5 * abs(
        (second[0] - first[0]) * (third[1] - first[1])
        - (third[0] - first[0]) * (second[1] - first[1])
    )


def _is_ear(
    previous: Vertex,
    current: Vertex,
    following: Vertex,
    polygon: Sequence[Vertex],
) -> bool:
    cross = (current[0] - previous[0]) * (following[1] - previous[1]) - (
        current[1] - previous[1]
    ) * (following[0] - previous[0])
    if cross <= 0.0:
        return False
    for vertex in polygon:
        if vertex in (previous, current, following):
            continue
        if _inside_triangle(vertex, previous, current, following):
            return False
    return True


def _inside_triangle(
    point: Vertex,
    first: Vertex,
    second: Vertex,
    third: Vertex,
) -> bool:
    first_side = (second[0] - first[0]) * (point[1] - first[1]) - (
        second[1] - first[1]
    ) * (point[0] - first[0])
    second_side = (third[0] - second[0]) * (point[1] - second[1]) - (
        third[1] - second[1]
    ) * (point[0] - second[0])
    third_side = (first[0] - third[0]) * (point[1] - third[1]) - (
        first[1] - third[1]
    ) * (point[0] - third[0])
    return (
        first_side >= -_COLLINEAR_EPSILON
        and second_side >= -_COLLINEAR_EPSILON
        and third_side >= -_COLLINEAR_EPSILON
    )
