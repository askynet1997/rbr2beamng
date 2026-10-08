"""Which TRK shape templates RBR collides with.

When a stage loads, RBR builds a BSP solid from each shape template
(``FUN_004f9cd0``) and never collides with a template that fails its checks.
This follows that builder step by step, in float32 like RBR.
"""

from __future__ import annotations

import numpy as np

from .models import ShapeCollisionMesh

_TOLERANCE = np.float32(0.1)
_SLOT_CORNERS = ((1, 2), (2, 0), (0, 1))


def _plane(points: np.ndarray, face: tuple[int, int, int]) -> tuple[np.ndarray, np.float32]:
    first, second, third = (points[index] for index in face)
    a = first - second
    b = third - second
    normal = np.array(
        (
            b[1] * a[2] - a[1] * b[2],
            b[2] * a[0] - a[2] * b[0],
            b[0] * a[1] - a[0] * b[1],
        ),
        dtype=np.float32,
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        normal = normal / np.sqrt(np.float32(normal @ normal))
    distance = np.float32(0.0) - (
        (second[2] * normal[2] + second[1] * normal[1]) + second[0] * normal[0]
    )
    return normal, distance


def shape_rejection(mesh: ShapeCollisionMesh) -> str | None:
    """Why RBR never collides with the template, or None when it may."""
    if "WATER_SOFT" in mesh.name:
        return "named WATER_SOFT"
    faces = [face.indices for face in mesh.faces]
    edges: list[list[int | None]] = []
    for face_index, (first, second, third) in enumerate(faces):
        for start, end in ((first, second), (second, third), (third, first)):
            for edge in edges:
                if edge[0] == end and edge[1] == start:
                    edge[3] = face_index
                    break
                if edge[0] == start and edge[1] == end:
                    return "inconsistently wound"
            else:
                edges.append([start, end, face_index, None])
    # RBR never writes the second face of an open edge and uses whatever byte
    # was there, so whether it collides with an open mesh is unknown.
    if any(edge[3] is None for edge in edges):
        return None
    neighbours: list[list[int | None]] = [[None, None, None] for _ in faces]
    for start, end, face_index, other in edges:
        for slot, (left, right) in enumerate(_SLOT_CORNERS):
            for owner, neighbour in ((face_index, other), (other, face_index)):
                corners = faces[owner]
                if (corners[left] == start and corners[right] == end) or (
                    corners[right] == start and corners[left] == end
                ):
                    if neighbours[owner][slot] is not None:
                        return "non-manifold"
                    neighbours[owner][slot] = neighbour

    points = np.asarray(mesh.vertices, dtype=np.float32)
    # Node k always splits by face k's plane, whichever face created it.
    planes = [_plane(points, face) for face in faces]
    children: dict[tuple[int, bool], int] = {}
    node_count = 0
    crossed = False

    def insert(face_index: int) -> None:
        nonlocal node_count, crossed
        corners = points[list(faces[face_index])]
        node = 0
        while node < node_count:
            normal, distance = planes[node]
            values = (
                (corners[:, 1] * normal[1] + corners[:, 2] * normal[2])
                + corners[:, 0] * normal[0]
                + distance
            )
            in_front = bool(np.any(values > _TOLERANCE))
            behind = bool(np.any(values < -_TOLERANCE))
            crossed = crossed or (in_front and behind)
            node = children.setdefault((node, in_front and not behind), node_count)
        node_count += 1

    visited = [False] * len(faces)

    def visit(face_index: int) -> None:
        visited[face_index] = True
        insert(face_index)
        for neighbour in neighbours[face_index]:
            if neighbour is not None and not visited[neighbour]:
                visit(neighbour)

    visit(0)
    if crossed:
        return "a face crosses a split plane"
    probe = np.float32(2.0) * max(np.float32(0.0), points[:, 0].max())
    node = 0
    while True:
        normal, distance = planes[node]
        behind = np.float32(0.0) <= np.float32(0.0) - (probe * normal[0] + distance)
        child = children.get((node, not behind))
        if child is None:
            return "inside out" if behind else None
        node = child
