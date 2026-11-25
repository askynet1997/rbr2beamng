from __future__ import annotations

import io
import re
import threading
import zipfile
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path, PurePosixPath, PureWindowsPath

import numpy as np
from PIL import Image

from ...beamng import write_collada_to_archive, write_json
from ...core import ConversionError, temporary_root
from ...filesystem import current_filesystem
from ...geometry import AssimpMeshLoader, _compact_mesh_part, convert_transform_matrices
from ...models import MeshPart, RbrStage
from ...rbr import (
    _file_index,
    _normalise_file_reference,
    _read_material_manifest,
    _resolve_indexed,
    get_section,
    get_value,
    load_transforms,
    parse_bool,
    read_ini,
)
from .mod import (
    ALGE,
    ALGE_BANNER,
    CERTINA,
    CERTINA_LAMP_CENTER,
    START_CLOCK_DIGITS_RECT,
    START_CLOCK_GREEN_RECT,
    START_CLOCK_RED_RECT,
    START_CLOCK_REGIONS,
    START_CLOCK_TEXTURE_SIZES,
    START_CLOCKS_FILENAME,
    TAG_HEUER,
    WARTHOG,
    start_clock_texture_tag,
)


# Metres along the face normals, keeps the overlays in front of the static clocks.
_OVERLAY_OFFSET = 0.005


def _rect(x: np.ndarray, y: np.ndarray, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    return (x >= x0) & (x <= x1) & (y >= y0) & (y <= y1)


def _dilate(mask: np.ndarray) -> np.ndarray:
    """mask grown by one pixel, to catch the blended edges of what it marks."""
    mask = mask.copy()
    mask[1:] |= mask[:-1].copy()
    mask[:-1] |= mask[1:].copy()
    mask[:, 1:] |= mask[:, :-1].copy()
    mask[:, :-1] |= mask[:, 1:].copy()
    return mask


def _tag_heuer_redrawn(x: np.ndarray, y: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    # the yellow hands, hour marks and minute ticks, and the dark hand outlines
    # inside the ticks, leave the face shading, the logo and the lettering
    red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    radius = np.hypot(x - 61.75, y - 73.0)
    marks = (radius <= 58.0) & (red > 110) & (green > 110) & (blue < 0.7 * green)
    outlines = (radius <= 50.0) & (pixels.mean(axis=2) < 55)
    mask = _dilate(marks | outlines)
    mask &= ~(_rect(x, y, 51, 41, 72, 59) | _rect(x, y, 55, 59, 65, 62) | _rect(x, y, 35, 88, 88, 98))
    return mask | _rect(x, y, 128, 82, 199, 128)


@cache
def _tag_heuer_masks() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    y, x = np.mgrid[0:256, 0:256] + 0.5
    marks = np.zeros(x.shape, dtype=bool)
    for hour in range(12):
        angle = hour / 12 * 2 * np.pi
        marks |= np.hypot(x - 61.75 - np.sin(angle) * 41.5, y - 73.0 + np.cos(angle) * 41.5) <= 2.5
    panel = _rect(x, y, 128, 82, 199, 128)
    face = (np.hypot(x - 61.75, y - 73.0) <= 30) & ~_rect(x, y, 50, 40, 74, 64)
    lamp = np.hypot(x - 164.5, y - 60.4) <= 9
    return marks, panel, face, lamp


def is_tag_heuer_texture(image: Image.Image) -> bool:
    """Whether a texture has the TAG Heuer board layout: yellow hour marks on a
    grey face, a dark digit panel and a red or orange lamp."""
    if image.width != image.height:
        return False
    pixels = np.asarray(
        image.convert("RGB").resize((256, 256), Image.Resampling.BILINEAR),
        dtype=np.float64,
    )
    marks, panel, face, lamp = _tag_heuer_masks()
    marks_rgb = pixels[marks]
    if not (
        np.mean(
            (marks_rgb[:, 0] > 110)
            & (marks_rgb[:, 1] > 110)
            & (marks_rgb[:, 2] < 0.7 * marks_rgb[:, 1])
        )
        > 0.8
    ):
        return False
    lamp_rgb = pixels[lamp]
    if not (np.mean(lamp_rgb[:, 0] - lamp_rgb[:, 2]) > 60):
        return False
    if not (np.median(pixels[panel].mean(axis=1)) < 70):
        return False
    return bool(30 < np.median(pixels[face].mean(axis=1)) < 140)


def _alge_redrawn(x: np.ndarray, y: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    # wall clock with its glow, start lights and countdown
    return _rect(x, y, 148, 30, 418, 96) | _rect(x, y, 182, 94, 219, 190) | _rect(x, y, 246, 108, 416, 176)


def _alge_banner_redrawn(x: np.ndarray, y: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    # the lit segments of both digit rows, the digit windows stay
    red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    return _dilate((red - np.maximum(green, blue) > 50) & _rect(x, y, 384, 86, 484, 144))


def _certina_redrawn(x: np.ndarray, y: np.ndarray, pixels: np.ndarray) -> np.ndarray:
    # everything lit on the face: hour marks, hands, hub and second LEDs with their glow, the lettering stays;
    # and the lamp with its haze
    red, green, blue = pixels[..., 0], pixels[..., 1], pixels[..., 2]
    radius = np.hypot(x - 1193, y - 394)
    lit = (pixels.mean(axis=2) > 30) & (radius <= 326)
    glow = (green - np.maximum(red, blue) > 12) & (radius <= 348)
    mask = lit | glow
    for _ in range(3):
        mask = _dilate(mask)
    lamp = np.hypot(x - CERTINA_LAMP_CENTER[0], y - CERTINA_LAMP_CENTER[1]) <= 95
    return (mask & (radius <= 348) & ~_rect(x, y, 1080, 474, 1310, 530)) | lamp


# Board elements the clock types redraw, from pixel centers in the design's
# reference texture coordinates and the source pixels.
_REDRAWN = {
    TAG_HEUER: _tag_heuer_redrawn,
    ALGE: _alge_redrawn,
    ALGE_BANNER: _alge_banner_redrawn,
    CERTINA: _certina_redrawn,
}
# RX materials of the boards, and of the parts set into them such as lamp
# lenses, by lowercase material name and diffuse texture stem.
_RX_CLOCK_MATERIALS = (
    (TAG_HEUER, lambda name, texture: name.endswith("\\startclock6\\material0")),
    (ALGE, lambda name, texture: texture.startswith("lahtotavarat")),
)
_RX_BOARD_INSETS = (lambda name, texture: name.endswith("\\startclock6\\material1"),)
# The Warthog start clock re-exported into an RX track (Serra d'Agua), by the
# last two components of the lowercase material name.
_RX_WARTHOG_ROLES = {
    ("reloj", "mat0"): "digits",
    ("reloj", "mat1"): "green",
    ("reloj", "mat2"): "housing",
    ("reloj", "mat3"): "red",
    ("reloj", "mat4"): "fixed digits",
}


@dataclass(frozen=True)
class StartClock:
    design: str
    # Overlay geometry, texcoords in the design's display textures.
    parts: tuple[MeshPart, ...]


@dataclass(frozen=True)
class StartClocks:
    clocks: tuple[StartClock, ...]
    # Design, then texture role.
    textures: dict[str, dict[str, Image.Image]]


def _face_cross(triangles: np.ndarray) -> np.ndarray:
    """Twice the area times the front normal of each triangle, from its winding."""
    return np.cross(triangles[:, 1] - triangles[:, 0], triangles[:, 2] - triangles[:, 0])


def _front_normal(parts: list[MeshPart]) -> np.ndarray:
    """Area-weighted front normal, source vertex normals can be arbitrary."""
    total = sum(
        _face_cross(
            np.asarray(part.vertices, dtype=np.float64)[np.asarray(part.faces, dtype=np.int64).reshape(-1, 3)]
        ).sum(axis=0)
        for part in parts
    )
    return total / max(float(np.linalg.norm(total)), 1e-12)


def _clip_polygon(points: list[np.ndarray], bounds: np.ndarray) -> list[np.ndarray]:
    for axis, limit, side in (
        (0, bounds[0], 1.0),
        (0, bounds[2], -1.0),
        (1, bounds[1], 1.0),
        (1, bounds[3], -1.0),
    ):
        clipped = []
        for index, current in enumerate(points):
            previous = points[index - 1]
            current_in = side * (current[-2 + axis] - limit) >= 0.0
            if current_in != (side * (previous[-2 + axis] - limit) >= 0.0):
                t = (limit - previous[-2 + axis]) / (
                    current[-2 + axis] - previous[-2 + axis]
                )
                clipped.append(previous + (current - previous) * t)
            if current_in:
                clipped.append(current)
        points = clipped
        if len(points) < 3:
            return []
    return points


def region_overlay(part: MeshPart, design: str) -> MeshPart | None:
    """The faces of part showing the design's region of their texture, clipped to it.

    Texcoords are remapped to span the design's display textures.
    """
    (reference_width, reference_height), (x0, y0, x1, y1) = START_CLOCK_REGIONS[design]
    bounds = np.array(
        (x0 / reference_width, y0 / reference_height, x1 / reference_width, y1 / reference_height)
    )
    vertices = np.asarray(part.vertices, dtype=np.float64)
    texcoords = np.asarray(part.texcoords, dtype=np.float64)
    attributes = np.column_stack(
        (
            vertices,
            np.asarray(part.normals, dtype=np.float64),
            np.asarray(part.colors, dtype=np.float64),
            texcoords[:, 0],
            1.0 - texcoords[:, 1],
        )
    )
    faces = np.asarray(part.faces, dtype=np.int64).reshape(-1, 3)
    covered = np.zeros(len(faces), dtype=bool)
    polygons = []
    for index, face in enumerate(faces):
        corners = attributes[face].copy()
        # the source texture wraps, bring the face into the tile of its center
        corners[:, -2:] -= np.floor(corners[:, -2:].mean(axis=0))
        low, high = corners[:, -2:].min(axis=0), corners[:, -2:].max(axis=0)
        if np.any(high <= bounds[:2]) or np.any(low >= bounds[2:]):
            continue
        polygon = _clip_polygon(list(corners), bounds)
        covered[index] = bool(polygon)
        polygons.extend(
            (polygon[0], polygon[corner], polygon[corner + 1])
            for corner in range(1, len(polygon) - 1)
        )
    if not polygons:
        return None
    triangles = np.asarray(polygons, dtype=np.float64)
    # stray faces of the mesh sides can share the board texels, keep the board plane
    cross = _face_cross(triangles[:, :, :3])
    lengths = np.linalg.norm(cross, axis=1)
    plane = cross.sum(axis=0)
    triangles = triangles[
        (lengths > 1e-12) & (cross @ plane > 0.9 * lengths * np.linalg.norm(plane))
    ]
    if not len(triangles):
        return None
    corners = triangles.reshape(-1, attributes.shape[1])
    normals = corners[:, 3:6]
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    normals = np.where(lengths > 1e-12, normals / np.maximum(lengths, 1e-12), (0.0, 0.0, 1.0))
    local = (corners[:, -2:] - bounds[:2]) / (bounds[2:] - bounds[:2])
    region_texcoords = np.column_stack((local[:, 0], 1.0 - local[:, 1])).astype(np.float32)
    overlay = replace(
        part,
        name=f"{part.name}_{design}",
        vertices=corners[:, :3].astype(np.float32),
        faces=np.arange(len(corners), dtype=np.uint32).reshape(-1, 3),
        normals=normals.astype(np.float32),
        colors=corners[:, 6:10].astype(np.float32),
        texcoords=region_texcoords,
        texcoord_sets=(region_texcoords,),
        semantic_uv_indices={},
        specular_strengths=None,
        blend_weights=None,
        collision_eligible=False,
        force_color_stream=True,
    )
    return with_board_insets(overlay, replace(part, faces=faces[~covered]))


def with_board_insets(overlay: MeshPart, part: MeshPart) -> MeshPart:
    """overlay plus the faces of part set into the board it covers, such as lamp
    lenses, textured by their place on the board."""
    faces = np.asarray(part.faces, dtype=np.int64).reshape(-1, 3)
    if not len(faces):
        return overlay
    board = np.asarray(overlay.vertices, dtype=np.float64)
    normal = _front_normal([overlay])
    plane = float(np.mean(board @ normal))
    mapping, *_ = np.linalg.lstsq(
        np.column_stack((board, np.ones(len(board)))),
        np.asarray(overlay.texcoords, dtype=np.float64),
        rcond=None,
    )
    vertices = np.asarray(part.vertices, dtype=np.float64)
    # flattened onto the board, so the overlay covers insets that bulge less than its offset
    depth = vertices @ normal - plane
    flat = vertices - np.outer(depth, normal)
    triangles = vertices[faces]
    cross = _face_cross(triangles)
    lengths = np.linalg.norm(cross, axis=1)
    texcoords = (
        np.column_stack((flat[faces].reshape(-1, 3), np.ones(len(faces) * 3))) @ mapping
    ).reshape(-1, 3, 2)
    inset = (
        (lengths > 1e-12)
        & (cross @ normal > 0.8 * lengths)
        & np.all(np.abs(depth[faces]) < _OVERLAY_OFFSET, axis=1)
        & np.all((texcoords > -0.01) & (texcoords < 1.01), axis=(1, 2))
    )
    if not inset.any():
        return overlay
    corners = faces[inset].reshape(-1)
    inset_texcoords = texcoords[inset].reshape(-1, 2).astype(np.float32)
    texcoords = np.concatenate((np.asarray(overlay.texcoords, dtype=np.float32), inset_texcoords))
    return replace(
        overlay,
        vertices=np.concatenate((board, flat[corners])).astype(np.float32),
        faces=np.concatenate(
            (
                np.asarray(overlay.faces, dtype=np.uint32).reshape(-1, 3),
                np.arange(len(board), len(board) + len(corners), dtype=np.uint32).reshape(-1, 3),
            )
        ),
        normals=np.concatenate(
            (np.asarray(overlay.normals, dtype=np.float32), np.asarray(part.normals, dtype=np.float32)[corners])
        ),
        colors=np.concatenate(
            (np.asarray(overlay.colors, dtype=np.float32), np.asarray(part.colors, dtype=np.float32)[corners])
        ),
        texcoords=texcoords,
        texcoord_sets=(texcoords,),
    )


def _inpaint(pixels: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Masked pixels filled smoothly from their surroundings, coarse to fine."""
    if not mask.any() or mask.all():
        return pixels
    height, width = mask.shape
    result = pixels.copy()
    if min(height, width) > 1:
        even_height, even_width = height + height % 2, width + width % 2
        weights = np.zeros((even_height, even_width), dtype=np.float32)
        weights[:height, :width] = ~mask
        values = np.zeros((even_height, even_width, pixels.shape[2]), dtype=np.float32)
        values[:height, :width] = pixels * weights[:height, :width, None]
        coarse_weights = weights.reshape(even_height // 2, 2, even_width // 2, 2).sum(axis=(1, 3))
        coarse = values.reshape(even_height // 2, 2, even_width // 2, 2, -1).sum(axis=(1, 3))
        coarse /= np.maximum(coarse_weights, 1e-6)[..., None]
        coarse = _inpaint(coarse, coarse_weights == 0)
        result[mask] = coarse.repeat(2, axis=0).repeat(2, axis=1)[:height, :width][mask]
    else:
        result[mask] = pixels[~mask].mean(axis=0)
    for _ in range(16):
        padded = np.pad(result, ((1, 1), (1, 1), (0, 0)), mode="edge")
        average = (padded[:-2, 1:-1] + padded[2:, 1:-1] + padded[1:-1, :-2] + padded[1:-1, 2:]) * 0.25
        result[mask] = average[mask]
    return result


def clock_background(design: str, image: Image.Image) -> Image.Image:
    """The design's region of a board texture, with the redrawn elements painted out."""
    (reference_width, reference_height), (x0, y0, x1, y1) = START_CLOCK_REGIONS[design]
    width, height = START_CLOCK_TEXTURE_SIZES[design]
    scale_x, scale_y = image.width / reference_width, image.height / reference_height
    box = (
        int(np.floor(x0 * scale_x)),
        int(np.floor(y0 * scale_y)),
        int(np.ceil(x1 * scale_x)),
        int(np.ceil(y1 * scale_y)),
    )
    pixels = np.asarray(image.convert("RGB").crop(box), dtype=np.float32)
    x, y = np.meshgrid(
        (box[0] + np.arange(pixels.shape[1]) + 0.5) / scale_x,
        (box[1] + np.arange(pixels.shape[0]) + 0.5) / scale_y,
    )
    pixels = _inpaint(pixels, _REDRAWN[design](x, y, pixels))
    cleaned = Image.fromarray(np.clip(pixels + 0.5, 0, 255).astype(np.uint8))
    return cleaned.resize(
        (width // 2, height // 2),
        Image.Resampling.LANCZOS,
        box=(
            x0 * scale_x - box[0],
            y0 * scale_y - box[1],
            x1 * scale_x - box[0],
            y1 * scale_y - box[1],
        ),
    )


def _transform_part(part: MeshPart, matrix: np.ndarray) -> MeshPart | None:
    basis = matrix[:3, :3]
    determinant = np.linalg.det(basis)
    if abs(determinant) < 1e-12:
        return None
    normals = np.asarray(part.normals, dtype=np.float64) @ np.linalg.inv(basis)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1e-12)
    faces = np.asarray(part.faces).reshape(-1, 3)
    return replace(
        part,
        vertices=np.asarray(part.vertices, dtype=np.float64) @ basis.T + matrix[:3, 3],
        normals=normals,
        faces=faces[:, (0, 2, 1)] if determinant < 0 else faces,
    )


def _warthog_overlay(
    part: MeshPart,
    rect: tuple[int, int, int, int],
    whole_texture: bool,
) -> MeshPart:
    """part with its texcoords span, or the whole texture, moved onto rect of the Warthog display textures."""
    texcoords = np.asarray(part.texcoords, dtype=np.float64)
    if whole_texture:
        # RX lamps can lie whole texture tiles away from [0, 1]
        texcoords = texcoords - np.floor(texcoords.mean(axis=0))
    else:
        # some stages extend the digits item over the housing with faces showing a single texel
        faces = np.asarray(part.faces, dtype=np.int64).reshape(-1, 3)
        corners = texcoords[faces]
        edge_1, edge_2 = corners[:, 1] - corners[:, 0], corners[:, 2] - corners[:, 0]
        spans = edge_1[:, 0] * edge_2[:, 1] - edge_1[:, 1] * edge_2[:, 0]
        part = _compact_mesh_part(replace(part, faces=faces[np.abs(spans) > 1e-12]))
        texcoords = np.asarray(part.texcoords, dtype=np.float64)
    (u0, v0), (u1, v1) = (
        ((0.0, 0.0), (1.0, 1.0))
        if whole_texture
        else (texcoords.min(axis=0), texcoords.max(axis=0))
    )
    x, y, width, height = rect
    texture_width, texture_height = START_CLOCK_TEXTURE_SIZES[WARTHOG]
    local_u = (texcoords[:, 0] - u0) / (u1 - u0)
    local_down = (v1 - texcoords[:, 1]) / (v1 - v0)
    region_texcoords = np.column_stack(
        (
            (x + local_u * width) / texture_width,
            1.0 - (y + local_down * height) / texture_height,
        )
    ).astype(np.float32)
    return replace(
        part,
        texcoords=region_texcoords,
        texcoord_sets=(region_texcoords,),
        semantic_uv_indices={},
        specular_strengths=None,
        force_color_stream=True,
    )


def _facing_out(parts: dict[str, MeshPart], housing: list[MeshPart]) -> dict[str, MeshPart]:
    """parts wound to face away from the housing they are set into.

    RBR draws them double-sided, and stages wind them either way.
    """
    if not housing:
        return parts
    digits = np.asarray(parts["digits"].vertices, dtype=np.float64)
    body = np.concatenate([np.asarray(part.vertices, dtype=np.float64) for part in housing])
    if np.mean((body - digits.mean(axis=0)) @ _front_normal([parts["digits"]])) <= 0:
        return parts
    return {
        role: replace(part, faces=np.asarray(part.faces)[:, ::-1].copy(), normals=-np.asarray(part.normals))
        for role, part in parts.items()
    }


def _near(parts: list[MeshPart], center: np.ndarray, distance: float = 1.0) -> list[MeshPart]:
    """parts centered within distance metres of center, nearest first."""
    distances = [float(np.linalg.norm(np.asarray(part.vertices).mean(axis=0) - center)) for part in parts]
    return [part for length, part in sorted(zip(distances, parts), key=lambda item: item[0]) if length < distance]


def warthog_clock(parts: dict[str, MeshPart], housing: list[MeshPart]) -> StartClock:
    """The Warthog clock of its digits, red and green lamp parts, set into housing."""
    parts = _facing_out(parts, housing)
    return StartClock(
        WARTHOG,
        tuple(
            _warthog_overlay(parts[role], rect, role != "digits")
            for role, rect in (
                ("digits", START_CLOCK_DIGITS_RECT),
                ("red", START_CLOCK_RED_RECT),
                ("green", START_CLOCK_GREEN_RECT),
            )
        ),
    )


def find_rx_start_clocks(
    stage: RbrStage,
    origin: np.ndarray,
    map_yaw_degrees: float,
    warnings: list[str],
) -> StartClocks | None:
    """TAG Heuer and ALGE-TIMING boards and Warthog clocks of an RX stage, as overlays in level space."""
    rules = {}
    inset_names = set()
    warthog_materials = {}
    for material in stage.materials:
        name = material.name.casefold()
        texture = material.diffuse_texture.stem.casefold() if material.diffuse_texture else ""
        for design, matches in _RX_CLOCK_MATERIALS:
            if matches(name, texture):
                rules[name] = (design, material)
        if any(matches(name, texture) for matches in _RX_BOARD_INSETS):
            inset_names.add(name)
        role = _RX_WARTHOG_ROLES.get(PureWindowsPath(name).parts[-2:])
        if role:
            warthog_materials[name] = role
    if not rules and not warthog_materials:
        return None
    # .x files may escape the backslashes of material names
    pattern = re.compile(
        b"|".join(
            rb"\\+".join(
                re.escape(component.encode("utf-8"))
                for component in PureWindowsPath(name).parts[-2:]
            )
            for name in (*rules, *inset_names, *warthog_materials)
        )
    )
    filesystem = current_filesystem()
    loader = AssimpMeshLoader()
    meshes: dict[
        Path,
        tuple[list[tuple[str, object, MeshPart]], list[MeshPart], dict[str, MeshPart]],
    ] = {}
    clocks: list[StartClock] = []
    insets: list[MeshPart] = []
    centers: list[tuple[str, np.ndarray]] = []
    textures: dict[str, dict[str, Image.Image]] = {}
    warthog_found: dict[str, list[MeshPart]] = {}

    def add(clock: StartClock) -> None:
        center = np.concatenate([part.vertices for part in clock.parts]).mean(axis=0)
        # far LODs repeat the clock in place
        if not any(
            other == clock.design and np.linalg.norm(center - position) < 0.1
            for other, position in centers
        ):
            centers.append((clock.design, center))
            clocks.append(clock)

    for stage_object in stage.objects:
        if not stage_object.visible or stage_object.clone_count < 1:
            continue
        path = stage_object.mesh_path
        if path not in meshes:
            meshes[path] = ([], [], {})
            if pattern.search(filesystem.read_bytes(path).lower()):
                for part in loader.load(path).parts:
                    name = part.material_name.casefold()
                    if name in inset_names:
                        meshes[path][1].append(part)
                        continue
                    if name in warthog_materials:
                        meshes[path][2][warthog_materials[name]] = part
                        continue
                    rule = rules.get(name)
                    overlay = rule and region_overlay(part, rule[0])
                    if overlay is not None:
                        meshes[path][0].append((*rule, overlay))
        boards, inset_parts, warthog_parts = meshes[path]
        if not boards and not inset_parts and not warthog_parts:
            continue
        matrices = convert_transform_matrices(
            load_transforms(stage_object),
            origin,
            map_yaw_degrees,
        )
        insets.extend(
            part
            for matrix in matrices
            for part in (_transform_part(inset, matrix) for inset in inset_parts)
            if part is not None
        )
        for design, material, overlay in boards:
            if design not in textures:
                try:
                    image = Image.open(io.BytesIO(filesystem.read_bytes(material.diffuse_texture)))
                    textures[design] = {"background": clock_background(design, image)}
                except (OSError, ValueError) as exc:
                    warnings.append(
                        f"Start clock texture {material.diffuse_texture} is unreadable ({exc}); "
                        "the clock stays static"
                    )
                    textures[design] = {}
            if not textures[design]:
                continue
            for matrix in matrices:
                part = _transform_part(overlay, matrix)
                if part is not None:
                    add(StartClock(design, (part,)))
        for matrix in matrices:
            for role, part in warthog_parts.items():
                part = _transform_part(part, matrix)
                if part is not None:
                    warthog_found.setdefault(role, []).append(part)
    # the digits, lamps and board of an RX Warthog clock can be separate objects
    for digits in warthog_found.get("digits", ()):
        center = np.asarray(digits.vertices).mean(axis=0)
        red = _near(warthog_found.get("red", []), center)
        green = _near(warthog_found.get("green", []), center)
        if not red or not green:
            continue
        housing = _near(warthog_found.get("housing", []), center)
        # quads of RBR's digit atlas spell a fixed time in front of the digits placeholder
        fixed = _near(warthog_found.get("fixed digits", []), center)
        if fixed:
            normal = _front_normal([digits])
            depth = (np.asarray(fixed[0].vertices).mean(axis=0) - center) @ normal
            digits = replace(digits, vertices=np.asarray(digits.vertices) + normal * depth)
        add(warthog_clock({"digits": digits, "red": red[0], "green": green[0]}, housing))
    if not clocks:
        return None
    for index, clock in enumerate(clocks):
        if clock.design == WARTHOG:
            continue
        (part,) = clock.parts
        for inset in insets:
            if np.linalg.norm(np.asarray(inset.vertices).mean(axis=0) - np.asarray(part.vertices).mean(axis=0)) < 1.0:
                part = with_board_insets(part, inset)
        clocks[index] = replace(clock, parts=(part,))
    return StartClocks(
        clocks=tuple(clocks),
        textures={design: images for design, images in textures.items() if images},
    )


def _rx_archive_index(
    archive: zipfile.ZipFile,
    folder_name: str,
    suffix: str | None = None,
) -> tuple[dict[str, str], dict[str, list[str]]]:
    paths: dict[str, str] = {}
    basenames: dict[str, list[str]] = {}
    folder_key = folder_name.casefold()
    suffix_key = suffix.casefold() if suffix else None
    for member in archive.infolist():
        if member.is_dir():
            continue
        parts = PurePosixPath(member.filename.replace("\\", "/")).parts
        if not parts:
            continue
        if parts[0].casefold() == folder_key:
            if len(parts) < 2:
                continue
            relative = PurePosixPath(*parts[1:])
        else:
            relative = PurePosixPath(*parts)
        if suffix_key and not relative.name.casefold().endswith(suffix_key):
            continue
        rel_key = _normalise_file_reference(str(relative)).casefold()
        paths[rel_key] = member.filename
        basenames.setdefault(relative.name.casefold(), []).append(member.filename)
    return paths, basenames


def _resolve_archive_member(
    index: tuple[dict[str, str], dict[str, list[str]]],
    value: str,
    *,
    suffix: str = "",
) -> str | None:
    paths, basenames = index
    reference = _normalise_file_reference(value)
    if suffix and not reference.casefold().endswith(suffix.casefold()):
        reference += suffix
    if "/" in reference:
        member = paths.get(reference.casefold())
        if member is not None:
            return member
    basename = PurePosixPath(reference).name.casefold()
    candidates = basenames.get(basename, ())
    return candidates[0] if len(candidates) == 1 else None


def _rx_texture_readable(stage_root: Path, diffuse: str) -> bool:
    if not diffuse:
        return False
    filesystem = current_filesystem()
    textures_dir = stage_root / "Textures"
    if filesystem.is_dir(textures_dir):
        resolution = _resolve_indexed(
            _file_index(textures_dir, recursive=True), diffuse
        )
        if resolution.path is None:
            return False
        try:
            Image.open(io.BytesIO(filesystem.read_bytes(resolution.path))).load()
            return True
        except (OSError, ValueError):
            return False
    textures_zip = stage_root / "Textures.zip"
    if not filesystem.is_file(textures_zip):
        return False
    try:
        with filesystem.open(textures_zip, "rb") as stream:
            with zipfile.ZipFile(stream) as archive:
                member = _resolve_archive_member(
                    _rx_archive_index(archive, "Textures"), diffuse
                )
                if member is None:
                    return False
                Image.open(io.BytesIO(archive.read(member))).load()
                return True
    except (OSError, ValueError, zipfile.BadZipFile):
        return False


def has_rx_start_clocks(
    stage_root: Path,
    cancelled: threading.Event | None = None,
) -> bool:
    """Fast check for whether an RX stage contains a supported start clock."""
    filesystem = current_filesystem()
    try:
        parser, root, count = _read_material_manifest(stage_root)
    except (ConversionError, OSError, ValueError):
        return False
    rules: dict[str, tuple[str, str]] = {}
    warthog_materials: dict[str, str] = {}
    for index in range(count):
        try:
            name = get_value(root, f"MAT_{index}")
            section = get_section(parser, name)
        except ConversionError:
            continue
        name_key = name.casefold()
        diffuse = get_value(section, "diffuseTexture", "")
        texture_stem = (
            PureWindowsPath(diffuse.replace("/", "\\")).stem.casefold()
            if diffuse
            else ""
        )
        for design, matches in _RX_CLOCK_MATERIALS:
            if matches(name_key, texture_stem):
                rules[name_key] = (design, diffuse)
        role = _RX_WARTHOG_ROLES.get(PureWindowsPath(name_key).parts[-2:])
        if role:
            warthog_materials[name_key] = role
    if not rules and not warthog_materials:
        return False
    if cancelled is not None and cancelled.is_set():
        return False
    try:
        objects_parser = read_ini(stage_root / "objectlist.ini")
        object_count = int(get_value(get_section(objects_parser, "OBJECTS"), "count"))
    except (ConversionError, OSError, ValueError):
        return False
    visible_meshes: set[str] = set()
    for index in range(object_count):
        try:
            section = get_section(objects_parser, f"Object_{index}")
            if not parse_bool(get_value(section, "Visible", "1")):
                continue
            if int(get_value(section, "cloneCount", "1")) < 1:
                continue
            mesh_name = get_value(section, "file")
        except (ConversionError, ValueError):
            continue
        if mesh_name:
            visible_meshes.add(mesh_name)
    if not visible_meshes:
        return False
    pattern = re.compile(
        b"|".join(
            rb"\\+".join(
                re.escape(component.encode("utf-8"))
                for component in PureWindowsPath(name).parts[-2:]
            )
            for name in (*rules, *warthog_materials)
        )
    )
    loader = AssimpMeshLoader()
    warthog_roles: set[str] = set()
    readable_cache: dict[str, bool] = {}

    def check_mesh(mesh_path: Path) -> bool:
        try:
            asset = loader.load(mesh_path)
        except (ConversionError, OSError):
            return False
        for part in asset.parts:
            part_name = part.material_name.casefold()
            if part_name in warthog_materials:
                warthog_roles.add(warthog_materials[part_name])
                if {"digits", "red", "green"} <= warthog_roles:
                    return True
                continue
            rule = rules.get(part_name)
            if rule is None:
                continue
            design, diffuse = rule
            if region_overlay(part, design) is None:
                continue
            if diffuse not in readable_cache:
                readable_cache[diffuse] = _rx_texture_readable(stage_root, diffuse)
            if readable_cache[diffuse]:
                return True
        return False

    objects_dir = stage_root / "Objects"
    if filesystem.is_dir(objects_dir):
        mesh_index = _file_index(objects_dir, ".x")
        seen_paths: set[Path] = set()
        for mesh_name in sorted(visible_meshes):
            if cancelled is not None and cancelled.is_set():
                return False
            resolution = _resolve_indexed(mesh_index, mesh_name, suffix=".x")
            if resolution.path is None or resolution.path in seen_paths:
                continue
            seen_paths.add(resolution.path)
            try:
                raw = filesystem.read_bytes(resolution.path)
            except OSError:
                continue
            if pattern.search(raw.lower()) and check_mesh(resolution.path):
                return True
        return False

    objects_zip = stage_root / "Objects.zip"
    if not filesystem.is_file(objects_zip):
        return False
    try:
        with filesystem.open(objects_zip, "rb") as stream:
            with zipfile.ZipFile(stream) as archive:
                archive_index = _rx_archive_index(archive, "Objects", suffix=".x")
                seen_members: set[str] = set()
                temp_ctx = None
                temp_root: Path | None = None
                try:
                    for mesh_name in sorted(visible_meshes):
                        if cancelled is not None and cancelled.is_set():
                            return False
                        member = _resolve_archive_member(
                            archive_index, mesh_name, suffix=".x"
                        )
                        if member is None or member in seen_members:
                            continue
                        seen_members.add(member)
                        try:
                            raw = archive.read(member)
                        except (OSError, KeyError, zipfile.BadZipFile):
                            continue
                        if not pattern.search(raw.lower()):
                            continue
                        if temp_root is None:
                            temp_ctx = filesystem.temporary_directory(
                                temporary_root(), prefix="rx-clock-"
                            )
                            temp_root = Path(temp_ctx.__enter__())
                        temp_mesh = temp_root / "mesh.x"
                        filesystem.write_bytes(temp_mesh, raw)
                        if check_mesh(temp_mesh):
                            return True
                finally:
                    if temp_ctx is not None:
                        temp_ctx.__exit__(None, None, None)
    except (OSError, zipfile.BadZipFile):
        return False
    return False


def _material(name: str, design: str) -> dict:
    return {
        "name": name,
        "mapTo": name,
        "class": "Material",
        "Stages": [
            {
                "baseColorMap": start_clock_texture_tag(design, "color"),
                "emissiveMap": start_clock_texture_tag(design, "glow"),
                "emissiveFactor": [1, 1, 1],
                "emissiveIntensityNits": 2400,
                "roughnessFactor": 1.0,
                "vertColor": True,
            },
            {},
            {},
            {},
        ],
        "castShadows": False,
        "version": 1.5,
    }


def write_start_clocks(
    archive: zipfile.ZipFile,
    archived_paths: set[str],
    level_dir: Path,
    level_id: str,
    start_clocks: StartClocks,
    origin: np.ndarray,
    version_suffix: str,
) -> None:
    """Write the overlay shapes and metadata driven by the RBR Countdown Clocks mod."""
    filesystem = current_filesystem()
    texture_dir = filesystem.write_path(level_dir / "art" / "textures")
    filesystem.mkdir(texture_dir, parents=True, exist_ok=True)
    designs = {clock.design: {"textures": {}} for clock in start_clocks.clocks}
    for design, textures in start_clocks.textures.items():
        entries = {}
        for role, image in textures.items():
            stem = f"rbr_start_clock_{design}_{role}"
            buffer = io.BytesIO()
            image.save(buffer, "PNG")
            filesystem.write_bytes(texture_dir / f"{stem}.color.png", buffer.getvalue())
            width, height = image.size
            entries[role] = {
                "path": f"/levels/{level_id}/art/textures/{stem}.color.dds",
                "width": width,
                "height": height,
            }
        designs[design] = {"textures": entries}
    materials = {}
    clocks = []
    for index, clock in enumerate(start_clocks.clocks):
        material_name = f"{level_id}_rbr_start_clock_{clock.design}"
        materials[material_name] = _material(material_name, clock.design)
        normal = _front_normal(clock.parts)
        parts = [
            replace(
                part,
                vertices=np.asarray(part.vertices, dtype=np.float64)
                + normal * _OVERLAY_OFFSET
                - origin,
                material_name=material_name,
            )
            for part in clock.parts
        ]
        vertices = np.concatenate([part.vertices for part in parts])
        center = (vertices.min(axis=0) + vertices.max(axis=0)) * 0.5
        parts = [
            replace(part, vertices=(part.vertices - center).astype(np.float32))
            for part in parts
        ]
        across = np.array((-normal[1], normal[0], 0.0))
        across /= max(float(np.linalg.norm(across)), 1e-12)
        archive_name = (
            f"levels/{level_id}/art/shapes/rbr_start_clock_{index}_{version_suffix}.dae"
        )
        write_collada_to_archive(
            archive,
            archive_name,
            parts,
            collision_parts=[],
            render_detail_size=2,
        )
        archived_paths.add(archive_name)
        clocks.append(
            {
                "design": clock.design,
                "shapeName": f"/{archive_name}",
                "position": [round(float(value), 4) for value in center],
                "normal": [round(float(value), 4) for value in normal],
                "width": round(float(np.ptp(vertices @ across)), 4),
            }
        )
    write_json(level_dir / "art" / "shapes" / "rbr_start_clock.materials.json", materials)
    write_json(level_dir / START_CLOCKS_FILENAME, {"designs": designs, "clocks": clocks})
