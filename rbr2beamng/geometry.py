from __future__ import annotations

import io
import math
import re
import hashlib
import time
from collections import Counter, defaultdict
from contextlib import nullcontext
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Callable, Iterable, Mapping, Sequence, TextIO
from xml.sax.saxutils import escape

import assimp_py
import numpy as np

from .core import ConversionError, slugify, temporary_root
from .filesystem import current_filesystem
from .models import (
    DEFAULT_WATER_NAME_MATCHES,
    MaterialVariant,
    MeshAsset,
    MeshPart,
    RbrMaterial,
    RbrStage,
    RbrSurface,
    SurfaceMap,
)
from .surface_profiles import current_surface_rules
from .profiling import profile_span
from .rbr import (
    SNOWWALL_BOTTOM_SURFACE_ID,
    SNOWWALL_SURFACE_ID,
    is_default_surface_map,
    is_snowwall_surface_map,
    is_water_material,
    surface_map_key,
)
from .surface_clip import (
    SELECTOR_MAP_CELLS,
    Region,
    clip_polygon_to_selector_regions,
    decompose_regions,
    merge_fragments,
    triangle_cost,
    triangulate,
)
from .xfile import (
    XCustomChannels,
    normalize_legacy_x_encoding,
    read_custom_channels_data,
)

ASSIMP_TO_BEAMNG = np.array(
    (
        (1.0, 0.0, 0.0),
        (0.0, 0.0, -1.0),
        (0.0, 1.0, 0.0),
    ),
    dtype=np.float64,
)

SOURCE_TO_BEAMNG = np.array(
    (
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
        (0.0, 1.0, 0.0),
    ),
    dtype=np.float64,
)

_IDENTITY3 = np.identity(3, dtype=np.float64)
_ORTHOGONAL_TOLERANCE = 2e-3 + 1e-5 * np.abs(_IDENTITY3)

ASSIMP_PROCESS_FLAGS = (
    assimp_py.Process_Triangulate
    | assimp_py.Process_JoinIdenticalVertices
    | assimp_py.Process_SortByPType
    | assimp_py.Process_GenSmoothNormals
    | assimp_py.Process_CalcTangentSpace
    | assimp_py.Process_FindInvalidData
    | assimp_py.Process_PreTransformVertices
    | assimp_py.Process_ValidateDataStructure
)
ASSIMP_GEOMETRY_FLAGS = (
    assimp_py.Process_Triangulate
    | assimp_py.Process_SortByPType
    | assimp_py.Process_PreTransformVertices
)


def source_position_to_beamng(position: Iterable[float]) -> np.ndarray:
    return SOURCE_TO_BEAMNG @ np.asarray(tuple(position), dtype=np.float64)


def source_sun_azimuth(direction: Iterable[float]) -> float | None:
    converted = source_position_to_beamng(direction)
    horizontal = -converted[:2]
    if not np.all(np.isfinite(horizontal)) or float(np.linalg.norm(horizontal)) < 1e-8:
        return None
    return math.degrees(math.atan2(float(horizontal[0]), float(horizontal[1]))) % 360.0


def signed_azimuth_delta(source_azimuth: float, target_azimuth: float) -> float:
    return (source_azimuth - target_azimuth + 180.0) % 360.0 - 180.0


def rotate_z(values: np.ndarray, yaw_degrees: float) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.shape[-1] != 3:
        raise ConversionError(
            f"Expected values with final dimension 3, got {array.shape}"
        )
    if abs(yaw_degrees) < 1e-12:
        return array.copy()
    angle = math.radians(yaw_degrees)
    cosine = math.cos(angle)
    sine = math.sin(angle)
    result = array.copy()
    result[..., 0] = cosine * array[..., 0] - sine * array[..., 1]
    result[..., 1] = sine * array[..., 0] + cosine * array[..., 1]
    return result


def assimp_vector_to_beamng(values: np.ndarray) -> np.ndarray:
    return np.asarray(values, dtype=np.float64) @ ASSIMP_TO_BEAMNG.T


def convert_transform_matrices(
    raw_matrices: np.ndarray,
    origin: np.ndarray,
    map_yaw_degrees: float = 0.0,
) -> np.ndarray:
    if raw_matrices.ndim != 3 or raw_matrices.shape[1:] != (4, 4):
        raise ConversionError(f"Expected transform array shaped (N, 4, 4), got {raw_matrices.shape}")

    source_to_beam = np.identity(4, dtype=np.float64)
    source_to_beam[:3, :3] = SOURCE_TO_BEAMNG
    inverse = np.linalg.inv(source_to_beam)
    column_matrices = np.asarray(raw_matrices, dtype=np.float64).transpose(0, 2, 1)
    converted = np.einsum("ij,njk,kl->nil", source_to_beam, column_matrices, inverse)
    converted[:, :3, 3] -= np.asarray(origin, dtype=np.float64)
    if abs(map_yaw_degrees) >= 1e-12:
        angle = math.radians(map_yaw_degrees)
        world_rotation = np.identity(4, dtype=np.float64)
        world_rotation[:2, :2] = (
            (math.cos(angle), -math.sin(angle)),
            (math.sin(angle), math.cos(angle)),
        )
        converted = np.einsum("ij,njk->nik", world_rotation, converted)
    return converted


def rbr_stuff_source_matrices(raw_matrices: np.ndarray) -> np.ndarray:
    """Source clone matrices rebuilt the way RBR places RX collision boxes.

    The RX plugin keeps only each row's length as scale and
    D3DXQuaternionRotationMatrix of the scaled matrix as rotation. That
    function expects a pure rotation, so any scale other than 1 turns the box
    slightly away from its visual object.
    """
    raw = np.asarray(raw_matrices, dtype=np.float32)
    m = raw[:, :3, :3]
    one = np.float32(1.0)
    quaternions = np.empty((len(raw), 4), dtype=np.float32)
    trace = m[:, 0, 0] + m[:, 1, 1] + m[:, 2, 2] + one
    largest = np.argmax(np.stack((m[:, 0, 0], m[:, 1, 1], m[:, 2, 2]), axis=1), axis=1)
    cases = (
        (trace > one, trace, lambda s: (
            (m[:, 1, 2] - m[:, 2, 1]) / s,
            (m[:, 2, 0] - m[:, 0, 2]) / s,
            (m[:, 0, 1] - m[:, 1, 0]) / s,
            np.float32(0.25) * s,
        )),
        ((trace <= one) & (largest == 0), one + m[:, 0, 0] - m[:, 1, 1] - m[:, 2, 2], lambda s: (
            np.float32(0.25) * s,
            (m[:, 0, 1] + m[:, 1, 0]) / s,
            (m[:, 0, 2] + m[:, 2, 0]) / s,
            (m[:, 1, 2] - m[:, 2, 1]) / s,
        )),
        ((trace <= one) & (largest == 1), one + m[:, 1, 1] - m[:, 0, 0] - m[:, 2, 2], lambda s: (
            (m[:, 0, 1] + m[:, 1, 0]) / s,
            np.float32(0.25) * s,
            (m[:, 1, 2] + m[:, 2, 1]) / s,
            (m[:, 2, 0] - m[:, 0, 2]) / s,
        )),
        ((trace <= one) & (largest == 2), one + m[:, 2, 2] - m[:, 0, 0] - m[:, 1, 1], lambda s: (
            (m[:, 0, 2] + m[:, 2, 0]) / s,
            (m[:, 1, 2] + m[:, 2, 1]) / s,
            np.float32(0.25) * s,
            (m[:, 0, 1] - m[:, 1, 0]) / s,
        )),
    )
    with np.errstate(divide="ignore", invalid="ignore"):
        for mask, radicand, components in cases:
            s = np.float32(2.0) * np.sqrt(radicand)
            values = np.stack(components(s), axis=1)
            quaternions[mask] = values[mask]
    quaternions /= np.linalg.norm(quaternions, axis=1, keepdims=True)
    x, y, z, w = quaternions.T
    rotation = np.stack(
        (
            np.stack((1 - 2 * (y * y + z * z), 2 * (x * y + z * w), 2 * (x * z - y * w)), axis=1),
            np.stack((2 * (x * y - z * w), 1 - 2 * (x * x + z * z), 2 * (y * z + x * w)), axis=1),
            np.stack((2 * (x * z + y * w), 2 * (y * z - x * w), 1 - 2 * (x * x + y * y)), axis=1),
        ),
        axis=1,
    )
    result = raw.astype(np.float64)
    result[:, :3, :3] = rotation * np.linalg.norm(m, axis=2)[:, :, None]
    return result


def decompose_matrices(
    matrices: np.ndarray,
) -> list[tuple[list[float], list[float], list[float], bool] | None]:
    """Position, column-major rotation, scale and forest compatibility per matrix.

    None marks a matrix that is not finite, has a zero scale axis, or does not
    decompose into finite values.
    """
    result: list[tuple[list[float], list[float], list[float], bool] | None] = [
        None
    ] * len(matrices)
    finite = np.all(np.isfinite(matrices), axis=(1, 2))
    basis = matrices[finite, :3, :3].astype(float)
    scale = np.linalg.norm(basis, axis=1)
    usable = ~np.any(scale < 1e-8, axis=1)
    indices = np.flatnonzero(finite)[usable]
    position = matrices[indices, :3, 3].astype(float)
    scale = scale[usable]
    rotation = basis[usable] / scale[:, None, :]
    mirrored = np.linalg.det(rotation) < 0
    scale[mirrored, 0] *= -1.0
    rotation[mirrored, :, 0] *= -1.0
    orthogonal = np.all(
        np.abs(np.matmul(rotation.transpose(0, 2, 1), rotation) - _IDENTITY3)
        <= _ORTHOGONAL_TOLERANCE,
        axis=(1, 2),
    )
    absolute_scale = np.abs(scale)
    uniform_scale = np.all(
        np.abs(absolute_scale - absolute_scale[:, :1])
        <= 2e-3 + 2e-3 * absolute_scale[:, :1],
        axis=1,
    )
    finite_values = (
        np.all(np.isfinite(position), axis=1)
        & np.all(np.isfinite(rotation), axis=(1, 2))
        & np.all(np.isfinite(scale), axis=1)
    )
    for index, values, keep in zip(
        indices.tolist(),
        zip(
            position.tolist(),
            rotation.transpose(0, 2, 1).reshape((-1, 9)).tolist(),
            scale.tolist(),
            (orthogonal & uniform_scale).tolist(),
        ),
        finite_values.tolist(),
    ):
        if keep:
            result[index] = values
    return result


def rbr_col_box_part(template: MeshPart, dimensions: tuple[float, ...]) -> MeshPart:
    """RX colBox (width, height, depth) as RBR's RX plugin builds it.

    Width and depth are full sizes centred on the origin; the box reaches from
    5% of its height below the origin up to its height.
    """
    values = np.asarray(dimensions, dtype=np.float32)
    if (
        values.shape != (3,)
        or not np.all(np.isfinite(values))
        or values[0] < 0.0
        or values[1] <= 0.0
        or values[2] < 0.0
    ):
        raise ConversionError(f"Invalid RBR collision box dimensions: {dimensions!r}")
    width, height, depth = values
    half_width = width * 0.5
    half_depth = depth * 0.5
    bottom = -0.05 * height
    vertices = np.asarray(
        (
            (-half_width, -half_depth, bottom),
            (half_width, -half_depth, bottom),
            (half_width, half_depth, bottom),
            (-half_width, half_depth, bottom),
            (-half_width, -half_depth, height),
            (half_width, -half_depth, height),
            (half_width, half_depth, height),
            (-half_width, half_depth, height),
        ),
        dtype=np.float32,
    )
    faces = np.asarray(
        (
            (0, 2, 1),
            (0, 3, 2),
            (4, 5, 6),
            (4, 6, 7),
            (0, 1, 5),
            (0, 5, 4),
            (3, 7, 6),
            (3, 6, 2),
            (0, 4, 7),
            (0, 7, 3),
            (1, 2, 6),
            (1, 6, 5),
        ),
        dtype=np.uint32,
    )
    return MeshPart(
        name=f"{template.name}_box",
        vertices=vertices,
        faces=faces,
        normals=np.zeros_like(vertices),
        texcoords=np.zeros((len(vertices), 2), dtype=np.float32),
        colors=np.ones((len(vertices), 4), dtype=np.float32),
        material_index=template.material_index,
        material_name=template.material_name,
        collision_eligible=True,
    )


def thicken_rbr_col_box_dimensions(
    dimensions: tuple[float, ...],
    *,
    maximum_gap: float = 0.2,
    target_thickness: float = 0.5,
) -> tuple[float, float, float]:
    width, height, depth = dimensions
    if width <= maximum_gap:
        width = max(width, target_thickness)
    if depth <= maximum_gap:
        depth = max(depth, target_thickness)
    return float(width), float(height), float(depth)


def spawn_rotation(
    direction: Iterable[float],
    map_yaw_degrees: float = 0.0,
) -> list[float]:
    source_direction = np.asarray(tuple(direction), dtype=np.float64)
    forward = -source_position_to_beamng(source_direction)
    forward[2] = 0.0
    length = float(np.linalg.norm(forward))
    if length < 1e-8:
        forward = np.array((0.0, 1.0, 0.0), dtype=np.float64)
    else:
        forward /= length
    forward = rotate_z(forward, map_yaw_degrees)
    right = np.array((forward[1], -forward[0], 0.0), dtype=np.float64)
    up = np.array((0.0, 0.0, 1.0), dtype=np.float64)
    return np.column_stack((right, forward, up)).flatten(order="F").tolist()


def quaternion_from_direction(direction: Iterable[float]) -> list[float]:
    forward = np.asarray(tuple(direction), dtype=np.float64)
    forward[2] = 0.0
    length = float(np.linalg.norm(forward))
    if length < 1e-8:
        forward = np.array((0.0, 1.0, 0.0), dtype=np.float64)
    else:
        forward /= length
    half_angle = math.atan2(float(forward[0]), float(forward[1])) * 0.5
    return [0.0, 0.0, math.sin(half_angle), math.cos(half_angle)]


def lod_detail_size(radius: float, distance: float) -> int:
    return max(
        2,
        int(round(max(radius, 0.0) * 3400.0 / max(distance, 1.0))),
    )


def _reshape(values: object, width: int, *, dtype: np.dtype = np.dtype(np.float32)) -> np.ndarray:
    return np.asarray(values, dtype=dtype).reshape((-1, width)).copy()


def remap_mesh_part(
    part: MeshPart,
    vertex_indices: np.ndarray,
    faces: np.ndarray,
    **changes: object,
) -> MeshPart:
    indices = np.asarray(vertex_indices)
    values: dict[str, object] = {
        "vertices": np.asarray(part.vertices)[indices],
        "faces": np.asarray(faces, dtype=np.uint32).reshape((-1, 3)),
        "normals": np.asarray(part.normals)[indices],
        "texcoords": np.asarray(part.texcoords)[indices],
        "colors": np.asarray(part.colors)[indices],
        "texcoord_sets": tuple(
            np.asarray(channel)[indices]
            for channel in part.texcoord_sets
        ),
        "specular_strengths": (
            np.asarray(part.specular_strengths)[indices]
            if part.specular_strengths is not None
            else None
        ),
        "blend_weights": (
            np.asarray(part.blend_weights)[indices]
            if part.blend_weights is not None
            else None
        ),
    }
    values.update(changes)
    return replace(part, **values)


def _material_name(material: dict[str, object], fallback: str) -> str:
    for key in ("TEXTURE_BASE", "NAME"):
        value = material.get(key)
        if isinstance(value, bytes):
            value = value.decode("utf-8", errors="replace")
        if isinstance(value, str) and value.strip() and value != "material0":
            return value.strip()
    textures = material.get("TEXTURES")
    if isinstance(textures, dict):
        for entries in textures.values():
            if entries:
                value = entries[0]
                if isinstance(value, bytes):
                    value = value.decode("utf-8", errors="replace")
                return str(value).strip()
    return fallback


class AssimpMeshLoader:
    def load(self, path: Path, *, geometry_only: bool = False) -> MeshAsset:
        """Load a mesh; ``geometry_only`` keeps positions, faces and materials."""
        filesystem = current_filesystem()
        path = filesystem.read_path(path)
        with profile_span(
            "read_x_custom_channels",
            category="asset",
            source=str(path),
        ) as custom_span:
            source_data = filesystem.read_bytes(path)
            custom_channels = (
                None
                if geometry_only
                else read_custom_channels_data(source_data)
            )
            custom_span.update(
                present=custom_channels is not None,
                uvSets=(
                    len(custom_channels.texcoords)
                    if custom_channels is not None
                    else 0
                ),
                hasBlendWeights=(
                    custom_channels is not None
                    and custom_channels.blend_weights is not None
                ),
            )
        normalized_data = normalize_legacy_x_encoding(source_data)
        flags = ASSIMP_GEOMETRY_FLAGS if geometry_only else ASSIMP_PROCESS_FLAGS
        if custom_channels is not None:
            flags &= ~assimp_py.Process_JoinIdenticalVertices
        with profile_span(
            "assimp_import",
            category="asset",
            source=str(path),
            processFlags=flags,
        ) as assimp_span:
            with (
                filesystem.temporary_directory(temporary_root(), prefix="x-")
                if normalized_data is not None
                else nullcontext(path)
            ) as import_root:
                import_path = path
                if normalized_data is not None:
                    import_path = filesystem.write_path(Path(import_root) / "mesh.x")
                    filesystem.write_bytes(import_path, normalized_data)
                try:
                    scene = assimp_py.import_file(str(import_path), flags)
                except Exception as exc:
                    raise ConversionError(
                        f"Assimp failed to load {path}: {exc}"
                    ) from exc
            assimp_span.update(
                meshes=len(scene.meshes),
                materials=len(scene.materials),
            )
        if custom_channels is not None and len(scene.meshes) != 1:
            raise ConversionError(
                f"BTB custom-channel mesh {path} produced {len(scene.meshes)} Assimp meshes"
            )
        parts: list[MeshPart] = []
        material_names = [
            _material_name(material, f"material_{index}")
            for index, material in enumerate(scene.materials)
        ]
        for index, mesh in enumerate(scene.meshes):
            vertices = _reshape(mesh.vertices, 3)
            vertices = assimp_vector_to_beamng(vertices).astype(np.float32)
            faces = np.asarray(mesh.indices, dtype=np.uint32).reshape((-1, 3)).copy()
            normals = _reshape(mesh.normals, 3) if mesh.normals else np.zeros_like(vertices)
            normals = assimp_vector_to_beamng(normals).astype(np.float32)
            normal_lengths = np.linalg.norm(normals, axis=1)
            normal_lengths[normal_lengths < 1e-8] = 1.0
            normals /= normal_lengths[:, None]
            if mesh.texcoords:
                components = int(mesh.num_uv_components[0])
                texcoords = _reshape(mesh.texcoords[0], components)[:, :2].astype(np.float32)
            else:
                texcoords = np.zeros((vertices.shape[0], 2), dtype=np.float32)
            if mesh.colors:
                colors = _reshape(mesh.colors[0], 4).astype(np.float32)
            else:
                colors = np.ones((vertices.shape[0], 4), dtype=np.float32)
            texcoord_sets: tuple[np.ndarray, ...] = (texcoords,)
            blend_weights = None
            if custom_channels is not None:
                (
                    vertices,
                    faces,
                    normals,
                    colors,
                    texcoord_sets,
                    blend_weights,
                ) = _attach_custom_channels(
                    path,
                    custom_channels,
                    vertices,
                    faces,
                    normals,
                    colors,
                    texcoords,
                )
                texcoords = texcoord_sets[0]
            material_index = int(mesh.material_index)
            parts.append(
                MeshPart(
                    name=mesh.name or f"part_{index}",
                    vertices=vertices,
                    faces=faces,
                    normals=normals,
                    texcoords=texcoords,
                    colors=colors,
                    material_index=material_index,
                    material_name=material_names[material_index],
                    texcoord_sets=texcoord_sets,
                    blend_weights=blend_weights,
                )
            )
        if not parts:
            raise ConversionError(f"Assimp loaded no meshes from {path}")
        return MeshAsset(path, parts, material_names)


def _attach_custom_channels(
    path: Path,
    custom: XCustomChannels,
    vertices: np.ndarray,
    faces: np.ndarray,
    normals: np.ndarray,
    colors: np.ndarray,
    assimp_texcoords: np.ndarray,
) -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
    tuple[np.ndarray, ...],
    np.ndarray | None,
]:
    source_indices = custom.faces.reshape(-1)
    if len(source_indices) != len(vertices):
        raise ConversionError(
            f"BTB custom-channel vertex count mismatch in {path}: "
            f"{len(source_indices)} source corners and {len(vertices)} Assimp vertices"
        )
    expanded_texcoords = []
    for channel in custom.texcoords:
        expanded = np.asarray(channel[source_indices], dtype=np.float32).copy()
        expanded[:, 1] = 1.0 - expanded[:, 1]
        expanded_texcoords.append(expanded)
    if expanded_texcoords and not np.allclose(
        expanded_texcoords[0],
        assimp_texcoords,
        rtol=1e-5,
        atol=1e-5,
    ):
        raise ConversionError(f"Unable to align BTB custom channels with Assimp vertices in {path}")
    blend_weights = (
        np.asarray(custom.blend_weights[source_indices], dtype=np.float32)
        if custom.blend_weights is not None
        else None
    )
    attributes = [vertices, normals, colors, *expanded_texcoords]
    if blend_weights is not None:
        attributes.append(blend_weights[:, None])
    if custom.tangents is not None:
        attributes.append(
            np.asarray(custom.tangents[source_indices], dtype=np.float32)
        )
    if custom.binormals is not None:
        attributes.append(
            np.asarray(custom.binormals[source_indices], dtype=np.float32)
        )
    combined = np.concatenate(attributes, axis=1)
    _unique, first_indices, inverse = np.unique(
        combined,
        axis=0,
        return_index=True,
        return_inverse=True,
    )
    first_indices = np.asarray(first_indices, dtype=np.uint32)
    return (
        vertices[first_indices],
        inverse[faces].astype(np.uint32),
        normals[first_indices],
        colors[first_indices],
        tuple(channel[first_indices] for channel in expanded_texcoords),
        blend_weights[first_indices] if blend_weights is not None else None,
    )


def _ground_type(surface_id: int, stage: RbrStage) -> str:
    surface = stage.surfaces.get(surface_id)
    return (
        surface.profile.ground_type
        if surface and surface.profile
        else current_surface_rules().unknown_ground_type
    )


def material_key(stage_id: str, material: RbrMaterial | None, ground_type: str) -> str:
    source_name = material.name if material else "missing"
    digest = hashlib.sha1(source_name.casefold().encode("utf-8")).hexdigest()[:8]
    return f"rbr_{slugify(stage_id)}_{slugify(source_name)[:80]}_{digest}_{ground_type.lower()}"


def _surface_ground_depth(surface: RbrSurface, ground_type: str) -> float:
    profile = surface.profile or current_surface_rules().unknown_profile
    return profile.ground_depth


SurfaceGroup = tuple[str, bool, bool, bool, float, bool, bool]
_GROUP_WATER = 2


def _painted_entirely_snowwall(
    material: RbrMaterial | None,
    stage: RbrStage,
) -> bool:
    surface_map = (
        stage.surface_maps.get(surface_map_key(material.diffuse_texture))
        if material and material.diffuse_texture
        else None
    ) or stage.unmatched_surface_map
    return surface_map is None or is_snowwall_surface_map(surface_map)


def _mapped_surface_group(
    surface_id: int,
    material: RbrMaterial | None,
    stage: RbrStage,
    transparent: bool,
    snowbank_floor: bool = False,
) -> SurfaceGroup:
    # Authors scatter Snowwall cells over snow textures, and RBR gives them the
    # same physics as the snow around them, so only a whole texture painted
    # Snowwall is a snowbank, and not where it is the floor under an added bank.
    if surface_id == SNOWWALL_SURFACE_ID and (
        snowbank_floor or not _painted_entirely_snowwall(material, stage)
    ):
        surface_id = SNOWWALL_BOTTOM_SURFACE_ID
    surface = stage.surfaces.get(surface_id)
    profile = (
        surface.profile
        if surface and surface.profile
        else current_surface_rules().unknown_profile
    )
    ground_type = profile.ground_type
    water = profile.water
    hard = profile.hard
    bendable = profile.bendable
    ground_depth = _surface_ground_depth(surface, ground_type) if surface else profile.ground_depth
    return (
        ground_type,
        hard,
        water,
        bendable,
        ground_depth,
        profile.snowbank,
        profile.collision_eligible,
    )


def _surface_group_lookup(
    material: RbrMaterial | None,
    stage: RbrStage,
    transparent: bool,
    snowbank_floor: bool,
) -> Callable[[int], SurfaceGroup]:
    groups: dict[int, SurfaceGroup] = {}

    def group_for_surface(surface_id: int) -> SurfaceGroup:
        group = groups.get(surface_id)
        if group is None:
            group = _mapped_surface_group(
                surface_id,
                material,
                stage,
                transparent,
                snowbank_floor,
            )
            groups[surface_id] = group
        return group

    return group_for_surface


def _material_variant_for_group(
    stage: RbrStage,
    material: RbrMaterial | None,
    group: SurfaceGroup,
    surface_ids: set[int],
) -> tuple[str, MaterialVariant]:
    (
        ground_type,
        hard,
        water,
        bendable,
        ground_depth,
        snowbank,
        _collision_eligible,
    ) = group
    state = f"{'h' if hard else 's'}{'w' if water else ''}{'b' if bendable else ''}"
    depth_state = f"d{round(ground_depth * 1000)}"
    bank_state = "_bank" if snowbank else ""
    key = (
        f"{material_key(stage.metadata.folder_name, material, ground_type)}_"
        f"{state}_{depth_state}{bank_state}"
    )
    return key, MaterialVariant(
        material=material,
        ground_type=ground_type,
        hard=hard,
        water=water,
        bendable=bendable,
        source_surface_ids=tuple(sorted(surface_ids)),
        ground_depth=ground_depth,
        snowbank=snowbank,
    )


def _collision_eligible(
    group: SurfaceGroup,
    transparent: bool,
) -> bool:
    (
        ground_type,
        hard,
        water,
        bendable,
        _ground_depth,
        snowbank,
        profile_collision_eligible,
    ) = group
    return profile_collision_eligible and (
        (hard and not transparent and not water and ground_type != "VOID")
        or ground_type == "SOFT_COLLISION_GENERAL"
        or snowbank
        or bendable
    )


def _surface_context(
    part: MeshPart,
    materials: dict[str, RbrMaterial],
    stage: RbrStage,
    water_name_matches: tuple[str, ...] | None,
    unmatched_surface_map: SurfaceMap | None = None,
) -> tuple[RbrMaterial | None, bool, bool, SurfaceMap | None]:
    material = materials.get(part.material_name.casefold())
    transparent = bool(
        material and "transparent" in material.technique.casefold()
    )
    surface_map = (
        stage.surface_maps.get(surface_map_key(material.diffuse_texture))
        if material and material.diffuse_texture
        else None
    )
    material_water = (
        (surface_map is None or is_default_surface_map(surface_map))
        and water_name_matches is not None
        and is_water_material(material, water_name_matches)
    )
    if material_water:
        surface_map = None
    elif surface_map is None:
        surface_map = unmatched_surface_map
    return (
        material,
        transparent,
        material_water,
        surface_map,
    )


def split_parts_by_surface(
    asset: MeshAsset,
    stage: RbrStage,
    water_name_matches: tuple[str, ...] | None = DEFAULT_WATER_NAME_MATCHES,
    unmatched_surface_map: SurfaceMap | None = None,
) -> tuple[list[MeshPart], dict[str, MaterialVariant]]:
    """Group faces by the surfaces RBR reads under them, cut at water edges.

    Parts whose texture has no map of its own use ``unmatched_surface_map``,
    or the unknown surface without one.
    """
    by_name = {material.name.casefold(): material for material in stage.materials}
    result: list[MeshPart] = []
    material_variants: dict[str, MaterialVariant] = {}
    for part in asset.parts:
        (
            material,
            transparent,
            material_water,
            surface_map,
        ) = _surface_context(
            part,
            by_name,
            stage,
            water_name_matches,
            unmatched_surface_map,
        )
        grouped_faces: dict[
            SurfaceGroup,
            list[np.ndarray],
        ] = defaultdict(list)
        grouped_ids: dict[SurfaceGroup, set[int]] = defaultdict(set)
        if surface_map is None:
            group = (
                ("WATER", False, True, False, 0.0, False, False)
                if material_water
                else (_ground_type(-1, stage), not transparent, False, False, 0.0, False, True)
            )
            grouped_faces[group].append(
                np.asarray(part.faces, dtype=np.uint32).reshape((-1, 3))
            )
        else:
            part, grouped_faces, grouped_ids = _clip_render_faces(
                part,
                surface_map,
                _surface_group_lookup(
                    material,
                    stage,
                    transparent,
                    part.snowbank_floor,
                ),
            )
        for group, faces in sorted(grouped_faces.items()):
            key, variant = _material_variant_for_group(
                stage,
                material,
                group,
                grouped_ids[group],
            )
            if material_water:
                variant = replace(
                    variant,
                    classification_fallback="water-name",
                )
            material_variants[key] = variant
            ground_type = variant.ground_type
            result.append(
                MeshPart(
                    name=f"{part.name}_{ground_type.lower()}",
                    vertices=part.vertices,
                    faces=np.concatenate(faces, axis=0).astype(np.uint32),
                    normals=part.normals,
                    texcoords=part.texcoords,
                    colors=part.colors,
                    material_index=part.material_index,
                    material_name=key,
                    collision_eligible=_collision_eligible(
                        group,
                        transparent,
                    ),
                    water=variant.water,
                    texcoord_sets=part.texcoord_sets,
                    blend_weights=part.blend_weights,
                )
            )
    return result, material_variants


def remove_overdrawn_faces(
    asset: MeshAsset,
    materials: Mapping[str, RbrMaterial],
) -> tuple[MeshAsset, int]:
    """Drop no-cull triangles that a later triangle of the same draw repeats.

    RBR draws them with ``ZFUNC LESSEQUAL``, so of two triangles sharing all
    three positions the later one covers the earlier from both sides. BeamNG
    reorders triangles at load, and would z-fight them instead.
    """
    parts: list[MeshPart] = []
    removed = 0
    for part in asset.parts:
        material = materials.get(part.material_name.casefold())
        faces = np.asarray(part.faces, dtype=np.uint32).reshape((-1, 3))
        if material is None or not material.double_sided or len(faces) < 2:
            parts.append(part)
            continue
        # Adding 0.0 folds -0.0 into 0.0, which np.unique compares bytewise.
        corners = np.asarray(part.vertices, dtype=np.float32)[faces] + np.float32(0.0)
        order = np.lexsort((corners[..., 2], corners[..., 1], corners[..., 0]))
        keys = np.take_along_axis(corners, order[..., None], axis=1).reshape((-1, 9))
        _unique, last_from_end = np.unique(keys[::-1], axis=0, return_index=True)
        keep = np.zeros(len(faces), dtype=bool)
        keep[len(faces) - 1 - last_from_end] = True
        if keep.all():
            parts.append(part)
            continue
        removed += int(len(faces) - np.count_nonzero(keep))
        parts.append(replace(part, faces=faces[keep]))
    return replace(asset, parts=parts), removed


@dataclass(frozen=True)
class _SurfacePolygon:
    group: SurfaceGroup
    surface_ids: tuple[int, ...]
    vertices: np.ndarray


# Clip vertices hold the barycentric weights of the source triangle followed by
# its interpolated selectors, which the surface map is addressed through.
_SELECTOR_U_FIELD = 3
_SELECTOR_V_FIELD = 4


def _surface_regions(
    cells: tuple[tuple[int, ...], ...],
) -> tuple[Region, ...]:
    if len(cells) != 16 or any(len(row) != 16 for row in cells):
        raise ConversionError("Surface maps must contain a 16x16 cell grid")
    return decompose_regions(cells)


def rx_collision_selectors(texcoords: np.ndarray) -> np.ndarray:
    """Per-vertex surface selectors that RBR's RX plugin compiles.

    The plugin stores one byte per collision vertex,
    ``(int)(15 * u) | ((int)(15 * v) << 4)`` in float32, with ``u`` and ``v``
    the source .X texture coordinates. Nothing wraps: coordinates past the
    texture edge keep only the low bits of their cell, and a negative ``u``
    sets the high nibble through sign extension. Returns ``(u, v)`` nibble
    pairs; ``texcoords`` holds converter UVs, whose V is flipped.
    """
    uv = np.asarray(texcoords, dtype=np.float32).reshape((-1, 2))
    fifteen = np.float32(15.0)
    u = np.trunc(fifteen * uv[:, 0]).astype(np.int64)
    v = np.trunc(fifteen * (np.float32(1.0) - uv[:, 1])).astype(np.int64)
    selector = (u | (v << 4)) & 0xFF
    return np.stack((selector & 0xF, selector >> 4), axis=1)


def _surface_triangles(
    selectors: np.ndarray,
    surface_map: SurfaceMap,
    group_for_surface,
) -> tuple[list[tuple[SurfaceGroup, tuple[int, ...]]], np.ndarray, int]:
    """Clip one triangle's corner selectors against the surface map.

    Returns each child triangle's group and surface ids, the children's corners
    as ``(n, 3, 3)`` barycentric weights of the source triangle, and how many
    triangles merging saved.
    """
    clip_triangle = tuple(
        (*weights, float(u), float(v))
        for weights, (u, v) in zip(
            ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            selectors,
        )
    )
    fragments = [
        (group_for_surface(surface_id), surface_id, polygon)
        for surface_id, polygon in clip_polygon_to_selector_regions(
            clip_triangle,
            _surface_regions(surface_map.cells),
            _SELECTOR_U_FIELD,
            _SELECTOR_V_FIELD,
        )
    ]
    # One surface's cells decompose into several rectangles, and different
    # surfaces often share a ground model, so neighbouring fragments form one
    # larger area whose outline needs fewer triangles than the pieces.
    surface_ids: dict[SurfaceGroup, set[int]] = {}
    for group, surface_id, _polygon in fragments:
        surface_ids.setdefault(group, set()).add(surface_id)
    merged = merge_fragments(
        [(group, polygon) for group, _surface_id, polygon in fragments]
    )
    saved = triangle_cost(
        polygon for _group, _surface_id, polygon in fragments
    ) - triangle_cost(outline for _group, outline in merged)
    labels: list[tuple[SurfaceGroup, tuple[int, ...]]] = []
    corners: list[list[tuple[float, ...]]] = []
    for group, outline in merged:
        identifiers = tuple(sorted(surface_ids[group]))
        for triangle_corners in triangulate(outline):
            labels.append((group, identifiers))
            corners.append([corner[:3] for corner in triangle_corners])
    return (
        labels,
        np.asarray(corners, dtype=np.float64).reshape((-1, 3, 3)),
        saved,
    )


def _exact_surface_polygons(
    triangle: np.ndarray,
    selectors: np.ndarray,
    surface_map: SurfaceMap,
    group_for_surface,
) -> tuple[list[_SurfacePolygon], int]:
    labels, weights, saved = _surface_triangles(
        selectors,
        surface_map,
        group_for_surface,
    )
    return (
        [
            _SurfacePolygon(group, identifiers, corner_weights @ triangle)
            for (group, identifiers), corner_weights in zip(labels, weights)
        ],
        saved,
    )


_IDENTITY_WEIGHTS = np.identity(3, dtype=np.float64)


def _clip_render_faces(
    part: MeshPart,
    surface_map: SurfaceMap,
    group_for_surface: Callable[[int], SurfaceGroup],
) -> tuple[
    MeshPart,
    dict[SurfaceGroup, list[np.ndarray]],
    dict[SurfaceGroup, set[int]],
]:
    """Group render faces by the surfaces RBR reads under them.

    A face takes the surface at the centroid of its selectors unless the
    surfaces under it disagree on water. Only those faces are cut, like
    collision; their pieces get vertices appended to the part, with every
    vertex attribute interpolated, so the rendered surface does not change.
    Faces keep their source order within each group.
    """
    faces = np.asarray(part.faces, dtype=np.uint32).reshape((-1, 3))
    face_selectors = rx_collision_selectors(part.texcoords)[faces]
    last_cell = SELECTOR_MAP_CELLS - 1
    centroid_cells = np.minimum(
        face_selectors.sum(axis=1) * SELECTOR_MAP_CELLS // (3 * 15),
        last_cell,
    )
    face_surfaces = np.asarray(surface_map.cells, dtype=np.int64)[
        centroid_cells[:, 1],
        centroid_cells[:, 0],
    ]
    # Water cells summed over every top-left rectangle of the map.
    water_counts = np.zeros((SELECTOR_MAP_CELLS + 1,) * 2, dtype=np.int64)
    water_counts[1:, 1:] = np.cumsum(
        np.cumsum(
            [
                [group_for_surface(surface_id)[_GROUP_WATER] for surface_id in row]
                for row in surface_map.cells
            ],
            axis=0,
        ),
        axis=1,
    )
    low = np.minimum(face_selectors.min(axis=1) * SELECTOR_MAP_CELLS // 15, last_cell)
    high = np.minimum(face_selectors.max(axis=1) * SELECTOR_MAP_CELLS // 15, last_cell) + 1
    water_cells = (
        water_counts[high[:, 1], high[:, 0]]
        - water_counts[low[:, 1], high[:, 0]]
        - water_counts[high[:, 1], low[:, 0]]
        + water_counts[low[:, 1], low[:, 0]]
    )
    cut = (water_cells > 0) & (water_cells < np.prod(high - low, axis=1))

    sources: dict[SurfaceGroup, list[np.ndarray]] = defaultdict(list)
    triangles: dict[SurfaceGroup, list[np.ndarray]] = defaultdict(list)
    grouped_ids: dict[SurfaceGroup, set[int]] = defaultdict(set)
    split_sources: list[int] = []
    split_groups: list[SurfaceGroup] = []
    split_weights: list[np.ndarray] = []
    for index in np.flatnonzero(cut):
        polygons, _merged_away = _exact_surface_polygons(
            _IDENTITY_WEIGHTS,
            face_selectors[index],
            surface_map,
            group_for_surface,
        )
        if len({polygon.group[_GROUP_WATER] for polygon in polygons}) < 2:
            cut[index] = False
            continue
        for polygon in polygons:
            grouped_ids[polygon.group].update(polygon.surface_ids)
            split_sources.append(int(index))
            split_groups.append(polygon.group)
            split_weights.append(polygon.vertices)

    for surface_id in np.unique(face_surfaces[~cut]):
        group = group_for_surface(int(surface_id))
        indices = np.flatnonzero(~cut & (face_surfaces == surface_id))
        sources[group].append(indices)
        triangles[group].append(faces[indices])
        grouped_ids[group].add(int(surface_id))

    if split_weights:
        weights = np.stack(split_weights)
        corners = faces[split_sources]

        def extend(values: np.ndarray) -> np.ndarray:
            values = np.asarray(values)
            added = np.einsum(
                "fij,fj...->fi...",
                weights,
                values[corners].astype(np.float64),
            )
            return np.concatenate(
                (
                    values,
                    added.reshape((-1, *values.shape[1:])).astype(values.dtype),
                )
            )

        added_faces = len(part.vertices) + np.arange(
            3 * len(split_weights),
            dtype=np.uint32,
        ).reshape((-1, 3))
        for position, group in enumerate(split_groups):
            sources[group].append(np.array((split_sources[position],)))
            triangles[group].append(added_faces[position : position + 1])
        part = replace(
            part,
            vertices=extend(part.vertices),
            normals=extend(part.normals),
            texcoords=extend(part.texcoords),
            colors=extend(part.colors),
            texcoord_sets=tuple(extend(values) for values in part.texcoord_sets),
            blend_weights=(
                None if part.blend_weights is None else extend(part.blend_weights)
            ),
        )

    grouped_faces: dict[SurfaceGroup, list[np.ndarray]] = {}
    for group, group_sources in sources.items():
        order = np.argsort(np.concatenate(group_sources), kind="stable")
        grouped_faces[group] = [np.concatenate(triangles[group])[order]]
    return part, grouped_faces, grouped_ids


def split_collision_parts_by_surface(
    asset: MeshAsset,
    stage: RbrStage,
    water_name_matches: tuple[str, ...] | None = DEFAULT_WATER_NAME_MATCHES,
) -> tuple[list[MeshPart], dict[str, MaterialVariant], int]:
    by_name = {material.name.casefold(): material for material in stage.materials}
    passthrough_parts: list[MeshPart] = []
    result_parts: list[MeshPart] = []
    material_variants: dict[str, MaterialVariant] = {}
    merged_triangle_count = 0
    for part in asset.parts:
        (
            material,
            transparent,
            material_water,
            surface_map,
        ) = _surface_context(
            part,
            by_name,
            stage,
            water_name_matches,
            stage.unmatched_surface_map,
        )
        if (
            surface_map is None
            or len({value for row in surface_map.cells for value in row}) <= 1
            or material_water
        ):
            passthrough_parts.append(part)
            continue

        group_for_surface = _surface_group_lookup(
            material,
            stage,
            transparent,
            part.snowbank_floor,
        )
        selectors = rx_collision_selectors(part.texcoords)
        attributes = np.concatenate(
            (
                part.vertices,
                part.normals,
                part.texcoords,
                part.colors,
            ),
            axis=1,
        ).astype(np.float64)

        # Faces with the same corner selectors split identically, so each
        # selector combination is clipped once into child triangles whose
        # corners are barycentric weights of the source triangle.
        faces = np.asarray(part.faces, dtype=np.int64).reshape((-1, 3))
        face_keys = (
            selectors[faces].reshape((-1, 6)) << np.arange(0, 24, 4)
        ).sum(axis=1)
        _unique_keys, first_faces, face_templates = np.unique(
            face_keys,
            return_index=True,
            return_inverse=True,
        )
        identity = np.identity(3, dtype=np.float64)
        group_codes: dict[SurfaceGroup, int] = {}
        grouped_ids: dict[int, set[int]] = defaultdict(set)
        template_weights: list[np.ndarray] = []
        template_codes: list[int] = []
        template_counts = np.zeros(len(first_faces), dtype=np.int64)
        template_saved = np.zeros(len(first_faces), dtype=np.int64)
        for template, face_index in enumerate(first_faces.tolist()):
            labels, weights, saved = _surface_triangles(
                selectors[faces[face_index]],
                surface_map,
                group_for_surface,
            )
            for group, identifiers in labels:
                code = group_codes.setdefault(group, len(group_codes))
                template_codes.append(code)
                grouped_ids[code].update(identifiers)
            template_weights.append(np.matmul(weights, identity))
            template_counts[template] = len(labels)
            template_saved[template] = saved
        merged_triangle_count += int(
            template_saved @ np.bincount(face_templates, minlength=len(first_faces))
        )
        child_counts = template_counts[face_templates]
        child_total = int(child_counts.sum())
        if not child_total:
            passthrough_parts.append(part)
            continue
        template_starts = np.cumsum(template_counts) - template_counts
        child_starts = np.cumsum(child_counts) - child_counts
        child_templates = np.repeat(
            template_starts[face_templates] - child_starts,
            child_counts,
        ) + np.arange(child_total)
        child_values = np.matmul(
            np.concatenate(template_weights)[child_templates],
            attributes[faces[np.repeat(np.arange(len(faces)), child_counts)]],
        )
        child_codes = np.asarray(template_codes, dtype=np.int64)[child_templates]
        for group in sorted(group_codes):
            code = group_codes[group]
            key, variant = _material_variant_for_group(
                stage,
                material,
                group,
                grouped_ids[code],
            )
            material_variants[key] = variant
            values = child_values[child_codes == code].reshape(
                (-1, attributes.shape[1])
            )
            result_parts.append(
                MeshPart(
                    name=f"{part.name}_{variant.ground_type.lower()}",
                    vertices=np.asarray(values[:, 0:3], dtype=np.float32),
                    faces=np.arange(len(values), dtype=np.uint32).reshape((-1, 3)),
                    normals=np.asarray(values[:, 3:6], dtype=np.float32),
                    texcoords=np.asarray(values[:, 6:8], dtype=np.float32),
                    colors=np.asarray(values[:, 8:12], dtype=np.float32),
                    material_index=part.material_index,
                    material_name=key,
                    collision_eligible=_collision_eligible(
                        group,
                        transparent,
                    ),
                    water=variant.water,
                )
            )

    if passthrough_parts:
        passthrough_result, passthrough_variants = split_parts_by_surface(
            MeshAsset(
                source=asset.source,
                parts=passthrough_parts,
                material_names=asset.material_names,
            ),
            stage,
            water_name_matches,
            stage.unmatched_surface_map,
        )
        result_parts.extend(passthrough_result)
        material_variants.update(passthrough_variants)
    return result_parts, material_variants, merged_triangle_count


def _xml_id(value: str) -> str:
    identifier = re.sub(r"[^A-Za-z0-9_.-]", "_", value)
    if not identifier or not (identifier[0].isalpha() or identifier[0] == "_"):
        identifier = f"id_{identifier}"
    return identifier


_FLOAT_CHUNK = 16384
_FLOAT_CHUNK_FORMAT = " ".join(("%.8g",) * _FLOAT_CHUNK)
_FLOAT_REPETITION_SAMPLE = 1024
_INDEX_CHUNK = 4096
_INDEX_CHUNK_FORMAT = " ".join(("%d",) * _INDEX_CHUNK)
_POWERS_OF_TEN = 10.0 ** np.arange(13)
_FOUR_DIGITS = np.array(
    [int.from_bytes(b"%04d" % value, "little") for value in range(10000)],
    dtype=np.uint64,
)
_TRAILING_ZEROS = np.array(
    [4] + [len(str(value)) - len(str(value).rstrip("0")) for value in range(1, 10000)],
    dtype=np.intp,
)
_BYTE_MASKS = np.array(
    [(1 << (8 * count)) - 1 for count in range(9)], dtype=np.uint64
)
_DECIMAL_POINTS = np.array(
    [ord(".") << (8 * count) for count in range(8)] + [0], dtype=np.uint64
)
# %g prefixes by sign and decimal exponent -4..7.
_FLOAT_PREFIXES = [
    ("-" if negative else "") + ("0." + "0" * (-exponent - 1) if exponent < 0 else "")
    for negative in (False, True)
    for exponent in range(-4, 8)
]
_FLOAT_PREFIX_WORDS = np.array(
    [int.from_bytes(prefix.encode("ascii"), "little") for prefix in _FLOAT_PREFIXES],
    dtype=np.uint64,
)
_FLOAT_PREFIX_BITS = np.array(
    [8 * len(prefix) for prefix in _FLOAT_PREFIXES], dtype=np.uint64
)
_FIELD_MASKS = np.arange(16) < np.arange(17)[:, None]


def _float32_text(values: np.ndarray) -> str:
    """``" ".join("%.8g" % value for value in values)`` for float32 ``values``.

    A float32 times 10**p is exact in float64 for p <= 12, so rounding that
    product to an integer gives %g's correctly rounded 8 significant digits.
    Each value's text is assembled little-endian in a 16-byte field. Values %g writes in
    exponent form, and non-finite values, are formatted one by one.
    """
    with np.errstate(divide="ignore", invalid="ignore"):
        magnitude = np.abs(values).astype(np.float64)
        power = 7 - np.floor(np.log10(magnitude))
    regular = (power >= 0) & (power <= 12)
    power = np.where(regular, power, 0).astype(np.intp)
    digits = np.rint(magnitude * _POWERS_OF_TEN[power])
    for _attempt in range(2):
        # log10 can be one off near powers of ten, and rounding can carry
        # into a ninth digit.
        wrong = np.flatnonzero(regular & ((digits < 1e7) | (digits >= 1e8)))
        if not len(wrong):
            break
        power[wrong] += np.where(digits[wrong] < 1e7, 1, -1)
        inside = (power[wrong] >= 0) & (power[wrong] <= 12)
        regular[wrong[~inside]] = False
        wrong = wrong[inside]
        digits[wrong] = np.rint(magnitude[wrong] * _POWERS_OF_TEN[power[wrong]])
    regular &= (digits >= 1e7) & (digits < 1e8) & (power <= 11)
    digits = np.where(regular, digits, 0).astype(np.intp)
    high = digits // 10000
    low = digits - high * 10000
    exponent = np.where(regular, 7 - power, 0)
    integer_digits = np.maximum(exponent + 1, 0)
    kept = np.maximum(
        8 - _TRAILING_ZEROS[low] - (low == 0) * _TRAILING_ZEROS[high],
        integer_digits,
    )
    inner_point = (kept > integer_digits) & (integer_digits > 0)
    text = (_FOUR_DIGITS[high] | (_FOUR_DIGITS[low] << np.uint64(32))) & _BYTE_MASKS[kept]
    integer_mask = _BYTE_MASKS[integer_digits]
    body = np.where(
        inner_point,
        (text & integer_mask)
        | _DECIMAL_POINTS[integer_digits]
        | ((text & ~integer_mask) << np.uint64(8)),
        text,
    )
    overflow = np.where(inner_point, text >> np.uint64(56), 0)
    prefix = np.signbit(values) * 12 + exponent + 4
    shift = _FLOAT_PREFIX_BITS[prefix]
    fields = np.empty((len(values), 2), dtype=np.uint64)
    fields[:, 0] = _FLOAT_PREFIX_WORDS[prefix] | (body << shift)
    # numpy shifts by 64 or more bits give 0.
    fields[:, 1] = (overflow << shift) | (body >> (np.uint64(64) - shift))
    length = (shift >> np.uint64(3)).astype(np.intp) + kept + inner_point
    chars = fields.view(np.uint8)
    chars[np.arange(len(values)), length] = ord(" ")
    for index in np.flatnonzero(~regular & (magnitude != 0)).tolist():
        special = b"%.8g " % float(values[index])
        chars[index, : len(special)] = np.frombuffer(special, dtype=np.uint8)
        length[index] = len(special) - 1
    return chars[_FIELD_MASKS[length + 1]][:-1].tobytes().decode("ascii")


def _write_floats(stream: TextIO, values: np.ndarray) -> None:
    source = np.asarray(values).reshape(-1)
    flat = np.asarray(source, dtype=np.float64)
    repetitive = False
    vectorized = False
    if flat.size >= _FLOAT_REPETITION_SAMPLE:
        sample = flat.view(np.uint64)[:: flat.size // _FLOAT_REPETITION_SAMPLE]
        repetitive = len(np.unique(sample)) * 4 <= len(sample)
        vectorized = not repetitive and source.dtype == np.float32
    for start in range(0, flat.size, _FLOAT_CHUNK):
        if start:
            stream.write(" ")
        chunk = flat[start : start + _FLOAT_CHUNK]
        if repetitive:
            unique, inverse = np.unique(chunk.view(np.uint64), return_inverse=True)
            unique_values = tuple(unique.view(np.float64).tolist())
            texts = np.array(
                (
                    _FLOAT_CHUNK_FORMAT[: 5 * len(unique_values) - 1]
                    % unique_values
                ).split(" "),
                dtype=object,
            )
            stream.write(" ".join(texts[inverse]))
        elif vectorized:
            stream.write(_float32_text(source[start : start + _FLOAT_CHUNK]))
        else:
            chunk_values = tuple(chunk.tolist())
            stream.write(
                _FLOAT_CHUNK_FORMAT[: 5 * len(chunk_values) - 1] % chunk_values
            )


def _write_collada_source(
    stream: TextIO,
    source_id: str,
    arrays: Iterable[np.ndarray],
    stride: int,
    params: tuple[str, ...],
    float_text: str | None = None,
) -> None:
    """Write a float source; ``float_text`` is the arrays' already formatted values."""
    arrays = tuple(np.asarray(values) for values in arrays)
    value_count = sum(values.size for values in arrays)
    array_id = f"{source_id}-array"
    stream.write(
        f'<source id="{source_id}"><float_array id="{array_id}" '
        f'count="{value_count}">'
    )
    if float_text is not None:
        stream.write(float_text)
    else:
        for index, values in enumerate(arrays):
            if index:
                stream.write(" ")
            _write_floats(stream, values)
    stream.write(
        f'</float_array><technique_common><accessor source="#{array_id}" '
        f'count="{value_count // stride}" stride="{stride}">'
    )
    for param in params:
        stream.write(f'<param name="{param}" type="float"/>')
    stream.write("</accessor></technique_common></source>\n")


def _write_indices(stream: TextIO, values: np.ndarray, offset: int = 0) -> None:
    flat = np.asarray(values).reshape(-1)
    for start in range(0, flat.size, _INDEX_CHUNK):
        if start:
            stream.write(" ")
        chunk = tuple(
            (flat[start : start + _INDEX_CHUNK].astype(np.int64) + offset).tolist()
        )
        stream.write(_INDEX_CHUNK_FORMAT[: 3 * len(chunk) - 1] % chunk)


def _collision_faces(part: MeshPart) -> np.ndarray:
    if not part.collision_eligible:
        return np.empty((0, 3), dtype=np.uint32)
    return part.faces


def _collision_part_digest(part: MeshPart, faces: np.ndarray) -> bytes:
    digest = hashlib.sha256()
    for values in (part.vertices, faces):
        array = np.asarray(values)
        digest.update(array.dtype.str.encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    digest.update(part.material_name.encode("utf-8"))
    return digest.digest()


def collision_shape_digest(parts: Iterable[MeshPart]) -> bytes:
    """Digest of everything a collision-only ``write_collada`` of ``parts``
    writes."""
    digest = hashlib.sha256()
    for part in parts:
        digest.update(_collision_part_digest(part, _collision_faces(part)))
    return digest.digest()


def _float32_sort_keys(values: np.ndarray) -> np.ndarray:
    # Adding +0.0 folds -0.0 into +0.0, so keys compare and order like the floats.
    bits = (values + np.float32(0.0)).view(np.uint32)
    return np.where(bits >> 31, ~bits, bits | np.uint32(0x80000000))


def _dense_rank(keys: np.ndarray) -> np.ndarray:
    order = np.argsort(keys)
    sorted_keys = keys[order]
    step = np.empty(len(keys), dtype=np.uint64)
    step[:1] = 0
    np.not_equal(sorted_keys[1:], sorted_keys[:-1], out=step[1:])
    del sorted_keys
    rank = np.empty(len(keys), dtype=np.uint64)
    rank[order] = np.cumsum(step, out=step)
    return rank


def _unique_float32_positions(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    keys = _float32_sort_keys(values[:, 0]).astype(np.uint64) << np.uint64(32)
    keys |= _float32_sort_keys(values[:, 1])
    keys = _dense_rank(keys)
    keys <<= np.uint64(32)
    keys |= _float32_sort_keys(values[:, 2])
    order = np.argsort(keys)
    sorted_keys = keys[order]
    del keys
    first = np.empty(len(order), dtype=bool)
    first[:1] = True
    np.not_equal(sorted_keys[1:], sorted_keys[:-1], out=first[1:])
    del sorted_keys
    inverse = np.empty(len(order), dtype=np.intp)
    inverse[order] = np.cumsum(first) - 1
    first_indices = np.minimum.reduceat(order, np.flatnonzero(first))
    return values[first_indices], first_indices, inverse


def _unique_rows(
    values: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``np.unique(values, axis=0, return_index=True, return_inverse=True)``."""
    if np.isnan(values).any():
        return np.unique(values, axis=0, return_index=True, return_inverse=True)
    if values.dtype == np.float32 and values.ndim == 2 and values.shape[1] == 3:
        return _unique_float32_positions(values)
    order = np.lexsort(values.T[::-1])
    sorted_values = values[order]
    first = np.empty(len(order), dtype=bool)
    first[:1] = True
    np.any(sorted_values[1:] != sorted_values[:-1], axis=1, out=first[1:])
    inverse = np.empty(len(order), dtype=np.intp)
    inverse[order] = np.cumsum(first) - 1
    return sorted_values[first], order[first], inverse


def _weld_collision_faces(
    vertices: np.ndarray,
    faces: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Unique positions used by ``faces``, the source vertex of each and the
    faces remapped onto them."""
    faces = np.asarray(faces, dtype=np.uint32).reshape((-1, 3))
    if not len(faces):
        return None
    used_vertices, compact_faces = np.unique(faces, return_inverse=True)
    compact_faces = compact_faces.reshape((-1, 3))
    compact_vertices = np.asarray(vertices)[used_vertices]
    unique_vertices, first_indices, inverse_vertices = _unique_rows(
        compact_vertices
    )
    return (
        unique_vertices,
        used_vertices[first_indices],
        inverse_vertices[compact_faces],
    )


def _collision_geometry_part(
    part: MeshPart,
    faces: np.ndarray,
) -> MeshPart | None:
    """Welded positions and faces only, which is all a Collada collision
    mesh stores."""
    welded = _weld_collision_faces(part.vertices, faces)
    if welded is None:
        return None
    unique_vertices, _source_vertices, welded_faces = welded
    return MeshPart(
        name=f"{part.name}_collision",
        vertices=np.asarray(unique_vertices, dtype=np.float32),
        faces=np.asarray(welded_faces, dtype=np.uint32).reshape((-1, 3)),
        normals=None,
        texcoords=None,
        colors=None,
        material_index=part.material_index,
        material_name=part.material_name,
    )


def _compact_collision_part(
    part: MeshPart,
    faces: np.ndarray,
) -> MeshPart | None:
    welded = _weld_collision_faces(part.vertices, faces)
    if welded is None:
        return None
    unique_vertices, attribute_indices, welded_faces = welded
    return remap_mesh_part(
        part,
        attribute_indices,
        welded_faces,
        name=f"{part.name}_collision",
        vertices=np.asarray(unique_vertices, dtype=np.float32),
        collision_eligible=True,
        water=False,
        texcoord_sets=(),
        semantic_uv_indices={},
        specular_strengths=None,
        blend_weights=None,
        force_color_stream=False,
        lod_group=None,
        lod_kind="any",
    )


def _compact_mesh_part(part: MeshPart) -> MeshPart:
    faces = np.asarray(part.faces, dtype=np.uint32).reshape((-1, 3))
    used_vertices, compact_faces = np.unique(faces, return_inverse=True)
    if len(used_vertices) == len(part.vertices):
        return part
    return remap_mesh_part(
        part,
        used_vertices,
        compact_faces,
    )


@dataclass(frozen=True)
class ThinWallShell:
    face_pairs: tuple[tuple[int, int], ...]
    source_faces: tuple[int, ...]


@dataclass(frozen=True)
class ThinWallTemplate:
    part: MeshPart
    shell: ThinWallShell


_GRID_MEMBER_CHUNK_COLUMNS = 1 << 18
_THIN_WALL_PAIR_CHUNK = 1 << 20


def _chunk_bounds(sizes: np.ndarray, limit: int) -> list[tuple[int, int]]:
    """Consecutive non-empty ranges whose summed sizes stay within ``limit``
    unless a single item is larger."""
    ends = np.cumsum(sizes)
    bounds: list[tuple[int, int]] = []
    start = 0
    while start < len(sizes):
        base = ends[start - 1] if start else 0
        end = max(
            start + 1,
            int(np.searchsorted(ends, base + limit, side="right")),
        )
        bounds.append((start, end))
        start = end
    return bounds


def _grid_cell_members(
    lower: np.ndarray,
    upper: np.ndarray,
    cells: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ascending indices of the boxes ``lower..upper`` that contain each cell.

    The boxes containing ``cells[i]`` are
    ``members[offsets[cell_index[i]]:offsets[cell_index[i] + 1]]``.
    """
    unique_cells, _first, cell_index = _unique_rows(cells)
    minimum = unique_cells.min(axis=0)
    size = unique_cells.max(axis=0) - minimum + 1
    if float(np.prod(size.astype(np.float64))) >= 2.0**62:
        cell_boxes = [
            np.flatnonzero(np.all((lower <= cell) & (upper >= cell), axis=1))
            for cell in unique_cells
        ]
        offsets = np.concatenate(
            ([0], np.cumsum([len(boxes) for boxes in cell_boxes]))
        )
        return cell_index, offsets, np.concatenate(cell_boxes)

    def keys(x: np.ndarray, y: np.ndarray, z: np.ndarray) -> np.ndarray:
        return ((x - minimum[0]) * size[1] + (y - minimum[1])) * size[2] + (
            z - minimum[2]
        )

    cell_keys = keys(unique_cells[:, 0], unique_cells[:, 1], unique_cells[:, 2])
    clipped_lower = np.maximum(lower, minimum)
    spans = np.maximum(
        np.minimum(upper, minimum + size - 1) - clipped_lower + 1, 0
    )
    columns = spans[:, 0] * spans[:, 1] * (spans[:, 2] > 0)
    box_keys = keys(clipped_lower[:, 0], clipped_lower[:, 1], clipped_lower[:, 2])
    del clipped_lower
    box_count = len(lower)
    index_type = (
        np.int32 if max(box_count, len(unique_cells)) < 2**31 else np.int64
    )
    hit_columns: list[tuple[np.ndarray, ...]] = []
    for start, end in _chunk_bounds(columns, _GRID_MEMBER_CHUNK_COLUMNS):
        box_columns = columns[start:end]
        boxes = np.repeat(np.arange(start, end), box_columns)
        x, y = np.divmod(
            np.arange(len(boxes))
            - np.repeat(np.cumsum(box_columns) - box_columns, box_columns),
            spans[boxes, 1],
        )
        low = box_keys[boxes] + (x * size[1] + y) * size[2]
        first = np.searchsorted(cell_keys, low, side="left")
        counts = (
            np.searchsorted(cell_keys, low + (spans[boxes, 2] - 1), side="right")
            - first
        )
        hit = counts > 0
        hit_columns.append(
            tuple(values[hit].astype(index_type) for values in (boxes, first, counts))
        )
    member_keys = np.empty(
        sum(
            int(counts.sum(dtype=np.int64))
            for _boxes, _first, counts in hit_columns
        ),
        dtype=np.int64,
    )
    position = 0
    for boxes, first, counts in hit_columns:
        ends = np.cumsum(counts, dtype=np.int64)
        count = int(ends[-1]) if len(ends) else 0
        member_cells = np.repeat(first, counts) + (
            np.arange(count) - np.repeat(ends - counts, counts)
        )
        np.add(
            member_cells * box_count,
            np.repeat(boxes, counts),
            out=member_keys[position : position + count],
        )
        position += count
    del hit_columns
    member_keys.sort()
    offsets = np.searchsorted(
        member_keys, np.arange(len(unique_cells) + 1) * box_count
    )
    member_keys %= box_count
    return cell_index, offsets, member_keys


def _sequential_dots(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    return (
        left[:, 0] * right[:, 0]
        + left[:, 1] * right[:, 1]
        + left[:, 2] * right[:, 2]
    )


def _nearest_opposed_faces(
    candidates: np.ndarray,
    grid_members: tuple[np.ndarray, np.ndarray, np.ndarray],
    normals: np.ndarray,
    triangles: np.ndarray,
    centroids: np.ndarray,
    maximum_gap: float,
    progress: Callable[[str], None] | None,
) -> tuple[np.ndarray, np.ndarray]:
    """Each candidate face with the nearest opposed face under its centroid.

    Faces sharing a grid cell with more than 16 other candidates take their
    point-in-triangle dot products from ``np.einsum``, which sums in a
    different order than the sequential products used for the others.
    """
    cell_index, member_offsets, members = grid_members
    member_starts = member_offsets[cell_index]
    member_counts = member_offsets[cell_index + 1] - member_starts
    rough_normals = [
        normals[candidates, axis].astype(np.float32) for axis in range(3)
    ]
    found_faces: list[np.ndarray] = []
    found_others: list[np.ndarray] = []
    last_progress_at = time.perf_counter()
    for start, end in _chunk_bounds(member_counts, _THIN_WALL_PAIR_CHUNK):
        if (
            progress is not None
            and time.perf_counter() - last_progress_at >= 5.0
        ):
            progress(
                "Matching thin wall faces: "
                f"{start:,}/{len(candidates):,} candidates"
            )
            last_progress_at = time.perf_counter()
        counts = member_counts[start:end]
        face_positions = np.repeat(np.arange(start, end), counts)
        other_positions = members[
            np.arange(len(face_positions))
            + np.repeat(member_starts[start:end] - (np.cumsum(counts) - counts), counts)
        ]
        is_self = other_positions == face_positions
        sequential_faces = (
            counts
            - np.bincount(face_positions[is_self] - start, minlength=end - start)
        ) <= 16
        x, y, z = (
            np.repeat(component[start:end], counts) * component[other_positions]
            for component in rough_normals
        )
        # Float32 dot products of unit normals are within 1e-6 of the float64
        # ones, so this keeps every pair the exact test below accepts.
        maybe_opposed = (x + y + z <= np.float32(-0.8999)) & ~is_self
        face_positions = face_positions[maybe_opposed]
        face_index = candidates[face_positions]
        other_index = candidates[other_positions[maybe_opposed]]
        face_normals = normals[face_index]
        other_normals = normals[other_index]
        normal_dots = _sequential_dots(face_normals, other_normals)

        opposed = normal_dots <= -0.9
        face_positions = face_positions[opposed]
        face_index = face_index[opposed]
        other_index = other_index[opposed]
        face_normals = face_normals[opposed]
        other_normals = other_normals[opposed]
        normal_dots = normal_dots[opposed]
        centroid = centroids[face_index]
        distances = (
            _sequential_dots(triangles[other_index, 0] - centroid, other_normals)
            / normal_dots
        )
        gaps = np.abs(distances)

        close = (gaps >= 0.005) & (gaps <= maximum_gap)
        face_positions = face_positions[close]
        face_index = face_index[close]
        other_index = other_index[close]
        gaps = gaps[close]
        distances = distances[close]
        other_triangles = triangles[other_index]
        points = centroid[close] + distances[:, None] * face_normals[close]
        first_edge = other_triangles[:, 1] - other_triangles[:, 0]
        second_edge = other_triangles[:, 2] - other_triangles[:, 0]
        relative = points - other_triangles[:, 0]
        sequential = sequential_faces[face_positions - start]

        def dots(left: np.ndarray, right: np.ndarray) -> np.ndarray:
            return np.where(
                sequential,
                _sequential_dots(left, right),
                np.einsum("ij,ij->i", left, right),
            )

        dot_00 = dots(first_edge, first_edge)
        dot_01 = dots(first_edge, second_edge)
        dot_11 = dots(second_edge, second_edge)
        dot_20 = dots(relative, first_edge)
        dot_21 = dots(relative, second_edge)
        denominator = dot_00 * dot_11 - dot_01 * dot_01
        inside = np.abs(denominator) >= 1e-12
        first = np.zeros_like(denominator)
        second = np.zeros_like(denominator)
        np.divide(
            dot_11 * dot_20 - dot_01 * dot_21,
            denominator,
            out=first,
            where=inside,
        )
        np.divide(
            dot_00 * dot_21 - dot_01 * dot_20,
            denominator,
            out=second,
            where=inside,
        )
        inside &= (
            (first >= -1e-5)
            & (second >= -1e-5)
            & (first + second <= 1.0 + 1e-5)
        )
        face_index = face_index[inside]
        other_index = other_index[inside]
        order = np.lexsort((gaps[inside], face_index))
        face_index = face_index[order]
        nearest = np.ones(len(order), dtype=bool)
        np.not_equal(face_index[1:], face_index[:-1], out=nearest[1:])
        found_faces.append(face_index[nearest])
        found_others.append(other_index[order][nearest])
    return np.concatenate(found_faces), np.concatenate(found_others)


def detect_thin_wall_shells(
    part: MeshPart,
    *,
    maximum_gap: float = 0.2,
    progress: Callable[[str], None] | None = None,
) -> tuple[ThinWallShell, ...]:
    faces = np.asarray(part.faces, dtype=np.uint32).reshape((-1, 3))
    if len(faces) < 2:
        return ()
    vertices = np.asarray(part.vertices, dtype=np.float64)
    triangles = vertices[faces]
    raw_normals = np.cross(
        triangles[:, 1] - triangles[:, 0],
        triangles[:, 2] - triangles[:, 0],
    )
    lengths = np.linalg.norm(raw_normals, axis=1)
    valid = lengths > 1e-8
    normals = np.zeros_like(raw_normals)
    normals[valid] = raw_normals[valid] / lengths[valid, None]
    areas = lengths * 0.5
    candidates = np.flatnonzero(
        valid
        & (areas >= 0.02)
        & (np.abs(normals[:, 2]) <= 0.65)
    )
    if len(candidates) < 2:
        return ()

    grid_size = max(0.5, maximum_gap * 2.0)
    candidate_triangles = triangles[candidates]
    grid_lower = np.floor(
        (
            np.min(candidate_triangles, axis=1)
            - maximum_gap
        )
        / grid_size
    ).astype(np.int64)
    grid_upper = np.floor(
        (
            np.max(candidate_triangles, axis=1)
            + maximum_gap
        )
        / grid_size
    ).astype(np.int64)
    centroids = np.mean(triangles, axis=1)
    candidate_cells = np.floor(centroids[candidates] / grid_size).astype(np.int64)
    nearest_faces, nearest_others = _nearest_opposed_faces(
        candidates,
        _grid_cell_members(grid_lower, grid_upper, candidate_cells),
        normals,
        triangles,
        centroids,
        maximum_gap,
        progress,
    )
    face_pairs = [
        tuple(pair)
        for pair in np.unique(
            np.sort(np.column_stack((nearest_faces, nearest_others)), axis=1),
            axis=0,
        ).tolist()
    ]
    if not face_pairs:
        return ()

    parent = list(range(len(face_pairs)))

    def root(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root = root(first)
        second_root = root(second)
        if first_root != second_root:
            parent[second_root] = first_root

    by_vertex: dict[int, list[int]] = defaultdict(list)
    for pair_index, (first, second) in enumerate(face_pairs):
        for vertex_index in np.unique(
            np.concatenate((faces[first], faces[second]))
        ):
            by_vertex[int(vertex_index)].append(pair_index)
    for pair_indices in by_vertex.values():
        for pair_index in pair_indices[1:]:
            union(pair_indices[0], pair_index)

    grouped: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for pair_index, pair in enumerate(face_pairs):
        grouped[root(pair_index)].append(pair)

    shells: list[ThinWallShell] = []
    for pairs in grouped.values():
        source_faces = sorted({face for pair in pairs for face in pair})
        component_vertices = vertices[
            np.unique(faces[source_faces].reshape(-1))
        ]
        extent = np.ptp(component_vertices, axis=0)
        paired_area = float(np.sum(areas[source_faces]) * 0.5)
        if (
            extent[2] < 0.25
            or extent[2] > 5.0
            or paired_area < 0.2
        ):
            continue
        shells.append(
            ThinWallShell(
                face_pairs=tuple(pairs),
                source_faces=tuple(source_faces),
            )
        )
    return tuple(shells)


def extract_thin_wall_templates(
    parts: Iterable[MeshPart],
    *,
    maximum_gap: float = 0.2,
    shell_cache: dict[bytes, tuple[ThinWallShell, ...]] | None = None,
    cache_stats: dict[str, int] | None = None,
    progress: Callable[[str], None] | None = None,
) -> tuple[list[MeshPart], list[ThinWallTemplate]]:
    cleaned_parts: list[MeshPart] = []
    templates: list[ThinWallTemplate] = []
    for part in parts:
        if not part.collision_eligible:
            cleaned_parts.append(part)
            continue
        cache_key = None
        if shell_cache is not None:
            digest = hashlib.sha256()
            for values in (part.vertices, part.faces):
                array = np.asarray(values)
                digest.update(array.dtype.str.encode("ascii"))
                digest.update(
                    np.asarray(array.shape, dtype=np.int64).tobytes()
                )
                digest.update(array.tobytes())
            digest.update(np.asarray(maximum_gap, dtype=np.float64).tobytes())
            cache_key = digest.digest()
        if cache_key is not None and cache_key in shell_cache:
            shells = shell_cache[cache_key]
            if cache_stats is not None:
                cache_stats["hits"] = cache_stats.get("hits", 0) + 1
        else:
            shells = detect_thin_wall_shells(
                part,
                maximum_gap=maximum_gap,
                progress=(
                    (
                        lambda message: progress(
                            f"Original thin wall {part.name}: {message}"
                        )
                    )
                    if progress is not None
                    else None
                ),
            )
            if cache_key is not None:
                shell_cache[cache_key] = shells
            if cache_stats is not None:
                cache_stats["misses"] = cache_stats.get("misses", 0) + 1
        if not shells:
            cleaned_parts.append(part)
            continue
        removed_faces = {
            face_index
            for shell in shells
            for face_index in shell.source_faces
        }
        keep_mask = np.ones(len(part.faces), dtype=bool)
        keep_mask[list(removed_faces)] = False
        if np.any(keep_mask):
            cleaned_parts.append(
                replace(part, faces=np.asarray(part.faces)[keep_mask])
            )
        templates.extend(ThinWallTemplate(part, shell) for shell in shells)
    return cleaned_parts, templates


_ROUTE_POINT_BATCH_ELEMENTS = 262_144
_ROUTE_BOUND_STRIDE = 16
_ROUTE_POINT_CHUNK = 256
_THIN_WALL_THICKNESS = 0.5


def _nearest_route_points(
    points: np.ndarray,
    route_positions: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(points, dtype=np.float64)
    route = np.asarray(route_positions, dtype=np.float64)
    nearest = np.empty_like(points)
    distances = np.empty(len(points), dtype=np.float64)
    if len(points) > _ROUTE_POINT_CHUNK and len(route) > 1:
        # Chunks of points near the same route samples keep the segment
        # bound below tight; each point's result is independent of its chunk.
        samples = route[::_ROUTE_BOUND_STRIDE, :2]
        sample_indices = np.concatenate(
            [
                np.argmin(
                    np.sum(
                        (points[begin : begin + _ROUTE_POINT_CHUNK, None, :2] - samples)
                        ** 2,
                        axis=2,
                    ),
                    axis=1,
                )
                for begin in range(0, len(points), _ROUTE_POINT_CHUNK)
            ]
        )
        order = np.argsort(sample_indices, kind="stable")
        for begin in range(0, len(points), _ROUTE_POINT_CHUNK):
            chunk = order[begin : begin + _ROUTE_POINT_CHUNK]
            nearest[chunk], distances[chunk] = _nearest_route_points(
                points[chunk],
                route,
            )
        return nearest, distances
    if len(route) == 1:
        nearest[:] = route[0]
        distances[:] = np.linalg.norm(points[:, :2] - route[0, :2], axis=1)
        return nearest, distances
    if not len(points):
        return nearest, distances
    starts = route[:-1]
    segment_vectors = route[1:] - starts
    # Every route vertex lies on a segment, so the farthest point's distance
    # to its nearest sampled vertex bounds how far any nearest segment can be.
    sample_delta = points[:, None, :2] - route[::_ROUTE_BOUND_STRIDE, :2]
    reach = float(
        np.sqrt(np.max(np.min(np.sum(sample_delta * sample_delta, axis=2), axis=1)))
    ) + 1e-3
    if np.isfinite(reach):
        near = np.all(
            (np.maximum(starts[:, :2], route[1:, :2]) >= points[:, :2].min(axis=0) - reach)
            & (np.minimum(starts[:, :2], route[1:, :2]) <= points[:, :2].max(axis=0) + reach),
            axis=1,
        )
        starts = starts[near]
        segment_vectors = segment_vectors[near]
    segment_lengths_squared = np.einsum(
        "ij,ij->i",
        segment_vectors[:, :2],
        segment_vectors[:, :2],
    )
    valid = segment_lengths_squared > 1e-12
    valid_starts = starts[valid, :2]
    valid_vectors = segment_vectors[valid, :2]
    valid_lengths_squared = segment_lengths_squared[valid]
    batch_size = max(1, _ROUTE_POINT_BATCH_ELEMENTS // len(starts))
    for begin in range(0, len(points), batch_size):
        batch = points[begin : begin + batch_size]
        count = len(batch)
        relative = (batch[:, None, :2] - valid_starts).reshape((-1, 2))
        parameters = np.zeros((count, len(starts)), dtype=np.float64)
        parameters[:, valid] = np.einsum(
            "ij,ij->i",
            relative,
            np.tile(valid_vectors, (count, 1)),
        ).reshape((count, len(valid_lengths_squared))) / valid_lengths_squared
        parameters = np.clip(parameters, 0.0, 1.0)
        projected = starts + parameters[:, :, None] * segment_vectors
        delta = projected[:, :, :2] - batch[:, None, :2]
        flat_delta = delta.reshape((-1, 2))
        nearest_indices = np.argmin(
            np.einsum("ij,ij->i", flat_delta, flat_delta).reshape(
                (count, len(starts))
            ),
            axis=1,
        )
        rows = np.arange(count)
        nearest[begin : begin + count] = projected[rows, nearest_indices]
        closest = delta[rows, nearest_indices]
        distances[begin : begin + count] = np.sqrt(
            np.matmul(closest[:, None, :], closest[:, :, None])[:, 0, 0]
        )
    return nearest, distances


def _thin_wall_world_vertices(
    part: MeshPart,
    matrix: np.ndarray | None,
) -> np.ndarray:
    source_vertices = np.asarray(part.vertices, dtype=np.float64)
    if matrix is None:
        return source_vertices
    homogeneous = np.column_stack(
        (source_vertices, np.ones(len(source_vertices), dtype=np.float64))
    )
    return (homogeneous @ np.asarray(matrix).T)[:, :3]


def _route_queries(
    queries: Sequence[np.ndarray],
    route_positions: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray]]:
    """``_nearest_route_points`` of each query, answered by one call."""
    nearest, distances = _nearest_route_points(
        np.concatenate(queries),
        route_positions,
    )
    bounds = np.cumsum([len(query) for query in queries])[:-1]
    return list(zip(np.split(nearest, bounds), np.split(distances, bounds)))


def _inflate_part_thin_walls(
    templates: Sequence[ThinWallTemplate],
    route_positions: np.ndarray,
    world_vertices: np.ndarray,
    names: Sequence[str],
) -> list[MeshPart | None]:
    """Inflate the thin walls of one part, with one route query per step."""
    source_faces = np.asarray(templates[0].part.faces, dtype=np.uint32)
    pair_faces = [
        np.asarray(template.shell.face_pairs, dtype=np.int64)
        for template in templates
    ]
    pair_queries = _route_queries(
        [
            np.mean(world_vertices[source_faces[faces.reshape(-1)]], axis=1)
            for faces in pair_faces
        ],
        route_positions,
    )
    inner_triangles: list[np.ndarray] = []
    for faces, (_nearest, pair_distances) in zip(pair_faces, pair_queries):
        pair_distances = pair_distances.reshape((-1, 2))
        inner_indices = np.where(
            pair_distances[:, 0] <= pair_distances[:, 1],
            faces[:, 0],
            faces[:, 1],
        )
        inner_indices = np.unique(inner_indices)
        inner_triangles.append(world_vertices[source_faces[inner_indices]].copy())
    inner_centroids = [np.mean(triangles, axis=1) for triangles in inner_triangles]
    centroid_queries = _route_queries(inner_centroids, route_positions)
    welded: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for triangles, centroids, (nearest_centroids, _distances) in zip(
        inner_triangles,
        inner_centroids,
        centroid_queries,
    ):
        roadward = nearest_centroids - centroids
        roadward[:, 2] = 0.0
        raw_normals = np.cross(
            triangles[:, 1] - triangles[:, 0],
            triangles[:, 2] - triangles[:, 0],
        )
        reverse = np.einsum("ij,ij->i", raw_normals, roadward) < 0
        triangles[reverse] = triangles[reverse][:, (0, 2, 1)]
        inner_vertices, inverse = np.unique(
            triangles.reshape((-1, 3)).astype(np.float32),
            axis=0,
            return_inverse=True,
        )
        welded.append(
            (
                inner_vertices,
                inverse.reshape((-1, 3)).astype(np.uint32),
                np.mean(inner_vertices, axis=0, keepdims=True),
            )
        )
    vertex_queries = _route_queries(
        [np.concatenate((vertices, center)) for vertices, _faces, center in welded],
        route_positions,
    )
    return [
        _inflated_thin_wall(
            templates[0].part,
            inner_vertices,
            inner_faces,
            nearest[:-1],
            component_center,
            nearest[-1:],
            name,
        )
        for (inner_vertices, inner_faces, component_center), (nearest, _distances), name in zip(
            welded,
            vertex_queries,
            names,
        )
    ]


def _inflated_thin_wall(
    part: MeshPart,
    inner_vertices: np.ndarray,
    inner_faces: np.ndarray,
    nearest_vertices: np.ndarray,
    component_center: np.ndarray,
    center_nearest: np.ndarray,
    name: str,
) -> MeshPart | None:
    outward = inner_vertices.astype(np.float64) - nearest_vertices
    outward[:, 2] = 0.0
    lengths = np.linalg.norm(outward, axis=1)
    fallback = component_center[0] - center_nearest[0]
    fallback[2] = 0.0
    fallback_length = float(np.linalg.norm(fallback))
    if fallback_length < 1e-8:
        return None
    fallback /= fallback_length
    invalid = lengths < 1e-8
    outward[~invalid] /= lengths[~invalid, None]
    outward[invalid] = fallback
    outer_vertices = (
        inner_vertices.astype(np.float64) + outward * _THIN_WALL_THICKNESS
    ).astype(np.float32)
    outer_offset = len(inner_vertices)
    outer_faces = inner_faces[:, (0, 2, 1)] + outer_offset

    edge_counts: dict[tuple[int, int], int] = defaultdict(int)
    oriented_edges: dict[tuple[int, int], tuple[int, int]] = {}
    for face in inner_faces.tolist():
        for first, second in (
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        ):
            key = tuple(sorted((first, second)))
            edge_counts[key] += 1
            oriented_edges[key] = first, second
    side_faces = []
    for key, count in edge_counts.items():
        if count != 1:
            continue
        first, second = oriented_edges[key]
        side_faces.extend(
            (
                (first, second + outer_offset, second),
                (first, first + outer_offset, second + outer_offset),
            )
        )
    if not side_faces:
        return None
    faces = np.concatenate(
        (
            inner_faces,
            outer_faces,
            np.asarray(side_faces, dtype=np.uint32),
        )
    )
    vertices = np.concatenate((inner_vertices, outer_vertices))
    texcoords = np.zeros((len(vertices), 2), dtype=np.float32)
    return MeshPart(
        name=name,
        vertices=vertices,
        faces=faces,
        normals=np.zeros((len(vertices), 3), dtype=np.float32),
        texcoords=texcoords,
        colors=np.ones((len(vertices), 4), dtype=np.float32),
        material_index=part.material_index,
        material_name=part.material_name,
        collision_eligible=True,
        water=False,
        texcoord_sets=(texcoords,),
    )


def _restore_thin_wall_template(
    template: ThinWallTemplate,
    *,
    matrix: np.ndarray | None,
    name: str,
) -> MeshPart:
    source_faces = np.asarray(template.part.faces, dtype=np.uint32)[
        list(template.shell.source_faces)
    ]
    restored = _compact_collision_part(template.part, source_faces)
    assert restored is not None
    vertices = np.asarray(restored.vertices, dtype=np.float64)
    if matrix is not None:
        homogeneous = np.column_stack(
            (vertices, np.ones(len(vertices), dtype=np.float64))
        )
        vertices = (homogeneous @ np.asarray(matrix, dtype=np.float64).T)[:, :3]
    return replace(
        restored,
        name=name,
        vertices=np.asarray(vertices, dtype=np.float32),
    )


def inflate_thin_wall_templates(
    templates: Iterable[ThinWallTemplate],
    route_positions: np.ndarray,
    *,
    matrix: np.ndarray | None = None,
    name_prefix: str = "thin_wall",
) -> tuple[list[MeshPart], dict[str, int]]:
    repairs: list[MeshPart] = []
    templates = tuple(templates)
    repaired = 0
    restored = 0
    removed_faces = 0
    generated_faces = 0
    inflated: list[MeshPart | None] = []
    start = 0
    while start < len(templates):
        part = templates[start].part
        end = start + 1
        while end < len(templates) and templates[end].part is part:
            end += 1
        inflated.extend(
            _inflate_part_thin_walls(
                templates[start:end],
                route_positions,
                _thin_wall_world_vertices(part, matrix),
                [f"{name_prefix}_{index}" for index in range(start, end)],
            )
        )
        start = end
    for index, (template, repair) in enumerate(zip(templates, inflated)):
        if repair is not None:
            repairs.append(repair)
            repaired += 1
            removed_faces += len(template.shell.source_faces)
            generated_faces += len(repair.faces)
        else:
            repairs.append(
                _restore_thin_wall_template(
                    template,
                    matrix=matrix,
                    name=f"{name_prefix}_{index}_original",
                )
            )
            restored += 1
    return repairs, {
        "detected": len(templates),
        "repaired": repaired,
        "restored": restored,
        "removedFaces": removed_faces,
        "generatedFaces": generated_faces,
    }


def _render_sources(
    part: MeshPart,
) -> tuple[list[tuple[str, np.ndarray, int, tuple[str, ...]]], bool]:
    """A render part's float sources in write order, and whether it has vertex colors."""
    has_vertex_colors = (
        part.force_color_stream
        or not np.all(np.asarray(part.colors) == 1.0)
    )
    texcoord_sets = part.texcoord_sets or (part.texcoords,)
    sources = [
        ("positions", part.vertices, 3, ("X", "Y", "Z")),
        ("normals", part.normals, 3, ("X", "Y", "Z")),
    ]
    sources.extend(
        (
            f"texcoords{texcoord_index}",
            texcoord_values,
            2,
            ("S", "T"),
        )
        for texcoord_index, texcoord_values in enumerate(texcoord_sets)
    )
    if has_vertex_colors:
        sources.append(("colors", part.colors, 4, ("R", "G", "B", "A")))
    return sources, has_vertex_colors


def write_collada(
    destination: Path | TextIO,
    parts: list[MeshPart],
    *,
    collision_only: bool = False,
    collision_parts: list[MeshPart] | None = None,
    render_detail_size: int | None = None,
    null_detail_sizes: tuple[int, ...] = (),
    progress: Callable[[str], None] | None = None,
    collision_compaction_cache: dict[bytes, MeshPart | None] | None = None,
) -> tuple[np.ndarray, np.ndarray, int]:
    if render_detail_size is not None and render_detail_size <= 0:
        raise ConversionError("Render detail size must be positive")
    if any(detail_size <= 0 for detail_size in null_detail_sizes):
        raise ConversionError("Null detail sizes must be positive")
    all_vertices = np.concatenate([part.vertices for part in parts], axis=0)
    bounds_min = all_vertices.min(axis=0)
    bounds_max = all_vertices.max(axis=0)
    collision_sources = collision_parts if collision_parts is not None else parts
    material_names = sorted(
        {
            part.material_name
            for part in [*parts, *collision_sources]
        }
    )
    entries: list[tuple[list[MeshPart], str]] = []
    if collision_only:
        compacted_collision_parts: list[MeshPart] = []
        for index, part in enumerate(collision_sources, 1):
            if (
                progress is not None
                and (
                    index == 1
                    or index % 100 == 0
                    or index == len(collision_sources)
                )
            ):
                progress(
                    "Compacting collision meshes: "
                    f"{index:,}/{len(collision_sources):,}"
                )
            faces = _collision_faces(part)
            part_digest = _collision_part_digest(part, faces)
            if (
                collision_compaction_cache is not None
                and part_digest in collision_compaction_cache
            ):
                collision_part = collision_compaction_cache[part_digest]
            else:
                collision_part = _collision_geometry_part(part, faces)
                if collision_compaction_cache is not None:
                    collision_compaction_cache[part_digest] = collision_part
            if collision_part:
                compacted_collision_parts.append(collision_part)
        if compacted_collision_parts:
            entries.append((compacted_collision_parts, "Colmesh-1"))
    else:
        entries.extend(
            (
                [compact_part],
                (
                    compact_part.name
                    if render_detail_size is None
                    else f"{compact_part.name}_L{render_detail_size}"
                ),
            )
            for compact_part in (_compact_mesh_part(part) for part in parts)
        )
        collision_index = 1
        for part in collision_sources:
            collision_part = _collision_geometry_part(part, _collision_faces(part))
            if collision_part is None:
                continue
            entries.append(
                (
                    [collision_part],
                    f"Colmesh-{collision_index}",
                )
            )
            collision_index += 1
    collision_triangle_count = sum(
        len(part.faces)
        for entry_parts, node_name in entries
        if node_name.casefold().startswith("colmesh")
        for part in entry_parts
    )
    render_sources = {
        part_index: _render_sources(entry_parts[0])
        for part_index, (entry_parts, node_name) in enumerate(entries)
        if not node_name.casefold().startswith("colmesh")
    }
    # Parts can share arrays, such as blend pass copies of a part, so each
    # shared array is formatted once and its text kept until its last use.
    remaining_uses = Counter(
        id(values)
        for sources, _has_vertex_colors in render_sources.values()
        for _source_name, values, _stride, _params in sources
    )
    shared_texts: dict[int, str] = {}

    stream_context = (
        current_filesystem().open(
            destination,
            "w",
            encoding="utf-8",
            newline="\n",
        )
        if isinstance(destination, Path)
        else nullcontext(destination)
    )
    with stream_context as stream:
        stream.write('<?xml version="1.0" encoding="utf-8"?>\n')
        stream.write('<COLLADA xmlns="http://www.collada.org/2005/11/COLLADASchema" version="1.4.1">\n')
        stream.write(
            "<asset><contributor><authoring_tool>RBR2BeamNG</authoring_tool></contributor>"
            "<unit name=\"meter\" meter=\"1\"/><up_axis>Z_UP</up_axis></asset>\n"
        )
        stream.write("<library_effects>\n")
        for material_name in material_names:
            material_id = _xml_id(material_name)
            stream.write(
                f'<effect id="{material_id}-effect"><profile_COMMON><technique sid="common"><phong>'
                "<diffuse><color>1 1 1 1</color></diffuse>"
                "</phong></technique></profile_COMMON></effect>\n"
            )
        stream.write("</library_effects>\n<library_materials>\n")
        for material_name in material_names:
            material_id = _xml_id(material_name)
            stream.write(
                f'<material id="{material_id}-material" name="{escape(material_name)}">'
                f'<instance_effect url="#{material_id}-effect"/></material>\n'
            )
        stream.write("</library_materials>\n<library_geometries>\n")
        for part_index, (entry_parts, node_name) in enumerate(entries):
            if (
                progress is not None
                and (
                    part_index == 0
                    or (part_index + 1) % 100 == 0
                    or part_index + 1 == len(entries)
                )
            ):
                progress(
                    "Writing collision geometry: "
                    f"{part_index + 1:,}/{len(entries):,}"
                )
            geometry_id = f"geometry_{part_index}"
            is_collision = node_name.casefold().startswith("colmesh")
            stream.write(f'<geometry id="{geometry_id}" name="{escape(node_name)}"><mesh>\n')
            if is_collision:
                source_id = f"{geometry_id}-positions"
                _write_collada_source(
                    stream,
                    source_id,
                    (part.vertices for part in entry_parts),
                    3,
                    ("X", "Y", "Z"),
                )
                stream.write(
                    f'<vertices id="{geometry_id}-vertices"><input '
                    f'semantic="POSITION" source="#{source_id}"/></vertices>\n'
                )
                vertex_offset = 0
                for part in entry_parts:
                    material_symbol = f"{_xml_id(part.material_name)}-symbol"
                    stream.write(
                        f'<triangles material="{material_symbol}" '
                        f'count="{len(part.faces)}"><input semantic="VERTEX" '
                        f'source="#{geometry_id}-vertices" offset="0"/><p>'
                    )
                    _write_indices(stream, part.faces, vertex_offset)
                    stream.write("</p></triangles>\n")
                    vertex_offset += len(part.vertices)
            else:
                part = entry_parts[0]
                sources, has_vertex_colors = render_sources[part_index]
                texcoord_sets = part.texcoord_sets or (part.texcoords,)
                for source_name, values, stride, params in sources:
                    key = id(values)
                    float_text = shared_texts.get(key)
                    remaining_uses[key] -= 1
                    if float_text is None and remaining_uses[key]:
                        buffer = io.StringIO()
                        _write_floats(buffer, values)
                        float_text = shared_texts[key] = buffer.getvalue()
                    if not remaining_uses[key]:
                        shared_texts.pop(key, None)
                    _write_collada_source(
                        stream,
                        f"{geometry_id}-{source_name}",
                        (values,),
                        stride,
                        params,
                        float_text,
                    )
                vertices_id = f"{geometry_id}-vertices"
                stream.write(
                    f'<vertices id="{vertices_id}"><input semantic="POSITION" source="#{geometry_id}-positions"/></vertices>\n'
                )
                material_symbol = f"{_xml_id(part.material_name)}-symbol"
                stream.write(f'<triangles material="{material_symbol}" count="{len(part.faces)}">')
                stream.write(f'<input semantic="VERTEX" source="#{vertices_id}" offset="0"/>')
                stream.write(f'<input semantic="NORMAL" source="#{geometry_id}-normals" offset="0"/>')
                for texcoord_index in range(len(texcoord_sets)):
                    stream.write(
                        f'<input semantic="TEXCOORD" '
                        f'source="#{geometry_id}-texcoords{texcoord_index}" '
                        f'offset="0" set="{texcoord_index}"/>'
                    )
                if has_vertex_colors:
                    stream.write(f'<input semantic="COLOR" source="#{geometry_id}-colors" offset="0" set="0"/>')
                stream.write("<p>")
                _write_indices(stream, part.faces)
                stream.write("</p></triangles>\n")
            stream.write("</mesh></geometry>\n")
        stream.write(
            "</library_geometries>\n<library_visual_scenes><visual_scene id=\"Scene\" name=\"Scene\">\n"
            '<node id="base00" name="base00" type="NODE"><node id="start01" name="start01" type="NODE">\n'
        )
        for part_index, (entry_parts, node_name) in enumerate(entries):
            stream.write(
                f'<node id="node_{part_index}" name="{escape(node_name)}" type="NODE">'
                f'<instance_geometry url="#geometry_{part_index}"><bind_material><technique_common>'
            )
            for material_name in dict.fromkeys(
                part.material_name for part in entry_parts
            ):
                material_id = _xml_id(material_name)
                stream.write(
                    f'<instance_material symbol="{material_id}-symbol" '
                    f'target="#{material_id}-material"/>'
                )
            stream.write(
                "</technique_common></bind_material></instance_geometry></node>\n"
            )
        requested_null_details = set(null_detail_sizes)
        if collision_only:
            requested_null_details.add(2)
        for detail_size in sorted(requested_null_details, reverse=True):
            stream.write(
                f'<node id="nulldetail{detail_size}" '
                f'name="nulldetail{detail_size}" type="NODE"/>\n'
            )
        stream.write(
            '</node></node></visual_scene></library_visual_scenes>'
            '<scene><instance_visual_scene url="#Scene"/></scene>\n</COLLADA>\n'
        )
    return bounds_min, bounds_max, collision_triangle_count


def driveline_range(stage: RbrStage) -> tuple[int, int]:
    spawn_position = source_position_to_beamng(stage.spawn.position)
    start_index = min(
        range(len(stage.driveline)),
        key=lambda index: np.linalg.norm(
            source_position_to_beamng(stage.driveline[index].position)
            - spawn_position
        ),
    )
    start_distance = stage.driveline[start_index].distance
    finish_distance = next(
        (
            note.distance
            for note in sorted(stage.pacenotes, key=lambda item: item.distance)
            if note.note_type == 22 and note.distance >= start_distance
        ),
        stage.driveline[-1].distance,
    )
    finish_index = min(range(len(stage.driveline)), key=lambda index: abs(stage.driveline[index].distance - finish_distance))
    if finish_index <= start_index:
        start_index, finish_index = 0, len(stage.driveline) - 1
    return start_index, finish_index


def quaternion_from_rotation(rotation_values: list[float]) -> list[float]:
    matrix = np.asarray(rotation_values, dtype=np.float64).reshape((3, 3), order="F")
    trace = float(np.trace(matrix))
    if trace > 0:
        scale = math.sqrt(trace + 1.0) * 2.0
        quaternion = np.array(
            (
                (matrix[2, 1] - matrix[1, 2]) / scale,
                (matrix[0, 2] - matrix[2, 0]) / scale,
                (matrix[1, 0] - matrix[0, 1]) / scale,
                0.25 * scale,
            )
        )
    else:
        index = int(np.argmax(np.diag(matrix)))
        if index == 0:
            scale = math.sqrt(1.0 + matrix[0, 0] - matrix[1, 1] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                (
                    0.25 * scale,
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[2, 1] - matrix[1, 2]) / scale,
                )
            )
        elif index == 1:
            scale = math.sqrt(1.0 + matrix[1, 1] - matrix[0, 0] - matrix[2, 2]) * 2.0
            quaternion = np.array(
                (
                    (matrix[0, 1] + matrix[1, 0]) / scale,
                    0.25 * scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    (matrix[0, 2] - matrix[2, 0]) / scale,
                )
            )
        else:
            scale = math.sqrt(1.0 + matrix[2, 2] - matrix[0, 0] - matrix[1, 1]) * 2.0
            quaternion = np.array(
                (
                    (matrix[0, 2] + matrix[2, 0]) / scale,
                    (matrix[1, 2] + matrix[2, 1]) / scale,
                    0.25 * scale,
                    (matrix[1, 0] - matrix[0, 1]) / scale,
                )
            )
    quaternion /= np.linalg.norm(quaternion)
    return quaternion.tolist()
