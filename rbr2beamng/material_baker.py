from __future__ import annotations

import hashlib
import math
from collections.abc import Iterator
from dataclasses import dataclass, replace
from functools import lru_cache
from pathlib import Path

import numpy as np
import xatlas
from PIL import Image

from .core import ConversionError, slugify
from .filesystem import current_filesystem
from .geometry import remap_mesh_part
from .models import MeshPart, PbrMaterialOverride, RbrMaterial
from .profiling import profile_span
from .rbr import texture_has_alpha


@dataclass(frozen=True)
class VertexLerpBake:
    part: MeshPart
    material: PbrMaterialOverride
    method: str


@dataclass(frozen=True)
class _EffectLayout:
    second_uv: int | None
    normal_uv: int | None = None
    specular_uv: int | None = None
    multiplier_uv: int | None = None
    additive_uv: int | None = None
    # RBR RLDoubleTexture*.pso: the blend weight is t1.a * v0.a, and output
    # alpha is v0.a * t0.a, lerped towards t1.a unless the shader is specular.
    rbr_double_texture: bool = False


@dataclass(frozen=True)
class AtlasBakeProfile:
    name: str
    target_resolution: int
    texels_per_unit: float
    max_dimension: int
    padding: int = 16


ATLAS_BAKE_PROFILE = AtlasBakeProfile(
    name="high",
    target_resolution=8192,
    texels_per_unit=64.0,
    max_dimension=8192,
)

BLEND_MASK_BAKE_PROFILES = {
    "btb_diffusevertlerp": AtlasBakeProfile("blend-mask", 1024, 8.0, 1024, 4),
    "btb_diffusebumpvertlerp": AtlasBakeProfile("blend-mask", 1024, 8.0, 1024, 4),
    "btb_diffusespecularvertlerp": AtlasBakeProfile("blend-mask", 1024, 8.0, 1024, 4),
    "btb_diffusebumpspecularvertlerp": AtlasBakeProfile(
        "blend-mask",
        1024,
        8.0,
        1024,
        4,
    ),
}

_EFFECT_LAYOUTS = {
    "btb_diffusevertlerp": _EffectLayout(second_uv=1),
    "btb_diffusebumpvertlerp": _EffectLayout(second_uv=2, normal_uv=1),
    "btb_diffusespecularvertlerp": _EffectLayout(second_uv=2, specular_uv=1),
    "btb_diffusebumpspecularvertlerp": _EffectLayout(
        second_uv=3,
        normal_uv=1,
        specular_uv=2,
    ),
    "btb_diffusemultiplyaddvertlerp": _EffectLayout(
        second_uv=3,
        multiplier_uv=1,
        additive_uv=2,
    ),
}
_CHUNKED_RASTER_PIXELS = 16_384
_RASTER_CHUNK_PIXELS = 32_768
_ORIGINAL_RBR_LEGACY_SPECULAR_CLEAR_COAT_ROUGHNESS = 0.5


def is_vertex_lerp_material(material: RbrMaterial | None) -> bool:
    return bool(
        material
        and material.effect.casefold() in _EFFECT_LAYOUTS
    )


def vertex_lerp_uv_indices(effect: str) -> tuple[int, ...]:
    """Return the source UV streams required by a supported vertex-lerp effect."""
    layout = _EFFECT_LAYOUTS.get(effect.casefold())
    if layout is None:
        raise ConversionError(f"Unsupported vertex-lerp effect {effect!r}")
    return tuple(
        sorted(
            index
            for index in {
                0,
                layout.second_uv,
                layout.normal_uv,
                layout.specular_uv,
                layout.multiplier_uv,
                layout.additive_uv,
            }
            if index is not None
        )
    )


def vertex_lerp_effect_needs_custom_material(effect: str) -> bool:
    """Whether an effect has source maps the standard PBR override cannot retain."""
    layout = _EFFECT_LAYOUTS.get(effect.casefold())
    if layout is None:
        raise ConversionError(f"Unsupported vertex-lerp effect {effect!r}")
    return layout.multiplier_uv is not None or layout.additive_uv is not None


def is_original_bake_material(
    part: MeshPart,
    material: RbrMaterial | None,
) -> bool:
    return bool(
        material
        and material.effect.casefold() == "rbr_original"
        and material.diffuse_texture is not None
        and (
            (
                material.specular_texture is not None
                and material.properties.get("originalSourceKind", "").casefold()
                != "geom"
            )
            or (
                material.second_diffuse_texture is not None
                and material.uses_alpha
            )
        )
    )


def prepare_original_material_part(
    part: MeshPart,
    material: RbrMaterial,
) -> MeshPart:
    required_semantics = ["diffuse_1"]
    if material.second_diffuse_texture is not None:
        required_semantics.append("diffuse_2")
    source_indices = [
        part.semantic_uv_indices[semantic]
        for semantic in required_semantics
        if semantic in part.semantic_uv_indices
    ]
    if not source_indices:
        source_indices = [0]
    source_indices = list(dict.fromkeys(source_indices))
    texcoord_sets = tuple(part.texcoord_sets[index] for index in source_indices)
    if len(texcoord_sets) > 2:
        raise ConversionError(
            f"Original mesh {part.name!r} requires {len(texcoord_sets)} render UV sets"
        )
    index_remap = {
        source_index: output_index
        for output_index, source_index in enumerate(source_indices)
    }
    return replace(
        part,
        texcoords=texcoord_sets[0],
        texcoord_sets=texcoord_sets,
        semantic_uv_indices={
            semantic: index_remap[source_index]
            for semantic, source_index in part.semantic_uv_indices.items()
            if source_index in index_remap
        },
        specular_strengths=None,
    )


@lru_cache(maxsize=64)
def _load_rgba(path: Path) -> np.ndarray:
    filesystem = current_filesystem()
    try:
        source_bytes = filesystem.stat(path).st_size
    except OSError:
        source_bytes = 0
    with profile_span(
        "load_source_texture",
        category="material",
        path=str(path),
        sourceBytes=source_bytes,
    ) as texture_span:
        try:
            with Image.open(filesystem.read_path(path)) as source:
                result = np.asarray(
                    source.convert("RGBA"),
                    dtype=np.float32,
                )
        except OSError as exc:
            raise ConversionError(
                f"Unable to read BTB blend texture {path}: {exc}"
            ) from exc
        texture_span.update(
            width=result.shape[1],
            height=result.shape[0],
        )
        return result


def clear_temporary_material_state() -> None:
    _load_rgba.cache_clear()
    texture_has_alpha.cache_clear()


def _wrap(values: np.ndarray, size: int) -> None:
    if size & (size - 1):
        values %= size
    else:
        values &= size - 1


def _sample(image: np.ndarray, uv: np.ndarray) -> np.ndarray:
    height, width, channels = image.shape
    one = np.float32(1.0)
    u = uv[:, 0]
    v = uv[:, 1]
    x = (
        (u - np.floor(u)) * np.float32(width)
        - np.float32(0.5)
    )
    y = (
        (one - (v - np.floor(v))) * np.float32(height)
        - np.float32(0.5)
    )
    floor_x = np.floor(x)
    floor_y = np.floor(y)
    x0 = floor_x.astype(np.int64)
    y0 = floor_y.astype(np.int64)
    fx = (x - floor_x)[:, None]
    fy = (y - floor_y)[:, None]
    x1 = x0 + 1
    y1 = y0 + 1
    _wrap(x0, width)
    _wrap(x1, width)
    _wrap(y0, height)
    _wrap(y1, height)
    texels = image.reshape((-1, channels)).view(
        np.dtype((np.void, channels * image.itemsize))
    ).ravel()

    def texel(index: np.ndarray) -> np.ndarray:
        return texels[index].view(image.dtype).reshape((-1, channels))

    row0 = y0 * width
    row1 = y1 * width
    top = texel(row0 + x0)
    top += (texel(row0 + x1) - top) * fx
    bottom = texel(row1 + x0)
    bottom += (texel(row1 + x1) - bottom) * fx
    top += (bottom - top) * fy
    top /= np.float32(255.0)
    return top


def _srgb_to_linear(values: np.ndarray) -> np.ndarray:
    low = values <= 0.04045
    result = values.copy()
    np.add(result, 0.055, out=result)
    np.divide(result, 1.055, out=result)
    np.power(result, 2.4, out=result)
    np.divide(
        values,
        12.92,
        out=result,
        where=low,
    )
    return result


def _linear_to_srgb(values: np.ndarray) -> np.ndarray:
    low = values <= 0.0031308
    result = np.maximum(values, 0.0)
    np.power(result, 1.0 / 2.4, out=result)
    np.multiply(result, 1.055, out=result)
    np.subtract(result, 0.055, out=result)
    np.multiply(
        values,
        12.92,
        out=result,
        where=low,
    )
    return result


def _triangle_raster_samples(
    face: np.ndarray,
    uv: np.ndarray,
    width: int,
    height: int,
) -> Iterator[
    tuple[
        np.ndarray,
        bool,
        np.ndarray | None,
        np.ndarray | None,
        np.ndarray | None,
    ]
]:
    triangle = np.column_stack(
        (
            uv[face, 0] * width,
            (1.0 - uv[face, 1]) * height,
        )
    )
    minimum = np.maximum(0, np.floor(triangle.min(axis=0)).astype(int))
    maximum = np.minimum(
        (width - 1, height - 1),
        np.ceil(triangle.max(axis=0)).astype(int),
    )
    if np.any(maximum < minimum):
        yield triangle, False, None, None, None
        return

    first = triangle[1] - triangle[0]
    second = triangle[2] - triangle[0]
    denominator = first[0] * second[1] - first[1] * second[0]
    if abs(float(denominator)) <= 1e-10:
        yield triangle, True, None, None, None
        return

    xs = np.arange(minimum[0], maximum[0] + 1)
    row_count = int(maximum[1] - minimum[1] + 1)
    rows_per_chunk = row_count
    if len(xs) * row_count >= _CHUNKED_RASTER_PIXELS:
        rows_per_chunk = max(
            1,
            _RASTER_CHUNK_PIXELS // max(1, len(xs)),
        )
    sampled = False
    for start_y in range(
        int(minimum[1]),
        int(maximum[1]) + 1,
        rows_per_chunk,
    ):
        ys = np.arange(
            start_y,
            min(int(maximum[1]) + 1, start_y + rows_per_chunk),
        )
        target_x = np.tile(xs, len(ys))
        target_y = np.repeat(ys, len(xs))
        relative_x = target_x + 0.5 - triangle[0, 0]
        relative_y = target_y + 0.5 - triangle[0, 1]
        weight1 = (
            relative_x * second[1] - relative_y * second[0]
        ) / denominator
        weight2 = (
            first[0] * relative_y - first[1] * relative_x
        ) / denominator
        weight0 = (1.0 - weight1 - weight2).astype(np.float32)
        weight1 = weight1.astype(np.float32)
        weight2 = weight2.astype(np.float32)
        inside = (
            (weight0 >= -1e-5)
            & (weight1 >= -1e-5)
            & (weight2 >= -1e-5)
        )
        if not np.any(inside):
            continue
        sampled = True
        yield (
            triangle,
            True,
            target_x[inside],
            target_y[inside],
            np.column_stack(
                (
                    weight0[inside],
                    weight1[inside],
                    weight2[inside],
                )
            ),
        )
    if not sampled:
        yield triangle, True, None, None, None


def _dilate(
    values: np.ndarray,
    coverage: np.ndarray,
    iterations: int,
    *,
    wrap: bool,
    fill_remaining: bool = False,
) -> np.ndarray:
    result = values
    covered = coverage.copy()
    height, width = covered.shape
    channels = result.shape[2]
    bytes_per_row = max(1, width * channels * result.dtype.itemsize)
    rows_per_chunk = max(1, min(128, 16 * 1024**2 // bytes_per_row))
    frontier: np.ndarray | None = None
    for _iteration in range(iterations):
        if np.all(covered):
            break
        if frontier is None:
            candidates = np.zeros_like(covered)
            candidates[:-1] |= covered[1:]
            candidates[1:] |= covered[:-1]
            candidates[:, :-1] |= covered[:, 1:]
            candidates[:, 1:] |= covered[:, :-1]
            if wrap:
                candidates[-1] |= covered[0]
                candidates[0] |= covered[-1]
                candidates[:, -1] |= covered[:, 0]
                candidates[:, 0] |= covered[:, -1]
            candidates &= ~covered
            candidate_indices = np.flatnonzero(candidates)
        else:
            frontier_y, frontier_x = np.divmod(frontier, width)
            if wrap:
                candidate_indices = np.concatenate(
                    (
                        ((frontier_y - 1) % height) * width + frontier_x,
                        ((frontier_y + 1) % height) * width + frontier_x,
                        frontier_y * width + (frontier_x - 1) % width,
                        frontier_y * width + (frontier_x + 1) % width,
                    )
                )
            else:
                candidate_parts = (
                    (frontier_y > 0, frontier - width),
                    (frontier_y + 1 < height, frontier + width),
                    (frontier_x > 0, frontier - 1),
                    (frontier_x + 1 < width, frontier + 1),
                )
                candidate_indices = np.concatenate(
                    tuple(
                        values[valid]
                        for valid, values in candidate_parts
                        if np.any(valid)
                    )
                )
            candidate_indices = np.unique(candidate_indices)
            candidate_indices = candidate_indices[
                ~covered.reshape(-1)[candidate_indices]
            ]
        if not len(candidate_indices):
            break

        target_y, target_x = np.divmod(candidate_indices, width)
        value_sum = np.zeros(
            (len(candidate_indices), channels),
            dtype=result.dtype,
        )
        neighbour_count = np.zeros(len(candidate_indices), dtype=np.uint8)
        neighbours = (
            (target_y + 1, target_x),
            (target_y - 1, target_x),
            (target_y, target_x + 1),
            (target_y, target_x - 1),
        )
        for source_y, source_x in neighbours:
            if wrap:
                source_y %= height
                source_x %= width
                valid = np.ones(len(candidate_indices), dtype=bool)
            else:
                valid = (
                    (source_y >= 0)
                    & (source_y < height)
                    & (source_x >= 0)
                    & (source_x < width)
                )
                source_y = np.clip(source_y, 0, height - 1)
                source_x = np.clip(source_x, 0, width - 1)
            source_covered = valid & covered[source_y, source_x]
            np.add(
                value_sum,
                result[source_y, source_x],
                out=value_sum,
                where=source_covered[:, None],
            )
            neighbour_count += source_covered

        fill = neighbour_count > 0
        if not np.any(fill):
            break
        np.divide(
            value_sum,
            neighbour_count[:, None],
            out=value_sum,
            where=fill[:, None],
        )
        target_y = target_y[fill]
        target_x = target_x[fill]
        result[target_y, target_x] = value_sum[fill]
        covered[target_y, target_x] = True
        frontier = candidate_indices[fill]
    if fill_remaining and np.any(coverage) and not np.all(covered):
        total = np.zeros(channels, dtype=result.dtype)
        count = 0
        for start in range(0, height, rows_per_chunk):
            end = min(height, start + rows_per_chunk)
            mask = coverage[start:end]
            total += np.sum(
                result[start:end],
                axis=(0, 1),
                dtype=result.dtype,
                where=mask[..., None],
            )
            count += int(np.count_nonzero(mask))
        mean = total / max(1, count)
        for start in range(0, height, rows_per_chunk):
            end = min(height, start + rows_per_chunk)
            np.copyto(
                result[start:end],
                mean,
                where=(~covered[start:end])[..., None],
            )
    return result


def _quantize_unorm8(values: np.ndarray) -> np.ndarray:
    np.clip(values, 0.0, 1.0, out=values)
    np.multiply(values, 255.0, out=values)
    np.rint(values, out=values)
    return values.astype(np.uint8)


def _next_pow2(value: int) -> int:
    return 1 << math.ceil(math.log2(max(1, int(value))))


def _atlas(
    part: MeshPart,
    profile: AtlasBakeProfile,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int, int]:
    vertices = np.asarray(part.vertices, dtype=np.float32)
    faces = np.asarray(part.faces, dtype=np.uint32).reshape((-1, 3))
    try:
        atlas = xatlas.Atlas()
        atlas.add_mesh(vertices, faces)
        pack_options = xatlas.PackOptions()
        pack_options.padding = profile.padding
        pack_options.resolution = profile.target_resolution
        pack_options.texels_per_unit = profile.texels_per_unit
        atlas.generate(pack_options=pack_options)
    except RuntimeError as exc:
        raise ConversionError(f"xatlas failed to unwrap {part.name}: {exc}") from exc
    if atlas.atlas_count != 1:
        raise ConversionError(f"Unable to create one atlas for {part.name}")
    if max(atlas.width, atlas.height) > profile.max_dimension:
        raise ConversionError(
            f"Atlas for {part.name} exceeds {profile.name!r} profile budget"
        )
    vertex_mapping, atlas_faces, atlas_uv = atlas[0]
    pack_width = max(1, int(atlas.width))
    pack_height = max(1, int(atlas.height))
    packed_uv = np.asarray(atlas_uv, dtype=np.float32)
    # Crop to used charts, then pad each axis to POT so texconv -pow2 is a no-op.
    pixel_uv = packed_uv * np.asarray((pack_width, pack_height), dtype=np.float32)
    crop_minimum = np.maximum(
        0.0,
        np.floor(pixel_uv.min(axis=0) - profile.padding),
    )
    crop_maximum = np.minimum(
        (pack_width, pack_height),
        np.ceil(pixel_uv.max(axis=0) + profile.padding),
    )
    used_size = np.maximum(1.0, crop_maximum - crop_minimum)
    width = min(
        profile.max_dimension,
        _next_pow2(int(math.ceil(float(used_size[0])))),
    )
    height = min(
        profile.max_dimension,
        _next_pow2(int(math.ceil(float(used_size[1])))),
    )
    cropped_uv = (pixel_uv - crop_minimum) / np.asarray(
        (width, height),
        dtype=np.float32,
    )
    return (
        np.asarray(vertex_mapping, dtype=np.uint32),
        np.asarray(atlas_faces, dtype=np.uint32).reshape((-1, 3)),
        np.asarray(cropped_uv, dtype=np.float32),
        width,
        height,
    )


def _rasterize_color_atlas(
    layout: _EffectLayout,
    faces: np.ndarray,
    atlas_uv: np.ndarray,
    width: int,
    height: int,
    source_uvs: tuple[np.ndarray, ...],
    blend: np.ndarray | None,
    diffuse_a: np.ndarray,
    diffuse_b: np.ndarray | None,
    multiplier: np.ndarray | None,
    additive: np.ndarray | None,
    specular: np.ndarray | None,
    specular_strengths: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, dict[str, int]]:
    color_output = np.zeros((height, width, 4), dtype=np.float32)
    clear_coat_output = (
        np.zeros((height, width, 1), dtype=np.float32)
        if specular is not None
        else None
    )
    coverage = np.zeros((height, width), dtype=bool)
    rasterized_faces = 0
    empty_faces = 0
    sampled_pixels = 0

    for face in faces:
        face_samples = 0
        for (
            _triangle,
            _in_bounds,
            target_x,
            target_y,
            barycentric,
        ) in _triangle_raster_samples(face, atlas_uv, width, height):
            if barycentric is None:
                continue
            uv_a = barycentric @ source_uvs[0][face]
            color_a = _sample(diffuse_a, uv_a)
            color = color_a.copy()
            linear_color = _srgb_to_linear(color_a[:, :3])
            if (
                diffuse_b is not None
                and layout.second_uv is not None
                and blend is not None
            ):
                uv_b = barycentric @ source_uvs[layout.second_uv][face]
                weight = np.clip(
                    barycentric @ blend[face],
                    0.0,
                    1.0,
                )[:, None]
                color_b = _sample(diffuse_b, uv_b)
                if layout.rbr_double_texture:
                    vertex_alpha = weight
                    weight = color_b[:, 3:4] * vertex_alpha
                    alpha = color_a[:, 3:4]
                    if layout.specular_uv is None:
                        alpha = alpha + weight * (color_b[:, 3:4] - alpha)
                    color[:, 3:4] = vertex_alpha * alpha
                linear_color *= 1.0 - weight
                linear_b = _srgb_to_linear(color_b[:, :3])
                linear_b *= weight
                linear_color += linear_b
            color[:, :3] = linear_color
            if multiplier is not None and layout.multiplier_uv is not None:
                uv = barycentric @ source_uvs[layout.multiplier_uv][face]
                multiplier_sample = _sample(multiplier, uv)
                color[:, :3] *= _srgb_to_linear(multiplier_sample[:, :3])
            if additive is not None and layout.additive_uv is not None:
                uv = barycentric @ source_uvs[layout.additive_uv][face]
                additive_sample = _sample(additive, uv)
                color[:, :3] += _srgb_to_linear(additive_sample[:, :3])
            np.clip(color[:, :3], 0.0, 1.0, out=color[:, :3])
            color[:, :3] = _linear_to_srgb(color[:, :3])
            np.clip(color, 0.0, 1.0, out=color)
            color_output[target_y, target_x] = color
            if clear_coat_output is not None and layout.specular_uv is not None:
                uv = barycentric @ source_uvs[layout.specular_uv][face]
                specular_sample = _sample(specular, uv)
                clear_coat_output[target_y, target_x, 0] = np.clip(
                    (
                        specular_sample[:, 0] * 0.2126
                        + specular_sample[:, 1] * 0.7152
                        + specular_sample[:, 2] * 0.0722
                    )
                    * (
                        np.clip(
                            barycentric @ specular_strengths[face],
                            0.0,
                            1.0,
                        )
                        if specular_strengths is not None
                        else 1.0
                    ),
                    0.0,
                    1.0,
                )
            coverage[target_y, target_x] = True
            face_samples += len(target_x)
        if face_samples:
            rasterized_faces += 1
            sampled_pixels += face_samples
        else:
            empty_faces += 1
    return (
        color_output,
        clear_coat_output,
        coverage,
        {
            "rasterizedFaces": rasterized_faces,
            "emptyFaces": empty_faces,
            "sampledPixels": sampled_pixels,
        },
    )


def _bake_atlas(
    part: MeshPart,
    material: RbrMaterial,
    layout: _EffectLayout,
    output_dir: Path,
    stem: str,
    *,
    preserve_vertex_rgb: bool = False,
    clear_coat_roughness: float | None = None,
) -> VertexLerpBake:
    with profile_span(
        "unwrap_atlas",
        category="material",
        part=part.name,
        faces=len(part.faces),
        atlasBakeProfile=ATLAS_BAKE_PROFILE.name,
        targetResolution=ATLAS_BAKE_PROFILE.target_resolution,
        maxResolution=ATLAS_BAKE_PROFILE.max_dimension,
        texelsPerUnit=ATLAS_BAKE_PROFILE.texels_per_unit,
    ) as atlas_span:
        mapping, faces, atlas_uv, width, height = _atlas(
            part,
            ATLAS_BAKE_PROFILE,
        )
        atlas_span.update(
            width=width,
            height=height,
            outputVertices=len(mapping),
            outputFaces=len(faces),
        )
    source_uvs = tuple(
        np.asarray(channel, dtype=np.float32)[mapping]
        for channel in part.texcoord_sets
    )
    blend = (
        np.asarray(part.blend_weights, dtype=np.float32)[mapping]
        if part.blend_weights is not None
        else None
    )
    diffuse_a = _load_rgba(material.diffuse_texture)
    diffuse_b = (
        np.ascontiguousarray(_load_rgba(material.second_diffuse_texture))
        if layout.second_uv is not None and material.second_diffuse_texture
        else None
    )
    multiplier = (
        np.ascontiguousarray(
            _load_rgba(material.multiplier_texture)[:, :, :3]
        )
        if layout.multiplier_uv is not None and material.multiplier_texture
        else None
    )
    additive = (
        np.ascontiguousarray(
            _load_rgba(material.additive_texture)[:, :, :3]
        )
        if layout.additive_uv is not None and material.additive_texture
        else None
    )
    specular = (
        np.ascontiguousarray(
            _load_rgba(material.specular_texture)[:, :, :3]
        )
        if layout.specular_uv is not None and material.specular_texture
        else None
    )
    working_bytes = height * width * (
        4 * np.dtype(np.float32).itemsize
        + np.dtype(bool).itemsize
        + (
            np.dtype(np.float32).itemsize
            if specular is not None
            else 0
        )
    )
    with profile_span(
        "rasterize_color_atlas",
        category="material",
        width=width,
        height=height,
        outputPixels=width * height,
        faces=len(faces),
        sourceTextureCount=sum(
            value is not None
            for value in (
                diffuse_a,
                diffuse_b,
                multiplier,
                additive,
                specular,
            )
        ),
        hasMultiplier=multiplier is not None,
        hasAdditive=additive is not None,
        hasSpecular=specular is not None,
        workingBytesEstimate=working_bytes,
    ) as raster_span:
        (
            color_output,
            clear_coat_output,
            coverage,
            raster_stats,
        ) = _rasterize_color_atlas(
            layout,
            faces,
            atlas_uv,
            width,
            height,
            source_uvs,
            blend,
            diffuse_a,
            diffuse_b,
            multiplier,
            additive,
            specular,
            (
                np.asarray(part.specular_strengths, dtype=np.float32)[mapping]
                if part.specular_strengths is not None
                else None
            ),
        )
        covered_pixels = int(np.count_nonzero(coverage))
        raster_span.update(
            **raster_stats,
            coveredPixels=covered_pixels,
            coverageRatio=round(
                float(covered_pixels) / max(1, width * height),
                8,
            ),
        )

    if not np.any(coverage):
        raise ConversionError(f"Texture atlas for {part.name} contains no rasterized triangles")
    # Fill leftover empty texels so sparse charts / mips do not sample pure black.
    with profile_span(
        "dilate_color_atlas",
        category="material",
        width=width,
        height=height,
        channels=4,
        coveredPixels=covered_pixels,
    ):
        color_output = _dilate(
            color_output,
            coverage,
            16,
            wrap=False,
            fill_remaining=True,
        )
    color_path = output_dir / f"{stem}_base.png"
    with profile_span(
        "encode_color_atlas",
        category="material",
        width=width,
        height=height,
    ) as encode_span:
        color_pixels = _quantize_unorm8(color_output)
        Image.fromarray(color_pixels, mode="RGBA").save(
            current_filesystem().write_path(color_path),
            compress_level=1,
        )
        encode_span.update(
            outputBytes=current_filesystem().stat(color_path).st_size,
        )
    del color_output, color_pixels
    clear_coat_path = None
    if clear_coat_output is not None:
        with profile_span(
            "dilate_specular_atlas",
            category="material",
            width=width,
            height=height,
            channels=1,
        ):
            clear_coat_output = _dilate(
                clear_coat_output,
                coverage,
                16,
                wrap=False,
                fill_remaining=True,
            )
        clear_coat_path = output_dir / f"{stem}_specular.png"
        with profile_span(
            "encode_specular_atlas",
            category="material",
            width=width,
            height=height,
        ) as encode_span:
            clear_coat_pixels = _quantize_unorm8(clear_coat_output)
            Image.fromarray(
                clear_coat_pixels[:, :, 0],
                mode="L",
            ).save(
                current_filesystem().write_path(clear_coat_path),
                compress_level=1,
            )
            encode_span.update(
                outputBytes=current_filesystem().stat(clear_coat_path).st_size
            )
        del clear_coat_output, clear_coat_pixels
    if layout.normal_uv is not None and material.normal_texture is not None:
        normal_uv = source_uvs[layout.normal_uv]
        texcoord_sets = (normal_uv, atlas_uv)
        base_color_uv = 1
        normal_texture = material.normal_texture
    else:
        texcoord_sets = (atlas_uv,)
        base_color_uv = 0
        normal_texture = None
    baked_colors = np.ones((len(mapping), 4), dtype=np.float32)
    if preserve_vertex_rgb:
        baked_colors[:, :3] = np.asarray(part.colors, dtype=np.float32)[mapping, :3]
    baked_part = remap_mesh_part(
        part,
        mapping,
        faces,
        texcoords=texcoord_sets[0],
        colors=baked_colors,
        texcoord_sets=texcoord_sets,
        semantic_uv_indices={},
        specular_strengths=None,
        blend_weights=None,
        force_color_stream=preserve_vertex_rgb,
    )
    return VertexLerpBake(
        part=baked_part,
        material=PbrMaterialOverride(
            base_color_texture=color_path,
            base_color_uv=base_color_uv,
            normal_texture=normal_texture,
            normal_uv=0,
            clear_coat_texture=clear_coat_path,
            clear_coat_uv=base_color_uv,
            clear_coat_roughness=clear_coat_roughness,
            base_vertex_color=preserve_vertex_rgb,
        ),
        method="atlas",
    )


def _rasterize_blend_mask(
    faces: np.ndarray,
    mask_uv: np.ndarray,
    blend_weights: np.ndarray,
    width: int,
    height: int,
    target: Path,
) -> None:
    accumulated = np.zeros((height, width), dtype=np.float64)
    counts = np.zeros((height, width), dtype=np.float64)
    for face in faces:
        for (
            triangle,
            in_bounds,
            target_x,
            target_y,
            barycentric,
        ) in _triangle_raster_samples(face, mask_uv, width, height):
            if barycentric is None:
                if in_bounds:
                    for vertex_index in range(3):
                        x, y = np.clip(
                            np.floor(triangle[vertex_index]).astype(int),
                            0,
                            (width - 1, height - 1),
                        )
                        accumulated[y, x] += float(
                            blend_weights[face[vertex_index]]
                        )
                        counts[y, x] += 1.0
                continue
            values = barycentric @ blend_weights[face]
            np.add.at(accumulated, (target_y, target_x), values)
            np.add.at(counts, (target_y, target_x), 1.0)
    coverage = counts > 0
    if not np.any(coverage):
        raise ConversionError(f"Blend mask for {target.name} has no coverage")
    mask = np.zeros((height, width, 1), dtype=np.float32)
    mask[coverage, 0] = (accumulated[coverage] / counts[coverage]).astype(np.float32)
    mask = _dilate(mask, coverage, 8, wrap=False, fill_remaining=True)
    Image.fromarray(
        np.clip(np.rint(mask[:, :, 0] * 255.0), 0, 255).astype(np.uint8),
        mode="L",
    ).save(current_filesystem().write_path(target), compress_level=1)


def _try_vertex_pbr_lerp(
    part: MeshPart,
    material: RbrMaterial,
    layout: _EffectLayout,
) -> VertexLerpBake | None:
    """Use stock PBR layers with the source blend in vertex alpha."""
    if (
        layout.multiplier_uv is not None
        or layout.additive_uv is not None
        or material.uses_alpha
    ):
        return None
    source_indices = (
        0,
        layout.second_uv,
        layout.normal_uv,
        layout.specular_uv,
    )
    texcoord_sets: list[np.ndarray] = []
    output_indices: dict[int, int] = {}
    for source_index in source_indices:
        if source_index is None:
            continue
        source_uv = np.asarray(part.texcoord_sets[source_index], dtype=np.float32)
        output_index = next(
            (
                index
                for index, output_uv in enumerate(texcoord_sets)
                if np.array_equal(source_uv, output_uv)
            ),
            None,
        )
        if output_index is None:
            if len(texcoord_sets) == 2:
                return None
            output_index = len(texcoord_sets)
            texcoord_sets.append(source_uv)
        output_indices[source_index] = output_index
    colors = np.ones_like(np.asarray(part.colors, dtype=np.float32))
    colors[:, 3] = np.clip(
        np.asarray(part.blend_weights, dtype=np.float32),
        0.0,
        1.0,
    )
    semantic_uv_indices = {
        semantic: output_indices[source_index]
        for semantic, source_index in part.semantic_uv_indices.items()
        if source_index in output_indices
    }
    return VertexLerpBake(
        part=replace(
            part,
            texcoords=texcoord_sets[0],
            colors=colors,
            texcoord_sets=tuple(texcoord_sets),
            semantic_uv_indices=semantic_uv_indices,
            blend_weights=None,
            force_color_stream=True,
        ),
        material=PbrMaterialOverride(
            base_color_texture=material.diffuse_texture,
            base_color_uv=output_indices[0],
            base_vertex_color=True,
            normal_texture=material.normal_texture,
            normal_uv=(
                output_indices[layout.normal_uv]
                if layout.normal_uv is not None
                else 0
            ),
            clear_coat_texture=material.specular_texture,
            clear_coat_uv=(
                output_indices[layout.specular_uv]
                if layout.specular_uv is not None
                else 0
            ),
            layer_color_texture=material.second_diffuse_texture,
            layer_color_uv=output_indices[layout.second_uv],
            layer_vertex_color=True,
        ),
        method="vertex-pbr",
    )


def _try_hybrid_vertex_lerp(
    part: MeshPart,
    material: RbrMaterial,
    layout: _EffectLayout,
    profile: AtlasBakeProfile,
    output_dir: Path,
    stem: str,
) -> VertexLerpBake | None:
    """Keep tiling albedos/normals; bake only a unique blend-weight mask.

    Full albedo atlases cannot match RBR sharpness on long tiled roads. BeamNG
    has two UV sets, so this path requires both diffuse layers to share UVs.
    """
    if layout.multiplier_uv is not None or layout.additive_uv is not None:
        return None
    uv_a = np.asarray(part.texcoord_sets[0], dtype=np.float32)
    uv_b = np.asarray(part.texcoord_sets[layout.second_uv], dtype=np.float32)
    if not np.allclose(uv_a, uv_b, rtol=1e-4, atol=1e-4):
        return None
    with profile_span(
        "unwrap_blend_mask",
        category="material",
        part=part.name,
        faces=len(part.faces),
        blendMaskProfile=profile.name,
        targetResolution=profile.target_resolution,
        texelsPerUnit=profile.texels_per_unit,
    ) as unwrap_span:
        mapping, faces, mask_uv, width, height = _atlas(part, profile)
        unwrap_span.update(
            width=width,
            height=height,
            outputVertices=len(mapping),
            outputFaces=len(faces),
        )
    mask_path = output_dir / f"{stem}_blend.png"
    with profile_span(
        "rasterize_blend_mask",
        category="material",
        part=part.name,
        faces=len(faces),
        width=width,
        height=height,
        outputPixels=width * height,
    ) as raster_span:
        _rasterize_blend_mask(
            faces,
            mask_uv,
            np.asarray(part.blend_weights, dtype=np.float32)[mapping],
            width,
            height,
            mask_path,
        )
        raster_span.update(
            outputBytes=current_filesystem().stat(mask_path).st_size
        )
    tiling_uv = uv_a[mapping]
    # UV0 = tiling albedos/normals (default sampler; no UseUV needed). UV1 = unique
    # blend mask. In-game A/B showed diffuseMapUseUV=1 still sampling the unique
    # set, so tiling must live on UV0 for RBR-like sharpness.
    return VertexLerpBake(
        part=remap_mesh_part(
            part,
            mapping,
            faces,
            texcoords=tiling_uv,
            colors=np.ones((len(mapping), 4), dtype=np.float32),
            texcoord_sets=(tiling_uv, mask_uv),
            blend_weights=None,
        ),
        material=PbrMaterialOverride(
            base_color_texture=material.diffuse_texture,
            base_color_uv=0,
            layer_color_texture=material.second_diffuse_texture,
            layer_color_uv=0,
            layer_opacity_texture=mask_path,
            layer_opacity_uv=1,
            normal_texture=(
                material.normal_texture if layout.normal_uv is not None else None
            ),
            normal_uv=0,
            clear_coat_texture=(
                material.specular_texture if layout.specular_uv is not None else None
            ),
            clear_coat_uv=0,
        ),
        method="hybrid",
    )


def bake_vertex_lerp(
    part: MeshPart,
    material: RbrMaterial,
    output_dir: Path,
    source: Path,
) -> VertexLerpBake:
    effect = material.effect.casefold()
    layout = _EFFECT_LAYOUTS.get(effect)
    if layout is None:
        raise ConversionError(f"Unsupported vertex-lerp effect {material.effect!r}")
    if part.blend_weights is None:
        raise ConversionError(f"Vertex-lerp mesh {source.name}:{part.name} has no blend weights")
    if len(part.texcoord_sets) <= layout.second_uv:
        raise ConversionError(
            f"Vertex-lerp mesh {source.name}:{part.name} has "
            f"{len(part.texcoord_sets)} UV sets; {material.effect} needs {layout.second_uv + 1}"
        )
    if material.diffuse_texture is None or material.second_diffuse_texture is None:
        raise ConversionError(f"Vertex-lerp material {material.name!r} lacks two diffuse textures")

    if effect == "btb_diffusevertlerp" and not material.uses_alpha:
        colors = np.ones_like(
            np.asarray(part.colors, dtype=np.float32)
        )
        colors[:, 3] = np.clip(
            np.asarray(part.blend_weights, dtype=np.float32),
            0.0,
            1.0,
        )
        return VertexLerpBake(
            part=replace(
                part,
                texcoords=np.asarray(part.texcoord_sets[0]),
                colors=colors,
                texcoord_sets=tuple(part.texcoord_sets[:2]),
                blend_weights=None,
                force_color_stream=True,
            ),
            material=PbrMaterialOverride.vertex_color_lerp(
                material.diffuse_texture,
                material.second_diffuse_texture,
            ),
            method="vertex",
        )

    vertex_pbr = _try_vertex_pbr_lerp(part, material, layout)
    if vertex_pbr is not None:
        return vertex_pbr

    current_filesystem().mkdir(output_dir, parents=True, exist_ok=True)
    digest = hashlib.sha1(
        f"{source}|{part.name}|{part.material_name}|{material.name}".casefold().encode(
            "utf-8"
        )
    ).hexdigest()[:10]
    stem = f"{slugify(source.stem)}_{slugify(part.name)}_{digest}"
    with profile_span(
        "try_hybrid_blend",
        category="material",
        part=part.name,
        faces=len(part.faces),
    ) as hybrid_span:
        try:
            hybrid = (
                None
                if (
                    material.uses_alpha
                    or (blend_mask_profile := BLEND_MASK_BAKE_PROFILES.get(effect))
                    is None
                )
                else _try_hybrid_vertex_lerp(
                    part,
                    material,
                    layout,
                    blend_mask_profile,
                    output_dir,
                    stem,
                )
            )
        except ConversionError:
            hybrid = None
        hybrid_span.update(succeeded=hybrid is not None)
    if hybrid is not None:
        return hybrid
    return _bake_atlas(
        part,
        material,
        layout,
        output_dir,
        stem,
    )


def bake_original_material(
    part: MeshPart,
    material: RbrMaterial,
    output_dir: Path,
    source: Path,
) -> VertexLerpBake:
    if material.effect.casefold() != "rbr_original":
        raise ConversionError(f"Unsupported Original RBR material {material.effect!r}")
    if material.diffuse_texture is None:
        raise ConversionError(f"Original material {material.name!r} has no diffuse texture")

    diffuse_uv = part.semantic_uv_indices.get("diffuse_1")
    second_uv = part.semantic_uv_indices.get("diffuse_2")
    specular_uv = part.semantic_uv_indices.get("specular")
    if diffuse_uv is None:
        raise ConversionError(f"Original mesh {source.name}:{part.name} has no diffuse UV")
    if material.second_diffuse_texture is not None and second_uv is None:
        raise ConversionError(
            f"Original mesh {source.name}:{part.name} has no second diffuse UV"
        )
    if material.specular_texture is not None and specular_uv is None:
        raise ConversionError(f"Original mesh {source.name}:{part.name} has no specular UV")

    source_uvs = tuple(part.texcoord_sets)
    ordered_indices = [diffuse_uv]
    for index in (second_uv, specular_uv):
        if index is not None and index not in ordered_indices:
            ordered_indices.append(index)
    ordered_uvs = tuple(source_uvs[index] for index in ordered_indices)
    index_remap = {
        source_index: output_index
        for output_index, source_index in enumerate(ordered_indices)
    }
    colors = np.asarray(part.colors, dtype=np.float32)
    bake_part = replace(
        part,
        texcoords=np.asarray(ordered_uvs[0]),
        texcoord_sets=ordered_uvs,
        blend_weights=(
            np.asarray(colors[:, 3], dtype=np.float32)
            if material.second_diffuse_texture is not None
            else None
        ),
    )
    layout = _EffectLayout(
        second_uv=(
            index_remap[second_uv]
            if material.second_diffuse_texture is not None
            else None
        ),
        specular_uv=(
            index_remap[specular_uv]
            if material.specular_texture is not None
            else None
        ),
        rbr_double_texture=material.second_diffuse_texture is not None,
    )
    current_filesystem().mkdir(output_dir, parents=True, exist_ok=True)
    digest = hashlib.sha1(
        f"{source}|{part.name}|{part.material_name}|{material.name}".casefold().encode(
            "utf-8"
        )
    ).hexdigest()[:10]
    stem = f"{slugify(source.stem)}_{slugify(part.name)}_{digest}"
    return _bake_atlas(
        bake_part,
        material,
        layout,
        output_dir,
        stem,
        preserve_vertex_rgb=True,
        clear_coat_roughness=(
            _ORIGINAL_RBR_LEGACY_SPECULAR_CLEAR_COAT_ROUGHNESS
            if material.specular_texture is not None
            else None
        ),
    )
