from __future__ import annotations

import struct
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .core import ConversionError
from .filesystem import current_filesystem


_TOKEN_NAMES = {
    0x0A: "{",
    0x0B: "}",
    0x0C: "(",
    0x0D: ")",
    0x0E: "[",
    0x0F: "]",
    0x10: "<",
    0x11: ">",
    0x12: ".",
    0x13: ",",
    0x14: ";",
    0x1F: "template",
    0x28: "WORD",
    0x29: "DWORD",
    0x2A: "FLOAT",
    0x2B: "DOUBLE",
    0x2C: "CHAR",
    0x2D: "UCHAR",
    0x2E: "SWORD",
    0x2F: "SDWORD",
    0x30: "void",
    0x31: "string",
    0x32: "unicode",
    0x33: "cstring",
    0x34: "array",
}


@dataclass(frozen=True)
class XCustomChannels:
    faces: np.ndarray
    texcoords: tuple[np.ndarray, ...]
    blend_weights: np.ndarray | None
    tangents: np.ndarray | None
    binormals: np.ndarray | None


class _TokenStream:
    def __init__(self, tokens: list[str | int | float]):
        self.tokens = tokens
        self.index = 0

    def read(self) -> str | int | float:
        if self.index >= len(self.tokens):
            raise ConversionError("Unexpected end of binary X file")
        value = self.tokens[self.index]
        self.index += 1
        return value

    def peek(self) -> str | int | float | None:
        return self.tokens[self.index] if self.index < len(self.tokens) else None

    def number(self) -> float:
        value = self.read()
        if not isinstance(value, (int, float)):
            raise ConversionError(f"Expected a number in binary X file, found {value!r}")
        return float(value)

    def integer(self) -> int:
        value = self.read()
        if not isinstance(value, int):
            raise ConversionError(f"Expected an integer in binary X file, found {value!r}")
        return value

    def object_head(self) -> None:
        if self.peek() != "{":
            self.read()
        if self.read() != "{":
            raise ConversionError("Expected an opening brace in binary X file")

    def closing_brace(self) -> None:
        if self.read() != "}":
            raise ConversionError("Expected a closing brace in binary X file")

    def skip_object(self) -> None:
        while self.peek() is not None and self.read() != "{":
            pass
        depth = 1
        while depth and self.peek() is not None:
            token = self.read()
            if token == "{":
                depth += 1
            elif token == "}":
                depth -= 1
        if depth:
            raise ConversionError("Unterminated object in binary X file")


def _binary_float_size(data: bytes) -> int:
    if len(data) < 16 or not data.startswith(b"xof "):
        raise ConversionError("Not a DirectX X file")
    if data[8:12] != b"bin ":
        raise ConversionError("BTB custom channels require an uncompressed binary X file")
    float_bits = int(data[12:16].decode("ascii", errors="strict"))
    if float_bits not in {32, 64}:
        raise ConversionError(f"Unsupported binary X float size: {float_bits}")
    return float_bits // 8


def _binary_token_end(data: bytes, offset: int, float_size: int) -> tuple[int, int]:
    if offset + 2 > len(data):
        raise ConversionError("Truncated binary X token")
    token = struct.unpack_from("<H", data, offset)[0]
    if token in {1, 2}:
        if offset + 6 > len(data):
            raise ConversionError("Truncated binary X token")
        length = struct.unpack_from("<I", data, offset + 2)[0]
        end = offset + 6 + length + (2 if token == 2 else 0)
    elif token == 3:
        end = offset + 6
    elif token == 5:
        end = offset + 18
    elif token in {6, 7}:
        if offset + 6 > len(data):
            raise ConversionError("Truncated binary X token")
        count = struct.unpack_from("<I", data, offset + 2)[0]
        end = offset + 6 + count * (4 if token == 6 else float_size)
    elif token in _TOKEN_NAMES:
        end = offset + 2
    else:
        raise ConversionError(f"Unsupported binary X token 0x{token:04x}")
    if end > len(data):
        raise ConversionError("Truncated binary X token")
    return token, end


def normalize_legacy_x_encoding(data: bytes) -> bytes | None:
    if len(data) < 16 or not data.startswith(b"xof "):
        return None
    if data[8:12] == b"txt ":
        try:
            data.decode("utf-8", errors="strict")
        except UnicodeDecodeError:
            return data.decode("cp1252", errors="replace").encode("utf-8")
        return None
    if data[8:12] != b"bin ":
        return None
    try:
        float_size = _binary_float_size(data)
        output = bytearray(data[:16])
        offset = 16
        changed = False
        while offset + 2 <= len(data):
            token, end = _binary_token_end(data, offset, float_size)
            if token in {1, 2}:
                length = struct.unpack_from("<I", data, offset + 2)[0]
                value_start = offset + 6
                value_end = value_start + length
                value = data[value_start:value_end]
                try:
                    value.decode("utf-8", errors="strict")
                except UnicodeDecodeError:
                    normalized = value.decode("cp1252", errors="replace").encode("utf-8")
                    output.extend(data[offset : offset + 2])
                    output.extend(struct.pack("<I", len(normalized)))
                    output.extend(normalized)
                    output.extend(data[value_end:end])
                    changed = True
                else:
                    output.extend(data[offset:end])
            else:
                output.extend(data[offset:end])
            offset = end
        output.extend(data[offset:])
    except (ConversionError, UnicodeDecodeError, ValueError):
        return None
    return bytes(output) if changed else None


def _binary_tokens(data: bytes) -> list[str | int | float]:
    float_size = _binary_float_size(data)
    float_format = "<f" if float_size == 4 else "<d"
    offset = 16
    tokens: list[str | int | float] = []

    while offset + 2 <= len(data):
        token, end = _binary_token_end(data, offset, float_size)
        if token in {1, 2}:
            length = struct.unpack_from("<I", data, offset + 2)[0]
            value_start = offset + 6
            tokens.append(
                data[value_start : value_start + length].decode(
                    "cp1252",
                    errors="replace",
                )
            )
        elif token == 3:
            tokens.append(struct.unpack_from("<I", data, offset + 2)[0])
        elif token == 5:
            pass
        elif token == 6:
            count = struct.unpack_from("<I", data, offset + 2)[0]
            tokens.extend(struct.unpack_from(f"<{count}I", data, offset + 6))
        elif token == 7:
            count = struct.unpack_from("<I", data, offset + 2)[0]
            for _index in range(count):
                value_offset = offset + 6 + _index * float_size
                tokens.append(struct.unpack_from(float_format, data, value_offset)[0])
        else:
            tokens.append(_TOKEN_NAMES[token])
        offset = end
    return tokens


def _parse_texture_coords(stream: _TokenStream, vertex_count: int) -> np.ndarray:
    stream.object_head()
    count = stream.integer()
    if count != vertex_count:
        raise ConversionError(
            f"Binary X texture coordinate count {count} does not match {vertex_count} vertices"
        )
    values = np.asarray(
        [stream.number() for _index in range(count * 2)],
        dtype=np.float32,
    ).reshape((-1, 2))
    stream.closing_brace()
    return values


def _decode_decl_data(
    stream: _TokenStream,
    vertex_count: int,
) -> tuple[dict[int, np.ndarray], np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    stream.object_head()
    element_count = stream.integer()
    elements = [
        (
            stream.integer(),
            stream.integer(),
            stream.integer(),
            stream.integer(),
        )
        for _index in range(element_count)
    ]
    dword_count = stream.integer()
    raw = np.asarray(
        [stream.integer() for _index in range(dword_count)],
        dtype=np.uint32,
    )
    stream.closing_brace()

    widths = {0: 1, 1: 2, 2: 3, 3: 4}
    try:
        stride = sum(widths[element[0]] for element in elements)
    except KeyError as exc:
        raise ConversionError(f"Unsupported BTB DeclData type {exc.args[0]}") from exc
    if dword_count != vertex_count * stride:
        raise ConversionError(
            f"BTB DeclData has {dword_count} DWORDs; expected {vertex_count * stride}"
        )
    values = raw.view(np.float32).reshape((vertex_count, stride))
    texcoords: dict[int, np.ndarray] = {}
    blend_weights = None
    tangents = None
    binormals = None
    offset = 0
    for data_type, method, usage, usage_index in elements:
        if method != 0:
            raise ConversionError(f"Unsupported BTB DeclData method {method}")
        width = widths[data_type]
        channel = values[:, offset : offset + width].copy()
        offset += width
        if usage == 1 and usage_index == 0 and width == 1:
            blend_weights = channel[:, 0]
        elif usage == 5 and width == 2:
            texcoords[usage_index] = channel
        elif usage == 6 and usage_index == 0 and width == 3:
            tangents = channel
        elif usage == 7 and usage_index == 0 and width == 3:
            binormals = channel
        elif usage in {0, 3}:
            continue
        else:
            raise ConversionError(
                f"Unsupported BTB DeclData semantic usage={usage}, index={usage_index}, width={width}"
            )
    return texcoords, blend_weights, tangents, binormals


def _decode_fvf_data(
    stream: _TokenStream,
    vertex_count: int,
) -> dict[int, np.ndarray]:
    stream.object_head()
    flags = stream.integer()
    dword_count = stream.integer()
    raw = np.asarray(
        [stream.integer() for _index in range(dword_count)],
        dtype=np.uint32,
    )
    stream.closing_brace()
    if vertex_count == 0 or dword_count % vertex_count:
        raise ConversionError("BTB FVFData does not align with the source vertex count")
    stride = dword_count // vertex_count
    texture_count = (flags >> 8) & 0xF
    if texture_count < 1 or stride != texture_count * 2:
        raise ConversionError(
            f"Unsupported BTB FVFData flags 0x{flags:x} and stride {stride}"
        )
    values = raw.view(np.float32).reshape((vertex_count, stride))
    return {
        index + 1: values[:, index * 2 : index * 2 + 2].copy()
        for index in range(stride // 2)
    }


def _parse_mesh(stream: _TokenStream) -> XCustomChannels | None:
    stream.object_head()
    vertex_count = stream.integer()
    for _index in range(vertex_count * 3):
        stream.number()
    face_count = stream.integer()
    faces: list[tuple[int, int, int]] = []
    for _index in range(face_count):
        count = stream.integer()
        indices = tuple(stream.integer() for _corner in range(count))
        if count != 3:
            raise ConversionError("BTB custom-channel meshes must contain only triangles")
        faces.append(indices)

    texcoords: dict[int, np.ndarray] = {}
    blend_weights = None
    tangents = None
    binormals = None
    found_custom_channels = False
    while True:
        object_name = stream.read()
        if object_name == "}":
            break
        if object_name == "MeshTextureCoords":
            index = 0
            while index in texcoords:
                index += 1
            texcoords[index] = _parse_texture_coords(stream, vertex_count)
        elif object_name == "DeclData":
            found_custom_channels = True
            extra_uvs, blend_weights, tangents, binormals = _decode_decl_data(
                stream,
                vertex_count,
            )
            texcoords.update(extra_uvs)
        elif object_name == "FVFData":
            found_custom_channels = True
            texcoords.update(_decode_fvf_data(stream, vertex_count))
        else:
            stream.skip_object()

    if not found_custom_channels:
        return None
    if not texcoords:
        texcoords[0] = np.zeros((vertex_count, 2), dtype=np.float32)
    max_index = max(texcoords)
    ordered_texcoords = tuple(
        texcoords.get(index, np.zeros((vertex_count, 2), dtype=np.float32))
        for index in range(max_index + 1)
    )
    return XCustomChannels(
        faces=np.asarray(faces, dtype=np.uint32).reshape((-1, 3)),
        texcoords=ordered_texcoords,
        blend_weights=blend_weights,
        tangents=tangents,
        binormals=binormals,
    )


def read_custom_channels_data(data: bytes) -> XCustomChannels | None:
    if b"DeclData" not in data and b"FVFData" not in data:
        return None
    if len(data) < 12 or data[8:12] != b"bin ":
        return None
    tokens = _binary_tokens(data)
    stream = _TokenStream(tokens)
    previous: str | int | float | None = None
    mesh_count = 0
    custom_channels = None
    while stream.peek() is not None:
        token = stream.read()
        if token == "Mesh" and previous != "template":
            mesh_count += 1
            parsed = _parse_mesh(stream)
            if parsed is not None:
                if custom_channels is not None:
                    return None
                custom_channels = parsed
        previous = token
    return custom_channels if mesh_count == 1 else None


def read_custom_channels(path: Path) -> XCustomChannels | None:
    return read_custom_channels_data(current_filesystem().read_bytes(path))
