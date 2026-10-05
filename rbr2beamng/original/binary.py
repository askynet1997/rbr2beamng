from __future__ import annotations

import math
import mmap
import os
import struct
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..filesystem import current_filesystem
from .models import Bounds, OriginalSegment, Vec2, Vec3


DEFAULT_COUNT_LIMIT = 10_000_000
DEFAULT_STRING_LIMIT = 1_048_576


class OriginalParseError(ValueError):
    def __init__(self, message: str, *, offset: int | None = None, context: str = ""):
        location = f" at 0x{offset:x}" if offset is not None else ""
        detail = f" ({context})" if context else ""
        super().__init__(f"{message}{location}{detail}")
        self.offset = offset
        self.context = context


class BoundsError(OriginalParseError):
    pass


class FormatError(OriginalParseError):
    pass


@dataclass(frozen=True)
class BinarySource:
    view: memoryview
    owner: object
    name: str


Source = str | os.PathLike[str] | bytes | bytearray | memoryview


def load_source(source: Source) -> BinarySource:
    if isinstance(source, (str, os.PathLike)):
        path = current_filesystem().read_path(source)
        try:
            file_size = current_filesystem().stat(path).st_size
            if file_size == 0:
                raw = b""
                return BinarySource(memoryview(raw), raw, str(path))
            with current_filesystem().open(path, "rb") as stream:
                mapped = mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ)
            return BinarySource(memoryview(mapped), mapped, str(path))
        except OSError as exc:
            raise OriginalParseError(f"Unable to open {path}: {exc}") from exc
    if isinstance(source, memoryview):
        return BinarySource(source.cast("B"), source, "<memory>")
    if isinstance(source, (bytes, bytearray)):
        return BinarySource(memoryview(source), source, "<bytes>")
    raise TypeError(f"Unsupported binary source: {type(source)!r}")


def checked_count(
    count: int,
    *,
    limit: int = DEFAULT_COUNT_LIMIT,
    offset: int | None = None,
    context: str = "count",
) -> int:
    if count < 0 or count > limit:
        raise FormatError(
            f"Invalid {context} {count}; maximum is {limit}",
            offset=offset,
            context=context,
        )
    return count


def require_finite(
    values: Iterable[float],
    *,
    offset: int,
    context: str,
) -> None:
    if not all(math.isfinite(float(value)) for value in values):
        raise FormatError(
            "Non-finite floating-point value",
            offset=offset,
            context=context,
        )


class BinaryReader:
    def __init__(
        self,
        data: bytes | bytearray | memoryview,
        *,
        start: int = 0,
        end: int | None = None,
        context: str = "",
    ):
        self._data = data if isinstance(data, memoryview) else memoryview(data)
        self._data = self._data.cast("B")
        actual_end = len(self._data) if end is None else end
        if start < 0 or actual_end < start or actual_end > len(self._data):
            raise BoundsError("Invalid reader range", offset=max(start, 0), context=context)
        self.start = start
        self.end = actual_end
        self.offset = start
        self.context = context

    @property
    def remaining(self) -> int:
        return self.end - self.offset

    def _require(self, size: int, *, offset: int | None = None) -> int:
        position = self.offset if offset is None else offset
        if size < 0 or position < self.start or position > self.end - size:
            raise BoundsError(
                f"Read of {size} bytes exceeds range ending at 0x{self.end:x}",
                offset=position,
                context=self.context,
            )
        return position

    def view(self, size: int) -> memoryview:
        position = self._require(size)
        self.offset += size
        return self._data[position : position + size]

    def read(self, size: int) -> bytes:
        return bytes(self.view(size))

    def view_at(self, offset: int, size: int) -> memoryview:
        self._require(size, offset=offset)
        return self._data[offset : offset + size]

    def unpack(self, fmt: str) -> tuple[Any, ...]:
        if not fmt:
            raise ValueError("Empty struct format")
        if fmt[0] not in "@=<>!":
            fmt = "<" + fmt
        elif fmt[0] != "<":
            raise ValueError("Original RBR fields must be little-endian")
        compiled = struct.Struct(fmt)
        position = self._require(compiled.size)
        values = compiled.unpack_from(self._data, position)
        self.offset += compiled.size
        return values

    def u8(self) -> int:
        return self.unpack("<B")[0]

    def u32(self) -> int:
        return self.unpack("<I")[0]

    def f32(self) -> float:
        return self.unpack("<f")[0]

    def vec2(self) -> Vec2:
        return self.unpack("<2f")  # type: ignore[return-value]

    def vec3(self) -> Vec3:
        return self.unpack("<3f")  # type: ignore[return-value]

    def bounds(self) -> Bounds:
        return Bounds(self.vec3(), self.vec3())

    def cstring(self, *, max_bytes: int = DEFAULT_STRING_LIMIT) -> str:
        max_end = min(self.end, self.offset + max_bytes + 1)
        position = self.offset
        while position < max_end and self._data[position] != 0:
            position += 1
        if position >= max_end or position >= self.end:
            raise FormatError(
                f"Unterminated or overlong Latin-1 string (limit {max_bytes})",
                offset=self.offset,
                context=self.context,
            )
        raw = self._data[self.offset : position]
        self.offset = position + 1
        return bytes(raw).decode("latin-1")

    def fixed_string(self, size: int) -> str:
        return self.read(size).split(b"\0", 1)[0].decode("latin-1")

    def array(self, dtype: Any, count: int, *, limit: int = DEFAULT_COUNT_LIMIT) -> Any:
        checked_count(count, limit=limit, offset=self.offset, context="array count")
        parsed_dtype = np.dtype(dtype)
        size = parsed_dtype.itemsize * count
        position = self._require(size)
        result = np.frombuffer(self._data, dtype=parsed_dtype, count=count, offset=position)
        self.offset += size
        return result

    def remaining_view(self) -> memoryview:
        return self.view(self.remaining)

    def require_end(self) -> None:
        if self.remaining:
            raise FormatError(
                f"{self.remaining} unexpected trailing bytes",
                offset=self.offset,
                context=self.context,
            )


def optional_index(value: int) -> int | None:
    return None if value == 0xFFFFFFFF else value


def scan_segments(
    data: bytes | bytearray | memoryview,
    *,
    context: str,
    count_limit: int = 100_000,
) -> tuple[OriginalSegment, ...]:
    reader = BinaryReader(data, context=context)
    segments: list[OriginalSegment] = []
    while reader.remaining:
        if reader.remaining < 16:
            raise BoundsError(
                "Truncated segment header",
                offset=reader.offset,
                context=context,
            )
        segment_offset = reader.offset
        header_size, category, kind, payload_size = reader.unpack("<4I")
        if header_size != 8:
            raise FormatError(
                f"Unexpected segment header size {header_size}; expected 8",
                offset=segment_offset,
                context=context,
            )
        checked_count(
            len(segments) + 1,
            limit=count_limit,
            offset=segment_offset,
            context="segment count",
        )
        payload = reader.view(payload_size)
        segments.append(
            OriginalSegment(
                offset=segment_offset,
                header_size=header_size,
                category=category,
                kind=kind,
                payload=payload,
            )
        )
    return tuple(segments)
