"""Read access to RSF-obfuscated stage files."""

from __future__ import annotations

import io
import tempfile
from typing import BinaryIO

import numpy as np

from .filesystem import FileSandbox, PathInput


_LONG_MASK = np.frombuffer(b"xEodFel20nbF049WGdgPEZKdirqpX8U3EtiJM", dtype=np.uint8)
_SHORT_MASK = np.frombuffer(b"Rbr", dtype=np.uint8)
_TABLE_SIZE = 21
# Offset of the per-file table for each version; the obfuscated content follows it.
_TABLE_OFFSETS = {0xA001: 4 + 0x171D, 0xA101: 4 + 0xE175}
# Least common multiple of the mask and table lengths.
_PERIOD = 777
_CHUNK = _PERIOD * 16384
_INDEX = np.arange(_PERIOD)
_MASK_STREAM = _LONG_MASK[_INDEX % len(_LONG_MASK)] ^ _SHORT_MASK[_INDEX % len(_SHORT_MASK)]
_ROTATE_THREE = (_LONG_MASK[_INDEX % len(_LONG_MASK)] & 1).astype(bool)


def _table_offset(stream: BinaryIO) -> int | None:
    offset = _TABLE_OFFSETS.get(int.from_bytes(stream.read(4), "little"))
    if offset is None or stream.seek(0, io.SEEK_END) < offset + _TABLE_SIZE:
        return None
    return offset


class _DeobfuscatingReader(io.RawIOBase):
    """Seekable original content of an RSF-obfuscated file, restored as it is read.

    ``fileno`` returns a temporary copy of the original content, for callers
    that map files.
    """

    def __init__(self, stream: BinaryIO, table_offset: int) -> None:
        super().__init__()
        self._file = stream
        self._copy: BinaryIO | None = None
        stream.seek(table_offset)
        table = np.frombuffer(stream.read(_TABLE_SIZE), dtype=np.uint8)
        self._stream = table[_INDEX % _TABLE_SIZE] ^ _MASK_STREAM
        self._start = table_offset + _TABLE_SIZE
        self._size = stream.seek(0, io.SEEK_END) - self._start
        self._position = 0

    def readable(self) -> bool:
        return True

    def seekable(self) -> bool:
        return True

    def tell(self) -> int:
        return self._position

    def seek(self, offset: int, whence: int = io.SEEK_SET) -> int:
        position = offset + (0, self._position, self._size)[whence]
        if position < 0:
            raise ValueError(f"negative seek position {position}")
        self._position = position
        return position

    def readinto(self, buffer) -> int:
        self._file.seek(self._start + self._position)
        count = self._file.readinto(buffer)
        if not count:
            return 0
        content = np.frombuffer(buffer, dtype=np.uint8, count=count)
        for begin in range(0, count, _CHUNK):
            block = content[begin:begin + _CHUNK]
            phase = (self._position + begin) % _PERIOD
            repeats = (phase + len(block)) // _PERIOD + 1
            value = block ^ np.tile(self._stream, repeats)[phase:phase + len(block)]
            block[:] = np.where(
                np.tile(_ROTATE_THREE, repeats)[phase:phase + len(block)],
                (value << 3) | (value >> 5),
                (value << 2) | (value >> 6),
            )
        self._position += count
        return count

    def readall(self) -> bytes:
        buffer = bytearray(max(self._size - self._position, 0))
        del buffer[self.readinto(buffer):]
        return bytes(buffer)

    def fileno(self) -> int:
        if self._copy is None:
            position = self._position
            self._position = 0
            self._copy = tempfile.TemporaryFile()
            while chunk := self.read(_CHUNK):
                self._copy.write(chunk)
            self._copy.flush()
            self._position = position
        return self._copy.fileno()

    def close(self) -> None:
        self._file.close()
        if self._copy is not None:
            self._copy.close()
        super().close()


class RsfDeobfuscatingSandbox(FileSandbox):
    """Reads RSF-obfuscated files as their original content.

    Paths and metadata stay those of the obfuscated files; only ``open``,
    ``read_text`` and ``read_bytes`` see the original content.
    """

    def open(
        self,
        path: PathInput,
        mode: str = "r",
        buffering: int = -1,
        encoding: str | None = None,
        errors: str | None = None,
        newline: str | None = None,
    ):
        arguments = (mode, buffering, encoding, errors, newline)
        if any(flag in mode for flag in ("w", "a", "x", "+")):
            return super().open(path, *arguments)
        stream = self.read_path(path).open("rb")
        table_offset = _table_offset(stream)
        if table_offset is None:
            stream.close()
            return super().open(path, *arguments)
        content = io.BufferedReader(_DeobfuscatingReader(stream, table_offset))
        if "b" in mode:
            return content
        return io.TextIOWrapper(content, encoding, errors, newline)

    def read_text(
        self,
        path: PathInput,
        encoding: str | None = None,
        errors: str | None = None,
    ) -> str:
        with self.open(path, encoding=encoding, errors=errors) as stream:
            return stream.read()

    def read_bytes(self, path: PathInput) -> bytes:
        with self.open(path, "rb") as stream:
            return stream.read()
