from __future__ import annotations

import os
import shutil
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from pathlib import Path


PathInput = str | os.PathLike[str]


class SandboxViolationError(PermissionError):
    pass


@dataclass(frozen=True)
class SandboxRoot:
    path: Path
    writable: bool


class FileSandbox:
    """Application-level path policy; it cannot prevent TOCTOU races."""

    def __init__(self) -> None:
        self._roots: list[SandboxRoot] = []

    @property
    def roots(self) -> tuple[SandboxRoot, ...]:
        return tuple(self._roots)

    def add_read_only(self, path: PathInput) -> Path:
        return self._add(path, writable=False)

    def add_read_write(self, path: PathInput) -> Path:
        return self._add(path, writable=True)

    def _add(self, path: PathInput, *, writable: bool) -> Path:
        root = self._canonical(path)
        for index, existing in enumerate(self._roots):
            if existing.path == root:
                if writable and not existing.writable:
                    self._roots[index] = SandboxRoot(root, writable=True)
                return root
        self._roots.append(SandboxRoot(root, writable))
        return root

    def read_path(self, path: PathInput) -> Path:
        return self._check(path, writable=False)

    def write_path(self, path: PathInput) -> Path:
        return self._check(path, writable=True)

    def _check(self, path: PathInput, *, writable: bool) -> Path:
        candidate = self._canonical(path)
        matches = [
            root
            for root in self._roots
            if self._contains(root.path, candidate)
        ]
        if matches:
            root = max(matches, key=lambda item: len(item.path.parts))
            if not writable or root.writable:
                return candidate
        access = "write" if writable else "read"
        raise SandboxViolationError(
            f"Filesystem sandbox denied {access} access to {candidate}"
        )

    @staticmethod
    def _canonical(path: PathInput) -> Path:
        try:
            return Path(path).expanduser().resolve(strict=False)
        except OSError as exc:
            raise SandboxViolationError(
                f"Filesystem sandbox could not resolve {path}"
            ) from exc

    @staticmethod
    def _contains(root: Path, candidate: Path) -> bool:
        depth = len(candidate.parts) - len(root.parts)
        if depth < 0:
            return False
        if depth == 0:
            return candidate == root
        return candidate.parents[depth - 1] == root

    def open(self, path: PathInput, mode: str = "r", *args, **kwargs):
        target = (
            self.write_path(path)
            if any(flag in mode for flag in ("w", "a", "x", "+"))
            else self.read_path(path)
        )
        return target.open(mode, *args, **kwargs)

    def read_text(self, path: PathInput, *args, **kwargs) -> str:
        return self.read_path(path).read_text(*args, **kwargs)

    def read_bytes(self, path: PathInput) -> bytes:
        return self.read_path(path).read_bytes()

    def write_text(self, path: PathInput, data: str, *args, **kwargs) -> int:
        return self.write_path(path).write_text(data, *args, **kwargs)

    def write_bytes(self, path: PathInput, data: bytes) -> int:
        return self.write_path(path).write_bytes(data)

    def exists(self, path: PathInput) -> bool:
        return self.read_path(path).exists()

    def is_file(self, path: PathInput) -> bool:
        return self.read_path(path).is_file()

    def is_dir(self, path: PathInput) -> bool:
        return self.read_path(path).is_dir()

    def stat(self, path: PathInput):
        return self.read_path(path).stat()

    def iterdir(self, path: PathInput) -> Iterator[Path]:
        return self.read_path(path).iterdir()

    def glob(self, path: PathInput, pattern: str) -> Iterator[Path]:
        return self.read_path(path).glob(pattern)

    def rglob(self, path: PathInput, pattern: str) -> Iterator[Path]:
        return self.read_path(path).rglob(pattern)

    def mkdir(self, path: PathInput, *args, **kwargs) -> None:
        self.write_path(path).mkdir(*args, **kwargs)

    def unlink(self, path: PathInput, *args, **kwargs) -> None:
        self.write_path(path).unlink(*args, **kwargs)

    def replace(self, source: PathInput, destination: PathInput) -> None:
        os.replace(self.write_path(source), self.write_path(destination))

    def copyfile(self, source: PathInput, destination: PathInput) -> None:
        shutil.copyfile(self.read_path(source), self.write_path(destination))

    def copy2(self, source: PathInput, destination: PathInput) -> None:
        shutil.copy2(self.read_path(source), self.write_path(destination))

    def symlink(self, _target: PathInput, _link: PathInput) -> None:
        raise SandboxViolationError(
            "Filesystem sandbox does not permit symbolic-link creation"
        )

    def temporary_directory(self, directory: PathInput, *, prefix: str) -> tempfile.TemporaryDirectory[str]:
        root = self.write_path(directory)
        root.mkdir(parents=True, exist_ok=True)
        return tempfile.TemporaryDirectory(prefix=prefix, dir=root)


_active_filesystem: ContextVar[FileSandbox | None] = ContextVar(
    "rbr2beamng_filesystem",
    default=None,
)


@contextmanager
def use_filesystem(filesystem: FileSandbox) -> Iterator[FileSandbox]:
    token = _active_filesystem.set(filesystem)
    try:
        yield filesystem
    finally:
        _active_filesystem.reset(token)


def current_filesystem() -> FileSandbox:
    filesystem = _active_filesystem.get()
    if filesystem is None:
        raise SandboxViolationError("No active filesystem sandbox")
    return filesystem
