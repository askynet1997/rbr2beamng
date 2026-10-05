from __future__ import annotations

import hashlib
import platform
import subprocess
import sys
from pathlib import Path
from typing import Callable

from PIL import Image

from .core import ConversionError
from .filesystem import current_filesystem
from .profiling import profile_span


TextureProgress = Callable[[int, int, Path], None]
CookedTextures = dict[tuple[tuple[str, ...], bytes], Path]


def cooked_texture_path(source: Path) -> Path:
    return source.with_suffix(".dds")


def _executable_path() -> Path:
    root = current_filesystem().read_path(__file__).parent / "bin"
    machine = platform.machine().casefold()
    if sys.platform == "win32" and machine in {"amd64", "x86_64"}:
        return root / "windows-x64" / "texconv.exe"
    if sys.platform.startswith("linux") and machine in {"amd64", "x86_64"}:
        return root / "linux-x64" / "texconv"
    if sys.platform == "darwin":
        return root / "macos-universal" / "texconv"
    raise ConversionError(
        f"Texture cooking is not available on {sys.platform}/{platform.machine()}"
    )


def _texture_arguments(source: Path, mode: str) -> list[str]:
    filesystem = current_filesystem()
    source = filesystem.read_path(source)
    output_directory = filesystem.write_path(source.parent)
    filesystem.write_path(cooked_texture_path(source))
    return [
        "-y",
        "-o",
        str(output_directory),
        *_texture_options(source, mode),
        "--",
        str(source),
    ]


def _texture_options(source: Path, mode: str) -> list[str]:
    name = source.name.casefold()
    if name.endswith(".color.png"):
        if mode not in {"RGB", "RGBA", "P"}:
            raise ConversionError(
                f"Color texture must be RGB or RGBA: {source} ({mode})"
            )
        options = ["-m", "0", "-srgb", "-f", "BC7_UNORM_SRGB", "-bc", "q"]
        if mode == "RGBA":
            options.insert(0, "-sepalpha")
    elif name.endswith(".normal.png"):
        if mode not in {"RGB", "RGBA"}:
            raise ConversionError(
                f"Normal texture must be RGB or RGBA: {source} ({mode})"
            )
        options = ["-m", "0", "-f", "BC5_UNORM"]
    elif name.endswith(".opacity.data.png"):
        if mode not in {"1", "L"}:
            raise ConversionError(
                f"Opacity texture must be grayscale: {source} ({mode})"
            )
        # Alpha tests compare exact source values; BC4 re-encoding moves
        # texels across the reference.
        options = ["-m", "0", "-f", "R8_UNORM"]
    elif name.endswith(".data.png"):
        if mode in {"1", "L"}:
            options = ["-m", "0", "-f", "BC4_UNORM"]
        elif mode in {"RGB", "RGBA"}:
            options = ["-m", "0", "-f", "BC7_UNORM", "-bc", "q"]
            if mode == "RGBA":
                options.insert(0, "-sepalpha")
        else:
            raise ConversionError(
                f"Data texture must be grayscale, RGB, or RGBA: {source} ({mode})"
            )
    else:
        raise ConversionError(f"Unsupported texture cooker filename: {source}")
    return options


def _run_texconv(arguments: list[str]) -> None:
    filesystem = current_filesystem()
    executable = _executable_path()
    if not filesystem.is_file(executable):
        raise ConversionError(
            f"Texture cooker executable is missing: {executable}"
        )
    try:
        completed = subprocess.run(
            [str(filesystem.read_path(executable)), *arguments],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=False,
            creationflags=(
                subprocess.CREATE_NO_WINDOW
                if sys.platform == "win32"
                else 0
            ),
        )
    except OSError as exc:
        raise ConversionError(
            f"Unable to run texture cooker {executable}: {exc}"
        ) from exc
    if completed.returncode != 0:
        message = (completed.stderr or completed.stdout).strip()
        raise ConversionError(
            f"Texture cooker failed with exit code {completed.returncode}: "
            f"{message or 'no error message'}"
        )


def _validate_dds(path: Path) -> None:
    filesystem = current_filesystem()
    try:
        with filesystem.open(path, "rb") as stream:
            header = stream.read(128)
    except OSError as exc:
        raise ConversionError(f"Texture cooker did not create {path}") from exc
    if len(header) < 128 or header[:4] != b"DDS ":
        raise ConversionError(f"Texture cooker created an invalid DDS file: {path}")
    height = int.from_bytes(header[12:16], "little")
    width = int.from_bytes(header[16:20], "little")
    if width <= 0 or height <= 0:
        raise ConversionError(f"Texture cooker created an invalid DDS size: {path}")


def _file_digest(path: Path) -> bytes:
    filesystem = current_filesystem()
    digest = hashlib.sha256()
    with filesystem.open(path, "rb") as stream:
        while chunk := stream.read(1024 * 1024):
            digest.update(chunk)
    return digest.digest()


def cook_textures(
    texture_dir: Path,
    progress: TextureProgress | None = None,
    cooked: CookedTextures | None = None,
) -> int:
    filesystem = current_filesystem()
    sources = sorted(
        (
            path
            for path in filesystem.rglob(texture_dir, "*.png")
            if path.name.casefold().endswith(
                (".color.png", ".normal.png", ".data.png")
            )
        ),
        key=lambda path: str(path).casefold(),
    )
    if not sources:
        return 0

    groups: dict[tuple[str, ...], list[Path]] = {}
    representatives: CookedTextures = {} if cooked is None else cooked
    duplicate_sources: dict[Path, Path] = {}
    for source in sources:
        with profile_span(
            "prepare_texture",
            category="texture",
            source=str(source),
            sourceBytes=filesystem.stat(source).st_size,
        ) as prepare_span:
            try:
                with Image.open(filesystem.read_path(source)) as image:
                    mode = image.mode
                    source_size = image.size
            except OSError as exc:
                raise ConversionError(
                    f"Unable to inspect texture {source}: {exc}"
                ) from exc
            arguments = _texture_arguments(source, mode)
            argument_prefix = tuple(arguments[:-2])
            signature = (
                tuple(_texture_options(source, mode)),
                _file_digest(source),
            )
            representative = representatives.get(signature)
            if representative is not None:
                duplicate_sources[source] = representative
                prepare_span.update(
                    mode=mode,
                    width=source_size[0],
                    height=source_size[1],
                    reusedFrom=str(representative),
                )
                continue
            representatives[signature] = source
            prepare_span.update(
                mode=mode,
                width=source_size[0],
                height=source_size[1],
            )
        groups.setdefault(argument_prefix, []).append(source)

    for argument_prefix, jobs in groups.items():
        for start in range(0, len(jobs), 32):
            batch = jobs[start : start + 32]
            arguments = [
                *argument_prefix,
                "--",
                *(str(source) for source in batch),
            ]
            try:
                with profile_span(
                    "texconv_batch",
                    category="texture",
                    firstSource=str(batch[0]),
                    count=len(batch),
                    sourceBytes=sum(
                        filesystem.stat(source).st_size
                        for source in batch
                    ),
                ) as batch_span:
                    _run_texconv(arguments)
                    batch_span.update(
                        outputBytes=sum(
                            filesystem.stat(cooked_texture_path(source)).st_size
                            for source in batch
                            if filesystem.is_file(cooked_texture_path(source))
                        ),
                    )
            except ConversionError as exc:
                raise ConversionError(
                    f"Unable to cook texture batch starting with {batch[0]}: {exc}"
                ) from exc
    for source, representative in duplicate_sources.items():
        filesystem.copyfile(
            cooked_texture_path(representative),
            cooked_texture_path(source),
        )
    for processed, source in enumerate(sources, 1):
        _validate_dds(cooked_texture_path(source))
        if progress:
            progress(processed, len(sources), source)

    with profile_span(
        "remove_source_textures",
        category="texture",
        count=len(sources),
    ):
        for source in sources:
            filesystem.unlink(source)
    return len(sources)
