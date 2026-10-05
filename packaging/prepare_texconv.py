from __future__ import annotations

import hashlib
import shutil
import tarfile
import urllib.request
import zipfile
from pathlib import Path


VERSION = "0.6.0"
ROOT = Path(__file__).resolve().parents[1]
BUILD = ROOT / "build" / "texconv"
PACKAGE_BIN = ROOT / "rbr2beamng" / "bin"
BASE_URL = (
    "https://github.com/matyalatte/Texconv-Custom-DLL/"
    f"releases/download/v{VERSION}"
)
ASSETS = {
    "windows-x64": (
        f"TexconvCustomDLL-v{VERSION}-Windows-x64.zip",
        "935d0b4bfc4cbeee57f4d533cc7eeec500c744b543b9b7b059143362a9f7846f",
        "texconv.exe",
    ),
    "linux-x64": (
        f"TexconvCustomDLL-v{VERSION}-Linux-x64.tar.xz",
        "9746a60e26f0ef811659113688e3924aba2b8c43560b6e2d6a5af2512fb70469",
        "texconv",
    ),
    "macos-universal": (
        f"TexconvCustomDLL-v{VERSION}-macOS-universal.tar.xz",
        "93143f338a9710adbc4646e280d98f44ceb0bb0e8cb34a45d3a113c221946112",
        "texconv",
    ),
}


def _download(name: str, expected_sha256: str) -> Path:
    target = BUILD / "downloads" / name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.is_file():
        temporary = target.with_suffix(target.suffix + ".tmp")
        urllib.request.urlretrieve(f"{BASE_URL}/{name}", temporary)
        temporary.replace(target)
    digest = hashlib.sha256(target.read_bytes()).hexdigest()
    if digest != expected_sha256:
        raise RuntimeError(f"Checksum mismatch for {target}")
    return target


def _extract(archive: Path, destination: Path) -> None:
    if destination.exists():
        shutil.rmtree(destination)
    destination.mkdir(parents=True)
    if archive.suffix == ".zip":
        with zipfile.ZipFile(archive) as source:
            source.extractall(destination)
    else:
        with tarfile.open(archive, "r:xz") as source:
            source.extractall(destination, filter="data")


def main() -> None:
    if PACKAGE_BIN.exists():
        shutil.rmtree(PACKAGE_BIN)
    PACKAGE_BIN.mkdir(parents=True, exist_ok=True)
    for platform_name, (archive_name, digest, executable_name) in ASSETS.items():
        archive = _download(archive_name, digest)
        extracted = BUILD / "extracted" / platform_name
        _extract(archive, extracted)
        executables = list(extracted.rglob(executable_name))
        if len(executables) != 1:
            raise RuntimeError(
                f"Expected one {executable_name} in {archive}, found {len(executables)}"
            )
        destination = PACKAGE_BIN / platform_name / executable_name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(executables[0], destination)
        for filename in ("LICENSE", "THIRD_PARTY_LICENSES.txt"):
            matches = list(extracted.rglob(filename))
            if len(matches) != 1:
                raise RuntimeError(
                    f"Expected one {filename} in {archive}, found {len(matches)}"
                )
            shutil.copy2(matches[0], destination.parent / filename)
    (PACKAGE_BIN / "VERSION.txt").write_text(f"{VERSION}\n", encoding="ascii")
    print(f"Prepared Texconv-Custom-DLL {VERSION}")


if __name__ == "__main__":
    main()
