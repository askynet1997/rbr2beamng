from __future__ import annotations

import shutil, subprocess, sys, uuid, zipfile
from pathlib import Path

from rbr2beamng import __version__

ROOT = Path(__file__).resolve().parents[1]
STAGING_ROOT = ROOT / "build" / f"release-{uuid.uuid4().hex}"
RELEASE_FOLDER = STAGING_ROOT / "rbr2beamng"
ARCHIVE = ROOT / "dist" / f"rbr2beamng_{__version__}_windows_x64.zip"


def _run(*arguments: str | Path) -> None:
    subprocess.run([str(argument) for argument in arguments], cwd=ROOT, check=True)


def _validate_environment() -> None:
    if sys.platform != "win32":
        raise SystemExit("Windows release builds must be run on Windows.")
    if sys.version_info[:3] != (3, 13, 15) or sys.maxsize <= 2**32:
        raise SystemExit("Windows release builds require 64-bit CPython 3.13.15.")


def _archive_release() -> None:
    ARCHIVE.parent.mkdir(parents=True, exist_ok=True)
    ARCHIVE.unlink(missing_ok=True)
    with zipfile.ZipFile(ARCHIVE, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9) as destination:
        destination.writestr(f"{RELEASE_FOLDER.name}/", b"")
        for source in sorted(RELEASE_FOLDER.rglob("*")):
            name = source.relative_to(STAGING_ROOT).as_posix()
            if source.is_dir():
                destination.writestr(f"{name}/", b"")
            else:
                destination.write(source, name)


def main() -> None:
    _validate_environment()
    _run(sys.executable, "packaging/prepare_texconv.py")
    _run(sys.executable, "packaging/collect_licenses.py")
    _run(sys.executable, "-m", "PyInstaller", "--noconfirm", "--clean", "--distpath", STAGING_ROOT, "--workpath", ROOT / "build" / "pyinstaller", ROOT / "packaging" / "rbr2beamng.spec")
    _run(RELEASE_FOLDER / "rbr2beamng-cli.exe", "--version")
    shutil.copytree(ROOT / "rbr2beamng" / "licenses", RELEASE_FOLDER / "licenses")
    shutil.copy2(ROOT / "README.md", RELEASE_FOLDER / "README.md")
    _archive_release()
    print(f"Built {ARCHIVE.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
