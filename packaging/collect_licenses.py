from __future__ import annotations

import importlib.metadata, re, shutil, sys, tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TEXCONV_PLATFORMS = ("windows-x64", "linux-x64", "macos-universal")
_REQUIREMENT_NAME = re.compile(r"\s*([A-Za-z0-9][A-Za-z0-9._-]*)")
_FILLER_PATH_PARTS = frozenset({"include", "lib", "libs", "share", "src"})


def _copy(source: Path, destination: Path) -> None:
    if not source.is_file():
        raise RuntimeError(f"Missing license file: {source}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)


def _normalized_name(name: str) -> str:
    return re.sub(r"[-_.]+", "-", name).casefold()


def _license_filename(name: str, path: Path) -> str:
    parts = list(path.parts)
    if parts and parts[0].endswith(".dist-info"):
        parts.pop(0)
    if parts and parts[0].casefold() == "licenses":
        parts.pop(0)
    if not parts:
        raise RuntimeError(f"License file has no name: {path}")
    filename = parts.pop()
    folders = []
    for part in parts:
        component = part.strip("_")
        if (
            component
            and component.casefold() not in _FILLER_PATH_PARTS
            and _normalized_name(component) != _normalized_name(name)
        ):
            folders.append(component)
    return "_".join((name, *folders[-3:], filename))


def _copy_flattened(
    source: Path,
    destination: Path,
    name: str,
    path: Path,
) -> None:
    base_destination = destination / _license_filename(name, path)
    numbered_destination = base_destination
    number = 2
    while numbered_destination.exists():
        if (
            numbered_destination.is_file()
            and numbered_destination.read_bytes() == source.read_bytes()
        ):
            return
        numbered_destination = base_destination.with_name(
            f"{base_destination.stem}_{number}{base_destination.suffix}"
        )
        number += 1
    _copy(source, numbered_destination)


def _project_dependency_names() -> list[str]:
    with (ROOT / "pyproject.toml").open("rb") as source:
        project = tomllib.load(source)["project"]
    requirements = list(project.get("dependencies", ()))
    for dependencies in project.get("optional-dependencies", {}).values():
        requirements.extend(dependencies)
    dependency_names = set()
    for requirement in requirements:
        match = _REQUIREMENT_NAME.match(requirement)
        if match is None:
            raise RuntimeError(f"Unsupported project dependency: {requirement}")
        dependency_names.add(match.group(1))
    return sorted(dependency_names, key=str.casefold)


def _copy_distribution_licenses(name: str, destination: Path) -> None:
    distribution = importlib.metadata.distribution(name)
    copied = False
    for entry in sorted(distribution.files or (), key=str):
        path = Path(str(entry))
        if not (
            path.name.casefold().startswith(("license", "copying", "notice"))
            or "licenses" in tuple(part.casefold() for part in path.parts)
        ):
            continue
        source = Path(distribution.locate_file(entry))
        if source.is_file():
            _copy_flattened(source, destination, name, path)
            copied = True
    if not copied:
        raise RuntimeError(f"No license files found for {name}")


def collect(destination: Path) -> None:
    if destination.exists():
        if not destination.is_dir():
            raise RuntimeError(f"License destination is not a directory: {destination}")
        shutil.rmtree(destination)
    destination.mkdir(parents=True)

    for platform_name in TEXCONV_PLATFORMS:
        directory = ROOT / "rbr2beamng" / "bin" / platform_name
        _copy_flattened(
            directory / "LICENSE",
            destination,
            "Texconv-Custom-DLL",
            Path(platform_name, "LICENSE"),
        )
        _copy_flattened(
            directory / "THIRD_PARTY_LICENSES.txt",
            destination,
            "Texconv-Custom-DLL",
            Path(platform_name, "THIRD_PARTY_LICENSES.txt"),
        )
    for name in _project_dependency_names():
        _copy_distribution_licenses(name, destination)

    _copy_flattened(
        Path(sys.base_prefix) / "LICENSE.txt",
        destination,
        "CPython",
        Path("LICENSE.txt"),
    )
    tcl_root = Path(sys.base_prefix) / "tcl"
    for path in tcl_root.rglob("license.terms") if tcl_root.is_dir() else ():
        _copy_flattened(
            path,
            destination,
            "CPython",
            Path("tcl") / path.relative_to(tcl_root),
        )
    _copy(
        ROOT / "packaging" / "licenses_extra.txt",
        destination / "Additional notices.txt",
    )


def main() -> None:
    destination = (
        Path(sys.argv[1])
        if len(sys.argv) > 1
        else ROOT / "rbr2beamng" / "licenses"
    )
    collect(destination.resolve())


if __name__ == "__main__":
    main()
