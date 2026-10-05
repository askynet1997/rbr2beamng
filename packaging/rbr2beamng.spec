import sys
from pathlib import Path

from PyInstaller.building.datastruct import TOC
from PyInstaller.utils.hooks import collect_all, collect_data_files, collect_submodules


project_root = Path(SPEC).resolve().parents[1]


def without_embedded_license_files(entries):
    return TOC(
        entry
        for entry in entries
        if not (
            entry[2] == "DATA"
            and (
                Path(entry[0]).name.casefold().startswith(("license", "copying", "notice"))
                or "licenses" in tuple(part.casefold() for part in Path(entry[0]).parts)
            )
        )
    )


assimp_datas, assimp_binaries, assimp_hidden = collect_all("assimp_py")
xatlas_datas, xatlas_binaries, xatlas_hidden = collect_all("xatlas")
texconv_executable = {
    "win32": "bin/windows-x64/texconv.exe",
    "linux": "bin/linux-x64/texconv",
    "darwin": "bin/macos-universal/texconv",
}[sys.platform]
converter_datas = collect_data_files(
    "rbr2beamng",
    includes=[
        "data/*",
        "bin/VERSION.txt",
    ],
)
converter_datas.append(
    (
        str(project_root / "rbr2beamng" / texconv_executable),
        str(Path("rbr2beamng") / Path(texconv_executable).parent),
    )
)

common = {
    "pathex": [str(project_root)],
    "binaries": assimp_binaries + xatlas_binaries,
    "datas": assimp_datas + xatlas_datas + converter_datas,
    # Plugins are found at runtime, so nothing imports them by name.
    "hiddenimports": assimp_hidden + xatlas_hidden + collect_submodules("rbr2beamng.plugins"),
    "hookspath": [],
    "hooksconfig": {},
    "runtime_hooks": [],
    "excludes": [],
    "noarchive": False,
    "optimize": 1,
}

cli_analysis = Analysis([str(project_root / "packaging" / "cli_entry.py")], **common)
cli_pyz = PYZ(cli_analysis.pure)
cli_exe = EXE(
    cli_pyz,
    cli_analysis.scripts,
    [],
    exclude_binaries=True,
    name="rbr2beamng-cli",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=True,
)

gui_analysis = Analysis([str(project_root / "packaging" / "gui_entry.py")], **common)
gui_pyz = PYZ(gui_analysis.pure)
gui_exe = EXE(
    gui_pyz,
    gui_analysis.scripts,
    [],
    exclude_binaries=True,
    name="rbr2beamng",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    console=False,
)
cli_analysis.datas = without_embedded_license_files(cli_analysis.datas)
gui_analysis.datas = without_embedded_license_files(gui_analysis.datas)

COLLECT(
    cli_exe,
    gui_exe,
    cli_analysis.binaries,
    cli_analysis.datas,
    gui_analysis.binaries,
    gui_analysis.datas,
    strip=False,
    upx=False,
    name="rbr2beamng",
)
