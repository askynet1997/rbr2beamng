"""Optional features: every module or package in this folder is loaded when present.

A plugin can define any of these:

COMPANION_MOD
    The CompanionMod that the GUI offers to install.
find_rx(stage, origin, map_yaw_degrees, warnings)
find_original(lbs, texture_ini, textures, warnings)
    What the plugin needs from an RX stage or an Original variant; write_level
    gets it as PluginLevel.found.
write_level(level: PluginLevel) -> dict[str, int]
    Adds the plugin's files to a converted level and returns stats to record.
"""

from __future__ import annotations

import importlib
import pkgutil
import zipfile
from collections.abc import Callable
from dataclasses import dataclass, replace
from functools import cache
from pathlib import Path
from types import ModuleType
from typing import TYPE_CHECKING

from ..profiling import profile_span

if TYPE_CHECKING:
    import numpy as np

    from ..beamng import WaterSpec
    from ..filesystem import FileSandbox
    from ..original.adapter import PreparedOriginalVariant


@dataclass(frozen=True)
class CompanionMod:
    title: str
    filename: str
    install: Callable[[Path, FileSandbox], None]


@dataclass(frozen=True)
class PluginLevel:
    archive: zipfile.ZipFile
    archived_paths: set[str]
    level_dir: Path
    level_id: str
    # Subtracted from the source positions find_original returns; zero for RX,
    # whose positions find_rx already places in the level.
    origin: np.ndarray
    version_suffix: str
    waters: list[WaterSpec]
    prepared: PreparedOriginalVariant | None
    progress: Callable[[str], None]
    found: object = None


@cache
def loaded() -> tuple[ModuleType, ...]:
    names = sorted({module.name for module in pkgutil.iter_modules(__path__)})
    return tuple(importlib.import_module(f"{__name__}.{name}") for name in names)


def companion_mods() -> list[CompanionMod]:
    return [plugin.COMPANION_MOD for plugin in loaded() if hasattr(plugin, "COMPANION_MOD")]


def find(hook: str, *arguments) -> dict[ModuleType, object]:
    found = {}
    for plugin in loaded():
        if hasattr(plugin, hook):
            with profile_span(hook, category="plugin", plugin=plugin.__name__):
                found[plugin] = getattr(plugin, hook)(*arguments)
    return found


def write_level(level: PluginLevel, found: dict[ModuleType, object]) -> dict[str, int]:
    stats: dict[str, int] = {}
    for plugin in loaded():
        if hasattr(plugin, "write_level"):
            with profile_span("write_level", category="plugin", plugin=plugin.__name__):
                stats.update(plugin.write_level(replace(level, found=found.get(plugin))))
    return stats
