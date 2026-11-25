"""Optional features: every module or package in this folder is loaded when present.

A plugin can define any of these:

COMPANION_MOD
    The CompanionMod that the GUI offers to install.
companion_mod_files(filename) -> Mapping[str, str]
    Adds extra archive entries to the named companion mod zip.
STAGE_INDICATOR
    A StageIndicator shown beside compatible stage names in the GUI, paired
    with ``scan_supported_stages(rbr_root, stages, cancelled)``.
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
import threading
import zipfile
from collections.abc import Callable, Iterable, Iterator
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
    from ..models import StageInspection
    from ..original.adapter import PreparedOriginalVariant


@dataclass(frozen=True)
class CompanionMod:
    title: str
    filename: str
    install: Callable[[Path, FileSandbox], None]
    indicator: str = ""


@dataclass(frozen=True)
class StageIndicator:
    symbol: str
    label: str
    scan_order: int = 0


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


def companion_mod_files(filename: str) -> dict[str, str]:
    entries: dict[str, str] = {}
    for plugin in loaded():
        hook = getattr(plugin, "companion_mod_files", None)
        if hook is not None and hook is not companion_mod_files:
            entries.update(hook(filename))
    return entries


def stage_indicators() -> list[StageIndicator]:
    return [
        plugin.STAGE_INDICATOR
        for plugin in loaded()
        if hasattr(plugin, "STAGE_INDICATOR")
        and hasattr(plugin, "scan_supported_stages")
    ]


def scan_stage_indicators(
    rbr_root: Path,
    stages: Iterable[StageInspection],
    cancelled: threading.Event,
) -> Iterator[tuple[str, StageIndicator]]:
    stage_list = tuple(stages)
    plugins_with_indicators = sorted(
        (
            plugin
            for plugin in loaded()
            if hasattr(plugin, "STAGE_INDICATOR")
            and hasattr(plugin, "scan_supported_stages")
        ),
        key=lambda plugin: plugin.STAGE_INDICATOR.scan_order,
    )
    for plugin in plugins_with_indicators:
        if cancelled.is_set():
            return
        indicator = plugin.STAGE_INDICATOR
        for source_key in plugin.scan_supported_stages(
            rbr_root, stage_list, cancelled
        ):
            if cancelled.is_set():
                return
            yield source_key, indicator


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
