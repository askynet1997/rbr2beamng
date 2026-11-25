"""Start clock overlays that the RBR Countdown Clocks mod animates in BeamNG.

Conversion modules load only when a conversion needs them, keeping them out of
the GUI's startup.
"""

from __future__ import annotations

import threading
from collections.abc import Iterable, Iterator
from pathlib import Path
from typing import TYPE_CHECKING

from .. import CompanionMod, PluginLevel, StageIndicator
from .mod import COUNTDOWN_CLOCKS_MOD_FILENAME, install_countdown_clocks_mod

if TYPE_CHECKING:
    from ...models import StageInspection

STAGE_INDICATOR = StageIndicator(
    symbol="⏱",
    label="Countdown clock",
    scan_order=1,
)

COMPANION_MOD = CompanionMod(
    "Countdown Clocks",
    COUNTDOWN_CLOCKS_MOD_FILENAME,
    install_countdown_clocks_mod,
    indicator=STAGE_INDICATOR.symbol,
)


def scan_supported_stages(
    rbr_root: Path,
    stages: Iterable[StageInspection],
    cancelled: threading.Event,
) -> Iterator[str]:
    stage_list = tuple(stages)
    if any(stage.source_format == "rx" and stage.valid for stage in stage_list):
        from .start_clocks import has_rx_start_clocks

        for stage in stage_list:
            if cancelled.is_set():
                return
            if (
                stage.source_format == "rx"
                and stage.valid
                and has_rx_start_clocks(stage.root, cancelled)
            ):
                yield stage.source_key
    if any(
        stage.source_format == "original" and stage.valid
        for stage in stage_list
    ):
        from .original import scan_original_start_clocks

        yield from scan_original_start_clocks(rbr_root, stage_list, cancelled)


def find_rx(stage, origin, map_yaw_degrees: float, warnings: list[str]):
    from .start_clocks import find_rx_start_clocks

    return find_rx_start_clocks(stage, origin, map_yaw_degrees, warnings)


def find_original(lbs, texture_ini, textures, warnings: list[str]):
    from .original import adapt_start_clocks

    return adapt_start_clocks(lbs, texture_ini, textures, warnings)


def write_level(level: PluginLevel) -> dict[str, int]:
    if level.found is None:
        return {}
    from .start_clocks import write_start_clocks

    level.progress("Writing start clock overlays")
    write_start_clocks(
        level.archive,
        level.archived_paths,
        level.level_dir,
        level.level_id,
        level.found,
        level.origin,
        level.version_suffix,
    )
    return {"countdownClocks": len(level.found.clocks)}
