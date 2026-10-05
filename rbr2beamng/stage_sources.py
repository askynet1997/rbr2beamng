from __future__ import annotations

from pathlib import Path

from .core import ConversionError, ProgressReporter, is_rbr_install
from .filesystem import current_filesystem
from .models import StageInspection
from .original.source import discover_original_stages
from .rbr import discover_stages as discover_rx_stages


def discover_stages(
    rbr_root: Path,
    reporter: ProgressReporter | None = None,
    *,
    inspect_original_variants: bool = True,
) -> list[StageInspection]:
    filesystem = current_filesystem()
    root = filesystem.read_path(rbr_root)
    if not is_rbr_install(root, filesystem):
        raise ConversionError(f"Not a valid RBR installation: {root}")
    result: list[StageInspection] = []
    if filesystem.is_dir(root / "RX_CONTENT" / "TRACKS"):
        result.extend(discover_rx_stages(root, reporter))
    if inspect_original_variants:
        result.extend(discover_original_stages(root, reporter))
    else:
        result.extend(
            discover_original_stages(
                root,
                reporter,
                inspect_variants=False,
            )
        )
    return sorted(
        result,
        key=lambda stage: (
            stage.metadata.name.casefold(),
            stage.source_format,
            stage.source_key,
        ),
    )

