from __future__ import annotations

import json
import math
import os
import queue
import re
import subprocess
import sys
import threading
import time
import tkinter as tk
import traceback
import webbrowser
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, replace
from pathlib import Path
from tkinter import filedialog, font as tkfont, messagebox, ttk

from PIL import Image, ImageOps, ImageTk

from . import __version__
from .core import (
    ConversionError,
    SETTINGS_BEAMNG_MODS_PATH,
    SETTINGS_FOLIAGE_GROUND_TYPES,
    SETTINGS_FOLIAGE_GROUND_TYPES_ENABLED,
    SETTINGS_FOLIAGE_NAME_FALLBACK,
    SETTINGS_FOLIAGE_NAME_MATCHES,
    SETTINGS_RBR_PATH,
    SETTINGS_SNOWBANK_NAME_FALLBACK,
    SETTINGS_SNOWBANK_NAME_MATCHES,
    SETTINGS_SNOWBANK_NAME_MESH_PATTERNS,
    SETTINGS_WATER_NAME_FALLBACK,
    SETTINGS_WATER_NAME_MATCHES,
    create_runtime_filesystem,
    find_beamng_mods_dir,
    find_rbr_install,
    format_duration,
    format_file_size,
    is_beamng_mods_dir,
    load_settings,
    save_settings,
    settings_path,
    slugify,
    stage_zip_name,
)
from .environment import DEFAULT_ENVIRONMENT_DATE, parse_environment_date
from .filesystem import FileSandbox, SandboxViolationError, use_filesystem
from .models import (
    DEFAULT_FOLIAGE_GROUND_TYPES,
    DEFAULT_FOLIAGE_NAME_MATCHES,
    DEFAULT_SNOWBANK_NAME_MATCHES,
    DEFAULT_SNOWBANK_NAME_MESH_PATTERNS,
    DEFAULT_WATER_NAME_MATCHES,
    StageDocument,
    StageLocation,
)
from .original.source import (
    inspect_original_stage,
    primary_original_tint,
)
from .pdf_viewer import PdfError, PdfViewer, pdf_text
from .plugins import CompanionMod, companion_mods
from .rbr import (
    PLAIN_TEXT_DOCUMENT_SUFFIXES,
    country_locations,
    location_for_country_code,
    read_rbr_surfaces,
)
from .surface_profiles import (
    DEFAULT_SURFACE_RULES,
    GROUND_TYPES,
    SOLID_GROUND_TYPES,
    SURFACE_COEFFICIENTS,
    SurfaceRules,
    bundled_surface_profile_catalog,
    load_surface_rules,
    surface_profile_override_path,
    surface_rules_data,
)
from .pacenote_overrides import (
    FIELDS_BY_KIND as PACENOTE_OVERRIDE_FIELDS_BY_KIND,
    load_pacenote_overrides,
    pacenote_override_kinds,
    pacenote_override_path,
)
from .pacenotes import (
    PACENOTE_OVERRIDE_KIND_LABELS,
    PACENOTE_REFERENCE_STATUS_LABELS,
    PacenoteReferenceEntry,
    PacenoteReferenceGroup,
    pacenote_conversion_reference,
    pacenote_field_choices,
    pacenote_installation_reference,
    pacenote_override_result,
)
from .stage_sources import discover_stages


_DARK_BACKGROUND = "#1b1b1b"
_DARK_CONTROL = "#333333"
_DARK_CONTROL_ACTIVE = "#454545"
_DARK_INPUT = "#111111"
_DARK_BORDER = "#555555"
_DARK_FOREGROUND = "#f2f2f2"
_DARK_MUTED = "#858585"
_DARK_SELECTION = "#0e639c"
_DARK_PROGRESS = "#18a637"
_DARK_SUCCESS_ACTIVE = "#22bd4b"
_DARK_SUCCESS_DISABLED = "#126c2a"
_DARK_DANGER = "#b83232"
_DARK_DANGER_ACTIVE = "#d64545"
_DARK_DANGER_DISABLED = "#5c3434"
_DARK_DANGER_TEXT = "#ff6b6b"
_DARK_DANGER_INPUT = "#3a1717"
_DARK_DISABLED_FOREGROUND = "#aaa0a0"
_DARK_SUCCESS_TEXT = "#4ade80"
_DARK_LINK = "#64b5f6"
_DARK_PACENOTE_MODIFIER = "#c084fc"
OptionStats = dict[str, bool | int | str]
_OPTION_STAT_COUNTER_KEYS = (
    "sourceSkyboxPartsRemoved",
    "waterObjects",
    "snowwallCollisionOverrideParts",
    "snowwallCollisionOverrideFaces",
    "thinWallComponentsDetected",
    "thinWallComponentsRepaired",
    "thinWallComponentsRestored",
    "thinWallFacesGenerated",
    "visualLodGroups",
    "brakeWallSegments",
    "brakeWallFaces",
)
_OPTION_STAT_BOOLEAN_KEYS = (
    "sourceSkyboxRemovalEnabled",
    "snowwallCollisionOverrideEnabled",
    "thinWallInflationEnabled",
    "visualLodsEnabled",
    "mapBorderBrakeWallsEnabled",
)
_ENVIRONMENT_OPTIONS = (
    ("M", "Morning", "Low sun, light fog"),
    ("N", "Noon", "Clear daylight"),
    ("O", "Overcast", "Heavy cloud cover"),
    ("E", "Evening", "Low evening sun"),
    ("S", "Custom source", "Neutral noon weather"),
)
_SURFACE_COLORS = {
    "gravel": "#c49a6c",
    "snow": "#d9efff",
    "tarmac": "#888888",
    "unknown": "#a970d6",
}
_EMAIL_DOMAIN_SEPARATOR = r"(?:\.|\(\s*dot\s*\)|\[\s*dot\s*\]|\bdot\b)"
_EMAIL_PATTERN = (
    r"(?<![A-Z0-9._%+-])"
    r"(?P<email_local>[A-Z0-9._%+-]+)"
    r"(?:\s*@\s*|\s*(?:\(\s*[A-Z0-9_-]+\s*\)|\[\s*[A-Z0-9_-]+\s*\])\s*|\s+\bat\b\s+)"
    rf"(?P<email_domain>[A-Z0-9-]+(?:\s*{_EMAIL_DOMAIN_SEPARATOR}\s*[A-Z0-9-]+)+)"
    r"(?![A-Z0-9.-])"
)
_LINK_PATTERN = re.compile(
    r"(?P<url>\b(?:https?://|www\.)[^\s<>{}\[\]\"']+)"
    rf"|(?P<email>{_EMAIL_PATTERN})",
    re.IGNORECASE,
)
_EMAIL_DOMAIN_SEPARATOR_PATTERN = re.compile(
    rf"\s*{_EMAIL_DOMAIN_SEPARATOR}\s*",
    re.IGNORECASE,
)
_COPYRIGHT_PATTERN = re.compile(
    r"^[^\r\n]*(?:©\s*\d{4}|copyright\s*(?:\(\s*c\s*\)\s*)?\d{4}|\(\s*c\s*\)\s*\S)[^\r\n]*\r?$",
    re.IGNORECASE | re.MULTILINE,
)
_LINK_TRAILING_PUNCTUATION = ".,;:!?)]}"
_STAGE_CREATOR_NOTICE_TEXT = (
    "Creating rally stages requires considerable time and effort. Please respect the "
    "wishes of modders and authors, including any licensing terms they've "
    "set for their work. If you're unsure whether you have permission to convert "
    "a stage, please contact its creator first."
)
_PROGRAM_DESCRIPTION = (
    "Convert RBR RX and Original RBR stages into BeamNG.drive levels"
)
_REPOSITORY_URL = "https://github.com/askynet1997/rbr2beamng"
_APP_TITLE = f"RBR2BeamNG v{__version__}"


def _title(name: str) -> str:
    return f"{name} - {_APP_TITLE}"


def _checkbox_text(selected: bool) -> str:
    return "☑" if selected else "☐"


def _foliage_name_matches(value: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            slugify(line.strip())
            for line in value.splitlines()
            if line.strip()
        )
    )


def _regex_patterns(value: str) -> tuple[str, ...]:
    patterns = tuple(
        dict.fromkeys(
            line.strip()
            for line in value.splitlines()
            if line.strip()
        )
    )
    for pattern in patterns:
        re.compile(pattern)
    return patterns


def _ground_types(value: str) -> tuple[str, ...]:
    ground_types = tuple(
        dict.fromkeys(
            line.strip().upper()
            for line in value.splitlines()
            if line.strip()
        )
    )
    unknown = [ground_type for ground_type in ground_types if ground_type not in GROUND_TYPES]
    if unknown:
        raise ValueError(
            f"Unknown BeamNG ground type: {', '.join(unknown)}. "
            f"Known types: {', '.join(sorted(GROUND_TYPES))}"
        )
    return ground_types


def _snowbank_name_matches(value: str) -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            name_match
            for line in value.splitlines()
            if (
                name_match := "".join(
                    char for char in line.casefold() if char.isalnum()
                )
            )
        )
    )


def _auto_hiding_scrollbar_setter(
    scrollbar: ttk.Scrollbar,
) -> Callable[[str, str], None]:
    def set_scrollbar(first: str, last: str) -> None:
        if float(first) <= 0.0 and float(last) >= 1.0:
            scrollbar.grid_remove()
        else:
            scrollbar.grid()
        scrollbar.set(first, last)

    return set_scrollbar


def _text_lines(text: tk.Text) -> list[str]:
    return [line.strip() for line in text.get("1.0", tk.END).splitlines() if line.strip()]


def _on_text_change(text: tk.Text, callback: Callable[[], None]) -> None:
    def changed(_event=None) -> None:
        if text.edit_modified():
            text.edit_modified(False)
            callback()

    text.bind("<<Modified>>", changed, add="+")


def _word_start_matches(value: str) -> tuple[str, ...]:
    matches: list[str] = []
    for line in value.splitlines():
        text = line.strip()
        body = text.lstrip("!").strip()
        if body:
            prefix = "!" if text.startswith("!") else ""
            matches.extend(
                prefix + word
                for word in re.findall(r"[a-z0-9]+", slugify(body))
            )
    return tuple(dict.fromkeys(matches))


def _configure_dark_theme(root: tk.Tk) -> None:
    root.configure(background=_DARK_BACKGROUND)
    style = ttk.Style(root)
    style.theme_use("clam")
    style.configure("TFrame", background=_DARK_BACKGROUND)
    style.configure("TLabel", background=_DARK_BACKGROUND, foreground=_DARK_FOREGROUND)
    style.configure("Muted.TLabel", foreground=_DARK_MUTED)
    style.configure(
        "TButton",
        background=_DARK_CONTROL,
        foreground=_DARK_FOREGROUND,
        bordercolor=_DARK_BORDER,
        focuscolor=_DARK_SELECTION,
        padding=(8, 4),
    )
    style.map(
        "TButton",
        background=[
            ("disabled", _DARK_BACKGROUND),
            ("pressed", _DARK_SELECTION),
            ("active", _DARK_CONTROL_ACTIVE),
        ],
        foreground=[("disabled", _DARK_MUTED)],
    )
    style.configure("Picker.TButton", padding=(3, 1))
    style.configure("Season.TButton", padding=(5, 2))
    style.map(
        "Season.TButton",
        background=[
            ("disabled", _DARK_BACKGROUND),
            ("selected", _DARK_SELECTION),
            ("pressed", _DARK_SELECTION),
            ("active", _DARK_CONTROL_ACTIVE),
        ],
        foreground=[("disabled", _DARK_MUTED)],
    )
    style.configure(
        "Success.TButton",
        background=_DARK_PROGRESS,
        foreground=_DARK_FOREGROUND,
    )
    style.map(
        "Success.TButton",
        background=[
            ("disabled", _DARK_SUCCESS_DISABLED),
            ("pressed", _DARK_SUCCESS_ACTIVE),
            ("active", _DARK_SUCCESS_ACTIVE),
        ],
        foreground=[("disabled", _DARK_MUTED)],
    )
    style.configure(
        "Action.TButton",
        background=_DARK_SELECTION,
        foreground=_DARK_FOREGROUND,
    )
    style.map(
        "Action.TButton",
        background=[
            ("disabled", _DARK_BACKGROUND),
            ("pressed", _DARK_LINK),
            ("active", _DARK_LINK),
        ],
        foreground=[("disabled", _DARK_MUTED)],
    )
    style.configure(
        "Danger.TButton",
        background=_DARK_DANGER,
        foreground=_DARK_FOREGROUND,
    )
    style.map(
        "Danger.TButton",
        background=[
            ("disabled", _DARK_DANGER_DISABLED),
            ("pressed", _DARK_DANGER_ACTIVE),
            ("active", _DARK_DANGER_ACTIVE),
        ],
        foreground=[("disabled", _DARK_DISABLED_FOREGROUND)],
    )
    style.configure(
        "TEntry",
        fieldbackground=_DARK_INPUT,
        foreground=_DARK_FOREGROUND,
        insertcolor=_DARK_FOREGROUND,
        bordercolor=_DARK_BORDER,
        lightcolor=_DARK_BORDER,
        darkcolor=_DARK_BORDER,
    )
    style.map(
        "TEntry",
        fieldbackground=[("disabled", _DARK_BACKGROUND)],
        foreground=[("disabled", _DARK_MUTED)],
    )
    style.configure(
        "TCombobox",
        fieldbackground=_DARK_INPUT,
        background=_DARK_CONTROL,
        foreground=_DARK_FOREGROUND,
        arrowcolor=_DARK_FOREGROUND,
        bordercolor=_DARK_BORDER,
        lightcolor=_DARK_BORDER,
        darkcolor=_DARK_BORDER,
    )
    style.map(
        "TCombobox",
        fieldbackground=[
            ("disabled", _DARK_BACKGROUND),
            ("readonly", _DARK_INPUT),
        ],
        background=[
            ("disabled", _DARK_BACKGROUND),
            ("readonly", _DARK_CONTROL),
        ],
        foreground=[
            ("disabled", _DARK_MUTED),
            ("readonly", _DARK_FOREGROUND),
        ],
    )
    root.option_add("*TCombobox*Listbox.background", _DARK_INPUT)
    root.option_add("*TCombobox*Listbox.foreground", _DARK_FOREGROUND)
    root.option_add("*TCombobox*Listbox.selectBackground", _DARK_SELECTION)
    root.option_add("*TCombobox*Listbox.selectForeground", _DARK_FOREGROUND)
    style.configure(
        "Error.TEntry",
        fieldbackground=_DARK_DANGER_INPUT,
        foreground=_DARK_DANGER_TEXT,
        insertcolor=_DARK_DANGER_TEXT,
        bordercolor=_DARK_DANGER,
        lightcolor=_DARK_DANGER,
        darkcolor=_DARK_DANGER,
    )
    style.configure(
        "Preview.TEntry",
        fieldbackground=_DARK_CONTROL_ACTIVE,
        foreground=_DARK_FOREGROUND,
        insertcolor=_DARK_FOREGROUND,
        bordercolor=_DARK_SELECTION,
        lightcolor=_DARK_SELECTION,
        darkcolor=_DARK_SELECTION,
    )
    style.map(
        "Preview.TEntry",
        fieldbackground=[("disabled", _DARK_BACKGROUND)],
        foreground=[("disabled", _DARK_MUTED)],
    )
    style.configure(
        "Treeview",
        background=_DARK_INPUT,
        fieldbackground=_DARK_INPUT,
        foreground=_DARK_FOREGROUND,
        bordercolor=_DARK_BORDER,
        rowheight=22,
    )
    style.map(
        "Treeview",
        background=[("selected", _DARK_SELECTION)],
        foreground=[("selected", _DARK_FOREGROUND)],
    )
    style.configure(
        "Treeview.Heading",
        background=_DARK_CONTROL,
        foreground=_DARK_FOREGROUND,
        bordercolor=_DARK_BORDER,
        relief=tk.FLAT,
    )
    style.map(
        "Treeview.Heading",
        background=[("pressed", _DARK_SELECTION), ("active", _DARK_CONTROL_ACTIVE)],
    )
    style.configure(
        "TNotebook",
        background=_DARK_BACKGROUND,
        bordercolor=_DARK_BORDER,
        tabmargins=(0, 0, 0, 0),
    )
    style.configure(
        "TNotebook.Tab",
        background=_DARK_CONTROL,
        foreground=_DARK_FOREGROUND,
        bordercolor=_DARK_BORDER,
        padding=(8, 4),
    )
    style.map(
        "TNotebook.Tab",
        background=[("selected", _DARK_SELECTION), ("active", _DARK_CONTROL_ACTIVE)],
        foreground=[("disabled", _DARK_MUTED)],
    )
    style.configure(
        "TCheckbutton",
        background=_DARK_BACKGROUND,
        foreground=_DARK_FOREGROUND,
        indicatorbackground=_DARK_INPUT,
        indicatorforeground=_DARK_FOREGROUND,
    )
    style.map(
        "TCheckbutton",
        background=[("active", _DARK_BACKGROUND)],
        foreground=[("disabled", _DARK_MUTED)],
        indicatorbackground=[("selected", _DARK_SELECTION)],
    )
    style.configure("Wrap.TCheckbutton", wraplength=340)
    style.configure("Header.TCheckbutton", wraplength=250)
    style.configure(
        "TRadiobutton",
        background=_DARK_BACKGROUND,
        foreground=_DARK_FOREGROUND,
        indicatorbackground=_DARK_INPUT,
        indicatorforeground=_DARK_FOREGROUND,
    )
    style.map(
        "TRadiobutton",
        background=[("active", _DARK_BACKGROUND)],
        foreground=[("disabled", _DARK_MUTED)],
        indicatorbackground=[("selected", _DARK_SELECTION)],
    )
    style.configure(
        "Danger.TLabel",
        foreground=_DARK_DANGER_TEXT,
    )
    style.configure(
        "TScrollbar",
        background=_DARK_CONTROL,
        troughcolor=_DARK_INPUT,
        bordercolor=_DARK_BACKGROUND,
        arrowcolor=_DARK_FOREGROUND,
    )
    style.map("TScrollbar", background=[("active", _DARK_CONTROL_ACTIVE)])


def _find_links(value: str) -> list[tuple[int, int, str]]:
    result: list[tuple[int, int, str]] = []
    for match in _LINK_PATTERN.finditer(value):
        label = match.group(0).rstrip(_LINK_TRAILING_PUNCTUATION)
        if not label:
            continue
        target = (
            "mailto:"
            + match.group("email_local")
            + "@"
            + _EMAIL_DOMAIN_SEPARATOR_PATTERN.sub(
                ".",
                match.group("email_domain"),
            )
            if match.group("email") is not None
            else label
        )
        if target.casefold().startswith("www."):
            target = f"https://{target}"
        result.append((match.start(), match.start() + len(label), target))
    return result


def _collect_contact_links(
    values: Iterable[str],
) -> tuple[list[tuple[str, str]], list[tuple[str, str]]]:
    emails: dict[str, tuple[str, str]] = {}
    urls: dict[str, tuple[str, str]] = {}
    for value in values:
        for start, end, target in _find_links(value):
            is_email = target.casefold().startswith("mailto:")
            label = target.removeprefix("mailto:") if is_email else value[start:end]
            links = emails if is_email else urls
            links.setdefault(target.casefold(), (label, target))
    return list(emails.values()), list(urls.values())


def _collect_copyright_notices(values: Iterable[str]) -> list[str]:
    notices: dict[str, str] = {}
    for value in values:
        for match in _COPYRIGHT_PATTERN.finditer(value):
            notice = " ".join(match.group(0).split())
            notices.setdefault(notice.casefold(), notice)
    return list(notices.values())


def _surface_label(value: str) -> str:
    return value.strip().casefold() or "unknown"


def _pacenote_reference_values(
    entry: PacenoteReferenceEntry,
) -> tuple[str, str, str]:
    return (
        entry.entry_type,
        PACENOTE_REFERENCE_STATUS_LABELS[entry.status],
        entry.result,
    )


def _pacenote_reference_tag(status: str) -> str:
    if status in {"unsupported", "loss"}:
        return "unsupported"
    if status == "route":
        return "route"
    if status in {"metadata", "unresolved"}:
        return "distance"
    if status == "conditional":
        return "modifier"
    return "pacenote"


def _filter_pacenote_reference_groups(
    groups: tuple[PacenoteReferenceGroup, ...],
    query: str,
    group_key: str,
) -> tuple[tuple[PacenoteReferenceGroup, tuple[PacenoteReferenceEntry, ...]], ...]:
    needle = query.strip().casefold()
    result = []
    for group in groups:
        if group_key and group.key != group_key:
            continue
        entries = tuple(
            entry
            for entry in group.entries
            if not needle
            or needle
            in " ".join(
                (
                    entry.source,
                    entry.entry_type,
                    entry.status,
                    entry.result,
                    entry.detail,
                )
            ).casefold()
        )
        if entries:
            result.append((group, entries))
    return tuple(result)


def _pacenote_preview_command(
    cli_command: list[str],
    rbr_root: Path,
    stage,
    output_dir: Path,
    original_variants: tuple[str, ...] = (),
) -> list[str]:
    stage_selector = (
        stage.source_key
        if stage.source_format == "original"
        else stage.metadata.folder_name
    )
    command = [
        *cli_command,
        "pacenote-visualizer",
        "--rbr-dir",
        str(rbr_root),
        "--stage",
        stage_selector,
        "--output",
        str(output_dir),
    ]
    for variant in original_variants:
        command.extend(("--original-variant", variant))
    return command


def _pacenote_preview_outputs(output: str) -> tuple[Path, ...]:
    return tuple(
        Path(line.removeprefix("Visualizer: ").strip())
        for line in output.splitlines()
        if line.startswith("Visualizer: ") and line.removeprefix("Visualizer: ").strip()
    )


def _surface_color(value: str) -> str:
    return _SURFACE_COLORS.get(_surface_label(value), _SURFACE_COLORS["unknown"])


def _stage_is_installed(
    beamng_mods_dir: str,
    folder_name: str,
    filesystem: FileSandbox,
    *,
    display_name: str = "",
    source_format: str = "rx",
) -> bool:
    if not beamng_mods_dir.strip():
        return False
    try:
        return filesystem.is_file(
            Path(beamng_mods_dir).expanduser()
            / stage_zip_name(
                folder_name,
                display_name=display_name,
                source_format=source_format,
            )
        )
    except OSError:
        return False


def _parse_temperatures(
    night: str,
    day: str,
) -> tuple[float, float] | None:
    try:
        values = (
            float(night.strip().replace(",", ".")),
            float(day.strip().replace(",", ".")),
        )
    except ValueError:
        return None
    if not all(math.isfinite(value) for value in values) or values[0] > values[1]:
        return None
    return values


def _parse_coordinates(
    latitude: str,
    longitude: str,
) -> tuple[float, float] | None:
    try:
        values = (
            float(latitude.strip().replace(",", ".")),
            float(longitude.strip().replace(",", ".")),
        )
    except ValueError:
        return None
    if (
        not all(math.isfinite(value) for value in values)
        or not -90.0 <= values[0] <= 90.0
        or not -180.0 <= values[1] <= 180.0
    ):
        return None
    return values


def _coordinate_fields_valid(latitude: str, longitude: str) -> bool:
    if not latitude.strip() and not longitude.strip():
        return True
    return _parse_coordinates(latitude, longitude) is not None


def _parse_map_altitude(value: str) -> float | None:
    text = value.strip().replace(",", ".")
    if not text:
        return None
    try:
        altitude = float(text)
    except ValueError:
        return None
    return altitude if math.isfinite(altitude) else None


def _parse_environment_date(value: str) -> str | None:
    try:
        parsed = parse_environment_date(value)
    except ConversionError:
        return None
    return parsed.isoformat() if parsed is not None else ""


def _environment_date_field_valid(value: str) -> bool:
    return _parse_environment_date(value) is not None


def _stage_coordinate_sources(
    stage,
) -> tuple[tuple[float, float] | None, tuple[float, float] | None]:
    location = getattr(stage, "location", None)
    if location is None:
        return None, None
    map_coordinates = (
        (location.latitude, location.longitude)
        if location.precision == "exact"
        else None
    )
    country_code = getattr(location, "country_code", "")
    country_location = (
        location_for_country_code(country_code)
        if country_code
        else None
    )
    country_coordinates = (
        (country_location.latitude, country_location.longitude)
        if country_location is not None
        else (
            (location.latitude, location.longitude)
            if location.precision == "country"
            else None
        )
    )
    return map_coordinates, country_coordinates


def _default_stage_coordinates(stage) -> tuple[float, float] | None:
    map_coordinates, country_coordinates = _stage_coordinate_sources(stage)
    if map_coordinates is not None:
        return map_coordinates
    if country_coordinates is not None:
        return country_coordinates
    return None


def _stage_map_altitude(stage) -> float | None:
    altitude = getattr(getattr(stage, "location", None), "altitude_meters", None)
    try:
        altitude = float(altitude)
    except (TypeError, ValueError):
        return None
    return altitude if math.isfinite(altitude) else None


def _coordinate_override(
    stage,
    coordinates: tuple[float, float] | None,
) -> tuple[float, float] | None:
    if coordinates is None:
        return None
    source = _default_stage_coordinates(stage)
    if source and all(
        math.isclose(value, source_value, abs_tol=1e-8)
        for value, source_value in zip(coordinates, source)
    ):
        return None
    return coordinates


def _format_coordinate(value: float) -> str:
    return f"{value:.8f}".rstrip("0").rstrip(".")


def _stage_detail_caption(stage) -> str:
    location = _stage_location_name(stage)
    return (
        f"{stage.metadata.name} - {location}"
        if location != "—"
        else stage.metadata.name
    )


def _snowbank_option_text() -> str:
    return "Give collision to snowbank surfaces the stage marks as non-colliding"


def _stage_location_name(stage) -> str:
    location = getattr(stage, "location", None)
    return (
        str(getattr(location, "country", "")).strip()
        or str(getattr(location, "country_code", "")).strip()
        or "—"
    )


def _stage_warning(stage) -> str:
    return "" if stage.valid else "\n".join(stage.issues)


def _stage_tree_tags(stage) -> tuple[str, ...]:
    return () if stage.valid else ("flagged",)


def _stage_conversion_action(stage, *, exists: bool, overwrite: bool) -> str:
    if not stage.valid:
        return "Can't convert"
    if not exists:
        return "New"
    return "Overwrite" if overwrite else "Skip"


def _stage_conversion_summary(actions: list[str]) -> str:
    new = actions.count("New")
    overwritten = actions.count("Overwrite")
    return f"{new + overwritten} to be converted ({new} new, {overwritten} overwritten)"


def _stage_legal_terms(stage) -> tuple[str, ...]:
    return tuple(
        document.title
        for document in getattr(stage, "documents", ())
        if document.path.name.casefold() != "track.ini"
    )


def _stage_display_values(stage) -> tuple[str, ...]:
    length = (
        f"{stage.metadata.length_km:.2f} km"
        if stage.metadata.length_km is not None
        else "—"
    )
    return (
        "Original" if stage.source_format == "original" else "RX",
        f"     {stage.metadata.surface_composition_text}",
        length,
        _stage_location_name(stage),
        stage.metadata.name,
        stage.metadata.author or "—",
        str(getattr(stage.metadata, "comment", "")).strip() or "—",
    )


def _stage_matches_search(stage, search: str) -> bool:
    query = search.strip().casefold()
    return not query or any(
        query in value.casefold()
        for value in (*_stage_display_values(stage), *_stage_legal_terms(stage))
    )


def _location_temperature_ranges(
    location: StageLocation | None,
) -> dict[str, tuple[float, float]]:
    if location is None:
        return {}
    result: dict[str, tuple[float, float]] = {}
    for season in ("summer", "autumn", "winter"):
        night = getattr(location, f"{season}_temperature_night", None)
        day = getattr(location, f"{season}_temperature_day", None)
        if night is not None and day is not None:
            result[season] = float(night), float(day)
    return result


def _stage_temperature_ranges(stage) -> dict[str, tuple[float, float]]:
    return _location_temperature_ranges(getattr(stage, "location", None))


def _location_profile_options(
    stage,
) -> tuple[tuple[str, StageLocation], ...]:
    location = getattr(stage, "location", None)
    if location is None:
        return ()
    return (
        (f"Detected: {_stage_location_name(stage)}", location),
    ) + tuple(
        (country_location.country, country_location)
        for country_location in country_locations()
    )


def _temperature_season_text(
    season: str,
    temperatures: tuple[float, float],
) -> str:
    return (
        f"{season.title()}: {temperatures[0]:g} to "
        f"{temperatures[1]:g}°C"
    )


def _temperature_season_for_surface(surface: str) -> str:
    return {
        "snow": "winter",
        "tarmac": "summer",
    }.get(surface.strip().casefold(), "autumn")


def _format_conversion_status(
    message: str,
    elapsed_seconds: float,
) -> str:
    return f"{_format_elapsed_timestamp(elapsed_seconds)}: {message}"


# Share of a stage's conversion time at which each converter phase starts and
# ends, measured from timestamped GUI conversion logs.
_RX_PHASE_SPANS = {
    "prepare": (0.0, 0.08),
    "parse": (0.0, 0.08),
    "geometry": (0.08, 0.75),
    "objects": (0.75, 0.85),
    "level": (0.85, 0.94),
    "textures": (0.94, 0.95),
    "validate": (0.95, 0.95),
    "install": (0.95, 1.0),
}
_ORIGINAL_PHASE_SPANS = {
    "prepare": (0.0, 0.06),
    "install": (0.90, 1.0),
}
_ORIGINAL_VARIANTS_SPAN = (0.06, 0.90)
_ORIGINAL_LEVEL_SHARE_OF_VARIANT = 0.8


def _stage_progress_span(
    source_format: str,
    phase: object,
    current: object,
    total: object,
) -> tuple[float, float] | None:
    """Return the stage share already done and the most this event can reach."""
    if phase == "complete":
        return 1.0, 1.0
    counted = (
        isinstance(current, int)
        and isinstance(total, int)
        and 0 < current <= total
    )
    if source_format == "original":
        if phase in ("original", "level", "rally") and counted:
            start, end = _ORIGINAL_VARIANTS_SPAN
            done = current - 1
            if phase == "original":
                begin, finish = done, done + _ORIGINAL_LEVEL_SHARE_OF_VARIANT
            else:
                begin, finish = done + _ORIGINAL_LEVEL_SHARE_OF_VARIANT, current
            scale = (end - start) / total
            return start + scale * begin, start + scale * finish
        span = _ORIGINAL_PHASE_SPANS.get(phase)
        counted = False
    else:
        span = _RX_PHASE_SPANS.get(phase)
    if span is None:
        return None
    start, end = span
    if not counted:
        return span
    step = (end - start) / total
    return start + step * (current - 1), start + step * current


def _estimate_remaining_seconds(
    *,
    elapsed: float,
    reached: float,
    reached_at: float,
    limit: float,
    stage_format: str,
    queued_formats: Iterable[str],
    completed_durations: Mapping[str, list[float]],
) -> float | None:
    history = completed_durations.get(stage_format)
    if reached > 0.0:
        projected = max(reached_at / reached, elapsed / limit)
    elif history:
        projected = sum(history) / len(history)
    else:
        return None
    remaining = max(0.0, projected - elapsed)
    for source_format in queued_formats:
        durations = completed_durations.get(source_format)
        remaining += sum(durations) / len(durations) if durations else projected
    return remaining


def _format_collision_merge_summary(
    collision_triangles: int | None,
    merged_away_triangles: int | None,
) -> str:
    if (
        collision_triangles is None
        or merged_away_triangles is None
    ):
        return ""
    before = collision_triangles + merged_away_triangles
    saved = merged_away_triangles / before * 100.0 if before else 0.0
    return (
        f"After optimization: {collision_triangles:,} triangles "
        f"({saved:.1f}% saved)\n"
    )


def _option_stat_bool(
    option_stats: Mapping[str, object],
    key: str,
) -> bool | None:
    value = option_stats.get(key)
    return value if isinstance(value, bool) else None


def _option_stat_count(
    option_stats: Mapping[str, object],
    key: str,
) -> int | None:
    value = option_stats.get(key)
    return (
        value
        if isinstance(value, int) and not isinstance(value, bool)
        else None
    )


def _option_stats_from_event(value: object) -> OptionStats | None:
    if not isinstance(value, dict):
        return None
    option_stats = {
        str(key): item
        for key, item in value.items()
        if isinstance(item, (bool, int, str))
    }
    return option_stats or None


def _aggregate_option_stats(
    option_stats_values: Iterable[Mapping[str, object]],
) -> OptionStats:
    values = tuple(option_stats_values)
    if not values:
        return {}
    result: OptionStats = {}
    source_formats = {
        value
        for option_stats in values
        if isinstance(
            value := option_stats.get("sourceFormat"),
            str,
        )
    }
    if source_formats:
        result["sourceFormat"] = (
            source_formats.pop()
            if len(source_formats) == 1
            else "mixed"
        )
    for key in _OPTION_STAT_COUNTER_KEYS:
        result[key] = sum(
            value
            for option_stats in values
            if isinstance(
                value := option_stats.get(key),
                int,
            )
            and not isinstance(value, bool)
        )
    for key in _OPTION_STAT_BOOLEAN_KEYS:
        flags = [
            value
            for option_stats in values
            if isinstance(
                value := option_stats.get(key),
                bool,
            )
        ]
        if flags:
            result[key] = all(flags)
    supported = [
        value
        for option_stats in values
        if isinstance(
            value := option_stats.get("mapBorderBrakeWallsSupported"),
            bool,
        )
    ]
    if supported:
        result["mapBorderBrakeWallsSupported"] = any(supported)
    return result


def _format_conversion_option_summary(
    option_stats: Mapping[str, object] | None,
) -> str:
    if not option_stats:
        return ""
    lines = []
    skybox_enabled = _option_stat_bool(
        option_stats,
        "sourceSkyboxRemovalEnabled",
    )
    skybox_parts = _option_stat_count(
        option_stats,
        "sourceSkyboxPartsRemoved",
    )
    if skybox_enabled is None:
        lines.append("Procedural sky/clouds: Unavailable")
    elif skybox_enabled:
        outcome = (
            f"{skybox_parts:,} dome meshes detected"
            if skybox_parts is not None
            else "Enabled"
        )
        lines.append(f"Procedural sky/clouds: {outcome}")
    else:
        lines.append("Procedural sky/clouds: Off")

    water_objects = _option_stat_count(option_stats, "waterObjects")
    lines.append(
        "Procedural water: "
        + (
            f"{water_objects:,} water meshes detected"
            if water_objects is not None
            else "Unavailable"
        )
    )
    snowwall_enabled = _option_stat_bool(
        option_stats,
        "snowwallCollisionOverrideEnabled",
    )
    snowwall_parts = _option_stat_count(
        option_stats,
        "snowwallCollisionOverrideParts",
    )
    snowwall_faces = _option_stat_count(
        option_stats,
        "snowwallCollisionOverrideFaces",
    )
    if snowwall_enabled is None:
        lines.append("Snowbank physics: Unavailable")
    elif snowwall_enabled:
        if snowwall_parts is None or snowwall_faces is None:
            outcome = "Enabled"
        else:
            outcome = f"{snowwall_parts:,} collision parts / {snowwall_faces:,} faces"
        lines.append(f"Snowbank physics: {outcome}")
    else:
        lines.append("Snowbank physics: Off")

    thin_walls_enabled = _option_stat_bool(
        option_stats,
        "thinWallInflationEnabled",
    )
    thin_walls_repaired = _option_stat_count(
        option_stats,
        "thinWallComponentsRepaired",
    )
    thin_walls_restored = _option_stat_count(
        option_stats,
        "thinWallComponentsRestored",
    )
    if thin_walls_enabled is None:
        lines.append("Thickened collision walls: Unavailable")
    elif thin_walls_enabled:
        outcome = (
            f"{thin_walls_repaired:,} walls"
            if thin_walls_repaired is not None
            else "Enabled"
        )
        if thin_walls_restored:
            outcome += f", {thin_walls_restored:,} unchanged"
        lines.append(f"Thickened collision walls: {outcome}")
    else:
        lines.append("Thickened collision walls: Off")

    visual_lods_enabled = _option_stat_bool(option_stats, "visualLodsEnabled")
    visual_lod_groups = _option_stat_count(option_stats, "visualLodGroups")
    if visual_lods_enabled is None:
        lines.append("Visual LODs: Unavailable")
    elif visual_lods_enabled:
        outcome = (
            f"{visual_lod_groups:,} output groups"
            if visual_lod_groups is not None
            else "Enabled"
        )
        lines.append(f"Visual LODs: {outcome}")
    else:
        lines.append("Visual LODs: Off")

    brake_walls_supported = _option_stat_bool(
        option_stats,
        "mapBorderBrakeWallsSupported",
    )
    brake_walls_enabled = _option_stat_bool(
        option_stats,
        "mapBorderBrakeWallsEnabled",
    )
    brake_wall_segments = _option_stat_count(
        option_stats,
        "brakeWallSegments",
    )
    brake_wall_faces = _option_stat_count(option_stats, "brakeWallFaces")
    if brake_walls_supported is False:
        lines.append("Invisible border walls: Not applicable to RX stage")
    elif brake_walls_enabled is None:
        lines.append("Invisible border walls: Unavailable")
    elif not brake_walls_enabled:
        lines.append("Invisible border walls: Off")
    elif brake_wall_segments is None or brake_wall_faces is None:
        lines.append("Invisible border walls: Enabled")
    else:
        lines.append(
            "Invisible border walls: "
            f"{brake_wall_segments:,} segments / {brake_wall_faces:,} faces"
        )
    return "\n".join(("Options:", *(f" - {line}" for line in lines))) + "\n"


def _format_batch_conversion_complete_summary(
    succeeded: int,
    failed: int,
    option_stats: Mapping[str, object] | None,
) -> str:
    summary = (
        f"Selected stages: {succeeded} succeeded, {failed} failed."
    )
    option_summary = _format_conversion_option_summary(option_stats)
    return (
        f"{summary}\n\n{option_summary.rstrip()}"
        if option_summary
        else summary
    )


def _format_conversion_complete_summary(
    elapsed_seconds: float,
    output_path: Path | None,
    zip_size_bytes: int | None,
    collision_triangles: int | None,
    merged_away_triangles: int | None,
    route_length_meters: float | None,
    sector_count: int | None,
    option_stats: Mapping[str, object] | None = None,
) -> str:
    size_text = (
        format_file_size(zip_size_bytes)
        if zip_size_bytes is not None
        else "Unavailable"
    )
    collision_text = (
        f"{collision_triangles:,}"
        if collision_triangles is not None
        else "Unavailable"
    )
    static_collision_text = (
        f"{collision_triangles + merged_away_triangles:,}"
        if collision_triangles is not None and merged_away_triangles is not None
        else collision_text
    )
    return (
        "The stage has been converted, installed in the BeamNG mods "
        "folder, and is ready to use.\n\n"
        f"Stage: {route_length_meters / 1000:.2f} km · {sector_count} sectors\n"
        f"Elapsed total: {format_duration(elapsed_seconds)}\n"
        f"ZIP size: {size_text}\n"
        f"Static collision: {static_collision_text} triangles\n"
        f"{_format_collision_merge_summary(collision_triangles, merged_away_triangles)}"
        f"{_format_conversion_option_summary(option_stats)}"
        f"\n{output_path}"
    )


def _format_completed_conversion_status(
    elapsed_seconds: float,
    output_path: Path | None,
    zip_size_bytes: int | None,
    collision_triangles: int | None,
) -> str:
    output_name = output_path.name if output_path is not None else "output.zip"
    size = (
        f"   {format_file_size(zip_size_bytes).replace(' ', '')}"
        if zip_size_bytes is not None
        else ""
    )
    coltris = (
        f" ({collision_triangles} coltris)"
        if collision_triangles is not None
        else ""
    )
    return (
        f"{_format_elapsed_timestamp(elapsed_seconds)}: "
        f"Conversion completed: {output_name}{size}{coltris}"
    )


def _format_elapsed_timestamp(seconds: float) -> str:
    total_milliseconds = max(0, int(seconds * 1000))
    total_seconds, milliseconds = divmod(total_milliseconds, 1000)
    minutes, seconds = divmod(total_seconds, 60)
    return f"{minutes:02d}:{seconds:02d}.{milliseconds:03d}"


def _centered_position(
    width: int,
    height: int,
    left: int,
    top: int,
    right: int,
    bottom: int,
) -> tuple[int, int]:
    x = left + max(0, (right - left - width) // 2)
    y = top + max(0, (bottom - top - height) // 2)
    return x, y


def _centered_geometry(
    width: int,
    height: int,
    screen_width: int,
    screen_height: int,
) -> str:
    x, y = _centered_position(width, height, 0, 0, screen_width, screen_height)
    return f"{width}x{height}+{x}+{y}"


def _current_monitor_work_area() -> tuple[int, int, int, int] | None:
    if sys.platform != "win32":
        return None

    import ctypes
    from ctypes import wintypes

    class MonitorInfo(ctypes.Structure):
        _fields_ = [
            ("cbSize", wintypes.DWORD),
            ("rcMonitor", wintypes.RECT),
            ("rcWork", wintypes.RECT),
            ("dwFlags", wintypes.DWORD),
        ]

    try:
        user32 = ctypes.windll.user32
        user32.GetCursorPos.argtypes = (ctypes.POINTER(wintypes.POINT),)
        user32.GetCursorPos.restype = wintypes.BOOL
        user32.MonitorFromPoint.argtypes = (wintypes.POINT, wintypes.DWORD)
        user32.MonitorFromPoint.restype = ctypes.c_void_p
        user32.GetMonitorInfoW.argtypes = (
            ctypes.c_void_p,
            ctypes.POINTER(MonitorInfo),
        )
        user32.GetMonitorInfoW.restype = wintypes.BOOL

        cursor = wintypes.POINT()
        if not user32.GetCursorPos(ctypes.byref(cursor)):
            return None
        monitor = user32.MonitorFromPoint(cursor, 2)
        if not monitor:
            return None
        monitor_info = MonitorInfo()
        monitor_info.cbSize = ctypes.sizeof(monitor_info)
        if not user32.GetMonitorInfoW(monitor, ctypes.byref(monitor_info)):
            return None
    except (AttributeError, OSError):
        return None

    work_area = monitor_info.rcWork
    return work_area.left, work_area.top, work_area.right, work_area.bottom


def _set_window_position(
    window: tk.Tk,
    x: int,
    y: int,
) -> bool:
    if sys.platform != "win32":
        return False

    import ctypes
    from ctypes import wintypes

    try:
        user32 = ctypes.windll.user32
        user32.SetWindowPos.argtypes = (
            wintypes.HWND,
            wintypes.HWND,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            ctypes.c_int,
            wintypes.UINT,
        )
        user32.SetWindowPos.restype = wintypes.BOOL
        return bool(
            user32.SetWindowPos(
                int(window.wm_frame(), 0),
                None,
                x,
                y,
                0,
                0,
                0x0001 | 0x0004 | 0x0010,
            )
        )
    except (AttributeError, OSError):
        return False


def _place_window_on_current_monitor(
    window: tk.Tk,
    width: int,
    height: int,
) -> None:
    work_area = _current_monitor_work_area()
    if work_area is None:
        window.geometry(
            _centered_geometry(
                width,
                height,
                window.winfo_screenwidth(),
                window.winfo_screenheight(),
            )
        )
        return

    x, y = _centered_position(width, height, *work_area)
    window.geometry(f"{width}x{height}")
    window.update_idletasks()
    if _set_window_position(window, x, y):
        return
    if x >= 0 and y >= 0:
        window.geometry(f"{width}x{height}+{x}+{y}")
        return
    window.geometry(
        _centered_geometry(
            width,
            height,
            window.winfo_screenwidth(),
            window.winfo_screenheight(),
        )
    )


def _licenses_directory() -> Path:
    if getattr(sys, "frozen", False):
        return Path(sys.executable).with_name("licenses")
    return Path(__file__).with_name("licenses")


def _open_path(path: Path) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Path does not exist: {path}")
    if sys.platform == "win32":
        os.startfile(path)
    elif sys.platform == "darwin":
        subprocess.Popen(["open", str(path)])
    else:
        subprocess.Popen(["xdg-open", str(path)])


def _timestamp_log_text(
    text: str,
    elapsed_seconds: float,
    stage_label: str = "",
) -> str:
    timestamp = _format_elapsed_timestamp(elapsed_seconds)
    prefix = f"{timestamp}: "
    if stage_label:
        prefix += f"{stage_label}: "
    return "".join(
        prefix + line
        for line in text.splitlines(keepends=True)
    )


def _can_decode(data: bytes, encoding: str) -> bool:
    try:
        data.decode(encoding)
        return True
    except UnicodeDecodeError:
        return False


@dataclass(frozen=True)
class _ConversionJob:
    stage: object
    command: list[str]
    destination: Path


class _ProgressDisplay(tk.Canvas):
    def __init__(self, parent) -> None:
        style = ttk.Style(parent)
        background = style.lookup("TFrame", "background") or "#f0f0f0"
        foreground = style.lookup("TLabel", "foreground") or "#111111"
        super().__init__(
            parent,
            height=20,
            background=background,
            highlightthickness=1,
            highlightbackground=_DARK_BORDER,
            takefocus=False,
        )
        self._value = 0.0
        self._text = ""
        self._eta = ""
        self._foreground = foreground
        self._fill = _DARK_SELECTION
        self._busy_job: str | None = None
        self._busy_phase = 0.0
        self.bind("<Configure>", self._redraw)

    def set(
        self,
        *,
        value: float | None = None,
        text: str | None = None,
        tone: str | None = None,
        eta: str | None = None,
    ) -> None:
        if value is not None:
            self.stop_busy()
            self._value = min(100.0, max(0.0, value))
        if text is not None:
            self._text = text
        if eta is not None:
            self._eta = eta
        if tone == "normal":
            self._fill = _DARK_SELECTION
        elif tone == "success":
            self._fill = _DARK_PROGRESS
        elif tone == "failure":
            self._fill = _DARK_DANGER
        self._redraw()

    def start_busy(self) -> None:
        if self._busy_job is None:
            self._busy_phase = 0.0
            self._busy_tick()

    def stop_busy(self) -> None:
        if self._busy_job is not None:
            self.after_cancel(self._busy_job)
            self._busy_job = None

    def _busy_tick(self) -> None:
        self._busy_phase = (self._busy_phase + 0.01) % 2.0
        self._redraw()
        self._busy_job = self.after(30, self._busy_tick)

    def _redraw(self, _event=None) -> None:
        self.delete("all")
        width = max(1, self.winfo_width())
        height = max(1, self.winfo_height())
        if self._busy_job is not None:
            segment = width * 0.2
            bounce = self._busy_phase if self._busy_phase <= 1.0 else 2.0 - self._busy_phase
            start = (width - segment) * bounce
            self.create_rectangle(
                start,
                0,
                start + segment,
                height,
                fill=self._fill,
                outline="",
            )
        elif (fill_width := width * self._value / 100.0) > 0:
            self.create_rectangle(0, 0, fill_width, height, fill=self._fill, outline="")
        self.create_text(
            8,
            height / 2,
            text=self._text,
            fill=self._foreground,
            font="TkDefaultFont",
            anchor=tk.W,
        )
        if self._eta:
            self.create_text(
                width - 8,
                height / 2,
                text=self._eta,
                fill=self._foreground,
                font="TkDefaultFont",
                anchor=tk.E,
            )


class ConverterApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        _configure_dark_theme(self.root)
        self.stage_caption_font = tkfont.nametofont("TkDefaultFont").copy()
        caption_font_size = self.stage_caption_font.cget("size")
        self.stage_caption_font.configure(
            size=caption_font_size + 5 if caption_font_size > 0 else caption_font_size - 5
        )
        self.root.title(_APP_TITLE)
        _place_window_on_current_monitor(self.root, 1500, 874)
        self.root.minsize(800, 600)
        self.root.state("zoomed")

        self.rbr_path = tk.StringVar()
        self.beamng_mods_dir = tk.StringVar()
        self.remove_source_skybox = tk.BooleanVar(value=True)
        self.use_water_name_fallback = tk.BooleanVar(value=True)
        self.water_name_matches = DEFAULT_WATER_NAME_MATCHES
        self.use_snowwall_collision_override = tk.BooleanVar(value=True)
        self.use_snowbank_name_fallback = tk.BooleanVar(value=True)
        self.snowbank_name_matches = DEFAULT_SNOWBANK_NAME_MATCHES
        self.snowbank_name_mesh_patterns = DEFAULT_SNOWBANK_NAME_MESH_PATTERNS
        self.inflate_thin_walls = tk.BooleanVar(value=True)
        self.use_visual_lods = tk.BooleanVar(value=False)
        self.use_map_border_brake_walls = tk.BooleanVar(value=True)
        self.use_foliage_name_fallback = tk.BooleanVar(value=True)
        self.foliage_name_matches = DEFAULT_FOLIAGE_NAME_MATCHES
        self.use_foliage_ground_types = tk.BooleanVar(value=True)
        self.foliage_ground_types = DEFAULT_FOLIAGE_GROUND_TYPES
        self.original_variant_values = {
            tint: tk.BooleanVar(value=False)
            for tint in ("M", "N", "O", "E", "S")
        }
        self.temperature_night = tk.StringVar(value="10")
        self.temperature_day = tk.StringVar(value="15")
        self.latitude = tk.StringVar()
        self.longitude = tk.StringVar()
        self.map_altitude = tk.StringVar()
        self.environment_date = tk.StringVar()
        self.batch_override_coordinates = tk.BooleanVar(value=False)
        self.batch_override_map_altitude = tk.BooleanVar(value=False)
        self.batch_override_environment_date = tk.BooleanVar(value=False)
        self.batch_override_temperatures = tk.BooleanVar(value=False)
        self.location_profile = tk.StringVar()
        self.location_profiles: dict[str, StageLocation] = {}
        self.selected_location: StageLocation | None = None
        self.temperature_season = tk.StringVar()
        self.temperature_ranges: dict[str, tuple[float, float]] = {}
        self._previewed_entries: tuple[ttk.Entry, ...] = ()
        self.stage_caption = tk.StringVar()
        self.stage_issue = tk.StringVar()
        self.stage_search = tk.StringVar()
        self.stage_thumbnail: ImageTk.PhotoImage | None = None
        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.stage_preload_generation = 0
        self.stage_preload_processed_keys: set[str] = set()
        self.stage_preload_cancel = threading.Event()
        self.stage_preload_requests: queue.Queue[str] = queue.Queue()
        self.stage_preload_thread: threading.Thread | None = None
        self.stage_preload_worker_lock = threading.Lock()
        self._stage_preload_active_generation: int | None = None
        self.process: subprocess.Popen[str] | None = None
        self.conversion_active = False
        self.pacenote_preview_active = False
        self.cancel_requested = False
        self.cancellation_logged = False
        self.conversion_started_at: float | None = None
        self.conversion_status = ""
        self.last_elapsed_second = -1
        self.completed_zip_size: int | None = None
        self.completed_collision_triangles: int | None = None
        self.completed_merged_away_triangles: int | None = None
        self.completed_route_length_meters: float | None = None
        self.completed_sector_count: int | None = None
        self.completed_option_stats: OptionStats | None = None
        self.stages = []
        self.inspected_original_stage_keys: set[str] = set()
        self.pending_current_conversion_key: str | None = None
        self.pending_batch_conversion_keys: set[str] = set()
        self.conversion_jobs: list[_ConversionJob] = []
        self.conversion_job_index = 0
        self.batch_failures = 0
        self.batch_failed_stage_names: list[str] = []
        self.batch_option_stats: list[OptionStats] = []
        self.completed_stage_durations: dict[str, list[float]] = {}
        self.stage_progress = (0.0, 0.0)
        self.stage_progress_reached_at = 0.0
        self.conversion_log_path: Path | None = None
        self.current_conversion_stage_name = ""
        self.last_output: Path | None = None
        self.selected_stage_keys: set[str] = set()
        self.sort_column: str | None = None
        self.sort_descending = False
        self.surface_markers: dict[str, tk.Frame] = {}
        self.environment_stage_root: str | None = None
        self.original_variant_stage_key: str | None = None
        self.filesystem = create_runtime_filesystem()

        self._build_ui()
        self.stage_search.trace_add("write", lambda *_: self._populate_tree())
        self.beamng_mods_dir.trace_add("write", self._on_beamng_mods_dir_changed)
        self.temperature_night.trace_add("write", self._on_temperature_changed)
        self.temperature_day.trace_add("write", self._on_temperature_changed)
        self.latitude.trace_add("write", lambda *_: self._update_convert_state())
        self.longitude.trace_add("write", lambda *_: self._update_convert_state())
        self.map_altitude.trace_add("write", lambda *_: self._update_convert_state())
        self.environment_date.trace_add("write", lambda *_: self._update_convert_state())
        self.batch_override_coordinates.trace_add(
            "write", lambda *_: self._update_convert_state()
        )
        self.batch_override_map_altitude.trace_add(
            "write", lambda *_: self._update_convert_state()
        )
        self.batch_override_environment_date.trace_add(
            "write", lambda *_: self._update_convert_state()
        )
        self.batch_override_temperatures.trace_add(
            "write", lambda *_: self._update_convert_state()
        )
        self._detect_paths()
        self.root.after_idle(self._focus_stage_search)
        self.root.after(100, self._drain_events)
        self.root.protocol("WM_DELETE_WINDOW", self._close)

    def _build_ui(self) -> None:
        outer = ttk.Frame(self.root, padding=12)
        outer.pack(fill=tk.BOTH, expand=True)
        outer.columnconfigure(0, weight=1)
        outer.rowconfigure(1, weight=1)

        paths = ttk.Frame(outer)
        paths.grid(row=0, column=0, sticky=tk.EW, pady=4)
        paths.columnconfigure(0, weight=1, uniform="path")
        paths.columnconfigure(1, weight=1, uniform="path")

        rbr_path = ttk.Frame(paths)
        rbr_path.grid(row=0, column=0, sticky=tk.EW, padx=(0, 8))
        rbr_path.columnconfigure(1, weight=1)
        ttk.Label(rbr_path, text="RBR install folder:").grid(row=0, column=0, sticky=tk.W, padx=(0, 8))
        rbr_picker = ttk.Frame(rbr_path)
        rbr_picker.grid(row=0, column=1, sticky=tk.EW)
        rbr_picker.columnconfigure(0, weight=1)
        ttk.Entry(rbr_picker, textvariable=self.rbr_path).grid(row=0, column=0, sticky=tk.EW)
        ttk.Button(
            rbr_picker,
            text="…",
            width=3,
            style="Picker.TButton",
            command=self._browse_rbr,
        ).grid(row=0, column=1, sticky=tk.NS)

        beamng_mods_path = ttk.Frame(paths)
        beamng_mods_path.grid(row=0, column=1, sticky=tk.EW, padx=(8, 0))
        beamng_mods_path.columnconfigure(1, weight=1)
        ttk.Label(beamng_mods_path, text="BeamNG mods folder:").grid(row=0, column=0, sticky=tk.W, padx=(0, 8))
        beamng_picker = ttk.Frame(beamng_mods_path)
        beamng_picker.grid(row=0, column=1, sticky=tk.EW)
        beamng_picker.columnconfigure(0, weight=1)
        self.beamng_mods_entry = ttk.Entry(
            beamng_picker,
            textvariable=self.beamng_mods_dir,
        )
        self.beamng_mods_entry.grid(row=0, column=0, sticky=tk.EW)
        ttk.Button(
            beamng_picker,
            text="…",
            width=3,
            style="Picker.TButton",
            command=self._browse_beamng_mods_dir,
        ).grid(row=0, column=1, sticky=tk.NS)

        content = ttk.Frame(outer)
        content.grid(row=1, column=0, sticky=tk.NSEW, pady=(4, 0))
        content.columnconfigure(1, weight=1)
        content.rowconfigure(1, weight=3, uniform="right_pane")
        content.rowconfigure(2, weight=1, uniform="right_pane")

        search = ttk.Frame(content)
        search.grid(row=0, column=1, columnspan=2, sticky=tk.EW)
        search.columnconfigure(2, weight=1)
        ttk.Label(search, text="Search:").grid(
            row=0,
            column=0,
            sticky=tk.W,
            padx=(0, 6),
        )
        self.stage_search_entry = ttk.Entry(
            search,
            textvariable=self.stage_search,
            width=30,
        )
        self.stage_search_entry.grid(
            row=0,
            column=1,
            sticky=tk.W,
        )
        self.stage_search_entry.bind("<Escape>", self._clear_stage_search)
        self.root.bind("<Control-f>", self._focus_stage_search)
        self.root.bind("<Control-F>", self._focus_stage_search)
        column = 3
        for mod in companion_mods():
            ttk.Button(
                search,
                text=f"Install 'RBR {mod.title}' mod",
                style="Action.TButton",
                command=lambda mod=mod: self._install_companion_mod(mod),
            ).grid(row=0, column=column, sticky=tk.E, padx=(8, 0))
            column += 1
        ttk.Button(search, text="Options", command=self._show_options).grid(
            row=0,
            column=column,
            sticky=tk.E,
            padx=(8, 0),
        )
        ttk.Button(search, text="About", command=self._show_about).grid(
            row=0,
            column=column + 1,
            sticky=tk.E,
            padx=(8, 0),
        )

        columns = (
            "selected",
            "format",
            "physics",
            "length",
            "location",
            "name",
            "author",
            "comment",
        )
        self.tree = ttk.Treeview(
            content,
            columns=columns,
            show="headings",
            selectmode="browse",
        )
        self.tree_headings = {
            "selected": "☐",
            "name": "Stage",
            "author": "Author",
            "location": "Location",
            "physics": "Surface",
            "length": "Length",
            "comment": "Comment",
            "format": "Format",
        }
        widths = {
            "selected": 32,
            "name": 240,
            "author": 180,
            "location": 120,
            "physics": 80,
            "length": 70,
            "comment": 190,
            "format": 70,
        }
        for column in columns:
            self.tree.heading(
                column,
                text=self.tree_headings[column],
                anchor=tk.CENTER if column == "selected" else tk.W,
                command=(
                    self._toggle_all_stages
                    if column == "selected"
                    else lambda selected_column=column: self._sort_tree(selected_column)
                ),
            )
            self.tree.column(
                column,
                width=widths[column],
                minwidth=32 if column == "selected" else 60,
                anchor=tk.CENTER if column == "selected" else tk.W,
                stretch=column in {"name", "author", "location", "comment"},
            )
        scrollbar = ttk.Scrollbar(
            content,
            orient=tk.VERTICAL,
            command=self._scroll_tree,
        )
        self.tree.configure(yscrollcommand=scrollbar.set)
        self.tree.tag_configure("flagged", foreground=_DARK_DANGER_TEXT)
        self.tree.grid(row=1, column=1, sticky=tk.NSEW, pady=(6, 0))
        scrollbar.grid(row=1, column=2, sticky=tk.NS, pady=(6, 0))
        self.tree.bind("<<TreeviewSelect>>", self._on_stage_selected)
        self.tree.bind("<Button-1>", self._on_tree_click, add="+")
        self.tree.bind("<Double-1>", self._on_stage_double_click)
        self.tree.bind("<Configure>", lambda _event: self._schedule_surface_marker_refresh())
        self.tree.bind("<MouseWheel>", lambda _event: self._schedule_surface_marker_refresh())
        self.tree.bind("<ButtonRelease-1>", lambda _event: self._schedule_surface_marker_refresh())
        for sequence in ("<Prior>", "<Next>", "<Home>", "<End>"):
            self.tree.bind(sequence, self._navigate_stage_list)

        details = ttk.Frame(content)
        details.grid(row=0, column=0, rowspan=4, sticky=tk.NSEW, padx=(0, 10))
        details.columnconfigure(0, weight=1)
        details.rowconfigure(1, weight=1)

        stage_frame = tk.Frame(
            details,
            background=_DARK_BACKGROUND,
            highlightbackground=_DARK_BORDER,
            highlightthickness=1,
            padx=8,
            pady=5,
        )
        stage_frame.grid(row=0, column=0, sticky=tk.EW)
        stage_frame.columnconfigure(0, weight=1)
        self.stage_caption_label = ttk.Label(
            stage_frame,
            textvariable=self.stage_caption,
            anchor=tk.CENTER,
            justify=tk.CENTER,
            wraplength=420,
            font=self.stage_caption_font,
        )
        self.stage_caption_label.grid(row=0, column=0, sticky=tk.EW, pady=(0, 6))
        self.stage_issue_label = ttk.Label(
            stage_frame,
            textvariable=self.stage_issue,
            anchor=tk.CENTER,
            justify=tk.CENTER,
            wraplength=420,
            style="Danger.TLabel",
        )
        self.stage_issue_label.grid(row=1, column=0, sticky=tk.EW, pady=(0, 6))
        self.stage_issue_label.grid_remove()
        self.stage_thumbnail_label = tk.Label(
            stage_frame,
            background=_DARK_INPUT,
            borderwidth=0,
            relief=tk.FLAT,
            highlightthickness=0,
        )
        self.stage_thumbnail_label.grid(row=2, column=0, sticky=tk.N)

        options_container = ttk.Frame(details)
        options_container.grid(row=1, column=0, sticky=tk.NSEW, pady=(6, 0))
        options_container.columnconfigure(0, weight=1)
        options_container.rowconfigure(0, weight=1)
        self.options_canvas = tk.Canvas(
            options_container,
            background=_DARK_BACKGROUND,
            borderwidth=0,
            highlightthickness=0,
            relief=tk.FLAT,
        )
        self.options_canvas.grid(row=0, column=0, sticky=tk.NSEW)
        self.options_scrollbar = ttk.Scrollbar(
            options_container,
            orient=tk.VERTICAL,
            command=self.options_canvas.yview,
        )
        self.options_scrollbar.grid(row=0, column=1, sticky=tk.NS)
        self.options_canvas.configure(yscrollcommand=self.options_scrollbar.set)
        options_content = tk.Frame(
            self.options_canvas,
            background=_DARK_BACKGROUND,
        )
        options_content.columnconfigure(0, weight=1)
        options_window = self.options_canvas.create_window(
            (0, 0),
            anchor=tk.NW,
            window=options_content,
        )

        def update_options_scroll_region(_event=None) -> None:
            self.options_canvas.configure(
                scrollregion=self.options_canvas.bbox(tk.ALL)
            )
            if options_content.winfo_reqheight() > self.options_canvas.winfo_height():
                self.options_scrollbar.grid()
            else:
                self.options_canvas.yview_moveto(0)
                self.options_scrollbar.grid_remove()

        def resize_options_content(event) -> None:
            self.options_canvas.itemconfigure(options_window, width=event.width)
            update_options_scroll_region()

        options_content.bind("<Configure>", update_options_scroll_region)
        self.options_canvas.bind("<Configure>", resize_options_content)

        self.variant_frame = tk.Frame(
            options_content,
            background=_DARK_BACKGROUND,
            highlightbackground=_DARK_BORDER,
            highlightthickness=1,
            padx=8,
            pady=5,
        )
        self.variant_frame.grid(row=0, column=0, sticky=tk.EW, pady=(0, 6))

        self.environment_frame = tk.Frame(
            options_content,
            background=_DARK_BACKGROUND,
            highlightbackground=_DARK_BORDER,
            highlightthickness=1,
            padx=8,
            pady=5,
        )
        self.environment_frame.grid(row=2, column=0, sticky=tk.EW)
        ttk.Label(
            self.environment_frame,
            text="Environment:",
        ).grid(row=0, column=0, columnspan=7, sticky=tk.W, pady=(0, 4))

        preset_frame = tk.Frame(
            options_content,
            background=_DARK_BACKGROUND,
            highlightbackground=_DARK_BORDER,
            highlightthickness=1,
            padx=8,
            pady=5,
        )
        preset_frame.grid(row=3, column=0, sticky=tk.EW, pady=(6, 0))
        ttk.Label(
            preset_frame,
            text="Preset:",
        ).grid(row=0, column=0, sticky=tk.W, pady=(0, 4))
        for column in range(3):
            preset_frame.columnconfigure(column, weight=1)

        ttk.Label(self.environment_frame, text="GPS").grid(
            row=1,
            column=0,
            sticky=tk.E,
            padx=(0, 6),
            pady=2,
        )
        ttk.Label(self.environment_frame, text="Lat").grid(
            row=1,
            column=1,
            sticky=tk.W,
        )
        self.latitude_entry = ttk.Entry(
            self.environment_frame,
            textvariable=self.latitude,
            width=5,
            justify=tk.RIGHT,
        )
        self.latitude_entry.grid(row=1, column=2, padx=(3, 0), pady=2)
        ttk.Label(self.environment_frame, text="°").grid(
            row=1,
            column=3,
            sticky=tk.W,
            padx=(0, 6),
        )
        ttk.Label(self.environment_frame, text="Lon").grid(
            row=1,
            column=4,
            sticky=tk.W,
        )
        self.longitude_entry = ttk.Entry(
            self.environment_frame,
            textvariable=self.longitude,
            width=5,
            justify=tk.RIGHT,
        )
        self.longitude_entry.grid(row=1, column=5, padx=(3, 0), pady=2)
        ttk.Label(self.environment_frame, text="°").grid(
            row=1,
            column=6,
            sticky=tk.W,
        )

        ttk.Label(self.environment_frame, text="Date").grid(
            row=2,
            column=0,
            sticky=tk.E,
            padx=(0, 6),
            pady=2,
        )
        ttk.Label(self.environment_frame, text="Local").grid(
            row=2,
            column=1,
            sticky=tk.W,
        )
        self.environment_date_entry = ttk.Entry(
            self.environment_frame,
            textvariable=self.environment_date,
            width=10,
            justify=tk.RIGHT,
        )
        self.environment_date_entry.grid(
            row=2,
            column=2,
            columnspan=3,
            sticky=tk.W,
            padx=(3, 0),
            pady=2,
        )
        ttk.Label(self.environment_frame, text="YYYY-MM-DD").grid(
            row=2,
            column=5,
            columnspan=2,
            sticky=tk.W,
            padx=(6, 0),
        )

        ttk.Label(self.environment_frame, text="Temperature").grid(
            row=3,
            column=0,
            sticky=tk.E,
            padx=(0, 6),
            pady=2,
        )
        ttk.Label(self.environment_frame, text="Day").grid(
            row=3,
            column=1,
            sticky=tk.W,
        )
        self.temperature_day_entry = ttk.Entry(
            self.environment_frame,
            textvariable=self.temperature_day,
            width=5,
            justify=tk.RIGHT,
        )
        self.temperature_day_entry.grid(row=3, column=2, padx=(3, 0), pady=2)
        ttk.Label(self.environment_frame, text="°C").grid(
            row=3,
            column=3,
            sticky=tk.W,
            padx=(0, 6),
        )
        ttk.Label(self.environment_frame, text="Night").grid(
            row=3,
            column=4,
            sticky=tk.W,
        )
        self.temperature_night_entry = ttk.Entry(
            self.environment_frame,
            textvariable=self.temperature_night,
            width=5,
            justify=tk.RIGHT,
        )
        self.temperature_night_entry.grid(row=3, column=5, padx=(3, 0), pady=2)
        ttk.Label(self.environment_frame, text="°C").grid(
            row=3,
            column=6,
            sticky=tk.W,
        )

        ttk.Label(self.environment_frame, text="Map altitude").grid(
            row=4,
            column=0,
            sticky=tk.E,
            padx=(0, 6),
            pady=2,
        )
        ttk.Label(self.environment_frame, text="MSL").grid(
            row=4,
            column=1,
            sticky=tk.W,
        )
        self.map_altitude_entry = ttk.Entry(
            self.environment_frame,
            textvariable=self.map_altitude,
            width=5,
            justify=tk.RIGHT,
        )
        self.map_altitude_entry.grid(row=4, column=2, padx=(3, 0), pady=2)
        ttk.Label(self.environment_frame, text="m").grid(
            row=4,
            column=3,
            sticky=tk.W,
            padx=(0, 6),
        )
        ttk.Label(
            self.environment_frame,
            text="Sea-level ground plane: 0 m",
        ).grid(
            row=4,
            column=4,
            columnspan=3,
            sticky=tk.W,
        )

        self.batch_environment_frame = tk.Frame(
            options_content,
            background=_DARK_BACKGROUND,
            highlightbackground=_DARK_BORDER,
            highlightthickness=1,
            padx=8,
            pady=5,
        )
        self.batch_environment_frame.grid(
            row=1,
            column=0,
            sticky=tk.EW,
            pady=(0, 6),
        )
        ttk.Label(
            self.batch_environment_frame,
            text="Multiple selected stages:",
        ).grid(row=0, column=0, sticky=tk.W, pady=(0, 4))
        self.batch_coordinates_override_check = ttk.Checkbutton(
            self.batch_environment_frame,
            text="Apply same GPS to all selected stages",
            variable=self.batch_override_coordinates,
        )
        self.batch_coordinates_override_check.grid(
            row=1,
            column=0,
            sticky=tk.W,
        )
        self.batch_map_altitude_override_check = ttk.Checkbutton(
            self.batch_environment_frame,
            text="Apply same map altitude to all selected stages",
            variable=self.batch_override_map_altitude,
        )
        self.batch_map_altitude_override_check.grid(
            row=2,
            column=0,
            sticky=tk.W,
        )
        self.batch_environment_date_override_check = ttk.Checkbutton(
            self.batch_environment_frame,
            text="Apply same date to all selected stages",
            variable=self.batch_override_environment_date,
        )
        self.batch_environment_date_override_check.grid(
            row=3,
            column=0,
            sticky=tk.W,
        )
        self.batch_temperature_override_check = ttk.Checkbutton(
            self.batch_environment_frame,
            text="Apply same temperatures to all selected stages",
            variable=self.batch_override_temperatures,
        )
        self.batch_temperature_override_check.grid(
            row=4,
            column=0,
            sticky=tk.W,
        )
        self._bind_entry_preview(
            self.batch_coordinates_override_check,
            (self.latitude_entry, self.longitude_entry),
        )
        self._bind_entry_preview(
            self.batch_map_altitude_override_check,
            (self.map_altitude_entry,),
        )
        self._bind_entry_preview(
            self.batch_environment_date_override_check,
            (self.environment_date_entry,),
        )
        self._bind_entry_preview(
            self.batch_temperature_override_check,
            (self.temperature_day_entry, self.temperature_night_entry),
        )
        self.batch_environment_frame.grid_remove()

        self.location_profile_selector = ttk.Combobox(
            preset_frame,
            textvariable=self.location_profile,
            state=tk.DISABLED,
            width=22,
        )
        self.location_profile_selector.grid(
            row=1,
            column=0,
            columnspan=2,
            sticky=tk.EW,
            padx=(0, 4),
        )
        self.location_profile_selector.bind(
            "<<ComboboxSelected>>",
            self._on_location_profile_selected,
        )
        self.country_coordinate_button = ttk.Button(
            preset_frame,
            text="GPS",
            command=self._use_selected_location_coordinates,
            state=tk.DISABLED,
        )
        self.country_coordinate_button.grid(
            row=1,
            column=2,
            sticky=tk.EW,
        )
        self._bind_entry_preview(
            self.country_coordinate_button,
            (self.latitude_entry, self.longitude_entry),
        )

        self.temperature_season_buttons: dict[str, ttk.Button] = {}
        for column, season in enumerate(("summer", "autumn", "winter")):
            button = ttk.Button(
                preset_frame,
                style="Season.TButton",
                command=lambda value=season: self._select_temperature_season(value),
            )
            button.grid(
                row=2,
                column=column,
                sticky=tk.EW,
                padx=(0, 4) if column < 2 else 0,
                pady=(3, 0),
            )
            self._bind_entry_preview(
                button,
                (
                    self.temperature_day_entry,
                    self.temperature_night_entry,
                    self.environment_date_entry,
                ),
            )
            self.temperature_season_buttons[season] = button

        self.original_variants_frame = ttk.Frame(self.variant_frame)
        self.original_variants_frame.grid(row=0, column=0, sticky=tk.W)
        ttk.Label(
            self.original_variants_frame,
            text="Extra levels with RBR's own lighting:",
        ).grid(row=0, column=0, columnspan=2, sticky=tk.W)
        self.original_variant_checks: dict[str, ttk.Checkbutton] = {}
        self.original_variant_summaries: dict[str, ttk.Label] = {}
        for tint, label, summary in _ENVIRONMENT_OPTIONS:
            check = ttk.Checkbutton(
                self.original_variants_frame,
                text=label,
                variable=self.original_variant_values[tint],
                command=self._update_convert_state,
            )
            self.original_variant_checks[tint] = check
            summary_label = ttk.Label(
                self.original_variants_frame,
                text=summary,
                style="Muted.TLabel",
            )
            self.original_variant_summaries[tint] = summary_label
        self.variant_frame.grid_remove()

        log_frame = ttk.Frame(content)
        log_frame.grid(row=2, column=1, columnspan=2, sticky=tk.NSEW, pady=(8, 0))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self.log = tk.Text(
            log_frame,
            height=10,
            wrap=tk.WORD,
            state=tk.DISABLED,
            background=_DARK_INPUT,
            foreground=_DARK_FOREGROUND,
            insertbackground=_DARK_FOREGROUND,
            selectbackground=_DARK_SELECTION,
            selectforeground=_DARK_FOREGROUND,
            highlightthickness=1,
            highlightbackground=_DARK_BORDER,
            highlightcolor=_DARK_SELECTION,
            relief=tk.FLAT,
        )
        log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log.yview)
        self.log.configure(yscrollcommand=log_scroll.set)
        self.log.grid(row=0, column=0, sticky=tk.NSEW)
        log_scroll.grid(row=0, column=1, sticky=tk.NS)

        left_controls = ttk.Frame(details)
        left_controls.grid(row=2, column=0, sticky=tk.EW, pady=(8, 0))
        left_controls.columnconfigure(1, weight=1)
        self.preview_pacenotes_button = ttk.Button(
            left_controls,
            text="Preview pacenotes",
            command=self._preview_pacenotes,
        )
        self.preview_pacenotes_button.grid(row=0, column=0, sticky=tk.NS, padx=(0, 8))
        self.convert_button = ttk.Button(
            left_controls,
            text="Convert",
            command=self._primary_action,
            style="Success.TButton",
        )
        self.convert_button.grid(row=0, column=1, sticky=tk.EW)

        def scroll_options(event) -> str:
            if options_content.winfo_reqheight() > self.options_canvas.winfo_height():
                self.options_canvas.yview_scroll(
                    -int(event.delta / 120),
                    "units",
                )
            return "break"

        def bind_options_scrollwheel(widget) -> None:
            widget.bind("<MouseWheel>", scroll_options, add="+")
            for child in widget.winfo_children():
                bind_options_scrollwheel(child)

        bind_options_scrollwheel(options_content)
        self.options_canvas.bind("<MouseWheel>", scroll_options)

        progress_row = ttk.Frame(content)
        progress_row.grid(row=3, column=1, columnspan=2, sticky=tk.EW, pady=(8, 0))
        progress_row.columnconfigure(1, weight=1)
        self.cancel_button = ttk.Button(
            progress_row,
            text="Cancel",
            command=self._cancel,
        )
        self.cancel_button.grid(row=0, column=0, sticky=tk.NS, padx=(0, 8))
        self.cancel_button.grid_remove()
        self.progress = _ProgressDisplay(progress_row)
        self.progress.grid(row=0, column=1, sticky=tk.EW)

    def _rebuild_filesystem(self) -> None:
        rbr_root = (
            Path(self.rbr_path.get()).expanduser()
            if self.rbr_path.get().strip()
            else None
        )
        beamng_mods_dir = (
            Path(self.beamng_mods_dir.get()).expanduser()
            if self.beamng_mods_dir.get().strip()
            else None
        )
        self.filesystem = create_runtime_filesystem(
            rbr_root=rbr_root,
            beamng_mods_dir=beamng_mods_dir,
        )

    def _detect_paths(self) -> None:
        settings = load_settings()
        snowbank_names_enabled = settings.get(SETTINGS_SNOWBANK_NAME_FALLBACK)
        if isinstance(snowbank_names_enabled, str):
            self.use_snowbank_name_fallback.set(
                snowbank_names_enabled.casefold() in {"1", "true", "yes", "on"}
            )
        stored_snowbank_matches = settings.get(SETTINGS_SNOWBANK_NAME_MATCHES)
        if isinstance(stored_snowbank_matches, str):
            self.snowbank_name_matches = _snowbank_name_matches(stored_snowbank_matches)
        stored_name_meshes = settings.get(SETTINGS_SNOWBANK_NAME_MESH_PATTERNS)
        if isinstance(stored_name_meshes, str):
            try:
                self.snowbank_name_mesh_patterns = _regex_patterns(stored_name_meshes)
            except re.error:
                pass
        water_fallback_enabled = settings.get(SETTINGS_WATER_NAME_FALLBACK)
        if isinstance(water_fallback_enabled, str):
            self.use_water_name_fallback.set(
                water_fallback_enabled.casefold() in {"1", "true", "yes", "on"}
            )
        stored_water_matches = settings.get(SETTINGS_WATER_NAME_MATCHES)
        if isinstance(stored_water_matches, str):
            self.water_name_matches = _word_start_matches(stored_water_matches)
        fallback_enabled = settings.get(SETTINGS_FOLIAGE_NAME_FALLBACK)
        if isinstance(fallback_enabled, str):
            self.use_foliage_name_fallback.set(
                fallback_enabled.casefold() in {"1", "true", "yes", "on"}
            )
        stored_matches = settings.get(SETTINGS_FOLIAGE_NAME_MATCHES)
        if isinstance(stored_matches, str):
            self.foliage_name_matches = _foliage_name_matches(stored_matches)
        ground_types_enabled = settings.get(SETTINGS_FOLIAGE_GROUND_TYPES_ENABLED)
        if isinstance(ground_types_enabled, str):
            self.use_foliage_ground_types.set(
                ground_types_enabled.casefold() in {"1", "true", "yes", "on"}
            )
        stored_ground_types = settings.get(SETTINGS_FOLIAGE_GROUND_TYPES)
        if isinstance(stored_ground_types, str):
            try:
                self.foliage_ground_types = _ground_types(stored_ground_types)
            except ValueError:
                pass
        self.progress.set(value=0, text="Loading stages…", tone="normal")
        self.root.update()
        rbr = find_rbr_install()
        if rbr:
            self.rbr_path.set(str(rbr))
        beamng_mods_dir = find_beamng_mods_dir()
        if beamng_mods_dir:
            self.beamng_mods_dir.set(str(beamng_mods_dir))
        self._save_paths()
        if self.rbr_path.get():
            self._refresh_stages()
        else:
            self.progress.set(value=0, text="", tone="normal")
        self._update_convert_state()

    def _save_paths(self) -> None:
        values: dict[str, str | None] = {
            SETTINGS_SNOWBANK_NAME_MATCHES: (
                None
                if self.snowbank_name_matches == DEFAULT_SNOWBANK_NAME_MATCHES
                else "\n".join(self.snowbank_name_matches)
            ),
            SETTINGS_SNOWBANK_NAME_FALLBACK: str(
                self.use_snowbank_name_fallback.get()
            ).lower(),
            SETTINGS_SNOWBANK_NAME_MESH_PATTERNS: (
                None
                if self.snowbank_name_mesh_patterns
                == DEFAULT_SNOWBANK_NAME_MESH_PATTERNS
                else "\n".join(self.snowbank_name_mesh_patterns)
            ),
            SETTINGS_WATER_NAME_FALLBACK: str(
                self.use_water_name_fallback.get()
            ).lower(),
            SETTINGS_WATER_NAME_MATCHES: (
                None
                if self.water_name_matches == DEFAULT_WATER_NAME_MATCHES
                else "\n".join(self.water_name_matches)
            ),
            SETTINGS_FOLIAGE_NAME_FALLBACK: str(
                self.use_foliage_name_fallback.get()
            ).lower(),
            SETTINGS_FOLIAGE_NAME_MATCHES: (
                None
                if self.foliage_name_matches == DEFAULT_FOLIAGE_NAME_MATCHES
                else "\n".join(self.foliage_name_matches)
            ),
            SETTINGS_FOLIAGE_GROUND_TYPES_ENABLED: str(
                self.use_foliage_ground_types.get()
            ).lower(),
            SETTINGS_FOLIAGE_GROUND_TYPES: (
                None
                if self.foliage_ground_types == DEFAULT_FOLIAGE_GROUND_TYPES
                else "\n".join(self.foliage_ground_types)
            ),
        }
        if self.rbr_path.get():
            path = str(Path(self.rbr_path.get()).expanduser().resolve())
            self.rbr_path.set(path)
            values[SETTINGS_RBR_PATH] = path
        if self.beamng_mods_dir.get():
            path = str(Path(self.beamng_mods_dir.get()).expanduser().resolve())
            self.beamng_mods_dir.set(path)
            values[SETTINGS_BEAMNG_MODS_PATH] = path
        self._rebuild_filesystem()
        if values:
            save_settings(values, self.filesystem)

    def _build_surface_profiles_tab(
        self,
        tab: ttk.Frame,
        window: tk.Toplevel,
    ) -> Callable[[], bool] | None:
        def show_unavailable(message: str) -> None:
            ttk.Label(tab, text=message, style="Muted.TLabel").grid(
                row=0,
                column=0,
                sticky=tk.W,
            )

        if not self.rbr_path.get():
            show_unavailable("Choose an RBR installation first.")
            return None
        try:
            with use_filesystem(self.filesystem):
                surfaces = read_rbr_surfaces(Path(self.rbr_path.get()))
                rules = load_surface_rules()
                bundled_catalog = bundled_surface_profile_catalog()
        except ConversionError as exc:
            show_unavailable(str(exc))
            return None
        if not surfaces:
            show_unavailable(
                "The selected installation has no readable physics.lsp "
                "in Physics/ or physics.rbz."
            )
            return None
        fingerprint = next(iter(surfaces.values())).physics_fingerprint

        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(1, weight=1)
        profiles: dict[int, dict[str, object]] = {}
        for surface_id, surface in surfaces.items():
            profile = surface.profile
            profiles[surface_id] = {
                "groundType": profile.ground_type if profile else rules.unknown_ground_type,
                "hard": profile.hard if profile else True,
                "bendable": profile.bendable if profile else False,
                "water": profile.water if profile else False,
                "collisionEligible": profile.collision_eligible if profile else True,
                "groundDepth": profile.ground_depth if profile else 0.0,
                "snowbank": profile.snowbank if profile else False,
            }
        default_profiles = {
            surface_id: {
                "groundType": profile.ground_type,
                "hard": profile.hard,
                "bendable": profile.bendable,
                "water": profile.water,
                "collisionEligible": profile.collision_eligible,
                "groundDepth": profile.ground_depth,
                "snowbank": profile.snowbank,
            }
            for surface_id, surface in surfaces.items()
            for profile, _status in (
                bundled_catalog.resolve(
                    surface.name,
                    surface.coefficients,
                    surface.flags,
                ),
            )
        }
        customized_ids = {
            surface_id
            for surface_id in surfaces
            if profiles[surface_id] != default_profiles[surface_id]
        }

        toolbar = ttk.Frame(tab)
        toolbar.grid(row=0, column=0, sticky=tk.EW, pady=(0, 4))
        toolbar.columnconfigure(1, weight=1)
        ttk.Label(toolbar, text="Search:").grid(
            row=0,
            column=0,
            sticky=tk.W,
            padx=(0, 6),
        )
        surface_search = tk.StringVar()
        surface_search_entry = ttk.Entry(
            toolbar,
            textvariable=surface_search,
            width=30,
        )
        surface_search_entry.grid(row=0, column=1, sticky=tk.W)
        ttk.Label(
            toolbar,
            text=f"physics.lsp SHA-256: {fingerprint}",
            style="Muted.TLabel",
        ).grid(row=0, column=2, sticky=tk.E)
        list_frame = ttk.Frame(tab)
        list_frame.grid(row=1, column=0, sticky=tk.NSEW, pady=(0, 6))
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        tree = ttk.Treeview(
            list_frame,
            columns=(
                "groundType",
                "groundDepth",
                "id",
                "name",
                "Type",
                "bendable",
                "hard",
                "valid",
                *(field for field in SURFACE_COEFFICIENTS if field != "Type"),
            ),
            show="headings",
            selectmode="browse",
        )
        source_columns = (
            ("groundType", "BeamNG groundmodel", 140),
            ("groundDepth", "Depth", 62),
            ("id", "RBR ID", 42),
            ("name", "RBR surface", 170),
            ("Type", "Type", 42),
            ("bendable", "Bendable", 55),
            ("hard", "Hard", 41),
            ("valid", "Valid", 41),
            *(
                (
                    field,
                    field,
                    42
                    if field in {"eHigh", "eLow", "Type"}
                    else 59
                    if field
                    in {
                        "CollMyHigh",
                        "CollMyLow",
                        "SoilDensity",
                        "SinkFactor",
                        "myKinetic",
                        "myStatic",
                    }
                    else 100
                    if field == "SolidGroundOffset"
                    else 84,
                )
                for field in SURFACE_COEFFICIENTS
                if field != "Type"
            ),
        )
        sort_column = "id"
        sort_reverse = False
        column_indexes = {
            column: index
            for index, (column, _label, _width) in enumerate(source_columns)
        }
        for column, label, width in source_columns:
            anchor = tk.W if column in {"name", "groundType"} else tk.CENTER
            tree.heading(
                column,
                text=f"{label} ↑" if column == sort_column else label,
                anchor=anchor,
                command=lambda selected_column=column: sort_surface_rows(
                    selected_column
                ),
            )
            tree.column(column, width=width, stretch=False, anchor=anchor)
        tree.tag_configure("localOverride", foreground="#ffb74d")
        tree.tag_configure("savedOverride", foreground=_DARK_LINK)

        def number_display(value: float | None) -> float | int | str:
            if value is None:
                return ""
            if value.is_integer():
                return int(value)
            return value

        def coefficient_display(surface, field: str) -> float | int | str:
            return number_display(surface.coefficients.get(field))

        def surface_row_values(surface_id: int) -> tuple[object, ...]:
            surface = surfaces[surface_id]
            return (
                profiles[surface_id]["groundType"],
                number_display(profiles[surface_id]["groundDepth"]),
                surface_id,
                surface.name,
                coefficient_display(surface, "Type"),
                "✓" if "+BENDABLE" in surface.flags else "",
                "✓" if "+HARD" in surface.flags and "-HARD" not in surface.flags else "",
                "✓" if "+VALID" in surface.flags else "",
                *(
                    coefficient_display(surface, field)
                    for field in SURFACE_COEFFICIENTS
                    if field != "Type"
                ),
            )
        tree.grid(row=0, column=0, sticky=tk.NSEW)
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=tree.yview)
        scrollbar.grid(
            row=0, column=1, sticky=tk.NS
        )
        tree.configure(yscrollcommand=scrollbar.set)
        horizontal_scrollbar = ttk.Scrollbar(
            list_frame,
            orient=tk.HORIZONTAL,
            command=tree.xview,
        )
        horizontal_scrollbar.grid(row=1, column=0, sticky=tk.EW)
        tree.configure(xscrollcommand=horizontal_scrollbar.set)
        ground_type = tk.StringVar()
        depth = tk.StringVar()
        ground_type_editor = ttk.Combobox(
            list_frame,
            textvariable=ground_type,
            values=tuple(sorted(GROUND_TYPES)),
            state="readonly",
        )
        depth_editor = ttk.Entry(list_frame, textvariable=depth)

        selected_id: int | None = None
        dirty_ids: set[int] = set()
        setting_values = False
        default_button: ttk.Button | None = None
        default_all_button: ttk.Button | None = None

        def differs_from_default(surface_id: int) -> bool:
            if surface_id != selected_id or surface_id not in dirty_ids:
                return surface_id in customized_ids
            default_profile = default_profiles[surface_id]
            try:
                edited_depth = float(depth.get())
            except ValueError:
                return True
            return (
                ground_type.get() != default_profile["groundType"]
                or edited_depth != default_profile["groundDepth"]
            )

        def update_row_tag(surface_id: int) -> None:
            tag = (
                "localOverride"
                if surface_id in dirty_ids
                else "savedOverride"
                if surface_id in customized_ids
                else ""
            )
            tree.item(str(surface_id), tags=(tag,) if tag else ())
            if default_button is not None and surface_id == selected_id:
                default_button.configure(
                    state=(
                        tk.NORMAL
                        if differs_from_default(surface_id)
                        else tk.DISABLED
                    )
                )
            if default_all_button is not None:
                default_all_button.configure(
                    state=(
                        tk.NORMAL
                        if any(
                            differs_from_default(changed_id)
                            for changed_id in customized_ids | dirty_ids
                        )
                        else tk.DISABLED
                    )
                )

        def show_surface(_event=None) -> None:
            nonlocal selected_id, setting_values
            selected = tree.selection()
            if not selected:
                return
            new_selected_id = int(selected[0])
            if selected_id is not None and new_selected_id != selected_id:
                if not apply_profile():
                    tree.selection_set(str(selected_id))
                    return
            selected_id = new_selected_id
            profile = profiles[selected_id]
            setting_values = True
            ground_type.set(str(profile["groundType"]))
            depth.set(str(number_display(profile["groundDepth"])))
            setting_values = False
            position_editors()
            update_row_tag(selected_id)

        def apply_profile() -> bool:
            if selected_id is None:
                return False
            try:
                ground_depth = float(depth.get())
            except ValueError:
                messagebox.showerror(_title("Surface physics"), "Ground depth must be a number.", parent=window)
                return False
            if not math.isfinite(ground_depth) or ground_depth < 0.0:
                messagebox.showerror(_title("Surface physics"), "Ground depth must be finite and non-negative.", parent=window)
                return False
            profile = dict(profiles[selected_id])
            target_ground_type = ground_type.get()
            default_profile = default_profiles[selected_id]
            if target_ground_type != profile["groundType"]:
                flags = surfaces[selected_id].flags
                source_hard = "+HARD" in flags and "-HARD" not in flags
                source_bendable = "+BENDABLE" in flags
                profile.update(
                    {
                        "hard": source_hard,
                        "bendable": source_bendable,
                        "collisionEligible": source_hard or source_bendable,
                        "snowbank": False,
                    }
                )
                if target_ground_type == default_profile["groundType"]:
                    profile.update(
                        {
                            field: default_profile[field]
                            for field in ("hard", "bendable", "collisionEligible", "snowbank")
                        }
                    )
                elif target_ground_type == "SNOWBANK":
                    profile.update(
                        {
                            "hard": False,
                            "bendable": True,
                            "collisionEligible": True,
                            "snowbank": True,
                        }
                    )
                elif target_ground_type in {"VOID", "WATER"}:
                    profile.update(
                        {
                            "hard": False,
                            "bendable": False,
                            "collisionEligible": False,
                        }
                    )
            profile.update(
                {
                    "groundType": target_ground_type,
                    "groundDepth": ground_depth,
                }
            )
            profile["water"] = target_ground_type == "WATER"
            profiles[selected_id] = profile
            tree.set(selected_id, "groundType", target_ground_type)
            tree.set(selected_id, "groundDepth", ground_depth)
            if profile != default_profile:
                customized_ids.add(selected_id)
            else:
                customized_ids.discard(selected_id)
            update_row_tag(selected_id)
            return True

        def position_editors(_event=None) -> None:
            if selected_id is None:
                ground_type_editor.place_forget()
                depth_editor.place_forget()
                return
            ground_bbox = tree.bbox(str(selected_id), "groundType")
            depth_bbox = tree.bbox(str(selected_id), "groundDepth")
            if ground_bbox:
                ground_type_editor.place(
                    x=ground_bbox[0],
                    y=ground_bbox[1],
                    width=ground_bbox[2],
                    height=ground_bbox[3],
                )
            else:
                ground_type_editor.place_forget()
            if depth_bbox:
                depth_editor.place(
                    x=depth_bbox[0],
                    y=depth_bbox[1],
                    width=depth_bbox[2],
                    height=depth_bbox[3],
                )
            else:
                depth_editor.place_forget()

        def save_profiles() -> bool:
            if selected_id is not None and not apply_profile():
                return False
            rules_data = surface_rules_data(SurfaceRules(unknown_type.get()))
            overrides = {
                f"override_{surfaces[surface_id].name}": {
                    **profile,
                    "sourceNames": [surfaces[surface_id].name],
                }
                for surface_id, profile in sorted(profiles.items())
                if profile != default_profiles[surface_id]
            }
            data = {
                "version": 2,
                **({"rules": rules_data} if rules_data else {}),
                "profiles": overrides,
            }
            try:
                if not overrides and not rules_data:
                    self.filesystem.unlink(
                        surface_profile_override_path(),
                        missing_ok=True,
                    )
                else:
                    self.filesystem.mkdir(
                        surface_profile_override_path().parent,
                        parents=True,
                        exist_ok=True,
                    )
                    self.filesystem.write_text(
                        surface_profile_override_path(),
                        json.dumps(data, indent=2) + "\n",
                        encoding="utf-8",
                    )
            except OSError as exc:
                messagebox.showerror(_title("Surface physics"), str(exc), parent=window)
                return False
            return True

        def restore_default() -> None:
            if selected_id is None:
                return
            profiles[selected_id] = dict(default_profiles[selected_id])
            customized_ids.discard(selected_id)
            dirty_ids.discard(selected_id)
            tree.set(selected_id, "groundType", profiles[selected_id]["groundType"])
            tree.set(
                selected_id,
                "groundDepth",
                number_display(profiles[selected_id]["groundDepth"]),
            )
            update_row_tag(selected_id)
            show_surface()

        def restore_all_defaults() -> None:
            nonlocal setting_values
            for surface_id, profile in default_profiles.items():
                profiles[surface_id] = dict(profile)
            customized_ids.clear()
            dirty_ids.clear()
            if selected_id is not None:
                profile = profiles[selected_id]
                setting_values = True
                ground_type.set(str(profile["groundType"]))
                depth.set(str(number_display(profile["groundDepth"])))
                setting_values = False
            refresh_surface_rows()

        def mark_dirty(*_args) -> None:
            if selected_id is None or setting_values:
                return
            dirty_ids.add(selected_id)
            update_row_tag(selected_id)

        def sort_surface_rows(column: str) -> None:
            nonlocal sort_column, sort_reverse
            sort_reverse = not sort_reverse if column == sort_column else False
            sort_column = column
            for heading_column, label, _width in source_columns:
                direction = (
                    " ↓"
                    if sort_reverse
                    else " ↑"
                    if heading_column == sort_column
                    else ""
                )
                tree.heading(heading_column, text=f"{label}{direction}")
            refresh_surface_rows()

        def refresh_surface_rows(*_args) -> None:
            nonlocal selected_id
            if selected_id is not None and not apply_profile():
                return
            selected_id = None
            children = tree.get_children()
            if children:
                tree.delete(*children)
            query = surface_search.get().strip().casefold()
            visible_surface_ids = [
                surface_id
                for surface_id in sorted(surfaces)
                if not query
                or any(
                    query in str(value).casefold()
                    for value in surface_row_values(surface_id)
                )
            ]
            column_index = column_indexes[sort_column]

            def sort_key(surface_id: int) -> tuple[bool, float | str]:
                value = surface_row_values(surface_id)[column_index]
                if isinstance(value, str):
                    return value == "", value.casefold()
                return False, value

            visible_surface_ids.sort(key=sort_key, reverse=sort_reverse)
            for surface_id in visible_surface_ids:
                tag = (
                    "localOverride"
                    if surface_id in dirty_ids
                    else "savedOverride"
                    if surface_id in customized_ids
                    else ""
                )
                tree.insert(
                    "",
                    tk.END,
                    iid=str(surface_id),
                    values=surface_row_values(surface_id),
                    tags=(tag,) if tag else (),
                )
            if visible_surface_ids:
                tree.selection_set(str(visible_surface_ids[0]))
            else:
                position_editors()

        ground_type.trace_add("write", mark_dirty)
        depth.trace_add("write", mark_dirty)
        surface_search.trace_add("write", refresh_surface_rows)
        tree.bind("<<TreeviewSelect>>", show_surface)
        tree.bind("<Configure>", position_editors)
        tree.bind(
            "<MouseWheel>",
            lambda _event: window.after_idle(position_editors),
        )

        def scroll_tree(scroll_command, *args) -> None:
            scroll_command(*args)
            window.after_idle(position_editors)

        scrollbar.configure(
            command=lambda *args: scroll_tree(tree.yview, *args)
        )
        horizontal_scrollbar.configure(
            command=lambda *args: scroll_tree(tree.xview, *args)
        )
        buttons = ttk.Frame(tab)
        buttons.grid(row=2, column=0, sticky=tk.W)
        default_button = ttk.Button(
            buttons,
            text="Restore default",
            command=restore_default,
            state=tk.DISABLED,
        )
        default_button.grid(
            row=0,
            column=0,
        )
        default_all_button = ttk.Button(
            buttons,
            text="Restore all defaults",
            command=restore_all_defaults,
            state=tk.DISABLED,
        )
        default_all_button.grid(
            row=0,
            column=1,
            padx=(6, 0),
        )

        rules_frame = tk.Frame(
            tab,
            background=_DARK_BACKGROUND,
            highlightbackground=_DARK_BORDER,
            highlightthickness=1,
            padx=8,
            pady=5,
        )
        rules_frame.grid(row=3, column=0, sticky=tk.EW, pady=(10, 0))
        rules_frame.columnconfigure(4, weight=1)
        solid_ground_types = tuple(sorted(SOLID_GROUND_TYPES))
        unknown_type = tk.StringVar(value=rules.unknown_ground_type)
        ttk.Label(
            rules_frame,
            text="Undefined surface IDs and surfaces of unknown Type use:",
        ).grid(row=0, column=0, sticky=tk.W, padx=(0, 6))
        ttk.Combobox(
            rules_frame,
            textvariable=unknown_type,
            values=solid_ground_types,
            state="readonly",
            width=24,
        ).grid(row=0, column=1, sticky=tk.W)

        def restore_rules() -> None:
            unknown_type.set(DEFAULT_SURFACE_RULES.unknown_ground_type)

        restore_rules_button = ttk.Button(
            rules_frame,
            text="Restore default",
            command=restore_rules,
        )
        restore_rules_button.grid(row=0, column=5, sticky=tk.E)

        def update_restore_rules_button(*_args) -> None:
            modified = unknown_type.get() != DEFAULT_SURFACE_RULES.unknown_ground_type
            restore_rules_button.configure(
                state=tk.NORMAL if modified else tk.DISABLED
            )

        unknown_type.trace_add("write", update_restore_rules_button)
        update_restore_rules_button()
        refresh_surface_rows()
        return save_profiles

    def _show_options(self) -> None:
        window = tk.Toplevel(self.root)
        window.withdraw()
        window.configure(background=_DARK_BACKGROUND)
        window.title(_title("Options"))
        window.transient(self.root)
        self._bind_escape_to_close(window)
        frame = ttk.Frame(window, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.columnconfigure(0, weight=1)
        frame.rowconfigure(0, weight=1)
        notebook = ttk.Notebook(frame)
        notebook.grid(row=0, column=0, sticky=tk.NSEW)
        copies: list[tuple[tk.BooleanVar, tk.BooleanVar]] = []

        def copy_of(variable: tk.BooleanVar) -> tk.BooleanVar:
            copy = tk.BooleanVar(master=window, value=variable.get())
            copies.append((variable, copy))
            return copy

        general_tab = ttk.Frame(notebook, padding=10)
        notebook.add(general_tab, text="General")
        for column in range(3):
            general_tab.columnconfigure(column, weight=1, uniform="option_panels")
        general_tab.rowconfigure(0, weight=1)
        general_tab.rowconfigure(1, weight=3)

        def option_panel(row: int, column: int, rowspan: int = 1) -> tk.Frame:
            panel = tk.Frame(
                general_tab,
                background=_DARK_BACKGROUND,
                highlightbackground=_DARK_BORDER,
                highlightthickness=1,
                padx=8,
                pady=5,
            )
            panel.grid(
                row=row,
                column=column,
                rowspan=rowspan,
                sticky=tk.NSEW,
                padx=(0, 10) if column < 2 else 0,
                pady=(0, 10) if row + rowspan < 2 else 0,
            )
            return panel

        general_panel = option_panel(0, 0, rowspan=2)
        for row, (text, variable) in enumerate((
            (
                "Use thicker collisions automatically (to avoid vehicle-eating objects)",
                self.inflate_thin_walls,
            ),
            (
                "Use Original RBR invisible walls (instead of falling out of bounds)",
                self.use_map_border_brake_walls,
            ),
            (
                "Use LOD (level of detail) meshes (performance may decrease, "
                "in-game overview map may break)",
                self.use_visual_lods,
            ),
        )):
            ttk.Checkbutton(
                general_panel,
                text=text,
                variable=copy_of(variable),
                style="Wrap.TCheckbutton",
            ).grid(row=row, column=0, sticky=tk.W, pady=(0 if row == 0 else 4, 0))

        def name_list(
            row: int,
            column: int,
            check_text: str,
            variable: tk.BooleanVar,
            hint: str,
            *lists: tuple[
                str | None,
                tk.BooleanVar | None,
                tuple[str, ...] | None,
                tuple[str, ...] | None,
            ],
        ) -> list[tk.Text]:
            panel = option_panel(row, column)
            panel.columnconfigure(0, weight=1)
            header = ttk.Frame(panel)
            header.grid(row=0, column=0, columnspan=2, sticky=tk.EW)
            header.columnconfigure(0, weight=1)
            header_enabled = copy_of(variable)
            ttk.Checkbutton(
                header,
                text=check_text,
                variable=header_enabled,
                style="Header.TCheckbutton",
            ).grid(row=0, column=0, sticky=tk.W)
            ttk.Label(
                panel,
                text=hint,
                style="Muted.TLabel",
                justify=tk.LEFT,
                wraplength=340,
            ).grid(row=1, column=0, columnspan=2, sticky=tk.W, pady=(2, 4))
            def restore_button(
                parent: tk.Misc,
                entries: list[tuple[tk.Text, tuple[str, ...]]],
            ) -> ttk.Button:
                def restore() -> None:
                    for text, default in entries:
                        state = text.cget("state")
                        text.configure(state=tk.NORMAL)
                        text.delete("1.0", tk.END)
                        text.insert("1.0", "\n".join(default))
                        text.configure(state=state)

                button = ttk.Button(parent, text="Restore default", command=restore)

                def update() -> None:
                    modified = any(
                        _text_lines(text) != list(default) for text, default in entries
                    )
                    button.configure(state=tk.NORMAL if modified else tk.DISABLED)

                for text, _default in entries:
                    _on_text_change(text, update)
                update()
                return button

            texts: list[tk.Text] = []
            groups: list[tuple[tk.Misc, list[tuple[tk.Text, tuple[str, ...]]]]] = [
                (header, [])
            ]
            enabled = header_enabled
            grid_row = 2
            for label, section_variable, value, _default in lists:
                section_label: ttk.Label | None = None
                if label:
                    if section_variable is not None:
                        enabled = copy_of(section_variable)
                        section_row = ttk.Frame(panel)
                        section_row.columnconfigure(0, weight=1)
                        ttk.Checkbutton(
                            section_row,
                            text=label,
                            variable=enabled,
                            style="Header.TCheckbutton",
                        ).grid(row=0, column=0, sticky=tk.W)
                        groups.append((section_row, []))
                        section_widget = section_row
                    else:
                        section_widget = section_label = ttk.Label(panel, text=label)
                    section_widget.grid(
                        row=grid_row,
                        column=0,
                        columnspan=2,
                        sticky=tk.EW,
                        pady=(6 if texts else 0, 2),
                    )
                    grid_row += 1
                if value is None:
                    continue
                panel.rowconfigure(grid_row, weight=max(1, len(_default)))
                text = tk.Text(
                    panel,
                    width=36,
                    height=8 // sum(section[2] is not None for section in lists),
                    wrap=tk.WORD,
                    background=_DARK_INPUT,
                    foreground=_DARK_FOREGROUND,
                    insertbackground=_DARK_FOREGROUND,
                    relief=tk.FLAT,
                )
                text.grid(row=grid_row, column=0, sticky=tk.NSEW)
                scrollbar = ttk.Scrollbar(
                    panel,
                    orient=tk.VERTICAL,
                    command=text.yview,
                )
                scrollbar.grid(row=grid_row, column=1, sticky=tk.NS)
                text.configure(yscrollcommand=_auto_hiding_scrollbar_setter(scrollbar))
                text.insert("1.0", "\n".join(value))
                texts.append(text)
                grid_row += 1
                groups[-1][1].append((text, _default))

                def update_enabled(
                    *_args,
                    text=text,
                    enabled=enabled,
                    section_label=section_label,
                ) -> None:
                    active = enabled.get()
                    text.configure(
                        state=tk.NORMAL if active else tk.DISABLED,
                        foreground=_DARK_FOREGROUND if active else _DARK_MUTED,
                        background=_DARK_INPUT if active else _DARK_BACKGROUND,
                    )
                    if section_label is not None:
                        section_label.configure(
                            style="TLabel" if active else "Muted.TLabel"
                        )

                enabled.trace_add("write", update_enabled)
                update_enabled()

            for parent, entries in groups:
                if entries:
                    restore_button(parent, entries).grid(row=0, column=1, sticky=tk.E)
            return texts

        name_list(
            0,
            1,
            "Remove sky domes and cloud layers based on geometry",
            self.remove_source_skybox,
            "BeamNG's own sky and clouds are always added; old sky objects can "
            "hide them. Removes source meshes that stay at least 100 m above "
            "a tenth of the route or more.",
        )
        (
            snowbank_text,
            snowbank_name_mesh_text,
        ) = name_list(
            1,
            1,
            _snowbank_option_text(),
            self.use_snowwall_collision_override,
            "Their snow becomes soft snowbank, and Snowwall ground directly "
            "below them stays firm snow.",
            (
                "Repair snow walls without surface data, guessed from names",
                self.use_snowbank_name_fallback,
                None,
                None,
            ),
            (
                "Snowbank names (any part of the name):",
                None,
                self.snowbank_name_matches,
                DEFAULT_SNOWBANK_NAME_MATCHES,
            ),
            (
                "Only on meshes named (regular expressions, matching the start):",
                None,
                self.snowbank_name_mesh_patterns,
                DEFAULT_SNOWBANK_NAME_MESH_PATTERNS,
            ),
        )
        foliage_text, foliage_ground_text = name_list(
            1,
            2,
            "Apply subsurface scattering based on name",
            self.use_foliage_name_fallback,
            "Names only apply to see-through objects the stage doesn't mark "
            "as vegetation.",
            (
                "Material and texture names (whole words):",
                None,
                self.foliage_name_matches,
                DEFAULT_FOLIAGE_NAME_MATCHES,
            ),
            (
                "Apply subsurface scattering based on surface type",
                self.use_foliage_ground_types,
                self.foliage_ground_types,
                DEFAULT_FOLIAGE_GROUND_TYPES,
            ),
        )
        [water_text] = name_list(
            0,
            2,
            "Convert to water/river/ocean objects based on geometry and name",
            self.use_water_name_fallback,
            "Only for objects the stage doesn't mark as water. One word start "
            "per line; lines starting with ! reject any name containing that text.",
            (None, None, self.water_name_matches, DEFAULT_WATER_NAME_MATCHES),
        )

        surface_tab = ttk.Frame(notebook, padding=10)
        notebook.add(surface_tab, text="Surface physics")
        save_surface_profiles = self._build_surface_profiles_tab(
            surface_tab,
            window,
        )
        pacenotes_tab = ttk.Frame(notebook, padding=10)
        notebook.add(pacenotes_tab, text="Pacenotes")
        save_pacenotes = self._build_pacenotes_tab(pacenotes_tab, window)

        def save() -> None:
            try:
                snowbank_name_mesh_patterns = _regex_patterns(
                    snowbank_name_mesh_text.get("1.0", tk.END)
                )
                foliage_ground_types = _ground_types(
                    foliage_ground_text.get("1.0", tk.END)
                )
            except (re.error, ValueError) as exc:
                notebook.select(general_tab)
                messagebox.showerror(
                    _title("Invalid name matching rule"),
                    str(exc),
                    parent=window,
                )
                return
            if save_surface_profiles is not None and not save_surface_profiles():
                notebook.select(surface_tab)
                return
            if not save_pacenotes():
                notebook.select(pacenotes_tab)
                return
            for variable, copy in copies:
                variable.set(copy.get())
            self.water_name_matches = _word_start_matches(
                water_text.get("1.0", tk.END)
            )
            self.snowbank_name_matches = _snowbank_name_matches(
                snowbank_text.get("1.0", tk.END)
            )
            self.snowbank_name_mesh_patterns = snowbank_name_mesh_patterns
            self.foliage_name_matches = _foliage_name_matches(
                foliage_text.get("1.0", tk.END)
            )
            self.foliage_ground_types = foliage_ground_types
            self._save_paths()
            window.destroy()

        footer = ttk.Frame(frame)
        footer.grid(row=1, column=0, sticky=tk.EW, pady=(10, 0))
        footer.columnconfigure(0, weight=1)
        ttk.Button(footer, text="OK", command=save, width=-10).grid(
            row=0,
            column=1,
        )
        ttk.Button(footer, text="Cancel", command=window.destroy, width=-10).grid(
            row=0,
            column=2,
            padx=(6, 0),
        )
        window.update_idletasks()
        width = 1280
        height = 780
        x, y = _centered_position(
            width,
            height,
            self.root.winfo_rootx(),
            self.root.winfo_rooty(),
            self.root.winfo_rootx() + self.root.winfo_width(),
            self.root.winfo_rooty() + self.root.winfo_height(),
        )
        window.geometry(f"{width}x{height}+{x}+{y}")
        window.minsize(900, 620)
        window.deiconify()
        window.grab_set()

    def _on_beamng_mods_dir_changed(self, *_args) -> None:
        self._rebuild_filesystem()
        self._update_convert_state()
        self._refresh_installed_markers()

    def _browse_rbr(self) -> None:
        path = filedialog.askdirectory(title=_title("Select the RBR installation"), initialdir=self.rbr_path.get() or None)
        if path:
            self.rbr_path.set(str(Path(path).resolve()))
            self._save_paths()
            self._refresh_stages()

    def _browse_beamng_mods_dir(self) -> None:
        path = filedialog.askdirectory(
            title=_title("Select the BeamNG mods folder"),
            initialdir=self.beamng_mods_dir.get() or None,
        )
        if path:
            self.beamng_mods_dir.set(str(Path(path).resolve()))
            self._save_paths()
            self._update_convert_state()

    def _install_companion_mod(self, mod: CompanionMod) -> None:
        title = f"'RBR {mod.title}' mod"
        mods_dir = self.beamng_mods_dir.get().strip()
        if not mods_dir:
            messagebox.showerror(
                _title(title),
                "Select the BeamNG mods folder first.",
                parent=self.root,
            )
            return
        try:
            destination = Path(mods_dir) / mod.filename
            mod.install(destination, self.filesystem)
        except (OSError, SandboxViolationError, RuntimeError) as exc:
            messagebox.showerror(
                _title(title),
                f"Could not install the {title}.\n\n{exc}",
                parent=self.root,
            )
            return
        messagebox.showinfo(
            _title(f"{title} installed"),
            f"Installed {destination.name} in:\n{destination.parent}",
            parent=self.root,
        )

    def _refresh_stages(self) -> None:
        self._cancel_stage_preload()
        self.inspected_original_stage_keys.clear()
        self.pending_current_conversion_key = None
        self.pending_batch_conversion_keys.clear()
        try:
            with use_filesystem(self.filesystem):
                self.stages = discover_stages(
                    Path(self.rbr_path.get()),
                    inspect_original_variants=False,
                )
        except (ConversionError, OSError) as exc:
            self.stages = []
            self.progress.set(value=0, text=str(exc), tone="normal")
            self._populate_tree()
            return
        if not self.conversion_active:
            self.progress.set(value=0, text="", tone="normal")
        self._populate_tree()
        self._update_stage_preload_progress()
        self._schedule_stage_preload()

    def _cancel_stage_preload(self) -> None:
        self.stage_preload_generation += 1
        self.stage_preload_processed_keys.clear()
        self.stage_preload_cancel.set()
        self.stage_preload_cancel = threading.Event()
        self.stage_preload_requests = queue.Queue()
        self._stage_preload_active_generation = None

    def _schedule_stage_preload(self) -> None:
        generation = self.stage_preload_generation
        if self.stages:
            self._stage_preload_active_generation = generation
            self._update_convert_state()
        self.root.after(
            100,
            lambda: self._start_stage_preload(generation),
        )

    def _start_stage_preload(self, generation: int) -> None:
        if (
            generation != self.stage_preload_generation
            or not self.rbr_path.get().strip()
            or not self.stages
        ):
            if generation == self._stage_preload_active_generation:
                self._stage_preload_active_generation = None
                self._update_convert_state()
            return
        rbr_root = Path(self.rbr_path.get()).expanduser()
        completed_keys = {
            stage.source_key
            for stage in self.stages
            if (
                stage.source_format == "original"
                and self._stage_preload_ready(stage)
            )
        }
        self.stage_preload_processed_keys.update(completed_keys)
        thread = threading.Thread(
            target=self._preload_stage_details,
            args=(
                generation,
                rbr_root,
                tuple(self.stages),
                completed_keys,
                self.stage_preload_requests,
                self.stage_preload_cancel,
            ),
            daemon=True,
        )
        self.stage_preload_thread = thread
        thread.start()

    def _preload_stage_details(
        self,
        generation: int,
        rbr_root: Path,
        stages,
        completed_keys: set[str],
        requests: queue.Queue[str],
        cancelled: threading.Event,
    ) -> None:
        try:
            with self.stage_preload_worker_lock:
                if cancelled.is_set():
                    return
                self._run_stage_preload(
                    generation,
                    rbr_root,
                    stages,
                    completed_keys,
                    requests,
                    cancelled,
                )
        finally:
            self.events.put(("stage_preload_finished", generation))

    def _run_stage_preload(
        self,
        generation: int,
        rbr_root: Path,
        stages,
        completed_keys: set[str],
        requests: queue.Queue[str],
        cancelled: threading.Event,
    ) -> None:
        stages_by_key = {
            stage.source_key: stage
            for stage in stages
        }
        pending_keys: list[str] = []
        stage_keys = iter(stages_by_key)
        filesystem = create_runtime_filesystem(
            rbr_root=rbr_root,
            include_runtime_write_roots=False,
        )
        with use_filesystem(filesystem):
            while not cancelled.is_set():
                try:
                    while True:
                        key = requests.get_nowait()
                        if key not in completed_keys and key not in pending_keys:
                            pending_keys.append(key)
                except queue.Empty:
                    pass
                if pending_keys:
                    key = pending_keys.pop(0)
                else:
                    key = next(
                        (
                            candidate
                            for candidate in stage_keys
                            if candidate not in completed_keys
                        ),
                        None,
                    )
                if key is None:
                    return
                stage = stages_by_key.get(key)
                if stage is None:
                    continue
                if stage.source_format != "original":
                    self.events.put(
                        (
                            "stage_preload_processed",
                            (generation, stage.source_key),
                        )
                    )
                    completed_keys.add(key)
                    continue
                try:
                    inspected_stage = inspect_original_stage(
                        rbr_root,
                        stage.source_key,
                    )
                except (ConversionError, OSError) as exc:
                    inspected_stage = replace(
                        stage,
                        valid=False,
                        issues=(str(exc),),
                    )
                self.events.put(
                    (
                        "original_stage_preloaded",
                        (generation, inspected_stage),
                    )
                )
                completed_keys.add(key)

    def _stage_preload_ready(self, stage) -> bool:
        if not stage:
            return True
        return (
            stage.source_format != "original"
            or stage.source_key in self.inspected_original_stage_keys
        )

    def _update_stage_preload_progress(self) -> None:
        stages = getattr(self, "stages", ())
        if (
            getattr(self, "conversion_active", False)
            or getattr(self, "pacenote_preview_active", False)
            or not stages
        ):
            return
        total = len(stages)
        processed = sum(
            stage.source_key in self.stage_preload_processed_keys
            for stage in stages
        )
        self.progress.set(
            value=processed / total * 100,
            text=f"Inspecting stages... ({processed}/{total})",
            tone="normal",
        )

    def _finish_stage_preload_progress(self, generation: int) -> None:
        if (
            generation != self.stage_preload_generation
            or generation != self._stage_preload_active_generation
        ):
            return
        self._stage_preload_active_generation = None
        if not self.conversion_active and not self.pacenote_preview_active:
            self.progress.set(
                value=0,
                text=f"Read {len(self.stages)} stages",
                tone="normal",
            )
        self._update_convert_state()

    def _request_stage_preload(self, stage) -> bool:
        if self._stage_preload_ready(stage):
            return True
        if stage:
            self.stage_preload_requests.put(stage.source_key)
        return False

    def _handle_original_stage_preloaded(self, payload) -> None:
        generation, inspected_stage = payload
        if generation != self.stage_preload_generation:
            return
        self.inspected_original_stage_keys.add(inspected_stage.source_key)
        self.stage_preload_processed_keys.add(inspected_stage.source_key)
        self._replace_stage(inspected_stage)
        selected_stage = self._selected_stage()
        if (
            selected_stage
            and selected_stage.source_key == inspected_stage.source_key
        ):
            self.original_variant_stage_key = None
            self._refresh_selected_stage()
        self._update_stage_preload_progress()
        self._resume_pending_stage_actions()

    def _handle_stage_preload_processed(self, payload) -> None:
        generation, source_key = payload
        if generation != self.stage_preload_generation:
            return
        self.stage_preload_processed_keys.add(source_key)
        self._update_stage_preload_progress()

    def _stage_tree_values(self, stage) -> list[str]:
        values = [
            _checkbox_text(stage.metadata.folder_name in self.selected_stage_keys)
        ]
        values.extend(_stage_display_values(stage))
        if _stage_is_installed(
            self.beamng_mods_dir.get(),
            stage.metadata.folder_name,
            self.filesystem,
            display_name=stage.metadata.name,
            source_format=stage.source_format,
        ):
            values[5] = f"✓ {values[5]}"
        return values

    def _replace_stage(self, inspected_stage) -> None:
        self.stages = [
            inspected_stage
            if stage.source_key == inspected_stage.source_key
            else stage
            for stage in self.stages
        ]
        item = inspected_stage.metadata.folder_name
        if self.tree.exists(item):
            self.tree.item(
                item,
                values=self._stage_tree_values(inspected_stage),
                tags=_stage_tree_tags(inspected_stage),
            )

    def _populate_tree(self) -> None:
        selected = self._selected_folder()
        for marker in self.surface_markers.values():
            marker.destroy()
        self.surface_markers.clear()
        self.tree.delete(*self.tree.get_children())
        for stage in self.stages:
            if not _stage_matches_search(stage, self.stage_search.get()):
                continue
            item = self.tree.insert(
                "",
                tk.END,
                iid=stage.metadata.folder_name,
                values=self._stage_tree_values(stage),
                tags=_stage_tree_tags(stage),
            )
            self._create_surface_marker(item, stage.metadata.physics)
            if selected == stage.metadata.folder_name:
                self.tree.selection_set(item)
        if self.sort_column:
            self._apply_tree_sort()
        if not self.tree.selection():
            items = self.tree.get_children()
            if items:
                self.tree.selection_set(items[0])
                self.tree.focus(items[0])
        self._refresh_selected_stage()
        self._update_select_heading()

    def _focus_stage_search(self, _event=None) -> str:
        self.stage_search_entry.focus_set()
        self.stage_search_entry.selection_range(0, tk.END)
        self.stage_search_entry.icursor(tk.END)
        return "break"

    def _clear_stage_search(self, _event=None) -> str:
        self.stage_search.set("")
        self.tree.focus_set()
        return "break"

    def _navigate_stage_list(self, event) -> str:
        items = self.tree.get_children()
        if not items:
            return "break"
        selected = self._selected_folder() or self.tree.focus()
        try:
            index = items.index(selected)
        except ValueError:
            index = 0
        if event.keysym == "Home":
            index = 0
        elif event.keysym == "End":
            index = len(items) - 1
        else:
            bounds = self.tree.bbox(items[index])
            row_height = bounds[3] if bounds else 22
            page_size = max(1, self.tree.winfo_height() // row_height - 1)
            offset = page_size if event.keysym == "Next" else -page_size
            index = min(len(items) - 1, max(0, index + offset))
        item = items[index]
        self.tree.selection_set(item)
        self.tree.focus(item)
        self.tree.see(item)
        self._schedule_surface_marker_refresh()
        return "break"

    def _on_tree_click(self, event) -> str | None:
        if (
            self.tree.identify_region(event.x, event.y) != "cell"
            or self.tree.identify_column(event.x) != "#1"
        ):
            return None
        item = self.tree.identify_row(event.y)
        if item:
            self._toggle_stage_selection(item)
        return "break"

    def _toggle_stage_selection(self, item: str) -> None:
        if item in self.selected_stage_keys:
            self.selected_stage_keys.remove(item)
        else:
            self.selected_stage_keys.add(item)
        if self.tree.exists(item):
            self.tree.set(item, "selected", _checkbox_text(item in self.selected_stage_keys))
        self._update_select_heading()
        self._update_convert_state()

    def _toggle_all_stages(self) -> None:
        items = self.tree.get_children()
        select_all = not items or not all(
            item in self.selected_stage_keys for item in items
        )
        for item in items:
            if select_all:
                self.selected_stage_keys.add(item)
            else:
                self.selected_stage_keys.discard(item)
            self.tree.set(item, "selected", _checkbox_text(select_all))
        self._update_select_heading()
        self._update_convert_state()

    def _update_select_heading(self) -> None:
        items = self.tree.get_children()
        selected = bool(items) and all(
            item in self.selected_stage_keys for item in items
        )
        self.tree.heading("selected", text=_checkbox_text(selected))

    def _refresh_selected_stage(self) -> None:
        self._schedule_surface_marker_refresh()
        self._update_stage_details()
        self._update_original_variants()
        self._update_environment_defaults()
        self._update_convert_state()

    def _create_surface_marker(self, item: str, surface: str) -> None:
        marker = tk.Frame(
            self.tree,
            width=10,
            height=10,
            background=_surface_color(surface),
            borderwidth=0,
            highlightthickness=0,
        )
        marker.bind(
            "<Button-1>",
            lambda _event, selected=item: self._select_surface_marker(selected),
        )
        marker.bind(
            "<Double-1>",
            lambda _event, selected=item: self._activate_surface_marker(selected),
        )
        self.surface_markers[item] = marker

    def _select_surface_marker(self, item: str) -> str:
        if self.tree.exists(item):
            self.tree.selection_set(item)
            self.tree.focus(item)
            self._update_convert_state()
        return "break"

    def _activate_surface_marker(self, item: str) -> str:
        self._select_surface_marker(item)
        self._request_stage_preload(self._selected_stage())
        self._refresh_selected_stage()
        self._start_conversion()
        return "break"

    def _schedule_surface_marker_refresh(self) -> None:
        self.root.after_idle(self._refresh_surface_markers)

    def _refresh_surface_markers(self) -> None:
        try:
            for item, marker in self.surface_markers.items():
                bounds = self.tree.bbox(item, "physics")
                if not bounds:
                    marker.place_forget()
                    continue
                x, y, _width, height = bounds
                marker.place(x=x + 6, y=y + max(0, (height - 10) // 2), width=10, height=10)
        except tk.TclError:
            pass

    def _scroll_tree(self, *args) -> None:
        self.tree.yview(*args)
        self._schedule_surface_marker_refresh()

    def _refresh_installed_markers(self) -> None:
        mods_dir = self.beamng_mods_dir.get()
        for stage in self.stages:
            item = stage.metadata.folder_name
            if not self.tree.exists(item):
                continue
            prefix = (
                "✓ "
                if _stage_is_installed(
                    mods_dir,
                    item,
                    self.filesystem,
                    display_name=stage.metadata.name,
                    source_format=stage.source_format,
                )
                else ""
            )
            self.tree.set(item, "name", prefix + stage.metadata.name)

    def _sort_tree(self, column: str) -> None:
        if self.sort_column == column:
            self.sort_descending = not self.sort_descending
        else:
            self.sort_column = column
            self.sort_descending = False
        self._apply_tree_sort()

    def _apply_tree_sort(self) -> None:
        column = self.sort_column
        if not column:
            return
        items = list(self.tree.get_children())
        stages = {stage.metadata.folder_name: stage for stage in self.stages}
        if column == "length":
            known = [item for item in items if stages[item].metadata.length_km is not None]
            unknown = [item for item in items if stages[item].metadata.length_km is None]
            known.sort(
                key=lambda item: stages[item].metadata.length_km,
                reverse=self.sort_descending,
            )
            items = known + unknown
        elif column == "physics":
            items.sort(
                key=lambda item: stages[item].metadata.physics.casefold(),
                reverse=self.sort_descending,
            )
        elif column == "name":
            items.sort(
                key=lambda item: stages[item].metadata.name.casefold(),
                reverse=self.sort_descending,
            )
        else:
            items.sort(
                key=lambda item: self.tree.set(item, column).casefold(),
                reverse=self.sort_descending,
            )
        for index, item in enumerate(items):
            self.tree.move(item, "", index)
        for heading_column, label in self.tree_headings.items():
            arrow = " ↓" if self.sort_descending else " ↑"
            self.tree.heading(
                heading_column,
                text=label + arrow if heading_column == column else label,
            )
        self._update_select_heading()
        self._schedule_surface_marker_refresh()

    def _selected_folder(self) -> str | None:
        selection = self.tree.selection()
        return selection[0] if selection else None

    def _on_stage_selected(self, _event=None) -> None:
        self._request_stage_preload(self._selected_stage())
        self._refresh_selected_stage()

    def _on_stage_double_click(self, event) -> str | None:
        if self.tree.identify_region(event.x, event.y) != "cell":
            return None
        item = self.tree.identify_row(event.y)
        if not item:
            return None
        self.tree.selection_set(item)
        self.tree.focus(item)
        self._request_stage_preload(self._selected_stage())
        self._refresh_selected_stage()
        self._start_conversion()
        return "break"

    def _selected_stage(self):
        folder = self._selected_folder()
        return next((stage for stage in self.stages if stage.metadata.folder_name == folder), None)

    def _selected_stages(self):
        return [
            stage
            for stage in self.stages
            if stage.valid
            and stage.metadata.folder_name in self.selected_stage_keys
        ]

    def _update_stage_details(self) -> None:
        stage = self._selected_stage()
        issue = _stage_warning(stage) if stage else ""
        self.stage_issue.set(issue)
        if issue:
            self.stage_issue_label.grid()
        else:
            self.stage_issue_label.grid_remove()
        if not stage:
            self.stage_caption.set("")
            image = Image.new("RGB", (260, 146), _DARK_INPUT)
        else:
            self.stage_caption.set(_stage_detail_caption(stage))
            try:
                if not stage.metadata.splashscreen:
                    raise OSError
                with Image.open(
                    self.filesystem.read_path(stage.metadata.splashscreen)
                ) as source:
                    image = ImageOps.fit(
                        source.convert("RGB"),
                        (260, 146),
                        method=Image.Resampling.LANCZOS,
                    )
            except OSError:
                image = Image.new("RGB", (260, 146), _DARK_INPUT)
        self.stage_thumbnail = ImageTk.PhotoImage(image, master=self.root)
        self.stage_thumbnail_label.configure(image=self.stage_thumbnail)

    def _update_original_variants(self) -> None:
        stage = self._selected_stage()
        if not stage or stage.source_format != "original" or len(stage.variants) < 2:
            self.variant_frame.grid_remove()
            self.original_variant_stage_key = None
            return
        self.variant_frame.grid()
        available = set(stage.variants) - {
            primary_original_tint(stage.variants)
        }
        if stage.source_key != self.original_variant_stage_key:
            for value in self.original_variant_values.values():
                value.set(False)
            self.original_variant_stage_key = stage.source_key
        for tint, check in self.original_variant_checks.items():
            if tint not in available:
                check.grid_remove()
                self.original_variant_summaries[tint].grid_remove()
        for index, (tint, _label, _summary) in enumerate(
            (option for option in _ENVIRONMENT_OPTIONS if option[0] in available),
            1,
        ):
            check = self.original_variant_checks[tint]
            summary = self.original_variant_summaries[tint]
            check.grid(row=index, column=0, sticky=tk.W)
            summary.grid(row=index, column=1, sticky=tk.W, padx=(8, 0))
            check.configure(state=tk.NORMAL)

    def _update_environment_defaults(self) -> None:
        stage = self._selected_stage()
        stage_key = (stage.source_key or str(stage.root)) if stage else None
        if stage_key == self.environment_stage_root:
            return
        self.environment_stage_root = stage_key
        with use_filesystem(self.filesystem):
            (
                map_coordinates,
                country_coordinates,
            ) = _stage_coordinate_sources(stage)
            self.location_profiles = dict(_location_profile_options(stage))
        self.location_profile_selector.configure(
            values=tuple(self.location_profiles),
        )
        if map_coordinates is not None:
            self._use_coordinates(map_coordinates)
        elif country_coordinates is not None:
            self._use_coordinates(country_coordinates)
        else:
            self.latitude.set("")
            self.longitude.set("")
        map_altitude = _stage_map_altitude(stage)
        if map_altitude is not None:
            self._use_map_altitude(map_altitude)
        else:
            self.map_altitude.set("")
        if stage:
            self.location_profile.set(next(iter(self.location_profiles), ""))
            self._on_location_profile_selected()
            default_season = _temperature_season_for_surface(
                stage.metadata.physics
            )
            self._select_temperature_season(default_season)
            self.environment_date.set(DEFAULT_ENVIRONMENT_DATE.isoformat())
        else:
            self.location_profile.set("")
            self.selected_location = None
            self.temperature_ranges = {}
            self.temperature_season.set("")
            self._refresh_temperature_season_buttons()
            self._update_convert_state()

    def _bind_entry_preview(
        self,
        button: ttk.Button,
        entries: tuple[ttk.Entry, ...],
    ) -> None:
        button.bind(
            "<Enter>",
            lambda _event: self._show_entry_preview(button, entries),
        )
        button.bind(
            "<Leave>",
            lambda _event: self._clear_entry_preview(entries),
        )

    def _show_entry_preview(
        self,
        button: ttk.Button,
        entries: tuple[ttk.Entry, ...],
    ) -> None:
        if button.instate(("disabled",)):
            return
        self._previewed_entries = entries
        for entry in entries:
            entry.configure(style="Preview.TEntry")

    def _clear_entry_preview(self, entries: tuple[ttk.Entry, ...]) -> None:
        if self._previewed_entries != entries:
            return
        self._previewed_entries = ()
        self._update_convert_state()

    def _on_location_profile_selected(self, _event=None) -> None:
        self.selected_location = self.location_profiles.get(
            self.location_profile.get()
        )
        self.temperature_ranges = _location_temperature_ranges(
            self.selected_location
        )
        self._on_temperature_changed()

    def _select_temperature_season(self, season: str) -> None:
        temperatures = self.temperature_ranges.get(season)
        if temperatures is None:
            return
        self.temperature_night.set(f"{temperatures[0]:g}")
        self.temperature_day.set(f"{temperatures[1]:g}")
        self.temperature_season.set(season)
        self._refresh_temperature_season_buttons()

    def _on_temperature_changed(self, *_args) -> None:
        temperatures = _parse_temperatures(
            self.temperature_night.get(),
            self.temperature_day.get(),
        )
        selected = ""
        if temperatures is not None:
            selected = next(
                (
                    season
                    for season, values in self.temperature_ranges.items()
                    if temperatures == values
                ),
                "",
            )
        self.temperature_season.set(selected)
        self._refresh_temperature_season_buttons()
        self._update_convert_state()

    def _refresh_temperature_season_buttons(self) -> None:
        selected = self.temperature_season.get()
        batch_selection = self._has_batch_selection()
        enabled = not batch_selection or self.batch_override_temperatures.get()
        for season, button in self.temperature_season_buttons.items():
            temperatures = self.temperature_ranges.get(season)
            button.configure(
                text=(
                    _temperature_season_text(season, temperatures)
                    if temperatures is not None
                    else f"{season.title()}: unavailable"
                ),
                state=(
                    tk.NORMAL
                    if temperatures is not None and enabled
                    else tk.DISABLED
                ),
            )
            button.state(["selected"] if season == selected else ["!selected"])

    def _use_coordinates(
        self,
        coordinates: tuple[float, float] | None,
    ) -> None:
        if coordinates is None:
            return
        self.latitude.set(_format_coordinate(coordinates[0]))
        self.longitude.set(_format_coordinate(coordinates[1]))

    def _use_map_altitude(self, altitude: float) -> None:
        self.map_altitude.set(f"{altitude:g}")

    def _use_selected_location_coordinates(self) -> None:
        if self.selected_location is None:
            return
        self._use_coordinates(
            (self.selected_location.latitude, self.selected_location.longitude)
        )
        self._use_map_altitude(self.selected_location.altitude_meters)

    def _has_batch_selection(self) -> bool:
        return len(getattr(self, "selected_stage_keys", ())) > 1

    def _bind_escape_to_close(self, window: tk.Toplevel) -> None:
        def close(_event=None) -> str:
            window.destroy()
            return "break"

        window.bind("<Escape>", close)

    def _show_about(self) -> None:
        window = tk.Toplevel(self.root)
        window.withdraw()
        window.configure(background=_DARK_BACKGROUND)
        window.title(_title("About"))
        window.transient(self.root)
        self._bind_escape_to_close(window)
        frame = ttk.Frame(window, padding=12)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.columnconfigure(0, weight=1)
        ttk.Label(
            frame,
            text=_APP_TITLE,
            anchor=tk.CENTER,
            justify=tk.CENTER,
            font=self.stage_caption_font,
        ).grid(row=0, column=0, columnspan=2, sticky=tk.EW, pady=(0, 4))
        ttk.Label(
            frame,
            text=_PROGRAM_DESCRIPTION,
            anchor=tk.CENTER,
            justify=tk.CENTER,
            wraplength=620,
        ).grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=(0, 8))
        repository_link = tk.Label(
            frame,
            text=_REPOSITORY_URL,
            background=_DARK_BACKGROUND,
            foreground=_DARK_LINK,
            cursor="hand2",
        )
        repository_link.grid(
            row=2,
            column=0,
            columnspan=2,
            sticky=tk.EW,
            pady=(0, 16),
        )

        def open_repository(_event=None) -> None:
            try:
                if not webbrowser.open(_REPOSITORY_URL, new=2):
                    raise OSError(
                        "No application is available to open the repository"
                    )
            except (OSError, webbrowser.Error) as exc:
                messagebox.showerror(
                    _title("Unable to open repository"),
                    str(exc),
                    parent=window,
                )

        repository_link.bind("<Button-1>", open_repository)
        ttk.Label(
            frame,
            text=_STAGE_CREATOR_NOTICE_TEXT,
            anchor=tk.CENTER,
            justify=tk.CENTER,
            wraplength=620,
            style="Danger.TLabel",
        ).grid(row=3, column=0, columnspan=2, sticky=tk.EW, pady=(0, 16))
        footer = ttk.Frame(frame)
        footer.grid(row=4, column=0, columnspan=2, sticky=tk.EW)
        footer.columnconfigure(1, weight=1)
        ttk.Button(
            footer,
            text="View third party licenses",
            command=lambda: self._open_licenses_directory(window),
        ).grid(row=0, column=0, sticky=tk.W)
        ttk.Button(footer, text="Close", command=window.destroy, width=-10).grid(
            row=0,
            column=2,
        )
        window.protocol("WM_DELETE_WINDOW", window.destroy)
        window.update_idletasks()
        width = 640
        height = window.winfo_reqheight()
        x = self.root.winfo_rootx() + (self.root.winfo_width() - width) // 2
        y = self.root.winfo_rooty() + (self.root.winfo_height() - height) // 2
        window.geometry(f"{width}x{height}+{x}+{y}")
        window.deiconify()
        window.grab_set()

    def _open_licenses_directory(self, parent: tk.Toplevel) -> None:
        try:
            _open_path(_licenses_directory())
        except OSError as exc:
            messagebox.showerror(
                _title("Unable to open third party licenses"),
                str(exc),
                parent=parent,
            )

    def _build_pacenotes_tab(
        self,
        tab: ttk.Frame,
        window: tk.Toplevel,
    ) -> Callable[[], bool]:
        def configure_tree(tree: ttk.Treeview) -> None:
            tree.heading("#0", text="RBR source", anchor=tk.W)
            tree.column("#0", width=270, minwidth=180, stretch=True)
            for column, heading, width in (
                ("type", "Type", 160),
                ("status", "Status", 150),
                ("result", "BeamNG result", 360),
            ):
                tree.heading(column, text=heading, anchor=tk.W)
                tree.column(
                    column,
                    width=width,
                    minwidth=100,
                    stretch=column == "result",
                )
            tree.tag_configure("group", foreground=_DARK_LINK)
            tree.tag_configure("unsupported", foreground=_DARK_DANGER_TEXT)
            tree.tag_configure("route", foreground=_DARK_LINK)
            tree.tag_configure("pacenote", foreground=_DARK_SUCCESS_TEXT)
            tree.tag_configure("modifier", foreground=_DARK_PACENOTE_MODIFIER)
            tree.tag_configure("distance", foreground=_DARK_FOREGROUND)
            tree.tag_configure("localOverride", foreground="#ffb74d")
            tree.tag_configure("savedOverride", foreground=_DARK_LINK)

        def set_detail(
            text: tk.Text,
            source: str,
            entry_type: str,
            status: str,
            result: str,
            detail: str,
        ) -> None:
            text.configure(state=tk.NORMAL)
            text.delete("1.0", tk.END)
            text.insert(tk.END, source + "\n", "title")
            if entry_type:
                text.insert(tk.END, f"Type: {entry_type}\n")
            if status:
                text.insert(tk.END, f"Status: {status}\n")
            if result:
                text.insert(tk.END, "\n" + result + "\n")
            if detail:
                text.insert(tk.END, "\n" + detail)
            text.configure(state=tk.DISABLED)

        def make_detail_text(parent: ttk.Frame) -> tk.Text:
            text = tk.Text(
                parent,
                background=_DARK_INPUT,
                borderwidth=0,
                foreground=_DARK_FOREGROUND,
                height=8,
                width=36,
                highlightthickness=1,
                highlightbackground=_DARK_BORDER,
                insertbackground=_DARK_FOREGROUND,
                padx=10,
                pady=10,
                relief=tk.FLAT,
                wrap=tk.WORD,
            )
            text.tag_configure("title", foreground=_DARK_LINK)
            text.configure(state=tk.DISABLED)
            return text

        reference_groups = pacenote_conversion_reference()
        rbr_root = self.rbr_path.get().strip()
        notices: list[str] = []
        install_groups: tuple[PacenoteReferenceGroup, ...] = ()
        try:
            with use_filesystem(self.filesystem):
                installation = pacenote_installation_reference(
                    Path(rbr_root) if rbr_root else None
                )
        except (OSError, ValueError) as exc:
            installation = None
            notices.append(f"Unable to inspect the Pacenote Plugin configuration: {exc}")
        if installation is not None:
            notices.append(
                installation.summary
                + (
                    f" Configuration: {installation.config_path}"
                    if installation.config_path
                    else ""
                )
            )
            install_groups = tuple(
                PacenoteReferenceGroup(key, title, summary, entries)
                for key, title, summary, entries in (
                    (
                        "installRecognized",
                        "Recognized in my RBR install",
                        "Calls defined by the installed Pacenote Plugin config that have a BeamNG rule.",
                        installation.recognized,
                    ),
                    (
                        "installUnsupported",
                        "Unsupported in my RBR install",
                        "Calls defined by the installed Pacenote Plugin config without a BeamNG rule.",
                        installation.unsupported,
                    ),
                )
                if entries
            )
        try:
            with use_filesystem(self.filesystem):
                saved = load_pacenote_overrides()
        except ConversionError as exc:
            saved = {}
            notices.append(f"Ignoring saved pacenote overrides: {exc}")
        defaults = {
            entry.override_key: entry.default_value
            for group in (*reference_groups, *install_groups)
            for entry in group.entries
            if entry.override_key
        }
        values = dict(defaults)
        values.update({key: value for key, value in saved.items() if key in defaults})
        other_overrides = {
            key: value for key, value in saved.items() if key not in defaults
        }
        dirty: set[str] = set()
        views: dict[str, tuple[PacenoteReferenceGroup, ...]] = {
            "All rules": reference_groups,
            **{group.title: (group,) for group in reference_groups},
        }
        if install_groups:
            views["Calls in my RBR install"] = install_groups
            unsupported_groups = tuple(
                group for group in install_groups if group.key == "installUnsupported"
            )
            if unsupported_groups:
                views["Unsupported calls in my RBR install"] = unsupported_groups

        tab.columnconfigure(0, weight=1)
        tab.rowconfigure(2, weight=1)
        toolbar = ttk.Frame(tab)
        toolbar.grid(row=0, column=0, sticky=tk.EW, pady=(0, 6))
        toolbar.columnconfigure(4, weight=1)
        ttk.Label(toolbar, text="Search:").grid(
            row=0,
            column=0,
            sticky=tk.W,
            padx=(0, 6),
        )
        search = tk.StringVar()
        ttk.Entry(toolbar, textvariable=search, width=30).grid(
            row=0,
            column=1,
            sticky=tk.W,
        )
        ttk.Label(toolbar, text="Show:").grid(
            row=0,
            column=2,
            sticky=tk.W,
            padx=(14, 6),
        )
        view = tk.StringVar(value="All rules")
        view_selector = ttk.Combobox(
            toolbar,
            state="readonly",
            textvariable=view,
            values=tuple(views),
            width=32,
        )
        view_selector.grid(row=0, column=3, sticky=tk.W)
        count = tk.StringVar()
        ttk.Label(toolbar, textvariable=count, style="Muted.TLabel").grid(
            row=0,
            column=5,
            sticky=tk.E,
        )
        ttk.Label(
            tab,
            text="\n".join(notices),
            style="Muted.TLabel",
            justify=tk.LEFT,
            wraplength=1100,
        ).grid(row=1, column=0, sticky=tk.W, pady=(0, 6))

        panes = ttk.Panedwindow(tab, orient=tk.HORIZONTAL)
        panes.grid(row=2, column=0, sticky=tk.NSEW)
        list_frame = ttk.Frame(panes)
        list_frame.columnconfigure(0, weight=1)
        list_frame.rowconfigure(0, weight=1)
        detail_frame = ttk.Frame(panes)
        detail_frame.columnconfigure(0, weight=1)
        detail_frame.rowconfigure(1, weight=1)
        panes.add(list_frame, weight=3)
        panes.add(detail_frame, weight=1)
        tree = ttk.Treeview(
            list_frame,
            columns=("type", "status", "result"),
            show="tree headings",
            selectmode="browse",
        )
        configure_tree(tree)
        tree.grid(row=0, column=0, sticky=tk.NSEW)
        y_scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL)
        y_scrollbar.grid(row=0, column=1, sticky=tk.NS)
        x_scrollbar = ttk.Scrollbar(list_frame, orient=tk.HORIZONTAL)
        x_scrollbar.grid(row=1, column=0, sticky=tk.EW)
        tree.configure(
            xscrollcommand=_auto_hiding_scrollbar_setter(x_scrollbar),
            yscrollcommand=y_scrollbar.set,
        )

        def fit_result_column() -> None:
            first, last = (float(value) for value in tree.xview())
            columns = ("#0", *tree.cget("columns"))
            content = sum(int(tree.column(column, "width")) for column in columns)
            overflow = math.ceil(content * (1.0 - (last - first)) - 1e-6)
            width = int(tree.column("result", "width"))
            shrink = min(overflow, width - int(tree.column("result", "minwidth")))
            if shrink > 0:
                tree.column("result", width=width - shrink)

        tree.bind(
            "<Configure>",
            lambda _event: window.after_idle(fit_result_column),
            add="+",
        )
        ttk.Label(detail_frame, text="Selected rule").grid(
            row=0,
            column=0,
            sticky=tk.W,
            pady=(0, 6),
        )
        detail_text = make_detail_text(detail_frame)
        detail_text.grid(row=1, column=0, sticky=tk.NSEW)
        overview = (
            "Select a rule to edit its BeamNG result below the rule details. "
            "Changed results are saved beside settings.json and used by every "
            "conversion and pacenote preview."
        )
        if installation is not None and installation.issues:
            overview += "\n\nPlugin configuration warnings:\n" + "\n".join(
                "• " + issue for issue in installation.issues
            )
        set_detail(detail_text, "Pacenote conversion rules", "", "", "", overview)
        form = tk.Frame(
            detail_frame,
            background=_DARK_BACKGROUND,
            highlightbackground=_DARK_BORDER,
            highlightthickness=1,
            padx=8,
            pady=5,
        )
        form.grid(row=2, column=0, sticky=tk.EW, pady=(6, 0))
        form.columnconfigure(1, weight=1)
        ttk.Label(form, text="BeamNG result").grid(
            row=0,
            column=0,
            columnspan=2,
            sticky=tk.W,
            pady=(0, 4),
        )
        kind_value = tk.StringVar()
        kind_selector = ttk.Combobox(form, textvariable=kind_value, state="readonly")
        ttk.Label(form, text="Result type").grid(row=1, column=0, sticky=tk.W, padx=(0, 6))
        kind_selector.grid(row=1, column=1, sticky=tk.EW, pady=2)
        field_labels = {
            "modifier": "Modifier",
            "caution": "Caution",
            "direction": "Direction",
            "descriptor": "Descriptor",
            "length": "Ends after (m)",
            "riskIntensity": "Severity",
            "shape": "Shape",
            "lengthMode": "Length call",
        }
        field_widgets: dict[str, tuple[ttk.Label, ttk.Entry | ttk.Combobox, tk.StringVar]] = {}
        for row, (field, label) in enumerate(field_labels.items(), 2):
            variable = tk.StringVar()
            widget = (
                ttk.Entry(form, textvariable=variable, width=10)
                if field == "length"
                else ttk.Combobox(
                    form,
                    textvariable=variable,
                    state="readonly",
                    values=tuple(text for text, _value in pacenote_field_choices(field)),
                )
            )
            field_label = ttk.Label(form, text=label)
            field_label.grid(row=row, column=0, sticky=tk.W, padx=(0, 6))
            widget.grid(row=row, column=1, sticky=tk.EW, pady=2)
            field_widgets[field] = (field_label, widget, variable)
        form.grid_remove()
        field_defaults: dict[str, object] = {
            "modifier": "",
            "caution": 0,
            "direction": 1,
            "descriptor": "",
            "length": 60.0,
            "riskIntensity": 0,
            "shape": "",
            "lengthMode": "",
        }
        form_key: str | None = None
        rows: dict[str, PacenoteReferenceEntry] = {}
        group_rows: dict[str, PacenoteReferenceGroup] = {}
        items_by_key: dict[str, list[str]] = {}
        default_button: ttk.Button | None = None
        default_all_button: ttk.Button | None = None

        def customized(key: str | None) -> bool:
            return key is not None and values[key] != defaults[key]

        def row_values(entry: PacenoteReferenceEntry) -> tuple[str, str, str]:
            key = entry.override_key
            if key is not None and customized(key):
                record = values[key]
                return (
                    (
                        PACENOTE_OVERRIDE_KIND_LABELS[str(record["kind"])]
                        if key.startswith(("legacy:", "call:"))
                        else entry.entry_type
                    ),
                    "Custom",
                    pacenote_override_result(record),
                )
            return _pacenote_reference_values(entry)

        def row_tags(entry: PacenoteReferenceEntry) -> tuple[str]:
            if customized(entry.override_key):
                return (
                    "localOverride"
                    if entry.override_key in dirty
                    else "savedOverride",
                )
            return (_pacenote_reference_tag(entry.status),)

        def selected_entry() -> PacenoteReferenceEntry | None:
            selection = tree.selection()
            return rows.get(selection[0]) if selection else None

        def update_buttons() -> None:
            entry = selected_entry()
            if default_button is not None:
                default_button.configure(
                    state=(
                        tk.NORMAL
                        if entry is not None and customized(entry.override_key)
                        else tk.DISABLED
                    )
                )
            if default_all_button is not None:
                default_all_button.configure(
                    state=(
                        tk.NORMAL
                        if any(customized(key) for key in defaults)
                        else tk.DISABLED
                    )
                )

        def choice_label(field: str, value: object) -> str:
            return next(
                text for text, choice in pacenote_field_choices(field) if choice == value
            )

        def choice_value(field: str, text: str) -> object:
            return next(
                choice for label, choice in pacenote_field_choices(field) if label == text
            )

        def load_form(key: str) -> None:
            nonlocal form_key
            form_key = key
            record = values[key]
            kinds = pacenote_override_kinds(key)
            kind_selector.configure(
                values=tuple(PACENOTE_OVERRIDE_KIND_LABELS[kind] for kind in kinds)
            )
            kind_value.set(PACENOTE_OVERRIDE_KIND_LABELS[str(record["kind"])])
            fields = PACENOTE_OVERRIDE_FIELDS_BY_KIND[str(record["kind"])]
            for field, (label, widget, variable) in field_widgets.items():
                if field not in fields:
                    label.grid_remove()
                    widget.grid_remove()
                    continue
                label.grid()
                widget.grid()
                variable.set(
                    f"{float(record[field]):g}"
                    if field == "length"
                    else choice_label(field, record[field])
                )
            form.grid()

        def record_for_kind(kind: str, key: str) -> dict[str, object]:
            record: dict[str, object] = {"kind": kind}
            for field in PACENOTE_OVERRIDE_FIELDS_BY_KIND[kind]:
                source = next(
                    (
                        candidate
                        for candidate in (values[key], defaults[key])
                        if field in candidate
                    ),
                    field_defaults,
                )
                record[field] = source[field]
            if kind == "modifier" and not record["modifier"]:
                record["modifier"] = pacenote_field_choices("modifier")[1][1]
            return record

        def apply_form(_event=None) -> None:
            if form_key is None:
                return
            kind = next(
                kind
                for kind in pacenote_override_kinds(form_key)
                if PACENOTE_OVERRIDE_KIND_LABELS[kind] == kind_value.get()
            )
            if kind != values[form_key]["kind"]:
                record = record_for_kind(kind, form_key)
            else:
                record = dict(values[form_key])
                for field in PACENOTE_OVERRIDE_FIELDS_BY_KIND[kind]:
                    text = field_widgets[field][2].get()
                    if field != "length":
                        record[field] = choice_value(field, text)
                        continue
                    try:
                        length = float(text)
                    except ValueError:
                        length = math.nan
                    if math.isfinite(length) and 0 < length <= 1000:
                        record[field] = length
                if kind == "modifier" and not record["modifier"]:
                    record["modifier"] = values[form_key]["modifier"]
            values[form_key] = record
            dirty.add(form_key)
            refresh_key(form_key)
            show_row()

        def show_row(_event=None) -> None:
            selection = tree.selection()
            if not selection:
                return
            entry = rows.get(selection[0])
            if entry is None:
                group = group_rows[selection[0]]
                set_detail(detail_text, group.title, "Rule group", "", group.summary, "")
                form.grid_remove()
            else:
                key = entry.override_key
                status, result = row_values(entry)[1:]
                detail = entry.detail
                if key:
                    if customized(key):
                        detail += f"\n\nDefault: {entry.result}."
                    load_form(key)
                else:
                    detail += (
                        "\n\nThis rule has no BeamNG result to choose: it sets stage "
                        "structure or keeps source data as metadata."
                    )
                    form.grid_remove()
                set_detail(
                    detail_text,
                    entry.source,
                    entry.entry_type,
                    status,
                    result,
                    detail,
                )
            update_buttons()

        def refresh_key(key: str) -> None:
            for item in items_by_key.get(key, ()):
                entry = rows[item]
                tree.item(item, values=row_values(entry), tags=row_tags(entry))

        def refresh_rows(*_args) -> None:
            tree.delete(*tree.get_children())
            rows.clear()
            group_rows.clear()
            items_by_key.clear()
            shown = 0
            for group, entries in _filter_pacenote_reference_groups(
                views[view.get()],
                search.get(),
                "",
            ):
                group_id = f"group:{group.key}"
                group_rows[group_id] = group
                tree.insert(
                    "",
                    tk.END,
                    iid=group_id,
                    text=f"{group.title} ({len(entries)})",
                    values=("", "", group.summary),
                    open=True,
                    tags=("group",),
                )
                for index, entry in enumerate(entries):
                    item = f"{group.key}:{index}"
                    rows[item] = entry
                    if entry.override_key:
                        items_by_key.setdefault(entry.override_key, []).append(item)
                    tree.insert(
                        group_id,
                        tk.END,
                        iid=item,
                        text=entry.source,
                        values=row_values(entry),
                        tags=row_tags(entry),
                    )
                    shown += 1
            count.set(f"{shown} rule" + ("" if shown == 1 else "s") + " shown")
            form.grid_remove()
            update_buttons()

        def restore_default() -> None:
            entry = selected_entry()
            if entry is None or not entry.override_key:
                return
            values[entry.override_key] = defaults[entry.override_key]
            refresh_key(entry.override_key)
            show_row()

        def restore_all_defaults() -> None:
            values.update(defaults)
            for key in defaults:
                refresh_key(key)
            show_row()
            update_buttons()

        y_scrollbar.configure(command=tree.yview)
        x_scrollbar.configure(command=tree.xview)
        tree.bind("<<TreeviewSelect>>", show_row)
        kind_selector.bind("<<ComboboxSelected>>", apply_form)
        for field, (_label, widget, _variable) in field_widgets.items():
            if field == "length":
                widget.bind("<Return>", apply_form)
                widget.bind("<FocusOut>", apply_form)
            else:
                widget.bind("<<ComboboxSelected>>", apply_form)
        search.trace_add("write", refresh_rows)
        view_selector.bind("<<ComboboxSelected>>", refresh_rows)
        buttons = ttk.Frame(tab)
        buttons.grid(row=3, column=0, sticky=tk.W, pady=(6, 0))
        default_button = ttk.Button(
            buttons,
            text="Restore default",
            command=restore_default,
            state=tk.DISABLED,
        )
        default_button.grid(row=0, column=0)
        default_all_button = ttk.Button(
            buttons,
            text="Restore all defaults",
            command=restore_all_defaults,
        )
        default_all_button.grid(row=0, column=1, padx=(6, 0))
        refresh_rows()

        def save_overrides() -> bool:
            overrides = {
                **other_overrides,
                **{key: value for key, value in values.items() if customized(key)},
            }
            path = pacenote_override_path()
            try:
                if not overrides:
                    self.filesystem.unlink(path, missing_ok=True)
                else:
                    self.filesystem.mkdir(path.parent, parents=True, exist_ok=True)
                    self.filesystem.write_text(
                        path,
                        json.dumps(
                            {"version": 1, "overrides": dict(sorted(overrides.items()))},
                            indent=2,
                        )
                        + "\n",
                        encoding="utf-8",
                    )
            except OSError as exc:
                messagebox.showerror(_title("Pacenotes"), str(exc), parent=window)
                return False
            return True

        return save_overrides

    def _rsf_licence_documents(self) -> tuple[StageDocument, ...]:
        path = Path(self.rbr_path.get()).expanduser() / "rsf_stage_licenses.pdf"
        try:
            return (StageDocument(path, path.name),) if self.filesystem.is_file(path) else ()
        except (OSError, SandboxViolationError):
            return ()

    def _stage_document_contents(self, stage) -> dict[Path, str]:
        document_contents: dict[Path, str] = {}
        for document in stage.documents:
            suffix = document.path.suffix.casefold()
            if suffix == ".pdf":
                try:
                    document_contents[document.path] = pdf_text(
                        self.filesystem.read_bytes(document.path)
                    )
                except (OSError, PdfError):
                    pass
                continue
            if suffix not in PLAIN_TEXT_DOCUMENT_SUFFIXES:
                continue
            try:
                raw = self.filesystem.read_bytes(document.path)
            except OSError:
                document_contents[document.path] = "Unable to read this document."
                continue
            document_contents[document.path] = next(
                (
                    raw.decode(encoding)
                    for encoding in ("utf-8-sig", "cp1252", "latin-1")
                    if _can_decode(raw, encoding)
                ),
                "Unable to decode this document.",
            )
        return document_contents

    def _confirm_stage_terms(self, stages, *, batch: bool = False) -> None:
        stages = tuple(stages)
        existing = {
            stage.metadata.folder_name
            for stage in stages
            if stage.valid and self.filesystem.exists(self._destination(stage))
        }
        rsf_documents = self._rsf_licence_documents()
        displayed_stage = None
        documents: tuple[StageDocument, ...] = ()
        contact_tags: list[str] = []
        window = tk.Toplevel(self.root)
        window.withdraw()
        window.configure(background=_DARK_BACKGROUND)
        window.title(_title("Review Permissions to Convert"))
        window.transient(self.root)
        self._bind_escape_to_close(window)
        frame = ttk.Frame(window, padding=10)
        frame.pack(fill=tk.BOTH, expand=True)
        frame.columnconfigure(0, weight=1, uniform="permissions_panel")
        frame.columnconfigure(1, weight=2, uniform="permissions_panel")
        frame.rowconfigure(0, weight=1)
        action_panel = ttk.Frame(frame)
        action_panel.grid(row=0, column=0, sticky=tk.NSEW, padx=(0, 4))
        action_panel.columnconfigure(0, weight=1)
        action_panel.rowconfigure(3, weight=1)
        summary = tk.StringVar()
        ttk.Label(action_panel, textvariable=summary, justify=tk.LEFT).grid(
            row=0,
            column=0,
            sticky=tk.W,
            pady=(0, 6),
        )
        overwrite_existing = tk.BooleanVar(value=True)
        if existing:
            ttk.Checkbutton(
                action_panel,
                text="Overwrite existing conversions",
                variable=overwrite_existing,
            ).grid(row=1, column=0, sticky=tk.W, pady=(0, 6))
        stage_list_panel = ttk.Frame(action_panel)
        stage_list_panel.grid(row=2, column=0, sticky=tk.NSEW, pady=(0, 8))
        stage_list_panel.columnconfigure(0, weight=1)
        stage_list = ttk.Treeview(
            stage_list_panel,
            columns=("stage", "action"),
            show="headings",
            selectmode="browse",
            height=min(len(stages), 10),
        )
        stage_list.heading("stage", text="Stage", anchor=tk.W)
        stage_list.heading("action", text="Action", anchor=tk.W)
        stage_list.column("stage", width=200, minwidth=60, stretch=True)
        stage_list.column("action", width=110, minwidth=60, stretch=False)
        stage_list.tag_configure("flagged", foreground=_DARK_DANGER_TEXT)
        stage_list.tag_configure("overwrite", foreground="#ffb74d")
        stage_list.tag_configure("skip", foreground=_DARK_MUTED)
        stage_list.grid(row=0, column=0, sticky=tk.NSEW)
        stage_list_scrollbar = ttk.Scrollbar(
            stage_list_panel,
            orient=tk.VERTICAL,
            command=stage_list.yview,
        )
        stage_list_scrollbar.grid(row=0, column=1, sticky=tk.NS)
        stage_list.configure(
            yscrollcommand=_auto_hiding_scrollbar_setter(stage_list_scrollbar)
        )
        for index, stage in enumerate(stages):
            stage_list.insert(
                "",
                tk.END,
                iid=str(index),
                values=(_stage_detail_caption(stage), ""),
            )
        contact_panel = ttk.Frame(action_panel)
        contact_panel.grid(row=3, column=0, sticky=tk.NSEW)
        contact_panel.columnconfigure(0, weight=1)
        contact_panel.rowconfigure(0, weight=1)
        document_panel = ttk.Frame(frame)
        document_panel.grid(row=0, column=1, sticky=tk.NSEW, padx=(4, 0))
        document_panel.columnconfigure(0, weight=1)
        document_panel.rowconfigure(1, weight=1)
        document_toolbar = ttk.Frame(document_panel)
        document_toolbar.grid(row=0, column=0, sticky=tk.EW, pady=(0, 8))
        document_toolbar.columnconfigure(0, weight=1)
        open_file_button = ttk.Button(document_toolbar, text="Open this file")
        open_file_button.grid(row=0, column=1)
        open_folder_button = ttk.Button(document_toolbar, text="Open folder")
        open_folder_button.grid(row=0, column=2, padx=(8, 0))
        document_tabs = ttk.Notebook(document_panel)
        document_tabs.grid(row=1, column=0, sticky=tk.NSEW)
        contact_text = tk.Text(
            contact_panel,
            width=32,
            height=8,
            wrap=tk.NONE,
            state=tk.DISABLED,
            background=_DARK_INPUT,
            foreground=_DARK_FOREGROUND,
            selectbackground=_DARK_SELECTION,
            selectforeground=_DARK_FOREGROUND,
            highlightthickness=1,
            highlightbackground=_DARK_BORDER,
            highlightcolor=_DARK_SELECTION,
            relief=tk.FLAT,
        )
        contact_text.grid(row=0, column=0, sticky=tk.NSEW)
        contact_scrollbar = ttk.Scrollbar(
            contact_panel,
            orient=tk.VERTICAL,
            command=contact_text.yview,
        )
        contact_scrollbar.grid(row=0, column=1, sticky=tk.NS)
        contact_horizontal_scrollbar = ttk.Scrollbar(
            contact_panel,
            orient=tk.HORIZONTAL,
            command=contact_text.xview,
        )
        contact_horizontal_scrollbar.grid(row=1, column=0, sticky=tk.EW)
        contact_text.configure(
            xscrollcommand=contact_horizontal_scrollbar.set,
            yscrollcommand=contact_scrollbar.set,
        )

        def open_link(target: str) -> None:
            try:
                if not webbrowser.open(target, new=2):
                    raise OSError(f"No application is available to open {target}")
            except (OSError, webbrowser.Error) as exc:
                messagebox.showerror(_title("Unable to open link"), str(exc), parent=window)

        def set_linked_text(
            widget: tk.Text,
            content: str,
            tags: list[str],
            tag_prefix: str,
        ) -> None:
            default_cursor = widget.cget("cursor")
            widget.configure(state=tk.NORMAL)
            for tag in tags:
                widget.tag_delete(tag)
            tags.clear()
            widget.delete("1.0", tk.END)
            widget.insert("1.0", content)
            for index, (start, end, target) in enumerate(_find_links(content)):
                tag = f"{tag_prefix}_{index}"
                tags.append(tag)
                widget.tag_add(tag, f"1.0+{start}c", f"1.0+{end}c")
                widget.tag_configure(tag, foreground=_DARK_LINK, underline=True)
                widget.tag_bind(
                    tag,
                    "<Enter>",
                    lambda _event, control=widget: control.configure(cursor="hand2"),
                )
                widget.tag_bind(
                    tag,
                    "<Leave>",
                    lambda _event, control=widget, cursor=default_cursor: control.configure(
                        cursor=cursor
                    ),
                )
                widget.tag_bind(
                    tag,
                    "<Button-1>",
                    lambda _event, link=target: open_link(link),
                )
            widget.configure(state=tk.DISABLED)

        def open_path(path: Path) -> None:
            try:
                path = self.filesystem.read_path(path)
                _open_path(path)
            except OSError as exc:
                messagebox.showerror(_title("Unable to open"), str(exc), parent=window)

        def add_tab(
            title: str,
            content: str,
            *,
            tag_prefix: str,
        ) -> None:
            panel = ttk.Frame(document_tabs)
            panel.columnconfigure(0, weight=1)
            panel.rowconfigure(0, weight=1)
            text = tk.Text(
                panel,
                width=68,
                height=17,
                wrap=tk.WORD,
                state=tk.DISABLED,
                background=_DARK_INPUT,
                foreground=_DARK_FOREGROUND,
                insertbackground=_DARK_FOREGROUND,
                selectbackground=_DARK_SELECTION,
                selectforeground=_DARK_FOREGROUND,
                highlightthickness=1,
                highlightbackground=_DARK_BORDER,
                highlightcolor=_DARK_SELECTION,
                relief=tk.FLAT,
            )
            text.grid(row=0, column=0, sticky=tk.NSEW)
            scrollbar = ttk.Scrollbar(panel, orient=tk.VERTICAL, command=text.yview)
            scrollbar.grid(row=0, column=1, sticky=tk.NS)
            text.configure(yscrollcommand=scrollbar.set)
            set_linked_text(text, content, [], tag_prefix)
            document_tabs.add(panel, text=title)

        def add_pdf_tab(document: StageDocument, *, tag_prefix: str) -> None:
            panel = ttk.Frame(document_tabs)
            try:
                viewer = PdfViewer(
                    panel,
                    self.filesystem.read_bytes(document.path),
                    displayed_stage.metadata.name,
                    background=_DARK_INPUT,
                    page_border=_DARK_BORDER,
                    match_color=_DARK_LINK,
                    current_color=_DARK_DANGER_ACTIVE,
                )
            except (OSError, PdfError) as exc:
                panel.destroy()
                add_tab(
                    document.title,
                    f"Unable to preview this PDF ({exc}). Use “Open this file” to read it.",
                    tag_prefix=tag_prefix,
                )
                return
            viewer.pack(fill=tk.BOTH, expand=True)
            document_tabs.add(panel, text=document.title)

        def selected_document():
            if not documents:
                return None
            index = document_tabs.index(document_tabs.select())
            if index >= len(documents):
                return None
            document = documents[index]
            return document if self.filesystem.is_file(document.path) else None

        def open_file() -> None:
            document = selected_document()
            if document:
                open_path(document.path)

        def show_document(_event=None) -> None:
            open_file_button.configure(
                state=tk.NORMAL if selected_document() else tk.DISABLED
            )

        def open_folder() -> None:
            document = selected_document()
            open_path(document.path.parent if document else displayed_stage.root)

        def show_stage(stage) -> None:
            nonlocal displayed_stage, documents
            displayed_stage = stage
            document_contents = self._stage_document_contents(stage)
            email_links, url_links = _collect_contact_links(
                (
                    stage.metadata.author_website,
                    stage.metadata.author_note,
                    *document_contents.values(),
                )
            )
            copyright_notices = _collect_copyright_notices(document_contents.values())
            for tab in document_tabs.tabs():
                document_tabs.nametowidget(tab).destroy()
            documents = (*stage.documents, *rsf_documents)
            if documents:
                for index, document in enumerate(documents):
                    suffix = document.path.suffix.casefold()
                    if suffix == ".pdf":
                        add_pdf_tab(document, tag_prefix=f"document_link_{index}")
                        continue
                    content = (
                        f"{suffix.removeprefix('.').upper()} document. Use “Open this file” to read it."
                        if suffix not in PLAIN_TEXT_DOCUMENT_SUFFIXES
                        else document_contents.get(document.path, "Unable to read this document.")
                    )
                    add_tab(
                        document.title,
                        content,
                        tag_prefix=f"document_link_{index}",
                    )
            else:
                add_tab(
                    "No relevant files",
                    "No readme, licence, attribution, copyright, terms, permission,\n"
                    "or credits files were found in this stage folder.",
                    tag_prefix="notice_link",
                )
            set_linked_text(
                contact_text,
                "\n".join(
                    [
                        "Author",
                        stage.metadata.author or "-",
                        "",
                        *(
                            ["Author note", stage.metadata.author_note, ""]
                            if stage.metadata.author_note
                            else []
                        ),
                        "Copyright and attribution",
                        *copyright_notices,
                        *([] if copyright_notices else ["-"]),
                        "",
                        "Email addresses",
                        *(label for label, _target in email_links),
                        *([] if email_links else ["-"]),
                        "",
                        "Websites",
                        *(label for label, _target in url_links),
                        *([] if url_links else ["-"]),
                    ]
                ),
                contact_tags,
                "contact_link",
            )
            warning = _stage_warning(stage)
            warning_label.configure(text=warning)
            if warning:
                warning_label.grid()
            else:
                warning_label.grid_remove()
            show_document()

        def select_stage(_event=None) -> None:
            selection = stage_list.selection()
            if selection and stages[int(selection[0])] is not displayed_stage:
                show_stage(stages[int(selection[0])])

        def stage_actions() -> list[str]:
            return [
                _stage_conversion_action(
                    stage,
                    exists=stage.metadata.folder_name in existing,
                    overwrite=overwrite_existing.get(),
                )
                for stage in stages
            ]

        def conversion_targets(actions: list[str]) -> list:
            return [
                stage
                for stage, action in zip(stages, actions)
                if action in ("New", "Overwrite")
            ]

        def refresh_actions(*_args) -> None:
            actions = stage_actions()
            for index, (stage, action) in enumerate(zip(stages, actions)):
                stage_list.set(str(index), "action", action)
                stage_list.item(
                    str(index),
                    tags=_stage_tree_tags(stage) or (action.casefold(),),
                )
            summary.set(_stage_conversion_summary(actions))
            targets = conversion_targets(actions)
            button_stages = targets or stages
            convert_button.configure(
                text=f"Agree and {self._convert_verb()} ({len(targets)})",
                state=tk.NORMAL if targets else tk.DISABLED,
                style=(
                    "Danger.TButton"
                    if not all(stage.valid for stage in button_stages)
                    else "Success.TButton"
                ),
            )

        def continue_conversion() -> None:
            targets = conversion_targets(stage_actions())
            overwrite = overwrite_existing.get()
            window.grab_release()
            window.destroy()
            self._begin_conversion(targets, batch=batch, overwrite_existing=overwrite)

        open_folder_button.configure(command=open_folder)
        open_file_button.configure(command=open_file)
        document_tabs.bind("<<NotebookTabChanged>>", show_document)
        buttons = ttk.Frame(frame)
        buttons.grid(row=1, column=0, columnspan=2, sticky=tk.EW, pady=(10, 0))
        buttons.columnconfigure(0, weight=1)
        warning_label = ttk.Label(
            buttons,
            justify=tk.RIGHT,
            wraplength=1000,
            style="Danger.TLabel",
            font=self.stage_caption_font,
        )
        warning_label.grid(row=0, column=0, columnspan=3, sticky=tk.E, pady=(0, 8))
        ttk.Label(
            buttons,
            text=_STAGE_CREATOR_NOTICE_TEXT,
            justify=tk.LEFT,
            wraplength=720,
            style="Danger.TLabel",
        ).grid(row=1, column=0, sticky=tk.W, padx=(0, 12))
        convert_button = ttk.Button(buttons, width=-10, command=continue_conversion)
        convert_button.grid(row=1, column=1)
        ttk.Button(buttons, text="Cancel", command=window.destroy, width=-10).grid(
            row=1,
            column=2,
            padx=(6, 0),
        )
        overwrite_existing.trace_add("write", refresh_actions)
        stage_list.bind("<<TreeviewSelect>>", select_stage)
        stage_list.selection_set("0")
        stage_list.focus("0")
        stage_list.focus_set()
        refresh_actions()
        show_stage(stages[0])
        window.protocol("WM_DELETE_WINDOW", window.destroy)
        window.update_idletasks()
        parent_width = self.root.winfo_width()
        width = 1275
        height = 720
        x = self.root.winfo_rootx() + (parent_width - width) // 2
        y = self.root.winfo_rooty() + (self.root.winfo_height() - height) // 2
        window.geometry(f"{width}x{height}+{x}+{y}")
        window.deiconify()
        window.grab_set()

    def _update_convert_state(self) -> None:
        batch_selection = self._has_batch_selection()
        coordinate_state = (
            tk.NORMAL
            if not batch_selection or self.batch_override_coordinates.get()
            else tk.DISABLED
        )
        map_altitude_state = (
            tk.NORMAL
            if not batch_selection or self.batch_override_map_altitude.get()
            else tk.DISABLED
        )
        date_state = (
            tk.NORMAL
            if not batch_selection or self.batch_override_environment_date.get()
            else tk.DISABLED
        )
        temperature_state = (
            tk.NORMAL
            if not batch_selection or self.batch_override_temperatures.get()
            else tk.DISABLED
        )
        if batch_selection:
            self.batch_environment_frame.grid()
        else:
            self.batch_environment_frame.grid_remove()
        self._update_original_variants()
        self._refresh_temperature_season_buttons()
        temperatures = _parse_temperatures(
            self.temperature_night.get(),
            self.temperature_day.get(),
        )
        coordinates_valid = _coordinate_fields_valid(
            self.latitude.get(),
            self.longitude.get(),
        )
        map_altitude_valid = _parse_map_altitude(self.map_altitude.get()) is not None
        environment_date_valid = _environment_date_field_valid(
            self.environment_date.get()
        )
        temperature_style = "TEntry" if temperatures else "Error.TEntry"
        coordinate_style = "TEntry" if coordinates_valid else "Error.TEntry"
        map_altitude_style = "TEntry" if map_altitude_valid else "Error.TEntry"
        date_style = "TEntry" if environment_date_valid else "Error.TEntry"
        self.temperature_night_entry.configure(
            state=temperature_state,
            style=temperature_style,
        )
        self.temperature_day_entry.configure(
            state=temperature_state,
            style=temperature_style,
        )
        self.latitude_entry.configure(
            state=coordinate_state,
            style=coordinate_style,
        )
        self.longitude_entry.configure(
            state=coordinate_state,
            style=coordinate_style,
        )
        self.map_altitude_entry.configure(
            state=map_altitude_state,
            style=map_altitude_style,
        )
        self.environment_date_entry.configure(
            state=date_state,
            style=date_style,
        )
        for entry in self._previewed_entries:
            entry.configure(style="Preview.TEntry")
        self.location_profile_selector.configure(
            state=(
                "readonly"
                if self.location_profiles
                and not batch_selection
                else tk.DISABLED
            )
        )
        self.country_coordinate_button.configure(
            text=(
                "GPS"
                if self.selected_location is None
                else (
                    f"GPS: {self.selected_location.latitude:g}°, "
                    f"{self.selected_location.longitude:g}°"
                )
            ),
            state=(
                tk.NORMAL
                if self.selected_location is not None
                and not batch_selection
                else tk.DISABLED
            )
        )
        beamng_mods_valid = bool(
            self.beamng_mods_dir.get()
            and is_beamng_mods_dir(
                Path(self.beamng_mods_dir.get()).expanduser(),
                self.filesystem,
            )
        )
        self.beamng_mods_entry.configure(
            style="TEntry" if beamng_mods_valid else "Error.TEntry"
        )
        if self.conversion_active:
            self.cancel_button.grid()
            self.cancel_button.configure(
                text="Cancelling…" if self.cancel_requested else "Cancel",
                state=tk.DISABLED if self.cancel_requested else tk.NORMAL,
                style="Danger.TButton" if self.cancel_requested else "TButton",
            )
        else:
            self.cancel_button.grid_remove()
        if self.pacenote_preview_active:
            self.convert_button.configure(
                text="Generating preview…",
                state=tk.DISABLED,
                style="TButton",
            )
            return
        if self._stage_preload_active_generation == self.stage_preload_generation:
            self.convert_button.configure(
                text="Reading stage list…",
                state=tk.DISABLED,
                style="TButton",
            )
            return
        target_stages, batch_targets = self._convert_targets()
        stage = target_stages[0] if target_stages and not batch_targets else None
        stage_preloading = bool(
            stage
            and stage.source_format == "original"
            and not self._stage_preload_ready(stage)
        )
        if stage and not stage.valid and not stage_preloading:
            self.convert_button.configure(
                text=self._convert_verb(),
                state=tk.NORMAL,
                style="Danger.TButton",
            )
            return
        original_variants_valid = bool(
            not stage
            or stage.source_format != "original"
            or not stage_preloading
        )
        enabled = bool(
            (target_stages if batch_targets else stage and stage.valid)
            and beamng_mods_valid
            and (
                not batch_selection
                or not self.batch_override_temperatures.get()
                or temperatures is not None
            )
            and (
                not batch_selection
                or not self.batch_override_coordinates.get()
                or coordinates_valid
            )
            and (
                not batch_selection
                or not self.batch_override_map_altitude.get()
                or map_altitude_valid
            )
            and (
                not batch_selection
                or not self.batch_override_environment_date.get()
                or environment_date_valid
            )
            and (
                batch_selection
                or (
                    temperatures is not None
                    and coordinates_valid
                    and map_altitude_valid
                    and environment_date_valid
                )
            )
            and original_variants_valid
        )
        self.convert_button.configure(
            text=(
                "Inspecting…"
                if stage_preloading
                else f"{self._convert_verb()} ({len(target_stages)})"
                if batch_targets
                else self._convert_verb()
            ),
            state=tk.NORMAL if enabled else tk.DISABLED,
            style="Success.TButton" if enabled else "TButton",
        )

    def _selected_original_variants(self, stage) -> tuple[str, ...]:
        if stage.source_format != "original" or not stage.variants:
            return ()
        primary = primary_original_tint(stage.variants)
        extras = tuple(
            tint
            for tint in stage.variants
            if tint != primary and self.original_variant_values[tint].get()
        )
        return (primary, *extras) if extras else ()

    def _cli_command(self) -> list[str]:
        if getattr(sys, "frozen", False):
            cli = Path(sys.executable).with_name("rbr2beamng-cli.exe")
            return [str(cli)]
        return [sys.executable, "-m", "rbr2beamng.cli"]

    def _preview_pacenotes(self) -> None:
        if self.pacenote_preview_active:
            return
        if self.conversion_active:
            messagebox.showinfo(
                _title("Conversion in progress"),
                "Wait for the current conversion to finish before generating a pacenote preview.",
                parent=self.root,
            )
            return
        stage = self._selected_stage()
        if stage is None:
            messagebox.showinfo(
                _title("No stage focused"),
                "Select a stage before generating a pacenote preview.",
                parent=self.root,
            )
            return
        if not stage.valid:
            messagebox.showerror(
                _title("Cannot preview stage"),
                "\n".join(stage.issues) or "The selected stage is invalid.",
                parent=self.root,
            )
            return

        original_variants = self._selected_original_variants(stage)

        try:
            rbr_root = self.filesystem.read_path(self.rbr_path.get())
            output_dir = self.filesystem.write_path(settings_path().parent / "logs")
            self.filesystem.mkdir(output_dir, parents=True, exist_ok=True)
        except (OSError, SandboxViolationError) as exc:
            messagebox.showerror(
                _title("Cannot prepare preview"),
                f"Unable to access the preview files:\n{exc}",
                parent=self.root,
            )
            return

        command = _pacenote_preview_command(
            self._cli_command(),
            rbr_root,
            stage,
            output_dir,
            original_variants,
        )
        self.pacenote_preview_active = True
        self.preview_pacenotes_button.configure(state=tk.DISABLED)
        self.progress.set(
            value=0,
            text=f"Generating pacenote preview for {stage.metadata.name}…",
            tone="normal",
        )
        self.progress.start_busy()
        self._update_convert_state()
        threading.Thread(
            target=self._run_pacenote_preview,
            args=(command, stage.metadata.name),
            daemon=True,
        ).start()

    def _run_pacenote_preview(
        self,
        command: list[str],
        stage_name: str,
    ) -> None:
        try:
            command = [
                str(self.filesystem.read_path(command[0])),
                *command[1:],
            ]
            result = subprocess.run(
                command,
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="replace",
                check=False,
                creationflags=(
                    subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
                ),
            )
            self.events.put(
                (
                    "pacenote_preview_finished",
                    (
                        stage_name,
                        result.returncode,
                        _pacenote_preview_outputs(result.stdout),
                        result.stdout,
                        result.stderr,
                    ),
                )
            )
        except Exception:
            self.events.put(
                (
                    "pacenote_preview_error",
                    (stage_name, traceback.format_exc().rstrip()),
                )
            )

    def _finish_pacenote_preview(
        self,
        payload: tuple[str, int, tuple[Path, ...], str, str],
    ) -> None:
        stage_name, return_code, outputs, stdout, stderr = payload
        self.progress.stop_busy()
        self.pacenote_preview_active = False
        self.preview_pacenotes_button.configure(
            state=tk.DISABLED if self.conversion_active else tk.NORMAL
        )
        self._update_convert_state()
        if return_code != 0 or not outputs:
            detail = "\n".join(part for part in (stderr, stdout) if part).strip()
            messagebox.showerror(
                _title("Pacenote preview failed"),
                (
                    f"Could not generate a pacenote preview for {stage_name}."
                    + (f"\n\n{detail[-4000:]}" if detail else "")
                ),
                parent=self.root,
            )
            self.progress.set(
                value=0,
                text=f"Pacenote preview failed: {stage_name}",
                tone="failure",
            )
            return

        opened: list[Path] = []
        errors: list[str] = []
        for output in outputs:
            try:
                _open_path(output)
                opened.append(output)
            except OSError as exc:
                errors.append(f"{output.name}: {exc}")
        if errors:
            messagebox.showerror(
                _title("Could not open pacenote preview"),
                "\n".join(errors),
                parent=self.root,
            )
        count = len(opened)
        if not count:
            self.progress.set(
                value=0,
                text=f"Pacenote preview could not be opened: {stage_name}",
                tone="failure",
            )
            return
        self.progress.set(
            value=100,
            text=(
                f"Opened pacenote preview: {opened[0].name}"
                if count == 1
                else f"Opened {count} pacenote previews"
            ),
            tone="success",
        )

    def _handle_pacenote_preview_error(self, payload: tuple[str, str]) -> None:
        stage_name, error = payload
        self.pacenote_preview_active = False
        self.preview_pacenotes_button.configure(
            state=tk.DISABLED if self.conversion_active else tk.NORMAL
        )
        self._update_convert_state()
        self.progress.set(
            value=0,
            text=f"Pacenote preview failed: {stage_name}",
            tone="failure",
        )
        messagebox.showerror(
            _title("Pacenote preview failed"),
            f"Could not generate a pacenote preview for {stage_name}.\n\n{error}",
            parent=self.root,
        )

    def _destination(self, stage) -> Path:
        mods_dir = Path(self.beamng_mods_dir.get())
        return self.filesystem.write_path(
            mods_dir
            / stage_zip_name(
                stage.metadata.folder_name,
                display_name=stage.metadata.name,
                source_format=stage.source_format,
            )
        )

    def _build_convert_command(
        self,
        folder: str,
        *,
        temperatures: tuple[float, float] | None,
        overwrite: bool,
        original_variants: tuple[str, ...] = (),
        coordinates: tuple[float, float] | None = None,
        map_altitude: float | None = None,
        environment_date: str | None = None,
    ) -> list[str]:
        rbr_root = self.filesystem.read_path(self.rbr_path.get())
        beamng_mods_dir = self.filesystem.write_path(
            self.beamng_mods_dir.get()
        )
        command = self._cli_command() + [
            "convert",
            "--rbr-dir",
            str(rbr_root),
            "--beamng-mods-dir",
            str(beamng_mods_dir),
            "--stage",
            folder,
            "--progress-json",
        ]
        if overwrite:
            command.append("--overwrite")
        if self.remove_source_skybox.get():
            command.append("--remove-source-skybox")
        if not self.use_water_name_fallback.get():
            command.append("--no-water-name-fallback")
        if self.water_name_matches != DEFAULT_WATER_NAME_MATCHES:
            command.extend(("--water-name-matches", *self.water_name_matches))
        if not self.use_snowwall_collision_override.get():
            command.append("--no-snowwall-collision-override")
        if not self.use_snowbank_name_fallback.get():
            command.append("--no-snowbank-name-fallback")
        if self.snowbank_name_matches != DEFAULT_SNOWBANK_NAME_MATCHES:
            command.extend(
                ("--snowbank-name-matches", *self.snowbank_name_matches)
            )
        if self.snowbank_name_mesh_patterns != DEFAULT_SNOWBANK_NAME_MESH_PATTERNS:
            command.extend(
                (
                    "--snowbank-name-mesh-patterns",
                    *self.snowbank_name_mesh_patterns,
                )
            )
        if not self.inflate_thin_walls.get():
            command.append("--no-thin-wall-inflation")
        if not self.use_visual_lods.get():
            command.append("--no-visual-lods")
        if not self.use_map_border_brake_walls.get():
            command.append("--no-map-border-brake-walls")
        if not self.use_foliage_name_fallback.get():
            command.append("--no-foliage-name-fallback")
        if self.foliage_name_matches != DEFAULT_FOLIAGE_NAME_MATCHES:
            command.extend(
                ("--foliage-name-matches", *self.foliage_name_matches)
            )
        if not self.use_foliage_ground_types.get():
            command.append("--no-foliage-ground-types")
        if self.foliage_ground_types != DEFAULT_FOLIAGE_GROUND_TYPES:
            command.extend(
                ("--foliage-ground-types", *self.foliage_ground_types)
            )
        if coordinates is not None:
            command.extend(
                (
                    "--latitude",
                    f"{coordinates[0]:.12g}",
                    "--longitude",
                    f"{coordinates[1]:.12g}",
                )
            )
        if map_altitude is not None:
            command.extend(("--map-altitude", f"{map_altitude:g}"))
        if environment_date is not None:
            command.extend(("--environment-date", environment_date))
        for tint in original_variants:
            command.extend(("--original-variant", tint))
        if temperatures is not None:
            command.extend(
                (
                    "--temperature-night",
                    f"{temperatures[0]:g}",
                    "--temperature-day",
                    f"{temperatures[1]:g}",
                )
            )
        return command

    def _convert_verb(self) -> str:
        return "Enqueue" if self.conversion_active else "Convert"

    def _queued_stage_keys(self) -> set[str]:
        if not self.conversion_active:
            return set()
        return {job.stage.metadata.folder_name for job in self.conversion_jobs}

    def _convert_targets(self) -> tuple[list, bool]:
        """Stages Convert would add now, and whether they use batch settings."""
        queued = self._queued_stage_keys()
        checked = self._selected_stages()
        if checked:
            return (
                [stage for stage in checked if stage.metadata.folder_name not in queued],
                self._has_batch_selection(),
            )
        stage = self._selected_stage()
        if stage is None or stage.metadata.folder_name in queued:
            return [], False
        return [stage], False

    def _primary_action(self) -> None:
        stages, batch = self._convert_targets()
        if batch:
            self._start_selected_conversion()
        elif stages:
            self._start_conversion(stages[0])

    def _start_conversion(self, stage=None) -> None:
        if self._stage_preload_active_generation == self.stage_preload_generation:
            return
        stage = stage or self._selected_stage()
        if not stage or stage.metadata.folder_name in self._queued_stage_keys():
            return
        if (
            stage.source_format == "original"
            and not self._stage_preload_ready(stage)
        ):
            self.pending_batch_conversion_keys.clear()
            self.pending_current_conversion_key = stage.metadata.folder_name
            self._request_stage_preload(stage)
            self._update_convert_state()
            return
        self._confirm_stage_terms((stage,))

    def _start_selected_conversion(self) -> None:
        stages = tuple(self._convert_targets()[0])
        if (
            self._stage_preload_active_generation == self.stage_preload_generation
            or not stages
        ):
            return
        pending = {
            stage.metadata.folder_name
            for stage in stages
            if (
                stage.source_format == "original"
                and not self._stage_preload_ready(stage)
            )
        }
        if pending:
            self.pending_current_conversion_key = None
            self.pending_batch_conversion_keys = pending
            for stage in stages:
                if stage.metadata.folder_name in pending:
                    self._request_stage_preload(stage)
            self._update_convert_state()
            return
        self._confirm_stage_terms(stages, batch=True)

    def _resume_pending_stage_actions(self) -> None:
        folder = self.pending_current_conversion_key
        if folder:
            stage = next(
                (
                    candidate
                    for candidate in self.stages
                    if candidate.metadata.folder_name == folder
                ),
                None,
            )
            if not stage or self._stage_preload_ready(stage):
                self.pending_current_conversion_key = None
                if stage:
                    self._confirm_stage_terms((stage,))
        if self.pending_batch_conversion_keys and all(
            folder in self.inspected_original_stage_keys
            for folder in self.pending_batch_conversion_keys
        ):
            self.pending_batch_conversion_keys.clear()
            self._start_selected_conversion()

    def _begin_conversion(
        self,
        stages,
        *,
        batch: bool,
        overwrite_existing: bool,
    ) -> None:
        self._save_paths()
        jobs = []
        for stage in stages:
            job = self._prepare_conversion(
                stage,
                batch=batch,
                overwrite_existing=overwrite_existing,
            )
            if job is None:
                return
            jobs.append(job)
        self._queue_conversion_jobs(jobs)

    def _queue_conversion_jobs(self, jobs: list[_ConversionJob]) -> None:
        if self.conversion_active:
            self.conversion_jobs.extend(jobs)
            for job in jobs:
                self._append_log(f"Queued: {job.stage.metadata.name}\n")
            self.progress.set(text=self._active_progress_text(), eta=self._eta_text())
            self._update_convert_state()
            return
        self._cancel_stage_preload()
        self.conversion_jobs = list(jobs)
        self.conversion_job_index = 0
        self.batch_failures = 0
        self.batch_failed_stage_names = []
        self.batch_option_stats = []
        self.completed_stage_durations = {}
        self._start_conversion_job(self.conversion_jobs[0])

    def _start_next_queued_job(self) -> bool:
        if self.conversion_job_index + 1 >= len(self.conversion_jobs):
            return False
        self.conversion_job_index += 1
        self._cancel_stage_preload()
        self._start_conversion_job(self.conversion_jobs[self.conversion_job_index])
        return True

    def _prepare_conversion(
        self,
        stage,
        *,
        batch: bool = False,
        overwrite_existing: bool = True,
    ) -> _ConversionJob | None:
        use_batch_temperatures = (
            not batch or self.batch_override_temperatures.get()
        )
        temperatures = _parse_temperatures(
            self.temperature_night.get(),
            self.temperature_day.get(),
        )
        if use_batch_temperatures and temperatures is None:
            messagebox.showerror(
                _title("Invalid temperatures"),
                "Night and day temperatures must be numbers, and night cannot exceed day.",
                parent=self.root,
            )
            return
        coordinates = _parse_coordinates(
            self.latitude.get(),
            self.longitude.get(),
        )
        use_batch_coordinates = (
            not batch or self.batch_override_coordinates.get()
        )
        if (
            use_batch_coordinates
            and
            coordinates is None
            and (self.latitude.get().strip() or self.longitude.get().strip())
        ):
            messagebox.showerror(
                _title("Invalid GPS coordinates"),
                "Latitude must be -90..90 and longitude must be -180..180.",
                parent=self.root,
            )
            return
        map_altitude = _parse_map_altitude(self.map_altitude.get())
        use_batch_map_altitude = (
            not batch or self.batch_override_map_altitude.get()
        )
        if use_batch_map_altitude and map_altitude is None:
            messagebox.showerror(
                _title("Invalid map altitude"),
                "Map altitude must be a finite number of meters.",
                parent=self.root,
            )
            return
        use_batch_environment_date = (
            not batch or self.batch_override_environment_date.get()
        )
        environment_date = _parse_environment_date(self.environment_date.get())
        if use_batch_environment_date and environment_date is None:
            messagebox.showerror(
                _title("Invalid environment date"),
                "Date must use YYYY-MM-DD format.",
                parent=self.root,
            )
            return
        destination = self._destination(stage)
        overwrite = overwrite_existing and self.filesystem.exists(destination)
        with use_filesystem(self.filesystem):
            coordinate_override = (
                _coordinate_override(stage, coordinates)
                if use_batch_coordinates
                else None
            )
        command = self._build_convert_command(
            stage.metadata.folder_name,
            temperatures=temperatures if use_batch_temperatures else None,
            overwrite=overwrite,
            coordinates=coordinate_override,
            map_altitude=map_altitude if use_batch_map_altitude else None,
            environment_date=(
                (environment_date or None)
                if use_batch_environment_date
                else None
            ),
            original_variants=(
                () if batch else self._selected_original_variants(stage)
            ),
        )
        return _ConversionJob(stage, command, destination)

    def _start_conversion_job(self, job: _ConversionJob) -> None:
        command = job.command
        self._reset_conversion_state()
        self.current_conversion_stage_name = job.stage.metadata.name
        try:
            log_directory = settings_path().parent / "logs"
            self.filesystem.mkdir(log_directory, parents=True, exist_ok=True)
            self.conversion_log_path = self.filesystem.write_path(
                log_directory / f"{job.destination.stem}.log"
            )
            self.filesystem.write_text(
                self.conversion_log_path,
                "",
                encoding="utf-8",
            )
        except OSError:
            self.conversion_log_path = None
        self._append_log(f"> {' '.join(command)}\n", elapsed=0)
        self.progress.set(
            value=0,
            text=self._active_progress_text(0),
            tone="normal",
            eta=self._eta_text(),
        )
        self.last_output = job.destination
        self._update_convert_state()
        thread = threading.Thread(
            target=self._run_process,
            args=(command,),
            daemon=True,
        )
        thread.start()

    def _reset_conversion_state(self) -> None:
        self.process = None
        self.conversion_active = True
        self.cancel_requested = False
        self.cancellation_logged = False
        self.conversion_started_at = time.monotonic()
        self.conversion_status = "Starting conversion…"
        self.stage_progress = (0.0, 0.0)
        self.stage_progress_reached_at = 0.0
        self.last_elapsed_second = -1
        self.completed_zip_size = None
        self.completed_collision_triangles = None
        self.completed_merged_away_triangles = None
        self.completed_route_length_meters = None
        self.completed_sector_count = None
        self.completed_option_stats = None

    def _elapsed_seconds(self) -> float:
        if self.conversion_started_at is None:
            return 0.0
        return time.monotonic() - self.conversion_started_at

    def _active_progress_text(self, elapsed: float | None = None) -> str:
        elapsed_seconds = self._elapsed_seconds() if elapsed is None else elapsed
        if self.completed_zip_size is not None:
            text = _format_completed_conversion_status(
                elapsed_seconds,
                self.last_output,
                self.completed_zip_size,
                self.completed_collision_triangles,
            )
        else:
            text = _format_conversion_status(
                self.conversion_status,
                elapsed_seconds,
            )
        return self._stage_progress_text(text)

    def _stage_progress_text(self, text: str) -> str:
        separator = ": " if getattr(self, "current_conversion_stage_name", "") else " "
        return f"{self._stage_progress_label()}{separator}{text}"

    def _stage_progress_label(self) -> str:
        stage_name = getattr(self, "current_conversion_stage_name", "")
        jobs = getattr(self, "conversion_jobs", [])
        if jobs:
            stage_count = f"[{self.conversion_job_index + 1}/{len(jobs)}]"
        else:
            stage_count = "[1/1]"
        return f"{stage_count} {stage_name}".rstrip()

    def _current_stage_format(self) -> str:
        return self.conversion_jobs[self.conversion_job_index].stage.source_format

    def _eta_text(self) -> str:
        if not self.conversion_active or self.cancel_requested:
            return ""
        reached, limit = self.stage_progress
        remaining = _estimate_remaining_seconds(
            elapsed=self._elapsed_seconds(),
            reached=reached,
            reached_at=self.stage_progress_reached_at,
            limit=limit,
            stage_format=self._current_stage_format(),
            queued_formats=[
                job.stage.source_format
                for job in self.conversion_jobs[self.conversion_job_index + 1:]
            ],
            completed_durations=self.completed_stage_durations,
        )
        return "" if remaining is None else f"~{format_duration(remaining)} left"

    def _update_elapsed_display(self) -> None:
        if not self.conversion_active or self.conversion_started_at is None:
            return
        elapsed_second = int(self._elapsed_seconds())
        if elapsed_second == self.last_elapsed_second:
            return
        self.last_elapsed_second = elapsed_second
        self.progress.set(text=self._active_progress_text(), eta=self._eta_text())

    def _run_process(self, command: list[str]) -> None:
        try:
            if self.cancel_requested:
                self.events.put(("finished", 130))
                return
            creation_flags = (
                subprocess.CREATE_NO_WINDOW if sys.platform == "win32" else 0
            )
            environment = os.environ.copy()
            environment["RBR2BEAMNG_GUI_RIGHTS_CONFIRMED"] = "1"
            environment["RBR2BEAMNG_GUI_LOG_ACTIVE"] = "1"
            command = [
                str(self.filesystem.read_path(command[0])),
                *command[1:],
            ]
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                encoding="utf-8",
                errors="replace",
                bufsize=1,
                creationflags=creation_flags,
                env=environment,
            )
            self.process = process
            if self.cancel_requested:
                process.terminate()
            assert process.stdout is not None
            for line in process.stdout:
                self.events.put(("line", line.rstrip("\r\n")))
            return_code = process.wait()
            if self.cancel_requested and return_code != 0:
                return_code = 130
            self.events.put(("finished", return_code))
        except Exception:
            self.events.put(("error", traceback.format_exc().rstrip()))
        finally:
            self.process = None

    def _handle_line(self, line: str) -> None:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            self._append_log(line + "\n")
            return
        if not isinstance(event, dict):
            self._append_log(line + "\n")
            return
        if "log" in event:
            self._write_log_file(f"{event['log']}\n")
            return
        message = str(event.get("message", ""))
        detail = event.get("detail")
        display = f"{message}: {detail}" if detail else message
        progress_display = message if event.get("phase") == "complete" else display
        self.conversion_status = progress_display
        zip_size = event.get("zipSizeBytes")
        if isinstance(zip_size, int) and zip_size >= 0:
            self.completed_zip_size = zip_size
        collision_triangles = event.get("collisionTriangles")
        if isinstance(collision_triangles, int) and collision_triangles >= 0:
            self.completed_collision_triangles = collision_triangles
        merged_away = event.get("mergedAwayTriangles")
        if isinstance(merged_away, int) and merged_away >= 0:
            self.completed_merged_away_triangles = merged_away
        route_length_meters = event.get("routeLengthMeters")
        if isinstance(route_length_meters, (int, float)) and route_length_meters >= 0:
            self.completed_route_length_meters = route_length_meters
        sector_count = event.get("sectorCount")
        if isinstance(sector_count, int) and sector_count > 0:
            self.completed_sector_count = sector_count
        option_stats = _option_stats_from_event(event.get("optionStats"))
        if option_stats is not None:
            self.completed_option_stats = option_stats
        elapsed = event.get("elapsed")
        if not isinstance(elapsed, (int, float)):
            elapsed = self._elapsed_seconds()
        value = None
        current = event.get("current")
        total = event.get("total")
        if isinstance(current, int) and isinstance(total, int) and total > 0:
            value = current / total * 100
        elif event.get("phase") == "complete":
            value = 100
        span = _stage_progress_span(
            self._current_stage_format(),
            event.get("phase"),
            current,
            total,
        )
        if span is not None and span > self.stage_progress:
            if span[0] > self.stage_progress[0]:
                self.stage_progress_reached_at = self._elapsed_seconds()
            self.stage_progress = span
        status = self._active_progress_text(float(elapsed))
        self.progress.set(value=value, text=status, eta=self._eta_text())
        severity = event.get("severity", "info")
        warning = "Warning: " if severity == "warning" else ""
        self._append_log(
            warning + display + "\n",
            elapsed=float(elapsed),
        )

    def _drain_events(self) -> None:
        try:
            while True:
                kind, payload = self.events.get_nowait()
                if kind == "line":
                    self._handle_line(str(payload))
                elif kind == "finished":
                    self._finished(int(payload))
                elif kind == "error":
                    self._append_log(f"Error: {payload}\n")
                    self._finished(2)
                elif kind == "original_stage_preloaded":
                    self._handle_original_stage_preloaded(payload)
                elif kind == "stage_preload_processed":
                    self._handle_stage_preload_processed(payload)
                elif kind == "stage_preload_finished":
                    self._finish_stage_preload_progress(int(payload))
                elif kind == "pacenote_preview_finished":
                    self._finish_pacenote_preview(payload)
                elif kind == "pacenote_preview_error":
                    self._handle_pacenote_preview_error(payload)
        except queue.Empty:
            pass
        self._update_elapsed_display()
        self.root.after(100, self._drain_events)

    def _finished(self, return_code: int) -> None:
        elapsed = self._elapsed_seconds()
        output = self.last_output
        zip_size = self.completed_zip_size
        collision_triangles = self.completed_collision_triangles
        merged_away = self.completed_merged_away_triangles
        route_length_meters = self.completed_route_length_meters
        sector_count = self.completed_sector_count
        option_stats = getattr(self, "completed_option_stats", None)
        cancelled = self.cancel_requested
        batch_stages = self.conversion_jobs
        batch_active = len(batch_stages) > 1
        self._clear_conversion_state()
        if return_code == 0:
            self.completed_stage_durations.setdefault(
                self._current_stage_format(),
                [],
            ).append(elapsed)
            if zip_size is None and output is not None:
                try:
                    zip_size = self.filesystem.stat(output).st_size
                except OSError:
                    zip_size = None
            self.progress.set(
                value=100,
                text=self._stage_progress_text(
                    _format_completed_conversion_status(
                        elapsed,
                        output,
                        zip_size,
                        collision_triangles,
                    )
                ),
                tone="success",
            )
            self._refresh_installed_markers()
            summary = _format_conversion_complete_summary(
                elapsed,
                output,
                zip_size,
                collision_triangles,
                merged_away,
                route_length_meters,
                sector_count,
                option_stats,
            )
            self._append_log(summary + "\n", elapsed=elapsed)
            self._write_conversion_result("SUCCESSFULL CONVERSION")
            if batch_active and not cancelled:
                if option_stats is not None:
                    self.batch_option_stats.append(option_stats)
                if self._start_next_queued_job():
                    return
                failures = self.batch_failures
                batch_option_stats = _aggregate_option_stats(
                    self.batch_option_stats
                )
                self.conversion_jobs = []
                self.batch_option_stats = []
                self.progress.set(
                    value=100,
                    text=(
                        f"Selected stages converted ({len(batch_stages) - failures} "
                        f"succeeded, {failures} failed)"
                    ),
                    tone="success" if failures == 0 else "failure",
                )
                if failures:
                    failed_stages = "\n".join(
                        f"- {name}" for name in self.batch_failed_stage_names
                    )
                    messagebox.showwarning(
                        _title("Batch conversion finished"),
                        f"{_format_batch_conversion_complete_summary(
                            len(batch_stages) - failures,
                            failures,
                            batch_option_stats,
                        )}\n\nReview the stage log files for details.\n\n"
                        f"Failed stages:\n{failed_stages}",
                        parent=self.root,
                    )
                else:
                    messagebox.showinfo(
                        _title("Batch conversion complete"),
                        _format_batch_conversion_complete_summary(
                            len(batch_stages),
                            0,
                            batch_option_stats,
                        ),
                        parent=self.root,
                    )
                self._update_convert_state()
                return
            messagebox.showinfo(_title("Conversion complete"), summary, parent=self.root)
        elif return_code == 130 or cancelled:
            self._log_cancellation(elapsed)
            self._write_conversion_result("FAILED CONVERSION")
            self.conversion_jobs = []
            self.batch_option_stats = []
            self.progress.set(
                text=self._stage_progress_text(
                    _format_conversion_status("Cancelled", elapsed)
                ),
                tone="normal",
            )
        else:
            self._write_conversion_result("FAILED CONVERSION")
            if batch_active and not cancelled:
                self.batch_failures += 1
                self.batch_failed_stage_names.append(
                    self.current_conversion_stage_name
                )
                if self._start_next_queued_job():
                    return
                failures = self.batch_failures
                batch_option_stats = _aggregate_option_stats(
                    self.batch_option_stats
                )
                self.conversion_jobs = []
                self.batch_option_stats = []
                self.progress.set(
                    value=100,
                    text=f"Batch conversion finished ({failures} failed)",
                    tone="failure",
                )
                messagebox.showwarning(
                    _title("Batch conversion finished"),
                    f"{_format_batch_conversion_complete_summary(
                        len(batch_stages) - failures,
                        failures,
                        batch_option_stats,
                    )}\n\nReview the stage log files for details.\n\n"
                    f"Failed stages:\n"
                    + "\n".join(
                        f"- {name}" for name in self.batch_failed_stage_names
                    ),
                    parent=self.root,
                )
                self._update_convert_state()
                return
            self.progress.set(
                value=100,
                text=self._stage_progress_text(
                    _format_conversion_status("Failed", elapsed)
                ),
                tone="failure",
            )
            messagebox.showerror(
                _title("Conversion failed"),
                "The conversion failed after "
                f"{format_duration(elapsed)}. Review the log for details.",
                parent=self.root,
            )

    def _clear_conversion_state(self) -> None:
        self.process = None
        self.conversion_active = False
        self.cancel_requested = False
        self.conversion_started_at = None
        self.conversion_status = ""
        self.completed_zip_size = None
        self.completed_collision_triangles = None
        self.completed_merged_away_triangles = None
        self.completed_route_length_meters = None
        self.completed_sector_count = None
        self.completed_option_stats = None
        self.progress.set(eta="")
        self._schedule_stage_preload()
        self._update_convert_state()

    def _cancel(self) -> None:
        if not self.conversion_active:
            return
        self.cancel_requested = True
        self._log_cancellation()
        self.conversion_status = "Cancelling…"
        self.progress.set(text=self._active_progress_text(), tone="normal", eta="")
        self._update_convert_state()
        self._terminate_process()

    def _terminate_process(self) -> None:
        process = self.process
        if process is not None and process.poll() is None:
            process.terminate()

    def _log_cancellation(self, elapsed: float | None = None) -> None:
        if self.cancellation_logged:
            return
        self._append_log("Conversion cancelled.\n", elapsed=elapsed)
        self.cancellation_logged = True

    def _append_log(
        self,
        text: str,
        *,
        elapsed: float | None = None,
    ) -> None:
        stamped = _timestamp_log_text(
            text,
            self._elapsed_seconds() if elapsed is None else elapsed,
            (
                self._stage_progress_label()
                if getattr(self, "current_conversion_stage_name", "")
                else ""
            ),
        )
        self.log.configure(state=tk.NORMAL)
        self.log.insert(tk.END, stamped)
        self.log.see(tk.END)
        self.log.configure(state=tk.DISABLED)
        self._write_log_file(stamped)

    def _write_conversion_result(self, result: str) -> None:
        self._write_log_file(f"{result}\n")

    def _write_log_file(self, text: str) -> None:
        conversion_log_path = getattr(self, "conversion_log_path", None)
        if conversion_log_path is None:
            return
        try:
            with self.filesystem.open(
                conversion_log_path,
                "a",
                encoding="utf-8",
            ) as stream:
                stream.write(text)
        except OSError:
            pass

    def _close(self) -> None:
        if self.conversion_active:
            if not messagebox.askyesno(
                _title("Cancel conversion?"),
                "A conversion is running. Cancel it and exit?",
                parent=self.root,
            ):
                return
            self.cancel_requested = True
            self._log_cancellation()
            self._write_conversion_result("FAILED CONVERSION")
            self._terminate_process()
        self._cancel_stage_preload()
        self.root.destroy()


def main() -> None:
    root = tk.Tk()
    root.withdraw()
    ConverterApp(root)
    root.deiconify()
    root.mainloop()


if __name__ == "__main__":
    main()
