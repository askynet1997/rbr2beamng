from __future__ import annotations

import configparser
import hashlib
import json
import time
from bisect import bisect_right
from collections import Counter, defaultdict
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

from .ini import config_parser
from .filesystem import SandboxViolationError, current_filesystem
from .models import Pacenote, RbrStage
from .pacenote_mapping_data import load_pacenote_mapping_data
from .pacenote_overrides import load_pacenote_overrides


NOTEBOOK_BASENAMES = ("structured", "casual", "enthusiast")
PACENOTE_CONFIG_HEADER_MAX_DISTANCE = 0.01
PACENOTE_PLUGIN_CONTROL_NOTE_IDS = frozenset(range(0xFE0, 0x1000))
PACENOTE_FAILURE_PREFIX = "Pacenote conversion failure: "
PACENOTE_WARNING_PREFIX = "Pacenote conversion warning: "
MODIFIER_SLOTS_FULL_PREFIX = "BeamNG "
MODIFIER_SLOTS_FULL_SEPARATOR = " modifier slots are full: "
UNSUPPORTED_BEAMNG_MODIFIER_PREFIX = "unsupported BeamNG modifier: "
NO_MODIFIER_CORNER_REASON = (
    "no BeamNG corner between the surrounding distance calls and link words"
)
NO_MODIFIER_PACENOTE_REASON = (
    "no BeamNG pacenote between the surrounding distance calls and link words"
)
PACENOTE_MAPPING_DATA = load_pacenote_mapping_data()
_LEGACY_ROWS_BY_ID = {
    int(row["id"]): row
    for row in PACENOTE_MAPPING_DATA["legacyDefinitions"]
}
_LEGACY_NOTE_IDS_BY_ROLE = {
    str(row["role"]): int(row["id"])
    for row in PACENOTE_MAPPING_DATA["legacyDefinitions"]
    if row.get("role")
}


def _legacy_note_id(role: str) -> int:
    try:
        return _LEGACY_NOTE_IDS_BY_ROLE[role]
    except KeyError as error:
        raise ValueError(f"Pacenote mapping data has no {role} legacy role") from error


DISTANCE_MARKER_NOTE_ID = _legacy_note_id("distanceMarker")
START_NOTE_ID = _legacy_note_id("start")
FINISH_NOTE_ID = _legacy_note_id("finish")
SPLIT_NOTE_ID = _legacy_note_id("split")
STOP_CONTROL_NOTE_ID = _legacy_note_id("stopControl")
UNMAPPED_NOTE_ID = _legacy_note_id("unmapped")
TIMING_NOTE_IDS = frozenset(
    {
        START_NOTE_ID,
        FINISH_NOTE_ID,
        SPLIT_NOTE_ID,
        STOP_CONTROL_NOTE_ID,
    }
)
LEGACY_CONVERSION_PRESENTATIONS = {
    int(row["id"]): (
        str(row["presentation"]["category"]),
        str(row["presentation"]["entry"]),
    )
    for row in PACENOTE_MAPPING_DATA["legacyDefinitions"]
    if row.get("presentation") and row.get("role") != "unmapped"
}
LEGACY_UNSUPPORTED_PRESENTATIONS = {
    int(row["id"]): (
        str(row["presentation"]["category"]),
        str(row["presentation"]["entry"]),
    )
    for row in PACENOTE_MAPPING_DATA["legacyDefinitions"]
    if row.get("presentation") and row.get("role") == "unmapped"
}
_FLAGS_BY_KEY = {
    str(flag["key"]): flag
    for flag in PACENOTE_MAPPING_DATA["flags"]
}


def _flag_mask(key: str) -> int:
    value = _FLAGS_BY_KEY[key]["mask"]
    if not isinstance(value, int):
        raise ValueError(f"Pacenote flag {key} has an invalid mask")
    return value


TIGHTENS_FLAG_MASK = _flag_mask("TIGHTENS")
LONG_FLAG_MASK = _flag_mask("LONG")
TIGHTENS_BAD_FLAG_MASK = _flag_mask("TIGHTENSBAD")
MAYBE_FLAG_MASK = _flag_mask("MAYBE")
EXPLICIT_SOUND_INDEX_FLAG_MASK = _flag_mask("EXPLICIT_SOUND_INDEX")
NO_LINK_OR_STICKY_FLAG_MASK = _flag_mask("NO_LINK_OR_STICKY")
EXPLICIT_SOUND_INDEX_VALUE_MASK = int(
    _FLAGS_BY_KEY["EXPLICIT_SOUND_INDEX"]["valueMask"]
)
EXPLICIT_SOUND_INDEX_VALUE_SHIFT = int(
    _FLAGS_BY_KEY["EXPLICIT_SOUND_INDEX"]["valueShift"]
)
STICKY_CONTROL_KINDS = frozenset(
    str(kind)
    for kind in _FLAGS_BY_KEY["NO_LINK_OR_STICKY"]["scopes"]
)
PHRASE_BOUNDARY_CONTROL_KINDS = frozenset({"AND", "INTO", "ONTO", "EMPTY_CALL"})
FIXED_CORNER_FLAG_NAMES = {
    TIGHTENS_FLAG_MASK: "tightens",
    TIGHTENS_BAD_FLAG_MASK: "tightens badly",
    LONG_FLAG_MASK: "long",
}
PLUGIN_FLAG_REGISTRY = {
    int(flag["mask"]): flag.get("registryName")
    for flag in PACENOTE_MAPPING_DATA["flags"]
    if flag.get("registryName") is not None
    or flag.get("registryState") == "empty"
}
PLUGIN_CONTROL_SPECS = {
    int(control["code"]): (
        str(control["kind"]),
        control.get("unit"),
        str(control["scope"]),
        bool(control["usesSignedFixedPointPayload"]),
        bool(control.get("usesSignedIntegerPayload", False)),
    )
    for control in PACENOTE_MAPPING_DATA["pluginControls"]
}
PLUGIN_CONTROL_CONVERSIONS = {
    int(control["code"]): (
        str(control["conversion"]["kind"]),
        str(control["conversion"]["value"]),
    )
    for control in PACENOTE_MAPPING_DATA["pluginControls"]
    if isinstance(control.get("conversion"), dict)
}
_PLUGIN_CONTROL_FIXED_CORNER_FLAG_BY_CONVERSION = {
    ("cornerShape", "tightens"): TIGHTENS_FLAG_MASK,
    ("cornerLengthMode", "longer"): LONG_FLAG_MASK,
}
PLUGIN_CONTROL_FIXED_CORNER_FLAGS = {
    code: _PLUGIN_CONTROL_FIXED_CORNER_FLAG_BY_CONVERSION[conversion]
    for code, conversion in PLUGIN_CONTROL_CONVERSIONS.items()
    if conversion in _PLUGIN_CONTROL_FIXED_CORNER_FLAG_BY_CONVERSION
}
PLUGIN_CONTROL_ATTACHED_MODIFIERS = {
    code: value
    for code, (kind, value) in PLUGIN_CONTROL_CONVERSIONS.items()
    if kind == "modifier"
}
PACENOTE_REFERENCE_STATUS_LABELS = {
    "converted": "Converted",
    "conditional": "Conditional",
    "route": "Route point",
    "metadata": "Metadata preserved",
    "loss": "May be lost",
    "unsupported": "Unsupported",
    "unresolved": "Unresolved",
    "recognized": "Recognized",
}


def format_pacenote_summary(stats: dict[str, int]) -> str:
    return (
        f"{stats['rallyPacenotes']}/{stats['rallySourcePacenotes']} converted, "
        f"{stats['rallySkippedPacenotes']} skipped; "
        f"{stats['rallyCustomPacenotes']} custom, "
        f"{stats['rallyInferredModifiers']} inferred modifiers; "
        f"{stats['rallyDistanceMarkersUsed']} distance markers"
    )


@dataclass(frozen=True)
class NoteDefinition:
    name: str
    direction: int = 0
    descriptor: str | None = None
    nominal_length: float = 2.0
    modifier: str | None = None
    caution: int = 0
    custom: bool = False
    spoken: bool = True
    audio_files: tuple[str, ...] = ()
    config_file: str | None = None
    source_name: str | None = None
    corner_shape: str | None = None
    corner_length_mode: str | None = None
    corner_risk_intensity: int = 0
    corner_caution: int = 0
    corner_modifier: CornerModifier | None = None

    @property
    def is_corner(self) -> bool:
        return self.direction != 0


@dataclass(frozen=True)
class CornerModifier:
    shape: str | None = None
    length_mode: str | None = None
    risk_intensity: int = 0
    caution: int = 0


@dataclass(frozen=True)
class PluginControl:
    code: int
    kind: str
    raw_payload: int | None
    value: float | int | None
    unit: str | None
    scope: str
    sticky: bool = False
    no_link: bool = False


@dataclass(frozen=True)
class PacenoteCatalog:
    definitions: dict[int, NoteDefinition]
    config_path: Path | None
    issues: tuple[str, ...]
    flag_modifiers: dict[int, str]
    dropped_flag_modifiers: dict[int, str]
    corner_flag_effects: dict[int, CornerModifier]
    known_flag_extra: int
    control_modifiers: dict[int, str]


@dataclass(frozen=True)
class PacenoteConversion:
    rbr_entry: str
    beamng_entry: str


@dataclass(frozen=True)
class PacenoteReferenceEntry:
    group: str
    source: str
    entry_type: str
    status: str
    result: str
    detail: str
    override_key: str | None = None
    default_value: dict[str, object] | None = field(default=None, hash=False)


@dataclass(frozen=True)
class PacenoteReferenceGroup:
    key: str
    title: str
    summary: str
    entries: tuple[PacenoteReferenceEntry, ...]


@dataclass(frozen=True)
class PacenoteInstallationReference:
    state: str
    summary: str
    config_path: str | None
    recognized: tuple[PacenoteReferenceEntry, ...]
    unsupported: tuple[PacenoteReferenceEntry, ...]
    issues: tuple[str, ...]


@dataclass(frozen=True)
class RallyNotebookResult:
    notebook: dict[str, object]
    spline: dict[str, object]
    stats: dict[str, int]
    warnings: tuple[str, ...]
    log_lines: tuple[str, ...]
    notes_lines: tuple[str, ...]
    visualizer_records: tuple[dict[str, object], ...]


def decode_plugin_control(
    note_type: int,
    flag: int = 0,
) -> PluginControl | None:
    raw_payload = None
    if note_type in PACENOTE_PLUGIN_CONTROL_NOTE_IDS:
        code = note_type
    else:
        code = note_type & 0xFFFF
        if note_type <= 0xFFFF or code not in PACENOTE_PLUGIN_CONTROL_NOTE_IDS:
            return None
        raw_payload = note_type >> 16
    (
        kind,
        unit,
        scope,
        decode_fixed_point_payload,
        decode_integer_payload,
    ) = PLUGIN_CONTROL_SPECS.get(
        code,
        ("UNKNOWN_PLUGIN_CONTROL", None, "unknown", False, False),
    )
    value = None
    if raw_payload is not None and (
        decode_fixed_point_payload or decode_integer_payload
    ):
        signed_payload = (
            raw_payload
            if raw_payload < 0x8000
            else raw_payload - 0x10000
        )
        value = (
            signed_payload / 100.0
            if decode_fixed_point_payload
            else signed_payload
        )
    sticky = (
        kind in STICKY_CONTROL_KINDS
        and bool(flag & NO_LINK_OR_STICKY_FLAG_MASK)
    )
    return PluginControl(
        code,
        kind,
        raw_payload,
        value,
        unit,
        scope,
        sticky,
        bool(flag & NO_LINK_OR_STICKY_FLAG_MASK) and not sticky,
    )


def known_flag_mask_for_note(note_type: int, flag: int) -> int:
    known_mask = KNOWN_FLAG_MASK
    if flag & EXPLICIT_SOUND_INDEX_FLAG_MASK:
        known_mask |= EXPLICIT_SOUND_INDEX_VALUE_MASK
    return known_mask


def _individual_flag_bits(mask: int) -> tuple[int, ...]:
    mask &= 0xFFFFFFFF
    bits: list[int] = []
    while mask:
        bit = mask & -mask
        bits.append(bit)
        mask &= ~bit
    return tuple(bits)


def _explicit_sound_index(flag: int) -> int | None:
    if not flag & EXPLICIT_SOUND_INDEX_FLAG_MASK:
        return None
    return (
        flag & EXPLICIT_SOUND_INDEX_VALUE_MASK
    ) >> EXPLICIT_SOUND_INDEX_VALUE_SHIFT


def _optional_string(value: object) -> str | None:
    return value if isinstance(value, str) else None


def _legacy_definition(row: dict[str, object]) -> NoteDefinition:
    return NoteDefinition(
        str(row["name"]),
        direction=int(row.get("direction", 0)),
        descriptor=_optional_string(row.get("descriptor")),
        nominal_length=float(row.get("nominalLength", 2.0)),
        modifier=_optional_string(row.get("modifier")),
        caution=int(row.get("caution", 0)),
        spoken=bool(row.get("spoken", True)),
    )


LEGACY_DEFINITIONS = {
    int(row["id"]): _legacy_definition(row)
    for row in PACENOTE_MAPPING_DATA["legacyDefinitions"]
}


CUSTOM_ALIASES = {
    str(name): (
        _optional_string(mapping.get("modifier")),
        int(mapping.get("caution", 0)),
    )
    for mapping in PACENOTE_MAPPING_DATA["customAliases"]
    for name in mapping["names"]
}

CUSTOM_CORNER_MODIFIERS = {
    str(name): CornerModifier(
        shape=_optional_string(mapping.get("shape")),
        length_mode=_optional_string(mapping.get("lengthMode")),
        risk_intensity=int(mapping.get("riskIntensity", 0)),
        caution=int(mapping.get("caution", 0)),
    )
    for mapping in PACENOTE_MAPPING_DATA["cornerModifiers"]
    for name in mapping["names"]
}

CUSTOM_CORNERS = {
    str(name): (
        int(mapping["direction"]),
        float(mapping["nominalLength"]),
        _optional_string(mapping.get("descriptor")),
    )
    for mapping in PACENOTE_MAPPING_DATA["customCorners"]
    for name in mapping["names"]
}


FLAG_MODIFIERS = {
    int(flag["mask"]): str(flag["conversion"]["value"])
    for flag in PACENOTE_MAPPING_DATA["flags"]
    if flag.get("status") == "supported"
    and flag.get("conversion", {}).get("kind") == "modifier"
}
UNSUPPORTED_FLAG_MODIFIERS = {
    int(flag["mask"]): str(flag["conversion"]["value"])
    for flag in PACENOTE_MAPPING_DATA["flags"]
    if flag.get("status") == "unsupported"
    and flag.get("conversion", {}).get("kind") == "droppedModifier"
}
UNSUPPORTED_BEAMNG_MODIFIERS = frozenset(UNSUPPORTED_FLAG_MODIFIERS.values())


def _modifier_slot_kind(is_corner: bool) -> str:
    return "post-corner" if is_corner else "pre-corner"


def _modifier_slots_full_issue(modifier: str, *, is_corner: bool) -> str:
    if modifier in UNSUPPORTED_BEAMNG_MODIFIERS:
        return UNSUPPORTED_BEAMNG_MODIFIER_PREFIX + modifier
    return (
        MODIFIER_SLOTS_FULL_PREFIX
        + _modifier_slot_kind(is_corner)
        + MODIFIER_SLOTS_FULL_SEPARATOR
        + modifier
    )


def _modifier_slots_full_reason(modifier: str, *, is_corner: bool) -> str:
    if modifier in UNSUPPORTED_BEAMNG_MODIFIERS:
        return "unsupported BeamNG modifier"
    return (
        MODIFIER_SLOTS_FULL_PREFIX
        + _modifier_slot_kind(is_corner)
        + " modifier slots are full"
    )
KNOWN_LOW_FLAG_BITS = frozenset(
    int(flag["mask"])
    for flag in PACENOTE_MAPPING_DATA["flags"]
    if flag.get("status") == "supported" and int(flag["mask"]) < 0x100
)
KNOWN_FLAG_MASK = sum(
    int(flag["mask"])
    for flag in PACENOTE_MAPPING_DATA["flags"]
    if flag.get("status") in {"supported", "unsupported", "scoped"}
)

BEAMNG_MODIFIERS = tuple(PACENOTE_MAPPING_DATA["beamngModifiers"])
BEAMNG_CORNER_DESCRIPTORS = tuple(PACENOTE_MAPPING_DATA["beamngCornerDescriptors"])
BEAMNG_CORNER_SHAPES = tuple(PACENOTE_MAPPING_DATA["beamngCornerShapes"])
BEAMNG_CORNER_LENGTH_MODES = tuple(PACENOTE_MAPPING_DATA["beamngCornerLengthModes"])
SUPPORTED_STRUCTURED_TYPES = frozenset({"caution", "corner", *BEAMNG_MODIFIERS})
_FLAGS_BY_MASK = {int(flag["mask"]): flag for flag in PACENOTE_MAPPING_DATA["flags"]}


def _flag_corner_effect(flag: dict[str, object]) -> CornerModifier | None:
    conversion = flag.get("conversion")
    if flag.get("status") != "supported" or not isinstance(conversion, dict):
        return None
    kind = conversion.get("kind")
    if kind == "cornerShape":
        return CornerModifier(shape=str(conversion["value"]))
    if kind == "tightensBad":
        return CornerModifier(shape="tightens", caution=1)
    if kind == "cornerLengthMode":
        return CornerModifier(length_mode=str(conversion["value"]))
    return None


CORNER_FLAG_EFFECTS = {
    int(flag["mask"]): effect
    for flag in PACENOTE_MAPPING_DATA["flags"]
    if (effect := _flag_corner_effect(flag)) is not None
}
LONG_CALL_MODIFIERS = {
    str(call): str(modifier)
    for call, modifier in _FLAGS_BY_KEY["LONG"].get("callModifiers", {}).items()
}
UNRESOLVED_FLAG_MASKS = frozenset(
    int(flag["mask"])
    for flag in PACENOTE_MAPPING_DATA["flags"]
    if flag.get("status") == "unresolved"
)
EDITABLE_FLAG_MASKS = frozenset(
    {
        *FLAG_MODIFIERS,
        *UNSUPPORTED_FLAG_MODIFIERS,
        *CORNER_FLAG_EFFECTS,
        *UNRESOLVED_FLAG_MASKS,
    }
)


def _fixed_corner_flag_mask(catalog: PacenoteCatalog) -> int:
    mask = 0
    for bit in catalog.corner_flag_effects:
        mask |= bit
    return mask


def _long_call_modifier(
    definition: NoteDefinition,
    flag: int,
    catalog: PacenoteCatalog,
) -> str | None:
    if (
        not flag & LONG_FLAG_MASK
        or catalog.corner_flag_effects.get(LONG_FLAG_MASK)
        != CORNER_FLAG_EFFECTS[LONG_FLAG_MASK]
    ):
        return None
    return LONG_CALL_MODIFIERS.get(definition.modifier or "")


def _corner_flag_name(mask: int) -> str:
    return FIXED_CORNER_FLAG_NAMES.get(mask) or _humanize_pacenote_text(
        str(_FLAGS_BY_MASK[mask]["key"])
    )


def pacenote_override_key(note_id: int, definition: NoteDefinition) -> str | None:
    if note_id in LEGACY_DEFINITIONS:
        role = _LEGACY_ROWS_BY_ID[note_id].get("role")
        return f"legacy:{note_id}" if role in {None, "unmapped"} else None
    if definition.custom and definition.source_name:
        return f"call:{definition.source_name}"
    return None


def _corner_change_record(effect: CornerModifier) -> dict[str, object]:
    return {
        "shape": effect.shape or "",
        "lengthMode": effect.length_mode or "",
        "riskIntensity": effect.risk_intensity,
        "caution": effect.caution,
    }


def _corner_change(record: dict[str, object]) -> CornerModifier:
    return CornerModifier(
        shape=str(record["shape"]) or None,
        length_mode=str(record["lengthMode"]) or None,
        risk_intensity=int(record["riskIntensity"]),
        caution=int(record["caution"]),
    )


def pacenote_definition_record(definition: NoteDefinition) -> dict[str, object]:
    if definition.is_corner:
        return {
            "kind": "corner",
            "direction": definition.direction,
            "descriptor": definition.descriptor or "",
            "length": definition.nominal_length,
            "shape": definition.corner_shape or "",
            "lengthMode": definition.corner_length_mode or "",
            "riskIntensity": definition.corner_risk_intensity,
            "caution": max(definition.caution, definition.corner_caution),
        }
    if definition.corner_modifier is not None:
        return {"kind": "cornerModifier", **_corner_change_record(definition.corner_modifier)}
    if definition.modifier or definition.caution:
        return {
            "kind": "call",
            "modifier": definition.modifier or "",
            "caution": definition.caution,
        }
    return {"kind": "none"}


def _definition_with_record(
    definition: NoteDefinition,
    record: dict[str, object],
) -> NoteDefinition:
    base = replace(
        definition,
        direction=0,
        descriptor=None,
        modifier=None,
        caution=0,
        spoken=True,
        corner_shape=None,
        corner_length_mode=None,
        corner_risk_intensity=0,
        corner_caution=0,
        corner_modifier=None,
    )
    kind = record["kind"]
    if kind == "corner":
        change = _corner_change(record)
        return replace(
            base,
            direction=int(record["direction"]),
            descriptor=str(record["descriptor"]) or None,
            nominal_length=float(record["length"]),
            corner_shape=change.shape,
            corner_length_mode=change.length_mode,
            corner_risk_intensity=change.risk_intensity,
            corner_caution=change.caution,
        )
    if kind == "cornerModifier":
        return replace(base, spoken=False, corner_modifier=_corner_change(record))
    if kind == "call":
        modifier = str(record["modifier"]) or None
        caution = int(record["caution"])
        return replace(
            base,
            modifier=modifier,
            caution=caution,
            spoken=bool(modifier or caution),
        )
    return replace(base, spoken=False)


def pacenote_flag_record(mask: int) -> dict[str, object]:
    if mask in FLAG_MODIFIERS:
        return {"kind": "modifier", "modifier": FLAG_MODIFIERS[mask]}
    if mask in CORNER_FLAG_EFFECTS:
        return {"kind": "cornerChange", **_corner_change_record(CORNER_FLAG_EFFECTS[mask])}
    return {"kind": "none"}


def _flag_effect_maps(
    overrides: dict[str, dict[str, object]],
) -> tuple[dict[int, str], dict[int, str], dict[int, CornerModifier], int]:
    modifiers: dict[int, str] = {}
    dropped: dict[int, str] = {}
    corner_effects: dict[int, CornerModifier] = {}
    known_extra = 0
    for flag in PACENOTE_MAPPING_DATA["flags"]:
        mask = int(flag["mask"])
        record = (
            overrides.get(f"flag:{flag['key']}")
            if mask in EDITABLE_FLAG_MASKS
            else None
        )
        if record is None:
            if mask in FLAG_MODIFIERS:
                modifiers[mask] = FLAG_MODIFIERS[mask]
            if mask in UNSUPPORTED_FLAG_MODIFIERS:
                dropped[mask] = UNSUPPORTED_FLAG_MODIFIERS[mask]
            if mask in CORNER_FLAG_EFFECTS:
                corner_effects[mask] = CORNER_FLAG_EFFECTS[mask]
            continue
        if record["kind"] == "modifier":
            modifiers[mask] = str(record["modifier"])
        elif record["kind"] == "cornerChange":
            corner_effects[mask] = _corner_change(record)
        if mask in UNRESOLVED_FLAG_MASKS and record["kind"] != "none":
            known_extra |= mask
    return modifiers, dropped, corner_effects, known_extra


def pacenote_control_key(code: int) -> str:
    return f"control:0x{code:X}"


def _control_modifier_map(overrides: dict[str, dict[str, object]]) -> dict[int, str]:
    result: dict[int, str] = {}
    for code, modifier in PLUGIN_CONTROL_ATTACHED_MODIFIERS.items():
        record = overrides.get(pacenote_control_key(code))
        if record is None:
            result[code] = modifier
        elif record["kind"] == "modifier":
            result[code] = str(record["modifier"])
    return result


def _corner_change_parts(record: dict[str, object]) -> list[str]:
    parts: list[str] = []
    if record["shape"]:
        parts.append(_humanize_pacenote_text(str(record["shape"])))
    if record["lengthMode"] == "skip":
        parts.append("no length call")
    elif record["lengthMode"]:
        parts.append(str(record["lengthMode"]))
    risk = int(record["riskIntensity"])
    if risk:
        parts.append("one step more severe" if risk > 0 else "one step less severe")
    if record["caution"]:
        parts.append(f"caution level {record['caution']}")
    return parts


PACENOTE_OVERRIDE_KIND_LABELS = {
    "corner": "Corner",
    "cornerModifier": "Change the adjacent corner",
    "cornerChange": "Change the corner",
    "call": "Modifier and caution",
    "modifier": "Modifier",
    "none": "Not converted",
}


def pacenote_field_choices(field: str) -> tuple[tuple[str, object], ...]:
    def named(values: tuple[str, ...]) -> tuple[tuple[str, object], ...]:
        return tuple(sorted((_humanize_pacenote_text(value), value) for value in values))

    return {
        "modifier": (("(none)", ""), *named(BEAMNG_MODIFIERS)),
        "caution": (("none", 0), ("level 1", 1), ("level 2", 2), ("level 3", 3)),
        "direction": (("left", -1), ("right", 1)),
        "descriptor": (("(none)", ""), *named(BEAMNG_CORNER_DESCRIPTORS)),
        "riskIntensity": (
            ("one step less severe", -1),
            ("as measured", 0),
            ("one step more severe", 1),
        ),
        "shape": (("(none)", ""), *named(BEAMNG_CORNER_SHAPES)),
        "lengthMode": (
            ("(none)", ""),
            ("shorter", "shorter"),
            ("longer", "longer"),
            ("no length call", "skip"),
        ),
    }[field]


def pacenote_override_result(record: dict[str, object]) -> str:
    kind = record["kind"]
    if kind == "corner":
        direction = "left" if int(record["direction"]) < 0 else "right"
        descriptor = (
            f" {_humanize_pacenote_text(str(record['descriptor']))}"
            if record["descriptor"]
            else ""
        )
        return ", ".join(
            [
                f"Creates a {direction}{descriptor} corner ending "
                f"{float(record['length']):g} m after the call",
                *_corner_change_parts(record),
            ]
        )
    if kind in {"cornerModifier", "cornerChange"}:
        parts = _corner_change_parts(record)
        return "Changes the corner: " + (", ".join(parts) if parts else "nothing")
    if kind in {"call", "modifier"}:
        parts = []
        if record["modifier"]:
            parts.append(f'adds the "{_humanize_pacenote_text(str(record["modifier"]))}" modifier')
        if record.get("caution"):
            parts.append(f"adds caution level {record['caution']}")
        if parts:
            text = " and ".join(parts)
            return text[0].upper() + text[1:]
    return "Not converted"


def _definition_conversion(
    definition: NoteDefinition,
) -> tuple[str, str]:
    if definition.is_corner:
        entry = (
            f"end=min({definition.nominal_length:g} m, available route distance); "
            "waypoint radius=8 m; direction from route"
        )
        if definition.descriptor:
            entry = (
                f'descriptor="{definition.descriptor}"; '
                f"end=min({definition.nominal_length:g} m, available route distance); "
                "waypoint radius=8 m; direction from route"
            )
        return "Corner", entry
    if definition.modifier:
        return "Modifier", f'type="{definition.modifier}"; slot 2 or 3'
    return "Caution", f'type="caution", level={definition.caution}; slot 1'


def supported_pacenote_conversions() -> tuple[PacenoteConversion, ...]:
    rows: list[PacenoteConversion] = []
    for note_id, definition in sorted(LEGACY_DEFINITIONS.items()):
        timing = LEGACY_CONVERSION_PRESENTATIONS.get(note_id)
        if timing is not None:
            category, entry = timing
        elif not definition.spoken:
            continue
        else:
            category, entry = _definition_conversion(definition)
        rows.append(
            PacenoteConversion(
                f'[ID {note_id}] "{definition.name}"',
                f"[{category.casefold()}] {entry}",
            )
        )
    for name, (modifier, caution) in sorted(CUSTOM_ALIASES.items()):
        definition = NoteDefinition(
            _humanize(name),
            modifier=modifier,
            caution=caution,
            custom=True,
        )
        category, entry = _definition_conversion(definition)
        rows.append(
            PacenoteConversion(
                f'PLUGIN: "{name}"',
                f"[{category.casefold()}] {entry}",
            )
        )
    for name, modifier in sorted(CUSTOM_CORNER_MODIFIERS.items()):
        details = []
        if modifier.shape:
            details.append(f'shape="{modifier.shape}"')
        if modifier.length_mode:
            details.append(f'length mode="{modifier.length_mode}"')
        if modifier.risk_intensity:
            details.append(
                "one intensity step "
                + ("higher" if modifier.risk_intensity > 0 else "lower")
            )
        if modifier.caution:
            details.append(f"caution level={modifier.caution}")
        rows.append(
            PacenoteConversion(
                f'PLUGIN: "{name}"',
                "[corner modifier] applies to an adjacent corner; "
                + "; ".join(details),
            )
        )
    for name, (direction, length, descriptor) in sorted(CUSTOM_CORNERS.items()):
        definition = NoteDefinition(
            _humanize(name),
            direction=direction,
            nominal_length=length,
            descriptor=descriptor,
            custom=True,
        )
        category, entry = _definition_conversion(definition)
        rows.append(
            PacenoteConversion(
                f'PLUGIN: "{name}"',
                f"[{category.casefold()}] {entry}",
            )
        )
    return tuple(rows)


def _humanize(name: str) -> str:
    return " ".join(name.strip().replace("-", "_").split("_")).casefold()


def _case_value(section: configparser.SectionProxy, name: str) -> str | None:
    expected = name.casefold()
    return next(
        (
            value.strip()
            for key, value in section.items()
            if key.casefold() == expected
        ),
        None,
    )


def _include_sort_key(item: tuple[str, str]) -> tuple[int, str]:
    key = item[0].casefold()
    suffix = key.removeprefix("file")
    return (int(suffix) if suffix.isdigit() else -1, key)


def _custom_definition(name: str) -> NoteDefinition:
    corner = CUSTOM_CORNERS.get(name)
    if corner is not None:
        direction, nominal_length, descriptor = corner
        return NoteDefinition(
            _humanize(name),
            direction=direction,
            descriptor=descriptor,
            nominal_length=nominal_length,
            custom=True,
            source_name=name,
        )
    corner_modifier = CUSTOM_CORNER_MODIFIERS.get(name)
    if corner_modifier is not None:
        return NoteDefinition(
            _humanize(name),
            custom=True,
            spoken=False,
            corner_modifier=corner_modifier,
            source_name=name,
        )
    modifier, caution = CUSTOM_ALIASES.get(name, (None, 0))
    return NoteDefinition(
        _humanize(name),
        modifier=modifier,
        caution=caution,
        custom=True,
        spoken=modifier is not None or caution > 0,
        source_name=name,
    )


def _audio_files(section: configparser.SectionProxy) -> tuple[str, ...]:
    return tuple(
        value.strip()
        for key, value in sorted(
            section.items(),
            key=lambda item: _include_sort_key(item),
        )
        if key.casefold().startswith("snd") and value.strip()
    )


def _diagnostic_details(
    note_id: int,
    definition: NoteDefinition | None,
    notes: list[Pacenote],
) -> str:
    name = definition.name if definition is not None else "unknown"
    parts = [f"RBR pacenote ID {note_id} ({name})"]
    if definition is not None and definition.audio_files:
        parts.append(f"audio: {', '.join(definition.audio_files)}")
    if definition is not None and definition.config_file:
        parts.append(f"config: {definition.config_file}")
    distances = sorted(note.distance for note in notes)
    if distances:
        parts.append(
            "distances: "
            + ", ".join(f"{distance:.1f} m" for distance in distances)
        )
    flags = sorted({note.flag for note in notes if note.flag})
    if flags:
        parts.append(
            "flags: " + ", ".join(hex(flag) for flag in flags)
        )
    return "; ".join(parts)


def _pacenote_failure(
    message: str,
    note: Pacenote,
    definition: NoteDefinition | None,
    reason: str,
) -> str:
    return (
        f"{PACENOTE_FAILURE_PREFIX}{message} "
        f"{_diagnostic_details(note.note_type, definition, [note])}; {reason}"
    )


def _pacenote_warning(
    message: str,
    note: Pacenote,
    definition: NoteDefinition | None,
) -> str:
    return (
        f"{PACENOTE_WARNING_PREFIX}{message}; "
        f"{_diagnostic_details(note.note_type, definition, [note])}"
    )


def _compact_log_value(value: object) -> str:
    return json.dumps(
        value,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _humanize_pacenote_text(value: str) -> str:
    text: list[str] = []
    for index, character in enumerate(value):
        if character == "_":
            text.append(" ")
        elif (
            index
            and character.isupper()
            and value[index - 1].islower()
        ):
            text.extend((" ", character.lower()))
        else:
            text.append(character.lower())
    return (
        "".join(text)
        .replace("dontcut", "don't cut")
        .replace("dont ", "don't ")
    )


def _human_distance(distance: float) -> str:
    return f"{distance:,.3f}".rstrip("0").rstrip(".") + " m"


def _human_flag_modifiers(flag: int) -> list[str]:
    modifiers = [
        _humanize_pacenote_text(modifier)
        for bit, modifier in FLAG_MODIFIERS.items()
        if flag & bit
    ]
    if flag & TIGHTENS_FLAG_MASK:
        modifiers.append("tightens")
    if flag & TIGHTENS_BAD_FLAG_MASK:
        modifiers.append("tightens badly")
    if flag & LONG_FLAG_MASK:
        modifiers.append("long")
    modifiers.extend(
        modifier
        for bit, modifier in UNSUPPORTED_FLAG_MODIFIERS.items()
        if flag & bit
    )
    if flag & NO_LINK_OR_STICKY_FLAG_MASK:
        modifiers.append("no link")
    return modifiers


def _fixed_corner_flag_names(flag: int) -> list[str]:
    return [
        _corner_flag_name(bit)
        for bit in _individual_flag_bits(flag)
    ]


def _human_control_description(control: PluginControl) -> str:
    if control.kind == "UNKNOWN_PLUGIN_CONTROL":
        return f"Plugin control {hex(control.code)}"
    if control.kind in {"INTO", "AND", "ONTO"}:
        return f"{control.kind.lower()} link"
    description = _humanize_pacenote_text(control.kind)
    if control.kind == "SOUND_INDEX" and control.value is not None:
        description += f" {control.value:+g}"
    elif control.value is not None and control.unit is not None:
        description += f" {control.value:+g} {control.unit}"
    if control.sticky:
        description += " (sticky)"
    elif control.no_link:
        description += " (no link)"
    return description


def _is_pacenote_config_header_fragment(note_type: int) -> bool:
    return (
        note_type > 0xFFFF
        and (note_type & 0xFFFF) in {0xFFE, 0xFFF}
    )


def pacenote_config_header_indexes(
    records: list[tuple[int, float, int]],
) -> frozenset[int]:
    indexes: list[int] = []
    codes: set[int] = set()
    previous_distance = -float("inf")
    start_index = (
        1
        if records
        and records[0][0] == START_NOTE_ID
        and records[0][2] == 0
        else 0
    )
    for index, (note_type, distance, flag) in enumerate(
        records[start_index:],
        start=start_index,
    ):
        payload = note_type >> 16
        payload_bytes = payload.to_bytes(2, "little")
        if (
            flag != 0
            or not _is_pacenote_config_header_fragment(note_type)
            or not 0 <= distance <= PACENOTE_CONFIG_HEADER_MAX_DISTANCE
            or distance < previous_distance
            or not all(
                byte == 0 or 0x20 <= byte <= 0x7E
                for byte in payload_bytes
            )
        ):
            break
        indexes.append(index)
        codes.add(note_type & 0xFFFF)
        previous_distance = distance
    return (
        frozenset(indexes)
        if len(indexes) >= 2 and codes == {0xFFE, 0xFFF}
        else frozenset()
    )


def _target_call_name(
    stage: RbrStage,
    catalog: PacenoteCatalog,
    source_index: int,
) -> str:
    if not 0 <= source_index < len(stage.pacenotes):
        return f"source {source_index}"
    note = stage.pacenotes[source_index]
    definition = catalog.definitions.get(note.note_type)
    if definition is not None:
        return definition.name
    return f"source {source_index}"


def _human_issue(issue: str) -> str:
    prefix = "unknown RBR pacenote flag bits "
    if issue.startswith(prefix):
        return "unknown flag " + issue.removeprefix(prefix)
    if issue.startswith(MODIFIER_SLOTS_FULL_PREFIX):
        slot_kind, separator, modifier = issue.removeprefix(
            MODIFIER_SLOTS_FULL_PREFIX
        ).partition(MODIFIER_SLOTS_FULL_SEPARATOR)
        if separator:
            return (
                "dropped "
                + modifier
                + " (BeamNG "
                + slot_kind
                + " modifier slots are full)"
            )
    if issue.startswith(UNSUPPORTED_BEAMNG_MODIFIER_PREFIX):
        return (
            "dropped "
            + issue.removeprefix(UNSUPPORTED_BEAMNG_MODIFIER_PREFIX)
            + " (unsupported BeamNG modifier)"
        )
    return issue


def _human_source_description(
    stage: RbrStage,
    catalog: PacenoteCatalog,
    note: Pacenote,
    definition: NoteDefinition | None,
    disposition: str,
    attached_corner_modifiers: dict[int, int],
    attached_fixed_corner_flags: dict[int, int],
    attached_plugin_control_modifiers: dict[int, int],
    plugin_controls: dict[int, PluginControl],
    preserved_plugin_controls: dict[int, int],
    preserved_start_markers: dict[int, int],
    config_header_note_ids: set[int],
    source_issues: dict[int, list[str]],
) -> str:
    note_id = id(note)
    control = plugin_controls.get(note_id)
    fixed_target_source_index = attached_fixed_corner_flags.get(note_id)
    target_source_index = attached_corner_modifiers.get(note_id)
    if target_source_index is None:
        target_source_index = fixed_target_source_index
    if target_source_index is None:
        target_source_index = attached_plugin_control_modifiers.get(note_id)
    if note_id in config_header_note_ids:
        description = "Pacenote config header"
    elif target_source_index is not None:
        target_description = _target_call_name(
            stage,
            catalog,
            target_source_index,
        )
        if fixed_target_source_index is not None:
            target_description += f" (source {target_source_index})"
        source_description = (
            definition.name
            if definition is not None
            else _humanize_pacenote_text(control.kind)
            if control is not None
            else "corner modifier"
        )
        description = (
            f"{source_description} → {target_description}"
        )
    elif control is not None:
        description = _human_control_description(control)
        target_source_index = preserved_plugin_controls.get(note_id)
        if target_source_index is not None:
            description += (
                f" → {_target_call_name(stage, catalog, target_source_index)}"
            )
    elif note_id in preserved_start_markers:
        description = (
            "initial distance marker → "
            + _target_call_name(
                stage,
                catalog,
                preserved_start_markers[note_id],
            )
        )
    elif note.note_type == DISTANCE_MARKER_NOTE_ID:
        description = "distance marker"
    elif disposition == "outside-rally-range":
        description = "outside rally range"
    elif definition is not None:
        description = definition.name
    else:
        description = f"unknown pacenote {hex(note.note_type)}"

    if (
        definition is not None
        and control is None
        and note.note_type != DISTANCE_MARKER_NOTE_ID
    ):
        modifiers = _human_flag_modifiers(note.flag)
        if modifiers:
            description += ", " + ", ".join(modifiers)
    issues = [
        _human_issue(issue)
        for issue in source_issues.get(note_id, [])
    ]
    if issues:
        description += " (" + "; ".join(issues) + ")"
    return description


def _human_beamng_description(entry: dict[str, object]) -> str:
    name = str(entry.get("name", "generated pacenote"))
    _, separator, base = name.partition(": ")
    description = base if separator else name
    structured = entry.get("structured")
    items = structured.get("items") if isinstance(structured, dict) else None
    if not isinstance(items, dict):
        return description

    additions: list[str] = []

    def add(value: str) -> None:
        if value and value.casefold() not in {
            item.casefold() for item in [description, *additions]
        }:
            additions.append(value)

    for item in items.values():
        if not isinstance(item, dict):
            continue
        item_type = item.get("type")
        if not isinstance(item_type, str):
            continue
        if item_type == "corner":
            shape = item.get("shape")
            if isinstance(shape, str):
                add(_humanize_pacenote_text(shape))
            length_mode = item.get("lengthMode")
            if length_mode == "shorter":
                add("short")
            elif length_mode == "longer":
                add("long")
            risk_intensity = item.get("riskIntensity")
            if risk_intensity == -1:
                add("minus")
            elif risk_intensity == 1:
                add("plus")
        elif item_type == "caution":
            level = item.get("level")
            add(
                "caution"
                if level in {None, 1}
                else f"caution {level}"
            )
        else:
            add(_humanize_pacenote_text(item_type))
    return description if not additions else description + ", " + ", ".join(additions)


def _human_log_line(
    prefix: str,
    distance: float,
    description: str,
    variant: str,
    *,
    at_start: bool = False,
) -> str:
    context = "" if variant == "base" else f"{variant} "
    location = "start" if at_start else _human_distance(distance)
    return f"{prefix} {context}{location}: {description}"


def _format_pacenote_log_lines(
    stage: RbrStage,
    catalog: PacenoteCatalog,
    candidate_note_ids: set[int],
    ignored_reasons: dict[int, list[str]],
    attached_corner_modifiers: dict[int, int],
    attached_fixed_corner_flags: dict[int, int],
    attached_plugin_control_modifiers: dict[int, int],
    plugin_controls: dict[int, PluginControl],
    preserved_plugin_controls: dict[int, int],
    preserved_start_markers: dict[int, int],
    config_header_note_ids: set[int],
    used_distance_marker_ids: set[int],
    source_issues: dict[int, list[str]],
    convertible: list[tuple[Pacenote, NoteDefinition]],
    notebook_notes: list[dict[str, object]],
) -> tuple[
    tuple[str, ...],
    tuple[str, ...],
    tuple[dict[str, object], ...],
]:
    variant = stage.source_variant or "base"
    source_indices = {
        id(note): index
        for index, note in enumerate(stage.pacenotes)
    }
    generated_by_source_index: dict[int, list[int]] = defaultdict(list)
    for entry_index, entry in enumerate(notebook_notes):
        metadata = entry.get("metadata")
        if not isinstance(metadata, dict):
            continue
        source_indexes = metadata.get("rbrSourceIndexes")
        if not isinstance(source_indexes, list):
            continue
        for source_index in source_indexes:
            if isinstance(source_index, int):
                generated_by_source_index[source_index].append(entry_index)
    generated = {
        id(note): (
            generated_by_source_index[source_indices[id(note)]][0],
            definition,
            notebook_notes[generated_by_source_index[source_indices[id(note)]][0]],
        )
        for note, definition in convertible
        if source_indices[id(note)] in generated_by_source_index
    }
    lines: list[str] = []
    notes_lines: list[str] = []
    source_records: list[dict[str, object]] = []
    beamng_records: list[dict[str, object]] = []
    for source_index, note in sorted(
        enumerate(stage.pacenotes),
        key=lambda item: (item[1].distance, item[0]),
    ):
        definition = catalog.definitions.get(note.note_type)
        reasons = ignored_reasons.get(id(note), [])
        generated_entry = generated.get(id(note))
        control = plugin_controls.get(id(note))
        if reasons:
            prefix = "[RBR-IGNORED]"
            disposition = "ignored"
        elif generated_entry is not None:
            prefix = "[RBR]"
            if id(note) in source_issues:
                disposition = "generated-with-issues"
            elif id(note) in attached_fixed_corner_flags:
                disposition = "generated-with-attached-fixed-corner-modifier"
            else:
                disposition = "generated"
        elif id(note) in attached_corner_modifiers:
            prefix = "[RBR]"
            disposition = "attached-corner-modifier"
        elif id(note) in attached_fixed_corner_flags:
            prefix = "[RBR]"
            disposition = "attached-fixed-corner-modifier"
        elif id(note) in attached_plugin_control_modifiers:
            prefix = "[RBR]"
            disposition = "attached-plugin-control-modifier"
        elif id(note) in preserved_plugin_controls:
            prefix = "[RBR]"
            disposition = "preserved-plugin-control"
        elif id(note) in preserved_start_markers:
            prefix = "[RBR]"
            disposition = "preserved-start-distance-marker"
        elif id(note) in config_header_note_ids:
            prefix = "[RBR]"
            disposition = "config-header"
        elif note.note_type == START_NOTE_ID:
            prefix = "[RBR]"
            disposition = "start-marker"
        elif note.note_type == SPLIT_NOTE_ID:
            prefix = "[RBR]"
            disposition = "split-marker"
        elif note.note_type == STOP_CONTROL_NOTE_ID:
            prefix = "[RBR]"
            disposition = "stop-marker"
        elif id(note) not in candidate_note_ids:
            prefix = "[RBR]"
            disposition = "outside-rally-range"
        elif note.note_type == DISTANCE_MARKER_NOTE_ID:
            prefix = "[RBR]"
            disposition = (
                "distance-marker-used"
                if id(note) in used_distance_marker_ids
                else "distance-marker-recorded"
            )
        else:
            prefix = "[RBR]"
            disposition = "recorded"
        fields = [
            f"sourceIndex={source_index}",
            f"variant={_compact_log_value(variant)}",
            f"type={note.note_type}",
            f"typeHex={hex(note.note_type)}",
            f"distance={note.distance:.9g}m",
            f"flags={hex(note.flag)}",
            f"call={_compact_log_value(definition.name if definition else control.kind if control else 'unknown')}",
            f"custom={str(bool(definition and definition.custom)).lower()}",
            f"audio={_compact_log_value(list(definition.audio_files) if definition else [])}",
            f"config={_compact_log_value(definition.config_file if definition else None)}",
            f"disposition={_compact_log_value(disposition)}",
        ]
        if reasons:
            fields.append(f"reason={_compact_log_value(reasons)}")
        target_source_index = attached_corner_modifiers.get(id(note))
        if target_source_index is None:
            target_source_index = attached_fixed_corner_flags.get(id(note))
        if target_source_index is None:
            target_source_index = attached_plugin_control_modifiers.get(id(note))
        if target_source_index is not None:
            fields.append(
                "targetSourceIndex="
                + str(target_source_index)
            )
        if control is not None:
            fields.extend(
                (
                    f"controlCode={hex(control.code)}",
                    f"controlKind={_compact_log_value(control.kind)}",
                    f"controlRawPayload={_compact_log_value(control.raw_payload)}",
                    f"controlValue={_compact_log_value(control.value)}",
                    f"controlUnit={_compact_log_value(control.unit)}",
                    f"controlScope={_compact_log_value(control.scope)}",
                )
            )
        if id(note) in preserved_plugin_controls:
            fields.append(
                "targetSourceIndex="
                + str(preserved_plugin_controls[id(note)])
            )
        if id(note) in preserved_start_markers:
            fields.append(
                "targetSourceIndex="
                + str(preserved_start_markers[id(note)])
            )
        if id(note) in source_issues:
            fields.append(
                f"issues={_compact_log_value(source_issues[id(note)])}"
            )
        lines.append(f"{prefix} {' '.join(fields)}")
        description = _human_source_description(
            stage,
            catalog,
            note,
            definition,
            disposition,
            attached_corner_modifiers,
            attached_fixed_corner_flags,
            attached_plugin_control_modifiers,
            plugin_controls,
            preserved_plugin_controls,
            preserved_start_markers,
            config_header_note_ids,
            source_issues,
        )
        if reasons:
            description += " — " + "; ".join(reasons)
        target_source_indexes = list(
            dict.fromkeys(
                target_source_index
                for target_source_index in (
                    attached_corner_modifiers.get(id(note)),
                    attached_fixed_corner_flags.get(id(note)),
                    attached_plugin_control_modifiers.get(id(note)),
                    preserved_plugin_controls.get(id(note)),
                    preserved_start_markers.get(id(note)),
                )
                if target_source_index is not None
            )
        )
        beamng_entry_indexes = list(
            dict.fromkeys(
                entry_index
                for target_source_index in [
                    source_index,
                    *target_source_indexes,
                ]
                for entry_index in generated_by_source_index.get(
                    target_source_index,
                    [],
                )
            )
        )
        source_records.append(
            {
                "mode": "rbr",
                "sourceIndex": source_index,
                "distance": note.distance,
                "type": note.note_type,
                "typeHex": hex(note.note_type),
                "flag": note.flag,
                "label": description,
                "cornerDirection": definition.direction if definition else 0,
                "disposition": disposition,
                "ignored": bool(reasons),
                "reasons": list(reasons),
                "issues": list(source_issues.get(id(note), [])),
                "targetSourceIndexes": target_source_indexes,
                "beamngEntryIndexes": beamng_entry_indexes,
                "mapVisible": (
                    id(note) in candidate_note_ids
                    and id(note) not in config_header_note_ids
                    and note.note_type not in TIMING_NOTE_IDS
                ),
            }
        )
        notes_lines.append(
            _human_log_line(
                prefix,
                note.distance,
                description,
                variant,
                at_start=id(note) in config_header_note_ids,
            )
        )
        if generated_entry is None:
            continue
        entry_index, generated_definition, entry = generated_entry
        metadata = entry["metadata"]
        fields = [
            f"entryIndex={entry_index}",
            f"variant={_compact_log_value(variant)}",
            f"sourceIndex={source_index}",
            f"sourceType={note.note_type}",
            f"sourceDistance={note.distance:.9g}m",
            f"endDistance={float(metadata['rbrEndDistance']):.9g}m",
            f"oldId={entry['oldId']}",
            f"pk={_compact_log_value(entry['pk'])}",
            f"name={_compact_log_value(entry['name'])}",
            f"audioMode={entry['audioMode']}",
            f"triggerType={entry['triggerType']}",
            f"slowCorner={str(entry['slowCorner']).lower()}",
            f"slowCornerReleaseType={entry['slowCornerReleaseType']}",
            f"includeLinkWord={str(entry['includeLinkWord']).lower()}",
            f"ignoreDistanceCalls={str(entry['ignoreDistanceCalls']).lower()}",
            f"distanceBeforeModifier={str(entry['distanceBeforeModifier']).lower()}",
            f"isolate={str(entry['isolate']).lower()}",
            f"todo={str(entry['todo']).lower()}",
            f"structured={_compact_log_value(entry['structured'])}",
            f"metadata={_compact_log_value(metadata)}",
            f"waypoints={_compact_log_value(entry['pacenoteWaypoints'])}",
        ]
        lines.append(f"[BEAMNG] {' '.join(fields)}")
        beamng_description = _human_beamng_description(entry)
        notes_lines.append(
            _human_log_line(
                "[BEAMNG]",
                note.distance,
                beamng_description,
                variant,
            )
        )
        beamng_records.append(
            {
                "mode": "beamng",
                "entryIndex": entry_index,
                "sourceIndex": source_index,
                "distance": note.distance,
                "endDistance": float(metadata["rbrEndDistance"]),
                "sourceType": note.note_type,
                "label": beamng_description,
                "cornerDirection": generated_definition.direction,
                "disposition": "generated",
                "ignored": False,
                "contributingSourceIndexes": [],
                "mapVisible": True,
            }
        )

    generated_entry_indexes = {
        entry_index
        for entry_index, _definition, _entry in generated.values()
    }
    for entry_index, entry in enumerate(notebook_notes):
        if entry_index in generated_entry_indexes:
            continue
        metadata = entry.get("metadata")
        if (
            not isinstance(metadata, dict)
            or metadata.get("rbrDistanceOnly") is not True
        ):
            continue
        source_indexes = metadata.get("rbrSourceIndexes")
        if not isinstance(source_indexes, list):
            source_indexes = []
        source_index = next(
            (
                value
                for value in source_indexes
                if isinstance(value, int)
                and 0 <= value < len(stage.pacenotes)
            ),
            None,
        )
        source_note = (
            stage.pacenotes[source_index]
            if source_index is not None
            else None
        )
        fields = [
            f"entryIndex={entry_index}",
            f"variant={_compact_log_value(variant)}",
            f"sourceIndex={_compact_log_value(source_index)}",
            f"sourceType={_compact_log_value(source_note.note_type if source_note else None)}",
            f"sourceDistance={float(metadata['rbrDistance']):.9g}m",
            f"endDistance={float(metadata['rbrEndDistance']):.9g}m",
            f"oldId={entry['oldId']}",
            f"pk={_compact_log_value(entry['pk'])}",
            f"name={_compact_log_value(entry['name'])}",
            f"audioMode={entry['audioMode']}",
            f"triggerType={entry['triggerType']}",
            f"slowCorner={str(entry['slowCorner']).lower()}",
            f"slowCornerReleaseType={entry['slowCornerReleaseType']}",
            f"includeLinkWord={str(entry['includeLinkWord']).lower()}",
            f"ignoreDistanceCalls={str(entry['ignoreDistanceCalls']).lower()}",
            f"distanceBeforeModifier={str(entry['distanceBeforeModifier']).lower()}",
            f"isolate={str(entry['isolate']).lower()}",
            f"todo={str(entry['todo']).lower()}",
            f"structured={_compact_log_value(entry['structured'])}",
            f"metadata={_compact_log_value(metadata)}",
            f"waypoints={_compact_log_value(entry['pacenoteWaypoints'])}",
        ]
        lines.append(f"[BEAMNG] {' '.join(fields)}")
        beamng_description = _human_beamng_description(entry)
        notes_lines.append(
            _human_log_line(
                "[BEAMNG]",
                float(metadata["rbrDistance"]),
                beamng_description,
                variant,
                at_start=True,
            )
        )
        beamng_records.append(
            {
                "mode": "beamng",
                "entryIndex": entry_index,
                "sourceIndex": source_index,
                "distance": float(metadata["rbrDistance"]),
                "endDistance": float(metadata["rbrEndDistance"]),
                "sourceType": source_note.note_type if source_note else None,
                "label": beamng_description,
                "cornerDirection": 0,
                "disposition": "generated",
                "ignored": False,
                "contributingSourceIndexes": [],
                "mapVisible": True,
            }
        )

    contributors_by_entry_index: dict[int, list[int]] = defaultdict(list)
    for source_record in source_records:
        source_index = int(source_record["sourceIndex"])
        for entry_index in source_record["beamngEntryIndexes"]:
            contributors_by_entry_index[int(entry_index)].append(source_index)
    for beamng_record in beamng_records:
        entry_index = int(beamng_record["entryIndex"])
        beamng_record["contributingSourceIndexes"] = (
            contributors_by_entry_index.get(entry_index, [])
        )
    return (
        tuple(lines),
        tuple(notes_lines),
        tuple([*source_records, *beamng_records]),
    )


def _resolve_config_path(root: Path, parent: Path, value: str) -> Path:
    try:
        candidate = current_filesystem().read_path(
            parent / value.replace("\\", "/")
        )
        candidate.relative_to(root)
    except (SandboxViolationError, ValueError):
        raise ValueError(f"Pacenote config include escapes its root: {value!r}")
    return candidate


def _read_plugin_definitions(
    config_root: Path,
    entry: Path,
) -> tuple[dict[int, NoteDefinition], list[str]]:
    filesystem = current_filesystem()
    root = filesystem.read_path(config_root)
    pending = [filesystem.read_path(entry)]
    visited: set[Path] = set()
    result: dict[int, NoteDefinition] = {}
    issues: list[str] = []
    while pending:
        path = pending.pop()
        if path in visited:
            continue
        if len(visited) >= 256:
            issues.append("Pacenote Plugin config exceeds 256 included files")
            break
        try:
            path.relative_to(root)
        except ValueError:
            issues.append(f"Pacenote Plugin config escapes its root: {path}")
            continue
        visited.add(path)
        try:
            text = filesystem.read_text(path, encoding="latin-1")
        except OSError as exc:
            issues.append(f"Unable to read Pacenote Plugin config {path.name}: {exc}")
            continue
        parser = config_parser()
        try:
            parser.read_string(text)
        except configparser.Error as exc:
            issues.append(f"Invalid Pacenote Plugin config {path.name}: {exc}")
            continue
        for section_name in parser.sections():
            section = parser[section_name]
            prefix, separator, name = section_name.partition("::")
            if separator and prefix.casefold() == "pacenote":
                raw_id = _case_value(section, "id")
                if raw_id is None:
                    continue
                try:
                    note_id = int(raw_id)
                except ValueError:
                    issues.append(
                        f"Pacenote Plugin config {path.name} has invalid ID {raw_id!r}"
                    )
                    continue
                definition = replace(
                    _custom_definition(name.upper()),
                    audio_files=_audio_files(section),
                    config_file=path.relative_to(root).as_posix(),
                )
                previous = result.get(note_id)
                if previous is not None and previous.name != definition.name:
                    issues.append(
                        f"Pacenote Plugin ID {note_id} is defined as both "
                        f"{previous.name!r} and {definition.name!r}"
                    )
                    continue
                result[note_id] = definition
                continue
            includes = sorted(
                (
                    (key, value)
                    for key, value in section.items()
                    if key.casefold().startswith("file")
                ),
                key=_include_sort_key,
                reverse=True,
            )
            for _key, value in includes:
                try:
                    pending.append(_resolve_config_path(root, path.parent, value))
                except ValueError as exc:
                    issues.append(str(exc))
    return result, issues


def rbr_install_root(stage: RbrStage) -> Path:
    return stage.root if stage.source_format == "original" else stage.root.parents[2]


def load_pacenote_catalog_from_root(
    rbr_root: Path,
    *,
    apply_overrides: bool = True,
) -> PacenoteCatalog:
    filesystem = current_filesystem()
    definitions = dict(LEGACY_DEFINITIONS)
    config_root = (
        rbr_root
        / "Plugins"
        / "Pacenote"
        / "config"
        / "pacenotes"
    )
    config_path = config_root / "Descriptive.ini"
    issues: list[str] = []
    if filesystem.is_file(config_path):
        custom, issues = _read_plugin_definitions(config_root, config_path)
        for note_id, definition in custom.items():
            existing = definitions.get(note_id)
            definitions[note_id] = (
                replace(
                    existing,
                    audio_files=definition.audio_files,
                    config_file=definition.config_file,
                    source_name=definition.source_name,
                )
                if existing is not None
                else definition
            )
    else:
        config_path = None
    overrides = load_pacenote_overrides() if apply_overrides else {}
    for note_id, definition in list(definitions.items()):
        key = pacenote_override_key(note_id, definition)
        if key in overrides:
            definitions[note_id] = _definition_with_record(definition, overrides[key])
    return PacenoteCatalog(
        definitions,
        config_path,
        tuple(issues),
        *_flag_effect_maps(overrides),
        _control_modifier_map(overrides),
    )


def load_pacenote_catalog(stage: RbrStage) -> PacenoteCatalog:
    return load_pacenote_catalog_from_root(rbr_install_root(stage))


def unsupported_pacenote_conversions(
    rbr_root: Path | None = None,
) -> tuple[PacenoteConversion, ...]:
    rows = [
        PacenoteConversion(
            f'[ID {note_id}] "{definition.name}"',
            (
                f"[{LEGACY_UNSUPPORTED_PRESENTATIONS[note_id][0].casefold()}] "
                f"{LEGACY_UNSUPPORTED_PRESENTATIONS[note_id][1]}"
            ),
        )
        for note_id, definition in sorted(LEGACY_DEFINITIONS.items())
        if not definition.spoken
        and note_id == UNMAPPED_NOTE_ID
    ]
    if rbr_root is None:
        return tuple(rows)
    catalog = load_pacenote_catalog_from_root(rbr_root)
    for note_id, definition in sorted(catalog.definitions.items()):
        if (
            not definition.custom
            or definition.spoken
            or definition.corner_modifier is not None
        ):
            continue
        details = ["[unsupported] no corresponding BeamNG entry"]
        if definition.audio_files:
            details.append(f'audio="{", ".join(definition.audio_files)}"')
        if definition.config_file:
            details.append(f'config="{definition.config_file}"')
        rows.append(
            PacenoteConversion(
                f'PLUGIN [ID {note_id}] "{definition.name}"',
                "; ".join(details),
            )
        )
    return tuple(rows)


def _legacy_reference_entries() -> tuple[PacenoteReferenceEntry, ...]:
    entries: list[PacenoteReferenceEntry] = []
    for row in PACENOTE_MAPPING_DATA["legacyDefinitions"]:
        note_id = int(row["id"])
        definition = LEGACY_DEFINITIONS[note_id]
        source = f'ID {note_id}: {row["name"]}'
        role = row.get("role")
        if role == "distanceMarker":
            entries.append(
                PacenoteReferenceEntry(
                    "legacy",
                    source,
                    "Distance rule",
                    "route",
                    "Ends the preceding converted note and lets BeamNG call the distance after it",
                    "It is used only between the preceding and following converted calls. Before the first call, it is preserved as metadata. Notes without one get no distance or link call, as in RBR.",
                )
            )
        elif role == "start":
            entries.append(
                PacenoteReferenceEntry(
                    "legacy",
                    source,
                    "Route point",
                    "route",
                    "Sets the stage start",
                    "This record is not spoken as a BeamNG pacenote.",
                )
            )
        elif role == "finish":
            entries.append(
                PacenoteReferenceEntry(
                    "legacy",
                    source,
                    "Route point and modifier",
                    "route",
                    "Sets the stage finish and adds a finish call",
                    "The finish record is both a route point and a spoken finish modifier.",
                )
            )
        elif role == "split":
            entries.append(
                PacenoteReferenceEntry(
                    "legacy",
                    source,
                    "Route point",
                    "route",
                    "Sets a split point",
                    "This record is not spoken as a BeamNG pacenote.",
                )
            )
        elif role == "stopControl":
            entries.append(
                PacenoteReferenceEntry(
                    "legacy",
                    source,
                    "Route point",
                    "route",
                    "Sets the stop-control finish point",
                    "This record is not spoken as a BeamNG pacenote.",
                )
            )
        elif role == "unmapped":
            entries.append(
                PacenoteReferenceEntry(
                    "legacy",
                    source,
                    "Unsupported call",
                    "unsupported",
                    "No BeamNG equivalent",
                    "The source record is retained in diagnostics but does not create a BeamNG pacenote.",
                    f"legacy:{note_id}",
                    pacenote_definition_record(definition),
                )
            )
        elif definition.is_corner:
            descriptor = (
                f" {definition.descriptor}" if definition.descriptor else ""
            )
            direction = "left" if definition.direction < 0 else "right"
            entries.append(
                PacenoteReferenceEntry(
                    "legacy",
                    source,
                    "Corner",
                    "converted",
                    f"Creates a {direction}{descriptor} corner",
                    f"Its default endpoint is {definition.nominal_length:g} m after the call, shortened when route space is limited.",
                    f"legacy:{note_id}",
                    pacenote_definition_record(definition),
                )
            )
        elif definition.modifier:
            entries.append(
                PacenoteReferenceEntry(
                    "legacy",
                    source,
                    "Modifier",
                    "converted",
                    f'Adds the "{_humanize_pacenote_text(definition.modifier)}" modifier',
                    "BeamNG uses available modifier slots on the generated note.",
                    f"legacy:{note_id}",
                    pacenote_definition_record(definition),
                )
            )
        else:
            entries.append(
                PacenoteReferenceEntry(
                    "legacy",
                    source,
                    "Caution",
                    "converted",
                    f"Adds caution level {definition.caution}",
                    "The caution is stored in the main caution slot of the generated note.",
                    f"legacy:{note_id}",
                    pacenote_definition_record(definition),
                )
            )
    return tuple(entries)


def _plugin_alias_reference_entries() -> tuple[PacenoteReferenceEntry, ...]:
    entries: list[PacenoteReferenceEntry] = []
    for mapping in PACENOTE_MAPPING_DATA["customAliases"]:
        modifier = _optional_string(mapping.get("modifier"))
        caution = int(mapping.get("caution", 0))
        for name in mapping["names"]:
            if modifier:
                result = f'Adds the "{_humanize_pacenote_text(modifier)}" modifier'
                entry_type = "Plugin modifier"
                detail = "The configured Plugin name is matched regardless of its installation-specific numeric ID."
            else:
                result = f"Adds caution level {caution}"
                entry_type = "Plugin caution"
                detail = "The configured Plugin name is matched regardless of its installation-specific numeric ID."
            entries.append(
                PacenoteReferenceEntry(
                    "pluginCalls",
                    str(name),
                    entry_type,
                    "converted",
                    result,
                    detail,
                    f"call:{name}",
                    pacenote_definition_record(_custom_definition(str(name))),
                )
            )
    return tuple(entries)


def _plugin_corner_reference_entries() -> tuple[PacenoteReferenceEntry, ...]:
    entries: list[PacenoteReferenceEntry] = []
    for mapping in PACENOTE_MAPPING_DATA["customCorners"]:
        direction = "left" if int(mapping["direction"]) < 0 else "right"
        descriptor = (
            f" {mapping['descriptor']}" if mapping.get("descriptor") else ""
        )
        length = float(mapping["nominalLength"])
        for name in mapping["names"]:
            entries.append(
                PacenoteReferenceEntry(
                    "pluginCorners",
                    str(name),
                    "Plugin corner",
                    "converted",
                    f"Creates a {direction}{descriptor} corner",
                    f"Its default endpoint is {length:g} m after the call, shortened when route space is limited.",
                    f"call:{name}",
                    pacenote_definition_record(_custom_definition(str(name))),
                )
            )
    return tuple(entries)


def _corner_modifier_result(mapping: dict[str, object]) -> str:
    changes: list[str] = []
    if shape := _optional_string(mapping.get("shape")):
        changes.append(_humanize_pacenote_text(shape))
    if length_mode := _optional_string(mapping.get("lengthMode")):
        changes.append(
            "shortens the corner"
            if length_mode == "shorter"
            else "lengthens the corner"
        )
    if risk_intensity := int(mapping.get("riskIntensity", 0)):
        changes.append(
            "lowers intensity"
            if risk_intensity < 0
            else "raises intensity"
        )
    if caution := int(mapping.get("caution", 0)):
        changes.append(f"adds caution level {caution}")
    return "Changes an adjacent corner: " + ", ".join(changes)


def _corner_modifier_reference_entries() -> tuple[PacenoteReferenceEntry, ...]:
    entries: list[PacenoteReferenceEntry] = []
    detail = (
        "Applies to the converted corner before it, or else after it, between "
        "the surrounding distance calls and link words. Otherwise it is "
        "skipped and reported."
    )
    for mapping in PACENOTE_MAPPING_DATA["cornerModifiers"]:
        for name in mapping["names"]:
            entries.append(
                PacenoteReferenceEntry(
                    "cornerModifiers",
                    str(name),
                    "Corner modifier",
                    "conditional",
                    _corner_modifier_result(mapping),
                    detail,
                    f"call:{name}",
                    pacenote_definition_record(_custom_definition(str(name))),
                )
            )
    return tuple(entries)


def _flag_reference_entry(flag: dict[str, object]) -> PacenoteReferenceEntry:
    conversion = flag.get("conversion")
    conversion_data = conversion if isinstance(conversion, dict) else {}
    kind = conversion_data.get("kind")
    value = _optional_string(conversion_data.get("value"))
    status = str(flag["status"])
    entry_status = {
        "supported": "converted",
        "unsupported": "loss",
        "scoped": "metadata",
        "unresolved": "unresolved",
    }[status]
    if kind == "modifier":
        result = f'Adds the "{_humanize_pacenote_text(value or "")}" modifier'
        detail = "It can be reported as dropped when the generated note has no free modifier slot."
    elif kind == "cornerShape":
        result = f'Changes a corner to "{_humanize_pacenote_text(value or "")}"'
        detail = "It applies only to a converted corner; otherwise the modifier is reported as dropped."
        entry_status = "conditional"
    elif kind == "tightensBad":
        result = "Tightens a corner and adds caution level 1"
        detail = "It applies only to a converted corner; otherwise the modifier is reported as dropped."
        entry_status = "conditional"
    elif kind == "cornerLengthMode":
        result = "Makes a corner longer"
        detail = "It applies only to a converted corner; otherwise the modifier is reported as dropped."
        for call, modifier in LONG_CALL_MODIFIERS.items():
            detail += (
                f' On a "{_humanize_pacenote_text(call)}" call it becomes '
                f'"{_humanize_pacenote_text(modifier)}" instead.'
            )
        entry_status = "conditional"
    elif kind == "droppedModifier":
        result = f'"{_humanize_pacenote_text(value or "")}" has no BeamNG equivalent'
        detail = "The source flag is recognized and reported as conversion loss."
    elif flag["key"] == "EXPLICIT_SOUND_INDEX":
        result = "Preserves the source sound-index selection as metadata"
        detail = (
            "Bits 16–18 select an RBR sound variation when this bit is set; "
            "BeamNG does not reproduce the source audio selection."
        )
    elif flag["key"] == "NO_LINK_OR_STICKY":
        result = "Preserves No Link or Sticky as metadata"
        scopes = ", ".join(
            _humanize_pacenote_text(str(scope))
            for scope in flag["scopes"]
        )
        detail = (
            f"Sticky applies to {scopes}. On other calls it is No Link, which "
            "stops RBR's link word after the call; BeamNG already speaks "
            "nothing there unless a distance marker follows."
        )
    else:
        result = "Preserves an unresolved flag"
        detail = "The Plugin registry has no confirmed behavior for this bit."
    registry_name = _optional_string(flag.get("registryName"))
    mask = int(flag["mask"])
    source = f"0x{mask:03X}"
    if registry_name:
        source += f" {registry_name}"
    editable = mask in EDITABLE_FLAG_MASKS
    return PacenoteReferenceEntry(
        "pluginFlags",
        source,
        "Plugin flag",
        entry_status,
        result,
        detail,
        f"flag:{flag['key']}" if editable else None,
        pacenote_flag_record(mask) if editable else None,
    )


def _plugin_control_reference_entries() -> tuple[PacenoteReferenceEntry, ...]:
    entries: list[PacenoteReferenceEntry] = []
    for control in PACENOTE_MAPPING_DATA["pluginControls"]:
        kind = str(control["kind"])
        conversion = control.get("conversion")
        if isinstance(conversion, dict):
            conversion_kind = str(conversion["kind"])
            conversion_value = str(conversion["value"])
            if conversion_kind == "cornerShape":
                result = (
                    f'Applies "{_humanize_pacenote_text(conversion_value)}" '
                    "to its corner"
                )
                detail = (
                    "It goes to the converted corner before it, or else after it, "
                    "between the surrounding distance calls and link words."
                )
            elif conversion_kind == "cornerLengthMode":
                result = "Makes its corner long"
                detail = (
                    "It goes to the converted corner before it, or else after it, "
                    "between the surrounding distance calls and link words."
                )
            else:
                result = (
                    f'Applies the "{_humanize_pacenote_text(conversion_value)}" '
                    "modifier to its pacenote"
                )
                detail = (
                    "It goes to the converted pacenote before it, or else after it, "
                    "between the surrounding distance calls and link words."
                )
            entry_status = "conditional"
        elif kind in {"INTO", "AND", "ONTO"}:
            result = "Preserves the source transition"
            detail = "RBR speaks it as its own call; BeamNG has no matching call."
            entry_status = "metadata"
        elif kind == "EMPTY_CALL":
            result = "Preserves the silent source call"
            detail = (
                "BeamNG does not synthesize its source link or distance behavior."
            )
            entry_status = "metadata"
        elif kind == "SOUND_INDEX":
            result = "Saves the source sound-index control"
            detail = (
                "Its signed payload and Sticky state are preserved as metadata; "
                "BeamNG does not reproduce RBR sound-variation selection."
            )
            entry_status = "metadata"
        else:
            result = "Saves the control with the next applicable note"
            detail = "Its value is preserved as metadata but does not override BeamNG timing."
            entry_status = "metadata"
        code = int(control["code"])
        if mask := PLUGIN_CONTROL_FIXED_CORNER_FLAGS.get(code):
            detail += (
                f" It follows the 0x{mask:03X} {_FLAGS_BY_MASK[mask]['key']} flag rule; "
                "edit that flag to change it."
            )
        editable = code in PLUGIN_CONTROL_ATTACHED_MODIFIERS
        entries.append(
            PacenoteReferenceEntry(
                "pluginControls",
                f"0x{code:X} {kind}",
                "Plugin control",
                entry_status,
                result,
                detail,
                pacenote_control_key(code) if editable else None,
                (
                    {"kind": "modifier", "modifier": PLUGIN_CONTROL_ATTACHED_MODIFIERS[code]}
                    if editable
                    else None
                ),
            )
        )
    return tuple(entries)


def pacenote_conversion_reference() -> tuple[PacenoteReferenceGroup, ...]:
    return (
        PacenoteReferenceGroup(
            "legacy",
            "Built-in RBR calls",
            "Legacy RBR IDs with a direct conversion rule.",
            _legacy_reference_entries(),
        ),
        PacenoteReferenceGroup(
            "pluginCalls",
            "Recognized Plugin calls",
            "Plugin names that map directly to BeamNG modifiers or cautions.",
            _plugin_alias_reference_entries(),
        ),
        PacenoteReferenceGroup(
            "pluginCorners",
            "Recognized Plugin corners",
            "Plugin names that create a complete BeamNG corner.",
            _plugin_corner_reference_entries(),
        ),
        PacenoteReferenceGroup(
            "cornerModifiers",
            "Plugin corner modifiers",
            "Plugin calls that require an adjacent converted corner.",
            _corner_modifier_reference_entries(),
        ),
        PacenoteReferenceGroup(
            "pluginFlags",
            "Fixed Plugin flags",
            "Flags that add detail, preserve metadata, or report conversion loss.",
            tuple(_flag_reference_entry(flag) for flag in PACENOTE_MAPPING_DATA["flags"]),
        ),
        PacenoteReferenceGroup(
            "pluginControls",
            "Plugin controls",
            "Source timing and transition controls that are retained as metadata.",
            _plugin_control_reference_entries(),
        ),
        PacenoteReferenceGroup(
            "specialCases",
            "Special and unsupported records",
            "Records that are preserved, skipped, or reported rather than converted to speech.",
            (
                PacenoteReferenceEntry(
                    "specialCases",
                    "0xFFE / 0xFFF configuration headers",
                    "Configuration metadata",
                    "metadata",
                    "Not spoken",
                    "A valid initial header run is treated as Plugin configuration data, not as pacenote controls.",
                ),
                PacenoteReferenceEntry(
                    "specialCases",
                    "Unknown type or unknown flag",
                    "Unknown record",
                    "unresolved",
                    "Preserved and reported",
                    "The converter keeps raw source details instead of guessing a BeamNG call.",
                ),
                PacenoteReferenceEntry(
                    "specialCases",
                    "Configured name without a rule",
                    "Unsupported Plugin call",
                    "unsupported",
                    "No BeamNG mapping",
                    "Its ID, source audio, and configuration file are shown in the installation tab when available.",
                ),
                PacenoteReferenceEntry(
                    "specialCases",
                    "Corner modifier without an adjacent corner",
                    "Conditional record",
                    "loss",
                    "Skipped and reported",
                    "A corner modifier needs a converted corner between the surrounding distance calls and link words.",
                ),
                PacenoteReferenceEntry(
                    "specialCases",
                    "Distance marker without a valid preceding note",
                    "Distance rule",
                    "loss",
                    "Preserved or reported without changing a BeamNG note",
                    "It cannot end a note before the first converted call, after skipped content, or outside the preceding note's range.",
                ),
                PacenoteReferenceEntry(
                    "specialCases",
                    "Plugin control without an applicable target",
                    "Plugin control",
                    "loss",
                    "Preserved and reported without attaching metadata",
                    "A timing control needs a following converted note. A transition control also needs a preceding converted note.",
                ),
                PacenoteReferenceEntry(
                    "specialCases",
                    "Record outside the stage start/finish range",
                    "Out-of-range record",
                    "metadata",
                    "Not converted for this stage range",
                    "The raw source record remains available in diagnostics.",
                ),
                PacenoteReferenceEntry(
                    "specialCases",
                    "Known future candidates, such as KEEP_IN",
                    "Not currently mapped",
                    "unsupported",
                    "Not converted today",
                    "Documentation candidates do not become supported until they are added to the JSON mapping data.",
                ),
            ),
        ),
    )


def _installation_entry(
    note_id: int,
    definition: NoteDefinition,
    *,
    unsupported: bool,
) -> PacenoteReferenceEntry:
    source = f"ID {note_id}: {definition.source_name or definition.name}"
    details = [f"Defined in {definition.config_file or 'the Plugin config'}"]
    if definition.audio_files:
        details.append("Audio: " + ", ".join(definition.audio_files))
    override_key = pacenote_override_key(note_id, definition)
    default_value = pacenote_definition_record(definition) if override_key else None
    if unsupported:
        return PacenoteReferenceEntry(
            "installation",
            source,
            "Configured Plugin call",
            "unsupported",
            "No current BeamNG mapping",
            ". ".join(details),
            override_key,
            default_value,
        )
    if definition.corner_modifier is not None:
        result = "Changes an adjacent converted corner"
        entry_type = "Configured corner modifier"
        status = "conditional"
    elif definition.is_corner:
        result = "Creates a BeamNG corner"
        entry_type = "Configured Plugin corner"
        status = "recognized"
    elif definition.modifier:
        result = f'Adds the "{_humanize_pacenote_text(definition.modifier)}" modifier'
        entry_type = "Configured Plugin call"
        status = "recognized"
    elif definition.caution:
        result = f"Adds caution level {definition.caution}"
        entry_type = "Configured Plugin call"
        status = "recognized"
    else:
        result = "Uses the built-in RBR behavior"
        entry_type = "Configured legacy ID"
        status = "recognized"
    return PacenoteReferenceEntry(
        "installation",
        source,
        entry_type,
        status,
        result,
        ". ".join(details),
        override_key,
        default_value,
    )


def pacenote_installation_reference(
    rbr_root: Path | None,
) -> PacenoteInstallationReference:
    if rbr_root is None:
        return PacenoteInstallationReference(
            "no-installation",
            "Choose an RBR install folder to inspect its Pacenote Plugin configuration.",
            None,
            (),
            (),
            (),
        )
    catalog = load_pacenote_catalog_from_root(rbr_root, apply_overrides=False)
    if catalog.config_path is None:
        return PacenoteInstallationReference(
            "no-config",
            "No Pacenote Plugin Descriptive.ini was found in this RBR installation.",
            None,
            (),
            (),
            catalog.issues,
        )
    configured = [
        (note_id, definition)
        for note_id, definition in catalog.definitions.items()
        if definition.config_file is not None
    ]
    recognized: list[PacenoteReferenceEntry] = []
    unsupported: list[PacenoteReferenceEntry] = []
    for note_id, definition in sorted(configured):
        entry = _installation_entry(
            note_id,
            definition,
            unsupported=(
                definition.custom
                and not definition.spoken
                and definition.corner_modifier is None
            ),
        )
        (unsupported if entry.status == "unsupported" else recognized).append(entry)
    summary = (
        f"Loaded {len(configured)} configured call"
        + ("" if len(configured) == 1 else "s")
        + f": {len(recognized)} recognized, {len(unsupported)} unsupported."
    )
    return PacenoteInstallationReference(
        "loaded",
        summary,
        str(catalog.config_path),
        tuple(recognized),
        tuple(unsupported),
        catalog.issues,
    )


def project_route(
    route: list[tuple[np.ndarray, np.ndarray, float]],
    distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    distances = [point[2] for point in route]
    if distance <= distances[0]:
        return route[0][0].copy(), route[0][1].copy()
    if distance >= distances[-1]:
        return route[-1][0].copy(), route[-1][1].copy()
    upper = bisect_right(distances, distance)
    lower = upper - 1
    span = distances[upper] - distances[lower]
    factor = 0.0 if span <= 1e-8 else (distance - distances[lower]) / span
    position = route[lower][0] + (route[upper][0] - route[lower][0]) * factor
    direction = route[lower][1] + (route[upper][1] - route[lower][1]) * factor
    length = float(np.linalg.norm(direction))
    if length > 1e-8:
        direction /= length
    return position, direction


def route_positions(
    route: list[tuple[np.ndarray, np.ndarray, float]],
    start_distance: float,
    end_distance: float,
) -> list[np.ndarray]:
    positions = [project_route(route, start_distance)[0]]
    positions.extend(
        position
        for position, _direction, distance in route
        if start_distance < distance < end_distance
    )
    positions.append(project_route(route, end_distance)[0])
    return positions


def route_distance(
    route: list[tuple[np.ndarray, np.ndarray, float]],
    start_distance: float,
    end_distance: float,
) -> float:
    if end_distance <= start_distance:
        return 0.0
    positions = route_positions(route, start_distance, end_distance)
    return sum(
        float(np.linalg.norm(end - start))
        for start, end in zip(positions, positions[1:])
    )


def _waypoint(
    old_id: int,
    waypoint_type: str,
    position: np.ndarray,
    normal: np.ndarray,
) -> dict[str, object]:
    return {
        "name": "corner start" if waypoint_type == "cornerStart" else "corner end",
        "normal": [float(value) for value in normal],
        "oldId": old_id,
        "pos": [float(value) for value in position],
        "radius": 8,
        "waypointType": waypoint_type,
    }


def _set_modifier(
    items: dict[str, dict[str, object]],
    modifier: str,
    *,
    corner: bool,
) -> bool:
    for slot in (("5", "6") if corner else ("2", "3")):
        if not items[slot]:
            items[slot] = {"type": modifier}
            return True
    return False


def _apply_corner_effect(
    items: dict[str, dict[str, object]],
    effect: CornerModifier,
) -> None:
    if effect.shape:
        items["4"]["shape"] = effect.shape
    if effect.length_mode:
        items["4"]["lengthMode"] = effect.length_mode
    if effect.risk_intensity:
        items["4"]["riskIntensity"] = effect.risk_intensity
    if effect.caution and not items["1"]:
        items["1"] = {"type": "caution", "level": effect.caution}


def _structured_items(
    definition: NoteDefinition,
    flag: int,
    *,
    attached_fixed_corner_flags: int = 0,
    attached_modifiers: tuple[str, ...] = (),
    known_flag_mask: int = KNOWN_FLAG_MASK,
    flag_modifiers: dict[int, str] = FLAG_MODIFIERS,
    dropped_flag_modifiers: dict[int, str] = UNSUPPORTED_FLAG_MODIFIERS,
    corner_flag_effects: dict[int, CornerModifier] = CORNER_FLAG_EFFECTS,
) -> tuple[dict[str, object], tuple[str, ...], int, tuple[str, ...]]:
    items: dict[str, dict[str, object]] = {
        str(index): {} for index in range(1, 7)
    }
    if definition.is_corner:
        corner: dict[str, object] = {"type": "corner"}
        if definition.descriptor:
            corner["descriptor"] = definition.descriptor
        items["4"] = corner
    caution = max(definition.caution, definition.corner_caution)
    if caution:
        items["1"] = {"type": "caution", "level": caution}
    dropped: list[str] = []
    if definition.modifier and not _set_modifier(
        items,
        definition.modifier,
        corner=definition.is_corner,
    ):
        dropped.append(definition.modifier)
    inferred: list[str] = []
    for bit, modifier in flag_modifiers.items():
        if flag & bit:
            inferred.append(modifier)
            if not _set_modifier(items, modifier, corner=definition.is_corner):
                dropped.append(modifier)
    for modifier in attached_modifiers:
        inferred.append(modifier)
        if not _set_modifier(items, modifier, corner=definition.is_corner):
            dropped.append(modifier)
    for bit, effect in corner_flag_effects.items():
        if flag & bit:
            name = _corner_flag_name(bit)
            inferred.append(name)
            if definition.is_corner:
                _apply_corner_effect(items, effect)
            else:
                dropped.append(name)
    for bit, modifier in dropped_flag_modifiers.items():
        if flag & bit:
            inferred.append(modifier)
            dropped.append(modifier)
    if definition.is_corner:
        if definition.corner_shape:
            items["4"]["shape"] = definition.corner_shape
        if definition.corner_length_mode:
            items["4"]["lengthMode"] = definition.corner_length_mode
        if definition.corner_risk_intensity:
            items["4"]["riskIntensity"] = definition.corner_risk_intensity
        for bit, effect in corner_flag_effects.items():
            if attached_fixed_corner_flags & bit:
                _apply_corner_effect(items, effect)
    return (
        {"schemaVersion": 3, "items": items},
        tuple(inferred),
        flag & ~known_flag_mask,
        tuple(dropped),
    )


def _apply_corner_modifier(
    definition: NoteDefinition,
    modifier: CornerModifier,
) -> NoteDefinition:
    return replace(
        definition,
        corner_shape=modifier.shape or definition.corner_shape,
        corner_length_mode=modifier.length_mode or definition.corner_length_mode,
        corner_risk_intensity=(
            modifier.risk_intensity
            if modifier.risk_intensity
            else definition.corner_risk_intensity
        ),
        corner_caution=max(definition.corner_caution, modifier.caution),
    )


def _corner_end_distance(
    note: Pacenote,
    definition: NoteDefinition,
    next_distance: float,
    finish_distance: float,
) -> float:
    available = max(2.0, min(next_distance - note.distance - 2.0, finish_distance - note.distance))
    return note.distance + min(definition.nominal_length, available)


def _notebook_pacenote(
    index: int,
    note: Pacenote,
    definition: NoteDefinition,
    start_distance: float,
    end_distance: float,
    distance_marker: float | None,
    route: list[tuple[np.ndarray, np.ndarray, float]],
    *,
    attached_fixed_corner_flags: int = 0,
    attached_modifiers: tuple[str, ...] = (),
    transferred_fixed_corner_flags: int = 0,
    catalog: PacenoteCatalog,
) -> tuple[dict[str, object], tuple[str, ...], int, tuple[str, ...]]:
    start_pos, start_normal = project_route(route, start_distance)
    end_pos, end_normal = project_route(route, end_distance)
    source_fixed_corner_flags = (
        note.flag & _fixed_corner_flag_mask(catalog)
        if not definition.is_corner
        else 0
    )
    conversion_flag = (
        note.flag
        & ~transferred_fixed_corner_flags
        & ~source_fixed_corner_flags
    ) | attached_fixed_corner_flags
    structured, inferred, unknown_bits, dropped = _structured_items(
        definition,
        conversion_flag,
        attached_fixed_corner_flags=attached_fixed_corner_flags,
        attached_modifiers=attached_modifiers,
        known_flag_mask=(
            known_flag_mask_for_note(note.note_type, note.flag)
            | catalog.known_flag_extra
        ),
        flag_modifiers=catalog.flag_modifiers,
        dropped_flag_modifiers=catalog.dropped_flag_modifiers,
        corner_flag_effects=catalog.corner_flag_effects,
    )
    old_id = index * 3 + 1
    metadata = {
        "rbrNoteType": note.note_type,
        "rbrDistance": note.distance,
        "rbrEndDistance": end_distance,
        "rbrFlag": note.flag,
        "rbrCall": definition.name,
        "rbrCustom": definition.custom,
        "rbrInferredModifiers": list(inferred),
    }
    if note.flag & NO_LINK_OR_STICKY_FLAG_MASK:
        metadata["rbrNoLink"] = True
    sound_index = _explicit_sound_index(note.flag)
    if sound_index is not None:
        metadata["rbrSoundIndex"] = sound_index
    if distance_marker is not None:
        metadata["rbrDistanceMarker"] = distance_marker
    return (
        {
            "audioMode": 1,
            "distanceBeforeModifier": False,
            "ignoreDistanceCalls": (
                note.note_type == FINISH_NOTE_ID or distance_marker is None
            ),
            "includeLinkWord": note.note_type != FINISH_NOTE_ID,
            "isolate": False,
            "metadata": metadata,
            "name": f"RBR {index + 1}: {definition.name}",
            "oldId": old_id,
            "pacenoteWaypoints": [
                _waypoint(old_id + 1, "cornerStart", start_pos, start_normal),
                _waypoint(old_id + 2, "cornerEnd", end_pos, end_normal),
            ],
            "pk": hashlib.sha1(
                f"{index}/{note.note_type}/{note.distance:.6f}/{note.flag}".encode()
            ).hexdigest()[:8],
            "slowCorner": bool(
                definition.is_corner
                and definition.descriptor in {"hairpin", "square"}
            ),
            "slowCornerReleaseType": 30,
            "structured": structured,
            "todo": False,
            "triggerType": 15 if index == 0 else 1,
        },
        inferred,
        unknown_bits,
        dropped,
    )


def _start_distance_only_pacenote(
    index: int,
    start_distance: float,
    route: list[tuple[np.ndarray, np.ndarray, float]],
    marker_source_indexes: list[int],
) -> dict[str, object]:
    start_pos, start_normal = project_route(route, start_distance)
    source_indexes = sorted(marker_source_indexes)
    source_index_text = ",".join(str(source_index) for source_index in source_indexes)
    old_id = index * 3 + 1
    return {
        "audioMode": 1,
        "distanceBeforeModifier": False,
        "ignoreDistanceCalls": False,
        "includeLinkWord": True,
        "isolate": False,
        "metadata": {
            "rbrDistance": start_distance,
            "rbrDistanceOnly": True,
            "rbrEndDistance": start_distance,
            "rbrInitialDistanceMarkerSourceIndexes": source_indexes,
            "rbrSourceIndexes": source_indexes,
        },
        "name": "RBR start distance",
        "oldId": old_id,
        "pacenoteWaypoints": [
            _waypoint(old_id + 1, "cornerStart", start_pos, start_normal),
            _waypoint(old_id + 2, "cornerEnd", start_pos, start_normal),
        ],
        "pk": hashlib.sha1(
            f"start-distance/{start_distance:.6f}/{source_index_text}".encode()
        ).hexdigest()[:8],
        "slowCorner": False,
        "slowCornerReleaseType": 30,
        "structured": {
            "schemaVersion": 3,
            "items": {str(index): {} for index in range(1, 7)},
        },
        "todo": False,
        "triggerType": 15,
    }


def _spline(
    route: list[tuple[np.ndarray, np.ndarray, float]],
    start_distance: float,
    finish_distance: float,
    stop_distance: float,
    extra_distances: list[float],
    surface: str,
) -> dict[str, object]:
    spline_start = max(route[0][2], start_distance)
    spline_end = min(
        route[-1][2],
        max(finish_distance + 20, stop_distance),
    )
    distances = [
        spline_start,
        start_distance,
        finish_distance,
        stop_distance,
        spline_end,
        *extra_distances,
    ]
    for _position, _direction, distance in route:
        if spline_start < distance < spline_end:
            distances.append(distance)
    projected = [
        project_route(route, distance)
        for distance in sorted(set(round(value, 4) for value in distances))
    ]
    return {
        "name": "RBR driveline",
        "version": 1,
        "isLoop": False,
        "nodes": [
            [float(value) for value in position]
            for position, _direction in projected
        ],
        "nmls": [
            [float(value) for value in direction]
            for _position, direction in projected
        ],
        "properties": {
            "liaisonAllocatedTimeMins": 5,
            "speedLimitKph": 100,
            "useRaycast": False,
            "vertRayRaise": 2,
        },
        "stats": {
            "raceDistanceKms": round(
                max(0, finish_distance - start_distance) / 1000,
                3,
            ),
            "surfacePercentages": {surface: 100},
        },
        "widths": [10] * len(projected),
    }


def build_rally_notebook(
    stage: RbrStage,
    route: list[tuple[np.ndarray, np.ndarray, float]],
    route_range: tuple[int, int],
    start_distance: float | None = None,
) -> RallyNotebookResult:
    catalog = load_pacenote_catalog(stage)
    if start_distance is None:
        start_distance = route[route_range[0]][2]
    start_distance = min(max(start_distance, route[0][2]), route[-1][2])
    route_notes = [
        note
        for _source_index, note in sorted(
            enumerate(stage.pacenotes),
            key=lambda item: (item[1].distance, item[0]),
        )
    ]
    finish_distance = next(
        (
            note.distance
            for note in route_notes
            if (
                note.note_type == FINISH_NOTE_ID
                and note.distance >= start_distance
            )
        ),
        route[route_range[1]][2],
    )
    finish_distance = min(max(finish_distance, start_distance), route[-1][2])
    stop_distance = next(
        (
            note.distance
            for note in route_notes
            if (
                note.note_type == STOP_CONTROL_NOTE_ID
                and note.distance >= finish_distance
            )
        ),
        finish_distance,
    )
    stop_distance = min(
        max(stop_distance, finish_distance),
        route[-1][2],
    )
    config_header_note_ids = {
        id(stage.pacenotes[index])
        for index in pacenote_config_header_indexes(
            [
                (note.note_type, note.distance, note.flag)
                for note in stage.pacenotes
            ]
        )
    }
    candidate_notes = [
        note
        for note in route_notes
        if start_distance <= note.distance <= finish_distance
        and note.note_type not in {
            START_NOTE_ID,
            SPLIT_NOTE_ID,
            STOP_CONTROL_NOTE_ID,
        }
        and id(note) not in config_header_note_ids
    ]
    plugin_controls = {
        id(note): control
        for note in candidate_notes
        if (control := decode_plugin_control(note.note_type, note.flag)) is not None
    }
    plugin_control_notes = [
        note
        for note in candidate_notes
        if id(note) in plugin_controls
    ]
    source_notes = [
        note
        for note in candidate_notes
        if id(note) not in plugin_controls
    ]
    candidate_note_ids = {id(note) for note in candidate_notes}
    source_indices = {
        id(note): index
        for index, note in enumerate(stage.pacenotes)
    }
    ignored_reasons: dict[int, list[str]] = defaultdict(list)
    source_issues: dict[int, list[str]] = defaultdict(list)
    unknown_flag_notes: dict[int, list[Pacenote]] = defaultdict(list)
    for note in plugin_control_notes:
        unknown = note.flag & ~(
            known_flag_mask_for_note(note.note_type, note.flag)
            | catalog.known_flag_extra
        )
        for bit in _individual_flag_bits(unknown):
            unknown_flag_notes[bit].append(note)
            source_issues[id(note)].append(
                f"unknown RBR flag bits {hex(bit)}"
            )
    attached_corner_modifiers: dict[int, int] = {}
    preserved_plugin_controls: dict[int, int] = {}
    preserved_start_markers: dict[int, int] = {}
    used_distance_marker_ids: set[int] = set()
    convertible: list[tuple[Pacenote, NoteDefinition]] = []
    unknown_ids: Counter[int] = Counter()
    unknown_notes: list[Pacenote] = []
    unspoken_notes: list[tuple[Pacenote, NoteDefinition]] = []
    unsupported_custom_notes: list[tuple[Pacenote, NoteDefinition]] = []
    unattached_corner_modifiers: dict[tuple[int, str], list[Pacenote]] = defaultdict(list)
    unconvertible_distance_markers: dict[str, list[Pacenote]] = defaultdict(list)
    distance_markers: dict[int, float] = {}
    distance_marker_notes: dict[int, Pacenote] = {}
    previous_content_note: Pacenote | None = None
    distance_marker_barrier: str | None = None
    for note in candidate_notes:
        control = plugin_controls.get(id(note))
        if control is not None:
            if control.kind == "EMPTY_CALL":
                previous_content_note = None
                distance_marker_barrier = "preceding EMPTY_CALL is a silent anchor"
            continue
        definition = catalog.definitions.get(note.note_type)
        if note.note_type == DISTANCE_MARKER_NOTE_ID:
            if previous_content_note is not None:
                distance_markers.setdefault(
                    id(previous_content_note),
                    note.distance,
                )
                distance_marker_notes.setdefault(
                    id(previous_content_note),
                    note,
                )
            else:
                unconvertible_distance_markers[
                    distance_marker_barrier or "no preceding content pacenote"
                ].append(note)
            previous_content_note = None
            distance_marker_barrier = None
        elif (
            note.note_type not in TIMING_NOTE_IDS
            and (definition is None or definition.corner_modifier is None)
        ):
            previous_content_note = note
            distance_marker_barrier = None
    for note in source_notes:
        definition = catalog.definitions.get(note.note_type)
        if note.note_type == DISTANCE_MARKER_NOTE_ID or (
            definition is not None and definition.corner_modifier is not None
        ):
            continue
        if definition is None:
            unknown_ids[note.note_type] += 1
            unknown_notes.append(note)
            ignored_reasons[id(note)].append("no matching definition")
        elif definition.spoken:
            if long_modifier := _long_call_modifier(definition, note.flag, catalog):
                definition = replace(definition, modifier=long_modifier)
            convertible.append((note, definition))
        elif definition.custom:
            unsupported_custom_notes.append((note, definition))
            ignored_reasons[id(note)].append("no BeamNG mapping")
        else:
            unspoken_notes.append((note, definition))
            ignored_reasons[id(note)].append("no BeamNG equivalent")
    convertible_definitions = {
        id(note): definition
        for note, definition in convertible
    }
    convertible_indexes = {
        id(note): index
        for index, (note, _definition) in enumerate(convertible)
    }
    fixed_corner_flag_mask = _fixed_corner_flag_mask(catalog)

    def fixed_flags_for_source(note: Pacenote) -> int:
        control = plugin_controls.get(id(note))
        if control is not None:
            return (
                PLUGIN_CONTROL_FIXED_CORNER_FLAGS.get(control.code, 0)
                & fixed_corner_flag_mask
            )
        definition = catalog.definitions.get(note.note_type)
        if definition is None or definition.is_corner:
            return 0
        fixed_flags = note.flag & fixed_corner_flag_mask
        if _long_call_modifier(definition, note.flag, catalog):
            fixed_flags &= ~LONG_FLAG_MASK
        return fixed_flags

    def needs_corner(note: Pacenote) -> bool | None:
        control = plugin_controls.get(id(note))
        if control is not None:
            if fixed_flags_for_source(note):
                return True
            return False if control.code in catalog.control_modifiers else None
        definition = catalog.definitions.get(note.note_type)
        if definition is not None and definition.corner_modifier is not None:
            return True
        return None

    def is_phrase_boundary(note: Pacenote) -> bool:
        control = plugin_controls.get(id(note))
        if control is not None:
            return control.kind in PHRASE_BOUNDARY_CONTROL_KINDS
        return note.note_type in {DISTANCE_MARKER_NOTE_ID, FINISH_NOTE_ID}

    def phrase_target(
        phrase: list[Pacenote],
        position: int,
        corner_only: bool,
    ) -> Pacenote | None:
        def is_target(note: Pacenote) -> bool:
            definition = convertible_definitions.get(id(note))
            return definition is not None and (definition.is_corner or not corner_only)

        before = (note for note in reversed(phrase[:position]) if is_target(note))
        after = (note for note in phrase[position + 1:] if is_target(note))
        return next(before, None) or next(after, None)

    # A standalone modifier belongs to the call before it in the same spoken
    # phrase, else the first one after it; distance calls and link words
    # separate phrases.
    standalone_modifier_targets: dict[int, Pacenote | None] = {}
    phrase: list[Pacenote] = []
    for note in (*candidate_notes, None):
        if note is not None and not is_phrase_boundary(note):
            phrase.append(note)
            continue
        for position, modifier_note in enumerate(phrase):
            corner_only = needs_corner(modifier_note)
            if corner_only is not None:
                standalone_modifier_targets[id(modifier_note)] = phrase_target(
                    phrase,
                    position,
                    corner_only,
                )
        phrase = []
    for note in source_notes:
        definition = catalog.definitions.get(note.note_type)
        if definition is None or definition.corner_modifier is None:
            continue
        target = standalone_modifier_targets.get(id(note))
        if target is None:
            unattached_corner_modifiers[
                (note.note_type, definition.name)
            ].append(note)
            ignored_reasons[id(note)].append(NO_MODIFIER_CORNER_REASON)
            continue
        index = convertible_indexes[id(target)]
        target_note, target_definition = convertible[index]
        convertible[index] = (
            target_note,
            _apply_corner_modifier(target_definition, definition.corner_modifier),
        )
        attached_corner_modifiers[id(note)] = source_indices[id(target)]
    attached_fixed_corner_flags: dict[int, int] = {}
    attached_fixed_corner_flag_targets: dict[int, int] = {}
    fixed_corner_flags_by_target: dict[int, int] = defaultdict(int)
    fixed_corner_flag_sources_by_target: dict[
        int, list[dict[str, object]]
    ] = defaultdict(list)
    unattached_fixed_corner_flags: dict[
        tuple[int, int, str], list[Pacenote]
    ] = defaultdict(list)

    def mark_unattached_fixed_corner_flags(
        note: Pacenote,
        fixed_flags: int,
        reason: str,
    ) -> None:
        unattached_fixed_corner_flags[
            (fixed_flags, note.note_type, reason)
        ].append(note)
        source_issues[id(note)].append(reason)

    def attach_fixed_corner_flags(
        note: Pacenote,
        fixed_flags: int,
        target: Pacenote,
    ) -> None:
        attached_fixed_corner_flags[id(note)] = fixed_flags
        attached_fixed_corner_flag_targets[id(note)] = source_indices[id(target)]
        fixed_corner_flags_by_target[id(target)] |= fixed_flags
        source: dict[str, object] = {
            "modifiers": _fixed_corner_flag_names(fixed_flags),
            "sourceDistance": note.distance,
            "sourceFlag": note.flag,
            "sourceIndex": source_indices[id(note)],
            "sourceType": note.note_type,
            "sourceTypeHex": hex(note.note_type),
        }
        control = plugin_controls.get(id(note))
        if control is not None:
            source["sourceControlKind"] = control.kind
        fixed_corner_flag_sources_by_target[id(target)].append(source)

    previous_convertible_corner: Pacenote | None = None
    source_rally_finished = False
    for note in candidate_notes:
        if note.note_type == FINISH_NOTE_ID:
            previous_convertible_corner = None
            source_rally_finished = True
            continue
        fixed_flags = fixed_flags_for_source(note)
        if id(note) in plugin_controls:
            if fixed_flags:
                target = standalone_modifier_targets.get(id(note))
                if target is None:
                    mark_unattached_fixed_corner_flags(
                        note,
                        fixed_flags,
                        NO_MODIFIER_CORNER_REASON,
                    )
                else:
                    attach_fixed_corner_flags(note, fixed_flags, target)
            continue
        if source_rally_finished:
            if fixed_flags:
                mark_unattached_fixed_corner_flags(
                    note,
                    fixed_flags,
                    "after finish marker",
                )
            continue
        converted_definition = convertible_definitions.get(id(note))
        if (
            converted_definition is not None
            and converted_definition.is_corner
        ):
            previous_convertible_corner = note
            continue
        if not fixed_flags:
            continue
        if previous_convertible_corner is None:
            mark_unattached_fixed_corner_flags(
                note,
                fixed_flags,
                "no preceding BeamNG corner for "
                + ", ".join(_fixed_corner_flag_names(fixed_flags)),
            )
        else:
            attach_fixed_corner_flags(note, fixed_flags, previous_convertible_corner)
    attached_plugin_control_modifiers: dict[int, int] = {}
    plugin_control_modifier_sources_by_target: dict[
        int, list[dict[str, object]]
    ] = defaultdict(list)
    for note in plugin_control_notes:
        control = plugin_controls[id(note)]
        modifier = catalog.control_modifiers.get(control.code)
        if modifier is None:
            continue
        target = standalone_modifier_targets.get(id(note))
        if target is None:
            source_issues[id(note)].append(NO_MODIFIER_PACENOTE_REASON)
            continue
        attached_plugin_control_modifiers[id(note)] = source_indices[id(target)]
        plugin_control_modifier_sources_by_target[id(target)].append(
            {
                "modifier": modifier,
                "sourceControlCode": control.code,
                "sourceControlCodeHex": hex(control.code),
                "sourceControlKind": control.kind,
                "sourceDistance": note.distance,
                "sourceFlag": note.flag,
                "sourceIndex": source_indices[id(note)],
                "sourceType": note.note_type,
                "sourceTypeHex": hex(note.note_type),
            }
        )
    plugin_controls_by_target: dict[int, list[dict[str, object]]] = defaultdict(list)
    for control_note in plugin_control_notes:
        control = plugin_controls[id(control_note)]
        if (
            id(control_note) in attached_fixed_corner_flags
            or id(control_note) in attached_plugin_control_modifiers
        ):
            continue
        control_source_index = source_indices[id(control_note)]
        target = next(
            (
                note
                for note, _definition in convertible
                if (note.distance, source_indices[id(note)])
                > (control_note.distance, control_source_index)
            ),
            None,
        )
        previous = next(
            (
                note
                for note, _definition in reversed(convertible)
                if (note.distance, source_indices[id(note)])
                < (control_note.distance, control_source_index)
            ),
            None,
        )
        if target is None:
            ignored_reasons[id(control_note)].append(
                "missing adjacent BeamNG pacenote for Plugin control"
            )
            continue
        preserved_plugin_controls[id(control_note)] = source_indices[id(target)]
        preserved_control: dict[str, object] = {
            "code": control.code,
            "codeHex": hex(control.code),
            "kind": control.kind,
            "distance": control_note.distance,
            "flag": control_note.flag,
            "payload": control.value,
            "rawPayload": control.raw_payload,
            "rawType": control_note.note_type,
            "rawTypeHex": hex(control_note.note_type),
            "scope": control.scope,
            "sourceIndex": control_source_index,
            "sticky": control.sticky,
            "targetSourceIndex": source_indices[id(target)],
            "unit": control.unit,
        }
        if control.no_link:
            preserved_control["noLink"] = True
        plugin_controls_by_target[id(target)].append(preserved_control)
        if previous is not None:
            plugin_controls_by_target[id(target)][-1][
                "previousSourceIndex"
            ] = source_indices[id(previous)]
    initial_distance_markers_by_target: dict[int, list[dict[str, object]]] = defaultdict(list)
    initial_distance_marker_notes: list[Pacenote] = []
    no_preceding_markers = unconvertible_distance_markers.get(
        "no preceding content pacenote",
        [],
    )
    if convertible:
        first_note = convertible[0][0]
        unresolved_markers: list[Pacenote] = []
        for marker in no_preceding_markers:
            if start_distance <= marker.distance < first_note.distance:
                preserved_start_markers[id(marker)] = source_indices[id(first_note)]
                initial_distance_marker_notes.append(marker)
                initial_distance_markers_by_target[id(first_note)].append(
                    {
                        "distance": marker.distance,
                        "distanceFromStart": marker.distance - start_distance,
                        "distanceToFirstPacenote": (
                            first_note.distance - marker.distance
                        ),
                        "flag": marker.flag,
                        "routeDistanceToFirstPacenote": route_distance(
                            route,
                            marker.distance,
                            first_note.distance,
                        ),
                        "sourceIndex": source_indices[id(marker)],
                    }
                )
            else:
                unresolved_markers.append(marker)
                ignored_reasons[id(marker)].append(
                    "no preceding content pacenote"
                )
        unconvertible_distance_markers[
            "no preceding content pacenote"
        ] = unresolved_markers
    else:
        for marker in no_preceding_markers:
            ignored_reasons[id(marker)].append(
                "no preceding content pacenote"
            )
    for content_note_id, marker_note in distance_marker_notes.items():
        if content_note_id not in convertible_indexes:
            unconvertible_distance_markers[
                "preceding content pacenote was not converted"
            ].append(marker_note)
            ignored_reasons[id(marker_note)].append(
                "preceding content pacenote was not converted"
            )
    notebook_notes: list[dict[str, object]] = []
    if initial_distance_marker_notes:
        notebook_notes.append(
            _start_distance_only_pacenote(
                0,
                start_distance,
                route,
                [
                    source_indices[id(marker)]
                    for marker in initial_distance_marker_notes
                ],
            )
        )
    notebook_index_offset = len(notebook_notes)
    inferred_count = 0
    dropped_modifiers: Counter[str] = Counter()
    dropped_modifier_notes: dict[tuple[str, int], list[Pacenote]] = defaultdict(list)
    custom_count = 0
    distance_marker_count = 0
    # RBR puts calls such as "keep in" or "slippy" just after the corner they
    # follow; only the next corner limits a corner's end.
    next_corner_distances: list[float] = []
    following_corner_distance = finish_distance
    for note, definition in reversed(convertible):
        next_corner_distances.append(following_corner_distance)
        if definition.is_corner:
            following_corner_distance = note.distance
    next_corner_distances.reverse()
    for index, (note, definition) in enumerate(convertible):
        next_distance = (
            convertible[index + 1][0].distance
            if index + 1 < len(convertible)
            else finish_distance
        )
        distance_marker = distance_markers.get(id(note))
        distance_marker_note = distance_marker_notes.get(id(note))
        if not (
            distance_marker is not None
            and note.distance < distance_marker < next_distance
        ):
            if distance_marker is not None:
                unconvertible_distance_markers[
                    "outside the preceding pacenote range"
                ].append(distance_marker_note or note)
                if distance_marker_note is not None:
                    ignored_reasons[id(distance_marker_note)].append(
                        "outside the preceding pacenote range"
                    )
            distance_marker = None
        if distance_marker is not None:
            end_distance = distance_marker
            if distance_marker_note is not None:
                used_distance_marker_ids.add(id(distance_marker_note))
            distance_marker_count += 1
        else:
            end_distance = (
                _corner_end_distance(
                    note,
                    definition,
                    next_corner_distances[index],
                    finish_distance,
                )
                if definition.is_corner
                else min(note.distance + 2, finish_distance)
            )
        transferred_fixed_corner_flags = attached_fixed_corner_flags.get(
            id(note),
            0,
        )
        attached_fixed_corner_flags_for_target = (
            fixed_corner_flags_by_target.get(id(note), 0)
        )
        attached_plugin_control_modifiers_for_target = tuple(
            str(source["modifier"])
            for source in plugin_control_modifier_sources_by_target.get(
                id(note),
                [],
            )
        )
        value, inferred, unknown, dropped = _notebook_pacenote(
            index + notebook_index_offset,
            note,
            definition,
            note.distance,
            max(note.distance, end_distance),
            distance_marker,
            route,
            attached_fixed_corner_flags=(
                attached_fixed_corner_flags_for_target
            ),
            attached_modifiers=attached_plugin_control_modifiers_for_target,
            transferred_fixed_corner_flags=transferred_fixed_corner_flags,
            catalog=catalog,
        )
        if fixed_corner_flag_sources_by_target.get(id(note)):
            value["metadata"]["rbrAttachedFixedCornerModifiers"] = (
                fixed_corner_flag_sources_by_target[id(note)]
            )
        if plugin_control_modifier_sources_by_target.get(id(note)):
            value["metadata"]["rbrAttachedPluginControlModifiers"] = (
                plugin_control_modifier_sources_by_target[id(note)]
            )
        target_source_index = attached_fixed_corner_flag_targets.get(
            id(note)
        )
        if target_source_index is not None:
            value["metadata"][
                "rbrFixedCornerModifierTargetSourceIndex"
            ] = target_source_index
        if plugin_controls_by_target.get(id(note)):
            value["metadata"]["rbrPluginControlsBefore"] = (
                plugin_controls_by_target[id(note)]
            )
        if initial_distance_markers_by_target.get(id(note)):
            value["metadata"]["rbrInitialDistanceMarkers"] = (
                initial_distance_markers_by_target[id(note)]
            )
        value["metadata"]["rbrSourceIndexes"] = [source_indices[id(note)]]
        notebook_notes.append(value)
        inferred_count += len(inferred)
        for modifier in dropped:
            dropped_modifiers[modifier] += 1
            dropped_modifier_notes[(modifier, note.note_type)].append(note)
            source_issues[id(note)].append(
                _modifier_slots_full_issue(
                    modifier,
                    is_corner=definition.is_corner,
                )
            )
        custom_count += int(definition.custom)
        for bit in _individual_flag_bits(unknown):
            unknown_flag_notes[bit].append(note)
            source_issues[id(note)].append(
                f"unknown RBR flag bits {hex(bit)}"
            )
    for pacenote, next_pacenote in zip(
        notebook_notes,
        notebook_notes[1:],
    ):
        if pacenote["ignoreDistanceCalls"]:
            continue
        source_distance = max(
            0.0,
            float(next_pacenote["metadata"]["rbrDistance"])
            - float(pacenote["metadata"]["rbrEndDistance"]),
        )
        distance = route_distance(
            route,
            float(pacenote["metadata"]["rbrEndDistance"]),
            float(next_pacenote["metadata"]["rbrDistance"]),
        )
        pacenote["metadata"]["rbrDistanceAfter"] = source_distance
        pacenote["metadata"]["beamngInferredDistanceAfter"] = distance
    timestamp = int(time.time())
    notebook = {
        "audioMode": 4,
        "authors": stage.metadata.author,
        "created_at": timestamp,
        "description": f"Converted RBR pacenotes for {stage.metadata.name}",
        "metadata": {
            "rbrSource": stage.metadata.folder_name,
            "rbrPacenoteConfig": (
                catalog.config_path.relative_to(
                    rbr_install_root(stage)
                ).as_posix()
                if catalog.config_path
                else None
            ),
        },
        "name": "RBR Pacenotes",
        "pacenotes": notebook_notes,
        "updated_at": timestamp,
        "version": "4",
    }
    handled_plugin_control_note_ids = (
        set(preserved_plugin_controls)
        | set(attached_fixed_corner_flags)
        | set(attached_plugin_control_modifiers)
    )
    warnings = list(catalog.issues)
    warnings.extend(
        _pacenote_failure(
            "Unable to preserve RBR Pacenote Plugin control record",
            note,
            None,
            "no following BeamNG pacenote",
        )
        for note in plugin_control_notes
        if (
            id(note) not in handled_plugin_control_note_ids
        )
    )
    warnings.extend(
        _pacenote_failure(
            "Unable to convert",
            note,
            None,
            "no matching definition",
        )
        for note in unknown_notes
    )
    warnings.extend(
        _pacenote_failure(
            "Unable to convert",
            note,
            definition,
            "no BeamNG equivalent",
        )
        for note, definition in unspoken_notes
    )
    warnings.extend(
        _pacenote_failure(
            "Unable to convert custom",
            note,
            definition,
            "no BeamNG mapping",
        )
        for note, definition in unsupported_custom_notes
    )
    warnings.extend(
        _pacenote_failure(
            "Unable to apply RBR corner modifier",
            note,
            catalog.definitions[note_id],
            NO_MODIFIER_CORNER_REASON,
        )
        for (note_id, _name), notes in sorted(unattached_corner_modifiers.items())
        for note in notes
    )
    warnings.extend(
        _pacenote_failure(
            "Unable to attach RBR fixed corner modifier "
            + ", ".join(_fixed_corner_flag_names(flags)),
            note,
            catalog.definitions.get(note_id),
            reason,
        )
        for (flags, note_id, reason), notes in sorted(
            unattached_fixed_corner_flags.items()
        )
        for note in notes
    )
    warnings.extend(
        _pacenote_failure(
            "Unable to convert",
            note,
            catalog.definitions.get(DISTANCE_MARKER_NOTE_ID),
            reason,
        )
        for reason, notes in sorted(unconvertible_distance_markers.items())
        for note in notes
    )
    for bits, notes in sorted(unknown_flag_notes.items()):
        warnings.extend(
            _pacenote_warning(
                f"Preserved unknown RBR pacenote flag bits {hex(bits)}",
                note,
                catalog.definitions.get(note.note_type),
            )
            for note in notes
        )
    if dropped_modifiers:
        warnings.extend(
            _pacenote_failure(
                f"Unable to convert RBR modifier {modifier}",
                note,
                catalog.definitions[note_id],
                _modifier_slots_full_reason(
                    modifier,
                    is_corner=catalog.definitions[note_id].is_corner,
                ),
            )
            for (modifier, note_id), notes in sorted(
                dropped_modifier_notes.items()
            )
            for note in notes
        )
    stats = {
        "rallySourcePacenotes": len(source_notes),
        "rallyPacenotes": len(notebook_notes),
        "rallyCustomPacenotes": custom_count,
        "rallyUnsupportedCustomPacenotes": (
            len(unsupported_custom_notes)
            + sum(len(notes) for notes in unattached_corner_modifiers.values())
        ),
        "rallySkippedPacenotes": len(source_notes) - len(notebook_notes),
        "rallyAttachedCornerModifiers": len(attached_corner_modifiers),
        "rallyAttachedFixedCornerModifiers": len(
            attached_fixed_corner_flag_targets
        ),
        "rallyAttachedPluginControlModifiers": len(
            set(attached_fixed_corner_flags).intersection(plugin_controls)
            | set(attached_plugin_control_modifiers)
        ),
        "rallyIgnoredPluginControlPacenotes": sum(
            id(note) not in handled_plugin_control_note_ids
            for note in plugin_control_notes
        ),
        "rallyPreservedPluginControls": len(preserved_plugin_controls),
        "rallyPacenoteConfigHeaders": len(config_header_note_ids),
        "rallyPreservedInitialDistanceMarkers": len(
            preserved_start_markers
        ),
        "rallyInferredModifiers": inferred_count,
        "rallyDistanceMarkersUsed": distance_marker_count,
        "rallyUnknownPacenoteIds": sum(unknown_ids.values()),
        "rallyUnknownFlagValues": len(unknown_flag_notes),
    }
    log_lines, notes_lines, visualizer_records = _format_pacenote_log_lines(
        stage,
        catalog,
        candidate_note_ids,
        ignored_reasons,
        attached_corner_modifiers,
        attached_fixed_corner_flag_targets,
        attached_plugin_control_modifiers,
        plugin_controls,
        preserved_plugin_controls,
        preserved_start_markers,
        config_header_note_ids,
        used_distance_marker_ids,
        source_issues,
        convertible,
        notebook_notes,
    )
    return RallyNotebookResult(
        notebook,
        _spline(
            route,
            start_distance,
            finish_distance,
            stop_distance,
            [note.distance for note, _definition in convertible],
            stage.metadata.physics,
        ),
        stats,
        tuple(warnings),
        log_lines,
        notes_lines,
        visualizer_records,
    )
