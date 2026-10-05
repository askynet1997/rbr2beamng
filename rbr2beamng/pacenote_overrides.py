from __future__ import annotations

import json
import math
from pathlib import Path

from .core import ConversionError, settings_path
from .filesystem import current_filesystem
from .pacenote_mapping_data import load_pacenote_mapping_data


CAUTION_LEVELS = (0, 1, 2, 3)
RISK_STEPS = (-1, 0, 1)
_CORNER_CHANGE_FIELDS = ("shape", "lengthMode", "riskIntensity", "caution")
FIELDS_BY_KIND = {
    "corner": ("direction", "descriptor", "length", *_CORNER_CHANGE_FIELDS),
    "cornerModifier": _CORNER_CHANGE_FIELDS,
    "cornerChange": _CORNER_CHANGE_FIELDS,
    "call": ("modifier", "caution"),
    "modifier": ("modifier",),
    "none": (),
}
KINDS_BY_PREFIX = {
    "legacy:": ("corner", "cornerModifier", "call", "none"),
    "call:": ("corner", "cornerModifier", "call", "none"),
    "flag:": ("modifier", "cornerChange", "none"),
    "control:": ("modifier", "none"),
}


def pacenote_override_path() -> Path:
    return settings_path().with_name("pacenote_overrides.json")


def pacenote_override_kinds(key: str) -> tuple[str, ...]:
    return next(
        (kinds for prefix, kinds in KINDS_BY_PREFIX.items() if key.startswith(prefix)),
        (),
    )


def _valid_field(field: str, value: object) -> bool:
    data = load_pacenote_mapping_data()
    if field in {"direction", "riskIntensity", "caution"}:
        allowed = {"direction": (-1, 1), "riskIntensity": RISK_STEPS, "caution": CAUTION_LEVELS}
        return type(value) is int and value in allowed[field]
    if field == "length":
        return (
            isinstance(value, (int, float))
            and not isinstance(value, bool)
            and math.isfinite(value)
            and 0 < value <= 1000
        )
    vocabulary = {
        "descriptor": "beamngCornerDescriptors",
        "shape": "beamngCornerShapes",
        "lengthMode": "beamngCornerLengthModes",
        "modifier": "beamngModifiers",
    }[field]
    return isinstance(value, str) and (value == "" or value in data[vocabulary])


def pacenote_override_error(key: str, record: object) -> str | None:
    if not isinstance(record, dict) or record.get("kind") not in pacenote_override_kinds(key):
        return f"Pacenote override {key!r} has an unsupported kind"
    fields = FIELDS_BY_KIND[record["kind"]]
    if set(record) != {"kind", *fields}:
        return f"Pacenote override {key!r} must have exactly: kind, {', '.join(fields)}"
    for field in fields:
        if not _valid_field(field, record[field]):
            return f"Pacenote override {key!r} has an unsupported {field} {record[field]!r}"
    if record["kind"] == "modifier" and not record["modifier"]:
        return f"Pacenote override {key!r} needs a BeamNG modifier"
    return None


def load_pacenote_overrides() -> dict[str, dict[str, object]]:
    path = pacenote_override_path()
    filesystem = current_filesystem()
    if not filesystem.is_file(path):
        return {}
    try:
        raw = json.loads(filesystem.read_text(path, encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConversionError(f"Unable to read pacenote overrides {path}: {exc}") from exc
    if (
        not isinstance(raw, dict)
        or raw.get("version") != 1
        or not isinstance(overrides := raw.get("overrides"), dict)
    ):
        raise ConversionError(f"Pacenote overrides {path} have an invalid schema")
    for key, record in overrides.items():
        if error := pacenote_override_error(str(key), record):
            raise ConversionError(error)
    return dict(overrides)
