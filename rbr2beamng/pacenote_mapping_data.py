from __future__ import annotations

import json
from functools import lru_cache
from importlib.resources import files
from typing import Any


_REQUIRED_LISTS = (
    "legacyDefinitions",
    "customAliases",
    "customCorners",
    "cornerModifiers",
    "flags",
    "pluginControls",
)
_LEGACY_ROLES = {
    "distanceMarker",
    "start",
    "finish",
    "split",
    "stopControl",
    "unmapped",
}


def _records(data: dict[str, Any], key: str) -> list[dict[str, Any]]:
    records = data.get(key)
    if not isinstance(records, list) or not all(
        isinstance(record, dict) for record in records
    ):
        raise ValueError(f"Pacenote mapping data has no {key} records")
    return records


def _unique(label: str, values: list[Any]) -> None:
    if len(values) != len(set(values)):
        raise ValueError(f"Pacenote mapping data has duplicate {label}")


def _expanded_names(groups: list[dict[str, Any]]) -> list[str]:
    names: list[str] = []
    for group in groups:
        group_names = group.get("names")
        if not isinstance(group_names, list) or not group_names or not all(
            isinstance(name, str) and name for name in group_names
        ):
            raise ValueError("Pacenote mapping data has invalid custom names")
        names.extend(group_names)
    return names


def _check_vocabulary(data: dict[str, Any], key: str, used: set[Any]) -> None:
    values = data.get(key)
    if not isinstance(values, list) or not values or not all(
        isinstance(value, str) and value for value in values
    ):
        raise ValueError(f"Pacenote mapping data has no {key}")
    _unique(key, values)
    unknown = sorted(str(value) for value in used - {None} - set(values))
    if unknown:
        raise ValueError(
            f"Pacenote mapping data uses values missing from {key}: " + ", ".join(unknown)
        )


def _validate(data: dict[str, Any]) -> None:
    legacy, aliases, corners, corner_modifiers, flags, controls = (
        _records(data, key) for key in _REQUIRED_LISTS
    )
    _unique("legacy IDs", [row.get("id") for row in legacy])
    _unique("Plugin names", _expanded_names([
        *aliases,
        *corners,
        *corner_modifiers,
    ]))
    _unique("flag keys", [row.get("key") for row in flags])
    _unique("flag masks", [row.get("mask") for row in flags])
    _unique("Plugin control codes", [row.get("code") for row in controls])
    conversions = [
        row["conversion"]
        for row in (*flags, *controls)
        if isinstance(row.get("conversion"), dict)
    ]
    _check_vocabulary(
        data,
        "beamngModifiers",
        {
            *(row.get("modifier") for row in (*legacy, *aliases)),
            *(item.get("value") for item in conversions if item.get("kind") == "modifier"),
            *(
                value
                for row in flags
                for pair in row.get("callModifiers", {}).items()
                for value in pair
            ),
        },
    )
    _check_vocabulary(
        data,
        "beamngCornerDescriptors",
        {row.get("descriptor") for row in (*legacy, *corners)},
    )
    _check_vocabulary(
        data,
        "beamngCornerShapes",
        {
            *(row.get("shape") for row in corner_modifiers),
            *(item.get("value") for item in conversions if item.get("kind") == "cornerShape"),
        },
    )
    _check_vocabulary(
        data,
        "beamngCornerLengthModes",
        {
            *(row.get("lengthMode") for row in corner_modifiers),
            *(
                item.get("value")
                for item in conversions
                if item.get("kind") == "cornerLengthMode"
            ),
        },
    )
    for row in legacy:
        role = row.get("role")
        if role is None:
            continue
        if role not in _LEGACY_ROLES:
            raise ValueError(f"Pacenote mapping data has invalid legacy role {role}")
        presentation = row.get("presentation")
        if not isinstance(presentation, dict) or not all(
            isinstance(presentation.get(key), str)
            for key in ("category", "entry", "documentation")
        ):
            raise ValueError(
                f"Pacenote mapping data has invalid presentation for role {role}"
            )


@lru_cache(maxsize=1)
def load_pacenote_mapping_data() -> dict[str, Any]:
    resource = files("rbr2beamng").joinpath("data", "pacenote_mappings.json")
    data = json.loads(resource.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schemaVersion") != 1:
        raise ValueError("Invalid pacenote mapping data schema")
    _validate(data)
    return data
