from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping

from .core import ConversionError, settings_path
from .filesystem import current_filesystem


SURFACE_COEFFICIENTS = (
    "myStatic",
    "myKinetic",
    "myStaticSoil",
    "myKineticSoil",
    "SinkFactor",
    "SolidGroundOffset",
    "SoilThickness",
    "SoilDensity",
    "RollingResistance",
    "WaterFactor",
    "BumpFrequency",
    "BumpAmplitude",
    "CollMyHigh",
    "CollMyLow",
    "eHigh",
    "eLow",
    "Type",
)
GROUND_TYPES = frozenset(
    {
        "ASPHALT",
        "ASPHALT_OLD",
        "ASPHALT_WET",
        "BRANCHES_STRONG",
        "DIRT",
        "DIRT_DUSTY",
        "DIRT_DUSTY_LOOSE",
        "GRASS",
        "GRAVEL",
        "GRAVEL_WET",
        "ICE",
        "LEAVES_STRONG",
        "LEAVES_THIN",
        "METAL",
        "MUD",
        "ROCK",
        "SAND",
        "SNOW",
        "SNOWBANK",
        "SOFT_COLLISION_GENERAL",
        "VOID",
        "WATER",
        "WOOD",
    }
)


@dataclass(frozen=True)
class SurfaceProfile:
    profile_id: str
    ground_type: str
    hard: bool
    bendable: bool
    water: bool
    collision_eligible: bool
    ground_depth: float = 0.0
    snowbank: bool = False


NEUTRAL_SURFACE_PROFILE = SurfaceProfile(
    "neutral",
    "GRAVEL",
    True,
    False,
    False,
    True,
    0.0,
)
SOLID_GROUND_TYPES = GROUND_TYPES - {"SNOWBANK", "VOID", "WATER"}


@dataclass(frozen=True)
class SurfaceRules:
    unknown_ground_type: str = NEUTRAL_SURFACE_PROFILE.ground_type

    @property
    def unknown_profile(self) -> SurfaceProfile:
        return replace(NEUTRAL_SURFACE_PROFILE, ground_type=self.unknown_ground_type)


DEFAULT_SURFACE_RULES = SurfaceRules()
_active_surface_rules: ContextVar[SurfaceRules | None] = ContextVar(
    "rbr2beamng_surface_rules",
    default=None,
)


@contextmanager
def use_surface_rules(rules: SurfaceRules) -> Iterator[SurfaceRules]:
    token = _active_surface_rules.set(rules)
    try:
        yield rules
    finally:
        _active_surface_rules.reset(token)


def current_surface_rules() -> SurfaceRules:
    return _active_surface_rules.get() or DEFAULT_SURFACE_RULES


def surface_rules_from_data(raw: Mapping[str, Any]) -> SurfaceRules:
    unknown = raw.get("unknownGroundType", DEFAULT_SURFACE_RULES.unknown_ground_type)
    if unknown not in SOLID_GROUND_TYPES:
        raise ConversionError(
            f"Surface rule unknownGroundType has unsupported ground type {unknown!r}"
        )
    return SurfaceRules(unknown)


def surface_rules_data(rules: SurfaceRules) -> dict[str, object]:
    data: dict[str, object] = {}
    if rules.unknown_ground_type != DEFAULT_SURFACE_RULES.unknown_ground_type:
        data["unknownGroundType"] = rules.unknown_ground_type
    return data


def load_surface_rules() -> SurfaceRules:
    path = surface_profile_override_path()
    if not current_filesystem().is_file(path):
        return DEFAULT_SURFACE_RULES
    raw = _load_catalog_data(path).get("rules", {})
    if not isinstance(raw, Mapping):
        raise ConversionError(f"Surface profile catalog {path} has invalid rules")
    return surface_rules_from_data(raw)


# (dry, wet) ground types per physics.lsp surface Type; Type 0 is generic.
_TYPE_GROUND_TYPES = {
    1: ("GRAVEL", "GRAVEL_WET"),
    2: ("ASPHALT", "ASPHALT_WET"),
    3: ("SNOW", "SNOW"),
    4: ("DIRT", "MUD"),
    5: ("GRASS", "GRASS"),
}


def derived_surface_profile(
    coefficients: Mapping[str, float],
    flags: tuple[str, ...],
) -> SurfaceProfile:
    hard = "+HARD" in flags and "-HARD" not in flags
    bendable = "+BENDABLE" in flags
    wet = coefficients.get("WaterFactor", 0.0) > 0.0
    if bendable:
        ground_type = "SOFT_COLLISION_GENERAL"
    elif not hard:
        ground_type = "WATER" if wet else "VOID"
    else:
        unknown = current_surface_rules().unknown_ground_type
        dry_type, wet_type = _TYPE_GROUND_TYPES.get(
            int(coefficients.get("Type", 0.0)),
            (unknown, unknown),
        )
        ground_type = wet_type if wet else dry_type
    return SurfaceProfile(
        "derived",
        ground_type,
        hard,
        bendable,
        ground_type == "WATER",
        hard or bendable,
    )


@dataclass(frozen=True)
class SurfaceProfileCatalog:
    version: int
    profiles: Mapping[str, SurfaceProfile]
    source_names: Mapping[str, str]

    def resolve(
        self,
        name: str,
        coefficients: Mapping[str, float],
        flags: tuple[str, ...],
    ) -> tuple[SurfaceProfile, str]:
        profile_id = self.source_names.get(name.casefold())
        if profile_id is None:
            return derived_surface_profile(coefficients, flags), "derived"
        return self.profiles[profile_id], "mapped"


def physics_lsp_fingerprint(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _profile_from_data(profile_id: str, raw: Mapping[str, Any]) -> SurfaceProfile:
    ground_type = raw.get("groundType")
    if ground_type not in GROUND_TYPES:
        raise ConversionError(f"Surface profile {profile_id!r} has unsupported ground type {ground_type!r}")
    depth = raw.get("groundDepth", 0.0)
    if not isinstance(depth, (int, float)) or not math.isfinite(depth) or depth < 0.0:
        raise ConversionError(f"Surface profile {profile_id!r} has invalid ground depth")
    flags = ("hard", "bendable", "water", "collisionEligible", "snowbank")
    if any(not isinstance(raw.get(flag, False), bool) for flag in flags):
        raise ConversionError(f"Surface profile {profile_id!r} has non-boolean behavior")
    return SurfaceProfile(
        profile_id,
        ground_type,
        raw.get("hard", False),
        raw.get("bendable", False),
        raw.get("water", ground_type == "WATER"),
        raw.get("collisionEligible", True),
        float(depth),
        raw.get("snowbank", False),
    )


def _load_catalog_data(path: Path) -> Mapping[str, Any]:
    filesystem = current_filesystem()
    try:
        raw = json.loads(filesystem.read_text(path, encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ConversionError(f"Unable to read surface profile catalog {path}: {exc}") from exc
    if not isinstance(raw, dict) or not isinstance(raw.get("version"), int):
        raise ConversionError(f"Surface profile catalog {path} has invalid schema")
    return raw


def _catalog_from_data(raw: Mapping[str, Any]) -> SurfaceProfileCatalog:
    raw_profiles = raw.get("profiles", {})
    if not isinstance(raw_profiles, dict):
        raise ConversionError("Surface profile catalog has invalid profiles")
    profiles = {
        profile_id: _profile_from_data(profile_id, profile)
        for profile_id, profile in raw_profiles.items()
        if isinstance(profile_id, str) and isinstance(profile, dict)
    }
    source_names: dict[str, str] = {}
    for profile_id in profiles:
        for name in raw_profiles[profile_id].get("sourceNames", ()):
            if not isinstance(name, str) or not name:
                raise ConversionError(f"Surface profile {profile_id!r} has invalid source name")
            if name.casefold() in source_names:
                raise ConversionError(f"Surface {name!r} is mapped more than once")
            source_names[name.casefold()] = profile_id
    return SurfaceProfileCatalog(raw["version"], profiles, source_names)


def bundled_surface_profile_catalog() -> SurfaceProfileCatalog:
    return _catalog_from_data(
        _load_catalog_data(Path(__file__).with_name("data") / "surface_profiles.json")
    )


def surface_profile_catalog() -> SurfaceProfileCatalog:
    catalog = bundled_surface_profile_catalog()
    override_path = surface_profile_override_path()
    if not current_filesystem().is_file(override_path):
        return catalog
    override = _catalog_from_data(_load_catalog_data(override_path))
    return SurfaceProfileCatalog(
        catalog.version,
        {**catalog.profiles, **override.profiles},
        {**catalog.source_names, **override.source_names},
    )


def surface_profile_override_path() -> Path:
    return settings_path().with_name("surface_profiles.json")
