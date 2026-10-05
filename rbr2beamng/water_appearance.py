from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from .models import WaterAppearance


DEFAULT_WATER_PROFILE = "beamng-default"


@dataclass(frozen=True)
class WaterProfileManifest:
    profiles: Mapping[str, WaterAppearance]
    source_ids: Mapping[str, str]
    default_profile: str = DEFAULT_WATER_PROFILE


DEFAULT_WATER_PROFILE_MANIFEST = WaterProfileManifest(
    profiles={DEFAULT_WATER_PROFILE: WaterAppearance()},
    source_ids={},
)


def water_profile_manifest_from_data(value: Mapping) -> WaterProfileManifest:
    profiles = {
        DEFAULT_WATER_PROFILE: WaterAppearance(),
        **{
            name: WaterAppearance(**profile)
            for name, profile in value.get("profiles", {}).items()
        },
    }
    return WaterProfileManifest(
        profiles,
        value.get("sourceIds", {}),
        value.get("defaultProfile", DEFAULT_WATER_PROFILE),
    )


def water_profile_manifest_data(profiles: WaterProfileManifest) -> dict[str, object]:
    return {
        "defaultProfile": profiles.default_profile,
        "profiles": {
            name: {
                field_name: value
                for field_name, value in vars(appearance).items()
                if value is not None
            }
            for name, appearance in sorted(profiles.profiles.items())
        },
        "sourceIds": dict(sorted(profiles.source_ids.items())),
    }


def water_appearance_for_source(
    source_id: str | None,
    profiles: WaterProfileManifest = DEFAULT_WATER_PROFILE_MANIFEST,
) -> WaterAppearance:
    return profiles.profiles[profiles.source_ids.get(source_id, profiles.default_profile)]
