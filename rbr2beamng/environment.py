from __future__ import annotations

import math
from dataclasses import dataclass, replace
from datetime import date, timedelta

from .core import ConversionError
from .models import StageLocation


WEATHER_PRESETS = {
    "morning": {
        "timeOfDay": 0.875,
        "cloudWindSpeed": 0.5,
        "cloudCover": 1.4,
        "fogDensity": 0.00025,
        "fogHeight": 500,
    },
    "noon": {
        "timeOfDay": 0.0,
        "cloudWindSpeed": 0.4,
        "cloudCover": 1.15,
        "fogDensity": 0.00001,
        "fogHeight": 150,
    },
    "evening": {
        "timeOfDay": 0.20833333333333334,
        "cloudWindSpeed": 0.45,
        "cloudCover": 1.25,
        "fogDensity": 0.00001,
        "fogHeight": 150,
    },
    "overcast": {
        "timeOfDay": 0.0,
        "cloudWindSpeed": 0.8,
        "cloudCover": 2.0,
        "fogDensity": 0.00005,
        "fogHeight": 150,
    },
    "day": {
        "timeOfDay": 0.0,
        "cloudWindSpeed": 0.6,
        "cloudCover": 1.575,
        "fogDensity": 0.00003,
        "fogHeight": 150,
    },
}

DEFAULT_ENVIRONMENT_DATE = date(2026, 3, 15)
_SUNRISE_ELEVATION = 0.0
_DEG = math.pi / 180.0


@dataclass(frozen=True)
class EnvironmentSettings:
    preset: str
    temperature_night: float
    temperature_day: float
    time_of_day: float
    cloud_wind_speed: float
    cloud_cover: float
    fog_density: float
    fog_height: float
    calendar_date: date
    utc_offset: float | None
    sun_azimuth: float | None
    sun_elevation: float | None
    warnings: tuple[str, ...] = ()


def location_override(
    location: StageLocation | None,
    latitude: float | None,
    longitude: float | None,
) -> StageLocation | None:
    if latitude is None and longitude is None:
        return location
    if latitude is None or longitude is None:
        raise ConversionError("Latitude and longitude must be provided together")
    if (
        not math.isfinite(latitude)
        or not math.isfinite(longitude)
        or not -90.0 <= latitude <= 90.0
        or not -180.0 <= longitude <= 180.0
    ):
        raise ConversionError(
            "Latitude must be -90..90 and longitude must be -180..180"
        )
    if location is not None:
        return replace(
            location,
            latitude=latitude,
            longitude=longitude,
            precision="exact",
        )
    return StageLocation(
        country_code="",
        country="Unknown",
        region="",
        latitude=latitude,
        longitude=longitude,
        utc_offset="0",
        dst_rule="",
        precision="exact",
    )


def preset_for_surface(surface: str) -> str:
    value = surface.strip().casefold()
    return value if value in {"snow", "tarmac", "gravel"} else "gravel"


def parse_environment_date(value: str | None) -> date | None:
    if value is None or not value.strip():
        return None
    try:
        return date.fromisoformat(value.strip())
    except ValueError as exc:
        raise ConversionError(
            "Environment date must use YYYY-MM-DD format"
        ) from exc


def _location_coordinates(
    location: StageLocation | None,
) -> tuple[float, float] | None:
    if location is None:
        return None
    try:
        latitude = float(location.latitude)
        longitude = float(location.longitude)
    except (AttributeError, TypeError, ValueError):
        return None
    if (
        not math.isfinite(latitude)
        or not math.isfinite(longitude)
        or not -90.0 <= latitude <= 90.0
        or not -180.0 <= longitude <= 180.0
    ):
        return None
    return latitude, longitude


def _sunday(year: int, month: int, occurrence: int) -> int:
    first = date(year, month, 1)
    first_sunday = 1 + (6 - first.weekday()) % 7
    return first_sunday + (occurrence - 1) * 7


def _last_sunday(year: int, month: int) -> int:
    if month == 12:
        next_month = date(year + 1, 1, 1)
    else:
        next_month = date(year, month + 1, 1)
    last_day = next_month - timedelta(days=1)
    return last_day.day - (last_day.weekday() + 1) % 7


def _dst_active(rule: str, calendar_date: date) -> bool:
    rule = rule.strip().casefold()
    year = calendar_date.year
    if rule == "eu":
        start = date(year, 3, _last_sunday(year, 3))
        end = date(year, 10, _last_sunday(year, 10))
        return start <= calendar_date < end
    if rule == "us":
        start = date(year, 3, _sunday(year, 3, 2))
        end = date(year, 11, _sunday(year, 11, 1))
        return start <= calendar_date < end
    if rule == "au":
        start = date(year, 10, _sunday(year, 10, 1))
        end = date(year, 4, _sunday(year, 4, 1))
        return calendar_date >= start or calendar_date < end
    return False


def effective_utc_offset(
    location: StageLocation | None,
    calendar_date: date,
) -> float | None:
    if location is None:
        return None
    try:
        standard_offset = float(getattr(location, "utc_offset"))
    except (AttributeError, TypeError, ValueError):
        return None
    return standard_offset + (
        1.0
        if _dst_active(getattr(location, "dst_rule", ""), calendar_date)
        else 0.0
    )


def _wrap_degrees(value: float) -> float:
    return value % 360.0


def _julian_date(calendar_date: date, utc_hours: float) -> float:
    year = calendar_date.year
    month = calendar_date.month
    if month <= 2:
        year -= 1
        month += 12
    century = math.floor(year / 100)
    correction = 2 - century + math.floor(century / 4)
    return (
        math.floor(365.25 * (year + 4716))
        + math.floor(30.6001 * (month + 1))
        + calendar_date.day
        + correction
        - 1524.5
        + utc_hours / 24.0
    )


def _refraction_degrees(elevation: float) -> float:
    clamped_elevation = max(-0.9, elevation)
    correction = 1.02 / math.tan(
        (clamped_elevation + 10.3 / (clamped_elevation + 5.11)) * _DEG
    ) / 60.0
    if elevation < -0.9:
        correction *= max(0.0, (elevation + 2.0) / 1.1)
    return correction


def solar_position(
    calendar_date: date,
    local_hours: float,
    latitude: float,
    longitude: float,
    utc_offset: float,
) -> tuple[float, float]:
    julian_day = _julian_date(calendar_date, local_hours - utc_offset)
    days = julian_day - 2451543.5
    perihelion = 282.9404 + 4.70935e-5 * days
    eccentricity = 0.016709 - 1.151e-9 * days
    mean_anomaly = _wrap_degrees(356.0470 + 0.9856002585 * days)
    obliquity = 23.4393 - 3.563e-7 * days
    eccentric_anomaly = mean_anomaly + math.degrees(
        eccentricity
        * math.sin(mean_anomaly * _DEG)
        * (1.0 + eccentricity * math.cos(mean_anomaly * _DEG))
    )
    orbital_x = math.cos(eccentric_anomaly * _DEG) - eccentricity
    orbital_y = math.sin(eccentric_anomaly * _DEG) * math.sqrt(
        1.0 - eccentricity * eccentricity
    )
    orbital_true_anomaly = math.degrees(math.atan2(orbital_y, orbital_x))
    orbital_radius = math.hypot(orbital_x, orbital_y)
    ecliptic_longitude = _wrap_degrees(orbital_true_anomaly + perihelion)
    ecliptic_x = orbital_radius * math.cos(ecliptic_longitude * _DEG)
    ecliptic_y = orbital_radius * math.sin(ecliptic_longitude * _DEG)
    equatorial_x = ecliptic_x
    equatorial_y = ecliptic_y * math.cos(obliquity * _DEG)
    equatorial_z = ecliptic_y * math.sin(obliquity * _DEG)
    right_ascension = math.atan2(equatorial_y, equatorial_x)
    declination = math.atan2(
        equatorial_z,
        math.hypot(equatorial_x, equatorial_y),
    )
    greenwich_sidereal_time = _wrap_degrees(
        280.46061837 + 360.98564736629 * (julian_day - 2451545.0)
    )
    local_sidereal_time = _wrap_degrees(
        greenwich_sidereal_time + longitude
    ) * _DEG
    latitude_radians = latitude * _DEG
    hour_angle = local_sidereal_time - right_ascension
    sine_elevation = max(
        -1.0,
        min(
            1.0,
            math.sin(latitude_radians) * math.sin(declination)
            + math.cos(latitude_radians)
            * math.cos(declination)
            * math.cos(hour_angle),
        ),
    )
    elevation = math.asin(sine_elevation)
    cosine_elevation = math.cos(elevation)
    azimuth = 0.0
    if abs(cosine_elevation) > 1e-6 and abs(math.cos(latitude_radians)) > 1e-6:
        azimuth = math.acos(
            max(
                -1.0,
                min(
                    1.0,
                    (
                        math.sin(declination)
                        - math.sin(latitude_radians) * sine_elevation
                    )
                    / (math.cos(latitude_radians) * cosine_elevation),
                ),
            )
        )
        if math.sin(hour_angle) > 0.0:
            azimuth = 2.0 * math.pi - azimuth
    elevation_degrees = math.degrees(elevation)
    return (
        _wrap_degrees(math.degrees(azimuth)),
        elevation_degrees + _refraction_degrees(elevation_degrees),
    )


def _solar_event_hours(
    calendar_date: date,
    latitude: float,
    longitude: float,
    utc_offset: float,
) -> tuple[float | None, float | None]:
    def elevation_at(local_hours: float) -> float:
        return solar_position(
            calendar_date,
            local_hours,
            latitude,
            longitude,
            utc_offset,
        )[1] - _SUNRISE_ELEVATION

    def refine_crossing(lower: float, upper: float) -> float:
        lower_value = elevation_at(lower)
        for _ in range(20):
            middle = (lower + upper) * 0.5
            middle_value = elevation_at(middle)
            if (lower_value <= 0.0) == (middle_value <= 0.0):
                lower = middle
                lower_value = middle_value
            else:
                upper = middle
        return (lower + upper) * 0.5

    sunrise = None
    sunset = None
    previous_hours = 0.0
    previous_value = elevation_at(previous_hours)
    for index in range(1, 145):
        current_hours = index / 6.0
        current_value = elevation_at(current_hours)
        if previous_value <= 0.0 < current_value:
            sunrise = refine_crossing(previous_hours, current_hours)
        elif previous_value >= 0.0 > current_value:
            sunset = refine_crossing(previous_hours, current_hours)
        previous_hours = current_hours
        previous_value = current_value
    return sunrise, sunset


def sunrise_sunset(
    calendar_date: date,
    latitude: float,
    longitude: float,
    utc_offset: float,
) -> tuple[float | None, float | None]:
    return _solar_event_hours(
        calendar_date,
        latitude,
        longitude,
        utc_offset,
    )


def _time_of_day_from_hours(local_hours: float) -> float:
    return (local_hours / 24.0 - 0.5) % 1.0


def _format_clock(local_hours: float) -> str:
    minutes = round((local_hours % 24.0) * 60.0) % (24 * 60)
    return f"{minutes // 60:02}:{minutes % 60:02}"


def _event_time_of_day(
    variant: str,
    calendar_date: date,
    location: StageLocation | None,
    utc_offset: float | None,
) -> tuple[float, tuple[str, ...]]:
    fallback_hours = 9.0 if variant == "M" else 17.0
    label = "Morning" if variant == "M" else "Evening"
    coordinates = _location_coordinates(location)
    if coordinates is None or utc_offset is None:
        return (
            _time_of_day_from_hours(fallback_hours),
            (
                f"{label} environment has no usable GPS/timezone; "
                f"using {_format_clock(fallback_hours)} local time",
            ),
        )
    sunrise, sunset = _solar_event_hours(
        calendar_date,
        coordinates[0],
        coordinates[1],
        utc_offset,
    )
    if sunrise is not None and sunset is not None:
        hours = sunrise + 0.5 if variant == "M" else sunset - 1.0
        return _time_of_day_from_hours(hours), ()
    _, noon_elevation = solar_position(
        calendar_date,
        12.0,
        coordinates[0],
        coordinates[1],
        utc_offset,
    )
    if noon_elevation > _SUNRISE_ELEVATION:
        return (
            _time_of_day_from_hours(fallback_hours),
            (
                f"{label} environment has no sunrise/sunset on "
                f"{calendar_date.isoformat()}; using {_format_clock(fallback_hours)} "
                "local time",
            ),
        )
    return (
        0.0,
        (
            f"{label} environment has no daylight on "
            f"{calendar_date.isoformat()}; using local noon",
        ),
    )


def location_temperatures_for_surface(
    surface: str,
    location: StageLocation | None,
) -> tuple[float, float] | None:
    if location is None:
        return None
    selected = preset_for_surface(surface)
    summer = (
        location.summer_temperature_night,
        location.summer_temperature_day,
    )
    winter = (
        location.winter_temperature_night,
        location.winter_temperature_day,
    )
    if selected == "snow":
        return winter if None not in winter else None
    if selected == "tarmac":
        return summer if None not in summer else None
    autumn = (
        location.autumn_temperature_night,
        location.autumn_temperature_day,
    )
    return autumn if None not in autumn else None


def resolve_environment_settings(
    surface: str,
    temperature_night: float | None = None,
    temperature_day: float | None = None,
    *,
    location: StageLocation | None = None,
    environment_variant: str | None = None,
    environment_date: str | None = None,
) -> EnvironmentSettings:
    selected = preset_for_surface(surface)
    location_temperatures = location_temperatures_for_surface(surface, location)
    if location_temperatures is None:
        if temperature_night is None or temperature_day is None:
            raise ConversionError(
                "Temperature data is unavailable; provide both night and day temperatures"
            )
        default_night, default_day = temperature_night, temperature_day
    else:
        default_night, default_day = location_temperatures
    night = float(default_night if temperature_night is None else temperature_night)
    day = float(default_day if temperature_day is None else temperature_day)
    if not math.isfinite(night) or not math.isfinite(day):
        raise ConversionError("Temperatures must be finite numbers")
    if night > day:
        raise ConversionError("Night temperature cannot exceed day temperature")
    environment = environment_values_for_variant(environment_variant)
    calendar_date = parse_environment_date(environment_date) or DEFAULT_ENVIRONMENT_DATE
    utc_offset = effective_utc_offset(location, calendar_date)
    variant = (environment_variant or "").strip().upper()
    time_of_day = environment["timeOfDay"]
    warnings: tuple[str, ...] = ()
    if variant in {"M", "E"}:
        time_of_day, warnings = _event_time_of_day(
            variant,
            calendar_date,
            location,
            utc_offset,
        )
    sun_azimuth = None
    sun_elevation = None
    coordinates = _location_coordinates(location)
    if coordinates is not None and utc_offset is not None:
        local_hours = ((time_of_day + 0.5) % 1.0) * 24.0
        sun_azimuth, sun_elevation = solar_position(
            calendar_date,
            local_hours,
            coordinates[0],
            coordinates[1],
            utc_offset,
        )
    return EnvironmentSettings(
        preset=selected,
        temperature_night=night,
        temperature_day=day,
        time_of_day=time_of_day,
        cloud_wind_speed=environment["cloudWindSpeed"],
        cloud_cover=environment["cloudCover"],
        fog_density=environment["fogDensity"],
        fog_height=environment["fogHeight"],
        calendar_date=calendar_date,
        utc_offset=utc_offset,
        sun_azimuth=sun_azimuth,
        sun_elevation=sun_elevation,
        warnings=warnings,
    )


def environment_values_for_variant(
    environment_variant: str | None,
) -> dict[str, float]:
    variant = (environment_variant or "").strip().upper()
    preset = {
        "D": "day",
        "M": "morning",
        "E": "evening",
        "O": "overcast",
    }.get(variant, "noon")
    return dict(WEATHER_PRESETS[preset])
