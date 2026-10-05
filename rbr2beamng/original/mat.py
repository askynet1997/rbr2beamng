from __future__ import annotations

import numpy as np

from .binary import BinaryReader, FormatError, Source, checked_count, load_source
from .models import MatFile, MaterialCondition, MaterialMap


SURFACES = {"dry", "damp", "wet"}
AGES = {"new", "normal", "worn"}


def _condition_identifier(identifier: str, offset: int) -> tuple[str, str, str]:
    parts = identifier.rsplit(None, 2)
    if len(parts) != 3:
        raise FormatError(
            f"Invalid MAT condition identifier {identifier!r}",
            offset=offset,
            context="MAT condition",
        )
    name, surface, age = parts[0], parts[1].casefold(), parts[2].casefold()
    if surface not in SURFACES or age not in AGES:
        raise FormatError(
            f"Invalid MAT condition suffix {surface!r} {age!r}",
            offset=offset,
            context="MAT condition",
        )
    return name, surface, age


def parse_mat(
    source: Source,
    *,
    max_conditions: int = 64,
    max_maps_per_condition: int = 256,
) -> MatFile:
    loaded = load_source(source)
    reader = BinaryReader(loaded.view, context=f"MAT {loaded.name}")
    count_offset = reader.offset
    condition_count = checked_count(
        reader.u32(),
        limit=max_conditions,
        offset=count_offset,
        context="MAT condition count",
    )
    conditions: list[MaterialCondition] = []
    for _ in range(condition_count):
        identifier_offset = reader.offset
        identifier = reader.cstring()
        name, surface, age = _condition_identifier(identifier, identifier_offset)
        map_count_offset = reader.offset
        map_count = checked_count(
            reader.u32(),
            limit=max_maps_per_condition,
            offset=map_count_offset,
            context="MAT map count",
        )
        maps: list[MaterialMap] = []
        for _ in range(map_count):
            dimension_offset = reader.offset
            width, height = reader.unpack("<II")
            if width != 16 or height != 16:
                raise FormatError(
                    f"MAT map is {width}x{height}; expected 16x16",
                    offset=dimension_offset,
                    context=identifier,
                )
            raw = reader.view(width * height)
            values = np.frombuffer(raw, dtype=np.uint8).reshape((height, width))
            maps.append(MaterialMap(width, height, values, raw))
        conditions.append(
            MaterialCondition(
                name=name,
                surface=surface,
                age=age,
                identifier=identifier,
                maps=tuple(maps),
            )
        )
    trailing = reader.remaining_view()
    return MatFile(tuple(conditions), trailing, loaded.view)
