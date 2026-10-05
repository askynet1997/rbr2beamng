from __future__ import annotations

import math

from .binary import BinaryReader, FormatError, Source, checked_count, load_source
from .models import (
    DlsAnimationSetDescriptor,
    DlsFile,
    OriginalPacenote,
)


DLS_HEADER = b"MINAATAD\x12\x00\x00\x00\x00\x00\x00\x00"
DLS_SECTION_COUNT = 10
ANIMATION_SETS_SECTION = 0
ANIMATION_NAMES_SECTION = 7
ANIMATION_SET_SECTION_COUNT = 8
PACENOTES_SECTION = 5
DESCRIPTOR_SIZE = 68


def _directory(
    data: memoryview,
) -> tuple[tuple[int | None, ...], dict[int, memoryview], dict[int, int]]:
    reader = BinaryReader(data, context="DLS header")
    header = reader.read(len(DLS_HEADER))
    if header != DLS_HEADER:
        raise FormatError("Invalid DLS header", offset=0, context="DLS")
    raw_offsets = reader.unpack(f"<{DLS_SECTION_COUNT}I")
    offsets: list[int | None] = []
    real_offsets: list[int] = []
    for index, raw_offset in enumerate(raw_offsets):
        if raw_offset in (0, 0xFFFFFFFF):
            offsets.append(None)
            continue
        if raw_offset < reader.offset or raw_offset > len(data):
            raise FormatError(
                f"DLS section {index} offset 0x{raw_offset:x} is out of bounds",
                offset=16 + index * 4,
                context="DLS directory",
            )
        if raw_offset % 4:
            raise FormatError(
                f"DLS section {index} offset is not 4-byte aligned",
                offset=16 + index * 4,
                context="DLS directory",
            )
        offsets.append(raw_offset)
        real_offsets.append(raw_offset)

    boundaries = sorted(set(real_offsets + [len(data)]))
    sections: dict[int, memoryview] = {}
    section_ends: dict[int, int] = {}
    for index, offset in enumerate(offsets):
        if offset is None:
            continue
        end = next((value for value in boundaries if value > offset), len(data))
        sections[index] = data[offset:end]
        section_ends[index] = end
    return tuple(offsets), sections, section_ends


def _parse_names(
    data: memoryview,
    offset: int | None,
    section_end: int | None,
) -> dict[int, str]:
    if offset is None:
        return {}
    reader = BinaryReader(
        data,
        start=offset,
        end=section_end,
        context="DLS animation names",
    )
    size_offset = reader.offset
    size = reader.u32()
    if size > reader.remaining:
        raise FormatError(
            f"DLS names blob declares {size} bytes with only {reader.remaining} available",
            offset=size_offset,
            context="DLS animation names",
        )
    blob = reader.view(size)
    if not blob or blob[-1] != 0:
        raise FormatError(
            "DLS names blob is not NUL-terminated",
            offset=reader.offset - size,
            context="DLS animation names",
        )
    names: dict[int, str] = {}
    position = 0
    while position < len(blob):
        end = position
        while end < len(blob) and blob[end] != 0:
            end += 1
        if end == len(blob):
            raise FormatError(
                "Unterminated DLS name",
                offset=offset + 4 + position,
                context="DLS animation names",
            )
        if end == position:
            if end != len(blob) - 1:
                raise FormatError(
                    "Unexpected empty DLS name",
                    offset=offset + 4 + position,
                    context="DLS animation names",
                )
            break
        names[position] = bytes(blob[position:end]).decode("latin-1")
        position = end + 1
    return names


def _next_boundary(address: int, landmarks: list[int], data_size: int) -> int:
    return next((item for item in landmarks if item > address), data_size)


def parse_dls(
    source: Source,
    *,
    max_animation_sets: int = 10_000,
    max_items_per_section: int = 2_000_000,
) -> DlsFile:
    loaded = load_source(source)
    if len(loaded.view) < len(DLS_HEADER) + DLS_SECTION_COUNT * 4:
        raise FormatError("DLS file is shorter than its header", offset=0, context="DLS")
    directory, sections, section_ends = _directory(loaded.view)
    names = _parse_names(
        loaded.view,
        directory[ANIMATION_NAMES_SECTION],
        section_ends.get(ANIMATION_NAMES_SECTION),
    )

    animation_offset = directory[ANIMATION_SETS_SECTION]
    descriptor_rows: list[tuple[int, int]] = []
    if animation_offset is not None:
        animation_reader = BinaryReader(
            loaded.view,
            start=animation_offset,
            end=section_ends[ANIMATION_SETS_SECTION],
            context="DLS animation sets",
        )
        count_offset = animation_reader.offset
        set_count = checked_count(
            animation_reader.u32(),
            limit=max_animation_sets,
            offset=count_offset,
            context="DLS animation set count",
        )
        for index in range(set_count):
            row_offset = animation_reader.offset
            descriptor_offset, descriptor_size = animation_reader.unpack("<II")
            if descriptor_size < DESCRIPTOR_SIZE:
                raise FormatError(
                    f"Animation-set descriptor {index} is {descriptor_size} bytes; "
                    f"minimum is {DESCRIPTOR_SIZE}",
                    offset=row_offset,
                    context="DLS animation sets",
                )
            if (
                descriptor_offset < len(DLS_HEADER) + DLS_SECTION_COUNT * 4
                or descriptor_offset > len(loaded.view) - DESCRIPTOR_SIZE
                or descriptor_offset + descriptor_size > len(loaded.view)
            ):
                raise FormatError(
                    f"Animation-set descriptor {index} is out of bounds",
                    offset=row_offset,
                    context="DLS animation sets",
                )
            descriptor_rows.append((descriptor_offset, descriptor_size))

    raw_descriptors: list[
        tuple[int, int, int, tuple[int, ...], tuple[int, ...]]
    ] = []
    landmarks = [len(loaded.view)]
    landmarks.extend(offset for offset in directory if offset is not None)
    landmarks.extend(offset for offset, _ in descriptor_rows)
    for descriptor_offset, descriptor_size in descriptor_rows:
        reader = BinaryReader(
            loaded.view,
            start=descriptor_offset,
            end=descriptor_offset + descriptor_size,
            context="DLS animation-set descriptor",
        )
        name_offset = reader.u32()
        counts = reader.unpack(f"<{ANIMATION_SET_SECTION_COUNT}I")
        offsets = reader.unpack(f"<{ANIMATION_SET_SECTION_COUNT}I")
        for section_index, count in enumerate(counts):
            checked_count(
                count,
                limit=max_items_per_section,
                offset=descriptor_offset + 4 + section_index * 4,
                context=f"animation set section {section_index} count",
            )
            section_offset = offsets[section_index]
            if section_offset not in (0, 0xFFFFFFFF):
                if section_offset > len(loaded.view):
                    raise FormatError(
                        f"Animation-set section {section_index} offset is out of bounds",
                        offset=descriptor_offset + 36 + section_index * 4,
                        context="DLS animation-set descriptor",
                    )
                landmarks.append(section_offset)
            elif count:
                raise FormatError(
                    f"Animation-set section {section_index} has count {count} "
                    "but no data offset",
                    offset=descriptor_offset + 36 + section_index * 4,
                    context="DLS animation-set descriptor",
                )
        raw_descriptors.append(
            (descriptor_offset, descriptor_size, name_offset, counts, offsets)
        )
    landmarks = sorted(set(landmarks))

    descriptors: list[DlsAnimationSetDescriptor] = []
    all_pacenotes: list[OriginalPacenote] = []
    driveline_pacenotes: list[OriginalPacenote] = []
    for descriptor_offset, descriptor_size, name_offset, counts, offsets in raw_descriptors:
        if name_offset not in names:
            raise FormatError(
                f"Animation-set name offset {name_offset} is not in the names table",
                offset=descriptor_offset,
                context="DLS animation-set descriptor",
            )
        name = names[name_offset]
        raw_sections: dict[int, memoryview] = {}
        for section_index, address in enumerate(offsets):
            if address in (0, 0xFFFFFFFF):
                continue
            end = _next_boundary(address, landmarks, len(loaded.view))
            raw_sections[section_index] = loaded.view[address:end]

        note_count = counts[PACENOTES_SECTION]
        note_address = offsets[PACENOTES_SECTION]
        notes: list[OriginalPacenote] = []
        if note_count:
            required_end = note_address + note_count * 12
            section_end = _next_boundary(note_address, landmarks, len(loaded.view))
            if required_end > section_end:
                raise FormatError(
                    f"{note_count} pacenotes exceed their animation-set section",
                    offset=note_address,
                    context=name,
                )
            note_reader = BinaryReader(
                loaded.view,
                start=note_address,
                end=required_end,
                context=f"DLS pacenotes {name}",
            )
            for _ in range(note_count):
                note_offset = note_reader.offset
                note_id, flags, distance = note_reader.unpack("<IIf")
                if not math.isfinite(distance):
                    raise FormatError(
                        "Non-finite pacenote distance",
                        offset=note_offset,
                        context=name,
                    )
                notes.append(OriginalPacenote(note_id, flags, distance, name))
        all_pacenotes.extend(notes)
        if name.casefold() == "driveline":
            driveline_pacenotes.extend(notes)
        descriptors.append(
            DlsAnimationSetDescriptor(
                name=name,
                name_offset=name_offset,
                descriptor_offset=descriptor_offset,
                descriptor_size=descriptor_size,
                counts=counts,
                offsets=offsets,
                raw_sections=raw_sections,
            )
        )

    return DlsFile(
        header=DLS_HEADER,
        directory=directory,
        sections=sections,
        names=names,
        animation_sets=tuple(descriptors),
        all_pacenotes=tuple(all_pacenotes),
        pacenotes=tuple(driveline_pacenotes),
        raw=loaded.view,
    )
