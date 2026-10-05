"""Deflate and checksum ZIP entries with zlib-ng, which zipfile has no option for."""

from __future__ import annotations

import zipfile

from zlib_ng import zlib_ng

ZIP_COMPRESSLEVEL = 2

_stdlib_get_compressor = zipfile._get_compressor


def _get_compressor(compress_type, compresslevel=None):
    if compress_type == zipfile.ZIP_DEFLATED:
        return zlib_ng.compressobj(
            zlib_ng.Z_DEFAULT_COMPRESSION if compresslevel is None else compresslevel,
            zlib_ng.DEFLATED,
            -15,
        )
    return _stdlib_get_compressor(compress_type, compresslevel)


zipfile._get_compressor = _get_compressor
zipfile.crc32 = zlib_ng.crc32
