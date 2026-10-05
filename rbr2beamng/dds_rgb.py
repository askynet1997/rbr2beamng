"""Vectorized replacement for Pillow's per-pixel uncompressed DDS decoder.

Pillow decodes uncompressed (bitmask RGB/RGBA) DDS files in pure Python, one
pixel at a time. This decoder computes the same bytes with numpy.
"""

from __future__ import annotations

import numpy as np
from PIL import DdsImagePlugin, Image


class DdsRgbDecoder(DdsImagePlugin.DdsRgbDecoder):
    def decode(self, buffer: Image.DecoderInput) -> tuple[int, int]:
        bitcount, masks = self.args
        bytecount = bitcount // 8
        if not bytecount:
            self.set_as_raw(bytearray())
            return -1, 0

        assert self.fd is not None
        data = self.fd.read(self.state.xsize * self.state.ysize * bytecount)
        count = len(data) // bytecount
        pixel_bytes = np.frombuffer(
            data,
            dtype=np.uint8,
            count=count * bytecount,
        ).reshape((count, bytecount))
        values = np.zeros(count, dtype=np.uint64)
        for index in range(bytecount):
            values |= pixel_bytes[:, index].astype(np.uint64) << (8 * index)

        channels = np.zeros((count, len(masks)), dtype=np.uint8)
        for channel, mask in enumerate(masks):
            offset = 0
            if mask != 0:
                while mask >> (offset + 1) << (offset + 1) == mask:
                    offset += 1
            total = mask >> offset
            if total:
                channels[:, channel] = (
                    ((values & mask) >> offset) / total * 255
                ).astype(np.uint8)
        self.set_as_raw(channels.tobytes())
        return -1, 0


Image.register_decoder("dds_rgb", DdsRgbDecoder)
# Stage textures can be as large as Direct3D allows (16384x16384).
Image.MAX_IMAGE_PIXELS = 16384 * 16384
