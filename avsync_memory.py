"""Bounded media analysis helpers. No additional runtime dependencies."""

from collections import OrderedDict
from io import BytesIO
import math
import os
import struct

import numpy as np
from scipy.io import wavfile


class WavChunks:
    """Read PCM/IEEE WAV blocks through SciPy without mapping a whole track.

    A small standalone WAV is presented to SciPy per block, retaining its sample
    format support (including packed 24-bit, extensible WAV, RIFX and RF64).
    Reading blocks rather than a whole-file mmap also bounds resident pages.
    """

    def __init__(self, path, seconds=1.0):
        if not math.isfinite(seconds) or seconds <= 0:
            raise ValueError("audio chunk seconds must be finite and positive")
        self.path = path
        with open(path, 'rb') as stream:
            header = stream.read(12)
            if len(header) != 12 or header[8:] != b'WAVE':
                raise ValueError("Not a WAV file")
            kind = header[:4]
            if kind not in (b'RIFF', b'RIFX', b'RF64'):
                raise ValueError("Unsupported WAV container")
            self.endian = '>' if kind == b'RIFX' else '<'
            self.kind = b'RIFX' if kind == b'RIFX' else b'RIFF'
            fmt = None
            rf64_data_size = None
            while True:
                chunk = stream.read(8)
                if len(chunk) != 8:
                    raise ValueError("WAV has no data chunk")
                name, size = struct.unpack(self.endian + '4sI', chunk)
                if name == b'ds64' and kind == b'RF64':
                    if size < 28:
                        raise ValueError("Invalid RF64 ds64 chunk")
                    ds64 = stream.read(28)
                    rf64_data_size = struct.unpack('<QQQI', ds64)[1]
                    stream.seek(size - 28 + (size & 1), 1)
                elif name == b'fmt ':
                    if size < 16 or size > 65536:
                        raise ValueError("Invalid WAV fmt chunk")
                    fmt = stream.read(size)
                    if len(fmt) != size:
                        raise ValueError("Truncated WAV format")
                    stream.seek(size & 1, 1)
                elif name == b'data':
                    if fmt is None:
                        raise ValueError("WAV data precedes format")
                    if kind == b'RF64' and size == 0xffffffff:
                        if rf64_data_size is None:
                            raise ValueError("RF64 data size missing")
                        size = rf64_data_size
                    self.offset = stream.tell()
                    self.size = min(size, os.fstat(stream.fileno()).st_size - self.offset)
                    break
                else:
                    stream.seek(size + (size & 1), 1)
        _, channels, self.rate, _, self.align, _ = struct.unpack(
            self.endian + 'HHIIHH', fmt[:16])
        if not self.rate or not self.align or not channels:
            raise ValueError("Invalid WAV rate/channel/block alignment")
        self.frames = self.size // self.align
        self.block_frames = max(1, int(self.rate * seconds))
        self.fmt_chunk = (b'fmt ' + struct.pack(self.endian + 'I', len(fmt))
                          + fmt + (b'\0' if len(fmt) & 1 else b''))

    def __iter__(self):
        with open(self.path, 'rb') as stream:
            stream.seek(self.offset)
            remaining = self.frames
            while remaining:
                count = min(remaining, self.block_frames)
                raw = stream.read(count * self.align)
                if len(raw) != count * self.align:
                    raise ValueError("Truncated WAV data")
                body = (b'WAVE' + self.fmt_chunk + b'data'
                        + struct.pack(self.endian + 'I', len(raw)) + raw
                        + (b'\0' if len(raw) & 1 else b''))
                block = BytesIO(self.kind + struct.pack(self.endian + 'I', len(body)) + body)
                _, data = wavfile.read(block)
                yield data
                remaining -= count


def audio_boundaries(path, db_threshold, chunk_seconds=1.0):
    """Sample-identical v14 thresholding using bounded blocks.

    Float64 arithmetic is deliberately retained *within a block*: float32 can
    change decisions at threshold edges. Clipped floats need a global peak
    pass before detection to preserve v14's whole-track normalization.
    """
    source = WavChunks(path, chunk_seconds)
    if not source.frames:
        return 0.0, 0.0
    floating = False
    peak = 0.0
    # Integers need no peak pass. Inspect the first block without retaining it.
    for block in source:
        floating = np.issubdtype(block.dtype, np.floating)
        if not floating:
            break
        local_peak = np.max(np.abs(block.astype(np.float64)))
        peak = np.maximum(peak, local_peak)  # Preserve global NaN behavior.
    if floating and peak == 0:
        return 0.0, 0.0
    threshold = 10.0 ** (db_threshold / 20.0)
    first = last = None
    offset = 0
    for block in source:
        values = block.astype(np.float64)
        if floating:
            if peak > 1.0:
                values /= peak
        else:
            info = np.iinfo(block.dtype)
            # Retain v14's uint8 convention for compatibility (not recentered).
            values /= max(abs(float(info.min)), abs(float(info.max)))
        np.abs(values, out=values)
        amplitude = values.max(axis=1) if values.ndim > 1 else values
        above = amplitude >= threshold
        if above.any():
            if first is None:
                first = offset + int(above.argmax())
            last = offset + len(above) - int(above[::-1].argmax())
        offset += len(block)
    if first is None:
        return 0.0, 0.0
    return first / source.rate, last / source.rate


class FrameCache:
    """Byte-bounded LRU of resized grayscale frames; never caches originals."""

    def __init__(self, folder, width, height, max_bytes):
        self.folder, self.width, self.height = folder, width, height
        self.max_bytes = max_bytes
        self.frames = OrderedDict()
        self.bytes = 0

    def get(self, name):
        import cv2
        if name in self.frames:
            self.frames.move_to_end(name)
            return self.frames[name]
        try:
            original = cv2.imread(os.path.join(self.folder, name), cv2.IMREAD_GRAYSCALE)
            if original is None:
                return None
            resized = cv2.resize(original, (self.width, self.height), interpolation=cv2.INTER_AREA)
        except (cv2.error, OSError, ValueError):
            return None
        if resized.nbytes <= self.max_bytes:
            while self.bytes + resized.nbytes > self.max_bytes:
                _, old = self.frames.popitem(last=False)
                self.bytes -= old.nbytes
            self.frames[name] = resized
            self.bytes += resized.nbytes
        return resized
