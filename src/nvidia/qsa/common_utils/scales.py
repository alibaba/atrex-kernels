"""Scalar FP8 scale validation shared by both NVIDIA targets."""
import math
import numbers
import struct


def _scale(name, value):
    if isinstance(value, bool) or not isinstance(value, numbers.Real):
        raise TypeError(f'{name} must be a positive Python scalar')
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f'{name} must be finite and positive')
    try:
        rounded = struct.unpack('=f', struct.pack('=f', value))[0]
    except (OverflowError, struct.error):
        raise ValueError(f'{name} must be representable as a positive float32') from None
    if not math.isfinite(rounded) or rounded <= 0:
        raise ValueError(f'{name} must be representable as a positive float32')
    return float(value)
