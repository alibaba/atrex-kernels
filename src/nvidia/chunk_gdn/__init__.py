"""NVIDIA Chunk-GDN implementation selection."""

from __future__ import annotations

from atrex.utils.device_target import detect_device_target


def select_chunk_gdn_implementation(device=None):
    """Return the eligible NVIDIA implementation module for ``device``."""
    target = detect_device_target(device)
    if target.family == "nvidia" and target.arch == "sm120":
        from . import sm120

        if sm120.is_available():
            return sm120
    return None


__all__ = ("select_chunk_gdn_implementation",)
