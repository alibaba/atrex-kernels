"""NVIDIA SM103 AKA M64 Chunk-GDN implementation."""

try:
    from .gdn_prefill import atrex_aka_chunk_gated_delta_rule_sm103_m64

    _has_sm103_aka_m64 = True
except ImportError:
    atrex_aka_chunk_gated_delta_rule_sm103_m64 = None  # type: ignore
    _has_sm103_aka_m64 = False


def is_available() -> bool:
    """Return whether the optional SM103 CuTeDSL implementation imported."""
    return _has_sm103_aka_m64


__all__ = [
    "atrex_aka_chunk_gated_delta_rule_sm103_m64",
    "is_available",
]
