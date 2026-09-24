"""NVIDIA SM103 AKA M64 Chunk-GDN implementation."""

try:
    from .gdn_prefill import atrex_aka_chunk_gated_delta_rule_sm103_m64
    from .fused_qk_l2norm import (
        fused_qk_l2_normalize_bf16,
        prewarm_fused_qk_l2_normalize_bf16,
    )

    _has_sm103_aka_m64 = True
except ImportError:
    atrex_aka_chunk_gated_delta_rule_sm103_m64 = None  # type: ignore
    fused_qk_l2_normalize_bf16 = None  # type: ignore
    prewarm_fused_qk_l2_normalize_bf16 = None  # type: ignore
    _has_sm103_aka_m64 = False


def is_available() -> bool:
    """Return whether the optional SM103 CuTeDSL implementation imported."""
    return _has_sm103_aka_m64


__all__ = [
    "atrex_aka_chunk_gated_delta_rule_sm103_m64",
    "fused_qk_l2_normalize_bf16",
    "is_available",
    "prewarm_fused_qk_l2_normalize_bf16",
]
