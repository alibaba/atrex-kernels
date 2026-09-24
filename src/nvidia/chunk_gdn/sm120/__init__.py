try:
    from .delta_rule import delta_rule_prefill_dsl as delta_rule_prefill_dsl_sm120
    from .fused_qk_l2norm import (
        fused_qk_l2_normalize_bf16,
        prewarm_fused_qk_l2_normalize_bf16,
    )

    _has_sm120_delta_rule_dsl = True
except ImportError:
    delta_rule_prefill_dsl_sm120 = None  # type: ignore
    fused_qk_l2_normalize_bf16 = None  # type: ignore
    prewarm_fused_qk_l2_normalize_bf16 = None  # type: ignore
    _has_sm120_delta_rule_dsl = False


def is_available() -> bool:
    """Return whether the optional SM120 CuTeDSL implementation imported."""
    return _has_sm120_delta_rule_dsl


__all__ = [
    "delta_rule_prefill_dsl_sm120",
    "fused_qk_l2_normalize_bf16",
    "is_available",
    "prewarm_fused_qk_l2_normalize_bf16",
]
