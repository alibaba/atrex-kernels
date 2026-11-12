"""Public QSA contracts with lazy, architecture-specific dispatch."""
from __future__ import annotations

from atrex.utils.device_target import detect_device_target
from functools import lru_cache
from contextlib import nullcontext

_NVIDIA_TARGETS = {}


@lru_cache(maxsize=1)
def _check_compiler():
    from importlib.metadata import version
    parts=version("nvidia-cutlass-dsl").split(".")
    if not ((4,6) <= tuple(map(int,parts[:2])) < (4,9)):
        raise NotImplementedError("qsa_compact requires CuTeDSL 4.6.x through 4.8.x")


def qsa_compact(q, k_cache, v_cache, indexer_block_ids, query_positions,
                block_table, token_to_req, use_prefill_config, output_gate,
                out=None, *, sm_scale=None, v_scale=1.0):
    """Compute gated block-sparse attention on NVIDIA SM100/SM103.

    Q: [T,Hq,256]; K/V: [P,S,Hkv,256]; compact int32 block IDs: [T,512].
    Each selected block contains four request-local tokens. Explicit positions,
    request IDs and a page table supply the causal and physical mapping.
    Gate and output are BF16. Returns out if supplied, otherwise a new tensor.

    NVIDIA supports BF16 and FP8 E4M3FN Q/K/V. For FP8, sm_scale is the merged
    QK multiplier (D**-0.5 * q_scale * k_scale); None uses D**-0.5. v_scale is
    V's dequantization multiplier, default 1. Both must be positive finite
    Python scalars representable in FP32. Quantize inputs before calling.
    BF16 requires default scale arguments.

    SM100 and SM103 select prefill, Q1 decode or Q4 MTP from shape.
    Prefill includes ragged batches. The compatibility
    argument use_prefill_config does not override that dispatch. Q1/Q4 support
    CUDA Graph after warmup with a preallocated BF16 out; scales are captured
    by value. Generic prefill capture is unsupported. Compilation is lazy.
    See src/nvidia/qsa/sm103/DESIGN.md for layouts, limits and examples.
    """
    import torch
    if not isinstance(q, torch.Tensor):
        raise ValueError("q must be a tensor")
    target_key = (q.device, detect_device_target)
    target = _NVIDIA_TARGETS.get(target_key)
    if target is None:
        target = detect_device_target(q.device)
        if target.family == "nvidia" and target.arch in ("sm100", "sm103"):
            _NVIDIA_TARGETS[target_key] = target
    if target.family != "nvidia" or target.arch not in ("sm100", "sm103"):
        raise NotImplementedError(
            f"qsa_compact requires NVIDIA SM100/SM103; got {target.family}/{target.arch}")
    if not target.runtime_version or int(target.runtime_version.split(".")[0]) < 13:
        raise NotImplementedError("qsa_compact requires CUDA 13 or later")
    _check_compiler()
    from atrex.src.nvidia.qsa.common_utils.contract import validate
    signature = validate(q, k_cache, v_cache, indexer_block_ids, query_positions,
                         block_table, token_to_req, output_gate, out)
    if q.dtype == torch.bfloat16:
        _require_default_scales(sm_scale, v_scale)
    if target.arch == "sm100":
        from atrex.src.nvidia.qsa.common_utils.mtp_domain import eligible
        guard = torch.cuda.device(q.device) if torch.cuda.current_device() != q.device.index else nullcontext()
        with guard:
            launch_options = {"_signature": signature}
            if eligible(q, k_cache, block_table):
                from atrex.src.nvidia.qsa.sm100.decode_runtime import forward
            else:
                from atrex.src.nvidia.qsa.sm100.runtime import forward
            return forward(q, k_cache, v_cache, indexer_block_ids, query_positions,
                           block_table, token_to_req, use_prefill_config, output_gate,
                           out, sm_scale=sm_scale, v_scale=v_scale, **launch_options)
    guard = torch.cuda.device(q.device) if torch.cuda.current_device() != q.device.index else nullcontext()
    with guard:
        if q.dtype == torch.float8_e4m3fn:
            from atrex.src.nvidia.qsa.sm103.fp8.runtime import forward
            return forward(q, k_cache, v_cache, indexer_block_ids, query_positions,
                           block_table, token_to_req, use_prefill_config, output_gate,
                           out, sm_scale=sm_scale, v_scale=v_scale, _signature=signature)
        from atrex.src.nvidia.qsa.sm103.runtime import forward
        return forward(q, k_cache, v_cache, indexer_block_ids, query_positions,
                       block_table, token_to_req, use_prefill_config, output_gate, out,
                       _signature=signature)


def _require_default_scales(sm_scale, v_scale):
    if sm_scale is None and type(v_scale) is float and v_scale == 1.0:
        return
    from numbers import Real
    if (sm_scale is not None or isinstance(v_scale, bool)
            or not isinstance(v_scale, Real) or v_scale != 1.0):
        raise ValueError("sm_scale/v_scale are supported only for NVIDIA FP8 Q/K/V")
