import atexit
import contextlib
import hashlib
import json
import os
import random
import re
import statistics
import subprocess
import sys

import pytest
import torch
from torch.profiler import profile as torch_profile, ProfilerActivity

import atrex

# ── constants ──────────────────────────────────────────────────────


def _sample_m_values(start, stop, count, *, required=(), seed=42):
    values = list(dict.fromkeys(required))
    pool = [m for m in range(start, stop + 1) if m not in values]
    values.extend(random.Random(seed).sample(pool, count - len(values)))
    return sorted(values)


def _sample_bucketed_m_values(
    start, stop, bucket_size, per_bucket, *, required=(), seed=42):
    values = list(dict.fromkeys(required))
    rng = random.Random(seed)
    for bucket_start in range(start, stop + 1, bucket_size):
        bucket_stop = min(bucket_start + bucket_size - 1, stop)
        pool = [
            m for m in range(bucket_start, bucket_stop + 1)
            if m not in values
        ]
        if len(pool) <= per_bucket:
            values.extend(pool)
        else:
            values.extend(rng.sample(pool, per_bucket))
    return sorted(dict.fromkeys(values))


NVFP4_VALIDATED_SHAPES = (
    ("qwen3_5_flash", 256, 8, 2048, 512),
    ("qwen3_5_flash_tp2", 256, 8, 2048, 256),
    ("qwen3_6_flash", 128, 8, 2048, 768),
    ("qwen3_6_flash_tp2", 128, 8, 2048, 384),
)

# e512_topk10 dedicated-path shape (E=512, topk=10, K=2560, N=320), ported from
# revision 90e4bf46. Deliberately NOT added to NVFP4_VALIDATED_SHAPES: that would
# sweep it through every dev M-value (unverified for this path, and costly at
# E=512). Instead it is folded into the standalone / flashinfer / cuda-graph /
# profile case lists below at a fixed branch-targeted M matrix, so the generic
# parametrized tests cover every implementation branch (across PIPELINES)
# without a dedicated per-shape test function.
E512_TOPK10_SHAPE = ("e512_topk10", 512, 10, 2560, 320)
# Branch-targeted M matrix; every value maps to a distinct implementation path
# or boundary of the e512_topk10 dispatch:
#   M=1    -> task29 fused split-K + BF16-boundary activation + task30 m1_row
#   M=2    -> smallest grouped_m16 gemm1 + task30 paired small-M finalize
#   M=16   -> compact routing + shared-SF staging (upper bound)
#   M=17   -> blocked routing + non-staged SF (lower bound)
#   M=128  -> production decode point
#   M=1024 -> fixed-order finalize (upper bound)
#   M=1025 -> dev large-M cutlass fused-finalize (lower bound)
#   M=2048 -> production prefill point
E512_TOPK10_M_VALUES = (1, 2, 16, 17, 128, 1024, 1025, 2048)
E512_TOPK10_VS_FI_M_VALUES = E512_TOPK10_M_VALUES
E512_TOPK10_GRAPH_M_VALUES = E512_TOPK10_M_VALUES
E512_TOPK10_PROFILE_M_VALUES = E512_TOPK10_M_VALUES

# Cosine-similarity gate against the FlashInfer reference.
#
# The dev shape keeps the original 0.99/0.99. The e512_topk10 shape
# (E=512, topk=10, inter=320) has a structurally lower correct-vs-correct
# floor: topk=10 sums ten independently 4-bit-quantized expert contributions
# and inter=320 gives each expert GEMM few accumulation terms, so NVFP4
# quantization noise -- not any kernel defect -- dominates the residual. The
# low-reference-norm "mixed" sentinel rows are the worst case. Measured across
# the full 8M x 3-sentinel matrix on the target GPU (diag job dv_78e9b0faad98,
# candidate revision 04e8edac): MIN global_cos=0.988869 (M=2/mixed), MIN worst per-token
# cos=0.985204 (M=2048/mixed). Each gate below sits ~0.005 under that measured
# floor: tight enough that a real defect (wrong expert/scale, a missed finalize
# -- cos << 0.98 or non-finite) still trips it, loose enough that correct-vs-
# correct quantization noise and large-M atomic-accumulation order do not.
E512_TOPK10_MIN_COS = 0.985
E512_TOPK10_MIN_ROW_COS = 0.98
DEFAULT_MIN_COS = 0.99
DEFAULT_MIN_ROW_COS = 0.99


def vs_reference_cos_thresholds(shape_name):
    """Per-shape (min_cos, min_row_cos) for the FlashInfer-reference gate."""
    if shape_name == E512_TOPK10_SHAPE[0]:
        return E512_TOPK10_MIN_COS, E512_TOPK10_MIN_ROW_COS
    return DEFAULT_MIN_COS, DEFAULT_MIN_ROW_COS

# Routing validity patterns. "mixed" puts zero-valid, one-valid, partially valid
# and fully valid tokens in the SAME batch, which is what a real decode batch
# looks like once capacity trimming emits -1 sentinels.
TOPK_SENTINEL_MODES = ("none", "partial", "all", "mixed")

M_VALUES_REQUIRED = (
    1, 2, 17, 33, 65, 127, 253,
    511, 512, 513,
    1024,
    2047, 2048, 2049,
    6000, 7000, 8000, 9000,
)
M_VALUES_ACCURACY = _sample_bucketed_m_values(
    1, 20000, 1000, 1, required=M_VALUES_REQUIRED, seed=42)
M_VALUES_STANDALONE_NVFP4 = M_VALUES_ACCURACY
M_VALUES_CUDA_GRAPH = M_VALUES_REQUIRED
M_VALUES_PROFILE = _sample_bucketed_m_values(
    6000, 9000, 1000, 1, required=M_VALUES_REQUIRED, seed=43)

PIPELINES = ["hybrid_v3", "hybrid_v5"]
INTERLEAVED_W1_PIPELINES = ("hybrid_v3", "hybrid_v5")


def _make_pipeline_shape_cases(m_values, shapes=NVFP4_VALIDATED_SHAPES):
    return [
        pytest.param(
            m,
            pipeline,
            shape_name,
            num_experts,
            topk,
            hidden_size,
            inter_size,
            id=f"M{m}-{pipeline}-{shape_name}",
        )
        for m in m_values
        for pipeline in PIPELINES
        for (
            shape_name,
            num_experts,
            topk,
            hidden_size,
            inter_size,
        ) in shapes
    ]


def _make_profile_shape_cases(m_values, shapes=NVFP4_VALIDATED_SHAPES):
    return [
        pytest.param(
            m,
            shape_name,
            num_experts,
            topk,
            hidden_size,
            inter_size,
            id=f"M{m}-{shape_name}",
        )
        for m in m_values
        for (
            shape_name,
            num_experts,
            topk,
            hidden_size,
            inter_size,
        ) in shapes
    ]


STANDALONE_NVFP4_CASES = (
    _make_pipeline_shape_cases(M_VALUES_STANDALONE_NVFP4)
    + _make_pipeline_shape_cases(E512_TOPK10_M_VALUES, [E512_TOPK10_SHAPE])
)
CUDA_GRAPH_NVFP4_CASES = (
    _make_pipeline_shape_cases(M_VALUES_CUDA_GRAPH)
    + _make_pipeline_shape_cases(
        E512_TOPK10_GRAPH_M_VALUES, [E512_TOPK10_SHAPE])
)
VS_FLASHINFER_NVFP4_CASES = (
    _make_pipeline_shape_cases(M_VALUES_ACCURACY)
    + _make_pipeline_shape_cases(
        E512_TOPK10_VS_FI_M_VALUES, [E512_TOPK10_SHAPE])
)
PROFILE_NVFP4_CASES = (
    _make_profile_shape_cases(M_VALUES_PROFILE)
    + _make_profile_shape_cases(
        E512_TOPK10_PROFILE_M_VALUES, [E512_TOPK10_SHAPE])
)

requires_sm120a = pytest.mark.skipif(
    not torch.cuda.is_available()
    or torch.cuda.get_device_capability() != (12, 0),
    reason="NVFP4 Fused MoE requires NVIDIA SM120",
)

def _flashinfer_fp4_quantize():
    try:
        from flashinfer import fp4_quantize
    except ImportError as error:
        raise RuntimeError(
            "NVFP4 fused MoE SM120 tests require flashinfer.fp4_quantize"
        ) from error
    return fp4_quantize


def _flashinfer_fused_moe():
    try:
        import flashinfer.fused_moe as fused_moe
    except ImportError as error:
        raise RuntimeError(
            "NVFP4 fused MoE SM120 tests require flashinfer.fused_moe"
        ) from error
    if not hasattr(fused_moe, "cutlass_fused_moe"):
        raise RuntimeError(
            "NVFP4 fused MoE SM120 tests require "
            "flashinfer.fused_moe.cutlass_fused_moe"
        )
    return fused_moe


# ── standalone data generation ─────────────────────────────────────


def make_standalone_nvfp4_data(M, E=256, topk=8, K=2048, N=512, device="cuda:0"):
    torch.manual_seed(42)
    a = torch.randn((M, K), device=device, dtype=torch.bfloat16) / 10

    score = torch.randn((M, E), device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(score, topk, dim=1)
    topk_weights = torch.softmax(topk_weights, dim=1).to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)

    w1_fp4 = torch.randint(0, 256, (E, 2 * N, K // 2), device=device, dtype=torch.uint8)
    w2_fp4 = torch.randint(0, 256, (E, K, N // 2), device=device, dtype=torch.uint8)

    valid_sf_bytes = [0x10, 0x20, 0x28, 0x30, 0x38, 0x3C, 0x3E]
    w1_blockscale = torch.tensor(
        valid_sf_bytes, device=device, dtype=torch.uint8
    )[torch.randint(0, len(valid_sf_bytes), (E, 2 * N, K // 16), device=device)].view(torch.float8_e4m3fn)
    w2_blockscale = torch.tensor(
        valid_sf_bytes, device=device, dtype=torch.uint8
    )[torch.randint(0, len(valid_sf_bytes), (E, K, N // 16), device=device)].view(torch.float8_e4m3fn)

    a1_gs = torch.full((E,), 1.47, device=device, dtype=torch.float32)
    a2_gs = torch.full((E,), 0.83, device=device, dtype=torch.float32)
    w1_gs = torch.rand((E,), device=device, dtype=torch.float32) * 1.5 + 0.5
    w2_gs = torch.rand((E,), device=device, dtype=torch.float32) * 1.5 + 0.5

    return dict(
        a=a, topk_weights=topk_weights, topk_ids=topk_ids,
        w1_fp4=w1_fp4, w2_fp4=w2_fp4,
        w1_blockscale=w1_blockscale, w2_blockscale=w2_blockscale,
        a1_gs=a1_gs, a2_gs=a2_gs, w1_gs=w1_gs, w2_gs=w2_gs,
    )


# ── FlashInfer-dependent data generation ──────────────────────────


def make_nvfp4_weights(num_experts, n, k, device="cuda:0"):
    fp4_quantize = _flashinfer_fp4_quantize()
    FLOAT4_E2M1_MAX = 6.0
    FLOAT8_E4M3_MAX = torch.finfo(torch.float8_e4m3fn).max

    dtype = torch.bfloat16
    w1_bf16 = torch.randn((num_experts, 2 * n, k), device=device, dtype=dtype) / 10
    w2_bf16 = torch.randn((num_experts, k, n), device=device, dtype=dtype) / 10

    w1_fp4 = torch.empty((num_experts, 2 * n, k // 2), device=device, dtype=torch.uint8)
    w2_fp4 = torch.empty((num_experts, k, n // 2), device=device, dtype=torch.uint8)

    quant_blocksize = 16
    w1_blockscale = torch.empty(
        (num_experts, 2 * n, k // quant_blocksize),
        device=device, dtype=torch.float8_e4m3fn,
    )
    w2_blockscale = torch.empty(
        (num_experts, k, n // quant_blocksize),
        device=device, dtype=torch.float8_e4m3fn,
    )

    w1_gs = torch.empty((num_experts,), device=device, dtype=torch.float32)
    w2_gs = torch.empty((num_experts,), device=device, dtype=torch.float32)

    for e in range(num_experts):
        w1_amax = torch.abs(w1_bf16[e]).max().to(torch.float32)
        w2_amax = torch.abs(w2_bf16[e]).max().to(torch.float32)
        w1_gs[e] = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w1_amax
        w2_gs[e] = FLOAT8_E4M3_MAX * FLOAT4_E2M1_MAX / w2_amax
        w1_fp4[e], w1_blockscale[e] = fp4_quantize(
            w1_bf16[e],
            w1_gs[e],
            is_sf_swizzled_layout=False,
        )
        w2_fp4[e], w2_blockscale[e] = fp4_quantize(
            w2_bf16[e],
            w2_gs[e],
            is_sf_swizzled_layout=False,
        )

    a1_gs = torch.ones((num_experts,), device=device, dtype=torch.float32)
    a2_gs = torch.ones((num_experts,), device=device, dtype=torch.float32)

    return {
        "w1_fp4": w1_fp4, "w2_fp4": w2_fp4,
        "w1_blockscale": w1_blockscale, "w2_blockscale": w2_blockscale,
        "w1_gs": w1_gs, "w2_gs": w2_gs,
        "a1_gs": a1_gs, "a2_gs": a2_gs,
    }

def make_direct_nvfp4_env(M, device="cuda:0"):
    return make_direct_nvfp4_env_shape(
        M,
        num_experts=256,
        topk=8,
        hidden_size=2048,
        inter_size=512,
        device=device,
    )


def apply_topk_sentinel_mode(topk_ids, topk_sentinel_mode):
    """Return a new [M, topk] int32 routing table for the requested mode.

    "mixed" cycles four per-token validity classes (zero / one / half / all
    slots valid) so a single batch exercises every finalize completion count at
    once. topk_weights are deliberately left untouched: the reference keeps the
    original weights and does not renormalize the surviving ones.
    """
    ids = topk_ids.clone()
    if topk_sentinel_mode == "none":
        return ids
    if topk_sentinel_mode == "partial":
        ids[:, 1::2] = -1
    elif topk_sentinel_mode == "all":
        ids.fill_(-1)
    elif topk_sentinel_mode == "mixed":
        for token in range(ids.shape[0]):
            pattern = token % 4
            if pattern == 0:
                ids[token, :] = -1
            elif pattern == 1:
                ids[token, 1:] = -1
            elif pattern == 2:
                ids[token, 1::2] = -1
    else:
        raise ValueError(f"unknown topk_sentinel_mode: {topk_sentinel_mode}")
    return ids


def valid_slot_counts(topk_ids):
    """Per-token number of non-sentinel (>= 0) top-k slots."""
    return (topk_ids >= 0).sum(dim=1)


_NVFP4_WEIGHTS_CACHE = {}
_W1_LAYOUT_CACHE = {}
# Dedicated seed, independent of the per-case torch.manual_seed(42) below, so a
# cached weight set is identical no matter which case populated it first.
_WEIGHTS_CACHE_SEED = 4242


def cached_nvfp4_weights(num_experts, inter_size, hidden_size, device="cuda:0"):
    """Deterministic NVFP4 weights, quantized once per (shape, device).

    ``make_nvfp4_weights`` loops over every expert in Python, so rebuilding it
    per parametrized case dominates the runtime of the M matrices (E=512 for
    e512_topk10, and the branch matrix multiplies the case count). The caller's
    RNG stream is saved and restored, so activation/routing draws stay exactly
    as reproducible as before caching. Callers must never mutate the returned
    tensors; the layout helpers below return copies.
    """
    key = (int(num_experts), int(inter_size), int(hidden_size), str(device))
    cached = _NVFP4_WEIGHTS_CACHE.get(key)
    if cached is not None:
        return cached
    cpu_state = torch.get_rng_state()
    cuda_states = (
        torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    )
    try:
        torch.manual_seed(_WEIGHTS_CACHE_SEED)
        weights = make_nvfp4_weights(
            num_experts, inter_size, hidden_size, device=str(device))
    finally:
        torch.set_rng_state(cpu_state)
        if cuda_states is not None:
            torch.cuda.set_rng_state_all(cuda_states)
    _NVFP4_WEIGHTS_CACHE[key] = weights
    return weights


def cached_w1_layout(weights, inter_size, kind):
    """Cached w1 gate/up relayout: "interleaved" (atrex) or "swapped" (ref).

    Both transforms are pure functions of the cached weight set, and the
    permutation they build is far more expensive than the GEMMs at E=512, so
    each layout is materialized once per process. Fresh tensors are returned by
    the underlying helpers, hence the cache is never aliased into a kernel that
    writes its inputs.
    """
    w1_fp4 = weights["w1_fp4"]
    key = (kind, tuple(w1_fp4.shape), int(inter_size), str(w1_fp4.device))
    cached = _W1_LAYOUT_CACHE.get(key)
    if cached is not None:
        return cached
    if kind == "interleaved":
        cached = interleave_gate_up(
            w1_fp4, weights["w1_blockscale"], inter_size)
    elif kind == "swapped":
        cached = swap_gate_up_halves(
            w1_fp4, weights["w1_blockscale"], inter_size)
    else:
        raise ValueError(f"unknown w1 layout kind: {kind}")
    _W1_LAYOUT_CACHE[key] = cached
    return cached


def make_direct_nvfp4_env_shape(
    M,
    num_experts,
    topk,
    hidden_size,
    inter_size,
    device="cuda:0",
    topk_sentinel_mode="none",
    weights=None,
):
    dtype = torch.bfloat16
    torch.manual_seed(42)
    if weights is None:
        weights = cached_nvfp4_weights(
            num_experts, inter_size, hidden_size, device=str(device))
    a = torch.randn((M, hidden_size), device=device, dtype=dtype) / 10
    score = torch.randn((M, num_experts), device=device, dtype=torch.float32)
    topk_weights, topk_ids = torch.topk(score, topk, dim=1)
    topk_weights = torch.softmax(topk_weights, dim=1).to(torch.float32)
    topk_ids = topk_ids.to(torch.int32)
    topk_ids = apply_topk_sentinel_mode(topk_ids, topk_sentinel_mode)

    return dict(
        weights=weights,
        a=a,
        topk_weights=topk_weights,
        topk_ids=topk_ids,
        dtype=dtype,
        device=device,
        topk_sentinel_mode=topk_sentinel_mode,
    )


def quantize_activation_nvfp4(hidden_states, global_scale):
    fp4_quantize = _flashinfer_fp4_quantize()

    return fp4_quantize(
        hidden_states,
        global_scale,
        is_sf_swizzled_layout=False,
    )


# ── profiling utilities ────────────────────────────────────────────


def flush_cache(size_mb=128, device="cuda:0", dtype=torch.int32, rounds=2):
    n = (size_mb * 1024 * 1024) // torch.tensor([], dtype=dtype).element_size()
    buf = torch.empty(n, device=device, dtype=dtype)
    for _ in range(rounds):
        buf.add_(1)
    torch.cuda.synchronize()
    return buf


def profile_cuda_kernels_ordered(fn, warmup=5, iters=20):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    with torch_profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                       acc_events=True) as prof:
        for _ in range(iters):
            torch.cuda.synchronize()
            fn()
            torch.cuda.synchronize()

    cuda_events = []
    for evt in prof.events():
        if evt.device_type == torch.autograd.DeviceType.CUDA and evt.device_time > 0:
            cuda_events.append((evt.name, evt.device_time))

    if len(cuda_events) == 0:
        return []

    kernels_per_iter = len(cuda_events) // iters
    per_iter = []
    for i in range(iters):
        start = i * kernels_per_iter
        end = start + kernels_per_iter
        per_iter.append([(name, dt) for name, dt in cuda_events[start:end]])

    return per_iter


def profile_e2e_cuda_events(fn, warmup=5, iters=20, setup_fn=None):
    for _ in range(warmup):
        if setup_fn:
            setup_fn()
        fn()
    torch.cuda.synchronize()

    times_us = []
    for _ in range(iters):
        if setup_fn:
            setup_fn()
            torch.cuda.synchronize()
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        start_evt.record()
        fn()
        end_evt.record()
        torch.cuda.synchronize()
        times_us.append(start_evt.elapsed_time(end_evt) * 1000.0)

    return times_us


def make_flashinfer_direct_nvfp4_runner(
    env,
    weights,
    hidden_states,
    output,
    *,
    swizzled_input_sf=True,
    tune_max_num_tokens=16384,
):
    fp4_quantize = _flashinfer_fp4_quantize()
    fused_moe = _flashinfer_fused_moe()

    a1_gs = weights["a1_gs"][0]
    a2_gs = weights["a2_gs"][0]
    hidden_states_fp4, input_sf = fp4_quantize(
        hidden_states,
        a1_gs,
        is_sf_swizzled_layout=swizzled_input_sf,
    )
    selected_experts = env["topk_ids"].to(torch.int)
    final_scales = env["topk_weights"]
    w1_fp4 = weights["w1_fp4"].contiguous().view(torch.long)
    w2_fp4 = weights["w2_fp4"].contiguous().view(torch.long)
    quant_scales = [
        a1_gs,
        weights["w1_blockscale"].view(torch.int32),
        torch.reciprocal(weights["a1_gs"] * weights["w1_gs"]),
        a2_gs,
        weights["w2_blockscale"].view(torch.int32),
        torch.reciprocal(weights["a2_gs"] * weights["w2_gs"]),
    ]

    def run():
        return fused_moe.cutlass_fused_moe(
            hidden_states_fp4,
            selected_experts,
            final_scales,
            w1_fp4,
            w2_fp4,
            output.dtype,
            quant_scales=quant_scales,
            input_sf=input_sf,
            output=output,
            tune_max_num_tokens=tune_max_num_tokens,
            swizzled_input_sf=swizzled_input_sf,
        )

    return run


def autotune_flashinfer_direct(run_fn):
    from flashinfer.autotuner import autotune

    for _ in range(3):
        run_fn()
    torch.cuda.synchronize()

    with torch.inference_mode(), autotune(True):
        run_fn()
    torch.cuda.synchronize()


# ── kernel classification ──────────────────────────────────────────


def _is_flush_cache_kernel(name):
    return "CUDAFunctorOnSelf_add" in name


def _is_cutlass_gemm(name):
    return name.startswith("_ZN7cutlass")


def classify_kernels_flashinfer(kernel_list):
    steps = {"routing": 0., "expand": 0., "gemm1": 0.,
             "activation": 0., "gemm2": 0., "finalize": 0.}
    cutlass_count = 0

    for name, dt in kernel_list:
        if _is_flush_cache_kernel(name):
            continue
        if _is_cutlass_gemm(name):
            cutlass_count += 1
            if cutlass_count == 1:
                steps["gemm1"] += dt
            else:
                steps["gemm2"] += dt
        elif "PrefixSum" in name or "FillFunctor" in name:
            steps["routing"] += dt
        elif "computeStrides" in name:
            steps["gemm1"] += dt
        elif "expandInputRows" in name or "cvt_fp16_to_fp4" in name:
            steps["expand"] += dt
        elif "Activation" in name or "doActivation" in name:
            steps["activation"] += dt
        elif "finalize" in name or "Memcpy" in name:
            steps["finalize"] += dt
        else:
            steps["routing"] += dt

    return steps


def classify_kernels_hybrid(kernel_list):
    steps = {"routing": 0., "expand": 0., "gemm1": 0.,
             "activation": 0., "gemm2": 0., "finalize": 0.}
    cutlass_count = 0

    for name, dt in kernel_list:
        if _is_flush_cache_kernel(name):
            continue
        if _is_cutlass_gemm(name) or "setup_cutlass_group_ptrs" in name:
            cutlass_count += 1
            if cutlass_count <= 2:
                steps["gemm1"] += dt
            else:
                steps["gemm2"] += dt
        elif "PrefixSum" in name or "FillFunctor" in name:
            steps["routing"] += dt
        elif "expandInputRows" in name:
            steps["expand"] += dt
        elif "Activation" in name or "doActivation" in name:
            steps["activation"] += dt
        elif "finalize" in name:
            steps["finalize"] += dt
        else:
            steps["routing"] += dt

    return steps


def classify_kernels_hybrid_v2(kernel_list):
    steps = {"routing": 0., "expand": 0., "gemm1": 0.,
             "activation": 0., "gemm2": 0., "finalize": 0.}

    for name, dt in kernel_list:
        if _is_flush_cache_kernel(name):
            continue
        if (_is_cutlass_gemm(name)
                or "setup_cutlass_group_ptrs" in name
                or "setup_fused_finalize_group_ptrs" in name):
            steps["gemm2"] += dt
        elif "gemm2_" in name or "task30" in name:
            steps["gemm2"] += dt
        elif ("grouped_gemm" in name
              or "compute_tile_info" in name
              or "gemm1_" in name
              or "task29" in name
              or "task13" in name):
            steps["gemm1"] += dt
        elif "doActivation" in name:
            steps["activation"] += dt
        elif "PrefixSum" in name or "FillFunctor" in name:
            steps["routing"] += dt
        elif "Memset" in name:
            steps["gemm2"] += dt
        elif "expandInputRows" in name:
            steps["expand"] += dt
        elif "finalize" in name:
            steps["finalize"] += dt
        elif "reduce_splitk" in name:
            steps["gemm1"] += dt
        else:
            steps["routing"] += dt

    return steps


# ── weight interleaving for hybrid_v3/v5 ──────────────────────────


def interleave_gate_up(w1_fp4, w1_blockscale, N, gran=8):
    E = w1_fp4.shape[0]

    def _interleave_rowmajor(t, N, gran):
        _, _, last = t.shape
        gate = t[:, :N, :].reshape(E, N // gran, gran, last)
        up   = t[:, N:, :].reshape(E, N // gran, gran, last)
        return torch.stack([up, gate], dim=2).reshape(E, 2 * N, last).contiguous()

    def _interleave_sf_swizzled(sf, N, gran):
        N_rows = 2 * N
        K_sf = sf.shape[2]
        sf_padded_K_sf = ((K_sf + 3) // 4) * 4
        sf_numKTiles = sf_padded_K_sf // 4
        mTileStride = sf_numKTiles * 512
        total_bytes = (N_rows // 128) * mTileStride

        rows_il = torch.arange(N_rows, dtype=torch.long)
        groups = rows_il // (2 * gran)
        within = rows_il % (2 * gran)
        is_gate = within >= gran
        rows_orig = torch.where(
            is_gate,
            groups * gran + (within - gran),
            N + groups * gran + within,
        )

        kvecs = torch.arange(K_sf, dtype=torch.long)
        ir_grid, kv_grid = torch.meshgrid(rows_il, kvecs, indexing='ij')
        ir_flat = ir_grid.reshape(-1)
        kv_flat = kv_grid.reshape(-1)
        orig_flat = rows_orig[ir_flat]

        def swizzled_offset(r, kv):
            tile = r // 128
            r_in_tile = r % 128
            sub_row = r_in_tile % 32
            half_tile = r_in_tile // 32
            k_sub_tile = kv // 4
            k_within = kv % 4
            return (tile * mTileStride + k_sub_tile * 512
                    + sub_row * 16 + half_tile * 4 + k_within)

        dst_offsets = swizzled_offset(ir_flat, kv_flat)
        src_offsets = swizzled_offset(orig_flat, kv_flat)

        gather_idx = torch.zeros(total_bytes, dtype=torch.long)
        gather_idx[dst_offsets] = src_offsets
        gather_idx = gather_idx.to(sf.device)

        sf_flat = sf.view(torch.uint8).reshape(E, -1)
        result = torch.gather(
            sf_flat, 1, gather_idx.unsqueeze(0).expand(E, -1))
        return result.reshape(sf.shape).view(sf.dtype)

    return (
        _interleave_rowmajor(w1_fp4, N, gran),
        _interleave_sf_swizzled(w1_blockscale, N, gran),
    )


def swap_gate_up_halves(w1_fp4, w1_blockscale, N):
    return (
        torch.cat([w1_fp4[:, N:, :], w1_fp4[:, :N, :]], dim=1).contiguous(),
        torch.cat([w1_blockscale[:, N:, :], w1_blockscale[:, :N, :]], dim=1).contiguous(),
    )


# ── helper: run atrex kernel ───────────────────────────────────────


def _launch_atrex(
    data,
    output,
    input_sf=None,
    pipeline="hybrid_v5",
    output_preinitialized=False,
):
    atrex.nvfp4_fused_moe(
        hidden_states=data["a"],
        w1_fp4=data["w1_fp4"],
        w2_fp4=data["w2_fp4"],
        w1_blockscale=data["w1_blockscale"],
        w2_blockscale=data["w2_blockscale"],
        a1_global_scale=data["a1_gs"],
        a2_global_scale=data["a2_gs"],
        w1_global_scale=data["w1_gs"],
        w2_global_scale=data["w2_gs"],
        topk_ids=data["topk_ids"],
        topk_weights=data["topk_weights"],
        output=output,
        input_sf=input_sf,
        pipeline=pipeline,
        output_preinitialized=output_preinitialized,
    )


def _run_atrex(
    data,
    M,
    device="cuda:0",
    input_sf=None,
    pipeline="hybrid_v5",
    hidden_size=None,
):
    if hidden_size is None:
        hidden_size = data["w2_fp4"].shape[1]
    output = torch.zeros((M, hidden_size), device=device, dtype=torch.bfloat16)
    _launch_atrex(
        data,
        output,
        input_sf=input_sf,
        pipeline=pipeline,
    )
    torch.cuda.synchronize()
    return output


def _run_atrex_cuda_graph(
    data,
    M,
    device="cuda:0",
    input_sf=None,
    pipeline="hybrid_v5",
    replays=8,
    hidden_size=None,
):
    if hidden_size is None:
        hidden_size = data["w2_fp4"].shape[1]
    output = torch.zeros((M, hidden_size), device=device, dtype=torch.bfloat16)

    # Warmup with the exact same shape/pipeline primes atrex static buffers and
    # custom TMA descriptors before CUDA graph capture.
    for _ in range(3):
        _launch_atrex(
            data,
            output,
            input_sf=input_sf,
            pipeline=pipeline,
        )
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _launch_atrex(
            data,
            output,
            input_sf=input_sf,
            pipeline=pipeline,
        )
    torch.cuda.synchronize()

    output.zero_()
    for _ in range(replays):
        graph.replay()
    torch.cuda.synchronize()
    return output


def _assert_repeatable_output(out1, out2, pipeline, min_cos=0.999):
    if pipeline in ("hybrid_v3", "hybrid_v5"):
        # Fused-finalize pipelines use atomic-add finalize; accumulation order can vary
        # across runs, so bitwise equality is too strict.
        cos = torch.nn.functional.cosine_similarity(
            out1.float().flatten(), out2.float().flatten(), dim=0).item()
        assert cos >= min_cos, (
            f"non-repeatable: two runs cos={cos:.6f}, expected >= {min_cos}")
        return cos

    assert torch.equal(out1, out2), "non-deterministic: two runs differ"
    return None


# ── output-buffer contracts ────────────────────────────────────────
#
# The public API has two output contracts, and both have to hold on a buffer the
# caller did NOT pre-zero:
#   output_preinitialized=False -> atrex owns every element of ``output``. A
#       token whose top-k slots are all sentinel (-1) contributes no expert row,
#       so no GEMM2 CTA ever finalizes it and routing's fused clear is the only
#       thing that can make its row zero.
#   output_preinitialized=True  -> ``output`` already holds another producer's
#       result (shared experts) and is accumulated into, so an empty row has to
#       keep its initial value bit for bit.
# Building ``output`` with torch.zeros hides a missing clear, which is why every
# accuracy case below runs the same reference comparison over all three fills.

OUTPUT_CONTRACTS = ("dirty_nonzero", "dirty_nan", "preinitialized")


def make_output_contract_buffer(contract, M, hidden_size, device):
    """Return ``(output, initial)`` for one output-buffer contract.

    ``initial`` is None unless the contract is preinitialized, in which case it
    holds the exact finite value every untouched element must keep. Dirty fills
    use magnitudes far outside the reference range plus non-finite entries, so a
    leaked element cannot be mistaken for a plausible result.
    """
    shape = (M, hidden_size)
    if contract == "preinitialized":
        # Multiples of 1/16 below 2 are exact in BF16, so "kept its initial
        # value" is asserted bit for bit rather than within a tolerance.
        initial = torch.full(
            shape, 0.5, device=device, dtype=torch.bfloat16)
        rows = (torch.arange(M, device=device) % 4).to(torch.bfloat16) * 0.25
        cols = (torch.arange(hidden_size, device=device) % 8).to(
            torch.bfloat16) * 0.0625
        initial += rows.unsqueeze(1)
        initial += cols.unsqueeze(0)
        return initial.clone(), initial
    if contract == "dirty_nan":
        dirty = torch.full(
            shape, float("nan"), device=device, dtype=torch.bfloat16)
        dirty[::2, ::3] = float("inf")
        return dirty, None
    if contract == "dirty_nonzero":
        dirty = torch.full(
            shape, -1.0e4, device=device, dtype=torch.bfloat16)
        dirty[1::2, :] = 1.0e4
        return dirty, None
    raise ValueError(f"unknown output contract: {contract}")


def assert_output_contract(
    nv, expected, *, empty_rows, initial, tag, min_cos=0.99, min_row_cos=0.99):
    """Compare one atrex output against its reference under a given contract.

    ``expected`` is the full [M, hidden] float reference (the routed FlashInfer
    result, plus ``initial`` for the preinitialized contract). ``empty_rows``
    marks tokens with no valid top-k slot: they receive no contribution at all,
    so they are compared exactly -- against zeros, or against the untouched
    initial value -- and never by cosine, which is undefined for a zero vector.
    Contributed rows are compared per token as well as over the whole tensor, so
    a handful of wrong rows cannot hide inside a global similarity.
    """
    nv_f32 = nv.float()
    assert torch.isfinite(nv_f32).all(), (
        f"[{tag}] output is not finite: "
        f"nan={int(torch.isnan(nv_f32).sum())}, "
        f"inf={int(torch.isinf(nv_f32).sum())}")

    if bool(empty_rows.any()):
        nv_empty = nv[empty_rows]
        want = (torch.zeros_like(nv_empty) if initial is None
                else initial[empty_rows])
        bad = (nv_empty != want).any(dim=1).nonzero().flatten().tolist()
        assert not bad, (
            f"[{tag}] tokens with no valid slot must keep "
            f"{'their exact initial value' if initial is not None else 'exact zeros'}"
            f"; {len(bad)} violating token(s), first: {bad[:8]}; "
            f"got={nv_empty[bad[0]][:4].tolist()}, "
            f"want={want[bad[0]][:4].tolist()}")

    contributed = ~empty_rows
    if not bool(contributed.any()):
        return None, None

    nv_c = nv_f32[contributed]
    exp_c = expected[contributed]
    cos = torch.nn.functional.cosine_similarity(
        nv_c.flatten(), exp_c.flatten(), dim=0).item()

    worst_row_cos = None
    row_norms = exp_c.norm(dim=1)
    usable = row_norms > 0
    if bool(usable.any()):
        row_cos = torch.nn.functional.cosine_similarity(
            nv_c[usable], exp_c[usable], dim=1)
        worst_row_cos = row_cos.min().item()
        worst_index = int(row_cos.argmin().item())
        assert worst_row_cos >= min_row_cos, (
            f"[{tag}] worst per-token cos={worst_row_cos:.6f} < {min_row_cos} "
            f"at contributed row {worst_index} "
            f"(reference norm={row_norms[usable][worst_index]:.6f})")

    assert cos >= min_cos, (
        f"[{tag}] cos_sim vs reference = {cos:.6f}, expected >= {min_cos}")
    return cos, worst_row_cos


def flashinfer_reference_output(
    env, weights, hidden_size, inter_size, pipeline, device):
    """FlashInfer cutlass_fused_moe reference for ``env``'s current routing.

    atrex consumes gate/up-interleaved w1 for the hybrid pipelines, while the
    reference consumes the plain layout with its two halves swapped, which is
    the same permutation expressed from the other side.
    """
    M = env["a"].shape[0]
    fi_output = torch.zeros(
        (M, hidden_size), device=device, dtype=torch.bfloat16)
    fi_weights = weights
    if pipeline in INTERLEAVED_W1_PIPELINES:
        fi_weights = dict(weights)
        fi_weights["w1_fp4"], fi_weights["w1_blockscale"] = cached_w1_layout(
            weights, inter_size, "swapped")
    fi_run = make_flashinfer_direct_nvfp4_runner(
        env,
        fi_weights,
        env["a"],
        fi_output,
        swizzled_input_sf=False,
        tune_max_num_tokens=16384,
    )
    fi_run()
    torch.cuda.synchronize()
    return fi_output


# ── test: standalone NVFP4 input ──────────────────────────────────


@requires_sm120a
@pytest.mark.parametrize(
    "M,pipeline,shape_name,num_experts,topk,hidden_size,inter_size",
    STANDALONE_NVFP4_CASES,
)
def test_atrex_fusedmoe_nvfp4(
    M, pipeline, shape_name, num_experts, topk, hidden_size, inter_size):
    device = "cuda:0"
    data = make_standalone_nvfp4_data(
        M, E=num_experts, topk=topk, K=hidden_size, N=inter_size,
        device=device)

    if pipeline in INTERLEAVED_W1_PIPELINES:
        data["w1_fp4"], data["w1_blockscale"] = interleave_gate_up(
            data["w1_fp4"], data["w1_blockscale"], inter_size)

    a_fp4, a_sf = quantize_activation_nvfp4(data["a"], data["a1_gs"][0])
    data_nvfp4 = dict(data)
    data_nvfp4["a"] = a_fp4

    out1 = _run_atrex(
        data_nvfp4, M, device, input_sf=a_sf, pipeline=pipeline,
        hidden_size=hidden_size)
    out2 = _run_atrex(
        data_nvfp4, M, device, input_sf=a_sf, pipeline=pipeline,
        hidden_size=hidden_size)

    assert not torch.isnan(out1).any(), "output contains NaN"
    assert not torch.isinf(out1).any(), "output contains Inf"
    assert out1.abs().sum() > 0, "output is all zeros"
    cos = _assert_repeatable_output(out1, out2, pipeline)

    cos_msg = f", repeat_cos={cos:.6f}" if cos is not None else ""
    print(f"\n[M={M}, {pipeline}, {shape_name}] NVFP4 input OK: "
          f"mean={out1.float().mean():.6f}, std={out1.float().std():.6f}, "
          f"sum={out1.float().abs().sum():.1f}{cos_msg}")


# ── test: CUDA graph capture/replay ───────────────────────────────


@requires_sm120a
@pytest.mark.parametrize(
    "M,pipeline,shape_name,num_experts,topk,hidden_size,inter_size",
    CUDA_GRAPH_NVFP4_CASES,
)
def test_atrex_fusedmoe_nvfp4_cuda_graph(
    M, pipeline, shape_name, num_experts, topk, hidden_size, inter_size):
    device = "cuda:0"
    data = make_standalone_nvfp4_data(
        M, E=num_experts, topk=topk, K=hidden_size, N=inter_size,
        device=device)

    if pipeline in INTERLEAVED_W1_PIPELINES:
        data["w1_fp4"], data["w1_blockscale"] = interleave_gate_up(
            data["w1_fp4"], data["w1_blockscale"], inter_size)

    a_fp4, a_sf = quantize_activation_nvfp4(data["a"], data["a1_gs"][0])
    data_nvfp4 = dict(data)
    data_nvfp4["a"] = a_fp4

    eager_out = _run_atrex(
        data_nvfp4, M, device, input_sf=a_sf, pipeline=pipeline,
        hidden_size=hidden_size)
    graph_out = _run_atrex_cuda_graph(
        data_nvfp4, M, device, input_sf=a_sf, pipeline=pipeline,
        hidden_size=hidden_size)

    assert not torch.isnan(graph_out).any(), "CUDA graph output contains NaN"
    assert not torch.isinf(graph_out).any(), "CUDA graph output contains Inf"
    assert graph_out.abs().sum() > 0, "CUDA graph output is all zeros"
    cos = _assert_repeatable_output(eager_out, graph_out, pipeline)

    cos_msg = f", graph_cos={cos:.6f}" if cos is not None else ""
    print(f"\n[M={M}, {pipeline}, {shape_name}] NVFP4 CUDA graph OK: "
          f"sum={graph_out.float().abs().sum():.1f}{cos_msg}")


# ── test: correctness vs FlashInfer (NVFP4 input) ─────────────────


@requires_sm120a
@pytest.mark.parametrize(
    "M,pipeline,shape_name,num_experts,topk,hidden_size,inter_size",
    VS_FLASHINFER_NVFP4_CASES,
)
@pytest.mark.parametrize("topk_sentinel_mode", TOPK_SENTINEL_MODES)
def test_vs_flashinfer_nvfp4(
    M, pipeline, shape_name, num_experts, topk, hidden_size, inter_size,
    topk_sentinel_mode):
    device = torch.device("cuda:0")
    env = make_direct_nvfp4_env_shape(
        M, num_experts, topk, hidden_size, inter_size, device=device,
        topk_sentinel_mode=topk_sentinel_mode)
    weights, a = env["weights"], env["a"]

    fi_output = flashinfer_reference_output(
        env, weights, hidden_size, inter_size, pipeline, device)
    fi_f32 = fi_output.float()
    base_tag = f"{pipeline}, {shape_name}, M={M}, sentinel={topk_sentinel_mode}"
    assert torch.isfinite(fi_f32).all(), (
        f"[{base_tag}] FlashInfer reference is not finite")

    empty_rows = valid_slot_counts(env["topk_ids"]) == 0
    if bool(empty_rows.any()):
        # Tokens with no valid slot contribute nothing, so the reference row
        # has to be zero before "atrex output is zero" means anything. The
        # tolerance is the one this comparison always used: allclose against
        # zeros, i.e. 1e-8 absolute.
        assert torch.allclose(
            fi_f32[empty_rows], torch.zeros_like(fi_f32[empty_rows]),
            atol=1e-8, rtol=0), (
            f"[{base_tag}] FlashInfer reference is not zero for the "
            f"{int(empty_rows.sum())} token(s) with no valid slot")

    if pipeline in INTERLEAVED_W1_PIPELINES:
        w1_fp4, w1_bs = cached_w1_layout(weights, inter_size, "interleaved")
    else:
        w1_fp4, w1_bs = weights["w1_fp4"], weights["w1_blockscale"]
    a_fp4, a_sf = quantize_activation_nvfp4(a, weights["a1_gs"][0])

    # Same weights, same routing, three output-buffer contracts: a dirty
    # non-finite fill and a dirty nonzero fill must both be fully overwritten
    # (output_preinitialized=False), and a finite preinitialized fill must be
    # accumulated into, with empty rows preserved exactly.
    summary = []
    for contract in OUTPUT_CONTRACTS:
        output, initial = make_output_contract_buffer(
            contract, M, hidden_size, device)
        atrex.nvfp4_fused_moe(
            hidden_states=a_fp4,
            w1_fp4=w1_fp4,
            w2_fp4=weights["w2_fp4"],
            w1_blockscale=w1_bs,
            w2_blockscale=weights["w2_blockscale"],
            a1_global_scale=weights["a1_gs"],
            a2_global_scale=weights["a2_gs"],
            w1_global_scale=weights["w1_gs"],
            w2_global_scale=weights["w2_gs"],
            topk_ids=env["topk_ids"],
            topk_weights=env["topk_weights"],
            output=output,
            input_sf=a_sf,
            pipeline=pipeline,
            output_preinitialized=(contract == "preinitialized"),
        )
        torch.cuda.synchronize()

        expected = fi_f32 if initial is None else fi_f32 + initial.float()
        min_cos, min_row_cos = vs_reference_cos_thresholds(shape_name)
        cos, row_cos = assert_output_contract(
            output,
            expected,
            empty_rows=empty_rows,
            initial=initial,
            tag=f"{base_tag}, {contract}",
            min_cos=min_cos,
            min_row_cos=min_row_cos,
        )
        summary.append(
            f"{contract}: cos={cos:.6f}, worst_token_cos={row_cos:.6f}"
            if row_cos is not None
            else (f"{contract}: cos={cos:.6f}, no usable per-token reference"
                  if cos is not None
                  else f"{contract}: all rows empty, exact match"))

    print(f"\n[{base_tag}] output contracts OK | " + " | ".join(summary))


# ── test: performance profiling (NVFP4 input) ─────────────────────


@requires_sm120a
@pytest.mark.parametrize(
    "M,shape_name,num_experts,topk,hidden_size,inter_size",
    PROFILE_NVFP4_CASES,
)
def test_profile_nvfp4(
    M, shape_name, num_experts, topk, hidden_size, inter_size):
    device = torch.device("cuda:0")
    env = make_direct_nvfp4_env_shape(
        M, num_experts, topk, hidden_size, inter_size, device=device)
    weights, a = env["weights"], env["a"]
    dtype = env["dtype"]

    warmup, iters = 10, 20
    flush_fn = lambda: flush_cache(128, device=str(device))

    a_fp4, a_sf = quantize_activation_nvfp4(a, weights["a1_gs"][0])

    # ── FlashInfer direct NVFP4 profiling ──
    # Keep this call path aligned with proj003 task22:
    # flashinfer.fp4_quantize(...) + fused_moe.cutlass_fused_moe(...).

    fi_output = torch.zeros((M, hidden_size), device=device, dtype=dtype)
    fi_nvfp4_fn_pure = make_flashinfer_direct_nvfp4_runner(
        env,
        weights,
        a,
        fi_output,
        swizzled_input_sf=True,
        tune_max_num_tokens=16384,
    )

    print(f"\n=== Profiling NVFP4 input: shape={shape_name}, M={M}, "
          f"E={num_experts}, topk={topk}, K={hidden_size}, N={inter_size}, "
          f"warmup={warmup}, iters={iters} ===")

    print("\nProfiling FlashInfer NVFP4 (direct)...")
    autotune_flashinfer_direct(fi_nvfp4_fn_pure)

    def fi_nvfp4_fn():
        flush_cache(128, device=str(device))
        return fi_nvfp4_fn_pure()

    fi_per_iter = profile_cuda_kernels_ordered(
        fi_nvfp4_fn, warmup=warmup, iters=iters)

    if fi_per_iter:
        print(f"FlashInfer NVFP4 kernel trace (iter 0, {len(fi_per_iter[0])} kernels):")
        for i, (name, dt) in enumerate(fi_per_iter[0]):
            print(f"  [{i:2d}] {dt:8.1f} us  {name[:90]}")

    fi_steps_all = [classify_kernels_flashinfer(kl) for kl in fi_per_iter]
    fi_steps = {k: sum(s[k] for s in fi_steps_all) / len(fi_steps_all)
                for k in fi_steps_all[0]} if fi_steps_all else {}

    fi_e2e_times = profile_e2e_cuda_events(
        fi_nvfp4_fn_pure, warmup=warmup, iters=iters, setup_fn=flush_fn)
    fi_e2e_avg = sum(fi_e2e_times) / len(fi_e2e_times)
    fi_e2e_min = min(fi_e2e_times)

    w1_fp4_interleaved, w1_bs_interleaved = cached_w1_layout(
        weights, inter_size, "interleaved")

    # -- Hybrid v3 NVFP4 profiling --

    v3_output = torch.zeros((M, hidden_size), device=device, dtype=dtype)

    def v3_nvfp4_fn():
        flush_cache(128, device=str(device))
        v3_output.zero_()
        atrex.nvfp4_fused_moe(
            hidden_states=a_fp4,
            w1_fp4=w1_fp4_interleaved, w2_fp4=weights["w2_fp4"],
            w1_blockscale=w1_bs_interleaved,
            w2_blockscale=weights["w2_blockscale"],
            a1_global_scale=weights["a1_gs"],
            a2_global_scale=weights["a2_gs"],
            w1_global_scale=weights["w1_gs"],
            w2_global_scale=weights["w2_gs"],
            topk_ids=env["topk_ids"],
            topk_weights=env["topk_weights"],
            output=v3_output,
            input_sf=a_sf,
            pipeline="hybrid_v3",
        )

    def v3_nvfp4_fn_pure():
        v3_output.zero_()
        atrex.nvfp4_fused_moe(
            hidden_states=a_fp4,
            w1_fp4=w1_fp4_interleaved, w2_fp4=weights["w2_fp4"],
            w1_blockscale=w1_bs_interleaved,
            w2_blockscale=weights["w2_blockscale"],
            a1_global_scale=weights["a1_gs"],
            a2_global_scale=weights["a2_gs"],
            w1_global_scale=weights["w1_gs"],
            w2_global_scale=weights["w2_gs"],
            topk_ids=env["topk_ids"],
            topk_weights=env["topk_weights"],
            output=v3_output,
            input_sf=a_sf,
            pipeline="hybrid_v3",
        )

    print("\nProfiling Hybrid v3 NVFP4...")
    v3_per_iter = profile_cuda_kernels_ordered(v3_nvfp4_fn, warmup=warmup, iters=iters)

    if v3_per_iter:
        print(f"Hybrid v3 NVFP4 kernel trace (iter 0, {len(v3_per_iter[0])} kernels):")
        for i, (name, dt) in enumerate(v3_per_iter[0]):
            print(f"  [{i:2d}] {dt:8.1f} us  {name[:90]}")

    v3_steps_all = [classify_kernels_hybrid_v2(kl) for kl in v3_per_iter]
    v3_steps = {k: sum(s[k] for s in v3_steps_all) / len(v3_steps_all)
                for k in v3_steps_all[0]} if v3_steps_all else {}

    v3_e2e_times = profile_e2e_cuda_events(
        v3_nvfp4_fn_pure, warmup=warmup, iters=iters, setup_fn=flush_fn)
    v3_e2e_avg = sum(v3_e2e_times) / len(v3_e2e_times)
    v3_e2e_min = min(v3_e2e_times)

    # -- Hybrid v5 NVFP4 profiling --

    v5_output = torch.zeros((M, hidden_size), device=device, dtype=dtype)

    def v5_nvfp4_fn():
        flush_cache(128, device=str(device))
        v5_output.zero_()
        atrex.nvfp4_fused_moe(
            hidden_states=a_fp4,
            w1_fp4=w1_fp4_interleaved, w2_fp4=weights["w2_fp4"],
            w1_blockscale=w1_bs_interleaved,
            w2_blockscale=weights["w2_blockscale"],
            a1_global_scale=weights["a1_gs"],
            a2_global_scale=weights["a2_gs"],
            w1_global_scale=weights["w1_gs"],
            w2_global_scale=weights["w2_gs"],
            topk_ids=env["topk_ids"],
            topk_weights=env["topk_weights"],
            output=v5_output,
            input_sf=a_sf,
            pipeline="hybrid_v5",
        )

    def v5_nvfp4_fn_pure():
        v5_output.zero_()
        atrex.nvfp4_fused_moe(
            hidden_states=a_fp4,
            w1_fp4=w1_fp4_interleaved, w2_fp4=weights["w2_fp4"],
            w1_blockscale=w1_bs_interleaved,
            w2_blockscale=weights["w2_blockscale"],
            a1_global_scale=weights["a1_gs"],
            a2_global_scale=weights["a2_gs"],
            w1_global_scale=weights["w1_gs"],
            w2_global_scale=weights["w2_gs"],
            topk_ids=env["topk_ids"],
            topk_weights=env["topk_weights"],
            output=v5_output,
            input_sf=a_sf,
            pipeline="hybrid_v5",
        )

    print("\nProfiling Hybrid v5 NVFP4...")
    v5_per_iter = profile_cuda_kernels_ordered(v5_nvfp4_fn, warmup=warmup, iters=iters)

    if v5_per_iter:
        print(f"Hybrid v5 NVFP4 kernel trace (iter 0, {len(v5_per_iter[0])} kernels):")
        for i, (name, dt) in enumerate(v5_per_iter[0]):
            print(f"  [{i:2d}] {dt:8.1f} us  {name[:90]}")

    v5_steps_all = [classify_kernels_hybrid_v2(kl) for kl in v5_per_iter]
    v5_steps = {k: sum(s[k] for s in v5_steps_all) / len(v5_steps_all)
                for k in v5_steps_all[0]} if v5_steps_all else {}

    v5_e2e_times = profile_e2e_cuda_events(
        v5_nvfp4_fn_pure, warmup=warmup, iters=iters, setup_fn=flush_fn)
    v5_e2e_avg = sum(v5_e2e_times) / len(v5_e2e_times)
    v5_e2e_min = min(v5_e2e_times)

    # ── Comparison table ──

    fi_total = sum(fi_steps.values()) if fi_steps else 0
    v3_total = sum(v3_steps.values()) if v3_steps else 0
    v5_total = sum(v5_steps.values()) if v5_steps else 0

    print(f"\n{'=' * 67}")
    print(f"  NVFP4 Input Comparison (flush_cache 128MB, iters={iters})")
    print(f"  shape={shape_name}, M={M}, E={num_experts}, topk={topk}, "
          f"K={hidden_size}, N={inter_size}")
    print(f"{'=' * 67}")

    step_names = ["routing", "expand", "gemm1", "activation", "gemm2", "finalize"]
    print(f"{'Step':>14s} | {'FI NVFP4':>12s} | {'Hy_v3 NVFP4':>12s} | {'Hy_v5 NVFP4':>12s}")
    print(f"{'-'*14}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")
    for step in step_names:
        fi_v = fi_steps.get(step, 0)
        v3_v = v3_steps.get(step, 0)
        v5_v = v5_steps.get(step, 0)
        print(f"{step:>14s} | {fi_v:10.1f}us | {v3_v:10.1f}us | {v5_v:10.1f}us")

    print(f"{'-'*14}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")
    print(f"{'kernel sum':>14s} | {fi_total:10.1f}us | {v3_total:10.1f}us | {v5_total:10.1f}us")
    print(f"{'e2e avg':>14s} | {fi_e2e_avg:10.1f}us | {v3_e2e_avg:10.1f}us | {v5_e2e_avg:10.1f}us")
    print(f"{'e2e min':>14s} | {fi_e2e_min:10.1f}us | {v3_e2e_min:10.1f}us | {v5_e2e_min:10.1f}us")
    if all(x > 0 for x in (fi_total, v3_total, v5_total)):
        print(f"{'-'*14}-+-{'-'*12}-+-{'-'*12}-+-{'-'*12}")
        print(f"{'speedup(kern)':>14s} | {'1.00x':>12s} | {fi_total/v3_total:10.2f}x | {fi_total/v5_total:10.2f}x")
        print(f"{'speedup(e2e)':>14s} | {'1.00x':>12s} | {fi_e2e_avg/v3_e2e_avg:10.2f}x | {fi_e2e_avg/v5_e2e_avg:10.2f}x")
        print(f"{'v5 vs v3':>14s} | {'---':>12s} | {'1.00x':>12s} | {v3_e2e_avg/v5_e2e_avg:10.2f}x")

    # Everything above is a report: it compares two different implementations
    # on one machine, so it cannot show whether this build regressed. The
    # migrated e512_topk10 path additionally has to defend its latency against
    # the revision it was ported from, which is what the ABBA gate below does
    # (two installed wheels, alternating blocks, hard +5% per case).
    if shape_name == E512_TOPK10_SHAPE[0]:
        gate_e512_topk10_abba(
            M, num_experts, topk, hidden_size, inter_size, list(PIPELINES))


# ── e512_topk10 (E=512, topk=10, hidden=2560, inter=320) ─────────
#
# Additive coverage for the e512_topk10 dedicated dispatch path ported from
# revision 90e4bf46. Skill §5: exactly one test module per operator target, so
# these live alongside the dev topk=8 tests rather than in a new file. Nothing
# here runs at collection time (JIT compiles on first call inside a test body).
#
# Shape-level correctness/accuracy AND the fi-vs-v3-vs-v5 perf comparison for
# e512_topk10 now ride the GENERIC parametrized tests: E512_TOPK10_SHAPE is
# folded into STANDALONE_NVFP4_CASES, VS_FLASHINFER_NVFP4_CASES and
# PROFILE_NVFP4_CASES (top of file) at curated M values, so
# test_atrex_fusedmoe_nvfp4 / test_vs_flashinfer_nvfp4 / test_profile_nvfp4
# cover it (across PIPELINES; hybrid_v3 == hybrid_v5 for this shape). This
# section keeps only the e512_topk10-SPECIFIC guards that no generic test can
# express (dedicated dispatch routing, topk=8 non-leak, detector, prewarm,
# kernel-name prefix) plus their helpers; the shape/M constants live near the
# case lists above.

# Discriminator for the e512_topk10 dedicated kernels. dev topk=8 uses
# atrex_task29/30_..._qwen3 (no "e512t10" token), so these two markers
# are unique to the ported path.
_E512_TOPK10_NEW_PATH_MARKERS = ("atrex_e512t10_", "e512t10")
_ATREX_MANGLED_RE = re.compile(r"^_ZN?\d*atrex_")


def _has_atrex_prefix(name):
    """Form-independent atrex_ prefix check for a profiler kernel name.

    torch profiler presents CUDA kernels either demangled (sometimes with a
    leading return type, e.g. "void atrex_e512t10_...") or mangled
    ("_ZN..atrex_e512t10_.."). Both encode the atrex_ symbol at the start of the
    operator's own namespace/function; a genuinely unprefixed kernel matches
    neither, which is the defect this skill §4 gate guards against.
    """
    stripped = name
    for ret in ("void ", "__global__ "):
        if stripped.startswith(ret):
            stripped = stripped[len(ret):]
    return stripped.startswith("atrex_") or bool(_ATREX_MANGLED_RE.match(name))


def _make_e512_topk10_data(M, device="cuda:0"):
    """Standalone (randint-weight) e512_topk10 inputs, flashinfer used only as
    the activation quantizer (topk-agnostic), never as a MoE reference."""
    _shape_name, E, topk, hidden_size, inter_size = E512_TOPK10_SHAPE
    data = make_standalone_nvfp4_data(
        M, E=E, topk=topk, K=hidden_size, N=inter_size, device=device)
    data["w1_fp4"], data["w1_blockscale"] = interleave_gate_up(
        data["w1_fp4"], data["w1_blockscale"], inter_size)
    a_fp4, a_sf = quantize_activation_nvfp4(data["a"], data["a1_gs"][0])
    data_nvfp4 = dict(data)
    data_nvfp4["a"] = a_fp4
    data_nvfp4["a_sf"] = a_sf
    data_nvfp4["hidden_size"] = hidden_size
    return data_nvfp4


def _profile_kernel_names_with_retry(fn, retries=3, warmup=3, iters=5):
    """Profile ``fn`` and return flattened CUDA kernel names, retrying cold CUPTI.

    The first torch.profiler CUDA (CUPTI) activation in a process can return an
    empty event set while CUPTI initializes. Retry (bounded) until events are
    observed so dispatch/prefix assertions judge real kernel launches, not a cold
    profiler; otherwise a test that happens to run first in a process (e.g. a
    targeted ``pytest -k`` selection) can see an empty list and fail with a
    misleading "no CUDA kernels" message. Later profiling calls are unaffected.
    """
    per_iter = []
    for _attempt in range(retries):
        per_iter = profile_cuda_kernels_ordered(fn, warmup=warmup, iters=iters)
        if per_iter:
            break
    return [name for kl in per_iter for name, _dt in kl]


def _profile_e512_topk10_kernel_names(M, pipeline="hybrid_v5"):
    data_nvfp4 = _make_e512_topk10_data(M, device="cuda:0")
    output = torch.zeros(
        (M, data_nvfp4["hidden_size"]), device="cuda:0",
        dtype=torch.bfloat16)

    def fn():
        _launch_atrex(
            data_nvfp4, output, input_sf=data_nvfp4["a_sf"],
            pipeline=pipeline)

    return _profile_kernel_names_with_retry(fn)


@requires_sm120a
def test_e512_topk10_dispatch_routes_to_dedicated_path():
    names = _profile_e512_topk10_kernel_names(128)
    new_path = sorted({
        n for n in names
        if any(m in n for m in _E512_TOPK10_NEW_PATH_MARKERS)
    })
    assert new_path, (
        "e512_topk10 (M=128) did not route to the dedicated atrex_e512t10_ path; "
        f"observed kernels: {sorted(set(names))}")
    assert any("task29" in n for n in new_path), (
        f"missing e512_topk10 task29 gemm1: {new_path}")
    assert any("task30" in n for n in new_path), (
        f"missing e512_topk10 task30 gemm2: {new_path}")
    print(f"\ne512_topk10 dedicated kernels observed: {new_path}")


@requires_sm120a
def test_topk8_shape_does_not_enter_e512_topk10_path():
    device = "cuda:0"
    M, E, topk, hidden_size, inter_size = 128, 256, 8, 2048, 512
    data = make_standalone_nvfp4_data(
        M, E=E, topk=topk, K=hidden_size, N=inter_size, device=device)
    data["w1_fp4"], data["w1_blockscale"] = interleave_gate_up(
        data["w1_fp4"], data["w1_blockscale"], inter_size)
    a_fp4, a_sf = quantize_activation_nvfp4(data["a"], data["a1_gs"][0])
    data["a"] = a_fp4
    output = torch.zeros(
        (M, hidden_size), device=device, dtype=torch.bfloat16)

    def fn():
        _launch_atrex(data, output, input_sf=a_sf, pipeline="hybrid_v5")

    # Reuse the bounded CUPTI cold-start retry so a solo run of this test
    # (pytest -k test_topk8_...) cannot fail with a misleading "no CUDA kernels"
    # message when the first profiler activation returns an empty event set.
    names = _profile_kernel_names_with_retry(fn)
    assert names, "topk=8 dispatch observed no CUDA kernels"
    leaked = sorted({
        n for n in names
        if any(m in n for m in _E512_TOPK10_NEW_PATH_MARKERS)
    })
    assert not leaked, (
        f"topk=8 shape wrongly entered the e512_topk10 path: {leaked}")
    print(f"\ntopk=8 (M=128) stayed on the dev path; "
          f"{len(set(names))} distinct kernels, no atrex_e512t10_/e512t10")


def test_e512_topk10_shape_detector_eligibility():
    # Pure detector test: no compile, no allocation, no device switch (skill §3).
    from atrex.api.nvfp4_fused_moe_sm120 import (
        _E512_TOPK10_TASK29_TASK30_MAX_M,
        _hybrid_v5_uses_compact_small_m_sf,
        _sm120_has_validated_shape,
        _sm120_is_e512_topk10_shape,
    )
    assert _sm120_is_e512_topk10_shape(512, 10, 2560, 320)
    for (E, topk, hidden, inter) in (
        (512, 10, 2560, 321), (512, 10, 2560, 256), (512, 8, 2560, 320),
        (256, 10, 2560, 320), (512, 10, 2048, 320), (128, 8, 2048, 768),
    ):
        assert not _sm120_is_e512_topk10_shape(E, topk, hidden, inter), (
            E, topk, hidden, inter)
    # validated-shape gate is additive: e512_topk10 accepted, dev shapes kept.
    assert _sm120_has_validated_shape(128, 512, 10, 2560, 320)
    assert _sm120_has_validated_shape(128, 256, 8, 2048, 512)
    assert not _sm120_has_validated_shape(128, 512, 10, 2560, 321)
    # compact shared-SF staging only for very small M on the e512_topk10 shape.
    assert _hybrid_v5_uses_compact_small_m_sf(16, 512, 10, 2560, 320)
    assert not _hybrid_v5_uses_compact_small_m_sf(17, 512, 10, 2560, 320)
    assert not _hybrid_v5_uses_compact_small_m_sf(16, 256, 8, 2048, 512)
    assert _E512_TOPK10_TASK29_TASK30_MAX_M == 1024


@requires_sm120a
def test_e512_topk10_prewarm_builds_dedicated_bindings():
    # First call JIT-compiles the extension; import alone must not compile.
    from atrex.api import nvfp4_fused_moe_sm120 as api
    mod = api.nvfp4_fused_moe_sm120_build()
    required = (
        "e512_topk10_task29_forward_fused",
        "e512_topk10_task29_forward_grouped_m16_fused_act",
        "e512_topk10_do_activation",
        "e512_topk10_task30_forward_fixed",
        "get_workspace_size_e512_topk10_task29_fused",
        "get_workspace_size_e512_topk10_task29_grouped_m16_fused_act",
        "get_workspace_size_e512_topk10_task30",
    )
    missing = [name for name in required if not hasattr(mod, name)]
    assert not missing, f"e512_topk10 bindings missing after build: {missing}"
    ws = int(mod.get_workspace_size_e512_topk10_task30(
        128, 512, 10, 2560, 320))
    assert ws >= 0
    print(f"\ne512_topk10 prewarm OK; task30 ws bytes for M=128: {ws}")


# ── profiler full-name gate ────────────────────────────────────────
#
# Every CUDA event observed inside a public API call is assigned to exactly one
# owner, and the owner decides what its name has to look like:
#   operator  -> this operator's own __global__ entries; must carry atrex_.
#   framework -> torch kernels/memcpy/memset the dispatcher launches around the
#                operator (workspace zero-fill, scale arithmetic).
#   cutlass   -> third-party GEMM adapter, reachable only from the M>1024
#                large-M path, whose launch site is dev's
#                kernels/down_gemm_fused_finalize.cu.
#   unknown   -> anything else, and a hard failure: an unattributed event is
#                exactly how an unprefixed kernel slips past a marker whitelist.
# Attribution is by prefix first, so an operator kernel that happens to mention
# "cutlass" (atrex_setup_cutlass_group_ptrs) is never misfiled as third-party.
_FRAMEWORK_NAME_PATTERNS = (
    "at::native::", "at::cuda::", "FillFunctor", "CUDAFunctor",
    "elementwise_kernel", "reduce_kernel", "Memset", "Memcpy",
    "CatArrayBatchedCopy", "direct_copy_kernel", "arange_kernel",
    "fill_kernel",
)
_CUTLASS_NAME_PATTERNS = (
    "_ZN7cutlass", "cutlass::device_kernel", "cutlass::gemm",
)

# Operator-owned stage families, derived from the sources this dispatch actually
# launches (routing_sort.cu, expand_input_rows.cu, e512_topk10_gemm1.cu,
# e512_topk10_activation.cu, e512_topk10_gemm2.cu, up_gate_gemm.cu) and
# confirmed against the observed profiler names for M=1/16/2048 (diag job
# dv_78e9b0faad98). Every pattern is a substring of both the mangled and the
# demangled form of the entry point.
#
# Routing has two mutually exclusive implementations, selected inside
# routing_sort.cu by token count, so it is split into two families and each
# branch asserts only the kernels that branch really launches:
#   routing_compact (1 <= M <= 16): the single fused single-CTA compact kernel;
#   routing_prefix  (M > 16):       the block -> global -> merge prefix-sum
#                                   pipeline; all three always launch together.
#                                   The global stage has a small and a
#                                   >2048-token large instantiation that share
#                                   the "globalExpertPrefixSum" stem, so the
#                                   stem is asserted rather than one variant.
_E512_STAGE_PATTERNS = {
    "routing_compact": ("routingSortCompactSmallMKernel",),
    "routing_prefix": (
        "blockExpertPrefixSumKernel",
        "globalExpertPrefixSum",
        "mergeExpertPrefixSumKernel",
    ),
    "expand": ("expandInputRowsKernel",),
    "gemm1_m1": ("gemm1_m1_splitk_fused_reduce_kernel",),
    "activation": ("doActivationKernel",),
    "gemm2_m1": ("gemm2_m1_row_kernel",),
    "gemm1_grouped_m16": ("gemm1_grouped_m16_fused_act_kernel",),
    "gemm2_small_m": ("gemm2_small_m_kernel",),
    "gemm1_large_m": (
        "compute_tile_info_v20_kernel",
        "grouped_gemm_v20_fused_act_kernel",
    ),
    "gemm2_large_m": ("setup_fused_finalize_group_ptrs",),
}

# routing_sort.cu takes the compact single-CTA branch for 1 <= num_tokens <= 16
# (with E=512, topk=10) and the three-kernel prefix-sum pipeline above that.
# This threshold is independent of the GEMM small-M/large-M split below.
_E512_ROUTING_COMPACT_MAX_M = 16


def _e512_small_m_max():
    from atrex.api.nvfp4_fused_moe_sm120 import (
        _E512_TOPK10_TASK29_TASK30_MAX_M,
    )
    return int(_E512_TOPK10_TASK29_TASK30_MAX_M)


def _e512_required_stages(M):
    """Stage families the dispatch must launch for this M, per branch.

    Routing: M<=16 uses the fused compact kernel, M>16 the block/global/merge
    prefix-sum pipeline (see _E512_STAGE_PATTERNS). GEMM: M==1 uses the split-K
    gemm1 with a separate BF16-boundary activation and the row-wise gemm2;
    1<M<=1024 fuses activation into both GEMMs, so no standalone activation
    kernel exists; M>1024 hands gemm2 to the dev cutlass fused-finalize path
    behind its group-pointer setup helper.
    """
    routing = (("routing_compact",) if M <= _E512_ROUTING_COMPACT_MAX_M
               else ("routing_prefix",))
    if M == 1:
        return routing + ("expand", "gemm1_m1", "activation", "gemm2_m1")
    if M <= _e512_small_m_max():
        return routing + ("expand", "gemm1_grouped_m16", "gemm2_small_m")
    return routing + ("expand", "gemm1_large_m", "gemm2_large_m")


@requires_sm120a
@pytest.mark.parametrize("pipeline", PIPELINES)
@pytest.mark.parametrize("M", E512_TOPK10_M_VALUES)
def test_e512_topk10_profiler_kernel_names_have_atrex_prefix(M, pipeline):
    names = _profile_e512_topk10_kernel_names(M, pipeline=pipeline)
    tag = f"e512_topk10 M={M} {pipeline}"
    assert names, f"profiler observed no CUDA kernels for {tag}"
    distinct = sorted(set(names))

    operator, framework, cutlass, unknown = [], [], [], []
    for name in distinct:
        if _has_atrex_prefix(name):
            operator.append(name)
        elif any(p in name for p in _FRAMEWORK_NAME_PATTERNS):
            framework.append(name)
        elif any(p in name for p in _CUTLASS_NAME_PATTERNS):
            cutlass.append(name)
        else:
            unknown.append(name)

    assert not unknown, (
        f"[{tag}] unattributed CUDA events; every observed kernel must be "
        f"operator-owned (atrex_), torch framework, or third-party cutlass: "
        f"{unknown}")
    assert operator, (
        f"[{tag}] no operator-owned kernel observed, so the dispatch cannot "
        f"have run; framework={framework}, cutlass={cutlass}")

    missing_prefix = [n for n in operator if not _has_atrex_prefix(n)]
    assert not missing_prefix, (
        f"[{tag}] every operator-owned kernel must carry the atrex_ prefix "
        f"(skill §4); missing: {missing_prefix}")

    if M <= _e512_small_m_max():
        assert not cutlass, (
            f"[{tag}] the small-M branch is fully operator-owned and must not "
            f"reach the cutlass adapter: {cutlass}")
    else:
        assert cutlass, (
            f"[{tag}] the large-M branch must reuse the dev cutlass "
            f"fused-finalize gemm2; observed operator kernels: {operator}")

    for stage in _e512_required_stages(M):
        for pattern in _E512_STAGE_PATTERNS[stage]:
            assert any(pattern in n for n in operator), (
                f"[{tag}] stage '{stage}' entry point '{pattern}' was never "
                f"launched; operator-owned kernels: {operator}")

    print(f"\n[{tag}] {len(distinct)} distinct CUDA events "
          f"({len(operator)} operator, {len(framework)} framework, "
          f"{len(cutlass)} cutlass, {len(unknown)} unknown)")
    for name in distinct:
        owner = ("operator" if name in operator else
                 "framework" if name in framework else
                 "cutlass" if name in cutlass else "unknown")
        print(f"  [{owner:>9s}] {name}")


# ── CUDA graph routing-state coverage ──────────────────────────────

_E512_GRAPH_SENTINEL_SEQUENCE = ("none", "partial", "all", "mixed", "none")


def _e512_graph_routing_variants(base_ids, topk):
    """Routing schedule replayed against one captured graph.

    M==1 has a single token, so "mixed" degenerates; it cycles explicit
    valid-slot counts instead, which is the same axis the finalize completion
    threshold depends on. Larger M walks the full sentinel-mode sequence twice:
    the second pass is what exposes a completion counter or an expert_rows
    buffer left over from the previous replay, because every static address is
    reused while the routing changes underneath it.
    """
    if base_ids.shape[0] == 1:
        variants = []
        for count in (topk, topk // 2, 1, 0, topk, 3, 0, topk // 2):
            ids = base_ids.clone()
            ids[:, count:] = -1
            variants.append((f"valid_slots={count}", ids))
        return variants
    variants = []
    for repeat in range(2):
        for mode in _E512_GRAPH_SENTINEL_SEQUENCE:
            variants.append(
                (f"{mode}#pass{repeat}",
                 apply_topk_sentinel_mode(base_ids, mode)))
    return variants


@requires_sm120a
@pytest.mark.parametrize("pipeline", PIPELINES)
@pytest.mark.parametrize("M", E512_TOPK10_GRAPH_M_VALUES)
def test_e512_topk10_cuda_graph_routing_state(M, pipeline):
    """Capture once, then change the routing in place at the captured address.

    Both output contracts are replayed against the same static input buffers:
    a non-preinitialized graph whose output is dirtied with NaN/Inf outside the
    graph before every replay (only captured nodes may clean it), and a
    preinitialized graph whose output is restored to a finite initial value
    that empty rows must preserve exactly. Nothing here resets internal
    completion counters or workspaces; the public API is the only entry point.
    """
    _shape_name, E, topk, hidden_size, inter_size = E512_TOPK10_SHAPE
    device = torch.device("cuda:0")
    env = make_direct_nvfp4_env_shape(
        M, E, topk, hidden_size, inter_size, device=device,
        topk_sentinel_mode="none")
    weights = env["weights"]
    base_ids = env["topk_ids"].clone()

    w1_fp4, w1_bs = cached_w1_layout(weights, inter_size, "interleaved")
    a_fp4, a_sf = quantize_activation_nvfp4(env["a"], weights["a1_gs"][0])

    topk_ids_static = base_ids.clone()
    data = {
        "a": a_fp4,
        "w1_fp4": w1_fp4,
        "w2_fp4": weights["w2_fp4"],
        "w1_blockscale": w1_bs,
        "w2_blockscale": weights["w2_blockscale"],
        "a1_gs": weights["a1_gs"],
        "a2_gs": weights["a2_gs"],
        "w1_gs": weights["w1_gs"],
        "w2_gs": weights["w2_gs"],
        "topk_ids": topk_ids_static,
        "topk_weights": env["topk_weights"],
    }
    output = torch.zeros(
        (M, hidden_size), device=device, dtype=torch.bfloat16)
    pre_output, initial = make_output_contract_buffer(
        "preinitialized", M, hidden_size, device)

    # Warmup on the captured shape primes atrex static buffers and the custom
    # TMA descriptors before capture.
    for _ in range(3):
        _launch_atrex(data, output, input_sf=a_sf, pipeline=pipeline)
        _launch_atrex(
            data, pre_output, input_sf=a_sf, pipeline=pipeline,
            output_preinitialized=True)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _launch_atrex(data, output, input_sf=a_sf, pipeline=pipeline)
    pre_graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(pre_graph):
        _launch_atrex(
            data, pre_output, input_sf=a_sf, pipeline=pipeline,
            output_preinitialized=True)
    torch.cuda.synchronize()

    variants = _e512_graph_routing_variants(base_ids, topk)
    for label, ids in variants:
        # Routing is rewritten in place at the address the graph captured; the
        # replay has to pick it up without a re-capture.
        topk_ids_static.copy_(ids)
        ref_env = dict(env)
        ref_env["topk_ids"] = ids
        fi_f32 = flashinfer_reference_output(
            ref_env, weights, hidden_size, inter_size, pipeline, device
        ).float()
        empty_rows = valid_slot_counts(ids) == 0

        output.fill_(float("nan"))
        output[::2, ::3] = float("inf")
        torch.cuda.synchronize()
        graph.replay()
        torch.cuda.synchronize()
        assert_output_contract(
            output, fi_f32, empty_rows=empty_rows, initial=None,
            tag=f"graph {pipeline} M={M} {label}",
            min_cos=E512_TOPK10_MIN_COS, min_row_cos=E512_TOPK10_MIN_ROW_COS)

        pre_output.copy_(initial)
        torch.cuda.synchronize()
        pre_graph.replay()
        torch.cuda.synchronize()
        assert_output_contract(
            pre_output, fi_f32 + initial.float(), empty_rows=empty_rows,
            initial=initial,
            tag=f"graph-preinit {pipeline} M={M} {label}",
            min_cos=E512_TOPK10_MIN_COS, min_row_cos=E512_TOPK10_MIN_ROW_COS)

    # Same routing replayed back to back: state reuse must be idempotent.
    # Atomic accumulation order can vary on the large-M cutlass path, so this
    # is a similarity check rather than a bitwise one.
    label, ids = variants[-1]
    topk_ids_static.copy_(ids)
    output.fill_(float("nan"))
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    first = output.clone()
    output.fill_(float("nan"))
    torch.cuda.synchronize()
    graph.replay()
    torch.cuda.synchronize()
    _assert_repeatable_output(first, output, pipeline)

    print(f"\n[graph {pipeline} M={M}] {len(variants)} routing variants "
          f"replayed on one captured graph, both output contracts verified "
          f"per replay; final labels: {[v[0] for v in variants]}")


# ── ABBA migration performance gate (e512_topk10) ─────────────────
#
# e512_topk10 was migrated from revision 90e4bf46, so its latency has to be
# defended against that revision instead of only reported next to FlashInfer.
# Both versions are measured from this file's private CLI worker mode
#
#     python test_nvfp4_fused_moe_sm120.py abba-worker
#
# one persistent subprocess per version, each with its own installed wheel and
# its own ATREX_JIT_CACHE_DIR, and never with a source tree on PYTHONPATH. Data
# generation, layout conversion and the public API call are therefore identical
# on both sides and only the installed atrex differs; the parent verifies that
# by comparing an input digest plus each worker's resolved package, JIT source
# and cache paths.
#
# Per case and pipeline the parent alternates A/B and B/A blocks across rounds,
# so slow clock drift lands on both versions rather than on one. Timing is the
# warm-cache eager CUDA-event convention already used by
# profile_e2e_cuda_events: no profiler attached, no cache flush, no weight
# quantization and no output zeroing inside the timed region -- only the public
# API call. The verdict is each version's median of block medians, and it
# hard-fails above +5% for any single case: no geometric mean, no offsetting a
# regression with a speedup elsewhere.
#
# Environment for a real run -- the gate refuses to guess any of the required
# ones, because a silently wrong interpreter or header set would turn the
# comparison into a report about nothing:
#   ATREX_ABBA_BASELINE_PYTHON    python of a venv holding the 90e4bf46 wheel
#   ATREX_ABBA_CANDIDATE_PYTHON   python of the candidate venv (default: ours)
#   ATREX_ABBA_BASELINE_CUTLASS_DIR / ATREX_ABBA_CANDIDATE_CUTLASS_DIR
#       per-side cutlass headers; unset means that side compiles against the
#       headers packaged inside its own wheel, which is what a user gets
#   ATREX_ABBA_HARNESS / _JIT_ROOT / _ROUNDS / _WARMUP / _ITERS /
#   _BASELINE_REVISION / _CANDIDATE_REVISION / _REPORT_DIR: optional

ABBA_WORKER_ARGV = "abba-worker"
ABBA_DEFAULT_ROUNDS = 5
ABBA_DEFAULT_WARMUP = 20
ABBA_DEFAULT_ITERS = 50
ABBA_MAX_RATIO = 1.05

_ABBA_ENV_BASELINE_PYTHON = "ATREX_ABBA_BASELINE_PYTHON"
_ABBA_ENV_CANDIDATE_PYTHON = "ATREX_ABBA_CANDIDATE_PYTHON"
_ABBA_ENV_HARNESS = "ATREX_ABBA_HARNESS"
_ABBA_ENV_JIT_ROOT = "ATREX_ABBA_JIT_ROOT"
_ABBA_ENV_ROUNDS = "ATREX_ABBA_ROUNDS"
_ABBA_ENV_WARMUP = "ATREX_ABBA_WARMUP"
_ABBA_ENV_ITERS = "ATREX_ABBA_ITERS"
_ABBA_ENV_BASELINE_REVISION = "ATREX_ABBA_BASELINE_REVISION"
_ABBA_ENV_CANDIDATE_REVISION = "ATREX_ABBA_CANDIDATE_REVISION"
_ABBA_ENV_BASELINE_CUTLASS = "ATREX_ABBA_BASELINE_CUTLASS_DIR"
_ABBA_ENV_CANDIDATE_CUTLASS = "ATREX_ABBA_CANDIDATE_CUTLASS_DIR"
_ABBA_ENV_REPORT = "ATREX_ABBA_REPORT_DIR"
_ABBA_DEFAULT_JIT_ROOT = "/tmp/atrex_abba_jit"


def _abba_emit(payload):
    sys.stdout.write(json.dumps(payload) + "\n")
    sys.stdout.flush()


def _abba_input_digest(entry):
    """Content digest of the measured workload, comparable across processes."""
    parts = [
        entry["a_fp4"].flatten()[:4096].cpu().numpy().tobytes(),
        entry["topk_ids"].cpu().numpy().tobytes(),
        entry["topk_weights"].cpu().numpy().tobytes(),
        entry["w1_fp4"][0, :8, :8].cpu().numpy().tobytes(),
        entry["w2_fp4"][0, :8, :8].cpu().numpy().tobytes(),
        entry["w1_blockscale"][0, :8, :8].contiguous().view(
            torch.uint8).cpu().numpy().tobytes(),
    ]
    return hashlib.sha256(b"".join(parts)).hexdigest()[:32]


def _abba_prepare_case(case):
    """Build one measured workload and prewarm every pipeline on it.

    The first call of a shape JIT-compiles the extension and builds the TMA
    descriptors, so prewarming happens here and is never charged to a block.
    """
    M = int(case["M"])
    hidden_size = int(case["hidden_size"])
    inter_size = int(case["inter_size"])
    env = make_direct_nvfp4_env_shape(
        M, int(case["num_experts"]), int(case["topk"]), hidden_size,
        inter_size, device="cuda:0", topk_sentinel_mode="none")
    weights = env["weights"]
    w1_fp4, w1_bs = cached_w1_layout(weights, inter_size, "interleaved")
    a_fp4, a_sf = quantize_activation_nvfp4(env["a"], weights["a1_gs"][0])
    output = torch.zeros(
        (M, hidden_size), device="cuda:0", dtype=torch.bfloat16)

    runs = {}
    for pipeline in case["pipelines"]:
        def run(pipeline=pipeline):
            atrex.nvfp4_fused_moe(
                hidden_states=a_fp4,
                w1_fp4=w1_fp4,
                w2_fp4=weights["w2_fp4"],
                w1_blockscale=w1_bs,
                w2_blockscale=weights["w2_blockscale"],
                a1_global_scale=weights["a1_gs"],
                a2_global_scale=weights["a2_gs"],
                w1_global_scale=weights["w1_gs"],
                w2_global_scale=weights["w2_gs"],
                topk_ids=env["topk_ids"],
                topk_weights=env["topk_weights"],
                output=output,
                input_sf=a_sf,
                pipeline=pipeline,
            )
        runs[pipeline] = run

    entry = {
        "runs": runs,
        "a_fp4": a_fp4,
        "topk_ids": env["topk_ids"],
        "topk_weights": env["topk_weights"],
        "w1_fp4": w1_fp4,
        "w2_fp4": weights["w2_fp4"],
        "w1_blockscale": w1_bs,
    }
    for run in runs.values():
        for _ in range(3):
            run()
    torch.cuda.synchronize()
    entry["digest"] = _abba_input_digest(entry)
    return entry


def _abba_measure(entry, pipeline, warmup, iters):
    """One measurement block: warm-cache eager CUDA events around the API."""
    run = entry["runs"][pipeline]
    for _ in range(warmup):
        run()
    torch.cuda.synchronize()

    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    samples = []
    for _ in range(iters):
        start.record()
        run()
        end.record()
        torch.cuda.synchronize()
        samples.append(start.elapsed_time(end) * 1000.0)

    mean = statistics.mean(samples)
    stdev = statistics.pstdev(samples) if len(samples) > 1 else 0.0
    return {
        "samples_us": samples,
        "median_us": statistics.median(samples),
        "min_us": min(samples),
        "max_us": max(samples),
        "mean_us": mean,
        "stdev_us": stdev,
        "cv": (stdev / mean) if mean > 0 else 0.0,
    }


def _abba_info(request):
    from atrex.core.compile_cu import _atrex_source_root, _jit_cache_root

    return {
        "ok": True,
        "label": request.get("label"),
        "revision": request.get("revision"),
        "python": sys.executable,
        "atrex_file": atrex.__file__,
        "source_root": str(_atrex_source_root()),
        "jit_cache": str(_jit_cache_root()),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "cutlass_dir": os.environ.get("ATREX_CUTLASS_DIR"),
        "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
    }


def _abba_worker_main():
    """Private CLI mode: serve measure requests over a line protocol."""
    cases = {}
    for line in sys.stdin:
        line = line.strip()
        if not line:
            continue
        try:
            request = json.loads(line)
        except json.JSONDecodeError as error:
            _abba_emit({"ok": False, "error": f"malformed request: {error}"})
            continue
        op = request.get("op")
        # stdout carries the protocol and nothing else: JIT build logs and any
        # library chatter are redirected to stderr, which the parent inherits.
        with contextlib.redirect_stdout(sys.stderr):
            try:
                if op == "info":
                    payload = _abba_info(request)
                elif op == "prepare":
                    entry = _abba_prepare_case(request["case"])
                    cases[request["case_id"]] = entry
                    payload = {"ok": True, "case_id": request["case_id"],
                               "digest": entry["digest"]}
                elif op == "measure":
                    payload = {"ok": True}
                    payload.update(_abba_measure(
                        cases[request["case_id"]], request["pipeline"],
                        int(request["warmup"]), int(request["iters"])))
                elif op == "exit":
                    payload = {"ok": True}
                else:
                    payload = {"ok": False, "error": f"unknown op: {op!r}"}
            except BaseException as error:  # keep the protocol alive
                payload = {"ok": False,
                           "error": f"{type(error).__name__}: {error}"}
        _abba_emit(payload)
        if op == "exit":
            break
    return 0


def _abba_gpu_state():
    """Read-only SM clock / temperature / utilization snapshot for the report."""
    try:
        completed = subprocess.run(
            ["nvidia-smi",
             "--query-gpu=clocks.sm,temperature.gpu,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=30)
    except Exception as error:
        return {"error": f"{type(error).__name__}: {error}"}
    if completed.returncode != 0:
        return {"error": f"nvidia-smi rc={completed.returncode}"}
    fields = [part.strip() for part in completed.stdout.strip().split(",")]
    return {
        "clock_sm_mhz": fields[0] if len(fields) > 0 else None,
        "temp_c": fields[1] if len(fields) > 1 else None,
        "util_pct": fields[2] if len(fields) > 2 else None,
    }


class _AbbaWorker:
    """One persistent measurement subprocess bound to one installed wheel."""

    def __init__(self, label, python_exe, harness, jit_cache, revision, cwd,
                 cutlass_dir=None):
        self.label = label
        self.python_exe = python_exe
        self.harness = harness
        self.jit_cache = jit_cache
        self.revision = revision
        self.cwd = cwd
        self.cutlass_dir = cutlass_dir
        self.proc = None
        self.info = None

    def start(self):
        if not os.path.exists(self.python_exe):
            raise RuntimeError(
                f"{self.label} python interpreter does not exist: "
                f"{self.python_exe}")
        os.makedirs(self.jit_cache, exist_ok=True)
        env = dict(os.environ)
        # A source tree on PYTHONPATH would silently shadow the installed wheel
        # and make the comparison meaningless.
        env.pop("PYTHONPATH", None)
        # The two revisions disagree on their cutlass pin (dev tracks v4.4.2,
        # 90e4bf46 needs >= 4.5), so each side's headers are declared here
        # instead of inherited from whatever the parent shell happens to export.
        # An undeclared side keeps only the headers packaged in its own wheel.
        for key in ("ATREX_CUTLASS_DIR", "CUTLASS_DIR"):
            env.pop(key, None)
        if self.cutlass_dir:
            env["ATREX_CUTLASS_DIR"] = self.cutlass_dir
            env["CUTLASS_DIR"] = self.cutlass_dir
        env["ATREX_JIT_CACHE_DIR"] = self.jit_cache
        self.proc = subprocess.Popen(
            [self.python_exe, self.harness, ABBA_WORKER_ARGV],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            text=True,
            bufsize=1,
            env=env,
            cwd=self.cwd,
        )
        self.info = self.request(
            {"op": "info", "label": self.label, "revision": self.revision})
        self._assert_installed_package()
        return self.info

    def _assert_installed_package(self):
        for key in ("atrex_file", "source_root"):
            path = self.info[key]
            assert "site-packages" in path, (
                f"[{self.label}] {key}={path} is not inside an installed "
                "wheel (site-packages); the gate measures installed packages, "
                "not a source tree")
        package_dir = os.path.dirname(self.info["atrex_file"])
        assert self.info["source_root"].startswith(package_dir + os.sep), (
            f"[{self.label}] JIT sources {self.info['source_root']} do not come "
            f"from the installed package {package_dir}")
        assert (os.path.realpath(self.info["jit_cache"])
                == os.path.realpath(self.jit_cache)), (
            f"[{self.label}] unexpected JIT cache {self.info['jit_cache']}, "
            f"requested {self.jit_cache}")

    def request(self, payload):
        self.proc.stdin.write(json.dumps(payload) + "\n")
        self.proc.stdin.flush()
        while True:
            line = self.proc.stdout.readline()
            if not line:
                raise RuntimeError(
                    f"[{self.label}] worker exited before replying to "
                    f"{payload.get('op')!r}")
            line = line.strip()
            if not line.startswith("{"):
                print(f"[abba:{self.label}] {line}", file=sys.stderr)
                continue
            reply = json.loads(line)
            if not reply.get("ok"):
                raise RuntimeError(
                    f"[{self.label}] worker failed on "
                    f"{payload.get('op')!r}: {reply.get('error')}")
            return reply

    def close(self):
        if self.proc is None:
            return
        try:
            with contextlib.suppress(Exception):
                self.request({"op": "exit"})
        finally:
            with contextlib.suppress(Exception):
                self.proc.stdin.close()
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=30)
            self.proc = None


# test_profile_nvfp4 is parametrized over every e512_topk10 M value, so the gate
# below runs once per M. One worker per version is started for the whole pytest
# process and reused: a fresh subprocess per M would pay eight CUDA context
# initializations per side and throw away the warm in-process JIT handle cache.
_ABBA_WORKERS = {}


def _abba_get_worker(label, python_exe, harness, jit_cache, revision, cwd,
                     cutlass_dir=None):
    key = (label, python_exe, jit_cache)
    worker = _ABBA_WORKERS.get(key)
    if worker is None:
        worker = _AbbaWorker(
            label, python_exe, harness, jit_cache, revision, cwd,
            cutlass_dir=cutlass_dir)
        worker.start()
        _ABBA_WORKERS[key] = worker
    return worker


def _abba_close_workers():
    while _ABBA_WORKERS:
        _ABBA_WORKERS.popitem()[1].close()


atexit.register(_abba_close_workers)


def _abba_env_int(name, default):
    raw = os.environ.get(name)
    if raw is None or not str(raw).strip():
        return default
    return int(raw)


def gate_e512_topk10_abba(
    M, num_experts, topk, hidden_size, inter_size, pipelines):
    """Hard migration gate for one e512_topk10 M value.

    Requires a baseline environment; a missing one is an error, never a skip and
    never a printed-only speedup, because the whole point of the gate is that
    the migrated path is compared against the revision it came from.
    """
    baseline_python = os.environ.get(_ABBA_ENV_BASELINE_PYTHON)
    if not baseline_python:
        raise RuntimeError(
            "e512_topk10 migration performance gate requires "
            f"{_ABBA_ENV_BASELINE_PYTHON} (a python from a venv with the "
            "90e4bf46 wheel installed). Refusing to report a speedup without "
            "the baseline it would be measured against.")
    candidate_python = (
        os.environ.get(_ABBA_ENV_CANDIDATE_PYTHON) or sys.executable)
    harness = os.path.abspath(
        os.environ.get(_ABBA_ENV_HARNESS) or __file__)
    cwd = os.path.dirname(harness)
    jit_root = os.environ.get(_ABBA_ENV_JIT_ROOT) or _ABBA_DEFAULT_JIT_ROOT
    rounds = _abba_env_int(_ABBA_ENV_ROUNDS, ABBA_DEFAULT_ROUNDS)
    warmup = _abba_env_int(_ABBA_ENV_WARMUP, ABBA_DEFAULT_WARMUP)
    iters = _abba_env_int(_ABBA_ENV_ITERS, ABBA_DEFAULT_ITERS)

    case_id = f"e512_topk10-M{M}"
    case = {
        "M": M, "num_experts": num_experts, "topk": topk,
        "hidden_size": hidden_size, "inter_size": inter_size,
        "pipelines": list(pipelines),
    }
    workers = {
        "baseline": _abba_get_worker(
            "baseline", baseline_python, harness,
            os.path.join(jit_root, "baseline"),
            os.environ.get(_ABBA_ENV_BASELINE_REVISION, "90e4bf46"), cwd,
            cutlass_dir=os.environ.get(_ABBA_ENV_BASELINE_CUTLASS)),
        "candidate": _abba_get_worker(
            "candidate", candidate_python, harness,
            os.path.join(jit_root, "candidate"),
            os.environ.get(_ABBA_ENV_CANDIDATE_REVISION, "local"), cwd,
            cutlass_dir=os.environ.get(_ABBA_ENV_CANDIDATE_CUTLASS)),
    }
    baseline = workers["baseline"]
    candidate = workers["candidate"]
    assert baseline.info["atrex_file"] != candidate.info["atrex_file"], (
        "both ABBA workers resolved the same atrex installation "
        f"({baseline.info['atrex_file']}); the two wheels are not isolated")
    assert baseline.info["jit_cache"] != candidate.info["jit_cache"], (
        "both ABBA workers share one JIT cache; compiled kernels would "
        "cross-contaminate the two versions")

    blocks = {}
    digests = {}
    for label, worker in workers.items():
        digests[label] = worker.request(
            {"op": "prepare", "case_id": case_id, "case": case})["digest"]
    assert digests["baseline"] == digests["candidate"], (
        f"ABBA workload differs between versions: {digests}")

    for pipeline in pipelines:
        blocks[pipeline] = {"baseline": [], "candidate": []}
        for round_index in range(rounds):
            # Alternate the block order every round so a drifting clock or a
            # warming card is charged to both versions equally.
            order = (("baseline", "candidate") if round_index % 2 == 0
                     else ("candidate", "baseline"))
            for label in order:
                reply = workers[label].request({
                    "op": "measure", "case_id": case_id,
                    "pipeline": pipeline, "warmup": warmup,
                    "iters": iters,
                })
                reply["round"] = round_index
                reply["gpu"] = _abba_gpu_state()
                blocks[pipeline][label].append(reply)

    report = {
        "case_id": case_id,
        "shape": [num_experts, topk, hidden_size, inter_size],
        "workload_digest": digests["baseline"],
        "rounds": rounds, "warmup": warmup, "iters": iters,
        "max_ratio": ABBA_MAX_RATIO,
        "workers": {label: worker.info for label, worker in workers.items()},
        "pipelines": {},
    }
    failures = []
    for pipeline in pipelines:
        medians = {}
        for label in ("baseline", "candidate"):
            values = [b["median_us"] for b in blocks[pipeline][label]]
            medians[label] = statistics.median(values)
        ratio = medians["candidate"] / medians["baseline"]
        report["pipelines"][pipeline] = {
            "block_medians_us": {
                label: [b["median_us"] for b in blocks[pipeline][label]]
                for label in ("baseline", "candidate")},
            "block_cv": {
                label: [round(b["cv"], 4) for b in blocks[pipeline][label]]
                for label in ("baseline", "candidate")},
            "gpu_state": {
                label: [b["gpu"] for b in blocks[pipeline][label]]
                for label in ("baseline", "candidate")},
            "median_us": medians,
            "ratio": ratio,
            "samples_us": {
                label: [b["samples_us"] for b in blocks[pipeline][label]]
                for label in ("baseline", "candidate")},
        }
        if ratio > ABBA_MAX_RATIO:
            failures.append(
                f"{case_id} {pipeline}: candidate/baseline = {ratio:.4f} "
                f"> {ABBA_MAX_RATIO} "
                f"(baseline median {medians['baseline']:.1f}us, "
                f"candidate median {medians['candidate']:.1f}us)")

    report_dir = os.environ.get(_ABBA_ENV_REPORT) or jit_root
    os.makedirs(report_dir, exist_ok=True)
    report_path = os.path.join(report_dir, f"abba_{case_id}.json")
    with open(report_path, "w") as handle:
        json.dump(report, handle, indent=2)

    print(f"\n=== ABBA migration gate: {case_id} "
          f"(E={num_experts}, topk={topk}, K={hidden_size}, N={inter_size}) ===")
    print(f"  baseline  : revision={baseline.revision} "
          f"atrex={baseline.info['atrex_file']} "
          f"cutlass={baseline.info.get('cutlass_dir') or 'packaged'}")
    print(f"  candidate : revision={candidate.revision} "
          f"atrex={candidate.info['atrex_file']} "
          f"cutlass={candidate.info.get('cutlass_dir') or 'packaged'}")
    print(f"  rounds={rounds} warmup={warmup} iters={iters} "
          f"workload_digest={digests['baseline']}")
    print(f"  device={candidate.info['device']} "
          f"torch={candidate.info['torch']} cuda={candidate.info['cuda']}")
    print(f"  {'pipeline':>12s} | {'baseline us':>12s} | {'candidate us':>13s} "
          f"| {'ratio':>7s} | {'gate':>6s}")
    for pipeline in pipelines:
        entry = report["pipelines"][pipeline]
        print(f"  {pipeline:>12s} | {entry['median_us']['baseline']:12.1f} "
              f"| {entry['median_us']['candidate']:13.1f} "
              f"| {entry['ratio']:7.4f} "
              f"| {'PASS' if entry['ratio'] <= ABBA_MAX_RATIO else 'FAIL':>6s}")
        print(f"  {'':>12s}   block medians baseline="
              f"{[round(v, 1) for v in entry['block_medians_us']['baseline']]} "
              f"candidate="
              f"{[round(v, 1) for v in entry['block_medians_us']['candidate']]}")
    print(f"  full samples and per-block clock/temperature: {report_path}")

    assert not failures, (
        "e512_topk10 migration performance regression beyond "
        f"{int((ABBA_MAX_RATIO - 1) * 100)}%:\n  " + "\n  ".join(failures))


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == ABBA_WORKER_ARGV:
        raise SystemExit(_abba_worker_main())
    raise SystemExit(
        f"usage: {os.path.basename(__file__)} {ABBA_WORKER_ARGV}\n"
        "this module is otherwise driven by pytest")
