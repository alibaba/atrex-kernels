"""NVIDIA SM120 Chunk-GDN public API and kernel tests.

Run directly to compare against FlashInfer on the canonical shape:
    python op_test/nvidia/chunk_gdn/test_chunk_gdn_sm120.py
"""

import argparse
import csv
import inspect
import io
import math
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

import pytest
torch = pytest.importorskip("torch")
import torch.nn.functional as F


_PROFILE_WORKLOAD_FLAG = "--profile-kernel-names"


def _requires_sm120():
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if torch.cuda.get_device_capability() != (12, 0):
        pytest.skip("SM120 CUDA device is required")


def _require_sm120_or_exit():
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    if torch.cuda.get_device_capability() != (12, 0):
        major, minor = torch.cuda.get_device_capability()
        raise SystemExit(f"SM120 CUDA device is required, got sm_{major}{minor}")


def _make_inputs(B=1, T=128, H=16, HV=32, K=128, V=128):
    torch.manual_seed(42)
    dtype = torch.bfloat16
    q = torch.randn(B, T, H, K, device="cuda", dtype=dtype)
    k = torch.randn(B, T, H, K, device="cuda", dtype=dtype)
    v = torch.randn(B, T, HV, V, device="cuda", dtype=dtype)
    g = F.logsigmoid(torch.randn(B, T, HV, device="cuda", dtype=dtype))
    beta = torch.sigmoid(torch.randn(B, T, HV, device="cuda", dtype=dtype))
    return q, k, v, g, beta


def test_chunk_gdn_public_api_exposes_state_checkpoints():
    from atrex.api.chunk_gdn_cutedsl import (
        can_use_chunk_gdn_fwd_cutedsl,
        chunk_gdn_fwd_cutedsl,
        chunk_gdn_fwd_cutedsl_prewarm_buckets,
    )

    checkpoint_parameters = {
        "state_checkpoints",
        "checkpoint_cu_starts",
        "checkpoint_every_n_tokens",
    }
    assert checkpoint_parameters.issubset(
        inspect.signature(chunk_gdn_fwd_cutedsl).parameters
    )
    assert checkpoint_parameters.issubset(
        inspect.signature(can_use_chunk_gdn_fwd_cutedsl).parameters
    )
    assert "include_state_checkpoints" in inspect.signature(
        chunk_gdn_fwd_cutedsl_prewarm_buckets
    ).parameters


def test_chunk_gdn_rejects_mismatched_checkpoint_buffer_rows():
    _requires_sm120()
    from atrex.api.chunk_gdn_cutedsl import (
        chunk_gdn_fwd_cutedsl,
        chunk_gdn_fwd_cutedsl_build,
    )

    q, k, v, g, beta = _make_inputs(T=64)
    ctx = chunk_gdn_fwd_cutedsl_build(
        B=1, H=16, HV=32, K=128, V=128, output_final_state=True
    )
    cu_seqlens = torch.tensor([0, 64], device="cuda", dtype=torch.int32)
    cu_seqlens_cpu = torch.tensor([0, 64], dtype=torch.int64)
    checkpoint_cu_starts = torch.tensor([0, 1], device="cuda", dtype=torch.int64)
    state_checkpoints = torch.empty(
        (0, 32, 128, 128), device="cuda", dtype=torch.float32
    )

    with pytest.raises(ValueError, match="checkpoint plan: expected 1, got 0"):
        chunk_gdn_fwd_cutedsl(
            ctx,
            q,
            k,
            v,
            g,
            beta,
            output_final_state=True,
            cu_seqlens=cu_seqlens,
            cu_seqlens_cpu=cu_seqlens_cpu,
            state_checkpoints=state_checkpoints,
            checkpoint_cu_starts=checkpoint_cu_starts,
            checkpoint_every_n_tokens=64,
        )


def test_ptx_version_fallback_downgrades_supported_minor_versions():
    from atrex.src.nvidia.chunk_gdn.sm120 import custom_compile_cache

    for requested in ("9.1", "9.2", "9.3"):
        ptx = f"// generated\n.version {requested}\n.target sm_120a\n"
        stderr = (
            f"ptxas fatal   : Unsupported .version {requested}; "
            "current version is '9.0'"
        )

        patched = custom_compile_cache._downgrade_unsupported_ptx_version(
            ptx, stderr
        )

        assert patched == "// generated\n.version 9.0\n.target sm_120a\n"


def test_ptx_version_fallback_ignores_unrelated_or_unsafe_errors():
    from atrex.src.nvidia.chunk_gdn.sm120 import custom_compile_cache

    cases = (
        (".version 9.3\n", "ptxas fatal: parsing error"),
        (
            ".version 9.2\n",
            "Unsupported .version 9.3; current version is '9.0'",
        ),
        (
            ".version 10.0\n",
            "Unsupported .version 10.0; current version is '9.0'",
        ),
        (
            ".version 9.0\n",
            "Unsupported .version 9.0; current version is '9.3'",
        ),
    )
    for ptx, stderr in cases:
        assert (
            custom_compile_cache._downgrade_unsupported_ptx_version(
                ptx, stderr
            )
            is None
        )


def test_ptx_assembler_retries_with_reported_version(monkeypatch):
    from atrex.src.nvidia.chunk_gdn.sm120 import custom_compile_cache

    attempts = []

    def fake_run(cmd, **kwargs):
        del kwargs
        ptx_path = Path(cmd[2])
        attempts.append(ptx_path.read_text())
        if len(attempts) == 1:
            return subprocess.CompletedProcess(
                cmd,
                1,
                stdout="",
                stderr=(
                    "ptxas fatal   : Unsupported .version 9.3; "
                    "current version is '9.0'"
                ),
            )
        Path(cmd[4]).write_bytes(b"patched-cubin")
        return subprocess.CompletedProcess(cmd, 0, stdout="", stderr="")

    monkeypatch.setattr(
        custom_compile_cache,
        "_find_ptxas",
        lambda: "/opt/cuda/bin/ptxas",
    )
    monkeypatch.setattr(custom_compile_cache.subprocess, "run", fake_run)

    cubin = custom_compile_cache._assemble_sm120a_cubin(
        ".version 9.3\n.target sm_120a\n"
    )

    assert cubin == b"patched-cubin"
    assert len(attempts) == 2
    assert ".version 9.3" in attempts[0]
    assert ".version 9.0" in attempts[1]


def _make_directional_inputs(B=1, T=128, H=16, HV=32, K=128, V=128, seed=42):
    """Mimic real-model statistics for stress-testing tail behaviour.

    Real GDN q/k come out of a conv1d + (optionally) l2norm pipeline. They
    have:
      - small l2 norm per token (l2(q) ~ 1-3 vs randn's sqrt(K) ~ 11)
      - a dominant low-rank / shared-direction component across tokens
        (conv1d outputs are correlated along the token axis)

    randn inputs are i.i.d. across tokens, so many tail-handling bugs average
    out statistically. Directional inputs do not, so this generator should
    expose hidden tail-kernel issues.
    """
    torch.manual_seed(seed)
    dtype = torch.bfloat16
    fp32 = torch.float32
    q_dir = torch.randn(1, 1, H, K, device="cuda", dtype=fp32) * 0.10
    k_dir = torch.randn(1, 1, H, K, device="cuda", dtype=fp32) * 0.07
    v_dir = torch.randn(1, 1, HV, V, device="cuda", dtype=fp32) * 0.50
    q_noise = 0.02 * torch.randn(B, T, H, K, device="cuda", dtype=fp32)
    k_noise = 0.02 * torch.randn(B, T, H, K, device="cuda", dtype=fp32)
    v_noise = 0.05 * torch.randn(B, T, HV, V, device="cuda", dtype=fp32)
    q = (q_dir.expand(B, T, H, K) + q_noise).to(dtype).contiguous()
    k = (k_dir.expand(B, T, H, K) + k_noise).to(dtype).contiguous()
    v = (v_dir.expand(B, T, HV, V) + v_noise).to(dtype).contiguous()
    g = F.logsigmoid(torch.randn(B, T, HV, device="cuda", dtype=dtype))
    beta = torch.sigmoid(torch.randn(B, T, HV, device="cuda", dtype=dtype))
    return q, k, v, g, beta


_INPUT_MAKERS = {
    "randn": _make_inputs,
    "directional": _make_directional_inputs,
}

_CANONICAL_MAX_ATREX_MS = {
    32: 1.00,
    48: 1.25,
    64: 1.50,
}
_CHECKPOINT_MAX_ATREX_MS = 4.00


def _input_stats(q, k, v, g, beta):
    return (
        f"q_l2[mean={q.float().norm(dim=-1).mean().item():.3f},"
        f"min={q.float().norm(dim=-1).min().item():.3f},"
        f"max={q.float().norm(dim=-1).max().item():.3f}] "
        f"k_l2[mean={k.float().norm(dim=-1).mean().item():.3f}] "
        f"v_std={v.float().std().item():.3f} "
        f"g[{g.float().min().item():.2f},{g.float().max().item():.2f}] "
        f"beta[{beta.float().min().item():.3f},{beta.float().max().item():.3f}]"
    )


def _get_flashinfer_chunk_gdn():
    import flashinfer

    return flashinfer.chunk_gated_delta_rule


def _reference_l2norm(x):
    x_fp32 = x.float()
    return (
        x_fp32
        * torch.rsqrt((x_fp32 * x_fp32).sum(dim=-1, keepdim=True) + 1e-6)
    ).to(x.dtype)


def _flashinfer_chunk_gdn(
    q,
    k,
    v,
    g,
    beta,
    scale,
    output_final_state=False,
    *,
    cu_seqlens=None,
    initial_state=None,
):
    """Adapt the ATREX public contract to FlashInfer's flat GDN API.

    FlashInfer SM120 expects pre-normalized Q/K and a positive alpha gate,
    whereas the public ATREX contract receives batched raw Q/K and log-gates.
    Keep these conversions explicit so the precision comparison is semantic.
    """
    chunk_gated_delta_rule = _get_flashinfer_chunk_gdn()
    if cu_seqlens is None:
        cu_seqlens = torch.tensor(
            [0, q.shape[1]], dtype=torch.int32, device=q.device
        )
    q_reference = _reference_l2norm(q.squeeze(0)).contiguous()
    k_reference = _reference_l2norm(k.squeeze(0)).contiguous()
    v_reference = v.squeeze(0).contiguous()
    alpha_reference = torch.exp(g.squeeze(0).float()).contiguous()
    beta_reference = beta.squeeze(0).float().contiguous()
    result = chunk_gated_delta_rule(
        q_reference,
        k_reference,
        v_reference,
        g=alpha_reference,
        beta=beta_reference,
        scale=scale,
        initial_state=initial_state,
        cu_seqlens=cu_seqlens,
        use_qk_l2norm_in_kernel=False,
        output_final_state=output_final_state,
        use_cp=False,
    )
    if output_final_state:
        output, final_state = result
    else:
        output, final_state = result, None
    return output.unsqueeze(0), final_state


def _bench_ms(fn, warmup=10, rep=30):
    torch.cuda.synchronize()
    with torch.inference_mode():
        for _ in range(warmup):
            fn()
        torch.cuda.synchronize()

        times = []
        for _ in range(rep):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            fn()
            end.record()
            end.synchronize()
            times.append(start.elapsed_time(end))

    times.sort()
    return times[len(times) // 2]


def _rel_err(out, ref):
    return ((out.float() - ref.float()).abs().max() / (ref.float().abs().max() + 1e-8)).item()


def _compare_cutedsl_vs_flashinfer(
    T=6144,
    HV=32,
    warmup=10,
    rep=30,
    min_speedup=None,
    max_atrex_ms=None,
    output_final_state=True,
    input_mode="randn",
    strict=True,
):
    import atrex

    B, H, K, V = 1, 16, 128, 128
    make_inputs = _INPUT_MAKERS[input_mode]
    q, k, v, g, beta = make_inputs(B, T, H, HV, K, V)
    scale = 1.0 / math.sqrt(K)
    cu = torch.tensor([0, T], dtype=torch.int32, device=q.device)
    ctx = atrex.chunk_gdn_fwd_cutedsl_build(
        B=B, H=H, HV=HV, K=K, V=V,
        seq_len=T, output_final_state=output_final_state, scale=scale,
    )

    def cutedsl_call():
        # Exercise ATREX's optimized single-sequence route. FlashInfer requires
        # cu_seqlens even for the same one-sequence semantic contract; passing
        # it to ATREX would intentionally select the different varlen kernel.
        return atrex.chunk_gdn_fwd_cutedsl(
            ctx, q, k, v, g, beta, scale,
            output_final_state=output_final_state,
        )

    def flashinfer_call():
        return _flashinfer_chunk_gdn(
            q,
            k,
            v,
            g,
            beta,
            scale,
            output_final_state=output_final_state,
            cu_seqlens=cu,
        )

    with torch.inference_mode():
        out, final_state = cutedsl_call()
        ref, ref_state = flashinfer_call()
    torch.cuda.synchronize()

    rel_err = _rel_err(out, ref)
    state_rel_err = None
    if output_final_state:
        assert final_state is not None
        assert ref_state is not None
        state_rel_err = _rel_err(final_state, ref_state)
    else:
        assert final_state is None
        assert ref_state is None

    cutedsl_ms = _bench_ms(cutedsl_call, warmup=warmup, rep=rep)
    flashinfer_ms = _bench_ms(flashinfer_call, warmup=warmup, rep=rep)
    speedup = flashinfer_ms / cutedsl_ms

    print(
        f"CuTeDSL GDN vs FlashInfer: T={T} HV={HV} input={input_mode} | "
        f"{_input_stats(q, k, v, g, beta)} | "
        f"rel_err={rel_err:.6e}, state_rel_err={state_rel_err}, "
        f"atrex_absmax={out.float().abs().max().item():.4f} "
        f"flashinfer_absmax={ref.float().abs().max().item():.4f}"
    )
    if final_state is not None and ref_state is not None:
        print(
            f"  state_absmax: atrex={final_state.float().abs().max().item():.4f} "
            f"flashinfer={ref_state.float().abs().max().item():.4f}"
        )
    print(
        f"  cutedsl={cutedsl_ms:.4f}ms, flashinfer={flashinfer_ms:.4f}ms, "
        f"speedup={speedup:.4f}x"
    )

    if strict:
        assert rel_err < 1e-2
        if state_rel_err is not None:
            assert state_rel_err < 1e-2
        if min_speedup is not None:
            assert speedup >= min_speedup
        if max_atrex_ms is not None:
            assert cutedsl_ms <= max_atrex_ms
    return {
        "rel_err": rel_err,
        "state_rel_err": state_rel_err,
        "cutedsl_ms": cutedsl_ms,
        "flashinfer_ms": flashinfer_ms,
        "speedup": speedup,
    }


def test_chunk_gdn_fwd_cutedsl_prewarm_public_api():
    _requires_sm120()
    import atrex
    from atrex.src.nvidia.chunk_gdn.sm120 import (
        custom_compile_cache,
        delta_rule,
    )

    delta_rule._DELTA_RULE_COMPILED_CACHE.clear()
    custom_compile_cache._in_mem_compile_cache.clear()
    atrex.chunk_gdn_fwd_cutedsl_prewarm_buckets(
        H=16,
        HV=32,
        include_state_checkpoints=True,
    )
    cached_cu_dtypes = {
        key[-1] for key in delta_rule._DELTA_RULE_COMPILED_CACHE
    }
    cached_checkpoint_modes = {
        key[3] for key in delta_rule._DELTA_RULE_COMPILED_CACHE
    }
    assert cached_cu_dtypes == {torch.int32, torch.int64}
    assert cached_checkpoint_modes == {False, True}
    assert len(delta_rule._DELTA_RULE_COMPILED_CACHE) == 16
    assert len(custom_compile_cache._in_mem_compile_cache) == 16


def test_fused_qk_l2norm_reuses_single_row_precompile_across_lengths():
    _requires_sm120()
    from atrex.src.nvidia.chunk_gdn.sm120.fused_qk_l2norm import (
        _compiled_cache,
        fused_qk_l2_normalize_bf16,
        prewarm_fused_qk_l2_normalize_bf16,
    )

    def eager_l2norm(x):
        xf = x.float()
        return (
            xf * torch.rsqrt((xf * xf).sum(dim=-1, keepdim=True) + 1e-6)
        ).to(x.dtype)

    _compiled_cache.clear()
    prewarm_fused_qk_l2_normalize_bf16(device="cuda")
    assert len(_compiled_cache) == 1
    executable_id = id(next(iter(_compiled_cache.values())))

    for rows in (24, 8192, 128, 1):
        torch.manual_seed(rows)
        q = torch.randn((rows, 128), device="cuda", dtype=torch.bfloat16)
        k = torch.randn_like(q)
        q_norm, k_norm = fused_qk_l2_normalize_bf16(q, k)
        torch.testing.assert_close(
            q_norm, eager_l2norm(q), rtol=1e-2, atol=2e-3
        )
        torch.testing.assert_close(
            k_norm, eager_l2norm(k), rtol=1e-2, atol=2e-3
        )
        assert len(_compiled_cache) == 1
        assert id(next(iter(_compiled_cache.values()))) == executable_id


def test_chunk_gdn_compile_cache_distinguishes_cu_seqlens_dtype():
    _requires_sm120()
    import atrex
    from atrex.src.nvidia.chunk_gdn.sm120 import (
        custom_compile_cache,
        delta_rule,
    )

    q, k, v, g, beta = _make_inputs(T=64)
    cu_cpu = torch.tensor([0, 64], dtype=torch.int64)
    ctx = atrex.chunk_gdn_fwd_cutedsl_build(
        B=1, H=16, HV=32, K=128, V=128, output_final_state=True
    )

    delta_rule._DELTA_RULE_COMPILED_CACHE.clear()
    custom_compile_cache._in_mem_compile_cache.clear()
    results = []
    for dtype in (torch.int32, torch.int64):
        cu_seqlens = cu_cpu.to(device=q.device, dtype=dtype)
        results.append(
            atrex.chunk_gdn_fwd_cutedsl(
                ctx,
                q,
                k,
                v,
                g,
                beta,
                output_final_state=True,
                cu_seqlens=cu_seqlens,
                cu_seqlens_cpu=cu_cpu,
            )
        )

    torch.cuda.synchronize()
    torch.testing.assert_close(results[0][0], results[1][0])
    torch.testing.assert_close(results[0][1], results[1][1])
    cached_cu_dtypes = {
        key[-1] for key in delta_rule._DELTA_RULE_COMPILED_CACHE
    }
    assert cached_cu_dtypes == {torch.int32, torch.int64}
    assert len(custom_compile_cache._in_mem_compile_cache) == 2


def test_chunk_gdn_public_checkpoint_stream_and_prefix_accuracy():
    _requires_sm120()
    import atrex

    B, H, HV, K, V = 1, 16, 32, 128, 128
    segments = (64, 128)
    T = sum(segments)
    interval = 64
    scale = 1.0 / math.sqrt(K)
    q_source, k_source, v_source, g_source, beta_source = (
        _make_directional_inputs(B, T, H, HV, K, V)
    )
    torch.cuda.synchronize()

    q = torch.empty_like(q_source)
    k = torch.empty_like(k_source)
    v = torch.empty_like(v_source)
    g = torch.empty_like(g_source)
    beta = torch.empty_like(beta_source)
    cu_cpu = torch.tensor([0, segments[0], T], dtype=torch.int64)
    cu = cu_cpu.to(device="cuda", dtype=torch.int32)
    expected_checkpoint_starts = (0, 1, 3)
    state_checkpoints = torch.full(
        (expected_checkpoint_starts[-1], HV, V, K),
        float("nan"),
        device="cuda",
        dtype=torch.float32,
    )
    # This is the request-level checkpoint plan used both by the ATREX writer
    # and by the caller below when it consumes state_checkpoints.
    checkpoint_cu_starts = torch.tensor(
        expected_checkpoint_starts, device="cuda", dtype=torch.int64
    )

    eligibility_kwargs = {
        "scale": scale,
        "output_final_state": True,
        "cu_seqlens": cu,
        "cu_seqlens_cpu": cu_cpu,
        "use_qk_l2norm_in_kernel": True,
        "state_checkpoints": state_checkpoints,
        "checkpoint_cu_starts": checkpoint_cu_starts,
        "checkpoint_every_n_tokens": interval,
    }
    assert atrex.can_use_chunk_gdn_fwd_cutedsl(
        q_source,
        k_source,
        v_source,
        g_source,
        beta_source,
        **eligibility_kwargs,
    )
    assert not atrex.can_use_chunk_gdn_fwd_cutedsl(
        q_source,
        k_source,
        v_source,
        g_source,
        beta_source,
        **{
            **eligibility_kwargs,
            "state_checkpoints": state_checkpoints[:-1],
        },
    )
    assert not atrex.can_use_chunk_gdn_fwd_cutedsl(
        q_source,
        k_source,
        v_source,
        g_source,
        beta_source,
        **{
            **eligibility_kwargs,
            "checkpoint_cu_starts": checkpoint_cu_starts.to(torch.int32),
        },
    )

    ctx = atrex.chunk_gdn_fwd_cutedsl_build(
        B=B, H=H, HV=HV, K=K, V=V, output_final_state=True
    )
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        torch.cuda._sleep(20_000_000)
        q.copy_(q_source)
        k.copy_(k_source)
        v.copy_(v_source)
        g.copy_(g_source)
        beta.copy_(beta_source)
        output, final_state = atrex.chunk_gdn_fwd_cutedsl(
            ctx,
            q,
            k,
            v,
            g,
            beta,
            scale=scale,
            output_final_state=True,
            cu_seqlens=cu,
            cu_seqlens_cpu=cu_cpu,
            state_checkpoints=state_checkpoints,
            checkpoint_cu_starts=checkpoint_cu_starts,
            checkpoint_every_n_tokens=interval,
        )

    stream.synchronize()
    assert torch.isfinite(output).all()
    assert torch.isfinite(final_state).all()
    assert torch.isfinite(state_checkpoints).all()
    torch.testing.assert_close(
        checkpoint_cu_starts,
        torch.tensor(
            expected_checkpoint_starts, device="cuda", dtype=torch.int64
        ),
    )

    seq_start = 0
    for seq_idx, seq_len in enumerate(segments):
        checkpoint_count = seq_len // interval
        for local_checkpoint_idx in range(checkpoint_count):
            prefix_len = (local_checkpoint_idx + 1) * interval
            prefix_slice = slice(seq_start, seq_start + prefix_len)
            prefix_cu_cpu = torch.tensor([0, prefix_len], dtype=torch.int64)
            prefix_cu = prefix_cu_cpu.to(device="cuda", dtype=torch.int32)
            _, prefix_final_state = atrex.chunk_gdn_fwd_cutedsl(
                ctx,
                q_source[:, prefix_slice].contiguous(),
                k_source[:, prefix_slice].contiguous(),
                v_source[:, prefix_slice].contiguous(),
                g_source[:, prefix_slice].contiguous(),
                beta_source[:, prefix_slice].contiguous(),
                scale=scale,
                output_final_state=True,
                cu_seqlens=prefix_cu,
                cu_seqlens_cpu=prefix_cu_cpu,
            )
            checkpoint_idx = (
                int(checkpoint_cu_starts[seq_idx].item())
                + local_checkpoint_idx
            )
            torch.testing.assert_close(
                state_checkpoints[checkpoint_idx],
                prefix_final_state[0],
                atol=1e-5,
                rtol=0,
            )
        seq_start += seq_len


def test_chunk_gdn_checkpoint_device_bounds_guard():
    _requires_sm120()
    import atrex

    B, T, H, HV, K, V = 1, 64, 16, 32, 128, 128
    q, k, v, g, beta = _make_inputs(B, T, H, HV, K, V)
    cu_cpu = torch.tensor([0, T], dtype=torch.int64)
    cu = cu_cpu.to(device=q.device, dtype=torch.int32)
    state_checkpoints = torch.full(
        (1, HV, V, K),
        float("nan"),
        device=q.device,
        dtype=torch.float32,
    )
    # The contents are deliberately corrupt. The public API trusts the
    # request-level plan, but the kernel must still contain any bad write index
    # instead of writing outside state_checkpoints.
    corrupt_checkpoint_cu_starts = torch.tensor(
        [2, 3], device=q.device, dtype=torch.int64
    )
    ctx = atrex.chunk_gdn_fwd_cutedsl_build(
        B=B, H=H, HV=HV, K=K, V=V, output_final_state=True
    )

    output, final_state = atrex.chunk_gdn_fwd_cutedsl(
        ctx,
        q,
        k,
        v,
        g,
        beta,
        output_final_state=True,
        cu_seqlens=cu,
        cu_seqlens_cpu=cu_cpu,
        state_checkpoints=state_checkpoints,
        checkpoint_cu_starts=corrupt_checkpoint_cu_starts,
        checkpoint_every_n_tokens=64,
    )
    torch.cuda.synchronize()

    assert torch.isfinite(output).all()
    assert torch.isfinite(final_state).all()
    assert torch.isnan(state_checkpoints).all()


def test_chunk_gdn_checkpoint_enabled_performance_upper_bound():
    _requires_sm120()
    import atrex

    B, T, H, HV, K, V = 1, 6144, 16, 32, 128, 128
    interval = 64
    q, k, v, g, beta = _make_inputs(B, T, H, HV, K, V)
    cu_cpu = torch.tensor([0, T], dtype=torch.int64)
    cu = cu_cpu.to(device=q.device, dtype=torch.int32)
    checkpoint_cu_starts = torch.tensor(
        [0, T // interval], device=q.device, dtype=torch.int64
    )
    state_checkpoints = torch.empty(
        (T // interval, HV, V, K),
        device=q.device,
        dtype=torch.float32,
    )
    scale = 1.0 / math.sqrt(K)
    ctx = atrex.chunk_gdn_fwd_cutedsl_build(
        B=B,
        H=H,
        HV=HV,
        K=K,
        V=V,
        seq_len=T,
        output_final_state=True,
        scale=scale,
    )

    def checkpoint_call():
        return atrex.chunk_gdn_fwd_cutedsl(
            ctx,
            q,
            k,
            v,
            g,
            beta,
            scale,
            output_final_state=True,
            cu_seqlens=cu,
            cu_seqlens_cpu=cu_cpu,
            state_checkpoints=state_checkpoints,
            checkpoint_cu_starts=checkpoint_cu_starts,
            checkpoint_every_n_tokens=interval,
        )

    output, final_state = checkpoint_call()
    torch.cuda.synchronize()
    assert torch.isfinite(output).all()
    assert torch.isfinite(final_state).all()
    assert torch.isfinite(state_checkpoints).all()

    checkpoint_ms = _bench_ms(checkpoint_call, warmup=5, rep=20)
    print(
        "ATREX checkpoint-enabled GDN: "
        f"T={T} HV={HV} interval={interval} {checkpoint_ms:.4f}ms "
        f"(limit={_CHECKPOINT_MAX_ATREX_MS:.2f}ms)"
    )
    assert checkpoint_ms <= _CHECKPOINT_MAX_ATREX_MS


def _profile_chunk_gdn_workload():
    # CuTeDSL may inspect argv while importing; hide this test-only mode flag.
    sys.argv = [sys.argv[0]]
    import atrex

    q, k, v, g, beta = _make_inputs(T=128, H=16, HV=32, K=128, V=128)
    scale = 1.0 / math.sqrt(128)
    context = atrex.chunk_gdn_fwd_cutedsl_build(
        B=1,
        H=16,
        HV=32,
        K=128,
        V=128,
        output_final_state=True,
        scale=scale,
    )
    cu_cpu = torch.tensor([0, 128], dtype=torch.int64)
    cu = cu_cpu.to(device=q.device, dtype=torch.int32)
    checkpoint_cu_starts = torch.tensor(
        [0, 2], device=q.device, dtype=torch.int64
    )
    state_checkpoints = torch.empty(
        (2, 32, 128, 128), device=q.device, dtype=torch.float32
    )

    result = None
    for _ in range(2):
        result = atrex.chunk_gdn_fwd_cutedsl(
            context,
            q,
            k,
            v,
            g,
            beta,
            scale,
            output_final_state=True,
            cu_seqlens=cu,
            cu_seqlens_cpu=cu_cpu,
            state_checkpoints=state_checkpoints,
            checkpoint_cu_starts=checkpoint_cu_starts,
            checkpoint_every_n_tokens=64,
        )
    torch.cuda.synchronize()
    assert result is not None
    output, final_state = result
    assert output.shape == (1, 128, 32, 128)
    assert final_state.shape == (1, 32, 128, 128)
    assert torch.isfinite(state_checkpoints).all()


def test_chunk_gdn_profiler_kernel_names_have_atrex_prefix():
    _requires_sm120()
    nsys = shutil.which("nsys")
    assert nsys is not None, "Nsight Systems is required for the SM120 gate"
    workload = Path(__file__)
    with tempfile.TemporaryDirectory(prefix="atrex_gdn_nsys_") as directory:
        report_base = Path(directory) / "chunk_gdn"
        subprocess.run(
            [
                nsys,
                "profile",
                "--trace=cuda",
                "--sample=none",
                "--cpuctxsw=none",
                "--force-overwrite=true",
                "--output",
                str(report_base),
                sys.executable,
                str(workload),
                _PROFILE_WORKLOAD_FLAG,
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        stats = subprocess.run(
            [
                nsys,
                "stats",
                "--report",
                "cuda_gpu_kern_sum",
                "--format",
                "csv",
                str(report_base.with_suffix(".nsys-rep")),
            ],
            check=True,
            text=True,
            capture_output=True,
        )

    stats_lines = stats.stdout.splitlines()
    header_index = next((
        index
        for index, line in enumerate(stats_lines)
        if "Time (%)" in line and "Name" in line
    ), None)
    assert header_index is not None, (
        "nsys stats output did not contain the expected CUDA kernel header: "
        f"{stats.stdout}"
    )
    rows = csv.DictReader(io.StringIO("\n".join(stats_lines[header_index:])))
    kernel_names = sorted(
        {row["Name"].strip() for row in rows if row.get("Name")}
    )
    gdn_kernel_names = [
        name
        for name in kernel_names
        if "gdn" in name.lower() or "delta_rule" in name.lower()
    ]
    assert gdn_kernel_names, (
        "Nsight Systems did not observe the CuTeDSL GDN kernels; "
        f"observed CUDA events: {kernel_names}"
    )
    assert all(name.startswith("atrex_") for name in gdn_kernel_names), (
        "all ATREX GDN kernels must have profiler-visible atrex_ prefixes; "
        f"observed: {gdn_kernel_names}"
    )
    print({"atrex_gdn_kernel_names": gdn_kernel_names})


@pytest.mark.parametrize("HV", [32, 48, 64])
@pytest.mark.parametrize("T", [1280, 6145, 16384])
def test_chunk_gdn_fwd_cutedsl_two_output_no_state_matches_flashinfer(T, HV):
    _requires_sm120()
    import atrex

    B, H, K, V = 1, 16, 128, 128
    q, k, v, g, beta = _make_inputs(B, T, H, HV, K, V)
    scale = 1.0 / math.sqrt(K)
    assert atrex.can_use_chunk_gdn_fwd_cutedsl(
        q, k, v, g, beta, scale,
        output_final_state=False,
        use_qk_l2norm_in_kernel=True,
    )

    ctx = atrex.chunk_gdn_fwd_cutedsl_build(B=B, H=H, HV=HV, K=K, V=V)
    out, final_state = atrex.chunk_gdn_fwd_cutedsl(ctx, q, k, v, g, beta, scale)
    ref, ref_state = _flashinfer_chunk_gdn(q, k, v, g, beta, scale)
    torch.cuda.synchronize()

    assert out.shape == (B, T, HV, V)
    assert out.dtype == torch.bfloat16
    assert final_state is None
    assert ref_state is None
    assert torch.isfinite(out.float()).all()

    assert _rel_err(out, ref) < 1e-2

@pytest.mark.parametrize(
    ("T", "HV"),
    [
        (1280, 32),
        (1280, 48),
        (1280, 64),
        (6173, 48),
        (16384, 32),
        (16384, 48),
        (16384, 64),
    ],
)
def test_chunk_gdn_fwd_cutedsl_final_state_matches_flashinfer(T, HV):
    _requires_sm120()
    import atrex

    B, H, K, V = 1, 16, 128, 128
    q, k, v, g, beta = _make_inputs(B, T, H, HV, K, V)
    scale = 1.0 / math.sqrt(K)
    assert atrex.can_use_chunk_gdn_fwd_cutedsl(
        q, k, v, g, beta, scale,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )

    ctx = atrex.chunk_gdn_fwd_cutedsl_build(B=B, H=H, HV=HV, K=K, V=V)
    out, final_state = atrex.chunk_gdn_fwd_cutedsl(
        ctx, q, k, v, g, beta, scale,
        output_final_state=True,
    )
    ref, ref_state = _flashinfer_chunk_gdn(
        q, k, v, g, beta, scale, output_final_state=True
    )
    torch.cuda.synchronize()

    assert out.shape == (B, T, HV, V)
    assert out.dtype == torch.bfloat16
    assert final_state.shape == (B, HV, V, K)
    assert final_state.dtype == torch.float32
    assert torch.isfinite(final_state).all()
    assert _rel_err(out, ref) < 1e-2
    assert _rel_err(final_state, ref_state) < 1e-2


@pytest.mark.parametrize("HV", [32, 48, 64])
def test_chunk_gdn_fwd_cutedsl_reuses_ctx_for_multiple_sequence_lengths(HV):
    _requires_sm120()
    import atrex

    B, H, K, V = 1, 16, 128, 128
    ctx = atrex.chunk_gdn_fwd_cutedsl_build(B=B, H=H, HV=HV, K=K, V=V)
    scale = 1.0 / math.sqrt(K)

    for T in (128, 256):
        q, k, v, g, beta = _make_inputs(B, T, H, HV, K, V)
        assert atrex.can_use_chunk_gdn_fwd_cutedsl(
            q, k, v, g, beta, scale,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
        )
        out, final_state = atrex.chunk_gdn_fwd_cutedsl(
            ctx, q, k, v, g, beta, scale,
            output_final_state=True,
        )
        ref, ref_state = _flashinfer_chunk_gdn(
            q, k, v, g, beta, scale,
            output_final_state=True,
        )
        torch.cuda.synchronize()

        assert out.shape == (B, T, HV, V)
        assert final_state.shape == (B, HV, V, K)
        assert _rel_err(out, ref) < 1e-2
        assert _rel_err(final_state, ref_state) < 1e-2


@pytest.mark.parametrize("HV", [32, 48, 64])
def test_chunk_gdn_fwd_cutedsl_precision_and_perf_vs_flashinfer(HV):
    _requires_sm120()
    _compare_cutedsl_vs_flashinfer(
        HV=HV,
        max_atrex_ms=_CANONICAL_MAX_ATREX_MS[HV],
    )


def test_chunk_gdn_fwd_cutedsl_rejects_invalid_state_and_options():
    _requires_sm120()
    import atrex

    B, T, H, HV, K, V = 1, 128, 16, 32, 128, 128
    q, k, v, g, beta = _make_inputs(B, T, H, HV, K, V)
    cu_cpu = torch.tensor([0, T // 2, T], dtype=torch.int32)
    cu = cu_cpu.to(q.device)
    valid_state = torch.zeros(
        (2, HV, V, K), device=q.device, dtype=torch.float32,
    )
    options = {
        "output_final_state": True,
        "use_qk_l2norm_in_kernel": True,
        "cu_seqlens": cu,
        "cu_seqlens_cpu": cu_cpu,
    }

    assert atrex.can_use_chunk_gdn_fwd_cutedsl(
        q, k, v, g, beta, initial_state=valid_state, **options,
    )
    assert not atrex.can_use_chunk_gdn_fwd_cutedsl(
        q, k, v, g, beta, initial_state=valid_state[:1], **options,
    )
    assert not atrex.can_use_chunk_gdn_fwd_cutedsl(
        q, k, v, g, beta, initial_state=valid_state.to(torch.bfloat16),
        **options,
    )
    assert not atrex.can_use_chunk_gdn_fwd_cutedsl(
        q, k, v, g, beta, initial_state=valid_state,
        head_first=True, **options,
    )
    assert not atrex.can_use_chunk_gdn_fwd_cutedsl(
        q, k, v.expand(2, -1, -1, -1), g, beta,
        initial_state=valid_state, **options,
    )


@pytest.mark.parametrize("HV", [32, 48])
def test_chunk_gdn_fwd_cutedsl_packed_cu_seqlens_matches_flashinfer(HV):
    _requires_sm120()
    import atrex

    B, H, K, V = 1, 16, 128, 128
    segments = (6103, 6106)
    T = sum(segments)
    q, k, v, g, beta = _make_directional_inputs(B, T, H, HV, K, V)
    scale = 1.0 / math.sqrt(K)
    cu_cpu = torch.tensor(
        [0, segments[0], T], dtype=torch.int32, device="cpu")
    cu = cu_cpu.to(device=q.device)

    assert atrex.can_use_chunk_gdn_fwd_cutedsl(
        q, k, v, g, beta, scale,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
        cu_seqlens=cu,
        cu_seqlens_cpu=cu_cpu,
        allow_padding=True,
    )

    ctx = atrex.chunk_gdn_fwd_cutedsl_build(B=B, H=H, HV=HV, K=K, V=V)
    out, final_state = atrex.chunk_gdn_fwd_cutedsl(
        ctx, q, k, v, g, beta, scale,
        output_final_state=True,
        cu_seqlens=cu,
        cu_seqlens_cpu=cu_cpu,
    )

    ref, ref_state = _flashinfer_chunk_gdn(
        q, k, v, g, beta, scale,
        cu_seqlens=cu,
        output_final_state=True,
    )
    torch.cuda.synchronize()

    assert out.shape == (B, T, HV, V)
    assert final_state.shape == (len(segments), HV, V, K)
    assert final_state.shape == ref_state.shape
    assert _rel_err(out, ref) < 1.5e-2
    assert _rel_err(final_state, ref_state) < 1e-2


def test_chunk_gdn_fwd_cutedsl_packed_ragged_matches_flashinfer():
    _requires_sm120()
    import atrex

    B, H, HV, K, V = 1, 16, 32, 128, 128
    segments = (1025, 1025)
    T = sum(segments)
    q, k, v, g, beta = _make_directional_inputs(B, T, H, HV, K, V)
    scale = 1.0 / math.sqrt(K)
    cu_cpu = torch.tensor([0, segments[0], T], dtype=torch.int32)
    cu = cu_cpu.to(q.device)
    ctx = atrex.chunk_gdn_fwd_cutedsl_build(B=B, H=H, HV=HV, K=K, V=V)
    out, final_state = atrex.chunk_gdn_fwd_cutedsl(
        ctx, q, k, v, g, beta, scale,
        output_final_state=True,
        cu_seqlens=cu,
        cu_seqlens_cpu=cu_cpu,
    )
    ref, ref_state = _flashinfer_chunk_gdn(
        q, k, v, g, beta, scale,
        cu_seqlens=cu,
        output_final_state=True,
    )
    torch.cuda.synchronize()

    assert torch.isfinite(out.float()).all()
    for end in (segments[0], T):
        # Guard the final partial BT=32 chunk of each sequence.
        assert _rel_err(out[:, end - 1:end], ref[:, end - 1:end]) < 1e-2
    assert _rel_err(out, ref) < 1e-2
    assert _rel_err(final_state, ref_state) < 1e-2


def test_chunk_gdn_fwd_cutedsl_single_ragged_init_state_matches_flashinfer():
    _requires_sm120()
    import atrex

    B, T, H, HV, K, V = 1, 1025, 16, 32, 128, 128
    q, k, v, g, beta = _make_directional_inputs(B, T, H, HV, K, V)
    scale = 1.0 / math.sqrt(K)
    cu_cpu = torch.tensor([0, T], dtype=torch.int32)
    cu = cu_cpu.to(q.device)
    initial_state = (
        0.01 * torch.randn(B, HV, V, K, device=q.device, dtype=torch.float32)
    ).contiguous()
    ctx = atrex.chunk_gdn_fwd_cutedsl_build(B=B, H=H, HV=HV, K=K, V=V)
    out, final_state = atrex.chunk_gdn_fwd_cutedsl(
        ctx, q, k, v, g, beta, scale,
        output_final_state=True,
        cu_seqlens=cu,
        cu_seqlens_cpu=cu_cpu,
        initial_state=initial_state,
    )
    ref, ref_state = _flashinfer_chunk_gdn(
        q, k, v, g, beta, scale,
        initial_state=initial_state,
        cu_seqlens=cu,
        output_final_state=True,
    )
    torch.cuda.synchronize()

    assert _rel_err(out[:, -1:], ref[:, -1:]) < 1e-2
    assert _rel_err(out, ref) < 1e-2
    assert _rel_err(final_state, ref_state) < 1e-2


@pytest.mark.parametrize("gate_order", ["fast_slow", "slow_fast"])
def test_chunk_gdn_fwd_cutedsl_packed_directional_gates_match_flashinfer(
    gate_order,
):
    _requires_sm120()
    import atrex

    B, H, HV, K, V = 1, 16, 32, 128, 128
    segments = (4096, 4096)
    T = sum(segments)
    q, k, v, _, beta = _make_directional_inputs(B, T, H, HV, K, V)
    scale = 1.0 / math.sqrt(K)
    cu_cpu = torch.tensor([0, segments[0], T], dtype=torch.int32)
    cu = cu_cpu.to(q.device)
    fast = torch.full(
        (B, segments[0], HV), -1.0, dtype=torch.bfloat16, device=q.device)
    slow = torch.full_like(fast, -1e-4)
    gates = (fast, slow) if gate_order == "fast_slow" else (slow, fast)
    g = torch.cat(gates, dim=1)
    ctx = atrex.chunk_gdn_fwd_cutedsl_build(B=B, H=H, HV=HV, K=K, V=V)
    out, final_state = atrex.chunk_gdn_fwd_cutedsl(
        ctx, q, k, v, g, beta, scale,
        output_final_state=True,
        cu_seqlens=cu,
        cu_seqlens_cpu=cu_cpu,
    )
    ref, ref_state = _flashinfer_chunk_gdn(
        q, k, v, g, beta, scale,
        cu_seqlens=cu,
        output_final_state=True,
    )
    torch.cuda.synchronize()

    assert _rel_err(out, ref) < 2e-2
    # This directional slow-decay stress shape amplifies the BF16 q/k
    # normalization-order difference between ATREX and FlashInfer.
    assert _rel_err(final_state, ref_state) < 2e-2


def test_chunk_gdn_fwd_cutedsl_reused_ctx_observes_gate_mutation():
    """CUDA-graph-style buffer replay may change a live tensor's contents."""
    _requires_sm120()
    import atrex

    B, T, H, HV, K, V = 1, 4096, 16, 32, 128, 128
    q, k, v, _, beta = _make_directional_inputs(B, T, H, HV, K, V)
    scale = 1.0 / math.sqrt(K)
    cu_cpu = torch.tensor([0, T], dtype=torch.int32)
    cu = cu_cpu.to(q.device)
    g = torch.full((B, T, HV), -1.0, dtype=torch.bfloat16, device=q.device)
    ctx = atrex.chunk_gdn_fwd_cutedsl_build(B=B, H=H, HV=HV, K=K, V=V)

    # Reuse the same context and input allocation with different gate values.
    atrex.chunk_gdn_fwd_cutedsl(
        ctx, q, k, v, g, beta, scale,
        output_final_state=True,
        cu_seqlens=cu,
        cu_seqlens_cpu=cu_cpu,
    )
    g.fill_(-1e-4)
    out, final_state = atrex.chunk_gdn_fwd_cutedsl(
        ctx, q, k, v, g, beta, scale,
        output_final_state=True,
        cu_seqlens=cu,
        cu_seqlens_cpu=cu_cpu,
    )
    ref, ref_state = _flashinfer_chunk_gdn(
        q, k, v, g, beta, scale,
        cu_seqlens=cu,
        output_final_state=True,
    )
    torch.cuda.synchronize()

    # This directional slow-decay stress shape amplifies the BF16 q/k
    # normalization-order difference between ATREX and FlashInfer.
    assert _rel_err(out, ref) < 2e-2
    assert _rel_err(final_state, ref_state) < 2e-2


@pytest.mark.parametrize("HV", [32, 48])
@pytest.mark.parametrize("segments", [(6106,), (3000, 3106)])
def test_delta_rule_prefill_dsl_sm120_split_v_matches_no_split(HV, segments):
    _requires_sm120()
    from atrex.src.nvidia.chunk_gdn.sm120 import (
        delta_rule_prefill_dsl_sm120,
    )

    assert delta_rule_prefill_dsl_sm120 is not None

    B, H, K, V = 1, 16, 128, 128
    T = sum(segments)
    q, k, v, g, beta = _make_directional_inputs(B, T, H, HV, K, V)
    scale = 1.0 / math.sqrt(K)
    q_flat = F.normalize(q.squeeze(0).float(), p=2.0, dim=-1).to(q.dtype).contiguous()
    k_flat = F.normalize(k.squeeze(0).float(), p=2.0, dim=-1).to(k.dtype).contiguous()
    v_flat = v.squeeze(0).contiguous()
    gate = torch.exp(g.squeeze(0).float()).contiguous()
    beta_flat = beta.squeeze(0).float().contiguous()
    cu_values = [0]
    for seg in segments:
        cu_values.append(cu_values[-1] + seg)
    cu = torch.tensor(cu_values, dtype=torch.int64, device=q.device)

    out_ref = torch.empty((T, HV, V), dtype=v.dtype, device=v.device)
    state_ref = torch.empty(
        (len(segments), HV, V, K), dtype=torch.float32, device=v.device)
    out_split = torch.empty_like(out_ref)
    state_split = torch.empty_like(state_ref)

    delta_rule_prefill_dsl_sm120(
        out_ref,
        state_ref,
        q_flat,
        k_flat,
        v_flat,
        None,
        gate,
        beta_flat,
        cu,
        scale,
        split_v_parts=1,
    )
    delta_rule_prefill_dsl_sm120(
        out_split,
        state_split,
        q_flat,
        k_flat,
        v_flat,
        None,
        gate,
        beta_flat,
        cu,
        scale,
        split_v_parts=2,
    )
    torch.cuda.synchronize()

    assert _rel_err(out_split, out_ref) < 2e-3
    assert _rel_err(state_split, state_ref) < 1e-4


@pytest.mark.parametrize("HV", [32, 48])
def test_delta_rule_prefill_dsl_sm120_head_major_gate_matches_row_major(HV):
    _requires_sm120()
    from atrex.src.nvidia.chunk_gdn.sm120 import (
        delta_rule_prefill_dsl_sm120,
    )

    assert delta_rule_prefill_dsl_sm120 is not None

    B, H, K, V = 1, 16, 128, 128
    T = 1024
    q, k, v, g, beta = _make_directional_inputs(B, T, H, HV, K, V)
    scale = 1.0 / math.sqrt(K)
    q_flat = F.normalize(q.squeeze(0).float(), p=2.0, dim=-1).to(q.dtype).contiguous()
    k_flat = F.normalize(k.squeeze(0).float(), p=2.0, dim=-1).to(k.dtype).contiguous()
    v_flat = v.squeeze(0).contiguous()
    gate_row = torch.exp(g.squeeze(0).float()).contiguous()
    beta_row = beta.squeeze(0).float().contiguous()
    gate_storage = gate_row.transpose(0, 1).contiguous()
    beta_storage = beta_row.transpose(0, 1).contiguous()
    gate_head_major = torch.as_strided(gate_storage, (T, HV), (1, T))
    beta_head_major = torch.as_strided(beta_storage, (T, HV), (1, T))
    cu = torch.tensor([0, T], dtype=torch.int64, device=q.device)

    out_row = torch.empty((T, HV, V), dtype=v.dtype, device=v.device)
    state_row = torch.empty((1, HV, V, K), dtype=torch.float32, device=v.device)
    out_head_major = torch.empty_like(out_row)
    state_head_major = torch.empty_like(state_row)

    delta_rule_prefill_dsl_sm120(
        out_row,
        state_row,
        q_flat,
        k_flat,
        v_flat,
        None,
        gate_row,
        beta_row,
        cu,
        scale,
        split_v_parts=2,
    )
    delta_rule_prefill_dsl_sm120(
        out_head_major,
        state_head_major,
        q_flat,
        k_flat,
        v_flat,
        None,
        gate_head_major,
        beta_head_major,
        cu,
        scale,
        split_v_parts=2,
    )
    torch.cuda.synchronize()

    assert _rel_err(out_head_major, out_row) == 0.0
    assert _rel_err(state_head_major, state_row) == 0.0


@pytest.mark.parametrize("HV", [32, 48])
@pytest.mark.parametrize("checkpoint_every_n_tokens", [64, 128])
def test_delta_rule_prefill_dsl_sm120_checkpoints_match_prefix(HV, checkpoint_every_n_tokens):
    _requires_sm120()
    from atrex.src.nvidia.chunk_gdn.sm120 import (
        delta_rule_prefill_dsl_sm120,
    )

    assert delta_rule_prefill_dsl_sm120 is not None

    B, H, K, V = 1, 16, 128, 128
    segments = (256, 384)
    T = sum(segments)
    q, k, v, g, beta = _make_directional_inputs(B, T, H, HV, K, V)
    scale = 1.0 / math.sqrt(K)
    q_flat = F.normalize(q.squeeze(0).float(), p=2.0, dim=-1).to(q.dtype).contiguous()
    k_flat = F.normalize(k.squeeze(0).float(), p=2.0, dim=-1).to(k.dtype).contiguous()
    v_flat = v.squeeze(0).contiguous()
    gate = torch.exp(g.squeeze(0).float()).contiguous()
    beta_flat = beta.squeeze(0).float().contiguous()
    cu = torch.tensor([0, segments[0], T], dtype=torch.int64, device=q.device)

    ckpt_counts = [seg // checkpoint_every_n_tokens for seg in segments]
    checkpoint_cu_starts = torch.tensor(
        [0, ckpt_counts[0], sum(ckpt_counts)],
        dtype=torch.int64,
        device=q.device,
    )
    state_checkpoints = torch.full(
        (sum(ckpt_counts), HV, V, K),
        float("nan"),
        dtype=torch.float32,
        device=q.device,
    )
    out = torch.empty((T, HV, V), dtype=v.dtype, device=v.device)
    state = torch.empty((len(segments), HV, V, K), dtype=torch.float32, device=v.device)

    delta_rule_prefill_dsl_sm120(
        out,
        state,
        q_flat,
        k_flat,
        v_flat,
        None,
        gate,
        beta_flat,
        cu,
        scale,
        state_checkpoints=state_checkpoints,
        checkpoint_cu_starts=checkpoint_cu_starts,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
    )
    torch.cuda.synchronize()
    assert torch.isfinite(state_checkpoints).all()

    starts = [0, segments[0]]
    for seq_idx, seq_start in enumerate(starts):
        for local_ckpt_idx in range(ckpt_counts[seq_idx]):
            prefix_len = (local_ckpt_idx + 1) * checkpoint_every_n_tokens
            prefix_out = torch.empty(
                (prefix_len, HV, V), dtype=v.dtype, device=v.device)
            prefix_state = torch.empty(
                (1, HV, V, K), dtype=torch.float32, device=v.device)
            prefix_cu = torch.tensor(
                [0, prefix_len], dtype=torch.int64, device=q.device)

            sl = slice(seq_start, seq_start + prefix_len)
            delta_rule_prefill_dsl_sm120(
                prefix_out,
                prefix_state,
                q_flat[sl].contiguous(),
                k_flat[sl].contiguous(),
                v_flat[sl].contiguous(),
                None,
                gate[sl].contiguous(),
                beta_flat[sl].contiguous(),
                prefix_cu,
                scale,
            )
            torch.cuda.synchronize()

            global_ckpt_idx = int(checkpoint_cu_starts[seq_idx].item()) + local_ckpt_idx
            torch.testing.assert_close(
                state_checkpoints[global_ckpt_idx],
                prefix_state[0],
                atol=1e-5,
                rtol=0,
            )


@pytest.mark.parametrize("HV", [32, 48, 64])
def test_chunk_gdn_fwd_cutedsl_tail_routing_uses_direct_only(HV):
    _requires_sm120()
    import atrex

    B, T, H, K, V = 1, 6103, 16, 128, 128
    q, k, v, g, beta = _make_directional_inputs(B, T, H, HV, K, V)
    scale = 1.0 / math.sqrt(K)

    direct_can_use = atrex.can_use_chunk_gdn_fwd_cutedsl(
        q,
        k,
        v,
        g,
        beta,
        scale,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )
    assert direct_can_use is (HV == 48)


def test_chunk_gdn_fwd_cutedsl_tail_directional():
    """Stress-test tail (T % 32 != 0) with realistic GDN-style inputs.

    Real-world q/k from conv1d have small l2 norm (~1-3) and a dominant
    shared direction across tokens. randn inputs are i.i.d. so many tail
    errors average out; directional inputs do not. If atrex's tail path leaks
    invalid values into the final_state accumulator, this test will fail
    (state_rel_err >> 1e-2) while the randn variant passes.
    """
    _requires_sm120()
    import atrex

    B, T, H, HV, K, V = 1, 6103, 16, 48, 128, 128
    q, k, v, g, beta = _make_directional_inputs(B, T, H, HV, K, V)
    scale = 1.0 / math.sqrt(K)
    assert atrex.can_use_chunk_gdn_fwd_cutedsl(
        q,
        k,
        v,
        g,
        beta,
        scale,
        output_final_state=True,
        use_qk_l2norm_in_kernel=True,
    )

    ctx = atrex.chunk_gdn_fwd_cutedsl_build(B=B, H=H, HV=HV, K=K, V=V)
    out, final_state = atrex.chunk_gdn_fwd_cutedsl(
        ctx, q, k, v, g, beta, scale,
        output_final_state=True,
    )
    ref, ref_state = _flashinfer_chunk_gdn(
        q, k, v, g, beta, scale, output_final_state=True
    )
    torch.cuda.synchronize()

    out_rel = _rel_err(out, ref)
    state_rel = _rel_err(final_state, ref_state)
    print(
        f"\n[directional T={T}] {_input_stats(q, k, v, g, beta)} | "
        f"out_rel={out_rel:.4e} state_rel={state_rel:.4e} | "
        f"out_absmax atrex={out.float().abs().max():.4f} "
        f"flashinfer={ref.float().abs().max():.4f} | "
        f"state_absmax atrex={final_state.float().abs().max():.4f} "
        f"flashinfer={ref_state.float().abs().max():.4f}"
    )

    assert torch.isfinite(out.float()).all()
    assert torch.isfinite(final_state).all()
    assert out_rel < 1e-2, f"output rel_err={out_rel:.4e} (>= 1e-2)"
    assert state_rel < 1e-2, f"state  rel_err={state_rel:.4e} (>= 1e-2)"


if __name__ == "__main__":
    if sys.argv[1:] == [_PROFILE_WORKLOAD_FLAG]:
        _require_sm120_or_exit()
        _profile_chunk_gdn_workload()
        raise SystemExit(0)

    parser = argparse.ArgumentParser(
        description="Compare SM120 CuTeDSL Chunk-GDN against FlashInfer on precision and performance."
    )
    parser.add_argument("--T", type=int, default=6144)
    parser.add_argument("--HV", type=int, default=32, choices=[32, 48, 64],
                        help="number of value heads")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=30)
    parser.add_argument("--min-speedup", type=float, default=None)
    parser.add_argument("--max-atrex-ms", type=float, default=None)
    parser.add_argument("--no-final-state", action="store_true")
    parser.add_argument(
        "--input-mode",
        choices=tuple(_INPUT_MAKERS.keys()),
        default="randn",
        help="randn=N(0,1) inputs; directional=shared-direction + small noise "
             "(stresses tail handling, mimics conv1d outputs)",
    )
    parser.add_argument(
        "--no-strict",
        action="store_true",
        help="Disable rel_err / speedup asserts (diagnostic mode)",
    )
    args = parser.parse_args()

    _require_sm120_or_exit()
    _compare_cutedsl_vs_flashinfer(
        T=args.T,
        HV=args.HV,
        warmup=args.warmup,
        rep=args.rep,
        min_speedup=args.min_speedup,
        max_atrex_ms=args.max_atrex_ms,
        output_final_state=not args.no_final_state,
        input_mode=args.input_mode,
        strict=not args.no_strict,
    )
