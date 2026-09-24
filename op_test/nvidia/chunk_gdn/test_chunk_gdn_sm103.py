"""NVIDIA SM103 AKA M64 Chunk-GDN public API and kernel tests."""

from __future__ import annotations

import argparse
import csv
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
_SCALE = 128**-0.5
_MAX_BALANCED_B3_MS = 1.0


def _requires_sm103() -> None:
    if not torch.cuda.is_available():
        pytest.skip("CUDA is not available")
    if torch.cuda.get_device_capability() != (10, 3):
        pytest.skip("SM103 CUDA device is required")


def _require_sm103_or_exit() -> None:
    if not torch.cuda.is_available():
        raise SystemExit("CUDA is not available")
    capability = torch.cuda.get_device_capability()
    if capability != (10, 3):
        raise SystemExit(f"SM103 CUDA device is required, got sm_{capability[0]}{capability[1]}")


def _make_inputs(
    lengths: tuple[int, ...],
    *,
    seed: int = 42,
    continuation: bool = False,
):
    torch.manual_seed(seed)
    total_t = sum(lengths)
    q = torch.randn((total_t, 4, 128), device="cuda", dtype=torch.float32)
    k = torch.randn_like(q)
    q = F.normalize(q, dim=-1).to(torch.bfloat16).contiguous().unsqueeze(0)
    k = F.normalize(k, dim=-1).to(torch.bfloat16).contiguous().unsqueeze(0)
    v = torch.randn(
        (1, total_t, 32, 128), device="cuda", dtype=torch.bfloat16
    )
    g = F.logsigmoid(torch.randn((1, total_t, 32), device="cuda"))
    beta = torch.sigmoid(torch.randn((1, total_t, 32), device="cuda"))
    initial_state = torch.zeros(
        (len(lengths), 32, 128, 128), device="cuda", dtype=torch.float32
    )
    if continuation:
        initial_state.normal_(mean=0.0, std=0.01)
    offsets = [0]
    for length in lengths:
        offsets.append(offsets[-1] + length)
    cu_cpu = torch.tensor(offsets, dtype=torch.int32)
    cu = cu_cpu.to(device="cuda")
    return q, k, v, g, beta, initial_state, cu, cu_cpu


@torch.inference_mode()
def _reference(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    initial_state: torch.Tensor,
    cu_cpu: torch.Tensor,
):
    q_flat = q.squeeze(0).float().repeat_interleave(8, dim=1)
    k_flat = k.squeeze(0).float().repeat_interleave(8, dim=1)
    v_flat = v.squeeze(0).float()
    alpha = torch.exp(g.squeeze(0).float())
    beta_flat = beta.squeeze(0).float()
    output = torch.empty_like(v_flat)
    final_state = torch.empty_like(initial_state)
    offsets = [int(value) for value in cu_cpu.tolist()]

    for seq_idx, (start, end) in enumerate(zip(offsets[:-1], offsets[1:])):
        # The public state layout is [H, V, K]; use [H, K, V] for Q/K matmuls.
        state = initial_state[seq_idx].transpose(-1, -2).float().clone()
        for token in range(start, end):
            state = alpha[token, :, None, None] * state
            predicted_v = torch.einsum("hk,hkv->hv", k_flat[token], state)
            delta_v = beta_flat[token, :, None] * (v_flat[token] - predicted_v)
            state = state + torch.einsum("hk,hv->hkv", k_flat[token], delta_v)
            output[token] = _SCALE * torch.einsum(
                "hk,hkv->hv", q_flat[token], state
            )
        final_state[seq_idx] = state.transpose(-1, -2)

    return output.to(torch.bfloat16).unsqueeze(0), final_state


def _build_context():
    import atrex

    return atrex.chunk_gdn_fwd_cutedsl_build(
        B=1,
        H=4,
        HV=32,
        K=128,
        V=128,
        output_final_state=True,
        scale=_SCALE,
    )


def _atrex_call(lengths: tuple[int, ...], *, seed=42, continuation=False):
    import atrex

    inputs = _make_inputs(lengths, seed=seed, continuation=continuation)
    q, k, v, g, beta, initial_state, cu, cu_cpu = inputs
    assert atrex.can_use_chunk_gdn_fwd_cutedsl(
        q,
        k,
        v,
        g,
        beta,
        scale=_SCALE,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu,
        cu_seqlens_cpu=cu_cpu,
        use_qk_l2norm_in_kernel=False,
    )
    output, final_state = atrex.chunk_gdn_fwd_cutedsl(
        _build_context(),
        q,
        k,
        v,
        g,
        beta,
        scale=_SCALE,
        output_final_state=True,
        cu_seqlens=cu,
        cu_seqlens_cpu=cu_cpu,
        qk_l2norm_already_applied=True,
        initial_state=initial_state,
    )
    return inputs, output, final_state


def _bench_ms(fn, *, warmup: int = 5, repetitions: int = 20) -> float:
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    samples.sort()
    return samples[len(samples) // 2]


def test_import_atrex_keeps_sm103_backend_lazy():
    code = (
        "import sys; import atrex; "
        "assert 'atrex.src.nvidia.chunk_gdn.sm103.gdn_prefill' not in sys.modules"
    )
    subprocess.run([sys.executable, "-c", code], check=True)


@pytest.mark.parametrize(
    ("lengths", "continuation"),
    [
        ((6,), False),
        ((61,), True),
        # B=1 and T % 128 == 64 exercises the synthetic paired-padding chunk.
        ((64,), False),
        ((128,), False),
        ((65, 63), True),
    ],
)
def test_chunk_gdn_sm103_matches_standalone_reference(lengths, continuation):
    _requires_sm103()
    inputs, output, final_state = _atrex_call(
        lengths, continuation=continuation
    )
    q, k, v, g, beta, initial_state, _, cu_cpu = inputs
    ref_output, ref_state = _reference(
        q, k, v, g, beta, initial_state, cu_cpu
    )
    torch.testing.assert_close(output, ref_output, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(final_state, ref_state, atol=5e-3, rtol=1e-3)


def test_chunk_gdn_sm103_accepts_int64_cu_seqlens():
    _requires_sm103()
    import atrex

    q, k, v, g, beta, initial_state, cu, cu_cpu = _make_inputs((17, 47))
    cu = cu.to(torch.int64)
    cu_cpu = cu_cpu.to(torch.int64)
    assert atrex.can_use_chunk_gdn_fwd_cutedsl(
        q,
        k,
        v,
        g,
        beta,
        scale=_SCALE,
        initial_state=initial_state,
        output_final_state=True,
        cu_seqlens=cu,
        cu_seqlens_cpu=cu_cpu,
        use_qk_l2norm_in_kernel=False,
    )
    output, final_state = atrex.chunk_gdn_fwd_cutedsl(
        _build_context(),
        q,
        k,
        v,
        g,
        beta,
        scale=_SCALE,
        output_final_state=True,
        cu_seqlens=cu,
        cu_seqlens_cpu=cu_cpu,
        qk_l2norm_already_applied=True,
        initial_state=initial_state,
    )
    assert output.shape == (1, 64, 32, 128)
    assert final_state.shape == (2, 32, 128, 128)


def test_chunk_gdn_sm103_rejects_unverified_options():
    _requires_sm103()
    import atrex

    q, k, v, g, beta, initial_state, cu, cu_cpu = _make_inputs((64,))
    common = {
        "scale": _SCALE,
        "initial_state": initial_state,
        "output_final_state": True,
        "cu_seqlens": cu,
        "cu_seqlens_cpu": cu_cpu,
    }
    assert atrex.can_use_chunk_gdn_fwd_cutedsl(
        q, k, v, g, beta, use_qk_l2norm_in_kernel=False, **common
    )
    assert not atrex.can_use_chunk_gdn_fwd_cutedsl(
        q, k, v, g, beta, use_qk_l2norm_in_kernel=True, **common
    )
    assert not atrex.can_use_chunk_gdn_fwd_cutedsl(
        q, k, v, g, beta, initial_state=None,
        scale=_SCALE, cu_seqlens=cu, cu_seqlens_cpu=cu_cpu,
        use_qk_l2norm_in_kernel=False,
    )
    assert not atrex.can_use_chunk_gdn_fwd_cutedsl(
        q, k, v, g.to(torch.bfloat16), beta,
        use_qk_l2norm_in_kernel=False, **common
    )
    assert not atrex.can_use_chunk_gdn_fwd_cutedsl(
        q, k, v, g, beta, scale=1.0,
        initial_state=initial_state, cu_seqlens=cu, cu_seqlens_cpu=cu_cpu,
        use_qk_l2norm_in_kernel=False,
    )
    assert not atrex.can_use_chunk_gdn_fwd_cutedsl(
        q, k, v, g, beta, state_checkpoints=torch.empty_like(initial_state),
        checkpoint_cu_starts=torch.tensor([0, 1], device="cuda", dtype=torch.int64),
        checkpoint_every_n_tokens=64,
        use_qk_l2norm_in_kernel=False, **common
    )


def test_chunk_gdn_sm103_forward_rejects_unnormalized_contract():
    _requires_sm103()
    import atrex

    q, k, v, g, beta, initial_state, cu, cu_cpu = _make_inputs((64,))
    with pytest.raises(ValueError, match="already be L2-normalized"):
        atrex.chunk_gdn_fwd_cutedsl(
            _build_context(),
            q,
            k,
            v,
            g,
            beta,
            scale=_SCALE,
            output_final_state=True,
            cu_seqlens=cu,
            cu_seqlens_cpu=cu_cpu,
            qk_l2norm_already_applied=False,
            initial_state=initial_state,
        )


def test_chunk_gdn_sm103_prewarm_public_api():
    _requires_sm103()
    import atrex

    atrex.chunk_gdn_fwd_cutedsl_prewarm_buckets(
        B=1,
        H=4,
        HV=32,
        K=128,
        V=128,
        output_final_state=True,
        scale=_SCALE,
        include_state_checkpoints=False,
    )
    _, output, final_state = _atrex_call((97,))
    assert torch.isfinite(output.float()).all()
    assert torch.isfinite(final_state).all()


def test_chunk_gdn_sm103_balanced_b3_performance_upper_bound():
    _requires_sm103()
    import atrex

    q, k, v, g, beta, initial_state, cu, cu_cpu = _make_inputs(
        (4224, 4224, 4224), continuation=True
    )
    ctx = _build_context()

    def call():
        return atrex.chunk_gdn_fwd_cutedsl(
            ctx,
            q,
            k,
            v,
            g,
            beta,
            scale=_SCALE,
            output_final_state=True,
            cu_seqlens=cu,
            cu_seqlens_cpu=cu_cpu,
            qk_l2norm_already_applied=True,
            initial_state=initial_state,
        )

    latency_ms = _bench_ms(call)
    print(
        f"SM103 AKA M64 balanced B=3 latency={latency_ms:.4f}ms "
        f"(limit={_MAX_BALANCED_B3_MS:.2f}ms)"
    )
    assert latency_ms <= _MAX_BALANCED_B3_MS


def _profile_chunk_gdn_sm103_workload() -> None:
    sys.argv = [sys.argv[0]]
    _atrex_call((128,))
    torch.cuda.synchronize()


def test_chunk_gdn_sm103_profiler_kernel_names_have_atrex_aka_prefix():
    _requires_sm103()
    nsys = shutil.which("nsys")
    assert nsys is not None, "Nsight Systems is required for the SM103 gate"
    workload = Path(__file__)
    with tempfile.TemporaryDirectory(prefix="atrex_aka_gdn_nsys_") as directory:
        report_base = Path(directory) / "chunk_gdn_sm103"
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
    lines = stats.stdout.splitlines()
    header_index = next(
        (
            index
            for index, line in enumerate(lines)
            if "Time (%)" in line and "Name" in line
        ),
        None,
    )
    assert header_index is not None, stats.stdout
    rows = csv.DictReader(io.StringIO("\n".join(lines[header_index:])))
    names = sorted({row["Name"].strip() for row in rows if row.get("Name")})
    aka_names = [name for name in names if "atrex_aka" in name]
    assert aka_names, f"no ATREX AKA kernel observed; CUDA events: {names}"
    assert all(name.startswith("atrex_aka_") for name in aka_names), aka_names
    print({"atrex_aka_kernel_names": aka_names})


if __name__ == "__main__":
    if sys.argv[1:] == [_PROFILE_WORKLOAD_FLAG]:
        _require_sm103_or_exit()
        _profile_chunk_gdn_sm103_workload()
        raise SystemExit(0)

    parser = argparse.ArgumentParser(description="Benchmark SM103 AKA M64 Chunk-GDN")
    parser.add_argument("--lengths", default="4224,4224,4224")
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--rep", type=int, default=100)
    args = parser.parse_args()
    _require_sm103_or_exit()
    lengths = tuple(int(value) for value in args.lengths.split(","))
    q, k, v, g, beta, initial_state, cu, cu_cpu = _make_inputs(
        lengths, continuation=True
    )
    ctx = _build_context()

    def benchmark_call():
        import atrex

        return atrex.chunk_gdn_fwd_cutedsl(
            ctx,
            q,
            k,
            v,
            g,
            beta,
            scale=_SCALE,
            output_final_state=True,
            cu_seqlens=cu,
            cu_seqlens_cpu=cu_cpu,
            qk_l2norm_already_applied=True,
            initial_state=initial_state,
        )

    print({"lengths": lengths, "atrex_ms": _bench_ms(
        benchmark_call, warmup=args.warmup, repetitions=args.rep
    )})
