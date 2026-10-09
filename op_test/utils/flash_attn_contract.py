"""Standalone FP32 reference and workload helpers for the unified FA contract."""

import math
import os
from pathlib import Path
import time

import torch


def error_metrics(actual, expected):
    """Diagnostics from the same input tensors; these do not set tolerances.

    A zero reference has undefined relative L2/norm ratio unless both outputs
    are zero. Return None for undefined/nonfinite metrics (valid JSON).
    """
    if actual.shape != expected.shape:
        raise ValueError("actual/reference shapes must match")
    actual = actual.detach().float().reshape(-1)
    expected = expected.detach().float().reshape(-1)
    finite = bool(torch.isfinite(actual).all() & torch.isfinite(expected).all())
    if not finite:
        return dict(finite=False, cosine=None, relative_l2=None,
                    norm_ratio=None, max_abs=None)
    norm_actual = torch.linalg.vector_norm(actual).item()
    norm_reference = torch.linalg.vector_norm(expected).item()
    error = torch.linalg.vector_norm(actual - expected).item()
    both_zero = norm_actual == norm_reference == 0
    cosine = (torch.dot(actual, expected).item() / (norm_actual * norm_reference)
              if norm_actual and norm_reference else float(both_zero))
    return dict(
        finite=True,
        cosine=max(-1.0, min(1.0, cosine)),
        relative_l2=error / norm_reference if norm_reference else (0.0 if both_zero else None),
        norm_ratio=norm_actual / norm_reference if norm_reference else None,
        max_abs=(actual - expected).abs().max().item() if actual.numel() else 0.0,
    )


def build_case(page_size, q_lens, kv_lens, *, layout="interleaved", fp8=True,
               hq=8, hkv=1):
    torch.manual_seed(1729)
    dtype = torch.float8_e4m3fn if fp8 else torch.bfloat16
    counts = [math.ceil(n / page_size) for n in kv_lens]
    pages = sum(counts)
    if layout == "interleaved":
        cache = (torch.randn(pages, 2, page_size, hkv, 256, device="cuda") * .4).to(dtype)
        k, v = cache[:, 0], cache[:, 1]
    elif layout == "strided":
        cache = (torch.randn(pages, 2, page_size * 2, hkv, 256, device="cuda") * .4).to(dtype)
        k, v = cache[:, 0, ::2], cache[:, 1, ::2]
    else:
        raise ValueError(layout)
    # Use a wider parent table to test row stride separately from logical width.
    parent = torch.full((len(counts), max(counts) + 3), -1, dtype=torch.int32, device="cuda")
    table = parent[:, :max(counts)]
    permutation = torch.randperm(pages, device="cuda").to(torch.int32)
    offset = 0
    for row, n in enumerate(counts):
        table[row, :n] = permutation[offset:offset+n]
        offset += n
    offsets = [0]
    for n in q_lens:
        offsets.append(offsets[-1] + n)
    q = (torch.randn(offsets[-1], hq, 256, device="cuda") * .4).to(dtype)
    scales = [torch.tensor(values, dtype=torch.float32, device="cuda").reshape(-1, 1)
              + torch.arange(hkv, dtype=torch.float32, device="cuda")[None, :] * .0625
              for values in ([.75 + .125*i for i in range(len(counts))],
                             [1.25 + .125*i for i in range(len(counts))],
                             [.5 + .25*i for i in range(len(counts))])]
    return dict(q=q, k=k, v=v, block_table=table,
                cu_seqlens_q=torch.tensor(offsets, dtype=torch.int32, device="cuda"),
                seqused_k=torch.tensor(kv_lens, dtype=torch.int32, device="cuda"),
                max_seqlen_q=max(q_lens), max_seqlen_k=max(kv_lens),
                softmax_scale=256**-.5, causal=True,
                q_descale=scales[0] if fp8 else None,
                k_descale=scales[1] if fp8 else None,
                v_descale=scales[2] if fp8 else None)


def reference(case):
    q, k, v = (case[x] for x in ("q", "k", "v"))
    hq, hkv = q.shape[1], k.shape[2]
    group = hq // hkv
    cu = case["cu_seqlens_q"].cpu().tolist()
    lengths = case["seqused_k"].cpu().tolist()
    result = torch.empty_like(q, dtype=torch.float32)
    lse = torch.empty((hq, q.shape[0]), dtype=torch.float32, device=q.device)
    for r, length in enumerate(lengths):
        nq = cu[r+1] - cu[r]
        if nq == 0:
            continue
        ids = case["block_table"][r, :math.ceil(length/k.shape[1])].long()
        qr = q[cu[r]:cu[r+1]].float()
        kr = k[ids].reshape(-1, hkv, 256)[:length].float().repeat_interleave(group, dim=1)
        vr = v[ids].reshape(-1, hkv, 256)[:length].float().repeat_interleave(group, dim=1)
        for name in ("q_descale", "k_descale", "v_descale"):
            if case[name] is not None:
                scale = case[name][r].repeat_interleave(group).reshape(1, hq, 1)
                if name == "q_descale": qr = qr * scale
                elif name == "k_descale": kr = kr * scale
                else: vr = vr * scale
        scores = torch.einsum("qhd,khd->hqk", qr, kr) * case["softmax_scale"]
        rows = torch.arange(nq, device=q.device) + length - nq
        cols = torch.arange(length, device=q.device)
        scores.masked_fill_(cols[None, :] > rows[:, None], -torch.inf)
        lse[:, cu[r]:cu[r+1]] = torch.logsumexp(scores, dim=-1)
        probability = torch.nan_to_num(torch.softmax(scores, dim=-1))
        result[cu[r]:cu[r+1]] = torch.einsum("hqk,khd->qhd", probability, vr)
    return result, lse


def check_case(case, splits=0, graph=False):
    from atrex import flash_attn_varlen_func

    out = torch.empty_like(case["q"], dtype=torch.bfloat16)
    kwargs = dict(case, out=out, return_softmax_lse=True, num_splits=splits)
    actual, lse = flash_attn_varlen_func(**kwargs)
    assert actual.data_ptr() == out.data_ptr()
    expected, expected_lse = reference(case)
    assert torch.isfinite(actual).all()
    torch.testing.assert_close(actual.float(), expected, atol=3e-2, rtol=5e-2)
    torch.testing.assert_close(lse, expected_lse, atol=3e-2, rtol=5e-2)
    if graph:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                flash_attn_varlen_func(**kwargs)
        torch.cuda.current_stream().wait_stream(stream)
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            graph_out, graph_lse = flash_attn_varlen_func(**kwargs)
        # Alter valid KV lengths while preserving graph-static upper bounds.
        qlens = case["cu_seqlens_q"][1:] - case["cu_seqlens_q"][:-1]
        case["seqused_k"].copy_(torch.maximum(case["seqused_k"] - 1, qlens))
        captured.replay()
        torch.cuda.synchronize()
        expected, expected_lse = reference(case)
        torch.testing.assert_close(graph_out.float(), expected, atol=3e-2, rtol=5e-2)
        torch.testing.assert_close(graph_lse, expected_lse, atol=3e-2, rtol=5e-2)


def check_eager_scratch_reuse(calls, *, rounds=100, tolerance_bytes=256 * 1024):
    """Check a fixed input/output working set, without flushing the allocator.

    Each entry is a layer/shape call with an already allocated output. Warm
    only the first layer of each shape, then exercise all layer addresses.
    Caller-owned tensors must all exist before sampling. This is an eager
    ownership regression, not a graph-pool or deployment capacity test.
    """
    for call in calls[0]:
        for _ in range(3):
            call()
    torch.cuda.synchronize()
    baseline = torch.cuda.memory_allocated()
    samples = []
    torch.cuda.reset_peak_memory_stats()
    for iteration in range(rounds):
        for layer in calls:
            for call in layer:
                call()
        torch.cuda.synchronize()
        if iteration in (0, rounds // 2, rounds - 1):
            samples.append(dict(
                iteration=iteration + 1,
                allocated=torch.cuda.memory_allocated(),
                reserved=torch.cuda.memory_reserved(),
                peak=torch.cuda.max_memory_allocated(),
            ))
    print("EAGER_SCRATCH_REUSE", dict(baseline=baseline, samples=samples))
    assert all(s["allocated"] <= baseline + tolerance_bytes for s in samples), (
        baseline, samples
    )


def profile_case(case):
    from atrex import flash_attn_varlen_func

    for _ in range(3):
        flash_attn_varlen_func(**case)
    torch.cuda.synchronize()
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,
                                            torch.profiler.ProfilerActivity.CUDA]) as prof:
        flash_attn_varlen_func(**case)
        torch.cuda.synchronize()
    names = [e.name for e in prof.events() if e.device_type == torch.autograd.DeviceType.CUDA]
    if trace_dir := os.environ.get("ATREX_FA_TRACE_DIR"):
        directory = Path(trace_dir)
        directory.mkdir(parents=True, exist_ok=True)
        prof.export_chrome_trace(str(directory / f"attention-{time.time_ns()}.json"))
    assert names, "Profiler exposed no GPU events; use the target native profiler"
    # Attribute Torch output initialization separately from custom kernels.
    # BF16 split/atomic decode calls out.zero_() (and optionally lse.fill_()).
    # Torch may implement those fills as a compute launch instead of memset.
    # Do not hide arbitrary elementwise kernels or vendor attention launches.
    print("GPU_EVENTS", names)
    kernels = [n for n in names if not n.lower().startswith(("memcpy", "memset"))]
    assert kernels, "Profiler exposed no compute kernels"
    torch_fills = [n for n in kernels
                   if n.startswith("void at::native::vectorized_elementwise_kernel<")
                   and "at::native::FillFunctor<" in n]
    if torch_fills:
        cpu_names = {e.name for e in prof.events()
                     if e.device_type == torch.autograd.DeviceType.CPU}
        assert "aten::fill_" in cpu_names, "Unattributed Torch fill launch"
    print("TORCH_INITIALIZATION_EVENTS", torch_fills)
    custom = [n for n in kernels if n not in torch_fills]
    assert custom, "Profiler exposed no custom attention kernels"
    assert all("atrex_" in name for name in custom), custom
    return names
