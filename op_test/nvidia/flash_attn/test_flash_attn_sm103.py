"""AKA SM103 BF16 q4 decode integration and regression gates."""

from __future__ import annotations

import csv
import gc
import importlib.metadata
import inspect
import io
import math
from pathlib import Path
import shutil
import statistics
import subprocess
import sys
import tempfile

import pytest
import torch


ATOL = 1e-2
RTOL = 5e-2
PROFILE_FLAG = "--profile-kernel-names"

# The ten q4 production shapes from the frozen Qwen3.7-Max TP4 inventory.
Q4_SEQUENCE_LENGTHS = (
    (1563, 12520, 14313, 2095, 3158, 2696, 4997, 6072, 2226, 565,
     5312, 13175, 2882, 17340, 11425, 840),
    (1555, 12509, 14304, 2085, 3150, 2686, 4989, 6068, 2220, 13893,
     5307, 13163, 2870, 17328, 11413, 835),
    (15070, 12926, 14788, 2513, 3748, 3115, 5403, 6643, 8725, 12043,
     5796, 8069, 29358, 2419, 12102, 1236, 10964, 1629),
    (15006, 12854, 14708, 2443, 3651, 3049, 5340, 6554, 8637, 11935,
     5723, 7961, 368, 1891, 11994, 1169, 2857, 2316, 2390),
    (1629, 12605, 14401, 2176, 3273, 2767, 5071, 6177, 2331, 11491,
     5403, 13300, 2980, 17459, 11550, 915, 2971, 4426, 8277, 14695),
    (14936, 12793, 14640, 2376, 3563, 2986, 5289, 6451, 8541, 11831,
     5657, 7860, 5341, 17719, 11890, 1111, 4563, 1464, 9876, 6560,
     2217, 2788, 1806, 310),
    (14909, 12767, 14619, 2357, 3528, 2956, 5268, 6418, 8511, 11795,
     5622, 7825, 5319, 17695, 11854, 1088, 4543, 6004, 665, 6532,
     2187, 2752, 1786, 1265, 285, 9841),
    (14960, 12813, 14667, 2398, 3595, 3004, 5307, 6487, 8573, 11867,
     5674, 7896, 5375, 17734, 11926, 1127, 4580, 242, 9902, 6584,
     2251, 2823, 1830, 332, 612, 581, 2322),
    (14877, 12739, 14595, 2333, 3491, 2929, 5236, 6398, 8481, 11751,
     5587, 7781, 5275, 17665, 11810, 1064, 4509, 11094, 249, 6506,
     2147, 269, 1751, 1232, 261, 9799, 645, 2709),
    (14905, 12765, 14618, 2355, 3524, 2952, 5264, 6417, 8509, 11791,
     5620, 7821, 5315, 17694, 11850, 1087, 4539, 11133, 272, 6528,
     2184, 2748, 1783, 1261, 284, 9839, 661, 6000),
)


@pytest.fixture(scope="module", autouse=True)
def require_sm103():
    assert torch.cuda.is_available(), "SM103 CUDA device is required"
    assert torch.cuda.get_device_capability() == (10, 3), "SM103 is required"


def _make_case(seq_lens, *, seed=42):
    torch.manual_seed(seed)
    seq_lens = list(seq_lens)
    batch = len(seq_lens)
    page_size = 128
    page_counts = [math.ceil(length / page_size) for length in seq_lens]
    total_pages = sum(page_counts)
    key = torch.randn(
        total_pages, page_size, 1, 256,
        device="cuda", dtype=torch.bfloat16,
    )
    value = torch.randn_like(key)
    physical_pages = torch.arange(
        total_pages - 1, -1, -1, device="cuda", dtype=torch.int32
    )
    page_table = torch.full(
        (batch, max(page_counts)), -1, device="cuda", dtype=torch.int32
    )
    cursor = 0
    for request, count in enumerate(page_counts):
        page_table[request, :count] = physical_pages[cursor : cursor + count]
        cursor += count
    q = torch.randn(
        batch * 4, 16, 256, device="cuda", dtype=torch.bfloat16
    )
    cu_seqlens_q = torch.arange(
        0, (batch + 1) * 4, 4, device="cuda", dtype=torch.int32
    )
    seqused_k = torch.tensor(seq_lens, device="cuda", dtype=torch.int32)
    out = torch.empty_like(q)
    return {
        "q": q,
        "k": key,
        "v": value,
        "cu_seqlens_q": cu_seqlens_q,
        "seqused_k": seqused_k,
        "page_table": page_table,
        "out": out,
        "max_seqlen_q": 4,
        "max_seqlen_k": page_table.shape[1] * page_size,
        "softmax_scale": 256**-0.5,
        "causal": True,
    }


def _launch_kwargs(case):
    return {
        "q": case["q"],
        "k": case["k"],
        "v": case["v"],
        "cu_seqlens_q": case["cu_seqlens_q"],
        "cu_seqlens_k": None,
        "qv": None,
        "seqused_k": case["seqused_k"],
        "max_seqlen_q": case["max_seqlen_q"],
        "max_seqlen_k": case["max_seqlen_k"],
        "page_table": case["page_table"],
        "softmax_scale": case["softmax_scale"],
        "causal": case["causal"],
        "softcap": 0.0,
        "window_size_left": None,
        "window_size_right": None,
        "learnable_sink": None,
        "out": case["out"],
        "return_lse": False,
        "q_descale": None,
        "k_descale": None,
        "v_descale": None,
        "num_splits": 0,
    }


def _runtime_kwargs(case):
    kwargs = _launch_kwargs(case)
    kwargs.pop("qv")
    return kwargs


def _reference(case):
    q = case["q"]
    key_cache = case["k"]
    value_cache = case["v"]
    page_table = case["page_table"]
    seq_lens = case["seqused_k"].tolist()
    output = torch.empty_like(q)
    for request, kv_length in enumerate(seq_lens):
        page_count = math.ceil(kv_length / 128)
        page_ids = page_table[request, :page_count].long()
        key = key_cache[page_ids].reshape(-1, 256)[:kv_length].float()
        value = value_cache[page_ids].reshape(-1, 256)[:kv_length].float()
        query = q[request * 4 : (request + 1) * 4].float()
        scores = torch.einsum("qhd,kd->hqk", query, key) * (256**-0.5)
        query_positions = torch.arange(4, device=q.device) + kv_length - 4
        key_positions = torch.arange(kv_length, device=q.device)
        scores.masked_fill_(
            key_positions[None, None, :] > query_positions[None, :, None],
            -torch.inf,
        )
        result = torch.einsum("hqk,kd->qhd", torch.softmax(scores, -1), value)
        output[request * 4 : (request + 1) * 4] = result.to(torch.bfloat16)
    return output


def _call(case):
    import atrex

    output, lse = atrex.flash_attn_varlen_func(**_launch_kwargs(case))
    assert lse is None
    return output


def test_public_signature_matches_vllm_fa4_lowering():
    import atrex
    from atrex.api.flash_attn import flash_attn_varlen_func

    expected = (
        "q", "k", "v", "qv", "cu_seqlens_q", "cu_seqlens_k",
        "seqused_q", "seqused_k", "max_seqlen_q", "max_seqlen_k",
        "min_seqlen_k", "page_table", "softmax_scale", "causal", "softcap",
        "window_size_left", "window_size_right", "learnable_sink", "tile_mn",
        "mma_pv_is_rs", "intra_wg_overlap", "num_threads", "num_splits",
        "pack_gqa", "_arch", "score_mod", "mask_mod", "block_sparse_tensors",
        "return_lse", "out", "lse", "aux_tensors", "aux_scalars",
        "q_descale", "k_descale", "v_descale", "gather_kv_indices",
        "output_scale",
    )
    assert tuple(inspect.signature(flash_attn_varlen_func).parameters) == expected
    assert callable(atrex.flash_attn_varlen_func)
    assert callable(atrex.can_use_flash_attn_varlen_func)

    case = _make_case([127 + index for index in range(16)])
    caller_kwargs = _launch_kwargs(case)
    q = caller_kwargs.pop("q")
    k = caller_kwargs.pop("k")
    v = caller_kwargs.pop("v")
    inspect.signature(flash_attn_varlen_func).bind(q, k, v, **caller_kwargs)


def test_import_atrex_is_lazy():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys, atrex; "
                "assert 'atrex.api.flash_attn' not in sys.modules; "
                "assert 'cutlass' not in sys.modules"
            ),
        ],
        check=True,
        text=True,
        capture_output=True,
    )
    assert result.stderr == ""


def test_cutedsl_version_is_validated():
    assert importlib.metadata.version("nvidia-cutlass-dsl") == "4.6.1"


def test_aka_q4_eligibility_is_strict():
    import atrex

    case = _make_case([127 + index for index in range(16)])
    launch_kwargs = _launch_kwargs(case)
    assert atrex.can_use_flash_attn_varlen_func(**launch_kwargs)

    rejected = (
        {"max_seqlen_q": 1},
        {"causal": False},
        {"softmax_scale": 0.1},
        {"num_splits": 1},
        {"return_lse": True},
        {"q_descale": torch.ones(1, device="cuda")},
        {"learnable_sink": torch.ones(16, device="cuda")},
    )
    for override in rejected:
        kwargs = dict(launch_kwargs)
        kwargs.update(override)
        assert not atrex.can_use_flash_attn_varlen_func(**kwargs)

    q1 = dict(case)
    q1["q"] = case["q"][::4].contiguous()
    q1["out"] = torch.empty_like(q1["q"])
    q1["cu_seqlens_q"] = torch.arange(
        17, device="cuda", dtype=torch.int32
    )
    q1["max_seqlen_q"] = 1
    assert not atrex.can_use_flash_attn_varlen_func(**_launch_kwargs(q1))
    with pytest.raises(NotImplementedError, match="AKA BF16 q4"):
        _call(q1)


@pytest.mark.parametrize("seq_lens", Q4_SEQUENCE_LENGTHS)
def test_production_q4_correctness_and_out_identity(seq_lens):
    case = _make_case(seq_lens, seed=len(seq_lens))
    expected = _reference(case)
    result = _call(case)
    torch.cuda.synchronize()
    assert result is case["out"]
    torch.testing.assert_close(result, expected, atol=ATOL, rtol=RTOL)


def test_poisoned_call_local_workspace():
    from atrex.src.nvidia.flash_attn.sm103 import aka_decode_runtime as runtime

    case = _make_case([129 + 7 * index for index in range(16)], seed=10)
    expected = _reference(case)
    device_index = case["q"].device.index or 0
    _, workspace_splits = runtime._atrex_aka_split_config(
        16, case["page_table"].shape[1], device_index
    )
    workspace = torch.full(
        (runtime._atrex_aka_workspace_elements(workspace_splits, 16),),
        float("nan"),
        dtype=torch.float32,
        device="cuda",
    )
    result, lse = runtime.atrex_aka_fa4_decode(
        **_runtime_kwargs(case), _workspace=workspace
    )
    torch.cuda.synchronize()
    assert result is case["out"]
    assert lse is None
    torch.testing.assert_close(result, expected, atol=ATOL, rtol=RTOL)


def test_two_streams_do_not_alias_workspace():
    case_a = _make_case([255 + index for index in range(16)], seed=20)
    case_b = _make_case([511 - index for index in range(16)], seed=21)
    expected_a = _reference(case_a)
    expected_b = _reference(case_b)
    torch.cuda.synchronize()
    stream_a, stream_b = torch.cuda.Stream(), torch.cuda.Stream()
    with torch.cuda.stream(stream_a):
        result_a = _call(case_a)
    with torch.cuda.stream(stream_b):
        result_b = _call(case_b)
    stream_a.synchronize()
    stream_b.synchronize()
    torch.testing.assert_close(result_a, expected_a, atol=ATOL, rtol=RTOL)
    torch.testing.assert_close(result_b, expected_b, atol=ATOL, rtol=RTOL)


def test_cuda_graph_replay_with_changing_seqused_k():
    long_lens = [511 + index for index in range(16)]
    short_lens = [129 + 3 * index for index in range(16)]
    case = _make_case(long_lens, seed=30)

    warmup_stream = torch.cuda.Stream()
    warmup_stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(warmup_stream):
        _call(case)
    torch.cuda.current_stream().wait_stream(warmup_stream)
    torch.cuda.synchronize()

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured_out = _call(case)
    assert captured_out is case["out"]

    for seq_lens in (short_lens, long_lens, short_lens):
        case["seqused_k"].copy_(
            torch.tensor(seq_lens, device="cuda", dtype=torch.int32)
        )
        graph.replay()
        torch.cuda.synchronize()
        expected = _reference(case)
        torch.testing.assert_close(case["out"], expected, atol=ATOL, rtol=RTOL)


def test_call_local_workspace_does_not_remain_allocated():
    from atrex.src.nvidia.flash_attn.sm103 import aka_decode_runtime as runtime

    cases = [
        _make_case([97 + batch] * batch, seed=100 + batch)
        for batch in range(16, 29)
    ]
    device_index = torch.cuda.current_device()
    runtime._atrex_aka_get_decode_kernel(device_index)
    torch.cuda.synchronize()
    gc.collect()
    baseline_allocated = torch.cuda.memory_allocated()

    reserved_samples = []
    for _ in range(3):
        for case in cases:
            _call(case)
        torch.cuda.synchronize()
        gc.collect()
        assert torch.cuda.memory_allocated() <= baseline_allocated + 1024**2
        reserved_samples.append(torch.cuda.memory_reserved())

    assert max(reserved_samples[1:]) - min(reserved_samples[1:]) <= 1024**2


def _benchmark_ms(call, *, warmup=10, repetitions=30):
    for _ in range(warmup):
        call()
    torch.cuda.synchronize()
    samples = []
    for _ in range(repetitions):
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        call()
        end.record()
        end.synchronize()
        samples.append(start.elapsed_time(end))
    return statistics.median(samples)


def test_public_api_bounded_performance():
    case = _make_case(Q4_SEQUENCE_LENGTHS[0], seed=50)
    latency_ms = _benchmark_ms(lambda: _call(case))
    print({"atrex_aka_q4_public_api_ms": latency_ms, "limit_ms": 0.15})
    assert latency_ms <= 0.15


def _profile_workload():
    sys.argv = [sys.argv[0]]
    case = _make_case([255 + index for index in range(16)], seed=60)
    _call(case)
    _call(case)
    torch.cuda.synchronize()
    assert torch.isfinite(case["out"]).all()


def test_profiler_observes_two_atrex_aka_kernels():
    nsys = shutil.which("nsys")
    assert nsys is not None, "Nsight Systems is required for the SM103 gate"
    with tempfile.TemporaryDirectory(prefix="atrex_fa4_nsys_") as directory:
        report_base = Path(directory) / "fa4_decode"
        subprocess.run(
            [
                nsys, "profile", "--trace=cuda", "--sample=none",
                "--cpuctxsw=none", "--force-overwrite=true", "--output",
                str(report_base), sys.executable, str(Path(__file__)), PROFILE_FLAG,
            ],
            check=True,
            text=True,
            capture_output=True,
        )
        stats = subprocess.run(
            [
                nsys, "stats", "--report", "cuda_gpu_kern_sum", "--format",
                "csv", str(report_base.with_suffix(".nsys-rep")),
            ],
            check=True,
            text=True,
            capture_output=True,
        )
    lines = stats.stdout.splitlines()
    header = next(
        (index for index, line in enumerate(lines) if "Time (%)" in line and "Name" in line),
        None,
    )
    assert header is not None, stats.stdout
    rows = csv.DictReader(io.StringIO("\n".join(lines[header:])))
    names = sorted({row["Name"].strip() for row in rows if row.get("Name")})
    aka_names = [name for name in names if "atrex_aka" in name]
    assert len(aka_names) == 2, f"expected decode + reduction, got {aka_names}"
    assert all(name.startswith("atrex_aka_") for name in aka_names)
    assert any("reduction" in name.lower() for name in aka_names)
    print({"atrex_aka_kernel_names": aka_names})


if __name__ == "__main__" and sys.argv[1:] == [PROFILE_FLAG]:
    assert torch.cuda.is_available()
    assert torch.cuda.get_device_capability() == (10, 3)
    _profile_workload()
