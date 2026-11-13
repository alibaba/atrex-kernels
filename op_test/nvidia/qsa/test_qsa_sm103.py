"""QSA correctness, profiling and paired performance tests on sm103."""

import gc
import json
import math
import os
from pathlib import Path
import shutil
import sqlite3
import subprocess
import sys
import pytest
import torch
import argparse
import hashlib
import importlib
import importlib.metadata
import statistics


ATOL, RTOL = 0.003, 0.02


CASES = [
    (128, "bf16", "q1"),
    (128, "fp8", "q1"),
    (1056, "bf16", "q4"),
    (1056, "fp8", "q4"),
    (2112, "bf16", "q4_union"),
    (2112, "fp8", "q4_union"),
    (16, "bf16", "prefill"),
    (16, "fp8", "prefill"),
]


def installed_api(architecture):
    import atrex

    package = Path(atrex.__file__).resolve()
    source = Path(__file__).resolve().parents[3] / "python"
    assert not package.is_relative_to(source), "Install the wheel before testing"
    root = os.environ.get("ATREX_TEST_INSTALL_ROOT")
    if root:
        assert package.is_relative_to(Path(root).resolve()), package
    assert (
        torch.cuda.get_device_capability()
        == {"sm100": (10, 0), "sm103": (10, 3)}[architecture]
    )
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    return atrex.qsa_compact


def launch_options(scales):
    return (
        dict(
            sm_scale=256**-0.5 * scales["q"] * scales["k_cache"],
            v_scale=scales["v_cache"],
        )
        if scales
        else {}
    )


def expand_torch(indexer_block_ids, query_positions):
    # Independent Torch expansion for the oracle, outside candidate runtime.
    rows, topk = indexer_block_ids.shape
    cols = torch.arange(topk * 4 + 3, device=indexer_block_ids.device)
    visible = query_positions.long() + 1
    complete = (visible // 4).clamp(0, topk)
    count = complete * 4
    rank = (cols // 4).clamp(max=topk - 1)
    selected = indexer_block_ids[:, rank].long() * 4 + cols % 4
    tail_start = (visible // 4) * 4
    tail_count = visible % 4
    token = torch.where(
        cols[None, :] < count[:, None],
        selected,
        tail_start[:, None] + cols[None, :] - count[:, None],
    )
    valid = (cols[None, :] < (count + tail_count)[:, None]) & (token >= 0)
    packed = torch.empty(
        (rows, topk * 4 + 4), device=indexer_block_ids.device, dtype=torch.int32
    )
    packed[:, :-1] = torch.where(valid, token, -1).int()
    packed[:, -1] = (count + tail_count).int()
    return packed


def slices(t):
    step = max(1, 16 * 1024**2 // (t[0].numel() * 4))
    for begin in range(0, len(t), step):
        yield slice(begin, begin + step)


def to_fp8(data, *, consume=False):
    """Per-tensor absmax calibration; preserve element strides and shared pools.

    Only actual Q/K/V entries are calibrated/converted, never padding in the
    underlying interleaved allocation. Calibration is outside all timed ranges.
    With consume=True, release converted source entries from the caller's dict
    as each tensor finishes. This bounds peak memory for full 128-GiB KV tests;
    payload, scales, strides and shared-pool semantics are unchanged.
    """
    converted = dict(data)
    pools = {}
    scales = {}
    details = {}
    output_shape, output_device = data["q"].shape, data["q"].device
    for key in ("q", "k_cache", "v_cache"):
        t = data[key]
        mx = torch.zeros((), device=t.device)
        for sl in slices(t):
            mx = torch.maximum(mx, t[sl].float().abs().amax())
        amax = float(mx.item())
        assert amax < float("inf"), (key, amax)
        scale = max(amax / 448.0, 1e-12)
        ptr = t.untyped_storage().data_ptr()
        if ptr not in pools:
            pools[ptr] = torch.empty(
                t.untyped_storage().nbytes() // t.element_size(),
                device=t.device,
                dtype=torch.float8_e4m3fn,
            )
        out = pools[ptr].as_strided(t.shape, t.stride(), t.storage_offset())
        zeros = total = 0
        for sl in slices(t):
            f = t[sl].float() / scale
            out[sl].copy_(f.clamp(-448, 448).to(torch.float8_e4m3fn))
            zeros += int(((out[sl].float() == 0) & (t[sl] != 0)).sum().item())
            total += f.numel()
        converted[key] = out
        scales[key] = scale
        details[key] = dict(
            amax=amax,
            scale=scale,
            underflow_nonzero_to_zero=zeros,
            elements=total,
            shape=list(t.shape),
            element_strides=list(t.stride()),
            storage_offset=t.storage_offset(),
        )
        if consume:
            del data[key]
            del t
    converted["out"] = (
        torch.empty(output_shape, device=output_device, dtype=torch.bfloat16)
        if consume
        else torch.empty_like(data["q"], dtype=torch.bfloat16)
    )
    return converted, scales, details


def oracle(d, scales):
    """Independent FP32 reference on dequantized selected entries, BF16 A/gate.

    FP8 is gathered as uint8 then reinterpreted; no whole-cache dequantization.
    The same equations and mask are used for BF16 and FP8.
    """
    q, k, v = d["q"], d["k_cache"], d["v_cache"]
    packed = expand_torch(d["indexer_block_ids"], d["query_positions"])
    table = d["block_table"]
    width = packed.shape[1] - 1
    result = torch.empty(q.shape, device=q.device, dtype=torch.bfloat16)
    sq, sk, sv = [scales.get(n, 1.0) for n in ("q", "k_cache", "v_cache")]
    for begin in range(0, len(q), 8):
        end = min(begin + 8, len(q))
        ids = packed[begin:end, :-1].long()
        req = d["token_to_req"][begin:end].long()
        lp = ids.clamp_min(0) // k.shape[1]
        off = ids.clamp_min(0) % k.shape[1]
        phys = table[
            req[:, None].clamp(0, len(table) - 1), lp.clamp(0, table.shape[1] - 1)
        ].long()
        valid = (
            (torch.arange(width, device=q.device)[None, :] < packed[begin:end, -1:])
            & (ids >= 0)
            & (req[:, None] >= 0)
            & (req[:, None] < len(table))
            & (lp < table.shape[1])
            & (phys >= 0)
            & (phys < len(k))
        )

        def gather(t):
            raw = t.view(torch.uint8) if t.dtype == torch.float8_e4m3fn else t
            x = raw[phys.clamp(0, len(k) - 1), off]
            if t.dtype == torch.float8_e4m3fn:
                x = x.view(torch.float8_e4m3fn)
            return torch.where(valid[:, :, None, None], x.float(), 0.0)

        keys = gather(k) * sk
        values = gather(v) * sv
        query = (q[begin:end].float() * sq).reshape(
            end - begin, k.shape[2], q.shape[1] // k.shape[2], q.shape[2]
        )
        logits = torch.einsum("rhgd,rthd->rhgt", query, keys) * (q.shape[2] ** -0.5)
        logits.masked_fill_(~valid[:, None, None, :], -torch.inf)
        p = torch.softmax(logits, -1).nan_to_num(nan=0.0)
        a = (
            torch.einsum("rhgt,rthd->rhgd", p, values)
            .reshape(end - begin, q.shape[1], q.shape[2])
            .to(torch.bfloat16)
        )
        result[begin:end] = (
            a.float() * torch.sigmoid(d["output_gate"][begin:end].float())
        ).to(torch.bfloat16)
    return result


def error(y, ref):
    a, b = y.float(), ref.float()
    diff = (a - b).abs()
    finite = torch.isfinite(a) & torch.isfinite(b)
    bad = (~finite) | (diff > 0.003 + 0.02 * b.abs())

    def number(x):
        value = float(x)
        return value if math.isfinite(value) else None

    return dict(
        passed=not bool(bad.any()),
        failed=int(bad.sum()),
        elements=y.numel(),
        failed_fraction=float(bad.float().mean()),
        finite=bool(finite.all()),
        max_abs=number(diff.nan_to_num(nan=float("inf")).max()),
        mean_abs=number(diff.nan_to_num(nan=float("inf")).mean()),
        relative_l2=number(
            torch.linalg.vector_norm(a - b)
            / torch.linalg.vector_norm(b).clamp_min(1e-30)
        ),
        atol=0.003,
        rtol=0.02,
    )


def _inputs(page, dtype, mode):
    torch.manual_seed(20261110 + page)
    batch, queries = {
        "q1": (8, 1),
        "q4": (32, 4),
        "q4_small": (3, 4),
        "q4_union": (64, 4),
        "prefill": (3, 0),
        "prefill_scramble": (33, 0),
    }[mode]
    rows = batch * queries if queries else (65 if batch == 33 else 17)
    hkv = 2 if mode == "q4_union" else 1
    heads = hkv * 8
    width = (131072 + page - 1) // page
    pages = 19
    # Shared K/V allocation, interleaved heads and a gap between pages.
    # This is a synthetic layout; it is not labelled as a production capture.
    pool = torch.empty((pages, page * 2, hkv, 512), device="cuda", dtype=torch.bfloat16)
    k, v = pool[:, :page, :, :256], pool[:, :page, :, 256:]
    k.copy_(torch.randn(k.shape, device="cuda") * 0.25)
    v.copy_(torch.randn(v.shape, device="cuda") * 0.5)
    q = (torch.randn((rows, heads, 256), device="cuda") * 0.25).bfloat16()
    ids = torch.full((rows, 512), -1, device="cuda", dtype=torch.int32)
    positions = torch.empty(rows, device="cuda", dtype=torch.int64)
    requests = (
        torch.arange(batch, device="cuda", dtype=torch.int32).repeat_interleave(queries)
        if queries
        else torch.arange(rows, device="cuda", dtype=torch.int32) % batch
    )
    table = (
        torch.arange(batch * width, device="cuda", dtype=torch.int32).reshape(
            batch, width
        )
        * 7
        + 3
    ) % pages
    data = dict(
        q=q,
        k_cache=k,
        v_cache=v,
        indexer_block_ids=ids,
        query_positions=positions,
        block_table=table,
        token_to_req=requests,
        use_prefill_config=not queries,
        output_gate=torch.randn_like(q),
        out=torch.empty_like(q),
    )
    scales = {}
    if dtype == "fp8":
        data, scales, _ = to_fp8(data, consume=True)
    return data, scales, queries


def _refresh(data, page, queries, revision):
    rows = len(data["q"])
    batch, width = data["block_table"].shape
    pages = len(data["k_cache"])
    table = (
        torch.arange(batch * width, device="cuda", dtype=torch.int32).reshape(
            batch, width
        )
        * 7
        + 3
        + revision
    ) % pages
    # In particular, small pages have invalid *interior* pages within one
    # bitmap word, while both endpoint pages can remain valid.
    table[:, 1::3] = -1
    table[:, 2::7] = pages + 2
    data["block_table"].copy_(table)
    lengths = (
        1,
        3,
        page - 1,
        page,
        page + 1,
        page + 3,
        2 * page - 1,
        2 * page + 1,
        4095,
        16384,
        65536,
        131072,
    )
    for row in range(rows):
        request = row // queries if queries else row % batch
        length = lengths[(request + revision) % len(lengths)]
        # A Q4 group's last query is at most 131072 tokens long.
        position = max(0, length - max(queries, 1)) + (row % queries if queries else 0)
        data["query_positions"][row] = position
        count = min(512, (position + 1) // 4)
        selected = torch.randperm((position + 1) // 4, device="cuda")[:count].int()
        data["indexer_block_ids"][row].fill_(-1)
        data["indexer_block_ids"][row, :count].copy_(selected)
    mapping = (
        torch.arange(batch, device="cuda", dtype=torch.int32).repeat_interleave(queries)
        if queries
        else torch.arange(rows, device="cuda", dtype=torch.int32) % batch
    )
    mapping = (mapping + revision) % batch
    if revision == 2:
        mapping.fill_(-1)
    elif queries:
        mapping[-queries:] = -1  # A padded request slot in the captured shape.
    else:
        mapping[::7] = -1
    data["token_to_req"].copy_(mapping)


def negative_pages(api):
    import pytest

    data, _, _ = _inputs(128, "bf16", "q1")
    for page in (0, 4, 8, 15, 17, 24, 127):
        with pytest.raises(ValueError, match="positive multiple of 16"):
            api(
                **{
                    **data,
                    "k_cache": data["k_cache"][:, :page],
                    "v_cache": data["v_cache"][:, :page],
                }
            )


def small_case(page=128, ratio=8, rows=65):
    torch.manual_seed(100000 + page + ratio)
    hkv = 2 if ratio == 12 else 1
    npages = (4096 + page - 1) // page
    k = torch.randn((3 * npages, page, hkv, 256), device="cuda", dtype=torch.bfloat16)
    v = torch.randn_like(k)
    table = torch.randperm(3 * npages, device="cuda").int().reshape(3, -1)
    positions = [
        0,
        1,
        2,
        3,
        7,
        127,
        128,
        1022,
        1023,
        2046,
        2047,
        2048,
        2050,
        3001,
        4000,
        4094,
    ]
    pos = torch.tensor(
        (positions * ((rows + 15) // 16))[:rows], device="cuda", dtype=torch.int64
    )
    ids = torch.full((rows, 512), -1, device="cuda", dtype=torch.int32)
    for row, p in enumerate(pos.cpu().tolist()):
        n = min(512, (p + 1) // 4)
        ids[row, :n] = torch.randperm((p + 1) // 4, device="cuda")[:n].int()
    q = torch.randn((rows, ratio * hkv, 256), device="cuda", dtype=torch.bfloat16)
    return dict(
        q=q,
        k_cache=k,
        v_cache=v,
        indexer_block_ids=ids,
        query_positions=pos,
        block_table=table,
        token_to_req=torch.arange(rows, device="cuda", dtype=torch.int32) % 3,
        use_prefill_config=True,
        output_gate=torch.randn_like(q),
    )


def decode_case(batch, queries, page=128, hkv=1, ratio=None):
    """Small physical KV, with every row using a full sparse selection."""
    torch.manual_seed(110500 + batch + queries + page + hkv)
    width = (4096 + page - 1) // page
    rows = batch * queries
    ratio = (8 if hkv == 1 else 12) if ratio is None else ratio
    k = torch.randn(
        (batch * width, page, hkv, 256), device="cuda", dtype=torch.bfloat16
    )
    q = torch.randn((rows, ratio * hkv, 256), device="cuda", dtype=torch.bfloat16)
    return dict(
        q=q,
        k_cache=k,
        v_cache=torch.randn_like(k),
        indexer_block_ids=torch.stack(
            [torch.randperm(1023, device="cuda")[:512] for _ in range(rows)]
        ).int(),
        query_positions=torch.full((rows,), 4094, device="cuda", dtype=torch.int64),
        block_table=torch.randperm(batch * width, device="cuda")
        .reshape(batch, width)
        .int(),
        token_to_req=torch.arange(
            batch, device="cuda", dtype=torch.int32
        ).repeat_interleave(queries),
        use_prefill_config=False,
        output_gate=torch.randn_like(q),
    )


def check(actual, expected):
    error = (actual.float() - expected.float()).abs()
    bad = (
        ~torch.isfinite(actual)
        | ~torch.isfinite(expected)
        | (error > ATOL + RTOL * expected.float().abs())
    )
    stats = dict(
        failed=int(bad.sum()),
        max_abs=float(error.max()),
        required_atol=float((error - RTOL * expected.float().abs()).max().clamp_min(0)),
    )
    assert stats["failed"] == 0, stats
    return stats


def capture(fn):
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            fn()
    stream.synchronize()
    start = torch.cuda.Event(enable_timing=True, external=True)
    end = torch.cuda.Event(enable_timing=True, external=True)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        start.record()
        fn()
        end.record()
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    return graph, start, end


def boundaries_and_metadata_refresh(api, page, ratio, dtype):
    data = small_case(page, ratio)
    scales = {}
    kwargs = {}
    if dtype == "fp8":
        data, scales, _ = to_fp8(data)
        kwargs = {
            "sm_scale": 256**-0.5 * scales["q"] * scales["k_cache"],
            "v_scale": scales["v_cache"],
        }
    data["out"] = torch.empty(data["q"].shape, device="cuda", dtype=torch.bfloat16)
    for refresh in (False, True):
        if refresh:
            data["indexer_block_ids"].fill_(-1)
            data["block_table"].copy_(data["block_table"].flip(0))
            data["token_to_req"].copy_((data["token_to_req"] + 1) % 3)
        expected = oracle(data, scales)
        y = api(**data, **kwargs)
        torch.cuda.synchronize()
        assert y is data["out"]
        stats = error(y, expected)
        print(dict(page=page, dtype=dtype, refresh=refresh, accuracy=stats))
        assert stats["finite"]
        if dtype == "bf16":
            assert stats["passed"], stats
        else:
            assert stats["relative_l2"] < 0.035, stats


def unsupported_target(api, monkeypatch, family, arch, runtime_version):
    """Reject unsupported targets before compiler checks or device launches."""
    import importlib
    from types import SimpleNamespace

    module = importlib.import_module("atrex.api.qsa")
    target = SimpleNamespace(family=family, arch=arch, runtime_version=runtime_version)
    monkeypatch.setattr(module, "detect_device_target", lambda device: target)

    def unexpected_compile():
        pytest.fail("unsupported targets must be rejected before compiler checks")

    monkeypatch.setattr(module, "_check_compiler", unexpected_compile)
    message = "CUDA 13" if arch in ("sm100", "sm103") else "NVIDIA SM100/SM103"
    with pytest.raises(NotImplementedError, match=message):
        api(torch.empty(1), None, None, None, None, None, None, False, None)


def negative_contract_and_capture(api):
    d = small_case()
    with pytest.raises(ValueError, match="share storage"):
        api(**d, out=d["q"])
    with pytest.raises(ValueError, match="positive multiple"):
        api(**{**d, "k_cache": d["k_cache"][:, :24], "v_cache": d["v_cache"][:, :24]})
    with pytest.raises(ValueError, match="1M tokens"):
        api(
            **{
                **d,
                "block_table": torch.zeros((3, 8193), device="cuda", dtype=torch.int32),
            }
        )
    f, _, _ = to_fp8(d)
    for scale in (0.0, -1.0, float("inf"), float("nan"), 1e-50, 1e50, True):
        with pytest.raises((TypeError, ValueError)):
            api(**f, sm_scale=scale)
    d["out"] = torch.empty_like(d["q"])
    graph = torch.cuda.CUDAGraph()
    with pytest.raises(NotImplementedError, match="graph capture"):
        with torch.cuda.graph(graph):
            api(**d)


def cache_releases_inputs(api):
    import weakref

    d = small_case(rows=66)
    refs = [weakref.ref(v) for v in d.values() if isinstance(v, torch.Tensor)]
    y = api(**d)
    refs.append(weakref.ref(y))
    torch.cuda.synchronize()
    del y, d
    gc.collect()
    assert all(r() is None for r in refs)


def warmed_contract_checks_live_metadata(api):
    d = decode_case(2, 1)
    d["out"] = torch.empty_like(d["q"])
    api(**d)
    torch.cuda.synchronize()
    d["q"].requires_grad_(True)
    with pytest.raises(ValueError, match="inference-only"):
        api(**d)
    d["q"].requires_grad_(False)
    with pytest.raises(ValueError, match="share storage"):
        api(**{**d, "out": d["q"]})
    ids = torch.empty((2, 1024), device="cuda", dtype=torch.int32)[:, ::2]
    with pytest.raises(ValueError, match="block IDs"):
        api(**{**d, "indexer_block_ids": ids})
    storage = torch.empty(d["q"].numel() + 1, device="cuda", dtype=torch.bfloat16)
    gate = storage[1:].view_as(d["q"])
    with pytest.raises(ValueError, match="gate must"):
        api(**{**d, "output_gate": gate})
    # Rebinding storage preserves shape/stride, but must invalidate alias checks.
    d["out"].set_(d["q"])
    with pytest.raises(ValueError, match="share storage"):
        api(**d)


def decode_same_layout_independent_streams(api):
    jobs = []
    for shift in (0.0, 0.25):
        d = decode_case(2, 1)
        d["q"].add_(shift)
        d["out"] = torch.empty_like(d["q"])
        expected = oracle(d, {})
        api(**d)
        stream = torch.cuda.Stream()
        jobs.append((d, expected, stream))
    torch.cuda.synchronize()
    for _ in range(3):
        for d, expected, stream in jobs:
            with torch.cuda.stream(stream):
                api(**d)
    torch.cuda.synchronize()
    for d, expected, stream in jobs:
        assert error(d["out"], expected)["passed"]


def decode_cached_scale_validation(api):
    d, _, _ = to_fp8(decode_case(2, 1))
    d["out"] = torch.empty(d["q"].shape, device="cuda", dtype=torch.bfloat16)
    api(**d, sm_scale=0.0625, v_scale=1.0)
    for value in (True, False, 0.0, -1.0, float("nan"), float("inf"), 1e-50, 1e50, []):
        with pytest.raises((TypeError, ValueError)):
            api(**d, sm_scale=0.0625, v_scale=value)


def strided_query_and_independent_streams(api):
    inputs = []
    streams = []
    outputs = []
    for shift in (0.0, 0.125):
        d = small_case(rows=67)
        storage = torch.empty((67, 8, 264), device="cuda", dtype=torch.bfloat16)
        d["q"] = storage[:, :, 8:].copy_(d["q"] + shift)
        assert d["q"].data_ptr() % 32 == 16
        d["query_positions"] = d["query_positions"].int()
        inputs.append(d)
        streams.append(torch.cuda.Stream())
    ready = torch.cuda.Event()
    ready.record()
    for d, stream in zip(inputs, streams):
        stream.wait_event(ready)
        with torch.cuda.stream(stream):
            outputs.append(api(**d))
    torch.cuda.synchronize()
    for d, y in zip(inputs, outputs):
        assert error(y, oracle(d, {}))["passed"]


def decode_capture_guards_and_independent_graphs(api, dtype):
    jobs = []
    for index in range(2):
        data = decode_case(7, 1, 128, 2)
        scales = {}
        kwargs = {}
        data["q"].mul_(1 + 0.25 * index)
        data["output_gate"].add_(index)
        if dtype == "fp8":
            data, scales, _ = to_fp8(data, consume=True)
            kwargs = {
                "sm_scale": 256**-0.5 * scales["q"] * scales["k_cache"],
                "v_scale": scales["v_cache"],
            }
        if index == 0:
            data.pop("out", None)
            with pytest.raises(RuntimeError, match="preallocated out"):
                with torch.cuda.graph(torch.cuda.CUDAGraph()):
                    api(**data, **kwargs)
            data["out"] = torch.empty(
                data["q"].shape, device="cuda", dtype=torch.bfloat16
            )
            with pytest.raises(RuntimeError, match="warm this Q1/Q4"):
                with torch.cuda.graph(torch.cuda.CUDAGraph()):
                    api(**data, **kwargs)
        else:
            data["out"] = torch.empty(
                data["q"].shape, device="cuda", dtype=torch.bfloat16
            )
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                api(**data, **kwargs)
        stream.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=stream):
            api(**data, **kwargs)
        jobs.append((data, scales, kwargs, graph))
    for data, scales, kwargs, graph in reversed(jobs):
        if dtype == "fp8":
            # An eager call with new scalar values must not change a captured graph.
            changed = dict(
                scales, q=scales["q"] * 1.25, v_cache=scales["v_cache"] * 0.75
            )
            y = api(
                **data,
                sm_scale=kwargs["sm_scale"] * 1.25,
                v_scale=kwargs["v_scale"] * 0.75,
            )
            assert error(y, oracle(data, changed))["relative_l2"] < 0.035
        data["indexer_block_ids"][:, ::19] = -1
        graph.replay()
        torch.cuda.synchronize()
        stats = error(data["out"], oracle(data, scales))
        assert stats["finite"], stats
        if dtype == "bf16":
            assert stats["passed"], stats
        else:
            assert stats["relative_l2"] < 0.035, stats


def decode_graph_uniform_attention(api, queries, batch, ratio, hkv, page, dtype):
    """Uniform attention exercises decode/MTP masks, gates and graph updates."""
    torch.manual_seed(103000 + batch + ratio + page)
    storage_dtype = torch.float8_e4m3fn if dtype == "fp8" else torch.bfloat16
    rows = queries * batch
    heads = ratio * hkv
    pages = 2 * batch
    token = torch.arange(pages * page, device="cuda").reshape(pages, page, 1, 1)
    head = torch.arange(hkv, device="cuda").reshape(1, 1, hkv, 1)
    column = torch.arange(256, device="cuda").reshape(1, 1, 1, 256)
    values = ((token % 7 + 2 * head + column % 3).float() / 4).to(storage_dtype)
    positions = ((torch.arange(batch, device="cuda") * 137) % (2 * page - 4))[:, None]
    positions = (positions + torch.arange(queries, device="cuda")[None, :]).reshape(-1)
    positions[0] = -1
    logical = 2 * page // 4
    count = ((positions + 1) // 4).clamp(0, 512)
    rank = torch.rand((rows, logical), device="cuda")
    rank.masked_fill_(
        torch.arange(logical, device="cuda")[None, :] >= count[:, None], 2.0
    )
    selected = rank.argsort(dim=1)[:, : min(512, logical)]
    ids = torch.full((rows, 512), -1, device="cuda", dtype=torch.int32)
    ids[:, : selected.shape[1]] = selected.int()
    ids.masked_fill_(torch.arange(512, device="cuda")[None, :] >= count[:, None], -1)
    data = dict(
        q=torch.zeros((rows, heads, 256), device="cuda", dtype=storage_dtype),
        k_cache=torch.zeros_like(values),
        v_cache=values,
        indexer_block_ids=ids,
        query_positions=positions,
        block_table=torch.randperm(pages, device="cuda").int().reshape(batch, 2),
        token_to_req=torch.arange(
            batch, device="cuda", dtype=torch.int32
        ).repeat_interleave(queries),
        use_prefill_config=False,
        output_gate=torch.zeros(
            (rows, heads, 256), device="cuda", dtype=torch.bfloat16
        ),
        out=torch.empty((rows, heads, 256), device="cuda", dtype=torch.bfloat16),
    )
    scales = {name: 1.0 for name in ("q", "k_cache", "v_cache")}
    gate_pattern = (
        torch.arange(rows * heads * 256, device="cuda").reshape(rows, heads, 256) % 65
        - 32
    ).float() / 4
    data["output_gate"].copy_(gate_pattern)
    api(**data)
    torch.cuda.synchronize()
    check(data["out"], oracle(data, scales))
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        api(**data)
    for replay in range(3):
        data["indexer_block_ids"][:, ::7] = -1
        data["token_to_req"].copy_(data["token_to_req"].roll(queries))
        data["output_gate"].copy_(
            gate_pattern.roll(7 * (replay + 1), dims=2) + (replay + 1) * 0.125
        )
        data["token_to_req"][-queries:] = -1
        data["out"].fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        check(data["out"], oracle(data, scales))
    graph.reset()


def profile_workload(architecture):
    api = installed_api(architecture)
    calls, retained = [], []
    for page, dtype, mode in CASES:
        data, scales, queries = _inputs(page, dtype, mode)
        _refresh(data, page, queries, 0)
        options = launch_options(scales)
        fn = lambda data=data, options=options: api(**data, **options)
        fn()
        if queries:
            captured = capture(fn)
            calls.append(captured[0].replay)
            # External CUDA Events must stay alive as long as their graph.
            retained.append((data, captured))
        else:
            calls.append(fn)
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStart()
    for fn in calls:
        fn()
    torch.cuda.synchronize()
    torch.cuda.cudart().cudaProfilerStop()
    for _, (graph, start, end) in retained:
        graph.reset()


def kernel_names(entry, output):
    nsys = shutil.which("nsys")
    assert nsys, "Nsight Systems is required for the kernel-name test"
    report = output / "qsa"
    subprocess.run(
        [
            nsys,
            "profile",
            "--trace=cuda",
            "--capture-range=cudaProfilerApi",
            "--capture-range-end=stop",
            "--cuda-graph-trace=node",
            "--sample=none",
            "--cpuctxsw=none",
            "--force-overwrite=true",
            "-o",
            str(report),
            sys.executable,
            entry,
            "--profile",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    database = report.with_suffix(".sqlite")
    subprocess.run(
        [
            nsys,
            "export",
            "--type=sqlite",
            "--force-overwrite=true",
            "-o",
            str(database),
            str(report.with_suffix(".nsys-rep")),
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    with sqlite3.connect(database) as connection:
        names = [
            r[0]
            for r in connection.execute(
                "SELECT s.value FROM CUPTI_ACTIVITY_KIND_KERNEL k "
                "JOIN StringIds s ON k.demangledName=s.id"
            )
        ]
    assert names and all(name.startswith("atrex_") for name in names), names
    (output / "kernel_names.json").write_text(json.dumps(names, indent=2))
    print({"kernel_events": len(names), "kernel_names": sorted(set(names))})


class WheelAPI:
    """Isolate installed module namespaces; activate outside measured calls."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.modules = {}

    def activate(self):
        for name in list(sys.modules):
            if name == "atrex" or name.startswith("atrex."):
                del sys.modules[name]
        sys.modules.update(self.modules)

    def __call__(self, **kwargs):
        self.activate()
        sys.path.insert(0, str(self.root))
        try:
            api = importlib.import_module("atrex")
            assert Path(api.__file__).resolve().is_relative_to(self.root), api.__file__
            return api.qsa_compact(**kwargs)
        finally:
            self.modules = {
                n: m
                for n, m in sys.modules.items()
                if n == "atrex" or n.startswith("atrex.")
            }
            sys.path.remove(str(self.root))


def identity(root, revision):
    root = Path(root).resolve()
    # Baselines before the API module rename still expose the same public API.
    api = root / "atrex/api/qsa.py"
    if not api.is_file():
        api = root / "atrex/api/qsa_compact.py"
    paths = [
        api,
        *sorted((root / "atrex/src/nvidia/qsa").rglob("*.py")),
    ]
    assert len(paths) > 1 and all(p.is_file() for p in paths), root
    hashes = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }
    return dict(
        revision=revision,
        root=str(root),
        source_digest=hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode()
        ).hexdigest(),
    )


def measure(apis, graphs, functions, queries, flush):
    timings = {n: {} for n in apis}
    for cache in ("warm", "cold_l2") if queries else ("eager",):
        samples = {n: [] for n in apis}
        for repeat in range(5):
            for sample in range(10):
                order = (
                    list(apis) if (repeat + sample) % 2 == 0 else list(reversed(apis))
                )
                for name in order:
                    if queries:
                        if cache == "cold_l2":
                            flush.zero_()
                            torch.cuda.synchronize()
                        graph, start, end = graphs[name]
                        graph.replay()
                        end.synchronize()
                    else:
                        apis[name].activate()
                        start, end = (
                            torch.cuda.Event(enable_timing=True) for _ in range(2)
                        )
                        start.record()
                        functions[name]()
                        end.record()
                        end.synchronize()
                    samples[name].append(start.elapsed_time(end) * 1000)
        for name in apis:
            timings[name][cache] = dict(
                median_us=statistics.median(samples[name]), samples_us=samples[name]
            )
    regression = {
        k: 100
        * (timings["current"][k]["median_us"] / timings["baseline"][k]["median_us"] - 1)
        for k in timings["current"]
    }
    return dict(timings=timings, regression_pct=regression)


def main(architecture):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regression", action="store_true", required=True)
    parser.add_argument("--current-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--baseline-revision", required=True)
    parser.add_argument("--current-revision", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", default="all")
    parser.add_argument("--reverse-capture", action="store_true")
    args = parser.parse_args()
    ids = (
        list(range(len(CASES)))
        if args.cases == "all"
        else [int(x) for x in args.cases.split(",")]
    )
    if (
        not ids
        or len(set(ids)) != len(ids)
        or any(i < 0 or i >= len(CASES) for i in ids)
    ):
        parser.error("--cases must contain unique indices from 0 to 7")
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert (
        torch.cuda.get_device_capability()
        == {"sm100": (10, 0), "sm103": (10, 3)}[architecture]
    )
    roots = dict(baseline=args.baseline_root, current=args.current_root)
    report = dict(
        architecture=architecture,
        device=str(torch.cuda.get_device_properties(0)),
        packages={n: identity(root, getattr(args, f"{n}_revision"))
                  for n, root in roots.items()},
        versions={
            n: importlib.metadata.version(n)
            for n in ("torch", "triton", "nvidia-cutlass-dsl", "apache-tvm-ffi")
        },
        measurement=dict(
            samples=50,
            order="per-sample alternating",
            cold_flush_bytes=256 * 1024 * 1024,
            reverse_capture=args.reverse_capture,
        ),
        cases=[],
        complete=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    failures = []
    flush = torch.empty(256 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    save()
    for case in ids:
        page, dtype, mode = CASES[case]
        print("CASE_START", case, page, dtype, mode, flush=True)
        row = dict(
            case=case, page=page, dtype=dtype, mode=mode, accuracy=[], complete=False
        )
        report["cases"].append(row)
        graphs = {}
        try:
            data, scales, queries = _inputs(page, dtype, mode)
            _refresh(data, page, queries, 0)
            kwargs = launch_options(scales)
            apis = {n: WheelAPI(root) for n, root in roots.items()}
            outputs = {n: torch.empty_like(data["out"]) for n in apis}
            functions = {}
            for name in reversed(apis) if args.reverse_capture else apis:
                api = apis[name]
                call = dict(data, out=outputs[name], **kwargs)
                api(**call)
                public = api.modules["atrex"].qsa_compact
                fn = lambda public=public, call=call: public(**call)
                functions[name] = fn
                if queries:
                    graphs[name] = capture(fn)
                else:
                    for _ in range(3):
                        fn()
                torch.cuda.synchronize()
            for revision in range(2):
                _refresh(data, page, queries, revision)
                expected = oracle(data, scales)
                metrics = {}
                for name, api in apis.items():
                    if queries:
                        graphs[name][0].replay()
                    else:
                        api(**dict(data, out=outputs[name], **kwargs))
                    torch.cuda.synchronize()
                    metrics[name] = error(outputs[name], expected)
                metrics["new_vs_old"] = error(outputs["current"], outputs["baseline"])
                metrics["bitwise_equal"] = torch.equal(
                    outputs["current"], outputs["baseline"]
                )
                row["accuracy"].append(metrics)
                save()
                assert all(metrics[n]["finite"] for n in apis), metrics
                assert metrics["new_vs_old"]["passed"], metrics
                if metrics["baseline"]["passed"]:
                    assert metrics["current"]["passed"], metrics
                assert (
                    metrics["current"]["relative_l2"]
                    <= metrics["baseline"]["relative_l2"] * 1.01 + 1e-7
                ), metrics
            row.update(measure(apis, graphs, functions, queries, flush))
            failures.extend(
                (case, metric, value)
                for metric, value in row["regression_pct"].items()
                if value > 5
            )
            row["complete"] = True
            print("CASE_COMPLETE", case, row["regression_pct"], flush=True)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            for graph, _, _ in graphs.values():
                graph.reset()
            save()
    report.update(
        complete=True, performance_pass=not failures, regressions_over5pct=failures
    )
    save()
    assert not failures, failures


@pytest.fixture
def api():
    return installed_api("sm103")


@pytest.mark.parametrize(
    "page,ratio", [(16, 8), (128, 8), (256, 12), (1056, 12), (2112, 8), (1056, 16)]
)
@pytest.mark.parametrize("dtype", ["bf16", "fp8"])
def test_prefill_reference(api, page, ratio, dtype):
    boundaries_and_metadata_refresh(api, page, ratio, dtype)


@pytest.mark.parametrize(
    "queries,batch,ratio,hkv,page,dtype",
    [
        (1, 3, 8, 1, 128, "bf16"),
        (1, 8, 12, 2, 16, "fp8"),
        (1, 128, 8, 1, 2112, "fp8"),
        (4, 3, 8, 1, 128, "fp8"),
        (4, 23, 8, 2, 256, "bf16"),
        (4, 64, 8, 1, 256, "fp8"),
        (4, 40, 12, 1, 1056, "bf16"),
        (4, 128, 8, 2, 2112, "fp8"),
        (1, 64, 16, 1, 2112, "bf16"),
        (1, 128, 16, 1, 2112, "fp8"),
        (4, 48, 16, 1, 1056, "bf16"),
        (4, 64, 16, 1, 128, "bf16"),
        (4, 40, 16, 1, 128, "fp8"),
        (4, 128, 16, 1, 2112, "fp8"),
    ],
)
def test_decode_graph(api, queries, batch, ratio, hkv, page, dtype):
    decode_graph_uniform_attention(
        api, queries, batch, ratio, hkv, page, dtype
    )


def test_invalid_pages(api):
    negative_pages(api)


@pytest.mark.parametrize(
    "family,arch,runtime_version",
    [("cpu", "cpu", None), ("nvidia", "sm90", "13.0"),
     ("nvidia", "sm100", "12.9"), ("nvidia", "sm103", None)],
)
def test_unsupported_target(api, monkeypatch, family, arch, runtime_version):
    unsupported_target(api, monkeypatch, family, arch, runtime_version)


def test_negative_contract_and_capture(api):
    negative_contract_and_capture(api)


def test_cache_releases_inputs(api):
    cache_releases_inputs(api)


def test_warmed_contract_checks_live_metadata(api):
    warmed_contract_checks_live_metadata(api)


def test_decode_independent_streams(api):
    decode_same_layout_independent_streams(api)


def test_decode_cached_scale_validation(api):
    decode_cached_scale_validation(api)


def test_strided_query_and_independent_streams(api):
    strided_query_and_independent_streams(api)


@pytest.mark.parametrize("dtype", ["bf16", "fp8"])
def test_capture_guards_and_independent_graphs(api, dtype):
    decode_capture_guards_and_independent_graphs(api, dtype)


def test_kernel_names(api, tmp_path):
    kernel_names(__file__, tmp_path)


@pytest.mark.parametrize("batch,hkv,ratio,page", [(23, 2, 8, 256)])
def test_strided_mtp_graph_metadata(api, batch, hkv, ratio, page, monkeypatch):
    """All KV partitions must use the same union after graph input updates."""
    import torch
    from atrex.src.nvidia.qsa.sm103 import runtime

    torch.manual_seed(20261109 + batch + page)
    rows = 4 * batch
    heads = ratio * hkv
    per_request = (8192 + page - 1) // page
    pages = batch * per_request
    length = per_request * page
    cache_shape = (pages, page, hkv, 256)
    cache_stride = (page * (hkv * 256 + 8), hkv * 256 + 8, 256, 1)
    k = torch.empty_strided(
        cache_shape, cache_stride, device="cuda", dtype=torch.bfloat16
    ).normal_()
    v = torch.empty_strided(
        cache_shape, cache_stride, device="cuda", dtype=torch.bfloat16
    ).normal_()
    ids = (
        torch.rand((rows, (length - 4) // 4), device="cuda")
        .argsort(dim=1)[:, :512]
        .int()
        .contiguous()
    )
    positions = (
        torch.full((batch, 1), length - 4, device="cuda", dtype=torch.int64)
        + torch.arange(4, device="cuda")[None, :]
    ).reshape(-1)
    q = torch.randn((rows, heads, 256), device="cuda", dtype=torch.bfloat16)
    data = dict(
        q=q,
        k_cache=k,
        v_cache=v,
        indexer_block_ids=ids,
        query_positions=positions,
        block_table=torch.randperm(pages, device="cuda")
        .int()
        .reshape(batch, per_request),
        token_to_req=torch.arange(
            batch, device="cuda", dtype=torch.int32
        ).repeat_interleave(4),
        use_prefill_config=False,
        output_gate=torch.randn_like(q),
        out=torch.empty_like(q),
    )
    compiled = []
    original_compile = runtime._compile

    def observed(function, args, specs):
        op = getattr(function, "__self__", None)
        if getattr(op, "deterministic_union", False):
            compiled.append((op.br, op.kv_splits))
        return original_compile(function, args, specs)

    monkeypatch.setattr(runtime, "_compile", observed)
    api(**data)
    torch.cuda.synchronize()
    assert (4, 3) in compiled
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        api(**data)
    for epoch in range(4):
        if epoch == 1:
            ids[:, ::17] = -1
            data["token_to_req"].copy_(data["token_to_req"].roll(4))
        elif epoch == 2:
            ids[::4] = -1
            positions[::4] = ((positions[::4] + 1) // 4) * 4 - 1
        elif epoch == 3:
            counts = torch.tensor(
                [0, 1, 7, 8, 9, 23, 24, 25, 63, 64, 65], device="cuda"
            )
            live = counts[torch.arange(rows, device="cuda") % len(counts)]
            ids.masked_fill_(
                torch.arange(512, device="cuda")[None, :] >= live[:, None], -1
            )
            positions.copy_(((positions + 1) // 4) * 4 - 1)
        expected = oracle(data, {})
        first = None
        for _ in range(2):
            data["out"].fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            check(data["out"], expected)
            if first is None:
                first = data["out"].clone()
            else:
                assert torch.equal(data["out"], first)
    graph.reset()


if __name__ == "__main__":
    if sys.argv[1:] == ["--profile"]:
        profile_workload("sm103")
    else:
        main("sm103")
