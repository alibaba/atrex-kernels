"""SM103 attention: public API, numerical/graph and poisoned-workspace regression.

Migrated from master FA4 tests, plus original-page/stride/descale contract cases.
Optional ATREX_FA_INVENTORY supplies a captured production inventory; the default
bounded workspace sweep below is synthetic and is not production performance evidence.
"""

from __future__ import annotations

import csv
import json
import math
import os
import weakref
from functools import partial
from pathlib import Path

import pytest
import torch


ATOL, RTOL = 3e-2, 5e-2


@pytest.fixture(scope="module", autouse=True)
def require_target():
    assert torch.cuda.is_available(), "SM103 GPU is required"
    assert torch.cuda.get_device_capability() == (10, 3)


@pytest.fixture(scope="module")
def rt():
    from atrex.src.nvidia.flash_attn.sm103 import decode_runtime
    return decode_runtime._fa4_decode_varlen, decode_runtime


def test_descale_type_identity():
    # CuTe DSL validates the annotated class, not just NamedTuple fields.
    from atrex.src.nvidia.flash_attn.sm103 import prefill_cutedsl, prefill_runtime
    assert prefill_runtime.DescaleTensors is prefill_cutedsl.DescaleTensors


# --------------------------------------------------------------------------- #
# Shared paged-cache builders and FP32 reference (varlen decode form).
# --------------------------------------------------------------------------- #
def build_decode_case(ctx_lens, q_lens, num_qo_heads, num_kv_heads, page_size, seed):
    torch.manual_seed(seed)
    batch = len(ctx_lens)
    npages = [math.ceil(c / page_size) for c in ctx_lens]
    key = torch.randn(
        sum(npages), page_size, num_kv_heads, 256,
        device="cuda", dtype=torch.bfloat16,
    )
    value = torch.randn_like(key)
    page_table = torch.zeros(
        batch, max(npages), device="cuda", dtype=torch.int32
    )
    off = 0
    for b, pc in enumerate(npages):
        page_table[b, :pc] = torch.arange(
            off, off + pc, device="cuda", dtype=torch.int32
        )
        off += pc
    cu = [0]
    for q in q_lens:
        cu.append(cu[-1] + q)
    query = torch.randn(
        cu[-1], num_qo_heads, 256, device="cuda", dtype=torch.bfloat16
    )
    cu_t = torch.tensor(cu, device="cuda", dtype=torch.int32)
    seqused = torch.tensor(ctx_lens, device="cuda", dtype=torch.int32)
    out = torch.empty_like(query)
    return query, key, value, page_table, cu_t, cu, seqused, out


def fp32_reference(query, key, value, page_table, ctx_lens, q_lens, cu):
    """Per-request FP32 paged attention over the varlen decode layout."""
    num_qo_heads = query.shape[1]
    num_kv_heads = key.shape[2]
    group = num_qo_heads // num_kv_heads
    page_size = key.shape[1]
    out = torch.empty_like(query)
    scale = 256**-0.5
    for b, ctx in enumerate(ctx_lens):
        ql = q_lens[b]
        if ql == 0:
            continue
        pc = math.ceil(ctx / page_size)
        pages = page_table[b, :pc].long()
        k = key[pages].reshape(-1, num_kv_heads, 256)[:ctx].float()
        v = value[pages].reshape(-1, num_kv_heads, 256)[:ctx].float()
        q = query[cu[b]: cu[b + 1]].float()
        q_pos = torch.arange(ql, device=q.device) + ctx - ql
        k_pos = torch.arange(ctx, device=q.device)
        mask = k_pos[None, :] > q_pos[:, None]
        for h in range(num_qo_heads):
            kh, vh = k[:, h // group], v[:, h // group]
            s = (q[:, h] @ kh.T) * scale
            s.masked_fill_(mask, -torch.inf)
            out[cu[b]: cu[b + 1], h] = (torch.softmax(s, -1) @ vh).to(
                torch.bfloat16
            )
    return out


def per_request_err(got, exp, q_lens, cu):
    return [
        (got[cu[b]: cu[b + 1]].float() - exp[cu[b]: cu[b + 1]].float())
        .abs().max().item()
        for b in range(len(q_lens))
    ]


def close_enough(got, exp) -> bool:
    return torch.allclose(got, exp, atol=ATOL, rtol=RTOL)


def ragged_patterns(batch: int, max_q: int) -> dict[str, list[int]]:
    """Query-length patterns that stress the prediction-tile overhang."""
    pats = {
        "uniform": [max_q] * batch,
        "cycle": [(i % max_q) + 1 for i in range(batch)],
        "first_short": [1] + [max_q] * (batch - 1),
        "last_short": [max_q] * (batch - 1) + [1],
        "alternating": [max_q if i % 2 == 0 else 1 for i in range(batch)],
        "all_one": [1] * batch,
        "descending": [max((max_q - i) % (max_q + 1), 1) for i in range(batch)],
    }
    if max_q >= 3:
        pats["mid_short"] = (
            [max_q] * (batch // 2) + [2] + [max_q] * (batch - batch // 2 - 1)
        )
    return {k: v for k, v in pats.items() if len(v) == batch}


def decode_kwargs(cu_t, seqused, max_q, page_table, out, page_size):
    return dict(
        cu_seqlens_q=cu_t,
        seqused_k=seqused,
        max_seqlen_q=max_q,
        max_seqlen_k=page_table.shape[1] * page_size,
        page_table=page_table,
        softmax_scale=256**-0.5,
        causal=True,
        return_lse=False,
        out=out,
    )


def gen_lengths(bs, splits, max_q, page_size):
    """Context lengths that leave some splits without a KV tile.

    ``short`` gives each request 1..splits-1 live tiles, so at least one split
    owns no tile (the bug's trigger); ``long`` covers every split, used to fill
    the whole workspace before a graph replay shrinks it back to ``short``.
    """
    max_live = max(1, splits - 1)
    short_lens = []
    for i in range(bs):
        live_tiles = 1 + (i % max_live)
        short_lens.append(max(1, live_tiles * page_size - 1 - (i % 64)))
    long_lens = [(splits + 2) * page_size - 13 * (i % 7) for i in range(bs)]
    q_lens = [max_q if i % 3 else max(max_q - 1, 1) for i in range(bs)]
    return short_lens, long_lens, q_lens


def dirty_free_block(decode_runtime, splits, batch, max_q, num_qo_heads):
    """Leave a NaN-filled block of the workspace's size in the allocator.

    The caching allocator hands a freed block of the same size to the next
    request, so this is what a first launch really sees.
    """
    nbytes = decode_runtime._decode_workspace_bytes(
        splits, batch, max_q, num_qo_heads
    )
    dirty = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
    dirty.fill_(0xFF)
    torch.cuda.synchronize()
    del dirty


def live_split_report(m_partial, q_lens):
    """Check every colmax row is defined and live splits form a prefix."""
    undefined = 0
    non_prefix = 0
    empty_request = 0
    live_counts: list[int] = []
    splits = m_partial.shape[0]
    for b, q_len in enumerate(q_lens):
        rows = m_partial[:, b, :q_len].reshape(splits, -1)
        finite = torch.isfinite(rows)
        marker = rows == -float("inf")
        undefined += int((~(finite | marker)).sum().item())
        live = finite.sum(dim=0)
        live_counts.append(int(live.max().item()))
        idx = torch.arange(splits, device=rows.device).unsqueeze(1)
        last_live = torch.where(
            finite, idx, torch.full_like(idx, -1)
        ).max(dim=0).values
        non_prefix += int((last_live + 1 != live).sum().item())
        empty_request += int((live == 0).sum().item())
    return (
        undefined,
        non_prefix,
        empty_request,
        min(live_counts),
        max(live_counts),
    )


# --------------------------------------------------------------------------- #
# Helpers for the migrated short-Q output/LSE case (separate paged layout).
# --------------------------------------------------------------------------- #
def _make_paged_kv(
    seq_lens: list[int],
    *,
    page_size: int = 128,
    num_kv_heads: int = 1,
    dtype: torch.dtype = torch.bfloat16,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    page_counts = [math.ceil(length / page_size) for length in seq_lens]
    total_pages = sum(page_counts)
    kv = torch.randn(
        total_pages, 2, page_size, num_kv_heads, 256,
        device="cuda", dtype=torch.bfloat16,
    )
    if dtype != torch.bfloat16:
        kv = kv.to(dtype)
    physical_pages = torch.randperm(total_pages, device="cuda")
    page_table = torch.empty(
        len(seq_lens), max(page_counts), device="cuda", dtype=torch.int32
    )
    offset = 0
    for row, page_count in enumerate(page_counts):
        page_table[row, :page_count] = physical_pages[
            offset : offset + page_count
        ].to(torch.int32)
        offset += page_count
    return kv[:, 0], kv[:, 1], page_table


def _gather_sequence(cache, page_table, row, length):
    page_count = math.ceil(length / cache.shape[1])
    pages = page_table[row, :page_count].long()
    return cache[pages].reshape(-1, cache.shape[2], cache.shape[3])[:length]


def _reference_output_lse(
    query, key, value, page_table, seq_lens, query_lens, scale
):
    total_q, num_qo_heads, _ = query.shape
    output = torch.empty(
        total_q, num_qo_heads, 256, device="cuda", dtype=torch.float32
    )
    lse = torch.empty(num_qo_heads, total_q, device="cuda", dtype=torch.float32)
    q_offset = 0
    group_size = num_qo_heads // key.shape[2]
    for row, (seq_len, query_len) in enumerate(zip(seq_lens, query_lens)):
        if query_len == 0:
            continue
        q = query[q_offset : q_offset + query_len].float()
        k = _gather_sequence(key, page_table, row, seq_len)
        v = _gather_sequence(value, page_table, row, seq_len)
        k = k.repeat_interleave(group_size, dim=1).float()
        v = v.repeat_interleave(group_size, dim=1).float()
        scores = torch.einsum("qhd,khd->hqk", q, k) * scale
        q_positions = torch.arange(query_len, device="cuda") + seq_len - query_len
        k_positions = torch.arange(seq_len, device="cuda")
        scores.masked_fill_(
            k_positions[None, :] > q_positions[:, None], -torch.inf
        )
        probabilities = torch.softmax(scores, dim=-1)
        output[q_offset : q_offset + query_len] = torch.einsum(
            "hqk,khd->qhd", probabilities, v
        )
        lse[:, q_offset : q_offset + query_len] = torch.logsumexp(scores, dim=-1)
        q_offset += query_len
    assert q_offset == total_q
    return output.to(torch.bfloat16), lse


def _assert_complete_close(
    output, lse, reference_output, reference_lse,
    *, output_atol, output_rtol, lse_atol, lse_rtol,
):
    assert torch.isfinite(output).all()
    assert torch.isfinite(lse).all()
    assert torch.isfinite(reference_output).all()
    assert torch.isfinite(reference_lse).all()
    torch.testing.assert_close(
        output, reference_output, atol=output_atol, rtol=output_rtol
    )
    torch.testing.assert_close(lse, reference_lse, atol=lse_atol, rtol=lse_rtol)


# --------------------------------------------------------------------------- #
# Inventory loaders for the poisoned-workspace sweep (16 qo / 1 kv head).
# --------------------------------------------------------------------------- #
def load_shapes(path: Path) -> list[dict]:
    with path.open(newline="") as handle:
        rows = [
            row
            for row in csv.DictReader(handle)
            if row["phase"] == "decode"
            and row["target_op"] == "fullattn_decode"
        ]
    for row in rows:
        row["bs"] = int(row["bs"])
        row["q_lens"] = json.loads(row["q_lens"])
        row["ctx_lens"] = json.loads(row["ctx_lens"])
    return rows


def make_pool(rows: list[dict], page_size: int):
    max_pages = max(
        sum(math.ceil(length / page_size) for length in row["ctx_lens"])
        for row in rows
    )
    torch.manual_seed(20260806)
    key = torch.randn(
        max_pages, page_size, 1, 256, device="cuda", dtype=torch.bfloat16
    )
    value = torch.randn_like(key)
    return key, value


def make_case(row: dict, key: torch.Tensor, value: torch.Tensor):
    bs = row["bs"]
    q_len = row["q_lens"][0]
    assert row["q_lens"] == [q_len] * bs
    page_size = key.shape[1]
    page_counts = [math.ceil(length / page_size) for length in row["ctx_lens"]]
    page_table = torch.zeros(
        bs, max(page_counts), device="cuda", dtype=torch.int32
    )
    offset = 0
    for batch_idx, page_count in enumerate(page_counts):
        page_table[batch_idx, :page_count] = torch.arange(
            offset, offset + page_count, device="cuda", dtype=torch.int32
        )
        offset += page_count
    query = torch.randn(
        bs * q_len, 16, 256, device="cuda", dtype=torch.bfloat16
    )
    cu_seqlens_q = torch.arange(
        0, (bs + 1) * q_len, q_len, device="cuda", dtype=torch.int32
    )
    seqused_k = torch.tensor(row["ctx_lens"], device="cuda", dtype=torch.int32)
    output = torch.empty_like(query)
    kwargs = dict(
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seqused_k,
        max_seqlen_q=q_len,
        max_seqlen_k=max(row["ctx_lens"]),
        page_table=page_table,
        causal=True,
        return_lse=False,
        out=output,
    )
    return query, key, value, output, kwargs


def inventory_reference(query, key, value, page_table, ctx_lens, q_len):
    output = torch.empty_like(query)
    scale = 256**-0.5
    for batch_idx, ctx_len in enumerate(ctx_lens):
        page_count = math.ceil(ctx_len / key.shape[1])
        pages = page_table[batch_idx, :page_count].long()
        k = key[pages].reshape(-1, 1, 256)[:ctx_len, 0].float()
        v = value[pages].reshape(-1, 1, 256)[:ctx_len, 0].float()
        q = query[batch_idx * q_len : (batch_idx + 1) * q_len].float()
        scores = torch.einsum("qhd,kd->hqk", q, k) * scale
        q_positions = torch.arange(q_len, device="cuda") + ctx_len - q_len
        k_positions = torch.arange(ctx_len, device="cuda")
        scores.masked_fill_(
            k_positions[None, :] > q_positions[:, None], -torch.inf
        )
        probabilities = torch.softmax(scores, dim=-1)
        output[batch_idx * q_len : (batch_idx + 1) * q_len] = torch.einsum(
            "hqk,kd->qhd", probabilities, v
        ).to(torch.bfloat16)
    return output


# --------------------------------------------------------------------------- #
# 1. short-Q eager + CUDA-Graph replay (migrated from tests/test_fa4_vllm_gpu).
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "fp8_e4m3"]
)
def test_p128_short_q_eager_and_graph_replay(
    rt, dtype: torch.dtype, monkeypatch: pytest.MonkeyPatch
) -> None:
    from atrex import flash_attn_varlen_func
    from atrex.src.nvidia.flash_attn.sm103 import launch

    def _reject_prefill_fallback(**kwargs):
        pytest.fail("supported BF16/FP8 short-Q must use the decode kernel")

    monkeypatch.setattr(launch, "_run_prefill", _reject_prefill_fallback)

    torch.manual_seed(20260806)
    query_lens = [0, 1, 2, 3, 4, 5]
    initial_seq_lens = [513] * len(query_lens)
    scale = 256**-0.5
    key, value, page_table = _make_paged_kv(initial_seq_lens, dtype=dtype)
    query = torch.randn(
        sum(query_lens), 16, 256, device="cuda", dtype=torch.bfloat16
    )
    if dtype != torch.bfloat16:
        query = query.to(dtype)
    cu_seqlens_q = torch.tensor(
        [0, 0, 1, 3, 6, 10, 15], device="cuda", dtype=torch.int32
    )
    seqused_k = torch.tensor(initial_seq_lens, device="cuda", dtype=torch.int32)
    output = torch.empty(query.shape, device="cuda", dtype=torch.bfloat16)
    kwargs = dict(
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seqused_k,
        max_seqlen_q=5,
        max_seqlen_k=max(initial_seq_lens),
        block_table=page_table,
        softmax_scale=scale,
        causal=True,
        return_softmax_lse=True,
        out=output,
    )

    actual_output, actual_lse = flash_attn_varlen_func(
        query, key, value, **kwargs
    )
    reference_output, reference_lse = _reference_output_lse(
        query, key, value, page_table, initial_seq_lens, query_lens, scale
    )
    assert actual_output is output
    lse = actual_lse
    _assert_complete_close(
        output, lse, reference_output, reference_lse,
        output_atol=3e-2, output_rtol=5e-2, lse_atol=2e-5, lse_rtol=2e-5,
    )

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        _, lse = flash_attn_varlen_func(query, key, value, **kwargs)

    replay_seq_lens = [257, 385, 513, 257, 385, 513]
    replay_query = torch.randn(query.shape, device="cuda", dtype=torch.bfloat16)
    if dtype != torch.bfloat16:
        replay_query = replay_query.to(dtype)
    query.copy_(replay_query)
    seqused_k.copy_(
        torch.tensor(replay_seq_lens, device="cuda", dtype=torch.int32)
    )
    page_table.copy_(page_table.roll(shifts=1, dims=1))
    graph.replay()
    torch.cuda.synchronize()
    reference_output, reference_lse = _reference_output_lse(
        query, key, value, page_table, replay_seq_lens, query_lens, scale
    )
    _assert_complete_close(
        output, lse, reference_output, reference_lse,
        output_atol=3e-2, output_rtol=5e-2, lse_atol=2e-5, lse_rtol=2e-5,
    )


# --------------------------------------------------------------------------- #
# 2. External reduction over empty splits on a poisoned/reused workspace.
#    Configs pick both reducer variants: splits==16 -> _Fa4DecodeReduce,
#    (heads16, splits 2/4) and (heads32, splits2) -> _Fa4DecodeReduceShared.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "hq,hk,splits",
    [
        pytest.param(16, 1, 16, id="h16k1s16-tiled"),
        pytest.param(32, 2, 16, id="h32k2s16-tiled"),
        pytest.param(16, 1, 4, id="h16k1s4-shared"),
        pytest.param(16, 1, 2, id="h16k1s2-shared"),
        pytest.param(32, 2, 2, id="h32k2s2-shared"),
    ],
)
def test_empty_split_poisoned_workspace(rt, monkeypatch, hq, hk, splits) -> None:
    decode, decode_runtime = rt
    page_size = 128
    monkeypatch.setattr(
        decode_runtime,
        "_vllm_decode_config",
        lambda *a, _s=splits, **k: (_s, "external"),
    )

    failures = []
    for bs in (16, 48):
        for max_q in (1, 2, 4):
            short_lens, long_lens, q_lens = gen_lengths(
                bs, splits, max_q, page_size
            )
            (query, key, value, page_table, cu_t, cu, seqused,
             out) = build_decode_case(
                long_lens, q_lens, hq, hk, page_size,
                seed=bs * 100 + max_q + splits,
            )
            kwargs = decode_kwargs(cu_t, seqused, max_q, page_table, out, page_size)
            seqused.copy_(
                torch.tensor(short_lens, device="cuda", dtype=torch.int32)
            )
            exp_short = fp32_reference(
                query, key, value, page_table, short_lens, q_lens, cu
            )

            # --- first launch on a workspace taken from dirty (NaN) memory ---
            dirty_free_block(decode_runtime, splits, bs, max_q, hq)
            out.zero_()
            # Observe a copy of the statistics, not a runtime-owned cache.
            observed_stats = []
            original_reduce = decode_runtime.fa4_decode_reduce_varlen_cutedsl

            def observe_reduce(o_partial, l_partial, m_partial, *args, **kw):
                observed_stats.append(m_partial.clone())
                return original_reduce(o_partial, l_partial, m_partial, *args, **kw)

            with monkeypatch.context() as patch:
                patch.setattr(
                    decode_runtime, "fa4_decode_reduce_varlen_cutedsl", observe_reduce
                )
                decode(query, key, value, **kwargs)
            torch.cuda.synchronize()
            dirty_err = max(per_request_err(out, exp_short, q_lens, cu))
            dirty_ok = close_enough(out, exp_short)

            assert len(observed_stats) == 1, "external reduction was not exercised"
            undefined, non_prefix, empty_request, _, _ = live_split_report(
                observed_stats.pop(), q_lens
            )
            rows_ok = undefined == 0 and non_prefix == 0 and empty_request == 0

            # --- capture at long context, replay after shrinking to short ---
            seqused.copy_(
                torch.tensor(long_lens, device="cuda", dtype=torch.int32)
            )
            run = lambda: decode(query, key, value, **kwargs)
            run()
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                run()
            graph.replay()  # every split now holds a long-context partial
            seqused.copy_(
                torch.tensor(short_lens, device="cuda", dtype=torch.int32)
            )
            out.zero_()
            graph.replay()
            torch.cuda.synchronize()
            shrink_err = max(per_request_err(out, exp_short, q_lens, cu))
            shrink_ok = close_enough(out, exp_short)
            del graph

            if not (dirty_ok and shrink_ok and rows_ok):
                failures.append(
                    f"bs={bs} maxq={max_q} undefined={undefined} "
                    f"non_prefix={non_prefix} empty={empty_request} "
                    f"dirty_err={dirty_err:.4f} shrink_err={shrink_err:.4f}"
                )

    assert not failures, (
        f"heads={hq}/{hk} splits={splits} 失败:\n  " + "\n  ".join(failures)
    )


# --------------------------------------------------------------------------- #
# 3. Ragged-Q full-output matrix across the three reduction paths.
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("reduction", ["dispatch", "atomic", "external"])
def test_ragged_q_matrix(rt, monkeypatch, reduction) -> None:
    decode, decode_runtime = rt
    real_cfg = decode_runtime._vllm_decode_config
    sm = torch.cuda.get_device_properties(0).multi_processor_count
    page_size, ctx = 128, 4096

    failures = []
    checked = 0
    for hq, hk in ((16, 1), (32, 2)):
        for bs in (16, 26):
            for max_q in (2, 4):
                base_splits, _ = real_cfg(bs, sm, max_q, hq, hk)
                splits = base_splits
                if reduction == "atomic" and splits & (splits - 1):
                    continue  # atomic needs a power-of-two cluster
                if reduction != "dispatch":
                    monkeypatch.setattr(
                        decode_runtime,
                        "_vllm_decode_config",
                        lambda *a, _s=splits, _r=reduction, **k: (_s, _r),
                    )
                for name, q_lens in ragged_patterns(bs, max_q).items():
                    ctx_lens = [
                        ctx - 37 * (i % 11) - (i % 3) for i in range(bs)
                    ]
                    (query, key, value, page_table, cu_t, cu, seqused,
                     out) = build_decode_case(
                        ctx_lens, q_lens, hq, hk, page_size,
                        seed=bs * 100 + max_q,
                    )
                    kwargs = decode_kwargs(
                        cu_t, seqused, max_q, page_table, out, page_size
                    )
                    exp = fp32_reference(
                        query, key, value, page_table, ctx_lens, q_lens, cu
                    )

                    out.zero_()
                    decode(query, key, value, **kwargs)
                    torch.cuda.synchronize()
                    eager_err = max(per_request_err(out, exp, q_lens, cu))
                    eager_ok = close_enough(out, exp)

                    run = lambda: decode(query, key, value, **kwargs)
                    run()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        run()
                    q2 = list(reversed(q_lens))
                    cu2 = [0]
                    for q in q2:
                        cu2.append(cu2[-1] + q)
                    cu_t.copy_(
                        torch.tensor(cu2, device="cuda", dtype=torch.int32)
                    )
                    query.copy_(torch.randn_like(query))
                    out.zero_()
                    graph.replay()
                    torch.cuda.synchronize()
                    exp2 = fp32_reference(
                        query, key, value, page_table, ctx_lens, q2, cu2
                    )
                    replay_err = max(per_request_err(out, exp2, q2, cu2))
                    replay_ok = close_enough(out, exp2)
                    del graph
                    cu_t.copy_(
                        torch.tensor(cu, device="cuda", dtype=torch.int32)
                    )

                    checked += 1
                    if not (eager_ok and replay_ok):
                        failures.append(
                            f"heads={hq}/{hk} bs={bs} maxq={max_q} "
                            f"pattern={name} splits={splits} "
                            f"eager_err={eager_err:.4f} "
                            f"replay_err={replay_err:.4f}"
                        )

    if reduction == "atomic" and checked == 0:
        pytest.fail("no power-of-two split schedule was exercised")
    assert not failures, (
        f"reduction={reduction} 失败:\n  " + "\n  ".join(failures)
    )


# --------------------------------------------------------------------------- #
# 4. Poisoned-workspace sweep over the real inventory decode shapes.
#    Sampled (every Nth shape) so it stays CI-fast; the full 609-shape sweep is
#    op_test/stress_fa4_decode_poison_inventory.py if kept locally.
# --------------------------------------------------------------------------- #
def test_poison_inventory(rt, monkeypatch) -> None:
    decode, decode_runtime = rt
    capacities = []
    original_slice = decode_runtime._slice_workspace

    def observe_slice(workspace, *args):
        capacities.append(workspace.numel())
        return original_slice(workspace, *args)

    monkeypatch.setattr(decode_runtime, "_slice_workspace", observe_slice)
    inventory = os.environ.get("ATREX_FA_INVENTORY")
    if inventory:
        rows = load_shapes(Path(inventory))
        assert rows, "The supplied inventory contains no decode cases"
    else:
        rows = [
            dict(case_id=f"synthetic-b{bs}-q{ql}-k{kl}", bs=bs,
                 q_lens=[ql] * bs,
                 ctx_lens=[kl - i % 17 for i in range(bs)])
            for bs in (16, 48) for ql in (1, 2, 4) for kl in (513, 4096)
        ]
    step = max(1, len(rows) // 40)
    rows = rows[::step]
    key, value = make_pool(rows, page_size=128)

    failures = []
    poisoned = 0
    for row in rows:
        capacities.clear()
        q_len = row["q_lens"][0]
        query, k, v, out, kwargs = make_case(row, key, value)
        decode(query, k, v, **kwargs)
        torch.cuda.synchronize()
        expected = inventory_reference(
            query, k, v, kwargs["page_table"], row["ctx_lens"], q_len
        )

        if capacities:
            assert len(capacities) == 1
            nbytes = capacities[0]
            dirty = torch.empty(nbytes, dtype=torch.uint8, device="cuda")
            dirty.fill_(0xFF)
            torch.cuda.synchronize()
            del dirty
            poisoned += 1

        out.zero_()
        decode(query, k, v, **kwargs)
        torch.cuda.synchronize()
        if int(out.isnan().sum().item()) or not close_enough(
            out.float(), expected.float()
        ):
            err = (out.float() - expected.float()).abs().max().item()
            failures.append(f"{row['case_id']} q={q_len} err={err:.4f}")

    assert poisoned > 0, "no external workspace was exercised"
    assert not failures, (
        f"{len(failures)}/{len(rows)} shapes 失败:\n  " + "\n  ".join(failures)
    )


def test_external_scratch_not_retained(rt, monkeypatch):
    from op_test.utils.flash_attn_contract import check_eager_scratch_reuse

    decode, runtime = rt
    monkeypatch.setattr(runtime, "_vllm_decode_config", lambda *a, **k: (4, "external"))
    original_slice = runtime._slice_workspace
    scratch_refs = []

    def observe_slice(workspace, *args):
        # Weak references do not change the lifetime under test.
        scratch_refs.append(weakref.ref(workspace))
        return original_slice(workspace, *args)

    monkeypatch.setattr(runtime, "_slice_workspace", observe_slice)
    calls = []
    for layer in range(4):
        layer_calls = []
        for q_len in (1, 4):
            q, k, v, table, cu_t, _, seq, out = build_decode_case(
                [513] * 16, [q_len] * 16, 16, 1, 128, seed=layer + q_len,
            )
            kwargs = decode_kwargs(cu_t, seq, q_len, table, out, 128)
            layer_calls.append(partial(decode, q, k, v, **kwargs))
        calls.append(layer_calls)
    check_eager_scratch_reuse(calls)
    assert scratch_refs, "external reduction was not exercised"
    assert all(ref() is None for ref in scratch_refs), "runtime retained scratch"


@pytest.mark.parametrize(
    "dtype", [torch.bfloat16, torch.float8_e4m3fn], ids=["bf16", "fp8_e4m3"]
)
def test_p128_prefill_output_lse_and_out_contract(
    dtype: torch.dtype
) -> None:
    from atrex import flash_attn_varlen_func

    torch.manual_seed(20260805)
    seq_lens = [129, 257]
    query_lens = [17, 33]
    scale = 256**-0.5
    key, value, page_table = _make_paged_kv(seq_lens, dtype=dtype)
    query = torch.randn(
        sum(query_lens), 16, 256, device="cuda", dtype=torch.bfloat16
    )
    if dtype != torch.bfloat16:
        query = query.to(dtype)
    cu_seqlens_q = torch.tensor(
        [0, query_lens[0], sum(query_lens)], device="cuda", dtype=torch.int32
    )
    seqused_k = torch.tensor(seq_lens, device="cuda", dtype=torch.int32)
    output = torch.empty(query.shape, device="cuda", dtype=torch.bfloat16)

    actual_output, actual_lse = flash_attn_varlen_func(
        query,
        key,
        value,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seqused_k,
        max_seqlen_q=max(query_lens),
        max_seqlen_k=max(seq_lens),
        block_table=page_table,
        softmax_scale=scale,
        causal=True,
        return_softmax_lse=True,
        out=output,
    )
    reference_output, reference_lse = _reference_output_lse(
        query, key, value, page_table, seq_lens, query_lens, scale
    )

    assert actual_output is output
    lse = actual_lse
    _assert_complete_close(
        output,
        lse,
        reference_output,
        reference_lse,
        output_atol=3e-2,
        output_rtol=5e-2,
        lse_atol=2e-3,
        lse_rtol=2e-3,
    )


from op_test.utils.flash_attn_contract import build_case, check_case, profile_case


def test_public_eligibility_and_fail_fast():
    from atrex import can_use_flash_attn_varlen_func, flash_attn_varlen_func

    case = build_case(128, [65, 3], [257, 513], hq=16)
    case["fa_version"] = 4
    assert can_use_flash_attn_varlen_func(**case)
    for update, match in (
        ({"fa_version": 3}, "fa_version"),
        ({"causal": False}, "causal"),
    ):
        kwargs = dict(case, **update)
        assert not can_use_flash_attn_varlen_func(**kwargs)
        with pytest.raises((NotImplementedError, ValueError), match=match):
            flash_attn_varlen_func(**kwargs)
    for bad_out in (
        torch.empty_like(case["q"]),
        torch.empty(
            (case["q"].shape[0] + 1, *case["q"].shape[1:]),
            dtype=torch.bfloat16,
            device="cuda",
        ),
        torch.empty(case["q"].shape, dtype=torch.bfloat16),
    ):
        kwargs = dict(case, out=bad_out)
        assert not can_use_flash_attn_varlen_func(**kwargs)
        with pytest.raises(ValueError, match="out"):
            flash_attn_varlen_func(**kwargs)


@pytest.mark.parametrize("page_size", [16, 32, 64, 128, 256])
@pytest.mark.parametrize("layout", ["interleaved", "strided"])
@pytest.mark.parametrize("fp8", [False, True])
@pytest.mark.parametrize("q_lens,kv_lens", [
    ([1, 1, 1], [127, 513, 1025]),
    ([4, 2, 1], [128, 511, 1024]),
    ([65, 129, 3], [65, 257, 513]),
])
def test_paged_contract(page_size, layout, fp8, q_lens, kv_lens):
    check_case(build_case(page_size, q_lens, kv_lens, layout=layout, fp8=fp8, hq=16))


@pytest.mark.parametrize("q_lens", [[1, 1], [65, 3]])
def test_hybrid_model_page_contract(q_lens):
    # Additional direct-API coverage of the hybrid allocator's logical block.
    # The model runner actually subdivides this into FA kernel blocks of 128.
    # Cross a page boundary and consume the original strided cache directly.
    check_case(build_case(4224, q_lens, [4225, 8451],
                          layout="strided", fp8=True, hq=32, hkv=2))


@pytest.mark.parametrize("q_lens", [[1, 1], [4, 2], [65, 3]])
def test_graph_replay(q_lens):
    check_case(build_case(128, q_lens, [257, 513], hq=16), graph=True)


@pytest.mark.parametrize("q_lens", [[1, 1], [4, 2], [65, 3]])
@pytest.mark.parametrize("fp8", [False, True])
def test_real_kernel_events(q_lens, fp8):
    profile_case(build_case(128, q_lens, [257, 513], hq=16, fp8=fp8))


@pytest.mark.parametrize("hq,hkv", [(8, 1), (16, 1), (32, 2)])
@pytest.mark.parametrize("q_lens", [[1], [4, 2], [65, 3]])
@pytest.mark.parametrize("broadcast_scales", [False, True])
def test_gqa_descale_layout(hq, hkv, q_lens, broadcast_scales):
    case = build_case(128, q_lens, [257, 513][:len(q_lens)], hq=hq, hkv=hkv)
    if broadcast_scales:
        for name in ("q_descale", "k_descale", "v_descale"):
            case[name] = case[name][0, 0].expand(len(q_lens), hkv)
    check_case(case)


def _benchmark_revision(api_contract, output_path):
    """Private migration-parity mode; execute identically for two installs.

    Only the intentional public-contract rename/return change is adapted.
    Identity descales keep this within the unmodified master's supported ABI.
    Shapes are representative of the requested model, not captured production
    tensors. Compilation and graph capture are excluded from timing.
    """
    import statistics
    import time
    import atrex
    from op_test.utils.flash_attn_contract import reference

    assert torch.cuda.get_device_capability() == (10, 3)
    cases = [(1, 256, 256), (1, 4096, 4096),
             (1, 1, 4096), (8, 1, 4096), (32, 1, 4096), (128, 1, 4096)]
    records = []
    for fp8 in (False, True):
        for batch, nq, nk in cases:
            case = build_case(128, [nq]*batch, [nk]*batch, hq=32, hkv=2, fp8=fp8)
            for key in ('q_descale', 'k_descale', 'v_descale'):
                case[key] = None
            out = torch.empty_like(case['q'], dtype=torch.bfloat16)
            kwargs = dict(case, out=out, num_splits=1)
            if api_contract == 'legacy':
                kwargs['page_table'] = kwargs.pop('block_table')
                kwargs.pop('fa_version', None)
                kwargs.pop('window_size', None)
                kwargs['return_lse'] = False
                kwargs['softcap'] = None
            else:
                kwargs['return_softmax_lse'] = False
            def invoke():
                result = atrex.flash_attn_varlen_func(**kwargs)
                return result[0] if api_contract == 'legacy' else result
            actual = invoke()
            expected, _ = reference(case)
            torch.testing.assert_close(actual.float(), expected, atol=ATOL, rtol=RTOL)
            stream = torch.cuda.Stream()
            stream.wait_stream(torch.cuda.current_stream())
            with torch.cuda.stream(stream):
                for _ in range(5):
                    invoke()
            torch.cuda.current_stream().wait_stream(stream)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                for _ in range(20):
                    invoke()
            for _ in range(5):
                graph.replay()
            torch.cuda.synchronize()
            warm_until = time.monotonic() + .5
            while time.monotonic() < warm_until:
                graph.replay()
                torch.cuda.synchronize()
            samples = []
            for _ in range(31):
                start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start.record()
                graph.replay()
                end.record()
                end.synchronize()
                samples.append(start.elapsed_time(end) / 20)
            record = dict(fp8=fp8, batch=batch, q=nq, kv=nk, median_ms=statistics.median(samples),
                          samples_ms=samples)
            records.append(record)
            print({k:v for k,v in record.items() if k != 'samples_ms'}, flush=True)
    with Path(output_path).open('x') as handle:
        json.dump(dict(atrex_path=atrex.__file__, contract=api_contract,
                       device=torch.cuda.get_device_name(), torch_version=torch.__version__,
                       cases=records), handle, indent=2)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark-contract', choices=['legacy', 'current'], required=True)
    parser.add_argument('--benchmark-output', required=True)
    args = parser.parse_args()
    _benchmark_revision(args.benchmark_contract, args.benchmark_output)
