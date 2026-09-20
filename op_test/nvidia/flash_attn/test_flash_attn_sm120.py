"""Tests for the SM120 (Blackwell GeForce / L20N) hd256 CuTeDSL flash-attention op.

Covers the full_attention layers of the Qwen3.5 hybrid models on sm120:
  - Qwen3.5-35B : q16 / kv2 / hd256 (QPK=8)
  - Qwen3.5-27B : q24 / kv4 / hd256 (QPK=6)
Exercises BF16 and FP8 inputs through the public `atrex.flash_attn_varlen_func` entry, which dispatches SM120 to
the vendored SM80-class kernel. Correctness is checked against an fp32 reference
from the same inputs. BF16 -> cos>0.999, rel_l1<0.01 (cosine is the primary gate).

Exercises: prefill (causal varlen), paged decode (packed-GQA batched + SplitKV), and the ragged
decode compact-grid path.

Run directly:
    python op_test/nvidia/flash_attn/test_flash_attn_sm120.py
"""

import math

import pytest
import torch


def _requires_sm120():
    assert torch.cuda.is_available(), "SM120 GPU is required"
    assert torch.cuda.get_device_capability() == (12, 0)
    from atrex import flash_attn_varlen_func  # noqa: F401


def _metrics(o, ref):
    d = o - ref
    rel_l1 = (d.abs().sum() / ref.abs().sum()).item()
    cos = torch.nn.functional.cosine_similarity(o.flatten(), ref.flatten(), dim=0).item()
    return rel_l1, cos, d.abs().max().item()


def test_sm120_eager_scratch_not_retained_per_layer_or_shape():
    from functools import partial
    from atrex import flash_attn_varlen_func
    from op_test.utils.flash_attn_contract import build_case, check_eager_scratch_reuse

    _requires_sm120()
    calls = []
    for _ in range(4):
        layer = []
        for q_lens, kv_lens, splits in (
            ([65, 3], [1025, 513], 1),
            ([4, 2], [513, 257], 4),
        ):
            case = build_case(64, q_lens, kv_lens, hq=16, hkv=2)
            out = torch.empty_like(case["q"], dtype=torch.bfloat16)
            layer.append(partial(flash_attn_varlen_func, **case,
                                 out=out, num_splits=splits))
        calls.append(layer)
    check_eager_scratch_reuse(calls)


@pytest.mark.parametrize(
    "total_q,hq,hkv,max_q,max_k,expected",
    [
        (1024, 16, 2, 1024, 65536, 8),
        (1024, 16, 2, 1024, 32768, 4),
        (2048, 16, 2, 2048, 65536, 4),
        (4096, 16, 2, 4096, 65536, 1),
        (1024, 16, 2, 1024, 1024, 1),
        (1024, 24, 4, 1024, 65536, 4),
    ],
)
def test_sm120_fp8_prefill_split_policy(total_q, hq, hkv, max_q, max_k, expected):
    from atrex.src.nvidia.flash_attn.sm120.launch import _sm120_fp8_prefill_splits

    assert _sm120_fp8_prefill_splits(
        total_q,
        hq,
        hkv,
        max_q,
        max_k,
        110,
        pack_gqa=True,
    ) == expected


@pytest.mark.parametrize(
    "batch,hkv,q_len,expected",
    [
        (1, 2, 1, 96),
        (4, 1, 1, 50),
        (4, 1, 4, 48),
        (16, 1, 1, 12),
        (64, 1, 1, 3),
    ],
)
def test_sm120_fp8_decode_split_policy(batch, hkv, q_len, expected):
    from atrex.src.nvidia.flash_attn.sm120.launch import _sm120_short_q_splits

    assert _sm120_short_q_splits(
        batch,
        hkv,
        200_000,
        110,
        tile_n=32,
        page_size=64,
        query_length=q_len,
        is_fp8=True,
    ) == expected


def _fp32_prefill_ref(q, k, v, cu, seqs, scale):
    """Per-batch causal softmax attention in fp32."""
    HQ, HKV = q.shape[1], k.shape[1]
    rep = HQ // HKV
    dev = q.device
    out = torch.empty(q.shape[0], HQ, q.shape[2], device=dev, dtype=torch.float32)
    for b in range(len(seqs)):
        s, e = cu[b].item(), cu[b + 1].item()
        L = e - s
        qf = q[s:e].float()
        kf = k[s:e].float().repeat_interleave(rep, dim=1)
        vf = v[s:e].float().repeat_interleave(rep, dim=1)
        sc = torch.einsum("ihd,jhd->hij", qf, kf) * scale
        mask = torch.triu(torch.ones(L, L, device=dev, dtype=torch.bool), 1)
        p = sc.masked_fill(mask, float("-inf")).softmax(-1)
        out[s:e] = torch.einsum("hij,jhd->ihd", p, vf)
    return out


def _fp32_paged_decode_ref(q, k_cache, v_cache, seqused_k, page_table, page_size, scale):
    """fp32 decode reference (one query token per sequence) from a paged KV cache."""
    nd, HQ, D = q.shape
    HKV = k_cache.shape[2]
    rep = HQ // HKV
    dev = q.device
    out = torch.empty(nd, HQ, D, device=dev, dtype=torch.float32)
    for i in range(nd):
        L = int(seqused_k[i].item())
        npages = (L + page_size - 1) // page_size
        ks, vs = [], []
        for p in range(npages):
            pg = int(page_table[i, p].item())
            take = min(page_size, L - p * page_size)
            ks.append(k_cache[pg, :take].float())
            vs.append(v_cache[pg, :take].float())
        kf = torch.cat(ks, 0).repeat_interleave(rep, dim=1)   # (L, HQ, D)
        vf = torch.cat(vs, 0).repeat_interleave(rep, dim=1)
        qf = q[i].float()                                     # (HQ, D)
        sc = torch.einsum("hd,jhd->hj", qf, kf) * scale       # (HQ, L)
        p = sc.softmax(-1)
        out[i] = torch.einsum("hj,jhd->hd", p, vf)
    return out


def _fp32_paged_prefill_ref(
    q, k_cache, v_cache, cu_q, qlens, kvlens, page_table, page_size,
    scale, q_descale, k_descale, v_descale,
):
    """Causal GQA reference with per-sequence/per-KV-head FP8 descales."""
    hq, hkv = q.shape[1], k_cache.shape[2]
    group = hq // hkv
    outputs = []
    for batch_idx, (q_len, kv_len) in enumerate(zip(qlens, kvlens)):
        q_begin = int(cu_q[batch_idx].item())
        qf = q[q_begin:q_begin + q_len].float()
        qf *= q_descale[batch_idx].repeat_interleave(group).reshape(1, hq, 1)
        num_pages = (kv_len + page_size - 1) // page_size
        page_ids = page_table[batch_idx, :num_pages].long()
        kf = k_cache[page_ids].reshape(-1, hkv, q.shape[-1])[:kv_len].float()
        vf = v_cache[page_ids].reshape(-1, hkv, q.shape[-1])[:kv_len].float()
        kf *= k_descale[batch_idx].reshape(1, hkv, 1)
        vf *= v_descale[batch_idx].reshape(1, hkv, 1)
        kf = kf.repeat_interleave(group, dim=1)
        vf = vf.repeat_interleave(group, dim=1)
        scores = torch.einsum("ihd,jhd->hij", qf, kf) * scale
        q_idx = torch.arange(q_len, device=q.device)[:, None]
        k_idx = torch.arange(kv_len, device=q.device)[None, :]
        scores.masked_fill_(k_idx > q_idx + kv_len - q_len, float("-inf"))
        outputs.append(torch.einsum("hij,jhd->ihd", scores.softmax(-1), vf))
    return torch.cat(outputs)


# ----------------------------------------------------------------------------- prefill
@pytest.mark.parametrize("HQ,HKV", [(16, 2), (24, 4)])   # 35B, 27B
@pytest.mark.parametrize("seqs", [[512], [256, 1024], [1, 4000, 900]])
def test_sm120_prefill_bf16(HQ, HKV, seqs):
    _requires_sm120()
    from atrex import flash_attn_varlen_func
    torch.manual_seed(0)
    dev, D = "cuda", 256
    scale = 1.0 / math.sqrt(D)
    total = sum(seqs)
    q = torch.randn(total, HQ, D, device=dev, dtype=torch.bfloat16)
    k = torch.randn(total, HKV, D, device=dev, dtype=torch.bfloat16)
    v = torch.randn(total, HKV, D, device=dev, dtype=torch.bfloat16)
    cu = torch.tensor([0] + torch.tensor(seqs).cumsum(0).tolist(), device=dev, dtype=torch.int32)
    o = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu,
        cu_seqlens_k=cu,
        max_seqlen_q=max(seqs),
        max_seqlen_k=max(seqs),
        softmax_scale=scale,
        causal=True,
        num_splits=0,
    )
    o = o.float()
    ref = _fp32_prefill_ref(q, k, v, cu, seqs, scale)
    rel_l1, cos, max_abs = _metrics(o, ref)
    assert cos > 0.999, f"HQ/HKV={HQ}/{HKV} seqs={seqs} cos={cos} max_abs={max_abs}"
    assert rel_l1 < 0.01, f"rel_l1={rel_l1} max_abs={max_abs}"


@pytest.mark.parametrize("dtype,min_cos,max_rel_l1", [
    (torch.float8_e4m3fn, 0.999, 0.04),
    (torch.float8_e5m2, 0.998, 0.07),
])
@pytest.mark.parametrize("page_size", [64, 128])
def test_sm120_paged_prefill_fp8(dtype, min_cos, max_rel_l1, page_size):
    _requires_sm120()
    from atrex import flash_attn_varlen_func

    torch.manual_seed(9)
    dev, hq, hkv, dim = "cuda", 16, 2, 256
    qlens, kvlens = [65, 97], [192, 320]
    page_counts = [(length + page_size - 1) // page_size for length in kvlens]
    q = (torch.randn(sum(qlens), hq, dim, device=dev) * 0.5).to(dtype)
    k = (torch.randn(sum(page_counts), page_size, hkv, dim, device=dev) * 0.5).to(dtype)
    v = (torch.randn(sum(page_counts), page_size, hkv, dim, device=dev) * 0.5).to(dtype)
    cu_q = torch.tensor([0, qlens[0], sum(qlens)], device=dev, dtype=torch.int32)
    cu_k = torch.tensor([0, kvlens[0], sum(kvlens)], device=dev, dtype=torch.int32)
    page_table = torch.zeros(2, max(page_counts), device=dev, dtype=torch.int32)
    page_begin = 0
    for batch_idx, count in enumerate(page_counts):
        page_table[batch_idx, :count] = torch.arange(
            page_begin, page_begin + count, device=dev, dtype=torch.int32
        )
        page_begin += count
    q_descale = torch.tensor([[0.7, 1.1], [0.9, 0.8]], device=dev)
    k_descale = torch.tensor([[0.8, 1.2], [0.6, 1.3]], device=dev)
    v_descale = torch.tensor([[0.6, 1.3], [1.2, 0.7]], device=dev)
    scale = 1.0 / math.sqrt(dim)

    out = flash_attn_varlen_func(
        q, k, v,
        cu_seqlens_q=cu_q,
        seqused_k=cu_k[1:] - cu_k[:-1],
        block_table=page_table,
        max_seqlen_q=max(qlens),
        max_seqlen_k=page_table.shape[1] * page_size,
        softmax_scale=scale,
        causal=True,
        num_splits=0,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    assert out.dtype == torch.bfloat16
    assert torch.isfinite(out).all()
    ref = _fp32_paged_prefill_ref(
        q, k, v, cu_q, qlens, kvlens, page_table, page_size,
        scale, q_descale, k_descale, v_descale,
    )
    rel_l1, cos, max_abs = _metrics(out.float(), ref)
    assert cos > min_cos, f"dtype={dtype} page={page_size} cos={cos} max_abs={max_abs}"
    assert rel_l1 < max_rel_l1, f"dtype={dtype} page={page_size} rel_l1={rel_l1}"


@pytest.mark.parametrize("page_size", [64, 128])
@pytest.mark.parametrize("q_len,kv_len", [(256, 256), (256, 4096)])
def test_sm120_paged_prefill_fp8_auto_schedule(page_size, q_len, kv_len):
    _requires_sm120()
    from atrex import flash_attn_varlen_func

    torch.manual_seed(11)
    dev, hq, hkv, dim = "cuda", 16, 2, 256
    num_pages = kv_len // page_size
    q = (torch.randn(q_len, hq, dim, device=dev) * 0.5).to(torch.float8_e4m3fn)
    k = (
        torch.randn(num_pages, page_size, hkv, dim, device=dev) * 0.5
    ).to(torch.float8_e4m3fn)
    v = (
        torch.randn(num_pages, page_size, hkv, dim, device=dev) * 0.5
    ).to(torch.float8_e4m3fn)
    cu_q = torch.tensor([0, q_len], device=dev, dtype=torch.int32)
    cu_k = torch.tensor([0, kv_len], device=dev, dtype=torch.int32)
    page_table = torch.arange(num_pages, device=dev, dtype=torch.int32).view(1, -1)
    descale = torch.ones(1, hkv, device=dev)
    scale = 1.0 / math.sqrt(dim)

    out = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        seqused_k=cu_k[1:] - cu_k[:-1],
        block_table=page_table,
        max_seqlen_q=q_len,
        max_seqlen_k=kv_len,
        softmax_scale=scale,
        causal=True,
        num_splits=0,
        q_descale=descale,
        k_descale=descale,
        v_descale=descale,
    )
    ref = _fp32_paged_prefill_ref(
        q,
        k,
        v,
        cu_q,
        [q_len],
        [kv_len],
        page_table,
        page_size,
        scale,
        descale,
        descale,
        descale,
    )
    rel_l1, cos, _ = _metrics(out.float(), ref)
    assert cos > 0.999
    assert rel_l1 < 0.04

    graph_out = torch.empty_like(out)
    flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        seqused_k=cu_k[1:] - cu_k[:-1],
        block_table=page_table,
        max_seqlen_q=q_len,
        max_seqlen_k=kv_len,
        softmax_scale=scale,
        causal=True,
        num_splits=0,
        q_descale=descale,
        k_descale=descale,
        v_descale=descale,
        out=graph_out,
    )
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay_out = flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_q,
            seqused_k=cu_k[1:] - cu_k[:-1],
            block_table=page_table,
            max_seqlen_q=q_len,
            max_seqlen_k=kv_len,
            softmax_scale=scale,
            causal=True,
            num_splits=0,
            q_descale=descale,
            k_descale=descale,
            v_descale=descale,
            out=graph_out,
        )
    graph.replay()
    torch.cuda.synchronize()
    replay_rel_l1, replay_cos, _ = _metrics(replay_out.float(), ref)
    assert replay_cos > 0.999
    assert replay_rel_l1 < 0.04


def test_sm120_paged_prefill_fp8_clc_cuda_graph():
    """CLC is graph-safe and bitwise matches the unchanged static path."""
    _requires_sm120()
    from atrex.src.nvidia.flash_attn.sm120.launch import forward

    torch.manual_seed(13)
    dev, page_size, hq, hkv, dim = "cuda", 64, 16, 2, 256
    qlens = [1024, 1024, 1024]
    kvlens = [1024, 4096, 32768]
    page_counts = [length // page_size for length in kvlens]
    q = (torch.randn(sum(qlens), hq, dim, device=dev) * 0.5).to(
        torch.float8_e4m3fn
    )
    k = (
        torch.randn(sum(page_counts), page_size, hkv, dim, device=dev) * 0.5
    ).to(torch.float8_e4m3fn)
    v = (
        torch.randn(sum(page_counts), page_size, hkv, dim, device=dev) * 0.5
    ).to(torch.float8_e4m3fn)
    cu_q = torch.tensor(
        [0] + torch.tensor(qlens).cumsum(0).tolist(), device=dev, dtype=torch.int32
    )
    seqused_k = torch.tensor(kvlens, device=dev, dtype=torch.int32)
    page_table = torch.zeros(
        len(qlens), max(page_counts), device=dev, dtype=torch.int32
    )
    page_begin = 0
    for batch_idx, count in enumerate(page_counts):
        page_table[batch_idx, :count] = torch.arange(
            page_begin, page_begin + count, device=dev, dtype=torch.int32
        )
        page_begin += count
    descale = torch.ones(len(qlens), hkv, device=dev)
    scale = 1.0 / math.sqrt(dim)

    def run(out, min_seqlen_k):
        return forward(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_q,
            seqused_k=seqused_k,
            page_table=page_table,
            max_seqlen_q=max(qlens),
            max_seqlen_k=max(kvlens),
            min_seqlen_k=min_seqlen_k,
            softmax_scale=scale,
            causal=True,
            num_splits=0,
            q_descale=descale,
            k_descale=descale,
            v_descale=descale,
            out=out,
        )

    static_out = torch.empty_like(q, dtype=torch.bfloat16)
    clc_out = torch.empty_like(static_out)
    run(static_out, None)
    run(clc_out, min(kvlens))
    torch.cuda.synchronize()
    assert torch.equal(clc_out, static_out)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay_out, _ = run(clc_out, min(kvlens))
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    assert replay_out.data_ptr() == clc_out.data_ptr()
    assert torch.equal(replay_out, static_out)


@pytest.mark.parametrize("page_size", [64, 128])
def test_sm120_paged_prefill_fp8_b64_varlen(page_size):
    """The generic scheduler handles B64 across both 31-request groups."""
    _requires_sm120()
    from atrex import flash_attn_varlen_func

    torch.manual_seed(15)
    dev, batch, hq, hkv, dim = "cuda", 64, 16, 2, 256
    qlens = [5 + batch_idx % 28 for batch_idx in range(batch)]
    kvlens = [q_len + 128 + (batch_idx % 8) * 64 for batch_idx, q_len in enumerate(qlens)]
    page_counts = [math.ceil(length / page_size) for length in kvlens]
    q = (torch.randn(sum(qlens), hq, dim, device=dev) * 0.5).to(
        torch.float8_e4m3fn
    )
    k = (
        torch.randn(sum(page_counts), page_size, hkv, dim, device=dev) * 0.5
    ).to(torch.float8_e4m3fn)
    v = (
        torch.randn(sum(page_counts), page_size, hkv, dim, device=dev) * 0.5
    ).to(torch.float8_e4m3fn)
    cu_q = torch.tensor(
        [0] + torch.tensor(qlens).cumsum(0).tolist(), device=dev, dtype=torch.int32
    )
    seqused_k = torch.tensor(kvlens, device=dev, dtype=torch.int32)
    page_table = torch.zeros(
        batch, max(page_counts), device=dev, dtype=torch.int32
    )
    page_begin = 0
    for batch_idx, count in enumerate(page_counts):
        page_table[batch_idx, :count] = torch.arange(
            page_begin, page_begin + count, device=dev, dtype=torch.int32
        )
        page_begin += count
    descale = torch.ones(batch, hkv, device=dev)
    scale = 1.0 / math.sqrt(dim)
    out = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        seqused_k=seqused_k,
        block_table=page_table,
        max_seqlen_q=max(qlens),
        max_seqlen_k=max(kvlens),
        softmax_scale=scale,
        causal=True,
        num_splits=0,
        q_descale=descale,
        k_descale=descale,
        v_descale=descale,
    )
    ref = _fp32_paged_prefill_ref(
        q,
        k,
        v,
        cu_q,
        qlens,
        kvlens,
        page_table,
        page_size,
        scale,
        descale,
        descale,
        descale,
    )
    rel_l1, cos, _ = _metrics(out.float(), ref)
    assert cos > 0.999
    assert rel_l1 < 0.04


@pytest.mark.parametrize("page_size", [64, 128])
def test_sm120_paged_prefill_fp8_b64_cuda_graph(page_size):
    """B64 device-side request ordering is dynamic and graph-safe."""
    _requires_sm120()
    from atrex import flash_attn_varlen_func
    from atrex.src.nvidia.flash_attn.sm120.runtime import _flash_attn_fwd

    torch.manual_seed(17)
    dev, batch, q_len, max_k = "cuda", 64, 8, 1024
    hq, hkv, dim = 16, 2, 256
    pages_per_request = math.ceil(max_k / page_size)
    q = (torch.randn(batch * q_len, hq, dim, device=dev) * 0.5).to(
        torch.float8_e4m3fn
    )
    k = (
        torch.randn(
            batch * pages_per_request, page_size, hkv, dim, device=dev
        )
        * 0.5
    ).to(torch.float8_e4m3fn)
    v = (
        torch.randn(
            batch * pages_per_request, page_size, hkv, dim, device=dev
        )
        * 0.5
    ).to(torch.float8_e4m3fn)
    cu_q = torch.arange(batch + 1, device=dev, dtype=torch.int32) * q_len
    seqused_k = torch.empty(batch, device=dev, dtype=torch.int32)
    page_table = torch.arange(
        batch * pages_per_request, device=dev, dtype=torch.int32
    ).reshape(batch, pages_per_request)
    descale = torch.ones(batch, hkv, device=dev)
    scale = 1.0 / math.sqrt(dim)
    out = torch.empty_like(q, dtype=torch.bfloat16)
    static_out = torch.empty_like(out)

    def run_public():
        return flash_attn_varlen_func(
            q,
            k,
            v,
            cu_seqlens_q=cu_q,
            seqused_k=seqused_k,
            block_table=page_table,
            max_seqlen_q=q_len,
            max_seqlen_k=256 * 1024,
            softmax_scale=scale,
            causal=True,
            num_splits=0,
            q_descale=descale,
            k_descale=descale,
            v_descale=descale,
            out=out,
        )

    def run_static():
        return _flash_attn_fwd(
            q,
            k,
            v,
            cu_seqlens_q=cu_q,
            seqused_k=seqused_k,
            page_table=page_table,
            max_seqlen_q=q_len,
            max_seqlen_k=256 * 1024,
            softmax_scale=scale,
            causal=True,
            num_splits=1,
            pack_gqa=True,
            q_descale=descale,
            k_descale=descale,
            v_descale=descale,
            out=static_out,
        )

    first_lengths = torch.tensor(
        [32] * 32 + [1024] * 32, device=dev, dtype=torch.int32
    )
    seqused_k.copy_(first_lengths)
    run_static()
    run_public()
    torch.cuda.synchronize()
    assert torch.equal(out, static_out)

    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        replay_out = run_public()
    for lengths in (
        first_lengths,
        first_lengths.flip(0),
        torch.full((batch,), 256, device=dev, dtype=torch.int32),
    ):
        seqused_k.copy_(lengths)
        graph.replay()
        run_static()
        torch.cuda.synchronize()
        assert replay_out.data_ptr() == out.data_ptr()
        assert torch.equal(replay_out, static_out)


# ----------------------------------------------------------------------------- decode
def _build_paged(kvlens, HKV, D, page_size, dev, seed=0):
    torch.manual_seed(seed)
    per = [(L + page_size - 1) // page_size for L in kvlens]
    total = sum(per)
    kc = torch.randn(total + 2, page_size, HKV, D, device=dev, dtype=torch.bfloat16)
    vc = torch.randn(total + 2, page_size, HKV, D, device=dev, dtype=torch.bfloat16)
    perm = torch.randperm(total, device=dev).to(torch.int32)
    bt = torch.zeros(len(kvlens), max(per), dtype=torch.int32, device=dev)
    o = 0
    for i, p in enumerate(per):
        bt[i, :p] = perm[o:o + p]; o += p
    sk = torch.tensor(kvlens, dtype=torch.int32, device=dev)
    return kc, vc, bt, sk


def _fp32_paged_mtp_ref(
    q,
    k_cache,
    v_cache,
    seqused_k,
    page_table,
    page_size,
    q_len,
    scale,
    q_descale,
    k_descale,
    v_descale,
):
    batch, hq, dim = q.shape[0] // q_len, q.shape[1], q.shape[2]
    hkv = k_cache.shape[2]
    group = hq // hkv
    query = q.view(batch, q_len, hq, dim)
    result = torch.empty_like(query, dtype=torch.float32)
    for batch_idx in range(batch):
        kv_len = int(seqused_k[batch_idx].item())
        num_pages = (kv_len + page_size - 1) // page_size
        page_ids = page_table[batch_idx, :num_pages].long()
        keys = k_cache[page_ids].reshape(-1, hkv, dim)[:kv_len].float()
        values = v_cache[page_ids].reshape(-1, hkv, dim)[:kv_len].float()
        keys *= k_descale[batch_idx].reshape(1, hkv, 1)
        values *= v_descale[batch_idx].reshape(1, hkv, 1)
        keys = keys.repeat_interleave(group, dim=1)
        values = values.repeat_interleave(group, dim=1)
        queries = query[batch_idx].float()
        queries *= q_descale[batch_idx].repeat_interleave(group).reshape(1, hq, 1)
        for query_idx in range(q_len):
            visible = kv_len - q_len + query_idx + 1
            scores = torch.einsum(
                "hd,khd->hk", queries[query_idx], keys[:visible]
            ) * scale
            probs = scores.softmax(dim=-1)
            result[batch_idx, query_idx] = torch.einsum(
                "hk,khd->hd", probs, values[:visible]
            )
    return result.reshape_as(q)


@pytest.mark.parametrize("page_size", [64, 128])
@pytest.mark.parametrize("q_len", [1, 2, 3, 4])
@pytest.mark.parametrize("hq,hkv", [(8, 1), (16, 2)])
def test_sm120_paged_decode_fp8_mtp(page_size, q_len, hq, hkv):
    _requires_sm120()
    from atrex import flash_attn_varlen_func

    torch.manual_seed(20260909 + page_size + q_len + hkv)
    dev, dim = "cuda", 256
    kvlens = [512, 769]
    pages_per_request = [math.ceil(length / page_size) for length in kvlens]
    total_pages = sum(pages_per_request)
    q = (
        torch.randn(len(kvlens) * q_len, hq, dim, device=dev) * 0.25
    ).to(torch.float8_e4m3fn)
    k = (
        torch.randn(total_pages, page_size, hkv, dim, device=dev) * 0.25
    ).to(torch.float8_e4m3fn)
    v = (
        torch.randn(total_pages, page_size, hkv, dim, device=dev) * 0.25
    ).to(torch.float8_e4m3fn)
    page_table = torch.zeros(
        len(kvlens), max(pages_per_request), device=dev, dtype=torch.int32
    )
    page_begin = 0
    for batch_idx, page_count in enumerate(pages_per_request):
        page_table[batch_idx, :page_count] = torch.arange(
            page_begin,
            page_begin + page_count,
            device=dev,
            dtype=torch.int32,
        )
        page_begin += page_count
    seqused_k = torch.tensor(kvlens, device=dev, dtype=torch.int32)
    cu_q = torch.arange(
        0,
        (len(kvlens) + 1) * q_len,
        q_len,
        device=dev,
        dtype=torch.int32,
    )
    q_descale = torch.linspace(0.7, 1.1, len(kvlens) * hkv, device=dev).reshape(
        len(kvlens), hkv
    )
    k_descale = torch.linspace(0.8, 1.2, len(kvlens) * hkv, device=dev).reshape(
        len(kvlens), hkv
    )
    v_descale = torch.linspace(0.6, 1.3, len(kvlens) * hkv, device=dev).reshape(
        len(kvlens), hkv
    )
    scale = 1.0 / math.sqrt(dim)
    out_buffer = torch.empty(
        len(kvlens) * q_len, hq, dim, device=dev, dtype=torch.bfloat16
    )
    out = flash_attn_varlen_func(
        q,
        k,
        v,
        cu_seqlens_q=cu_q,
        seqused_k=seqused_k,
        block_table=page_table,
        max_seqlen_q=q_len,
        max_seqlen_k=page_table.shape[1] * page_size,
        softmax_scale=scale,
        causal=True,
        num_splits=0,
        q_descale=q_descale,
        k_descale=k_descale,
        v_descale=v_descale,
        out=out_buffer,
    )
    assert out.data_ptr() == out_buffer.data_ptr()
    assert out.dtype == torch.bfloat16
    ref = _fp32_paged_mtp_ref(
        q,
        k,
        v,
        seqused_k,
        page_table,
        page_size,
        q_len,
        scale,
        q_descale,
        k_descale,
        v_descale,
    )
    rel_l1, cos, max_abs = _metrics(out.float(), ref)
    assert cos > 0.999, (
        f"Q{q_len} HQ/HKV={hq}/{hkv} page={page_size} "
        f"cos={cos} max_abs={max_abs}"
    )
    assert rel_l1 < 0.04, (
        f"Q{q_len} HQ/HKV={hq}/{hkv} page={page_size} "
        f"rel_l1={rel_l1} max_abs={max_abs}"
    )

    if (page_size, q_len, hq, hkv) == (64, 4, 8, 1):
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            flash_attn_varlen_func(
                q,
                k,
                v,
                cu_seqlens_q=cu_q,
                seqused_k=seqused_k,
                block_table=page_table,
                max_seqlen_q=q_len,
                max_seqlen_k=page_table.shape[1] * page_size,
                softmax_scale=scale,
                causal=True,
                num_splits=0,
                q_descale=q_descale,
                k_descale=k_descale,
                v_descale=v_descale,
                out=out_buffer,
            )
        replay_lengths = torch.tensor([449, 640], device=dev, dtype=torch.int32)
        seqused_k.copy_(replay_lengths)
        graph.replay()
        torch.cuda.synchronize()
        replay_ref = _fp32_paged_mtp_ref(
            q,
            k,
            v,
            seqused_k,
            page_table,
            page_size,
            q_len,
            scale,
            q_descale,
            k_descale,
            v_descale,
        )
        replay_rel_l1, replay_cos, _ = _metrics(out_buffer.float(), replay_ref)
        assert replay_cos > 0.999
        assert replay_rel_l1 < 0.04


@pytest.mark.parametrize("HQ,HKV", [(16, 2), (24, 4)])   # 35B, 27B
@pytest.mark.parametrize("kvlens", [
    [8192] * 8,                              # uniform decode
    [6000] * 4 + [12000] * 4,                # mild spread (bucketing path)
    [4000] * 6 + [60000] * 2 + [900] * 4,    # ragged (compact-grid path)
])
@pytest.mark.parametrize("page_size", [16, 48, 64, 128, 256])
def test_sm120_paged_decode_bf16(HQ, HKV, kvlens, page_size):
    _requires_sm120()
    from atrex import flash_attn_varlen_func
    dev, D = "cuda", 256
    scale = 1.0 / math.sqrt(D)
    kc, vc, bt, sk = _build_paged(kvlens, HKV, D, page_size, dev)
    nd = len(kvlens)
    torch.manual_seed(1)
    q = torch.randn(nd, HQ, D, device=dev, dtype=torch.bfloat16)
    cu = torch.arange(nd + 1, device=dev, dtype=torch.int32)
    out = torch.empty_like(q)
    o = flash_attn_varlen_func(
        q,
        kc,
        vc,
        cu_seqlens_q=cu,
        seqused_k=sk,
        block_table=bt,
        max_seqlen_q=1,
        max_seqlen_k=bt.shape[1] * page_size,
        softmax_scale=scale,
        causal=True,
        num_splits=0,
        out=out,
    )
    assert o.data_ptr() == out.data_ptr()
    o = o.float()
    ref = _fp32_paged_decode_ref(q, kc, vc, sk, bt, page_size, scale)
    rel_l1, cos, max_abs = _metrics(o, ref)
    assert cos > 0.999, f"HQ/HKV={HQ}/{HKV} kvlens[:3]={kvlens[:3]} page={page_size} cos={cos} max_abs={max_abs}"
    assert rel_l1 < 0.02, f"rel_l1={rel_l1} max_abs={max_abs}"


# --------------------------------------------------------------- pack_gqa + split-KV coexistence
@pytest.mark.parametrize("HQ,HKV", [(16, 2), (24, 4)])   # 35B, 27B
@pytest.mark.parametrize("kvlens", [
    [2048] * 8,                              # uniform decode
    [128, 512, 2048, 4096, 8192, 300, 1500, 6000],   # ragged decode
])
def test_sm120_packgqa_split_matches_nopack(HQ, HKV, kvlens, monkeypatch):
    """The fp32 packed-scatter O store lets pack_gqa and split-KV coexist (upstream leaves this
    NotImplementedError). It must (a) match the fp32 reference and (b) be bit-for-bit identical to
    the pack_gqa=False split path, since the packed layout only changes *addressing*, not values."""
    _requires_sm120()
    monkeypatch.setenv("ATREX_FA4_PGSPLIT", "1")
    from atrex.src.nvidia.flash_attn.sm120.runtime import _flash_attn_fwd
    dev, D, page_size = "cuda", 256, 64
    scale = 1.0 / math.sqrt(D)
    kc, vc, bt, sk = _build_paged(kvlens, HKV, D, page_size, dev)
    nd = len(kvlens)
    torch.manual_seed(1)
    q = torch.randn(nd, HQ, D, device=dev, dtype=torch.bfloat16)
    cu = torch.arange(nd + 1, device=dev, dtype=torch.int32)   # one query token per sequence
    mk = bt.shape[1] * page_size

    def run(pack_gqa):
        out = torch.empty(nd, HQ, D, device=dev, dtype=torch.bfloat16)
        _flash_attn_fwd(q, kc, vc, cu_seqlens_q=cu, seqused_k=sk, page_table=bt,
                        max_seqlen_q=1, max_seqlen_k=mk, softmax_scale=scale, causal=False,
                        pack_gqa=pack_gqa, num_splits=4, out=out)
        return out.float()

    ref = _fp32_paged_decode_ref(q, kc, vc, sk, bt, page_size, scale)
    o_pack = run(True)
    o_nopack = run(False)
    _, cos, max_abs = _metrics(o_pack, ref)
    assert cos > 0.999, f"HQ/HKV={HQ}/{HKV} pack_gqa+split cos={cos} max_abs={max_abs}"
    # addressing-only change -> identical values
    assert torch.equal(o_pack, o_nopack), (
        f"pack_gqa+split differs from nopack+split "
        f"(max {(o_pack - o_nopack).abs().max().item():.2e})")


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v", "-s"]))
