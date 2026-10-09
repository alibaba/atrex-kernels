"""Private BF16/FP8 short-Q runtime for the unified FA4 varlen entry."""

from __future__ import annotations

import functools
import math
from typing import Literal

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.typing import BFloat16, Float32, Int32

from atrex.src.nvidia.flash_attn.common_utils.cutlass_dsl_cache import (
    use_filesystem_cutlass_dsl_version_hash,
)
from atrex.src.nvidia.flash_attn.sm103.decode_cutedsl import (
    CausalMask,
    GroupedQueryAttentionDecodePaged,
)
from atrex.src.nvidia.flash_attn.sm103.decode_reduce_cutedsl import (
    fa4_decode_reduce_varlen_cutedsl,
)

HEAD_DIM = 256
GROUPED_HEAD_TILE = 16
VLLM_SEQUENCE_TILE = 128
SUPPORTED_PAGE_SIZES = (16, 32, 64, 128, 256)
SUPPORTED_DTYPES = (torch.bfloat16, torch.float8_e4m3fn)
MAX_QUERY_LENGTH = 32
SUPPORTED_QUERY_LENGTHS = tuple(range(1, MAX_QUERY_LENGTH + 1))
_vllm_decode_kernels: dict[tuple, object] = {}

def _slice_workspace(
    workspace: torch.Tensor,
    kv_splits: int,
    batch_size: int,
    q_len_per_req: int,
    num_qo_heads: int,
):
    o_shape = (
        kv_splits,
        batch_size,
        q_len_per_req,
        num_qo_heads,
        HEAD_DIM,
    )
    stat_shape = o_shape[:-1]
    workspace_f32 = workspace.view(torch.float32)
    required = math.prod(o_shape) + 2 * math.prod(stat_shape)
    if workspace_f32.numel() < required:
        raise RuntimeError(
            f"decode workspace needs {required * 4} bytes, got {workspace.nbytes}"
        )
    start = 0
    end = math.prod(o_shape)
    o_partial = workspace_f32[start:end].view(o_shape)
    start, end = end, end + math.prod(stat_shape)
    l_partial = workspace_f32[start:end].view(stat_shape)
    start, end = end, end + math.prod(stat_shape)
    m_partial = workspace_f32[start:end].view(stat_shape)
    return o_partial, l_partial, m_partial


def _decode_workspace_bytes(
    kv_splits: int,
    batch_size: int,
    q_len_per_req: int,
    num_qo_heads: int,
) -> int:
    """Return external-reduction workspace capacity in bytes."""
    output_elements = (
        kv_splits
        * batch_size
        * q_len_per_req
        * num_qo_heads
        * HEAD_DIM
    )
    partial_stat_elements = (
        kv_splits * batch_size * q_len_per_req * num_qo_heads
    )
    return (output_elements + 2 * partial_stat_elements) * torch.float32.itemsize


def _prediction_tile(q_len_per_req: int) -> int:
    """Return the power-of-two kernel tile covering the logical query length."""
    if q_len_per_req not in SUPPORTED_QUERY_LENGTHS:
        raise ValueError(
            "Atrex FA4 decode query length must be in "
            f"[1, {MAX_QUERY_LENGTH}], got {q_len_per_req}"
        )
    return 1 << (q_len_per_req - 1).bit_length()


def _supported_packed_rows(
    q_len_per_req: int,
    num_qo_heads: int,
    num_kv_heads: int,
) -> tuple[int, ...]:
    """Return packed-row variants that are valid for a static head shape."""
    _prediction_tile(q_len_per_req)
    if num_qo_heads <= 0 or num_kv_heads <= 0:
        raise ValueError("decode head counts must be positive")
    if num_qo_heads % num_kv_heads != 0:
        raise ValueError("num_qo_heads must be divisible by num_kv_heads")
    grouped_heads = num_qo_heads // num_kv_heads
    # Q4 with two or four grouped-query heads must use an exact logical N
    # tile.  Reusing N32 makes the TMA tile at least twice as wide as the
    # per-KV-head Q/O slice and can alias the neighboring KV head.  N8/N16
    # are native kernel tiles and keep these small-GQA shapes both correct and
    # efficient without encoding a concrete model head count.
    if q_len_per_req == 4 and grouped_heads in (2, 4):
        return (q_len_per_req * grouped_heads,)
    if q_len_per_req == 4 and grouped_heads == GROUPED_HEAD_TILE:
        return (32, 64)
    return (32,)


def _packed_rows(
    batch_size: int,
    q_len_per_req: int,
    num_qo_heads: int,
    num_kv_heads: int,
) -> int:
    """Select the measured packed query/head rows for one graph batch."""
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    variants = _supported_packed_rows(
        q_len_per_req, num_qo_heads, num_kv_heads
    )
    if len(variants) == 1:
        return variants[0]

    # Q4 packs prediction and grouped-query rows into one MMA N dimension.
    # N32 wins at small batches because it creates more independent CTAs;
    # N64 wins once request parallelism is already sufficient.  TP1 has two
    # KV heads and crosses that point one graph bucket earlier than TP2.
    if num_qo_heads == 32 and num_kv_heads == 2:
        return 64 if batch_size >= 16 else 32
    if num_qo_heads == 16 and num_kv_heads == 1:
        return 64 if batch_size >= 16 else 32
    return 32


def _decode_p_stages(q_len_per_req: int) -> int:
    """Use the measured shallower BMM2 pipeline for Q4 verification."""
    return 2 if q_len_per_req == 4 else 4


def _decode_sequence_tile(
    q_len_per_req: int,
    num_qo_heads: int,
    num_kv_heads: int,
    *,
    page_size: int | None = None,
    packed_rows: int | None = None,
    q_dtype: torch.dtype | None = None,
    kv_dtype: torch.dtype | None = None,
) -> int:
    """Select the measured decode KV sequence tile.

    P128 FP8 decode with a 32-row Q/head tile benefits from consuming two
    physical pages per CTA loop iteration.  Keep the 128-token tile for wider
    packed rows and every unmeasured page/dtype combination.
    """
    if q_len_per_req == 1 and num_qo_heads == 32 and num_kv_heads == 2:
        return 256
    if (
        page_size == 128
        and q_dtype == kv_dtype == torch.float8_e4m3fn
        and packed_rows == 32
        and 1 <= q_len_per_req <= 5
    ):
        return 256
    return VLLM_SEQUENCE_TILE


@functools.cache
def _compile_kernel(
    cache_page_size: int,
    reduction: Literal["external", "atomic", "none"],
    sequence_tile: int,
    softmax_warpgroups: int,
    p_stages: int,
    q_len_per_req: int,
    num_qo_heads: int,
    num_kv_heads: int,
    packed_rows: int,
    q_dtype: torch.dtype,
    kv_dtype: torch.dtype,
    device_index: int | None = None,
    device_capability: tuple[int, int] | None = None,
    ragged_q: bool = False,
    return_lse: bool = False,
):
    # These arguments intentionally participate in functools.cache's key. CuTe
    # compilation targets the active CUDA device, so a callable compiled for
    # one architecture must never be reused for another one.
    del device_index, device_capability
    if cache_page_size not in SUPPORTED_PAGE_SIZES:
        raise ValueError(
            f"Atrex FA4 decode requires page size in {SUPPORTED_PAGE_SIZES}"
        )
    if q_dtype not in SUPPORTED_DTYPES or kv_dtype not in SUPPORTED_DTYPES:
        raise TypeError(f"unsupported Q/KV dtypes: {q_dtype}/{kv_dtype}")
    if q_dtype != kv_dtype:
        raise TypeError("Atrex FA4 decode requires matching Q/KV dtypes")
    if q_len_per_req not in SUPPORTED_QUERY_LENGTHS:
        raise ValueError(
            "Atrex FA4 decode query length must be one of "
            f"{SUPPORTED_QUERY_LENGTHS}, got {q_len_per_req}"
        )
    # The native mainloop tile is at most 128 tokens. A physical page256 is
    # consumed as two zero-copy page128 subpages; the kernel translates each
    # logical subpage through the original physical-page table.
    kernel_page_size = min(cache_page_size, 128)
    page_table_factor = cache_page_size // kernel_page_size
    cute_q_dtype = BFloat16 if q_dtype == torch.bfloat16 else cutlass.Float8E4M3FN
    cute_kv_dtype = BFloat16 if kv_dtype == torch.bfloat16 else cutlass.Float8E4M3FN
    prediction_tile = _prediction_tile(q_len_per_req)
    supported_packed_rows = _supported_packed_rows(
        q_len_per_req, num_qo_heads, num_kv_heads
    )
    if packed_rows not in supported_packed_rows:
        raise ValueError(
            "packed_rows is not valid for the decode shape: "
            f"{packed_rows} not in {supported_packed_rows}"
        )
    grouped_head_tile = min(
        GROUPED_HEAD_TILE, packed_rows // prediction_tile
    )
    fmha = GroupedQueryAttentionDecodePaged(
        page_size=kernel_page_size,
        headdim=HEAD_DIM,
        grouped_head_tile=grouped_head_tile,
        prediction_tile=prediction_tile,
        sequence_tile=sequence_tile,
        reduction_mode=reduction,
        softmax_warpgroups=softmax_warpgroups,
        page_table_factor=page_table_factor,
        p_stages=p_stages,
    )
    fmha.atrex_sm103_decode_kernel.set_name_prefix("atrex")
    mask = CausalMask()
    sym_splits = cute.sym_int()
    sym_batch = cute.sym_int()
    sym_batch_plus_one = cute.sym_int()
    sym_pages = cute.sym_int()
    sym_max_pages = cute.sym_int()
    sym_total_q = cute.sym_int()

    seqlens = cute.runtime.make_fake_compact_tensor(
        Int32, (sym_batch,), assumed_align=16
    )
    page_table = cute.runtime.make_fake_tensor(
        Int32,
        (sym_batch, sym_max_pages),
        stride=(cute.sym_int(), 1),
        assumed_align=4,
    )
    k = cute.runtime.make_fake_tensor(
        cute_kv_dtype,
        (sym_pages, cache_page_size, num_kv_heads, HEAD_DIM),
        stride=(cute.sym_int(), cute.sym_int(), cute.sym_int(), 1),
        assumed_align=16,
    )
    v = cute.runtime.make_fake_tensor(
        cute_kv_dtype,
        (sym_pages, cache_page_size, num_kv_heads, HEAD_DIM),
        stride=(cute.sym_int(), cute.sym_int(), cute.sym_int(), 1),
        assumed_align=16,
    )
    if ragged_q:
        cu_seqlens_q = cute.runtime.make_fake_compact_tensor(
            Int32, (sym_batch_plus_one,), assumed_align=16
        )
        q_shape = (1, sym_total_q, num_qo_heads, HEAD_DIM)
    else:
        cu_seqlens_q = None
        q_shape = (sym_batch, q_len_per_req, num_qo_heads, HEAD_DIM)
    q = cute.runtime.make_fake_compact_tensor(
        cute_q_dtype,
        q_shape,
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    o = cute.runtime.make_fake_compact_tensor(
        BFloat16,
        q_shape,
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    if return_lse and reduction != "external":
        if ragged_q:
            lse = cute.runtime.make_fake_tensor(
                Float32,
                (1, sym_total_q, num_qo_heads),
                stride=(cute.sym_int(), 1, cute.sym_int()),
                assumed_align=4,
            )
        else:
            lse = cute.runtime.make_fake_compact_tensor(
                Float32,
                (sym_batch, q_len_per_req, num_qo_heads),
                stride_order=(2, 1, 0),
                assumed_align=4,
            )
    else:
        lse = None

    if reduction == "external":
        o_partial = cute.runtime.make_fake_compact_tensor(
            Float32,
            (
                sym_splits,
                sym_batch,
                q_len_per_req,
                num_qo_heads,
                HEAD_DIM,
            ),
            stride_order=(4, 3, 2, 1, 0),
            assumed_align=16,
        )
        l_partial = cute.runtime.make_fake_compact_tensor(
            Float32,
            (sym_splits, sym_batch, q_len_per_req, num_qo_heads),
            stride_order=(3, 2, 1, 0),
            assumed_align=16,
        )
        m_partial = cute.runtime.make_fake_compact_tensor(
            Float32,
            (sym_splits, sym_batch, q_len_per_req, num_qo_heads),
            stride_order=(3, 2, 1, 0),
            assumed_align=16,
        )
    else:
        o_partial = l_partial = m_partial = None

    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    with use_filesystem_cutlass_dsl_version_hash():
        return cute.compile(
            fmha,
            Int32(1),
            seqlens,
            cu_seqlens_q,
            page_table,
            k,
            v,
            q,
            o,
            lse,
            o_partial,
            l_partial,
            m_partial,
            None,  # attention sinks
            mask,
            Float32(1.0),
            Float32(1.0),
            None,  # skip-softmax threshold
            stream,
            True,
            options="--enable-tvm-ffi --opt-level 3",
        )


def _get_vllm_decode_kernel(
    q: torch.Tensor,
    *,
    page_size: int,
    reduction: Literal["external", "atomic", "none"],
    packed_rows: int,
    num_qo_heads: int,
    num_kv_heads: int,
    kv_dtype: torch.dtype,
    q_len_per_req: int = 1,
    ragged_q: bool = False,
    return_lse: bool = False,
):
    """Compile during warmup and require an exact cache hit during capture."""
    device_index = q.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    device_capability = torch.cuda.get_device_capability(q.device)
    if packed_rows not in _supported_packed_rows(
        q_len_per_req, num_qo_heads, num_kv_heads
    ):
        raise ValueError("packed_rows is not valid for the requested decode shape")
    p_stages = _decode_p_stages(q_len_per_req)
    sequence_tile = _decode_sequence_tile(
        q_len_per_req,
        num_qo_heads,
        num_kv_heads,
        page_size=page_size,
        packed_rows=packed_rows,
        q_dtype=q.dtype,
        kv_dtype=kv_dtype,
    )
    key = (
        device_index,
        device_capability,
        page_size,
        reduction,
        sequence_tile,
        p_stages,
        q_len_per_req,
        num_qo_heads,
        num_kv_heads,
        packed_rows,
        q.dtype,
        kv_dtype,
        ragged_q,
        return_lse,
    )
    kernel = _vllm_decode_kernels.get(key)
    if kernel is not None:
        return kernel
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "Atrex FA4 decode kernel must be compiled before graph capture"
        )
    with torch.cuda.device(q.device):
        kernel = _compile_kernel(
            page_size,
            reduction,
            sequence_tile,
            1,
            p_stages,
            q_len_per_req,
            num_qo_heads,
            num_kv_heads,
            packed_rows,
            q.dtype,
            kv_dtype,
            device_index,
            device_capability,
            ragged_q,
            return_lse,
        )
    _vllm_decode_kernels[key] = kernel
    return kernel


def _vllm_decode_config(
    batch_size: int,
    sm_count: int,
    q_len_per_req: int,
    num_qo_heads: int,
    num_kv_heads: int,
) -> tuple[int, Literal["external", "atomic", "none"]]:
    """Select a graph-stable decode launch from static dimensions.

    KV length never changes the selected kernel or reduction topology.  Decode
    grows the cache monotonically, so a graph captured at one context length is
    replayed at every later one: any topology frozen against a KV length is
    stale by construction.  The one host-visible bound a caller can supply is
    also not a per-graph maximum -- the vLLM binding replaces max_seqlen_k with
    the full page-table capacity, precisely so a captured graph cannot go stale
    -- which makes it a deployment-wide constant carrying no per-graph signal.

    Its only sound use is as a ceiling, never as an estimate: ``real <= cap``
    supports ``splits = min(splits, cap_tiles)`` (splits beyond the tile count
    own no work), but never ``splits = max(splits, cap_tiles / 16)``, which
    reads an upper bound as if it were the actual length.  The latter looks
    profitable in benchmarks, which pass a real context length, and then
    selects an untested schedule in production.  The ceiling itself needs a
    capacity below 2048 tokens to bind at all, so it is left unimplemented.

    Match the TRTLLM-gen scheduling model by counting independent
    query/head/batch CTAs, then adding enough KV splits to fill an SM wave.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if sm_count <= 0:
        raise ValueError("sm_count must be positive")
    prediction_tile = _prediction_tile(q_len_per_req)
    packed_rows = _packed_rows(
        batch_size,
        q_len_per_req,
        num_qo_heads,
        num_kv_heads,
    )
    grouped_head_tile = min(
        GROUPED_HEAD_TILE, packed_rows // prediction_tile
    )
    grouped_heads = num_qo_heads // num_kv_heads
    query_head_tiles = math.ceil(grouped_heads / grouped_head_tile) * math.ceil(
        q_len_per_req / prediction_tile
    )
    base_ctas = batch_size * num_kv_heads * query_head_tiles

    if q_len_per_req == 1:
        occupancy_splits = max(1, math.ceil(sm_count / base_ctas))
    else:
        occupancy_splits = max(1, sm_count // base_ctas)
    target_splits = min(16, occupancy_splits)

    reduction: Literal["external", "atomic", "none"] = "atomic"
    head_shape = (num_qo_heads, num_kv_heads)
    if q_len_per_req == 1 and head_shape == (32, 2):
        # TP1 Q1: external reduction removes DSM synchronization at the two
        # medium-small graph buckets.  Larger buckets need fewer splits.
        if batch_size <= 2:
            split_cap = 16
        elif batch_size <= 4:
            split_cap, reduction = 16, "external"
        elif batch_size <= 8:
            split_cap, reduction = 8, "external"
        elif batch_size <= 16:
            split_cap = 4
        else:
            split_cap = 2
        target_splits = min(target_splits, split_cap)
    elif q_len_per_req == 1 and head_shape == (16, 1):
        # TP2 Q1 has half as many independent KV-head CTAs.  Split8 at B8 is
        # faster than the generic split16 choice.  The large PD-disaggregated
        # decode buckets need more sequence parallelism: B16--B19 prefer a
        # full split16 cluster, while B20 and above prefer split8.
        if batch_size <= 4:
            split_cap = 16
        elif batch_size <= 8:
            split_cap = 8
        elif batch_size < 16:
            split_cap, reduction = 4, "external"
        elif batch_size <= 19:
            split_cap = 16
            target_splits = 16
        elif batch_size <= 32:
            split_cap = 8
            target_splits = 8
        else:
            # Beyond the measured B<=32 schedule, PD-disaggregated decode pairs
            # a large batch with long, ragged KV.  Request CTAs alone fill the
            # SM wave, but occupancy-driven splits then collapse to split2 and
            # leave each CTA a very long serial KV loop (~3x slower than split8
            # at B48-128, ctx~28k).  Hold the measured split8 instead: KV length
            # is not a usable dispatch input, so assume the long-KV case that
            # PD-disaggregated decode actually serves.
            split_cap = 8
            target_splits = 8
        target_splits = min(target_splits, split_cap)
    elif q_len_per_req == 4 and head_shape == (32, 2):
        # TP1 Q4 uses N32 plus independent split CTAs for B1-B4, then changes
        # to N64 with atomic cluster reduction once request parallelism grows.
        # For a graph bucket containing five to eight N128 KV tiles, atomic S4
        # gives every split useful work and removes the external workspace and
        # reducer.  This is consistently faster across B1--B4; longer buckets
        # retain the measured occupancy-oriented schedule below.
        if batch_size == 1:
            split_cap, reduction = 16, "external"
        elif batch_size <= 2:
            split_cap, reduction = 16, "external"
        elif batch_size <= 4:
            split_cap, reduction = 8, "external"
        elif batch_size <= 8:
            split_cap = 4
        elif batch_size <= 16:
            split_cap = 4
        else:
            split_cap = 2
        target_splits = min(target_splits, split_cap)
    elif q_len_per_req == 4 and head_shape == (16, 1):
        # TP2 Q4 switches to N64 at B16, where the wider query tile avoids the
        # packed-N32 performance cliff.  Q4 always reduces externally, which has
        # no co-resident cluster to pay for, so B16--B26 affords split16.  Past
        # that the doubled FP32 partial round trip outweighs the extra sequence
        # parallelism and split8 wins again.
        if batch_size <= 2:
            split_cap = 16
        elif batch_size <= 4:
            split_cap, reduction = 16, "external"
        elif batch_size <= 8:
            split_cap, reduction = 8, "external"
        elif batch_size < 16:
            split_cap, reduction = 4, "external"
        elif batch_size <= 26:
            split_cap = 16
            target_splits = 16
        elif batch_size <= 28:
            split_cap = 8
            target_splits = 8
        else:
            # B>28 Q4 is outside the measured 609 range but arises in
            # PD-disaggregated decode.  Hold the measured split8 so long-context
            # large batches are not starved of KV parallelism (the fixed
            # split4/split2 tail was ~2-3x slower at B>=48, ctx~28k).
            split_cap = 8
            target_splits = 8
        target_splits = min(target_splits, split_cap)
    elif q_len_per_req == 2 and head_shape == (32, 2):
        if batch_size == 4:
            target_splits = min(target_splits, 8)
        elif batch_size == 8:
            target_splits = min(target_splits, 8)
            reduction = "external"
        elif batch_size >= 16:
            target_splits = min(target_splits, 4)
    elif q_len_per_req == 2 and head_shape == (16, 1):
        if batch_size == 8:
            target_splits = min(target_splits, 8)
        elif batch_size >= 16:
            target_splits = min(target_splits, 4)
    elif q_len_per_req == 3 and head_shape == (32, 2):
        if batch_size == 2:
            target_splits = min(target_splits, 16)
            reduction = "external"
        elif batch_size == 4:
            target_splits = min(target_splits, 8)
            reduction = "external"
        elif batch_size >= 16:
            target_splits = min(target_splits, 4)
    elif q_len_per_req == 3 and head_shape == (16, 1):
        if batch_size == 4:
            target_splits = min(target_splits, 16)
            reduction = "external"
        elif batch_size >= 16:
            target_splits = min(target_splits, 4)
    elif batch_size >= 16:
        # Q2/Q3 retain the formula-based MTP schedule while avoiding large DSM
        # clusters after request/head parallelism already fills an SM wave.
        target_splits = min(target_splits, 4)
    # Generic Q4 decode uses enough sequence parallelism to keep at least 64
    # CTAs active.  This structural rule selects S8/S4/S4 for the B8 H8/1,
    # H16/2 and H24/4 families respectively, while head-specific production
    # schedules above retain their measured overrides.
    if q_len_per_req == 4 and head_shape not in ((32, 2), (16, 1)):
        parallel_splits = max(1, math.ceil(64 / base_ctas))
        structural_splits = 1 << (parallel_splits - 1).bit_length()
        target_splits = min(16, structural_splits)
    # Atomic cluster reduction supports power-of-two split counts. Rounding
    # down avoids capturing CTAs that cannot contribute useful KV work.
    kv_splits = 1 << (target_splits.bit_length() - 1)
    if kv_splits == 1:
        reduction = "none"
    return kv_splits, reduction


def _fa4_decode_varlen(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    qv: torch.Tensor | None = None,
    cu_seqlens_q: torch.Tensor | None = None,
    cu_seqlens_k: torch.Tensor | None = None,
    seqused_q: torch.Tensor | None = None,
    seqused_k: torch.Tensor | None = None,
    max_seqlen_q: int | None = None,
    max_seqlen_k: int | None = None,
    min_seqlen_k: int | None = None,
    page_table: torch.Tensor | None = None,
    softmax_scale: float | None = None,
    causal: bool = False,
    softcap: float | None = None,
    window_size_left: int | None = None,
    window_size_right: int | None = None,
    learnable_sink: torch.Tensor | None = None,
    tile_mn=None,
    mma_pv_is_rs=None,
    intra_wg_overlap=None,
    num_threads: int = 384,
    num_splits: int = 1,
    pack_gqa: bool | None = None,
    _arch: int | None = None,
    score_mod=None,
    mask_mod=None,
    block_sparse_tensors=None,
    return_lse: bool = False,
    out: torch.Tensor | None = None,
    lse: torch.Tensor | None = None,
    aux_tensors=None,
    aux_scalars=None,
    q_descale: torch.Tensor | None = None,
    k_descale: torch.Tensor | None = None,
    v_descale: torch.Tensor | None = None,
    gather_kv_indices: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor | None]:
    """Run BF16/FP8 paged ragged Q0--Q5 decode/MTP from device metadata.

    Unsupported static features raise ``NotImplementedError`` so the unified
    Atrex entry can route them to Atrex's 2CTA varlen kernel. Runtime query and
    KV lengths are never read on the host.
    """
    unsupported = (
        qv is not None
        or cu_seqlens_k is not None
        or seqused_q is not None
        or min_seqlen_k is not None
        or learnable_sink is not None
        or tile_mn is not None
        or mma_pv_is_rs is not None
        or intra_wg_overlap is not None
        or score_mod is not None
        or mask_mod is not None
        or block_sparse_tensors is not None
        or aux_tensors is not None
        or aux_scalars is not None
        or gather_kv_indices is not None
        or q_descale is not None
        or k_descale is not None
        or v_descale is not None
    )
    if unsupported:
        raise NotImplementedError("Atrex short-Q decode does not support modifiers")
    if not causal or softcap not in (None, 0.0):
        raise NotImplementedError("Atrex short-Q decode requires causal attention")
    if window_size_left is not None or window_size_right is not None:
        raise NotImplementedError("Atrex short-Q decode does not support local attention")
    if pack_gqa is False:
        raise NotImplementedError("Atrex short-Q decode requires packed GQA")
    del num_threads, num_splits

    if max_seqlen_q is None or not 1 <= max_seqlen_q <= 5:
        raise NotImplementedError("Atrex short-Q decode requires 1 <= max_seqlen_q <= 5")
    if cu_seqlens_q is None or seqused_k is None or page_table is None:
        raise NotImplementedError(
            "Atrex short-Q decode requires cu_seqlens_q, seqused_k, and page_table"
        )
    if q.ndim != 3 or q.shape[-1] != HEAD_DIM:
        raise NotImplementedError("Atrex short-Q decode requires packed HD256 Q")
    if k.ndim != 4 or v.shape != k.shape or k.shape[-1] != HEAD_DIM:
        raise NotImplementedError("Atrex short-Q decode requires paged HD256 K/V")
    _, page_size, num_kv_heads, _ = k.shape
    num_qo_heads = q.shape[1]
    if page_size not in SUPPORTED_PAGE_SIZES:
        raise NotImplementedError(f"unsupported Atrex decode page size {page_size}")
    if num_qo_heads <= 0 or num_kv_heads <= 0 or num_qo_heads % num_kv_heads:
        raise NotImplementedError("Atrex short-Q decode requires grouped-query heads")
    if q.dtype not in SUPPORTED_DTYPES or k.dtype != q.dtype or v.dtype != q.dtype:
        raise NotImplementedError(
            "Atrex short-Q decode requires matching BF16 or FP8 E4M3 Q/K/V"
        )
    if q.device.type != "cuda" or k.device != q.device or v.device != q.device:
        raise ValueError("Q/K/V must share one CUDA device")
    if torch.cuda.get_device_capability(q.device) != (10, 3):
        raise NotImplementedError("Atrex short-Q decode requires SM103")
    if _arch is not None and _arch != 103:
        raise NotImplementedError("Atrex short-Q decode requires SM103")
    if not q.is_contiguous():
        raise NotImplementedError("Atrex short-Q decode requires contiguous packed Q")
    if k.stride(-1) != 1 or v.stride(-1) != 1:
        raise ValueError("K/V head_dim must be contiguous")

    if cu_seqlens_q.dtype != torch.int32 or cu_seqlens_q.ndim != 1:
        raise ValueError("cu_seqlens_q must be int32 [batch + 1]")
    batch_size = cu_seqlens_q.numel() - 1
    if batch_size <= 0 or cu_seqlens_q.stride(0) != 1:
        raise ValueError("cu_seqlens_q must be non-empty and contiguous")
    if seqused_k.dtype != torch.int32 or seqused_k.shape != (batch_size,):
        raise ValueError("seqused_k must be int32 [batch]")
    if seqused_k.stride(0) != 1:
        raise ValueError("seqused_k must be contiguous")
    if page_table.dtype != torch.int32 or page_table.ndim != 2:
        raise ValueError("page_table must be int32 [batch, max_pages]")
    if page_table.shape[0] != batch_size or page_table.stride(-1) != 1:
        raise ValueError("page_table batch/last-dimension layout is invalid")
    if any(t.device != q.device for t in (cu_seqlens_q, seqused_k, page_table)):
        raise ValueError("decode metadata must be on the Q device")
    if max_seqlen_k is not None and max_seqlen_k <= 0:
        raise ValueError("max_seqlen_k must be positive")

    expected_out = (q.shape[0], num_qo_heads, HEAD_DIM)
    if out is None:
        out = torch.empty(expected_out, dtype=torch.bfloat16, device=q.device)
    elif out.shape != expected_out or out.dtype != torch.bfloat16:
        raise ValueError("Atrex decode out must be BF16 with the packed Q shape")
    if out.device != q.device or not out.is_contiguous():
        raise ValueError("Atrex decode out must be contiguous on the Q device")
    wants_lse = return_lse or lse is not None
    if lse is None and wants_lse:
        lse = torch.empty(
            (num_qo_heads, q.shape[0]), dtype=torch.float32, device=q.device
        )
    if lse is not None:
        if lse.shape != (num_qo_heads, q.shape[0]) or lse.dtype != torch.float32:
            raise ValueError("Atrex decode lse must be FP32 [heads, total_q]")
        if lse.device != q.device or not lse.is_contiguous():
            raise ValueError("Atrex decode lse must be contiguous on the Q device")
        lse.fill_(float("-inf"))

    scale = HEAD_DIM**-0.5 if softmax_scale is None else softmax_scale
    props = torch.cuda.get_device_properties(q.device)
    kv_splits, reduction = _vllm_decode_config(
        batch_size,
        props.multi_processor_count,
        max_seqlen_q,
        num_qo_heads,
        num_kv_heads,
    )
    # LSE is only produced by the external reducer, which converts the
    # mainloop's base-2 statistic into PAI-compatible natural log.
    #
    # Q > 1 also stays on the external reducer for now.  Atomic reduce-add
    # does handle ragged Q -- the kernel zeroes the normalization of query
    # rows a request does not own, so they contribute exactly zero to the
    # neighbour they overhang into -- but configurations with a packed query
    # tile below 64 rows (o_stages == 2) still show a sporadic whole-request
    # deviation (<= 0.055 abs, roughly one run in ten) that predates that fix
    # and is not root-caused.  No static predicate separates the clean cases:
    # the failing tile width depends on head shape, not on max_seqlen_q.
    # External reduces FP32 partials deterministically and passed the full
    # ragged-Q matrix (248 combos, max err 0.0010), so Q > 1 keeps it until
    # the atomic anomaly is resolved.
    if max_seqlen_q > 1 or wants_lse:
        reduction = "external"
    packed_rows = _packed_rows(
        batch_size,
        max_seqlen_q,
        num_qo_heads,
        num_kv_heads,
    )
    # Atomic accumulation and external/ragged reduction retain explicit
    # clearing.  A non-split kernel overwrites non-empty requests and performs
    # device-side zero stores for empty requests, preserving graph replay
    # semantics without adding a separate memset launch to the fast path.
    if reduction != "none":
        out.zero_()
    kernel = _get_vllm_decode_kernel(
        q,
        page_size=page_size,
        reduction=reduction,
        packed_rows=packed_rows,
        q_len_per_req=max_seqlen_q,
        num_qo_heads=num_qo_heads,
        num_kv_heads=num_kv_heads,
        kv_dtype=k.dtype,
        ragged_q=True,
        return_lse=wants_lse,
    )

    o_partial = l_partial = m_partial = None
    if reduction == "external":
        required_bytes = _decode_workspace_bytes(
            kv_splits, batch_size, max_seqlen_q, num_qo_heads
        )
        # Empty splits refresh their statistics but may not store O. The
        # reducer skips O reads when their weight is zero, so no workspace
        # initialization is needed. Keep scratch local to the allocator/graph
        # pool rather than retaining a per-layer cache.
        workspace = torch.empty(
            required_bytes, dtype=torch.uint8, device=q.device,
        )
        o_partial, l_partial, m_partial = _slice_workspace(
            workspace,
            kv_splits,
            batch_size,
            max_seqlen_q,
            num_qo_heads,
        )

    kernel_lse = None
    if wants_lse and reduction in ("atomic", "none"):
        assert lse is not None
        kernel_lse = lse.transpose(0, 1).unsqueeze(0)
    kernel(
        kv_splits,
        seqused_k,
        cu_seqlens_q,
        page_table,
        k,
        v,
        q.unsqueeze(0),
        out.unsqueeze(0),
        kernel_lse,
        o_partial,
        l_partial,
        m_partial,
        None,
        CausalMask(),
        Float32(scale),
        Float32(1.0),
        None,
        not (
            reduction == "atomic"
            and max_seqlen_q == 1
            and num_qo_heads == 32
            and num_kv_heads == 2
        ),
    )
    if reduction == "external":
        assert o_partial is not None
        assert l_partial is not None
        assert m_partial is not None
        fa4_decode_reduce_varlen_cutedsl(
            o_partial,
            l_partial,
            m_partial,
            cu_seqlens_q,
            out,
            lse,
        )
    return out, lse if wants_lse else None
