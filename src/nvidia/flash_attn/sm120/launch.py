"""SM120 launch policy migrated from l20n-fa4 (68fd128c)."""

_SM120_RUNTIME_BALANCED_MAX_BATCH = 64
_SM120_RUNTIME_BALANCED_Q1_GRID = 96
_SM120_RUNTIME_BALANCED_MTP_GRID = 92
_SM120_RUNTIME_BALANCED_HIGH_BATCH = 8
_SM120_FP8_DECODE_GRID = 192
_SM120_FP8_DECODE_B4_Q1_GRID = 200

def _sm120_forward():
    """Lazily load the L20N SM120 forward implementation."""
    from atrex.src.nvidia.flash_attn.sm120.runtime import _flash_attn_fwd

    return _flash_attn_fwd

def _sm120_short_q_splits(
    num_sequences,
    num_kv_heads,
    max_k,
    sm_count,
    tile_n=32,
    page_size=None,
    query_length=1,
    is_fp8=False,
):
    """Choose native short-Q SplitKV parallelism from graph-static dimensions.

    Each (sequence, KV-head) pair contributes one independent CTA before KV
    splitting. BF16 reaches its bandwidth plateau near one CTA wave. FP8 needs
    more independent memory requests because each K/V tile carries half as
    many bytes, so its Q1-Q4 schedules target roughly two CTA waves.
    """
    base_ctas = max(1, num_sequences * num_kv_heads)
    kv_tiles = max(1, (max_k + tile_n - 1) // tile_n)
    if is_fp8 and page_size in (64, 128):
        # FP8 halves the K/V bytes per tile, so a single CTA wave does not
        # expose enough independent memory requests on SM120. Q1/Q2 use a
        # nearly two-wave grid; Q3/Q4 need fewer splits because each CTA has
        # more query rows and correspondingly more QK/PV work.
        target_ctas = min(
            2 * sm_count,
            _SM120_FP8_DECODE_B4_Q1_GRID
            if query_length == 1 and base_ctas == 4
            else _SM120_FP8_DECODE_GRID,
        )
        occupancy_splits = (target_ctas + base_ctas - 1) // base_ctas
        return max(1, min(96, occupancy_splits, kv_tiles))
    if (
        page_size == 128
        and query_length in (2, 3, 4)
        and num_sequences == 1
        and num_kv_heads == 1
    ):
        # Packed speculative verification needs one nearly full SM wave. Keep
        # the topology independent of the replay-time KV length, as in the
        # SM103 decode path.
        return min(96, sm_count, kv_tiles)
    if (
        page_size == 128
        and query_length == 1
        and num_sequences == 1
        and num_kv_heads == 1
    ):
        # Select from the caller's graph-static KV capacity, never replay-time
        # seqused_k. This keeps vLLM's full-capacity graph stable while direct
        # operator graphs can use the measured short-KV saturation points.
        if max_k <= 64 * 1024:
            return min(128, kv_tiles)
        if max_k <= 128 * 1024:
            return min(64, kv_tiles)
        return min(96, kv_tiles)
    if (
        page_size == 128
        and num_sequences == 4
        and num_kv_heads == 1
        and 128 * 1024 <= max_k < 256 * 1024
        and query_length in (1, 2, 3, 4)
    ):
        # S32 hits a repeatable scheduling cliff for C4 in this context
        # range (about 4% at 192K).  Q1 reaches the bandwidth plateau with
        # S24; packed speculative verification needs S36.
        tuned_splits = 24 if query_length == 1 else 36
        return min(tuned_splits, kv_tiles)
    if page_size == 64 and num_kv_heads == 1 and query_length in (1, 2, 3, 4):
        # M64/N64 permits one resident CTA per SM. Select a graph-stable grid
        # from batch/head parallelism instead of KV length: S32 underfills C1,
        # while B4/S32 leaves a tail wave. One nearly full SM wave avoids both
        # cliffs and remains stable from 32K through 512K.
        target_ctas = min(96, sm_count)
        occupancy_splits = (target_ctas + base_ctas - 1) // base_ctas
        return max(1, min(target_ctas, occupancy_splits, kv_tiles))
    if max_k >= 256 * 1024:
        # C1 is sensitive to the N-tile-specific saturation point; larger N
        # tiles need fewer CTAs to saturate DRAM. The tuned C4 Q1 shape uses
        # two full SM waves; other multi-sequence shapes retain one wave.
        if base_ctas == 1:
            target_ctas = {
                128: 96,
                1056: 88,
                1152: 84,
            }.get(page_size, {64: 84, 48: 88, 32: 92, 16: 96}.get(tile_n, 92))
        elif (
            page_size == 128
            and query_length in (1, 4)
            and num_sequences == 4
            and num_kv_heads == 1
        ):
            # At 512K, 55 splits give 220 useful CTAs (two full SM waves),
            # matching the fastest FA2 schedule for the TP2 Qwen3.5 shape.
            target_ctas = 2 * sm_count
        else:
            target_ctas = 96
        occupancy_splits = (target_ctas + base_ctas - 1) // base_ctas
        return max(1, min(128, occupancy_splits, kv_tiles))
    occupancy_splits = (2 * sm_count + base_ctas - 1) // base_ctas
    return max(1, min(32, occupancy_splits, kv_tiles))

def _sm120_fp8_prefill_splits(
    total_q,
    num_q_heads,
    num_kv_heads,
    max_q,
    max_k,
    sm_count,
    *,
    pack_gqa,
    tile_m=64,
    tile_n=32,
):
    """Select SplitKV only for underfilled short-Q FP8 prefill shapes.

    The policy uses graph-static dimensions only. Splitting is useful when a
    short prefix-cache query leaves too few independent Q tiles; it is avoided
    once Q supplies enough work because the FP32 partial buffers and combine
    kernel then cost more than the finer KV scheduling saves.
    """
    if (
        max_q is None
        or max_k is None
        or max_q < 256
        or max_q > 2048
        or max_k < max(4096, 4 * max_q)
        or num_q_heads % num_kv_heads != 0
    ):
        return 1
    qhead_per_kvhead = num_q_heads // num_kv_heads
    packed_rows = total_q * qhead_per_kvhead if pack_gqa else total_q
    scheduled_heads = num_kv_heads if pack_gqa else num_q_heads
    base_ctas = ((packed_rows + tile_m - 1) // tile_m) * scheduled_heads
    if base_ctas > 5 * sm_count:
        return 1
    splits = 8 if base_ctas <= 3 * sm_count and max_k >= 48 * 1024 else 4
    kv_tiles = max(1, (max_k + tile_n - 1) // tile_n)
    return min(splits, kv_tiles)

def _sm120_paged_tile_n(page_size: int) -> int:
    """Largest validated N tile that remains inside one physical KV page."""
    # Native page128 decode is fastest with N32; its smaller Q/K/V footprint
    # and split96 schedule mirror the dedicated SM103 decode organization.
    if page_size == 128:
        return 32
    # Hybrid Qwen page sizes are large but N64 carries a repeatable scheduling
    # cliff at long context; N48 wins for both page1056 and page1152.
    if page_size >= 1024 and page_size % 48 == 0:
        return 48
    for tile_n in (64, 48, 32, 16):
        if page_size % tile_n == 0:
            return tile_n
    return 0

def make_sm120_short_q_ragged_metadata(
    kv_lengths,
    *,
    query_length=1,
    max_splits=None,
    device=None,
):
    """Build graph inputs for a load-balanced SM120 Q1 or Q4 launch.

    Build this metadata outside CUDA graph capture and pass the returned
    tensors as ``num_splits_dynamic`` and ``work_map``. The compact work map
    uses one encoded int32 per CTA. Q1/M16 and Q4/M32 with one KV head both
    have ``m_block == head == 0``.
    """
    import math

    import torch

    lengths = [int(length) for length in kv_lengths]
    if query_length not in (1, 4):
        raise ValueError("compact SM120 metadata supports query_length 1 or 4")
    if not lengths or min(lengths) <= 0:
        raise ValueError("kv_lengths must contain positive lengths")
    if len(lengths) >= (1 << 15):
        raise ValueError("too many sequences for compact Q1 work encoding")
    if max_splits is None:
        max_splits = 48 if query_length == 4 else 96
    if not 1 < max_splits <= 128:
        raise ValueError("max_splits must be in [2, 128]")
    max_length = max(lengths)
    split_counts = [
        max(1, math.ceil(max_splits * length / max_length))
        for length in lengths
    ]
    encoded_work = [
        (sequence << 16) | split
        for sequence, split_count in enumerate(split_counts)
        for split in range(split_count)
    ]
    return (
        torch.tensor(split_counts, dtype=torch.int32, device=device),
        torch.tensor(encoded_work, dtype=torch.int32, device=device).view(-1, 1),
    )

def make_sm120_q1_ragged_metadata(kv_lengths, *, max_splits=96, device=None):
    """Backward-compatible Q1 wrapper for ragged SM120 metadata."""
    return make_sm120_short_q_ragged_metadata(
        kv_lengths,
        query_length=1,
        max_splits=max_splits,
        device=device,
    )

def _run_sm120(**kwargs):
    """Run the L20N path and preserve the public ``(out, lse)`` ABI.

    Host dispatch is based only on graph-static metadata and follows two
    routes:

    * Q1/Q2/Q3/Q4 short-Q -> native causal packed-GQA + SplitKV;
    * prefill and other attention shapes -> the general causal-varlen kernel.

    All routes keep Q, output and paged-KV storage owned by the caller.
    """
    import torch

    q = kwargs["q"]
    k = kwargs["k"]
    v = kwargs["v"]
    out = kwargs.get("out")
    cu_seqlens_q = kwargs.get("cu_seqlens_q")
    seqused_k = kwargs.get("seqused_k")
    page_table = kwargs.get("page_table")
    max_seqlen_q = kwargs.get("max_seqlen_q")
    requested_splits = kwargs.get("num_splits") or 0
    num_splits_dynamic = kwargs.get("num_splits_dynamic")
    work_map = kwargs.get("work_map")

    # Keep speculative verification as one native causal attention problem.
    # Q positions and grouped query heads are packed together inside the
    # kernel, matching the SM103 short-Q design without using its SM103-only
    # tcgen05/TMEM instructions.  page32 requires tile_n=32: tile_n=64 would
    # make blocks_per_page zero in the SM120 paged-KV mapper.
    can_run_native_short_q = (
        max_seqlen_q in (1, 2, 3, 4)
        and kwargs.get("causal")
        and q is not None
        and k is not None
        and v is not None
        and cu_seqlens_q is not None
        and seqused_k is not None
        and page_table is not None
        and kwargs.get("qv") is None
        and kwargs.get("learnable_sink") is None
        and not kwargs.get("return_lse")
        and not kwargs.get("softcap")
        and kwargs.get("window_size_left") is None
        and kwargs.get("window_size_right") is None
        and q.ndim == 3
        and k.ndim == 4
        and q.dtype in (
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        )
        and q.is_contiguous()
        and (out is None or out.is_contiguous())
    )
    if can_run_native_short_q:
        num_sequences = cu_seqlens_q.shape[0] - 1
        page_size = k.shape[1]
        tile_n = _sm120_paged_tile_n(page_size)
        if (
            num_sequences > 0
            and (out is None or out.shape == q.shape)
            and tile_n > 0
        ):
            is_fp8 = q.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
            if is_fp8 and page_size % 32 == 0:
                tile_n = 32
            num_kv_heads = k.shape[-2]
            sm_count = torch.cuda.get_device_properties(
                q.device
            ).multi_processor_count
            runtime_balanced_splits = (
                requested_splits == 0
                and 0 < num_sequences <= _SM120_RUNTIME_BALANCED_MAX_BATCH
                and num_sequences <= sm_count
                and num_kv_heads == 1
                and (page_size == 64 or (is_fp8 and page_size == 128))
                and q.shape[0] == num_sequences * max_seqlen_q
                and num_splits_dynamic is None
                and work_map is None
            )
            runtime_balanced_grid_size = 0
            if runtime_balanced_splits:
                if is_fp8:
                    runtime_balanced_grid_size = min(
                        2 * sm_count,
                        _SM120_FP8_DECODE_B4_Q1_GRID
                        if max_seqlen_q == 1 and num_sequences == 4
                        else _SM120_FP8_DECODE_GRID,
                    )
                elif num_sequences >= _SM120_RUNTIME_BALANCED_HIGH_BATCH:
                    runtime_balanced_grid_size = min(2 * sm_count, 256)
                else:
                    runtime_balanced_grid_size = min(
                        _SM120_RUNTIME_BALANCED_Q1_GRID
                        if max_seqlen_q == 1
                        else _SM120_RUNTIME_BALANCED_MTP_GRID,
                        sm_count,
                    )
            max_k = kwargs.get("max_seqlen_k") or page_table.shape[1] * page_size
            num_splits = (
                runtime_balanced_grid_size
                if runtime_balanced_splits
                else requested_splits
                or _sm120_short_q_splits(
                    num_sequences,
                    num_kv_heads,
                    max_k,
                    sm_count,
                    tile_n,
                    page_size,
                    max_seqlen_q,
                    is_fp8=is_fp8,
                )
            )
            if runtime_balanced_splits and max_seqlen_q == 1:
                tile_n = 32
            tile_m = (
                64
                if is_fp8
                else 32
                if (page_size == 128 and max_seqlen_q == 4)
                or (runtime_balanced_splits and max_seqlen_q == 1)
                else 16 if page_size == 128 else 64
            )
            short_q_kwargs = dict(kwargs)
            short_q_kwargs.update(
                tile_mn=(tile_m, tile_n),
                mma_pv_is_rs=False,
                intra_wg_overlap=True,
                pack_gqa=True,
                num_splits=num_splits,
                _runtime_balanced_splits=runtime_balanced_splits,
                _runtime_balanced_grid_size=runtime_balanced_grid_size,
                _runtime_balanced_batch_size=num_sequences,
                _allow_pack_gqa_split=True,
                _arch=120,
            )
            result = _sm120_forward()(**short_q_kwargs)
            return result[0], result[1]

    # General route for causal prefill, prefix-cache work and attention groups
    # not covered by the native short-Q specialization. A caller value of
    # num_splits=0 delegates the graph-static FP8 split policy to Atrex.
    sm120_kwargs = dict(kwargs)
    if (
        requested_splits == 0
        and q is not None
        and k is not None
        and q.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    ):
        num_q_heads = q.shape[-2]
        num_kv_heads = k.shape[-2]
        use_pack_gqa = kwargs.get("pack_gqa")
        if use_pack_gqa is None:
            use_pack_gqa = num_q_heads > num_kv_heads
        page_size = k.shape[1] if page_table is not None and k.ndim == 4 else None
        max_k = kwargs.get("max_seqlen_k")
        if max_k is None and page_size is not None:
            max_k = page_table.shape[1] * page_size
        sm120_kwargs["num_splits"] = (
            _sm120_fp8_prefill_splits(
                q.shape[0],
                num_q_heads,
                num_kv_heads,
                max_seqlen_q,
                max_k,
                torch.cuda.get_device_properties(q.device).multi_processor_count,
                pack_gqa=use_pack_gqa,
            )
            if kwargs.get("causal")
            and page_size in (64, 128)
            and cu_seqlens_q is not None
            and page_table is not None
            else 1
        )
    else:
        sm120_kwargs["num_splits"] = max(1, requested_splits)
    num_sequences = cu_seqlens_q.shape[0] - 1 if cu_seqlens_q is not None else 0
    min_k = kwargs.get("min_seqlen_k")
    max_k = kwargs.get("max_seqlen_k")
    page_size = k.shape[1] if page_table is not None and k is not None and k.ndim == 4 else None
    # Single-CTA CLC only wins at the measured page64 scheduling cliff. Keep
    # the static kernel for all other shapes, especially page128 and mild skew.
    sm120_kwargs["_use_clc"] = (
        sm120_kwargs["num_splits"] == 1
        and q is not None
        and q.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        and kwargs.get("causal")
        and page_size == 64
        and 3 <= num_sequences <= 4
        and max_seqlen_q == 1024
        and q.shape[0] == num_sequences * max_seqlen_q
        and min_k is not None
        and min_k > 0
        and max_k is not None
        and max_k >= 32 * min_k
        and not kwargs.get("return_lse")
        and kwargs.get("softcap") is None
        and kwargs.get("window_size_left") is None
        and kwargs.get("window_size_right") is None
    )
    # B64 has enough CTAs that fine-grained CLC only adds synchronization.
    # Preserve batch-local KV reuse and move whole requests only when a
    # device-side check detects both extreme skew and a heavy final wave.
    use_request_order = (
        sm120_kwargs["num_splits"] == 1
        and q is not None
        and q.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        and kwargs.get("causal")
        and page_size in (64, 128)
        and num_sequences == 64
        and max_seqlen_q is not None
        and 5 <= max_seqlen_q <= 2048
        and max_k is not None
        and max_k >= 256 * 1024
        and q.shape[0] == num_sequences * max_seqlen_q
        and seqused_k is not None
        and num_splits_dynamic is None
        and work_map is None
        and not kwargs.get("return_lse")
        and kwargs.get("softcap") is None
        and kwargs.get("window_size_left") is None
        and kwargs.get("window_size_right") is None
    )
    if use_request_order:
        sorted_k, request_order_i64 = torch.sort(
            seqused_k, descending=True, stable=True
        )
        has_extreme_skew = sorted_k[0] >= 32 * sorted_k[-1]
        has_heavy_tail = 4 * seqused_k[-8:].sum() >= seqused_k.sum()
        use_sorted_order = has_extreme_skew & has_heavy_tail
        # Both the flag and all indices are overwritten before the kernel.
        # Invocation-local storage also participates in graph-pool reuse.
        request_order = torch.empty(
            seqused_k.shape[0] + 1, device=seqused_k.device, dtype=torch.int32,
        )
        request_order[0].copy_(use_sorted_order)
        request_order[1:].copy_(request_order_i64)
        sm120_kwargs["request_order"] = request_order
    else:
        sm120_kwargs["request_order"] = None
    sm120_kwargs["_arch"] = 120
    result = _sm120_forward()(**sm120_kwargs)
    return result[0], result[1]

forward = _run_sm120
