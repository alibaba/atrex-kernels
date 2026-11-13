"""BF16 prefill, decode and MTP dispatch with call-local scratch."""
from __future__ import annotations

import torch

import cutlass
from cutlass.cute.runtime import make_fake_stream
from ..common_utils.prefill_launch import PrefillLaunch, arena_sizes, launch_key, remember, scratch
from ..common_utils.runtime import _br_lane_cap, _compile, _cute_stream, _select_br, _sms_count

from .qsa_kernel import QSAForward
from ..common_utils.capacity import MAX_PREFILL_TOKENS, scratch_sizes
from ..common_utils.mtp_domain import eligible as mtp_eligible

_COMPILE_CACHE: dict = {}
_PRE_CACHE: dict = {}
_LAUNCH_CACHE: dict = {}

def _mtp_splits(rows: int, hkv: int, selected_blocks: int, sms: int) -> int:
    # Public shape only: at most 512 complete blocks plus a causal tail.
    # Balance whole 96-token tiles without launching an extra CTA wave.
    tiles = (selected_blocks * 4 + 3 + 95) // 96
    slots = max(1, sms // (rows * hkv))
    tiles_per_split = max(1, (tiles + slots - 1) // slots)
    return max(1, (tiles + tiles_per_split - 1) // tiles_per_split)


def run(
    q: torch.Tensor,
    k_cache: torch.Tensor,
    v_cache: torch.Tensor,
    indexer_block_ids: torch.Tensor,
    query_positions: torch.Tensor,
    block_table: torch.Tensor,
    token_to_req: torch.Tensor,
    use_prefill_config: bool,
    output_gate: torch.Tensor,
    out: torch.Tensor | None = None,
    _kvmajor_override: bool | None = None,
    _signature=None,
) -> torch.Tensor:
    plan_key = launch_key(_signature, None, 1.0, _kvmajor_override)
    hit = _LAUNCH_CACHE.get(plan_key)
    if hit is not None:
        return hit(q, k_cache, v_cache, indexer_block_ids, query_positions,
                   block_table, token_to_req, output_gate, out)
    capturing = torch.cuda.is_current_stream_capturing()
    if capturing and out is None:
        raise RuntimeError('CUDA graph capture requires a preallocated out tensor')
    logical_indices = indexer_block_ids
    if indexer_block_ids.dtype != torch.int32 or indexer_block_ids.ndim != 2:
        raise ValueError("indexer_block_ids must be int32 [Q, block_topk]")
    if query_positions.shape != (q.shape[0],) or query_positions.dtype not in (torch.int32, torch.int64):
        raise ValueError("query_positions must be int32/int64 [Q]")
    if not query_positions.is_contiguous() or not indexer_block_ids.is_contiguous():
        raise ValueError("compact IDs and positions must be contiguous")
    if indexer_block_ids.shape[0] != q.shape[0] or indexer_block_ids.shape[1] != 512:
        raise ValueError("this validated contract requires 512 selected blocks per query")
    if any(t.device != q.device for t in (indexer_block_ids, query_positions)):
        raise ValueError("compact inputs must be on the query device")
    if (not output_gate.is_contiguous() or output_gate.data_ptr()%32
            or (out is not None and (not out.is_contiguous() or out.data_ptr()%32))):
        raise ValueError("QSA requires contiguous, 32-byte-aligned gate/output")
    mtp = mtp_eligible(q, k_cache, block_table)
    if _kvmajor_override is None:
        # Q1 keeps a request's KV heads adjacent for paged-cache locality.
        _kvmajor_override = (not (mtp and q.shape[0] == block_table.shape[0])
                             and block_table.shape[0] > 1 and k_cache.shape[2] > 1)
    del use_prefill_config  # The public API selects the path from tensor shapes.
    if out is None:
        out = torch.empty(q.shape, dtype=q.dtype, device=q.device)

    R, H, D = q.shape
    P, S, Hkv, Dk = k_cache.shape
    kv_splits = _mtp_splits(R, Hkv, indexer_block_ids.shape[1], _sms_count(q.device)) if mtp else 1
    _km = bool(_kvmajor_override) and Hkv > 1
    _local_prefill = not mtp
    if _local_prefill and block_table.shape[1] > (MAX_PREFILL_TOKENS+S-1)//S:
        raise ValueError("prefill requires at most 1M tokens per request, rounded up to a page")
    _request_count = block_table.shape[0]
    _request_span = 0
    if block_table.shape[0] > 1:
        _request_span = block_table.shape[1] * S
        block_table = block_table.contiguous().view(1, -1)
    if v_cache.stride() != k_cache.stride() or v_cache.shape != k_cache.shape:
        raise NotImplementedError(
            "QSA requires k_cache/v_cache to share shape and strides")
    for t in (output_gate, out):
        if t.stride(2) != 1 or t.stride(1) % 8 or t.stride(0) % 8:
            raise NotImplementedError(
                "QSA requires 16B-compatible gate/out strides")
    sel_width = logical_indices.shape[1] * 4 + 3
    bt_width = block_table.shape[1]
    if logical_indices.data_ptr() % 16 or logical_indices.stride(1) != 1:
        raise NotImplementedError(
            "QSA requires contiguous 16B-aligned logical_indices")
    group = H // Hkv
    softmax_scale = float(D) ** -0.5

    _kst = 2
    _vst = 2
    _ssts = 2
    _ao = False
    _aoh_eff = True
    _pb = False
    _peel = 2
    _rpre = 1
    _gw = 2
    _srt = True
    _so = 6 if Hkv == 1 else 7
    _sp = 0
    _dr = 2
    _rs = False
    _ms_eff = True
    _qd = 3
    _qpf = 1
    _gpf = 1

    _sms = _sms_count(q.device)
    # The public contract guarantees four-token groups and aligned pages;
    # the page table above is now a single request-local address space.
    br = _select_br(R, Hkv, _sms, _br_lane_cap(group))
    _balanced_mtp = False
    _strided_mtp_union = False
    if mtp:
        # Require at least half an SM wave after Q4 coalescing. On 148 SMs,
        # B64/Hkv1 regresses from insufficient CTA parallelism, whereas
        # B128/Hkv1 and B64/Hkv2 both supply 128 independent union CTAs.
        # Q1 requests own independent KV pages: retain one row per CTA and
        # use KV splitting to expose parallelism for small decode batches.
        br = 4 if (R==4*_request_count and _request_count>=64 and
            2*_request_count*Hkv>=_sms and 4*group<=64) else 1
        from ..common_utils.decode_dispatch import _select_decode_config, _standard_layout, _serving_layout
        override = _select_decode_config(
            R, _request_count, Hkv, group, indexer_block_ids.shape[1], _sms, False,
            sm100=False, page=S, standard_layout=_standard_layout(
                q, k_cache, v_cache, indexer_block_ids, query_positions,
                block_table, token_to_req, output_gate, out),
            serving_layout=_serving_layout(q, k_cache, v_cache, indexer_block_ids,
                                          query_positions, block_table, token_to_req,
                                          output_gate, out),
            single_layer_layout=_serving_layout(q, k_cache, v_cache, indexer_block_ids,
                query_positions, block_table, token_to_req, output_gate, out, layers=1),
            kv_capacity=S * (block_table.numel() // _request_count))
        if override is not None:
            br = override[0]
        assert br in (1,2,4)
        if br>1:kv_splits=1
        _balanced_mtp = R==4*_request_count and br==1 and _sms<R*Hkv<=3*_sms//2
        if _balanced_mtp:kv_splits=2
        # Three Q4 partitions fit one CTA wave. Sharing KV amortizes the
        # gather cost when the 128-byte physical-address path is unavailable.
        _strided_mtp_union = bool(_balanced_mtp and group in (8,12)
            and 3*_request_count*Hkv<=_sms
            and (k_cache.stride(0)%64 or k_cache.stride(1)%64))
        if _strided_mtp_union:
            br=4
            kv_splits=3
        if override is not None:
            kv_splits = override[1]
            _balanced_mtp = _balanced_mtp and kv_splits == 2
            _strided_mtp_union = _strided_mtp_union and kv_splits == 3
    maxe = br * (sel_width // 4 + 1)
    nblocks = (R + br - 1) // br
    use_bmp = (block_table.shape[0] == 1
               and S >= 16 and S % 4 == 0
               and (sel_width + 1) % 4 == 0 and br <= 16)
    _wsm_eff = False
    _s0, _s1 = k_cache.stride(0), k_cache.stride(1)
    _q0, _q1 = _s0 // 64, _s1 // 64
    _ri_eff = False
    _em_eff = use_bmp
    tmax = (maxe + 15) // 16

    zero_pad_q = mtp
    active_softmax = zero_pad_q
    units = nblocks * Hkv * kv_splits
    _launched_ctas=max(1,units if mtp else min(_sms,units))
    _static_short=bool(_ms_eff and units<=2*_launched_ctas)
    _ordered = _request_count > 1 and not _static_short
    direct_pre = mtp
    # Only public shape/device properties select the measured union variants.
    # A separate prepass amortizes metadata across heads once Q4 blocks alone
    # exceed one SM wave; smaller grids avoid its launch/global-memory cost.
    union_separate=bool(mtp and br==4 and
        (Hkv>1 and nblocks>_sms))
    ratio16_split = mtp and group == 16 and Hkv == 1 and br == 4 and kv_splits > 1
    ratio16_union = mtp and group == 16 and br in (2, 4) and Hkv == 1 and (kv_splits == 1 or ratio16_split)
    union_v_warps=(2 if Hkv>1 else 1)
    if _strided_mtp_union:union_v_warps=1
    fused_meta = mtp and not union_separate
    local_meta = fused_meta
    single_unit_mma = mtp
    m_tile = 64 if mtp else 128
    decode_fixed=bool(mtp and R==_request_count and local_meta)
    decode_gate=bool(mtp and ((br==1 and group<=16) or (br==2 and group<=8))
                     and kv_splits==1 and m_tile==64 and single_unit_mma)
    decode_l2=64 if mtp and (R==_request_count or _balanced_mtp) else 128
    union_physical=bool((_balanced_mtp or decode_fixed) and
                        (fused_meta or union_separate) and _s0%64==0 and _s1%64==0
                        and (P-1)*_q0+max(S-4,0)*_q1<2**31)
    decode_fixed=decode_fixed and union_physical
    if capturing and not (mtp and _static_short and single_unit_mma
                           and (fused_meta or union_separate)):
        raise NotImplementedError('CUDA graph capture requires the static Q1/Q4 decode path; dynamic host tickets are unsupported')
    assert m_tile in (64,128)
    _pdl = not direct_pre
    _q_load_align = (32 if q.stride(-1) == 1 and q.data_ptr() % 32 == 0
                     and all((stride*q.element_size()) % 32 == 0
                             for stride in q.stride()[:2]) else 16)
    key = (
        q.device,
        _sms,
        tuple(q.shape),
        tuple(q.stride()),
        _q_load_align,
        tuple(k_cache.shape),
        tuple(k_cache.stride()),
        tuple(v_cache.stride()),
        tuple(logical_indices.shape),
        tuple(logical_indices.stride()),
        tuple(query_positions.shape),
        tuple(query_positions.stride()),
        query_positions.dtype,
        tuple(block_table.shape),
        tuple(block_table.stride()),
        tuple(token_to_req.shape),
        tuple(token_to_req.stride()),
        tuple(output_gate.stride()),
        tuple(out.stride()),
        q.dtype,
        k_cache.dtype,
        logical_indices.dtype,
        _kst,
        _vst,
        _ssts,
        _pdl,
        _ao,
        _aoh_eff,
        _pb,
        _em_eff,
        _peel,
        _rpre,
        _gw,
        _srt,
        _ri_eff,
        _so,
        _sp,
        _dr,
        _rs,
        _ms_eff,
        _qd,
        _qpf,
        _gpf,
        br,
        maxe,
        nblocks,
        _ordered,
        _km,
        _static_short,
        kv_splits,
        zero_pad_q,
        active_softmax,
        fused_meta,
        local_meta,
        m_tile,
        single_unit_mma,
        _request_span,
        union_v_warps,
        union_physical,
        union_separate,
        decode_fixed,
        decode_gate,
        decode_l2,
    )
    pre_key = (key, _local_prefill, _request_span)

    # Each invocation owns its scratch. PyTorch's stream-aware allocator
    # handles eager reuse; capture allocations belong to the graph pool.
    # No graph or model can retain these buffers through a module-global cache.
    ws_size, count_size, mask_size = scratch_sizes(R, br, Hkv, kv_splits, _rpre)
    # Fused decode consumes IDs/positions/request IDs directly. Retain the
    # bounds check above, but allocate metadata only for a real prepass.
    if _local_prefill:
        sizes = arena_sizes(ws_size, count_size, mask_size)
        workspace = scratch(q.device, sizes)
        mWS, mCount, mRM, mTicket = workspace
    else:
        if not fused_meta:
            mWS = torch.empty(ws_size, dtype=torch.int32, device=q.device)
            mCount = torch.empty(count_size, dtype=torch.int32, device=q.device)
            mRM = torch.empty(mask_size, dtype=torch.int32, device=q.device)
        mTicket = torch.empty(1, dtype=torch.int32, device=q.device)
        if not _static_short:
            # Dynamic prefill starts a fresh ticket epoch on every invocation.
            # Static Q1/Q4 never reads the ticket and needs no initialization.
            from ..common_utils.schedule import atrex_qsa_reset_ticket
            atrex_qsa_reset_ticket[(1,)](mTicket, num_warps=1)
    kernel_gate, kernel_out = output_gate, out
    if kv_splits > 1:
        kernel_gate = torch.empty((R*kv_splits,H,2), dtype=torch.float32, device=q.device)
        kernel_out = torch.empty((R*kv_splits,H,D), dtype=torch.float32, device=q.device)
    mQ, mK, mV = q, k_cache, v_cache
    mLI, mPos, mBT, mT2R = logical_indices, query_positions, block_table, token_to_req
    mGate, mOut = kernel_gate, kernel_out
    scale = cutlass.Float32(softmax_scale)
    nb = cutlass.Int32(nblocks)
    rows = cutlass.Int32(R)
    bt_rows = cutlass.Int32(_request_count)
    num_pages = cutlass.Int32(P)
    nct = cutlass.Int32(_launched_ctas)
    stream = make_fake_stream(use_tvm_ffi_env_stream=True) if _local_prefill else _cute_stream()

    pre_compiled = _PRE_CACHE.get(pre_key)
    if capturing and (key not in _COMPILE_CACHE
                      or (union_separate and pre_compiled is None)):
        raise RuntimeError('Compile and warm this Q1/Q4 configuration before CUDA graph capture')
    if pre_compiled is None and union_separate:
        from ..common_utils.union_preprocess import UnionPreprocess
        pre=UnionPreprocess(S,maxe,((maxe+23)//24)*24,2*tmax,_request_span,
                            True,False,union_physical)
        pre.atrex_union.set_name_prefix('atrex')
        pre_compiled = _compile(pre.__call__,
            (mLI,mPos,mT2R,mBT,mK,mWS,mCount,mRM,stream),
            [(16,1),(4,0),(4,0),(4,1),(16,3),(4,0),(4,0),(4,0)])
        _PRE_CACHE[pre_key]=pre_compiled
    elif pre_compiled is None and not direct_pre:
        from ..common_utils.local_bitmap import LocalBitmapPreprocess
        pre = LocalBitmapPreprocess(br, S, bt_width, _request_span,
                                    pdl=(_pdl and not _ordered))
        pre.atrex_qsa_prefill_local_bitmap.set_name_prefix("atrex")
        pre_compiled = _compile(pre.__call__,
            (mLI, mPos, mBT, mT2R, mWS, mCount, mRM, rows, bt_rows,
             num_pages, cutlass.Int32(_q0), cutlass.Int32(_q1), stream),
            [(16,1),(4,0),(4,1),(4,0),(4,0),(4,0),(4,0)])
        _PRE_CACHE[pre_key] = pre_compiled

    callWS,callCount,callRM=(mLI,mPos,mT2R) if fused_meta else (mWS,mCount,mRM)
    main_compiled = _COMPILE_CACHE.get(key)
    tmk = tmv = cutlass.Int64(0)
    if main_compiled is None:
        kernel_cls = QSAForward
        if m_tile == 64:
            from .mtp_m64 import QSAForward as kernel_cls
            if br == 2:
                from ..common_utils.decode_bf16 import QSAForward as kernel_cls
        kernel_options={"decode_reuse_v":decode_gate and br==1,"address_hoist":_balanced_mtp and union_physical,"deterministic_union":_strided_mtp_union or ratio16_split,"grouped_union":ratio16_union} if m_tile==64 else {}
        if ratio16_union and br == 2:
            kernel_options.update(pair_gate_stage=True, pair_direct_masks=True)
        op = kernel_cls(**kernel_options, group=group, head_dim=D, page_size=S,
                        static_short=_static_short, q_load_align=_q_load_align, kv_splits=kv_splits,
                        zero_pad_q=zero_pad_q, active_softmax=active_softmax, fused_metadata=fused_meta, local_metadata=local_meta, single_unit_mma=single_unit_mma,
                        metadata_request_span=_request_span if fused_meta else 0,
                        union_hash_mix=True,union_full_first=ratio16_union and br == 2,
                        union_v_warps=union_v_warps,
                        union_physical=union_physical,
                        decode_fixed=decode_fixed,decode_gate=decode_gate,decode_l2=decode_l2,
                        n_tile=96,ent_tile=24,
                        sel_width=sel_width, bt_width=bt_width, br=br, maxe=maxe,
                        hkv=Hkv,
                        k_stages=_kst, v_stages=_vst, s_stages=_ssts,
                        use_pdl=_pdl, arrive_on=_ao,
                        aohyb=_aoh_eff,
                        pair_bar=_pb, emask=_em_eff, peel=_peel,
                        gw=_gw, sum_tree=_srt,
                        rowidx=_ri_eff, sopt=_so, spin=_sp,
                        drain=_dr, rsplit=_rs,
                        mmasched=_ms_eff, qdepth=_qd,
                        qpf=_qpf, gpf=_gpf, ordered=_ordered, kvmajor=_km,
                        partial_bound=_em_eff,
                        single_bt=use_bmp and block_table.shape[0]==1,
                        compact_ws=use_bmp and block_table.shape[0]==1
                                   and not _ri_eff and not _wsm_eff)
        if m_tile == 64 and br == 2:
            op.atrex_qsa_bf16_decode.set_name_prefix("atrex")
        else:
            op.atrex_qsa_sm103_attention.set_name_prefix("atrex")
        main_compiled = _compile(op.__call__,
            (mQ, mK, mV, callWS, callCount, callRM, mTicket, mBT, mGate,
             mOut, scale, nb, nct, cutlass.Int32(0), cutlass.Int64(0),
             cutlass.Int64(0), stream),
            [(_q_load_align,2),(16,3),(16,3),
             (16,1) if fused_meta else (4,0), (4,0),(4,0),(4,0),
             (4,1),(16,2),(16,2)])
        _COMPILE_CACHE[key] = main_compiled
    if _local_prefill:
        plan = PrefillLaunch(
            pre_compiled, main_compiled, sizes,
            (rows, bt_rows, num_pages, cutlass.Int32(_q0), cutlass.Int32(_q1)),
            (scale, nb, nct, cutlass.Int32(0), tmk, tmv),
            nblocks, _static_short, _ordered)
        remember(_LAUNCH_CACHE, plan_key, plan)
        return plan(q, k_cache, v_cache, indexer_block_ids, query_positions,
                    block_table, token_to_req, output_gate, out, workspace)
    if union_separate:
        pre_compiled(mLI,mPos,mT2R,mBT,mK,mWS,mCount,mRM,stream)
    elif fused_meta:
        pass
    else:
        pre_compiled(mLI, mPos, mBT, mT2R, mWS, mCount, mRM, rows, bt_rows,
                     num_pages, cutlass.Int32(_q0), cutlass.Int32(_q1), stream)
    if _ordered:
        from ..common_utils.schedule import schedule
        schedule(mCount, nblocks, _pdl)
    tbase = cutlass.Int32(0)
    main_compiled(mQ, mK, mV, callWS, callCount, callRM, mTicket, mBT, mGate, mOut,
                  scale, nb, nct, tbase, tmk, tmv, stream)
    if kv_splits > 1:
        from ..common_utils.mtp_merge import merge
        merge(kernel_out, kernel_gate, output_gate, out, kv_splits, wide=_strided_mtp_union or ratio16_split)
    return out


# Host compilation/cache updates are serialized; GPU scratch is call-local.
# The lock covers enqueue only, never a device synchronization.
from threading import RLock
_HOST_LOCK = RLock()

def forward(*args, **kwargs):
    with _HOST_LOCK:
        return run(*args, **kwargs)
