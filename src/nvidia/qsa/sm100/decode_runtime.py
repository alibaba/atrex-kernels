"""SM100 Q1/Q4 dispatch with call-local scratch and warmed graph capture."""
from threading import RLock

import cutlass
from cutlass.cute.runtime import make_fake_stream
import torch

from ..common_utils.capacity import scratch_sizes
from ..common_utils.mtp_domain import eligible
from ..common_utils.runtime import _compile, _sms_count
from ..common_utils.decode_dispatch import _select_decode_config, _standard_layout, _serving_layout
from ..common_utils.scales import _scale

_COMPILE_CACHE = {}
_PRE_CACHE = {}
_LAUNCH_CACHE = {}
_HOST_LOCK = RLock()


def _launch(spec, q, k, v, ids, pos, table, req, gate, out):
    main, pre, scalars, splits, sizes, merger, merge_args, order_args = spec
    flat_table = table.view(1, -1)
    if pre is not None:
        ws, count, mask = (torch.empty(n, device=q.device, dtype=torch.int32) for n in sizes)
        pre(ids, pos, req, flat_table, k, ws, count, mask)
        metadata = (ws, count, mask)
    else:
        metadata = (ids, pos, req)
    # Static CTA scheduling never accesses the ticket; retain its tensor ABI.
    ticket = req
    if order_args is not None:
        from .unit_order import work_order
        ticket = work_order(metadata[1], *order_args)
    partial, stats = out, gate
    if splits > 1:
        rows, heads, dim = q.shape
        stats = torch.empty((rows * splits, heads, 2), device=q.device, dtype=torch.float32)
        partial = torch.empty((rows * splits, heads, dim), device=q.device, dtype=torch.float32)
    main(q, k, v, *metadata, ticket, flat_table, stats, partial, *scalars)
    if merger is not None:
        merger(partial, stats, gate, out, *merge_args)
    return out


def _configuration(rows, requests, hkv, group, selected_blocks, sms, fp8, override=None):
    """Balance Q4 KV reuse with the number of available SMs."""
    tile = 128 if fp8 else 96
    tiles = (selected_blocks * 4 + 3 + tile - 1) // tile
    slots = max(1, sms // (rows * hkv))
    tiles_per_split = max(1, (tiles + slots - 1) // slots)
    splits = max(1, (tiles + tiles_per_split - 1) // tiles_per_split)
    br = 4 if (rows == 4 * requests and requests >= 64
               and 2 * requests * hkv >= sms and 4 * group <= 64) else 1
    if fp8 and rows == 4 * requests and hkv == 1 and 32 <= requests < 128:
        br = 2
    if (not fp8 and br == 1 and rows == 4 * requests and requests >= 64
            and hkv == 1 and 2 * group <= 16):
        br = 2
    if override is not None:
        br = override[0]
    if br > 1:
        splits = 1
    if not fp8 and br == 1 and rows == 4 * requests and sms < rows * hkv <= 3 * sms // 2:
        splits = 2
    if fp8 and br == 2 and group == 12 and hkv == 1 and sms < (rows // br) * hkv <= 3 * sms // 2:
        splits = 2
    if override is not None:
        splits = override[1]
    blocks = (rows + br - 1) // br
    separate = br == 4 and hkv > 1 and blocks > sms
    if fp8 and br == 2 and hkv == 1 and (blocks * hkv > sms or splits > 1):
        separate = True
    stages = 3 if fp8 and rows == requests and splits == 1 else 2
    gate = (splits == 1 and br * group <= 16
            and (br in (1, 2) if fp8 else (rows == requests or br == 2)))
    return br, splits, separate, stages, gate


def _run(q, k_cache, v_cache, indexer_block_ids, query_positions,
         block_table, token_to_req, use_prefill_config, output_gate, out=None,
         *, sm_scale=None, v_scale=1.0, _signature=None):
    del use_prefill_config
    launch_key = (_signature, sm_scale, type(sm_scale), v_scale, type(v_scale))
    if _signature is not None and out is not None:
        try:
            spec = _LAUNCH_CACHE.get(launch_key)
        except TypeError:
            spec = None  # Invalid non-scalar scales use the normal validator.
        if spec is not None:
            return _launch(spec, q, k_cache, v_cache, indexer_block_ids,
                           query_positions, block_table, token_to_req, output_gate, out)
    if not eligible(q, k_cache, block_table):
        raise ValueError('SM100 decode requires Q1/Q4 with batch <= 256')
    capturing = torch.cuda.is_current_stream_capturing()
    if capturing and out is None:
        raise RuntimeError('CUDA graph capture requires a preallocated out tensor')
    if out is None:
        out = torch.empty(q.shape, device=q.device, dtype=torch.bfloat16)
    fp8 = q.dtype == torch.float8_e4m3fn
    scale = _scale('sm_scale', 256**-0.5 if sm_scale is None else sm_scale)
    v_scale = _scale('v_scale', v_scale)
    rows, heads, dim = q.shape
    pages, page, hkv, _ = k_cache.shape
    requests, width = block_table.shape
    group = heads // hkv
    sms = _sms_count(q.device)
    override = _select_decode_config(
        rows, requests, hkv, group, indexer_block_ids.shape[1], sms, fp8,
        sm100=True, page=page, standard_layout=_standard_layout(
            q, k_cache, v_cache, indexer_block_ids, query_positions,
            block_table, token_to_req, output_gate, out),
        serving_layout=_serving_layout(q, k_cache, v_cache, indexer_block_ids,
                                      query_positions, block_table, token_to_req,
                                      output_gate, out))
    br, splits, separate, stages, decode_gate = _configuration(
        rows, requests, hkv, group, indexer_block_ids.shape[1], sms, fp8, override)
    fp8_mtp_mid_batch = (fp8 and override == (4, 1) and rows == 4 * requests
                         and hkv == 1 and 80 <= requests <= 128
                         and page in (128, 256, 1056, 2112))
    balanced_mtp = (not fp8 and br == 1 and splits == 2 and rows == 4 * requests
                    and sms < rows * hkv <= 3 * sms // 2)
    balanced_fp8 = fp8 and br == 2 and splits == 2
    address32_ok = (
        0 < k_cache.stride(0) < 2**31 and k_cache.stride(0) % 64 == 0
        and 0 < k_cache.stride(1) < 2**31 and k_cache.stride(1) % 64 == 0
        and (pages - 1) * (k_cache.stride(0) // 64)
            + (page - 1) * (k_cache.stride(1) // 64) < 2**31)
    deep_fp8 = (fp8 and br == 4 and splits == 1 and hkv == 2
                and group == 12 and requests > sms and address32_ok)
    if deep_fp8 or fp8_mtp_mid_batch:
        stages = 3
    unit_order = (fp8 and br == 4 and splits == 1 and (hkv > 1 or group > 8)
                  and (rows // br) * hkv > sms and not (hkv > 1 and group > 8 and rows // br > sms))
    if unit_order:
        separate = True
    order_args = ((rows // br), hkv) if unit_order else None
    gate_factor = fp8 and splits == 1 and (br == 2 or rows == requests) and br * group < 64
    if gate_factor:
        decode_gate = True
    # A 64-byte unit retains the full address range with one shuffle.
    relative_address32 = (fp8 and ((br == 2 and hkv == 1) or deep_fp8 or fp8_mtp_mid_batch)
                          and address32_ok)
    blocks = (rows + br - 1) // br
    ncta = blocks * hkv * splits
    maxe = br * 513
    tmax = (maxe + 15) // 16
    request_span = width * page if requests > 1 else 0
    table = block_table.view(1, -1)
    fused = not separate
    physical = ((rows == requests or (not fp8 and br == 1 and (splits == 1 or balanced_mtp))) and fused
                and k_cache.stride(0) % 64 == 0 and k_cache.stride(1) % 64 == 0
                and (pages - 1) * (k_cache.stride(0) // 64)
                + max(page - 4, 0) * (k_cache.stride(1) // 64) < 2**31)
    q_align = 32 if q.data_ptr() % 32 == 0 and all(
        stride * q.element_size() % 32 == 0 for stride in q.stride()[:2]) else 16
    inputs = (q, k_cache, v_cache, indexer_block_ids, query_positions,
              block_table, token_to_req, output_gate, out)
    key = (q.device, sms, q_align, br, splits, separate, stages, physical,
           tuple((tuple(t.shape), tuple(t.stride()), t.dtype) for t in inputs))
    main = _COMPILE_CACHE.get(key)
    pre = _PRE_CACHE.get(key)
    if capturing and (main is None or (separate and pre is None)):
        raise RuntimeError('Compile and warm this Q1/Q4 configuration before CUDA graph capture')
    if separate:
        ws_size, count_size, mask_size = scratch_sizes(rows, br, hkv, splits, 1)
        ws = torch.empty(ws_size, device=q.device, dtype=torch.int32)
        count = torch.empty(count_size, device=q.device, dtype=torch.int32)
        mask = torch.empty(mask_size, device=q.device, dtype=torch.int32)
    ticket = token_to_req
    kernel_gate, kernel_out = output_gate, out
    if splits > 1:
        kernel_gate = torch.empty((rows * splits, heads, 2), device=q.device, dtype=torch.float32)
        kernel_out = torch.empty((rows * splits, heads, dim), device=q.device, dtype=torch.float32)
    if separate:
        pre_args = (indexer_block_ids, query_positions, token_to_req, table,
                    k_cache, ws, count, mask)
    if separate and pre is None:
        if fp8:
            from ..common_utils.fp8.decode.union_preprocess import UnionPreprocess
        else:
            from ..common_utils.union_preprocess import UnionPreprocess
        entries = 32 if fp8 else 24
        op = UnionPreprocess(page, maxe, ((maxe + entries - 1) // entries) * entries,
                             2 * tmax, request_span, True, False, False, **({"metadata_ready": br == 2 and separate and not unit_order, "direct_masks": br == 4 or balanced_fp8, "prefetch_k_tiles": 0 if balanced_fp8 else 2, "use_pdl": fp8 and separate and not unit_order and (br == 2 or (br == 4 and hkv > 1 and group > 8))} if fp8 else {}))
        op.atrex_union.set_name_prefix('atrex')
        pre = _compile(op.__call__, pre_args + (make_fake_stream(use_tvm_ffi_env_stream=True),),
                       [(16, 1), (4, 0), (4, 0), (4, 1), (16, 3), (4, 0), (4, 0), (4, 0)])
        _PRE_CACHE[key] = pre
    meta = (indexer_block_ids, query_positions, token_to_req) if fused else (ws, count, mask)
    args = (q, k_cache, v_cache, *meta, ticket, table, kernel_gate, kernel_out,
            cutlass.Float32(scale))
    if fp8:
        args += (cutlass.Float32(v_scale),)
    args += (cutlass.Int32(blocks), cutlass.Int32(ncta), cutlass.Int32(0),
             cutlass.Int64(0), cutlass.Int64(0))
    if main is None:
        if fp8:
            from .decode_fp8 import QSAForward
        else:
            from ..common_utils.decode_bf16 import QSAForward
        kernel_options = dict(gate_factor=gate_factor, unit_order=unit_order,
                              relative_address32=relative_address32, softmax_packed=balanced_fp8 or fp8_mtp_mid_batch) if fp8 else dict(address_hoist=balanced_mtp and physical)
        op = QSAForward(**kernel_options,
            group=group, head_dim=dim, page_size=page, static_short=True,
            q_load_align=q_align, kv_splits=splits, zero_pad_q=True,
            active_softmax=True, fused_metadata=fused, local_metadata=fused,
            single_unit_mma=True, metadata_request_span=request_span if fused else 0,
            union_hash_mix=True, union_full_first=False,
            union_v_warps=2 if hkv > 1 else 1, union_physical=physical,
            decode_fixed=physical and rows == requests, decode_gate=decode_gate,
            decode_l2=64 if rows == requests or balanced_mtp else 128,
            n_tile=128 if fp8 else 96, ent_tile=32 if fp8 else 24,
            sel_width=2051, bt_width=table.shape[1], br=br, maxe=maxe, hkv=hkv,
            k_stages=stages, v_stages=stages, s_stages=2,
            use_pdl=fp8 and separate and not unit_order and (br == 2 or (br == 4 and hkv > 1 and group > 8)), arrive_on=False, aohyb=True,
            pair_bar=False, emask=True, peel=2, gw=2,
            sum_tree=True, rowidx=False, sopt=6 if hkv == 1 else 7,
            spin=2 if fp8 and hkv == 1 and br > 1 else 0, drain=2, rsplit=False,
            mmasched=True, qdepth=3, qpf=1, gpf=1,
            ordered=False, kvmajor=(not fp8 and requests > 1 and hkv > 1
                and not (rows == requests and requests > sms)),
            partial_bound=True, single_bt=True, compact_ws=True)
        entry = op.atrex_qsa_sm100_decode if fp8 else op.atrex_qsa_bf16_decode
        entry.set_name_prefix('atrex')
        main = _compile(op.__call__, args + (make_fake_stream(use_tvm_ffi_env_stream=True),),
                        [(q_align, 2), (16, 3), (16, 3),
                         (16, 1) if fused else (4, 0), (4, 0), (4, 0), (4, 0),
                         (4, 1), (16, 2), (16, 2)])
        _COMPILE_CACHE[key] = main
    merger = None
    merge_args = ()
    if splits > 1:
        import triton
        if fp8:
            from ..common_utils.fp8.decode.mtp_merge import atrex_mtp_merge, MERGE_BLOCK_D, MERGE_WARPS
            if balanced_fp8:
                MERGE_BLOCK_D = 256
        else:
            from ..common_utils.mtp_merge import atrex_mtp_merge, merge_block_d, MERGE_WARPS
            MERGE_BLOCK_D = merge_block_d(rows, heads, splits)
        merge_args = (heads, splits, triton.next_power_of_2(splits), dim, MERGE_BLOCK_D)
        if fp8:
            merge_args += (v_scale,)
        grid = (rows * heads, dim // MERGE_BLOCK_D, 1)
        compiled_merge = atrex_mtp_merge.warmup(
            kernel_out, kernel_gate, output_gate, out, *merge_args,
            grid=grid, num_warps=MERGE_WARPS)
        merger = compiled_merge[grid]
    if _signature is not None:
        # Cache code and scalar launch metadata only; scratch stays call-local.
        if len(_LAUNCH_CACHE) >= 128:
            _LAUNCH_CACHE.clear()
        _LAUNCH_CACHE[launch_key] = (
            main, pre if separate else None, args[10:], splits,
            (ws_size, count_size, mask_size) if separate else None, merger, merge_args, order_args)
    if separate:
        pre(*pre_args)
    if order_args is not None:
        from .unit_order import work_order
        ticket = work_order(meta[1], *order_args)
        args = args[:6] + (ticket,) + args[7:]
    main(*args)
    if merger is not None:
        merger(kernel_out, kernel_gate, output_gate, out, *merge_args)
    return out


def forward(*args, **kwargs):
    with _HOST_LOCK:
        return _run(*args, **kwargs)
