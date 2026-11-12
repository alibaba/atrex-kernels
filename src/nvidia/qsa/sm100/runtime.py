"""SM100 prefill dispatch with call-local scratch and tensor-free JIT caches."""
from __future__ import annotations

from threading import RLock

import torch
import cutlass
from cutlass.cute.runtime import make_fake_stream
from ..common_utils.scales import _scale
from ..common_utils.prefill_launch import PrefillLaunch, arena_sizes, launch_key, remember, scratch

from ..common_utils.capacity import scratch_sizes
from ..common_utils.local_bitmap import LocalBitmapPreprocess
from ..common_utils.runtime import _compile, _select_br, _sms_count

_COMPILE_CACHE = {}
_PRE_CACHE = {}
_LAUNCH_CACHE = {}
_HOST_LOCK = RLock()


def _run(q, k_cache, v_cache, indexer_block_ids, query_positions,
         block_table, token_to_req, use_prefill_config, output_gate, out=None,
         *, sm_scale=None, v_scale=1.0, _signature=None):
    plan_key = launch_key(_signature, sm_scale, v_scale)
    hit = _LAUNCH_CACHE.get(plan_key)
    if hit is not None:
        return hit(q, k_cache, v_cache, indexer_block_ids, query_positions,
                   block_table, token_to_req, output_gate, out)
    # The public API performs metadata-only tensor and architecture validation.
    if torch.cuda.is_current_stream_capturing():
        raise NotImplementedError('SM100 QSA prefill does not support CUDA graph capture')
    del use_prefill_config
    fp8 = q.dtype == torch.float8_e4m3fn
    softmax_scale = _scale('sm_scale', 256**-0.5 if sm_scale is None else sm_scale)
    v_scale = _scale('v_scale', v_scale)
    if out is None:
        out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    rows, heads, dim = q.shape
    pages, page_size, hkv, _ = k_cache.shape
    requests, table_width = block_table.shape
    request_span = table_width * page_size if requests > 1 else 0
    table = block_table.view(1, -1)
    group = heads // hkv
    sms = _sms_count(q.device)
    br = _select_br(rows, hkv, sms, min(128 // group, 16))
    nblocks = (rows + br - 1) // br
    maxe = br * 513
    ncta = min(sms, nblocks * hkv)
    static_short = nblocks * hkv <= 2 * ncta
    ordered = requests > 1 and not static_short
    q_align = 32 if (q.data_ptr() % 32 == 0 and all(
        stride * q.element_size() % 32 == 0 for stride in q.stride()[:2])) else 16
    inputs = (q, k_cache, v_cache, indexer_block_ids, query_positions,
              block_table, token_to_req, output_gate, out)
    key = (q.device, sms, q_align, br, static_short, ordered,
           tuple((tuple(t.shape), tuple(t.stride()), t.dtype) for t in inputs))
    ws_size, count_size, mask_size = scratch_sizes(rows, br, hkv)
    sizes = arena_sizes(ws_size, count_size, mask_size)
    workspace = scratch(q.device, sizes)
    ws, count, mask, ticket = workspace
    stream = make_fake_stream(use_tvm_ffi_env_stream=True)
    pre_args = (indexer_block_ids, query_positions, table, token_to_req,
                ws, count, mask, cutlass.Int32(rows), cutlass.Int32(requests),
                cutlass.Int32(pages), cutlass.Int32(k_cache.stride(0) // 64),
                cutlass.Int32(k_cache.stride(1) // 64), stream)
    pre = _PRE_CACHE.get(key)
    if pre is None:
        op = LocalBitmapPreprocess(br, page_size, table.shape[1], request_span,
                                   pdl=not ordered, full_tile_groups=32 if fp8 else 24)
        op.atrex_qsa_prefill_local_bitmap.set_name_prefix('atrex')
        pre = _compile(op.__call__, pre_args,
                       [(16, 1), (4, 0), (4, 1), (4, 0), (4, 0), (4, 0), (4, 0)])
        _PRE_CACHE[key] = pre
    args = (q, k_cache, v_cache, ws, count, mask, ticket, table, output_gate, out,
            cutlass.Float32(softmax_scale))
    if fp8:
        args += (cutlass.Float32(v_scale),)
    args += (cutlass.Int32(nblocks), cutlass.Int32(ncta), cutlass.Int32(0),
             cutlass.Int64(0), cutlass.Int64(0), stream)
    main = _COMPILE_CACHE.get(key)
    if main is None:
        if fp8:
            from .prefill_fp8 import QSAForward
        else:
            from .prefill_bf16 import QSAForward
        op = QSAForward(
            group=group, head_dim=dim, page_size=page_size,
            static_short=static_short, q_load_align=q_align,
            n_tile=128 if fp8 else 96, ent_tile=32 if fp8 else 24,
            sel_width=2051, bt_width=table.shape[1], br=br, maxe=maxe, hkv=hkv,
            k_stages=2, v_stages=2, s_stages=2,
            use_pdl=True, arrive_on=False, aohyb=True,
            pair_bar=False, emask=True, peel=2, gw=2,
            sum_tree=True, rowidx=False, sopt=6 if hkv == 1 else 7,
            spin=0, drain=2, rsplit=False,
            mmasched=True, qdepth=3, qpf=1, gpf=1,
            ordered=ordered, kvmajor=requests > 1 and hkv > 1,
            partial_bound=True, single_bt=True, compact_ws=True)
        op.atrex_qsa_sm100_attention.set_name_prefix('atrex')
        main = _compile(op.__call__, args,
                        [(q_align, 2), (16, 3), (16, 3), (4, 0), (4, 0),
                         (4, 0), (4, 0), (4, 1), (16, 2), (16, 2)])
        _COMPILE_CACHE[key] = main
    plan = PrefillLaunch(pre, main, sizes, pre_args[7:-1], args[10:-1],
                         nblocks, static_short, ordered)
    remember(_LAUNCH_CACHE, plan_key, plan)
    return plan(q, k_cache, v_cache, indexer_block_ids, query_positions,
                block_table, token_to_req, output_gate, out, workspace)


def forward(*args, **kwargs):
    # Serialize compilation and enqueue, never synchronize the GPU under the lock.
    with _HOST_LOCK:
        return _run(*args, **kwargs)
