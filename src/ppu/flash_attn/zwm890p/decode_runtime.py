"""Direct paged FP8 decode; the prefill implementation is never used here."""

import ctypes
from functools import lru_cache
import math

import torch


_MIN_PAGE_REORDER_KV = 64 * 1024
_MIN_SHORT_PAGE_REORDER_KV = 16 * 1024
_MAX_SHORT_PAGE_REORDER_KV = 32 * 1024 + 4
_DecodePointers = ctypes.c_void_p * 13
_DecodeMetadata = ctypes.c_int64 * 26


@lru_cache(maxsize=None)
def _compute_units(device):
    return torch.cuda.get_device_properties(device).multi_processor_count


def validate(call):
    q, k, v = (call[name] for name in ("q", "k", "v"))
    table = call["block_table"]
    offsets, lengths = call["cu_seqlens_q"], call["seqused_k"]
    batch = offsets.numel() - 1
    query = call["max_seqlen_q"]
    if call.get("fa_version") not in (None, 3):
        raise NotImplementedError("PPU decode requires fa_version=3")
    if q.dtype != torch.float8_e4m3fn or q.ndim != 3 or q.shape[-1] != 256:
        raise NotImplementedError("PPU decode requires FP8 E4M3 packed HD256 Q")
    if table is None or k.ndim != 4 or k.shape[-1] != 256 or v.shape != k.shape:
        raise NotImplementedError("PPU decode requires paged HD256 K/V")
    if k.dtype != q.dtype or v.dtype != q.dtype:
        raise NotImplementedError("PPU decode requires matching FP8 E4M3 Q/K/V")
    page_size, kv_heads = k.shape[1:3]
    query_heads = q.shape[1]
    if page_size not in (64, 128, 256):
        raise NotImplementedError("PPU decode supports page sizes 64, 128, and 256")
    if kv_heads <= 0 or query_heads not in (6 * kv_heads, 8 * kv_heads, 12 * kv_heads, 16 * kv_heads):
        raise NotImplementedError("PPU decode supports GQA=6, 8, 12, or 16")
    if query not in range(1, 6) or not 0 < batch <= 992 or not batch <= q.shape[0] <= batch * query:
        raise NotImplementedError("PPU decode supports nonempty Q1-Q5 requests and batch <= 992")
    if lengths is None or lengths.shape != (batch,) or table.ndim != 2 or table.shape[0] != batch:
        raise ValueError("Inconsistent PPU decode sequence metadata")
    if not offsets.is_contiguous() or not lengths.is_contiguous() or table.stride(-1) != 1:
        raise ValueError("PPU decode metadata requires contiguous rows")
    if not 0 < call["max_seqlen_k"] <= table.shape[1] * page_size:
        raise ValueError("PPU decode block_table must cover positive max_seqlen_k")
    if not call["causal"] or call["window_size"] not in (None, (-1, -1), [-1, -1]):
        raise NotImplementedError("PPU decode requires causal full-window attention")
    if call["softcap"] or call["logits_soft_cap"] or call["s_aux"] is not None:
        raise NotImplementedError("PPU decode does not support softcap or sinks")
    if call["return_softmax_lse"]:
        raise NotImplementedError("PPU decode does not expose softmax LSE")
    splits = call["num_splits"]
    if not isinstance(splits, int) or splits not in range(33):
        raise NotImplementedError("PPU decode supports num_splits from 0 to 32")
    for tensor in (q, k, v):
        if tensor.stride(-1) != 1 or any(stride <= 0 for stride in tensor.stride()) or tensor.data_ptr() % 16:
            raise ValueError("PPU decode requires positive strides and aligned pointers")
        if any(stride % 16 for stride in tensor.stride()[:-1]):
            raise ValueError("PPU decode requires 16-byte aligned vector strides")
    out = call["out"]
    if out is not None and (not out.is_contiguous() or out.data_ptr() % 16):
        raise ValueError("PPU decode requires contiguous aligned BF16 out")
    for name in ("q_descale", "k_descale", "v_descale"):
        scale = call[name]
        if scale is not None and (
            scale.dtype != torch.float32 or scale.shape != (batch, kv_heads) or scale.device != q.device
            or any(stride < 0 for stride in scale.stride())
        ):
            raise ValueError("PPU decode descales must be float32 [batch,KV heads] on Q's device")
    scale = call["softmax_scale"]
    if scale is not None and (not math.isfinite(scale) or scale <= 0):
        raise ValueError("PPU decode requires a positive finite softmax_scale")


def forward(
    *, q, k, v, cu_seqlens_q, cu_seqlens_k, seqused_k,
    max_seqlen_q, max_seqlen_k, page_table, softmax_scale, causal,
    softcap, window_size_left, window_size_right, learnable_sink,
    out, return_lse, q_descale, k_descale, v_descale, num_splits, fa_version=None,
):
    from .launch import _normalized_call

    validate(_normalized_call(locals()))
    out = _forward_validated(
        q, k, v, cu_seqlens_q, seqused_k, max_seqlen_q, max_seqlen_k, page_table,
        softmax_scale, out, q_descale, k_descale, v_descale, num_splits,
    )
    return out, None


def _forward_validated(
    q, k, v, cu_seqlens_q, seqused_k, max_seqlen_q, max_seqlen_k, page_table,
    softmax_scale, out, q_descale, k_descale, v_descale, num_splits,
):
    """Launch after the caller validates the public decode contract."""
    from .decode_build import load_library

    device = q.device
    total, query_heads, head_dim = q.shape
    cache_pages, page_size, kv_heads, _ = k.shape
    variant = "q1" if max_seqlen_q == 1 else "q4"
    if max_seqlen_q == 16:
        variant = "q16_dual"
    elif max_seqlen_q == 4 and query_heads // kv_heads == 16:
        variant = "q4_wide"
    batch = cu_seqlens_q.numel() - 1
    compute_units = _compute_units(device)
    library = load_library(variant, device)
    if out is None:
        out = torch.empty(q.shape, device=device, dtype=torch.bfloat16)
    if num_splits == 0 and (query_heads, kv_heads, page_size) == (32, 2, 64) and max_seqlen_k >= 16 * 1024:
        if batch == 1 and total == max_seqlen_q and max_seqlen_q in (1, 4):
            # The batch-one grid has only two KV heads. More static shards
            # expose enough CTAs without running a device-side scheduling pass.
            short_history = max_seqlen_k <= 16 * 1024
            if max_seqlen_q == 1:
                num_splits = 64 if short_history else 96
            else:
                num_splits = 48 if short_history else 64
        elif max_seqlen_q == 1 and batch in (8, 16):
            num_splits = 96 // batch
    selected_splits = num_splits or 32
    table_width = page_table.shape[1]
    sort_short_history = (
        batch >= 64
        and _MIN_SHORT_PAGE_REORDER_KV <= max_seqlen_k <= _MAX_SHORT_PAGE_REORDER_KV
        and table_width <= 513
    )
    sort_history = (
        variant == "q4_wide"
        and num_splits == 0
        and kv_heads == 2
        and page_size == 64
        and batch >= 8
        and (max_seqlen_k >= _MIN_PAGE_REORDER_KV or sort_short_history)
        and table_width <= 4096
    )
    partial = partial_lse = None
    if selected_splits > 1:
        partial = torch.empty((selected_splits, query_heads, total, head_dim), device=device, dtype=torch.float32)
        partial_lse = torch.empty((selected_splits, query_heads, total), device=device, dtype=torch.float32)
    plan = None
    if num_splits == 0:
        plan_entries = 4 + batch + batch * kv_heads * selected_splits
        if sort_history:
            plan_entries += batch * table_width
        plan = torch.empty(plan_entries, device=device, dtype=torch.int32)
    tensors = (q, k, v, out, cu_seqlens_q, seqused_k, page_table,
               q_descale, k_descale, v_descale, partial, partial_lse, plan)
    pointers = _DecodePointers(*(tensor.data_ptr() if tensor is not None else 0 for tensor in tensors))
    descale_strides = [
        stride
        for scale in (q_descale, k_descale, v_descale)
        for stride in ((0, 0) if scale is None else scale.stride())
    ]
    metadata = _DecodeMetadata(
        batch, total, max_seqlen_q, table_width * page_size, cache_pages, page_size,
        page_table.stride(0), *q.stride()[:2], *k.stride()[:3], *v.stride()[:3], *out.stride()[:2],
        *descale_strides, compute_units, query_heads, kv_heads,
    )
    # A private ABI selector enables the native sorter and its appended workspace.
    native_splits = -1 if sort_history else num_splits
    status = library.atrex_ppu_decode(
        pointers, metadata, softmax_scale, native_splits,
        torch.cuda.current_stream(device).cuda_stream,
    )
    if status:
        raise RuntimeError(f"PPU FP8 decode launch failed: {status}")
    return out
