"""Direct paged FP8 decode; the prefill implementation is never used here."""

import ctypes
from functools import lru_cache
import math

import torch

from . import decode_build


_MIN_PAGE_REORDER_KV = 64 * 1024
_MIN_SHORT_PAGE_REORDER_KV = 16 * 1024
_MAX_SHORT_PAGE_REORDER_KV = 32 * 1024 + 4
_DecodePointers = ctypes.c_void_p * 13
_DecodeMetadata = ctypes.c_int64 * 26
_get_current_raw_stream = getattr(torch._C, "_cuda_getCurrentRawStream", None)


@lru_cache(maxsize=None)
def _compute_units(device):
    return torch.cuda.get_device_properties(device).multi_processor_count


def validate(call, *, common_validated=False):
    q, k, v = call["q"], call["k"], call["v"]
    table = call["block_table"]
    offsets, lengths = call["cu_seqlens_q"], call["seqused_k"]
    batch = offsets.numel() - 1
    query = call["max_seqlen_q"]
    if call.get("fa_version") not in (None, 3):
        raise NotImplementedError("PPU decode requires fa_version=3")
    if q.dtype != torch.float8_e4m3fn or (not common_validated and (q.ndim != 3 or q.shape[-1] != 256)):
        raise NotImplementedError("PPU decode requires FP8 E4M3 packed HD256 Q")
    if table is None or (not common_validated and (k.ndim != 4 or k.shape[-1] != 256 or v.shape != k.shape)):
        raise NotImplementedError("PPU decode requires paged HD256 K/V")
    if not common_validated and (k.dtype != q.dtype or v.dtype != q.dtype):
        raise NotImplementedError("PPU decode requires matching FP8 E4M3 Q/K/V")
    page_size, kv_heads = k.shape[1:3]
    query_heads = q.shape[1]
    if page_size not in (64, 128, 256):
        raise NotImplementedError("PPU decode supports page sizes 64, 128, and 256")
    if kv_heads <= 0 or query_heads not in (6 * kv_heads, 8 * kv_heads, 12 * kv_heads, 16 * kv_heads):
        raise NotImplementedError("PPU decode supports GQA=6, 8, 12, or 16")
    if query not in range(1, 6) or not 0 < batch <= 992 or not batch <= q.shape[0] <= batch * query:
        raise NotImplementedError("PPU decode supports nonempty Q1-Q5 requests and batch <= 992")
    if not common_validated and (
        lengths is None or lengths.shape != (batch,) or table.ndim != 2 or table.shape[0] != batch
    ):
        raise ValueError("Inconsistent PPU decode sequence metadata")
    if not offsets.is_contiguous() or not lengths.is_contiguous() or (
        not common_validated and table.stride(-1) != 1
    ):
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
    q_row, q_head, q_last = q.stride()
    k_page, k_row, k_head, k_last = k.stride()
    v_page, v_row, v_head, v_last = v.stride()
    # Combine offsets before testing for 16-byte vector alignment.
    if (q_last != 1 or k_last != 1 or v_last != 1
            or min(q_row, q_head, k_page, k_row, k_head, v_page, v_row, v_head) <= 0
            or (q.data_ptr() | k.data_ptr() | v.data_ptr()) % 16):
        raise ValueError("PPU decode requires positive strides and aligned pointers")
    if (q_row | q_head | k_page | k_row | k_head | v_page | v_row | v_head) % 16:
        raise ValueError("PPU decode requires 16-byte aligned vector strides")
    out = call["out"]
    if out is not None and (out.data_ptr() % 16 or (not common_validated and not out.is_contiguous())):
        raise ValueError("PPU decode requires contiguous aligned BF16 out")
    device = q.device
    for scale in (call["q_descale"], call["k_descale"], call["v_descale"]):
        if scale is not None and (
            scale.dtype != torch.float32 or scale.shape != (batch, kv_heads) or scale.device != device
            or min(scale.stride()) < 0
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
    device = q.device
    total, query_heads, head_dim = q.shape
    cache_pages, page_size, kv_heads, _ = k.shape
    batch = cu_seqlens_q.numel() - 1
    variant = "q1" if max_seqlen_q == 1 else "q4"
    if max_seqlen_q == 16:
        variant = "q16_history" if batch == 128 and max_seqlen_k - max_seqlen_q >= 64 * 1024 else "q16_dual"
    elif max_seqlen_q == 4 and query_heads // kv_heads == 16:
        variant = "q4_wide"
    compute_units = _compute_units(device)
    table_capacity = page_table.shape[1] * page_size
    air_q4 = (max_seqlen_q == 4 and (query_heads, kv_heads, page_size) == (24, 2, 256)
              and batch in (64, 128) and total == 4 * batch and num_splits == 0)
    if air_q4 and ((batch == 64 and table_capacity <= 8192) or (batch == 128 and table_capacity <= 4096)):
        library = decode_build.load_library("q4_air_columns", device)
        num_splits = 1
    elif air_q4 and table_capacity == 8192:
        # The device planner skips inactive rows on padded graph replay.
        library = decode_build.load_library("q4_wide", device)
    else:
        library = decode_build.load_library(variant, device)
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
    if num_splits == 0 and (query_heads, kv_heads, page_size) == (24, 2, 256) and max_seqlen_k >= 1024:
        if batch == 1 and total == max_seqlen_q:
            if max_seqlen_q == 1:
                num_splits = 32 if max_seqlen_k <= 8 * 1024 else (64 if max_seqlen_k <= 32 * 1024 else 96)
            elif max_seqlen_q == 4:
                if max_seqlen_k <= 8 * 1024:
                    num_splits = 32
                else:
                    num_splits = 48 if max_seqlen_k <= 16 * 1024 else 64
        elif batch == 8 and max_seqlen_q == 1:
            num_splits = 4 if max_seqlen_k <= 2 * 1024 else 12
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
        # The sorter costs more than it saves at batch 16 and the first long-KV threshold.
        and (batch != 16 or max_seqlen_k > _MIN_PAGE_REORDER_KV)
        and (max_seqlen_k >= _MIN_PAGE_REORDER_KV or sort_short_history)
        and table_width <= 4096
    )
    partial_pointer = lse_pointer = 0
    if selected_splits > 1:
        rows = selected_splits * query_heads * total
        scratch = torch.empty(rows * (head_dim + 1), device=device, dtype=torch.float32)
        partial_pointer = scratch.data_ptr()
        lse_pointer = partial_pointer + rows * head_dim * scratch.element_size()
    plan = None
    if num_splits == 0:
        plan_entries = 4 + batch + batch * kv_heads * selected_splits
        if sort_history:
            plan_entries += batch * table_width
        plan = torch.empty(plan_entries, device=device, dtype=torch.int32)
    pointers = _DecodePointers(
        q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr(),
        cu_seqlens_q.data_ptr(), seqused_k.data_ptr(), page_table.data_ptr(),
        q_descale.data_ptr() if q_descale is not None else 0,
        k_descale.data_ptr() if k_descale is not None else 0,
        v_descale.data_ptr() if v_descale is not None else 0,
        partial_pointer, lse_pointer, plan.data_ptr() if plan is not None else 0,
    )
    metadata = _DecodeMetadata(
        batch, total, max_seqlen_q, table_width * page_size, cache_pages, page_size,
        page_table.stride(0), *q.stride()[:2], *k.stride()[:3], *v.stride()[:3], *out.stride()[:2],
        *((0, 0) if q_descale is None else q_descale.stride()),
        *((0, 0) if k_descale is None else k_descale.stride()),
        *((0, 0) if v_descale is None else v_descale.stride()),
        compute_units, query_heads, kv_heads,
    )
    # A private ABI selector enables the native sorter and its appended workspace.
    native_splits = -1 if sort_history else num_splits
    status = library.atrex_ppu_decode(
        pointers, metadata, softmax_scale, native_splits,
        _get_current_raw_stream(device.index) if _get_current_raw_stream is not None
        else torch.cuda.current_stream(device).cuda_stream,
    )
    if status:
        raise RuntimeError(f"PPU FP8 decode launch failed: {status}")
    return out
