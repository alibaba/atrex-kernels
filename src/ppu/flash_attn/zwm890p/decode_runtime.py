"""Direct paged FP8 decode; the prefill implementation is never used here."""

import ctypes
import math

import torch


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
    if kv_heads <= 0 or query_heads not in (8 * kv_heads, 16 * kv_heads):
        raise NotImplementedError("PPU decode currently supports GQA=8 or GQA=16")
    if query not in (1, 4) or not 0 < batch <= 992 or not batch <= q.shape[0] <= batch * query:
        raise NotImplementedError("PPU decode supports nonempty Q1/Q4 requests and batch <= 992")
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
    if query == 4 and (splits > 1 or (splits == 0 and call["max_seqlen_k"] < 1024)):
        raise NotImplementedError("PPU Q4 decode supports automatic long-KV or num_splits=1")
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
    out, return_lse, q_descale, k_descale, v_descale, num_splits,
):
    from .launch import _normalized_call

    validate(_normalized_call(locals()))
    from .decode_build import load_library

    variant = "q1" if max_seqlen_q == 1 else "q4"
    library = load_library(variant, q.device)
    batch = cu_seqlens_q.numel() - 1
    total = q.shape[0]
    query_heads, head_dim = q.shape[1:]
    page_size, kv_heads = k.shape[1:3]
    if out is None:
        out = torch.empty(q.shape, device=q.device, dtype=torch.bfloat16)
    if any(scale is None for scale in (q_descale, k_descale, v_descale)):
        unit = torch.ones((batch, kv_heads), device=q.device, dtype=torch.float32)
        q_descale = unit if q_descale is None else q_descale
        k_descale = unit if k_descale is None else k_descale
        v_descale = unit if v_descale is None else v_descale
    allocated_splits = num_splits or 32
    lse = torch.empty((query_heads, total), device=q.device, dtype=torch.float32)
    partial = torch.empty((allocated_splits, query_heads, total, head_dim), device=q.device, dtype=torch.float32)
    partial_lse = torch.empty((allocated_splits, query_heads, total), device=q.device, dtype=torch.float32)
    semaphore = torch.zeros(32 + batch, device=q.device, dtype=torch.int32)
    dynamic_splits = torch.empty(batch + 1, device=q.device, dtype=torch.int32)
    tensors = (q, k, v, out, cu_seqlens_q, seqused_k, page_table,
               q_descale, k_descale, v_descale, lse, partial, partial_lse,
               semaphore, dynamic_splits)
    pointers = (ctypes.c_void_p * len(tensors))(*(tensor.data_ptr() for tensor in tensors))
    values = [batch, total, max_seqlen_q, page_table.shape[1] * page_size, k.shape[0], page_size,
              page_table.stride(0), q.stride(0), q.stride(1), *k.stride()[:3],
              *v.stride()[:3], out.stride(0), out.stride(1),
              *q_descale.stride(), *k_descale.stride(), *v_descale.stride(),
              torch.cuda.get_device_properties(q.device).multi_processor_count, query_heads, kv_heads]
    metadata = (ctypes.c_int64 * len(values))(*values)
    status = library.atrex_ppu_decode(
        pointers, metadata, softmax_scale, num_splits, 5 if max_seqlen_q == 1 else 4,
        torch.cuda.current_stream(q.device).cuda_stream,
    )
    if status:
        raise RuntimeError(f"PPU FP8 decode launch failed: {status}")
    return out, None
