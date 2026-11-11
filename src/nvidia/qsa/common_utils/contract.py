"""Reuse successful static checks without retaining tensors or their storage."""
import torch


_VALIDATED = set()


def validate(q, k, v, ids, pos, table, req, gate, out=None):
    inputs = (q, k, v, ids, pos, table, req, gate)
    tensors = inputs if out is None else inputs + (out,)
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        return _validate(*inputs, out)
    # Every property used by the contract participates, including alignment,
    # gradients and output aliasing. Tensor contents remain device metadata.
    signature = tuple((t.shape, t.stride(), t.dtype, t.device,
                       t.requires_grad, t.data_ptr() % 32) for t in tensors)
    aliases = False
    if out is not None:
        storage = out.untyped_storage().data_ptr()
        aliases = any(storage == t.untyped_storage().data_ptr() for t in inputs)
    key = (signature, aliases)
    if key not in _VALIDATED:
        _validate(*inputs, out)
        if len(_VALIDATED) >= 128:
            _VALIDATED.clear()
        _VALIDATED.add(key)
    return signature


def _validate(q, k, v, ids, pos, table, req, gate, out=None):
    import torch
    tensors = (q, k, v, ids, pos, table, req, gate)
    if not all(isinstance(t, torch.Tensor) for t in tensors):
        raise ValueError("all inputs except use_prefill_config must be tensors")
    if any(t.device != q.device for t in tensors):
        raise ValueError("all tensors must be on the query device")
    if any(t.requires_grad for t in tensors):
        raise ValueError("qsa_compact is inference-only")
    if q.ndim != 3 or not 1 <= q.shape[0] < 2**31 or q.shape[2] != 256:
        raise ValueError("q must be [T,Hq,256] with positive signed-int32 T")
    if k.ndim != 4 or v.shape != k.shape or v.stride() != k.stride():
        raise ValueError("K/V must have identical four-dimensional shapes and strides")
    p, s, hkv, d = k.shape
    if p < 1 or s < 16 or s % 16 or hkv not in (1, 2) or d != 256:
        raise ValueError("K/V require P>0, page size a positive multiple of 16, Hkv=1/2, D=256")
    if q.shape[1] not in (8*hkv, 12*hkv):
        raise ValueError("GQA ratio must be 8 or 12")
    if q.dtype not in (torch.bfloat16, torch.float8_e4m3fn) or any(t.dtype != q.dtype for t in (k,v)):
        raise ValueError("Q/K/V must all be BF16 or all be FP8 E4M3FN")
    if gate.dtype != torch.bfloat16:
        raise ValueError("gate must be BF16")
    for t in (q, k, v):
        if t.stride(-1) != 1 or t.data_ptr() % 16 or any((x*t.element_size()) % 16 for x in t.stride()[:-1]):
            raise ValueError("Q/K/V require contiguous D and 16-byte-aligned rows/base")
        if any(t.stride(i) < t.stride(i+1)*t.shape[i+1] for i in range(t.ndim-1)):
            raise ValueError("Q/K/V must be non-overlapping in dimension order")
    t = q.shape[0]
    if ids.shape != (t,512) or ids.dtype != torch.int32 or not ids.is_contiguous() or ids.data_ptr()%16:
        raise ValueError("block IDs must be contiguous, 16-byte aligned int32 [T,512]")
    if pos.shape != (t,) or pos.dtype not in (torch.int32,torch.int64) or not pos.is_contiguous():
        raise ValueError("positions must be contiguous int32/int64 [T]")
    if req.shape != (t,) or req.dtype != torch.int32 or not req.is_contiguous():
        raise ValueError("request IDs must be contiguous int32 [T]")
    if table.ndim != 2 or table.dtype != torch.int32 or not table.is_contiguous() or table.shape[0] < 1 or table.shape[1] < 1:
        raise ValueError("page table must be contiguous int32 [B>0,L>0]")
    if table.numel() * s >= 2**31:
        raise ValueError("flattened page-table token capacity must be below 2**31")
    from .mtp_domain import eligible
    from .capacity import MAX_PREFILL_TOKENS
    if not eligible(q, k, table) and table.shape[1] > (MAX_PREFILL_TOKENS+s-1)//s:
        raise ValueError("prefill requires at most 1M tokens per request, rounded up to a page")
    if gate.shape != q.shape or not gate.is_contiguous() or gate.data_ptr()%32:
        raise ValueError("gate must match Q, contiguous and 32-byte aligned")
    if out is not None:
        if not isinstance(out,torch.Tensor) or out.device != q.device or out.dtype != torch.bfloat16 or out.shape != q.shape or not out.is_contiguous() or out.data_ptr()%32 or out.requires_grad:
            raise ValueError("out must be contiguous aligned BF16 on Q device, matching Q")
        if any(out.untyped_storage().data_ptr()==x.untyped_storage().data_ptr() for x in tensors):
            raise ValueError("out must not share storage with any input")
