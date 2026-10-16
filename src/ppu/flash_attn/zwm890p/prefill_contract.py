"""Side-effect-free eligibility for the unchanged original prefill kernel."""

import torch


def validate(call, *, eligibility_only=False):
    q, k, v = (call[name] for name in ("q", "k", "v"))
    table, lengths = call["block_table"], call["seqused_k"]
    batch = call["cu_seqlens_q"].numel() - 1
    if q.dtype != torch.float8_e4m3fn:
        raise NotImplementedError("PPU attention requires FP8 E4M3 Q/K/V")
    if tuple(q.shape[1:]) != (8, 256) or not q.is_contiguous():
        raise NotImplementedError("PPU prefill requires contiguous Q [tokens,8,256]")
    if table is None or tuple(k.shape[2:]) != (1, 256):
        raise NotImplementedError("PPU prefill requires paged K/V [blocks,page_size,1,256]")
    if k.shape[1] < 64 or k.shape[1] % 64:
        raise NotImplementedError("PPU prep_kv requires page_size to be a multiple of 64")
    block_elements = k.shape[1] * k.shape[2] * k.shape[3]
    layouts = []
    for tensor in (k, v):
        if any(stride <= 0 for stride in tensor.stride()):
            raise ValueError("PPU KV requires positive strides")
        if tuple(tensor.stride()[1:]) != (256, 256, 1):
            raise NotImplementedError("PPU prep_kv requires contiguous rows within each block")
        if tensor.stride(0) % block_elements:
            raise NotImplementedError("PPU prep_kv requires an integral block stride")
        if tensor.numel() and tensor.data_ptr() % 16:
            raise ValueError("PPU prep_kv requires 16-byte aligned cache pointers")
        layouts.append(tensor.stride(0) // block_elements)
    if layouts[0] != layouts[1]:
        raise NotImplementedError("PPU prep_kv requires matching K/V block layouts")
    if not call["causal"] or call["window_size"] not in (None, (-1, -1), [-1, -1]):
        raise NotImplementedError("PPU attention supports causal full-window attention")
    if call["softcap"] or call["s_aux"] is not None:
        raise NotImplementedError("PPU attention does not support softcap or sinks")
    if call["return_softmax_lse"]:
        raise NotImplementedError("PPU attention does not return softmax LSE")
    if lengths is None or lengths.shape != (batch,) or table.shape[0] != batch:
        raise ValueError("PPU sequence metadata batch dimensions must match")
    if call["max_seqlen_k"] > table.shape[1] * k.shape[1]:
        raise ValueError("PPU block_table does not cover max_seqlen_k")
    if eligibility_only and any(call[name] is not None for name in ("q_descale", "k_descale", "v_descale")):
        raise NotImplementedError("PPU can_use cannot prove device descales are fixed at one")
    scale = call["softmax_scale"]
    if scale is not None and abs(float(scale) - 256**-0.5) > 1e-12:
        raise NotImplementedError("PPU attention requires softmax_scale=1/sqrt(256)")
