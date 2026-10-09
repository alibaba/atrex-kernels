"""PPU raw-FP8 prefill: original prep_kv -> attention -> optional combine.

The native implementation is restored from 19916c26, with profiler symbol
renaming only. Internal 64-token page views, page-table remapping and K/V
packing do not modify the framework cache or its metadata. Temporary buffers
belong to the invocation, including its CUDA graph pool.

The source kernel supports only implicit or explicit unit descales and does not
return LSE. Scales must stay equal to one throughout any captured graph's life.
"""

import weakref

import torch

from atrex.core.compile_cu import cuda_kernel


_UNIT_DESCALES = {}


def _check_unit_descale(scale, requests, device):
    """Check fixed unit scales, not a dequantization operation.

    Cache only live, versioned Tensor views, never naked device pointers.
    Unversioned inference tensors are checked on every eager invocation.
    Graphs require prevalidated, versioned scales that remain fixed at one.
    """
    if scale is None:
        return
    if scale.shape != (requests, 1) or scale.dtype != torch.float32 or scale.device != device:
        raise ValueError("PPU descales must be float32 [requests,1] on the Q device")
    owner = scale if scale._base is None else scale._base
    try:
        version = scale._version
    except RuntimeError:  # Inference tensors have no mutation version counter.
        version = None
    key = (id(owner), scale.storage_offset(), tuple(scale.shape), tuple(scale.stride()))
    cached = _UNIT_DESCALES.get(key)
    if version is not None and cached is not None and cached[0]() is owner and cached[1] == version:
        return
    if torch.cuda.is_current_stream_capturing():
        raise NotImplementedError("Prevalidate fixed, versioned unit PPU descales before graph capture")
    if not bool(torch.all(scale == 1.0).item()):
        raise NotImplementedError("Original PPU kernel supports only unit descales")
    if version is not None:
        _UNIT_DESCALES[key] = (weakref.ref(owner), version)


@cuda_kernel(
    sources=("ppu/flash_attn/zwm890p/kernel.cu",),
    module_name="atrex_ppu_flash_attn_prepared",
    function_name="attn",
)
def _launch(*args):
    """Bound to the native entry on first call."""


def _default_splits(tokens, requests, max_seqlen_q, max_seqlen_k):
    """Preserve the source revision's graph-static launch policy."""
    if requests == 1 or max_seqlen_q > 16:
        return 1
    query_blocks = (tokens + 31) // 32 + requests
    waves = (156 + query_blocks // 2) // query_blocks
    splits = min(waves, 8) if waves >= 4 else 1
    return max(splits, min((max_seqlen_k + 16383) // 16384, 8))


def _page_view(tensor):
    """Original zero-copy view of interleaved blocks as 64-token pages."""
    num_blocks, block_size, num_heads, head_dim = map(int, tensor.shape)
    block_elements = block_size * num_heads * head_dim
    if tuple(tensor.stride()[1:]) != (num_heads * head_dim, head_dim, 1):
        raise NotImplementedError("Original PPU prep_kv requires contiguous rows within each block")
    if tensor.stride(0) <= 0 or tensor.stride(0) % block_elements:
        raise NotImplementedError("Original PPU prep_kv requires an integral block stride")
    interleave = tensor.stride(0) // block_elements
    subpages = block_size // 64
    virtual_stride = interleave * subpages
    if virtual_stride == 1 or num_blocks == 0:
        return tensor, virtual_stride, subpages
    page_elements = 64 * num_heads * head_dim
    view = torch.as_strided(
        tensor,
        (virtual_stride * (num_blocks - 1) + subpages, 64, num_heads, head_dim),
        (page_elements, num_heads * head_dim, head_dim, 1),
        tensor.storage_offset(),
    )
    return view, virtual_stride, subpages


def _remap_block_table(table, virtual_stride, subpages):
    """Adapt only an internal table; never mutate vLLM's block_table."""
    if virtual_stride == 1:
        return table
    remapped = torch.empty(
        (table.shape[0], table.shape[1] * subpages),
        dtype=torch.int32, device=table.device,
    )
    expanded = remapped.view(*table.shape, subpages)
    expanded.copy_(table.unsqueeze(-1)).mul_(virtual_stride)
    if subpages > 1:
        expanded.add_(torch.arange(subpages, dtype=torch.int32, device=table.device))
    return remapped


def forward(
    *, q, k, v, cu_seqlens_q, cu_seqlens_k, seqused_k,
    max_seqlen_q, max_seqlen_k, page_table, softmax_scale, causal,
    softcap, window_size_left, window_size_right, learnable_sink,
    out, return_lse, q_descale, k_descale, v_descale, num_splits,
):
    if q.dtype != torch.float8_e4m3fn or k.dtype != q.dtype or v.dtype != q.dtype:
        raise NotImplementedError("PPU native attention requires FP8 E4M3 Q/K/V")
    if q.ndim != 3 or tuple(q.shape[1:]) != (8, 256) or not q.is_contiguous():
        raise NotImplementedError("PPU attention requires contiguous Q [tokens,8,256]")
    if k.ndim != 4 or tuple(k.shape[2:]) != (1, 256) or v.shape != k.shape:
        raise NotImplementedError("PPU attention requires paged K/V [blocks,page_size,1,256]")
    if k.shape[1] < 64 or k.shape[1] % 64:
        raise NotImplementedError("Original PPU prep_kv requires page_size to be a multiple of 64")
    for tensor in (k, v):
        if tensor.stride(-1) != 1 or any(s <= 0 for s in tensor.stride()):
            raise ValueError("PPU KV requires positive strides and contiguous head dimensions")
        if tensor.numel() and tensor.data_ptr() % 16:
            raise ValueError("Original PPU prep_kv requires 16-byte aligned cache pointers")
    key_view, virtual_stride, subpages = _page_view(k)
    value_view, value_stride, value_subpages = _page_view(v)
    if (virtual_stride, subpages) != (value_stride, value_subpages):
        raise NotImplementedError("Original PPU prep_kv requires matching K/V block layouts")
    if not causal or window_size_left is not None or window_size_right is not None:
        raise NotImplementedError("PPU attention currently supports causal full-window attention")
    if softcap or learnable_sink is not None:
        raise NotImplementedError("PPU attention does not support softcap or sinks")
    if cu_seqlens_k is not None or page_table is None or seqused_k is None:
        raise NotImplementedError("PPU attention requires paged KV and seqused_k")
    requests = cu_seqlens_q.numel() - 1
    if cu_seqlens_q.ndim != 1 or requests < 0:
        raise ValueError("Invalid cu_seqlens_q")
    if seqused_k.shape != (requests,) or page_table.ndim != 2 or page_table.shape[0] != requests:
        raise ValueError("Sequence metadata batch dimensions must match")
    for tensor in (cu_seqlens_q, seqused_k, page_table):
        if tensor.dtype != torch.int32 or tensor.device != q.device or tensor.stride(-1) != 1:
            raise ValueError("Sequence metadata must be device int32 with contiguous rows")
    if not cu_seqlens_q.is_contiguous() or not seqused_k.is_contiguous():
        raise ValueError("Sequence lengths must be contiguous")
    if max_seqlen_k < 0 or max_seqlen_k > page_table.shape[1] * k.shape[1]:
        raise ValueError("Page table does not cover max_seqlen_k")
    if max_seqlen_q < 0:
        raise ValueError("max_seqlen_q must be nonnegative")
    for scale in (q_descale, k_descale, v_descale):
        _check_unit_descale(scale, requests, q.device)
    if abs(float(softmax_scale) - 256**-0.5) > 1e-12:
        raise NotImplementedError("Original PPU kernel requires softmax_scale=1/sqrt(256)")
    if return_lse:
        raise NotImplementedError("Original PPU kernel does not return softmax LSE")
    if out is None:
        out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
    elif out.shape != q.shape or out.dtype != torch.bfloat16 or out.device != q.device or not out.is_contiguous():
        raise ValueError("out must be contiguous BF16 with Q's shape and device")
    if q.shape[0] == 0:
        return out, None
    if requests == 0:
        raise ValueError("Nonempty Q requires sequence metadata")
    if max_seqlen_k == 0:
        out.zero_()
        return out, None
    # Keep the existing long-prefill policy. Short-Q calls may split the KV
    # range; all requests are processed without host-side batch partitioning.
    if num_splits == 0:
        num_splits = _default_splits(q.shape[0], requests, max_seqlen_q, max_seqlen_k)
    num_splits = min(num_splits, max(1, (max_seqlen_k + 63) // 64))
    extension = _launch.build(q.device)
    max_pages = (max_seqlen_k + 63) // 64
    remapped_table = _remap_block_table(page_table, virtual_stride, subpages)
    # Preserve the original prep_kv layout and native argument contract.
    packed_k = torch.empty(requests * max_pages * 16384, dtype=torch.uint8, device=q.device)
    packed_v = torch.empty_like(packed_k)
    k_scale = torch.empty(requests * max_pages, dtype=torch.float32, device=q.device)
    v_amax = torch.empty(requests * 256, dtype=torch.float32, device=q.device)
    partial = torch.empty(
        num_splits * q.shape[0] * 8 * 272 if num_splits > 1 else 0,
        dtype=torch.float32, device=q.device,
    )
    out.zero_()
    extension.prep_kv(
        key_view, value_view, remapped_table, seqused_k,
        packed_k, packed_v, k_scale, v_amax, max_pages,
    )
    extension.attn(
        q, packed_k, packed_v, k_scale, v_amax, cu_seqlens_q, seqused_k,
        out, partial, max_pages, (q.shape[0] + 31) // 32 + requests, num_splits,
    )
    if num_splits > 1:
        extension.combine(partial, out, num_splits)
    return out, None
