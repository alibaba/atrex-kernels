"""Native paged FP8 prefill with invocation-local layout conversion and scratch."""

import torch

from atrex.core.compile_cu import cuda_kernel


@cuda_kernel(
    sources=(
        "ppu/flash_attn/zwm890p/kernel.cu",
        "ppu/flash_attn/zwm890p/prefill_general.cu",
    ),
    include_dirs=("ppu/flash_attn/zwm890p/include",),
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
    from .launch import _normalized_call
    from .prefill_contract import validate

    validate(_normalized_call(locals()))
    key_view, virtual_stride, subpages = _page_view(k)
    value_view = _page_view(v)[0]
    if cu_seqlens_k is not None:
        raise NotImplementedError("PPU attention requires paged KV and seqused_k")
    requests = cu_seqlens_q.numel() - 1
    if cu_seqlens_q.ndim != 1 or requests < 0:
        raise ValueError("Invalid cu_seqlens_q")
    if page_table.ndim != 2:
        raise ValueError("Sequence metadata batch dimensions must match")
    for tensor in (cu_seqlens_q, seqused_k, page_table):
        if tensor.dtype != torch.int32 or tensor.device != q.device or tensor.stride(-1) != 1:
            raise ValueError("Sequence metadata must be device int32 with contiguous rows")
    if not cu_seqlens_q.is_contiguous() or not seqused_k.is_contiguous():
        raise ValueError("Sequence lengths must be contiguous")
    if max_seqlen_k < 0:
        raise ValueError("Page table does not cover max_seqlen_k")
    if max_seqlen_q < 0:
        raise ValueError("max_seqlen_q must be nonnegative")
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
    kv_heads = k.shape[2]
    descales = torch.empty((0,), dtype=torch.float32, device=q.device)
    if any(scale is not None for scale in (q_descale, k_descale, v_descale)):
        descales = torch.empty((3, requests, kv_heads), dtype=torch.float32, device=q.device)
        for index, scale in enumerate((q_descale, k_descale, v_descale)):
            if scale is None:
                descales[index].fill_(1.0)
            else:
                descales[index].copy_(scale)
    packed_k = torch.empty(requests * kv_heads * max_pages * 16384, dtype=torch.uint8, device=q.device)
    packed_v = torch.empty_like(packed_k)
    k_scale = torch.empty(requests * kv_heads * max_pages, dtype=torch.float32, device=q.device)
    v_amax = torch.empty(requests * kv_heads * 256, dtype=torch.float32, device=q.device)
    partial = torch.empty(
        num_splits * q.shape[0] * q.shape[1] * 272 if num_splits > 1 else 0,
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
        kv_heads, descales, softmax_scale,
    )
    if num_splits > 1:
        extension.combine(partial, out, num_splits)
    return out, None
