"""Native paged FP8 prefill with invocation-local layout conversion and scratch."""

from functools import lru_cache
from math import prod

import torch

from atrex.core.compile_cu import cuda_kernel


@cuda_kernel(
    sources=(
        "ppu/flash_attn/zwm890p/prefill.cu",
        "ppu/flash_attn/zwm890p/prefill_general.cu",
    ),
    include_dirs=("ppu/flash_attn/zwm890p/include",),
    module_name="atrex_ppu_flash_attn_prepared",
    function_name="attn",
)
def _launch(*args):
    """Bound to the native entry on first call."""


@lru_cache(maxsize=None)
def _extension(device):
    return _launch.build(device)


# Includes packed K/V, page remapping, scales and grouping metadata. Inputs,
# the returned output and the separate FA Split-KV partials are not prep scratch.
# This is a grouping target, not a limit on supported request sizes.
_PREP_WORKSPACE_BYTES = 8 * 1024**3


@cuda_kernel(
    sources=("ppu/flash_attn/zwm890p/prefill_workspace.cu",),
    module_name="atrex_ppu_prefill_workspace",
    function_name="prepare",
)
def _workspace_launch(*args):
    """Graph-replayable metadata planning; no attention arithmetic."""


@lru_cache(maxsize=None)
def _workspace_extension(device):
    return _workspace_launch.build(device)


def _aligned_bytes(size):
    return (size + 511) // 512 * 512


def _group_layout(batch, capacity, heads, pages, scaled, direct_k=False):
    minimum = max(1, capacity // 2) if capacity < batch else batch
    slots = (batch + minimum - 1) // minimum
    # One allocation owns every prep temporary, including full-call descales.
    packed_pages = capacity * heads * pages
    specifications = (
        ("k", (0 if direct_k else packed_pages * 16384,), torch.uint8, 1),
        ("v", (packed_pages * 16384,), torch.uint8, 1),
        ("k_scale", (0 if direct_k else packed_pages,), torch.float32, 4),
        ("v_amax", (0 if direct_k else capacity * heads * 256,), torch.float32, 4),
        ("table", (capacity, pages), torch.int32, 4),
        ("cu", (capacity + 1,), torch.int32, 4),
        ("lengths", (capacity,), torch.int32, 4),
        ("scales", (3, batch, heads) if scaled else (0,), torch.float32, 4),
        ("group_scales", (3, capacity, heads) if scaled else (0,), torch.float32, 4),
        ("cost", (batch + 1,), torch.int64, 8),
        ("groups", (batch + 1,), torch.int32, 4),
        ("next", (batch + 1,), torch.int32, 4),
        ("starts", (slots + 1,), torch.int32, 4),
    )
    layout, total = {}, 0
    for name, shape, dtype, width in specifications:
        size = prod(shape) * width
        layout[name] = (total, size, shape, dtype)
        total += _aligned_bytes(size)
    return layout, total, minimum, slots


def _group_workspace(batch, heads, pages, scaled, device, direct_k=False):
    packed_planes = 1 if direct_k else 2
    lower, upper = 1, min(batch, _PREP_WORKSPACE_BYTES // (packed_planes * heads * pages * 16384))
    capacity = 0
    while lower <= upper:
        middle = (lower + upper) // 2
        if _group_layout(batch, middle, heads, pages, scaled, direct_k)[1] <= _PREP_WORKSPACE_BYTES:
            capacity, lower = middle, middle + 1
        else:
            upper = middle - 1
    # Requests cannot be split here. Allow one request above the target and
    # let the allocator report an OOM only if the actual allocation fails.
    capacity = max(1, capacity)
    layout, size, minimum, slots = _group_layout(batch, capacity, heads, pages, scaled, direct_k)
    storage = torch.empty(size, dtype=torch.uint8, device=device)
    views = {
        name: storage[offset:offset + length].view(dtype).view(shape)
        for name, (offset, length, shape, dtype) in layout.items()
    }
    return views, capacity, minimum, slots


@lru_cache(maxsize=None)
def _group_geometry(query_heads, kv_heads, device):
    # Mirror the existing main kernel's CTA geometry, without changing its ABI
    # or binary. Its 128-KiB shared memory and launch bounds cap residency.
    packed = query_heads == 6 and kv_heads == 1
    ratio = query_heads // kv_heads
    head_group = 4 if packed or ratio % 4 == 0 else 6 if ratio % 6 == 0 else 2
    properties = torch.cuda.get_device_properties(device)
    blocks = min(
        properties.shared_memory_per_multiprocessor // (4 * 2 * 64 * 256),
        properties.max_threads_per_multi_processor // (head_group * 64),
        1 if head_group > 4 else 2,
    )
    return (query_heads if packed else 1, 32 * head_group if packed else 32,
            1 if packed else query_heads // head_group,
            max(1, properties.multi_processor_count * blocks))


def _default_splits(tokens, requests, max_seqlen_q, max_seqlen_k, *, query_heads=0, kv_heads=0, page_size=0):
    """Select a launch policy from graph-static shape bounds."""
    if requests == 1 and (query_heads, kv_heads, page_size) == (32, 2, 64):
        if max_seqlen_q == 128:
            # A single request launches 32 CTAs. Deeper history amortizes the
            # merge and gains utilization from shorter KV shards.
            return 2 if max_seqlen_k - max_seqlen_q <= 32 * 1024 else 12
        if max_seqlen_q == 512:
            # Three splits place 384 CTAs into five two-CTA waves on 39 CUs.
            return 3
    if (requests == 1 and (query_heads, kv_heads, page_size) == (6, 1, 64)
            and 32 <= max_seqlen_q <= 512 and tokens == max_seqlen_q
            and max_seqlen_k - max_seqlen_q >= 16 * 1024):
        # The packed Air kernel puts 128 token-head rows in each CTA.
        ctas = (tokens * query_heads + 127) // 128
        return min(12, (78 + ctas - 1) // ctas)
    if (requests == 8 and (query_heads, kv_heads, page_size) == (6, 1, 64)
            and tokens == requests * max_seqlen_q):
        prefix = max_seqlen_k - max_seqlen_q
        if max_seqlen_q == 512 and prefix >= 64 * 1024:
            # 384 packed CTAs nearly fill five two-CTA waves on 39 CUs.
            return 2
        if 64 <= max_seqlen_q <= 128 and prefix >= 16 * 1024:
            return 2 if max_seqlen_q == 64 else 3
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


def _fill_descales(descales, sources):
    if descales.numel():
        for index, scale in enumerate(sources):
            if scale is None:
                descales[index].fill_(1.0)
            else:
                descales[index].copy_(scale)


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
    if (num_splits == 0 and max_seqlen_q == 16 and q.shape[0] == 16 * requests
            and requests in (1, 8, 64, 128) and q.shape[1] == 32 and k.shape[1:3] == (64, 2)
            and max_seqlen_k >= 16 * 1024 and softmax_scale in (None, 0.0625)
            and q.data_ptr() % 16 == 0 and out.data_ptr() % 16 == 0
            and q_descale is None and k_descale is None and v_descale is None):
        from .decode_runtime import _forward_validated

        # Small query tiles can consume the original pages. Keep long-Q
        # prefill on the prepared layout, which its mainloop reuses across CTAs.
        if requests == 1:
            splits = 19  # Two KV heads fill 38 of the 39 compute units.
        elif requests == 8:
            splits = 2 if max_seqlen_k <= 32 * 1024 + 16 else 12
        else:
            splits = 1 if max_seqlen_k <= 32 * 1024 + 16 else 3
        out = _forward_validated(
            q, k, v, cu_seqlens_q, seqused_k, max_seqlen_q, max_seqlen_k,
            page_table, 0.0625, out, q_descale, k_descale, v_descale, splits,
        )
        return out, None
    key_view, virtual_stride, subpages = _page_view(k)
    value_view = _page_view(v)[0]
    # Keep the existing long-prefill policy. Short-Q calls may split the KV
    # range; all requests are processed without host-side batch partitioning.
    automatic_splits = num_splits == 0
    if automatic_splits:
        num_splits = _default_splits(
            q.shape[0], requests, max_seqlen_q, max_seqlen_k,
            query_heads=q.shape[1], kv_heads=k.shape[2], page_size=k.shape[1],
        )
    num_splits = min(num_splits, max(1, (max_seqlen_k + 63) // 64))
    extension = _extension(q.device)
    max_pages = (max_seqlen_k + 63) // 64
    kv_heads = k.shape[2]
    scales = (q_descale, k_descale, v_descale)
    scaled = any(scale is not None for scale in scales)
    descale_shape = (3, requests, kv_heads) if scaled else (0,)
    query_blocks = (q.shape[0] + 31) // 32 + requests
    heads = (q.shape[1], kv_heads)
    prefix = max_seqlen_k - max_seqlen_q
    single_auto = automatic_splits and requests == 1 and q.shape[0] == max_seqlen_q
    direct_q128 = max_seqlen_q == 128 and (
        (heads == (32, 2) and (requests in (8, 64, 128) or single_auto))
        or (heads == (8, 1) and (requests == 8 or (single_auto and prefix >= 64 * 1024)))
    )
    direct_q512 = max_seqlen_q == 512 and (
        (heads == (32, 2) and requests == 8)
        or (heads == (8, 1) and prefix >= 64 * 1024 and (requests == 8 or single_auto))
    )
    direct_k = (
        (direct_q128 or direct_q512) and (num_splits == 1 or single_auto) and k.shape[1] == 64
        and max_seqlen_k >= 16 * 1024 and key_view.data_ptr() % 128 == 0
        and softmax_scale in (None, 0.0625) and not scaled
    )
    if direct_k:
        if single_auto and heads == (8, 1):
            # Fill at least 78 CTA slots instead of launching only 8 or 32 CTAs.
            num_splits = 10 if direct_q128 else 3
        elif (direct_q128 and automatic_splits and heads == (8, 1) and requests == 8
              and q.shape[0] == 128 * requests and prefix >= 64 * 1024):
            # Six shards nearly fill five two-CTA waves for this eight-head batch.
            num_splits = 6

    air_short_paged = (
        automatic_splits and heads == (6, 1) and k.shape[1] == 64
        and q.shape[0] == requests * max_seqlen_q
        and ((requests == 8 and 64 <= max_seqlen_q <= 128 and prefix >= 16 * 1024)
             or (requests == 72 and max_seqlen_q == 88 and prefix >= 64 * 1024))
        and (k.data_ptr() | v.data_ptr()) % 128 == 0
    )
    compact_air = air_short_paged and requests == 8 and 64 < max_seqlen_q <= 96
    if compact_air:
        # Four warps and two buffers keep two splits resident for these short queries.
        num_splits = 2
    partial = torch.empty(
        num_splits * q.shape[0] * q.shape[1] * 272 if num_splits > 1 else 0,
        dtype=torch.float32, device=q.device,
    )
    if air_short_paged:
        descales = torch.empty(descale_shape, dtype=torch.float32, device=q.device)
        _fill_descales(descales, scales)
        out.zero_()
        extension.attn_air_direct_kv(q, k, v, page_table, cu_seqlens_q, seqused_k, out,
                                     partial, num_splits, descales, softmax_scale, compact_air)
        if num_splits > 1:
            extension.combine(partial, out, num_splits)
        return out, None
    packed_pages = requests * kv_heads * max_pages
    packed_bytes = packed_pages * 16384
    full_bytes = sum(_aligned_bytes(size) for size in (
        0 if direct_k else packed_bytes,
        packed_bytes,
        0 if direct_k else packed_pages * 4,
        0 if direct_k else requests * kv_heads * 256 * 4,
        3 * requests * kv_heads * 4 if scaled else 0,
        requests * page_table.shape[1] * subpages * 4 if virtual_stride != 1 else 0,
        subpages * 4 if virtual_stride != 1 and subpages > 1 else 0,
    ))
    workspace = None
    slots = 1
    if full_bytes > _PREP_WORKSPACE_BYTES:
        workspace, capacity, minimum, slots = _group_workspace(
            requests, kv_heads, max_pages, scaled, q.device, direct_k,
        )
        descales = workspace["scales"]
        packed_k, packed_v = workspace["k"], workspace["v"]
        k_scale, v_amax = workspace["k_scale"], workspace["v_amax"]
        remapped_table = workspace["table"]
    else:
        remapped_table = _remap_block_table(page_table, virtual_stride, subpages)
        descales = torch.empty(descale_shape, dtype=torch.float32, device=q.device)
        packed_v = torch.empty(packed_bytes, dtype=torch.uint8, device=q.device)
        if not direct_k:
            packed_k = torch.empty_like(packed_v)
            k_scale = torch.empty(packed_pages, dtype=torch.float32, device=q.device)
            v_amax = torch.empty(requests * kv_heads * 256, dtype=torch.float32, device=q.device)
    _fill_descales(descales, scales)

    full_q_tiles = q.shape[0] == max_seqlen_q * requests and (
        direct_k or (heads == (32, 2) and max_seqlen_q == 512
                     and not scaled and softmax_scale in (None, 0.0625))
    )
    if workspace is None and full_q_tiles:
        query_blocks = q.shape[0] // 32
    else:
        out.zero_()

    group_cu, group_lengths, group_scales = cu_seqlens_q, seqused_k, descales
    if workspace is not None:
        planner = _workspace_extension(q.device)
        query_multiplier, query_tile, head_groups, resident = _group_geometry(q.shape[1], kv_heads, q.device)
        planner.plan(
            cu_seqlens_q, seqused_k, workspace["cost"], workspace["groups"],
            workspace["next"], workspace["starts"], capacity, minimum,
            query_multiplier, query_tile, head_groups, num_splits, resident,
        )
        group_cu, group_lengths, group_scales = workspace["cu"], workspace["lengths"], workspace["group_scales"]
        query_blocks = (q.shape[0] + 31) // 32 + capacity

    # Group count is graph-static. Each replay rebuilds live group metadata and
    # reuses the same packed workspace, with global Q offsets for the writes.
    for slot in range(slots):
        if workspace is not None:
            planner.prepare(
                page_table, cu_seqlens_q, seqused_k, descales, workspace["starts"],
                remapped_table, group_cu, group_lengths, group_scales,
                slot, virtual_stride, subpages,
            )
        if direct_k:
            extension.prep_v_direct_k(value_view, remapped_table, group_lengths, packed_v, max_pages)
            extension.attn_direct_k(
                q, key_view, packed_v, remapped_table, group_cu, group_lengths,
                out, partial, max_pages, query_blocks, num_splits,
            )
        else:
            extension.prep_kv(
                key_view, value_view, remapped_table, group_lengths,
                packed_k, packed_v, k_scale, v_amax, max_pages,
            )
            extension.attn(
                q, packed_k, packed_v, k_scale, v_amax, group_cu, group_lengths,
                out, partial, max_pages, query_blocks, num_splits,
                kv_heads, group_scales, softmax_scale,
            )
    if num_splits > 1:
        extension.combine(partial, out, num_splits)
    return out, None
