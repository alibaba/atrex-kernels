"""Private runtime for the AKA SM103 BF16 q4 decode specialization."""

from __future__ import annotations

import math
from threading import Lock

import cutlass
import cutlass.cute as cute
import torch
from cutlass.cute.typing import BFloat16, Float32, Int32

from .aka_decode_cutedsl import (
    CausalMask,
    GroupedQueryAttentionDecode,
    GroupedQueryAttentionDecodePaged,
    warp_threads,
)


_ATREX_AKA_NUM_Q_HEADS = 16
_ATREX_AKA_NUM_KV_HEADS = 1
_ATREX_AKA_HEAD_DIM = 256
_ATREX_AKA_QUERY_LENGTH = 4
_ATREX_AKA_EXTERNAL_PAGE_SIZE = 128
_ATREX_AKA_KERNEL_PAGE_SIZE = 64
_ATREX_AKA_SPLIT_CAP = 32

_ATREX_AKA_KERNEL_CACHE: dict[tuple[int, tuple[int, int]], object] = {}
_ATREX_AKA_KERNEL_CACHE_LOCK = Lock()


def _atrex_aka_workspace_elements(
    workspace_splits: int,
    batch_size: int,
) -> int:
    output = (
        workspace_splits
        * batch_size
        * _ATREX_AKA_QUERY_LENGTH
        * _ATREX_AKA_NUM_Q_HEADS
        * _ATREX_AKA_HEAD_DIM
    )
    partial_stats = (
        workspace_splits
        * batch_size
        * _ATREX_AKA_QUERY_LENGTH
        * _ATREX_AKA_NUM_Q_HEADS
    )
    return output + 2 * partial_stats


def _atrex_aka_slice_workspace(
    workspace: torch.Tensor,
    workspace_splits: int,
    batch_size: int,
):
    o_shape = (
        workspace_splits,
        batch_size,
        _ATREX_AKA_QUERY_LENGTH,
        _ATREX_AKA_NUM_Q_HEADS,
        _ATREX_AKA_HEAD_DIM,
    )
    stat_shape = o_shape[:-1]
    o_elements = math.prod(o_shape)
    stat_elements = math.prod(stat_shape)
    required = o_elements + 2 * stat_elements
    if workspace.dtype != torch.float32 or not workspace.is_contiguous():
        raise ValueError("AKA decode workspace must be contiguous float32")
    if workspace.numel() < required:
        raise ValueError(
            f"AKA decode workspace has {workspace.numel()} elements; needs {required}"
        )
    o_partial = workspace[:o_elements].view(o_shape)
    l_partial = workspace[o_elements : o_elements + stat_elements].view(stat_shape)
    m_partial = workspace[
        o_elements + stat_elements : o_elements + 2 * stat_elements
    ].view(stat_shape)
    return o_partial, l_partial, m_partial


def _atrex_aka_split_config(
    batch_size: int,
    max_pages: int,
    device_index: int,
) -> tuple[int, int]:
    # Q4 uses two prediction rows per MMA tile. Reserving half a CTA per
    # request keeps the balanced launch inside one hardware wave.
    prediction_groups = 2
    sm_count = torch.cuda.get_device_properties(device_index).multi_processor_count
    split_target = max(1, sm_count // prediction_groups - batch_size // 2)
    table_tiles = math.ceil(
        max_pages * _ATREX_AKA_EXTERNAL_PAGE_SIZE / 128
    )
    max_splits = max(
        1,
        min(
            _ATREX_AKA_SPLIT_CAP // prediction_groups,
            split_target,
            table_tiles,
        ),
    )
    # The final plane carries the per-request split count into reduction.
    return split_target, max_splits + 1


def _atrex_aka_compile_decode(
    device_index: int,
    device_capability: tuple[int, int],
):
    """Compile the fixed q4 kernel for one CUDA device context."""
    del device_capability  # It remains part of the cache key at the caller.
    grouped_head_tile = _ATREX_AKA_NUM_Q_HEADS
    prediction_tile = 2
    sequence_tile = 128
    softmax_warpgroups = 1
    cluster_kv = 2
    kernel = GroupedQueryAttentionDecodePaged(
        _ATREX_AKA_KERNEL_PAGE_SIZE,
        _ATREX_AKA_HEAD_DIM,
        grouped_head_tile,
        prediction_tile=prediction_tile,
        sequence_tile=sequence_tile,
        reduction_mode="kernel",
        softmax_warpgroups=softmax_warpgroups,
        table_page_size=_ATREX_AKA_EXTERNAL_PAGE_SIZE,
        cluster_kv=cluster_kv,
        single_warp_batch=True,
    )
    kernel.decode.set_name_prefix("atrex_aka")
    GroupedQueryAttentionDecode.reduction_kernel.set_name_prefix("atrex_aka")

    sym_splits = cute.sym_int()
    sym_batch = cute.sym_int()
    sym_pages = cute.sym_int()
    sym_table = cute.sym_int()
    dtype = cutlass.BFloat16
    acc_dtype = cutlass.Float32
    seqlens = cute.runtime.make_fake_compact_tensor(
        Int32, (sym_batch,), assumed_align=16
    )
    page_table = cute.runtime.make_fake_compact_tensor(
        Int32, (sym_table,), assumed_align=16
    )
    key_cache = cute.runtime.make_fake_tensor(
        dtype,
        (
            sym_pages,
            _ATREX_AKA_KERNEL_PAGE_SIZE,
            _ATREX_AKA_NUM_KV_HEADS,
            _ATREX_AKA_HEAD_DIM,
        ),
        stride=(cute.sym_int(), cute.sym_int(), cute.sym_int(), 1),
        assumed_align=16,
    )
    value_cache = cute.runtime.make_fake_tensor(
        dtype,
        (
            sym_pages,
            _ATREX_AKA_KERNEL_PAGE_SIZE,
            _ATREX_AKA_NUM_KV_HEADS,
            _ATREX_AKA_HEAD_DIM,
        ),
        stride=(cute.sym_int(), cute.sym_int(), cute.sym_int(), 1),
        assumed_align=16,
    )
    q = cute.runtime.make_fake_compact_tensor(
        dtype,
        (
            sym_batch,
            _ATREX_AKA_QUERY_LENGTH,
            _ATREX_AKA_NUM_Q_HEADS,
            _ATREX_AKA_HEAD_DIM,
        ),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    out = cute.runtime.make_fake_compact_tensor(
        dtype,
        (
            sym_batch,
            _ATREX_AKA_QUERY_LENGTH,
            _ATREX_AKA_NUM_Q_HEADS,
            _ATREX_AKA_HEAD_DIM,
        ),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    o_partial = cute.runtime.make_fake_compact_tensor(
        acc_dtype,
        (
            sym_splits,
            sym_batch,
            _ATREX_AKA_QUERY_LENGTH,
            _ATREX_AKA_NUM_Q_HEADS,
            _ATREX_AKA_HEAD_DIM,
        ),
        stride_order=(4, 3, 2, 1, 0),
        assumed_align=16,
    )
    l_partial = cute.runtime.make_fake_compact_tensor(
        acc_dtype,
        (
            sym_splits,
            sym_batch,
            _ATREX_AKA_QUERY_LENGTH,
            _ATREX_AKA_NUM_Q_HEADS,
        ),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    m_partial = cute.runtime.make_fake_compact_tensor(
        acc_dtype,
        (
            sym_splits,
            sym_batch,
            _ATREX_AKA_QUERY_LENGTH,
            _ATREX_AKA_NUM_Q_HEADS,
        ),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    with torch.cuda.device(device_index):
        return cute.compile(
            kernel,
            Int32(1),
            seqlens,
            Int32(1),
            page_table,
            key_cache,
            value_cache,
            q,
            out,
            None,
            o_partial,
            l_partial,
            m_partial,
            None,
            CausalMask(),
            Float32(_ATREX_AKA_HEAD_DIM**-0.5),
            Float32(1.0),
            None,
            stream,
            True,
            options="--enable-tvm-ffi --opt-level 3",
        )


def _atrex_aka_get_decode_kernel(device_index: int):
    capability = torch.cuda.get_device_capability(device_index)
    key = (device_index, capability)
    compiled = _ATREX_AKA_KERNEL_CACHE.get(key)
    if compiled is not None:
        return compiled
    if torch.cuda.is_current_stream_capturing():
        raise RuntimeError(
            "AKA FA4 decode must be called once eagerly before CUDA Graph capture"
        )
    with _ATREX_AKA_KERNEL_CACHE_LOCK:
        compiled = _ATREX_AKA_KERNEL_CACHE.get(key)
        if compiled is None:
            compiled = _atrex_aka_compile_decode(device_index, capability)
            _ATREX_AKA_KERNEL_CACHE[key] = compiled
    return compiled


def atrex_aka_fa4_decode(
    *,
    q,
    k,
    v,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_k,
    max_seqlen_q,
    max_seqlen_k,
    page_table,
    softmax_scale,
    causal,
    softcap,
    window_size_left,
    window_size_right,
    learnable_sink,
    out,
    return_lse,
    q_descale,
    k_descale,
    v_descale,
    num_splits,
    _workspace: torch.Tensor | None = None,
):
    """Run the fixed AKA q4 specialization and return ``(out, None)``.

    ``_workspace`` is private test injection used to prove that the kernels do
    not read stale scratch values. Production calls always allocate call-local
    scratch through PyTorch's stream-aware caching/graph allocator.
    """
    del (
        cu_seqlens_q,
        cu_seqlens_k,
        max_seqlen_q,
        max_seqlen_k,
        causal,
        softcap,
        window_size_left,
        window_size_right,
        learnable_sink,
        return_lse,
        q_descale,
        k_descale,
        v_descale,
        num_splits,
    )
    batch_size = seqused_k.numel()
    device_index = q.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    split_target, workspace_splits = _atrex_aka_split_config(
        batch_size, page_table.shape[1], device_index
    )
    compiled = _atrex_aka_get_decode_kernel(device_index)

    if out is None:
        out = torch.empty_like(q)
    required = _atrex_aka_workspace_elements(workspace_splits, batch_size)
    if _workspace is None:
        workspace = torch.empty(required, dtype=torch.float32, device=q.device)
    else:
        if _workspace.device != q.device:
            raise ValueError("AKA decode workspace must be on the Q device")
        workspace = _workspace
    o_partial, l_partial, m_partial = _atrex_aka_slice_workspace(
        workspace, workspace_splits, batch_size
    )

    q_view = q.view(
        batch_size,
        _ATREX_AKA_QUERY_LENGTH,
        _ATREX_AKA_NUM_Q_HEADS,
        _ATREX_AKA_HEAD_DIM,
    )
    out_view = out.view_as(q_view)
    key_view = k.view(
        k.shape[0] * (_ATREX_AKA_EXTERNAL_PAGE_SIZE // _ATREX_AKA_KERNEL_PAGE_SIZE),
        _ATREX_AKA_KERNEL_PAGE_SIZE,
        _ATREX_AKA_NUM_KV_HEADS,
        _ATREX_AKA_HEAD_DIM,
    )
    value_view = v.view_as(key_view)
    compiled(
        split_target,
        seqused_k,
        page_table.stride(0),
        page_table.reshape(-1),
        key_view,
        value_view,
        q_view,
        out_view,
        None,
        o_partial,
        l_partial,
        m_partial,
        None,
        CausalMask(),
        Float32(float(softmax_scale)),
        Float32(1.0),
        None,
        True,
    )
    return out, None


__all__ = ("atrex_aka_fa4_decode",)
