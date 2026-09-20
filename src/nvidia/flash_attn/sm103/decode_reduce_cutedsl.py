"""CuTeDSL reduction for split-K FA4 decode partial outputs."""

from __future__ import annotations

import functools
import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass.cute.typing import BFloat16, Float32, Int32, Optional

from atrex.api.cutlass_dsl_compat import (
    use_filesystem_cutlass_dsl_version_hash,
)


HEAD_DIM = 256
SUPPORTED_SPLITS = (1, 2, 4, 8, 16)
LN2 = 0.6931471805599453


class _Fa4DecodeReduceShared:
    def __init__(self, num_splits: int, store_lse: bool):
        self.num_splits = num_splits
        self.store_lse = store_lse

    @cute.jit
    def __call__(
        self,
        mO_partial: cute.Tensor,
        mL_partial: cute.Tensor,
        mM_partial: cute.Tensor,
        mCuSeqlensQ: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        scale_o: Float32,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class SharedStorage:
            weights: cute.struct.Align[
                cute.struct.MemRange[Float32, self.num_splits], 16
            ]
            inv_sum: cute.struct.Align[cute.struct.MemRange[Float32, 1], 4]
            final_lse: cute.struct.Align[cute.struct.MemRange[Float32, 1], 4]

        dense_rows = (
            mO_partial.shape[1]
            * mO_partial.shape[2]
            * mO_partial.shape[3]
        )
        self.atrex_sm103_decode_reduce_kernel(
            mO_partial,
            mL_partial,
            mM_partial,
            mCuSeqlensQ,
            mO,
            mLSE,
            scale_o,
            SharedStorage,
        ).launch(
            grid=(dense_rows, 1, 1),
            block=(HEAD_DIM, 1, 1),
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def atrex_sm103_decode_reduce_kernel(
        self,
        mO_partial: cute.Tensor,
        mL_partial: cute.Tensor,
        mM_partial: cute.Tensor,
        mCuSeqlensQ: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        scale_o: Float32,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        dense_row, _, _ = cute.arch.block_idx()
        lane = tidx % 32
        warp = tidx // 32
        max_query = mO_partial.shape[2]
        num_heads = mO_partial.shape[3]
        head = dense_row % num_heads
        query_slot = (dense_row // num_heads) % max_query
        batch = dense_row // (num_heads * max_query)
        query_start = mCuSeqlensQ[batch]
        query_end = mCuSeqlensQ[batch + 1]

        if query_slot < query_end - query_start:
            storage = utils.SmemAllocator().allocate(SharedStorage)
            weights = storage.weights.get_tensor((self.num_splits,))
            inv_sum_smem = storage.inv_sum.get_tensor((1,))
            lse_smem = storage.final_lse.get_tensor((1,))
            if warp == 0:
                partial_max = -Float32.inf
                partial_sum = Float32(0.0)
                if lane < self.num_splits:
                    partial_max = mM_partial[
                        lane, batch, query_slot, head
                    ]
                    partial_sum = mL_partial[
                        lane, batch, query_slot, head
                    ]
                final_max = cute.arch.warp_reduction_max(partial_max)
                safe_final_max = (
                    final_max if final_max > -Float32.inf else Float32(0.0)
                )
                correction = Float32(0.0)
                if lane < self.num_splits and partial_max > -Float32.inf:
                    correction = cute.math.exp2(
                        partial_max - safe_final_max,
                        fastmath=True,
                    )
                if lane < self.num_splits:
                    weights[lane] = correction
                final_sum = cute.arch.warp_reduction_sum(
                    correction * partial_sum
                )
                if lane == 0:
                    inv_sum_smem[0] = (
                        cute.arch.rcp_approx(final_sum)
                        if final_sum > Float32(0.0)
                        else Float32(0.0)
                    )
                    lse_smem[0] = (
                        (safe_final_max + cute.math.log2(final_sum))
                        * Float32(LN2)
                        if final_sum > Float32(0.0)
                        else -Float32.inf
                    )
            cute.arch.sync_threads()
            output = Float32(0.0)
            for split in cutlass.range_constexpr(self.num_splits):
                output += (
                    weights[split]
                    * mO_partial[split, batch, query_slot, head, tidx]
                )
            packed_token = query_start + query_slot
            mO[packed_token, head, tidx] = (
                output * inv_sum_smem[0] * scale_o
            ).to(BFloat16)
            if cutlass.const_expr(self.store_lse):
                if tidx == 0:
                    mLSE[head, packed_token] = lse_smem[0]


class _Fa4DecodeReduce:
    def __init__(self, num_splits: int, store_lse: bool):
        self.num_splits = num_splits
        self.store_lse = store_lse
        self.num_threads = min(256, max(64, num_splits * 16))

    @cute.jit
    def __call__(
        self,
        mO_partial: cute.Tensor,
        mL_partial: cute.Tensor,
        mM_partial: cute.Tensor,
        mCuSeqlensQ: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        scale_o: Float32,
        stream: cuda.CUstream,
    ):
        if cutlass.const_expr(mO_partial.element_type is not Float32):
            raise TypeError("partial output must be Float32")
        if cutlass.const_expr(mL_partial.element_type is not Float32):
            raise TypeError("partial sum must be Float32")
        if cutlass.const_expr(mM_partial.element_type is not Float32):
            raise TypeError("partial max must be Float32")
        if cutlass.const_expr(mO.element_type is not BFloat16):
            raise TypeError("output must be BFloat16")

        batch_size = mO_partial.shape[1]
        max_query = mO_partial.shape[2]
        num_heads = mO_partial.shape[3]
        dense_rows = batch_size * max_query * num_heads
        self.atrex_sm103_decode_varlen_reduce_kernel(
            mO_partial,
            mL_partial,
            mM_partial,
            mCuSeqlensQ,
            mO,
            mLSE,
            scale_o,
        ).launch(
            grid=(dense_rows, 1, 1),
            block=(self.num_threads, 1, 1),
            stream=stream,
        )

    @cute.kernel
    def atrex_sm103_decode_varlen_reduce_kernel(
        self,
        mO_partial: cute.Tensor,
        mL_partial: cute.Tensor,
        mM_partial: cute.Tensor,
        mCuSeqlensQ: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        scale_o: Float32,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        dense_row, _, _ = cute.arch.block_idx()
        lane = tidx % 32

        max_query = mO_partial.shape[2]
        num_heads = mO_partial.shape[3]
        head = dense_row % num_heads
        query_slot = (dense_row // num_heads) % max_query
        batch = dense_row // (num_heads * max_query)
        query_start = mCuSeqlensQ[batch]
        query_end = mCuSeqlensQ[batch + 1]

        if query_slot < query_end - query_start:
            partial_max = -Float32.inf
            partial_sum = Float32(0.0)
            if lane < self.num_splits:
                partial_max = mM_partial[lane, batch, query_slot, head]
                partial_sum = mL_partial[lane, batch, query_slot, head]

            final_max = cute.arch.warp_reduction_max(partial_max)
            safe_final_max = (
                final_max if final_max > -Float32.inf else Float32(0.0)
            )
            correction = Float32(0.0)
            if lane < self.num_splits and partial_max > -Float32.inf:
                correction = cute.math.exp2(
                    partial_max - safe_final_max,
                    fastmath=True,
                )
            final_sum = cute.arch.warp_reduction_sum(
                correction * partial_sum
            )
            inv_sum = (
                cute.arch.rcp_approx(final_sum)
                if final_sum > Float32(0.0)
                else Float32(0.0)
            )
            packed_token = query_start + query_slot
            values_per_thread = HEAD_DIM // self.num_threads
            for value in cutlass.range_constexpr(values_per_thread):
                dimension = tidx + value * self.num_threads
                output = Float32(0.0)
                for split in cutlass.range_constexpr(self.num_splits):
                    split_correction = cute.arch.shuffle_sync(
                        correction,
                        split,
                    )
                    output += (
                        split_correction
                        * mO_partial[
                            split,
                            batch,
                            query_slot,
                            head,
                            dimension,
                        ]
                    )
                mO[packed_token, head, dimension] = (
                    output * inv_sum * scale_o
                ).to(BFloat16)
            if cutlass.const_expr(self.store_lse):
                if tidx == 0:
                    mLSE[head, packed_token] = (
                        (safe_final_max + cute.math.log2(final_sum))
                        * Float32(LN2)
                        if final_sum > Float32(0.0)
                        else -Float32.inf
                    )


@functools.cache
def _compile_reducer(
    num_splits: int,
    num_heads: int,
    max_query: int,
    store_lse: bool,
    device_index: int,
    device_capability: tuple[int, int],
):
    del device_index, device_capability
    if num_splits not in SUPPORTED_SPLITS:
        raise ValueError(f"unsupported num_splits={num_splits}")

    sym_batch = cute.sym_int()
    sym_total_q = cute.sym_int()
    sym_batch_plus_one = cute.sym_int()
    o_partial = cute.runtime.make_fake_compact_tensor(
        Float32,
        (num_splits, sym_batch, max_query, num_heads, HEAD_DIM),
        stride_order=(4, 3, 2, 1, 0),
        assumed_align=16,
    )
    stat_partial = cute.runtime.make_fake_compact_tensor(
        Float32,
        (num_splits, sym_batch, max_query, num_heads),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    cu_seqlens_q = cute.runtime.make_fake_compact_tensor(
        Int32,
        (sym_batch_plus_one,),
        assumed_align=16,
    )
    output = cute.runtime.make_fake_compact_tensor(
        BFloat16,
        (sym_total_q, num_heads, HEAD_DIM),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    lse = (
        cute.runtime.make_fake_compact_tensor(
            Float32,
            (num_heads, sym_total_q),
            stride_order=(1, 0),
            assumed_align=16,
        )
        if store_lse
        else None
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    use_shared = (
        (num_heads == 16 and num_splits in (2, 4))
        or (num_heads == 32 and num_splits == 2)
    )
    if use_shared:
        reducer = _Fa4DecodeReduceShared(num_splits, store_lse)
    else:
        reducer = _Fa4DecodeReduce(num_splits, store_lse)
    with use_filesystem_cutlass_dsl_version_hash():
        return cute.compile(
            reducer,
            o_partial,
            stat_partial,
            stat_partial,
            cu_seqlens_q,
            output,
            lse,
            Float32(1.0),
            stream,
            options="--enable-tvm-ffi --opt-level 3",
        )


def fa4_decode_reduce_varlen_cutedsl(
    o_partial: torch.Tensor,
    l_partial: torch.Tensor,
    m_partial: torch.Tensor,
    cu_seqlens_q: torch.Tensor,
    output: torch.Tensor,
    lse: torch.Tensor | None = None,
    *,
    scale_o: float = 1.0,
) -> None:
    """Merge dense split-K rows directly into packed varlen output/LSE."""
    if o_partial.ndim != 5 or o_partial.shape[-1] != HEAD_DIM:
        raise ValueError(
            "o_partial must have shape [splits, batch, max_query, heads, 256]"
        )
    if l_partial.shape != o_partial.shape[:-1] or m_partial.shape != l_partial.shape:
        raise ValueError("partial statistics must match o_partial without head_dim")
    num_splits, _, max_query, num_heads, head_dim = o_partial.shape
    del head_dim
    batch_size = o_partial.shape[1]
    if cu_seqlens_q.shape != (batch_size + 1,) or cu_seqlens_q.dtype != torch.int32:
        raise ValueError("cu_seqlens_q must be contiguous int32 [batch + 1]")
    if cu_seqlens_q.stride(0) != 1:
        raise ValueError("cu_seqlens_q must be contiguous")
    if output.ndim != 3 or output.shape[1:] != (num_heads, HEAD_DIM):
        raise ValueError("output must have shape [total_q, heads, 256]")
    if output.dtype != torch.bfloat16:
        raise TypeError("FA4 decode output must use BF16")
    if any(t.dtype != torch.float32 for t in (o_partial, l_partial, m_partial)):
        raise TypeError("FA4 decode partial workspace must use FP32")
    tensors = (o_partial, l_partial, m_partial, cu_seqlens_q, output)
    if not all(t.device == output.device for t in tensors):
        raise ValueError("FA4 decode reduction tensors must share one device")
    if not all(t.is_contiguous() for t in tensors):
        raise ValueError("FA4 decode reduction tensors must be contiguous")
    if lse is not None:
        if lse.shape != (num_heads, output.shape[0]) or lse.dtype != torch.float32:
            raise ValueError("lse must be FP32 [heads, total_q]")
        if lse.device != output.device or not lse.is_contiguous():
            raise ValueError("lse must be contiguous on the output device")
    if num_splits not in SUPPORTED_SPLITS:
        raise ValueError(
            "varlen reduction supports "
            + "/".join(str(split) for split in SUPPORTED_SPLITS)
            + f" splits, got {num_splits}"
        )

    device_index = output.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    compiled = _compile_reducer(
        num_splits,
        num_heads,
        max_query,
        lse is not None,
        device_index,
        torch.cuda.get_device_capability(output.device),
    )
    compiled(
        o_partial,
        l_partial,
        m_partial,
        cu_seqlens_q,
        output,
        lse,
        Float32(scale_o),
    )
