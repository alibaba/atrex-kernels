"""Small-row SplitKV reducer for SM120 Q1--Q4 decode."""

from __future__ import annotations

import functools
import math

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
import torch
from cutlass.cute.typing import BFloat16, Float32, Int32, Optional

from atrex.src.nvidia.flash_attn.common_utils.cutlass_dsl_cache import (
    use_filesystem_cutlass_dsl_version_hash,
)


LOG2_E = math.log2(math.e)


class _Sm120Q1Reduce:
    def __init__(
        self, num_splits: int, tokens_per_sequence: int, head_dim: int,
        parallel_columns: bool,
    ):
        self.num_splits = num_splits
        self.tokens_per_sequence = tokens_per_sequence
        self.head_dim = head_dim
        self.column_tile = 32 if head_dim in (128, 256) and parallel_columns else head_dim

    @cute.jit
    def __call__(
        self,
        mO_partial: cute.Tensor,
        mLSE_partial: cute.Tensor,
        mNumSplitsDynamic: Optional[cute.Tensor],
        mO: cute.Tensor,
        stream: cuda.CUstream,
    ):
        @cute.struct
        class SharedStorage:
            weights: cute.struct.Align[
                cute.struct.MemRange[Float32, self.num_splits], 16
            ]
            inv_sum: cute.struct.Align[cute.struct.MemRange[Float32, 1], 4]

        rows = mO_partial.shape[1] * mO_partial.shape[2]
        self.atrex_sm120_decode_reduce_kernel(
            mO_partial,
            mLSE_partial,
            mNumSplitsDynamic,
            mO,
            SharedStorage,
        ).launch(
            grid=(rows * (self.head_dim // self.column_tile), 1, 1),
            block=(self.column_tile, 1, 1),
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
        )

    @cute.kernel
    def atrex_sm120_decode_reduce_kernel(
        self,
        mO_partial: cute.Tensor,
        mLSE_partial: cute.Tensor,
        mNumSplitsDynamic: Optional[cute.Tensor],
        mO: cute.Tensor,
        SharedStorage: cutlass.Constexpr,
    ):
        tidx, _, _ = cute.arch.thread_idx()
        row, _, _ = cute.arch.block_idx()
        column = tidx
        if cutlass.const_expr(self.head_dim == 128 or self.column_tile < self.head_dim):
            tile = row
            row = tile // (self.head_dim // self.column_tile)
            column += (tile % (self.head_dim // self.column_tile)) * self.column_tile
        lane = tidx % 32
        num_heads = mO_partial.shape[2]
        token = row // num_heads
        head = row - token * num_heads
        num_splits = Int32(self.num_splits)
        if cutlass.const_expr(mNumSplitsDynamic is not None):
            batch = token // self.tokens_per_sequence
            num_splits = mNumSplitsDynamic[batch]
        storage = utils.SmemAllocator().allocate(SharedStorage)
        weights = storage.weights.get_tensor((self.num_splits,))
        inv_sum = storage.inv_sum.get_tensor((1,))

        if tidx < 32:
            local_max = -Float32.inf
            for offset in cutlass.range_constexpr((self.num_splits + 31) // 32):
                split = lane + offset * 32
                if split < num_splits:
                    local_max = cutlass.max(
                        local_max, mLSE_partial[split, head, token]
                    )
            final_max = cute.arch.warp_reduction_max(local_max)
            safe_max = final_max if final_max > -Float32.inf else Float32(0.0)
            local_sum = Float32(0.0)
            for offset in cutlass.range_constexpr((self.num_splits + 31) // 32):
                split = lane + offset * 32
                if split < self.num_splits:
                    weight = Float32(0.0)
                    if split < num_splits:
                        lse = mLSE_partial[split, head, token]
                        weight = (
                            cute.math.exp2(
                                (lse - safe_max) * Float32(LOG2_E),
                                fastmath=True,
                            )
                            if lse > -Float32.inf
                            else Float32(0.0)
                        )
                    weights[split] = weight
                    local_sum += weight
            final_sum = cute.arch.warp_reduction_sum(local_sum)
            if lane == 0:
                inv_sum[0] = (
                    cute.arch.rcp_approx(final_sum)
                    if final_sum > Float32(0.0)
                    else Float32(0.0)
                )

        cute.arch.sync_threads()
        output = Float32(0.0)
        if cutlass.const_expr(mNumSplitsDynamic is not None):
            # Keep fixed, unrolled load groups as in the SM103 reducer while
            # avoiding reads from most inactive ragged-batch split slots.
            if num_splits <= 24:
                for split in cutlass.range_constexpr(min(self.num_splits, 24)):
                    if split < num_splits:
                        output += (
                            weights[split]
                            * mO_partial[split, token, head, column]
                        )
            elif num_splits <= 40:
                for split in cutlass.range_constexpr(min(self.num_splits, 40)):
                    if split < num_splits:
                        output += (
                            weights[split]
                            * mO_partial[split, token, head, column]
                        )
            elif num_splits <= 48:
                for split in cutlass.range_constexpr(min(self.num_splits, 48)):
                    if split < num_splits:
                        output += (
                            weights[split]
                            * mO_partial[split, token, head, column]
                        )
            elif num_splits <= 64:
                for split in cutlass.range_constexpr(min(self.num_splits, 64)):
                    if split < num_splits:
                        output += (
                            weights[split]
                            * mO_partial[split, token, head, column]
                        )
            elif num_splits <= 80:
                for split in cutlass.range_constexpr(min(self.num_splits, 80)):
                    if split < num_splits:
                        output += (
                            weights[split]
                            * mO_partial[split, token, head, column]
                        )
            else:
                for split in cutlass.range_constexpr(self.num_splits):
                    if split < num_splits:
                        output += (
                            weights[split]
                            * mO_partial[split, token, head, column]
                        )
        else:
            for split in cutlass.range_constexpr(self.num_splits):
                output += weights[split] * mO_partial[split, token, head, column]
        mO[token, head, column] = (output * inv_sum[0]).to(BFloat16)


@functools.cache
def _compile(
    num_splits: int,
    num_heads: int,
    device_index: int,
    dynamic_splits: bool,
    tokens_per_sequence: int,
    head_dim: int,
    parallel_columns: bool,
):
    del device_index
    sym_total_q = cute.sym_int()
    sym_batch = cute.sym_int()
    o_partial = cute.runtime.make_fake_compact_tensor(
        Float32,
        (num_splits, sym_total_q, num_heads, head_dim),
        stride_order=(3, 2, 1, 0),
        assumed_align=16,
    )
    lse_partial = cute.runtime.make_fake_compact_tensor(
        Float32,
        (num_splits, num_heads, sym_total_q),
        stride_order=(2, 1, 0),
        assumed_align=4,
    )
    output = cute.runtime.make_fake_compact_tensor(
        BFloat16,
        (sym_total_q, num_heads, head_dim),
        stride_order=(2, 1, 0),
        assumed_align=16,
    )
    num_splits_dynamic = (
        cute.runtime.make_fake_compact_tensor(
            Int32,
            (sym_batch,),
            assumed_align=4,
        )
        if dynamic_splits
        else None
    )
    stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)
    reducer = _Sm120Q1Reduce(num_splits, tokens_per_sequence, head_dim, parallel_columns)
    reducer.atrex_sm120_decode_reduce_kernel.set_name_prefix("atrex")
    with use_filesystem_cutlass_dsl_version_hash():
        return cute.compile(
            reducer,
            o_partial,
            lse_partial,
            num_splits_dynamic,
            output,
            stream,
            options="--enable-tvm-ffi --opt-level 3",
        )


def sm120_q1_reduce(
    o_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    output: torch.Tensor,
    num_splits_dynamic: torch.Tensor | None = None,
    tokens_per_sequence: int = 1,
    parallel_columns: bool = False,
) -> None:
    num_splits, total_q, num_heads, head_dim = o_partial.shape
    if head_dim not in (128, 256) or output.shape != (total_q, num_heads, head_dim):
        raise ValueError("invalid SM120 Q1 reduction shape")
    if lse_partial.shape != (num_splits, num_heads, total_q):
        raise ValueError("invalid SM120 Q1 LSE partial shape")
    if tokens_per_sequence <= 0 or total_q % tokens_per_sequence:
        raise ValueError("invalid SM120 short-Q tokens_per_sequence")
    if num_splits_dynamic is not None:
        if num_splits_dynamic.shape != (total_q // tokens_per_sequence,):
            raise ValueError("invalid SM120 Q1 dynamic split shape")
        if num_splits_dynamic.dtype != torch.int32:
            raise ValueError("SM120 Q1 dynamic split counts must be int32")
    device_index = output.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    _compile(
        num_splits,
        num_heads,
        device_index,
        num_splits_dynamic is not None,
        tokens_per_sequence,
        head_dim,
        parallel_columns,
    )(
        o_partial,
        lse_partial,
        num_splits_dynamic,
        output,
    )
