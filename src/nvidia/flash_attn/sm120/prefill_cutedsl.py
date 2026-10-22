# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.
# A reimplementation of
# https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_fwd_kernel_sm80.h
# and https://github.com/Dao-AILab/flash-attention/blob/main/hopper/flash_fwd_kernel_sm90.h
# from Cutlass C++ to Cute-DSL.
# Built on Cute-DSL example: https://github.com/NVIDIA/cutlass/blob/main/examples/python/CuTeDSL/ampere/flash_attention_v2.py

import math
import os
from types import SimpleNamespace
from typing import Type, Callable, Optional
from functools import partial

import cuda.bindings.driver as cuda

import cutlass
import cutlass.cute as cute
from cutlass import Float32, Int32, Int64, const_expr
from cutlass._mlir import ir
from cutlass._mlir.dialects import llvm
from cutlass.cute.nvgpu import cpasync, warp
import cutlass.utils as utils_basic
from cutlass.base_dsl.arch import Arch
from cutlass.cutlass_dsl import BaseDSL, T, dsl_user_op

from quack import copy_utils
from quack import layout_utils

from flash_attn.cute import ampere_helpers as sm80_utils
from flash_attn.cute.cute_dsl_utils import assume_tensor_aligned
from flash_attn.cute import utils
from flash_attn.cute.mask import AttentionMask
from flash_attn.cute.softmax import Softmax, apply_score_mod_inner
# Use the local cu_seqlens phantom-tile OOB clamp; other helpers come from
# the public flash-attn-4 package.
from atrex.src.nvidia.flash_attn.sm120.seqlen_info import SeqlenInfoQK
from flash_attn.cute.block_info import BlockInfo
from flash_attn.cute.pack_gqa import PackGQA, pack_gqa_layout
from flash_attn.cute.named_barrier import NamedBarrierFwd
from flash_attn.cute.block_sparsity import BlockSparseTensors
from flash_attn.cute.tile_scheduler import SingleTileScheduler, SingleTileVarlenScheduler, TileSchedulerArguments
from flash_attn.cute.utils import AuxData


@dsl_user_op
def _ldmatrix_m16n16_x2_trans_b8(ptr: cute.Pointer, *, loc=None, ip=None):
    """Load a 16x32 byte tile and return its four packed FP8 registers."""
    ptr_i32 = ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    result = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32,i32,i32,i32)>"),
        [ptr_i32],
        "ldmatrix.sync.aligned.m16n16.x2.trans.shared.b8 "
        "{$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Int32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip))
        for i in range(4)
    )


@dsl_user_op
def _ldmatrix_m8n8_x4_b16(ptr: cute.Pointer, *, loc=None, ip=None):
    """Load four packed 32-bit registers for an FP8 MMA A fragment."""
    ptr_i32 = ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    result = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32,i32,i32,i32)>"),
        [ptr_i32],
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 "
        "{$0, $1, $2, $3}, [$4];",
        "=r,=r,=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Int32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip))
        for i in range(4)
    )


@dsl_user_op
def _ldmatrix_m8n8_x2_b16(ptr: cute.Pointer, *, loc=None, ip=None):
    ptr_i32 = ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    result = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(i32,i32)>"),
        [ptr_i32],
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {$0, $1}, [$2];",
        "=r,=r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Int32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip))
        for i in range(2)
    )


@dsl_user_op
def _mma_m16n8k32_f32_fp8(
    a, b, c, fp8_dtype: cutlass.Constexpr, *, reuse_accumulator=False, loc=None, ip=None
):
    """Ada-style FP8 warp MMA used by SM120 without relying on NVVM lowering."""
    args = [
        *(Int32(x).ir_value(loc=loc, ip=ip) for x in a),
        *(Int32(x).ir_value(loc=loc, ip=ip) for x in b),
        *(Float32(x).ir_value(loc=loc, ip=ip) for x in c),
    ]
    fp8_name = "e4m3" if fp8_dtype is cutlass.Float8E4M3FN else "e5m2"
    result = llvm.inline_asm(
        ir.Type.parse("!llvm.struct<(f32,f32,f32,f32)>"),
        args,
        f"mma.sync.aligned.m16n8k32.row.col.f32.{fp8_name}.{fp8_name}.f32 "
        "{$0, $1, $2, $3}, {$4, $5, $6, $7}, {$8, $9}, "
        "{$10, $11, $12, $13};",
        (
            "=f,=f,=f,=f,r,r,r,r,r,r,0,1,2,3" if reuse_accumulator
            else "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f"
        ),
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return tuple(
        Float32(llvm.extractvalue(T.f32(), result, [i], loc=loc, ip=ip))
        for i in range(4)
    )


@dsl_user_op
def _prmt_b32(a, b, selector, *, loc=None, ip=None):
    """Pack two pairs of FP8 bytes without a shared-memory round trip."""
    result = llvm.inline_asm(
        T.i32(),
        [Int32(x).ir_value(loc=loc, ip=ip) for x in (a, b, selector)],
        "prmt.b32 $0, $1, $2, $3;",
        "=r,r,r,r",
        has_side_effects=False,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )
    return Int32(result)


class FlashAttentionForwardBase:

    def __init__(
        self,
        dtype: Type[cutlass.Numeric],
        head_dim: int,
        head_dim_v: Optional[int] = None,
        qhead_per_kvhead: int = 1,
        is_causal: bool = False,
        is_local: bool = False,
        lpt: bool = False,
        pack_gqa: bool = True,
        tile_m: int = 128,
        tile_n: int = 128,
        num_stages: int = 1,
        num_threads: int = 128,
        Q_in_regs: bool = False,
        score_mod: Optional[cutlass.Constexpr] = None,
        mask_mod: Optional[cutlass.Constexpr] = None,
        has_aux_tensors: bool = False,
        q_subtile_factor: int = 1,
        is_split_kv: bool = False,
        page_size: Optional[int] = None,   # PAGED
        dynamic_splits: bool = False,       # DYNSPLIT
        compact_grid: bool = False,          # COMPACTGRID
        compact_short_q: bool = False,
        runtime_balanced_splits: bool = False,
        runtime_balanced_grid_size: int = 0,
        runtime_balanced_batch_size: int = 0,
    ):
        """Initializes the configuration for a flash attention kernel.

        All contiguous dimensions must be at least 16 bytes aligned, which means that the head dimension
        should be a multiple of 8.

        :param head_dim: head dimension
        :type head_dim: int
        :param tile_m: m block size
        :type tile_m: int
        :param tile_n: n block size
        :type tile_n: int
        :param num_threads: number of threads
        :type num_threads: int
        :param is_causal: is causal
        :param score_mod: A callable that takes the attention scores and applies a modification.
            Callable signature: ``score_mod(scores, batch_idx, head_idx, q_idx, kv_idx, aux_tensors) -> Any``
        :param mask_mod: A callable that takes the attention scores and returns a boolean representing whether that score should be masked.
            Callable signature: ``mask_mod(batch_idx, head_idx, q_idx, kv_idx, aux_tensors) -> Boolean``
        """
        self.dtype = dtype
        self.is_fp8 = dtype in (cutlass.Float8E4M3FN, cutlass.Float8E5M2)
        self.output_dtype = cutlass.BFloat16 if self.is_fp8 else dtype
        # Online softmax probabilities are <=1. Scale HD128 E4M3 P into
        # [0,256] before PV to avoid flushing small probabilities to zero;
        # undo this constant after normalization without changing the LSE.
        self.fp8_p_scale = 256.0 if self.is_fp8 and head_dim == 128 else 1.0
        # PAGED: page-aligned paged KV. page_size % tile_n == 0 is enforced on the host, so one
        # n_block never straddles two pages and a tile stays a single contiguous gmem read.
        self.page_size = page_size
        self.paged_kv = page_size is not None
        self.blocks_per_page = (page_size // tile_n) if page_size is not None else 1
        # DYNSPLIT: read the split count per sequence instead of using one scalar for the batch.
        self.dynamic_splits = dynamic_splits
        # COMPACTGRID: take (m_block, head, batch, split) from a host-built work map instead of
        # decoding it from a rectangular blockIdx, so only useful CTAs are launched.
        self.compact_grid = compact_grid
        self.compact_short_q = compact_short_q
        self.runtime_balanced_splits = runtime_balanced_splits
        self.runtime_balanced_grid_size = runtime_balanced_grid_size
        self.runtime_balanced_batch_size = runtime_balanced_batch_size
        # padding head_dim to a multiple of 16 as k_block_size
        hdim_multiple_of = 16
        self.tile_hdim = int(math.ceil(head_dim / hdim_multiple_of) * hdim_multiple_of)
        head_dim_v = head_dim_v if head_dim_v is not None else head_dim
        self.same_hdim_kv = head_dim == head_dim_v
        self.tile_hdimv = int(math.ceil(head_dim_v / hdim_multiple_of) * hdim_multiple_of)
        # Can save registers (and hence be faster) if we don't have to check hdim predication
        self.check_hdim_oob = head_dim != self.tile_hdim
        self.check_hdim_v_oob = head_dim_v != self.tile_hdimv
        self.qhead_per_kvhead = qhead_per_kvhead
        self.is_causal = is_causal
        self.is_local = is_local
        self.lpt = lpt
        self.pack_gqa = pack_gqa
        self.tile_m = tile_m
        self.tile_n = tile_n
        self.num_threads = num_threads
        self.num_stages = num_stages
        self.q_subtile_factor = q_subtile_factor
        self.is_split_kv = is_split_kv          # SPLITKV
        self.Q_in_regs = Q_in_regs
        self.score_mod = score_mod
        self.mask_mod = mask_mod
        self.qk_acc_dtype = Float32
        self.score_vec_size: cutlass.Constexpr = getattr(
            score_mod, "__vec_size__", 1 if cutlass.const_expr(has_aux_tensors) else 2
        )
        if self.score_vec_size > 2:
            raise ValueError(
                f"score_mod vec_size {self.score_vec_size} not supported on Sm80/90/120 "
                "due to accumulator thread ownership pattern."
            )
        self.mask_vec_size: cutlass.Constexpr = getattr(mask_mod, "__vec_size__", 1)
        if self.mask_vec_size > 1:
            raise ValueError(
                f"mask_mod vec_size {self.mask_vec_size} not supported on Sm80/90/120 "
                "due to accumulator thread ownership pattern."
            )
        self.arch = BaseDSL._get_dsl().get_arch_enum()

    @staticmethod
    def can_implement(
        dtype,
        head_dim,
        head_dim_v,
        tile_m,
        tile_n,
        num_stages,
        num_threads,
        is_causal,
        Q_in_regs=False,
    ) -> bool:
        """Check if the kernel can be implemented with the given parameters.

        :param dtype: data type
        :type dtype: cutlass.Numeric
        :param head_dim: head dimension
        :type head_dim: int
        :param tile_m: m block size
        :type tile_m: int
        :param tile_n: n block size
        :type tile_n: int
        :param num_threads: number of threads
        :type num_threads: int
        :param is_causal: is causal
        :type is_causal: bool

        :return: True if the kernel can be implemented, False otherwise
        :rtype: bool
        """
        if dtype not in [
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        ]:
            return False
        if head_dim % 8 != 0:
            return False
        if head_dim_v % 8 != 0:
            return False
        if tile_n % 16 != 0:
            return False
        if num_threads % 32 != 0:
            return False
        # Check if block size setting is out of shared memory capacity
        # Shared memory usage: Q tile + (K tile + V tile) where K and V use the same tile size
        input_bytes = dtype.width // 8
        if dtype.width == 8:
            smem_usage_Q = tile_m * (head_dim + 16)
            smem_usage_K = tile_n * (head_dim + 16) * num_stages
            smem_usage_V = tile_n * head_dim_v * num_stages
            smem_usage_P = (
                tile_m * (tile_n + 16) if head_dim != 128 or is_causal else 1
            )
        else:
            smem_usage_Q = tile_m * head_dim * input_bytes
            smem_usage_K = tile_n * head_dim * num_stages * input_bytes
            smem_usage_V = tile_n * head_dim_v * num_stages * input_bytes
            smem_usage_P = 0
        smem_usage_QV = (
            (smem_usage_Q + smem_usage_V) if not Q_in_regs else max(smem_usage_Q, smem_usage_V)
        )
        smem_usage = smem_usage_QV + smem_usage_K + smem_usage_P
        # TODO: sm86 and sm89
        smem_capacity = utils_basic.get_smem_capacity_in_bytes("sm_80")
        if smem_usage > smem_capacity:
            return False
        # Check if twice the block size is divisible by the number of threads
        compact_short_q = is_causal and (tile_m, tile_n, num_threads) in (
            (16, 32, 64),
            (32, 32, 128),
        )
        if not compact_short_q and (tile_m * 2) % num_threads != 0:
            return False
        return True

    def _check_type(
        self,
        mQ_type: Type[cutlass.Numeric],
        mK_type: Type[cutlass.Numeric],
        mV_type: Type[cutlass.Numeric],
        mO_type: Type[cutlass.Numeric],
        mLSE_type: Type[cutlass.Numeric] | None,
        mCuSeqlensQ_type: Type[cutlass.Numeric] | None,
        mCuSeqlensK_type: Type[cutlass.Numeric] | None,
        mSeqUsedQ_type: Type[cutlass.Numeric] | None,
        mSeqUsedK_type: Type[cutlass.Numeric] | None,
    ):
        # Get the data type and check if it is fp16 or bf16
        if const_expr(not (mQ_type == mK_type == mV_type)):
            raise TypeError("Q/K/V tensors must have the same data type")
        if const_expr(self.is_split_kv and mO_type != Float32):
            raise TypeError("SplitKV partial output tensor must be Float32")
        if const_expr(not self.is_split_kv and mO_type != self.output_dtype):
            raise TypeError("Output tensor has an unsupported data type")
        if const_expr(
            mQ_type not in (
                cutlass.Float16,
                cutlass.BFloat16,
                cutlass.Float8E4M3FN,
                cutlass.Float8E5M2,
            )
        ):
            raise TypeError("Only Float16, BFloat16, or FP8 is supported")
        if const_expr(mLSE_type not in [None, Float32]):
            raise TypeError("LSE tensor must be Float32")
        if const_expr(mCuSeqlensQ_type not in [None, Int32]):
            raise TypeError("cu_seqlens_q tensor must be Int32")
        if const_expr(mCuSeqlensK_type not in [None, Int32]):
            raise TypeError("cu_seqlens_k tensor must be Int32")
        if const_expr(mSeqUsedQ_type not in [None, Int32]):
            raise TypeError("seqused_q tensor must be Int32")
        if const_expr(mSeqUsedK_type not in [None, Int32]):
            raise TypeError("seqused_k tensor must be Int32")
        assert mQ_type == self.dtype

    def _setup_attributes(self):
        # ///////////////////////////////////////////////////////////////////////////////
        # Shared memory layout: Q/K/V
        # ///////////////////////////////////////////////////////////////////////////////
        sQ_layout_atom, sK_layout_atom, sV_layout_atom, sO_layout_atom, sP_layout_atom = (
            self._get_smem_layout_atom()
        )
        self.sQ_layout = cute.tile_to_shape(
            sQ_layout_atom,
            (self.tile_m, self.tile_hdim),
            (0, 1),
        )
        self.sK_layout = cute.tile_to_shape(
            sK_layout_atom,
            (self.tile_n, self.tile_hdim, self.num_stages),
            (0, 1, 2),
        )
        self.sV_layout = cute.tile_to_shape(
            sV_layout_atom,
            (self.tile_n, self.tile_hdimv, self.num_stages),
            (0, 1, 2),
        )
        self.sO_layout = cute.tile_to_shape(
            sO_layout_atom,
            (self.tile_m, self.tile_hdimv),
            (0, 1),
        )
        if const_expr(sP_layout_atom is not None):
            self.sP_layout = cute.tile_to_shape(
                sP_layout_atom,
                (self.tile_m, self.tile_n),
                (0, 1),
            )
        else:
            self.sP_layout = None
        # ///////////////////////////////////////////////////////////////////////////////
        # GMEM Tiled copy:
        # ///////////////////////////////////////////////////////////////////////////////
        # Thread layouts for copies
        universal_copy_bits = 128
        async_copy_elems = universal_copy_bits // self.dtype.width
        # atom_async_copy: async copy atom for QKV load
        atom_async_copy = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        # atom_universal_copy: universal copy atom for O store
        atom_universal_copy = cute.make_copy_atom(
            cute.nvgpu.CopyUniversalOp(),
            self.output_dtype,
            num_bits_per_copy=universal_copy_bits,
        )
        # tQ_layout and tK_layout: thread layout for QK load
        tQK_shape_dim_1 = sQ_layout_atom.outer.shape[1] // async_copy_elems
        assert self.num_Q_load_threads % tQK_shape_dim_1 == 0, (
            "num_threads must be divisible by tQK_shape_dim_1"
        )
        assert self.num_producer_threads % tQK_shape_dim_1 == 0, (
            "num_threads must be divisible by tQK_shape_dim_1"
        )
        tQ_layout = cute.make_ordered_layout(
            (self.num_Q_load_threads // tQK_shape_dim_1, tQK_shape_dim_1),
            order=(1, 0),
        )
        tK_layout = cute.make_ordered_layout(
            (self.num_producer_threads // tQK_shape_dim_1, tQK_shape_dim_1),
            order=(1, 0),
        )
        # So that we don't have to check if we overshoot kBlockM when we load Q
        assert self.tile_m % tQ_layout.shape[0] == 0
        tV_shape_dim_1 = sV_layout_atom.outer.shape[1] // async_copy_elems
        tV_layout = cute.make_ordered_layout(
            (self.num_producer_threads // tV_shape_dim_1, tV_shape_dim_1),
            order=(1, 0),
        )
        # TODO: need a different layout for O if O dtype is not the same as V dtype
        # tO_layout: thread layout for O store
        output_copy_elems = universal_copy_bits // self.output_dtype.width
        tO_shape_dim_1 = sO_layout_atom.outer.shape[1] // output_copy_elems
        tO_layout = cute.make_ordered_layout(
            (self.num_epilogue_threads // tO_shape_dim_1, tO_shape_dim_1),
            order=(1, 0),
        )
        # So that we don't have to check if we overshoot kBlockM when we store O
        assert self.tile_m % tO_layout.shape[0] == 0

        # Value layouts for copies
        vQKV_layout = cute.make_layout((1, async_copy_elems))
        vO_layout = cute.make_layout((1, output_copy_elems))

        self.gmem_tiled_copy_Q = cute.make_tiled_copy_tv(atom_async_copy, tQ_layout, vQKV_layout)
        self.gmem_tiled_copy_K = cute.make_tiled_copy_tv(atom_async_copy, tK_layout, vQKV_layout)
        self.gmem_tiled_copy_V = cute.make_tiled_copy_tv(atom_async_copy, tV_layout, vQKV_layout)
        # gmem_tiled_copy_O: tiled copy for O store
        self.gmem_tiled_copy_O = cute.make_tiled_copy_tv(atom_universal_copy, tO_layout, vO_layout)

    def _get_smem_layout_atom(self):
        raise NotImplementedError()

    def _get_tiled_mma(self):
        raise NotImplementedError()

    def _get_shared_storage_cls(self):
        raise NotImplementedError()

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        """Configures and launches the flash attention kernel.

        mQ/mK/mV/mO has same data types(supports fp16 and bf16) and same layout:
        (batch_size, seqlen_q, num_head, head_dim):(_, _, _, 1)
        """
        raise NotImplementedError()

    @cute.jit
    def epilogue(
        self,
        acc_O: cute.Tensor,
        lse: cute.Tensor,
        split_idx: Int32,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        sO: cute.Tensor,
        seqlen: SeqlenInfoQK,
        gmem_tiled_copy_O: cute.TiledCopy,
        tma_atom_O: Optional[cute.CopyAtom],
        tiled_mma: cute.TiledMma,
        tidx: Int32,
        m_block: Int32,
        head_idx: Int32,
        batch_idx: Int32,
    ):
        # SplitKV stores the FP32 accumulator directly to its partial
        # workspace. Only the non-split output path needs the BF16 conversion
        # and register-to-shared-memory round trip.
        if const_expr(not self.is_split_kv and not self.is_fp8):
            rO = cute.make_fragment_like(acc_O, self.output_dtype)
            rO.store(acc_O.load().to(self.output_dtype))
            # Make sure all threads have finished reading V
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.Epilogue),
                number_of_threads=self.num_epilogue_threads,
            )
            smem_copy_atom_O = utils.get_smem_store_atom(
                self.arch.major * 10 + self.arch.minor, self.output_dtype
            )
            smem_thr_copy_O = cute.make_tiled_copy_C(
                smem_copy_atom_O, tiled_mma
            ).get_slice(tidx)
            taccOrO = smem_thr_copy_O.retile(rO)
            taccOsO = smem_thr_copy_O.partition_D(sO)
            # copy acc O from rmem to smem with the smem copy atom
            cute.copy(smem_copy_atom_O, taccOrO, taccOsO)

        cO = cute.make_identity_tensor((self.tile_m, self.tile_hdimv))
        pack_gqa = PackGQA(
            self.tile_m, self.tile_hdimv, self.check_hdim_v_oob, self.qhead_per_kvhead
        )

        # Write LSE from rmem -> gmem
        if const_expr(mLSE is not None):
            if const_expr(self.is_split_kv):
                # SPLITKV: mLSE is (..., head, [batch,] split) -> pick this split too
                mLSE_cur = seqlen.offset_batch_Q(mLSE, batch_idx, dim=2)[None, head_idx, split_idx]
            else:
                mLSE_cur = seqlen.offset_batch_Q(mLSE, batch_idx, dim=2)[None, head_idx]
            if const_expr(not self.pack_gqa):
                gLSE = cute.local_tile(mLSE_cur, (self.tile_m,), (m_block,))
                gLSE_expanded_layout = cute.append(
                    gLSE.layout, cute.make_layout((self.tile_hdimv,), stride=(0,))
                )
                gLSE_expanded = cute.make_tensor(gLSE.iterator, gLSE_expanded_layout)
                thr_mma = tiled_mma.get_slice(tidx)
                taccOgLSE = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(gLSE_expanded))
                assert cute.size(taccOgLSE, mode=[0]) == cute.size(lse)
                taccOcO = layout_utils.reshape_acc_to_mn(thr_mma.partition_C(cO))
                t0accOcO = layout_utils.reshape_acc_to_mn(thr_mma.get_slice(0).partition_C(cO))
                # Only the thread corresponding to column 0 writes out the lse to gmem
                if taccOcO[0][1] == 0:
                    for m in cutlass.range(cute.size(taccOgLSE.shape[1]), unroll_full=True):
                        if (
                            t0accOcO[m, 0][0]
                            < seqlen.seqlen_q - m_block * self.tile_m - taccOcO[0][0]
                        ):
                            taccOgLSE[m, 0] = lse[m]
            else:
                pack_gqa.store_LSE(mLSE_cur, lse, tiled_mma, tidx, m_block, seqlen.seqlen_q)

        ragged = False  # force domain_offset (rank-preserving) path for varlen
        if const_expr(self.is_split_kv):
            # SPLITKV: mO is (s, d, head, [batch,] split) -> pick this split too
            mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3, ragged=ragged)[
                None, None, head_idx, split_idx
            ]
        elif const_expr(self.pack_gqa):
            if const_expr(not seqlen.has_cu_seqlens_q):
                mO_cur = mO[None, None, head_idx, batch_idx]
            else:
                mO_cur = cute.domain_offset(
                    ((0, seqlen.offset_q), 0), mO[None, None, head_idx]
                )
        else:
            mO_cur = seqlen.offset_batch_Q(mO, batch_idx, dim=3, ragged=ragged)[
                None, None, head_idx
            ]
        # thr_mma = tiled_mma.get_slice(tidx)
        # taccOgO = thr_mma.partition_C(gO)
        # cute.autovec_copy(rO, taccOgO)
        # sync to make sure all smem stores are done
        if const_expr(self.use_tma_O):
            # ensure smem writes are visible to TMA
            cute.arch.fence_view_async_shared()
            cute.arch.barrier_arrive(
                barrier_id=int(NamedBarrierFwd.Epilogue),
                number_of_threads=self.num_epilogue_threads + cute.arch.WARP_SIZE,
            )
            gO = cute.local_tile(mO_cur, (self.tile_m, self.tile_hdimv), (m_block, 0))
            store_O, _, _ = copy_utils.tma_get_copy_fn(
                tma_atom_O, 0, cute.make_layout(1), sO, gO, single_stage=True
            )
            warp_idx = cute.arch.make_warp_uniform(cute.arch.warp_idx())
            if warp_idx == 4:
                cute.arch.barrier(
                    barrier_id=int(NamedBarrierFwd.Epilogue),
                    number_of_threads=self.num_epilogue_threads + cute.arch.WARP_SIZE,
                )
                store_O()
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
        elif const_expr(self.is_split_kv):
            # SPLITKV: out_partial is fp32 -> store straight from the fp32 accumulator, no sO
            # round-trip and no dtype conversion. acc_O and partition_C(gO) share a layout
            # (that is why upstream's commented-out autovec_copy above would have worked).
            #
            # pack_gqa + split: mO_cur carries the packed ((qhead_per_kvhead, seqlen_q), headdim)
            # compound strides from pack_gqa_layout, i.e. packed row r addresses head r%qpk, token
            # r//qpk in the physically-normal (total_q, nheads, headdim) partial buffer. local_tile +
            # partition_C thread the compound stride through unchanged, and the direct element write
            # below honours arbitrary strides, so the SAME MMA-C store scatters each packed row to
            # its (head, token) slot. This is the fp32 pointer-scatter the upstream SM80 kernel
            # leaves as NotImplementedError; the combine kernel needs no change because the partial
            # buffer is physically normal-layout. We widen the row limit to packed rows and always
            # mask (packed decode tiles -- QPK rows -- are far from a full 64-row tile).
            thr_mma_O = tiled_mma.get_slice(tidx)
            taccOcO = thr_mma_O.partition_C(cO)
            if const_expr(self.pack_gqa):
                # ``cute.local_tile`` requires the packed leading mode to
                # divide tile_m.  QPK=6 (Qwen3.5-27B) does not divide 64, so
                # form each destination pointer from the logical packed row.
                # This also makes the scatter semantics explicit and works
                # for every GQA ratio.
                row_limit = seqlen.seqlen_q * self.qhead_per_kvhead - m_block * self.tile_m
                for i in cutlass.range(cute.size(acc_O.shape), unroll_full=True):
                    row_in_tile = taccOcO[i][0]
                    if row_in_tile < row_limit:
                        packed_row = m_block * self.tile_m + row_in_tile
                        query_idx = packed_row // self.qhead_per_kvhead
                        query_head = packed_row - query_idx * self.qhead_per_kvhead
                        out_ptr = utils.elem_pointer(
                            mO_cur,
                            ((query_head, query_idx), taccOcO[i][1]),
                        )
                        out_ptr[0] = acc_O[i]
            else:
                gO = cute.local_tile(
                    mO_cur, (self.tile_m, self.tile_hdimv), (m_block, 0)
                )
                taccOgO = thr_mma_O.partition_C(gO)
                row_limit = seqlen.seqlen_q - m_block * self.tile_m
                if row_limit >= self.tile_m:
                    cute.autovec_copy(acc_O, taccOgO)
                else:
                    for i in cutlass.range(cute.size(acc_O.shape), unroll_full=True):
                        if taccOcO[i][0] < row_limit:
                            taccOgO[i] = acc_O[i]
        elif const_expr(self.is_fp8):
            # FP8 inputs produce BF16 output. Store the accumulator directly
            # instead of reserving a 32-KiB BF16 sO tile and round-tripping it
            # through shared memory. Besides removing the epilogue barrier,
            # this lets the 32-row FP8 kernel keep multiple CTAs resident.
            thr_mma_O = tiled_mma.get_slice(tidx)
            taccOcO = thr_mma_O.partition_C(cO)
            if const_expr(self.pack_gqa):
                row_limit = (
                    seqlen.seqlen_q * self.qhead_per_kvhead
                    - m_block * self.tile_m
                )
                for i in cutlass.range(cute.size(acc_O.shape), unroll_full=True):
                    row_in_tile = taccOcO[i][0]
                    if row_in_tile < row_limit:
                        packed_row = m_block * self.tile_m + row_in_tile
                        query_idx = packed_row // self.qhead_per_kvhead
                        query_head = (
                            packed_row - query_idx * self.qhead_per_kvhead
                        )
                        out_ptr = utils.elem_pointer(
                            mO_cur,
                            ((query_head, query_idx), taccOcO[i][1]),
                        )
                        out_ptr[0] = self.output_dtype(acc_O[i])
            else:
                gO = cute.local_tile(
                    mO_cur, (self.tile_m, self.tile_hdimv), (m_block, 0)
                )
                taccOgO = thr_mma_O.partition_C(gO)
                row_limit = seqlen.seqlen_q - m_block * self.tile_m
                for i in cutlass.range(cute.size(acc_O.shape), unroll_full=True):
                    if taccOcO[i][0] < row_limit:
                        taccOgO[i] = self.output_dtype(acc_O[i])
        else:
            cute.arch.barrier(
                barrier_id=int(NamedBarrierFwd.Epilogue),
                number_of_threads=self.num_epilogue_threads,
            )
            gmem_thr_copy_O = gmem_tiled_copy_O.get_slice(tidx)
            tOsO = gmem_thr_copy_O.partition_S(sO)
            tOrO = cute.make_fragment_like(tOsO, self.output_dtype)
            # load acc O from smem to rmem for wider vectorization
            cute.autovec_copy(tOsO, tOrO)
            if const_expr(not self.pack_gqa):
                gO = cute.local_tile(mO_cur, (self.tile_m, self.tile_hdimv), (m_block, 0))
                tOgO = gmem_thr_copy_O.partition_D(gO)
                tOcO = gmem_thr_copy_O.partition_S(cO)
                t0OcO = gmem_tiled_copy_O.get_slice(0).partition_S(cO)
                tOpO = utils.predicate_k(tOcO, limit=mO.shape[1])
                # copy acc O from rmem to gmem
                for rest_m in cutlass.range_constexpr(cute.size(tOrO.shape[1])):
                    if (
                        t0OcO[0, rest_m, 0][0]
                        < seqlen.seqlen_q - m_block * self.tile_m - tOcO[0][0]
                    ):
                        cute.copy(
                            gmem_tiled_copy_O,
                            tOrO[None, rest_m, None],
                            tOgO[None, rest_m, None],
                            pred=tOpO[None, rest_m, None]
                            if const_expr(self.check_hdim_v_oob)
                            else None,
                        )
            else:
                pack_gqa.store_O(mO_cur, tOrO, gmem_tiled_copy_O, tidx, m_block, seqlen.seqlen_q)

    @cute.jit
    def advance_pipeline(self, pipeline_index):
        return pipeline_index + 1 if pipeline_index < self.num_stages - 1 else 0

    @cute.jit
    def load_Q(
        self,
        gmem_thr_copy: cute.TiledCopy,
        gQ: cute.Tensor,
        sQ: cute.Tensor,
        block: Int32,
        seqlen: Int32,
        headdim: Int32,
    ):
        tQsQ, tQgQ = gmem_thr_copy.partition_D(sQ), gmem_thr_copy.partition_S(gQ)
        cQ = cute.make_identity_tensor((self.tile_m, self.tile_hdim))
        tQcQ = gmem_thr_copy.partition_S(cQ)
        t0QcQ = gmem_thr_copy.get_slice(0).partition_S(cQ)
        tQpQ = utils.predicate_k(tQcQ, limit=headdim)
        for m in cutlass.range_constexpr(cute.size(tQsQ.shape[1])):
            # Instead of using tQcQ, we using t0QcQ and subtract the offset from the limit
            # (seqlen - block * kBlockM). This is because the entries of t0QcQ are known at compile time.
            if t0QcQ[0, m, 0][0] < seqlen - block * self.tile_m - tQcQ[0][0]:
                cute.copy(
                    gmem_thr_copy,
                    tQgQ[None, m, None],
                    tQsQ[None, m, None],
                    pred=tQpQ[None, m, None] if const_expr(self.check_hdim_oob) else None,
                )
            # We don't need to clear the sQ smem tiles since we'll only write out the valid outputs

    @cute.jit
    def _kv_block_tile(
        self,
        tXgX: cute.Tensor,
        block: Int32,
        mKV_pages: Optional[cute.Tensor],
        mPageTable: Optional[cute.Tensor],
        page_batch_idx: Int32,
        gmem_thr_copy,
        tile_hdim: cutlass.Constexpr,
    ):
        """PAGED: (CPY, N, K) gmem source for one n_block.

        Non-paged: just the block slice of the pre-partitioned tensor (identical codegen to the
        original in-loop indexing). Paged: page_size % tile_n == 0 means the block sits inside a
        single page, so we re-tile that page -- one int32 page-table read, no gather.
        """
        if const_expr(not self.paged_kv):
            return tXgX[None, None, None, block]
        if const_expr(self.blocks_per_page == 1):
            page_slot, sub_block = block, 0
        else:
            page_slot = block // self.blocks_per_page
            sub_block = block % self.blocks_per_page
        page = mPageTable[page_batch_idx, page_slot]
        gX = cute.local_tile(
            mKV_pages[None, None, page], (self.tile_n, tile_hdim), (sub_block, 0)
        )
        return gmem_thr_copy.partition_S(gX)

    @cute.jit
    def load_K(
        self,
        gmem_tiled_copy: cute.TiledCopy,
        tKgK: cute.Tensor,
        tKsK: cute.Tensor,
        tKcK: cute.Tensor,
        t0KcK: cute.Tensor,
        tKpK: cute.Tensor,
        block: Int32,
        smem_pipe_write: Int32,
        seqlen: Int32,
        need_predicates: cutlass.Constexpr,
        mKV_pages: Optional[cute.Tensor] = None,   # PAGED
        mPageTable: Optional[cute.Tensor] = None,  # PAGED
        page_batch_idx: Int32 = 0,                 # PAGED
        gmem_thr_copy: Optional[cute.TiledCopy] = None,  # PAGED
    ):
        # PAGED: point this tile at page_table[batch, block // blocks_per_page]. Hoisting the
        # block slice out of the copy loop is a no-op for the contiguous path.
        tKg = self._kv_block_tile(
            tKgK, block, mKV_pages, mPageTable, page_batch_idx, gmem_thr_copy, self.tile_hdim
        )
        # Do we need to check if we overshoot kBlockN when we load K?
        is_even_n_smem_k = self.tile_n % gmem_tiled_copy.tiler_mn[0].shape == 0
        if const_expr(need_predicates or not is_even_n_smem_k):
            # Instead of using tKcK, we using t0KcK and subtract the offset from the limit
            # (seqlen - block * kBlockN). This is because the entries of t0KcK are known at compile time.
            if const_expr(is_even_n_smem_k):
                seqlen_limit = seqlen - block * self.tile_n
            else:
                if const_expr(not need_predicates):
                    seqlen_limit = self.tile_n
                else:
                    seqlen_limit = cutlass.min(seqlen - block * self.tile_n, self.tile_n)
            seqlen_limit -= tKcK[0][0]
            for n in cutlass.range_constexpr(cute.size(tKsK.shape[1])):
                if t0KcK[0, n, 0][0] < seqlen_limit:
                    cute.copy(
                        gmem_tiled_copy,
                        tKg[None, n, None],
                        tKsK[
                            None, n, None, smem_pipe_write if const_expr(self.num_stages > 1) else 0
                        ],
                        pred=tKpK[None, n, None] if const_expr(self.check_hdim_oob) else None,
                    )
                # We don't need to clear the sK smem tiles since we'll mask out the scores anyway.
        else:
            cute.copy(
                gmem_tiled_copy,
                tKg,
                tKsK[None, None, None, smem_pipe_write if const_expr(self.num_stages > 1) else 0],
                pred=tKpK if const_expr(self.check_hdim_oob) else None,
            )

    @cute.jit
    def load_V(
        self,
        gmem_tiled_copy: cute.TiledCopy,
        tVgV: cute.Tensor,
        tVsV: cute.Tensor,
        tVcV: cute.Tensor,
        t0VcV: cute.Tensor,
        tVpV: cute.Tensor,
        block: Int32,
        smem_pipe_write: Int32,
        seqlen: Int32,
        need_predicates: cutlass.Constexpr,
        mKV_pages: Optional[cute.Tensor] = None,   # PAGED
        mPageTable: Optional[cute.Tensor] = None,  # PAGED
        page_batch_idx: Int32 = 0,                 # PAGED
        gmem_thr_copy: Optional[cute.TiledCopy] = None,  # PAGED
    ):
        tVg = self._kv_block_tile(
            tVgV, block, mKV_pages, mPageTable, page_batch_idx, gmem_thr_copy, self.tile_hdimv
        )
        # Do we need to check if we overshoot kBlockN when we load V?
        is_even_n_smem_v = self.tile_n % gmem_tiled_copy.tiler_mn[0].shape == 0
        if const_expr(need_predicates or not is_even_n_smem_v):
            for n in cutlass.range_constexpr(cute.size(tVsV.shape[1])):
                # If kBlockN doesn't evenly divide the tiled copy, only the last `n` needs to be checked
                if (
                    is_even_n_smem_v
                    or n < cute.size(tVsV.shape[1]) - 1
                    or tVcV[0, n, 0][0] < self.tile_n
                ):
                    predicate = tVpV[None, n, None] if const_expr(self.check_hdim_v_oob) else None
                    if const_expr(need_predicates):
                        seqlen_limit = seqlen - block * self.tile_n - tVcV[0][0]
                        predicate_n = t0VcV[0, n, 0][0] < seqlen_limit
                        predicate = cute.make_fragment_like(tVpV[None, 0, None])
                        for k in cutlass.range_constexpr(cute.size(predicate.shape[1])):
                            for i in cutlass.range_constexpr(cute.size(predicate.shape[0])):
                                predicate[i, k] = (
                                    tVpV[i, n, k] if const_expr(self.check_hdim_v_oob) else True
                                ) and predicate_n
                    cute.copy(
                        gmem_tiled_copy,
                        tVg[None, n, None],
                        tVsV[
                            None, n, None, smem_pipe_write if const_expr(self.num_stages > 1) else 0
                        ],
                        pred=predicate,
                    )
        else:
            cute.copy(
                gmem_tiled_copy,
                tVg,
                tVsV[None, None, None, smem_pipe_write if const_expr(self.num_stages > 1) else 0],
                pred=tVpV if const_expr(self.check_hdim_v_oob) else None,
            )


class FlashAttentionForwardSm80(FlashAttentionForwardBase):
    def _get_smem_layout_atom(self):
        if const_expr(self.is_fp8):
            def fp8_layout_atom(k_dim):
                return cute.make_composed_layout(
                    cute.make_swizzle(0, 4, 4),
                    0,
                    cute.make_layout((16, k_dim), stride=(k_dim + 16, 1)),
                )

            sQ_layout_atom = fp8_layout_atom(self.tile_hdim)
        else:
            sQ_layout_atom = sm80_utils.get_smem_layout_atom(
                self.dtype, self.tile_hdim
            )
        sK_layout_atom = sQ_layout_atom
        sV_layout_atom = (
            cute.make_composed_layout(
                cute.make_swizzle(4, 4, 4),
                0,
                cute.make_layout(
                    (16, self.tile_hdimv), stride=(self.tile_hdimv, 1)
                ),
            )
            if const_expr(self.is_fp8)
            else sm80_utils.get_smem_layout_atom(self.dtype, self.tile_hdimv)
        )
        sO_layout_atom = sm80_utils.get_smem_layout_atom(
            self.output_dtype, self.tile_hdimv
        )
        sP_layout_atom = (
            cute.make_composed_layout(
                cute.make_swizzle(0, 4, 4),
                0,
                cute.make_layout(
                    (16, self.tile_n), stride=(self.tile_n + 16, 1)
                ),
            )
            if const_expr(self.is_fp8)
            else None
        )
        return sQ_layout_atom, sK_layout_atom, sV_layout_atom, sO_layout_atom, sP_layout_atom

    def _get_tiled_mma(self):
        # FP8 arithmetic is emitted with inline PTX below. A BF16 tiled-MMA
        # descriptor supplies the identical m16n8 accumulator/thread mapping
        # without depending on CuTe's CUDA-version-gated FP8 lowering.
        mma_dtype = cutlass.BFloat16 if const_expr(self.is_fp8) else self.dtype
        mma_op = warp.MmaF16BF16Op(mma_dtype, Float32, (16, 8, 16))
        mma_k = 16
        tiled_mma_qk = cute.make_tiled_mma(
            mma_op,
            (self.num_threads // 32, 1, 1),
            permutation_mnk=(self.num_threads // 32 * 16, 16, mma_k),
        )
        tiled_mma_pv = cute.make_tiled_mma(
            mma_op,
            (self.num_threads // 32, 1, 1),
            permutation_mnk=(self.num_threads // 32 * 16, 16, mma_k),
        )
        return tiled_mma_qk, tiled_mma_pv

    def _get_shared_storage_cls(self):
        q_bytes = cute.cosize(self.sQ_layout) * self.dtype.width // 8
        o_bytes = cute.cosize(self.sO_layout) * self.output_dtype.width // 8
        qo_bytes = q_bytes if const_expr(self.is_fp8) else max(q_bytes, o_bytes)
        sQ_struct = cute.struct.Align[
            cute.struct.MemRange[cutlass.Int8, qo_bytes], 1024
        ]
        sK_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cute.cosize(self.sK_layout)], 1024
        ]
        sV_struct = cute.struct.Align[
            cute.struct.MemRange[self.dtype, cute.cosize(self.sV_layout)], 1024
        ]
        sP_struct = cute.struct.Align[
            cute.struct.MemRange[
                self.dtype,
                cute.cosize(self.sP_layout)
                if const_expr(self.is_fp8 and (self.tile_hdim != 128 or self.is_causal))
                else 1,
            ],
            1024,
        ]
        qv_bytes = max(
            qo_bytes,
            cute.cosize(self.sV_layout) * self.dtype.width // 8,
        )
        sQV_struct = cute.struct.Align[
            cute.struct.MemRange[cutlass.Int8, qv_bytes], 1024
        ]

        @cute.struct
        class SharedStorageQKV:
            sV: sV_struct
            sQ: sQ_struct
            sK: sK_struct

        @cute.struct
        class SharedStorageSharedQV:
            sQ: sQV_struct
            sK: sK_struct

        @cute.struct
        class SharedStorageQKVFP8:
            sV: sV_struct
            sQ: sQ_struct
            sK: sK_struct
            sP: sP_struct

        @cute.struct
        class SharedStorageSharedQVFP8:
            sQ: sQV_struct
            sK: sK_struct
            sP: sP_struct

        if const_expr(self.is_fp8):
            return SharedStorageQKVFP8 if const_expr(not self.Q_in_regs) else SharedStorageSharedQVFP8
        return SharedStorageQKV if const_expr(not self.Q_in_regs) else SharedStorageSharedQV

    @cute.jit
    def __call__(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        softmax_scale: Float32,
        mCuSeqlensQ: Optional[cute.Tensor] = None,
        mCuSeqlensK: Optional[cute.Tensor] = None,
        mSeqUsedQ: Optional[cute.Tensor] = None,
        mSeqUsedK: Optional[cute.Tensor] = None,
        mPageTable: Optional[cute.Tensor] = None,
        mNumSplitsDynamic: Optional[cute.Tensor] = None,   # DYNSPLIT
        mWorkMap: Optional[cute.Tensor] = None,   # COMPACTGRID
        mQDescale: Optional[cute.Tensor] = None,
        mKDescale: Optional[cute.Tensor] = None,
        mVDescale: Optional[cute.Tensor] = None,
        window_size_left: Optional[Int32] = None,
        window_size_right: Optional[Int32] = None,
        learnable_sink: Optional[cute.Tensor] = None,
        blocksparse_tensors: Optional[BlockSparseTensors] = None,
        aux_data: AuxData = AuxData(),
        # Always keep stream as the last parameter (EnvStream: obtained implicitly via TVM FFI).
        stream: cuda.CUstream = None,
    ):
        """Configures and launches the flash attention kernel.

        mQ/mK/mV/mO has same data types(supports fp16 and bf16) and same layout:
        (batch_size, seqlen_q, num_head, head_dim):(_, _, _, 1)
        """
        assert learnable_sink is None, "Learnable sink is not supported in this kernel"
        self._check_type(
            *(t.element_type if t is not None else None for t in (mQ, mK, mV, mO, mLSE, mCuSeqlensQ, mCuSeqlensK, mSeqUsedQ, mSeqUsedK))
        )
        tiled_mma_qk, tiled_mma_pv = self._get_tiled_mma()
        self.num_mma_threads = tiled_mma_pv.size
        self.num_producer_threads = self.num_threads
        self.num_Q_load_threads = self.num_threads
        self.num_epilogue_threads = self.num_threads
        # self.use_tma_O = self.arch >= 90 and mCuSeqlensQ is None
        self.use_tma_O = False  # sm120: force autovec store (no TMA-O atom)
        self._setup_attributes()
        SharedStorage = self._get_shared_storage_cls()
        mQ, mK, mV, mO = [assume_tensor_aligned(t) for t in (mQ, mK, mV, mO)]
        # SPLITKV: num_splits is the LEADING dim of the user-layout out_partial.
        num_splits = mO.shape[0] if const_expr(self.is_split_kv) else Int32(1)
        # Layout permutation: 4D non-varlen vs 3D varlen
        QO_layout_transpose = [1, 3, 2, 0] if const_expr(mCuSeqlensQ is None) else [0, 2, 1]
        # SPLITKV: O carries a leading num_splits dim -> keep it as the LAST mode so all the
        # existing (s,d,h,b) indexing keeps working and we just append [None,...,split_idx].
        O_layout_transpose = (
            ([2, 4, 3, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 3, 2, 0])
            if const_expr(self.is_split_kv) else QO_layout_transpose
        )
        KV_layout_transpose = (
            [1, 3, 2, 0]
            if const_expr(mPageTable is not None or mCuSeqlensK is None)
            else [0, 2, 1]
        )
        mQ = cute.make_tensor(mQ.iterator, cute.select(mQ.layout, mode=QO_layout_transpose))
        mO = cute.make_tensor(mO.iterator, cute.select(mO.layout, mode=O_layout_transpose))
        mK, mV = [
            cute.make_tensor(t.iterator, cute.select(t.layout, mode=KV_layout_transpose))
            for t in (mK, mV)
        ]
        if const_expr(mLSE is not None):
            LSE_layout_transpose = (
                ([3, 2, 1, 0] if const_expr(mCuSeqlensQ is None) else [2, 1, 0])
                if const_expr(self.is_split_kv)
                else ([2, 1, 0] if const_expr(mCuSeqlensQ is None) else [1, 0])
            )
            mLSE = cute.make_tensor(mLSE.iterator, cute.select(mLSE.layout, mode=LSE_layout_transpose))
        # PACKGQA: fold qhead_per_kvhead into the seqlen mode -> mode 0 becomes
        # (qhead_per_kvhead, seqlen_q) and the head mode shrinks to nheads_kv. This is the call
        # every other kernel in the repo makes and this one was missing.
        if const_expr(self.pack_gqa):
            nheads_kv = mK.shape[2]
            mQ = pack_gqa_layout(mQ, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            mO = pack_gqa_layout(mO, self.qhead_per_kvhead, nheads_kv, head_idx=2)
            if const_expr(mLSE is not None):
                mLSE = pack_gqa_layout(mLSE, self.qhead_per_kvhead, nheads_kv, head_idx=1)
        # TileScheduler for varlen, simple grid for non-varlen
        if const_expr(mCuSeqlensQ is not None or mSeqUsedQ is not None):
            TileScheduler = SingleTileVarlenScheduler
        else:
            TileScheduler = SingleTileScheduler
        num_batch = (
            mCuSeqlensQ.shape[0] - 1
            if const_expr(mCuSeqlensQ is not None)
            else mQ.shape[3]
        )
        tile_sched_args = TileSchedulerArguments(
            num_block=cute.ceil_div(cute.size(mQ.shape[0]), self.tile_m),
            num_head=cute.size(mQ.shape[2]),
            num_batch=num_batch,
            num_splits=num_splits,
            is_split_kv=self.is_split_kv,
            seqlen_k=0,
            headdim=mQ.shape[1],
            headdim_v=mV.shape[1],
            total_q=cute.size(mQ.shape[0])
            if const_expr(mCuSeqlensQ is not None)
            else cute.size(mQ.shape[0]) * cute.size(mQ.shape[3]),
            tile_shape_mn=(self.tile_m, self.tile_n),
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
            mCuSeqlensQ=mCuSeqlensQ,
            mSeqUsedQ=mSeqUsedQ,
            # Match SM103's causal ordering so heavy Q tiles do not form the
            # final scheduling wave.
            lpt=self.lpt,
        )
        tile_sched_params = TileScheduler.to_underlying_arguments(tile_sched_args)
        grid_dim = TileScheduler.get_grid_shape(tile_sched_params)
        if const_expr(self.runtime_balanced_splits):
            # Use a fixed launch shape while mapping CTAs from device-resident
            # sequence lengths. This keeps CUDA Graph replay stable.
            grid_dim = (self.runtime_balanced_grid_size, Int32(1), Int32(1))
        elif const_expr(self.compact_grid):
            # COMPACTGRID: one CTA per row of the work map, nothing wasted.
            grid_dim = (cute.size(mWorkMap.shape[0]), Int32(1), Int32(1))
        softmax_scale_log2, softmax_scale = utils.compute_softmax_scale_log2(softmax_scale, self.score_mod)
        fastdiv_mods = utils.compute_fastdiv_mods(mQ, mK, self.qhead_per_kvhead, self.pack_gqa, aux_data.tensors)

        self.atrex_sm120_prefill_kernel(
            mQ,
            mK,
            mV,
            mO,
            mLSE,
            mCuSeqlensQ,
            mCuSeqlensK,
            mSeqUsedQ,
            mSeqUsedK,
            mPageTable,   # PAGED
            mNumSplitsDynamic,   # DYNSPLIT
            mWorkMap,   # COMPACTGRID
            mQDescale,
            mKDescale,
            mVDescale,
            softmax_scale_log2,
            softmax_scale,
            window_size_left,
            window_size_right,
            self.sQ_layout,
            self.sK_layout,
            self.sV_layout,
            self.sO_layout,
            self.sP_layout,
            self.gmem_tiled_copy_Q,
            self.gmem_tiled_copy_K,
            self.gmem_tiled_copy_V,
            self.gmem_tiled_copy_O,
            tiled_mma_qk,
            tiled_mma_pv,
            SharedStorage,
            tile_sched_params,
            TileScheduler,
            aux_data,
            fastdiv_mods,
            num_splits,          # SPLITKV
        ).launch(
            grid=grid_dim,
            block=[self.num_threads, 1, 1],
            smem=SharedStorage.size_in_bytes(),
            stream=stream,
            min_blocks_per_mp=(
                2 if self.is_fp8 and self.tile_hdim == 128
                and self.tile_m == 64 and self.num_threads == 128
                and not self.is_causal else 0
            ),
        )

    @cute.kernel
    def atrex_sm120_prefill_kernel(
        self,
        mQ: cute.Tensor,
        mK: cute.Tensor,
        mV: cute.Tensor,
        mO: cute.Tensor,
        mLSE: Optional[cute.Tensor],
        mCuSeqlensQ: Optional[cute.Tensor],
        mCuSeqlensK: Optional[cute.Tensor],
        mSeqUsedQ: Optional[cute.Tensor],
        mSeqUsedK: Optional[cute.Tensor],
        mPageTable: Optional[cute.Tensor],   # PAGED
        mNumSplitsDynamic: Optional[cute.Tensor],   # DYNSPLIT
        mWorkMap: Optional[cute.Tensor],   # COMPACTGRID
        mQDescale: Optional[cute.Tensor],
        mKDescale: Optional[cute.Tensor],
        mVDescale: Optional[cute.Tensor],
        softmax_scale_log2: Float32,
        softmax_scale: Optional[Float32],
        window_size_left: Optional[Int32],
        window_size_right: Optional[Int32],
        sQ_layout: cute.ComposedLayout,
        sK_layout: cute.ComposedLayout,
        sV_layout: cute.ComposedLayout,
        sO_layout: cute.ComposedLayout,
        sP_layout: cute.ComposedLayout | None,
        gmem_tiled_copy_Q: cute.TiledCopy,
        gmem_tiled_copy_K: cute.TiledCopy,
        gmem_tiled_copy_V: cute.TiledCopy,
        gmem_tiled_copy_O: cute.TiledCopy,
        tiled_mma_qk: cute.TiledMma,
        tiled_mma_pv: cute.TiledMma,
        SharedStorage: cutlass.Constexpr,
        tile_sched_params,
        TileScheduler: cutlass.Constexpr[Callable],
        aux_data: AuxData = AuxData(),
        fastdiv_mods=None,
        num_splits: Int32 = Int32(1),   # SPLITKV
    ):
        # Thread index, block index
        tidx, _, _ = cute.arch.thread_idx()

        if const_expr(self.runtime_balanced_splits):
            j, _, _ = cute.arch.block_idx()
            num_batch = self.runtime_balanced_batch_size
            m_block = Int32(0)
            num_head = Int32(0)
            batch_size = Int32(0)
            split_idx = j
            ns_b = Int32(self.runtime_balanced_grid_size)
            if const_expr(1 < self.runtime_balanced_batch_size <= 32):
                lane_idx = cute.arch.lane_idx()
                request_n_blocks = Int32(0)
                if lane_idx < num_batch:
                    request_n_blocks = cute.ceil_div(
                        mSeqUsedK[lane_idx], self.tile_n
                    )
                n_blocks_cumulative = request_n_blocks
                for scan_step in cutlass.range_constexpr(
                    (self.runtime_balanced_batch_size - 1).bit_length()
                ):
                    offset = 1 << scan_step
                    partial_sum = cute.arch.shuffle_sync_up(
                        n_blocks_cumulative,
                        offset=offset,
                        mask_and_clamp=0,
                    )
                    if lane_idx >= offset:
                        n_blocks_cumulative += partial_sum
                total_n_blocks = cute.arch.shuffle_sync(
                    n_blocks_cumulative,
                    self.runtime_balanced_batch_size - 1,
                )
                total_n_blocks = cutlass.max(total_n_blocks, 1)
                extra_work = self.runtime_balanced_grid_size - num_batch
                extra_cumulative = (
                    n_blocks_cumulative * extra_work // total_n_blocks
                )
                prior_extra = cute.arch.shuffle_sync_up(
                    extra_cumulative,
                    offset=1,
                    mask_and_clamp=0,
                )
                if lane_idx == 0:
                    prior_extra = Int32(0)
                request_splits = Int32(0)
                splits_cumulative = Int32(self.runtime_balanced_grid_size)
                if lane_idx < num_batch:
                    request_splits = (
                        Int32(1) + extra_cumulative - prior_extra
                    )
                    splits_cumulative = lane_idx + 1 + extra_cumulative
                is_prior_request = lane_idx < num_batch
                if is_prior_request:
                    is_prior_request = splits_cumulative <= j
                batch_size = cute.arch.popc(
                    cute.arch.vote_ballot_sync(is_prior_request)
                )
                work_begin = (
                    Int32(0)
                    if batch_size == 0
                    else cute.arch.shuffle_sync(
                        splits_cumulative, batch_size - 1
                    )
                )
                split_idx = j - work_begin
                ns_b = cute.arch.shuffle_sync(request_splits, batch_size)
            elif const_expr(32 < self.runtime_balanced_batch_size <= 64):
                # Compute the compact high-batch map in lane 0 of each warp
                # and broadcast it. A tiny serial scan is insignificant next
                # to long-KV attention, while avoiding a second scheduling
                # kernel and remaining safe under CUDA Graph replay.
                lane_idx = cute.arch.lane_idx()
                total_n_blocks = Int32(0)
                cumulative_blocks = Int32(0)
                prior_work_end = Int32(0)
                extra_work = Int32(
                    self.runtime_balanced_grid_size - num_batch
                )
                if lane_idx == 0:
                    for request_idx in cutlass.range(
                        self.runtime_balanced_batch_size, unroll=1
                    ):
                        total_n_blocks += cute.ceil_div(
                            mSeqUsedK[request_idx], self.tile_n
                        )
                    total_n_blocks = cutlass.max(total_n_blocks, 1)
                    for request_idx in cutlass.range(
                        self.runtime_balanced_batch_size, unroll=1
                    ):
                        cumulative_blocks += cute.ceil_div(
                            mSeqUsedK[request_idx], self.tile_n
                        )
                        work_end = (
                            request_idx
                            + 1
                            + cumulative_blocks * extra_work // total_n_blocks
                        )
                        if j >= prior_work_end:
                            if j < work_end:
                                batch_size = request_idx
                                split_idx = j - prior_work_end
                                ns_b = work_end - prior_work_end
                        prior_work_end = work_end
                batch_size = cute.arch.shuffle_sync(batch_size, 0)
                split_idx = cute.arch.shuffle_sync(split_idx, 0)
                ns_b = cute.arch.shuffle_sync(ns_b, 0)
            if tidx == 0:
                if split_idx == 0:
                    mNumSplitsDynamic[batch_size] = ns_b
        elif const_expr(self.compact_grid):
            # COMPACTGRID: the scheduler's rectangular decode is ignored; row j of the work map
            # says exactly which (m_block, head, sequence, split) this CTA owns.
            j, _, _ = cute.arch.block_idx()
            if const_expr(self.compact_short_q):
                m_block = Int32(0)
                num_head = Int32(0)
                encoded_work = mWorkMap[j, 0]
                batch_size = encoded_work >> 16
                split_idx = encoded_work & 0xFFFF
            else:
                m_block = mWorkMap[j, 0]
                num_head = mWorkMap[j, 1]
                batch_size = mWorkMap[j, 2]
                split_idx = mWorkMap[j, 3]
        else:
            tile_scheduler = TileScheduler.create(tile_sched_params)
            work_tile = tile_scheduler.initial_work_tile_info()
            m_block, num_head, batch_size, split_idx = work_tile.tile_idx

        block_info = BlockInfo(
            self.tile_m,
            self.tile_n,
            self.is_causal,
            self.is_local,
            self.is_split_kv,  # SPLITKV
            window_size_left,
            window_size_right,
            qhead_per_kvhead_packgqa=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        seqlen = SeqlenInfoQK.create(
            batch_idx=batch_size,
            seqlen_q_static=mQ.shape[0],
            seqlen_k_static=mK.shape[0],
            mCuSeqlensQ=mCuSeqlensQ,
            mCuSeqlensK=mCuSeqlensK,
            mSeqUsedQ=mSeqUsedQ,
            mSeqUsedK=mSeqUsedK,
        )
        # DYNSPLIT: this sequence's own split count. Splits beyond it come back with
        # n_block_max <= n_block_min -- the empty-split case the SplitKV patch already handles
        # (acc_O=0, lse=-inf; combine's scale>0 guard drops them), so nothing else has to change.
        if const_expr(not self.runtime_balanced_splits):
            ns_b = num_splits
            if const_expr(self.dynamic_splits):
                ns_b = mNumSplitsDynamic[batch_size]
        n_block_min, n_block_max = block_info.get_n_block_min_max(
            seqlen, m_block, split_idx, ns_b
        )
        # For varlen, wasted grid tiles (where batch_idx >= num_batch) will have
        # seqlen_q=seqlen_k=0 and n_block_max=0.  Clamp to 0 so we don't use a
        # negative block index for K/V loads; the load/store predicates already
        # guard all memory accesses when seqlen is 0.
        n_block = cutlass.max(n_block_max - 1, 0)

        # ///////////////////////////////////////////////////////////////////////////////
        # Get the appropriate tiles for this thread block.
        # ///////////////////////////////////////////////////////////////////////////////
        blkQ_shape = (self.tile_m, self.tile_hdim)
        blkK_shape = (self.tile_n, self.tile_hdim)
        blkV_shape = (self.tile_n, self.tile_hdimv)
        # PACKGQA: grid iterates nheads_kv, so num_head IS the kv head already.
        num_head_kv = num_head if const_expr(self.pack_gqa) else num_head // self.qhead_per_kvhead
        if const_expr(self.pack_gqa):
            if const_expr(not seqlen.has_cu_seqlens_q):
                mQ_cur = mQ[None, None, num_head, batch_size]
            else:
                # explicit nested-mode offset instead of offset_batch_Q (whose rank-3
                # domain_offset with None coords misbehaves here)
                mQ_cur = cute.domain_offset(
                    ((0, seqlen.offset_q), 0), mQ[None, None, num_head]
                )
        elif const_expr(not seqlen.has_cu_seqlens_q):
            mQ_cur = mQ[None, None, num_head, batch_size]
        else:
            # keep the ORIGINAL rank-2 domain_offset: routing the unpacked varlen path through
            # offset_batch_Q (rank-3 domain_offset with None coords) gave illegal accesses once
            # m_block > 0 and offset_q > 0 (4x2048 varlen prefill).
            mQ_cur = cute.domain_offset((seqlen.offset_q, 0), mQ[None, None, num_head])
        mK_pages, mV_pages = None, None
        if const_expr(self.paged_kv):
            # PAGED: mK/mV are (page_size, head_dim, head_kv, num_pages) after the layout
            # transpose -- the 4th mode is the PAGE index, not the batch. Keep it whole and let
            # load_K/load_V pick the page for each n_block. mK_cur/mV_cur below are only used to
            # build the copy partitions and predicates, whose layouts do not depend on the page.
            mK_pages = mK[None, None, num_head_kv, None]
            mV_pages = mV[None, None, num_head_kv, None]
            mK_cur = mK[None, None, num_head_kv, 0]
            mV_cur = mV[None, None, num_head_kv, 0]
        elif const_expr(not seqlen.has_cu_seqlens_k):
            mK_cur = mK[None, None, num_head_kv, batch_size]
            mV_cur = mV[None, None, num_head_kv, batch_size]
        else:
            mK_cur = cute.domain_offset((seqlen.offset_k, 0), mK[None, None, num_head_kv])
            mV_cur = cute.domain_offset((seqlen.offset_k, 0), mV[None, None, num_head_kv])
        gQ = (
            cute.local_tile(mQ_cur, blkQ_shape, (m_block, 0))
            if const_expr(not self.pack_gqa)
            else None
        )
        gK = cute.local_tile(mK_cur, blkK_shape, (None, 0))
        gV = cute.local_tile(mV_cur, blkV_shape, (None, 0))

        # ///////////////////////////////////////////////////////////////////////////////
        # Get shared memory buffer
        # ///////////////////////////////////////////////////////////////////////////////
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        sQ = cute.make_tensor(
            cute.recast_ptr(storage.sQ.data_ptr(), dtype=self.dtype), sQ_layout
        )
        sK = storage.sK.get_tensor(sK_layout)
        if const_expr(not self.Q_in_regs):
            sV = storage.sV.get_tensor(sV_layout)
        else:
            sV = cute.make_tensor(cute.recast_ptr(sQ.iterator, dtype=self.dtype), sV_layout)
        sP = storage.sP.get_tensor(sP_layout) if const_expr(self.is_fp8) else None
        # Transpose view of V to tensor with layout (head_dim_v, tile_n) for tiled mma
        sVt = layout_utils.transpose_view(sV)

        gmem_thr_copy_K = gmem_tiled_copy_K.get_slice(tidx)
        gmem_thr_copy_V = gmem_tiled_copy_V.get_slice(tidx)
        # (CPY_Atom, CPY_N, CPY_K, n_block)
        tKsK, tKgK = gmem_thr_copy_K.partition_D(sK), gmem_thr_copy_K.partition_S(gK)
        # (CPY_Atom, CPY_N, CPY_K, n_block)
        tVsV, tVgV = gmem_thr_copy_V.partition_D(sV), gmem_thr_copy_V.partition_S(gV)

        # ///////////////////////////////////////////////////////////////////////////////
        # Tile MMA compute thread partitions and allocate accumulators
        # ///////////////////////////////////////////////////////////////////////////////
        thr_mma_qk = tiled_mma_qk.get_slice(tidx)
        thr_mma_pv = tiled_mma_pv.get_slice(tidx)
        if const_expr(self.is_fp8):
            tSrQ = (
                cute.make_rmem_tensor((4, self.tile_hdim // 32), Int32)
                if const_expr(self.tile_hdim == 128 and not self.is_causal
                              and self.tile_m == (self.num_threads // 32) * 16
                              and not self.is_split_kv)
                else None
            )
            tSrK = None
            tOrVt = None
        else:
            tSrQ = thr_mma_qk.make_fragment_A(thr_mma_qk.partition_A(sQ))
            tSrK = thr_mma_qk.make_fragment_B(thr_mma_qk.partition_B(sK[None, None, 0]))
            tOrVt = thr_mma_pv.make_fragment_B(thr_mma_pv.partition_B(sVt[None, None, 0]))
        acc_shape_O = thr_mma_pv.partition_shape_C((self.tile_m, self.tile_hdimv))
        acc_O = cute.make_rmem_tensor(acc_shape_O, Float32)
        acc_O.fill(0.0)

        # ///////////////////////////////////////////////////////////////////////////////
        # Smem copy atom tiling
        # ///////////////////////////////////////////////////////////////////////////////
        # mma.sync.m16n8k32 consumes four packed 32-bit registers for A and
        # two for B.  Each ldmatrix.b16 result therefore carries two adjacent
        # FP8 values.  The m8n16 instruction is for sub-byte unpacking and is
        # not the native FP8 fragment layout.
        if const_expr(self.is_fp8):
            smem_thr_copy_Q = None
            smem_thr_copy_K = None
            smem_thr_copy_V = None
            tSsQ = None
            tSsK = None
            tOsVt = None
        else:
            smem_copy_atom_QK = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=False, num_matrices=4),
                self.dtype,
            )
            smem_copy_atom_V = cute.make_copy_atom(
                warp.LdMatrix8x8x16bOp(transpose=True, num_matrices=4),
                self.dtype,
            )
            smem_thr_copy_Q = utils.make_tiled_copy_A(
                smem_copy_atom_QK, tiled_mma_qk
            ).get_slice(tidx)
            smem_thr_copy_K = utils.make_tiled_copy_B(
                smem_copy_atom_QK, tiled_mma_qk
            ).get_slice(tidx)
            smem_thr_copy_V = utils.make_tiled_copy_B(
                smem_copy_atom_V, tiled_mma_pv
            ).get_slice(tidx)
            tSsQ = smem_thr_copy_Q.partition_S(sQ)
            tSsK = smem_thr_copy_K.partition_S(sK)
            tOsVt = smem_thr_copy_V.partition_S(sVt)

        # ///////////////////////////////////////////////////////////////////////////////
        # Predicate: Mark indices that need to copy when problem_shape isn't a multiple
        # of tile_shape
        # ///////////////////////////////////////////////////////////////////////////////
        # Construct identity layout for KV
        cK = cute.make_identity_tensor((self.tile_n, self.tile_hdim))
        tKcK = gmem_thr_copy_K.partition_S(cK)
        t0KcK = gmem_thr_copy_K.get_slice(0).partition_S(cK)
        if const_expr(self.tile_hdim == self.tile_hdimv):
            tVcV = tKcK
            t0VcV = t0KcK
        else:
            cV = cute.make_identity_tensor((self.tile_n, self.tile_hdimv))
            tVcV = gmem_thr_copy_V.partition_S(cV)
            t0VcV = gmem_thr_copy_V.get_slice(0).partition_S(cV)
        # Allocate predicate tensors for m and n, here we only allocate the tile of k, and
        # use "if" on the mn dimension.
        # This is to reduce register pressure and gets 2-3% performance gain.
        tKpK = utils.predicate_k(tKcK, limit=mK.shape[1])
        if const_expr(self.same_hdim_kv):
            tVpV = tKpK
        else:
            tVpV = utils.predicate_k(tVcV, limit=mV.shape[1])

        qk_descale = Float32(1.0)
        v_descale = Float32(1.0)
        if const_expr(self.is_fp8):
            if const_expr(mQDescale is not None):
                qk_descale *= Float32(mQDescale[batch_size, num_head_kv])
            if const_expr(mKDescale is not None):
                qk_descale *= Float32(mKDescale[batch_size, num_head_kv])
            if const_expr(mVDescale is not None):
                v_descale = Float32(mVDescale[batch_size, num_head_kv])
            softmax_scale_log2 *= qk_descale
            if const_expr(softmax_scale is not None):
                softmax_scale *= qk_descale

        # shape: (atom_v_m * rest_m)
        softmax = Softmax.create(
            softmax_scale_log2,
            num_rows=acc_O.shape[0][0] * acc_O.shape[1],
            softmax_scale=softmax_scale,
        )
        softmax.reset()

        # group parameters for compute_one_n_block
        mma_params = SimpleNamespace(
            thr_mma_qk=thr_mma_qk,
            thr_mma_pv=thr_mma_pv,
            tSrQ=tSrQ,
            tSrK=tSrK,
            tOrVt=tOrVt,
            acc_O=acc_O,
        )
        smem_copy_params = SimpleNamespace(
            smem_thr_copy_Q=smem_thr_copy_Q,
            smem_thr_copy_K=smem_thr_copy_K,
            smem_thr_copy_V=smem_thr_copy_V,
            tSsQ=tSsQ,
            tSsK=tSsK,
            tOsVt=tOsVt,
            sQ=sQ,
            sK=sK,
            sV=sV,
            sP=sP,
        )
        load_K = partial(
            self.load_K, gmem_tiled_copy_K, tKgK, tKsK, tKcK, t0KcK, tKpK, seqlen=seqlen.seqlen_k,
            mKV_pages=mK_pages, mPageTable=mPageTable, page_batch_idx=batch_size,  # PAGED
            gmem_thr_copy=gmem_thr_copy_K,
        )
        load_V = partial(
            self.load_V, gmem_tiled_copy_V, tVgV, tVsV, tVcV, t0VcV, tVpV, seqlen=seqlen.seqlen_k,
            mKV_pages=mV_pages, mPageTable=mPageTable, page_batch_idx=batch_size,  # PAGED
            gmem_thr_copy=gmem_thr_copy_V,
        )

        compute_one_n_block = partial(
            self.compute_one_n_block,
            mma_params=mma_params,
            smem_copy_params=smem_copy_params,
            softmax=softmax,
            load_K=load_K,
            load_V=load_V,
            score_mod=self.score_mod,
            batch_idx=batch_size,
            head_idx=num_head,
            m_block=m_block,
            aux_data=aux_data,
            fastdiv_mods=fastdiv_mods,
            n_block_min=n_block_min,   # PREFETCHCLAMP
        )

        # ///////////////////////////////////////////////////////////////////////////////
        # Prologue
        # ///////////////////////////////////////////////////////////////////////////////
        # Start async loads of the last mn-tile, where we take care of the mn residue
        if const_expr(not self.pack_gqa):
            gmem_thr_copy_Q = gmem_tiled_copy_Q.get_slice(tidx)
            self.load_Q(
                gmem_thr_copy_Q, gQ, sQ, m_block, seqlen=seqlen.seqlen_q, headdim=mQ.shape[1]
            )
        else:
            # PACKGQA: rows of the tile come from qhead_per_kvhead different heads, so the load
            # is a per-row pointer gather instead of a contiguous tile copy.
            PackGQA(
                self.tile_m, self.tile_hdim, self.check_hdim_oob, self.qhead_per_kvhead
            ).load_Q(mQ_cur, sQ, gmem_tiled_copy_Q, tidx, m_block, seqlen.seqlen_q)
        cute.arch.cp_async_commit_group()

        def preprocess_Q():
            # Q-in-registers preprocess runs after only Q and K[0] have
            # been committed. Waiting on the full future pipeline depth
            # would leave Q pending when stages > 1, then read uninitialized
            # shared data before its storage is reused for V.
            cute.arch.cp_async_wait_group(
                1 if const_expr(self.tile_hdim == 128 and self.Q_in_regs)
                else self.num_stages * 2 - 1
            )
            if const_expr(self.Q_in_regs):
                cute.arch.barrier()
                tSrQ_copy_view = smem_thr_copy_Q.retile(tSrQ)
                cute.copy(smem_thr_copy_Q, tSsQ, tSrQ_copy_view)

        # If Q_in_regs, we load Q, then load 1 stage of K, then (optionally) rotate Q and
        # read from smem_q to registers, then load V.
        # If !Q_in_regs, we load Q, load all stages of K & V, then (optionally) rotate Q.
        if const_expr(self.Q_in_regs):
            load_K(n_block, smem_pipe_write=0, need_predicates=True)
            cute.arch.cp_async_commit_group()
            preprocess_Q()
            cute.arch.barrier()  # Make sure all threads have read smem_q before loading V

        for stage in cutlass.range_constexpr(self.num_stages):
            # PREFETCHCLAMP: `>= n_block_min` instead of `>= 0` -- see the guards in
            # compute_one_n_block. commit_group() stays unconditional so the pipeline depth the
            # first preprocess_Q() waits on does not change.
            if const_expr(not self.Q_in_regs or stage > 0):
                if stage == 0 or n_block - stage >= n_block_min:
                    load_K(n_block - stage, smem_pipe_write=stage, need_predicates=stage == 0)
                cute.arch.cp_async_commit_group()
            if const_expr(stage < self.num_stages - 1):
                if stage == 0 or n_block - stage >= n_block_min:
                    load_V(n_block - stage, smem_pipe_write=stage, need_predicates=stage == 0)
                cute.arch.cp_async_commit_group()
        if const_expr(not self.Q_in_regs):
            preprocess_Q()

        # ///////////////////////////////////////////////////////////////////////////////
        # Mainloop
        # ///////////////////////////////////////////////////////////////////////////////
        # Start processing of the first n-block.
        # For performance reason, we separate out two kinds of iterations:
        # those that need masking on S, and those that don't.
        # We need masking on S for the very last block when K and V has length not multiple of tile_n.
        # We also need masking on S if it's causal, for the last several blocks.
        mask = AttentionMask(
            self.tile_m,
            self.tile_n,
            seqlen,
            window_size_left,
            window_size_right,
            self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )
        mask_fn = partial(
            mask.apply_mask,
            batch_idx=batch_size,
            head_idx=num_head,
            m_block=m_block,
            thr_mma=thr_mma_qk,
            mask_causal=self.is_causal,
            mask_local=self.is_local,
            aux_data=aux_data,
            fastdiv_mods=fastdiv_mods if const_expr(self.mask_mod is not None) else None,
        )

        # First iteration with seqlen masking
        smem_pipe_read = Int32(0)
        smem_pipe_write = Int32(self.num_stages - 1)
        # NOREGRESS: the empty-split guard is a runtime branch around a big inlined region;
        # only the SplitKV path needs it, and paying it unconditionally cost ~8% on the
        # tile_m=128 prefill config.
        if const_expr(self.is_split_kv):
            if n_block_max > n_block_min:  # SPLITKV: empty trailing split does nothing
                compute_one_n_block(
                    n_block,
                    smem_pipe_read,
                    smem_pipe_write,
                    is_first_n_block=True,
                    seqlen=seqlen,
                    mask_fn=partial(mask_fn, mask_mod=self.mask_mod, mask_seqlen=True),
                )
        else:
            compute_one_n_block(
                n_block,
                smem_pipe_read,
                smem_pipe_write,
                is_first_n_block=True,
                seqlen=seqlen,
                mask_fn=partial(mask_fn, mask_mod=self.mask_mod, mask_seqlen=True),
            )
        smem_pipe_read = self.advance_pipeline(smem_pipe_read)
        smem_pipe_write = self.advance_pipeline(smem_pipe_write)
        # Next couple of iterations with causal masking
        if const_expr(self.is_causal or self.is_local):
            n_block_min_causal_local_mask = block_info.get_n_block_min_causal_local_mask(
                seqlen, m_block, n_block_min
            )
            # NOREGRESS: original trip count when not splitting.
            if const_expr(self.is_split_kv):
                n_mask_count = cutlass.max(n_block_max - 1 - n_block_min_causal_local_mask, 0)
            else:
                n_mask_count = n_block_max - 1 - n_block_min_causal_local_mask
            for n_tile in cutlass.range(n_mask_count, unroll=1):
                n_block = n_block_max - 2 - n_tile
                compute_one_n_block(
                    n_block,
                    smem_pipe_read,
                    smem_pipe_write,
                    seqlen=seqlen,
                    mask_fn=partial(mask_fn, mask_mod=self.mask_mod, mask_seqlen=True),
                )
                smem_pipe_read = self.advance_pipeline(smem_pipe_read)
                smem_pipe_write = self.advance_pipeline(smem_pipe_write)
        # The remaining iterations have no masking
        # SPLITKV: trip count is RELATIVE to this split's n_block_min (was: absolute n_block),
        # and clamped so an EMPTY trailing split iterates zero times.
        # NOREGRESS: keep the original absolute trip count when not splitting.
        if const_expr(self.is_split_kv):
            n_tile_count = cutlass.max(n_block - n_block_min, 0)
        else:
            n_tile_count = n_block
        for n_tile in cutlass.range(n_tile_count, unroll=1):
            compute_one_n_block(
                n_block - n_tile - 1, smem_pipe_read, smem_pipe_write,
                seqlen=seqlen, is_first_n_block=False,
                mask_fn=partial(mask_fn, mask_mod=self.mask_mod, mask_seqlen=False)
            )
            smem_pipe_read = self.advance_pipeline(smem_pipe_read)
            smem_pipe_write = self.advance_pipeline(smem_pipe_write)
        # TODO: local

        # normalize acc_O by row_sum and calculate the lse
        row_scale = softmax.finalize()
        if const_expr(self.is_fp8):
            if const_expr(self.tile_hdim == 128):
                row_scale.store(row_scale.load() * (v_descale / self.fp8_p_scale))
            else:
                row_scale.store(row_scale.load() * v_descale)
        softmax.rescale_O(acc_O, row_scale)

        # ///////////////////////////////////////////////////////////////////////////////
        # Epilogue
        # ///////////////////////////////////////////////////////////////////////////////
        # reuse sQ's data iterator
        sO = cute.make_tensor(
            cute.recast_ptr(sQ.iterator, dtype=self.output_dtype), sO_layout
        )
        self.epilogue(
            acc_O,
            softmax.row_sum,
            split_idx,
            mO,
            mLSE,
            sO,
            seqlen,
            gmem_tiled_copy_O,
            None,
            tiled_mma_pv,
            tidx,
            m_block,
            num_head,
            batch_size,
        )


    @cute.jit
    def compute_one_n_block(
        self,
        n_block: Int32,
        smem_pipe_read: Int32,
        smem_pipe_write: Int32,
        mma_params: SimpleNamespace,
        smem_copy_params: SimpleNamespace,
        softmax: Softmax,
        load_K: Callable,
        load_V: Callable,
        score_mod: Callable | None,
        batch_idx: cutlass.Int32,
        head_idx: cutlass.Int32,
        m_block: cutlass.Int32,
        seqlen: SeqlenInfoQK,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=None,
        mask_fn: Optional[Callable] = None,
        is_first_n_block: cutlass.Constexpr = False,
        check_inf: cutlass.Constexpr = True,
        # PREFETCHCLAMP: lowest KV block this split owns. Prefetches below it are pure wasted
        # DRAM traffic (the mainloop stops before ever computing on them). Defaults to the python
        # int 0, so the non-split path keeps the original constant-folded guards.
        n_block_min: Int32 = 0,
    ):
        """Compute one n_block of S/O.

        This function provides different variants for processing the first n block versus
        subsequent blocks.
        """

        def sync():
            cute.arch.cp_async_wait_group(self.num_stages * 2 - 2)
            cute.arch.barrier()

        acc_shape_S = mma_params.thr_mma_qk.partition_shape_C((self.tile_m, self.tile_n))
        acc_S = cute.make_rmem_tensor(acc_shape_S, Float32)
        acc_S.fill(0.0)
        # wait for smem tile QK before mma calculation for S
        sync()

        # need predicates for the first tile
        def load_V_next():
            if self.num_stages == 1 or n_block - self.num_stages + 1 >= n_block_min:   # PREFETCHCLAMP
                load_V(
                    n_block - self.num_stages + 1,
                    smem_pipe_write,
                    need_predicates=is_first_n_block and self.num_stages == 1,
                )
            cute.arch.cp_async_commit_group()

        load_V_next()
        if const_expr(self.is_fp8):
            # HD128 specialization is removed at compile time for HD256.
            if const_expr(self.tile_hdim == 128):
                lane = cute.arch.thread_idx()[0] % 32
                warp_idx = cute.arch.thread_idx()[0] // 32
                matrix_id = lane // 8
                k_stage = smem_pipe_read if const_expr(self.num_stages > 1) else 0
                q_row = (
                    warp_idx * 16
                    + lane % 8
                    + (matrix_id % 2) * 8
                )
                k_row_in_tile = lane % 8
                k_col_half = ((lane // 8) % 2) * 16
                for kk in cutlass.range_constexpr(self.tile_hdim // 32):
                    q_col = kk * 32 + (matrix_id // 2) * 16
                    q_fragments = cute.make_rmem_tensor((4, cute.size(acc_S.shape[1])), Int32)
                    for m in cutlass.range_constexpr(cute.size(acc_S.shape[1])):
                        if const_expr(mma_params.tSrQ is not None):
                            if const_expr(is_first_n_block):
                                q_regs = _ldmatrix_m8n8_x4_b16(
                                    utils.elem_pointer(smem_copy_params.sQ, (q_row, q_col)).align(16)
                                )
                                for reg in cutlass.range_constexpr(4):
                                    mma_params.tSrQ[reg, kk] = q_regs[reg]
                            for reg in cutlass.range_constexpr(4):
                                q_fragments[reg, m] = mma_params.tSrQ[reg, kk]
                        else:
                            q_regs = _ldmatrix_m8n8_x4_b16(
                                utils.elem_pointer(
                                    smem_copy_params.sQ,
                                    (q_row + m * (self.num_threads // 32) * 16, q_col),
                                ).align(16)
                            )
                            for reg in cutlass.range_constexpr(4):
                                q_fragments[reg, m] = q_regs[reg]
                    for n in cutlass.range_constexpr(self.tile_n // 8):
                        k_row = n * 8 + k_row_in_tile
                        b = _ldmatrix_m8n8_x2_b16(
                            utils.elem_pointer(
                                smem_copy_params.sK,
                                (k_row, kk * 32 + k_col_half, k_stage),
                            ).align(16)
                        )
                        for m in cutlass.range_constexpr(cute.size(acc_S.shape[1])):
                            a = tuple(q_fragments[reg, m] for reg in range(4))
                            c = (
                                acc_S[(0, 0), m, n], acc_S[(1, 0), m, n],
                                acc_S[(0, 1), m, n], acc_S[(1, 1), m, n],
                            )
                            d = _mma_m16n8k32_f32_fp8(
                                a, b, c, self.dtype, reuse_accumulator=True
                            )
                            acc_S[(0, 0), m, n] = d[0]
                            acc_S[(1, 0), m, n] = d[1]
                            acc_S[(0, 1), m, n] = d[2]
                            acc_S[(1, 1), m, n] = d[3]
            else:
                lane = cute.arch.thread_idx()[0] % 32
                warp_idx = cute.arch.thread_idx()[0] // 32
                matrix_id = lane // 8
                k_stage = smem_pipe_read if const_expr(self.num_stages > 1) else 0
                q_row = (
                    warp_idx * 16
                    + lane % 8
                    + (matrix_id % 2) * 8
                )
                k_row_in_tile = lane % 8
                k_col_half = ((lane // 8) % 2) * 16
                for kk in cutlass.range_constexpr(self.tile_hdim // 32):
                    q_col = kk * 32 + (matrix_id // 2) * 16
                    a = _ldmatrix_m8n8_x4_b16(
                        utils.elem_pointer(
                            smem_copy_params.sQ, (q_row, q_col)
                        ).align(16)
                    )
                    for n in cutlass.range_constexpr(self.tile_n // 8):
                        k_row = n * 8 + k_row_in_tile
                        b = _ldmatrix_m8n8_x2_b16(
                            utils.elem_pointer(
                                smem_copy_params.sK,
                                (k_row, kk * 32 + k_col_half, k_stage),
                            ).align(16)
                        )
                        c = (
                            acc_S[(0, 0), 0, n],
                            acc_S[(1, 0), 0, n],
                            acc_S[(0, 1), 0, n],
                            acc_S[(1, 1), 0, n],
                        )
                        d = _mma_m16n8k32_f32_fp8(a, b, c, self.dtype)
                        acc_S[(0, 0), 0, n] = d[0]
                        acc_S[(1, 0), 0, n] = d[1]
                        acc_S[(0, 1), 0, n] = d[2]
                        acc_S[(1, 1), 0, n] = d[3]
        else:
            sm80_utils.gemm(
                mma_params.thr_mma_qk,
                acc_S,
                mma_params.tSrQ,
                mma_params.tSrK,
                smem_copy_params.tSsQ,
                smem_copy_params.tSsK[
                    None, None, None, smem_pipe_read if const_expr(self.num_stages > 1) else 0
                ],
                smem_copy_params.smem_thr_copy_Q,
                smem_copy_params.smem_thr_copy_K,
                A_in_regs=self.Q_in_regs,
            )
        if const_expr(score_mod is not None):
            self.apply_score_mod(
                mma_params.thr_mma_qk,
                batch_idx,
                head_idx,
                m_block,
                acc_S,
                n_block,
                softmax_scale=softmax.softmax_scale,
                seqlen=seqlen,
                aux_data=aux_data,
                fastdiv_mods=fastdiv_mods,
            )

        smem_pipe_write = self.advance_pipeline(smem_pipe_write)

        def load_K_next():
            if n_block - self.num_stages >= n_block_min:   # PREFETCHCLAMP
                load_K(n_block - self.num_stages, smem_pipe_write, need_predicates=False)
            cute.arch.cp_async_commit_group()

        # wait for smem tile V for O
        if const_expr(self.num_stages == 1):
            sync()
            load_K_next()
        if const_expr(mask_fn is not None):
            mask_fn(acc_S, n_block=n_block)
        row_scale = softmax.online_softmax(acc_S, is_first=is_first_n_block, check_inf=check_inf)
        if const_expr(
            self.is_fp8 and self.tile_hdim == 128
            and not self.is_causal and self.tile_m >= 64
        ):
            # A long KV scan rarely raises the running maximum after its
            # first few tiles. Multiplication by exactly one is unnecessary;
            # keep the original FP32 update whenever a row actually changes.
            acc_O_mn = layout_utils.reshape_acc_to_mn(mma_params.acc_O)
            for r in cutlass.range_constexpr(cute.size(row_scale)):
                if row_scale[r] != 1.0:
                    acc_O_mn[r, None].store(
                        acc_O_mn[r, None].load() * row_scale[r]
                    )
        else:
            softmax.rescale_O(mma_params.acc_O, row_scale)
        rP = cute.make_fragment_like(acc_S, self.dtype)
        if const_expr(self.fp8_p_scale != 1.0):
            rP.store((acc_S.load() * self.fp8_p_scale).to(self.dtype))
        else:
            rP.store(acc_S.load().to(self.dtype))
        if const_expr(self.is_fp8):
            # HD128 specialization is removed at compile time for HD256.
            if const_expr(self.tile_hdim == 128):
                lane = cute.arch.thread_idx()[0] % 32
                warp_idx = cute.arch.thread_idx()[0] // 32
                lane_group = lane // 4
                lane_in_group = lane % 4
                p_layout = cute.make_layout(
                    ((4, 2, 2), cute.size(acc_S.shape[1]), self.tile_n // 32),
                    stride=((1, 4, 8), 16, 16 * cute.size(acc_S.shape[1])),
                )
                tOrP = cute.make_rmem_tensor(p_layout, self.dtype)
                tOrP_i32 = cute.recast_tensor(tOrP, Int32)
                if const_expr(self.tile_hdim == 128 and not self.is_causal):
                    # Each accumulator lane owns two adjacent columns in two rows.
                    # FP8 MMA A needs four adjacent columns from the same rows.
                    # Exchange paired columns within each four-lane row group.
                    rP_i16 = cute.recast_tensor(rP, cutlass.Int16)
                    source_lane = lane_group * 4 + (lane_in_group % 2) * 2
                    selector = Int32(0x5410) + (lane_in_group // 2) * 0x2222
                    for m in cutlass.range_constexpr(cute.size(acc_S.shape[1])):
                        for kb in cutlass.range_constexpr(self.tile_n // 32):
                            for half_k in cutlass.range_constexpr(2):
                                for half_m in cutlass.range_constexpr(2):
                                    n = kb * 4 + half_k * 2
                                    packed = (
                                        (Int32(rP_i16[(0, half_m), m, n]) & 0xFFFF)
                                        | (Int32(rP_i16[(0, half_m), m, n + 1]) << 16)
                                    )
                                    even = cute.arch.shuffle_sync(packed, source_lane)
                                    odd = cute.arch.shuffle_sync(packed, source_lane + 1)
                                    tOrP_i32[(0, half_m, half_k), m, kb] = _prmt_b32(
                                        even, odd, selector
                                    )
                else:
                    for n in cutlass.range_constexpr(self.tile_n // 8):
                        col = n * 8 + lane_in_group * 2
                        smem_copy_params.sP[warp_idx * 16 + lane_group, col] = rP[(0, 0), 0, n]
                        smem_copy_params.sP[warp_idx * 16 + lane_group, col + 1] = rP[(1, 0), 0, n]
                        smem_copy_params.sP[warp_idx * 16 + lane_group + 8, col] = rP[(0, 1), 0, n]
                        smem_copy_params.sP[warp_idx * 16 + lane_group + 8, col + 1] = rP[(1, 1), 0, n]
                    cute.arch.sync_warp()
                    matrix_id = lane // 8
                    p_row = warp_idx * 16 + lane % 8 + (matrix_id % 2) * 8
                    for kb in cutlass.range_constexpr(self.tile_n // 32):
                        p_col = kb * 32 + (matrix_id // 2) * 16
                        regs = _ldmatrix_m8n8_x4_b16(
                            utils.elem_pointer(smem_copy_params.sP, (p_row, p_col)).align(16)
                        )
                        for reg in cutlass.range_constexpr(4):
                            tOrP_i32[(0, reg % 2, reg // 2), 0, kb] = regs[reg]
            else:
                lane = cute.arch.thread_idx()[0] % 32
                warp_idx = cute.arch.thread_idx()[0] // 32
                lane_group = lane // 4
                lane_in_group = lane % 4
                for n in cutlass.range_constexpr(self.tile_n // 8):
                    col = n * 8 + lane_in_group * 2
                    smem_copy_params.sP[warp_idx * 16 + lane_group, col] = rP[(0, 0), 0, n]
                    smem_copy_params.sP[warp_idx * 16 + lane_group, col + 1] = rP[(1, 0), 0, n]
                    smem_copy_params.sP[warp_idx * 16 + lane_group + 8, col] = rP[(0, 1), 0, n]
                    smem_copy_params.sP[warp_idx * 16 + lane_group + 8, col + 1] = rP[(1, 1), 0, n]
                cute.arch.sync_warp()
                p_layout = cute.make_layout(
                    ((4, 2, 2), 1, self.tile_n // 32),
                    stride=((1, 4, 8), 0, 16),
                )
                tOrP = cute.make_rmem_tensor(p_layout, self.dtype)
                tOrP_i32 = cute.recast_tensor(tOrP, Int32)
                matrix_id = lane // 8
                p_row = warp_idx * 16 + lane % 8 + (matrix_id % 2) * 8
                for kb in cutlass.range_constexpr(self.tile_n // 32):
                    p_col = kb * 32 + (matrix_id // 2) * 16
                    regs = _ldmatrix_m8n8_x4_b16(
                        utils.elem_pointer(smem_copy_params.sP, (p_row, p_col)).align(16)
                    )
                    for reg in cutlass.range_constexpr(4):
                        tOrP_i32[(0, reg % 2, reg // 2), 0, kb] = regs[reg]
        else:
            tOrP = layout_utils.reshape_acc_to_frgA(rP)
        if const_expr(self.num_stages > 1):
            sync()
            load_K_next()
        if const_expr(self.is_fp8):
            # HD128 specialization is removed at compile time for HD256.
            if const_expr(self.tile_hdim == 128):
                lane = cute.arch.thread_idx()[0] % 32
                lane_row = lane % 16
                lane_matrix_col = (lane // 16) * 16
                v_stage = (
                    smem_pipe_read if const_expr(self.num_stages > 1) else 0
                )
                for kb in cutlass.range_constexpr(self.tile_n // 32):
                    for group in cutlass.range_constexpr(self.tile_hdimv // 32):
                        col = group * 32 + lane_matrix_col
                        b0 = _ldmatrix_m16n16_x2_trans_b8(
                            utils.elem_pointer(
                                smem_copy_params.sV,
                                (kb * 32 + lane_row, col, v_stage),
                            ).align(16)
                        )
                        b1 = _ldmatrix_m16n16_x2_trans_b8(
                            utils.elem_pointer(
                                smem_copy_params.sV,
                                (kb * 32 + 16 + lane_row, col, v_stage),
                            ).align(16)
                        )
                        for m in cutlass.range_constexpr(cute.size(acc_S.shape[1])):
                            a = (
                                tOrP_i32[(0, 0, 0), m, kb], tOrP_i32[(0, 1, 0), m, kb],
                                tOrP_i32[(0, 0, 1), m, kb], tOrP_i32[(0, 1, 1), m, kb],
                            )
                            for local_n in cutlass.range_constexpr(4):
                                n = group * 4 + local_n
                                b = (b0[local_n], b1[local_n])
                                c = (
                                    mma_params.acc_O[(0, 0), m, n], mma_params.acc_O[(1, 0), m, n],
                                    mma_params.acc_O[(0, 1), m, n], mma_params.acc_O[(1, 1), m, n],
                                )
                                d = _mma_m16n8k32_f32_fp8(
                                    a, b, c, self.dtype, reuse_accumulator=True
                                )
                                mma_params.acc_O[(0, 0), m, n] = d[0]
                                mma_params.acc_O[(1, 0), m, n] = d[1]
                                mma_params.acc_O[(0, 1), m, n] = d[2]
                                mma_params.acc_O[(1, 1), m, n] = d[3]
            else:
                lane = cute.arch.thread_idx()[0] % 32
                lane_row = lane % 16
                lane_matrix_col = (lane // 16) * 16
                v_stage = (
                    smem_pipe_read if const_expr(self.num_stages > 1) else 0
                )
                for kb in cutlass.range_constexpr(self.tile_n // 32):
                    a = (
                        tOrP_i32[(0, 0, 0), 0, kb],
                        tOrP_i32[(0, 1, 0), 0, kb],
                        tOrP_i32[(0, 0, 1), 0, kb],
                        tOrP_i32[(0, 1, 1), 0, kb],
                    )
                    for group in cutlass.range_constexpr(self.tile_hdimv // 32):
                        col = group * 32 + lane_matrix_col
                        b0 = _ldmatrix_m16n16_x2_trans_b8(
                            utils.elem_pointer(
                                smem_copy_params.sV,
                                (kb * 32 + lane_row, col, v_stage),
                            ).align(16)
                        )
                        b1 = _ldmatrix_m16n16_x2_trans_b8(
                            utils.elem_pointer(
                                smem_copy_params.sV,
                                (kb * 32 + 16 + lane_row, col, v_stage),
                            ).align(16)
                        )
                        for local_n in cutlass.range_constexpr(4):
                            n = group * 4 + local_n
                            b = (b0[local_n], b1[local_n])
                            c = (
                                mma_params.acc_O[(0, 0), 0, n],
                                mma_params.acc_O[(1, 0), 0, n],
                                mma_params.acc_O[(0, 1), 0, n],
                                mma_params.acc_O[(1, 1), 0, n],
                            )
                            d = _mma_m16n8k32_f32_fp8(a, b, c, self.dtype)
                            mma_params.acc_O[(0, 0), 0, n] = d[0]
                            mma_params.acc_O[(1, 0), 0, n] = d[1]
                            mma_params.acc_O[(0, 1), 0, n] = d[2]
                            mma_params.acc_O[(1, 1), 0, n] = d[3]
        else:
            tOsV_stage = smem_copy_params.tOsVt[
                None, None, None,
                smem_pipe_read if const_expr(self.num_stages > 1) else 0,
            ]
            sm80_utils.gemm_rs(
                mma_params.thr_mma_pv,
                mma_params.acc_O,
                tOrP,
                mma_params.tOrVt,
                tOsV_stage,
                smem_copy_params.smem_thr_copy_V,
            )
        # if const_expr(self.num_stages > 1):
        #     load_K_next()
    @cute.jit
    def apply_score_mod(
        self,
        thr_mma_qk,
        batch_idx,
        head_idx,
        m_block,
        acc_S,
        n_block,
        softmax_scale,
        seqlen,
        aux_data: AuxData = AuxData(),
        fastdiv_mods=None,
    ):
        # Prepare index tensor
        cS = cute.make_identity_tensor((self.tile_m, self.tile_n))
        cS = cute.domain_offset((m_block * self.tile_m, n_block * self.tile_n), cS)
        tScS = thr_mma_qk.partition_C(cS)

        apply_score_mod_inner(
            acc_S,
            tScS,
            self.score_mod,
            batch_idx,
            head_idx,
            softmax_scale,
            self.score_vec_size,
            self.qk_acc_dtype,
            aux_data,
            fastdiv_mods,
            seqlen_info=seqlen,
            constant_q_idx=None,
            qhead_per_kvhead=self.qhead_per_kvhead if const_expr(self.pack_gqa) else 1,
        )


# SM90 forward pass moved to flash_fwd_sm90.py; re-export for backward compatibility

# =====================================================================================
# ATREX PORT: SM120 (Blackwell GeForce) forward. Same SM80-era MMA, 99 KB SMEM.
# Subclasses the in-file FlashAttentionForwardSm80 (vendored above) and overrides the SMEM
# capacity check. Inheriting this implementation keeps the CpAsync code paths (no TMA-O on sm120).
# =====================================================================================

class FlashAttentionForwardSm120(FlashAttentionForwardSm80):
    @staticmethod
    def can_implement(
        dtype,
        head_dim,
        head_dim_v,
        tile_m,
        tile_n,
        num_stages,
        num_threads,
        is_causal,
        Q_in_regs=False,
    ) -> bool:
        """Check if the kernel can be implemented on SM120.

        Same logic as SM80 but uses SM120's shared memory capacity (99 KB).
        """
        if dtype not in [
            cutlass.Float16,
            cutlass.BFloat16,
            cutlass.Float8E4M3FN,
            cutlass.Float8E5M2,
        ]:
            return False
        if head_dim % 8 != 0:
            return False
        if head_dim_v % 8 != 0:
            return False
        if tile_n % 16 != 0:
            return False
        if num_threads % 32 != 0:
            return False
        # Shared memory usage: Q + K + V + FP8 probability staging.
        input_bytes = dtype.width // 8
        if dtype.width == 8:
            smem_usage_Q = tile_m * (head_dim + 16)
            smem_usage_K = tile_n * (head_dim + 16) * num_stages
            smem_usage_V = tile_n * head_dim_v * num_stages
            smem_usage_P = (
                tile_m * (tile_n + 16) if head_dim != 128 or is_causal else 1
            )
        else:
            smem_usage_Q = tile_m * head_dim * input_bytes
            smem_usage_K = tile_n * head_dim * num_stages * input_bytes
            smem_usage_V = tile_n * head_dim_v * num_stages * input_bytes
            smem_usage_P = 0
        smem_usage_QV = (
            (smem_usage_Q + smem_usage_V) if not Q_in_regs else max(smem_usage_Q, smem_usage_V)
        )
        smem_usage = smem_usage_QV + smem_usage_K + smem_usage_P
        # SM120 has 99 KB shared memory (vs 163 KB on SM80)
        smem_capacity = utils_basic.get_smem_capacity_in_bytes("sm_120")
        if smem_usage > smem_capacity:
            return False
        compact_short_q = is_causal and (tile_m, tile_n, num_threads) in (
            (16, 32, 64),
            (32, 32, 128),
        )
        if not compact_short_q and (tile_m * 2) % num_threads != 0:
            return False
        return True

def __getattr__(name):
    if name == "FlashAttentionForwardSm90":
        from flash_attn.cute.flash_fwd_sm90 import FlashAttentionForwardSm90
        return FlashAttentionForwardSm90
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

# HD256 FP8 TMA prefill. The existing HD128/BF16/decode implementation above
# is unchanged. NVIDIA-derived kernel body: BSD-3-Clause; ATREX wrappers:
# Apache-2.0. See NOTICE for the original source revision and license terms.
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# Copyright 2026 Alibaba Group
from enum import Enum as _tma_Enum
from cutlass import Boolean as _tma_Boolean
from cutlass._mlir.dialects import nvvm as _tma_nvvm
from cutlass.cute.arch.nvvm_wrappers import shuffle_sync_op as _tma_shuffle_sync_op

# prefill_tma_primitives.py — Copyright 2026 Alibaba Group; SPDX-License-Identifier: Apache-2.0
class _tma_Shfl(_tma_Enum):
    IDX = "idx"
    UP = "up"
    DOWN = "down"
    BFLY = "bfly"


class _tma_MMALayout(_tma_Enum):
    ROW = "row"
    COL = "col"


class _tma_SetMaxRegisterAction(_tma_Enum):
    INCREASE = "increase"
    DECREASE = "decrease"


@dsl_user_op
def _tma_shfl_sync(thread_mask, val, offset, mask_and_clamp=0x1F,
              kind=_tma_Shfl.IDX, *, loc=None, ip=None):
    return _tma_shuffle_sync_op(
        val, offset, mask=thread_mask, mask_and_clamp=mask_and_clamp,
        kind=getattr(_tma_nvvm.ShflKind, kind.value), loc=loc, ip=ip,
    )


@dsl_user_op
def _tma_elect_sync(*, loc=None, ip=None):
    return _tma_Boolean(_tma_nvvm.elect_sync(loc=loc, ip=ip))


@dsl_user_op
def _tma_setmaxregister(reg_count: cutlass.Constexpr,
                   action: cutlass.Constexpr, *, loc=None, ip=None):
    if action is _tma_SetMaxRegisterAction.INCREASE:
        cute.arch.setmaxregister_increase(reg_count, loc=loc, ip=ip)
    elif action is _tma_SetMaxRegisterAction.DECREASE:
        cute.arch.setmaxregister_decrease(reg_count, loc=loc, ip=ip)
    else:
        raise ValueError(f"unsupported setmaxregister action: {action}")


@dsl_user_op
def _tma_mbarrier_init(ptr, count, *, loc=None, ip=None):
    cute.arch.mbarrier_init(ptr, count, loc=loc, ip=ip)


@dsl_user_op
def _tma_mbarrier_arrive(ptr, count=1, *, loc=None, ip=None):
    # count is an arrival decrement, not a peer CTA rank.
    cute.arch.mbarrier_arrive(ptr, arrive_count=count, loc=loc, ip=ip)


@dsl_user_op
def _tma_mbarrier_arrive_expect_tx(ptr, tx_count, *, loc=None, ip=None):
    cute.arch.mbarrier_arrive_and_expect_tx(ptr, tx_count, loc=loc, ip=ip)


@dsl_user_op
def _tma_fence_mbarrier_init(*, loc=None, ip=None):
    cute.arch.mbarrier_init_fence(loc=loc, ip=ip)


@dsl_user_op
def _tma_barrier_cta_sync(barrier_id=0, thread_count=None, *, loc=None, ip=None):
    cute.arch.barrier(
        barrier_id=barrier_id, number_of_threads=thread_count, loc=loc, ip=ip,
    )


def _tma__shared_address(ptr, *, loc=None, ip=None):
    return Int32(ptr.toint(loc=loc, ip=ip)).ir_value(loc=loc, ip=ip)


@dsl_user_op
def _tma_mbarrier_try_wait_parity(ptr, phase, time_limit=None, *, loc=None, ip=None):
    if time_limit is None:
        return cute.arch.mbarrier_try_wait(ptr, phase, loc=loc, ip=ip)
    # PTX timeHint is an unsigned32-bit nanosecond suspension hint, not cycles.
    result = llvm.inline_asm(
        T.i32(),
        [_tma__shared_address(ptr, loc=loc, ip=ip),
         Int32(phase).ir_value(loc=loc, ip=ip),
         Int32(time_limit).ir_value(loc=loc, ip=ip)],
        "{ .reg .pred done; "
        "mbarrier.try_wait.parity.acquire.cta.shared::cta.b64 done, [$1], $2, $3; "
        "selp.u32 $0, 1, 0, done; }",
        "=r,r,r,r",
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    return _tma_Boolean(Int32(result))


def _tma_ldmatrix_instruction(nregs, layout, shape="m8n8", src_format="b16"):
    """Validate the supported subset and return its exact PTX instruction."""
    trans = ".trans" if layout is _tma_MMALayout.COL else ""
    if shape == "m8n8" and src_format == "b16" and nregs in (1, 2, 4):
        matrices = nregs
    elif shape == "m16n16" and src_format == "b8" and nregs in (2, 4):
        if layout is not _tma_MMALayout.COL:
            raise ValueError("m16n16.b8 requires the transpose layout")
        matrices = nregs // 2
    else:
        raise ValueError(f"unsupported ldmatrix combination: {nregs}, {layout}, {shape}, {src_format}")
    return f"ldmatrix.sync.aligned.{shape}.x{matrices}{trans}.shared::cta.{src_format}"


@dsl_user_op
def _tma_ldmatrix(ptr, nregs: cutlass.Constexpr, layout=_tma_MMALayout.ROW,
             *, shape="m8n8", src_format="b16", loc=None, ip=None):
    instruction = _tma_ldmatrix_instruction(nregs, layout, shape, src_format)
    outputs = ", ".join(f"${i}" for i in range(nregs))
    result_type = (T.i32() if nregs == 1 else
                   ir.Type.parse("!llvm.struct<(" + ",".join(["i32"] * nregs) + ")>"))
    result = llvm.inline_asm(
        result_type, [_tma__shared_address(ptr, loc=loc, ip=ip)],
        instruction + " {" + outputs + "}, [$" + str(nregs) + "];",
        ",".join(["=r"] * nregs + ["r"]),
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    if nregs == 1:
        return (Int32(result),)
    return tuple(Int32(llvm.extractvalue(T.i32(), result, [i], loc=loc, ip=ip))
                 for i in range(nregs))


@dsl_user_op
def _tma_stmatrix(ptr, values, layout=_tma_MMALayout.ROW, *, loc=None, ip=None):
    nregs = len(values)
    if nregs not in (1, 2, 4):
        raise ValueError(f"unsupported stmatrix register count: {nregs}")
    trans = ".trans" if layout is _tma_MMALayout.COL else ""
    inputs = ", ".join(f"${i + 1}" for i in range(nregs))
    llvm.inline_asm(
        ir.Type.parse("!llvm.void"),
        [_tma__shared_address(ptr, loc=loc, ip=ip)] +
        [Int32(value).ir_value(loc=loc, ip=ip) for value in values],
        f"stmatrix.sync.aligned.m8n8.x{nregs}{trans}.shared::cta.b16 [$0], "
        "{" + inputs + "};",
        ",".join(["r"] * (nregs + 1)),
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )


@dsl_user_op
def _tma_load_ext(ptr, *, dtype=cutlass.Int32, l2_cache_hint=None, loc=None, ip=None):
    # The attention call site only requires one packed32-bit FP8 word.
    if dtype is not cutlass.Int32:
        raise ValueError(f"load_ext subset supports Int32 only, got {dtype}")
    args = [Int64(ptr.toint(loc=loc, ip=ip)).ir_value(loc=loc, ip=ip)]
    if l2_cache_hint is None:
        ptx, constraints = "ld.global.b32 $0, [$1];", "=r,l"
    else:
        args.append(Int64(l2_cache_hint).ir_value(loc=loc, ip=ip))
        ptx, constraints = "ld.global.L2::cache_hint.b32 $0, [$1], $2;", "=r,l,l"
    result = llvm.inline_asm(
        T.i32(), args, ptx, constraints,
        has_side_effects=True, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT, loc=loc, ip=ip,
    )
    return Int32(result)


# prefill_tma_copy.py — Copyright 2026 Alibaba Group; SPDX-License-Identifier: Apache-2.0
@cute.jit
def _tma_tma_smem_layout462(d_chunk: cutlass.Constexpr[int],
                        rows: cutlass.Constexpr[int],
                        paged: cutlass.Constexpr[bool]):
    base = cute.make_layout((d_chunk, rows, 1, 1), stride=(1, d_chunk, 0, 0))
    if cutlass.const_expr(d_chunk == 128):
        return cute.make_composed_layout(cute.make_swizzle(3, 4, 3), 0, base)
    elif cutlass.const_expr(d_chunk == 64):
        return cute.make_composed_layout(cute.make_swizzle(2, 4, 3), 0, base)
    else:
        return base


@cute.jit
def _tma_make_tma462(x: cute.Tensor, d_chunk: cutlass.Constexpr[int],
                rows: cutlass.Constexpr[int], paged: cutlass.Constexpr[bool]):
    # Reorder view metadata only; preserve NHD or interleaved vLLM storage.
    view = cute.make_tensor(x.iterator, cute.select(x.layout, mode=[3, 1, 2, 0]))
    tiler = (d_chunk, rows, 1, 1)
    return cpasync.make_tiled_tma_atom(
        cpasync.CopyBulkTensorTileG2SOp(), view,
        _tma_tma_smem_layout462(d_chunk, rows, paged), tiler,
    )


@cute.jit
def _tma_copy_tma462(s_dst: cute.Pointer, atom: cute.CopyAtom, tma_tensor: cute.Tensor,
                coord: tuple, mbar: cute.Pointer,
                d_chunk: cutlass.Constexpr[int], rows: cutlass.Constexpr[int],
                paged: cutlass.Constexpr[bool], cache_policy: cutlass.Int64 | None):
    tiler = (d_chunk, rows, 1, 1)
    # Original raw descriptor coords were (D, H, S, P).
    tile_coord = (coord[0] // d_chunk, coord[2] // rows, coord[1], coord[3])
    g_tile = cute.local_tile(tma_tensor, tiler, tile_coord)
    s_tile = cute.make_tensor(s_dst, _tma_tma_smem_layout462(d_chunk, rows, paged))
    # Group the four logical modes so partition forms one rectangular box.
    rank = cute.rank(s_tile)
    tS, tG = cpasync.tma_partition(
        atom, 0, cute.make_layout(1),
        cute.group_modes(s_tile, 0, rank), cute.group_modes(g_tile, 0, rank),
    )
    # CUTLASS 4.6.2 automatically elects one lane for a TMA copy.
    # Caller MUST NOT wrap this call in a lane-level elect_sync branch.
    cute.copy(atom, tG, tS, tma_bar_ptr=mbar, cache_policy=cache_policy)


# prefill_tma_helpers.py — Copyright 2026 Alibaba Group; SPDX-License-Identifier: Apache-2.0
@cute.jit
def _tma_atrex_row_reduce(value: cutlass.Float32, maximum: cutlass.Constexpr[bool], full: cutlass.Constexpr[bool]):
    clamp = 0x1F if cutlass.const_expr(full) else 0x1C03
    for step in cutlass.range_constexpr(2):
        other = _tma_shfl_sync(thread_mask=0xFFFFFFFF, val=value,
                                offset=2 >> step, mask_and_clamp=clamp,
                                kind=_tma_Shfl.BFLY)
        if cutlass.const_expr(maximum):
            value = cute.arch.fmax(value, other)
        else:
            value = value + other
    return value


@cute.jit
def _tma_atrex_row_max(value: cutlass.Float32):
    return _tma_atrex_row_reduce(value, True, False)


@cute.jit
def _tma_atrex_row_sum(value: cutlass.Float32):
    return _tma_atrex_row_reduce(value, False, False)


@cute.jit
def _tma_atrex_mma_e4m3_f32(
    a0: cutlass.Int32, a1: cutlass.Int32, a2: cutlass.Int32, a3: cutlass.Int32,
    b0: cutlass.Int32, b1: cutlass.Int32,
    c0: cutlass.Float32, c1: cutlass.Float32, c2: cutlass.Float32, c3: cutlass.Float32,
    ab_dtype: cutlass.Constexpr[Type[cutlass.Numeric]],
):
    if cutlass.const_expr(ab_dtype != cutlass.Float8E4M3FN):
        raise TypeError('The FP8 prefill kernel supports E4M3FN only')
    # Four packed A words and two packed B words describe m16n8k32 E4M3.
    # The result and accumulator remain FP32; no f16/bf16 MMA is used.
    return cute.arch.inline_ptx(
        'mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 '
        '{$0,$1,$2,$3}, {$8,$9,$10,$11}, {$12,$13}, {$4,$5,$6,$7};',
        write_only_types=[cutlass.Float32] * 4,
        read_only_args=[c0,c1,c2,c3,a0,a1,a2,a3,b0,b1],
    )


@cute.jit
def _tma_atrex_smem_column(row: cutlass.Int32, col: cutlass.Int32,
                     row_stride: cutlass.Constexpr[int], elem_bytes: cutlass.Constexpr[int]):
    # Each physical row permutes 16-byte chunks within a TMA swizzle segment.
    pitch_bytes = row_stride * elem_bytes
    if cutlass.const_expr(pitch_bytes % 128 == 0):
        segment_bytes, row_bits, row_shift = 128, 3, 0
    elif cutlass.const_expr(pitch_bytes % 64 == 0):
        segment_bytes, row_bits, row_shift = 64, 2, 1
    else:
        segment_bytes, row_bits, row_shift = 32, 1, 0
    segment_elems = segment_bytes // elem_bytes
    segment_start = (col // segment_elems) * segment_elems
    xor_elems = ((row >> row_shift) & ((1 << row_bits) - 1)) * (16 // elem_bytes)
    return segment_start + ((col - segment_start) ^ xor_elems)


@cute.jit
def _tma_atrex_pack_word(src: tuple, dtype: cutlass.Constexpr[Type[cutlass.Numeric]]):
    lanes = cutlass.Vector.from_elements(src, dtype=dtype)
    packed = lanes.bitcast(cutlass.Int32)
    return packed[0]


def _tma_atrex_ceildiv(n, divisor):
    return (n + divisor - 1) // divisor


@cute.jit
def _tma_atrex_encode_fp8_pair(upper: cutlass.Float32, lower: cutlass.Float32,
                         dtype: cutlass.Constexpr[Type[cutlass.Numeric]]):
    if cutlass.const_expr(dtype != cutlass.Float8E4M3FN):
        raise TypeError('E4M3FN probability format required')
    return cute.arch.inline_ptx(
        'cvt.rn.satfinite.e4m3x2.f32 $0, $1, $2;',
        write_only_types=[cutlass.Uint16], read_only_args=[upper, lower],
    )


@cute.jit
def _tma_atrex_join_fp8_pairs(low: cutlass.Uint16, high: cutlass.Uint16):
    return cute.arch.inline_ptx('mov.b32 $0, {$1,$2};',
                               write_only_types=[cutlass.Int32], read_only_args=[low,high])


_tma_atrex_pair_add = partial(cute.arch.add_packed_f32x2, ftz=False, rnd='rn')
_tma_atrex_pair_mul = partial(cute.arch.mul_packed_f32x2, ftz=False, rnd='rn')
_tma_atrex_pair_fma = partial(cute.arch.fma_packed_f32x2, ftz=False, rnd='rn')


@cute.jit
def _tma_atrex_mask_v_word(word: cutlass.Int32, first_k: cutlass.Int32, seqlen_k: cutlass.Int32):
    """Zero invalid FP8 bytes before MMA, including NaN page padding."""
    valid_bytes = seqlen_k - first_k
    keep = cutlass.Int32(-1)
    if valid_bytes < 4:
        keep = cutlass.Int32(0)
        if valid_bytes > 0:
            keep = (cutlass.Int32(1) << (valid_bytes * 8)) - cutlass.Int32(1)
    return word & keep


# prefill_tma_cutedsl.py — NVIDIA BSD-3-Clause kernel body, adapted from FlashInfer e44067b1

class _tma_FlashAttentionForwardSm120Fp8Tma:
    """Private causal E4M3 D256/Q16..128/KV64/128 paged specialization.

    The constructor fixes paged storage, distributed row state and three K/V
    slots. N64/Q64 requests cap232 and a two-block launch hint; N128 retains
    cap240 and a one-block hint. Dense execution is enforced by ``__call__``.
    Private call signatures remain compatible with the runtime adapter.
    """

    SEQ_Q_TILES = [128, 64, 32, 16]
    SEQ_KV_TILES = [128, 64]
    MMA_TILER = (16, 8, 32)  # mma.sync.aligned.m16n8k32
    # Barrier 0 is the CTA-wide initialization barrier; barrier 1 synchronizes
    # compute warps before the K/V storage is aliased for the output epilogue.
    COMPUTE_BARRIER_ID = 1
    # K and V share a three-slot circular TMA pipeline. D256/N128 uses
    # 96 KiB; N64 uses 48 KiB before the BF16 output alias is considered.
    KV_PIPELINE_STAGES = 3

    def __init__(
        self,
        in_dtype: Type[cutlass.Numeric] = cutlass.Float8E4M3FN,
        out_dtype: Type[cutlass.Numeric] = cutlass.BFloat16,
        is_causal: bool = True,
        head_tile: int = 256,
        kv_tile: int = SEQ_KV_TILES[0],
        q_tile: int = SEQ_Q_TILES[0],
        use_paged_kv: bool = True,
        num_tokens_per_page: int = 64,
        balanced_scheduler: bool = True,
        pack_gqa: bool = False,
        num_splits: int = 1,
    ) -> None:
        """Configure the fixed private specialization selected by ATREX."""
        if (
            in_dtype != cutlass.Float8E4M3FN
            or num_splits not in (1, 2, 3, 4, 8, 16, 32)
            or out_dtype != (cutlass.BFloat16 if num_splits == 1 else cutlass.Float32)
            or is_causal is not True
            or head_tile != 256
            or q_tile not in self.SEQ_Q_TILES
            or kv_tile not in (64, 128)
            or use_paged_kv is not True
            or balanced_scheduler is not True
            or num_tokens_per_page not in (64, 128)
        ):
            raise ValueError(
                "SM120 TMA prefill requires causal paged E4M3/BF16, "
                "D256/Q16/32/64/128/KV64/128, balanced scheduling and page64/page128"
            )
        self.in_dtype = in_dtype
        self.out_dtype = out_dtype
        self.is_causal = is_causal
        self.use_paged_kv = use_paged_kv
        self.balanced_scheduler = balanced_scheduler
        self.pack_gqa = pack_gqa
        self.num_splits = num_splits
        # All causal paths use bottom-right alignment, so Q is positioned at
        # the end of its logical K/V sequence when their lengths differ.
        self.has_causal_q_offset = self.is_causal

        self.head_tile = head_tile
        self.q_tile = q_tile
        self.kv_tile = kv_tile
        self.kv_pipeline_stages = 3
        self.num_tokens_per_page = num_tokens_per_page
        # The block-table width is a runtime tensor dimension. Do not specialize
        # on it merely to cache one row in a warp; direct loads keep one compiled
        # artifact valid for every page-table capacity.
        self.cache_page_ids = False

        # Q128/Q64 keep separate compute/load warpgroups and register donation.
        # Q16/Q32 fit compute, load and padding warps in one warpgroup; all
        # setmaxregister operations are disabled there to keep it converged.
        self.compute_warp_ids: tuple[int, ...]
        if self.q_tile == 128:
            self.compute_warp_ids = (0, 1, 2, 3, 4, 5, 6, 7)
            self.load_warp_id = 8
            self.empty_warp_ids = (9, 10, 11)
            self.num_warps = 12
        elif self.q_tile == 64:
            self.compute_warp_ids = (0, 1, 2, 3)
            self.load_warp_id = 4
            self.empty_warp_ids = (5, 6, 7)
            self.num_warps = 8
        else:
            self.compute_warp_ids = tuple(range(self.q_tile // 16))
            self.load_warp_id = len(self.compute_warp_ids)
            self.empty_warp_ids = tuple(range(self.load_warp_id + 1, 4))
            self.num_warps = 4
        self.num_compute_warps = len(self.compute_warp_ids)
        self.threads_per_cta = cute.arch.WARP_SIZE * self.num_warps
        self.threads_compute = cute.arch.WARP_SIZE * self.num_compute_warps

        self._setup_attributes()

    def _setup_attributes(self) -> None:
        """Compute derived tile, MMA, and TMA constants from the configuration."""

        self.qo_tile_elems = self.q_tile * self.head_tile
        self.kv_tile_elems = self.kv_tile * self.head_tile
        if self.num_splits > 1:
            # Float32 partials store directly from MMA registers, without sO.
            self.smem_elems = self.kv_tile_elems * self.kv_pipeline_stages
        else:
            self.smem_elems = _tma_atrex_ceildiv(
                max(
                    self.kv_tile_elems * self.kv_pipeline_stages * self.in_dtype.bytes,
                    self.qo_tile_elems * self.out_dtype.bytes,
                ),
                self.in_dtype.bytes,
            )

        # MMA
        self.qk_k_frags = self.kv_tile // self.MMA_TILER[1]
        self.qk_d_frags = self.head_tile // self.MMA_TILER[2]
        self.pv_v_frags = self.kv_tile // self.MMA_TILER[2]
        self.pv_d_frags = self.head_tile // self.MMA_TILER[1]

        # TMA
        tma_copy_head_bytes = self.head_tile * self.in_dtype.bytes
        self.tma_swizzle = 128
        self.tma_copy_iters = tma_copy_head_bytes // 128

        self.tma_copy_head_per_iter = self.head_tile // self.tma_copy_iters
        self.tma_copy_elems_per_iter = self.kv_tile_elems // self.tma_copy_iters
        self.tma_copy_rows = min(self.kv_tile, self.num_tokens_per_page)
        self.tma_copy_elems_per_page_iter = (
            self.tma_copy_rows * self.tma_copy_head_per_iter
        )
        self.kv_page_chunks = _tma_atrex_ceildiv(self.kv_tile, self.num_tokens_per_page)

    @cute.jit
    def _get_swizzled_col(
        self, row: cutlass.Int32, col: cutlass.Int32
    ) -> cutlass.Int32:
        """Return the SMEM column for K/V tiles loaded by this kernel's TMA layout."""
        smem_col = col
        smem_col = _tma_atrex_smem_column(
            row,
            col,
            self.tma_copy_head_per_iter,
            self.in_dtype.bytes,
        )
        return smem_col

    @cute.jit
    def get_kv_stage_ptr(
        self, sKV: cute.Pointer, stage: cutlass.Int32
    ) -> cutlass.Pointer:
        """Select one of the three K/V ring slots without a dynamic GEP."""
        ptr = sKV
        if stage == 1:
            ptr = (sKV + self.kv_tile_elems)
        elif stage == 2:
            ptr = (sKV + 2 * self.kv_tile_elems)
        return ptr

    @cute.jit
    def get_mbar_stage_ptr(
        self, mbars: cute.Pointer, stage: cutlass.Int32
    ) -> cutlass.Pointer:
        """Select one of three ring mbarriers without a dynamic GEP."""
        ptr = mbars
        if stage == 1:
            ptr = (mbars + 1)
        elif stage == 2:
            ptr = (mbars + 2)
        return ptr


    @cute.jit
    def resolve_paged_kv_tile_pages(
        self,
        batch_idx: cutlass.Int32,
        seq_coord: cutlass.Int32,
        seqlen_kv: cutlass.Int32,
        block_tables: cute.Tensor,
        cached_page_id: cutlass.Int32,
    ) -> tuple:
        """Resolve shared K/V physical pages for one logical K/V tile."""
        page_ids = cute.make_rmem_tensor(cute.make_layout(self.kv_page_chunks), cutlass.Int32)
        block_tables_ptr = cutlass.Pointer(block_tables.iterator, dtype=block_tables.dtype)
        block_table_offset = batch_idx * cutlass.Int32(block_tables.shape[1])
        base_logical_page = seq_coord // self.num_tokens_per_page
        safe_page_count = cute.math.max(
            (seqlen_kv + self.num_tokens_per_page - 1) // self.num_tokens_per_page,
            cutlass.Int32(1),
        )

        for seq_chunk in cutlass.range_constexpr(self.kv_page_chunks):
            logical_page = base_logical_page + cutlass.Int32(seq_chunk)
            if logical_page >= safe_page_count:
                # TMA still needs a legal full-page address for a partial
                # tile. K tail values are masked before softmax, so clamping
                # cannot make them logically valid. Paged pools, including
                # unused slots, can contain NaNs: V tail masking discards invalid tokens.
                logical_page = safe_page_count - cutlass.Int32(1)
            page_ids[seq_chunk] = cutlass.Int32(
                block_tables_ptr[block_table_offset + logical_page]
            )

        return page_ids

    @cute.jit
    def load_one_paged_kv_tile(
        self,
        s_dst: cutlass.Pointer,
        tma_desc_ptr: tuple,
        mbar_arrived: cute.Pointer,
        page_ids: cute.Tensor,
        kv_head_idx: cutlass.Int32,
        l2_cache_hint: cutlass.Int64 | None,
        seq_coord: cutlass.Int32 = cutlass.Int32(0),
    ) -> None:
        """Launch TMA loads for one logical K/V tile from pre-resolved pages.

        The logical paged view is
        ``(num_pages, num_tokens_per_page, Hkv, D)``; its TMA descriptor
        retains the physical NHD or interleaved vLLM strides. The
        destination SMEM layout matches the contiguous path:
        ``[D chunk][KV row][D within chunk]``.
        """
        if _tma_elect_sync():
            _tma_mbarrier_arrive_expect_tx(
                mbar_arrived, self.kv_tile_elems * self.in_dtype.bytes
            )

        token_in_page = cutlass.Int32(0)
        if cutlass.const_expr(self.kv_tile < self.num_tokens_per_page):
            # N64/page128 alternates between the two half-page coordinates.
            token_in_page = seq_coord % self.num_tokens_per_page
        for issue_chunk in cutlass.range_constexpr(self.kv_page_chunks):
            # Preserve the tuned reverse TMA issue order, while deriving page_dst
            # from the logical chunk so MMA still sees left-to-right sequence rows.
            seq_chunk = self.kv_page_chunks - 1 - issue_chunk
            page_id = page_ids[seq_chunk]
            page_dst = seq_chunk * self.tma_copy_elems_per_page_iter

            for i in cutlass.range_constexpr(self.tma_copy_iters):
                head_offset = i * self.tma_copy_head_per_iter
                _tma_copy_tma462(s_dst + i * self.tma_copy_elems_per_iter + page_dst, tma_desc_ptr[0], tma_desc_ptr[1], (
                            head_offset,
                            kv_head_idx,
                            token_in_page,
                            page_id,
                        ), mbar_arrived, self.tma_copy_head_per_iter, self.tma_copy_rows, self.use_paged_kv, l2_cache_hint)

    @cute.jit
    def load_q_tile(
        self,
        basic_params: SimpleNamespace,
    ) -> cute.Tensor:
        """Load the warp-owned Q tile from GMEM into MMA A registers.

        :param basic_params: Per-CTA tensor metadata, lane mapping, and Q base
            offsets.
        :return: Packed Q fragments arranged for ``mma.sync`` A operands.
        """

        # Each lane loads a contiguous 4-element vector. The shuffles below
        # transpose the four row quads into the m16n8k32 A-fragment layout.
        def load_q_fragment(col0_in_cta: cutlass.Int32):
            q_regs_per_frag = [
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
                cutlass.Int32(0),
            ]
            for row_quad in cutlass.range_constexpr(self.MMA_TILER[0] // 4):
                row_in_cta = (
                    basic_params.q_warp_row0 + row_quad * 4 + basic_params.lane_div8
                )
                col_in_cta = col0_in_cta + basic_params.lane_mod8 * 4
                cur_q_seq_idx = basic_params.q_seq_idx + row_in_cta
                if (
                    cur_q_seq_idx < basic_params.seqlen_q
                    and col_in_cta < basic_params.head_dim
                ):
                    if cutlass.const_expr(self.pack_gqa):
                        q_src = (
                            basic_params.q_ptr
                            + basic_params.q_head_base
                            + (cur_q_seq_idx // basic_params.q_heads_per_kv_head)
                            * basic_params.q_seq_stride
                            + (cur_q_seq_idx % basic_params.q_heads_per_kv_head)
                            * basic_params.head_dim
                            + col_in_cta
                        )
                    else:
                        q_src = (
                            basic_params.q_ptr
                            + basic_params.q_head_base
                            + cur_q_seq_idx * basic_params.q_seq_stride
                            + col_in_cta
                        )
                    q_packed = _tma_load_ext(
                        q_src,
                        dtype=cutlass.Int32,
                        l2_cache_hint=basic_params.q_l2_cache_hint,
                    )
                    q_regs_per_frag[row_quad] = q_packed
            return q_regs_per_frag

        q_regs = cute.make_rmem_tensor(cute.make_layout(self.qk_d_frags * 4), cutlass.Int32)
        src_lane_lo = (basic_params.lane_div4 % 4) * 8 + basic_params.lane_mod4
        src_lane_hi = (basic_params.lane_div4 % 4) * 8 + (basic_params.lane_mod4 + 4)

        q_regs_offset = 0
        col0_in_cta = 0
        for _ in cutlass.range_constexpr(self.qk_d_frags):
            q_rows = load_q_fragment(col0_in_cta)
            q_rows_lo = [
                _tma_shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=q_rows[0],
                    offset=src_lane_lo,
                    mask_and_clamp=0x1F,
                    kind=_tma_Shfl.IDX,
                ),
                _tma_shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=q_rows[1],
                    offset=src_lane_lo,
                    mask_and_clamp=0x1F,
                    kind=_tma_Shfl.IDX,
                ),
                _tma_shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=q_rows[2],
                    offset=src_lane_lo,
                    mask_and_clamp=0x1F,
                    kind=_tma_Shfl.IDX,
                ),
                _tma_shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=q_rows[3],
                    offset=src_lane_lo,
                    mask_and_clamp=0x1F,
                    kind=_tma_Shfl.IDX,
                ),
            ]
            q_rows_hi = [
                _tma_shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=q_rows[0],
                    offset=src_lane_hi,
                    mask_and_clamp=0x1F,
                    kind=_tma_Shfl.IDX,
                ),
                _tma_shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=q_rows[1],
                    offset=src_lane_hi,
                    mask_and_clamp=0x1F,
                    kind=_tma_Shfl.IDX,
                ),
                _tma_shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=q_rows[2],
                    offset=src_lane_hi,
                    mask_and_clamp=0x1F,
                    kind=_tma_Shfl.IDX,
                ),
                _tma_shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=q_rows[3],
                    offset=src_lane_hi,
                    mask_and_clamp=0x1F,
                    kind=_tma_Shfl.IDX,
                ),
            ]

            if basic_params.lane_div4 < 4:
                q_regs[q_regs_offset + 0] = q_rows_lo[0]
                q_regs[q_regs_offset + 1] = q_rows_lo[2]
                q_regs[q_regs_offset + 2] = q_rows_hi[0]
                q_regs[q_regs_offset + 3] = q_rows_hi[2]
            else:
                q_regs[q_regs_offset + 0] = q_rows_lo[1]
                q_regs[q_regs_offset + 1] = q_rows_lo[3]
                q_regs[q_regs_offset + 2] = q_rows_hi[1]
                q_regs[q_regs_offset + 3] = q_rows_hi[3]

            col0_in_cta += self.MMA_TILER[2]
            q_regs_offset += 4

        return q_regs

    @cute.jit
    def mma_qk(
        self,
        basic_params: SimpleNamespace,
        mma_params: SimpleNamespace,
        q_regs: cute.Tensor,
        sK: cutlass.Pointer,
    ) -> cute.Tensor:
        """Compute ``S = Q @ K.T``.

        Q fragments are supplied in registers by ``load_q_tile``. K fragments
        are read from the TMA-populated ``sK`` tile with ``ldmatrix``.

        :param basic_params: Per-CTA tensor metadata and lane mapping.
        :param mma_params: Shared K/V ring and local O accumulator state.
        :param q_regs: Register-resident packed Q fragments.
        :param sK: Shared-memory pointer to this tile's K operand.
        :return: Register-resident QK score fragments.
        """
        s_regs = cute.make_rmem_tensor(cute.make_layout(self.qk_k_frags * 4), cutlass.Float32)
        for i in cutlass.range_constexpr(self.qk_k_frags * 4):
            s_regs[i] = cutlass.Float32(0.0)

        k_row_in_frag_pair = (basic_params.lane_div8 // 2) * 8 + basic_params.lane_mod8
        k_col_in_frag_pair = (basic_params.lane_div8 % 2) * 16

        def load_k_fragments(dst, d_offset: cutlass.Constexpr[int]):
            for k_pair in cutlass.range_constexpr(self.qk_k_frags // 2):
                k_row = k_pair * 16 + k_row_in_frag_pair
                k_col = d_offset + k_col_in_frag_pair
                k_chunk = k_col // self.tma_copy_head_per_iter
                k_col_in_chunk = k_col % self.tma_copy_head_per_iter
                sK_ptr = (
                    sK
                    + k_chunk * self.tma_copy_elems_per_iter
                    + k_row * self.tma_copy_head_per_iter
                    + self._get_swizzled_col(k_row, k_col_in_chunk)
                )
                k_vec = _tma_ldmatrix(sK_ptr, 4, _tma_MMALayout.ROW)
                dst[k_pair * 2] = (k_vec[0], k_vec[1])
                dst[k_pair * 2 + 1] = (k_vec[2], k_vec[3])

        k_cur = [None] * self.qk_k_frags
        load_k_fragments(k_cur, 0)

        for d_frag in cutlass.range_constexpr(self.qk_d_frags):
            # The last wraparound load is discarded; identical unrolled
            # iterations keep the register-load/MMA pipeline branch-free.
            d_next = ((d_frag + 1) % self.qk_d_frags) * self.MMA_TILER[2]
            k_next = [None] * self.qk_k_frags
            load_k_fragments(k_next, d_next)
            q_off = d_frag * 4
            for k_frag in cutlass.range_constexpr(self.qk_k_frags):
                s_off = k_frag * 4
                array462_mma_673 = _tma_atrex_mma_e4m3_f32(
                    q_regs[q_off + 0],
                    q_regs[q_off + 1],
                    q_regs[q_off + 2],
                    q_regs[q_off + 3],
                    k_cur[k_frag][0],
                    k_cur[k_frag][1],
                    s_regs[s_off + 0],
                    s_regs[s_off + 1],
                    s_regs[s_off + 2],
                    s_regs[s_off + 3],
                    self.in_dtype,
                )
                s_regs[s_off + 0] = array462_mma_673[0]
                s_regs[s_off + 1] = array462_mma_673[1]
                s_regs[s_off + 2] = array462_mma_673[2]
                s_regs[s_off + 3] = array462_mma_673[3]
            k_cur = k_next

        return s_regs


    @cute.jit
    def online_softmax(
        self,
        basic_params: SimpleNamespace,
        mma_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        s_regs: cute.Tensor,
        row_state: cutlass.Float32,
        kv_seq_idx: cutlass.Int32,
        is_masked_frontier_tile: cutlass.Constexpr[bool],
        uses_softmax_init_path: cutlass.Constexpr[bool],
    ) -> tuple:
        """Online softmax and stage packed P in registers for the PV MMA.

        :param basic_params: Per-CTA tensor metadata and lane mapping.
        :param mma_params: Local output accumulator state to rescale.
        :param softmax_params: Row max/sum state and log2 softmax scale.
        :param s_regs: Register-resident QK score fragments.
        :param row_state: One persistent scalar per lane. Lanes 0/1 in each
            threadquad hold row maxima; lanes 2/3 hold row sums.
        :param kv_seq_idx: Logical row offset within the current K/V sequence.
        :param is_masked_frontier_tile: Marks the causal/K-tail frontier. It
            enables causal limits and explicit K/V-length bounds.
        :param uses_softmax_init_path: Select the cheaper update that bypasses
            merging prior state. The generic path is also correct for a first
            full tile because row state and O start at their neutral values.
        :return: Register-resident P fragments and updated distributed state.
        """
        o_regs = mma_params.o_regs
        softmax_scale_log2 = softmax_params.softmax_scale_log2

        p_regs = cute.make_rmem_tensor(cute.make_layout(self.qk_k_frags * 2), cutlass.Uint16)

        # Each lane owns four S registers split across two Q rows after Q@K^T.
        for row_half in cutlass.range_constexpr(2):
            s_reg_idx_lo = row_half * 2
            s_reg_idx_hi = row_half * 2 + 1

            q_row_in_cta = (
                basic_params.q_warp_row0 + basic_params.lane_div4 + row_half * 8
            )

            # Compute the valid K-column limit for the current Q row.
            valid_cols = basic_params.seqlen_k
            has_valid_cols = True
            if cutlass.const_expr(is_masked_frontier_tile):
                if cutlass.const_expr(self.pack_gqa):
                    valid_cols = cute.math.min(
                        (basic_params.q_seq_idx + q_row_in_cta)
                        // basic_params.q_heads_per_kv_head
                        + basic_params.causal_q_offset
                        + 1,
                        basic_params.seqlen_k,
                    )
                else:
                    valid_cols = cute.math.min(
                        basic_params.q_seq_idx
                        + q_row_in_cta
                        + basic_params.causal_q_offset
                        + 1,
                        basic_params.seqlen_k,
                    )
                has_valid_cols = kv_seq_idx < valid_cols

            # Reduce max across this lane's S values for the current Q row.
            cur_max0 = -cutlass.Float32.inf
            cur_max1 = -cutlass.Float32.inf
            for k_frag in cutlass.range_constexpr(self.qk_k_frags):
                s_off = k_frag * 4
                s0 = s_regs[s_off + s_reg_idx_lo]
                s1 = s_regs[s_off + s_reg_idx_hi]
                if cutlass.const_expr(is_masked_frontier_tile):
                    k_col0 = kv_seq_idx + k_frag * 8 + 2 * basic_params.lane_mod4
                    k_col1 = k_col0 + 1
                    valid0 = k_col0 < valid_cols if has_valid_cols else False
                    valid1 = k_col1 < valid_cols if has_valid_cols else False
                    if not valid0:
                        s0 = -cutlass.Float32.inf
                    if not valid1:
                        s1 = -cutlass.Float32.inf
                s_regs[s_off + s_reg_idx_lo] = s0
                s_regs[s_off + s_reg_idx_hi] = s1
                cur_max0 = cute.arch.fmax(cur_max0, s0)
                cur_max1 = cute.arch.fmax(cur_max1, s1)

            cur_max = cute.arch.fmax(cur_max0, cur_max1)
            row_max_prev = -cutlass.Float32.inf
            if cutlass.const_expr(not uses_softmax_init_path):
                row_max_prev = _tma_shfl_sync(
                    thread_mask=0xFFFFFFFF,
                    val=row_state,
                    offset=row_half,
                    mask_and_clamp=0x1C03,
                    kind=_tma_Shfl.IDX,
                )
            cur_max = _tma_atrex_row_max(cur_max)

            # Compute the correction in the max-holder lane and broadcast it
            # once to the other three lanes that own this logical row.
            old_scale = cutlass.Float32(1.0)
            if cutlass.const_expr(uses_softmax_init_path):
                new_max = cur_max
            else:
                new_max = cute.arch.fmax(row_max_prev, cur_max)
                old_scale_candidate = cute.math.exp2(
                    (row_max_prev - new_max) * softmax_scale_log2,
                    fastmath=True,
                )
                if cutlass.const_expr(is_masked_frontier_tile):
                    if new_max == -cutlass.Float32.inf:
                        old_scale_candidate = cutlass.Float32(1.0)
                old_scale = old_scale_candidate

            if cutlass.const_expr(not uses_softmax_init_path):
                for d_frag in cutlass.range_constexpr(self.pv_d_frags):
                    o_off = d_frag * 4 + row_half * 2
                    if old_scale < 1.0:
                        o_regs[o_off + 0], o_regs[o_off + 1] = _tma_atrex_pair_mul(
                            (o_regs[o_off + 0], o_regs[o_off + 1]),
                            (old_scale, old_scale),
                        )

            # Compute P, accumulate the per-lane partial sum, and stage P.
            exp_max = new_max
            if cutlass.const_expr(is_masked_frontier_tile):
                # A fully masked row has -inf max; use a finite subtraction
                # anchor to avoid -inf - -inf = NaN while preserving zero P.
                if exp_max == -cutlass.Float32.inf:
                    exp_max = cutlass.Float32(0.0)
            neg_exp_max_scaled = -(exp_max * softmax_scale_log2)
            tile_sum0 = cutlass.Float32(0.0)
            tile_sum1 = cutlass.Float32(0.0)
            p_even0 = cutlass.Float32(0.0)
            p_even1 = cutlass.Float32(0.0)
            for k_frag in cutlass.range_constexpr(self.qk_k_frags):
                s_off = k_frag * 4
                s0 = s_regs[s_off + s_reg_idx_lo]
                s1 = s_regs[s_off + s_reg_idx_hi]
                in0, in1 = _tma_atrex_pair_fma(
                    (s0, s1),
                    (softmax_scale_log2, softmax_scale_log2),
                    (neg_exp_max_scaled, neg_exp_max_scaled),
                )
                p0 = cute.math.exp2(in0, fastmath=True)
                p1 = cute.math.exp2(in1, fastmath=True)
                tile_sum0, tile_sum1 = _tma_atrex_pair_add(
                    (tile_sum0, tile_sum1),
                    (p0, p1),
                )
                p_regs[k_frag * 2 + row_half] = _tma_atrex_encode_fp8_pair(
                    p1, p0, self.in_dtype
                )

            # Reduce tile_sum across the four lanes that own one Q row.
            tile_sum = tile_sum0 + tile_sum1
            if basic_params.lane_mod4 == row_half + 2:
                tile_sum = tile_sum + row_state * old_scale
            tile_sum = _tma_atrex_row_sum(tile_sum)

            # Correct row_sum and rescale O when row_max changes.
            if basic_params.lane_mod4 == row_half:
                row_state = new_max
            if basic_params.lane_mod4 == row_half + 2:
                row_state = tile_sum

        return p_regs, row_state


    @cute.jit
    def mma_pv(
        self,
        basic_params: SimpleNamespace,
        mma_params: SimpleNamespace,
        p_regs: cute.Tensor,
        sV: cutlass.Pointer,
        kv_seq_idx: cutlass.Int32,
        mask_tail: cutlass.Constexpr[bool],
    ) -> None:
        """Compute ``O += P @ V``.

        P fragments are already packed in registers. V fragments are streamed
        from the TMA-populated ``sV`` tile with ``ldmatrix``.

        :param basic_params: Per-CTA tensor metadata and lane mapping.
        :param mma_params: Shared K/V ring and local output accumulator state.
        :param p_regs: Register-resident FP8 P fragments from softmax.
        :param sV: Shared-memory pointer to this tile's V operand.
        """
        o_regs = mma_params.o_regs

        def load_v_fragment_pair(
            v_seq_offset: cutlass.Constexpr[int],
            d_pair: cutlass.Constexpr[int],
        ):
            v_lane = basic_params.lane
            v_row = v_seq_offset + v_lane
            v_col = d_pair * 16
            v_chunk = v_col // self.tma_copy_head_per_iter
            v_col_in_chunk = v_col % self.tma_copy_head_per_iter
            sV_ptr = (
                sV
                + v_chunk * self.tma_copy_elems_per_iter
                + v_row * self.tma_copy_head_per_iter
                + self._get_swizzled_col(v_row, v_col_in_chunk)
            )
            v_vec = _tma_ldmatrix(
                sV_ptr,
                4,
                _tma_MMALayout.COL,
                shape="m16n16",
                src_format="b8",
            )
            w0, w1, w2, w3 = v_vec[0], v_vec[1], v_vec[2], v_vec[3]
            if cutlass.const_expr(mask_tail):
                if kv_seq_idx + self.kv_tile > basic_params.seqlen_k:
                    first_k = kv_seq_idx + v_seq_offset + basic_params.lane_mod4 * 4
                    w0 = _tma_atrex_mask_v_word(w0, first_k, basic_params.seqlen_k)
                    w1 = _tma_atrex_mask_v_word(w1, first_k, basic_params.seqlen_k)
                    w2 = _tma_atrex_mask_v_word(w2, first_k + 16, basic_params.seqlen_k)
                    w3 = _tma_atrex_mask_v_word(w3, first_k + 16, basic_params.seqlen_k)
            return (w0, w2), (w1, w3)

        def pack_p_cols(
            k_frag0: cutlass.Constexpr[int],
            row_half: cutlass.Constexpr[int],
        ) -> cutlass.Int32:

            packed_pairs = _tma_atrex_join_fp8_pairs(
                p_regs[k_frag0 * 2 + row_half],
                p_regs[(k_frag0 + 1) * 2 + row_half],
            )
            pair_lane = (basic_params.lane_mod4 % 2) * 2
            src0 = pair_lane
            src1 = src0 + 1
            lo = _tma_shfl_sync(
                thread_mask=0xFFFFFFFF,
                val=packed_pairs,
                offset=src0,
                mask_and_clamp=0x1C03,
                kind=_tma_Shfl.IDX,
            )
            hi = _tma_shfl_sync(
                thread_mask=0xFFFFFFFF,
                val=packed_pairs,
                offset=src1,
                mask_and_clamp=0x1C03,
                kind=_tma_Shfl.IDX,
            )
            selector = cutlass.Int32(0x5410) + (
                basic_params.lane_mod4 // 2
            ) * cutlass.Int32(0x2222)
            return cute.arch.inline_ptx(
                "prmt.b32 $0, $1, $2, $3;",
                write_only_types=[cutlass.Int32],
                read_only_args=[lo, hi, selector],
            )

        if cutlass.const_expr(
            self.head_tile == 256
            and self.q_tile == 64
            and self.kv_tile == 64
            and self.num_splits == 1
            and mask_tail
        ):
            # Keep only two D pairs of V live at a time in the masked frontier.
            # Each output fragment still accumulates v_frag 0 before v_frag 1.
            # The unmasked hot loop retains the existing prefetch schedule.
            p_fragments = [None] * self.pv_v_frags
            for v_frag in cutlass.range_constexpr(self.pv_v_frags):
                p_fragments[v_frag] = (
                    pack_p_cols(v_frag * 4 + 0, 0),
                    pack_p_cols(v_frag * 4 + 0, 1),
                    pack_p_cols(v_frag * 4 + 2, 0),
                    pack_p_cols(v_frag * 4 + 2, 1),
                )

            v_cur = [None] * 4
            for d_in_group in cutlass.range_constexpr(2):
                v_cur[d_in_group * 2], v_cur[d_in_group * 2 + 1] = load_v_fragment_pair(
                    0, d_in_group
                )
            for d_group in cutlass.range_constexpr(self.pv_d_frags // 4):
                for v_frag in cutlass.range_constexpr(self.pv_v_frags):
                    prefetch_next = cutlass.const_expr(
                        v_frag + 1 < self.pv_v_frags
                        or d_group + 1 < self.pv_d_frags // 4
                    )
                    if cutlass.const_expr(prefetch_next):
                        next_v_frag = (v_frag + 1) % self.pv_v_frags
                        next_d_group = d_group + (v_frag + 1) // self.pv_v_frags
                        v_next = [None] * 4
                    p_reg0, p_reg1, p_reg2, p_reg3 = p_fragments[v_frag]
                    for d_in_group in cutlass.range_constexpr(2):
                        if cutlass.const_expr(prefetch_next):
                            v_next[d_in_group * 2], v_next[d_in_group * 2 + 1] = load_v_fragment_pair(
                                next_v_frag * self.MMA_TILER[2],
                                next_d_group * 2 + d_in_group,
                            )
                        for d_in_pair in cutlass.range_constexpr(2):
                            d_frag = d_group * 4 + d_in_group * 2 + d_in_pair
                            v_index = d_in_group * 2 + d_in_pair
                            o_off = d_frag * 4
                            array462_mma_1356 = _tma_atrex_mma_e4m3_f32(
                                p_reg0,
                                p_reg1,
                                p_reg2,
                                p_reg3,
                                v_cur[v_index][0],
                                v_cur[v_index][1],
                                o_regs[o_off + 0],
                                o_regs[o_off + 1],
                                o_regs[o_off + 2],
                                o_regs[o_off + 3],
                                self.in_dtype,
                            )
                            o_regs[o_off + 0] = array462_mma_1356[0]
                            o_regs[o_off + 1] = array462_mma_1356[1]
                            o_regs[o_off + 2] = array462_mma_1356[2]
                            o_regs[o_off + 3] = array462_mma_1356[3]
                    if cutlass.const_expr(prefetch_next):
                        v_cur = v_next
        else:
            v_cur = [None] * self.pv_d_frags
            for d_pair in cutlass.range_constexpr(self.pv_d_frags // 2):
                v_cur[d_pair * 2], v_cur[d_pair * 2 + 1] = load_v_fragment_pair(0, d_pair)

            for v_frag in cutlass.range_constexpr(self.pv_v_frags):
                prefetch_next = cutlass.const_expr(
                    not (
                        self.head_tile == 256 and self.q_tile in (16, 32, 64, 128) and self.kv_tile in (64, 128)
                    )
                    or v_frag + 1 < self.pv_v_frags
                )
                if cutlass.const_expr(prefetch_next):
                    v_seq_off_next = ((v_frag + 1) % self.pv_v_frags) * self.MMA_TILER[2]
                    v_next = [None] * self.pv_d_frags

                p_reg0 = pack_p_cols(v_frag * 4 + 0, 0)
                p_reg1 = pack_p_cols(v_frag * 4 + 0, 1)
                p_reg2 = pack_p_cols(v_frag * 4 + 2, 0)
                p_reg3 = pack_p_cols(v_frag * 4 + 2, 1)
                # Interleave each next-fragment x2 prefetch with the two MMAs that
                # consume the matching current D pair. This exposes independent
                # load and MMA work while preserving the x2 carrier pairing.
                for d_pair in cutlass.range_constexpr(self.pv_d_frags // 2):
                    if cutlass.const_expr(prefetch_next):
                        v_next[d_pair * 2], v_next[d_pair * 2 + 1] = load_v_fragment_pair(
                            v_seq_off_next, d_pair
                        )
                    for d_in_pair in cutlass.range_constexpr(2):
                        d_frag = d_pair * 2 + d_in_pair
                        o_off = d_frag * 4
                        array462_mma_1356 = _tma_atrex_mma_e4m3_f32(
                            p_reg0,
                            p_reg1,
                            p_reg2,
                            p_reg3,
                            v_cur[d_frag][0],
                            v_cur[d_frag][1],
                            o_regs[o_off + 0],
                            o_regs[o_off + 1],
                            o_regs[o_off + 2],
                            o_regs[o_off + 3],
                            self.in_dtype,
                        )
                        o_regs[o_off + 0] = array462_mma_1356[0]
                        o_regs[o_off + 1] = array462_mma_1356[1]
                        o_regs[o_off + 2] = array462_mma_1356[2]
                        o_regs[o_off + 3] = array462_mma_1356[3]
                if cutlass.const_expr(prefetch_next):
                    v_cur = v_next


    @cute.jit
    def wait_pipeline_barrier(
        self,
        mbar: cutlass.Pointer,
        phase: cutlass.Int32,
    ) -> None:
        """Block on one dynamically selected circular-pipeline barrier."""
        while not _tma_mbarrier_try_wait_parity(mbar, phase, time_limit=10_000_000):
            pass


    @cute.jit
    def compute_one_kv_tile(
        self,
        basic_params: SimpleNamespace,
        mma_params: SimpleNamespace,
        softmax_params: SimpleNamespace,
        q_regs: cute.Tensor,
        row_state: cutlass.Float32,
        kv_tile_idx: cutlass.Int32,
        k_stage: cutlass.Int32,
        is_masked_frontier_tile: cutlass.Constexpr[bool],
        uses_softmax_init_path: cutlass.Constexpr[bool],
    ) -> cutlass.Float32:
        """One compute-side iteration of the right-to-left FMHA prefill loop.

        K and V are consecutive operands in one three-slot circular pipeline:
        ``K0, V0, K1, V1, ...``. Online softmax sits between consuming K and V
        because it has no V dependency, so the producer can fill later slots
        while scalar work runs.

        :param basic_params: Per-CTA tensor metadata, lane mapping, and K/V
            pipeline mbarriers.
        :param mma_params: Shared-memory tiles and local MMA state.
        :param softmax_params: Online softmax row state.
        :param q_regs: Register-resident packed Q fragments.
        :param row_state: One distributed max/sum scalar per lane.
        :param kv_tile_idx: K/V tile index processed by this iteration.
        :param k_stage: K slot in the repeating ``0, 2, 1`` ring schedule.
        :param is_masked_frontier_tile: Marks the causal/K-tail frontier for
            causal limits and explicit K/V-length bounds.
        :param uses_softmax_init_path: Whether to bypass the neutral initial
            softmax/O state instead of using the generic merge path.
        """

        # Six operands span one three-tile period. Carry the K stage directly;
        # its low bit is the K phase. V is the next slot, and its phase is a
        # divide-by-two shift. This avoids selecting four values from a state
        # table on every inner-loop iteration.
        k_phase = k_stage & cutlass.Int32(1)
        v_stage = k_stage + cutlass.Int32(1)
        if v_stage == self.kv_pipeline_stages:
            v_stage = cutlass.Int32(0)
        v_phase = (k_stage + cutlass.Int32(1)) // cutlass.Int32(2)

        k_arrived = self.get_mbar_stage_ptr(basic_params.mbar_arrived, k_stage)
        self.wait_pipeline_barrier(k_arrived, k_phase)

        sK = self.get_kv_stage_ptr(mma_params.sKV, k_stage)
        s_regs = self.mma_qk(basic_params, mma_params, q_regs, sK)
        if _tma_elect_sync():
            k_consumed = self.get_mbar_stage_ptr(basic_params.mbar_consumed, k_stage)
            _tma_mbarrier_arrive(
                k_consumed,
                count=cute.arch.WARP_SIZE,
            )

        kv_seq_idx = kv_tile_idx * self.kv_tile
        v_arrived = self.get_mbar_stage_ptr(basic_params.mbar_arrived, v_stage)
        sV = self.get_kv_stage_ptr(mma_params.sKV, v_stage)
        p_regs, row_state = self.online_softmax(
            basic_params,
            mma_params,
            softmax_params,
            s_regs,
            row_state,
            kv_seq_idx,
            is_masked_frontier_tile,
            uses_softmax_init_path,
        )
        self.wait_pipeline_barrier(v_arrived, v_phase)
        self.mma_pv(basic_params, mma_params, p_regs, sV, kv_seq_idx, is_masked_frontier_tile)
        if _tma_elect_sync():
            v_consumed = self.get_mbar_stage_ptr(basic_params.mbar_consumed, v_stage)
            _tma_mbarrier_arrive(
                v_consumed,
                count=cute.arch.WARP_SIZE,
            )
        return row_state


    @cute.kernel
    def atrex_fp8_prefill_nhd_tma_kernel(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        o: cute.Tensor,
        lse: cute.Tensor | None,
        tma_k_desc: cute.CopyAtom,
        tma_v_desc: cute.CopyAtom,
        tma_k_tensor: cute.Tensor,
        tma_v_tensor: cute.Tensor,
        softmax_scale_log2: cutlass.Float32,
        output_scale: cutlass.Float32,
        skip_softmax_threshold: cute.Tensor | None,
        seqlens_kv: cute.Tensor | None = None,
        cu_seqlens_q: cute.Tensor | None = None,
        block_tables: cute.Tensor | None = None,
        cu_seqlens_k: cute.Tensor | None = None,
        q_descale: cute.Tensor | None = None,
        k_descale: cute.Tensor | None = None,
        v_descale: cute.Tensor | None = None,
    ) -> None:
        """SM120 FMHA prefill FP8 kernel.

        :param q: Query tensor.
        :param k: Key tensor.
        :param v: Value tensor.
        :param o: Output tensor.
        :param lse: Optional packed log2 LSE tensor ``[Hq,total_q] or [B,Hq,Q]``.
        :param tma_k_desc: Tensor map descriptor for K.
        :param tma_v_desc: Tensor map descriptor for V.
        :param softmax_scale_log2: ``softmax_scale * log2(e)``, pre-folded host-side.
        :param output_scale: Scalar V dequantization scale folded into the
            final normalization.
        :param skip_softmax_threshold: Reserved compatibility argument; must be None.
        :param seqlens_kv: Optional runtime K/V length for each request.
        :param cu_seqlens_q: Optional cumulative offsets for packed Q/O.
        :param block_tables: Shared K/V page indices with shape ``[B,max_pages]``.
        :param cu_seqlens_k: Optional cumulative offsets for packed K/V.
        """
        tidx, _, _ = cute.arch.thread_idx()
        block_x, block_y, block_z = cute.arch.block_idx()
        q_head_idx = block_x
        q_tile_idx = block_y
        # Causal Q tiles have monotonically increasing K/V work. Schedule
        # the longest rows first so the final waves do not consist only of
        # the most expensive CTAs. This preserves the logical tile mapping
        # while improving load balance for every causal sequence length.
        _, grid_y, _ = cute.arch.grid_dim()
        q_tile_idx = grid_y - cutlass.Int32(1) - block_y
        if cutlass.const_expr(self.num_splits > 1):
            batch_idx = block_z // self.num_splits
            split_idx = block_z % self.num_splits
        else:
            batch_idx = block_z
        q_seq_idx = q_tile_idx * self.q_tile

        warp = cute.arch.make_warp_uniform(cute.arch.warp_idx())
        lane = tidx % cute.arch.WARP_SIZE
        lane_div4 = lane // 4
        lane_mod4 = lane % 4
        lane_div8 = lane // 8
        lane_mod8 = lane % 8
        lane_div16 = lane // 16

        q_token_base = cutlass.Int32(0)
        causal_q_offset = cutlass.Int32(0)

        cute.arch.griddepcontrol_wait()
        q_token_base = cutlass.Int32(cu_seqlens_q[batch_idx])
        seqlen_q = cutlass.Int32(cu_seqlens_q[batch_idx + 1]) - q_token_base
        logical_seqlen_q = seqlen_q
        packed_row_count = logical_seqlen_q
        num_heads_q = q.shape[1]
        head_dim = q.shape[2]

        seqlen_k = cutlass.Int32(seqlens_kv[batch_idx])
        num_heads_kv = k.shape[2]

        causal_q_offset = seqlen_k - logical_seqlen_q
        q_ptr = cutlass.Pointer(q.iterator, dtype=q.dtype)
        o_ptr = cutlass.Pointer(o.iterator, dtype=o.dtype)

        q_heads_per_kv_head = num_heads_q // num_heads_kv
        if cutlass.const_expr(self.pack_gqa):
            # A CTA owns one KV group; its M rows interleave that group's heads.
            kv_head_idx = q_head_idx
            q_head_idx = kv_head_idx * q_heads_per_kv_head
            packed_row_count = logical_seqlen_q * q_heads_per_kv_head
        else:
            kv_head_idx = q_head_idx
            if cutlass.const_expr(num_heads_q != num_heads_kv):
                # Contiguous equal-sized Q-head groups share one K/V head, matching
                # repeat_interleave semantics without materializing repeated K/V.
                q_heads_per_kv_head = num_heads_q // num_heads_kv
                kv_head_idx = q_head_idx // q_heads_per_kv_head

        # ATREX scales are FP32 [B,Hkv], indexed by the logical KV head.
        qk_descale = cutlass.Float32(1.0)
        if cutlass.const_expr(q_descale is not None):
            qk_descale = qk_descale * cutlass.Float32(q_descale[batch_idx, kv_head_idx])
        if cutlass.const_expr(k_descale is not None):
            qk_descale = qk_descale * cutlass.Float32(k_descale[batch_idx, kv_head_idx])
        softmax_scale_log2 = softmax_scale_log2 * qk_descale
        if cutlass.const_expr(v_descale is not None):
            output_scale = output_scale * cutlass.Float32(v_descale[batch_idx, kv_head_idx])

        q_seq_stride = num_heads_q * head_dim
        q_head_base = cutlass.Int64(q_token_base) * cutlass.Int64(q_seq_stride) + cutlass.Int64(q_head_idx) * head_dim
        o_head_base = q_head_base
        lse_head_base = q_head_idx
        if cutlass.const_expr(self.num_splits > 1):
            # Cast before multiplying; the partial plane may exceed 2 GiB.
            o_head_base = o_head_base + (
                cutlass.Int64(split_idx) * cutlass.Int64(q.shape[0])
                * cutlass.Int64(num_heads_q) * cutlass.Int64(head_dim)
            )
            lse_head_base = q_head_idx + split_idx * num_heads_q

        num_kv_tiles = _tma_atrex_ceildiv(seqlen_k, self.kv_tile)
        if cutlass.const_expr(self.pack_gqa):
            # The final packed M row supplies a conservative logical frontier.
            last_q_token = (q_seq_idx + self.q_tile - 1) // q_heads_per_kv_head
            num_kv_tiles_causal = _tma_atrex_ceildiv(
                last_q_token + causal_q_offset + 1, self.kv_tile
            )
        else:
            num_kv_tiles_causal = _tma_atrex_ceildiv(
                q_seq_idx + causal_q_offset + self.q_tile, self.kv_tile
            )
        num_kv_tiles_causal = cute.math.max(num_kv_tiles_causal, cutlass.Int32(0))
        num_kv_tiles = cute.math.min(num_kv_tiles, num_kv_tiles_causal)

        kv_tile_begin = cutlass.Int32(0)
        kv_tile_end = num_kv_tiles
        if cutlass.const_expr(self.num_splits > 1):
            # Partition the current Q tile's causal-bound KV tiles without overlap.
            kv_tile_begin = (num_kv_tiles * split_idx) // self.num_splits
            kv_tile_end = (num_kv_tiles * (split_idx + 1)) // self.num_splits
            num_kv_tiles = kv_tile_end - kv_tile_begin

        # The epilogue later aliases this storage as sO after compute warps
        # finish consuming the final K/V tile.
        array462_smem = utils_basic.SmemAllocator()
        sKV = array462_smem.allocate_array(k.dtype, self.smem_elems, byte_alignment=128)
        s_mbar = array462_smem.allocate_array(cutlass.Int64, self.kv_pipeline_stages * 2, byte_alignment=8)
        mbar_arrived = s_mbar
        mbar_consumed = (s_mbar + self.kv_pipeline_stages)
        tma_k_desc_ptr = (tma_k_desc, tma_k_tensor)
        tma_v_desc_ptr = (tma_v_desc, tma_v_tensor)

        # Initialize the K/V pipeline only for CTAs that have both a Q row and
        # at least one K/V token. The condition is CTA-uniform, so zero-length
        # requests skip every TMA/mbarrier/MMA operation without an early
        # return (early kernel returns are unsupported by CuTe DSL).
        if q_seq_idx < packed_row_count and seqlen_k > 0 and num_kv_tiles > 0:
            if warp == self.load_warp_id:
                if _tma_elect_sync():
                    cpasync.prefetch_descriptor(tma_k_desc)
                    cpasync.prefetch_descriptor(tma_v_desc)
                if _tma_elect_sync():
                    _tma_mbarrier_init((mbar_arrived + 0), 1)
                    _tma_mbarrier_init((mbar_arrived + 1), 1)
                    _tma_mbarrier_init((mbar_arrived + 2), 1)
                    _tma_mbarrier_init((mbar_consumed + 0), self.threads_compute)
                    _tma_mbarrier_init((mbar_consumed + 1), self.threads_compute)
                    _tma_mbarrier_init(
                        (mbar_consumed + 2), self.threads_compute
                    )
            _tma_fence_mbarrier_init()
            _tma_barrier_cta_sync(0)

        # /////////////////////////////////////////////////////////////////////////////
        #  LOAD K/V
        # /////////////////////////////////////////////////////////////////////////////
        if warp == self.load_warp_id and q_seq_idx < packed_row_count and seqlen_k > 0 and num_kv_tiles > 0:
            if cutlass.const_expr(self.q_tile >= 64):
                _tma_setmaxregister(24, _tma_SetMaxRegisterAction.DECREASE)

            kv_l2_cache_hint = cutlass.Int64(0)
            kv_l2_cache_hint = cute.arch.inline_ptx(
                "createpolicy.fractional.L2::evict_last.b64 $0, 1.0;",
                write_only_types=[cutlass.Int64],
                read_only_args=[],
            )
            cached_page = cutlass.Int32(0)
            # Treat K and V as one ordered operand stream. Three ring slots
            # initially hold K0, V0, and K1. Thereafter the producer reuses
            # a slot as soon as compute signals that operand consumed.
            operand_idx = cutlass.Int32(0)
            stage = cutlass.Int32(0)
            empty_phase = cutlass.Int32(0)
            num_operands = num_kv_tiles * cutlass.Int32(2)
            while operand_idx < num_operands:
                if operand_idx >= self.kv_pipeline_stages:
                    empty = self.get_mbar_stage_ptr(mbar_consumed, stage)
                    self.wait_pipeline_barrier(empty, empty_phase)

                tile_iteration = operand_idx // cutlass.Int32(2)
                if cutlass.const_expr(self.num_splits > 1):
                    kv_seq_idx = (
                        kv_tile_end - cutlass.Int32(1) - tile_iteration
                    ) * self.kv_tile
                else:
                    kv_seq_idx = (
                        num_kv_tiles - cutlass.Int32(1) - tile_iteration
                    ) * self.kv_tile
                page_ids = self.resolve_paged_kv_tile_pages(
                    batch_idx,
                    kv_seq_idx,
                    seqlen_k,
                    block_tables,
                    cached_page,
                )

                s_operand = self.get_kv_stage_ptr(sKV, stage)
                arrived = self.get_mbar_stage_ptr(mbar_arrived, stage)
                is_v_operand = operand_idx & cutlass.Int32(1)
                if is_v_operand == 0:
                    if cutlass.const_expr(self.kv_tile < self.num_tokens_per_page):
                        self.load_one_paged_kv_tile(
                            s_operand,
                            tma_k_desc_ptr,
                            arrived,
                            page_ids,
                            kv_head_idx,
                            kv_l2_cache_hint,
                            kv_seq_idx,
                        )
                    else:
                        self.load_one_paged_kv_tile(
                            s_operand,
                            tma_k_desc_ptr,
                            arrived,
                            page_ids,
                            kv_head_idx,
                            kv_l2_cache_hint,
                        )
                else:
                    if cutlass.const_expr(self.kv_tile < self.num_tokens_per_page):
                        self.load_one_paged_kv_tile(
                            s_operand,
                            tma_v_desc_ptr,
                            arrived,
                            page_ids,
                            kv_head_idx,
                            kv_l2_cache_hint,
                            kv_seq_idx,
                        )
                    else:
                        self.load_one_paged_kv_tile(
                            s_operand,
                            tma_v_desc_ptr,
                            arrived,
                            page_ids,
                            kv_head_idx,
                            kv_l2_cache_hint,
                        )
                operand_idx += 1
                stage += 1
                if stage == self.kv_pipeline_stages:
                    stage = cutlass.Int32(0)
                    # The first wrap only fills the initially empty slots.
                    # Every later wrap advances each stage by one generation.
                    if operand_idx > self.kv_pipeline_stages:
                        empty_phase ^= cutlass.Int32(1)
        # /////////////////////////////////////////////////////////////////////////////
        #  COMPUTE
        # /////////////////////////////////////////////////////////////////////////////
        elif warp < self.load_warp_id and q_seq_idx < packed_row_count and seqlen_k > 0 and num_kv_tiles > 0:
            if cutlass.const_expr(self.q_tile >= 64):
                # N64/Q64 keeps the requested warpgroup register budget
                # within 32K registers per CTA. Residency still depends on
                # the compiler allocation and launch resource limits.
                if cutlass.const_expr(self.kv_tile == 64 and self.q_tile == 64):
                    _tma_setmaxregister(232, _tma_SetMaxRegisterAction.INCREASE)
                else:
                    _tma_setmaxregister(240, _tma_SetMaxRegisterAction.INCREASE)

            q_l2_cache_hint = cutlass.Int64(0)
            q_l2_cache_hint = cute.arch.inline_ptx(
                "createpolicy.fractional.L2::evict_first.b64 $0, 1.0;",
                write_only_types=[cutlass.Int64],
                read_only_args=[],
            )
            compute_warp_idx = warp
            q_warp_row0 = compute_warp_idx * self.MMA_TILER[0]

            # The D256 tiles retain distributed state: four per-lane
            # scalars cross the Q128 register cliff. Lanes 0/1
            # hold maxima and lanes 2/3 hold sums.
            row_state = cutlass.Float32(0.0)
            if lane_mod4 < 2:
                row_state = -cutlass.Float32.inf

            # Per-lane fp32 accumulator for O = P @ V.
            o_regs = cute.make_rmem_tensor(cute.make_layout(self.pv_d_frags * 4), cutlass.Float32)
            for i in cutlass.range_constexpr(self.pv_d_frags * 4):
                o_regs[i] = 0.0

            basic_params = SimpleNamespace(
                # Problem shape and logical Q coordinates.
                seqlen_q=packed_row_count,
                q_heads_per_kv_head=q_heads_per_kv_head,
                seqlen_k=seqlen_k,
                head_dim=head_dim,
                q_ptr=q_ptr,
                q_seq_idx=q_seq_idx,
                q_head_base=q_head_base,
                causal_q_offset=causal_q_offset,
                q_seq_stride=q_seq_stride,
                q_l2_cache_hint=q_l2_cache_hint,
                # Compute-warp row and lane ownership.
                q_warp_row0=q_warp_row0,
                lane=lane,
                lane_div4=lane_div4,
                lane_mod4=lane_mod4,
                lane_div8=lane_div8,
                lane_mod8=lane_mod8,
                # Unified K/V ring arrival and reuse handshakes.
                mbar_arrived=mbar_arrived,
                mbar_consumed=mbar_consumed,
            )
            mma_params = SimpleNamespace(
                sKV=sKV,
                o_regs=o_regs,
            )

            softmax_params = SimpleNamespace(
                softmax_scale_log2=softmax_scale_log2,
            )

            # Load Q into registers.
            q_regs = self.load_q_tile(basic_params)

            # K/V runs right-to-left, so tail/causal boundary tiles form a short
            # masked prefix. Once that prefix is consumed, every remaining tile
            # is fully valid in both ragged and paged storage, so the hot loop
            # can skip per-score bounds checks.
            if cutlass.const_expr(self.num_splits > 1):
                kv_tile_idx = kv_tile_end - 1
            else:
                kv_tile_idx = num_kv_tiles - 1
            masked_kv_tile_count = 1
            # Packed requests can be rectangular, so reserve one full
            # extra overlap for the bottom-right causal offset.
            masked_kv_tile_count = _tma_atrex_ceildiv(
                self.q_tile + self.kv_tile - 1, self.kv_tile
            )

            # Phase 1: potentially masked iterations.
            pipeline_k_stage = cutlass.Int32(0)
            for step in cutlass.range_constexpr(masked_kv_tile_count):
                if kv_tile_idx >= kv_tile_begin:
                    row_state = self.compute_one_kv_tile(
                        basic_params,
                        mma_params,
                        softmax_params,
                        q_regs,
                        row_state,
                        kv_tile_idx,
                        pipeline_k_stage,
                        is_masked_frontier_tile=True,
                        uses_softmax_init_path=(step == 0),
                    )
                kv_tile_idx -= 1
                if pipeline_k_stage == 0:
                    pipeline_k_stage = cutlass.Int32(2)
                else:
                    pipeline_k_stage -= 1

            # Phase 2: remaining tiles outside the explicit mask frontier.
            while kv_tile_idx >= kv_tile_begin:
                row_state = self.compute_one_kv_tile(
                    basic_params,
                    mma_params,
                    softmax_params,
                    q_regs,
                    row_state,
                    kv_tile_idx,
                    pipeline_k_stage,
                    is_masked_frontier_tile=False,
                    uses_softmax_init_path=False,
                )
                kv_tile_idx -= 1
                if pipeline_k_stage == 0:
                    pipeline_k_stage = cutlass.Int32(2)
                else:
                    pipeline_k_stage -= 1

            _tma_barrier_cta_sync(
                self.COMPUTE_BARRIER_ID, thread_count=self.threads_compute
            )

            quad_lane_base = lane_div4 * 4
            row_max_0 = _tma_shfl_sync(
                thread_mask=0xFFFFFFFF,
                val=row_state,
                offset=quad_lane_base,
                mask_and_clamp=0x1F,
                kind=_tma_Shfl.IDX,
            )
            row_max_1 = _tma_shfl_sync(
                thread_mask=0xFFFFFFFF,
                val=row_state,
                offset=quad_lane_base + 1,
                mask_and_clamp=0x1F,
                kind=_tma_Shfl.IDX,
            )
            row_sum_0 = _tma_shfl_sync(
                thread_mask=0xFFFFFFFF,
                val=row_state,
                offset=quad_lane_base + 2,
                mask_and_clamp=0x1F,
                kind=_tma_Shfl.IDX,
            )
            row_sum_1 = _tma_shfl_sync(
                thread_mask=0xFFFFFFFF,
                val=row_state,
                offset=quad_lane_base + 3,
                mask_and_clamp=0x1F,
                kind=_tma_Shfl.IDX,
            )

            # The four lanes in each lane quad hold identical final online
            # softmax state for one Q row. Have one leader write each of the
            # two rows owned by the quad. FlashInfer LSE is log2-base:
            #
            #   log2(sum(exp(score * scale)))
            #     = row_max * scale * log2(e) + log2(row_sum).
            if cutlass.const_expr(lse is not None):
                if lane_mod4 == 0:
                    for row_half in cutlass.range_constexpr(2):
                        lse_q_seq_idx = (
                            q_seq_idx + q_warp_row0 + lane_div4 + row_half * 8
                        )
                        if lse_q_seq_idx < packed_row_count:
                            if cutlass.const_expr(row_half == 0):
                                lse_row_max = row_max_0
                                lse_row_sum = row_sum_0
                            else:
                                lse_row_max = row_max_1
                                lse_row_sum = row_sum_1
                            lse_natural = (
                                lse_row_max * softmax_scale_log2
                                + cute.math.log2(lse_row_sum, fastmath=True)
                            ) * math.log(2.0)
                            if cutlass.const_expr(self.pack_gqa):
                                lse_token = lse_q_seq_idx // q_heads_per_kv_head
                                lse_head = lse_head_base + lse_q_seq_idx % q_heads_per_kv_head
                                if cutlass.const_expr(cute.rank(lse) == 3):
                                    lse[batch_idx, lse_head, lse_token] = lse_natural
                                else:
                                    lse[lse_head, q_token_base + lse_token] = lse_natural
                            else:
                                if cutlass.const_expr(cute.rank(lse) == 3):
                                    lse[batch_idx, lse_head_base, lse_q_seq_idx] = lse_natural
                                else:
                                    lse[lse_head_base, q_token_base + lse_q_seq_idx] = lse_natural

            if cutlass.const_expr(self.num_splits > 1):
                # SplitKV partials must remain FP32 for the existing combine ABI.
                row_sum_inv_0 = cutlass.Float32(0.0)
                row_sum_inv_1 = cutlass.Float32(0.0)
                if row_sum_0 > 0.0:
                    row_sum_inv_0 = cute.math.rcp(row_sum_0, approx=True, ftz=True) * output_scale
                if row_sum_1 > 0.0:
                    row_sum_inv_1 = cute.math.rcp(row_sum_1, approx=True, ftz=True) * output_scale
                for d_frag in cutlass.range_constexpr(self.pv_d_frags):
                    for row_half in cutlass.range_constexpr(2):
                        partial_q_row = (
                            q_seq_idx + q_warp_row0 + lane_div4 + row_half * 8
                        )
                        partial_col = d_frag * 8 + lane_mod4 * 2
                        if cutlass.const_expr(row_half == 0):
                            row_sum_inv = row_sum_inv_0
                        else:
                            row_sum_inv = row_sum_inv_1
                        o_off = d_frag * 4 + row_half * 2
                        partial_vec = cutlass.Vector.from_elements(
                            (o_regs[o_off] * row_sum_inv,
                             o_regs[o_off + 1] * row_sum_inv),
                            dtype=cutlass.Float32,
                        )
                        if cutlass.const_expr(self.pack_gqa):
                            partial_row_offset = (
                                (partial_q_row // q_heads_per_kv_head) * q_seq_stride
                                + (partial_q_row % q_heads_per_kv_head) * head_dim
                            )
                        else:
                            partial_row_offset = partial_q_row * q_seq_stride
                        if partial_q_row < packed_row_count:
                            (
                                o_ptr + o_head_base + partial_row_offset + partial_col
                            ).store(partial_vec, alignment=8)
            else:
                # No compute warp may alias sKV as sO while a peer still reads the
                # final sV tile. After that lifetime barrier, normalize O, stage an
                # stmatrix-friendly layout, and store 128b per lane to GMEM.
                sO = cutlass.Pointer(sKV, dtype=o.dtype)
                # Bottom-right rectangular causal attention can have empty rows.
                # Their neutral accumulator must normalize to zero, not 0*inf.
                row_sum_inv_0 = cutlass.Float32(0.0)
                row_sum_inv_1 = cutlass.Float32(0.0)
                if row_sum_0 > 0.0:
                    row_sum_inv_0 = cute.math.rcp(row_sum_0, approx=True, ftz=True) * output_scale
                if row_sum_1 > 0.0:
                    row_sum_inv_1 = cute.math.rcp(row_sum_1, approx=True, ftz=True) * output_scale
                for d_frag_pair in cutlass.range_constexpr(self.pv_d_frags // 2):
                    o_off = (d_frag_pair * 2) * 4
                    o_scaled0 = _tma_atrex_pair_mul(
                        (o_regs[o_off + 0], o_regs[o_off + 1]),
                        (row_sum_inv_0, row_sum_inv_0),
                    )
                    o_scaled1 = _tma_atrex_pair_mul(
                        (o_regs[o_off + 2], o_regs[o_off + 3]),
                        (row_sum_inv_1, row_sum_inv_1),
                    )
                    o_scaled2 = _tma_atrex_pair_mul(
                        (o_regs[o_off + 4], o_regs[o_off + 5]),
                        (row_sum_inv_0, row_sum_inv_0),
                    )
                    o_scaled3 = _tma_atrex_pair_mul(
                        (o_regs[o_off + 6], o_regs[o_off + 7]),
                        (row_sum_inv_1, row_sum_inv_1),
                    )
                    o_packed0 = _tma_atrex_pack_word(
                        (o.dtype(o_scaled0[0]), o.dtype(o_scaled0[1])),
                        o.dtype,
                    )
                    o_packed1 = _tma_atrex_pack_word(
                        (o.dtype(o_scaled1[0]), o.dtype(o_scaled1[1])),
                        o.dtype,
                    )
                    o_packed2 = _tma_atrex_pack_word(
                        (o.dtype(o_scaled2[0]), o.dtype(o_scaled2[1])),
                        o.dtype,
                    )
                    o_packed3 = _tma_atrex_pack_word(
                        (o.dtype(o_scaled3[0]), o.dtype(o_scaled3[1])),
                        o.dtype,
                    )
                    sO_ptr = (
                        sO
                        + compute_warp_idx * (self.pv_d_frags // 2) * (16 * 16)
                        + d_frag_pair * (16 * 16)
                        + lane * 8
                    )
                    _tma_stmatrix(
                        sO_ptr,
                        [o_packed0, o_packed1, o_packed2, o_packed3],
                        _tma_MMALayout.ROW,
                    )

                store_row = lane_mod8 + ((lane_div8) % 2) * 8
                store_col = lane_div16 * 8
                store_q_seq_idx = q_seq_idx + q_warp_row0 + store_row
                for d_frag_pair in cutlass.range_constexpr(self.pv_d_frags // 2):
                    store_col_in_cta = d_frag_pair * 16 + store_col
                    sO_ptr = (
                        sO
                        + compute_warp_idx * (self.pv_d_frags // 2) * (16 * 16)
                        + d_frag_pair * (16 * 16)
                        + lane * 8
                    )
                    o_vec = sO_ptr.load(count=8, alignment=16)
                    if cutlass.const_expr(self.pack_gqa):
                        gO_ptr = (
                            o_ptr
                            + o_head_base
                            + (store_q_seq_idx // q_heads_per_kv_head) * q_seq_stride
                            + (store_q_seq_idx % q_heads_per_kv_head) * head_dim
                            + store_col_in_cta
                        )
                    else:
                        gO_ptr = (
                            o_ptr
                            + o_head_base
                            + store_q_seq_idx * q_seq_stride
                            + store_col_in_cta
                        )
                    if store_q_seq_idx < packed_row_count and store_col_in_cta < head_dim:
                        gO_ptr.store(o_vec, alignment=16)

        # A zero-length K/V sequence has a mathematically empty attention
        # reduction. This is ensured by writing O=0 for the
        # valid Q rows in this CTA. Compute warps own disjoint groups of 16
        # rows, and the two lane half-warps cover one aligned 16-element output
        # vector per row, so this path needs neither SMEM nor a CTA barrier.
        elif warp < self.load_warp_id and q_seq_idx < packed_row_count:
            if cutlass.const_expr(self.q_tile >= 64):
                _tma_setmaxregister(24, _tma_SetMaxRegisterAction.DECREASE)
            zero_vector = cutlass.Vector.from_elements(
                tuple(self.out_dtype(0.0) for _ in range(8)),
                dtype=self.out_dtype,
            )
            zero_q_seq_idx = (
                q_seq_idx + warp * self.MMA_TILER[0] + lane % self.MMA_TILER[0]
            )
            if cutlass.const_expr(lse is not None):
                if lane < self.MMA_TILER[0] and zero_q_seq_idx < packed_row_count:
                    if cutlass.const_expr(self.pack_gqa):
                        zero_token = zero_q_seq_idx // q_heads_per_kv_head
                        zero_head = lse_head_base + zero_q_seq_idx % q_heads_per_kv_head
                        if cutlass.const_expr(cute.rank(lse) == 3):
                            lse[batch_idx, zero_head, zero_token] = -cutlass.Float32.inf
                        else:
                            lse[zero_head, q_token_base + zero_token] = -cutlass.Float32.inf
                    else:
                        if cutlass.const_expr(cute.rank(lse) == 3):
                            lse[batch_idx, lse_head_base, zero_q_seq_idx] = -cutlass.Float32.inf
                        else:
                            lse[lse_head_base, q_token_base + zero_q_seq_idx] = -cutlass.Float32.inf
            for head_chunk in cutlass.range_constexpr(self.head_tile // 16):
                zero_head_col = head_chunk * 16 + lane_div16 * 8
                if zero_q_seq_idx < packed_row_count:
                    if cutlass.const_expr(self.pack_gqa):
                        (
                            o_ptr
                            + o_head_base
                            + (zero_q_seq_idx // q_heads_per_kv_head) * q_seq_stride
                            + (zero_q_seq_idx % q_heads_per_kv_head) * head_dim
                            + zero_head_col
                        ).store(zero_vector, alignment=16)
                    else:
                        (
                            o_ptr
                            + o_head_base
                            + zero_q_seq_idx * q_seq_stride
                            + zero_head_col
                        ).store(zero_vector, alignment=16)

        # /////////////////////////////////////////////////////////////////////////////
        #  EMPTY / INACTIVE CTA
        # /////////////////////////////////////////////////////////////////////////////
        else:
            if cutlass.const_expr(self.q_tile >= 64):
                _tma_setmaxregister(24, _tma_SetMaxRegisterAction.DECREASE)

        cute.arch.griddepcontrol_launch_dependents()

    @cute.jit
    def __call__(
        self,
        q: cute.Tensor,
        k: cute.Tensor,
        v: cute.Tensor,
        o: cute.Tensor,
        lse: cute.Tensor | None,
        softmax_scale_log2: cutlass.Float32,
        output_scale: cutlass.Float32,
        skip_softmax_threshold: cute.Tensor | None,
        stream: cuda.CUstream,
        seqlens_kv: cute.Tensor | None = None,
        cu_seqlens_q: cute.Tensor | None = None,
        block_tables: cute.Tensor | None = None,
        cu_seqlens_k: cute.Tensor | None = None,
        max_seqlen_q: cutlass.Int32 | None = None,
        use_pdl: bool = False,
        q_descale: cute.Tensor | None = None,
        k_descale: cute.Tensor | None = None,
        v_descale: cute.Tensor | None = None,
    ) -> None:
        """Launch the SM120 PRIM FMHA FP8 kernel.

        :param q: Packed query ``(total_q, Hq, D)``.
        :param k: Packed key ``(total_k, Hkv, D)`` or NHD paged K pool
            ``(num_pages, num_tokens_per_page, Hkv, D)`` with supported physical
            strides. Paged pools may contain NaNs in unused tail-page slots.
        :param v: Packed or paged value with the same shape and tail-padding
            contract as K.
        :param o: Packed output ``(total_q, Hq, D)``.
        :param lse: Optional head-major float32 natural-log LSE ``(total_q,Hq)``.
        :param softmax_scale_log2: ``softmax_scale * log2(e)``.
        :param output_scale: Scalar multiplier applied to normalized output.
        :param skip_softmax_threshold: Reserved compatibility argument; must be None.
        :param stream: CUDA stream used for the launch.
        :param seqlens_kv: Runtime K/V lengths, one Int32 per request.
        :param cu_seqlens_q: Packed-Q cumulative Int32 offsets.
        :param block_tables: Shared K/V page indices with shape
            ``(B, max_num_pages_per_seq_kv)``.
        :param cu_seqlens_k: Packed-K/V cumulative Int32 offsets.
        :param max_seqlen_q: Contiguous-varlen launch-grid bound.
        :param use_pdl: Whether to use Programmatic Dependent Launch.

        This private launcher accepts only the constructor's fixed paged
        configuration. Optional skip-softmax execution is not supported.
        """
        if cutlass.const_expr(skip_softmax_threshold is not None):
            raise ValueError("SM120 TMA prefill requires dense softmax execution")
        head_dim = q.shape[2]
        output_head_dim = o.shape[2]
        num_heads_q = q.shape[1]
        batch_size = cute.size(cu_seqlens_q) - 1
        launch_batch_size = batch_size
        if cutlass.const_expr(self.num_splits > 1):
            if cutlass.const_expr(o.dtype != cutlass.Float32):
                raise ValueError("SplitKV requires Float32 partial output")
            if cutlass.const_expr(lse is None):
                raise ValueError("SplitKV requires internal Float32 LSE partials")
            if cutlass.const_expr(lse is not None):
                if cutlass.const_expr(cute.rank(lse) != 2 or lse.dtype != cutlass.Float32):
                    raise ValueError("SplitKV LSE must have rank two and Float32 dtype")
                if cutlass.const_expr(lse.shape[0] != self.num_splits * num_heads_q):
                    raise ValueError("SplitKV LSE leading dimension must be S*Hq")
            launch_batch_size = batch_size * self.num_splits
        if cutlass.const_expr(
            head_dim != k.shape[3]
            or head_dim != v.shape[3]
            or head_dim != output_head_dim
            or head_dim != self.head_tile
            or k.shape[1] != self.num_tokens_per_page
            or v.shape[1] != self.num_tokens_per_page
        ):
            raise ValueError(
                "runtime paged tensors must match the compiled structural config"
            )
        tma_k_desc = _tma_make_tma462(k, self.tma_copy_head_per_iter, self.tma_copy_rows, True)
        tma_v_desc = _tma_make_tma462(v, self.tma_copy_head_per_iter, self.tma_copy_rows, True)
        if cutlass.const_expr(self.pack_gqa):
            num_heads_kv = k.shape[2]
            if cutlass.const_expr(num_heads_kv <= 0 or num_heads_q % num_heads_kv != 0):
                raise ValueError("packed GQA requires Hq divisible by positive Hkv")
            q_heads_per_kv_head = num_heads_q // num_heads_kv
            packed_max_seqlen_q = max_seqlen_q * q_heads_per_kv_head
            grid = (
                num_heads_kv,
                cute.ceil_div(packed_max_seqlen_q, self.q_tile),
                launch_batch_size,
            )
        else:
            # Keep Q heads adjacent in launch order. GQA/MQA heads that share
            # K/V are scheduled close together, while Q tiles run from the
            # longest causal rows to the shortest.
            grid = (
                num_heads_q,
                cute.ceil_div(max_seqlen_q, self.q_tile),
                launch_batch_size,
            )

        self.atrex_fp8_prefill_nhd_tma_kernel(
            q,
            k,
            v,
            o,
            lse,
            tma_k_desc.atom,
            tma_v_desc.atom,
            tma_k_desc.tma_tensor,
            tma_v_desc.tma_tensor,
            softmax_scale_log2,
            output_scale,
            skip_softmax_threshold,
            seqlens_kv,
            cu_seqlens_q,
            block_tables,
            cu_seqlens_k,
            q_descale,
            k_descale,
            v_descale,
        ).launch(
            grid=grid,
            block=(self.threads_per_cta, 1, 1),
            stream=stream,
            # Only the narrow N64/Q64 specialization requests two blocks;
            # this hint does not assert measured CTA residency.
            min_blocks_per_mp=(
                2 if self.kv_tile == 64 and self.q_tile == 64 else 1
            ),
            use_pdl=use_pdl,
        )
