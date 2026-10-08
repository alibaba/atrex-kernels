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
# ATREX PORT: keep the local seqlen_info OOB clamp; use the installed FA4 package
# for the other CuTe helpers.
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
    a, b, c, fp8_dtype: cutlass.Constexpr, *, loc=None, ip=None
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
        "=f,=f,=f,=f,r,r,r,r,r,r,f,f,f,f",
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
def _clc_init_query_sm120(
    mbarrier_addr: cute.Pointer,
    clc_response_ptr: cute.Pointer,
    *,
    loc=None,
    ip=None,
):
    response_i32 = clc_response_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    mbarrier_i32 = mbarrier_addr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [response_i32, mbarrier_i32],
        "mbarrier.init.shared::cta.b64 [$1], 1;\n\t"
        "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [$0], [$1];\n\t"
        "mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 _, [$1], 16;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _clc_next_query_sm120(
    mbarrier_addr: cute.Pointer,
    clc_response_ptr: cute.Pointer,
    *,
    loc=None,
    ip=None,
):
    response_i32 = clc_response_ptr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    mbarrier_i32 = mbarrier_addr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [response_i32, mbarrier_i32],
        "fence.proxy.async.shared::cta;\n\t"
        "clusterlaunchcontrol.try_cancel.async.shared::cta.mbarrier::complete_tx::bytes.b128 [$0], [$1];\n\t"
        "mbarrier.arrive.expect_tx.relaxed.cta.shared::cta.b64 _, [$1], 16;",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


@dsl_user_op
def _clc_wait_sm120(
    mbarrier_addr: cute.Pointer,
    phase: Int32,
    *,
    loc=None,
    ip=None,
):
    mbarrier_i32 = mbarrier_addr.toint(loc=loc, ip=ip).ir_value(loc=loc, ip=ip)
    llvm.inline_asm(
        None,
        [mbarrier_i32, phase.ir_value(loc=loc, ip=ip)],
        "{\n\t.reg .pred p;\nwaitLoop:\n\t"
        "mbarrier.try_wait.parity.relaxed.cta.shared::cta.b64 p, [$0], $1;\n\t"
        "@!p bra waitLoop;\n}",
        "r,r",
        has_side_effects=True,
        is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT,
        loc=loc,
        ip=ip,
    )


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
        use_clc: bool = False,
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
        reorder_batch: bool = False,
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
        self.reorder_batch = reorder_batch
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
        self.use_clc = use_clc
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
            smem_usage_P = tile_m * (tile_n + 16)
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
                cute.cosize(self.sP_layout) if const_expr(self.is_fp8) else 1,
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

        @cute.struct
        class SharedStorageQKVFP8Clc:
            sV: sV_struct
            sQ: sQ_struct
            sK: sK_struct
            sP: sP_struct
            clc_mbar: cute.struct.MemRange[Int64, 1]
            clc_response: cute.struct.Align[cute.struct.MemRange[Int32, 4], 16]
            clc_coord: cute.struct.Align[cute.struct.MemRange[Int32, 2], 8]

        if const_expr(self.is_fp8):
            if const_expr(self.use_clc):
                return SharedStorageQKVFP8Clc
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
        mRequestOrder: Optional[cute.Tensor] = None,
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
        if const_expr(self.use_clc):
            assert TileScheduler is SingleTileVarlenScheduler
        if const_expr(self.reorder_batch):
            assert TileScheduler is SingleTileVarlenScheduler
            assert not self.use_clc and not self.compact_grid
            assert not self.runtime_balanced_splits and not self.is_split_kv
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

        launch_kernel = self.atrex_sm120_prefill_clc_kernel if const_expr(self.use_clc) else self.atrex_sm120_prefill_kernel
        launch_kernel(
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
            mRequestOrder,
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
        mRequestOrder: Optional[cute.Tensor],
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
            if const_expr(self.reorder_batch):
                reordered_batch = batch_size
                if cute.arch.lane_idx() == 0:
                    if mRequestOrder[0] != 0:
                        reordered_batch = Int32(mRequestOrder[batch_size + 1])
                batch_size = cute.arch.shuffle_sync(reordered_batch, 0)

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
            tSrQ = None
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
            cute.arch.cp_async_wait_group(self.num_stages * 2 - 1)
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

    @cute.kernel
    def atrex_sm120_prefill_clc_kernel(
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
        mRequestOrder: Optional[cute.Tensor],
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
        tidx, _, _ = cute.arch.thread_idx()
        smem = cutlass.utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        clc_mbar_ptr = storage.clc_mbar.data_ptr()
        clc_response_ptr = storage.clc_response.data_ptr()
        clc_coord_ptr = storage.clc_coord.data_ptr()
        sQ = cute.make_tensor(
            cute.recast_ptr(storage.sQ.data_ptr(), dtype=self.dtype), sQ_layout
        )
        sK = storage.sK.get_tensor(sK_layout)
        sV = storage.sV.get_tensor(sV_layout)
        sP = storage.sP.get_tensor(sP_layout)
        sVt = layout_utils.transpose_view(sV)

        # CLC is restricted by the host to uniform-Q, non-split packed-GQA
        # prefill.  That makes each batch occupy the same contiguous range in
        # the flattened launch, so canceled block coordinates can be decoded
        # directly without rerunning the generic warp-prefix varlen mapper.
        num_batch = tile_sched_params.num_batch
        num_head_sched = tile_sched_params.num_head
        num_m_blocks = cute.ceil_div(
            tile_sched_params.total_q // num_batch, self.tile_m
        )
        blocks_per_batch = num_m_blocks * num_head_sched
        tile_idx = cute.arch.block_idx()[0]
        batch_size = tile_idx // blocks_per_batch
        mh_block = tile_idx - batch_size * blocks_per_batch
        num_head = mh_block // num_m_blocks
        m_block = mh_block - num_head * num_m_blocks
        split_idx = Int32(0)
        work_valid = tile_idx < cute.arch.grid_dim()[0]
        clc_phase = Int32(0)
        if tidx == 0:
            _clc_init_query_sm120(clc_mbar_ptr, clc_response_ptr)

        while work_valid:
            self.process_tile_clc(
                m_block,
                num_head,
                batch_size,
                split_idx,
                mQ,
                mK,
                mV,
                mO,
                mLSE,
                mCuSeqlensQ,
                mCuSeqlensK,
                mSeqUsedQ,
                mSeqUsedK,
                mPageTable,
                mNumSplitsDynamic,
                mQDescale,
                mKDescale,
                mVDescale,
                softmax_scale_log2,
                softmax_scale,
                window_size_left,
                window_size_right,
                sQ,
                sK,
                sV,
                sP,
                sVt,
                sO_layout,
                gmem_tiled_copy_Q,
                gmem_tiled_copy_K,
                gmem_tiled_copy_V,
                gmem_tiled_copy_O,
                tiled_mma_qk,
                tiled_mma_pv,
                aux_data,
                fastdiv_mods,
                num_splits,
                tidx,
            )
            if tidx == 0:
                _clc_wait_sm120(clc_mbar_ptr, clc_phase)
                clc_x, _, _, clc_valid = cute.arch.clc_response(clc_response_ptr)
                cute.arch.fence_proxy("async.shared", space="cta")
                clc_coord_ptr[0] = clc_x
                clc_coord_ptr[1] = Int32(clc_valid)
            cute.arch.barrier()
            # SM120 issues CLC directly rather than constructing the SM100
            # multicast scheduler state.
            next_tile_idx = cute.arch.grid_dim()[0]
            clc_valid_i32 = clc_coord_ptr[1]
            if clc_valid_i32 != Int32(0):
                next_tile_idx = clc_coord_ptr[0]
            batch_size = next_tile_idx // blocks_per_batch
            mh_block = next_tile_idx - batch_size * blocks_per_batch
            num_head = mh_block // num_m_blocks
            m_block = mh_block - num_head * num_m_blocks
            clc_phase ^= 1
            work_valid = clc_valid_i32 != Int32(0)
            if work_valid:
                if tidx == 0:
                    _clc_next_query_sm120(clc_mbar_ptr, clc_response_ptr)

    @cute.jit
    def process_tile_clc(
        self: cutlass.Constexpr,
        m_block,
        num_head,
        batch_size,
        split_idx,
        mQ,
        mK,
        mV,
        mO,
        mLSE,
        mCuSeqlensQ,
        mCuSeqlensK,
        mSeqUsedQ,
        mSeqUsedK,
        mPageTable,
        mNumSplitsDynamic,
        mQDescale,
        mKDescale,
        mVDescale,
        softmax_scale_log2,
        softmax_scale,
        window_size_left,
        window_size_right,
        sQ,
        sK,
        sV,
        sP,
        sVt,
        sO_layout,
        gmem_tiled_copy_Q,
        gmem_tiled_copy_K,
        gmem_tiled_copy_V,
        gmem_tiled_copy_O,
        tiled_mma_qk,
        tiled_mma_pv,
        aux_data,
        fastdiv_mods,
        num_splits,
        tidx,
        *,
        loc=None,
        ip=None,
    ):
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
            tSrQ = None
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
        tile_softmax_scale_log2 = softmax_scale_log2
        tile_softmax_scale = softmax_scale
        if const_expr(self.is_fp8):
            if const_expr(mQDescale is not None):
                qk_descale *= Float32(mQDescale[batch_size, num_head_kv])
            if const_expr(mKDescale is not None):
                qk_descale *= Float32(mKDescale[batch_size, num_head_kv])
            if const_expr(mVDescale is not None):
                v_descale = Float32(mVDescale[batch_size, num_head_kv])
            tile_softmax_scale_log2 *= qk_descale
            if const_expr(tile_softmax_scale is not None):
                tile_softmax_scale *= qk_descale

        # shape: (atom_v_m * rest_m)
        softmax = Softmax.create(
            tile_softmax_scale_log2,
            num_rows=acc_O.shape[0][0] * acc_O.shape[1],
            softmax_scale=tile_softmax_scale,
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
            cute.arch.cp_async_wait_group(self.num_stages * 2 - 1)
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
        softmax.rescale_O(mma_params.acc_O, row_scale)
        rP = cute.make_fragment_like(acc_S, self.dtype)
        rP.store(acc_S.load().to(self.dtype))
        if const_expr(self.is_fp8):
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
# ATREX PORT: SM120 (Blackwell GeForce / L20N) forward. Same SM80-era MMA, 99 KB SMEM.
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
            smem_usage_P = tile_m * (tile_n + 16)
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
