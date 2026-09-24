# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause

import math
from abc import ABC, abstractmethod
from dataclasses import dataclass, is_dataclass
from functools import partial
from typing import Literal, Tuple, Type, cast
import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import OperandMajorMode, tcgen05
from cutlass.pipeline import Agent, CooperativeGroup, NamedBarrier as nbar
from cutlass.cute.typing import BFloat16, Float16, Float32, Int8, Int32, Int64, Optional, Union

@dataclass(frozen=True)
class MaskSpec:
    has_window_left: bool = False
    has_window_right: bool = False

    @property
    def has_left_bound(self) -> bool:
        return self.has_window_left

    @property
    def has_right_bound(self) -> bool:
        return self.has_window_right

    @property
    def needs_masking(self) -> bool:
        return True

@cute.jit
def get_kv_block_range(spec: MaskSpec, blk_coord: cute.Coord, tile_shape: cute.Shape, seqlen_k: Int32, seqlen_q: Int32, window_left: Int32, window_right: Int32) -> tuple[Int32, Int32]:
    qk_offset = seqlen_k - seqlen_q
    start_block = 0
    if cutlass.const_expr(spec.has_window_left):
        first_q = blk_coord[0] * tile_shape[0] + qk_offset
        min_kv = cutlass.max(0, first_q - window_left)
        start_block = min_kv // tile_shape[1]
    last_q = (blk_coord[0] + 1) * tile_shape[0] - 1 + qk_offset
    end_elem = seqlen_k
    if cutlass.const_expr(spec.has_window_right):
        end_elem = cutlass.min(seqlen_k, last_q + window_right + 1)
    end_block = cute.ceil_div(end_elem, tile_shape[1])
    return (start_block, end_block)

@cute.jit
def get_trip_count(spec: MaskSpec, blk_coord: cute.Coord, tile_shape: cute.Shape, seqlen_k: Int32, seqlen_q: Int32, window_left: Int32, window_right: Int32) -> Int32:
    (start_block, end_block) = get_kv_block_range(spec, blk_coord, tile_shape, seqlen_k, seqlen_q, window_left, window_right)
    return end_block - start_block

@cute.jit
def get_peel_sections(spec: MaskSpec, blk_coord: cute.Coord, tile_shape: cute.Shape, seqlen_k: Int32, seqlen_q: Int32, window_left: Int32, window_right: Int32) -> tuple[Int32, Int32, Int32, Int32, Int32]:
    stage_tiler = (tile_shape[0] // 2, tile_shape[1])
    (lo0, hi0) = get_kv_block_range(spec, (blk_coord[0] * 2, blk_coord[1], blk_coord[2]), stage_tiler, seqlen_k, seqlen_q, window_left, window_right)
    (lo1, hi1) = get_kv_block_range(spec, (blk_coord[0] * 2 + 1, blk_coord[1], blk_coord[2]), stage_tiler, seqlen_k, seqlen_q, window_left, window_right)
    lo1_clamped = cutlass.min(lo1, hi0)
    head = lo1_clamped - lo0
    main = hi0 - lo1_clamped
    borrow = cutlass.min(head, cutlass.max(0, 1 - main))
    return (lo0, head - borrow, main + borrow, hi1 - hi0, lo1 - lo1_clamped + borrow)

@cute.jit
def get_stage_peel_segments(spec: MaskSpec, blk_coord: cute.Coord, stage: int, tile_shape: cute.Shape, seqlen_k: Int32, seqlen_q: Int32, window_left: Int32, window_right: Int32) -> tuple[Int32, Int32, Int32, Int32, Int32, Int32]:
    (union_start, head, _, tail, stage1_extra) = get_peel_sections(spec, blk_coord, tile_shape, seqlen_k, seqlen_q, window_left, window_right)
    (_, masked_left, unmasked, masked_right) = get_trip_segments(spec, (blk_coord[0] * 2 + stage, blk_coord[1], blk_coord[2]), (tile_shape[0] // 2, tile_shape[1]), seqlen_k, seqlen_q, window_left, window_right)
    if cutlass.const_expr(stage == 0):
        return (union_start, masked_left, unmasked, masked_right, Int32(0), tail)
    else:
        return (union_start + head, masked_left + stage1_extra, unmasked, masked_right, head, Int32(0))

@cute.jit
def get_trip_segments(spec: MaskSpec, blk_coord: cute.Coord, tile_shape: cute.Shape, seqlen_k: Int32, seqlen_q: Int32, window_left: Int32, window_right: Int32) -> tuple[Int32, Int32, Int32, Int32]:
    (start_block, end_block) = get_kv_block_range(spec, blk_coord, tile_shape, seqlen_k, seqlen_q, window_left, window_right)
    if cutlass.const_expr(not spec.needs_masking):
        return (start_block, 0, end_block - start_block, 0)
    else:
        qk_offset = seqlen_k - seqlen_q
        first_q = blk_coord[0] * tile_shape[0] + qk_offset
        last_q = first_q + tile_shape[0] - 1
        lo_max = 0
        if cutlass.const_expr(spec.has_window_left):
            lo_max = cutlass.max(0, last_q - window_left)
        hi_min = seqlen_k
        if cutlass.const_expr(spec.has_window_right):
            hi_min = cutlass.min(seqlen_k, first_q + window_right + 1)
        unmasked_start = cute.ceil_div(lo_max, tile_shape[1])
        unmasked_end = hi_min // tile_shape[1]
        unmasked_start = cutlass.min(cutlass.max(unmasked_start, start_block), end_block)
        unmasked_end = cutlass.min(cutlass.max(unmasked_end, unmasked_start), end_block)
        return (start_block, unmasked_start - start_block, unmasked_end - unmasked_start, end_block - unmasked_end)

@cute.jit
def apply_mask(spec: MaskSpec, acc_qk: cute.Tensor, index_qk: cute.Tensor, seqlen_k: Int32, causal_offset: Int32, index_qk_static: cute.Tensor, window_left: Int32, window_right: Int32) -> None:
    if cutlass.const_expr(spec.needs_masking):
        base_k = index_qk[0][1] - index_qk_static[0][1]
        row_q = index_qk[0][0] + causal_offset
        hi = seqlen_k
        if cutlass.const_expr(spec.has_window_right):
            hi = cutlass.min(row_q + window_right + 1, seqlen_k)
        hi_rel = hi - base_k
        if cutlass.const_expr(spec.has_window_left):
            lo_rel = row_q - window_left - base_k
            for i in range(cute.size(acc_qk)):
                k = index_qk_static[i][1]
                acc_qk[i] = cutlass.select_((k < lo_rel) | (k >= hi_rel), -Float32.inf, acc_qk[i])
        else:
            for i in range(cute.size(acc_qk)):
                acc_qk[i] = cutlass.select_(index_qk_static[i][1] >= hi_rel, -Float32.inf, acc_qk[i])

class AttentionMask(ABC):

    def __post_init__(self):
        assert is_dataclass(self), f'{type(self)} must be dataclass'

    def can_implement(self, seqlen_q, seqlen_kv, tile_q, tile_kv):
        return

    @cute.jit
    @abstractmethod
    def is_oob_kv(self, idx_q, idx_kv, seqlen_q, seqlen_kv) -> bool:
        ...

    @cute.jit
    @abstractmethod
    def get_range_args(self, seqlen_q, seqlen_kv, tile_q, tile_kv, num_tiles_kv, num_iters_kv, kv_splits, kv_split_idx, warpgroups_kv, warpgroup_kv_idx) -> tuple[tuple[Int32, Int32, Int32, bool], ...]:
        ...

@dataclass(frozen=True)
class DenseMask(AttentionMask):

    @cute.jit
    def is_oob_kv(self, idx_q, idx_kv, seqlen_q, seqlen_kv) -> bool:
        return idx_kv >= seqlen_kv

    @cute.jit
    def get_range_args(self, seqlen_q, seqlen_kv, tile_q, tile_kv, num_tiles_kv, num_iters_kv, kv_splits, kv_split_idx, warpgroups_kv, warpgroup_kv_idx) -> tuple[tuple[Int32, Int32, Int32, bool], ...]:
        is_last_split = kv_split_idx == (num_tiles_kv - 1) % kv_splits
        is_last_phase = warpgroup_kv_idx == (num_iters_kv - 1) % warpgroups_kv
        unmasked_start = warpgroup_kv_idx
        unmasked_end = num_iters_kv
        unmasked_step = warpgroups_kv
        masked_start = masked_end = masked_step = 0
        if seqlen_kv % tile_kv != 0 and is_last_split and is_last_phase:
            unmasked_end -= 1
            masked_start = num_tiles_kv - 1
            masked_end = num_tiles_kv
            masked_step = 1
        return ((unmasked_start, unmasked_end, unmasked_step, False), (masked_start, masked_end, masked_step, True))

@dataclass(frozen=True)
class CausalMask(AttentionMask):

    def can_implement(self, seqlen_q, seqlen_kv, tile_q, tile_kv):
        if seqlen_q > tile_kv:
            raise ValueError(f'seqlen_q({seqlen_q}) with causal mask can be at most tile_kv({tile_kv})')

    @cute.jit
    def is_oob_kv(self, idx_q, idx_kv, seqlen_q, seqlen_kv) -> bool:
        idx_current = seqlen_kv - seqlen_q + idx_q
        return idx_kv > idx_current

    @cute.jit
    def get_range_args(self, seqlen_q, seqlen_kv, tile_q, tile_kv, num_tiles_kv, num_iters_kv, kv_splits, kv_split_idx, warpgroups_kv, warpgroup_kv_idx) -> tuple[tuple[Int32, Int32, Int32, bool], ...]:
        is_last_split = kv_split_idx == (num_tiles_kv - 1) % kv_splits
        is_prev_split = kv_split_idx == (num_tiles_kv - 2) % kv_splits
        is_last_phase = warpgroup_kv_idx == (num_iters_kv - 1) % warpgroups_kv
        is_prev_phase = warpgroup_kv_idx == (num_iters_kv - 2) % warpgroups_kv
        unmasked_start = warpgroup_kv_idx
        unmasked_end = num_iters_kv
        unmasked_step = warpgroups_kv
        masked_start = masked_end = masked_step = 0
        if 0 < seqlen_kv % tile_kv < seqlen_q:
            if kv_splits == 1:
                unmasked_end -= 2
                masked_start = num_tiles_kv - 2 + (0 if is_prev_phase else 1)
                masked_end = num_tiles_kv
                masked_step = warpgroups_kv
            elif (is_last_split or is_prev_split) and is_last_phase:
                unmasked_end -= 1
                masked_start = num_tiles_kv - 1 if is_last_split else num_tiles_kv - 2
                masked_end = num_tiles_kv
                masked_step = 2
        elif seqlen_q > 1 or seqlen_kv % tile_kv != 0:
            if is_last_split and is_last_phase:
                unmasked_end -= 1
                masked_start = num_tiles_kv - 1
                masked_end = num_tiles_kv
                masked_step = 1
        return ((unmasked_start, unmasked_end, unmasked_step, False), (masked_start, masked_end, masked_step, True))
mma_modes = (0, 1, 2)
mma_dice = (None, None, None)
warp_threads = 32
warpgroup_warps = 4
warpgroup_threads = 128
max_reduction_iters = 4
max_reduction_rows = 8
min_reduction_ctas_per_request = 16
min_f32 = Float32(-3.4028234663852886e+38)
log2_e = math.log2(math.e)
exp2 = partial(cute.math.exp2, fastmath=True)
warp_fmax = partial(cute.arch.warp_redux_sync, kind='fmax', nan=True)
smem_fmax = partial(cute.arch.atomic_fmax, sem='relaxed', scope='cta')

class GroupedQueryAttentionDecode:

    @staticmethod
    def gqa_pack(t_bshd: cute.Tensor, h_k: int):
        (d, h_q, s_q, b) = tuple(reversed(t_bshd.shape))[:4]
        (stride_d, stride_h, stride_s, stride_b) = tuple(reversed(t_bshd.stride))[:4]
        has_partial = cute.rank(t_bshd) == 5
        b_partial = b * t_bshd.shape[0] if has_partial else b
        h_g = h_q // h_k
        gqa_shape = (b_partial, (h_g, s_q), h_k, d)
        gqa_stride = (stride_b, (stride_h, stride_s), stride_h * h_g, stride_d)
        gqa_layout = cute.make_layout(gqa_shape, stride=gqa_stride)
        return cute.make_tensor(t_bshd.iterator, gqa_layout)

    @staticmethod
    def gemm_view(t_bshd: cute.Tensor, s_first: bool):
        sdhb = (1, 3, 2, 0)
        dshb = (3, 1, 2, 0)
        reorder = sdhb if s_first else dshb
        mT_layout = cute.select(t_bshd.layout, reorder)
        mT_layout = cute.group_modes(mT_layout, 2, 4)
        return cute.make_tensor(t_bshd.iterator, mT_layout)

    @staticmethod
    def gemm_view_bsh(t_bsh: cute.Tensor, h_k: int):
        (h_q, s_q, b) = tuple(reversed(t_bsh.shape))[:3]
        (stride_h, stride_s, stride_b) = tuple(reversed(t_bsh.stride))[:3]
        h_g = h_q // h_k
        mT_shape = ((h_g, s_q), (h_k, b))
        mT_stride = ((stride_h, stride_s), (stride_h * h_g, stride_b))
        has_partial = cute.rank(t_bsh) == 4
        mT_shape += (t_bsh.shape[0],) if has_partial else ()
        mT_stride += (t_bsh.stride[0],) if has_partial else ()
        mT_layout = cute.make_layout(mT_shape, stride=mT_stride)
        return cute.make_tensor(t_bsh.iterator, mT_layout)

    @staticmethod
    @cute.jit
    def reduction_none(lane_store_max: bool, lane_idx: Int32, sM_final_nbar: nbar, sL_final_nbar: nbar, sM: cute.Tensor, sL: cute.Tensor, gL: Optional[cute.Tensor], sSink: Optional[cute.Tensor], scale_o: Float32):
        store_lse = gL is not None
        colmax = Float32(0)
        colsum = Float32(0)
        sM_final_nbar.arrive_and_wait()
        if cutlass.const_expr(store_lse):
            if lane_store_max:
                colmax = sM[lane_idx]
        sL_final_nbar.arrive_and_wait()
        if lane_store_max:
            sL_lane = sL[lane_idx, None]
            colsum = sL_lane[0] + sL_lane[1] + sL_lane[2] + sL_lane[3]
            if cutlass.const_expr(sSink is not None):
                colsum += exp2(log2_e * sSink[lane_idx] - colmax)
            normalization = cute.arch.rcp_approx(colsum) * scale_o
            sM[lane_idx] = normalization
        sM_final_nbar.arrive()
        if cutlass.const_expr(store_lse):
            if lane_store_max:
                gL[lane_idx] = colmax + cute.math.log2(colsum)

    @staticmethod
    @cute.jit
    def reduction_epilogue(blk_tile_hp: Tuple[int, int], coord_hp: Tuple[Int32, Int32], coord_hb: Tuple[Int32, Int32], kv_split_idx: Int32, kv_splits: Int32, metadata_idx: Int32, store_partial: bool, lane_idx: Int32, sM_final_nbar: nbar, sL_final_nbar: nbar, sM: cute.Tensor, sL: cute.Tensor, mM_partial: cute.Tensor, mL_partial: cute.Tensor):
        coord_h = (coord_hp, coord_hb, kv_split_idx)
        gM_partial = cute.local_tile(mM_partial, (blk_tile_hp,), coord_h)
        gL_partial = cute.local_tile(mL_partial, (blk_tile_hp,), coord_h)
        (blk_tile_h, blk_tile_p) = blk_tile_hp
        blk_tile_n = blk_tile_h * blk_tile_p
        lane_store_max = blk_tile_n == warp_threads or lane_idx < blk_tile_n
        (grouped_heads, prediction) = mM_partial.shape[0]
        cM = cute.make_identity_tensor(mM_partial.shape[0])
        cM = cute.local_tile(cM, blk_tile_hp, coord_hp)
        (idx_hg, idx_p) = cM[lane_idx]
        lane_store_max &= idx_hg < grouped_heads
        lane_store_max &= idx_p < prediction
        lane_store_max &= store_partial
        cute.arch.fence_acq_rel_cta()
        sM_final_nbar.arrive_and_wait()
        if lane_store_max:
            gM_partial[lane_idx] = sM[lane_idx]
        if lane_idx == 0 and kv_split_idx == 0 and coord_hp[0] == 0 and coord_hp[1] == 0 and coord_hb[0] == 0:
            mM_partial[(0, 0), (0, coord_hb[1]), metadata_idx] = Float32(kv_splits)
        sL_final_nbar.arrive_and_wait()
        if lane_store_max:
            sL_lane_wg = sL[lane_idx, None]
            gL_partial[lane_idx] = sL_lane_wg[0] + sL_lane_wg[1] + sL_lane_wg[2] + sL_lane_wg[3]

    @staticmethod
    @cute.jit
    def reduction_cluster(blk_tile_n: int, kv_splits: Int32, kv_split_idx: Int32, lane_idx: Int32, sM_final_nbar: nbar, sL_final_nbar: nbar, reduction_mbars_ptr: cute.Pointer, sM: cute.Tensor, sL: cute.Tensor, sR: cute.Tensor, gL: Optional[cute.Tensor], sSink: Optional[cute.Tensor], scale_o: Float32):
        acc_dtype = sM.dtype
        colmax_bits = blk_tile_n * acc_dtype.width
        copy_vec_bits = min(colmax_bits, 128)
        dsmem_store_threads = colmax_bits // copy_vec_bits
        dsmem_store_values = copy_vec_bits // acc_dtype.width
        dsmem_store_atom_r = cute.make_copy_atom(cute.nvgpu.cpasync.CopyDsmemStoreOp(), acc_dtype, num_bits_per_copy=copy_vec_bits)
        dsmem_store_r = cute.make_tiled_copy(dsmem_store_atom_r, cute.make_ordered_layout((dsmem_store_threads, dsmem_store_values), order=(1, 0)), (blk_tile_n,))
        thr_store_r = dsmem_store_r.get_slice(lane_idx)
        tRsM = thr_store_r.partition_S(sM)
        tRsL = thr_store_r.partition_S(sL)
        tRsR = thr_store_r.partition_S(sR)
        tRrM_shape = thr_store_r.partition_D(sM).shape
        tRrM_final = cute.make_rmem_tensor(tRrM_shape, acc_dtype)
        tRrM_prev = cute.make_rmem_tensor(tRrM_shape, acc_dtype)
        tRrL_final = cute.make_rmem_tensor(tRrM_shape, acc_dtype)
        cute.arch.fence_acq_rel_cta()
        sM_final_nbar.arrive_and_wait()
        is_reduction_lane = lane_idx < dsmem_store_threads
        if is_reduction_lane:
            tRrM_prev.store(tRsM.load())
            tRrM_final.store(tRrM_prev.load())
            for i in cutlass.range_constexpr(max_reduction_iters):
                xor_mask = 1 << i
                if xor_mask < kv_splits:
                    peer_idx = kv_split_idx ^ xor_mask
                    tRsR_local = tRsR[None, None, i, 0]
                    tRsR_peer = cute.make_tensor(cute.arch.map_dsmem_ptr(tRsR_local.iterator, peer_idx), tRsR_local.layout)
                    local_mbar = reduction_mbars_ptr + i
                    peer_mbar = cute.arch.map_dsmem_ptr(local_mbar, peer_idx)
                    cute.copy(dsmem_store_atom_r, tRrM_final, tRsR_peer, mbar_ptr=peer_mbar)
                    cute.arch.fence_acq_rel_cta()
                    cute.arch.mbarrier_wait(local_mbar, phase=0)
                    tRrR = tRsR_local.load()
                    for j in cutlass.range_constexpr(cute.size(tRrM_final)):
                        tRrM_final[j] = cute.arch.fmax(tRrM_final[j], tRrR[j])
        sL_final_nbar.arrive_and_wait()
        if is_reduction_lane:
            colsum = tRsL[None, None, 0].load()
            for i in cutlass.range_constexpr(1, warpgroup_warps, 1):
                colsum += tRsL[None, None, i].load()
            correction = exp2(tRrM_prev.load() - tRrM_final.load())
            correction = correction.reshape(colsum.shape)
            colsum *= correction
            for i in cutlass.range_constexpr(max_reduction_iters):
                xor_mask = 1 << i
                if xor_mask < kv_splits:
                    peer_idx = kv_split_idx ^ xor_mask
                    tRrL_local = cute.make_rmem_tensor(tRrM_shape, acc_dtype)
                    tRrL_local.store(colsum)
                    tRsR_local = tRsR[None, None, i, 1]
                    tRsR_peer = cute.make_tensor(cute.arch.map_dsmem_ptr(tRsR_local.iterator, peer_idx), tRsR_local.layout)
                    local_mbar = reduction_mbars_ptr + max_reduction_iters + i
                    peer_mbar = cute.arch.map_dsmem_ptr(local_mbar, peer_idx)
                    cute.copy(dsmem_store_atom_r, tRrL_local, tRsR_peer, mbar_ptr=peer_mbar)
                    cute.arch.fence_acq_rel_cta()
                    cute.arch.mbarrier_wait(local_mbar, phase=0)
                    colsum += tRsR_local.load()
            if cutlass.const_expr(sSink is not None):
                tRsSink = thr_store_r.partition_S(sSink)
                tRrSink = tRsSink.load().reshape(tRrM_final.shape)
                sink_prob = exp2(log2_e * tRrSink - tRrM_final.load())
                colsum += sink_prob.reshape(colsum.shape)
            rcp_colsum = cute.make_rmem_tensor(colsum.shape, acc_dtype)
            for i in cutlass.range(cute.size(colsum.shape)):
                rcp_colsum[i] = cute.arch.rcp_approx(colsum[i])
            tRsM.store(correction * rcp_colsum.load() * scale_o)
            if cutlass.const_expr(gL is not None):
                tRrL_final.store(colsum)
        sM_final_nbar.arrive()
        if cutlass.const_expr(gL is not None):
            tRgL = thr_store_r.partition_D(gL)
            if kv_split_idx == 0 and is_reduction_lane:
                lse = tRrM_final.load() + cute.math.log2(tRrL_final.load())
                tRgL.store(lse)

    @staticmethod
    @cute.jit
    def launch_reduction(d_per_blk: int, o_bshd: cute.Tensor, l_bsh: Optional[cute.Tensor], o_partial_bshd: cute.Tensor, l_partial_bsh: cute.Tensor, m_partial_bsh: cute.Tensor, sink_h: Optional[cute.Tensor], scale_o: Float32, stream: cuda.CUstream, enable_pdl: bool=True):
        (splits, b, s_q, h_q, d) = o_partial_bshd.shape

        def reverse(t: cute.Tensor):
            modes = tuple(reversed(range(cute.rank(t))))
            layout = cute.select(t.layout, modes)
            return cute.make_tensor(t.iterator, layout)
        o_dhsb = reverse(o_bshd)
        l_hsb = reverse(l_bsh) if l_bsh is not None else None
        o_partial_dhsb = reverse(o_partial_bshd)
        l_partial_hsb = reverse(l_partial_bsh)
        m_partial_hsb = reverse(m_partial_bsh)
        d_per_thr = 128 // o_bshd.dtype.width
        thr_per_row = d_per_blk // d_per_thr
        rows = h_q * s_q
        rows_per_blk = math.gcd(rows, min(max_reduction_rows, max(1, rows // min_reduction_ctas_per_request)))
        thr_per_blk = thr_per_row * rows_per_blk
        d_blks = cute.ceil_div(d, d_per_blk)
        smem_bytes = rows_per_blk * (splits * 2 + 1) * Float32.width // 8
        GroupedQueryAttentionDecode.reduction_kernel((thr_per_row, d_per_thr, d_per_blk, rows_per_blk), o_dhsb, l_hsb, o_partial_dhsb, l_partial_hsb, m_partial_hsb, sink_h, scale_o).launch(grid=[d_blks, rows // rows_per_blk, b], block=[thr_per_blk, 1, 1], cluster=[1, 1, 1], stream=stream, smem=smem_bytes, min_blocks_per_mp=1, use_pdl=enable_pdl)

    @staticmethod
    @cute.kernel
    def reduction_kernel(tile_d: cute.Tile, o_dhsb: cute.Tensor, l_hsb: Optional[cute.Tensor], o_partial_dhsb: cute.Tensor, l_partial_hsb: cute.Tensor, m_partial_hsb: cute.Tensor, sink_h: Optional[cute.Tensor], scale_o: Float32):
        (thr_per_row, d_per_thr, d_per_blk, rows_per_blk) = tile_d
        (d, h_q, s_q, b, splits) = o_partial_dhsb.shape
        (d_blk_idx, coord_row_blk, coord_b) = cute.arch.block_idx()
        (tidx, _, _) = cute.arch.thread_idx()
        row_idx = tidx // thr_per_row
        row_tidx = tidx % thr_per_row
        (coord_h, coord_s) = cute.idx2crd(coord_row_blk * rows_per_blk + row_idx, (h_q, s_q))
        not_oob_d = True
        if d % d_per_blk != 0:
            not_oob_d = d_blk_idx * d_per_blk + row_tidx * d_per_thr < d
        coord_o = (d_blk_idx, coord_h, coord_s, coord_b, None)
        gO = cute.local_tile(o_dhsb, (d_per_blk,), coord_o[:-1])
        gO_partial = cute.local_tile(o_partial_dhsb, (d_per_blk,), coord_o)
        gM_partial = cute.local_tile(m_partial_hsb, (1,), coord_o[1:])
        gL_partial = cute.local_tile(l_partial_hsb, (1,), coord_o[1:])
        gM_partial_0 = gM_partial[None, row_tidx]
        gL_partial_0 = gL_partial[None, row_tidx]
        smem_ptr = cute.arch.get_dyn_smem(Float32)
        partial_layout = cute.make_layout((1, splits))
        sM_partial = cute.make_tensor(smem_ptr + row_idx * splits, partial_layout)
        sL_partial = cute.make_tensor(smem_ptr + (rows_per_blk + row_idx) * splits, partial_layout)
        sL_partial_0 = sL_partial[None, row_tidx]
        sM_partial_0 = sM_partial[None, row_tidx]
        scalar_layout = cute.make_layout(1)
        use_sink = sink_h is not None
        if cutlass.const_expr(use_sink):
            gSink = cute.local_tile(sink_h, (1,), (coord_h,))
            sSink = cute.make_tensor(smem_ptr + rows_per_blk * splits * 2 + row_idx, scalar_layout)
        cpasync_atom = cute.make_copy_atom(cute.nvgpu.cpasync.CopyG2SOp(), Float32, num_bits_per_copy=32)
        copy_atom = cute.make_copy_atom(cute.nvgpu.CopyUniversalOp(), Float32)
        tv_layout = cute.make_ordered_layout((thr_per_row, d_per_thr), order=(1, 0))
        tiled_copy = cute.make_tiled_copy(copy_atom, tv_layout, (d_per_blk,))
        thr_copy = tiled_copy.get_slice(row_tidx)
        tCgO_partial = thr_copy.partition_S(gO_partial)
        tCgO_partial = tCgO_partial[None, 0, None]
        tCgO = thr_copy.partition_D(gO)
        tCgO = tCgO[None, 0]
        tCrO_final = cute.zeros_like(tCgO, Float32)
        cute.arch.fence_acq_rel_cta()
        cute.arch.griddepcontrol_wait()
        splits_b = Int32(m_partial_hsb[0, 0, coord_b, splits - 1])
        if cutlass.const_expr(use_sink):
            if row_tidx == 0:
                cute.copy(cpasync_atom, gSink, sSink)
        if row_tidx < splits_b:
            cute.copy(cpasync_atom, gL_partial_0, sL_partial_0)
            cute.copy(cpasync_atom, gM_partial_0, sM_partial_0)
        for split_idx in cutlass.range(thr_per_row + row_tidx, splits_b, thr_per_row):
            gL_partial_n = gL_partial[None, split_idx]
            sL_partial_n = sL_partial[None, split_idx]
            cute.copy(cpasync_atom, gL_partial_n, sL_partial_n)
            gM_partial_n = gM_partial[None, split_idx]
            sM_partial_n = sM_partial[None, split_idx]
            cute.copy(cpasync_atom, gM_partial_n, sM_partial_n)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.sync_threads()
        max_final = -Float32.inf
        for split_idx in cutlass.range(splits_b, unroll=8):
            max_final = cute.arch.fmax(max_final, sM_partial[0, split_idx])
        sum_final = Float32(0)
        if cutlass.const_expr(use_sink):
            sum_final = exp2(log2_e * sSink[0] - max_final)
        if max_final > -Float32.inf and not_oob_d:
            for split_idx in cutlass.range(splits_b, unroll=8):
                max_partial = sM_partial[0, split_idx]
                if max_partial > -Float32.inf:
                    correction = exp2(max_partial - max_final)
                    sum_final += correction * sL_partial[0, split_idx]
                    tCrO_final += correction * tCgO_partial[None, split_idx].load()
            tCrO_final *= cute.arch.rcp_approx(sum_final) * scale_o
        cute.arch.griddepcontrol_launch_dependents()
        if not_oob_d:
            tCgO.store(tCrO_final.to(o_dhsb.dtype))
        if cutlass.const_expr(l_hsb is not None):
            if d_blk_idx == 0 and row_tidx == 0:
                l_hsb[coord_h, coord_s, coord_b] = max_final + cute.math.log2(sum_final)
        return
GqaDecode = GroupedQueryAttentionDecode
warp_or = partial(cute.arch.warp_redux_sync, kind='or')
debug_blasst = False
gmem_add = partial(cute.arch.atomic_add, sem='relaxed', scope='gpu')

@cute.jit
def flatten_split_work_single_warp(seqlens: cute.Tensor, batches: Int32, sequence_tile: Int32, split_target: Int32, max_splits: Int32, cta_idx: Int32, lane_idx: Int32) -> tuple[bool, Int32, Int32, Int32, Int32]:
    lane_seqlen = Int32(0)
    if lane_idx < batches:
        lane_seqlen = seqlens[lane_idx]
    total_tiles = cute.ceil_div(lane_seqlen, sequence_tile)
    for i in cutlass.range_constexpr(5):
        total_tiles += cute.arch.shuffle_sync_bfly(total_tiles, offset=1 << i)
    span = cutlass.max(Int32(1), cute.ceil_div(total_tiles, split_target))
    lane_splits = Int32(0)
    if lane_idx < batches:
        lane_splits = cutlass.min(max_splits, cute.ceil_div(cute.ceil_div(lane_seqlen, sequence_tile), span))
    inclusive_splits = lane_splits
    for i in cutlass.range_constexpr(5):
        offset = 1 << i
        peer_splits = cute.arch.shuffle_sync(inclusive_splits, cutlass.max(Int32(0), lane_idx - offset))
        if lane_idx >= offset:
            inclusive_splits += peer_splits
    lane_start = inclusive_splits - lane_splits
    found = (lane_splits > 0) & (cta_idx >= lane_start) & (cta_idx < inclusive_splits)
    return (found, lane_idx, cta_idx - lane_start, lane_splits, lane_seqlen)

@cute.jit
def flatten_split_work(seqlens: cute.Tensor, batches: Int32, sequence_tile: Int32, split_target: Int32, max_splits: Int32, cta_idx: Int32, lane_idx: Int32) -> tuple[bool, Int32, Int32, Int32, Int32]:
    total_tiles = Int32(0)
    for base in cutlass.range(0, batches, warp_threads):
        lane_tiles = Int32(0)
        if base + lane_idx < batches:
            lane_tiles = cute.ceil_div(seqlens[base + lane_idx], sequence_tile)
        for i in cutlass.range_constexpr(5):
            lane_tiles += cute.arch.shuffle_sync_bfly(lane_tiles, offset=1 << i)
        total_tiles += lane_tiles
    span = cutlass.max(Int32(1), cute.ceil_div(total_tiles, split_target))
    acc = Int32(0)
    found = Int32(0)
    coord_b = Int32(0)
    split_idx = Int32(0)
    splits = Int32(1)
    seqlen = Int32(0)
    for base in cutlass.range(0, batches, warp_threads):
        lane_seqlen = Int32(0)
        lane_splits = Int32(0)
        if base + lane_idx < batches:
            lane_seqlen = seqlens[base + lane_idx]
            lane_splits = cutlass.min(max_splits, cute.ceil_div(cute.ceil_div(lane_seqlen, sequence_tile), span))
        inclusive_splits = lane_splits
        for i in cutlass.range_constexpr(5):
            offset = 1 << i
            peer_splits = cute.arch.shuffle_sync(inclusive_splits, cutlass.max(Int32(0), lane_idx - offset))
            if lane_idx >= offset:
                inclusive_splits += peer_splits
        lane_start = acc + inclusive_splits - lane_splits
        lane_end = acc + inclusive_splits
        if found == 0 and lane_splits > 0 and cta_idx >= lane_start and cta_idx < lane_end:
            coord_b = base + lane_idx
            split_idx = cta_idx - lane_start
            splits = lane_splits
            seqlen = lane_seqlen
            found = Int32(1)
        acc += cute.arch.shuffle_sync(inclusive_splits, warp_threads - 1)
    return (found != 0, coord_b, split_idx, splits, seqlen)

class GroupedQueryAttentionDecodePaged:

    def __init__(self, page_size, headdim, grouped_head_tile, prediction_tile=1, sequence_tile=256, reduction_mode: Literal['kernel', 'atomic', 'none']='kernel', softmax_warpgroups=1, table_page_size=None, cluster_kv=1, single_warp_batch=False):
        self.headdim = headdim
        self.grouped_head_tile = grouped_head_tile
        self.page_size = page_size
        self.table_page_size = page_size if table_page_size is None else table_page_size
        self.subpages_per_table_page = self.table_page_size // self.page_size
        self.prediction_tile = prediction_tile
        self.sequence_tile = sequence_tile
        self.do_kernel_red = reduction_mode == 'kernel'
        self.do_atomic_red = reduction_mode == 'atomic'
        self.do_none_red = reduction_mode == 'none' or reduction_mode is None
        self.softmax_warpgroups = softmax_warpgroups
        self.cluster_kv = cluster_kv
        self.single_warp_batch = single_warp_batch
        self.threads_per_cta = (2 + softmax_warpgroups) * warpgroup_threads
        assert headdim > 0 and headdim % 64 == 0
        assert grouped_head_tile * prediction_tile in (1, 2, 4, 8, 16, 32)
        assert sequence_tile > 0 and sequence_tile % 128 == 0
        assert page_size in (8, 16, 32, 64)
        assert self.table_page_size >= page_size
        assert self.table_page_size % page_size == 0
        assert self.softmax_warpgroups in (1, 2)
        assert cluster_kv in (1, 2, 4)
        assert self.do_kernel_red ^ self.do_atomic_red ^ self.do_none_red

    def can_implement(self, kv_splits, qo_shape, kv_shape, qkv_dtype, o_dtype, mask_config, threshold_scale_factor):
        GqaDecode.can_implement(self, kv_splits, qo_shape, kv_shape, qkv_dtype, o_dtype, mask_config)
        if threshold_scale_factor is not None and (not threshold_scale_factor > 0):
            raise ValueError(f'threshold_scale_factor must be None or > 0, got {threshold_scale_factor}')

    @cute.jit
    def __call__(self, split_target: Int32, seqlens: Union[cute.Tensor, Int32], table_stride: Int32, page_table: cute.Tensor, k_bshd: cute.Tensor, v_bshd: cute.Tensor, q_bshd: cute.Tensor, o_bshd: cute.Tensor, l_bsh: Optional[cute.Tensor], o_partial_bshd: Optional[cute.Tensor], l_partial_bsh: Optional[cute.Tensor], m_partial_bsh: Optional[cute.Tensor], sink_h: Optional[cute.Tensor], mask_config, scale_s: Float32, scale_o: Float32, threshold_scale_factor: Optional[Float32], stream: cuda.CUstream, enable_pdl: bool=True):
        assert not self.do_atomic_red, 'cluster-atomic reduction cannot express per-request split counts'
        mma_dtype = q_bshd.dtype
        acc_dtype = Float32
        assert k_bshd.dtype == v_bshd.dtype == mma_dtype
        blk_tile_s = self.sequence_tile
        blk_tile_h = self.grouped_head_tile
        blk_tile_p = self.prediction_tile
        blk_tile_d = self.headdim
        blk_tile_shpd = (blk_tile_s, blk_tile_h, blk_tile_p, blk_tile_d)
        mma_tile_m = 128
        mma_tile_k = 128 * 8 // mma_dtype.width
        min_mma_tile_n = 16 if mma_dtype.width == 8 else 8
        blk_tile_n = blk_tile_h * blk_tile_p
        mma_tile_n = max(min_mma_tile_n, blk_tile_n)
        mma_tile_mnk = (mma_tile_m, mma_tile_n, mma_tile_k)
        tiles_sm = blk_tile_s // mma_tile_m
        tiles_dm = math.ceil(blk_tile_d / mma_tile_m)
        tiles_dk = math.ceil(blk_tile_d / mma_tile_k)
        pages_s = blk_tile_s // self.page_size
        assert blk_tile_s % mma_tile_m == 0
        assert mma_tile_n % blk_tile_n == 0
        tiled_mma_kq = sm100_utils.make_trivial_tiled_mma(mma_dtype, mma_dtype, OperandMajorMode.K, OperandMajorMode.K, acc_dtype, tcgen05.CtaGroup.ONE, mma_tile_mnk[:2])
        tiled_mma_vp = sm100_utils.make_trivial_tiled_mma(mma_dtype, mma_dtype, OperandMajorMode.MN, OperandMajorMode.MN, acc_dtype, tcgen05.CtaGroup.ONE, mma_tile_mnk[:2])
        self.pt_stages = pt_stages = 4
        self.sp_stages = sp_stages = 4
        self.p_stages = p_stages = 4
        self.o_stages = o_stages = 2
        tmem_capacity_cols = cute.arch.get_max_tmem_alloc_cols('sm_100')
        tmem_s_stage_cols = tiles_sm * mma_tile_n
        tmem_alloc_cols = mma_tile_n * o_stages
        tmem_alloc_cols += tiles_dm * mma_tile_n * o_stages
        max_s_stages = (tmem_capacity_cols - tmem_alloc_cols) // tmem_s_stage_cols
        self.s_stages = s_stages = min(max_s_stages, p_stages)
        tmem_alloc_cols += tmem_s_stage_cols * s_stages
        tmem_alloc_cols = 2 ** math.ceil(math.log2(tmem_alloc_cols))
        self.tmem_alloc_cols = tmem_alloc_cols
        assert tmem_alloc_cols <= tmem_capacity_cols
        smem_alloc_bits = 0
        mbarrier_bits = Int64.width
        pipe_stage_bits = mbarrier_bits * 2
        mk_stage_bits = mma_tile_m * mma_tile_k * mma_dtype.width
        nk_stage_bits = mma_tile_n * mma_tile_k * mma_dtype.width
        mn_stage_bits = mma_tile_m * mma_tile_n * mma_dtype.width
        smem_alloc_bits += Int32.width
        assert isinstance(seqlens, cute.Tensor), 'per-request split mapping requires a request-length tensor'
        smem_alloc_bits += Int32.width * 4
        smem_alloc_bits += pt_stages * pages_s * Int32.width
        if cutlass.const_expr(threshold_scale_factor is not None):
            smem_alloc_bits += sp_stages * (Int32.width + pipe_stage_bits)
        smem_alloc_bits += blk_tile_n * acc_dtype.width
        smem_alloc_bits += blk_tile_n * warpgroup_warps * acc_dtype.width
        if cutlass.const_expr(self.do_atomic_red):
            smem_alloc_bits += max_reduction_iters * blk_tile_n * acc_dtype.width * 2
            smem_alloc_bits += max_reduction_iters * mbarrier_bits * 2
        smem_alloc_bits += tiles_dk * nk_stage_bits + mbarrier_bits
        smem_alloc_bits += s_stages * pipe_stage_bits
        smem_alloc_bits += p_stages * (tiles_sm * mn_stage_bits + pipe_stage_bits)
        smem_alloc_bits += o_stages * pipe_stage_bits
        alignment_bits = 1024 - smem_alloc_bits % 1024
        smem_capacity_bits = utils.get_smem_capacity_in_bytes('sm_100') * 8
        remaining_bits = smem_capacity_bits - smem_alloc_bits - alignment_bits
        kv_stages = remaining_bits // mk_stage_bits
        kv_stages -= 1 if kv_stages * pipe_stage_bits > alignment_bits else 0
        h_k = k_bshd.shape[2]
        o_bshd_ = o_partial_bshd if self.do_kernel_red else o_bshd
        mQ_nkl = GqaDecode.gemm_view(GqaDecode.gqa_pack(q_bshd, h_k), True)
        mK_mkl = GqaDecode.gemm_view(k_bshd, True)
        mV_mkl = GqaDecode.gemm_view(v_bshd, False)
        mO_mnl = GqaDecode.gemm_view(GqaDecode.gqa_pack(o_bshd_, h_k), False)
        smem_layout_q = sm100_utils.make_smem_layout_b(tiled_mma_kq, mma_tile_mnk, mma_dtype, tiles_dk)
        smem_layout_k_mma = sm100_utils.make_smem_layout_a(tiled_mma_kq, mma_tile_mnk, mma_dtype, kv_stages)
        smem_layout_v_mma = sm100_utils.make_smem_layout_a(tiled_mma_vp, mma_tile_mnk, mma_dtype, kv_stages)
        smem_layout_k_mk = cute.composition(smem_layout_k_mma, cute.make_layout((mma_tile_m, mma_tile_k, kv_stages)))
        smem_layout_v_mk = cute.composition(smem_layout_v_mma, cute.make_layout((mma_tile_m, mma_tile_k, kv_stages)))
        smem_layout_k_tma = cute.tiled_divide(smem_layout_k_mk, (self.page_size, mma_tile_k))
        smem_layout_k_tma = cute.select(smem_layout_k_tma, [0, 1, 3])
        smem_layout_v_tma = cute.tiled_divide(smem_layout_v_mk, (mma_tile_m, self.page_size))
        smem_layout_v_tma = cute.select(smem_layout_v_tma, [0, 2, 3])
        o_smem_dtype = mO_mnl.dtype
        smem_layout_atom_o = tcgen05.make_smem_layout_atom(tcgen05.mma.SmemLayoutAtomKind.MN_SW128, o_smem_dtype)
        smem_layout_o = cute.tile_to_shape(smem_layout_atom_o, (max(blk_tile_d, mma_tile_m), mma_tile_n), order=(1, 0))
        smem_layout_o = cute.flat_divide(smem_layout_o, (mma_tile_m, mma_tile_n))
        tma_load_op = cute.nvgpu.cpasync.CopyBulkTensorTileG2SOp()
        tma_store_op = cute.nvgpu.cpasync.CopyReduceBulkTensorTileS2GOp() if self.do_atomic_red else cute.nvgpu.cpasync.CopyBulkTensorTileS2GOp()
        tma_tile_n = (blk_tile_h, mma_tile_n // blk_tile_h)
        tma_tile_mnk = (mma_tile_m, tma_tile_n, mma_tile_k)
        (tma_atom_q, tma_tensor_q) = cute.nvgpu.make_tiled_tma_atom_B(tma_load_op, mQ_nkl, cute.select(smem_layout_q, mma_modes), tma_tile_mnk, tiled_mma_kq)
        (tma_atom_k, tma_tensor_k) = cute.nvgpu.cpasync.make_tiled_tma_atom(tma_load_op, mK_mkl, smem_layout_k_tma[0], (self.page_size, mma_tile_k))
        (tma_atom_v, tma_tensor_v) = cute.nvgpu.cpasync.make_tiled_tma_atom(tma_load_op, mV_mkl, smem_layout_v_tma[0], (mma_tile_m, self.page_size))
        (tma_atom_o, tma_tensor_o) = cute.nvgpu.cpasync.make_tiled_tma_atom(tma_store_op, mO_mnl, cute.select(smem_layout_o, mode=[0, 1]), tma_tile_mnk[:2])
        mL_nl = None if l_bsh is None else GqaDecode.gemm_view_bsh(l_bsh, h_k)
        assert l_bsh is None or l_bsh.dtype == acc_dtype
        mM_partial_nl = mL_partial_nl = None
        if cutlass.const_expr(self.do_kernel_red):
            assert m_partial_bsh.dtype == l_partial_bsh.dtype == o_partial_bshd.dtype == acc_dtype
            mM_partial_nl = GqaDecode.gemm_view_bsh(m_partial_bsh, h_k)
            mL_partial_nl = GqaDecode.gemm_view_bsh(l_partial_bsh, h_k)
        if cutlass.const_expr(sink_h is not None):
            assert sink_h.dtype == acc_dtype
            h_g = sink_h.shape[0] // h_k
            mSink = cute.make_tensor(sink_h.iterator, cute.make_layout((h_g, h_k), stride=(1, h_g)))
        else:
            mSink = None
        scale_s_log2_e = scale_s * log2_e
        enable_blasst = threshold_scale_factor is not None
        log2_threshold_scale_factor = Float32(cute.math.log2(threshold_scale_factor)) if enable_blasst else None
        n_tiles = cute.ceil_div(mQ_nkl.shape[0], (blk_tile_h, blk_tile_p))
        (h_k_count, batches) = mQ_nkl.shape[2]
        max_splits = Int32(1) if self.do_none_red else Int32(o_partial_bshd.shape[0]) - Int32(1)
        grid_y = cute.size(n_tiles)
        grid_z = cute.size(h_k_count)
        grid_x = batches if self.do_none_red else split_target + batches
        grid = (grid_y, grid_x, grid_z)
        assert grid_y % self.cluster_kv == 0, f'head/prediction tiles {grid_y} must be divisible by kv cluster {self.cluster_kv}'
        self.decode(blk_tile_shpd, mma_tile_mnk, tiled_mma_kq, tiled_mma_vp, mma_dtype, o_smem_dtype, seqlens, table_stride, max_splits, page_table.iterator, smem_layout_k_mma, smem_layout_k_tma, tma_atom_k, tma_tensor_k, smem_layout_v_mma, smem_layout_v_tma, tma_atom_v, tma_tensor_v, smem_layout_q, tma_atom_q, tma_tensor_q, smem_layout_o, tma_atom_o, tma_tensor_o, mL_nl, mL_partial_nl, mM_partial_nl, mSink, mask_config, scale_s_log2_e, scale_o, log2_threshold_scale_factor).launch(grid=grid, block=[self.threads_per_cta, 1, 1], cluster=[self.cluster_kv, 1, 1], stream=stream, min_blocks_per_mp=1, use_pdl=enable_pdl)
        if cutlass.const_expr(self.do_kernel_red):
            GqaDecode.launch_reduction(self.headdim, o_bshd, l_bsh, o_partial_bshd, l_partial_bsh, m_partial_bsh, sink_h, scale_o, stream, enable_pdl)

    @cute.kernel
    def decode(self, blk_tile_shpd: cute.Tile, mma_tile_mnk: cute.Tile, tiled_mma_kq: cute.TiledMma, tiled_mma_vp: cute.TiledMma, mma_dtype: Type[cutlass.Numeric], out_dtype: Type[cutlass.Numeric], seqlens: Union[cute.Tensor, Int32], table_stride: Int32, max_splits: Int32, page_table_iter: cute.Pointer, smem_layout_k_mma: cute.ComposedLayout, smem_layout_k_tma: cute.ComposedLayout, tma_atom_k: cute.CopyAtom, mK: cute.Tensor, smem_layout_v_mma: cute.ComposedLayout, smem_layout_v_tma: cute.ComposedLayout, tma_atom_v: cute.CopyAtom, mV: cute.Tensor, smem_layout_q: cute.ComposedLayout, tma_atom_q: cute.CopyAtom, mQ: cute.Tensor, smem_layout_o: cute.ComposedLayout, tma_atom_o: cute.CopyAtom, mO: cute.Tensor, mL: Optional[cute.Tensor], mL_partial: Optional[cute.Tensor], mM_partial: Optional[cute.Tensor], mSink: Optional[cute.Tensor], mask_config: AttentionMask, scale_s_log2_e: Float32, scale_o: Float32, log2_threshold_scale_factor: Optional[Float32]):
        svector_align = 16
        stensor_align = 128
        smem = utils.SmemAllocator()
        mcast_coord = 0
        mcast_layout = cute.make_layout((1, 1, 1, 1))
        q_dtype = k_dtype = mma_dtype
        o_dtype = out_dtype
        acc_dtype = Float32
        (blk_tile_s, blk_tile_h, blk_tile_p, blk_tile_d) = blk_tile_shpd
        blk_tile_hp = (blk_tile_h, blk_tile_p)
        blk_tile_n = blk_tile_h * blk_tile_p
        (mma_tile_m, mma_tile_n, mma_tile_k) = mma_tile_mnk
        tiles_sm = blk_tile_s // mma_tile_m
        tiles_sk = blk_tile_s // mma_tile_k
        tiles_dm = cute.ceil_div(blk_tile_d, mma_tile_m)
        tiles_dk = cute.ceil_div(blk_tile_d, mma_tile_k)
        page_size = self.page_size
        pages_s = blk_tile_s // page_size
        pages_m = mma_tile_m // page_size
        pages_k = mma_tile_k // page_size
        do_kernel_red = self.do_kernel_red
        do_atomic_red = self.do_atomic_red
        do_none_red = self.do_none_red
        store_lse = mL is not None
        enable_blasst = log2_threshold_scale_factor is not None
        use_sink = mSink is not None
        warpgroup_id = 0
        mma_kq_warp_id = warpgroup_id * warpgroup_warps + 0
        mma_vp_warp_id = warpgroup_id * warpgroup_warps + 1
        tma_qk_warp_id = warpgroup_id * warpgroup_warps + 2
        tma_vo_warp_id = warpgroup_id * warpgroup_warps + 3
        reduction_warp_id = mma_kq_warp_id
        warpgroup_id += 1
        softmax_warpgroups = self.softmax_warpgroups
        softmax_warpgroup_ids = tuple(range(warpgroup_id, warpgroup_id + softmax_warpgroups))
        warpgroup_id += softmax_warpgroups
        assert softmax_warpgroups in (1, 2)
        correction_warpgroup_id = warpgroup_id
        warpgroup_id += 1
        assert self.threads_per_cta == warpgroup_id * warpgroup_threads
        use_reg_reconfig = blk_tile_n > 16
        max_sw_regs_per_wg_thread = 256
        max_hw_regs_per_wg_thread = 64 * 1024 // warpgroup_threads
        mma_tma_regs = 64
        softmax_regs = 120
        correction_regs = min(208, max_sw_regs_per_wg_thread, max_hw_regs_per_wg_thread - mma_tma_regs - softmax_regs * softmax_warpgroups)
        assert mma_tma_regs + softmax_regs * softmax_warpgroups + correction_regs <= max_hw_regs_per_wg_thread
        (tiles_hz, grid_x, _) = cute.arch.grid_dim()
        (coord_hp, cta_idx, coord_hk) = cute.arch.block_idx()
        (tidx, _, _) = cute.arch.thread_idx()
        lane_idx = cute.arch.lane_idx()
        warp_idx = cute.arch.make_warp_uniform(tidx // warp_threads)
        warpgroup_idx = cute.arch.make_warp_uniform(tidx // warpgroup_threads)
        warpgroup_tidx = tidx % warpgroup_threads
        warpgroup_widx = warp_idx % warpgroup_warps
        init_warp = 1
        (grouped_heads, prediction) = mQ.shape[0]
        (heads_k, batches) = mQ.shape[2]
        tiles_hp = cute.ceil_div(mQ.shape[0], blk_tile_hp)
        coord_hp = cute.idx2crd(coord_hp, tiles_hp)
        (coord_hg, coord_p) = coord_hp
        cpasync_atom = cute.make_copy_atom(cute.nvgpu.cpasync.CopyG2SOp(), Int32, num_bits_per_copy=32)
        map_smem = smem.allocate_tensor(Int32, cute.make_layout(4), svector_align)
        split_target = cutlass.max(Int32(1), grid_x - batches)
        if warp_idx == 0:
            cute.arch.griddepcontrol_wait()
            if lane_idx == 0:
                map_smem[0] = Int32(0)
                map_smem[1] = Int32(0)
                map_smem[2] = Int32(1)
                map_smem[3] = Int32(0)
            cute.arch.sync_warp()
            if cutlass.const_expr(self.single_warp_batch):
                (map_found, map_b, map_split, map_splits, map_seqlen) = flatten_split_work_single_warp(seqlens, batches, blk_tile_s, split_target, max_splits, cta_idx, lane_idx)
            else:
                (map_found, map_b, map_split, map_splits, map_seqlen) = flatten_split_work(seqlens, batches, blk_tile_s, split_target, max_splits, cta_idx, lane_idx)
            if map_found:
                map_smem[0] = map_b
                map_smem[1] = map_split
                map_smem[2] = map_splits
                map_smem[3] = map_seqlen
        cute.arch.sync_threads()
        coord_b = map_smem[0]
        kv_split_idx = map_smem[1]
        kv_splits = map_smem[2]
        seqlen = map_smem[3]
        no_work = seqlen == 0
        coord_hb = (coord_hk, coord_b)
        if warp_idx == init_warp:
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_q)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_k)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_v)
            cute.nvgpu.cpasync.prefetch_descriptor(tma_atom_o)
        init_warp += 1
        tmem_alloc_cols = self.tmem_alloc_cols
        tmem_ptr_smem_ptr = smem.allocate_array(Int32)
        if warp_idx == init_warp and not no_work:
            cute.arch.alloc_tmem(tmem_alloc_cols, tmem_ptr_smem_ptr)
        init_warp += 1
        softmax_threads = warpgroup_threads
        correction_threads = warpgroup_threads
        reduction_threads = warp_threads
        mma_threads = warp_threads
        tma_threads = warp_threads
        tma_order_k_nbar = nbar(1, tma_threads + tma_threads)
        tma_order_v_nbar = nbar(2, tma_threads + tma_threads)
        mma_order_kq_nbar = nbar(3, mma_threads + mma_threads)
        mma_order_vp_nbar = nbar(4, mma_threads + mma_threads)
        sM_producer_nbar = nbar(5, softmax_threads + correction_threads)
        sM_consumer_nbar = nbar(6, softmax_threads + correction_threads)
        tL_producer_nbar = nbar(7, softmax_threads + correction_threads)
        tL_consumer_nbar = nbar(9, softmax_threads + correction_threads)
        sM_final_nbar = nbar(11, correction_threads + reduction_threads)
        sL_final_nbar = nbar(12, correction_threads + reduction_threads)
        sO_final_nbar = nbar(13, correction_threads + tma_threads)
        sSP_producer_nbar = nbar(14, softmax_threads)
        sM_mutex_nbar = nbar(14, softmax_threads * softmax_warpgroups)
        pdl_store_nbar = nbar(15, tma_threads + reduction_threads)

        def with_phase(nbar_, phase):
            return nbar(nbar_.barrier_id + phase, nbar_.num_threads)
        thr_cg = lambda t: CooperativeGroup(Agent.Thread, t)
        elect_one_cooperative = thr_cg(1)
        warpgroup_cooperative = thr_cg(warpgroup_threads)
        mma_group = elect_one_cooperative
        tma_group = elect_one_cooperative
        softmax_group = warpgroup_cooperative
        correction_group = warpgroup_cooperative
        if cutlass.const_expr(do_atomic_red):
            reduction_mbars_ptr = smem.allocate_array(Int64, max_reduction_iters * 2)
            if warp_idx == init_warp:
                if lane_idx < max_reduction_iters * 2:
                    mbar_ptr = reduction_mbars_ptr + lane_idx
                    arrive_count = 1
                    expect_tx_bytes = blk_tile_n * acc_dtype.width // 8
                    cute.arch.mbarrier_init(mbar_ptr, arrive_count)
                    cute.arch.mbarrier_init_fence()
                    cute.arch.mbarrier_arrive_and_expect_tx(mbar_ptr, expect_tx_bytes)
            init_warp += 1
            cute.arch.cluster_arrive_relaxed()
        q_load_mbar = smem.allocate_array(Int64, 1)
        if warp_idx == init_warp:
            expect_tx_bytes = cute.size_in_bytes(q_dtype, smem_layout_q)
            with cute.arch.elect_one():
                cute.arch.mbarrier_init(q_load_mbar, 1)
                cute.arch.mbarrier_init_fence()
                cute.arch.mbarrier_arrive_and_expect_tx(q_load_mbar, expect_tx_bytes)
        init_warp += 1
        kv_stages = smem_layout_k_tma.shape[-1]
        kv_stage_bytes = mma_tile_m * mma_tile_k * k_dtype.width // 8
        kv_pipeline_ptr = smem.allocate_array(Int64, kv_stages * 2)
        (kv_producer, kv_consumer) = pipeline.PipelineTmaUmma.create(num_stages=kv_stages, producer_group=tma_group, consumer_group=mma_group, tx_count=kv_stage_bytes, barrier_storage=kv_pipeline_ptr, cta_layout_vmnk=mcast_layout, defer_sync=True).make_participants()
        s_stages = self.s_stages
        s_pipeline_ptr = smem.allocate_array(Int64, s_stages * 2)
        (s_producer, s_consumer) = pipeline.PipelineUmmaAsync.create(num_stages=self.s_stages, producer_group=mma_group, consumer_group=softmax_group, barrier_storage=s_pipeline_ptr, defer_sync=True).make_participants()
        p_stages = self.p_stages
        p_pipeline_ptr = smem.allocate_array(Int64, p_stages * 2)
        (p_producer, p_consumer) = pipeline.PipelineAsyncUmma.create(num_stages=self.p_stages, producer_group=softmax_group, consumer_group=mma_group, barrier_storage=p_pipeline_ptr, defer_sync=True).make_participants()
        o_stages = self.o_stages
        o_pipeline_ptr = smem.allocate_array(Int64, o_stages * 2)
        (o_producer, o_consumer) = pipeline.PipelineUmmaAsync.create(num_stages=o_stages, producer_group=mma_group, consumer_group=correction_group, barrier_storage=o_pipeline_ptr, defer_sync=True).make_participants()
        if cutlass.const_expr(enable_blasst):
            skip_group = thr_cg(correction_threads + (mma_threads + tma_threads) * 2)
            sp_stages = self.sp_stages
            sp_pipeline_ptr = smem.allocate_array(Int64, sp_stages * 2)
            (sp_producer, sp_consumer) = pipeline.PipelineAsync.create(num_stages=sp_stages, producer_group=softmax_group, consumer_group=skip_group, barrier_storage=sp_pipeline_ptr, defer_sync=True).make_participants()
            sSP_i32 = smem.allocate_tensor(Int32, cute.make_layout(sp_stages))
            sSP_i8 = cute.make_tensor(cute.recast_ptr(sSP_i32.iterator, dtype=Int8), cute.make_layout((warpgroup_warps, sp_stages)))
        thrblk_mma_kq = tiled_mma_kq.get_slice(0)
        thrblk_mma_vp = tiled_mma_vp.get_slice(0)
        tAsK = smem.allocate_tensor(k_dtype, smem_layout_k_mma.outer, stensor_align, smem_layout_k_mma.inner)
        tAsV = cute.make_tensor(tAsK.iterator, smem_layout_v_mma.outer)
        tBsQ = smem.allocate_tensor(q_dtype, smem_layout_q.outer, stensor_align, smem_layout_q.inner)
        tCtS_shape = tiled_mma_kq.partition_shape_C((mma_tile_m, mma_tile_n, tiles_sm, s_stages))
        tCtS = thrblk_mma_kq.make_fragment_C(tCtS_shape)
        blk_tile_nm = (None, mma_tile_n, mma_tile_m * tiles_sm)
        tBsP_nm_layout = sm100_utils.make_smem_layout_b(tiled_mma_vp, blk_tile_nm, mma_dtype, p_stages)
        tBsP_nm = smem.allocate_tensor(mma_dtype, tBsP_nm_layout.outer, stensor_align, tBsP_nm_layout.inner)
        tBsP_nk_tile = thrblk_mma_vp.partition_shape_B((mma_tile_n, mma_tile_k))
        tBsP_nk = cute.local_tile(tBsP_nm, tBsP_nk_tile, (0, 0, None, None))
        tCsP_tile = cute.make_ordered_layout(tCtS_shape, order=((2, 0), 3, 1, 4, 5))
        tCsP = cute.composition(tBsP_nm, tCsP_tile)
        sO_iterator = cute.recast_ptr(tAsK.iterator, smem_layout_o.inner, dtype=o_dtype)
        sO_mma = cute.make_tensor(sO_iterator, smem_layout_o.outer)
        tCsO = thrblk_mma_vp.partition_C(sO_mma)
        tCsO = tCsO[mma_dice + (None, 0)]
        tCtO = thrblk_mma_vp.make_fragment_C((*tCsO.shape, o_stages))
        pt_stages = self.pt_stages
        sPT_layout = cute.make_layout((pages_s, pt_stages))
        sPT = smem.allocate_tensor(Int32, sPT_layout, svector_align)
        sM_layout = cute.make_layout(blk_tile_n)
        sM = smem.allocate_tensor(acc_dtype, sM_layout, svector_align)
        lane_store_max = blk_tile_n == warp_threads or lane_idx < blk_tile_n
        if warp_idx == init_warp:
            if lane_store_max:
                sM[lane_idx] = -Float32.inf
        init_warp += 1
        sL_layout = cute.make_layout((blk_tile_n, warpgroup_warps))
        sL = smem.allocate_tensor(acc_dtype, sL_layout, svector_align)
        if warp_idx == init_warp:
            for i in cutlass.range_constexpr(0, cute.size(sL), warp_threads):
                if i + lane_idx < cute.size(sL):
                    sL[i + lane_idx] = Float32(0)
        init_warp += 1
        if cutlass.const_expr(use_sink and (not do_kernel_red)):
            sSink_layout = cute.make_layout((blk_tile_hp,), stride=((1, 0),))
            sSink = smem.allocate_tensor(acc_dtype, sSink_layout, svector_align)
            gSink = cute.local_tile(mSink, (blk_tile_h,), (coord_hg, coord_hk))
            sSink_lane = cute.local_tile(sSink[(None, 0),], (1,), (lane_idx,))
            gSink_lane = cute.local_tile(gSink, (1,), (lane_idx,))
            if warp_idx == reduction_warp_id and lane_idx < blk_tile_h:
                cute.copy(cpasync_atom, gSink_lane, sSink_lane)
            init_warp += 1
        else:
            sSink = None
        tCtL_shape = tiled_mma_kq.partition_shape_C((mma_tile_m, mma_tile_n, o_stages))
        tCtL = thrblk_mma_kq.make_fragment_C(tCtL_shape)
        if cutlass.const_expr(do_atomic_red):
            sR_layout = cute.make_layout((blk_tile_n, max_reduction_iters, 2))
            sR = smem.allocate_tensor(acc_dtype, sR_layout, svector_align)
        if cutlass.const_expr(do_atomic_red):
            cute.arch.cluster_wait()
        cute.arch.sync_threads()
        assert init_warp <= self.threads_per_cta // warp_threads, f'used {init_warp} init warps, {self.threads_per_cta // warp_threads} warps available'
        page_count = cute.ceil_div(seqlen, page_size)
        table_offset = coord_b * table_stride
        tiles_s = cute.ceil_div(seqlen, blk_tile_s)
        iters_s = cute.ceil_div(tiles_s - kv_split_idx, kv_splits)
        exit_early = no_work
        prefetch_iters = min(2, s_stages - 1)
        assert pt_stages > prefetch_iters + 1
        tmem_ptr = cute.arch.retrieve_tmem_ptr(Int32, 16, tmem_ptr_smem_ptr)
        tmem_offset = 0
        tCtS = cute.make_tensor(cute.recast_ptr(tmem_ptr + tmem_offset, dtype=acc_dtype), tCtS.layout)
        tmem_offset += tcgen05.find_tmem_tensor_col_offset(tCtS)
        tCtL = cute.make_tensor(cute.recast_ptr(tmem_ptr + tmem_offset, dtype=acc_dtype), tCtL.layout)
        tmem_offset += tcgen05.find_tmem_tensor_col_offset(tCtL)
        tCtO = cute.make_tensor(cute.recast_ptr(tmem_ptr + tmem_offset, dtype=acc_dtype), tCtO.layout)
        tmem_offset += tcgen05.find_tmem_tensor_col_offset(tCtO)
        assert tmem_offset <= tmem_alloc_cols, f'\t{tmem_offset} tmem cols used, {tmem_alloc_cols} tmem cols allocated'
        if exit_early:
            if warp_idx == mma_vp_warp_id:
                cute.arch.relinquish_tmem_alloc_permit()
        elif warp_idx == tma_qk_warp_id:
            if cutlass.const_expr(use_reg_reconfig):
                cute.arch.setmaxregister_decrease(mma_tma_regs)
            gQ = cute.local_tile(mQ, tiler=(blk_tile_hp, blk_tile_d), coord=(coord_hp, 0, coord_hb))
            gQ_mma = cute.local_tile(gQ, (mma_tile_n, mma_tile_k), coord=(0, None))
            tBgQ = thrblk_mma_kq.partition_B(gQ_mma)
            (tBsQ_tma, tBgQ_tma) = cute.nvgpu.cpasync.tma_partition(tma_atom_q, mcast_coord, mcast_layout, smem_tensor=cute.group_modes(tBsQ, 0, 3), gmem_tensor=cute.group_modes(tBgQ, 0, 3))
            sK = cute.make_tensor(tAsK.iterator, smem_layout_k_tma.outer)
            gK = cute.local_tile(mK, (page_size, mma_tile_k), coord=(0, None, (coord_hk, None)))
            (sK_tma, gK_tma) = cute.nvgpu.cpasync.tma_partition(tma_atom_k, mcast_coord, mcast_layout, smem_tensor=sK, gmem_tensor=cute.group_modes(gK, 0, 2))
            pt_load = cute.make_tiled_copy(cpasync_atom, cute.make_ordered_layout((pages_s, 1), order=(1, 0)), (pages_s,))
            lane_load_page = lane_idx < pages_s
            thr_pt_load = pt_load.get_slice(lane_idx)
            tPTsPT = thr_pt_load.partition_D(sPT)
            cute.arch.griddepcontrol_wait()
            if lane_load_page:
                logical_page_idx = kv_split_idx * pages_s + lane_idx
                if logical_page_idx < page_count:
                    table_page_idx = logical_page_idx // self.subpages_per_table_page
                    gPT = cute.make_tensor(page_table_iter + table_offset + table_page_idx, cute.make_layout(1))
                    cute.copy(cpasync_atom, gPT, tPTsPT[None, 0, 0])
                else:
                    tPTsPT[0] = -1
            cute.arch.sync_warp()
            cute.arch.cp_async_commit_group()
            cute.copy(tma_atom_q, tBgQ_tma, tBsQ_tma, tma_bar_ptr=q_load_mbar)
            pt_index = 0
            for s in cutlass.range(iters_s):
                pt_index_next = 0 if pt_index == pt_stages - 1 else pt_index + 1
                if s < iters_s - 1 and lane_load_page:
                    tile_s_next = (s + 1) * kv_splits + kv_split_idx
                    logical_page_idx = tile_s_next * pages_s + lane_idx
                    virt_page_idx_smem = tPTsPT[None, 0, pt_index_next]
                    if logical_page_idx < page_count:
                        table_page_idx = logical_page_idx // self.subpages_per_table_page
                        gPT = cute.make_tensor(page_table_iter + table_offset + table_page_idx, cute.make_layout(1))
                        cute.copy(cpasync_atom, gPT, virt_page_idx_smem)
                    else:
                        virt_page_idx_smem[0] = -1
                cute.arch.sync_warp()
                cute.arch.cp_async_commit_group()
                cute.arch.cp_async_wait_group(1)
                rPT = sPT[None, pt_index].load().reshape((pages_m, tiles_sm))
                pt_index = pt_index_next
                tma_order_v_nbar.arrive_and_wait()
                kv_token = kv_producer.try_acquire()
                for sm in cutlass.range_constexpr(tiles_sm):
                    for dk in cutlass.range_constexpr(tiles_dk):
                        kv_handle = kv_producer.acquire_and_advance(kv_token)
                        is_last_iter = sm == tiles_sm - 1 and dk == tiles_dk - 1
                        if is_last_iter:
                            tma_order_k_nbar.arrive()
                        else:
                            kv_token = kv_producer.try_acquire()
                        for pm in cutlass.range_constexpr(pages_m):
                            virtual_page_idx = rPT[pm, sm]
                            if virtual_page_idx >= 0:
                                logical_page_idx = (s * kv_splits + kv_split_idx) * pages_s + sm * pages_m + pm
                                virtual_page_idx = virtual_page_idx * self.subpages_per_table_page + logical_page_idx % self.subpages_per_table_page
                            cute.copy(tma_atom_k, gK_tma[None, dk, virtual_page_idx], sK_tma[None, pm, kv_handle.index], tma_bar_ptr=kv_handle.barrier)
                if s >= prefetch_iters:
                    keep_tile = not enable_blasst
                    if cutlass.const_expr(enable_blasst):
                        sp_handle = sp_consumer.wait_and_advance()
                        keep_tile = sSP_i32[sp_handle.index] != 0
                        sp_handle.release()
                    if keep_tile:
                        for _ in cutlass.range_constexpr(tiles_dm * tiles_sk):
                            kv_producer.advance()
            for s in cutlass.range_constexpr(prefetch_iters):
                tma_order_v_nbar.arrive_and_wait()
                tma_order_k_nbar.arrive()
                if cutlass.const_expr(enable_blasst):
                    if s < min(prefetch_iters, iters_s):
                        sp_handle = sp_consumer.wait_and_advance()
                        sp_handle.release()
        elif warp_idx == tma_vo_warp_id:
            if cutlass.const_expr(use_reg_reconfig):
                cute.arch.setmaxregister_decrease(mma_tma_regs)
            sV = cute.make_tensor(tAsV.iterator, smem_layout_v_tma.outer)
            gV = cute.local_tile(mV, (mma_tile_m, page_size), coord=(None, 0, (coord_hk, None)))
            (sV_tma, gV_tma) = cute.nvgpu.cpasync.tma_partition(tma_atom_v, mcast_coord, mcast_layout, smem_tensor=sV, gmem_tensor=cute.group_modes(gV, 0, 2))
            tma_order_v_nbar.arrive()
            for s in cutlass.range_constexpr(prefetch_iters):
                if s < iters_s:
                    for _ in cutlass.range_constexpr(tiles_sm * tiles_dk):
                        kv_producer.advance()
                tma_order_k_nbar.arrive_and_wait()
                tma_order_v_nbar.arrive()
            pt_index = 0
            for s in cutlass.range(iters_s):
                if s < iters_s - prefetch_iters:
                    for _ in cutlass.range_constexpr(tiles_sm * tiles_dk):
                        kv_producer.advance()
                keep_tile = not enable_blasst
                if cutlass.const_expr(enable_blasst):
                    sp_handle = sp_consumer.wait_and_advance()
                    keep_tile = sSP_i32[sp_handle.index] != 0
                    sp_handle.release()
                    if not keep_tile:
                        tma_order_k_nbar.arrive_and_wait()
                        tma_order_v_nbar.arrive()
                if keep_tile:
                    tma_order_k_nbar.arrive_and_wait()
                    rPT = sPT[None, pt_index].load().reshape((pages_k, tiles_sk))
                    kv_token = kv_producer.try_acquire()
                    for sk in cutlass.range_constexpr(tiles_sk):
                        for dm in cutlass.range_constexpr(tiles_dm):
                            kv_handle = kv_producer.acquire_and_advance(kv_token)
                            is_last_iter = sk == tiles_sk - 1 and dm == tiles_dm - 1
                            if is_last_iter:
                                tma_order_v_nbar.arrive()
                            else:
                                kv_token = kv_producer.try_acquire()
                            for pk in cutlass.range_constexpr(pages_k):
                                virtual_page_idx = rPT[pk, sk]
                                if virtual_page_idx >= 0:
                                    logical_page_idx = (s * kv_splits + kv_split_idx) * pages_s + sk * pages_k + pk
                                    virtual_page_idx = virtual_page_idx * self.subpages_per_table_page + logical_page_idx % self.subpages_per_table_page
                                cute.copy(tma_atom_v, gV_tma[None, dm, virtual_page_idx], sV_tma[None, pk, kv_handle.index], tma_bar_ptr=kv_handle.barrier)
                pt_index = 0 if pt_index == pt_stages - 1 else pt_index + 1
            coord_b_partial = kv_split_idx * batches + coord_b if do_kernel_red else coord_b
            gO = cute.local_tile(mO, tiler=(blk_tile_d, blk_tile_hp), coord=(0, coord_hp, (coord_hk, coord_b_partial)))
            gO_mma = cute.flat_divide(gO, (mma_tile_m, mma_tile_n))
            (sO_tma, gO_tma) = cute.nvgpu.cpasync.tma_partition(tma_atom_o, mcast_coord, mcast_layout, smem_tensor=cute.group_modes(sO_mma, 0, 2), gmem_tensor=cute.group_modes(gO_mma, 0, 2))
            sO_final_nbar.arrive_and_wait()
            cute.copy(tma_atom_o, sO_tma, gO_tma)
            if cutlass.const_expr(use_reg_reconfig):
                cute.arch.cp_async_bulk_commit_group()
                cute.arch.cp_async_bulk_wait_group(0, read=True)
                pdl_store_nbar.arrive()
        elif warp_idx == mma_kq_warp_id:
            if cutlass.const_expr(use_reg_reconfig):
                cute.arch.setmaxregister_decrease(mma_tma_regs)
            tAsK_desc = thrblk_mma_kq.make_fragment_A(tAsK)
            tBsQ_desc = thrblk_mma_kq.make_fragment_B(tBsQ)
            cute.arch.mbarrier_wait(q_load_mbar, phase=0)
            for s in cutlass.range(iters_s):
                s_token = s_producer.try_acquire()
                mma_order_vp_nbar.arrive_and_wait()
                k_token = kv_consumer.try_wait()
                s_handle = s_producer.acquire_and_advance(s_token)
                for sm in cutlass.range_constexpr(tiles_sm):
                    tiled_mma_kq.set(tcgen05.Field.ACCUMULATE, False)
                    for dk in cutlass.range_constexpr(tiles_dk):
                        k_handle = kv_consumer.wait_and_advance(k_token)
                        is_last_iter = sm == tiles_sm - 1 and dk == tiles_dk - 1
                        if is_last_iter:
                            mma_order_kq_nbar.arrive()
                        else:
                            k_token = kv_consumer.try_wait()
                        mmas_k = cute.size(tAsK.shape[2])
                        for mma_k in cutlass.range_constexpr(mmas_k):
                            cute.gemm(tiled_mma_kq, tCtS[mma_dice + (sm, s_handle.index)], tAsK_desc[None, None, mma_k, k_handle.index], tBsQ_desc[None, None, mma_k, dk], tCtS[mma_dice + (sm, s_handle.index)])
                            if dk == 0 and mma_k == 0:
                                tiled_mma_kq.set(tcgen05.Field.ACCUMULATE, True)
                        k_handle.release()
                s_handle.commit()
                if s >= prefetch_iters:
                    keep_tile = not enable_blasst
                    if cutlass.const_expr(enable_blasst):
                        sp_handle = sp_consumer.wait_and_advance()
                        keep_tile = sSP_i32[sp_handle.index] != 0
                        sp_handle.release()
                    if keep_tile:
                        for _ in cutlass.range_constexpr(tiles_dm * tiles_sk):
                            kv_consumer.advance()
            for s in cutlass.range_constexpr(prefetch_iters):
                mma_order_vp_nbar.arrive_and_wait()
                mma_order_kq_nbar.arrive()
                if cutlass.const_expr(enable_blasst):
                    if s < min(prefetch_iters, iters_s):
                        sp_handle = sp_consumer.wait_and_advance()
                        sp_handle.release()
        elif warp_idx == mma_vp_warp_id:
            if cutlass.const_expr(use_reg_reconfig):
                cute.arch.setmaxregister_decrease(mma_tma_regs)
            tiled_mma_vp.set(tcgen05.Field.ACCUMULATE, True)
            tAsV_desc = thrblk_mma_vp.make_fragment_A(tAsV)
            tBsP_desc = thrblk_mma_vp.make_fragment_B(tBsP_nk)
            mma_order_vp_nbar.arrive()
            for s in cutlass.range_constexpr(prefetch_iters):
                if s < iters_s:
                    for _ in cutlass.range_constexpr(tiles_sm * tiles_dk):
                        kv_consumer.advance()
                mma_order_kq_nbar.arrive_and_wait()
                mma_order_vp_nbar.arrive()
            for s in cutlass.range(iters_s):
                if s < iters_s - prefetch_iters:
                    for _ in cutlass.range_constexpr(tiles_sm * tiles_dk):
                        kv_consumer.advance()
                keep_tile = not enable_blasst
                if cutlass.const_expr(enable_blasst):
                    sp_handle = sp_consumer.wait_and_advance()
                    keep_tile = sSP_i32[sp_handle.index] != 0
                    sp_handle.release()
                    if not keep_tile:
                        mma_order_kq_nbar.arrive_and_wait()
                        mma_order_vp_nbar.arrive()
                if keep_tile:
                    p_token = p_consumer.try_wait()
                    o_token = o_producer.try_acquire()
                    mma_order_kq_nbar.arrive_and_wait()
                    v_token = kv_consumer.try_wait()
                    p_handle = p_consumer.wait_and_advance(p_token)
                    o_handle = o_producer.acquire_and_advance(o_token)
                    for sk in cutlass.range_constexpr(tiles_sk):
                        for dm in cutlass.range_constexpr(tiles_dm):
                            v_handle = kv_consumer.wait_and_advance(v_token)
                            is_last_iter = sk == tiles_sk - 1 and dm == tiles_dm - 1
                            if is_last_iter:
                                mma_order_vp_nbar.arrive()
                            else:
                                v_token = kv_consumer.try_wait()
                            mmas_k = cute.size(tAsV.shape[2])
                            for mma_k in cutlass.range_constexpr(mmas_k):
                                cute.gemm(tiled_mma_vp, tCtO[mma_dice + (dm, o_handle.index)], tAsV_desc[None, None, mma_k, v_handle.index], tBsP_desc[None, None, mma_k, sk, p_handle.index], tCtO[mma_dice + (dm, o_handle.index)])
                            v_handle.release()
                    p_handle.release()
                    o_handle.commit()
            if iters_s == 1:
                o_producer.commit()
                o_producer.advance()
            o_producer.tail()
            cute.arch.relinquish_tmem_alloc_permit()
            cute.arch.dealloc_tmem(tmem_ptr, tmem_alloc_cols)
        elif warpgroup_idx in softmax_warpgroup_ids:
            if cutlass.const_expr(use_reg_reconfig):
                cute.arch.setmaxregister_decrease(softmax_regs)
            softmax_phase = 0
            if cutlass.const_expr(softmax_warpgroups == 2):
                softmax_phase = (warpgroup_idx - 1) % softmax_warpgroups
                sM_acquire_nbar = with_phase(sM_mutex_nbar, softmax_phase)
                sM_release_nbar = with_phase(sM_mutex_nbar, softmax_phase ^ 1)
                if softmax_phase == 1:
                    s_consumer.advance()
                    p_producer.advance()
                    sM_release_nbar.arrive()
            if iters_s == 1 and softmax_phase == softmax_warpgroups - 1:
                with_phase(tL_producer_nbar, 1).arrive()
            assert not (enable_blasst and softmax_warpgroups != 1), 'blasst only supports 1 softmax wg'
            tmem_repeat_op_s = blk_tile_n
            if cutlass.const_expr(mma_tile_n == blk_tile_n and tiles_sm in (2, 4)):
                tmem_repeat_op_s *= tiles_sm
            tmem_repeat_op_s = tcgen05.Repetition(tmem_repeat_op_s)
            tmem_load_op_s = tcgen05.Ld32x32bOp(tmem_repeat_op_s)
            tmem_load_atom_s = cute.make_copy_atom(tmem_load_op_s, acc_dtype)
            tCtS_stage = tCtS[mma_dice + (None, 0)]
            tmem_load_s = tcgen05.make_tmem_copy(tmem_load_atom_s, tCtS_stage)
            thr_load_s = tmem_load_s.get_slice(warpgroup_tidx)
            tStS = thr_load_s.partition_S(tCtS)
            tSsP = thr_load_s.partition_D(tCsP)
            tStS = tStS[None, 0, 0, 0, None, None]
            tSsP = tSsP[None, 0, 0, 0, None, None]
            tmem_repeat_op_l = tcgen05.Repetition(blk_tile_n)
            tmem_load_op_l = tcgen05.Ld32x32bOp(tmem_repeat_op_l)
            tmem_store_op_l = tcgen05.St32x32bOp(tmem_repeat_op_l)
            tmem_load_atom_l = cute.make_copy_atom(tmem_load_op_l, acc_dtype)
            tmem_store_atom_l = cute.make_copy_atom(tmem_store_op_l, acc_dtype)
            tCtL_phase = tCtL[mma_dice + (softmax_phase,)]
            tmem_load_l = tcgen05.make_tmem_copy(tmem_load_atom_l, tCtL_phase)
            thr_load_l = tmem_load_l.get_slice(warpgroup_tidx)
            tStL = thr_load_l.partition_S(tCtL)
            tStL = tStL[None, 0, 0, 0, None]
            tSrL_shape = thr_load_l.partition_D(tCtL_phase).shape[:1]
            range_args = mask_config.get_range_args(prediction, seqlen, blk_tile_p, blk_tile_s, tiles_s, iters_s, kv_splits, kv_split_idx, softmax_warpgroups, softmax_phase)
            num_mask_phases = len(range_args)
            if cutlass.const_expr(enable_blasst):
                sM_lane_prev = -Float32.inf
                log2_threshold_p = log2_threshold_scale_factor - cute.math.log2(Float32(seqlen))
            loop_idx = 0
            for mask_phase in cutlass.range_constexpr(num_mask_phases):
                (start, stop, step, is_masked) = range_args[mask_phase]
                for coord_s in cutlass.range(start, stop, step):
                    s_token = s_consumer.try_wait()
                    s_handle = s_consumer.wait_and_advance(s_token)
                    tStS_s = tStS[None, None, s_handle.index]
                    tSrS_s = cute.make_rmem_tensor(tSsP.shape[:-1], acc_dtype)
                    cute.copy(tmem_load_atom_s, tStS_s, tSrS_s)
                    cute.arch.fence_view_async_tmem_load()
                    s_handle.release()
                    if cutlass.const_expr(is_masked):
                        masked = cute.make_rmem_tensor((blk_tile_h, blk_tile_p, tiles_sm), acc_dtype)
                        masked.store(tSrS_s.load().reshape(masked.shape))
                        offset_p = coord_p * blk_tile_p
                        offset_s = coord_s * blk_tile_s + warpgroup_tidx
                        for sm in cutlass.range_constexpr(tiles_sm):
                            for p in cutlass.range_constexpr(blk_tile_p):
                                idx_q = offset_p + p
                                idx_kv = offset_s + sm * mma_tile_m
                                is_oob_kv = mask_config.is_oob_kv(idx_q, idx_kv, prediction, seqlen)
                                mask = -Float32.inf if is_oob_kv else Float32(0)
                                masked_p = masked[None, p, sm]
                                masked_p.store(masked_p.load() + mask)
                        scores = masked.load().reshape((blk_tile_n, tiles_sm))
                    else:
                        scores = tSrS_s.load().reshape((blk_tile_n, tiles_sm))
                    rM = cute.make_rmem_tensor_like(sM)
                    rM.store(scores.reduce(cute.ReductionOp.MAX, init_val=min_f32, reduction_profile=(None, 0)))
                    rM_lane = Float32(0)
                    for n in cutlass.range_constexpr(blk_tile_n):
                        rM[n] = warp_fmax(rM[n])
                        if n == lane_idx:
                            rM_lane = rM[n]
                    rM_lane *= scale_s_log2_e
                    keep_tile = not enable_blasst
                    if cutlass.const_expr(enable_blasst):
                        lane_keep_tile = rM_lane - sM_lane_prev >= log2_threshold_p
                        lane_keep_tile &= lane_store_max
                        lane_keep_tile |= loop_idx < o_stages
                        warp_keep_tile = warp_or(Int32(lane_keep_tile))
                        loop_idx += 1
                        sp_handle = sp_producer.acquire_and_advance()
                        with cute.arch.elect_one():
                            sSP_i8[warpgroup_widx, sp_handle.index] = Int8(warp_keep_tile)
                        sp_handle.commit()
                        sSP_producer_nbar.sync()
                        keep_tile = sSP_i32[sp_handle.index] != 0
                    if keep_tile:
                        p_token = p_producer.try_acquire()
                        if cutlass.const_expr(softmax_warpgroups == 2):
                            sM_acquire_nbar.arrive_and_wait()
                        sM_consumer_nbar.arrive_and_wait()
                        if lane_store_max:
                            smem_fmax(sM.iterator + sM.layout(lane_idx), rM_lane)
                        p_handle = p_producer.acquire_and_advance(p_token)
                        tSsP_s = tSsP[None, None, p_handle.index]
                        sM_producer_nbar.arrive_and_wait()
                        colmax = sM.load()
                        if cutlass.const_expr(enable_blasst):
                            if lane_store_max:
                                sM_lane_prev = sM[lane_idx]
                        if cutlass.const_expr(softmax_warpgroups == 2):
                            sM_release_nbar.arrive()
                        probs = exp2(scale_s_log2_e * scores - colmax)
                        tSsP_s.store(probs.to(mma_dtype).reshape(tSsP_s.shape))
                        cute.arch.fence_view_async_shared()
                        p_handle.commit()
                        colsum = probs[None, 0]
                        for sm in cutlass.range_constexpr(1, tiles_sm, 1):
                            colsum += probs[None, sm]
                        tSrL = cute.make_rmem_tensor(tSrL_shape, acc_dtype)
                        tSrL.store(colsum.reshape(tSrL.shape))
                        with_phase(tL_consumer_nbar, softmax_phase).arrive_and_wait()
                        cute.copy(tmem_store_atom_l, tSrL, tStL[None, softmax_phase])
                        cute.arch.fence_view_async_tmem_store()
                        with_phase(tL_producer_nbar, softmax_phase).arrive()
                        if cutlass.const_expr(softmax_warpgroups == 2):
                            s_consumer.advance()
                            p_producer.advance()
                        else:
                            softmax_phase ^= 1
        elif warpgroup_idx == correction_warpgroup_id:
            if cutlass.const_expr(use_reg_reconfig):
                cute.arch.setmaxregister_increase(correction_regs)
            tmem_repeat_op_o = tcgen05.Repetition(blk_tile_n)
            tmem_load_op_o = tcgen05.Ld32x32bOp(tmem_repeat_op_o)
            tmem_store_op_o = tcgen05.St32x32bOp(tmem_repeat_op_o)
            tmem_load_atom_o = cute.make_copy_atom(tmem_load_op_o, acc_dtype)
            tmem_store_atom_o = cute.make_copy_atom(tmem_store_op_o, acc_dtype)
            tCtO_dm = tCtO[mma_dice + (0, 0)]
            tmem_load_o = tcgen05.make_tmem_copy(tmem_load_atom_o, tCtO_dm)
            thr_load_o = tmem_load_o.get_slice(warpgroup_tidx)
            tOtO = thr_load_o.partition_S(tCtO)
            tOsO = thr_load_o.partition_D(tCsO)
            tOtL = thr_load_o.partition_S(tCtL)
            tOtO = tOtO[None, 0, 0, 0, None, None]
            tOsO = tOsO[None, 0, 0, 0, None]
            tOtL = tOtL[None, 0, 0, 0, None]

            def colsum_load(phase, blk_tile_n=blk_tile_n, tOtL=tOtL, tOrO_shape=tOsO.shape[:1], tmem_load_atom_o=tmem_load_atom_o, tL_producer_nbar=tL_producer_nbar, tL_consumer_nbar=tL_consumer_nbar):
                with_phase(tL_producer_nbar, phase).arrive_and_wait()
                tOtL_s = tOtL[None, phase]
                tOrL_s = cute.make_rmem_tensor(tOrO_shape, Float32)
                cute.copy(tmem_load_atom_o, tOtL_s, tOrL_s)
                cute.arch.fence_view_async_tmem_load()
                with_phase(tL_consumer_nbar, phase).arrive()
                return tOrL_s.load().reshape(blk_tile_n)
            tOrO = cute.make_rmem_tensor(tOsO.shape, acc_dtype)
            tOrO.fill(Float32(0))
            for phase in cutlass.range_constexpr(o_stages):
                cute.copy(tmem_store_atom_o, tOrO, tOtO[None, None, phase])
            cute.copy(tmem_store_atom_o, tOrO[None, 0], tOtL[None, 1])
            cute.arch.fence_view_async_tmem_store()
            sM_consumer_nbar.arrive()
            for phase in cutlass.range_constexpr(o_stages):
                with_phase(tL_consumer_nbar, phase).arrive()
            colsum_p = cute.make_rmem_tensor((blk_tile_n, o_stages), Float32)
            (colsum_0, colsum_1) = (colsum_p[None, 0], colsum_p[None, 1])
            colsum_p.fill(Float32(0))
            sM_lane_prev_prev = sM_lane_prev = Float32(0)
            for s in cutlass.range_constexpr(o_stages):
                sM_lane_prev_prev = sM_lane_prev
                if not (s == 1 and iters_s == 1):
                    if cutlass.const_expr(enable_blasst):
                        sp_handle = sp_consumer.wait_and_advance()
                        sp_handle.release()
                    sM_producer_nbar.arrive_and_wait()
                    if lane_store_max:
                        sM_lane_prev = sM[lane_idx]
                    sM_consumer_nbar.arrive()
            softmax_phase = 0
            unroll = o_stages if not enable_blasst else 1
            keep_tile = not enable_blasst
            for s in cutlass.range(iters_s - o_stages, unroll=unroll):
                if cutlass.const_expr(enable_blasst):
                    sp_handle = sp_consumer.wait_and_advance()
                    keep_tile = sSP_i32[sp_handle.index] != 0
                    sp_handle.release()
                if keep_tile:
                    colsum_s = colsum_load(softmax_phase)
                    sM_producer_nbar.arrive_and_wait()
                    if s == iters_s - o_stages - 1:
                        sM_final_nbar.arrive()
                    sM_lane = Float32(0)
                    if lane_store_max:
                        sM_lane = sM[lane_idx]
                    sM_consumer_nbar.arrive()
                    o_token = o_consumer.try_wait()
                    o_handle = o_consumer.wait_and_advance(o_token)
                    correction_lane = exp2(sM_lane_prev_prev - sM_lane)
                    correction = cute.make_rmem_tensor_like(sM)
                    for n in cutlass.range_constexpr(blk_tile_n):
                        correction[n] = cute.arch.shuffle_sync(correction_lane, n)
                    correction = correction.load()
                    (sM_lane_prev_prev, sM_lane_prev) = (sM_lane_prev, sM_lane)
                    correction_o = correction.reshape(tOsO.shape[:1])
                    for dm in cutlass.range_constexpr(tiles_dm):
                        tOtO_dm = tOtO[None, dm, softmax_phase]
                        tOrO_dm = cute.make_rmem_tensor(tOsO.shape[:1], acc_dtype)
                        cute.copy(tmem_load_atom_o, tOtO_dm, tOrO_dm)
                        tOrO_dm.store(correction_o * tOrO_dm.load())
                        cute.copy(tmem_store_atom_o, tOrO_dm, tOtO_dm)
                    cute.arch.fence_view_async_tmem_store()
                    o_handle.release()
                    colsum_s *= correction
                    if softmax_phase == 0:
                        colsum_0.store(correction * colsum_0.load() + colsum_s)
                    elif softmax_phase == 1:
                        colsum_1.store(correction * colsum_1.load() + colsum_s)
                    softmax_phase ^= 1
            if not keep_tile or iters_s <= o_stages:
                sM_final_nbar.arrive()
            correction_lane = exp2(sM_lane_prev_prev - sM_lane_prev)
            correction = cute.make_rmem_tensor_like(sM)
            for n in cutlass.range_constexpr(blk_tile_n):
                correction[n] = cute.arch.shuffle_sync(correction_lane, n)
            correction = correction.load()
            tail_phase = softmax_phase if enable_blasst else iters_s % o_stages
            for phase in cutlass.range_constexpr(o_stages):
                if tail_phase == phase:
                    colsum_prev = colsum_load(phase)
                    colsum_final = colsum_load(phase ^ 1)
                    colsum_prev += colsum_p[None, phase].load()
                    colsum_final += colsum_p[None, phase ^ 1].load()
                    colsum_final += correction * colsum_prev
                    rL_lane = Float32(0.0)
                    for n in cutlass.range_constexpr(blk_tile_n):
                        rL_n = cute.arch.warp_reduction_sum(colsum_final[n])
                        if n == lane_idx:
                            rL_lane = rL_n
                    if lane_store_max:
                        sL[lane_idx, warpgroup_widx] = rL_lane
                    sL_final_nbar.arrive_and_wait()
            tOrO_tail = cute.make_rmem_tensor((*tOsO.shape, o_stages), acc_dtype)
            for s in cutlass.range_constexpr(o_stages):
                o_handle = o_consumer.wait_and_advance()
                tOtO_s = tOtO[None, None, tail_phase ^ s]
                tOrO_s = tOrO_tail[None, None, s]
                cute.copy(tmem_load_atom_o, tOtO_s, tOrO_s)
                cute.arch.fence_view_async_tmem_load()
                o_handle.release()
            tOrO_prev = tOrO_tail[None, None, 0].load()
            tOrO_final = tOrO_tail[None, None, 1].load()
            output_prev = tOrO_prev.reshape((blk_tile_n, tiles_dm))
            output_final = tOrO_final.reshape((blk_tile_n, tiles_dm))
            output_final += correction * output_prev
            if cutlass.const_expr(do_atomic_red or do_none_red):
                sM_final_nbar.arrive_and_wait()
                normalization = sM.load()
                output_final *= normalization
            tOsO.store(output_final.to(o_dtype).reshape(tOsO.shape))
            cute.arch.fence_view_async_shared()
            sO_final_nbar.arrive()
            if cutlass.const_expr(enable_blasst and debug_blasst):
                tiles_skipped_ptr = seqlens.iterator + batches
                tiles_kept = o_consumer.current_handle().count if iters_s > 1 else 1
                tiles_skipped = iters_s - tiles_kept
                if warpgroup_tidx == 0:
                    gmem_add(tiles_skipped_ptr, Int32(tiles_skipped))
        if warp_idx == reduction_warp_id:
            if not no_work:
                if cutlass.const_expr(sSink is not None):
                    cute.arch.cp_async_commit_group()
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.sync_warp()
                if cutlass.const_expr(do_kernel_red):
                    GqaDecode.reduction_epilogue(blk_tile_hp, coord_hp, coord_hb, kv_split_idx, kv_splits, max_splits, True, lane_idx, sM_final_nbar, sL_final_nbar, sM, sL, mM_partial, mL_partial)
                elif cutlass.const_expr(do_atomic_red):
                    gL = None
                    if cutlass.const_expr(store_lse):
                        gL = cute.local_tile(mL, (blk_tile_hp,), (coord_hp, coord_hb))
                    GqaDecode.reduction_cluster(blk_tile_n, kv_splits, kv_split_idx, lane_idx, sM_final_nbar, sL_final_nbar, reduction_mbars_ptr, sM, sL, sR, gL, sSink, scale_o)
                elif cutlass.const_expr(do_none_red):
                    gL = None
                    if cutlass.const_expr(store_lse):
                        gL = cute.local_tile(mL, (blk_tile_hp,), (coord_hp, coord_hb))
                    GqaDecode.reduction_none(lane_store_max, lane_idx, sM_final_nbar, sL_final_nbar, sM, sL, gL, sSink, scale_o)
                if cutlass.const_expr(use_reg_reconfig):
                    pdl_store_nbar.arrive_and_wait()
            cute.arch.griddepcontrol_launch_dependents()
        return
