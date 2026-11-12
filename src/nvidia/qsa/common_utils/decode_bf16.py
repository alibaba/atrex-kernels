"""SM100/SM103 BF16 Q1/Q4 attention with FP32 online-softmax reductions."""
from __future__ import annotations

import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
import cutlass.utils.blackwell_helpers as sm100_utils
from cutlass.cute.nvgpu import cpasync, tcgen05
from cutlass.cute.typing import Float32, Int32, Int64

from .mtp_m64 import QSAForward as Base
from .qsa_kernel import publish_bias
from .layouts import _elem_pointer, LOG2_E
from .native_metadata import build_native_metadata
from . import qsa_sm_ptx


class QSAForward(Base):
    def __init__(self, *args, **kwargs):
        rows_per_cta = kwargs.get('br', 1)
        if rows_per_cta == 2:
            # Reuse Q4's allocation and roles with the supplied pair-sized
            # union bound. Only the row and mask strides differ.
            kwargs['br'] = 4
        super().__init__(*args, **kwargs)
        if rows_per_cta == 2:
            assert self.group <= 12 and self.kv_splits == 1
            assert self.fused_metadata and self.local_metadata
            self.br = 2
            self.RM = 2 * self.RB
        assert self.m_tile == 64 and self.zero_pad_q and self.active_softmax
        assert self.single_unit_mma and self.static_short
        assert not (self.use_pdl or self.arrive_on or self.pair_bar or self.rowidx or self.rsplit or self.ordered)
        assert self.mmasched
        self.sm100_q1_address_hoist = self.sm100_q1_address_hoist or (self.br == 1 and self.union_physical and self.kv_splits == 1)
        self.sm100_q1_single_unit = self.br == 1 and self.union_physical and self.kv_splits == 1
        self.q4_gate_stage = (self.br == 4 and self.hkv > 1
                              and not self.fused_metadata and self.kv_splits == 1
                              and (self.br * self.group) % 16 == 0)

    @cute.jit
    def __call__(self, mQ, mK, mV, mWS, mCount, mRM, mTicket, mBT, mGate, mOut, softmax_scale: Float32, nblocks: Int32, ncta: Int32, tbase: Int32, tmapK: Int64, tmapV: Int64, stream):
        self.scale_log2 = softmax_scale * Float32(LOG2_E)
        M, N, D = (64, self.n_tile, self.head_dim)
        qk_tiler = (M, N, D)
        pv_tiler = (M, D, N)
        qk_mma = sm100_utils.make_trivial_tiled_mma(self.dtype, self.dtype, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K, self.acc, tcgen05.CtaGroup.ONE, qk_tiler[:2], tcgen05.OperandSource.TMEM)
        qk_sm_mma = sm100_utils.make_trivial_tiled_mma(self.dtype, self.dtype, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.K, self.acc, tcgen05.CtaGroup.ONE, qk_tiler[:2], tcgen05.OperandSource.SMEM)
        pv_mma = sm100_utils.make_trivial_tiled_mma(self.dtype, self.dtype, tcgen05.OperandMajorMode.K, tcgen05.OperandMajorMode.MN, self.acc, tcgen05.CtaGroup.ONE, pv_tiler[:2], tcgen05.OperandSource.TMEM)
        q_tmem_lay = sm100_utils.make_smem_layout_a(qk_mma, (M, N, D // 2), self.dtype, 1)
        q_smem_lay = sm100_utils.make_smem_layout_a(qk_sm_mma, (M, N, D // 2), self.dtype, 1)
        k_lay = sm100_utils.make_smem_layout_b(qk_mma, qk_tiler, self.dtype, self.k_stages)
        v_lay = sm100_utils.make_smem_layout_b(pv_mma, pv_tiler, self.dtype, self.v_stages)
        p_tmem_lay = sm100_utils.make_smem_layout_a(pv_mma, pv_tiler, self.dtype, 1)
        self.atrex_qsa_bf16_decode(qk_mma, qk_sm_mma, pv_mma, q_tmem_lay, q_smem_lay, k_lay, v_lay, p_tmem_lay, mQ, mK, mV, mWS, mCount, mRM, mTicket, mBT, mGate, mOut, self.scale_log2, nblocks, ncta, tbase, tmapK, tmapV).launch(grid=(ncta, 1, 1), block=[self.threads, 1, 1], cluster=(1, 1, 1), stream=stream, min_blocks_per_mp=1, use_pdl=False)

    @cute.kernel
    def atrex_qsa_bf16_decode(self, qk_mma, qk_sm_mma, pv_mma, q_tmem_lay, q_smem_lay, k_lay, v_lay, p_tmem_lay, mQ, mK, mV, mWS, mCount, mRM, mTicket, mBT, mGate, mOut, scale_log2: Float32, nblocks: Int32, ncta: Int32, tbase: Int32, tmapK: Int64, tmapV: Int64):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(tidx // 32)
        cta, _, _ = cute.arch.block_idx()
        M, N, D = (64, self.n_tile, self.head_dim)
        group, page = (self.group, self.page_size)
        br, maxe, etile = (self.br, self.maxe, self.ent_tile)
        kst, vst, ssts = (self.k_stages, self.v_stages, self.s_stages)
        sm_threads = 32 * self.SOFTMAX_WARPS
        Hkv = Int32(mK.shape[2])
        total_units = nblocks * Hkv * self.kv_splits

        @cute.struct
        class SharedStorage:
            kfull: cute.struct.MemRange[Int64, kst]
            kfree: cute.struct.MemRange[Int64, kst]
            vfull: cute.struct.MemRange[Int64, vst]
            vfree: cute.struct.MemRange[Int64, vst]
            sfull: cute.struct.MemRange[Int64, ssts]
            pfull: cute.struct.MemRange[Int64, 2 * ssts]
            pvdone: cute.struct.MemRange[Int64, 4]
            qready: Int64
            odrained: Int64
            unit_posted: Int64
            unit_taken: Int64
            gstaged: Int64
            gconsumed: Int64
            sUnit: cute.struct.MemRange[Int32, 1]
            sMax: cute.struct.MemRange[Float32, 1 if self.fused_metadata else 256]
            sSum: cute.struct.MemRange[Float32, 1 if self.fused_metadata else 256]
            tmem_holding_buf: Int32
        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        if cutlass.const_expr(self.fused_metadata and self.br == 1):
            meta = smem.allocate_tensor(Int32, cute.make_layout((self.local_entries + 12 + self.RB + 8,)), 16)
            block = self._unit_blk(Int32(cta), nblocks, mCount)
            if cutlass.const_expr(self.decode_fixed):
                from .decode_metadata import build_decode_metadata
                mWS, mCount, mRM = build_decode_metadata(mWS, mCount, mRM, mBT, mK, meta, tidx, block, Int32(cta), nblocks, self.kv_splits, self.local_entries, self.RB, self.page_size, self.local_metadata, self.metadata_request_span)
            else:
                mWS, mCount, mRM = build_native_metadata(mWS, mCount, mRM, mBT, mK, meta, tidx, block, Int32(cta), nblocks, self.kv_splits, self.local_entries, self.RB, self.page_size, self.local_metadata, self.metadata_request_span, physical=self.union_physical)
        sK = smem.allocate_tensor(self.dtype, k_lay.outer, 128, k_lay.inner)
        sV = smem.allocate_tensor(self.dtype, v_lay.outer, 128, v_lay.inner)
        sQ = smem.allocate_tensor(self.dtype, q_smem_lay.outer, 128, q_smem_lay.inner)
        if cutlass.const_expr(self.fused_metadata and self.br in (2, 4)):
            from .union_metadata import build_union_metadata
            meta = smem.allocate_tensor(Int32, cute.make_layout((self.local_entries + 12 + 4 * self.RB + 8,)), 16)
            scratch = cute.make_tensor(cute.make_ptr(Int32, sK.iterator.toint(), cute.AddressSpace.smem, assumed_align=16), cute.make_layout((8192,)))
            block = self._unit_blk(Int32(cta), nblocks, mCount)
            if cutlass.const_expr(self.br == 2):
                from .decode_union import build_pair_metadata
                mWS, mCount, mRM = build_pair_metadata(mWS, mCount, mRM, mBT, mK, meta, scratch, tidx, block, self.local_entries, self.RB, self.page_size, self.metadata_request_span, self.union_hash_mix, self.union_full_first, self.union_physical)
            else:
                mWS, mCount, mRM = build_union_metadata(mWS, mCount, mRM, mBT, mK, meta, scratch, tidx, block, self.local_entries, self.RB, self.page_size, self.metadata_request_span, self.union_hash_mix, self.union_full_first, self.union_physical)
        sG = None
        if cutlass.const_expr(self.q4_gate_stage):
            sG = cute.make_tensor(cute.make_ptr(self.dtype, sK.iterator.toint(), cute.AddressSpace.smem, assumed_align=128), cute.make_layout((16384,)))
        if cutlass.const_expr(self.decode_gate):
            sG = smem.allocate_tensor(self.dtype, cute.make_layout((4096,)), 128)
        kfull = storage.kfull.data_ptr()
        kfree = storage.sfull.data_ptr()
        vfull = storage.vfull.data_ptr()
        vfree = storage.pvdone.data_ptr()
        sfull = storage.sfull.data_ptr()
        pfull = storage.pfull.data_ptr()
        pvdone = storage.pvdone.data_ptr()
        qready = storage.qready.ptr
        odrained = storage.odrained.ptr
        unit_posted = storage.unit_posted.ptr
        unit_taken = storage.unit_taken.ptr
        gstaged = storage.gstaged.ptr
        gconsumed = storage.gconsumed.ptr
        sUnit = storage.sUnit.data_ptr()
        sMax = storage.sMax.data_ptr()
        sSum = storage.sSum.data_ptr()
        if tidx == 0:
            for i in cutlass.range_constexpr(kst):
                cute.arch.mbarrier_init(kfull + i, 64)
            for i in cutlass.range_constexpr(vst):
                cute.arch.mbarrier_init(vfull + i, 32 * self.gwr)
            for i in cutlass.range_constexpr(ssts):
                cute.arch.mbarrier_init(sfull + i, 1)
                cute.arch.mbarrier_init(pfull + i, 32 if self.decode_gate else sm_threads)
                cute.arch.mbarrier_init(pfull + ssts + i, 32 if self.decode_gate else sm_threads)
            for i in cutlass.range_constexpr(4):
                cute.arch.mbarrier_init(pvdone + i, 1)
            cute.arch.mbarrier_init(qready, sm_threads)
            cute.arch.mbarrier_init(odrained, sm_threads)
            cute.arch.mbarrier_init(unit_posted, 1)
            cute.arch.mbarrier_init(unit_taken, 32 * self.take_warps)
            if cutlass.const_expr(self.decode_gate):
                cute.arch.mbarrier_init(gstaged, 96)
            elif cutlass.const_expr(self.q4_gate_stage):
                cute.arch.mbarrier_init(gstaged, 64)
            cute.arch.mbarrier_init_fence()
        tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=self.threads)
        tmem = utils.TmemAllocator(storage.tmem_holding_buf.ptr, barrier_for_retrieve=tmem_bar)
        tmem.allocate(cute.arch.get_max_tmem_alloc_cols(self.arch_str))
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc)
        qk_thr = qk_mma.get_slice(0)
        pv_thr = pv_mma.get_slice(0)
        q_col = N * ssts
        o_col = N * ssts + D // 4
        s_frag = qk_thr.make_fragment_C(cute.make_layout(qk_thr.partition_shape_C((M, N))))
        s_staged = cute.append(s_frag.layout, cute.make_layout((ssts,), stride=(N,)))
        tStS = cute.make_tensor(tmem_ptr, s_staged)
        o_frag = pv_thr.make_fragment_C(cute.make_layout(pv_thr.partition_shape_C((M, D))))
        tOtO = cute.make_tensor(tmem_ptr + o_col, o_frag.layout)
        q_ptr = tmem_ptr + q_col
        tQ_tm = cute.make_tensor(q_ptr, q_tmem_lay.outer)
        tQrQ_1 = qk_thr.make_fragment_A(tQ_tm)[None, None, None, 0]
        tQrQ = cute.make_tensor(cute.recast_ptr(q_ptr, dtype=self.dtype), tQrQ_1.layout)
        tSrK = qk_thr.make_fragment_B(sK)
        tQsQ = qk_sm_mma.get_slice(0).make_fragment_A(sQ)
        tOrV = pv_thr.make_fragment_B(sV)
        tP_tm = cute.make_tensor(tStS.iterator, p_tmem_lay.outer)
        tOrP_1 = pv_thr.make_fragment_A(tP_tm)[None, None, None, 0]
        tOrP = cute.make_tensor(cute.recast_ptr(tStS.iterator, dtype=self.dtype), cute.append(tOrP_1.layout, cute.make_layout((ssts,), stride=(N * 2,))))
        qk_idesc = pv_idesc = 0
        k_desc_hi = v_desc_hi = 0
        k_desc_lo0 = v_desc_lo0 = Int32(0)
        k_stage_d = v_stage_d = 0
        offs_bk = offs_bv = offs_aq = offs_ap = (0,)
        tmem_base = Int32(0)
        atom = cute.make_copy_atom(cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL), self.dtype, num_bits_per_copy=128)
        val_layout = cute.make_layout((1, self.copy_elems))
        kv_thr_layout = cute.make_ordered_layout((32 // self.gptr, self.gptr), order=(1, 0))
        kvgcopy = cute.make_tiled_copy_tv(atom, kv_thr_layout, val_layout)
        lane = tidx % 32
        kvgthr = kvgcopy.get_slice(lane)
        voff = _elem_pointer(mV, (0, 0, 0, 0)).toint() - _elem_pointer(mK, (0, 0, 0, 0)).toint()
        st1b = Int64(mK.stride[1]) * Int64(2)
        cute.arch.sync_threads()
        if warp_idx < self.SOFTMAX_WARPS:
            cute.arch.setmaxregister_increase(208)
            if cutlass.const_expr(self.decode_gate):
                if warp_idx == 0:
                    self._softmax_split16(tidx, warp_idx // 4, unit_posted, unit_taken, sUnit, total_units, nblocks, mQ, mWS, mCount, mRM, mGate, mOut, tStS, tOtO, sfull, pfull, pvdone, qready, odrained, sMax, sSum, sG, gstaged, gconsumed, q_col, sQ, scale_log2, M, N, D, group, br, maxe, etile, ssts)
                else:
                    if cutlass.const_expr(self.br == 2):
                        self._idle_pair_softmax(tidx, unit_posted, unit_taken, sUnit, total_units, nblocks, mQ, mCount, tStS, q_col, sQ, qready, mGate, sG, gstaged)
                    else:
                        self._idle_decode_softmax(tidx, unit_posted, unit_taken, sUnit, total_units, nblocks, mQ, mCount, tStS, q_col, sQ, qready, mGate, sG, gstaged)
            else:
                self._softmax_split16(tidx, warp_idx // 4, unit_posted, unit_taken, sUnit, total_units, nblocks, mQ, mWS, mCount, mRM, mGate, mOut, tStS, tOtO, sfull, pfull, pvdone, qready, odrained, sMax, sSum, sG, gstaged, gconsumed, q_col, sQ, scale_log2, M, N, D, group, br, maxe, etile, ssts)
        else:
            cute.arch.setmaxregister_decrease(88)
            if warp_idx == self.MMA_WARP:
                self._mma_role(unit_posted, unit_taken, sUnit, total_units, nblocks, mCount, mTicket, tbase, qk_mma, qk_sm_mma, pv_mma, tStS, tOtO, tQrQ, tQsQ, tSrK, tOrP, tOrV, kfull, kfree, vfull, vfree, sfull, pfull, pvdone, qready, odrained, etile, kst, vst, ssts, qk_idesc, pv_idesc, k_desc_hi, v_desc_hi, k_desc_lo0, v_desc_lo0, k_stage_d, v_stage_d, offs_aq, offs_bk, offs_ap, offs_bv, tmem_base, q_col, o_col)
            elif warp_idx == self.KLOAD_WARP:
                self._gather_role(mK, sK, kvgcopy, kvgthr, mWS, mBT, mCount, unit_posted, unit_taken, sUnit, total_units, nblocks, page, D, Int64(0), st1b, kfull, kfree, kst, lane, maxe, etile, False, 0, 2, tmapK)
                if cutlass.const_expr(self.q4_gate_stage):
                    self._stage_q4_gate(mGate, sG, mCount, Int32(cta), nblocks, sfull, gstaged, lane + 0)

            elif warp_idx == self.VLOAD_WARP:
                self._gather_role(mK, sV, kvgcopy, kvgthr, mWS, mBT, mCount, unit_posted, unit_taken, sUnit, total_units, nblocks, page, D, voff, st1b, vfull, vfree, vst, lane, maxe, etile, True, 0, self.gwr, tmapV)
            elif warp_idx == self.KLOAD_WARP2:
                self._gather_role(mK, sK, kvgcopy, kvgthr, mWS, mBT, mCount, unit_posted, unit_taken, sUnit, total_units, nblocks, page, D, Int64(0), st1b, kfull, kfree, kst, lane, maxe, etile, False, 1, 2)
                if cutlass.const_expr(self.q4_gate_stage):
                    self._stage_q4_gate(mGate, sG, mCount, Int32(cta), nblocks, sfull, gstaged, lane + 32)

            elif warp_idx == self.VLOAD_WARP2:
                if cutlass.const_expr(self.gwr == 2):
                    self._gather_role(mK, sV, kvgcopy, kvgthr, mWS, mBT, mCount, unit_posted, unit_taken, sUnit, total_units, nblocks, page, D, voff, st1b, vfull, vfree, vst, lane, maxe, etile, True, 1, 2)
        cute.arch.sync_threads()
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)

    @cute.jit
    def _softmax_split16(self, tidx, wg, unit_posted, unit_taken, sUnit, total_units, nblocks, mQ, mWS, mCount, mRM, mGate, mOut, tStS, tOtO, sfull, pfull, pvdone, qready, odrained, sMax, sSum, sG, gstaged, gconsumed, q_col, sQ, scale_log2, M, N, D, group, br, maxe, etile, ssts):
        """Contiguous48 columns/thread; each warp owns16 complete rows.

            Transport/max/BF16 store mapping independently passed split16_probe.
            XOR16 exchanges replace the old cross-WG shared-memory barriers.
            Both WGs now rescale/drain disjoint rows, while WG0 still refills Q
            using the original128-row mapping. All256 P arrivals gate PV.
            """
        sg = tidx % 128
        lane = tidx % 32
        half = lane // 16
        L = self._logical_row(sg, wg)
        base = cute.make_ptr(self.acc, (tStS.iterator + (wg * 16 + sg // 32 * 32) * 65536).toint(), cute.AddressSpace.tmem, assumed_align=16)
        rS = cute.make_rmem_tensor((48,), self.acc)
        rSelected = cute.make_rmem_tensor((12,), self.acc)
        rP = cute.make_rmem_tensor((48,), self.dtype)
        rPW = cute.make_tensor(cute.recast_ptr(rP.iterator, dtype=self.acc), cute.make_layout((24,)))
        rows = Int32(mQ.shape[0])
        ph = Int32(0)
        gt = Int32(0)
        q_ptr = tStS.iterator + q_col
        self._mwait(unit_posted, 0)
        u = Int32(sUnit.load())
        cute.arch.mbarrier_arrive(unit_taken)
        u0 = cutlass.min(u, total_units - 1)
        b0 = self._unit_blk(u0, nblocks, mCount)
        h0 = self._unit_kvh(u0, nblocks)
        if wg == 0:
            self._refill_q_tmem(sQ, mQ, q_ptr, None, None, None, None, b0 * br, h0, group, group * cutlass.min(rows - b0 * br, Int32(br)), sg)
            cute.arch.fence_view_async_tmem_store()
        else:
            self._refill_q_smem(sQ, mQ, b0 * br, h0, group, group * cutlass.min(rows - b0 * br, Int32(br)), sg)
        cute.arch.mbarrier_arrive(qready)
        while u < total_units:
            blk = self._unit_blk(u, nblocks, mCount)
            kvh = self._unit_kvh(u, nblocks)
            row0 = blk * br
            lane_rows = group * cutlass.min(rows - row0, Int32(br))
            entry_begin, count = self._kv_slice(self._full_count(mCount, blk), u)
            tile_begin = entry_begin // etile
            num_tiles = (count + etile - 1) // etile
            mref = Float32(-cutlass.Float32.inf)
            row_sum = Float32(0)
            nfast = cutlass.max(cutlass.min(num_tiles - 1, self._fast_count(mCount, nblocks, blk) - tile_begin), Int32(0))
            rr = cutlass.min(L // group, Int32(br - 1))
            moff = blk * self.RM + rr * self.RB
            if cutlass.const_expr(self.local_metadata):
                moff = rr * self.RB
            first_w = tile_begin * 3 + half * 3 // 2
            pref_a = Int32(mRM[moff + first_w])
            pref_b = Int32(mRM[moff + first_w + 1])
            for span in cutlass.range_constexpr(2):
                begin = Int32(0) if span == 0 else nfast
                trip = nfast if span == 0 else num_tiles - nfast
                for jj in cutlass.range(trip):
                    tile = begin + jj
                    g = gt + tile
                    s = g % ssts
                    a = pref_a
                    b = pref_b
                    next_w = (tile_begin + tile + 1) * 3 + half * 3 // 2
                    pref_a = Int32(mRM[moff + cutlass.min(next_w, Int32(self.RB - 1))])
                    pref_b = Int32(mRM[moff + cutlass.min(next_w + 1, Int32(self.RB - 1))])
                    lo = cutlass.select_(half == 0, a, a >> 16 & Int32(65535) | b << 16)
                    hi = cutlass.select_(half == 0, b & Int32(65535), b >> 16 & Int32(65535))
                    if cutlass.const_expr(span == 1):
                        nv = cutlass.max(cutlass.min((count - tile * 24 - half * 12) * 4, Int32(48)), Int32(0))
                        lo = lo & cutlass.select_(nv >= 32, Int32(-1), (Int32(1) << cutlass.min(nv, Int32(31))) - 1)
                        hi = hi & (Int32(1) << cutlass.max(nv - 32, Int32(0))) - 1
                    self._mwait(sfull + s, g // ssts & 1)
                    live_warp = Int32(1)
                    live_warp = Int32(L - lane % 16 < lane_rows)
                    if live_warp != 0:
                        if cutlass.const_expr(span == 0):
                            for chunk in cutlass.range_constexpr(3):
                                vals = cute.make_tensor(rS.iterator + chunk * 16, cute.make_layout((16,)))
                                qsa_sm_ptx.split_load(Int32((base + s * N + chunk * 16).toint()), vals, 16, 48)
                        else:
                            for chunk in cutlass.range_constexpr(3):
                                vals = cute.make_tensor(rS.iterator + chunk * 16, cute.make_layout((16,)))
                                qsa_sm_ptx.split_load(Int32((base + s * N + chunk * 16).toint()), vals, 16, 48)
                        cute.arch.fence_view_async_tmem_load()
                        bits = [lo, hi]
                        if cutlass.const_expr(span == 0):
                            for ent in cutlass.range_constexpr(12):
                                quad = cute.arch.fmax(cute.arch.fmax(Float32(rS[ent * 4]), Float32(rS[ent * 4 + 1])), cute.arch.fmax(Float32(rS[ent * 4 + 2]), Float32(rS[ent * 4 + 3])))
                                valid = bits[ent >> 3] >> 4 * (ent & 7) + 3 & 1 != 0
                                rSelected[ent] = cutlass.select_(valid, quad, Float32(-cutlass.Float32.inf))
                            part = rSelected.load().reduce(cute.ReductionOp.MAX, Float32(-cutlass.Float32.inf), 0) * scale_log2
                        else:
                            for j in cutlass.range_constexpr(48):
                                rS[j] = cutlass.select_(bits[j >> 5] >> (j & 31) & 1 != 0, Float32(rS[j]), Float32(-cutlass.Float32.inf))
                            part = rS.load().reduce(cute.ReductionOp.MAX, Float32(-cutlass.Float32.inf), 0) * scale_log2
                        mx = cute.arch.fmax(part, Float32(cute.arch.shuffle_sync(part, lane ^ 16, mask_and_clamp=31)))
                        first = mref == Float32(-cutlass.Float32.inf)
                        vote = cute.arch.vote_any_sync(mx > mref)
                        newref = cute.arch.fmax(mx, mref)
                        neg = Float32(0) - newref
                        if cutlass.const_expr(span == 0):
                            biases = {}
                            for ent in cutlass.range_constexpr(12):
                                biasbits = bits[ent >> 3] << 28 - 4 * (ent & 7)
                                biases[ent] = cutlass.select_(biasbits < Int32(0), neg, Float32(-cutlass.Float32.inf))
                            for pair in cutlass.range_constexpr(16):
                                x, y = cute.arch.fma_packed_f32x2((Float32(rS[pair * 2]), Float32(rS[pair * 2 + 1])), (scale_log2, scale_log2), (biases[pair // 2], biases[pair // 2]))
                                rS[pair * 2] = cute.math.exp2(x, fastmath=True)
                                rS[pair * 2 + 1] = cute.math.exp2(y, fastmath=True)
                            src_first = cute.make_tensor(rS.iterator, cute.make_layout((32,)))
                            dst_first = cute.make_tensor(rP.iterator, cute.make_layout((32,)))
                            dst_first.store(src_first.load().to(self.dtype))
                            for chunk in cutlass.range_constexpr(2):
                                words = cute.make_tensor(rPW.iterator + chunk * 8, cute.make_layout((8,)))
                                qsa_sm_ptx.split_store(Int32((base + s * N + chunk * 8).toint()), words, 8, 24)
                            if vote and tile > 0:
                                self._mwait(pvdone + (g - 1 & 3), (g - 1) // 4 & 1)
                                delta = cutlass.select_(first, Float32(0), mref - newref)
                                alpha = cute.math.exp2(delta, fastmath=True)
                                self._rescale_split16(tOtO, wg, sg, alpha)
                                row_sum = row_sum * alpha
                            cute.arch.fence_view_async_tmem_store()
                            tailneg = publish_bias(pfull + s, neg)
                            for ent in cutlass.range_constexpr(8, 12):
                                biasbits = bits[ent >> 3] << 28 - 4 * (ent & 7)
                                biases[ent] = cutlass.select_(biasbits < Int32(0), tailneg, Float32(-cutlass.Float32.inf))
                            for pair in cutlass.range_constexpr(16, 24):
                                x, y = cute.arch.fma_packed_f32x2((Float32(rS[pair * 2]), Float32(rS[pair * 2 + 1])), (scale_log2, scale_log2), (biases[pair // 2], biases[pair // 2]))
                                rS[pair * 2] = cute.math.exp2(x, fastmath=True)
                                rS[pair * 2 + 1] = cute.math.exp2(y, fastmath=True)
                            src_last = cute.make_tensor(rS.iterator + 32, cute.make_layout((16,)))
                            dst_last = cute.make_tensor(rP.iterator + 32, cute.make_layout((16,)))
                            dst_last.store(src_last.load().to(self.dtype))
                            words = cute.make_tensor(rPW.iterator + 16, cute.make_layout((8,)))
                            qsa_sm_ptx.split_store(Int32((base + s * N + 16).toint()), words, 8, 24)
                        else:
                            for j in cutlass.range_constexpr(48):
                                prob = cute.math.exp2(rS[j] * scale_log2 + neg, fastmath=True)
                                rS[j] = cutlass.select_(bits[j >> 5] >> (j & 31) & 1 != 0, prob, Float32(0))
                            rP.store(rS.load().to(self.dtype))
                            for chunk in cutlass.range_constexpr(3):
                                words = cute.make_tensor(rPW.iterator + chunk * 8, cute.make_layout((8,)))
                                qsa_sm_ptx.split_store(Int32((base + s * N + chunk * 8).toint()), words, 8, 24)
                            if vote and tile > 0:
                                self._mwait(pvdone + (g - 1 & 3), (g - 1) // 4 & 1)
                                delta = cutlass.select_(first, Float32(0), mref - newref)
                                alpha = cute.math.exp2(delta, fastmath=True)
                                self._rescale_split16(tOtO, wg, sg, alpha)
                                row_sum = row_sum * alpha
                            cute.arch.fence_view_async_tmem_store()
                            cute.arch.mbarrier_arrive(pfull + s)
                        tsum = self._sum_packed(rS, 48)
                        row_sum += tsum
                        mref = newref
                        cute.arch.fence_view_async_tmem_store()
                        cute.arch.mbarrier_arrive(pfull + ssts + s)
                    else:
                        cute.arch.mbarrier_arrive(pfull + s)
                        cute.arch.mbarrier_arrive(pfull + ssts + s)
            gt_next = gt + num_tiles
            ph += 1
            self._mwait(unit_posted, ph & 1)
            u2 = Int32(sUnit.load())
            cute.arch.mbarrier_arrive(unit_taken)
            u2 = total_units
            row_sum += Float32(cute.arch.shuffle_sync(row_sum, lane ^ 16, mask_and_clamp=31))
            inv = cutlass.select_(row_sum != Float32(0), cute.arch.rcp_approx(row_sum), Float32(0))
            if u2 < total_units:
                bn = self._unit_blk(u2, nblocks, mCount)
                hn = self._unit_kvh(u2, nblocks)
                if wg == 0:
                    self._refill_q_tmem(sQ, mQ, q_ptr, None, None, None, None, bn * br, hn, group, group * cutlass.min(rows - bn * br, Int32(br)), sg)
                    cute.arch.fence_view_async_tmem_store()
                else:
                    self._refill_q_smem(sQ, mQ, bn * br, hn, group, group * cutlass.min(rows - bn * br, Int32(br)), sg)
                cute.arch.mbarrier_arrive(qready)
            if gt_next > 0:
                self._mwait(pvdone + (gt_next - 1 & 3), (gt_next - 1) // 4 & 1)
            if cutlass.const_expr(self.kv_splits > 1):
                drain_live = Int32(1)
                drain_live = Int32(L - lane % 16 < lane_rows)
                if drain_live != 0:
                    self._drain_mtp_partial(tOtO, wg, sg, L, row0, kvh, group, lane_rows, mref, row_sum, u, mGate, mOut)
            else:
                self._drain_split16(tOtO, wg, sg, L, row0, kvh, group, lane_rows, inv, mGate, mOut, sG, gstaged)
            gt = gt_next
            u = u2

    @cute.jit
    def _drain_split16(self,tOtO,wg,sg,L,row0,kvh,group,lane_rows,inv,mGate,mOut,sG=None,gstaged=None):
        ptr=cute.make_ptr(self.acc,(tOtO.iterator+(wg*16+(sg//32)*32)*65536).toint(),
            cute.AddressSpace.tmem,assumed_align=16)
        r=cute.make_rmem_tensor((32,),self.acc)
        rg=cute.make_rmem_tensor((32,),self.dtype);rw=cute.make_rmem_tensor((32,),self.dtype)
        if cutlass.const_expr(self.decode_gate or self.q4_gate_stage):self._mwait(gstaged,0)
        for chunk in cutlass.range_constexpr(4):
            qsa_sm_ptx.split_load(Int32((ptr+chunk*32).toint()),r,32,128)
            cute.arch.fence_view_async_tmem_load()
            if L<lane_rows:
                row=row0+L//group;head=kvh*group+L%group;col=(sg%32//16)*128+chunk*32
                gp=cute.make_ptr(self.dtype,_elem_pointer(mGate,(row,head,col)).toint(),cute.AddressSpace.gmem,assumed_align=32)
                op=cute.make_ptr(self.dtype,_elem_pointer(mOut,(row,head,col)).toint(),cute.AddressSpace.gmem,assumed_align=32)
                if cutlass.const_expr(self.decode_gate or self.q4_gate_stage):
                    for part in cutlass.range_constexpr(4):
                        src=cute.make_tensor(sG.iterator+(sg//32)*4096+(chunk*4+part)*256+(sg%32)*8,cute.make_layout((8,)))
                        dst=cute.make_tensor(rg.iterator+part*8,cute.make_layout((8,)))
                        cute.autovec_copy(src,dst)
                else:
                    cute.autovec_copy(cute.make_tensor(gp,cute.make_layout((32,))),rg)
                zero2=(Float32(0),Float32(0));half2=(Float32(.5),Float32(.5))
                for pair in cutlass.range_constexpr(16):
                    j=pair*2
                    o0,o1=cute.arch.fma_packed_f32x2((Float32(r[j]),Float32(r[j+1])),(inv,inv),zero2)
                    a0=Float32(cutlass.select_(inv!=Float32(0),o0,Float32(0))).to(self.dtype)
                    a1=Float32(cutlass.select_(inv!=Float32(0),o1,Float32(0))).to(self.dtype)
                    g0,g1=cute.arch.fma_packed_f32x2((Float32(rg[j]),Float32(rg[j+1])),half2,zero2)
                    t0=qsa_sm_ptx.tanh_approx(g0);t1=qsa_sm_ptx.tanh_approx(g1)
                    s0,s1=cute.arch.fma_packed_f32x2((t0,t1),half2,half2)
                    y0,y1=cute.arch.fma_packed_f32x2((a0.to(Float32),a1.to(Float32)),(s0,s1),zero2)
                    rw[j]=y0.to(self.dtype);rw[j+1]=y1.to(self.dtype)
                cute.autovec_copy(rw,cute.make_tensor(op,cute.make_layout((32,))))

    @cute.jit
    def _stage_q4_gate(self, mGate, sG, mCount, unit, nblocks, sfull, gstaged, helper):
        block = self._unit_blk(unit, nblocks, mCount)
        kvh = self._unit_kvh(unit, nblocks)
        count = self._full_count(mCount, block)
        tiles = (count + self.ent_tile - 1) // self.ent_tile
        if tiles > 0:
            last = tiles - 1
            self._mwait(sfull + last % self.s_stages, (last // self.s_stages) & 1)
        vectors = self.br * self.group * 32
        for iteration in cutlass.range_constexpr((self.br * self.group * 32 + 63) // 64):
            i = helper + iteration * 64
            group_warp = i // 512
            local = i % 512
            glane = local % 32
            logical = group_warp * 16 + glane % 16
            col = (glane // 16) * 128 + (local // 32) * 8
            if i < vectors:
                src = _elem_pointer(mGate, (block * self.br + logical // self.group,
                                           kvh * self.group + logical % self.group, col)).toint()
                dst = Int32(Int64((sG.iterator + i * 8).toint()) & 0x3ffff)
                qsa_sm_ptx.cp_async_16(dst, src)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.fence_view_async_shared()
        cute.arch.mbarrier_arrive(gstaged)

    @cute.jit
    def _idle_pair_softmax(self, tidx, unit_posted, unit_taken, sUnit, total_units,
                           nblocks, mQ, mCount, tStS, q_col, sQ, qready, mGate, sG, gstaged):
        assert self.br == 2 and self.br * self.group <= 16 and self.single_unit_mma
        sg = tidx % 128
        self._mwait(unit_posted, 0)
        unit = Int32(sUnit.load())
        cute.arch.mbarrier_arrive(unit_taken)
        block = self._unit_blk(unit, nblocks, mCount)
        kvh = self._unit_kvh(unit, nblocks)
        self._refill_q_tmem(sQ, mQ, tStS.iterator + q_col, None, None, None, None,
                            block * 2, kvh, self.group, Int32(2 * self.group), sg)
        cute.arch.fence_view_async_tmem_store()
        cute.arch.mbarrier_arrive(qready)
        for iteration in cutlass.range_constexpr(6):
            index = tidx - 32 + iteration * 96
            if index < 512:
                lane = index % 32
                logical = lane % 16
                col = (lane // 16) * 128 + (index // 32) * 8
                safe = cutlass.min(logical, Int32(2 * self.group - 1))
                src = _elem_pointer(mGate, (block * 2 + safe // self.group,
                                            kvh * self.group + safe % self.group, col)).toint()
                dst = Int32(Int64((sG.iterator + index * 8).toint()) & 0x3ffff)
                size = cutlass.select_(logical < 2 * self.group, Int32(16), Int32(0))
                qsa_sm_ptx.cp_async_16_zfill(dst, src, size)
        cute.arch.cp_async_commit_group()
        cute.arch.cp_async_wait_group(0)
        cute.arch.fence_view_async_shared()
        cute.arch.mbarrier_arrive(gstaged)
        self._mwait(unit_posted, 1)
        cute.arch.mbarrier_arrive(unit_taken)
