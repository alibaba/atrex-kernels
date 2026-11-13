"""V(g) storage becomes free at the existing PV-retirement event(g).

Producer g waits event(g-Vst) at slot((g-Vst)&3), phase((g-Vst)//4)&1.
The producer cannot lap its event: later PV requires later V gathered by
this same sequential producer. Shared storage layout remains unchanged.
"""
from __future__ import annotations
import cutlass
import cutlass.cute as cute
import cutlass.pipeline as pipeline
import cutlass.utils as utils
from cutlass.cute.nvgpu import cpasync,tcgen05
from cutlass.cute.typing import Float32,Int32,Int64
from ....common_utils.fp8.decode.native_metadata import build_native_metadata
from ....common_utils.fp8.decode.layouts import _elem_pointer
from ....common_utils import qsa_sm_ptx


from cutlass._mlir.dialects import llvm


import cutlass.utils.blackwell_helpers as sm100_utils
from ....common_utils.fp8.decode.layouts import LOG2_E
from ....common_utils.fp8.decode.qsa_kernel import QSAForward as Shared, publish_bias

@cute.jit
def _load_quad_max(addr:Int32,values:cute.Tensor):
    result=llvm.inline_asm(llvm.StructType.get_literal([Float32.mlir_type]*5),
        [addr.ir_value()],
        "tcgen05.ld.red.sync.aligned.16x32bx2.x4.f32.max {$0,$1,$2,$3}, $4, [$5], 64;",
        "=f,=f,=f,=f,=f,r",has_side_effects=True,is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT)
    for i in cutlass.range_constexpr(4):
        values[i]=Float32(llvm.extractvalue(Float32.mlir_type,result,[i]))
    return Float32(llvm.extractvalue(Float32.mlir_type,result,[4]))


class QSAForward(Shared):
    @cute.jit
    def __call__(self, mQ, mK, mV, mWS, mCount, mRM, mTicket, mBT, mGate,
                 mOut, softmax_scale: Float32, v_scale: Float32, nblocks: Int32, ncta: Int32,
                 tbase: Int32, tmapK: Int64, tmapV: Int64, stream):
        self.scale_log2 = softmax_scale * Float32(LOG2_E)
        M, N, D = self.m_tile, self.n_tile, self.head_dim
        qk_tiler = (M, N, D)
        pv_tiler = (M, D, N)
        qk_mma = sm100_utils.make_trivial_tiled_mma(
            self.dtype, self.dtype, tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K, self.acc, tcgen05.CtaGroup.ONE,
            qk_tiler[:2], tcgen05.OperandSource.TMEM)
        qk_sm_mma = sm100_utils.make_trivial_tiled_mma(
            self.dtype,self.dtype,tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.K,self.acc,tcgen05.CtaGroup.ONE,
            qk_tiler[:2],tcgen05.OperandSource.SMEM)
        # The PV A operand (P) is read from TMEM, aliasing the drained S stage.
        pv_mma = sm100_utils.make_trivial_tiled_mma(
            self.dtype, self.dtype, tcgen05.OperandMajorMode.K,
            tcgen05.OperandMajorMode.MN, self.acc, tcgen05.CtaGroup.ONE,
            pv_tiler[:2], tcgen05.OperandSource.TMEM)
        # V5: the QK A operand (Q) is ALSO read from TMEM (its own resident
        # column range), freeing the full 64KB sQ SMEM buffer for deeper K/V
        # gather rings.  Q is refilled by the softmax warpgroup with LDG ->
        # register -> tcgen05.st once per unit.
        q_tmem_lay = sm100_utils.make_smem_layout_a(qk_mma, (M,N,D//2),
                                                    self.dtype, 1)
        q_smem_lay = sm100_utils.make_smem_layout_a(qk_sm_mma,(M,N,D),self.dtype,1)
        k_lay = sm100_utils.make_smem_layout_b(qk_mma, qk_tiler, self.dtype,
                                               self.k_stages)
        v_lay = sm100_utils.make_smem_layout_b(pv_mma, pv_tiler, self.dtype,
                                               self.v_stages)
        # Single-stage P layout: the stage stride is applied by hand in the
        # kernel, because make_smem_layout_a packs stages back-to-back (N/2
        # columns apart) while the S/P ping-pong stages are N columns apart.
        p_tmem_lay = sm100_utils.make_smem_layout_a(pv_mma, pv_tiler, self.dtype, 1)
        self.atrex_qsa_sm103_attention(qk_mma, qk_sm_mma, pv_mma, q_tmem_lay, q_smem_lay, k_lay, v_lay, p_tmem_lay,
                    mQ, mK, mV, mWS, mCount, mRM, mTicket, mBT, mGate, mOut,
                    self.scale_log2, v_scale, nblocks, ncta, tbase, tmapK,
                    tmapV).launch(
            grid=(ncta, 1, 1), block=[self.threads, 1, 1],
            cluster=(1, 1, 1), stream=stream, min_blocks_per_mp=1,
            use_pdl=self.use_pdl)

    @cute.kernel
    def atrex_qsa_sm103_attention(self, qk_mma, qk_sm_mma, pv_mma, q_tmem_lay, q_smem_lay, k_lay, v_lay, p_tmem_lay,
               mQ, mK, mV, mWS, mCount, mRM, mTicket, mBT, mGate, mOut,
               scale_log2: Float32, v_scale: Float32, nblocks: Int32, ncta: Int32,
               tbase: Int32, tmapK: Int64, tmapV: Int64):
        tidx, _, _ = cute.arch.thread_idx()
        warp_idx = cute.arch.make_warp_uniform(tidx // 32)
        cta, _, _ = cute.arch.block_idx()
        M, N, D = self.m_tile, self.n_tile, self.head_dim
        group, page = self.group, self.page_size
        br, maxe, etile = self.br, self.maxe, self.ent_tile
        kst, vst, ssts = self.k_stages, self.v_stages, self.s_stages
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
            sMax: cute.struct.MemRange[Float32, 1 if self.fused_metadata else 256]   # per-tile half-max swap
            sSum: cute.struct.MemRange[Float32, 1 if self.fused_metadata else 256]   # epilogue row_sum swap
            tmem_holding_buf: Int32

        smem = utils.SmemAllocator()
        storage = smem.allocate(SharedStorage)
        if cutlass.const_expr(self.fused_metadata and self.br==1):
            meta=smem.allocate_tensor(Int32,cute.make_layout((self.local_entries+12+self.RB+8,)),16)
            block=self._unit_blk(Int32(cta),nblocks,mCount)
            if cutlass.const_expr(self.decode_fixed):
                from ....common_utils.fp8.decode.decode_metadata import build_decode_metadata
                mWS,mCount,mRM=build_decode_metadata(mWS,mCount,mRM,mBT,mK,meta,tidx,block,Int32(cta),nblocks,
                    self.kv_splits,self.local_entries,self.RB,self.page_size,self.local_metadata,self.metadata_request_span)
            else:
                mWS,mCount,mRM=build_native_metadata(mWS,mCount,mRM,mBT,mK,meta,tidx,block,Int32(cta),nblocks,
                    self.kv_splits,self.local_entries,self.RB,self.page_size,self.local_metadata,self.metadata_request_span)
        sK = smem.allocate_tensor(self.dtype, k_lay.outer, 128, k_lay.inner)
        sV = smem.allocate_tensor(self.dtype, v_lay.outer, 128, v_lay.inner)
        sQ = smem.allocate_tensor(self.dtype,q_smem_lay.outer,128,q_smem_lay.inner)
        if cutlass.const_expr(self.fused_metadata and self.br in (2,4)):
            from ....common_utils.fp8.decode.union_metadata import build_union_metadata
            meta=smem.allocate_tensor(Int32,cute.make_layout((self.local_entries+12+4*self.RB+8,)),16)
            scratch=cute.make_tensor(cute.make_ptr(Int32,sK.iterator.toint(),cute.AddressSpace.smem,assumed_align=16),cute.make_layout((8192,)))
            block=self._unit_blk(Int32(cta),nblocks,mCount)
            mWS,mCount,mRM=build_union_metadata(mWS,mCount,mRM,mBT,mK,meta,scratch,tidx,block,
                self.local_entries,self.RB,self.page_size,self.metadata_request_span,self.union_hash_mix,self.union_full_first,self.union_physical,self.br,
                direct_masks=self.group == 16 and getattr(self, "grouped_union", False), grouped_insert=getattr(self, "grouped_union", False))
        sG = None
        if cutlass.const_expr(getattr(self, "q4_gate_stage", False)):
            sG = cute.make_tensor(cute.make_ptr(cutlass.BFloat16, sK.iterator.toint(), cute.AddressSpace.smem, assumed_align=128), cute.make_layout((16384,)))
        if cutlass.const_expr(self.decode_gate):
            if cutlass.const_expr(self.gate_factor):
                sG=smem.allocate_tensor(Float32,cute.make_layout((256*self.br*self.group,)),128)
            else:
                sG=smem.allocate_tensor(cutlass.BFloat16,cute.make_layout((4096*self.live_softmax_warps,)),128)

        kfull = storage.kfull.data_ptr()
        kfree = storage.kfree.data_ptr() if cutlass.const_expr(kst!=ssts) else storage.sfull.data_ptr()
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
                cute.arch.mbarrier_init(kfull + i, 32*self.k_load_warps)
                if cutlass.const_expr(kst!=ssts):cute.arch.mbarrier_init(kfree + i, 1)
            for i in cutlass.range_constexpr(vst):
                cute.arch.mbarrier_init(vfull + i, 32 * self.gwr)
            for i in cutlass.range_constexpr(ssts):
                cute.arch.mbarrier_init(sfull + i, 1)
                cute.arch.mbarrier_init(pfull + i, 32*self.live_softmax_warps if self.m_tile==64 else sm_threads)
                cute.arch.mbarrier_init(pfull + ssts + i, 32*self.live_softmax_warps if self.m_tile==64 else sm_threads)
            for i in cutlass.range_constexpr(4):
                cute.arch.mbarrier_init(pvdone + i, 1)
            cute.arch.mbarrier_init(qready, sm_threads)
            cute.arch.mbarrier_init(odrained, sm_threads)
            cute.arch.mbarrier_init(unit_posted, 1)
            cute.arch.mbarrier_init(unit_taken, 32 * self.take_warps)
            if cutlass.const_expr(self.decode_gate):
                cute.arch.mbarrier_init(gstaged,32*(4-self.live_softmax_warps))
            if cutlass.const_expr(getattr(self, "q4_gate_stage", False)):
                cute.arch.mbarrier_init(gstaged, 64)
            cute.arch.mbarrier_init_fence()

        tmem_bar = pipeline.NamedBarrier(barrier_id=1, num_threads=self.threads)
        tmem = utils.TmemAllocator(storage.tmem_holding_buf.ptr,
                                   barrier_for_retrieve=tmem_bar)
        tmem.allocate(cute.arch.get_max_tmem_alloc_cols(self.arch_str))
        tmem.wait_for_alloc()
        tmem_ptr = tmem.retrieve_ptr(self.acc)

        qk_thr = qk_mma.get_slice(0)
        pv_thr = pv_mma.get_slice(0)
        # FP8 N128: S/P stages own columns [0,256); O owns [256,512).
        # Q is fully resident in SMEM. No TMEM capacity is reserved for Q.
        # make_fragment_C derives the TMEM lane/column encoding; the stage mode
        # is appended with an N-column stride so stage s sits at column s*N.
        q_col = N * ssts
        o_col = N * ssts
        s_frag = qk_thr.make_fragment_C(cute.make_layout(qk_thr.partition_shape_C((M, N))))
        s_staged = cute.append(s_frag.layout, cute.make_layout((ssts,), stride=(N,)))
        tStS = cute.make_tensor(tmem_ptr, s_staged)
        o_frag = pv_thr.make_fragment_C(cute.make_layout(pv_thr.partition_shape_C((M, D))))
        tOtO = cute.make_tensor(tmem_ptr + o_col, o_frag.layout)

        # Inherited half-D fragment describes the atom count only; the QK
        # issue path below reads the full Q from tQsQ in shared memory.
        q_ptr = tmem_ptr + q_col
        tQ_tm = cute.make_tensor(q_ptr, q_tmem_lay.outer)
        tQrQ_1 = qk_thr.make_fragment_A(tQ_tm)[(None, None, None, 0)]
        tQrQ = cute.make_tensor(cute.recast_ptr(q_ptr, dtype=self.dtype),
                                tQrQ_1.layout)
        tSrK = qk_thr.make_fragment_B(sK)
        tQsQ = qk_sm_mma.get_slice(0).make_fragment_A(sQ)
        tOrV = pv_thr.make_fragment_B(sV)
        # PV A operand: E4M3 P packed four-per-word inside the S TMEM columns.
        # The fragment layout is built for one stage, then a stage mode with the
        # real S/P ping-pong stride is appended: P stage s lives at TMEM column
        # s*N, which is s*N*4 E4M3 elements from the base.
        tP_tm = cute.make_tensor(tStS.iterator, p_tmem_lay.outer)
        tOrP_1 = pv_thr.make_fragment_A(tP_tm)[(None, None, None, 0)]
        tOrP = cute.make_tensor(
            cute.recast_ptr(tStS.iterator, dtype=self.dtype),
            cute.append(tOrP_1.layout, cute.make_layout((ssts,), stride=(N * 4,))))

        # ---- precomputed PTX-MMA descriptors (trace-time constants) ----
        # The single-inline-asm k-loop issue replaces the per-atom cute.gemm
        # descriptor construction on the MMA warp (e8 trace: 377 cyc/tile of
        # issue work inside the critical 1-tile kfree->regather->kfull loop).
        qk_idesc = pv_idesc = 0
        k_desc_hi = v_desc_hi = 0
        k_desc_lo0 = v_desc_lo0 = Int32(0)
        k_stage_d = v_stage_d = 0
        offs_bk = offs_bv = offs_aq = offs_ap = (0,)
        tmem_base = Int32(0)

        # ---- 32-thread gather copy: one warp per K/V ring ----
        atom = cute.make_copy_atom(
            cpasync.CopyG2SOp(cache_mode=cpasync.LoadCacheMode.GLOBAL),
            self.dtype, num_bits_per_copy=128)
        val_layout = cute.make_layout((1, self.copy_elems))
        kv_thr_layout = cute.make_ordered_layout((32 // self.gptr, self.gptr),
                                                 order=(1, 0))
        kvgcopy = cute.make_tiled_copy_tv(atom, kv_thr_layout, val_layout)
        lane = tidx % 32
        kvgthr = kvgcopy.get_slice(lane)

        voff = _elem_pointer(mV, (0, 0, 0, 0)).toint() - \
            _elem_pointer(mK, (0, 0, 0, 0)).toint()
        st1b = Int64(mK.stride[1])  # bytes per token step in a page


        cute.arch.sync_threads()   # mbarrier init + TMEM ptr visible to all

        # ==================== role split =====================================
        if warp_idx < self.SOFTMAX_WARPS:
            # Stay within the CTA's allocated pool, not just the SM total.
            #192*256+96*128=61440, leaving slack versus168*384=64512.
            cute.arch.setmaxregister_increase(208)
            if cutlass.const_expr(self.m_tile==64 and self.single_unit_mma):
                if warp_idx<self.live_softmax_warps:
                    self._softmax_split16(
                        tidx, warp_idx // 4, unit_posted, unit_taken, sUnit,
                        total_units, nblocks,
                        mQ, mWS, mCount, mRM, mGate, mOut,
                        tStS, tOtO, sfull, pfull, pvdone, qready, odrained,
                        sMax, sSum, sG, gstaged, gconsumed,
                        q_col, sQ,
                        scale_log2, v_scale, M, N, D, group, br, maxe, etile, ssts)
                else:
                    self._idle_decode_softmax(tidx,unit_posted,unit_taken,sUnit,total_units,
                        nblocks,mQ,mCount,tStS,q_col,sQ,qready,mGate,sG,gstaged)
                    if cutlass.const_expr(getattr(self,"decode_reuse_v",False)):
                        if warp_idx==1:
                            self._gather_role(mK,sV,kvgcopy,kvgthr,mWS,mBT,mCount,
                                unit_posted,unit_taken,sUnit,total_units,nblocks,page,D,voff,st1b,
                                vfull,vfree,vst,lane,maxe,etile,True,1,2,tmapV)
            else:
                self._softmax_split16(
                    tidx, warp_idx // 4, unit_posted, unit_taken, sUnit,
                    total_units, nblocks,
                    mQ, mWS, mCount, mRM, mGate, mOut,
                    tStS, tOtO, sfull, pfull, pvdone, qready, odrained,
                    sMax, sSum, sG, gstaged, gconsumed,
                    q_col, sQ,
                    scale_log2, v_scale, M, N, D, group, br, maxe, etile, ssts)

        else:
            cute.arch.setmaxregister_decrease(88)
            if warp_idx == self.MMA_WARP:
                self._mma_role(unit_posted, unit_taken, sUnit, total_units,
                               nblocks, mCount, mTicket, tbase,
                               qk_mma, qk_sm_mma, pv_mma, tStS, tOtO, tQrQ, tQsQ, tSrK,
                               tOrP, tOrV, kfull, kfree, vfull, vfree,
                               sfull, pfull, pvdone, qready, odrained,
                               etile, kst, vst, ssts,
                               qk_idesc, pv_idesc, k_desc_hi, v_desc_hi,
                               k_desc_lo0, v_desc_lo0, k_stage_d, v_stage_d,
                               offs_aq, offs_bk, offs_ap, offs_bv, tmem_base,
                               q_col, o_col)
            elif warp_idx == self.KLOAD_WARP:
                self._gather_role(mK, sK, kvgcopy, kvgthr, mWS, mBT, mCount,
                                  unit_posted, unit_taken, sUnit, total_units,
                                  nblocks, page, D, Int64(0), st1b, kfull,
                                  kfree, kst, lane, maxe, etile, False,
                                  0, self.k_load_warps, tmapK)
                if cutlass.const_expr(getattr(self, "q4_gate_stage", False)):
                    self._stage_q4_gate(mGate, sG, mCount, Int32(cta), nblocks, sfull, gstaged, lane)
            elif warp_idx == self.VLOAD_WARP:
                self._gather_role(mK, sV, kvgcopy, kvgthr, mWS, mBT, mCount,
                                  unit_posted, unit_taken, sUnit, total_units,
                                  nblocks, page, D, voff, st1b, vfull, vfree,
                                  vst, lane, maxe, etile, True,
                                  0, self.gwr, tmapV)
            elif warp_idx == self.KLOAD_WARP2:
                # Entries12..23, with both K warps contributing32 arrivals.
                self._gather_role(mK, sK, kvgcopy, kvgthr, mWS, mBT,
                                  mCount, unit_posted, unit_taken, sUnit,
                                  total_units, nblocks, page, D, Int64(0),
                                  st1b, kfull, kfree, kst, lane, maxe,
                                  etile, False, 1, self.k_load_warps)
                if cutlass.const_expr(getattr(self, "q4_gate_stage", False)):
                    self._stage_q4_gate(mGate, sG, mCount, Int32(cta), nblocks, sfull, gstaged, lane + 32)
            elif warp_idx == self.VLOAD_WARP2:
                # gw=4 only: the second V-ring half
                if cutlass.const_expr(self.gwr == 2):
                    self._gather_role(mK, sV, kvgcopy, kvgthr, mWS, mBT,
                                      mCount, unit_posted, unit_taken, sUnit,
                                      total_units, nblocks, page, D, voff,
                                      st1b, vfull, vfree, vst, lane, maxe,
                                      etile, True, 1, 2)
            elif warp_idx == self.KLOAD_WARP3:
                if cutlass.const_expr(self.k_load_warps==4):
                    self._gather_role(mK,sK,kvgcopy,kvgthr,mWS,mBT,mCount,
                        unit_posted,unit_taken,sUnit,total_units,nblocks,page,D,
                        Int64(0),st1b,kfull,kfree,kst,lane,maxe,etile,False,
                        2,self.k_load_warps)
            elif warp_idx == self.KLOAD_WARP4:
                if cutlass.const_expr(self.k_load_warps==4):
                    self._gather_role(mK,sK,kvgcopy,kvgthr,mWS,mBT,mCount,
                        unit_posted,unit_taken,sUnit,total_units,nblocks,page,D,
                        Int64(0),st1b,kfull,kfree,kst,lane,maxe,etile,False,
                        3,self.k_load_warps)

        cute.arch.sync_threads()   # all roles done; TMEM can be released
        if cutlass.const_expr(getattr(self,"metadata_ready",False)):
            cute.arch.griddepcontrol_wait()
        tmem.relinquish_alloc_permit()
        tmem.free(tmem_ptr)

    @cute.jit
    def _softmax_split16(self,tidx,wg,unit_posted,unit_taken,sUnit,
                      total_units,nblocks,mQ,mWS,mCount,mRM,mGate,mOut,
                      tStS,tOtO,sfull,pfull,pvdone,qready,odrained,
                      sMax,sSum,sG,gstaged,gconsumed,q_col,sQ,
                      scale_log2,v_scale,M,N,D,group,br,maxe,etile,ssts):
        """64 scores/thread, two lanes per row, P scaled by 448 and E4M3.

        XOR16 combines the row max/sum. Both warpgroups rescale and drain
        disjoint rows, and cooperatively stage the full Q in shared memory.
        """
        sg=tidx%128;lane=tidx%32;half=lane//16
        L=self._logical_row(sg,wg)
        base=cute.make_ptr(self.acc,
            (tStS.iterator+(wg*16+(sg//32)*32)*65536).toint(),
            cute.AddressSpace.tmem,assumed_align=16)
        ld=cute.make_copy_atom(tcgen05.LdRed16x32bx2Op(tcgen05.Repetition(16),
            redOp=tcgen05.TmemLoadRedOp.MAX,half_split_off=64),self.acc)
        rS=cute.make_rmem_tensor((64,),self.acc)
        rMx=cute.make_rmem_tensor((4,),self.acc)
        rSelected=cute.make_rmem_tensor((16,),self.acc)
        rP=cute.make_rmem_tensor((64,),self.dtype)
        rPW=cute.make_tensor(cute.recast_ptr(rP.iterator,dtype=self.acc),cute.make_layout((12,)))
        rows=Int32(mQ.shape[0]);ph=Int32(0);gt=Int32(0)
        q_ptr=tStS.iterator+q_col
        self._mwait(unit_posted,0);u=Int32(sUnit.load())
        cute.arch.mbarrier_arrive(unit_taken)
        u0=cutlass.min(u,total_units-1)
        # Ordered block lookup depends on the sort output, unlike the
        # default reverse walk. Gate BEFORE Q refill's first lookup.
        if cutlass.const_expr(self.use_pdl and self.ordered):
            cute.arch.griddepcontrol_wait()
        b0=self._unit_blk(u0,nblocks,mCount);h0=self._unit_kvh(u0,nblocks)
        # Both softmax warpgroups stage the full FP8 Q tile asynchronously.
        self._refill_q_smem(sQ,mQ,b0*br,h0,group,
            group*cutlass.min(rows-b0*br,Int32(br)),tidx)
        cute.arch.mbarrier_arrive(qready)
        if cutlass.const_expr(getattr(self,"metadata_ready",False)):
            self._acquire_metadata(mCount,self._unit_blk(u0,nblocks,mCount))
        elif cutlass.const_expr(self.use_pdl and not self.ordered):cute.arch.griddepcontrol_wait()
        while u<total_units:
            blk=self._unit_blk(u,nblocks,mCount);kvh=self._unit_kvh(u,nblocks)
            row0=blk*br;lane_rows=group*cutlass.min(rows-row0,Int32(br))
            entry_begin,count=self._kv_slice(self._full_count(mCount,blk),u)
            tile_begin=entry_begin//etile
            num_tiles=(count+etile-1)//etile
            mref=Float32(-cutlass.Float32.inf);row_sum=Float32(0)
            nfast=cutlass.max(cutlass.min(num_tiles-1,
                self._fast_count(mCount,nblocks,blk)-tile_begin),Int32(0))
            rr=cutlass.min(L//group,Int32(br-1));moff=blk*self.RM+rr*self.RB
            if cutlass.const_expr(self.local_metadata):moff=rr*self.RB
            # Seed tile0; both words are inside each row's allocated RB.
            first_w=tile_begin*4+half*2
            pref_a=Int32(mRM[moff+first_w]);pref_b=Int32(mRM[moff+first_w+1])
            for span in cutlass.range_constexpr(2):
                begin=Int32(0) if span==0 else nfast
                trip=nfast if span==0 else num_tiles-nfast
                for jj in cutlass.range(trip):
                    tile=begin+jj;g=gt+tile;s=g%ssts
                    a=pref_a;b=pref_b
                    # Carry only2 words across the iteration. Clamp the final
                    # unused prefetch to this row; never touch another row.
                    next_w=(tile_begin+tile+1)*4+half*2
                    pref_a=Int32(mRM[moff+cutlass.min(next_w,Int32(self.RB-1))])
                    pref_b=Int32(mRM[moff+cutlass.min(next_w+1,Int32(self.RB-1))])
                    lo=a
                    hi=b
                    if cutlass.const_expr(span==1):
                        nv=cutlass.max(cutlass.min((count-tile*32-half*16)*4,Int32(64)),Int32(0))
                        lo=lo&cutlass.select_(nv>=32,Int32(-1),(Int32(1)<<cutlass.min(nv,Int32(31)))-1)
                        hi=hi&cutlass.select_(nv>=64,Int32(-1),(Int32(1)<<cutlass.min(cutlass.max(nv-32,Int32(0)),Int32(31)))-1)
                    self._mwait(sfull+s,(g//ssts)&1)
                    live_warp=Int32(1)
                    if cutlass.const_expr(self.active_softmax):
                        live_warp=Int32((L-lane%16)<lane_rows)
                    if live_warp!=0:
                        if cutlass.const_expr(span==0):
                            if cutlass.const_expr(self.softmax_wide):
                                for chunk in cutlass.range_constexpr(4):
                                    vals=cute.make_tensor(rS.iterator+chunk*16,cute.make_layout((16,)))
                                    qsa_sm_ptx.split_load(Int32((base+s*N+chunk*16).toint()),vals,16,64)
                                for ent in cutlass.range_constexpr(16):
                                    x=cute.arch.fmax(Float32(rS[ent*4]),Float32(rS[ent*4+1]))
                                    y=cute.arch.fmax(Float32(rS[ent*4+2]),Float32(rS[ent*4+3]))
                                    rSelected[ent]=cute.arch.fmax(x,y)
                            else:
                                for ent in cutlass.range_constexpr(16):
                                    vals=cute.make_tensor(rS.iterator+ent*4,cute.make_layout((4,)))
                                    rSelected[ent]=_load_quad_max(Int32((base+s*N+ent*4).toint()),vals)
                        else:
                            for chunk in cutlass.range_constexpr(4):
                                vals=cute.make_tensor(rS.iterator+chunk*16,cute.make_layout((16,)))
                                qsa_sm_ptx.split_load(Int32((base+s*N+chunk*16).toint()),vals,16,64)
                        cute.arch.fence_view_async_tmem_load()
                        bits=[lo,hi]
                        if cutlass.const_expr(span==0):
                            for ent in cutlass.range_constexpr(16):
                                quad=Float32(rSelected[ent])
                                valid=((bits[ent>>3]>>(4*(ent&7)+3))&1)!=0
                                rSelected[ent]=cutlass.select_(valid,quad,Float32(-cutlass.Float32.inf))
                            part=rSelected.load().reduce(cute.ReductionOp.MAX,Float32(-cutlass.Float32.inf),0)*scale_log2
                        else:
                            for j in cutlass.range_constexpr(64):
                                rS[j]=cutlass.select_(((bits[j>>5]>>(j&31))&1)!=0,Float32(rS[j]),Float32(-cutlass.Float32.inf))
                            part=rS.load().reduce(cute.ReductionOp.MAX,Float32(-cutlass.Float32.inf),0)*scale_log2
                        mx=cute.arch.fmax(part,Float32(cute.arch.shuffle_sync(part,lane^16,mask_and_clamp=31)))
                        first=mref==Float32(-cutlass.Float32.inf)
                        vote=cute.arch.vote_any_sync(mx>mref)
                        newref=cute.arch.fmax(mx,mref)
                        # PrimTS P scale=448. The denominator uses unrounded FP32.
                        neg=Float32(8.807354922057604)-newref
                        if cutlass.const_expr(span==0):
                            biases={}
                            for ent in cutlass.range_constexpr(16):
                                biasbits=bits[ent>>3]<<(28-4*(ent&7))
                                biases[ent]=cutlass.select_(biasbits<Int32(0),neg,Float32(-cutlass.Float32.inf))
                            for pair in cutlass.range_constexpr(16):
                                x,y=cute.arch.fma_packed_f32x2((Float32(rS[pair*2]),Float32(rS[pair*2+1])),
                                    (scale_log2,scale_log2),(biases[pair//2],biases[pair//2]))
                                rS[pair*2]=cute.math.exp2(x,fastmath=True)
                                rS[pair*2+1]=cute.math.exp2(y,fastmath=True)
                            src_first=cute.make_tensor(rS.iterator,cute.make_layout((32,)))
                            dst_first=cute.make_tensor(rP.iterator,cute.make_layout((32,)))
                            dst_first.store(src_first.load().to(self.dtype))
                            for chunk in cutlass.range_constexpr(2):
                                words=cute.make_tensor(rPW.iterator+chunk*4,cute.make_layout((4,)))
                                qsa_sm_ptx.split_store(Int32((base+s*N+chunk*4).toint()),words,4,16)
                            if vote and tile>0:
                                self._mwait(pvdone+((g-1)&3),((g-1)//4)&1)
                                delta=cutlass.select_(first,Float32(0),mref-newref)
                                alpha=cute.math.exp2(delta,fastmath=True)
                                self._rescale_split16(tOtO,wg,sg,alpha)
                                row_sum=row_sum*alpha
                            cute.arch.fence_view_async_tmem_store()
                            # K32 atoms 0 and 2 have all their P operands ready here.
                            tailneg=publish_bias(pfull+s,neg)
                            for ent in cutlass.range_constexpr(8,16):
                                biasbits=bits[ent>>3]<<(28-4*(ent&7))
                                biases[ent]=cutlass.select_(biasbits<Int32(0),tailneg,Float32(-cutlass.Float32.inf))
                            for pair in cutlass.range_constexpr(16,32):
                                x,y=cute.arch.fma_packed_f32x2((Float32(rS[pair*2]),Float32(rS[pair*2+1])),
                                    (scale_log2,scale_log2),(biases[pair//2],biases[pair//2]))
                                rS[pair*2]=cute.math.exp2(x,fastmath=True)
                                rS[pair*2+1]=cute.math.exp2(y,fastmath=True)
                            src_last=cute.make_tensor(rS.iterator+32,cute.make_layout((32,)))
                            dst_last=cute.make_tensor(rP.iterator+32,cute.make_layout((32,)))
                            dst_last.store(src_last.load().to(self.dtype))
                            for chunk in cutlass.range_constexpr(2):
                                words=cute.make_tensor(rPW.iterator+8+chunk*4,cute.make_layout((4,)))
                                qsa_sm_ptx.split_store(Int32((base+s*N+8+chunk*4).toint()),words,4,16)
                        else:
                            if cutlass.const_expr(self.softmax_packed):
                                for pair in cutlass.range_constexpr(32):
                                    x,y=cute.arch.fma_packed_f32x2((Float32(rS[pair*2]),Float32(rS[pair*2+1])),(scale_log2,scale_log2),(neg,neg))
                                    px=cute.math.exp2(x,fastmath=True)
                                    py=cute.math.exp2(y,fastmath=True)
                                    j=pair*2
                                    rS[j]=cutlass.select_(((bits[j>>5]>>(j&31))&1)!=0,px,Float32(0))
                                    rS[j+1]=cutlass.select_(((bits[(j+1)>>5]>>((j+1)&31))&1)!=0,py,Float32(0))
                            else:
                                for j in cutlass.range_constexpr(64):
                                    prob=cute.math.exp2(rS[j]*scale_log2+neg,fastmath=True)
                                    rS[j]=cutlass.select_(((bits[j>>5]>>(j&31))&1)!=0,prob,Float32(0))
                            rP.store(rS.load().to(self.dtype))
                            for chunk in cutlass.range_constexpr(4):
                                words=cute.make_tensor(rPW.iterator+chunk*4,cute.make_layout((4,)))
                                qsa_sm_ptx.split_store(Int32((base+s*N+chunk*4).toint()),words,4,16)
                            if vote and tile>0:
                                self._mwait(pvdone+((g-1)&3),((g-1)//4)&1)
                                delta=cutlass.select_(first,Float32(0),mref-newref)
                                alpha=cute.math.exp2(delta,fastmath=True)
                                self._rescale_split16(tOtO,wg,sg,alpha)
                                row_sum=row_sum*alpha
                            cute.arch.fence_view_async_tmem_store()
                            cute.arch.mbarrier_arrive(pfull+s)
                        # The balanced reduction sees all 64 unrounded FP32 P values.
                        tsum=self._sum_packed(rS,64)
                        row_sum+=tsum;mref=newref
                        cute.arch.fence_view_async_tmem_store();cute.arch.mbarrier_arrive(pfull+ssts+s)
                    else:
                        # Zero-padded Q makes this warp's S (and aliased P)
                        # zero. Preserve every barrier arrival and phase.
                        cute.arch.mbarrier_arrive(pfull+s)
                        cute.arch.mbarrier_arrive(pfull+ssts+s)
            gt_next=gt+num_tiles;ph+=1
            self._mwait(unit_posted,ph&1);u2=Int32(sUnit.load())
            cute.arch.mbarrier_arrive(unit_taken)
            # Keep the terminal ticket arrival; this static CTA has no next unit.
            if cutlass.const_expr(self.single_unit_mma and self.single_pass):u2=total_units
            row_sum+=Float32(cute.arch.shuffle_sync(row_sum,lane^16,mask_and_clamp=31))
            inv=cutlass.select_(row_sum!=Float32(0),cute.arch.rcp_approx(row_sum),Float32(0))
            if u2<total_units:
                bn=self._unit_blk(u2,nblocks,mCount);hn=self._unit_kvh(u2,nblocks)
                self._refill_q_smem(sQ,mQ,bn*br,hn,group,
                    group*cutlass.min(rows-bn*br,Int32(br)),tidx)
                cute.arch.mbarrier_arrive(qready)
            # Final QK retired before softmax read its S tile. The next Q
            # can overwrite Q SMEM while final PV reads P in[0,256), V,
            # and O in[256,512).
            # Keep O drain behind final PV retirement. Next QK is enqueued
            # by the same MMA warp after all previous PV operations.
            if gt_next>0:self._mwait(pvdone+((gt_next-1)&3),((gt_next-1)//4)&1)
            if cutlass.const_expr(self.kv_splits > 1):
                drain_live=Int32(1)
                if cutlass.const_expr(self.active_softmax):
                    drain_live=Int32((L-lane%16)<lane_rows)
                if drain_live!=0:
                    self._drain_mtp_partial(tOtO,wg,sg,L,row0,kvh,group,lane_rows,
                                            mref,row_sum,u,mGate,mOut)
            else:
                self._drain_split16(tOtO,wg,sg,L,row0,kvh,group,lane_rows,inv*v_scale,mGate,mOut,sG,gstaged)
            gt=gt_next;u=u2
