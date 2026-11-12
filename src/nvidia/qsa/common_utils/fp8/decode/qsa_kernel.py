"""V(g) storage becomes free at the existing PV-retirement event(g).

Producer g waits event(g-Vst) at slot((g-Vst)&3), phase((g-Vst)//4)&1.
The producer cannot lap its event: later PV requires later V gathered by
this same sequential producer. Shared storage layout remains unchanged.
"""
from __future__ import annotations
import cutlass
import cutlass.cute as cute
from cutlass.cute.nvgpu import tcgen05
from cutlass.cute.typing import Float32,Int32,Int64
from .layouts import QSAForward as Base
from .layouts import _elem_pointer, _shfl32, _shfl
from ... import qsa_sm_ptx


from cutlass._mlir.dialects import llvm


def _copy_address_plan(layout):
    # Ordinary Python executes at trace time; all offsets are layout constants.
    n=cute.size(layout,mode=[1]);k=cute.size(layout,mode=[2])
    offsets=tuple(int(cute.crd2idx((0,i,j),layout)) for i in range(n) for j in range(k))
    residues=tuple(sorted(set(x%1024 for x in offsets)))
    assert len(residues)<=4,(layout,residues)
    ids=tuple(residues.index(x%1024) for x in offsets)
    high=tuple(x-x%1024 for x in offsets)
    return residues,ids,high

@cute.jit
def _copy16(dst:Int32,src:Int64):
    # Prefetch hint only: each lane still copies exactly16 bytes.
    llvm.inline_asm(None,[dst.ir_value(),src.ir_value()],
        "cp.async.cg.shared.global.L2::128B [$0], [$1], 16;",
        "r,l,~{memory}",has_side_effects=True,is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT)

@cute.jit
def _copy16_policy(dst:Int32,src:Int64,amount:cutlass.Constexpr):
    if cutlass.const_expr(amount==128):
        _copy16(dst,src)
    else:
        hint='' if amount==0 else f'.L2::{amount}B'
        llvm.inline_asm(None,[dst.ir_value(),src.ir_value()],
            f"cp.async.cg.shared.global{hint} [$0], [$1], 16;",
            "r,l,~{memory}",has_side_effects=True,is_align_stack=False,
            asm_dialect=llvm.AsmDialect.AD_ATT)

@cute.jit
def publish_bias(bar,neg:Float32):
    b=Int32(Int64(bar.toint())&0x3ffff)
    return Float32(llvm.inline_asm(Float32.mlir_type,[b.ir_value(),neg.ir_value()],
        "{ .reg .b64 state; .reg .b32 zero; "
        "mbarrier.arrive.shared::cta.b64 state, [$1]; "
        "cvt.u32.u64 zero,state; and.b32 zero,zero,0x80000000; "
        "add.rn.f32 $0,$2,zero; }",
        "=f,r,f",has_side_effects=True,is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT))


class QSAForward(Base):
    def __init__(self,*args,static_short=False,q_load_align=16,kv_splits=1,zero_pad_q=False,active_softmax=False,fused_metadata=False,local_metadata=False,single_unit_mma=False,metadata_request_span=0,union_hash_mix=False,union_full_first=False,union_v_warps=1,union_physical=False,decode_fixed=False,decode_gate=False,decode_l2=128,softmax_packed=False,softmax_wide=False,gate_factor=False,single_pass=False,**kwargs):
        super().__init__(*args,**kwargs)
        self.single_pass=bool(single_pass)
        self.gate_factor=bool(gate_factor)
        self.k_load_warps=2
        self.KLOAD_WARP3=self.KLOAD_WARP4=-1
        self.softmax_packed=bool(softmax_packed)
        self.softmax_wide=bool(softmax_wide)
        self.decode_fixed=bool(decode_fixed)
        self.decode_gate=bool(decode_gate)
        self.decode_single_softmax=self.decode_gate
        self.decode_l2=int(decode_l2)
        assert self.decode_l2 in (0,64,128,256)
        self.kv_splits=int(kv_splits)
        self.fused_metadata=bool(fused_metadata)
        self.metadata_request_span=int(metadata_request_span)
        self.union_hash_mix=bool(union_hash_mix)
        self.union_full_first=bool(union_full_first)
        self.union_v_warps=int(union_v_warps)
        self.union_physical=bool(union_physical)
        self.single_unit_mma=bool(single_unit_mma)
        assert not self.single_unit_mma or self.fused_metadata or static_short
        self.local_metadata=bool(local_metadata) and self.fused_metadata
        self.local_entries=((self.maxe+31)//32+self.kv_splits-1)//self.kv_splits*32
        if self.fused_metadata:
            assert self.br in (1,2,4) and self.single_bt and not self.ordered
            assert self.br==1 or (self.kv_splits==1 and self.local_metadata)
        self.zero_pad_q=bool(zero_pad_q)
        self.active_softmax=bool(active_softmax)
        assert not self.active_softmax or self.zero_pad_q
        assert self.kv_splits >= 1
        assert 2<=self.k_stages<=4 and self.s_stages==2
        assert self.v_stages<=4
        self.static_short=bool(static_short)
        assert not self.single_pass or (self.m_tile==64 and self.single_unit_mma and self.static_short)
        assert not self.gate_factor or (self.decode_gate and self.m_tile==64 and self.single_unit_mma and self.kv_splits==1)
        assert q_load_align in (16,32)
        self.q_load_align=int(q_load_align)
        assert self.threads==384, "208x256+88x128 fits the384-thread allocated pool"
        assert self.n_tile == 128 and self.s_stages == 2


    @cute.jit
    def _acquire_metadata(self, counts, block: Int32):
        address = Int64((counts.iterator + block).toint())
        count = Int32(-1)
        while count < 0:
            count = Int32(llvm.inline_asm(Int32.mlir_type, [address.ir_value()],
                "ld.acquire.gpu.global.u32 $0, [$1];", "=r,l,~{memory}",
                has_side_effects=True, is_align_stack=False,
                asm_dialect=llvm.AsmDialect.AD_ATT))

    @cute.jit
    def _mma_role(self, unit_posted, unit_taken, sUnit, total_units, nblocks,
                  mCount, mTicket, tbase, qk_mma, qk_sm_mma, pv_mma, tStS, tOtO, tQrQ, tQsQ,
                  tSrK,
                  tOrP, tOrV, kfull, kfree, vfull, vfree,
                  sfull, pfull, pvdone, qready, odrained,
                  etile, kst, vst, ssts,
                  qk_idesc, pv_idesc, k_desc_hi, v_desc_hi,
                  k_desc_lo0, v_desc_lo0, k_stage_d, v_stage_d,
                  offs_aq, offs_bk, offs_ap, offs_bv, tmem_base,
                  q_col, o_col):
        """One warp issues every tcgen05 MMA, QK ahead of PV.

        Iteration ``i`` of a unit issues QK(i) then PV(i-ahead), so the
        in-order tensor-core queue runs ``QK(0), QK(1), PV(0), QK(2), PV(1),
        ...``: the softmax warpgroup gets a QK window per tile and the TC
        queue stays fed while it works.  Tile indices are CTA-cumulative
        across units (``gt``), so ring/stage phases never reset.  With
        ssts >= 3 the aliased S/P stage overwritten by QK(i) was last read by
        PV(i-ssts), which in-order execution retired before QK(i-1) <=
        PV(i-2) < QK(i).  With ssts == 2 (V5: TMEM Q claims 128 columns)
        ahead falls back to 1: QK(i+1) is issued after PV(i-1), and the
        in-order queue still retires PV(i-1) before QK(i+1) rewrites the
        stage.
        """
        ph = Int32(0)
        if cutlass.const_expr(self.mmasched):
            # ---- e34 QSA_MMASCHED: this warp is also the ticket scheduler.
            # Prologue posting of ticket 0; every other role waits
            # unit_posted(0) exactly as it did from the dedicated warp. ----
            tidx_m, _, _ = cute.arch.thread_idx()
            u_t_m = Int32(0)
            if tidx_m == 32 * self.MMA_WARP:
                if cutlass.const_expr(self.static_short):
                    initial_cta, _, _ = cute.arch.block_idx()
                    u_t_m=Int32(initial_cta)
                    if cutlass.const_expr(getattr(self,"unit_order",False)):
                        u_t_m=Int32(mTicket[initial_cta])
                else:
                    u_t_m = cute.arch.atomic_add(
                        mTicket.iterator, Int32(1), sem="relaxed",
                        scope="gpu") - tbase
                (sUnit).store(u_t_m)
                cute.arch.mbarrier_arrive(unit_posted)
            u = Int32(_shfl32(u_t_m, 0))
        else:
            self._mwait(unit_posted, 0)
            u = Int32(sUnit.load())
            cute.arch.mbarrier_arrive(unit_taken)
        # PDL gate before the first mCount read (loop head); qready/TMEM Q is
        # preprocess-independent and stays above the gate.
        if cutlass.const_expr(getattr(self,"metadata_ready",False)):
            self._acquire_metadata(mCount, self._unit_blk(u, nblocks, mCount))
        elif cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
        self._mwait(qready, 0)      # resident TMEM Q for unit 0
        nkq = cute.size(tQrQ, mode=[2])
        nkp = cute.size(tOrV, mode=[2])
        gt = Int32(0)
        if cutlass.const_expr(self.mmasched):
            u_next = Int32(0)
        while u < total_units:
            if ph > 0:
                self._mwait(qready, ph & 1)
            if cutlass.const_expr(self.mmasched):
                # ---- e34 SCHEDULER DUTY, and why it sits HERE and not at the
                # loop end.  Posting ticket ph+1 at the END of unit ph
                # DEADLOCKS (measured: every QSA_MMASCHED arm hung, e34 J8/J9):
                # the gather warps run up to k_stages tiles AHEAD of this warp
                # and signal kfull(g) only at their iteration g+lag, so the
                # arrivals for unit ph's LAST lag tiles are issued from unit
                # ph+1's iterations - a gather warp parked on unit_posted(ph+1)
                # never sends them, and this warp never finishes unit ph's
                # tiles, so neither side can proceed.  The dedicated scheduler
                # warp avoids that by publishing a ticket ~a full unit ahead of
                # every consumer, which is exactly what posting HERE (right
                # after qready, i.e. before this unit's tiles are issued)
                # reproduces.  qready(ph) also proves unit_taken phase ph is
                # complete - the softmax reads ticket ph before it refills Q -
                # so the sUnit overwrite is safe and the unit_taken wait free.
                # The atomicAdd's ~700-cycle latency is exposed once per unit
                # (~2us of main over ~5.5 units/CTA); that is the price of the
                # 352-thread launch's +16 registers.
                self._mwait(unit_taken, ph & 1)
                tidx_n, _, _ = cute.arch.thread_idx()
                u_t_n = Int32(0)
                if tidx_n == 32 * self.MMA_WARP:
                    if cutlass.const_expr(self.static_short):
                        cta, _, _ = cute.arch.block_idx()
                        width, _, _ = cute.arch.grid_dim()
                        wave=ph+Int32(1)
                        start=wave*width
                        count=cutlass.max(cutlass.min(total_units-start,Int32(width)),Int32(0))
                        offset=cutlass.select_((wave&1)!=0,count-Int32(1)-cta,Int32(cta))
                        u_t_n=cutlass.select_(cta<count,start+offset,total_units)
                    else:
                        u_t_n = cute.arch.atomic_add(
                            mTicket.iterator, Int32(1), sem="relaxed",
                            scope="gpu") - tbase
                    (sUnit).store(u_t_n)
                    cute.arch.mbarrier_arrive(unit_posted)
                u_next = Int32(_shfl32(u_t_n, 0))
            blk = self._unit_blk(u, nblocks, mCount)
            _, count = self._kv_slice(self._full_count(mCount,blk), u)
            num_tiles = (count + etile - 1) // etile
            # QK runs two tiles ahead of the PV slot when the S/P ring has
            # >= 3 stages; with ssts == 2 the stage QK(i+1) would overwrite is
            # still owned by PV(i-1), so the order falls back to 1-ahead.
            ahead = 2 if ssts >= 3 else 1
            for i in cutlass.range(num_tiles + ahead):
                if i < num_tiles:
                    g = gt + i
                    s = g % ssts
                    ks = g % kst
                    self._mwait(kfull + ks, (g // kst) & 1)
                    tStS_s = tStS[(None, None, None, s)]
                    qk_sm_mma.set(tcgen05.Field.ACCUMULATE,False)
                    for kp in cutlass.range_constexpr(2*nkq):
                        cute.gemm(qk_sm_mma,tStS_s,tQsQ[(None,None,kp,0)],
                                  tSrK[(None,None,kp,ks)],tStS_s)
                        qk_sm_mma.set(tcgen05.Field.ACCUMULATE,True)
                    with cute.arch.elect_one():
                        tcgen05.commit(sfull + s, None, tcgen05.CtaGroup.ONE)
                        if cutlass.const_expr(kst!=ssts):tcgen05.commit(kfree + ks, None, tcgen05.CtaGroup.ONE)
                if i >= ahead:
                    j = i - ahead
                    g = gt + j
                    ps = g % ssts
                    vs = g % vst
                    self._mwait(pfull + ps, (g // ssts) & 1)
                    self._mwait(vfull + vs, (g // vst) & 1)
                    pv_mma.set(tcgen05.Field.ACCUMULATE, j != 0)
                    for step in cutlass.range_constexpr(4):
                        if cutlass.const_expr(step==2):
                            self._mwait(pfull+ssts+ps,(g//ssts)&1)
                        kp=step*2 if step<2 else (step-2)*2+1
                        cute.gemm(pv_mma, tOtO, tOrP[(None, None, kp, ps)],
                                  tOrV[(None, None, kp, vs)], tOtO)
                        pv_mma.set(tcgen05.Field.ACCUMULATE, True)
                    # Deferred releases: both fire only once the PV MMA has
                    # really read the V stage and the P columns, so producers
                    # may reuse them; ofree tells the softmax warpgroup that O
                    # is quiescent (epilogue drain / rare rescale).
                    with cute.arch.elect_one():
                        # PV-retirement ring: barrier g&3 sees every 4th PV,
                        # so a waiter can never observe it aliased two phases
                        # (the next commit to the same barrier is PV(g+3),
                        # gated on the waiter's own progress).
                        tcgen05.commit(pvdone + (g & 3), None,
                                       tcgen05.CtaGroup.ONE)
            gt += num_tiles
            if cutlass.const_expr(self.mmasched):
                # Ticket ph+1 was already published at the TOP of this
                # iteration (see the gather-ring deadlock note there), so the
                # unit boundary is just a register hand-off: no unit_posted
                # wait, no unit_taken arrival (this warp is the scheduler and
                # is excluded from the 32*take_warps count).
                u = u_next
                ph += 1
            else:
                ph += 1
                self._mwait(unit_posted, ph & 1)
                u = Int32(sUnit.load())
                cute.arch.mbarrier_arrive(unit_taken)
            # The host admits fused metadata only when every CTA has one
            # unit. Keep next-ticket publication for gather drain, but make
            # the impossible MMA backedge explicit to shorten live ranges.
            if cutlass.const_expr(self.single_unit_mma):u=total_units




    @cute.jit
    def _logical_row(self,sg,wg):
        return (sg//32)*32+sg%16+wg*16

    @cute.jit
    def _rescale_split16(self,tOtO,wg,sg,alpha):
        ptr=cute.make_ptr(self.acc,(tOtO.iterator+(wg*16+(sg//32)*32)*65536).toint(),
            cute.AddressSpace.tmem,assumed_align=16)
        scratch=cute.make_rmem_tensor((64,),self.acc)
        first=cute.make_tensor(scratch.iterator,cute.make_layout((32,)))
        qsa_sm_ptx.split_load(Int32(ptr.toint()),first,32,128)
        cute.arch.fence_view_async_tmem_load()
        for chunk in cutlass.range_constexpr(4):
            r=cute.make_tensor(scratch.iterator+(chunk&1)*32,cute.make_layout((32,)))
            if cutlass.const_expr(chunk<3):
                nxt=cute.make_tensor(scratch.iterator+((chunk+1)&1)*32,cute.make_layout((32,)))
                qsa_sm_ptx.split_load(Int32((ptr+(chunk+1)*32).toint()),nxt,32,128)
            for pair in cutlass.range_constexpr(16):
                x,y=cute.arch.fma_packed_f32x2((Float32(r[pair*2]),Float32(r[pair*2+1])),
                    (alpha,alpha),(Float32(0),Float32(0)))
                r[pair*2]=x;r[pair*2+1]=y
            qsa_sm_ptx.split_store(Int32((ptr+chunk*32).toint()),r,32,128)
            if cutlass.const_expr(chunk<3):cute.arch.fence_view_async_tmem_load()
        # Caller waits once for all pending P and O stores before publication.


    @cute.jit
    def _gather_role(self, mK, sX, kvgcopy, kvgthr, mWS, mBT, mCount,
                     unit_posted, unit_taken, sUnit, total_units, nblocks,
                     page, D, voff, st1b, full_bar, free_bar, stages, lane,
                     maxe, etile, transpose, half: cutlass.Constexpr = 0,
                     nhalf: cutlass.Constexpr = 1, tmap=None):
        """One warp fills a K (or V) ring stage per tile with cp.async
        gathers, flowing seamlessly across unit boundaries.

        Lane ``l`` resolves union entry ``l`` of the tile (etile <= 32) into a
        token byte address; the token rows each lane copies are then reached
        by warp shuffles of those addresses, so the block-table lookup happens
        once per entry instead of once per copy.  Entries past ``count`` clamp
        to entry 0 so every SMEM slot is refreshed each tile: masked columns
        then contribute P=0 x finite V = 0, never 0 x stale-NaN = NaN.

        Lagged arrivals: full_bar for tile g fires
        at iteration g+lag (lag = stages-1).  V5.5 performs that signal at
        the TOP of iteration g+lag, using cp_async_wait_group(lag-1): the
        groups outstanding before that iteration's commit are exactly
        g-lag..g+lag-1 relative to the signal, so wait_group(lag-1) proves
        group g landed - identical confirmation data to V5's post-commit
        wait_group(lag), but the ~900-cycle resolve+LDGSTS issue block of
        the signaling iteration no longer sits INSIDE the consumer's
        kfree->regather->kfull feedback cycle (e8 trace: that loop paced
        the kernel at ~2290 cyc/tile with D=1532 of MMA kfull waiting
        while every throughput unit stayed below 42%).  Up to ``lag``
        groups remain in flight, so the issue cadence still decouples from
        the gather round-trip latency (period >= (L+d)/lag).  Stage safety:
        tile g+1 issues into stage (g+1)%stages only after free_bar proves
        QK/PV(g+1-stages) retired.  Arrivals for the last ``lag`` tiles
        drain after the terminal ticket in tile order (wait_group counts
        strictly decrease), matching the consumer's cumulative-index phase
        parity.
        """
        N = self.n_tile
        ce = self.copy_elems
        gptr = self.gptr
        lag = stages - 1
        cND = cute.make_identity_tensor((N, D))
        tXcX = kvgthr.partition_S(cND)
        npass = cute.size(tXcX, mode=[1])
        nchunk = cute.size(tXcX, mode=[2])
        # gw=4: this warp owns the ``half``-th group of ``eh`` union entries of
        # every tile, i.e. passes [hoff, hoff+eh) of the copy partition, and its
        # lanes 0..eh-1 resolve those entries (the shuffle source for pass
        # hoff+jj is lane jj).  nhalf=1 reproduces the champion exactly.
        eh = npass // nhalf
        hoff = half * eh
        # At most four live swizzled bases, instead of one percopy.
        tXsX0=kvgthr.partition_D(self._flatten(sX[(None,None,None,0)],transpose))
        residues,copy_resid,copy_high=_copy_address_plan(tXsX0.layout)
        stage_bytes=int(sX.layout.stride[-1])
        assert stage_bytes%1024==0
        sw=tXsX0.iterator.type.swizzle_type
        assert sw.num_base==4 and sw.num_shift==3 and sw.num_bits<=3
        copy_bases=cute.make_rmem_tensor((len(residues),),Int32)
        raw_base=Int32(Int64(tXsX0.iterator.toint())&0x3ffff)
        for ri in cutlass.range_constexpr(len(residues)):
            raw=raw_base+Int32(residues[ri])
            copy_bases[ri]=raw^((raw>>3)&Int32(((1<<sw.num_bits)-1)<<4))
        bt_w = self.bt_width
        if cutlass.const_expr(self.rowidx or self.union_physical or getattr(self, "relative_address32", False)):
            # Offset unit is 64 elements = 64 FP8 bytes, so cache strides must
            # be multiples of 64 elements (host-guarded); >>6 is then exact.
            q0k = Int32(mK.stride[0]) >> 6
            q1k = Int32(mK.stride[1]) >> 6
            # Every entry P2a let through has a live physical page and
            # loff <= page-4 (page is a multiple of the 4-token group), so a
            # legal off128 is <= omax; stale words past ``count`` are clamped
            # into range and can never form an out-of-bounds LDGSTS address.
            omax = (Int32(mK.shape[0]) - 1) * q0k + (page - 4) * q1k
            lane_q1 = Int32(lane // gptr) * q1k
        ph = Int32(0)
        if cutlass.const_expr(getattr(self,"decode_reuse_v",False) and transpose and half==1):
            # This warp already consumed the initial ticket while loading Q.
            # The scheduler may have published the terminal ticket by now.
            assert self.static_short and self.single_unit_mma and self.br==1
            initial_cta,_,_=cute.arch.block_idx()
            u=Int32(initial_cta)
        else:
            self._mwait(unit_posted,0)
            u=Int32(sUnit.load())
            cute.arch.mbarrier_arrive(unit_taken)
        # PDL gate: the mWS prime loads below are this warp's first reads of
        # preprocess output.
        if cutlass.const_expr(getattr(self,"metadata_ready",False)):
            self._acquire_metadata(mCount, self._unit_blk(u, nblocks, mCount))
        elif cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_wait()
        gt = Int32(0)
        # CHAIN29 LIVENESS (ported from fit_v47_drain): exclusive lower bound
        # of the tile indices whose full_bar arrival is STILL OWED.  Zero
        # unless a short unit (see the bounded per-unit drain below) settled
        # its own arrivals early, so on every full-budget shape the guards it
        # feeds stay exactly as champion-written and the instruction stream is
        # unchanged.
        arr_base = Int32(0)
        entry_stride = 1 if self.compact_ws else 4
        group_word = 0 if self.compact_ws else 1
        while u < total_units:
            blk = self._unit_blk(u, nblocks, mCount)
            kvh = self._unit_kvh(u, nblocks)
            entry_begin, count = self._kv_slice(self._full_count(mCount,blk), u)
            num_tiles = (count + etile - 1) // etile
            ws_base = (blk * maxe + entry_begin) * entry_stride
            if cutlass.const_expr(self.local_metadata):ws_base=Int32(0)
            last_eb = (num_tiles - 1) * etile

            # ---- 3-stage entry-resolution pipeline: at iteration i the mWS
            # words for tile i+2 and the mBT lookup for tile i+1 are ISSUED,
            # while tile i fills from the address resolved at i-1.  Both
            # dependent global-load latencies therefore hide under a full
            # tile of cp.async issue work instead of pacing the ring. ----
            # prime stage W (tile 0 + tile 1 words) and stage M (tile 0 bt).
            eid0 = hoff + lane
            src0 = cutlass.select_(eid0 < count, eid0, Int32(0))
            eid1 = cutlass.min(etile, last_eb) + hoff + lane
            src1 = cutlass.select_((eid1 >= 0) & (eid1 < count), eid1, Int32(0))
            if cutlass.const_expr(getattr(self, "relative_address32", False)):
                assert D == 256 and gptr == 8 and ce == 16
                assert not self.rowidx and not self.union_physical
                kbase = (_elem_pointer(mK, (0, 0, kvh, 0)).toint() + voff
                         + Int64(lane // gptr) * st1b + Int64(lane % gptr) * Int64(ce))
            if cutlass.const_expr(self.rowidx or self.union_physical):
                kbase = _elem_pointer(mK, (0, 0, kvh, 0)).toint() + voff
                if cutlass.const_expr(getattr(self, "sm100_q1_address_hoist", False)):
                    assert D == 256 and ce == 16 and gptr == 8 and self.union_physical
                    kbase += Int64(lane_q1) * Int64(64) + Int64(lane % gptr) * Int64(16)
                o0 = cutlass.min(cutlass.max(
                    Int32(mWS[ws_base + src0 * entry_stride]), Int32(0)), omax)
                o1 = cutlass.min(cutlass.max(
                    Int32(mWS[ws_base + src1 * entry_stride]), Int32(0)), omax)
            else:
                wr0 = Int32(0) if self.single_bt else Int32(mWS[ws_base + src0 * entry_stride])
                wg0 = Int32(mWS[ws_base + src0 * entry_stride + group_word])
                wr1 = Int32(0) if self.single_bt else Int32(mWS[ws_base + src1 * entry_stride])
                wg1 = Int32(mWS[ws_base + src1 * entry_stride + group_word])
                # clamp the prime lookups so a degenerate unit (count == 0, only
                # possible for an empty union) cannot index mBT out of bounds
                req0 = Int32(0) if self.single_bt else cutlass.min(cutlass.max(wr0, Int32(0)), cute.size(mBT.shape[0]) - 1)
                lidx0 = wg0 << 2
                lpage0 = cutlass.min(cutlass.max(lidx0 // page, Int32(0)),
                                     bt_w - 1)
                loff0 = lidx0 - (lidx0 // page) * page
                mp0 = Int32(mBT[req0, lpage0])

            loop_tiles = num_tiles
            # e3 (v4) QSA_AOHYB: per-unit signalling form.  A unit with
            # num_tiles >= lag never needs the CHAIN29 drain (its own iterations
            # reach g+lag), so it may take the unthrottled fire-and-forget form;
            # a partial-prefix unit keeps the champion's commit_group + waited
            # lagged arrival + unit-end drain, i.e. the deadlock fix untouched.
            # Traced only for the hybrid so every other arm's instruction stream
            # is byte-identical to the shipped champion producer.
            if cutlass.const_expr(self.aohyb):
                noinc_u = num_tiles >= lag
            for i in cutlass.range(loop_tiles):
                g = gt + i
                st = g % stages
                # ---- PRE-ISSUE ARRIVAL (V5.5): signal tile g-lag at the TOP
                # of the iteration, before the resolve/issue block.  The
                # confirmation is identical (wait_group(lag-1) proves group
                # g-lag landed: outstanding before this iteration's commit
                # are groups g-lag..g-1), but the ~900-cycle LDGSTS issue
                # block no longer sits inside the consumer's critical
                # free->regather->full feedback cycle (e8 trace: that loop,
                # not any throughput unit, paced the kernel at ~2290cyc/tile
                # with D=1532 of MMA-warp kfull waiting).
                # e14 arrive_on: this blocking confirmation is REPLACED by a
                # fire-and-forget cp.async.mbarrier.arrive.noinc tied to this
                # tile's copies below - wait_group(lag-1) here would still
                # pace the producer loop at (L+d)/lag on the lag-2 ring.
                # e19 tma: completion is direct (complete_tx on landing), so
                # there is no producer-side confirmation at all.
                if cutlass.const_expr(not self.arrive_on and (not self.aohyb)):
                    if g - lag >= arr_base:
                        cute.arch.cp_async_wait_group(lag - 1)
                        cute.arch.fence_view_async_shared()
                        cute.arch.mbarrier_arrive(full_bar + (g - lag) % stages)
                e2 = cutlass.min((i + 2) * etile, last_eb)
                eid2 = e2 + hoff + lane
                src2 = cutlass.select_(eid2 < count, eid2, Int32(0))
                if cutlass.const_expr(self.rowidx or self.union_physical):
                    # ONE independent global load per tile (no mBT round trip).
                    o2 = cutlass.min(cutlass.max(
                        Int32(mWS[ws_base + src2 * entry_stride]), Int32(0)), omax)
                    myoff = o0
                else:
                    wr2 = Int32(0) if self.single_bt else Int32(mWS[ws_base + src2 * entry_stride])
                    wg2 = Int32(mWS[ws_base + src2 * entry_stride + group_word])
                    # stage M for tile i+1 (wr1/wg1 landed a full iteration ago)
                    lidx1 = wg1 << 2
                    lpage1 = lidx1 // page
                    loff1 = lidx1 - lpage1 * page
                    mp1 = Int32(mBT[wr1, lpage1])
                    # Resolve the physical byte address for the current tile.
                    if cutlass.const_expr(getattr(self, "relative_address32", False)):
                        myoff = mp0 * q0k + loff0 * q1k
                    else:
                        myaddr = _elem_pointer(
                            mK, (mp0, loff0, kvh, 0)).toint() + voff
                if g >= stages:
                    if cutlass.const_expr(transpose):
                        self._mwait(free_bar + ((g-stages)&3), ((g-stages)//4)&1)
                    else:
                        self._mwait(free_bar + st, ((g // stages) - 1) & 1)
                tXsX=tXsX0
                for jj in cutlass.range_constexpr(eh):
                    mm = hoff + jj
                    if cutlass.const_expr(getattr(self, "sm100_q1_address_hoist", False)):
                        addr = kbase + Int64(_shfl32(myoff, jj)) * Int64(64)
                    elif cutlass.const_expr(self.rowidx or self.union_physical):
                        addr = kbase + Int64(
                            _shfl32(myoff, jj) + lane_q1) * Int64(64)
                    elif cutlass.const_expr(getattr(self, "relative_address32", False)):
                        addr = kbase + Int64(_shfl32(myoff, jj)) * Int64(64)
                    else:
                        addr = (_shfl(myaddr, jj, 32)
                                + Int64(lane // gptr) * st1b)
                    gptr_t = cute.make_ptr(self.dtype, addr, cute.AddressSpace.gmem,
                                           assumed_align=16)
                    mg = cute.make_tensor(gptr_t, cute.make_layout((D,)))
                    mgc = cute.tiled_divide(mg, (ce,))
                    for k in cutlass.range_constexpr(nchunk):
                        ki = tXcX[0, 0, k][1] // ce
                        srck = cute.make_tensor(mgc[None, ki].iterator,
                                                tXsX[(None, mm, k)].layout)
                        dstk=tXsX[(None,mm,k)]
                        assert cute.size(srck)==16 and cute.size(dstk)==16
                        ci=mm*nchunk+k
                        dst_addr=copy_bases[copy_resid[ci]]+Int32(copy_high[ci])+Int32(st)*Int32(stage_bytes)
                        if cutlass.const_expr(getattr(self, "relative_address32", False) or getattr(self, "sm100_q1_address_hoist", False)):
                            _copy16_policy(dst_addr, addr + Int64(k * ce * gptr), self.decode_l2)
                        else:
                            _copy16_policy(dst_addr,Int64(srck.iterator.toint()),self.decode_l2)
                if cutlass.const_expr(self.arrive_on):
                    # Fire-and-forget arrival: fires when THIS thread's prior
                    # cp.async copies (exactly this tile's - the next tile's
                    # are not issued yet) have landed, with the completion's
                    # release semantics.  All 32 lanes arrive -> the barrier's
                    # 32-count phase flips exactly as with the waited form.
                    cute.arch.cp_async_mbarrier_arrive_noinc(full_bar + st)
                elif cutlass.const_expr(self.aohyb):
                    # e3 (v4) hybrid: the unthrottled form for a full-budget
                    # unit, the champion's commit_group for a partial-prefix
                    # unit so the CHAIN29 drain below still owns its
                    # arrivals.  One runtime predicate per tile, evaluated
                    # once per unit into noinc_u.
                    if noinc_u:
                        cute.arch.cp_async_mbarrier_arrive_noinc(
                            full_bar + st)
                    else:
                        cute.arch.cp_async_commit_group()
                else:
                    cute.arch.cp_async_commit_group()
                if cutlass.const_expr(self.rowidx or self.union_physical):
                    o0 = o1
                    o1 = o2
                else:
                    wr1 = wr2
                    wg1 = wg2
                    mp0 = mp1
                    loff0 = loff1
            # ---- CHAIN29 BOUNDED PER-UNIT DRAIN, ported from the fit_v47
            # drain arm (FINDINGS_v47_hang.md §5/§14) to this bf16 champion's
            # geometry.  A unit with num_tiles < lag cannot deliver its OWN
            # last ``lag`` arrivals: full_bar(g) fires at iteration g+lag,
            # which lands in a LATER unit, and that cross-unit dependency
            # deadlocks QSAForward.  v33 ships entile 16 with KST=3 / VST=4,
            # so lag = max(kst-1, vst-1) = 3 and the exposed window is
            # num_tiles <= 2, i.e. union count <= 32 - WIDER than fit_v47's
            # (entile 24, kst 2 / vst 3, lag 2, count <= 24).  On the s220
            # count ladder 3,5,8,10,13,15,18,20,23,25,28,30,33,... that
            # predicts blocks B=0..11 dead and B=12 (count 33) live, whereas
            # v47 is dead only for B=0..8: B=9/10/11 are the discriminating
            # cells.  chain23 C5 already measured the entile-16/lag-3 geometry
            # hanging at count 3 (num_tiles 1); the num_tiles-2 half of this
            # window is new.  Settle the owed arrivals HERE instead,
            # depth-preservingly (KST/VST untouched, no SMEM change):
            # wait_group(0) is legal because every group this warp committed
            # is for a tile < gt + num_tiles, and the guard is dynamic and
            # never taken when num_tiles >= lag, so full-budget shapes pay one
            # predicate and the instruction stream is unchanged.
            if cutlass.const_expr(not self.arrive_on):
                if num_tiles < lag:
                    cute.arch.cp_async_wait_group(0)
                    cute.arch.fence_view_async_shared()
                    for l in cutlass.range_constexpr(lag):
                        if gt + num_tiles - lag + l >= arr_base:
                            cute.arch.mbarrier_arrive(
                                full_bar
                                + (gt + num_tiles - lag + l) % stages)
                # select_ (not an assignment inside the dynamic if) so the
                # watermark stays a plain loop-carried value.
                if cutlass.const_expr(self.aohyb):
                    # e3 (v4): under the hybrid the watermark advances on EVERY
                    # unit - a short one just settled its own owed arrivals above
                    # and a long one owes nothing (all of its arrivals fired on
                    # landing), so no later drain may re-arrive any of them.
                    arr_base = gt + num_tiles
                else:
                    arr_base = cutlass.select_(num_tiles < lag,
                                               gt + num_tiles, arr_base)
            gt += num_tiles
            ph += 1
            self._mwait(unit_posted, ph & 1)
            u = Int32(sUnit.load())
            cute.arch.mbarrier_arrive(unit_taken)
            if cutlass.const_expr(getattr(self, "relative_address32", False) or getattr(self, "sm100_q1_address_hoist", False)):
                assert self.static_short and self.single_unit_mma
                u = total_units
        # ---- drain the last ``lag`` arrivals in tile order (waited form
        # only; arrive-on arrivals were tied at issue and fire on landing;
        # outstanding asynchronous copies complete before the final arrivals.
        if cutlass.const_expr(not self.arrive_on):
            for l in cutlass.range_constexpr(lag):
                if gt - lag + l >= arr_base:
                    cute.arch.cp_async_wait_group(lag - 1 - l)
                    cute.arch.fence_view_async_shared()
                    cute.arch.mbarrier_arrive(full_bar + (gt - lag + l) % stages)


    @cute.jit
    def _refill_q_tmem(self, sQ,mQ, q_ptr, sP_lay, tiled_st, thr_st, rP_w,
                       row0, kvh, group, lane_rows, sg,
                       c_lo: cutlass.Constexpr = 0,
                       nc_exec: cutlass.Constexpr = -1):
        """Refill the resident TMEM Q operand for one unit (WG-A only, or
        chunk-split across both warpgroups under e19 qref=2).

        Thread ``sg`` owns TMEM lane ``sg`` = (row r = sg//group, head h =
        sg%group) and streams its Q row through registers in 32-dim (16 TMEM
        word) chunks: LDG chunk c+1 is issued before chunk c's tcgen05.st so
        the global-load latency overlaps the store pipeline.  Lanes past
        ``lane_rows`` clamp to the last live (row, head): their garbage Q
        rows stay in bounds, never reach the epilogue (valid_lane mask) and
        cannot contaminate live rows (MMA output rows are independent).
        ``rP_w`` supplies the exact register word-view layout the P t2st
        path uses (same St32x32b Repetition(16) value shape).
        ``c_lo``/``nc_exec`` select the chunk subrange this caller streams
        (-1 = all D//32 chunks = the champion single-WG form).

        e34 QSA_QDEPTH: the LDG pipeline DEPTH of this stream (2 = the
        champion's single-ahead double buffer, 3 = two chunks ahead).  The
        refill is a pure latency chain: with depth d the steady-state
        per-chunk period is >= L/(d-1) for a cold-DRAM round trip L, so at
        d=2 each of the D//32 = 8 chunks pays a FULL ~800-cycle miss (8L =
        ~6400 cycles = the ~3.8us/boundary that the e34 noqref(512) probe
        priced at +20.8us main), while d=3 halves it and d=4 would take it to
        L/3.  e31's QSA_QPF L2 prefetch recovered only -2.1us of that because
        the 2-6us prefetch lead is inside the window where the gather streams
        ~38MB through the 126MB L2 (its own A/B proof: QPF=2 at a ~45us lead
        is +2.78us WORSE), so the latency has to be hidden by DEPTH, not by
        warming.  Cost: one extra 16-register buffer - exactly the +16 that
        QSA_MMASCHED's 352-thread launch frees at the REG cap (65536/384 = 170
        -> ptxas 168; 65536/352 = 186 -> 184).  Bitwise identical (the chunk
        order, values and tcgen05.st sequence are unchanged; only the issue
        distance of the loads moves).
        """
        D = self.head_dim
        nchunk = D // 64  # first128 Q dims occupy64 packed TMEM columns
        nc = nchunk if nc_exec < 0 else nc_exec
        nbuf = self.qdepth
        assert nbuf <= nc + 1, "qdepth deeper than the chunk stream"
        r_act = cutlass.max(lane_rows // group, Int32(1))
        r = sg // group
        h = sg - r * group
        rr = cutlass.min(r, r_act - 1)
        qbase = _elem_pointer(mQ, (row0 + rr, kvh * group + h, 0)).toint()
        # Q's32-column refill needs a16-word store atom independent of the
        # N96 probability path's24-word shape.
        sP_lay=cute.make_layout((self.m_tile,8),stride=(65536,1))
        qt=cute.make_tensor(q_ptr,sP_lay)
        qst=cute.make_copy_atom(tcgen05.St32x32bOp(tcgen05.Repetition(8)),self.acc)
        tiled_st=tcgen05.make_tmem_copy(qst,qt)
        thr_st=tiled_st.get_slice(sg)
        word_shape=thr_st.partition_S(cute.make_identity_tensor((self.m_tile,8))).shape
        rQa = cute.make_rmem_tensor((32,), self.dtype)
        rQb = cute.make_rmem_tensor((32,), self.dtype)
        bufs = [rQa, rQb]
        if cutlass.const_expr(nbuf >= 3):
            rQc = cute.make_rmem_tensor((32,), self.dtype)
            bufs = [rQa, rQb, rQc]
        # prologue: put the first nbuf-1 chunks in flight before the first
        # tcgen05.st (nbuf=2 reproduces the champion's single gQa load).
        for p in cutlass.range_constexpr(nbuf - 1):
            if cutlass.const_expr(p < nc):
                gQp = cute.make_tensor(
                    cute.make_ptr(self.dtype, qbase + Int64((c_lo + p) * 32),
                                  cute.AddressSpace.gmem, assumed_align=self.q_load_align),
                    cute.make_layout((32,)))
                self._load_mtp_q(gQp, bufs[p], sg, lane_rows)
        for ci in cutlass.range_constexpr(nc):
            c = c_lo + ci
            cur = bufs[ci % nbuf]
            if cutlass.const_expr(ci + nbuf - 1 < nc):
                gQn = cute.make_tensor(
                    cute.make_ptr(self.dtype,
                                  qbase + Int64((c + nbuf - 1) * 32),
                                  cute.AddressSpace.gmem, assumed_align=self.q_load_align),
                    cute.make_layout((32,)))
                self._load_mtp_q(gQn, bufs[(ci + nbuf - 1) % nbuf], sg, lane_rows)
            cur_w = cute.make_tensor(
                cute.recast_ptr(cur.iterator, dtype=self.acc), word_shape)
            dest = cute.make_tensor(q_ptr + c * 8, sP_lay)
            cute.copy(tiled_st, cur_w, thr_st.partition_D(dest))


    @cute.jit
    def _stage_q_smem(self,sQ,mQ,row0,kvh,group,lane_rows,sg):
        # 8 adjacent lanes cooperate on one Qrow. Preserve16Bcopy
        # alignment while testing lower shared-memory wavefront amplification.
        flat=self._flatten(sQ[(None,None,None,0)],False)
        sw=sQ.iterator.type.swizzle_type
        assert sw.num_base==4 and sw.num_shift==3
        swmask=((1<<sw.num_bits)-1)<<4
        r_act=cutlass.max(lane_rows//group,Int32(1))
        for rowpass in cutlass.range_constexpr(4):
            qlane=sg//8+rowpass*(self.SOFTMAX_WARPS*4)
            qr=qlane//group;qh=qlane-qr*group
            rr=cutlass.min(qr,r_act-1)
            qbase=_elem_pointer(mQ,(row0+rr,kvh*group+qh,0)).toint()
            for colpass in cutlass.range_constexpr(2):
                col=((sg&7)+colpass*8)*16
                dst=flat.iterator+cute.crd2idx((qlane,col),flat.layout)
                address=Int32(Int64(dst.toint())&0x3FFFF)
                address=address^((address>>3)&Int32(swmask))
                if cutlass.const_expr(self.zero_pad_q):
                    source_bytes=cutlass.select_(qlane<lane_rows,Int32(16),Int32(0))
                    qsa_sm_ptx.cp_async_16_zfill(address,qbase+Int64(col),source_bytes)
                else:
                    qsa_sm_ptx.cp_async_16(address,qbase+Int64(col))
        cute.arch.cp_async_commit_group()

    @cute.jit
    def _refill_q_smem(self,sQ,mQ,row0,kvh,group,lane_rows,sg):
        # The second warpgroup uses the identical linear128-row Q mapping;
        # it owns dedicated Q SMEM, disjoint from WG0's Q TMEM stores.
        r_act=cutlass.max(lane_rows//group,Int32(1))
        rr=cutlass.min(sg//group,r_act-1)
        hh=sg-(sg//group)*group
        qbase=_elem_pointer(mQ,(row0+rr,kvh*group+hh,0)).toint()
        self._stage_q_smem(sQ,mQ,row0,kvh,group,lane_rows,sg)
        cute.arch.cp_async_wait_group(0)
        cute.arch.fence_view_async_shared()


    @cute.jit
    def _full_count(self,mCount,blk):
        if cutlass.const_expr(self.local_metadata):return Int32(mCount[0])
        else:return Int32(mCount[blk])

    @cute.jit
    def _fast_count(self,mCount,nblocks,blk):
        if cutlass.const_expr(self.local_metadata):return Int32(mCount[8])
        else:return Int32(mCount[2*nblocks+blk])

    @cute.jit
    def _kv_slice(self, full_count, u):
        # Partition whole attention tiles; no selected token is dropped.
        tiles = (full_count + self.ent_tile - 1) // self.ent_tile
        chunk = (tiles + self.kv_splits - 1) // self.kv_splits
        begin = (u % self.kv_splits) * chunk * self.ent_tile
        count = cutlass.max(cutlass.min(full_count-begin, chunk*self.ent_tile), Int32(0))
        # Empty partitions still execute the original harmless metadata prime.
        # Point it inside this row's allocation rather than beyond its union.
        safe_begin = cutlass.min(begin, cutlass.max(full_count-1, Int32(0)))
        return safe_begin, count

    @cute.jit
    def _drain_mtp_partial(self,tOtO,wg,sg,L,row0,kvh,group,lane_rows,
                           mref,row_sum,u,mStats,mPartial):
        ptr=cute.make_ptr(self.acc,(tOtO.iterator+(wg*16+(sg//32)*32)*65536).toint(),
            cute.AddressSpace.tmem,assumed_align=16)
        values=cute.make_rmem_tensor((32,),self.acc)
        split=u%self.kv_splits
        if L<lane_rows:
            row=(row0+L//group)*self.kv_splits+split
            head=kvh*group+L%group
            if sg%32<16:
                mStats[row,head,0]=mref
                mStats[row,head,1]=row_sum
        for chunk in cutlass.range_constexpr(4):
            qsa_sm_ptx.split_load(Int32((ptr+chunk*32).toint()),values,32,128)
            cute.arch.fence_view_async_tmem_load()
            if L<lane_rows:
                row=(row0+L//group)*self.kv_splits+split
                head=kvh*group+L%group
                col=(sg%32//16)*128+chunk*32
                op=cute.make_ptr(self.acc,_elem_pointer(mPartial,(row,head,col)).toint(),
                    cute.AddressSpace.gmem,assumed_align=16)
                for j in cutlass.range_constexpr(32):
                    values[j]=cutlass.select_(row_sum>Float32(0),values[j],Float32(0))
                cute.autovec_copy(values,cute.make_tensor(op,cute.make_layout((32,))))


    @cute.jit
    def _load_mtp_q(self, source, values, lane, lane_rows):
        if cutlass.const_expr(self.zero_pad_q):
            for j in cutlass.range_constexpr(32):
                values[j]=self.dtype(0)
            if lane<lane_rows:
                cute.autovec_copy(source,values)
        else:
            cute.autovec_copy(source,values)
