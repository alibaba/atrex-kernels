"""BF16 M64 decode/MTP roles with zero-padded query rows."""
import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Int32, Int64
from .qsa_kernel import QSAForward as M128
from .layouts import _elem_pointer
from . import qsa_sm_ptx


class QSAForward(M128):
    def __init__(self,*args,decode_reuse_v=False,address_hoist=False,grouped_union=False,**kwargs):
        self.grouped_union=grouped_union
        kwargs['m_tile']=64
        super().__init__(*args,**kwargs)
        assert self.br in (1,4) and self.br*self.group<=64 and self.kv_splits>=1 and self.gw==2
        self.sm100_q1_address_hoist = bool(address_hoist and self.br==1 and self.union_physical)
        self.SOFTMAX_WARPS=4
        self.MMA_WARP=4
        self.KLOAD_WARP=5
        self.VLOAD_WARP=6
        self.KLOAD_WARP2=7
        self.VLOAD_WARP2=-1
        self.NWARPS=8
        self.take_warps=7
        self.threads=256
        self.decode_reuse_v=bool(decode_reuse_v and self.decode_gate and self.br==1)
        if self.decode_reuse_v:
            self.gwr=2
        if self.br==4 and self.union_v_warps==2:
            self.VLOAD_WARP2=8
            self.gwr=2
            self.NWARPS=12
            self.take_warps=8
            self.threads=384

    @cute.jit
    def _logical_row(self,sg,wg):
        return (sg//32)*16+sg%16

    @cute.jit
    def _refill_q_tmem(self,sQ,mQ,q_ptr,sP_lay,tiled_st,thr_st,rP_w,
                       row0,kvh,group,lane_rows,sg,
                       c_lo:cutlass.Constexpr=0,nc_exec:cutlass.Constexpr=-1):
        # Each half-warp owns 64 dimensions of both Q halves. Logical rows
        # 16*w..16*w+15 reside in TMEM data paths 32*w..32*w+15.
        lane=sg%32;half=lane//16;logical=(sg//32)*16+lane%16
        r_act=cutlass.max(lane_rows//group,Int32(1))
        rr=cutlass.min(logical//group,r_act-1);head=logical%group
        qbase=_elem_pointer(mQ,(row0+rr,kvh*group+head,0)).toint()
        flat=self._flatten(sQ[(None,None,None,0)],False)
        sw=sQ.iterator.type.swizzle_type
        assert sw.num_base==4 and sw.num_shift==3
        swmask=((1<<sw.num_bits)-1)<<4
        for chunk in cutlass.range_constexpr(8):
            dst=flat.iterator+cute.crd2idx((logical,half*64+chunk*8),flat.layout)
            addr=Int32(Int64(dst.toint())&0x3ffff)
            addr=addr^((addr>>3)&Int32(swmask))
            if cutlass.const_expr(self.zero_pad_q):
                size=cutlass.select_(logical<lane_rows,Int32(16),Int32(0))
                qsa_sm_ptx.cp_async_16_zfill(addr,qbase+Int64(256+half*128+chunk*16),size)
            else:
                qsa_sm_ptx.cp_async_16(addr,qbase+Int64(256+half*128+chunk*16))
        cute.arch.cp_async_commit_group()
        r0=cute.make_rmem_tensor((32,),self.dtype)
        r1=cute.make_rmem_tensor((32,),self.dtype)
        for chunk in cutlass.range_constexpr(2):
            cur=r0 if chunk==0 else r1
            src=cute.make_tensor(cute.make_ptr(self.dtype,qbase+Int64(half*128+chunk*64),
                cute.AddressSpace.gmem,assumed_align=self.q_load_align),cute.make_layout((32,)))
            self._load_mtp_q(src,cur,logical,lane_rows)
        for chunk in cutlass.range_constexpr(2):
            cur=r0 if chunk==0 else r1
            words=cute.make_tensor(cute.recast_ptr(cur.iterator,dtype=self.acc),cute.make_layout((16,)))
            addr=Int32((q_ptr+(sg//32)*32*65536+chunk*16).toint())
            qsa_sm_ptx.split_store(addr,words,16,32)
        cute.arch.cp_async_wait_group(0)
        cute.arch.fence_view_async_shared()


    @cute.jit
    def _idle_decode_softmax(self,tidx,unit_posted,unit_taken,sUnit,total_units,
                            nblocks,mQ,mCount,tStS,q_col,sQ,qready,mGate,sG,gstaged):
        # Keep the complete warpgroup's register and Q initialization protocol.
        # Only warp zero owns live Q1 rows; other warps can leave the tile loop.
        assert self.br==1 and self.group<=16 and self.single_unit_mma
        sg=tidx%128
        self._mwait(unit_posted,0)
        u=Int32(sUnit.load());cute.arch.mbarrier_arrive(unit_taken)
        b=self._unit_blk(u,nblocks,mCount);h=self._unit_kvh(u,nblocks)
        self._refill_q_tmem(sQ,mQ,tStS.iterator+q_col,None,None,None,None,
            b,h,self.group,Int32(self.group),sg)
        cute.arch.fence_view_async_tmem_store()
        cute.arch.mbarrier_arrive(qready)
        if cutlass.const_expr(self.decode_gate):
            # Chunk-major layout gives every 8-element vector a bank-contiguous
            # lane span when the active softmax warp drains it later.
            for iteration in cutlass.range_constexpr(6):
                i=tidx-32+iteration*96
                if i<512:
                    glane=i%32;head=glane%16
                    col=(glane//16)*128+(i//32)*8
                    safe_head=cutlass.min(head,Int32(self.group-1))
                    src=_elem_pointer(mGate,(b,h*self.group+safe_head,col)).toint()
                    dst=Int32(Int64((sG.iterator+i*8).toint())&0x3ffff)
                    size=cutlass.select_(head<self.group,Int32(16),Int32(0))
                    qsa_sm_ptx.cp_async_16_zfill(dst,src,size)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.fence_view_async_shared()
            cute.arch.mbarrier_arrive(gstaged)
        # Warp one continues as a V producer after Q and gate initialization.
        if not (self.decode_reuse_v and tidx//32==1):
            self._mwait(unit_posted,1)
            cute.arch.mbarrier_arrive(unit_taken)
