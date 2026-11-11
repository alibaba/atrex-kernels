"""FP8 M64 decode/MTP roles with complete queries in shared memory."""
import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Int32, Int64, Float32
from .qsa_kernel import QSAForward as M128
from .layouts import _elem_pointer
from ... import qsa_sm_ptx


class QSAForward(M128):
    def __init__(self,*args,decode_reuse_v=False,k_load_warps=2,relative_address32=False,metadata_ready=False,**kwargs):
        self.metadata_ready=metadata_ready
        self.relative_address32=relative_address32 or getattr(self,'relative_address32',False)
        kwargs['m_tile']=64
        super().__init__(*args,**kwargs)
        assert self.br in (1,2,4) and self.br*self.group<=64 and self.kv_splits>=1 and self.gw==2
        self.SOFTMAX_WARPS=4
        self.live_softmax_warps=(self.br*self.group+15)//16
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

        assert k_load_warps in (2,4)
        if k_load_warps==4:
            # Same K entries and order, divided across four producers.
            assert self.br==4 and self.hkv in (1,2)
            self.k_load_warps=4
            extra_v = self.VLOAD_WARP2 >= 0
            self.KLOAD_WARP3,self.KLOAD_WARP4=(9,10) if extra_v else (8,9)
            self.NWARPS,self.threads,self.take_warps=12,384,9+int(extra_v)

    @cute.jit
    def _logical_row(self,sg,wg):
        return (sg//32)*16+sg%16

    @cute.jit
    def _idle_decode_softmax(self,tidx,unit_posted,unit_taken,sUnit,total_units,
                            nblocks,mQ,mCount,tStS,q_col,sQ,qready,mGate,sG,gstaged):
        # Keep the complete warpgroup's register and Q initialization protocol.
        # Only warps owning live M64 rows enter the tile loop.
        assert self.br*self.group<=64 and self.single_unit_mma
        sg=tidx%128
        self._mwait(unit_posted,0)
        u=Int32(sUnit.load());cute.arch.mbarrier_arrive(unit_taken)
        b=self._unit_blk(u,nblocks,mCount);h=self._unit_kvh(u,nblocks)
        self._refill_q_smem(sQ,mQ,b*self.br,h,self.group,Int32(self.br*self.group),sg)
        cute.arch.fence_view_async_tmem_store()
        cute.arch.mbarrier_arrive(qready)
        if cutlass.const_expr(self.gate_factor):
            self._stage_gate_factor(tidx,b,h,mGate,sG)
            cute.arch.mbarrier_arrive(gstaged)
        elif cutlass.const_expr(self.decode_gate):
            if cutlass.const_expr(self.br==1):
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
            else:
                # The idle softmax warp(s) prefetch every live row's gate while
                # active warps execute attention. One 4096-BF16 slab per warp.
                helpers=32*(4-self.live_softmax_warps)
                vectors=512*self.live_softmax_warps
                for iteration in cutlass.range_constexpr((vectors+helpers-1)//helpers):
                    i=tidx-32*self.live_softmax_warps+iteration*helpers
                    if i<vectors:
                        group_warp=i//512;local=i%512;glane=local%32
                        gate_row=group_warp*16+glane%16
                        col=(glane//16)*128+(local//32)*8
                        safe=cutlass.min(gate_row,Int32(self.br*self.group-1))
                        src=_elem_pointer(mGate,(b*self.br+safe//self.group,
                            h*self.group+safe%self.group,col)).toint()
                        dst=Int32(Int64((sG.iterator+i*8).toint())&0x3ffff)
                        size=cutlass.select_(gate_row<self.br*self.group,Int32(16),Int32(0))
                        qsa_sm_ptx.cp_async_16_zfill(dst,src,size)
            cute.arch.cp_async_commit_group()
            cute.arch.cp_async_wait_group(0)
            cute.arch.fence_view_async_shared()
            cute.arch.mbarrier_arrive(gstaged)
        # Warp one continues as a V producer after Q and gate initialization.
        if not (self.decode_reuse_v and tidx//32==1):
            self._mwait(unit_posted,1)
            cute.arch.mbarrier_arrive(unit_taken)

    @cute.jit
    def _stage_gate_factor(self,tidx,block,kvh,mGate,sG):
        # The original FP32 tanh sigmoid, moved ahead of the output drain.
        # Keep the second V producer free to enter its gather pipeline.
        helper_begin=64 if self.decode_reuse_v else 32*self.live_softmax_warps
        helpers=128-helper_begin
        lane_rows=self.br*self.group
        rg=cute.make_rmem_tensor((8,),cutlass.BFloat16)
        factors=cute.make_rmem_tensor((8,),Float32)
        if tidx>=helper_begin:
            for iteration in cutlass.range_constexpr((lane_rows*32+helpers-1)//helpers):
                vector=tidx-helper_begin+iteration*helpers
                if vector<lane_rows*32:
                    row=(vector%(2*lane_rows))%lane_rows
                    half=(vector%(2*lane_rows))//lane_rows
                    col=half*128+(vector//(2*lane_rows))*8
                    ptr=cute.make_ptr(cutlass.BFloat16,
                        _elem_pointer(mGate,(block*self.br+row//self.group,kvh*self.group+row%self.group,col)).toint(),
                        cute.AddressSpace.gmem,assumed_align=16)
                    cute.autovec_copy(cute.make_tensor(ptr,cute.make_layout((8,))),rg)
                    for pair in cutlass.range_constexpr(4):
                        j=pair*2
                        g0,g1=cute.arch.fma_packed_f32x2((Float32(rg[j]),Float32(rg[j+1])),(Float32(.5),Float32(.5)),(Float32(0),Float32(0)))
                        t0=qsa_sm_ptx.tanh_approx(g0);t1=qsa_sm_ptx.tanh_approx(g1)
                        s0,s1=cute.arch.fma_packed_f32x2((t0,t1),(Float32(.5),Float32(.5)),(Float32(.5),Float32(.5)))
                        factors[j]=s0;factors[j+1]=s1
                    cute.autovec_copy(factors,cute.make_tensor(sG.iterator+vector*8,cute.make_layout((8,))))
