"""Build one Q4 union shared by all KV heads; counted in the public call."""
import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Int32, Int64
from cutlass._mlir.dialects import llvm
from cutlass import utils
from .union_metadata import build_union_metadata

class UnionPreprocess:
    def __init__(self,page,maxe,entries,rb,request_span,hash_mix,full_first,physical,direct_masks=False,use_pdl=False,metadata_ready=False,prefetch_k_tiles=2):
        self.br=maxe//513
        assert self.br in (2,4)
        self.use_pdl=use_pdl
        self.metadata_ready=metadata_ready
        assert not metadata_ready or (use_pdl and self.br==2)
        self.direct_masks=direct_masks
        self.prefetch_k_tiles=prefetch_k_tiles
        self.page,self.maxe,self.entries,self.rb=page,maxe,entries,rb
        self.request_span,self.hash_mix,self.full_first,self.physical=request_span,hash_mix,full_first,physical

    @cute.jit
    def __call__(self,mIds,mPos,mReq,mBT,mK,mWS,mCount,mRM,stream):
        self.atrex_union(mIds,mPos,mReq,mBT,mK,mWS,mCount,mRM).launch(
            grid=(mIds.shape[0]//self.br,1,1),block=(256,1,1),stream=stream,use_pdl=False)

    @cute.kernel
    def atrex_union(self,mIds,mPos,mReq,mBT,mK,mWS,mCount,mRM):
        tidx,_,_=cute.arch.thread_idx();blk,_,_=cute.arch.block_idx()
        if cutlass.const_expr(self.metadata_ready):
            # One writer initializes this CTA's count before the first trigger.
            # The ordinary primary launch also orders prior caller/replay work.
            if tidx==0:
                mCount[blk]=Int32(-1)
                llvm.inline_asm(None,[],"fence.release.gpu;","~{memory}",
                    has_side_effects=True,is_align_stack=False,asm_dialect=llvm.AsmDialect.AD_ATT)
                cute.arch.griddepcontrol_launch_dependents()
        elif cutlass.const_expr(self.use_pdl):
            cute.arch.griddepcontrol_launch_dependents()
        smem=utils.SmemAllocator()
        meta=smem.allocate_tensor(Int32,cute.make_layout((self.entries+12+4*self.rb+8,)),16)
        scratch=smem.allocate_tensor(Int32,cute.make_layout((2048*self.br,)),16)
        ws,count,masks=build_union_metadata(mIds,mPos,mReq,mBT,mK,meta,scratch,tidx,blk,
            self.entries,self.rb,self.page,self.request_span,self.hash_mix,self.full_first,self.physical,self.br,direct_masks=self.direct_masks,live_rows_only=self.br==2,prefetch_k_tiles=self.prefetch_k_tiles if self.use_pdl and self.br==2 else 0)
        total=Int32(count[0]);nblocks=Int32(mIds.shape[0])//self.br
        for i in cutlass.range_constexpr((self.maxe+255)//256):
            idx=tidx+i*256
            if idx<self.maxe:
                value=Int32(0)
                if idx<total:value=Int32(ws[idx])
                mWS[blk*self.maxe+idx]=value
        for i in cutlass.range_constexpr((self.br*self.rb+255)//256):
            idx=tidx+i*256
            if idx<self.br*self.rb:mRM[blk*self.br*self.rb+idx]=Int32(masks[idx])
        if cutlass.const_expr(self.metadata_ready):
            if tidx==0:
                mCount[nblocks+blk]=blk
                mCount[2*nblocks+blk]=Int32(count[8])
            # All workspace, masks and auxiliary count stores precede release.
            cute.arch.sync_threads()
            if tidx==0:
                address=Int64((mCount.iterator+blk).toint())
                llvm.inline_asm(None,[address.ir_value(),total.ir_value()],
                    "st.release.gpu.global.u32 [$0], $1;","l,r,~{memory}",
                    has_side_effects=True,is_align_stack=False,asm_dialect=llvm.AsmDialect.AD_ATT)
        else:
            if tidx==0:
                mCount[blk]=total
                mCount[nblocks+blk]=blk
                mCount[2*nblocks+blk]=Int32(count[8])
