"""Build one Q4 union shared by all KV heads; counted in the public call."""
import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Int32
from cutlass import utils
from .union_metadata import build_union_metadata

class UnionPreprocess:
    def __init__(self,page,maxe,entries,rb,request_span,hash_mix,full_first,physical):
        self.page,self.maxe,self.entries,self.rb=page,maxe,entries,rb
        self.request_span,self.hash_mix,self.full_first,self.physical=request_span,hash_mix,full_first,physical

    @cute.jit
    def __call__(self,mIds,mPos,mReq,mBT,mK,mWS,mCount,mRM,stream):
        self.atrex_union(mIds,mPos,mReq,mBT,mK,mWS,mCount,mRM).launch(
            grid=(mIds.shape[0]//4,1,1),block=(256,1,1),stream=stream)

    @cute.kernel
    def atrex_union(self,mIds,mPos,mReq,mBT,mK,mWS,mCount,mRM):
        tidx,_,_=cute.arch.thread_idx();blk,_,_=cute.arch.block_idx()
        smem=utils.SmemAllocator()
        meta=smem.allocate_tensor(Int32,cute.make_layout((self.entries+12+4*self.rb+8,)),16)
        scratch=smem.allocate_tensor(Int32,cute.make_layout((8192,)),16)
        ws,count,masks=build_union_metadata(mIds,mPos,mReq,mBT,mK,meta,scratch,tidx,blk,
            self.entries,self.rb,self.page,self.request_span,self.hash_mix,self.full_first,self.physical)
        total=Int32(count[0]);nblocks=Int32(mIds.shape[0])//4
        for i in cutlass.range_constexpr((self.maxe+255)//256):
            idx=tidx+i*256
            if idx<self.maxe:
                value=Int32(0)
                if idx<total:value=Int32(ws[idx])
                mWS[blk*self.maxe+idx]=value
        for i in cutlass.range_constexpr((4*self.rb+255)//256):
            idx=tidx+i*256
            if idx<4*self.rb:mRM[blk*4*self.rb+idx]=Int32(masks[idx])
        if tidx==0:
            mCount[blk]=total
            mCount[nblocks+blk]=blk
            mCount[2*nblocks+blk]=Int32(count[8])
