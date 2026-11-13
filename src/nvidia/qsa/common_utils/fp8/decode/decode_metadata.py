"""Q1 fixed-rank metadata: preserve holes as masked safe entries, no scan."""
import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Int32,Int64
from .native_metadata import _ballot32

@cute.jit
def _expand8(bits:Int32):
    value=Int32(0)
    for b in cutlass.range_constexpr(8):
        value|=cutlass.select_(((bits>>b)&1)!=0,Int32(15)<<Int32(4*b),Int32(0))
    return value

@cute.jit
def build_decode_metadata(mIds,mPos,mReq,mBT,mK,meta,tidx,blk,unit,nblocks,
                          splits:cutlass.Constexpr,local_entries:cutlass.Constexpr,
                          rb:cutlass.Constexpr,page:cutlass.Constexpr,local_view:cutlass.Constexpr,
                          request_span:cutlass.Constexpr=0,physical:cutlass.Constexpr=True):
    assert physical and local_view and rb==66
    lane,warp=tidx%32,tidx//32
    visible=Int64(mPos[blk])+1
    complete=Int32(cutlass.min(cutlass.max(visible//4,Int64(0)),Int64(512)))
    tail_id,tail_count=Int32(visible//4),Int32(visible%4)
    request=Int32(mReq[blk]);width,pages=Int32(mBT.shape[1]),Int32(mK.shape[0])
    request_valid=request==0;page_bias=Int32(0)
    if cutlass.const_expr(request_span>0):
        width=Int32(request_span//page)
        requests=Int32(mBT.shape[1])//width
        request_valid=(request>=0)&(request<requests)
        page_bias=cutlass.min(cutlass.max(request,Int32(0)),requests-1)*width
    q0,q1=Int32(mK.stride[0])//64,Int32(mK.stride[1])//64
    tail_page=tail_id//(page//4)
    tail_safe=cutlass.min(cutlass.max(tail_page,Int32(0)),width-1)
    tail_pp=Int32(mBT[0,page_bias+tail_safe])
    tail_live=Int32((tail_count>0)&(tail_id>=0)&request_valid&(tail_page>=0)&(tail_page<width)&(tail_pp>=0)&(tail_pp<pages))
    total=cutlass.select_(request_valid,complete+tail_live,Int32(0))
    tiles=(total+31)//32;chunk=(tiles+splits-1)//splits
    begin=(unit%splits)*chunk*32
    count=cutlass.max(cutlass.min(total-begin,chunk*32),Int32(0))
    rm_offset=local_entries+12
    tail_mask=cutlass.select_(tail_live!=0,((Int32(1)<<tail_count)-1)<<((complete%8)*4),Int32(0))
    if tidx<256:
        col0,col1=warp*64+lane,warp*64+lane+32
        gid0,gid1=Int32(mIds[blk,col0]),Int32(mIds[blk,col1])
        lp0,lp1=gid0//(page//4),gid1//(page//4)
        safe0=cutlass.min(cutlass.max(lp0,Int32(0)),width-1)
        safe1=cutlass.min(cutlass.max(lp1,Int32(0)),width-1)
        pp0,pp1=Int32(mBT[0,page_bias+safe0]),Int32(mBT[0,page_bias+safe1])
        valid0=Int32((col0<complete)&(gid0>=0)&request_valid&(lp0>=0)&(lp0<width)&(pp0>=0)&(pp0<pages))
        valid1=Int32((col1<complete)&(gid1>=0)&request_valid&(lp1>=0)&(lp1<width)&(pp1>=0)&(pp1<pages))
        ballot0,ballot1=_ballot32(valid0),_ballot32(valid1)
        off0=cutlass.select_(valid0!=0,pp0*q0+(gid0%(page//4))*4*q1,Int32(0))
        off1=cutlass.select_(valid1!=0,pp1*q0+(gid1%(page//4))*4*q1,Int32(0))
        if (col0<complete)&(col0>=begin)&(col0<begin+count):meta[col0-begin]=off0
        if (col1<complete)&(col1>=begin)&(col1<begin+count):meta[col1-begin]=off1
        if lane<4:
            w0,w1=warp*8+lane,warp*8+lane+4
            mask0,mask1=_expand8((ballot0>>(lane*8))&255),_expand8((ballot1>>(lane*8))&255)
            meta[rm_offset+w0]=mask0|cutlass.select_(w0==complete//8,tail_mask,Int32(0))
            meta[rm_offset+w1]=mask1|cutlass.select_(w1==complete//8,tail_mask,Int32(0))
    if tidx==0:
        if (tail_live!=0)&(complete>=begin)&(complete<begin+count):
            meta[complete-begin]=tail_pp*q0+(tail_id%(page//4))*4*q1
        if count==0:meta[0]=Int32(0)
        meta[local_entries]=total
        meta[local_entries+8]=cutlass.select_(tail_live!=0,complete//32,Int32(33))
        meta[rm_offset+64]=cutlass.select_(complete==512,tail_mask,Int32(0))
        meta[rm_offset+65]=Int32(0)
    cute.arch.sync_threads()
    return (cute.make_tensor(meta.iterator,cute.make_layout((local_entries,))),
            cute.make_tensor(meta.iterator+local_entries,cute.make_layout((12,))),
            cute.make_tensor(meta.iterator+rm_offset,cute.make_layout((rb,))))
