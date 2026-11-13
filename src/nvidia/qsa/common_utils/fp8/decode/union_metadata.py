"""Q4 union in temporary K shared storage, preserving per-row token masks."""
import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Int32,Int64
from cutlass._mlir.dialects import llvm
from ... import qsa_sm_ptx

@cute.jit
def _cas(pointer,old:Int32,new:Int32):
    address=Int32(Int64(pointer.toint())&0x3ffff)
    return Int32(llvm.inline_asm(Int32.mlir_type,[address.ir_value(),old.ir_value(),new.ir_value()],
        'atom.shared.cas.b32 $0, [$1], $2, $3;','=r,r,r,r',
        has_side_effects=True,is_align_stack=False,asm_dialect=llvm.AsmDialect.AD_ATT))

@cute.jit
def _partial(membership:Int32):
    result=Int32(0)
    for rr in cutlass.range_constexpr(4):
        nibble=(membership>>(rr*4))&15
        result=result|Int32((nibble!=0)&(nibble!=15))
    return result

@cute.jit
def build_union_metadata(mIds,mPos,mReq,mBT,mK,meta,scratch,tidx,blk,
                         entries:cutlass.Constexpr,rb:cutlass.Constexpr,
                         page:cutlass.Constexpr,request_span:cutlass.Constexpr,
                         hash_mix:cutlass.Constexpr=False,full_first:cutlass.Constexpr=False,
                         physical:cutlass.Constexpr=False,br:cutlass.Constexpr=4,direct_masks:cutlass.Constexpr=False,live_rows_only:cutlass.Constexpr=False,prefetch_k_tiles:cutlass.Constexpr=0,grouped_insert:cutlass.Constexpr=False):
    # At most 4*(512+tail) distinct keys in 4096 slots. Full flattened keys
    # include request identity, so arbitrary request/row permutations are safe.
    # Keep the appended causal tail in a separate namespace: its token may
    # also occur in a selected block, and the oracle counts both occurrences.
    # The MTP domain bounds flattened tokens below 2**31, leaving group bit29
    # free for this tag. The tag is removed before storing gather addresses.
    slots=1024*br
    per_thread=slots//256
    if tidx<256:
        for i in cutlass.range_constexpr(per_thread):
            scratch[tidx+i*256]=Int32(-1)
            scratch[slots+tidx+i*256]=Int32(0)
    count_off=entries;mask_off=entries+12;scan_off=mask_off+4*rb
    if cutlass.const_expr(not direct_masks):
        for i in cutlass.range_constexpr((4*rb+255)//256):
            idx=tidx+i*256
            if (idx<4*rb)&(tidx<256):meta[mask_off+idx]=Int32(0)
    if tidx==0:meta[count_off+8]=Int32(1000000)
    cute.arch.sync_threads()
    width=Int32(mBT.shape[1]);requests=Int32(1)
    if cutlass.const_expr(request_span>0):
        width=Int32(request_span//page);requests=Int32(mBT.shape[1])//width
    if cutlass.const_expr(grouped_insert):
        from ...union_insert import insert_query_columns
        insert_query_columns(mIds, mPos, mReq, mBT, mK, scratch, tidx, blk,
                             page, request_span, br, hash_mix, hash_shift=5 if page == 128 and br == 2 else 6)
    else:
        for part in cutlass.range_constexpr((br*513+255)//256):
            item=tidx+part*256
            if (item<br*513)&(tidx<256):
                rr=item//512;col=item%512
                if item>=br*512:rr=item-br*512
                row=blk*br+rr
                visible=Int64(mPos[row])+Int64(1)
                complete=cutlass.min(cutlass.max(visible//4,Int64(0)),Int64(512))
                gid=Int32(0);nibble=Int32(15);live=Int32(0)
                if item<br*512:
                    gid=Int32(mIds[row,col]);live=Int32(col<complete)
                else:
                    gid=Int32(visible//4);nibble=(Int32(1)<<Int32(visible%4))-1
                    live=Int32((visible>0)&(nibble!=0))
                request=Int32(mReq[row])
                safe_req=cutlass.min(cutlass.max(request,Int32(0)),requests-1)
                lp=gid//(page//4);safe_lp=cutlass.min(cutlass.max(lp,Int32(0)),width-1)
                physical_page=Int32(mBT[0,safe_req*width+safe_lp])
                live=live&Int32((gid>=0)&(lp>=0)&(lp<width)&(request>=0)&(request<requests)&(physical_page>=0)&(physical_page<Int32(mK.shape[0])))
                if live!=0:
                    key=gid+safe_req*width*(page//4)
                    if item>=br*512:key=key|Int32(1<<29)
                    slot=key&(slots-1)
                    if cutlass.const_expr(hash_mix):slot=((key^(key>>12))*Int32(-1640531535))&(slots-1)
                    previous=_cas(scratch.iterator+slot,Int32(-1),key)
                    while (previous!=Int32(-1))&(previous!=key):
                        slot=(slot+1)&(slots-1);previous=_cas(scratch.iterator+slot,Int32(-1),key)
                    cute.arch.atomic_or(scratch.iterator+slots+slot,nibble<<(rr*4),sem='relaxed',scope='cta')
    cute.arch.sync_threads()
    if cutlass.const_expr(direct_masks):
        # The scan barrier retires these reads before scratch is reused below.
        assert br in (2,4)
        cached_keys=cute.make_rmem_tensor((per_thread,),Int32)
        cached_members=cute.make_rmem_tensor((per_thread,),Int32)
        for i in cutlass.range_constexpr(per_thread):
            source_thread=cutlass.min(tidx,Int32(255))
            cached_keys[i]=Int32(scratch[source_thread*per_thread+i])
            cached_members[i]=Int32(scratch[slots+source_thread*per_thread+i])
    local=Int32(0)
    if tidx<256:
        for i in cutlass.range_constexpr(per_thread):
            if cutlass.const_expr(direct_masks):
                live=Int32(cached_keys[i]!=Int32(-1))
            else:
                live=Int32(scratch[tidx*per_thread+i]!=Int32(-1))
            if cutlass.const_expr(full_first):live=live&Int32(_partial(Int32(scratch[slots+tidx*per_thread+i]))==0)
            local+=live
    lane=tidx%32;warp=tidx//32;inclusive=local
    for offset in cutlass.range_constexpr(5):
        distance=1<<offset
        other=Int32(cute.arch.shuffle_sync(inclusive,cutlass.max(lane-distance,Int32(0)),mask_and_clamp=31))
        inclusive+=cutlass.select_(lane>=distance,other,Int32(0))
    if (lane==31)&(tidx<256):meta[scan_off+warp]=inclusive
    cute.arch.sync_threads()
    prefix=inclusive-local;total=Int32(0)
    for w in cutlass.range_constexpr(8):
        number=Int32(meta[scan_off+w]);total+=number
        prefix+=cutlass.select_(warp>w,number,Int32(0))
    if cutlass.const_expr(full_first):
        if tidx==0:
            meta[count_off+9]=total
            meta[count_off+8]=total//32
        cute.arch.sync_threads()
    rank=prefix
    for i in cutlass.range_constexpr(per_thread):
        safe_thread=cutlass.min(tidx,Int32(255))
        if cutlass.const_expr(direct_masks):
            key=Int32(cached_keys[i]);membership=Int32(cached_members[i])
        else:
            key=Int32(scratch[safe_thread*per_thread+i]);membership=Int32(scratch[slots+safe_thread*per_thread+i])
        if tidx>=256:key=Int32(-1)
        if key!=Int32(-1):
            partial=_partial(membership)
            output_rank=rank
            if cutlass.const_expr(full_first):
                if partial!=0:
                    output_rank=Int32(cute.arch.atomic_add(meta.iterator+count_off+9,Int32(1),sem='relaxed',scope='cta'))
                else:rank+=1
            else:rank+=1
            logical_key=key&Int32((1<<29)-1)
            if cutlass.const_expr(prefetch_k_tiles > 0):
                assert br == 2 and not physical and not full_first
                if (output_rank < prefetch_k_tiles * 32) & (Int32(mK.shape[2]) == 1):
                    # The hash insertion validated this complete physical
                    # four-token group, including a separately tagged tail.
                    lp = logical_key // (page // 4)
                    pp = Int32(mBT[0, lp])
                    token = (logical_key % (page // 4)) * 4
                    address = (mK.iterator.toint() + Int64(pp) * Int64(mK.stride[0])
                               + Int64(token) * Int64(mK.stride[1]))
                    for prefetch_item in cutlass.range_constexpr(8):
                        qsa_sm_ptx.prefetch_l2(address
                            + Int64(prefetch_item // 2) * Int64(mK.stride[1]) + Int64(prefetch_item % 2) * Int64(128))
            encoded=logical_key
            if cutlass.const_expr(physical):
                lp=logical_key//(page//4);pp=Int32(mBT[0,lp])
                encoded=Int32(Int64(pp)*(Int64(mK.stride[0])//64)+Int64(logical_key%(page//4))*4*(Int64(mK.stride[1])//64))
            meta[output_rank]=encoded
            if cutlass.const_expr(direct_masks):
                scratch[output_rank]=membership
            else:
                # Unused rows have zero membership and zero-initialized masks.
                for rr in cutlass.range_constexpr(br if live_rows_only else 4):
                    nibble=(membership>>(rr*4))&15
                    cute.arch.atomic_or(meta.iterator+mask_off+rr*rb+output_rank//8,nibble<<((output_rank%8)*4),sem='relaxed',scope='cta')
            if cutlass.const_expr(not full_first):
                if partial!=0:
                    cute.arch.atomic_min(meta.iterator+count_off+8,output_rank//32,sem='relaxed',scope='cta')
    if cutlass.const_expr(full_first):
        cute.arch.sync_threads()
        total=Int32(meta[count_off+9])
    if cutlass.const_expr(direct_masks):
        cute.arch.sync_threads()
        for step in cutlass.range_constexpr((rb+255)//256):
            word=tidx+step*256
            if (word<rb)&(tidx<256):
                packed=cute.make_rmem_tensor((4,),Int32)
                packed.fill(Int32(0))
                for mask_item in cutlass.range_constexpr(8):
                    rank_id=word*8+mask_item
                    bits=Int32(0)
                    if rank_id<total:bits=Int32(scratch[rank_id])
                    for rr in cutlass.range_constexpr(4):
                        packed[rr]=Int32(packed[rr])|(((bits>>(rr*4))&15)<<(mask_item*4))
                for rr in cutlass.range_constexpr(4):meta[mask_off+rr*rb+word]=Int32(packed[rr])
    if tidx==0:
        meta[count_off]=total
        if total==0:meta[0]=Int32(0)
    cute.arch.sync_threads()
    ws=cute.make_tensor(meta.iterator,cute.make_layout((entries,)))
    counts=cute.make_tensor(meta.iterator+count_off,cute.make_layout((12,)))
    masks=cute.make_tensor(meta.iterator+mask_off,cute.make_layout((4*rb,)))
    return ws,counts,masks
