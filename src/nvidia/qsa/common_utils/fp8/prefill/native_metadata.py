"""Build one CTA's stable compact metadata directly from the public inputs."""
import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Int32, Int64
from cutlass._mlir.dialects import llvm


@cute.jit
def _ballot32(value: Int32):
    return Int32(llvm.inline_asm(Int32.mlir_type, [value.ir_value()],
        '{ .reg .pred p; setp.ne.u32 p, $1, 0; vote.ballot.sync.b32 $0, p, 0xffffffff; }',
        '=r,r', has_side_effects=False, is_align_stack=False,
        asm_dialect=llvm.AsmDialect.AD_ATT))


@cute.jit
def build_native_metadata(mIds, mPos, mReq, mBT, mK, meta, tidx, blk, unit, nblocks,
                          splits: cutlass.Constexpr, local_entries: cutlass.Constexpr,
                          rb: cutlass.Constexpr, page: cutlass.Constexpr, local_view: cutlass.Constexpr,
                          request_span: cutlass.Constexpr = 0):
    # Only one unit per CTA is admitted by the host. Compact all IDs to find
    # stable ranks, but retain only the entries consumed by this partition.
    lane, warp = tidx % 32, tidx // 32
    visible = Int64(mPos[blk])+Int64(1)
    complete = cutlass.min(cutlass.max(visible//4, Int64(0)), Int64(512))
    tail_id, tail_count = Int32(visible//4), Int32(visible%4)
    request = Int32(mReq[blk])
    width, pages = Int32(mBT.shape[1]), Int32(mK.shape[0])
    request_valid = request == 0
    page_bias = Int32(0)
    if cutlass.const_expr(request_span > 0):
        width = Int32(request_span // page)
        requests = Int32(mBT.shape[1]) // width
        request_valid = (request >= 0) & (request < requests)
        page_bias = cutlass.min(cutlass.max(request, Int32(0)), requests-1)*width
    gid0, gid1 = Int32(0), Int32(0)
    valid0, valid1 = Int32(0), Int32(0)
    ballot0, ballot1 = Int32(0), Int32(0)
    count_offset = local_entries
    rm_offset = local_entries+12
    scan_offset = rm_offset+rb
    if tidx < 256:
        col0, col1 = warp*64+lane, warp*64+lane+32
        gid0, gid1 = Int32(mIds[blk, col0]), Int32(mIds[blk, col1])
        lp0, lp1 = gid0//(page//4), gid1//(page//4)
        safe0 = cutlass.min(cutlass.max(lp0, Int32(0)), width-1)
        safe1 = cutlass.min(cutlass.max(lp1, Int32(0)), width-1)
        pp0, pp1 = Int32(mBT[0, page_bias+safe0]), Int32(mBT[0, page_bias+safe1])
        valid0 = Int32((col0<complete)&(gid0>=0)&request_valid&(lp0>=0)&(lp0<width)&(pp0>=0)&(pp0<pages))
        valid1 = Int32((col1<complete)&(gid1>=0)&request_valid&(lp1>=0)&(lp1<width)&(pp1>=0)&(pp1<pages))
        ballot0, ballot1 = _ballot32(valid0), _ballot32(valid1)
        if lane == 0:
            meta[scan_offset+warp] = Int32(cute.arch.popc(ballot0))+Int32(cute.arch.popc(ballot1))
    cute.arch.sync_threads()
    prefix, full = Int32(0), Int32(0)
    for w in cutlass.range_constexpr(8):
        wc = Int32(meta[scan_offset+w])
        prefix += cutlass.select_(warp>w, wc, Int32(0))
        full += wc
    tail_page = tail_id//(page//4)
    tail_safe = cutlass.min(cutlass.max(tail_page, Int32(0)), width-1)
    tail_physical = Int32(mBT[0, page_bias+tail_safe])
    tail_live = Int32((tail_count>0)&(tail_id>=0)&request_valid&(tail_page>=0)&(tail_page<width)&(tail_physical>=0)&(tail_physical<pages))
    total = full+tail_live
    tiles = (total+23)//24
    chunk = (tiles+splits-1)//splits
    begin = (unit%splits)*chunk*24
    count = cutlass.max(cutlass.min(total-begin, chunk*24), Int32(0))
    safe_begin = cutlass.min(begin, cutlass.max(total-1, Int32(0)))
    if tidx < 256:
        lower_lanes = (Int32(1)<<lane)-1
        rank0 = prefix+Int32(cute.arch.popc(ballot0&lower_lanes))
        rank1 = prefix+Int32(cute.arch.popc(ballot0))+Int32(cute.arch.popc(ballot1&lower_lanes))
        if (valid0!=0)&(rank0>=begin)&(rank0<begin+count):
            meta[rank0-begin] = gid0+page_bias*(page//4)
        if (valid1!=0)&(rank1>=begin)&(rank1<begin+count):
            meta[rank1-begin] = gid1+page_bias*(page//4)
    if tidx == 0:
        if (tail_live!=0)&(full>=begin)&(full<begin+count):
            meta[full-begin] = tail_id+page_bias*(page//4)
        if count == 0:
            meta[0] = Int32(0)
        meta[count_offset] = total
        fast_offset = Int32(8) if cutlass.const_expr(local_view) else 2*nblocks
        meta[count_offset+fast_offset] = cutlass.select_(tail_live!=0, full//24, Int32(33))
    if tidx < rb:
        bits = cutlass.min(cutlass.max(full*4-tidx*32, Int32(0)), Int32(32))
        full_mask = cutlass.select_(bits==32, Int32(-1), (Int32(1)<<cutlass.min(bits, Int32(31)))-1)
        tail_mask = ((Int32(1)<<tail_count)-1)<<((full%8)*4)
        meta[rm_offset+tidx] = full_mask|cutlass.select_((tidx==full//8)&(tail_live!=0), tail_mask, Int32(0))
    cute.arch.sync_threads()
    # Bias views to retain C76's row/partition address formulas. All consumed
    # addresses resolve inside the shared allocation; padding is never read.
    if cutlass.const_expr(local_view):
        ws = cute.make_tensor(meta.iterator, cute.make_layout((local_entries,)))
        counts = cute.make_tensor(meta.iterator+count_offset, cute.make_layout((12,)))
        masks = cute.make_tensor(meta.iterator+rm_offset, cute.make_layout((rb,)))
    else:
        ws = cute.make_tensor(meta.iterator-blk*513-safe_begin, cute.make_layout((2052,)))
        counts = cute.make_tensor(meta.iterator+count_offset-blk, cute.make_layout((12,)))
        masks = cute.make_tensor(meta.iterator+rm_offset-blk*rb, cute.make_layout((264,)))
    return ws, counts, masks
