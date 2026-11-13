"""Combine membership for equal per-column keys before shared hash insertion."""
import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Int32, Int64
from cutlass._mlir.dialects import llvm


@cute.jit
def _cas(pointer, old: Int32, new: Int32):
    address = Int32(Int64(pointer.toint()) & 0x3ffff)
    return Int32(llvm.inline_asm(Int32.mlir_type,
        [address.ir_value(), old.ir_value(), new.ir_value()],
        'atom.shared.cas.b32 $0, [$1], $2, $3;', '=r,r,r,r',
        has_side_effects=True, is_align_stack=False, asm_dialect=llvm.AsmDialect.AD_ATT))


@cute.jit
def insert_query_columns(ids, pos, req, table, cache, scratch, tidx, block,
                         page: cutlass.Constexpr, request_span: cutlass.Constexpr,
                         br: cutlass.Constexpr, hash_mix: cutlass.Constexpr,
                         deterministic: cutlass.Constexpr = False,
                         hash_shift: cutlass.Constexpr = 6):
    slots = 1024 * br
    if cutlass.const_expr(deterministic):
        saved_keys = cute.make_rmem_tensor((3 * br,), Int32)
        saved_masks = cute.make_rmem_tensor((3 * br,), Int32)
        saved_keys.fill(Int32(-1))
        saved_masks.fill(Int32(0))
    width = Int32(table.shape[1])
    requests = Int32(1)
    if cutlass.const_expr(request_span > 0):
        width = Int32(request_span // page)
        requests = Int32(table.shape[1]) // width
    for part in cutlass.range_constexpr(3):
        column = tidx + part * 256
        if (tidx < 256) & (column <= 512):
            keys = cute.make_rmem_tensor((br,), Int32)
            masks = cute.make_rmem_tensor((br,), Int32)
            for row in cutlass.range_constexpr(br):
                query = block * br + row
                visible = Int64(pos[query]) + Int64(1)
                complete = cutlass.min(cutlass.max(visible // 4, Int64(0)), Int64(512))
                gid = Int32(visible // 4)
                mask = (Int32(1) << Int32(visible % 4)) - 1
                valid = (visible > 0) & (mask != 0)
                if column < 512:
                    gid = Int32(ids[query, column])
                    mask = Int32(15)
                    valid = column < complete
                request = Int32(req[query])
                safe_request = cutlass.min(cutlass.max(request, Int32(0)), requests - 1)
                logical_page = gid // (page // 4)
                safe_page = cutlass.min(cutlass.max(logical_page, Int32(0)), width - 1)
                physical = Int32(table[0, safe_request * width + safe_page])
                valid = valid & (gid >= 0) & (logical_page >= 0) & (logical_page < width)
                valid = valid & (request >= 0) & (request < requests) & (physical >= 0) & (physical < Int32(cache.shape[0]))
                key = gid + safe_request * width * (page // 4)
                if column == 512:
                    key = key | Int32(1 << 29)
                keys[row] = cutlass.select_(valid, key, Int32(-1))
                masks[row] = mask << (row * 4)
            for row in cutlass.range_constexpr(br):
                key = Int32(keys[row])
                first = key >= 0
                member = Int32(masks[row])
                for previous in cutlass.range_constexpr(row):
                    first = first & (Int32(keys[previous]) != key)
                for following in cutlass.range_constexpr(row + 1, br):
                    member = member | cutlass.select_(Int32(keys[following]) == key, Int32(masks[following]), Int32(0))
                if first:
                    slot = key & (slots - 1)
                    if cutlass.const_expr(hash_mix):
                        slot = ((key ^ (key >> hash_shift)) * Int32(-1640531535)) & (slots - 1)
                    if cutlass.const_expr(deterministic):
                        # Ordered displacement gives every independent KV split
                        # the same entries. Attach membership after keys settle.
                        saved_keys[part * br + row] = key
                        saved_masks[part * br + row] = member
                        current = key
                        done = Int32(0)
                        while done == 0:
                            previous = Int32(cute.arch.atomic_min(
                                scratch.iterator + slot, current, sem='relaxed', scope='cta'))
                            done = Int32((previous == Int32(2147483647)) | (previous == current))
                            current = cutlass.max(previous, current)
                            slot = (slot + 1) & (slots - 1)
                    else:
                        previous = _cas(scratch.iterator + slot, Int32(-1), key)
                        while (previous != Int32(-1)) & (previous != key):
                            slot = (slot + 1) & (slots - 1)
                            previous = _cas(scratch.iterator + slot, Int32(-1), key)
                        cute.arch.atomic_or(scratch.iterator + slots + slot, member,
                                            sem='relaxed', scope='cta')
    if cutlass.const_expr(deterministic):
        cute.arch.sync_threads()
        for item in cutlass.range_constexpr(3 * br):
            key = Int32(saved_keys[item])
            if (tidx < 256) & (key >= 0):
                slot = key & (slots - 1)
                if cutlass.const_expr(hash_mix):
                    slot = ((key ^ (key >> hash_shift)) * Int32(-1640531535)) & (slots - 1)
                while Int32(scratch[slot]) != key:
                    slot = (slot + 1) & (slots - 1)
                cute.arch.atomic_or(scratch.iterator + slots + slot, Int32(saved_masks[item]),
                                    sem='relaxed', scope='cta')
