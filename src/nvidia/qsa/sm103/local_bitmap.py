"""Request-local bitmap; emits the unchanged C76 workspace ABI, full first.

A query tile may contain arbitrary request IDs. Reuse a bitmap for one
request, or two adjacent requests when small enough. A bijective permutation
and request interleaving improve KV reuse without hash lookup/probing. Full
groups precede at most BR partial groups. Storage does not scale with batch.
"""
import cutlass
import cutlass.cute as cute
import cutlass.utils as utils
from cutlass.cute.typing import Int32, Int64

from .warp import _shfl32
from . import qsa_sm_ptx


ORDER = "interleaved_full"
LARGE_FACTOR = 4093


class LocalBitmapPreprocess:
    def __init__(self, br, page_size, bt_width, request_span, pdl=False,
                 threads=1024, full_tile_groups=24):
        assert 1 <= br <= 16 and page_size >= 16 and page_size % 16 == 0
        assert full_tile_groups in (24, 32)
        self.full_tile_groups = full_tile_groups
        self.br, self.page = br, page_size
        self.width = request_span // page_size if request_span else bt_width
        self.groups = self.width * (page_size // 4)
        self.scramble = bool(request_span and (bt_width // self.width >= 32 or request_span >= 1048576))
        self.window = 2 if self.scramble and self.groups <= 65536 else 1
        self.bitmap_groups = self.groups * self.window
        import math
        self.factor = LARGE_FACTOR if self.groups > 65536 else 257
        while math.gcd(self.factor, self.groups) != 1:
            self.factor += 2
        self.inverse = pow(self.factor, -1, self.groups)
        self.emit_scan = self.scramble and (self.groups <= 65536 or self.groups & (self.groups - 1) == 0)
        self.threads = threads
        self.wpt = ((self.bitmap_groups + 31) // 32 + threads - 1) // threads
        self.words = self.wpt * threads
        self.maxe = br * 513
        self.rb = 2 * ((self.maxe + 15) // 16)
        self.buckets = (self.maxe + 63) // 64
        self.pdl = pdl
        self.smem = self.words * 8 + br * self.rb * 4 + br * 16 + threads // 32 * 4 + self.buckets * 4 + 16 + br * 4 + br * br * 4
        assert self.smem <= 232448
        assert self.bitmap_groups < (1 << 27)
        assert (self.groups - 1) * self.factor < (1 << 31)

    @cute.jit
    def __call__(self, ids, pos, bt, req, ws, counts, rm, rows: Int32,
                 requests: Int32, pages: Int32, q0: Int32, q1: Int32, stream):
        self.atrex_qsa_prefill_local_bitmap(ids, pos, bt, req, ws, counts, rm,
                                            rows, requests, pages).launch(
            grid=((rows + self.br - 1) // self.br, 1, 1),
            block=(self.threads, 1, 1), stream=stream,
            min_blocks_per_mp=min(2048 // self.threads, 232448 // self.smem, 3))

    @cute.jit
    def _encode(self, group):
        if cutlass.const_expr(self.scramble):
            return ((group % self.groups * self.factor) % self.groups) * self.window + group // self.groups
        return group

    @cute.jit
    def _decode(self, group):
        if cutlass.const_expr(self.scramble):
            return Int32((Int64(group // self.window) * self.inverse) % self.groups) + (group % self.window) * self.groups
        return group

    @cute.jit
    def _index(self, word):
        # Keep logical ascending ranks while spreading power-of-two scan
        # strides over banks. This permutes only the low five word bits.
        if cutlass.const_expr(self.wpt >= 4 and self.wpt & (self.wpt - 1) == 0):
            return word ^ ((word >> 5) & 31)
        return word

    @cute.jit
    def _alive(self, bt, request, page, pages):
        phys = Int32(-1)
        if (page >= 0) and (page < self.width * self.window):
            phys = Int32(bt[0, request * self.width + page])
        return (phys >= 0) and (phys < pages)

    @cute.kernel
    def atrex_qsa_prefill_local_bitmap(self, ids, pos, bt, req, ws, counts, rm,
                                      rows: Int32, requests: Int32, pages: Int32):
        tid, _, _ = cute.arch.thread_idx()
        blk, _, _ = cute.arch.block_idx()
        br, thr, wpt = self.br, self.threads, self.wpt
        rb, maxe, nw = self.rb, self.maxe, thr // 32
        lane, warp = tid % 32, tid // 32
        nitems = (maxe + thr - 1) // thr
        if cutlass.const_expr(self.pdl):
            cute.arch.griddepcontrol_launch_dependents()

        @cute.struct
        class Storage:
            words: cute.struct.MemRange[Int64, self.words]
            masks: cute.struct.MemRange[Int32, br * rb]
            row_req: cute.struct.MemRange[Int32, br]
            row_complete: cute.struct.MemRange[Int32, br]
            row_tail: cute.struct.MemRange[Int32, br]
            row_nibble: cute.struct.MemRange[Int32, br]
            warp_sum: cute.struct.MemRange[Int32, nw]
            bucket: cute.struct.MemRange[Int32, self.buckets]
            total: Int32
            partial_keys: cute.struct.MemRange[Int32, br]
            partial_masks: cute.struct.MemRange[Int32, br * br]
            partial_count: Int32

        storage = utils.SmemAllocator().allocate(Storage)
        pairs = storage.words.data_ptr()
        words = cute.make_ptr(Int32, pairs.toint(), cute.AddressSpace.smem, assumed_align=8)
        masks = storage.masks.data_ptr()
        sums = storage.warp_sum.data_ptr()
        buckets = storage.bucket.data_ptr()
        own_req = requests
        if lane < br:
            if blk * br + lane < rows:
                value = Int32(req[blk * br + lane])
                if (value >= 0) and (value < requests):
                    own_req = value
        current = own_req
        for d in cutlass.range_constexpr(5):
            current = cutlass.min(current, _shfl32(current, lane ^ (1 << d)))
        if tid < br:
            complete, tail, nibble = Int32(0), Int32(-1), Int32(0)
            if own_req < requests:
                visible = Int64(pos[blk * br + tid]) + Int64(1)
                complete = Int32(cutlass.min(cutlass.max(visible // 4, Int64(0)), Int64(512)))
                if (visible > 0) and (visible // 4 < self.groups):
                    tail = Int32(visible // 4)
                    nibble = (Int32(1) << Int32(visible % 4)) - 1
            (storage.row_req.data_ptr() + tid).store(own_req)
            (storage.row_complete.data_ptr() + tid).store(complete)
            (storage.row_tail.data_ptr() + tid).store(tail)
            (storage.row_nibble.data_ptr() + tid).store(nibble)
        for i in cutlass.range_constexpr((br * rb + thr - 1) // thr):
            j = tid + i * thr
            if j < br * rb:
                (masks + j).store(Int32(0))
        for i in cutlass.range_constexpr((br * br + thr - 1) // thr):
            j = tid + i * thr
            if j < br * br:
                (storage.partial_masks.data_ptr() + j).store(Int32(0))
        cute.arch.sync_threads()
        # At most BR distinct partial keys. Sort/deduplicate them once; the
        # main bitmap then ranks only full groups, without a second bitmap.
        key = Int32(0x7FFFFFFF)
        tail_key, tail_req, tail_nibble = Int32(-1), Int32(0), Int32(0)
        pcount, prank, other_key, other_unique = Int32(0), Int32(0), Int32(0), Int32(0)
        unique = Int32(0) == 1
        if warp == 0:
            if lane < br:
                tail_req = (storage.row_req.data_ptr() + lane).load()
                tail_key = (storage.row_tail.data_ptr() + lane).load()
                tail_nibble = (storage.row_nibble.data_ptr() + lane).load()
                if (tail_req < requests) and (tail_key >= 0) and (tail_nibble != 0):
                    if self._alive(bt, tail_req, tail_key // (self.page // 4), pages):
                        key = tail_req * self.groups + tail_key
            unique = key != Int32(0x7FFFFFFF)
            for r in cutlass.range_constexpr(br):
                other_key = _shfl32(key, Int32(r))
                unique = unique and not ((r < lane) and (other_key == key))
            pcount, prank = Int32(0), Int32(0)
            for r in cutlass.range_constexpr(br):
                other_key = _shfl32(key, Int32(r))
                other_unique = _shfl32(Int32(unique), Int32(r))
                pcount += other_unique
                prank += cutlass.select_(other_key < key, other_unique, Int32(0))
            if unique:
                (storage.partial_keys.data_ptr() + prank).store(key)
            if lane == 0:
                storage.partial_count.ptr.store(pcount)
        cute.arch.sync_threads()
        partial_count = storage.partial_count.ptr.load()
        offset = Int32(0)
        # Stable scalar joins across the data-dependent request loop.
        row_req = Int32(0)
        gid = Int32(0)
        nibble = Int32(0)
        packed = Int32(0)
        item = Int32(0)
        rr = Int32(0)
        col = Int32(0)
        w = Int32(0)
        b = Int32(0)
        lo = Int32(0)
        hi = Int32(0)
        split = Int32(0)
        field = Int32(0)
        mask = Int32(0)
        subtotal = Int32(0)
        incl = Int32(0)
        delta = Int32(0)
        other = Int32(0)
        total = Int32(0)
        prefix = Int32(0)
        run = Int32(0)
        next_bucket = Int32(0)
        count = Int32(0)
        rank = Int32(0)
        shift = Int32(0)
        bucket = Int32(0)
        mid = Int32(0)
        bit = Int32(0)
        following = Int32(0)
        j = Int32(0)
        alive_lo = Int32(0) == 1
        alive_hi = Int32(0) == 1
        pair = Int64(0)
        pk = Int32(0)
        global_gid = Int32(0)
        valid_page = Int32(0) == 1
        remaining = Int32(0)
        emit_rank = Int32(0)
        lowest = Int32(0)
        while current < requests:
            for i in cutlass.range_constexpr(wpt):
                (pairs + tid + i * thr).store(Int64(0))
            cute.arch.sync_threads()
            carry = [Int32(0) for _ in range(nitems)]
            for i in cutlass.range_constexpr(nitems):
                item = tid + i * thr
                rr, col = item // 512, item % 512
                if cutlass.const_expr((i + 1) * thr > br * 512):
                    if item >= br * 512:
                        rr, col = item - br * 512, Int32(512)
                packed = Int32(0)
                if rr < br:
                    row_req = (storage.row_req.data_ptr() + rr).load()
                    if (row_req >= current) and (row_req < current + self.window):
                        gid, nibble = Int32(-1), Int32(15)
                        if item < br * 512:
                            if col < (storage.row_complete.data_ptr() + rr).load():
                                gid = Int32(ids[blk * br + rr, col])
                        else:
                            gid = (storage.row_tail.data_ptr() + rr).load()
                            nibble = (storage.row_nibble.data_ptr() + rr).load()
                        if (gid >= 0) and (gid < self.groups) and (nibble != 0):
                            gid += (row_req - current) * self.groups
                            valid_page = True
                            # A 128-token bitmap word can span up to eight
                            # small pages. Validate each selected four-token
                            # block before insertion instead of masking only
                            # the word's first and last pages below.
                            if cutlass.const_expr(self.scramble or self.page < 128):
                                valid_page = self._alive(bt, current, gid // (self.page // 4), pages)
                            if valid_page:
                                gid = self._encode(gid)
                                cute.arch.atomic_or(words + 2 * self._index(gid >> 5), Int32(1) << (gid & 31), sem="relaxed", scope="cta")
                                packed = (gid << 4) | nibble
                carry[i] = packed
            cute.arch.sync_threads()

            if tid < partial_count:
                pk = (storage.partial_keys.data_ptr() + tid).load() - current * self.groups
                if (pk >= 0) and (pk < self.bitmap_groups):
                    pk = self._encode(pk)
                    cute.arch.atomic_and(words + 2 * self._index(pk >> 5), ~(Int32(1) << (pk & 31)), sem="relaxed", scope="cta")
            cute.arch.sync_threads()
            bits = [Int32(0) for _ in range(wpt)]
            subtotal = Int32(0)
            for i in cutlass.range_constexpr(wpt):
                w = tid * wpt + i
                b = (words + 2 * self._index(w)).load()
                if cutlass.const_expr(not self.scramble and self.page >= 128):
                    if b != 0:
                        lo = w * 128 // self.page
                        hi = (w * 128 + 127) // self.page
                        alive_lo = self._alive(bt, current, lo, pages)
                        if cutlass.const_expr(self.page % 128 == 0):
                            b = cutlass.select_(alive_lo, b, Int32(0))
                        else:
                            alive_hi = self._alive(bt, current, hi, pages)
                            split = (lo + 1) * (self.page // 4) - w * 32
                            field = (Int32(1) << cutlass.min(split, Int32(31))) - 1
                            mask = cutlass.select_(alive_lo, field, Int32(0)) | cutlass.select_(alive_hi, ~field, Int32(0))
                            mask = cutlass.select_(hi == lo, cutlass.select_(alive_lo, Int32(-1), Int32(0)), mask)
                            b = b & mask
                (words + 2 * self._index(w)).store(b)
                bits[i] = b
                subtotal += Int32(cute.arch.popc(b))
            incl = subtotal
            for d in cutlass.range_constexpr(5):
                delta = 1 << d
                other = _shfl32(incl, cutlass.max(lane - delta, Int32(0)))
                incl += cutlass.select_(lane >= delta, other, Int32(0))
            if lane == 31:
                (sums + warp).store(incl)
            cute.arch.sync_threads()
            if warp == 0:
                total = Int32(0)
                if lane < nw:
                    total = (sums + lane).load()
                prefix = total
                for d in cutlass.range_constexpr(5):
                    delta = 1 << d
                    other = _shfl32(prefix, cutlass.max(lane - delta, Int32(0)))
                    prefix += cutlass.select_(lane >= delta, other, Int32(0))
                if lane < nw:
                    (sums + lane).store(prefix - total)
                if lane == 31:
                    storage.total.ptr.store(prefix)
            cute.arch.sync_threads()
            run = incl - subtotal + (sums + warp).load()
            for i in cutlass.range_constexpr(wpt):
                w = tid * wpt + i
                (words + 2 * self._index(w) + 1).store(run)
                count = Int32(cute.arch.popc(bits[i]))
                if cutlass.const_expr(self.emit_scan):
                    # A bijective bitmap permutation spreads selected groups
                    # over words. Emit directly from the prefix scan, avoiding
                    # serial rank-to-word binary searches for long KV.
                    remaining = bits[i]
                    emit_rank = offset + run
                    while remaining != 0:
                        lowest = remaining & (Int32(0) - remaining)
                        bit = Int32(cute.arch.popc(lowest - 1))
                        ws[blk * maxe + emit_rank] = current * self.groups + self._decode(w * 32 + bit)
                        emit_rank += 1
                        remaining = remaining & (remaining - 1)
                else:
                    next_bucket = ((run + 63) >> 6) << 6
                    if next_bucket < run + count:
                        (buckets + (next_bucket >> 6)).store(w)
                run += count
            cute.arch.sync_threads()

            for i in cutlass.range_constexpr(nitems):
                packed = carry[i]
                if packed != 0:
                    gid, nibble = packed >> 4, packed & 15
                    pair = (pairs + self._index(gid >> 5)).load()
                    b, prefix = Int32(pair), Int32(pair >> 32)
                    shift = gid & 31
                    item = tid + i * thr
                    rr = item // 512
                    if cutlass.const_expr((i + 1) * thr > br * 512):
                        if item >= br * 512:
                            rr = item - br * 512
                    if ((b >> shift) & 1) != 0:
                        rank = offset + prefix + Int32(cute.arch.popc(b & ((Int32(1) << shift) - 1)))
                        cute.arch.atomic_or(masks + rr * rb + rank // 8, nibble << (4 * (rank & 7)), sem="relaxed", scope="cta")
                    else:
                        global_gid = current * self.groups + self._decode(gid)
                        for j in cutlass.range(partial_count):
                            if global_gid == (storage.partial_keys.data_ptr() + j).load():
                                cute.arch.atomic_or(storage.partial_masks.data_ptr() + j * br + rr, nibble, sem="relaxed", scope="cta")
            total = storage.total.ptr.load()
            if cutlass.const_expr(not self.emit_scan):
                for i in cutlass.range_constexpr(nitems):
                    rank = tid + i * thr
                    if rank < total:
                        bucket = rank >> 6
                        lo = (buckets + bucket).load()
                        hi = Int32(self.words - 1)
                        if ((bucket + 1) << 6) < total:
                            hi = (buckets + bucket + 1).load()
                        while lo < hi:
                            mid = (lo + hi + 1) >> 1
                            prefix = (words + 2 * self._index(mid) + 1).load()
                            if prefix <= rank:
                                lo = mid
                            else:
                                hi = mid - 1
                        pair = (pairs + self._index(lo)).load()
                        bit = qsa_sm_ptx.fns_b32(Int32(pair), Int32(0), rank - Int32(pair >> 32) + 1)
                        ws[blk * maxe + offset + rank] = current * self.groups + self._decode(lo * 32 + bit)
            offset += total
            following = cutlass.select_(own_req >= current + self.window, own_req, requests)
            for d in cutlass.range_constexpr(5):
                following = cutlass.min(following, _shfl32(following, lane ^ (1 << d)))
            current = following
            cute.arch.sync_threads()
        if tid < partial_count:
            ws[blk * maxe + offset + tid] = (storage.partial_keys.data_ptr() + tid).load()
        if tid < br * partial_count:
            prank, rr = tid // br, tid % br
            nibble = (storage.partial_masks.data_ptr() + tid).load()
            rank = offset + prank
            cute.arch.atomic_or(masks + rr * rb + rank // 8, nibble << (4 * (rank & 7)), sem="relaxed", scope="cta")
        cute.arch.sync_threads()
        if tid == 0:
            counts[blk] = offset + partial_count
            counts[2 * ((rows + br - 1) // br) + blk] = offset // self.full_tile_groups
        for i in cutlass.range_constexpr((br * rb + thr - 1) // thr):
            j = tid + i * thr
            if j < br * rb:
                rm[blk * br * rb + j] = (masks + j).load()
