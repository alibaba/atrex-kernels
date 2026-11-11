"""Shared device primitives for the SM100/SM103 QSA kernel families."""
from __future__ import annotations

import cutlass
import cutlass.cute as cute
from cutlass.cute.typing import Float32, Int32, Int64

@cute.jit
def _elem_pointer(x: cute.Tensor, coord) -> cute.Pointer:
    return x.iterator + cute.crd2idx(coord, x.layout)


@cute.jit
def _shfl(value: Int64, src_lane: Int32, width: cutlass.Constexpr) -> Int64:
    mask_and_clamp = ((32 - width) << 8) | 31
    return Int64(cute.arch.shuffle_sync(value, src_lane, mask_and_clamp=mask_and_clamp))


@cute.jit
def _shfl32(value: Int32, src_lane: Int32) -> Int32:
    """Broadcast a 32-bit word within a warp before widening its address."""
    return Int32(cute.arch.shuffle_sync(value, src_lane, mask_and_clamp=31))


class DeviceHelpers:
    """Warp reductions, metadata masks and unit-to-request mapping."""


    @cute.jit
    def _colbits_pack(self, raw, count, sh_lo32, ebase, etile, nw):
        """Pack pre-loaded raw nibble words into this thread's validity bits.

        Entry ``ebase + ee`` contributes its 4 lane nibbles at bits ``4*ee``
        of the returned word list; entries at/after ``count`` contribute 0.
        """
        nlive = cutlass.min(Int32(etile), count - ebase)
        ws = []
        for w in cutlass.range_constexpr(nw):
            acc = Int32(0)
            for ee in cutlass.range_constexpr(8):
                k = w * 8 + ee
                nib = (raw[k] >> sh_lo32) & Int32(0xF)
                nib = cutlass.select_(k < nlive, nib, Int32(0))
                acc = acc | (nib << (ee * 4))
            ws.append(acc)
        return ws

    @cute.jit
    def _colbits_raw(self, mWS, wbase, ebase, nw):
        """Issue the ``nw`` raw nibble-word LDGs for the tile at ``ebase``.

        Returned as a register list; the caller defers the shift/OR packing
        to a later iteration so the global-load latency overlaps real work.
        ``ebase`` must already be clamped into this unit's entry range (dead
        tails re-read the last live tile's words: harmless, they are masked).
        """
        raw = []
        for ee in cutlass.range_constexpr(nw * 8):
            raw.append(Int32(mWS[wbase + (ebase + ee) * 4]))
        return raw

    @cute.jit
    def _flatten(self, sX: cute.Tensor, transpose: cutlass.Constexpr) -> cute.Tensor:
        lay = cute.make_layout(
            (sX.shape[0][0], (sX.shape[0][1], sX.shape[2])),
            stride=(sX.stride[0][0], (sX.stride[0][1], sX.stride[2])))
        t = cute.make_tensor(sX.iterator, lay)
        if cutlass.const_expr(transpose):
            t = cute.make_tensor(t.iterator, cute.select(t.layout, mode=[1, 0]))
        return t

    @cute.jit
    def _mwait(self, bar, ph):
        """Wait for an mbarrier phase using the selected lowering."""
        if cutlass.const_expr(self.spin == 0):
            cute.arch.mbarrier_wait(bar, ph)
        elif cutlass.const_expr(self.spin == 1):
            ok = cute.arch.mbarrier_try_wait(bar, ph)
            while not ok:
                ok = cute.arch.mbarrier_try_wait(bar, ph)
        else:
            ok = cute.arch.mbarrier_test_wait(bar, ph)
            while not ok:
                ok = cute.arch.mbarrier_test_wait(bar, ph)

    @cute.jit
    def _sum_packed(self, rS, n: cutlass.Constexpr):
        """Sum FP32 registers with a balanced tree of packed additions.

        Reassociation preserves the algorithm but can change FP32 rounding."""
        pr = []
        for i in cutlass.range_constexpr(n // 2):
            pr.append((Float32(rS[2 * i]), Float32(rS[2 * i + 1])))
        for lvl in cutlass.range_constexpr(8):
            if cutlass.const_expr(len(pr) > 1):
                nxt = []
                for i in cutlass.range_constexpr(len(pr) // 2):
                    nxt.append(cute.arch.add_packed_f32x2(pr[2 * i],
                                                          pr[2 * i + 1]))
                if cutlass.const_expr(len(pr) % 2 == 1):
                    nxt.append(pr[-1])
                pr = nxt
        a, b = pr[0]
        return a + b

    @cute.jit
    def _sum_tree(self, rS, n: cutlass.Constexpr):
        """Sum FP32 registers with a balanced scalar tree.

        Reassociation shortens the dependency chain and can change rounding."""
        vals = []
        for i in cutlass.range_constexpr(n):
            vals.append(Float32(rS[i]))
        for _ in cutlass.range_constexpr(8):
            if cutlass.const_expr(len(vals) > 1):
                m = len(vals) // 2
                nxt = []
                for i in cutlass.range_constexpr(m):
                    nxt.append(vals[2 * i] + vals[2 * i + 1])
                if cutlass.const_expr(len(vals) % 2 == 1):
                    nxt.append(vals[-1])
                vals = nxt
        return vals[0]

    @cute.jit
    def _unit_blk(self, u, nblocks, mCount):
        """Map a split unit to a query block, respecting optional stored ordering."""
        u = u // self.kv_splits
        if cutlass.const_expr(self.ordered):
            index = u % nblocks if cutlass.const_expr(self.kvmajor) else u // self.hkv
            index = cutlass.min(cutlass.max(index, Int32(0)), nblocks - 1)
            return Int32(mCount[nblocks + index])
        if cutlass.const_expr(self.kvmajor):
            return nblocks - 1 - (u % nblocks)
        if cutlass.const_expr(self.hkv == 1):
            return nblocks - 1 - u
        if cutlass.const_expr(self.hkv == 2):
            return nblocks - 1 - (u >> 1)
        return nblocks - 1 - (u // self.hkv)

    @cute.jit
    def _unit_kvh(self, u, nblocks):
        """Map a split unit to a KV head; adjacent tickets share blocks by default."""
        u = u // self.kv_splits
        if cutlass.const_expr(self.kvmajor):
            return u // nblocks
        if cutlass.const_expr(self.hkv == 1):
            return Int32(0)
        if cutlass.const_expr(self.hkv == 2):
            return u & 1
        return u % self.hkv
