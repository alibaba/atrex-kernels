"""Warp exchange used by request-local bitmap preprocessing."""
import cutlass.cute as cute
from cutlass.cute.typing import Int32


@cute.jit
def _shfl32(value: Int32, src_lane: Int32) -> Int32:
    return Int32(cute.arch.shuffle_sync(value, src_lane, mask_and_clamp=31))
