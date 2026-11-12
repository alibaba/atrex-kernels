"""Merge FP32 partition accumulators before BF16 rounding and gate."""
import triton
import triton.language as tl

MERGE_BLOCK_D = 64
MERGE_WARPS = 1

def merge_block_d(rows, heads, splits):
    return 128 if splits == 2 and rows * heads >= 1024 else MERGE_BLOCK_D


@triton.jit
def atrex_mtp_merge(Partial, Stats, Gate, Out, H: tl.constexpr,
                    SPLITS: tl.constexpr, BS: tl.constexpr, D: tl.constexpr,
                    BD: tl.constexpr):
    rh = tl.program_id(0)
    row, head = rh // H, rh % H
    s = tl.arange(0, BS)
    d = tl.program_id(1)*BD+tl.arange(0, BD)
    base = ((row*SPLITS+s)*H+head)
    maxima = tl.load(Stats+base*2, s<SPLITS, other=-float('inf'))
    sums = tl.load(Stats+base*2+1, s<SPLITS, other=0.)
    maximum = tl.max(maxima, 0)
    maximum = tl.where(maximum == -float('inf'), 0., maximum)
    weights = tl.exp2(maxima-maximum)
    denominator = tl.sum(sums*weights, 0)
    partial = tl.load(Partial+base[:, None]*D+d[None, :], s[:, None]<SPLITS, other=0.)
    numerator = tl.sum(partial*weights[:, None], 0)
    result = tl.where(denominator>0, numerator/denominator, 0.)
    rounded = result.to(tl.bfloat16).to(tl.float32)
    gate = tl.load(Gate+rh*D+d).to(tl.float32)
    output = rounded/(1.+tl.exp(-gate))
    tl.store(Out+rh*D+d, output.to(tl.bfloat16))


def merge(partial, stats, gate, out, splits, wide=False):
    block_d = 256 if wide else merge_block_d(gate.shape[0], gate.shape[1], splits)
    atrex_mtp_merge[(gate.shape[0]*gate.shape[1], gate.shape[2]//block_d)](
        partial, stats, gate, out, gate.shape[1], splits,
        triton.next_power_of_2(splits), gate.shape[2], block_d, num_warps=MERGE_WARPS)
