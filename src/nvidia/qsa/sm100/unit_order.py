"""Order independent CTAs by current union size, within graph timing."""
import torch
import triton
import triton.language as tl

@triton.jit
def atrex_qsa_unit_order(Count, Ticket, NB: tl.constexpr, HKV: tl.constexpr,
                         BLOCK: tl.constexpr):
    b = tl.arange(0, BLOCK)
    score = tl.load(Count + b, b < NB, other=0).to(tl.int64)
    # Preserve the existing reversed block order for ties; padding sorts last.
    key = tl.where(b < NB, tl.maximum(score, 0) * 1024 + b, -1)
    ordered = tl.sort(key, descending=True)
    block = ordered % 1024
    for h in tl.static_range(HKV):
        tl.store(Ticket + b * HKV + h, (NB - 1 - block) * HKV + h, b < NB)

def work_order(count, blocks, hkv):
    ticket = torch.empty((blocks * hkv,), device=count.device, dtype=torch.int32)
    atrex_qsa_unit_order[(1,)](count, ticket, blocks, hkv,
                              triton.next_power_of_2(blocks), num_warps=4)
    return ticket
