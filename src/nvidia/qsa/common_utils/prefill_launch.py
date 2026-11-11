"""Tensor-free eager prefill launch plans; scratch belongs to each invocation."""
from dataclasses import dataclass

import torch

from ..sm103.schedule import atrex_qsa_reset_ticket, schedule


def launch_key(signature, sm_scale=None, v_scale=1.0, kvmajor=None):
    # Only already validated ordinary scalars enter the fast cache. Keeping
    # types in the key prevents True from aliasing a previously accepted 1.
    # Unusual scalar objects go through the runtime's existing validation.
    if (signature is None or type(sm_scale) not in (type(None), int, float)
            or type(v_scale) not in (int, float)):
        return None
    return (signature, sm_scale, type(sm_scale), v_scale, type(v_scale), kvmajor)


def remember(cache, key, plan):
    if key is not None:
        if len(cache) >= 128:
            cache.clear()
        cache[key] = plan


def arena_sizes(ws, count, mask):
    # Preserve the caching allocator's 256-byte base alignment for every
    # buffer, including warp-wide metadata loads. Cache these padded extents.
    return tuple((size + 63) // 64 * 64 for size in (ws, count, mask, 1))


def scratch(device, sizes):
    # split creates aligned views in one Torch call. The caching allocator
    # owns the arena on this stream; nothing is retained between calls.
    return torch.empty(sum(sizes), dtype=torch.int32, device=device).split(sizes)


@dataclass(frozen=True)
class PrefillLaunch:
    pre: object
    main: object
    sizes: tuple
    pre_tail: tuple
    main_tail: tuple
    nblocks: int
    static_short: bool
    ordered: bool

    def __call__(self, q, k, v, ids, pos, table, req, gate, out=None,
                 workspace=None):
        if torch.cuda.is_current_stream_capturing():
            raise NotImplementedError('QSA prefill does not support CUDA graph capture')
        if out is None:
            out = torch.empty(q.shape, dtype=torch.bfloat16, device=q.device)
        ws, count, mask, ticket = scratch(q.device, self.sizes) if workspace is None else workspace
        if not self.static_short:
            atrex_qsa_reset_ticket[(1,)](ticket, num_warps=1)
        table = table.view(1, -1)
        self.pre(ids, pos, table, req, ws, count, mask, *self.pre_tail)
        if self.ordered:
            schedule(count, self.nblocks, True)
        self.main(q, k, v, ws, count, mask, ticket, table, gate, out,
                  *self.main_tail)
        return out
