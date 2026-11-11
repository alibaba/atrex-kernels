"""Shared host planning and CuTe compilation for SM100/SM103 QSA."""
from __future__ import annotations

import torch
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

_BR_CACHE: dict = {}
_SMS_CACHE: dict = {}


def _sms_count(device) -> int:
    """Cache immutable device properties without retaining caller tensors."""
    k = str(device)
    v = _SMS_CACHE.get(k)
    if v is None:
        v = torch.cuda.get_device_properties(device).multi_processor_count
        _SMS_CACHE[k] = v
    return v


def _br_lane_cap(group: int) -> int:
    """Limit real query/head lanes to one 128-lane MMA tile."""
    return max(1, min(128 // max(1, group), 16))


def _select_br(rows: int, hkv: int, sms: int, br_cap: int,
               pre_slots: int = 2) -> int:
    """Minimize CTA waves, then preprocess waves, then row-block size."""
    sms = max(1, int(sms))
    slots = max(1, int(pre_slots) * sms)
    ck = (int(rows), int(hkv), sms, int(br_cap), slots)
    hit = _BR_CACHE.get(ck)
    if hit is not None:
        return hit
    best = None
    for br in range(1, int(br_cap) + 1):
        nblocks = -(-int(rows) // br)
        w_main = -(-(nblocks * int(hkv)) // sms)
        w_pre = -(-nblocks // slots)
        key = (w_main, w_pre, br)
        if best is None or key < best[0]:
            best = (key, br)
    _BR_CACHE[ck] = best[1]
    return best[1]


def _cute_stream() -> "cute.runtime.Stream":
    import cuda.bindings.driver as cuda_drv
    return cuda_drv.CUstream(torch.cuda.current_stream().cuda_stream)


def _compile(fn, args, tensor_specs):
    """Build descriptors during JIT; launches consume live Torch tensors.

    TVM-FFI keeps caller buffers out of the compilation cache and allows each
    layer to use its own tensor addresses after configuration warmup.
    """
    compile_args = list(args)
    for i, (align, leading_dim) in enumerate(tensor_specs):
        compile_args[i] = from_dlpack(
            args[i], assumed_align=align, enable_tvm_ffi=True
        ).mark_layout_dynamic(leading_dim=leading_dim)
    return cute.compile(fn, *compile_args, options="--enable-tvm-ffi")
