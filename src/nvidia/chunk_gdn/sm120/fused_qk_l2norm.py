"""Fused Q/K bf16 L2 normalization for SM120 Chunk-GDN prefill."""

from __future__ import annotations

import contextlib
import functools
import hashlib
import threading

import torch
import cutlass
import cutlass.cute as cute
import cutlass.cutlass_dsl.cutlass as _cutlass_dsl
from cutlass.cute.runtime import from_dlpack


@functools.lru_cache(maxsize=1)
def _stable_cutlass_dsl_version_hash():
    return hashlib.sha256(b"atrex-gdn-sm120-fused-qk-l2norm-v4-dynamic-layout")


@contextlib.contextmanager
def _stable_cutlass_dsl_version():
    """Scope the legacy CUTLASS version workaround to this compilation."""
    original = _cutlass_dsl.CutlassBaseDSL.get_version

    def stable_version(_self):
        return _stable_cutlass_dsl_version_hash()

    _cutlass_dsl.CutlassBaseDSL.get_version = stable_version
    try:
        yield
    finally:
        _cutlass_dsl.CutlassBaseDSL.get_version = original

K_DIM = 128
VALUES_PER_LANE = 4
NUM_WARPS = 4
NUM_THREADS = NUM_WARPS * 32
_SM120A_COMPILE_OPTIONS = (cute.GPUArch("sm_120a"),)


@cute.kernel
def atrex_gdn_fused_qk_l2norm_bf16_kernel(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mQOut: cute.Tensor,
    mKOut: cute.Tensor,
    rows: cutlass.Int32,
):
    """Normalize one contiguous ``[rows, 128]`` Q/K pair per warp."""
    tidx, _, _ = cute.arch.thread_idx()
    lane_id = tidx % 32
    warp_id = tidx // 32
    row = cute.arch.block_idx()[0] * NUM_WARPS + warp_id

    if row < rows:
        q_bf16 = cute.make_rmem_tensor(
            cute.make_layout((VALUES_PER_LANE,), stride=(1,)),
            cutlass.BFloat16,
        )
        k_bf16 = cute.make_rmem_tensor(
            cute.make_layout((VALUES_PER_LANE,), stride=(1,)),
            cutlass.BFloat16,
        )
        q_tile = cute.local_tile(
            mQ,
            (1, VALUES_PER_LANE),
            (row, lane_id),
        )
        k_tile = cute.local_tile(
            mK,
            (1, VALUES_PER_LANE),
            (row, lane_id),
        )
        cute.autovec_copy(q_tile, q_bf16)
        cute.autovec_copy(k_tile, k_bf16)

        q_fp32 = cute.make_rmem_tensor(
            cute.make_layout((VALUES_PER_LANE,), stride=(1,)),
            cutlass.Float32,
        )
        k_fp32 = cute.make_rmem_tensor(
            cute.make_layout((VALUES_PER_LANE,), stride=(1,)),
            cutlass.Float32,
        )
        sum_q = cutlass.Float32(0.0)
        sum_k = cutlass.Float32(0.0)
        for ki in cutlass.range_constexpr(VALUES_PER_LANE):
            q_fp32[ki] = cutlass.Float32(q_bf16[ki])
            k_fp32[ki] = cutlass.Float32(k_bf16[ki])
            sum_q = sum_q + q_fp32[ki] * q_fp32[ki]
            sum_k = sum_k + k_fp32[ki] * k_fp32[ki]

        for offset in [16, 8, 4, 2, 1]:
            sum_q = sum_q + cute.arch.shuffle_sync_bfly(
                sum_q, offset=offset, mask=-1, mask_and_clamp=31,
            )
            sum_k = sum_k + cute.arch.shuffle_sync_bfly(
                sum_k, offset=offset, mask=-1, mask_and_clamp=31,
            )
        inv_q = cute.rsqrt(sum_q + cutlass.Float32(1e-6))
        inv_k = cute.rsqrt(sum_k + cutlass.Float32(1e-6))

        for ki in cutlass.range_constexpr(VALUES_PER_LANE):
            q_bf16[ki] = (q_fp32[ki] * inv_q).to(cutlass.BFloat16)
            k_bf16[ki] = (k_fp32[ki] * inv_k).to(cutlass.BFloat16)
        cute.autovec_copy(
            q_bf16,
            cute.local_tile(
                mQOut,
                (1, VALUES_PER_LANE),
                (row, lane_id),
            ),
        )
        cute.autovec_copy(
            k_bf16,
            cute.local_tile(
                mKOut,
                (1, VALUES_PER_LANE),
                (row, lane_id),
            ),
        )


atrex_gdn_fused_qk_l2norm_bf16_kernel.set_name_prefix("atrex")


@cute.jit
def launch_fused_qk_l2norm_bf16(
    mQ: cute.Tensor,
    mK: cute.Tensor,
    mQOut: cute.Tensor,
    mKOut: cute.Tensor,
    rows: cutlass.Int32,
    stream,
):
    atrex_gdn_fused_qk_l2norm_bf16_kernel(
        mQ, mK, mQOut, mKOut, rows,
    ).launch(
        grid=(cute.ceil_div(rows, NUM_WARPS), 1, 1),
        block=(NUM_THREADS, 1, 1),
        stream=stream,
    )


_compiled_cache: dict = {}
_compiled_cache_lock = threading.Lock()


def fused_qk_l2_normalize_bf16(
    q: torch.Tensor,
    k: torch.Tensor,
    *,
    q_out: torch.Tensor | None = None,
    k_out: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Normalize a Q/K pair and return tensors with the original shape."""
    if q.device.type != "cuda" or k.device != q.device:
        raise ValueError("q and k must be CUDA tensors on the same device")
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16:
        raise TypeError("q and k must both be torch.bfloat16")
    if q.shape != k.shape or q.ndim < 2 or q.shape[-1] != K_DIM:
        raise ValueError(
            "q and k must have identical [..., 128] shapes, got "
            f"q={tuple(q.shape)}, k={tuple(k.shape)}"
        )

    q_contig = q.contiguous()
    k_contig = k.contiguous()
    q_out_supplied = q_out is not None
    k_out_supplied = k_out is not None
    if q_out is None:
        q_out = torch.empty_like(q_contig)
    if k_out is None:
        k_out = torch.empty_like(k_contig)
    for name, out in (("q_out", q_out), ("k_out", k_out)):
        if (
            out.device != q.device
            or out.dtype != torch.bfloat16
            or out.shape != q.shape
            or not out.is_contiguous()
        ):
            raise ValueError(
                f"{name} must be contiguous bf16 on {q.device} with shape "
                f"{tuple(q.shape)}"
            )

    # The current CuTeDSL runtime launch is not recorded by torch CUDA Graph
    # capture. Preserve exact semantics with the original eager formula while
    # capturing; ordinary execution still uses the fused kernel below.
    if torch.cuda.is_current_stream_capturing():
        q_fp32 = q_contig.float()
        k_fp32 = k_contig.float()
        q_normalized = (
            q_fp32
            * torch.rsqrt(
                (q_fp32 * q_fp32).sum(dim=-1, keepdim=True) + 1e-6
            )
        ).to(q.dtype)
        k_normalized = (
            k_fp32
            * torch.rsqrt(
                (k_fp32 * k_fp32).sum(dim=-1, keepdim=True) + 1e-6
            )
        ).to(k.dtype)
        if q_out_supplied:
            q_out.copy_(q_normalized)
        else:
            q_out = q_normalized
        if k_out_supplied:
            k_out.copy_(k_normalized)
        else:
            k_out = k_normalized
        return q_out, k_out

    q_2d = q_contig.view(-1, K_DIM)
    k_2d = k_contig.view(-1, K_DIM)
    q_out_2d = q_out.view(-1, K_DIM)
    k_out_2d = k_out.view(-1, K_DIM)
    if q_2d.stride() != (K_DIM, 1) or k_2d.stride() != (K_DIM, 1):
        raise ValueError("flattened q and k must have stride (128, 1)")
    if q_out_2d.stride() != (K_DIM, 1) or k_out_2d.stride() != (K_DIM, 1):
        raise ValueError("flattened outputs must have stride (128, 1)")
    rows = int(q_2d.shape[0])
    import cuda.bindings.driver as cuda_driver

    stream_val = torch.cuda.current_stream(device=q.device).cuda_stream
    stream = cuda_driver.CUstream(stream_val)
    mQ = from_dlpack(q_2d, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    mK = from_dlpack(k_2d, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    mQOut = from_dlpack(q_out_2d, assumed_align=16).mark_layout_dynamic(leading_dim=1)
    mKOut = from_dlpack(k_out_2d, assumed_align=16).mark_layout_dynamic(leading_dim=1)

    device_index = q.device.index
    if device_index is None:
        device_index = torch.cuda.current_device()
    key = (
        "atrex-fused-qk-l2norm-bf16-d128-v4-dynamic-layout",
        int(device_index),
        q.dtype,
        K_DIM,
    )
    with _compiled_cache_lock:
        compiled = _compiled_cache.get(key)
        if compiled is None:
            with _stable_cutlass_dsl_version():
                compiled = cute.compile[_SM120A_COMPILE_OPTIONS](
                    launch_fused_qk_l2norm_bf16,
                    mQ,
                    mK,
                    mQOut,
                    mKOut,
                    cutlass.Int32(rows),
                    stream,
                )
            _compiled_cache[key] = compiled
    compiled(mQ, mK, mQOut, mKOut, cutlass.Int32(rows), stream)
    return q_out, k_out


def prewarm_fused_qk_l2_normalize_bf16(
    *,
    device: torch.device | str | int = "cuda",
) -> None:
    """Compile the dynamic-layout D=128 specialization from exactly rows=1."""
    device = torch.device(device)
    q = torch.zeros((1, K_DIM), device=device, dtype=torch.bfloat16)
    k = torch.zeros_like(q)
    fused_qk_l2_normalize_bf16(q, k)
