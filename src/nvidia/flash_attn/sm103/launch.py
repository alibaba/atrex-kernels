"""SM103 FlashAttention launch policy with an AKA-only q4 branch."""

from __future__ import annotations

import math

import torch

from atrex.utils.device_target import detect_device_target


_ATREX_AKA_BATCH_RANGE = range(16, 29)
_ATREX_AKA_Q_LEN = 4
_ATREX_AKA_Q_HEADS = 16
_ATREX_AKA_KV_HEADS = 1
_ATREX_AKA_HEAD_DIM = 256
_ATREX_AKA_PAGE_SIZE = 128
_ATREX_AKA_SCALE = _ATREX_AKA_HEAD_DIM**-0.5


def _cuda_runtime_major(version: str | None) -> int | None:
    if version is None:
        return None
    try:
        return int(version.split(".", 1)[0])
    except ValueError:
        return None


def _is_aligned(tensor: torch.Tensor, alignment: int = 16) -> bool:
    return tensor.data_ptr() % alignment == 0


def can_use_atrex_aka_fa4_decode(**kwargs) -> bool:
    """Return whether this call exactly matches the verified AKA q4 domain."""
    try:
        q, k, v = kwargs["q"], kwargs["k"], kwargs["v"]
        cu_seqlens_q = kwargs.get("cu_seqlens_q")
        seqused_k = kwargs.get("seqused_k")
        page_table = kwargs.get("page_table")
        out = kwargs.get("out")
        if q.device.type != "cuda":
            return False
        target = detect_device_target(q.device)
        if (
            target.family != "nvidia"
            or target.arch != "sm103"
            or (_cuda_runtime_major(target.runtime_version) or 0) < 13
        ):
            return False
        if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
            return False
        if q.ndim != 3 or k.ndim != 4 or v.shape != k.shape:
            return False
        if tuple(q.shape[1:]) != (_ATREX_AKA_Q_HEADS, _ATREX_AKA_HEAD_DIM):
            return False
        if tuple(k.shape[1:]) != (
            _ATREX_AKA_PAGE_SIZE,
            _ATREX_AKA_KV_HEADS,
            _ATREX_AKA_HEAD_DIM,
        ):
            return False
        if k.shape[0] <= 0:
            return False
        if any(t.device != q.device for t in (k, v)):
            return False
        if not all(t.is_contiguous() and _is_aligned(t) for t in (q, k, v)):
            return False

        if seqused_k is None or page_table is None or cu_seqlens_q is None:
            return False
        batch_size = seqused_k.numel()
        if batch_size not in _ATREX_AKA_BATCH_RANGE:
            return False
        if kwargs.get("max_seqlen_q") != _ATREX_AKA_Q_LEN:
            return False
        if q.shape[0] != batch_size * _ATREX_AKA_Q_LEN:
            return False
        if (
            cu_seqlens_q.device != q.device
            or cu_seqlens_q.dtype != torch.int32
            or cu_seqlens_q.ndim != 1
            or cu_seqlens_q.numel() != batch_size + 1
            or not cu_seqlens_q.is_contiguous()
            or not _is_aligned(cu_seqlens_q)
        ):
            return False
        if (
            seqused_k.device != q.device
            or seqused_k.dtype != torch.int32
            or seqused_k.ndim != 1
            or not seqused_k.is_contiguous()
            or not _is_aligned(seqused_k)
        ):
            return False
        if (
            page_table.device != q.device
            or page_table.dtype != torch.int32
            or page_table.ndim != 2
            or page_table.shape[0] != batch_size
            or page_table.shape[1] <= 0
            or not page_table.is_contiguous()
            or not _is_aligned(page_table)
        ):
            return False
        max_seqlen_k = kwargs.get("max_seqlen_k")
        if (
            not isinstance(max_seqlen_k, int)
            or max_seqlen_k <= 0
            or max_seqlen_k > page_table.shape[1] * _ATREX_AKA_PAGE_SIZE
        ):
            return False

        if kwargs.get("cu_seqlens_k") is not None:
            return False
        if kwargs.get("causal") is not True:
            return False
        softmax_scale = kwargs.get("softmax_scale")
        if softmax_scale is None:
            softmax_scale = _ATREX_AKA_SCALE
        if not math.isclose(
            float(softmax_scale),
            _ATREX_AKA_SCALE,
            rel_tol=0.0,
            abs_tol=1e-12,
        ):
            return False
        if kwargs.get("return_lse") is not False:
            return False
        if kwargs.get("num_splits") != 0:
            return False
        if kwargs.get("softcap") not in (None, 0, 0.0):
            return False
        if kwargs.get("window_size_left") is not None:
            return False
        if kwargs.get("window_size_right") is not None:
            return False
        if kwargs.get("learnable_sink") is not None:
            return False
        if any(
            kwargs.get(name) is not None
            for name in (
                "qv",
                "seqused_q",
                "min_seqlen_k",
                "tile_mn",
                "mma_pv_is_rs",
                "intra_wg_overlap",
                "pack_gqa",
                "_arch",
                "score_mod",
                "mask_mod",
                "block_sparse_tensors",
                "lse",
                "aux_tensors",
                "aux_scalars",
                "q_descale",
                "k_descale",
                "v_descale",
                "gather_kv_indices",
                "output_scale",
            )
        ):
            return False
        if kwargs.get("num_threads", 384) != 384:
            return False

        if out is not None:
            if (
                out.device != q.device
                or out.dtype != torch.bfloat16
                or out.shape != q.shape
                or not out.is_contiguous()
                or not _is_aligned(out)
                or out.data_ptr() == q.data_ptr()
            ):
                return False
        return True
    except (AttributeError, KeyError, TypeError, ValueError):
        return False


def forward(**kwargs):
    """Dispatch the unified SM103 interface to the AKA q4 specialization."""
    if can_use_atrex_aka_fa4_decode(**kwargs):
        from .aka_decode_runtime import atrex_aka_fa4_decode

        return atrex_aka_fa4_decode(
            q=kwargs["q"],
            k=kwargs["k"],
            v=kwargs["v"],
            cu_seqlens_q=kwargs.get("cu_seqlens_q"),
            cu_seqlens_k=kwargs.get("cu_seqlens_k"),
            seqused_k=kwargs.get("seqused_k"),
            max_seqlen_q=kwargs.get("max_seqlen_q"),
            max_seqlen_k=kwargs.get("max_seqlen_k"),
            page_table=kwargs.get("page_table"),
            softmax_scale=(
                _ATREX_AKA_SCALE
                if kwargs.get("softmax_scale") is None
                else kwargs["softmax_scale"]
            ),
            causal=kwargs.get("causal"),
            softcap=kwargs.get("softcap"),
            window_size_left=kwargs.get("window_size_left"),
            window_size_right=kwargs.get("window_size_right"),
            learnable_sink=kwargs.get("learnable_sink"),
            out=kwargs.get("out"),
            return_lse=kwargs.get("return_lse"),
            q_descale=kwargs.get("q_descale"),
            k_descale=kwargs.get("k_descale"),
            v_descale=kwargs.get("v_descale"),
            num_splits=kwargs.get("num_splits"),
        )
    raise NotImplementedError(
        "No ATREX SM103 FlashAttention implementation supports this call; "
        "the current external build contains only the AKA BF16 q4 fast path"
    )


__all__ = ("forward",)
