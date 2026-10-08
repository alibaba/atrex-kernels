"""Validated SM103 launch policy."""

import math
from threading import Lock

import torch

from atrex.utils.device_target import detect_device_target

_PREFILL_JIT_KEYS: set[tuple] = set()
_PREFILL_JIT_LOCK = Lock()

_AKA_BATCH_RANGE = range(16, 29)
_AKA_Q_LEN = 4
_AKA_Q_HEADS = 16
_AKA_KV_HEADS = 1
_AKA_HEAD_DIM = 256
_AKA_PAGE_SIZE = 128
_AKA_SCALE = _AKA_HEAD_DIM**-0.5


def _cuda_runtime_major(version):
    if version is None:
        return None
    try:
        return int(version.split(".", 1)[0])
    except ValueError:
        return None


def _is_aligned(tensor, alignment=16):
    return tensor.data_ptr() % alignment == 0


def _can_use_aka_q4(**kwargs):
    """Return whether the unified call matches the AKA BF16 q4 fast path."""
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
        if tuple(q.shape[1:]) != (_AKA_Q_HEADS, _AKA_HEAD_DIM):
            return False
        if tuple(k.shape[1:]) != (
            _AKA_PAGE_SIZE,
            _AKA_KV_HEADS,
            _AKA_HEAD_DIM,
        ):
            return False
        if k.shape[0] <= 0 or any(t.device != q.device for t in (k, v)):
            return False
        if not all(t.is_contiguous() and _is_aligned(t) for t in (q, k, v)):
            return False

        if seqused_k is None or page_table is None or cu_seqlens_q is None:
            return False
        batch_size = seqused_k.numel()
        if batch_size not in _AKA_BATCH_RANGE:
            return False
        if kwargs.get("max_seqlen_q") != _AKA_Q_LEN:
            return False
        if q.shape[0] != batch_size * _AKA_Q_LEN:
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
            or max_seqlen_k > page_table.shape[1] * _AKA_PAGE_SIZE
        ):
            return False

        if kwargs.get("cu_seqlens_k") is not None:
            return False
        if kwargs.get("causal") is not True:
            return False
        softmax_scale = kwargs.get("softmax_scale")
        if softmax_scale is None:
            softmax_scale = _AKA_SCALE
        if not math.isclose(
            float(softmax_scale), _AKA_SCALE, rel_tol=0.0, abs_tol=1e-12
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
            for name in ("q_descale", "k_descale", "v_descale")
        ):
            return False

        if out is not None and (
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

def _ensure_prefill_jit(
    q,
    k,
    v,
    *,
    cu_seqlens_q,
    cu_seqlens_k,
    seqused_q,
    seqused_k,
    page_table,
    softmax_scale,
    causal,
    pack_gqa,
    softcap,
    learnable_sink,
    q_descale=None,
    k_descale=None,
    v_descale=None,
):
    """Prepare the default-feature prefill specialization once per static key.

    A short-Q call supplies the static properties needed by default-feature
    prefill (softcap=0, no attention sink). Serving startup must exercise each
    short-Q shape before graph capture and before accepting requests. A failed
    mandatory precompile is deliberately reported, not silently bypassed.
    Non-default features compile their own specialization on first use.
    """
    if softcap not in (None, 0, 0.0) or learnable_sink is not None:
        return
    if (
        q is None
        or k is None
        or v is None
        or q.device.type != "cuda"
    ):
        return

    device_index = q.device.index
    key = (
        device_index,
        q.dtype,
        k.dtype,
        v.dtype,
        q.shape[-2:],
        k.shape[-2:],
        v.shape[-2:],
        k.shape[1] if k.ndim == 4 else None,
        cu_seqlens_q is not None,
        cu_seqlens_k is not None,
        seqused_q is not None,
        seqused_k is not None,
        page_table is not None,
        causal,
        pack_gqa,
        tuple(None if s is None else (s.dtype, tuple(s.shape), s.stride())
              for s in (q_descale, k_descale, v_descale)),
    )
    if key in _PREFILL_JIT_KEYS:
        return

    with _PREFILL_JIT_LOCK:
        if key in _PREFILL_JIT_KEYS:
            return
        if torch.cuda.is_current_stream_capturing():
            raise RuntimeError(
                "SM103 prefill specialization was not compiled before CUDA "
                "graph capture; warm up the short-Q shape during startup"
            )
        _prefill_forward()(
            q,
            k,
            v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_q=seqused_q,
            seqused_k=seqused_k,
            page_table=page_table,
            softmax_scale=softmax_scale,
            causal=causal,
            num_splits=1,
            pack_gqa=pack_gqa,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            _prepare_only=True,
        )
        _PREFILL_JIT_KEYS.add(key)

def _prefill_forward():
    """Lazily load the PAI-compatible 2CTA forward entry.

    Keeping this import out of module initialization lets ``import atrex`` work
    in CPU-only environments, where CuTe DSL is deliberately unavailable.
    """
    from atrex.src.nvidia.flash_attn.sm103.prefill_runtime import _flash_attn_fwd

    return _flash_attn_fwd

def _short_q_forward():
    """Lazily load the ragged short-Q decode entry.

    The decode module owns validation of its supported static shapes.  An
    unsupported shape must be reported as ``NotImplementedError`` so this
    wrapper can route it to Atrex's 2CTA path, never to the PAI wheel.
    """
    from atrex.src.nvidia.flash_attn.sm103.decode_runtime import _fa4_decode_varlen

    return _fa4_decode_varlen

def _run_prefill(**kwargs):
    """Run 2CTA prefill and normalize its private four-value result."""
    result = _prefill_forward()(**kwargs)
    return result[0], result[1]

def _run_short_q(**kwargs):
    """Run the decode binding, whose public ABI is also ``(out, lse)``."""
    return _short_q_forward()(**kwargs)


def _run_aka_q4(**kwargs):
    """Run the dev AKA specialization through the unified private ABI."""
    from atrex.src.nvidia.flash_attn.sm103.aka_decode_runtime import (
        atrex_aka_fa4_decode,
    )

    return atrex_aka_fa4_decode(**kwargs)

def forward(**kwargs):
    q, k, v = kwargs["q"], kwargs["k"], kwargs["v"]
    max_seqlen_q = kwargs.get("max_seqlen_q")
    if max_seqlen_q is not None and 1 <= max_seqlen_q <= 5:
        _ensure_prefill_jit(
            q, k, v,
            cu_seqlens_q=kwargs.get("cu_seqlens_q"),
            cu_seqlens_k=kwargs.get("cu_seqlens_k"),
            seqused_q=kwargs.get("seqused_q"),
            seqused_k=kwargs.get("seqused_k"),
            page_table=kwargs.get("page_table"),
            softmax_scale=kwargs.get("softmax_scale"),
            causal=kwargs.get("causal", False),
            pack_gqa=kwargs.get("pack_gqa"),
            softcap=kwargs.get("softcap"),
            learnable_sink=kwargs.get("learnable_sink"),
            q_descale=kwargs.get("q_descale"),
            k_descale=kwargs.get("k_descale"),
            v_descale=kwargs.get("v_descale"),
        )
        if _can_use_aka_q4(**kwargs):
            return _run_aka_q4(**kwargs)
        try:
            return _run_short_q(**kwargs)
        except NotImplementedError:
            # Unsupported short-Q features use the general Atrex kernel.
            pass
    # This implementation is non-split; vLLM's maximum split count is a hint.
    kwargs["num_splits"] = 1
    return _run_prefill(**kwargs)
