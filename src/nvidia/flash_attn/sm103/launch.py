"""Validated SM103 launch policy."""

from threading import Lock

_PREFILL_JIT_KEYS: set[tuple] = set()
_PREFILL_JIT_LOCK = Lock()

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
    q_descale=None,
    k_descale=None,
    v_descale=None,
):
    """Compile the matching prefill specialization from a short-Q call.

    vLLM naturally compiles the decode/MTP specializations while preparing
    CUDA graphs.  A decode-shaped call already carries every static property
    needed by the sequence-length-independent prefill compiler, so use it to
    populate the prefill cache as well.  This keeps the first real prefill
    request off the compilation path without launching a dummy prefill.
    """
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
    if kwargs.get("num_splits") == 0:
        # vLLM uses 0 to delegate split selection to the operator. The Atrex
        # 2CTA prefill path is intentionally non-split, represented as 1 by
        # its private interface.
        kwargs = dict(kwargs)
        kwargs["num_splits"] = 1
    result = _prefill_forward()(**kwargs)
    return result[0], result[1]

def _run_short_q(**kwargs):
    """Run the decode binding, whose public ABI is also ``(out, lse)``."""
    return _short_q_forward()(**kwargs)

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
            q_descale=kwargs.get("q_descale"),
            k_descale=kwargs.get("k_descale"),
            v_descale=kwargs.get("v_descale"),
        )
        try:
            return _run_short_q(**kwargs)
        except NotImplementedError:
            # Unsupported short-Q features use the general Atrex kernel.
            pass
    # This implementation is non-split; vLLM's maximum split count is a hint.
    kwargs["num_splits"] = 1
    return _run_prefill(**kwargs)
