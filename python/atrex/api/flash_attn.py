"""Unified vLLM-compatible paged/varlen attention entry.

Framework tensors and sequence metadata are consumed as supplied. Architecture
launchers own kernel selection and temporary storage, not framework scheduling.
Importing this module does not import Torch or compile a hardware implementation.
"""

from atrex.utils.device_target import detect_device_target


def flash_attn_varlen_func(
    q,
    k,
    v,
    max_seqlen_q,
    cu_seqlens_q,
    max_seqlen_k,
    cu_seqlens_k=None,
    seqused_k=None,
    q_v=None,
    dropout_p=0.0,
    softmax_scale=None,
    causal=False,
    window_size=None,
    softcap=0.0,
    alibi_slopes=None,
    deterministic=False,
    return_attn_probs=False,
    block_table=None,
    return_softmax_lse=False,
    out=None,
    scheduler_metadata=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    num_splits=0,
    output_scale=None,
    fa_version=3,
    s_aux=None,
    cp_world_size=1,
    cp_rank=0,
    cp_tot_seqused_k=None,
    mask_mod=None,
    aux_tensors=None,
    dynamic_causal=None,
    *,
    qv=None,
    logits_soft_cap=None,
    num_prefill=-1,
    max_seqlen_k_decode=0,
    max_seqlen_k_prefill=0,
):
    """Run attention, returning ``out`` or ``(out, lse)`` on request.

    Paged K/V have logical shape [blocks, page_size, kv_heads, dim].
    ``block_table`` indexes those physical blocks; ``seqused_k`` bounds the
    valid tokens. No CPU readback or page-table conversion is performed here.
    ``fa_version`` labels the caller's API, not the hardware implementation.
    Vendor scheduler metadata is opaque and is not consumed by Atrex kernels.
    Prefill counts are not used to repartition a batch. Only forward inference
    is supported; attention probabilities and context parallelism are not.
    """
    import torch

    del scheduler_metadata, deterministic, num_prefill
    del max_seqlen_k_decode, max_seqlen_k_prefill
    if fa_version not in (2, 3, 4):
        raise ValueError(f"Unsupported FlashAttention API version {fa_version}")
    if dropout_p != 0 or return_attn_probs:
        raise NotImplementedError("Atrex attention supports inference without dropout/probabilities")
    if cp_world_size != 1 or cp_rank != 0 or cp_tot_seqused_k is not None:
        raise NotImplementedError("Atrex attention does not yet support context parallelism")
    if alibi_slopes is not None:
        raise NotImplementedError("Atrex attention does not support ALiBi")
    if q_v is not None or qv is not None:
        raise NotImplementedError("Atrex standard attention does not support auxiliary Q/MLA")
    if output_scale is not None:
        raise NotImplementedError("Atrex attention does not support quantized output")
    if mask_mod is not None or aux_tensors is not None:
        raise NotImplementedError("Atrex attention does not support custom mask modifiers")
    if dynamic_causal is not None:
        raise NotImplementedError("Atrex attention does not support per-sequence causal masks")
    if (cu_seqlens_k is None) == (seqused_k is None):
        raise ValueError("Provide exactly one of cu_seqlens_k and seqused_k")
    if block_table is not None and seqused_k is None:
        raise ValueError("Paged KV requires seqused_k")
    if q.ndim != 3 or k.ndim != (4 if block_table is not None else 3) or v.ndim != k.ndim:
        raise ValueError("Expected packed Q and packed or paged K/V")
    if q.device != k.device or q.device != v.device:
        raise ValueError("Q/K/V must share one device")
    if q.dtype != k.dtype or k.dtype != v.dtype:
        raise NotImplementedError("Atrex attention requires matching Q/K/V dtypes")
    if q.dtype not in (torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2):
        raise NotImplementedError(f"Unsupported attention dtype {q.dtype}")
    if q.shape[-1] != k.shape[-1] or k.shape[:-1] != v.shape[:-1]:
        raise ValueError("Incompatible Q/K/V dimensions")
    if k.shape[-2] <= 0 or q.shape[-2] % k.shape[-2]:
        raise ValueError("Q heads must be divisible by KV heads")
    if any(t.stride(-1) != 1 for t in (q, k, v)):
        raise ValueError("Q/K/V head dimensions must be contiguous")
    if num_splits < 0:
        raise ValueError("num_splits must be nonnegative")
    window = (-1, -1) if window_size is None else tuple(window_size)
    if len(window) != 2:
        raise ValueError("window_size must contain two values")
    scale = q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
    if logits_soft_cap is not None:
        softcap = logits_soft_cap
    # Descale describes FP8 storage only. vLLM FA3 also supplies these for BF16.
    if q.dtype == torch.bfloat16:
        q_descale = k_descale = v_descale = None

    target = detect_device_target(q.device)
    if target.family == "nvidia" and target.arch in ("sm100", "sm103"):
        from atrex.src.nvidia.flash_attn.sm103.launch import forward
    elif target.family == "nvidia" and target.arch == "sm120":
        from atrex.src.nvidia.flash_attn.sm120.launch import forward
    elif target.family == "alibaba_ppu" and target.arch == "zwm890p":
        from atrex.src.ppu.flash_attn.zwm890p.runtime import forward
    else:
        raise NotImplementedError(f"Atrex attention does not support {target.family}/{target.arch}")

    # Launch and compilation must use the input device, not an unrelated
    # process-current device. The caller's device is restored on return.
    with torch.cuda.device(q.device):
        output, lse = forward(
            q=q, k=k, v=v,
            cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k,
            seqused_k=seqused_k, max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k, page_table=block_table,
            softmax_scale=scale, causal=causal, softcap=softcap,
            window_size_left=window[0] if window[0] >= 0 else None,
            window_size_right=window[1] if window[1] >= 0 else None,
            learnable_sink=s_aux, out=out, return_lse=return_softmax_lse,
            q_descale=q_descale, k_descale=k_descale, v_descale=v_descale,
            num_splits=num_splits,
        )
    return (output, lse) if return_softmax_lse else output
