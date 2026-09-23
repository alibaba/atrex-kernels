"""vLLM-compatible paged/varlen attention with hardware dispatch."""

from atrex.utils.device_target import detect_device_target


_TARGET_FA_VERSIONS = {
    ("nvidia", "sm103"): frozenset({4}),
    ("nvidia", "sm120"): frozenset({3}),
    ("alibaba_ppu", "zwm890p"): frozenset({2, 3}),
}


def _validate_flash_attn_call(call):
    """Validate the public contract without compiling or allocating buffers."""
    import torch

    eligibility_only = call.get("_eligibility_only", False)
    q, k, v = (call[name] for name in ("q", "k", "v"))
    cu_seqlens_q = call["cu_seqlens_q"]
    cu_seqlens_k = call["cu_seqlens_k"]
    seqused_k = call["seqused_k"]
    block_table = call["block_table"]
    fa_version = call["fa_version"]

    if call["deterministic"] is not False:
        raise NotImplementedError("Atrex attention does not support deterministic=True")
    if call["scheduler_metadata"] is not None:
        raise NotImplementedError(
            "Atrex owns its kernel schedule and does not consume vendor "
            "scheduler_metadata"
        )
    if (
        call["num_prefill"] != -1
        or call["max_seqlen_k_decode"] != 0
        or call["max_seqlen_k_prefill"] != 0
    ):
        raise NotImplementedError(
            "Atrex does not consume provider-specific prefill/decode counters"
        )
    if call["dropout_p"] != 0 or call["return_attn_probs"]:
        raise NotImplementedError(
            "Atrex attention supports inference without dropout/probabilities"
        )
    if (
        call["cp_world_size"] != 1
        or call["cp_rank"] != 0
        or call["cp_tot_seqused_k"] is not None
    ):
        raise NotImplementedError(
            "Atrex attention does not yet support context parallelism"
        )
    if call["alibi_slopes"] is not None:
        raise NotImplementedError("Atrex attention does not support ALiBi")
    if call["q_v"] is not None or call["qv"] is not None:
        raise NotImplementedError(
            "Atrex standard attention does not support auxiliary Q/MLA"
        )
    if call["output_scale"] is not None:
        raise NotImplementedError("Atrex attention does not support quantized output")
    if call["mask_mod"] is not None or call["aux_tensors"] is not None:
        raise NotImplementedError(
            "Atrex attention does not support custom mask modifiers"
        )
    if call["dynamic_causal"] is not None:
        raise NotImplementedError(
            "Atrex attention does not support per-sequence causal masks"
        )
    if (cu_seqlens_k is None) == (seqused_k is None):
        raise ValueError("Provide exactly one of cu_seqlens_k and seqused_k")
    if block_table is not None and seqused_k is None:
        raise ValueError("Paged KV requires seqused_k")
    if q.ndim != 3 or k.ndim != (4 if block_table is not None else 3):
        raise ValueError("Expected packed Q and packed or paged K/V")
    if v.ndim != k.ndim:
        raise ValueError("K/V must have the same rank")
    if q.device != k.device or q.device != v.device:
        raise ValueError("Q/K/V must share one device")
    if q.dtype != k.dtype or k.dtype != v.dtype:
        raise NotImplementedError("Atrex attention requires matching Q/K/V dtypes")
    if q.dtype not in (torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2):
        raise NotImplementedError(f"Unsupported attention dtype {q.dtype}")
    if q.shape[-1] != 256 or k.shape[-1] != 256 or v.shape[-1] != 256:
        raise NotImplementedError("Atrex attention currently requires head_dim=256")
    if k.shape[:-1] != v.shape[:-1]:
        raise ValueError("Incompatible K/V dimensions")
    if k.shape[-2] <= 0 or q.shape[-2] % k.shape[-2]:
        raise ValueError("Q heads must be divisible by KV heads")
    if any(tensor.stride(-1) != 1 for tensor in (q, k, v)):
        raise ValueError("Q/K/V head dimensions must be contiguous")
    if call["num_splits"] < 0:
        raise ValueError("num_splits must be nonnegative")
    if cu_seqlens_q is None:
        raise ValueError("Atrex varlen attention requires cu_seqlens_q")

    for tensor, name in (
        (cu_seqlens_q, "cu_seqlens_q"),
        (cu_seqlens_k, "cu_seqlens_k"),
        (seqused_k, "seqused_k"),
        (block_table, "block_table"),
    ):
        if tensor is None:
            continue
        if tensor.device != q.device or tensor.dtype != torch.int32:
            raise ValueError(f"{name} must be int32 on the Q device")
        if tensor.ndim not in (1, 2) or tensor.stride(-1) != 1:
            raise ValueError(f"{name} must have contiguous rows")

    window = (-1, -1) if call["window_size"] is None else tuple(call["window_size"])
    if len(window) != 2:
        raise ValueError("window_size must contain two values")
    softcap = call["softcap"]
    logits_soft_cap = call["logits_soft_cap"]
    if logits_soft_cap is not None:
        if softcap not in (0, 0.0, logits_soft_cap):
            raise ValueError("softcap and logits_soft_cap disagree")
        softcap = logits_soft_cap
    if q.dtype == torch.bfloat16 and any(
        call[name] is not None for name in ("q_descale", "k_descale", "v_descale")
    ):
        raise NotImplementedError("BF16 attention does not consume FP8 descales")

    target = detect_device_target(q.device)
    target_key = (target.family, target.arch)
    supported_versions = _TARGET_FA_VERSIONS.get(target_key)
    if supported_versions is None:
        raise NotImplementedError(
            "Atrex attention supports only nvidia/sm103, nvidia/sm120, and "
            f"alibaba_ppu/zwm890p; detected {target.family}/{target.arch}"
        )
    if fa_version is not None and fa_version not in supported_versions:
        versions = ", ".join(str(version) for version in sorted(supported_versions))
        raise NotImplementedError(
            f"Atrex {target.family}/{target.arch} supports fa_version {versions}; "
            f"received {fa_version}"
        )

    if target_key in (("nvidia", "sm103"), ("nvidia", "sm120")):
        if window != (-1, -1):
            raise NotImplementedError(
                f"Atrex {target.arch} attention does not support sliding windows"
            )
    else:
        if q.dtype != torch.float8_e4m3fn:
            raise NotImplementedError("PPU attention requires FP8 E4M3 Q/K/V")
        if tuple(q.shape[1:]) != (8, 256):
            raise NotImplementedError("PPU attention requires Q [tokens,8,256]")
        if block_table is None or tuple(k.shape[2:]) != (1, 256):
            raise NotImplementedError(
                "PPU attention requires paged K/V [blocks,page_size,1,256]"
            )
        if k.shape[1] < 64 or k.shape[1] % 64:
            raise NotImplementedError(
                "PPU prep_kv requires page_size to be a multiple of 64"
            )
        if not call["causal"] or window != (-1, -1):
            raise NotImplementedError(
                "PPU attention supports causal full-window attention"
            )
        if softcap or call["s_aux"] is not None:
            raise NotImplementedError("PPU attention does not support softcap or sinks")
        if call["return_softmax_lse"]:
            raise NotImplementedError("PPU attention does not return softmax LSE")
        if eligibility_only and any(
            call[name] is not None
            for name in ("q_descale", "k_descale", "v_descale")
        ):
            raise NotImplementedError(
                "PPU can_use cannot prove device descales are fixed at one"
            )
        softmax_scale = call["softmax_scale"]
        if softmax_scale is not None and abs(float(softmax_scale) - 256**-0.5) > 1e-12:
            raise NotImplementedError("PPU attention requires softmax_scale=1/sqrt(256)")

    return target, window, softcap


def can_use_flash_attn_varlen_func(
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
    fa_version=None,
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
    """Return whether ATREX can execute this exact call without side effects."""
    try:
        call = locals()
        call["_eligibility_only"] = True
        _validate_flash_attn_call(call)
    except (
        AttributeError,
        ImportError,
        NotImplementedError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return False
    return True


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
    fa_version=None,
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
    """Run attention and return ``out`` or ``(out, lse)`` when requested."""
    import torch

    target, window, softcap = _validate_flash_attn_call(locals())
    if target.family == "nvidia" and target.arch == "sm103":
        from atrex.src.nvidia.flash_attn.sm103.launch import forward
    elif target.family == "nvidia" and target.arch == "sm120":
        from atrex.src.nvidia.flash_attn.sm120.launch import forward
    elif target.family == "alibaba_ppu" and target.arch == "zwm890p":
        from atrex.src.ppu.flash_attn.zwm890p.runtime import forward
    else:  # Kept exhaustive with _TARGET_FA_VERSIONS.
        raise AssertionError(f"unhandled validated target {target}")

    with torch.cuda.device(q.device):
        output, lse = forward(
            q=q,
            k=k,
            v=v,
            cu_seqlens_q=cu_seqlens_q,
            cu_seqlens_k=cu_seqlens_k,
            seqused_k=seqused_k,
            max_seqlen_q=max_seqlen_q,
            max_seqlen_k=max_seqlen_k,
            page_table=block_table,
            softmax_scale=(
                q.shape[-1] ** -0.5 if softmax_scale is None else softmax_scale
            ),
            causal=causal,
            softcap=softcap,
            window_size_left=window[0] if window[0] >= 0 else None,
            window_size_right=window[1] if window[1] >= 0 else None,
            learnable_sink=s_aux,
            out=out,
            return_lse=return_softmax_lse,
            q_descale=q_descale,
            k_descale=k_descale,
            v_descale=v_descale,
            num_splits=num_splits,
        )
    return (output, lse) if return_softmax_lse else output


__all__ = ("can_use_flash_attn_varlen_func", "flash_attn_varlen_func")
