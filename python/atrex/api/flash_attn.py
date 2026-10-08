"""vLLM-compatible paged/varlen attention with hardware dispatch."""

from atrex.utils.device_target import detect_device_target


_TARGET_FA_VERSIONS = {
    ("nvidia", "sm103"): frozenset({4}),
    ("nvidia", "sm120"): frozenset({3}),
    ("alibaba_ppu", "zwm890p"): frozenset({2, 3}),
}


def _validate_out(call, torch, *, require_contiguous=False):
    out = call["out"]
    if out is None:
        return
    q = call["q"]
    expected_dtype = (
        torch.bfloat16
        if q.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        else q.dtype
    )
    if out.shape != q.shape or out.dtype != expected_dtype or out.device != q.device:
        raise ValueError(
            "out must have Q's shape/device and the backend output dtype"
        )
    if out.stride(-1) != 1 or (require_contiguous and not out.is_contiguous()):
        raise ValueError("out has an unsupported layout")


def _validate_descales(call, torch, batch_size, num_kv_heads):
    q = call["q"]
    scales = (call["q_descale"], call["k_descale"], call["v_descale"])
    if q.dtype == torch.bfloat16:
        if any(scale is not None for scale in scales):
            raise NotImplementedError("BF16 attention does not consume FP8 descales")
        return
    for scale in scales:
        if scale is None:
            continue
        if (
            scale.shape != (batch_size, num_kv_heads)
            or scale.dtype != torch.float32
            or scale.device != q.device
        ):
            raise ValueError(
                "FP8 descales must be float32 [batch,num_kv_heads] on the Q device"
            )


def _validate_nvidia_call(call, torch, target_key, batch_size, window):
    q, k = call["q"], call["k"]
    block_table = call["block_table"]
    if window != (-1, -1):
        raise NotImplementedError(
            f"Atrex {target_key[1]} attention does not support sliding windows"
        )
    if call["s_aux"] is not None:
        sink = call["s_aux"]
        if (
            sink.shape != (q.shape[-2],)
            or sink.dtype != torch.bfloat16
            or sink.device != q.device
        ):
            raise ValueError("s_aux must be BF16 [num_q_heads] on the Q device")
    _validate_out(
        call,
        torch,
        require_contiguous=(
            target_key == ("nvidia", "sm103")
            and 1 <= call["max_seqlen_q"] <= 5
        ),
    )
    _validate_descales(call, torch, batch_size, k.shape[-2])

    if target_key == ("nvidia", "sm103"):
        if not call["causal"]:
            raise NotImplementedError("Atrex SM103 attention requires causal=True")
        return

    if call["s_aux"] is not None:
        raise NotImplementedError("Atrex SM120 attention does not support s_aux")
    if call["num_splits"] > 256:
        raise NotImplementedError("Atrex SM120 attention supports num_splits <= 256")
    if block_table is not None and k.shape[1] % 16:
        raise NotImplementedError(
            "Atrex SM120 paged attention requires page_size divisible by 16"
        )


def _validate_ppu_call(call, torch, batch_size, window, softcap, eligibility_only):
    q, k, v = (call[name] for name in ("q", "k", "v"))
    block_table = call["block_table"]
    seqused_k = call["seqused_k"]
    if q.dtype != torch.float8_e4m3fn:
        raise NotImplementedError("PPU attention requires FP8 E4M3 Q/K/V")
    if tuple(q.shape[1:]) != (8, 256) or not q.is_contiguous():
        raise NotImplementedError("PPU attention requires contiguous Q [tokens,8,256]")
    if block_table is None or tuple(k.shape[2:]) != (1, 256):
        raise NotImplementedError(
            "PPU attention requires paged K/V [blocks,page_size,1,256]"
        )
    if k.shape[1] < 64 or k.shape[1] % 64:
        raise NotImplementedError(
            "PPU prep_kv requires page_size to be a multiple of 64"
        )
    block_elements = k.shape[1] * k.shape[2] * k.shape[3]
    layouts = []
    for tensor in (k, v):
        if any(stride <= 0 for stride in tensor.stride()):
            raise ValueError("PPU KV requires positive strides")
        if tuple(tensor.stride()[1:]) != (256, 256, 1):
            raise NotImplementedError(
                "PPU prep_kv requires contiguous rows within each block"
            )
        if tensor.stride(0) % block_elements:
            raise NotImplementedError(
                "PPU prep_kv requires an integral block stride"
            )
        if tensor.numel() and tensor.data_ptr() % 16:
            raise ValueError("PPU prep_kv requires 16-byte aligned cache pointers")
        layouts.append(tensor.stride(0) // block_elements)
    if layouts[0] != layouts[1]:
        raise NotImplementedError("PPU prep_kv requires matching K/V block layouts")
    if not call["causal"] or window != (-1, -1):
        raise NotImplementedError(
            "PPU attention supports causal full-window attention"
        )
    if softcap or call["s_aux"] is not None:
        raise NotImplementedError("PPU attention does not support softcap or sinks")
    if call["return_softmax_lse"]:
        raise NotImplementedError("PPU attention does not return softmax LSE")
    if seqused_k.shape != (batch_size,) or block_table.shape[0] != batch_size:
        raise ValueError("PPU sequence metadata batch dimensions must match")
    if call["max_seqlen_k"] > block_table.shape[1] * k.shape[1]:
        raise ValueError("PPU block_table does not cover max_seqlen_k")
    _validate_out(call, torch, require_contiguous=True)
    if eligibility_only and any(
        call[name] is not None for name in ("q_descale", "k_descale", "v_descale")
    ):
        raise NotImplementedError(
            "PPU can_use cannot prove device descales are fixed at one"
        )
    softmax_scale = call["softmax_scale"]
    if softmax_scale is not None and abs(float(softmax_scale) - 256**-0.5) > 1e-12:
        raise NotImplementedError("PPU attention requires softmax_scale=1/sqrt(256)")


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
    if q.shape[-2] <= 0:
        raise ValueError("Q must contain at least one attention head")
    if any(tensor.stride(-1) != 1 for tensor in (q, k, v)):
        raise ValueError("Q/K/V head dimensions must be contiguous")
    if q.dtype in (torch.float8_e4m3fn, torch.float8_e5m2) and any(
        tensor.requires_grad for tensor in (q, k, v)
    ):
        raise NotImplementedError("Atrex FP8 attention is forward-only")
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

    if cu_seqlens_q.ndim != 1:
        raise ValueError("cu_seqlens_q must be one-dimensional")
    batch_size = cu_seqlens_q.shape[0] - 1
    if batch_size < 0:
        raise ValueError("Invalid cu_seqlens_q")
    if q.shape[0] and batch_size == 0:
        raise ValueError("Nonempty Q requires sequence metadata")
    if cu_seqlens_k is not None and cu_seqlens_k.shape != (batch_size + 1,):
        raise ValueError("cu_seqlens_k must have shape [batch+1]")
    if seqused_k is not None and seqused_k.shape != (batch_size,):
        raise ValueError("seqused_k must have shape [batch]")
    if block_table is not None and (
        block_table.ndim != 2 or block_table.shape[0] != batch_size
    ):
        raise ValueError("block_table must have shape [batch,max_pages]")
    if block_table is not None and k.shape[1] <= 0:
        raise ValueError("Paged K/V requires a positive page size")
    if call["max_seqlen_q"] < 0 or call["max_seqlen_k"] < 0:
        raise ValueError("maximum sequence lengths must be nonnegative")

    if target_key in (("nvidia", "sm103"), ("nvidia", "sm120")):
        _validate_nvidia_call(call, torch, target_key, batch_size, window)
    else:
        _validate_ppu_call(
            call, torch, batch_size, window, softcap, eligibility_only
        )

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
