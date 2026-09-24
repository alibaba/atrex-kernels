"""vLLM FA4 lower-level interface with hardware dispatch."""

from atrex.utils.device_target import detect_device_target


def can_use_flash_attn_varlen_func(
    q,
    k,
    v,
    qv=None,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    min_seqlen_k=None,
    page_table=None,
    softmax_scale=None,
    causal=False,
    softcap=None,
    window_size_left=None,
    window_size_right=None,
    learnable_sink=None,
    tile_mn=None,
    mma_pv_is_rs=None,
    intra_wg_overlap=None,
    num_threads=384,
    num_splits=1,
    pack_gqa=None,
    _arch=None,
    score_mod=None,
    mask_mod=None,
    block_sparse_tensors=None,
    return_lse=False,
    out=None,
    lse=None,
    aux_tensors=None,
    aux_scalars=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    gather_kv_indices=None,
    output_scale=None,
):
    """Return whether ATREX can execute this lower-level FA4 call."""
    call = locals()
    try:
        target = detect_device_target(q.device)
        if target.family != "nvidia" or target.arch != "sm103":
            return False
        from atrex.src.nvidia.flash_attn.sm103.launch import (
            can_use_atrex_aka_fa4_decode,
        )

        return can_use_atrex_aka_fa4_decode(**call)
    except (
        AttributeError,
        ImportError,
        NotImplementedError,
        RuntimeError,
        TypeError,
        ValueError,
    ):
        return False


def flash_attn_varlen_func(
    q,
    k,
    v,
    qv=None,
    cu_seqlens_q=None,
    cu_seqlens_k=None,
    seqused_q=None,
    seqused_k=None,
    max_seqlen_q=None,
    max_seqlen_k=None,
    min_seqlen_k=None,
    page_table=None,
    softmax_scale=None,
    causal=False,
    softcap=None,
    window_size_left=None,
    window_size_right=None,
    learnable_sink=None,
    tile_mn=None,
    mma_pv_is_rs=None,
    intra_wg_overlap=None,
    num_threads=384,
    num_splits=1,
    pack_gqa=None,
    _arch=None,
    score_mod=None,
    mask_mod=None,
    block_sparse_tensors=None,
    return_lse=False,
    out=None,
    lse=None,
    aux_tensors=None,
    aux_scalars=None,
    q_descale=None,
    k_descale=None,
    v_descale=None,
    gather_kv_indices=None,
    output_scale=None,
):
    """Run FA4 and always return its lower-level ``(out, lse)`` tuple."""
    import torch

    call = locals()
    call.pop("torch")
    from atrex.src.nvidia.flash_attn.sm103.launch import forward

    with torch.cuda.device(q.device):
        return forward(**call)


__all__ = ("can_use_flash_attn_varlen_func", "flash_attn_varlen_func")
