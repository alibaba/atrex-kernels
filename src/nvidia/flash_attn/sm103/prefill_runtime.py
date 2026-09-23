# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.

import inspect
import math
import os
from functools import lru_cache
from typing import Optional, Tuple, Callable

import torch
from torch._guards import active_fake_mode

import cutlass
import cutlass.cute as cute
from cutlass import Float32
from cutlass.cutlass_dsl import T, dsl_user_op
from cutlass._mlir.dialects import nvvm

from atrex.src.nvidia.flash_attn.common_utils.cutlass_dsl_cache import (
    use_filesystem_cutlass_dsl_version_hash,
)
from atrex._vendor.flash_attn.cute.cache_utils import get_jit_cache


if os.environ.get("CUTE_DSL_PTXAS_PATH", None) is not None:
    from atrex._vendor.flash_attn.cute import cute_dsl_ptxas  # noqa: F401

    # Patch to dump ptx and then use system ptxas to compile to cubin
    cute_dsl_ptxas.patch()


from atrex._vendor.flash_attn.cute import utils
from atrex._vendor.flash_attn.cute import fa_logging
from atrex._vendor.flash_attn.cute.cute_dsl_utils import (
    get_aux_tensor_metadata,
    to_cute_aux_tensor,
    to_cute_tensor,
)
from atrex.src.nvidia.flash_attn.sm103.prefill_cutedsl import (
    DescaleTensors,
    FlashAttentionForwardHd256_2CTA_Sm103,
)

from atrex._vendor.flash_attn.cute.utils import AuxData
from atrex._vendor.flash_attn.cute.block_sparsity import (
    BlockSparseTensorsTorch,
    to_cute_block_sparse_tensors,
    normalize_block_sparse_config,
)


def _patch_flash_attn_fmax_for_cutlass_dsl_45x() -> None:
    """Adapt the private FA4 beta20-derived fmax helper to the CuTeDSL 4.5.x ABI.

    The 4.5.x generated NVVM binding takes the result type as its first
    argument.  Flash-attn-4 b20 was published with the newer two-operand
    spelling, so using a genuinely isolated 4.5.x installation otherwise
    fails while lowering the softmax reduction.
    """
    parameters = tuple(inspect.signature(nvvm.fmax).parameters)
    if not parameters or parameters[0] != "res":
        return

    @dsl_user_op
    def fmax_45x(a, b, c=None, *, loc=None, ip=None):
        return Float32(
            nvvm.fmax(
                T.f32(),
                Float32(a).ir_value(loc=loc, ip=ip),
                Float32(b).ir_value(loc=loc, ip=ip),
                c=Float32(c).ir_value(loc=loc, ip=ip) if c is not None else None,
                loc=loc,
                ip=ip,
            )
        )

    utils.fmax = fmax_45x


_patch_flash_attn_fmax_for_cutlass_dsl_45x()


def is_fake_mode() -> bool:
    return active_fake_mode() is not None


def _parse_arch_str(arch_str):
    """Parse arch string (e.g. 'sm_80', 'sm_90a', '80', '100') to int (e.g. 80, 90, 100)."""
    import re
    match = re.match(r"^(?:sm_?|SM_?)?(\d+)(\d)([af]?)$", arch_str)
    if not match:
        raise ValueError(f"Invalid arch format: {arch_str}")
    major, minor, _ = match.groups()
    return int(major) * 10 + int(minor)


@lru_cache(maxsize=None)
def _get_device_arch(device_index):
    """Cached device arch check.

    Override with FLASH_ATTENTION_ARCH (e.g. 'sm_80' or '80') to select which
    kernel path to use (SM80/SM90/SM100/SM120) independently of the compilation
    target (CUTE_DSL_ARCH).

    For CPU-only compilation (no GPU), set both:
      FLASH_ATTENTION_ARCH=sm_80  (kernel selection)
      CUTE_DSL_ARCH=sm_80         (compilation target)
    """
    arch_override = os.environ.get("FLASH_ATTENTION_ARCH", None)
    if arch_override is not None:
        return _parse_arch_str(arch_override)
    major, minor = torch.cuda.get_device_capability(device_index)
    return major * 10 + int(minor)


def maybe_contiguous(x):
    if x is None:
        return None

    # PyTorch considers singleton dimensions contiguous regardless of their
    # stride.  In particular, vLLM expands scalar FP8 descales to [batch,
    # kv_heads], which produces a [1, 1] view with strides (0, 0) for a
    # single-request MQA batch.  ``contiguous()`` is a no-op for that view,
    # while CuTe requires the marked leading dimension to have unit stride.
    # Canonicalize singleton strides without allocating before handling a
    # genuinely non-contiguous final dimension.
    shape = x.shape
    strides = list(x.stride())
    expected_stride = 1
    changed = False
    for dim in range(len(shape) - 1, -1, -1):
        if shape[dim] == 1 and strides[dim] != expected_stride:
            strides[dim] = expected_stride
            changed = True
        expected_stride = strides[dim] * shape[dim]
    if changed:
        x = x.as_strided(shape, strides)
    return x.contiguous() if x.stride(-1) != 1 else x


def _validate_tensor(t, name, expected_shape, expected_dtype, expected_device):
    assert t.shape == expected_shape, f"{name} shape {t.shape} != expected {expected_shape}"
    assert t.dtype == expected_dtype, f"{name} dtype {t.dtype} != expected {expected_dtype}"
    assert t.device == expected_device, f"{name} device {t.device} != expected {expected_device}"
    if not is_fake_mode():
        assert t.is_cuda, f"{name} must be on CUDA"

torch2cute_dtype_map = {
    torch.float16: cutlass.Float16,
    torch.bfloat16: cutlass.BFloat16,
    torch.float32: cutlass.Float32,
    torch.float8_e4m3fn: cutlass.Float8E4M3FN,
    torch.float8_e5m2: cutlass.Float8E5M2,
}


def _resolve_causal_local_window(causal, window_size_left, window_size_right, mask_mod=None):
    """Resolve causal/local/window settings into canonical form.

    Returns (causal, local, window_size_left, window_size_right).
    """
    if mask_mod is not None:
        return False, False, window_size_left, window_size_right
    if causal:
        window_size_right = 0
    if window_size_left is not None and window_size_right is not None and window_size_left + window_size_right < 0:
        window_size_left = None
        window_size_right = None
    if window_size_left is not None or window_size_right is not None:
        if window_size_left is None and window_size_right == 0:
            causal, local = True, False
            window_size_right = None
        else:
            causal, local = False, True
    else:
        local = False
    return causal, local, window_size_left, window_size_right


def _flash_attn_fwd(
    q: Optional[torch.Tensor],
    k: Optional[torch.Tensor],
    v: torch.Tensor,
    qv: Optional[torch.Tensor] = None,
    cu_seqlens_q: Optional[torch.Tensor] = None,
    cu_seqlens_k: Optional[torch.Tensor] = None,
    seqused_q: Optional[torch.Tensor] = None,
    seqused_k: Optional[torch.Tensor] = None,
    max_seqlen_q: Optional[int] = None,
    max_seqlen_k: Optional[int] = None,
    min_seqlen_k: Optional[int] = None,
    page_table: Optional[torch.Tensor] = None,
    softmax_scale: Optional[float] = None,
    causal: bool = False,
    softcap: Optional[float] = None,
    window_size_left: Optional[int] = None,
    window_size_right: Optional[int] = None,
    learnable_sink: Optional[torch.Tensor] = None,
    tile_mn: Optional[Tuple[int, int]] = None,
    mma_pv_is_rs: Optional[bool] = None,
    intra_wg_overlap: Optional[bool] = None,
    num_threads: int = 384,
    num_splits: int = 1,
    pack_gqa: Optional[bool] = None,
    _arch: Optional[int] = None,
    score_mod: Optional[Callable] = None,
    mask_mod: Optional[Callable] = None,
    block_sparse_tensors: Optional[BlockSparseTensorsTorch] = None,
    return_lse: bool = False,
    out: Optional[torch.Tensor] = None,
    lse: Optional[torch.Tensor] = None,
    aux_tensors: Optional[list[torch.Tensor]] = None,
    aux_scalars: Optional[tuple] = None,
    q_descale: Optional[torch.Tensor] = None,
    k_descale: Optional[torch.Tensor] = None,
    v_descale: Optional[torch.Tensor] = None,
    gather_kv_indices: Optional[torch.Tensor] = None,
    _prepare_only: bool = False,
) -> Optional[
    Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]
]:
    """Forward pass for FlashAttention.

    Args:
        ...
        score_mod: A callable that takes the attention scores and applies a modification.
        mask_mod: A callable that takes token position information and selectively masks
        block_sparse_tensors: A tuple of tensors used for block sparsity.
        return_lse: Whether to return the log softmax of the attention scores. If set to True will always calculate
            The returned LSE supports taking gradient.
        out: Optional pre-allocated output tensor. If None, will be allocated internally.
        lse: Optional pre-allocated log-sum-exp tensor. If None, will be allocated when needed.
        aux_tensors: Some score_mods will want to read from global aux_tensors. This is how we thread them through to the inner kernel.
        aux_scalars: Runtime scalar captures used by score_mod or mask_mod.
    """
    aux_scalars = tuple(aux_scalars) if aux_scalars else None
    q, k, v, qv = [maybe_contiguous(t) for t in (q, k, v, qv)]
    assert q is not None or qv is not None
    assert v is not None
    q_descale, k_descale, v_descale = [maybe_contiguous(t) for t in (q_descale, k_descale, v_descale)]
    q_shape = q.shape if q is not None else qv.shape
    num_head, head_dim = q_shape[-2:]
    if cu_seqlens_q is None:
        batch_size, seqlen_q = q_shape[:2]
        total_q = batch_size * seqlen_q
    else:
        batch_size = cu_seqlens_q.shape[0] - 1
        seqlen_q = None
        total_q = q_shape[0]
    if page_table is not None:
        assert cu_seqlens_k is None, "page_table is not supported with cu_seqlens_k"
        assert page_table.dtype == torch.int32, "page_table must be int32"
        assert page_table.stride(-1) == 1, "page_table must be contiguous in the last dimension"
        max_num_pages_per_seq = page_table.shape[1]
        assert page_table.shape == (batch_size, max_num_pages_per_seq)
        num_pages, page_size = v.shape[:2]
        seqlen_k = num_pages * page_size
    else:
        num_pages, page_size = None, None
        seqlen_k = v.shape[-3]
    num_head_kv = v.shape[-2]
    head_dim_v = v.shape[-1]
    if cu_seqlens_k is None:
        if page_table is None:
            assert k is None or k.shape == (batch_size, seqlen_k, num_head_kv, head_dim)
            assert v.shape == (batch_size, seqlen_k, num_head_kv, head_dim_v)
        else:
            assert k is None or k.shape == (num_pages, page_size, num_head_kv, head_dim)
            assert v.shape == (num_pages, page_size, num_head_kv, head_dim_v)
    else:
        assert k is None or k.shape == (seqlen_k, num_head_kv, head_dim)
        assert v.shape == (seqlen_k, num_head_kv, head_dim_v)
        assert cu_seqlens_k.shape == (batch_size + 1,), (
            "cu_seqlens_k must have shape (batch_size + 1,)"
        )

    if cu_seqlens_q is not None:
        assert cu_seqlens_q.shape == (batch_size + 1,), (
            "cu_seqlens_q must have shape (batch_size + 1,)"
        )
    assert seqused_q is None or seqused_q.shape == (batch_size,), (
        "seqused_q must have shape (batch_size,)"
    )
    assert seqused_k is None or seqused_k.shape == (batch_size,), (
        "seqused_k must have shape (batch_size,)"
    )
    assert v.dtype in [torch.float16, torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2], (
        "inputs must be float16, bfloat16, fp8 e4m3fn, or fp8 e5m2"
    )

    q_dtype = q.dtype if q is not None else qv.dtype
    assert k is None or k.dtype == v.dtype, (
        f"K/V dtype mismatch: {k.dtype} != {v.dtype}"
    )
    mixed_bf16_kv_fp8_q = (
        q is not None
        and q_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
        and v.dtype == torch.bfloat16
    )
    assert mixed_bf16_kv_fp8_q or q_dtype == v.dtype, (
        f"unsupported Q/KV dtype combination: {q_dtype}/{v.dtype}"
    )

    for t in [cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k]:
        if t is not None:
            assert t.dtype == torch.int32, (
                "cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k must be int32"
            )
            assert t.stride(0) == 1, (
                "cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k must be contiguous"
            )
    if learnable_sink is not None:
        assert learnable_sink.shape == (num_head,)
        assert learnable_sink.dtype == torch.bfloat16, "learnable_sink must be bfloat16"

    if not is_fake_mode():
        assert all(
            t is None or t.is_cuda
            for t in (
                q,
                k,
                v,
                qv,
                q_descale,
                k_descale,
                v_descale,
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_q,
                seqused_k,
                page_table,
                learnable_sink,
            )
        ), "inputs must be on CUDA device"
    arch = _get_device_arch(v.device.index) if _arch is None else _arch
    assert arch in (100, 103), "Atrex FA4 prefill only supports SM100 and SM103"
    assert num_head % num_head_kv == 0, "num_head must be divisible by num_head_kv"
    if softmax_scale is None:
        softmax_scale = (
            1.0 / math.sqrt(head_dim) if qv is None or q is None
            else 1.0 / math.sqrt(head_dim + head_dim_v)
        )
    if softcap == 0.0:
        softcap = None
    qhead_per_kvhead = num_head // num_head_kv
    if pack_gqa is None:
        pack_gqa = qhead_per_kvhead > 1

    is_fp8 = q_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    requires_grad = any(t is not None and t.requires_grad for t in [q, k, v, qv])
    if is_fp8 and requires_grad:
        raise NotImplementedError("FA4 CuTe FP8 backward is not supported yet (forward-only).")
    assert (
        head_dim == 256
        and head_dim_v == 256
        and v.dtype in (torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2)
    ), (
        "Atrex FA4 prefill only supports BF16 or FP8 HD256 on SM100/SM103 with the 2CTA kernel"
    )
    out_torch_dtype = torch.bfloat16 if is_fp8 else q_dtype
    device = v.device
    q_batch_seqlen_shape = (batch_size, seqlen_q) if cu_seqlens_q is None else (total_q,)

    if qv is None:
        lse_shape = (batch_size, num_head, seqlen_q) if cu_seqlens_q is None else (num_head, total_q)
    else:
        # num_head contiguous better for MQA in MLA absorbed
        lse_shape = (batch_size, seqlen_q, num_head) if cu_seqlens_q is None else (total_q, num_head)

    if out is None:
        out = torch.empty(
            *q_batch_seqlen_shape, num_head, head_dim_v, dtype=out_torch_dtype, device=device
        )
    else:
        _validate_tensor(out, "out", (*q_batch_seqlen_shape, num_head, head_dim_v), out_torch_dtype, device)

    if lse is None:
        lse = (
            torch.empty(lse_shape, dtype=torch.float32, device=device)
            if requires_grad or return_lse
            else None
        )
    elif lse is not None:
        _validate_tensor(lse, "lse", lse_shape, torch.float32, device)

    if seqlen_k == 0 or total_q == 0:
        out.zero_()
        if lse is not None:
            lse.fill_(float("-inf"))
        return out, lse, None, None

    if is_fp8:
        for t, name in ((q_descale, "q_descale"), (k_descale, "k_descale"), (v_descale, "v_descale")):
            if t is not None:
                _validate_tensor(t, name, (batch_size, num_head_kv), torch.float32, device)
    else:
        assert q_descale is None and k_descale is None and v_descale is None, (
            "q_descale/k_descale/v_descale are only supported for FP8 inputs"
        )

    dtype = torch2cute_dtype_map[q_dtype]
    use_block_sparsity = block_sparse_tensors is not None

    causal, local, window_size_left, window_size_right = _resolve_causal_local_window(
        causal, window_size_left, window_size_right, mask_mod
    )

    current_stream = cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True)

    # BF16 hd256 needs a deeper K/V pipeline than tile_n=128 can fit alongside
    # the Q/O buffers (the 64-wide layout has capacity for six stages and uses
    # the tuned five-stage cap, versus three 128-wide stages). On SM103 the
    # 128x64 tile is faster at both short and long causal-varlen shapes. FP8
    # keeps its tuned 128x128 tile.
    tile_m, tile_n = (128, 128 if is_fp8 else 64)
    if tile_mn is not None:
        tile_m, tile_n = tile_mn
    if mma_pv_is_rs is None:
        mma_pv_is_rs = True
    if intra_wg_overlap is None:
        intra_wg_overlap = True

    assert causal and not local, "Atrex FA4 HD256 prefill requires causal attention"
    assert num_splits == 1, "Atrex FA4 HD256 2CTA prefill does not support split-KV"
    assert cu_seqlens_q is not None, "Atrex FA4 HD256 prefill requires varlen Q"
    assert cu_seqlens_k is not None or page_table is not None, (
        "Atrex FA4 HD256 prefill requires varlen or paged K/V"
    )
    input_variant = "fp8" if is_fp8 else "bf16"
    fwd_kernel_variant = f"hd256_{input_variant}_2cta_sm{arch}_clc_v1_default"
    q_stage = 1

    use_2cta_instrs = True

    if softcap is not None:
        assert score_mod is None, "softcap and score_mod cannot be used together"
        score_mod = utils.create_softcap_scoremod(softcap)

    # hash score and mask mods for compile cache
    score_mod_hash = utils.hash_callable(score_mod) if score_mod is not None else False
    mask_mod_hash = utils.hash_callable(mask_mod) if mask_mod is not None else False

    if use_block_sparsity:
        # NB: pack_gqa requires block sparse head dim == 1 (broadcasted)
        head_dim_idx = 0 if block_sparse_tensors.mask_block_cnt.ndim == 2 else 1
        if pack_gqa and block_sparse_tensors.mask_block_cnt.shape[head_dim_idx] != 1:
            pack_gqa = False
        if cu_seqlens_q is not None:
            assert block_sparse_tensors.cu_total_m_blocks is not None, (
                "Varlen block sparsity requires block_sparse_tensors.cu_total_m_blocks."
            )

    # See get_broadcast_dims for why this is needed in compile key
    block_sparse_broadcast_pattern = None
    normalized_block_sparse_tensors = None
    q_subtile_factor = 1
    if block_sparse_tensors is not None:
        (
            normalized_block_sparse_tensors,
            block_sparse_broadcast_pattern,
            q_subtile_factor,
        ) = normalize_block_sparse_config(
            block_sparse_tensors,
            batch_size=batch_size,
            num_head=num_head,
            seqlen_q=seqlen_q,
            seqlen_k=seqlen_k,
            block_size=(tile_m, tile_n),
            q_stage=q_stage,
        )
    if aux_tensors is not None:
        aux_tensor_metadata = get_aux_tensor_metadata(aux_tensors)
    else:
        aux_tensor_metadata = None
    aux_scalar_metadata = tuple(type(s) for s in aux_scalars) if aux_scalars is not None else None

    if qv is not None:
        assert arch // 10 in [10, 11], "only support Blackwell arch with qv"
        assert q is None or qv.shape[:-1] == q.shape[:-1]
        assert qv.shape[-1] == head_dim_v
        assert head_dim_v == 512
        assert q is None or head_dim == 64
        assert not local, "local not yet supported with qv"
        assert q_descale is None and k_descale is None and v_descale is None, (
            "q_descale/k_descale/v_descale are not yet supported with qv"
        )
        assert tile_n == 128

        assert learnable_sink is None
        assert softcap is None
        assert score_mod is None
        assert mask_mod is None

        if page_table is not None:
            assert gather_kv_indices is None, "paged KV + topk sparsity not yet supported together"

        qv = maybe_contiguous(qv)

        gather_kv_length = 2048  # dummy value
        sparse_kv = gather_kv_indices is not None
        # always use kv bitmask by default (handles -1 sentinel)
        disable_sparse_kv_bitmask = False
        if sparse_kv:
            assert gather_kv_indices.shape[:-1] == qv.shape[:-2]
            gather_kv_length = gather_kv_indices.shape[-1]
            assert gather_kv_length % 128 == 0
            # if min_seqlen_k is None or causal:
            #     disable_sparse_kv_bitmask = False
            # else:
            #     # seqlen_k_boundary = min_seqlen_k - max_seqlen_q + 1 if causal else min_seqlen_k
            #     seqlen_k_boundary = min_seqlen_k
            #     disable_sparse_kv_bitmask = seqlen_k_boundary >= gather_kv_length

        if requires_grad and sparse_kv:
            if cu_seqlens_q is None:
                p = torch.empty(batch_size, seqlen_q, num_head, gather_kv_length, dtype=q_dtype, device=device)
                row_max = torch.empty(batch_size, seqlen_q, gather_kv_length//128, num_head, dtype=torch.float32, device=device)
            else:
                p = torch.empty(total_q, num_head, gather_kv_length, dtype=q_dtype, device=device)
                row_max = torch.empty(total_q, gather_kv_length//128, num_head, dtype=torch.float32, device=device)
        else:
            p = row_max = None
    else:
        assert gather_kv_indices is None, "gather_kv_indices is only supported with qv"
        gather_kv_length = None
        sparse_kv = None
        disable_sparse_kv_bitmask = None
        p = row_max = None

    paged_kv_small_page = page_size is not None and page_size % tile_n != 0
    compile_key = (
        device.index,
        dtype,
        v.dtype,
        head_dim,
        head_dim_v,
        qhead_per_kvhead,
        num_head_kv,
        causal,
        score_mod_hash,
        mask_mod_hash,
        use_block_sparsity,
        block_sparse_broadcast_pattern,
        aux_tensor_metadata,
        aux_scalar_metadata,
        lse is None,
        cu_seqlens_q is None,
        cu_seqlens_k is None,
        seqused_q is None,
        seqused_k is None,
        page_table is not None,
        window_size_left is not None,
        window_size_right is not None,
        learnable_sink is not None,
        q_descale is not None,
        k_descale is not None,
        v_descale is not None,
        block_sparse_tensors is None or block_sparse_tensors.cu_total_m_blocks is None,
        block_sparse_tensors is None or block_sparse_tensors.cu_block_idx_offsets is None,
        tile_m,
        tile_n,
        q_stage,
        num_threads,
        pack_gqa,
        arch,
        paged_kv_small_page,
        page_size,
        use_2cta_instrs,
        q_subtile_factor,
        mma_pv_is_rs,
        intra_wg_overlap,
        fwd_kernel_variant,
        q is not None,
        qv is not None,
        p is not None,
        row_max is not None,
        gather_kv_length,
        sparse_kv,
        disable_sparse_kv_bitmask,
        fa_logging.get_fa_log_level(),
    )

    if compile_key not in _flash_attn_fwd.compile_cache:
        (
            cu_seqlens_q_tensor,
            cu_seqlens_k_tensor,
            seqused_q_tensor,
            seqused_k_tensor,
            learnable_sink_tensor,
        ) = [
            to_cute_tensor(t, assumed_align=4, leading_dim=0)
            if t is not None
            else None
            for t in (cu_seqlens_q, cu_seqlens_k, seqused_q, seqused_k, learnable_sink)
        ]
        page_table_tensor = (
            to_cute_tensor(page_table, assumed_align=4, leading_dim=1)
            if page_table is not None
            else None
        )
        q_tensor, k_tensor, v_tensor, o_tensor = [
            to_cute_tensor(t) for t in (q, k, v, out)
        ]
        lse_tensor = to_cute_tensor(lse, assumed_align=4)

        q_descale_tensor, k_descale_tensor, v_descale_tensor = (
            to_cute_tensor(t, assumed_align=4, leading_dim=1)
            for t in (q_descale, k_descale, v_descale)
        )
        descale_tensors_tensor = (
            DescaleTensors(
                q_descale=q_descale_tensor,
                k_descale=k_descale_tensor,
                v_descale=v_descale_tensor,
            )
            if q_descale_tensor is not None
            or k_descale_tensor is not None
            or v_descale_tensor is not None
            else None
        )

        sparse_tensors = None
        if normalized_block_sparse_tensors is not None:
            sparse_tensors = to_cute_block_sparse_tensors(normalized_block_sparse_tensors)

        cute_aux_tensors = None
        aux_tensor_metadata = None
        if aux_tensors is not None:
            cute_aux_tensors = [to_cute_aux_tensor(buf) for buf in aux_tensors]

        qv_tensor = to_cute_tensor(qv)
        gather_kv_indices_tensor = to_cute_tensor(gather_kv_indices)
        p_tensor = to_cute_tensor(p)
        row_max_tensor = to_cute_tensor(row_max)

        fa_fwd = FlashAttentionForwardHd256_2CTA_Sm103(
            head_dim,
            head_dim_v,
            qhead_per_kvhead=qhead_per_kvhead,
            is_causal=causal,
            is_local=local,
            is_split_kv=False,
            pack_gqa=pack_gqa,
            m_block_size=tile_m,
            n_block_size=tile_n,
            q_stage=q_stage,
            is_persistent=not causal
                and not local
                and cu_seqlens_q is None
                and seqused_q is None,
            score_mod=score_mod,
            mask_mod=mask_mod,
            has_aux_tensors=aux_tensors is not None,
            paged_kv_non_tma=paged_kv_small_page,
            paged_kv_page_size=page_size,
            is_varlen_q=cu_seqlens_q is not None or seqused_q is not None,
            q_subtile_factor=q_subtile_factor,
            use_2cta_instrs=use_2cta_instrs,
            dedicated_clc_warp=False,
            register_config=None,
        )
        fa_fwd.atrex_sm103_prefill_kernel.set_name_prefix("atrex")
        # TODO: check @can_implement
        if qv is not None:
            with use_filesystem_cutlass_dsl_version_hash():
                compiled_kernel = cute.compile(
                    fa_fwd,
                    q_tensor,
                    qv_tensor,
                    k_tensor,
                    v_tensor,
                    o_tensor,
                    lse_tensor,
                    softmax_scale,
                    p_tensor,
                    row_max_tensor,
                    cu_seqlens_q_tensor,
                    cu_seqlens_k_tensor,
                    seqused_q_tensor,
                    seqused_k_tensor,
                    gather_kv_indices_tensor,
                    page_table_tensor,
                    window_size_left,
                    window_size_right,
                    current_stream,
                    options="--enable-tvm-ffi",
                )
            _flash_attn_fwd.compile_cache[compile_key] = compiled_kernel
        else:
            compile_args = [
                fa_fwd,
                q_tensor,
                k_tensor,
                v_tensor,
                o_tensor,
                lse_tensor,
                softmax_scale,
                cu_seqlens_q_tensor,
                cu_seqlens_k_tensor,
                seqused_q_tensor,
                seqused_k_tensor,
                page_table_tensor,
                window_size_left,
                window_size_right,
                learnable_sink_tensor,
            ]
            if arch // 10 in [10, 11]:
                compile_args.append(descale_tensors_tensor)
            compile_args.extend([
                sparse_tensors,
                AuxData(cute_aux_tensors, aux_scalars),
            ])
            compile_args.append(current_stream)
            with use_filesystem_cutlass_dsl_version_hash():
                compiled_kernel = cute.compile(
                    *compile_args, options="--enable-tvm-ffi"
                )
            _flash_attn_fwd.compile_cache[compile_key] = compiled_kernel

    if _prepare_only:
        assert not is_fake_mode(), "Prepared execution requires a CUDA device"
        assert q is not None and qv is None, "Prepared execution requires standard Q/K/V"
        assert lse is None, "Prepared execution does not support auxiliary outputs"
        # FP8 preparation compiles the same descale-bearing signature as the
        # subsequent launch; validation already happened before compilation.
        assert normalized_block_sparse_tensors is None and aux_tensors is None and not aux_scalars
        assert window_size_left is None and window_size_right is None and learnable_sink is None
        return None

    if not is_fake_mode():
        q_call, k_call, v_call, qv_call = [
            t.detach() if t is not None else None
            for t in (q, k, v, qv)
        ]
        if is_fp8:
            # need uint8 workaround until we pin torch >= 2.11.0 where fp8 export is supported
            q_call, k_call, v_call, qv_call = [
                t.view(torch.uint8)
                if t is not None
                and t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
                else t
                for t in (q_call, k_call, v_call, qv_call)
            ]
        descale_tensors = (
            DescaleTensors(q_descale=q_descale, k_descale=k_descale, v_descale=v_descale)
            if q_descale is not None or k_descale is not None or v_descale is not None
            else None
        )
        if qv is not None:
            _flash_attn_fwd.compile_cache[compile_key](
                q_call,
                qv_call,
                k_call,
                v_call,
                out.detach(),
                lse,
                softmax_scale,
                p,
                row_max,
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_q,
                seqused_k,
                gather_kv_indices,
                page_table,
                window_size_left,
                window_size_right,
            )
        else:
            call_args = [
                q_call,
                k_call,
                v_call,
                out.detach(),
                lse,
                softmax_scale,
                cu_seqlens_q,
                cu_seqlens_k,
                seqused_q,
                seqused_k,
                page_table,
                window_size_left,
                window_size_right,
                learnable_sink,
            ]
            if arch // 10 in [10, 11]:
                call_args.append(descale_tensors)
            call_args.extend([
                (
                    normalized_block_sparse_tensors.mask_block_cnt,
                    normalized_block_sparse_tensors.mask_block_idx,
                    normalized_block_sparse_tensors.full_block_cnt,
                    normalized_block_sparse_tensors.full_block_idx,
                    normalized_block_sparse_tensors.cu_total_m_blocks,
                    normalized_block_sparse_tensors.cu_block_idx_offsets,
                    normalized_block_sparse_tensors.dq_write_order,
                    normalized_block_sparse_tensors.dq_write_order_full,
                )
                if normalized_block_sparse_tensors is not None
                else None,
                AuxData(aux_tensors, aux_scalars),
            ])
            _flash_attn_fwd.compile_cache[compile_key](*call_args)
    return out, lse, p, row_max


_flash_attn_fwd.compile_cache = get_jit_cache("fwd")
