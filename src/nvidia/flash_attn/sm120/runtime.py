# Copyright (c) 2025, Jay Shah, Ganesh Bikshandi, Ying Zhang, Vijay Thakkar, Pradeep Ramani, Tri Dao.

import os
import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional, Tuple, Callable

import torch

import cutlass
import cutlass.cute as cute
from cutlass import Int32, Float32

from quack.compile_utils import make_fake_tensor as fake_tensor
from atrex.src.nvidia.flash_attn.common_utils.cutlass_dsl_cache import (
    use_filesystem_cutlass_dsl_version_hash,
)
from flash_attn.cute.cache_utils import get_jit_cache
from flash_attn.cute.testing import is_fake_mode


if os.environ.get("CUTE_DSL_PTXAS_PATH", None) is not None:
    from flash_attn.cute import cute_dsl_ptxas  # noqa: F401

    # Patch to dump ptx and then use system ptxas to compile to cubin
    cute_dsl_ptxas.patch()


from flash_attn.cute import utils
from flash_attn.cute import fa_logging
from flash_attn.cute.cute_dsl_utils import (
    get_aux_tensor_metadata,
    to_cute_aux_tensor,
    to_cute_tensor,
)
from atrex.src.nvidia.flash_attn.sm120.prefill_cutedsl import FlashAttentionForwardSm120
from flash_attn.cute.flash_fwd_combine import FlashAttentionForwardCombine

from flash_attn.cute.utils import AuxData
from flash_attn.cute.block_sparsity import (
    BlockSparseTensorsTorch,
    to_cute_block_sparse_tensors,
    normalize_block_sparse_config,
)


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


@dataclass(frozen=True)
class FwdConfig:
    m_block_size: int
    n_block_size: int
    mma_pv_is_rs: bool
    intra_wg_overlap: bool


def _sm120_paged_tile_n(page_size: int, candidates: tuple[int, ...]) -> int:
    """Pick the first validated N tile contained by one physical KV page."""
    tile_n = next((n for n in candidates if page_size % n == 0), None)
    if tile_n is None:
        raise ValueError(
            f"Atrex FA SM120 does not support paged KV page_size={page_size}; "
            f"page_size must be divisible by one of {candidates}"
        )
    return tile_n


def maybe_contiguous(x):
    return x.contiguous() if x is not None and x.stride(-1) != 1 else x


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
    # ATREX PORT sm120: explicit params (not module globals) for the split-KV/compact-grid path.
    # num_splits_dynamic: int32 CUDA tensor [num_batch], per-sequence split count (DYNSPLIT).
    # work_map: int32 CUDA tensor [rows,4] = (m_block, head, batch, split), or the Q1/HKV1
    # compact form [rows,1] = ((batch << 16) | split), host work map (COMPACTGRID);
    #   when given the grid launches exactly `rows` CTAs. Both sm120-only; ignored on other arches.
    num_splits_dynamic: Optional[torch.Tensor] = None,
    work_map: Optional[torch.Tensor] = None,
    _runtime_balanced_splits: bool = False,
    _runtime_balanced_grid_size: int = 0,
    _runtime_balanced_batch_size: int = 0,
    _allow_pack_gqa_split: bool = False,
    _sm120_num_stages: Optional[int] = None,
    _sm120_q_in_regs: Optional[bool] = None,
    _sm120_num_threads: Optional[int] = None,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
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
    if qv is not None:
        raise NotImplementedError("Atrex SM120 attention does not support auxiliary Q/MLA")
    aux_scalars = tuple(aux_scalars) if aux_scalars else None
    q, k, v = [maybe_contiguous(t) for t in (q, k, v)]
    assert q is not None
    assert v is not None
    q_descale, k_descale, v_descale = [maybe_contiguous(t) for t in (q_descale, k_descale, v_descale)]
    q_shape = q.shape
    num_head, head_dim = q_shape[-2:]
    if cu_seqlens_q is None:
        batch_size, seqlen_q = q_shape[:2]
        total_q = batch_size * seqlen_q
    else:
        batch_size = cu_seqlens_q.shape[0] - 1
        seqlen_q = None
        total_q = q_shape[0]
    if total_q == 0:
        q_dtype = q.dtype
        out_dtype = (
            torch.bfloat16
            if q_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
            else q_dtype
        )
        if out is None:
            out = torch.empty(
                *q_shape[:-1], v.shape[-1], dtype=out_dtype, device=v.device
            )
        if lse is None and return_lse:
            lse_shape = (
                (batch_size, num_head, seqlen_q)
                if cu_seqlens_q is None
                else (num_head, total_q)
            )
            lse = torch.empty(lse_shape, dtype=torch.float32, device=v.device)
        return out, lse, None, None
    if page_table is not None:
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
    if page_table is not None:
        assert k is None or k.shape == (num_pages, page_size, num_head_kv, head_dim)
        assert v.shape == (num_pages, page_size, num_head_kv, head_dim_v)
    elif cu_seqlens_k is None:
        assert k is None or k.shape == (batch_size, seqlen_k, num_head_kv, head_dim)
        assert v.shape == (batch_size, seqlen_k, num_head_kv, head_dim_v)
    else:
        assert k is None or k.shape == (seqlen_k, num_head_kv, head_dim)
        assert v.shape == (seqlen_k, num_head_kv, head_dim_v)
    if cu_seqlens_k is not None:
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

    q_dtype = q.dtype
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
    if arch != 120:
        raise NotImplementedError(
            f"the SM120 FlashAttention runtime cannot compile for arch={arch}"
        )
    assert num_head % num_head_kv == 0, "num_head must be divisible by num_head_kv"
    if softmax_scale is None:
        softmax_scale = 1.0 / math.sqrt(head_dim)
    if softcap == 0.0:
        softcap = None
    qhead_per_kvhead = num_head // num_head_kv
    if pack_gqa is None:
        pack_gqa = qhead_per_kvhead > 1

    is_fp8 = q_dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
    requires_grad = any(t is not None and t.requires_grad for t in [q, k, v])
    if is_fp8 and requires_grad:
        raise NotImplementedError("FA4 CuTe FP8 backward is not supported yet (forward-only).")
    # SM120 uses warp-level mma.sync for BF16 and FP8 E4M3/E5M2.
    if not (
        (head_dim == head_dim_v == 256
         or (head_dim == head_dim_v == 128 and v.dtype in (
             torch.bfloat16, torch.float8_e4m3fn
         )))
        and v.dtype
        in (torch.bfloat16, torch.float8_e4m3fn, torch.float8_e5m2)
    ):
        raise NotImplementedError(
            "Atrex SM120 attention supports BF16/FP8 E4M3 HD128 and BF16/FP8 HD256"
        )
    out_torch_dtype = torch.bfloat16 if is_fp8 else q_dtype
    device = v.device
    q_batch_seqlen_shape = (batch_size, seqlen_q) if cu_seqlens_q is None else (total_q,)

    lse_shape = (batch_size, num_head, seqlen_q) if cu_seqlens_q is None else (num_head, total_q)

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

    # SM80/SM120 uses warp MMA. Native page128 Q1 uses a compact M16 tile and
    # only needs two warps; all other paths keep the four-warp configuration.
    num_threads = 64 if tile_mn == (16, 32) else 128

    # BF16 hd256 needs a deeper K/V pipeline than tile_n=128 can fit alongside
    # the Q/O buffers (the 64-wide layout has capacity for six stages and uses
    # the tuned five-stage cap, versus three 128-wide stages). On SM103 the
    # 128x64 tile is faster at both short and long causal-varlen shapes. FP8
    # keeps its tuned 128x128 tile.
    fwd_cfg = FwdConfig(128, 128, True, True)
    # num_stages / Q_in_regs are constructor parameters of the SM120 kernel.
    sm120_num_stages = 1
    sm120_q_in_regs = False
    if tile_mn is None:
        # Validated hd256 policy (48-shape sweep): prefill/prefix-cache
        # (causal) = 64x64 stages=1; decode = 64x32 stages=2.
        if causal:
            # A paged tile must not cross a physical page because the kernel
            # resolves one page-table entry per N tile.
            causal_tile_n = 32 if is_fp8 else 64
            if page_size is not None:
                causal_tile_n = _sm120_paged_tile_n(
                    page_size,
                    (32, 16) if is_fp8 else (64, 48, 32, 16),
                )
            fwd_cfg = FwdConfig(64, causal_tile_n, True, True)
            sm120_num_stages = 1
        else:
            decode_tile_n = 32
            if page_size is not None:
                decode_tile_n = _sm120_paged_tile_n(page_size, (32, 16))
            fwd_cfg = FwdConfig(64, decode_tile_n, True, True)
            sm120_num_stages = 2
            if (
                head_dim == 128
                and num_head == 10
                and num_head_kv == 2
                and max_seqlen_q is not None
                and max_seqlen_q >= 256
                and total_q >= 8192
            ):
                # HD128 HQ10/HKV2 Q8K/K64K: 13.54ms -> 11.87ms in the
                # same-GPU sweep; two smaller-SMEM CTAs feed warp MMA.
                if is_fp8:
                    # Two M16 fragments per warp reuse the same K/V tile.
                    # M256/N64 with two stages reduces long-context traffic.
                    prefill_tile_n = _sm120_paged_tile_n(page_size, (64, 32))
                    fwd_cfg = FwdConfig(256, prefill_tile_n, True, True)
                    num_threads = 256
                else:
                    fwd_cfg = FwdConfig(128, decode_tile_n, True, True)
                sm120_num_stages = 2 if is_fp8 else 1
    else:
        fwd_cfg = FwdConfig(tile_mn[0], tile_mn[1], fwd_cfg.mma_pv_is_rs, fwd_cfg.intra_wg_overlap)
        if (
            is_fp8
            and causal
            and max_seqlen_q == 1
            and page_size == 64
            and num_head_kv == 1
        ):
            sm120_num_stages = 2
    tile_m, tile_n = fwd_cfg.m_block_size, fwd_cfg.n_block_size
    if head_dim == 128:
        if _sm120_num_stages is not None:
            if _sm120_num_stages < 1:
                raise ValueError("SM120 stages must be positive")
            sm120_num_stages = _sm120_num_stages
        if _sm120_q_in_regs is not None:
            sm120_q_in_regs = _sm120_q_in_regs
        if _sm120_num_threads is not None:
            if _sm120_num_threads not in (64, 128, 256):
                raise ValueError("SM120 attention supports 64, 128 or 256 threads")
            num_threads = _sm120_num_threads
    if page_size is not None and page_size % tile_n != 0:
        raise ValueError(
            f"Atrex FA SM120 requires page_size ({page_size}) to be divisible "
            f"by tile_n ({tile_n})"
        )
    if mma_pv_is_rs is None:
        mma_pv_is_rs = fwd_cfg.mma_pv_is_rs
    if intra_wg_overlap is None:
        intra_wg_overlap = fwd_cfg.intra_wg_overlap

    # Prefill is causal varlen; decode is non-causal packed GQA with SplitKV.
    assert not local, "Atrex FA4 sm120 does not support local/sliding-window attention"
    assert cu_seqlens_k is not None or page_table is not None, (
        "Atrex FA4 sm120 requires varlen or paged K/V"
    )
    assert num_splits <= 256, "num_splits must be <= 256 (combine kernel limit)"
    if num_splits_dynamic is not None:
        _validate_tensor(
            num_splits_dynamic,
            "num_splits_dynamic",
            (batch_size,),
            torch.int32,
            device,
        )
    if work_map is not None:
        if (
            work_map.device != device
            or work_map.dtype != torch.int32
            or work_map.ndim != 2
            or work_map.shape[1] not in (1, 4)
            or not work_map.is_contiguous()
        ):
            raise ValueError(
                "SM120 work_map must be contiguous int32 CUDA [rows, 1|4]"
            )
        if work_map.shape[1] == 1 and not (
            num_splits_dynamic is not None
            and cu_seqlens_q is not None
            and max_seqlen_q in (1, 4)
            and total_q == batch_size * max_seqlen_q
            and num_head_kv == 1
        ):
            raise ValueError(
                "the compact short-Q work map requires ragged Q1/Q4, one KV head, "
                "and dynamic split counts"
            )
    # Split-KV and packed GQA can coexist only through the opt-in scatter path.
    is_split_kv = num_splits > 1
    _pgsplit = (
        _allow_pack_gqa_split
        or os.environ.get("ATREX_FA4_PGSPLIT", "0") == "1"
    )
    if is_split_kv and not (_pgsplit and pack_gqa):
        pack_gqa = False
    q_stage = 1
    use_2cta_instrs = False
    input_variant = "fp8" if is_fp8 else "bf16"
    fwd_kernel_variant = (
        f"hd{head_dim}_{input_variant}_sm120_v1{'_split' if is_split_kv else ''}"
        f"{'_pg' if (is_split_kv and pack_gqa) else ''}"
    )

    use_lpt = (
        causal
        and not is_split_kv
        and max_seqlen_q is not None
        and max_seqlen_k is not None
        and 2 * max_seqlen_q >= max_seqlen_k
    )
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

    assert gather_kv_indices is None, "gather_kv_indices is only supported with qv"
    gather_kv_length = None
    sparse_kv = None
    disable_sparse_kv_bitmask = None
    p = row_max = None

    # ATREX PORT sm120: fp32 partial buffers for the split-KV combine. Partials MUST be fp32 (the
    # combine kernel's bf16-partial path faults on this build). Keep them invocation-local so eager
    # execution can reuse storage through PyTorch's caching allocator and CUDA graphs can own the
    # storage through their private pools. The kernels write every launched split; compact grids
    # initialize LSE for slots that are not launched, and the dynamic short-Q reducer never reads
    # O for a split outside the current request's runtime count.
    out_partial = lse_partial = runtime_num_splits = None
    kernel_num_splits_dynamic = num_splits_dynamic
    if is_split_kv:
        out_partial = torch.empty(
            num_splits,
            *q_batch_seqlen_shape,
            num_head,
            head_dim_v,
            dtype=torch.float32,
            device=device,
        )
        lse_partial = torch.empty(
            num_splits,
            *lse_shape,
            dtype=torch.float32,
            device=device,
        )
        if _runtime_balanced_splits:
            runtime_num_splits = torch.empty(
                batch_size, dtype=torch.int32, device=device
            )
            kernel_num_splits_dynamic = runtime_num_splits
        use_short_q_dynamic_reducer = (
            cu_seqlens_q is not None
            and max_seqlen_q in (1, 2, 3, 4)
            and total_q == batch_size * max_seqlen_q
            and lse is None
            and num_splits <= 256
            and kernel_num_splits_dynamic is not None
        )
        if work_map is not None and not use_short_q_dynamic_reducer:
            lse_partial.fill_(float("-inf"))

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
        sm120_num_stages,
        sm120_q_in_regs,
        is_split_kv,
        pack_gqa,
        use_lpt,
        arch,
        page_size,
        # ATREX PORT sm120: DYNSPLIT / COMPACTGRID are constexpr in the kernel; omitting them would
        # let a compact-grid launch reuse a non-compact cubin (wrong grid decode). page-aligned
        # Paged-vs-not is already keyed by page_size above.
        kernel_num_splits_dynamic is not None if is_split_kv else False,
        work_map.shape[1] if work_map is not None else 0,
        _runtime_balanced_splits,
        _runtime_balanced_grid_size,
        _runtime_balanced_batch_size,
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

    if arch == 120 and is_fp8 and head_dim == 256:
        # CuTe specializes zero strides in the normalized descale tensors.
        # Presence alone cannot distinguish a broadcast batch from compact
        # per-sequence scales with otherwise identical launch metadata.
        compile_key += tuple(
            None if tensor is None else tuple(tensor.stride())
            for tensor in (q_descale, k_descale, v_descale)
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
        # ATREX PORT sm120: per-sequence split counts (DYNSPLIT) and host work map (COMPACTGRID).
        nsd_tensor = (
            to_cute_tensor(
                kernel_num_splits_dynamic, assumed_align=4, leading_dim=0
            )
            if kernel_num_splits_dynamic is not None
            else None
        )
        wm_tensor = (
            to_cute_tensor(work_map, assumed_align=4, leading_dim=1)
            if work_map is not None
            else None
        )
        q_tensor, k_tensor, v_tensor, o_tensor = [
            to_cute_tensor(t) for t in (q, k, v, out if not is_split_kv else out_partial)
        ]
        if is_split_kv:
            lse_tensor = to_cute_tensor(lse_partial, assumed_align=4)
        else:
            lse_tensor = to_cute_tensor(lse, assumed_align=4)

        q_descale_tensor, k_descale_tensor, v_descale_tensor = (
            to_cute_tensor(t, assumed_align=4, leading_dim=1)
            for t in (q_descale, k_descale, v_descale)
        )
        sparse_tensors = None
        if normalized_block_sparse_tensors is not None:
            sparse_tensors = to_cute_block_sparse_tensors(normalized_block_sparse_tensors)

        cute_aux_tensors = None
        aux_tensor_metadata = None
        if aux_tensors is not None:
            cute_aux_tensors = [to_cute_aux_tensor(buf) for buf in aux_tensors]

        if not FlashAttentionForwardSm120.can_implement(
            dtype,
            head_dim,
            head_dim_v,
            tile_m,
            tile_n,
            sm120_num_stages,
            num_threads,
            causal,
            sm120_q_in_regs,
        ):
            raise ValueError(
                "Atrex FA SM120 configuration exceeds kernel resource limits: "
                f"dtype={dtype}, head_dim={head_dim}, head_dim_v={head_dim_v}, "
                f"tile=({tile_m}, {tile_n}), stages={sm120_num_stages}, "
                f"threads={num_threads}, causal={causal}, "
                f"Q_in_regs={sm120_q_in_regs}"
            )
        fa_fwd = FlashAttentionForwardSm120(
            dtype,
            head_dim,
            head_dim_v,
            qhead_per_kvhead=qhead_per_kvhead,
            is_causal=causal,
            is_local=local,
            lpt=use_lpt,
            pack_gqa=pack_gqa,
            tile_m=tile_m,
            tile_n=tile_n,
            num_stages=sm120_num_stages,
            num_threads=num_threads,
            Q_in_regs=sm120_q_in_regs,
            score_mod=score_mod,
            mask_mod=mask_mod,
            has_aux_tensors=aux_tensors is not None,
            is_split_kv=is_split_kv,
            page_size=page_size if page_table is not None else None,
            dynamic_splits=kernel_num_splits_dynamic is not None,
            compact_grid=work_map is not None,
            compact_short_q=work_map is not None and work_map.shape[1] == 1,
            runtime_balanced_splits=_runtime_balanced_splits,
            runtime_balanced_grid_size=_runtime_balanced_grid_size,
            runtime_balanced_batch_size=_runtime_balanced_batch_size,
        )
        fa_fwd.atrex_sm120_prefill_kernel.set_name_prefix("atrex")
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
            nsd_tensor,
            wm_tensor,
            q_descale_tensor,
            k_descale_tensor,
            v_descale_tensor,
            window_size_left,
            window_size_right,
            learnable_sink_tensor,
        ]
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

    if not is_fake_mode():
        q_call, k_call, v_call = [
            t.detach() if t is not None else None
            for t in (q, k, v)
        ]
        if is_fp8:
            # need uint8 workaround until we pin torch >= 2.11.0 where fp8 export is supported
            q_call, k_call, v_call = [
                t.view(torch.uint8)
                if t is not None
                and t.dtype in (torch.float8_e4m3fn, torch.float8_e5m2)
                else t
                for t in (q_call, k_call, v_call)
            ]
        call_args = [
            q_call,
            k_call,
            v_call,
            out.detach() if not is_split_kv else out_partial,
            lse_partial if is_split_kv else lse,
            softmax_scale,
            cu_seqlens_q,
            cu_seqlens_k,
            seqused_q,
            seqused_k,
            page_table,
            kernel_num_splits_dynamic,
            work_map,
            q_descale,
            k_descale,
            v_descale,
            window_size_left,
            window_size_right,
            learnable_sink,
        ]
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
    if is_split_kv:
        use_short_q_reducer = (
            cu_seqlens_q is not None
            and max_seqlen_q in (1, 2, 3, 4)
            and total_q == batch_size * max_seqlen_q
            and lse is None
            and num_splits <= 256
            and (work_map is None or kernel_num_splits_dynamic is not None)
        )
        if use_short_q_reducer:
            from atrex.src.nvidia.flash_attn.sm120.decode_reduce import sm120_q1_reduce

            sm120_q1_reduce(
                out_partial,
                lse_partial,
                out,
                kernel_num_splits_dynamic,
                max_seqlen_q,
                # Split HD128 output columns across more SMs for the small
                # FP8 cache: the original reducer launches only 10 Q1 CTAs.
                parallel_columns=(
                    is_fp8 and head_dim == 128 and num_head == 10
                    and num_head_kv == 2 and batch_size == 1
                    and max_seqlen_k is not None and max_seqlen_k <= 65536
                    and num_splits >= 32 and kernel_num_splits_dynamic is None
                ),
            )
        else:
            _flash_attn_fwd_combine(
                out_partial,
                lse_partial.transpose(-1, -2),
                out,
                lse.transpose(-1, -2) if lse is not None else None,
                cu_seqlens_q,
                seqused_q,
            )
    return out, lse, p, row_max


_flash_attn_fwd.compile_cache = get_jit_cache("fwd")


# =====================================================================================
# ATREX PORT sm120: split-KV combine. Vendored verbatim from the flash-attn cute interface
# (flash-attn-4 provides the FlashAttentionForwardCombine kernel class; its Python wrappers live in
# the interface). Only reached on the sm120 split-KV path (is_split_kv True).
# =====================================================================================
def _compile_fwd_combine(
    dtype, dtype_partial, head_dim, tile_m, k_block_size, log_max_splits,
    has_cu_seqlens, has_seqused, has_lse, has_varlen_batch_idx,
):
    """Compile fwd combine kernel using cute fake tensors (no real GPU tensors needed)."""
    sym = cute.sym_int
    div = 128 // dtype_partial.width  # 16-byte alignment in elements

    fa_combine = FlashAttentionForwardCombine(
        dtype=dtype,
        dtype_partial=dtype_partial,
        head_dim=head_dim,
        tile_m=tile_m,
        k_block_size=k_block_size,
        log_max_splits=log_max_splits,
    )
    fa_combine.kernel.set_name_prefix("atrex")
    if not fa_combine.can_implement(
        dtype, dtype_partial, head_dim, tile_m, k_block_size, log_max_splits,
        num_threads=256,
    ):
        raise RuntimeError(
            "FlashAttention combine kernel cannot be implemented with given parameters"
        )

    if has_cu_seqlens:
        # Varlen: (num_splits, total_q, nheads, headdim)
        num_splits, total_q, nheads = sym(), sym(), sym()
        mO_partial = fake_tensor(dtype_partial, (num_splits, total_q, nheads, head_dim), divisibility=div)
        mLSE_partial = fake_tensor(Float32, (num_splits, total_q, nheads), divisibility=1, leading_dim=1)
        mO = fake_tensor(dtype, (total_q, nheads, head_dim), divisibility=div)
        mLSE = fake_tensor(Float32, (total_q, nheads), divisibility=1, leading_dim=0) if has_lse else None
    else:
        # Batched: (num_splits, batch, seqlen, nheads, headdim)
        num_splits, batch, seqlen, nheads = sym(), sym(), sym(), sym()
        mO_partial = fake_tensor(dtype_partial, (num_splits, batch, seqlen, nheads, head_dim), divisibility=div)
        mLSE_partial = fake_tensor(Float32, (num_splits, batch, seqlen, nheads), divisibility=1, leading_dim=2)
        mO = fake_tensor(dtype, (batch, seqlen, nheads, head_dim), divisibility=div)
        mLSE = fake_tensor(Float32, (batch, seqlen, nheads), divisibility=1, leading_dim=1) if has_lse else None
        batch = mO_partial.shape[1]

    batch_for_1d = batch if not has_cu_seqlens else sym()
    batchp1 = sym()
    mCuSeqlens = fake_tensor(Int32, (batchp1,), divisibility=1) if has_cu_seqlens else None
    mSeqused = fake_tensor(Int32, (batch_for_1d,), divisibility=1) if has_seqused else None
    mNumSplitsDynamic = None  # Not parametrized in compile_key
    mVarlenBatchIdx = fake_tensor(Int32, (batch_for_1d,), divisibility=1) if has_varlen_batch_idx else None
    mSemaphore = None  # Not parametrized in compile_key

    with use_filesystem_cutlass_dsl_version_hash():
        return cute.compile(
            fa_combine,
            mO_partial, mLSE_partial, mO, mLSE,
            mCuSeqlens, mSeqused, mNumSplitsDynamic, mVarlenBatchIdx, mSemaphore,
            cute.runtime.make_fake_stream(use_tvm_ffi_env_stream=True),
            options="--enable-tvm-ffi",
        )


def _flash_attn_fwd_combine(
    out_partial: torch.Tensor,
    lse_partial: torch.Tensor,
    out: torch.Tensor,
    lse: Optional[torch.Tensor] = None,
    cu_seqlens: Optional[torch.Tensor] = None,
    seqused: Optional[torch.Tensor] = None,
    num_splits_dynamic_ptr: Optional[torch.Tensor] = None,
    varlen_batch_idx: Optional[torch.Tensor] = None,
    semaphore_to_reset: Optional[torch.Tensor] = None,
) -> None:
    """Combine partial outputs / LSE from multiple split-KV splits into final out/lse."""
    assert out_partial.dtype in [torch.float16, torch.bfloat16, torch.float32], (
        "out_partial must be fp16, bf16, or fp32"
    )
    if not is_fake_mode():
        assert out_partial.is_cuda and lse_partial.is_cuda, "tensors must be on CUDA device"
    for t, name in [
        (cu_seqlens, "cu_seqlens"),
        (seqused, "seqused"),
        (num_splits_dynamic_ptr, "num_splits_dynamic_ptr"),
    ]:
        if t is not None:
            if not is_fake_mode():
                assert t.is_cuda, f"{name} must be on CUDA device"
            assert t.is_contiguous(), f"{name} must be contiguous"
    head_dim = out_partial.shape[-1]
    num_splits = out_partial.shape[0]
    assert num_splits <= 256
    k_block_size = 64 if head_dim <= 64 else 128
    tile_m = 8 if k_block_size % 128 == 0 else (16 if k_block_size % 64 == 0 else 32)
    log_max_splits = max(math.ceil(math.log2(num_splits)), 4)
    if tile_m == 8:
        log_max_splits = max(log_max_splits, 5)

    dtype = torch2cute_dtype_map[out.dtype]
    dtype_partial = torch2cute_dtype_map[out_partial.dtype]
    compile_key = (
        dtype,
        dtype_partial,
        head_dim,
        tile_m,
        k_block_size,
        log_max_splits,
        cu_seqlens is not None,
        seqused is not None,
        lse is not None,
        varlen_batch_idx is not None,
    )
    device_key = (out.device.index, *compile_key)
    if device_key not in _flash_attn_fwd_combine.compile_cache:
        _flash_attn_fwd_combine.compile_cache[device_key] = _compile_fwd_combine(
            *compile_key
        )
    if not is_fake_mode():
        _flash_attn_fwd_combine.compile_cache[device_key](
            out_partial, lse_partial, out, lse,
            cu_seqlens, seqused, num_splits_dynamic_ptr, varlen_batch_idx,
            semaphore_to_reset,
        )


_flash_attn_fwd_combine.compile_cache = get_jit_cache("fwd_combine")


# HD256 FP8 TMA prefill. The existing HD128/BF16/decode implementation above
# is unchanged. NVIDIA-derived kernel body: BSD-3-Clause; ATREX wrappers:
# Apache-2.0. See NOTICE for the original source revision and license terms.
import functools as _tma_functools
from cutlass.cute.runtime import make_fake_compact_tensor as _tma_make_fake_compact_tensor
from cutlass.cute.runtime import make_fake_stream as _tma_make_fake_stream
from .prefill_cutedsl import _tma_FlashAttentionForwardSm120Fp8Tma

# prefill_tma_runtime.py — Copyright 2026 Alibaba Group; SPDX-License-Identifier: Apache-2.0
def _select_tma_prefill_config(max_q, batch, hq, hkv, sm_count, max_kv=0):
    """Choose short-Q tiling, GQA reuse and bounded SplitKV parallelism."""
    if max_q >= 4096:
        return 128, False, 1

    def waves(rows, heads, tile):
        blocks = batch * heads * ((rows + tile - 1) // tile)
        return (blocks + sm_count - 1) // sm_count

    group = hq // hkv
    plain_waves = waves(max_q, hq, 128)
    packed_waves = waves(max_q * group, hkv, 128)
    pack_gqa = group > 1 and packed_waves < plain_waves
    rows, heads = (max_q * group, hkv) if pack_gqa else (max_q, hq)
    target_waves = waves(rows, heads, 128)
    q_tile = next(
        tile for tile in ((32, 64, 128) if pack_gqa else (16, 32, 64, 128))
        if waves(rows, heads, tile) <= target_waves
    )
    short_kv = 0 < max_kv <= 16384
    medium_kv = max_q >= 128 and 6144 <= max_kv <= 16384
    if short_kv and max_q >= 128:
        q_tile = 128
    blocks = batch * heads * ((rows + q_tile - 1) // q_tile)
    num_splits = 1
    # Packed groups with enough parallelism already approach the bandwidth
    # limit. Split only their poorly filled second wave; low-batch unpacked
    # tiles also benefit from partitioning a long KV scan.
    if not pack_gqa or sm_count < blocks < 2 * sm_count:
        limit = min(
            32,
            max(1, 5 * sm_count // blocks),
            max(1, ((max_kv + 127) // 128) // 4),
        )
        if short_kv and q_tile >= 64:
            # Restrict candidates before comparing waves per split. Clipping a
            # winner afterwards can retain more partials for the same waves.
            wave_budget = 5 if medium_kv else 1
            limit = min(limit, max(1, wave_budget * sm_count // blocks))
            if wave_budget > 1:
                limit = min(limit, 4)
        best_waves = (blocks + sm_count - 1) // sm_count
        # Three partitions fill two waves for 64 CTAs on 110 SMs; four
        # partitions need three waves plus the same combine launch.
        candidates = (2, 3, 4) if medium_kv else (2, 4, 8, 16, 32)
        for splits in candidates:
            if splits > limit:
                break
            split_waves = (blocks * splits + sm_count - 1) // sm_count
            if split_waves * num_splits < best_waves * splits:
                num_splits, best_waves = splits, split_waves
    return q_tile, pack_gqa, num_splits


def _select_tma_prefill_launch_config(max_q, batch, hq, hkv, sm_count, max_kv=0):
    """Use narrow TMA tiles for short-KV, unsplit prefill."""
    q_tile, pack_gqa, num_splits = _select_tma_prefill_config(
        max_q, batch, hq, hkv, sm_count, max_kv,
    )
    kv_tile = 128
    if (max_q < 4096 and 0 < max_kv <= 7168 and q_tile == 128
            and not pack_gqa and num_splits == 1):
        q_tile, kv_tile = 64, 64
    return q_tile, pack_gqa, num_splits, kv_tile


@_tma_functools.cache
def _compile_tma_prefill(
    hq, hkv, page_size, kv_strides,
    q_descale_strides, k_descale_strides, v_descale_strides,
    q_tile, pack_gqa, num_splits, device_index, kv_tile=128,
):
    # Device belongs in the cache key even though compilation uses the current
    # CUDA device selected by the public API. Lengths remain runtime metadata.
    kernel = _tma_FlashAttentionForwardSm120Fp8Tma(
        in_dtype=cutlass.Float8E4M3FN,
        out_dtype=cutlass.BFloat16 if num_splits == 1 else cutlass.Float32,
        is_causal=True,
        head_tile=256,
        kv_tile=kv_tile,
        q_tile=q_tile,
        use_paged_kv=True,
        num_tokens_per_page=page_size,
        balanced_scheduler=True,
        pack_gqa=pack_gqa,
        num_splits=num_splits,
    )
    total_q, batch, pages, capacity, cu_size = (cute.sym_int() for _ in range(5))
    q = _tma_make_fake_compact_tensor(
        cutlass.Float8E4M3FN, (total_q, hq, 256),
        stride_order=(2, 1, 0), assumed_align=16,
    )
    out = _tma_make_fake_compact_tensor(
        cutlass.BFloat16 if num_splits == 1 else cutlass.Float32,
        (total_q if num_splits == 1 else cute.sym_int(), hq, 256),
        stride_order=(2, 1, 0), assumed_align=16,
    )
    lse = (
        _tma_make_fake_compact_tensor(
            cutlass.Float32, (num_splits * hq, total_q),
            stride_order=(1, 0), assumed_align=4,
        )
        if num_splits > 1 else None
    )

    def kv():
        # K/V share inner strides but retain independent symbolic page strides.
        # Both NHD pools and vLLM's interleaved HND views stay caller-owned.
        return cute.runtime.make_fake_tensor(
            cutlass.Float8E4M3FN, (pages, page_size, hkv, 256),
            stride=(cute.sym_int(divisibility=16), *kv_strides, 1), assumed_align=16,
        )

    seq = _tma_make_fake_compact_tensor(cutlass.Int32, (batch,), assumed_align=4)
    cu = _tma_make_fake_compact_tensor(cutlass.Int32, (cu_size,), assumed_align=4)
    table = _tma_make_fake_compact_tensor(
        cutlass.Int32, (batch, capacity), stride_order=(1, 0), assumed_align=4,
    )

    def descale(strides):
        if strides is None:
            return None
        if strides == (hkv, 1):
            # Preserve the existing compact specialization for NHD benchmarks.
            return _tma_make_fake_compact_tensor(
                cutlass.Float32, (batch, hkv),
                stride_order=(1, 0), assumed_align=4,
            )
        # Static zero strides preserve expanded per-tensor/per-head descales.
        return cute.runtime.make_fake_tensor(
            cutlass.Float32, (batch, hkv), stride=strides, assumed_align=4,
        )

    kernel.atrex_fp8_prefill_nhd_tma_kernel.set_name_prefix("atrex")
    with use_filesystem_cutlass_dsl_version_hash():
        return cute.compile(
            kernel, q, kv(), kv(), out, lse,
            cutlass.Float32(1.0), cutlass.Float32(1.0),
            None,  # All attention tiles execute; no skip-softmax approximation.
            _tma_make_fake_stream(use_tvm_ffi_env_stream=True),
            seq, cu, table, None, cutlass.Int32(1), False,
            descale(q_descale_strides), descale(k_descale_strides), descale(v_descale_strides),
            options="--enable-tvm-ffi",
        )


def _forward_tma_prefill(**kwargs):
    """Run the page64/page128 causal prefill specialization."""
    q, k, v = (kwargs[name] for name in ("q", "k", "v"))
    out = kwargs.get("out")
    if out is None:
        out = torch.empty_like(q, dtype=torch.bfloat16)
    q_scale, k_scale, v_scale = (
        kwargs.get(name) for name in ("q_descale", "k_descale", "v_descale")
    )
    # Long prefill always uses this configuration. Avoid looking up hardware
    # and sequence metadata that the selector would immediately discard.
    q_tile, pack_gqa, num_splits, kv_tile = 128, False, 1, 128
    if kwargs["max_seqlen_q"] < 4096:
        q_tile, pack_gqa, num_splits, kv_tile = _select_tma_prefill_launch_config(
            kwargs["max_seqlen_q"], kwargs["cu_seqlens_q"].numel() - 1,
            q.shape[1], k.shape[2],
            torch.cuda.get_device_properties(q.device).multi_processor_count,
            kwargs.get("max_seqlen_k") or kwargs["page_table"].shape[1] * k.shape[1],
        )
    if kwargs.get("num_splits") == 1:
        num_splits = 1
    out_partial = lse_partial = None
    kernel_out, kernel_lse = out, None
    if num_splits > 1:
        # The established combine path requires FP32 partials on this build.
        # Keep both allocations local so CUDA graphs own their captured storage.
        out_partial = torch.empty(
            (num_splits, *q.shape), dtype=torch.float32, device=q.device,
        )
        lse_partial = torch.empty(
            (num_splits, q.shape[1], q.shape[0]),
            dtype=torch.float32, device=q.device,
        )
        kernel_out = out_partial.view(-1, q.shape[1], q.shape[2])
        kernel_lse = lse_partial.view(-1, q.shape[0])
    compiled = _compile_tma_prefill(
        q.shape[1], k.shape[2], k.shape[1], (k.stride(1), k.stride(2)),
        None if q_scale is None else q_scale.stride(),
        None if k_scale is None else k_scale.stride(),
        None if v_scale is None else v_scale.stride(),
        q_tile, pack_gqa, num_splits, q.device.index or 0, kv_tile=kv_tile,
    )
    # The TVM-FFI runtime supplies the current stream through EnvStream.
    compiled(
        q, k, v, kernel_out, kernel_lse,
        cutlass.Float32(kwargs["softmax_scale"] * math.log2(math.e)),
        cutlass.Float32(1.0), None,
        kwargs["seqused_k"], kwargs["cu_seqlens_q"], kwargs["page_table"],
        None, cutlass.Int32(kwargs["max_seqlen_q"]), False,
        q_scale, k_scale, v_scale,
    )
    if num_splits > 1:
        _flash_attn_fwd_combine(
            out_partial, lse_partial.transpose(-1, -2), out,
            cu_seqlens=kwargs["cu_seqlens_q"],
        )
    return out, None
