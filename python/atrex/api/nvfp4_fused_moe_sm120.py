import logging
import os
import sys
from functools import lru_cache
from pathlib import Path
from typing import Optional

import torch
from torch import Tensor

from atrex.core.compile_cu import cuda_kernel

log = logging.getLogger(__name__)

_SOURCE_DIR = "nvidia/nvfp4_fused_moe/sm120"
_MODULE_NAME = "nvfp4_fused_moe_sm120_steps"
_FUSED_FINALIZE_M_THRESHOLD = 2048
_E512_TOPK10_TASK29_TASK30_MAX_M = 1024
_SOURCES = (
    f"{_SOURCE_DIR}/nvfp4_fused_moe_pybind.cu",
    f"{_SOURCE_DIR}/pybind_common.cu",
    f"{_SOURCE_DIR}/routing_pybind.cu",
    f"{_SOURCE_DIR}/expand_input_rows_pybind.cu",
    f"{_SOURCE_DIR}/up_gate_pybind.cu",
    f"{_SOURCE_DIR}/down_pybind.cu",
    f"{_SOURCE_DIR}/e512_topk10_pybind.cu",
    f"{_SOURCE_DIR}/kernels/routing_sort.cu",
    f"{_SOURCE_DIR}/kernels/expand_input_rows.cu",
    f"{_SOURCE_DIR}/kernels/up_gate_gemm.cu",
    f"{_SOURCE_DIR}/kernels/up_gate_activation.cu",
    f"{_SOURCE_DIR}/kernels/finalize_moe_routing.cu",
    f"{_SOURCE_DIR}/kernels/down_gemm.cu",
    f"{_SOURCE_DIR}/kernels/down_gemm_fused_finalize.cu",
    f"{_SOURCE_DIR}/kernels/task13_gemm1_gather_cuda.cu",
    f"{_SOURCE_DIR}/kernels/task29_gemm1_small_m_cuda.cu",
    f"{_SOURCE_DIR}/kernels/task30_gemm2_small_m_cuda.cu",
    f"{_SOURCE_DIR}/kernels/e512_topk10_gemm1.cu",
    f"{_SOURCE_DIR}/kernels/e512_topk10_gemm2.cu",
    f"{_SOURCE_DIR}/kernels/e512_topk10_activation.cu",
)
_INCLUDE_DIRS = (
    _SOURCE_DIR,
    f"{_SOURCE_DIR}/include",
)
_EXTRA_CFLAGS = ("-std=c++20",)
_EXTRA_CUDA_CFLAGS = (
    "-std=c++20",
    "--expt-relaxed-constexpr",
    "-diag-suppress", "20012",
    "-diag-suppress", "20013",
    "-diag-suppress", "20015",
    "-static-global-template-stub=false",
    "-use_fast_math",
    "-DNDEBUG",
    "-DFLASHINFER_ENABLE_FP8_E8M0",
    "-DFLASHINFER_ENABLE_FP4_E2M1",
    "-DFLASHINFER_ENABLE_F16",
    "-DFLASHINFER_ENABLE_BF16",
    "-DFLASHINFER_ENABLE_FP8_E4M3",
    "-DFLASHINFER_ENABLE_FP8_E5M2",
    "-DENABLE_BF16",
    "-DENABLE_FP8",
    "-DENABLE_FP4",
    "-DCOMPILE_BLACKWELL_TMA_GEMMS",
    "-DCOMPILE_BLACKWELL_SM120_TMA_GROUPED_GEMMS",
    "-DUSING_OSS_CUTLASS_MOE_GEMM",
    "-DCUTLASS_ENABLE_GDC_FOR_SM100=1",
)


def _require_tensor(tensor: Optional[Tensor], name: str) -> Tensor:
    if tensor is None:
        raise ValueError(f"{name} is required")
    return tensor


def _require_cuda_contiguous(tensor: Tensor, name: str) -> None:
    if not tensor.is_cuda:
        raise ValueError(f"{name} must be a CUDA tensor")
    if not tensor.is_contiguous():
        raise ValueError(f"{name} must be contiguous")


def _require_dtype(tensor: Tensor, name: str, dtype: torch.dtype) -> None:
    if tensor.dtype != dtype:
        raise ValueError(f"{name} must have dtype {dtype}, got {tensor.dtype}")


def _check_device(name: str, tensor: Optional[Tensor], device: torch.device) -> None:
    if tensor is not None and tensor.device != device:
        raise ValueError(f"{name} must be on {device}, got {tensor.device}")


def _step_sync_enabled() -> bool:
    for name in (
        "ATREX_NVFP4_SM120_STEP_SYNC",
        "ATREX_NVFP4_HYBRID_V5_STEP_SYNC",
        "ATREX_NVFP4_HYBRID_V3_STEP_SYNC",
    ):
        if os.environ.get(name, "0").strip().lower() in {
            "1", "true", "yes", "on",
        }:
            return True
    return False


def _maybe_sync(label: str) -> None:
    if _step_sync_enabled():
        torch.cuda.synchronize()
        if (
            os.environ.get("ATREX_NVFP4_SM120_STEP_SYNC_VERBOSE")
            or os.environ.get("ATREX_NVFP4_HYBRID_V5_STEP_SYNC_VERBOSE")
            or os.environ.get("ATREX_NVFP4_HYBRID_V3_STEP_SYNC_VERBOSE")
        ):
            print(f"[atrex sm120 step] synced after {label}",
                  file=sys.stderr, flush=True)


@lru_cache(maxsize=1)
def _cutlass_external_include_dirs() -> tuple[str, ...]:
    package_root = Path(__file__).resolve().parents[1]
    repository_root = Path(__file__).resolve().parents[3]
    roots = (
        package_root / "third_party" / "cutlass",
        repository_root / "third_party" / "cutlass",
    )
    checked = []
    for cutlass_root in dict.fromkeys(root.resolve() for root in roots):
        include_dir = cutlass_root / "include"
        tools_include_dir = cutlass_root / "tools" / "util" / "include"
        if include_dir.is_dir() and tools_include_dir.is_dir():
            return (str(include_dir), str(tools_include_dir))
        checked.append(
            f"{include_dir} and {tools_include_dir}"
        )
    raise RuntimeError(
        "NVFP4 fused MoE SM120 JIT requires packaged or source-tree "
        "third_party/cutlass headers. Checked: "
        + "; ".join(checked)
    )


@lru_cache(maxsize=4)
def _extension_entry(cutlass_include_dirs: tuple[str, ...]):
    @cuda_kernel(
        sources=_SOURCES,
        module_name=_MODULE_NAME,
        function_name="get_num_blocks_per_seq",
        include_dirs=_INCLUDE_DIRS,
        external_include_dirs=cutlass_include_dirs,
        extra_cflags=_EXTRA_CFLAGS,
        extra_cuda_cflags=_EXTRA_CUDA_CFLAGS,
        extra_ldflags=("-lcuda",),
        cuda_arch_suffix="f",
    )
    def _nvfp4_fused_moe_sm120_extension_marker(*args, **kwargs):
        raise AssertionError("compiled extension marker must not execute")

    return _nvfp4_fused_moe_sm120_extension_marker


@lru_cache(maxsize=1)
def _build_and_load(device=None):
    log.info("JIT compiling %s for NVIDIA SM120 FP4 ...", _MODULE_NAME)
    return _extension_entry(_cutlass_external_include_dirs()).build(device)


def _hybrid_v5_gather_min_m() -> int:
    env = os.environ.get("HYBRID_V5_GATHER_MIN_M")
    if env is None or env.strip() == "":
        env = os.environ.get("HYBRID_V7_GATHER_MIN_M")
    if env is None or env.strip() == "":
        return 512
    try:
        value = int(env)
    except ValueError:
        return 512
    if value < 0 or value > 1_000_000_000:
        return 512
    return value


def _sm120_is_e512_topk10_shape(
    E: int, topk: int, hidden_size: int, inter_size: int
) -> bool:
    """Pure shape probe for the e512_topk10 dedicated path.

    Skill §3: a detector must not compile, allocate large buffers, switch
    devices, or mutate tensors; this only compares static shape integers.
    """
    return (E, topk, hidden_size, inter_size) == (512, 10, 2560, 320)


def _sm120_has_validated_shape(
    M: int, E: int, topk: int, hidden_size: int, inter_size: int
) -> bool:
    del M
    if _sm120_is_e512_topk10_shape(E, topk, hidden_size, inter_size):
        return True
    return (
        topk == 8
        and hidden_size == 2048
        and ((E == 256 and inter_size == 512)
             or (E == 256 and inter_size == 256)
             or (E == 128 and inter_size == 768)
             or (E == 128 and inter_size == 384))
    )


def _hybrid_v5_uses_compact_small_m_sf(
    M: int, E: int, topk: int, hidden_size: int, inter_size: int
) -> bool:
    """e512_topk10 compact shared-SF staging for very small M (1..16).

    Intentionally pipeline-agnostic: despite the historical ``_hybrid_v5_``
    name, this gate keys only on (M, shape) and never on the ``pipeline`` arg.
    The e512_topk10 dedicated kernels always require the compact shared-SF
    layout for M <= 16, so nvfp4_fused_moe calls this bare (with no
    ``use_hybrid_v5`` guard, unlike the dev-trunk _hybrid_v5_uses_* helpers);
    consequently hybrid_v3 == hybrid_v5 for this shape. The operator test
    parametrizes standalone/vs_flashinfer over PIPELINES to lock that in.
    """
    return (
        1 <= M <= 16
        and _sm120_is_e512_topk10_shape(E, topk, hidden_size, inter_size)
    )


def _hybrid_v5_uses_phase6_small_m(
    M: int, E: int, topk: int, hidden_size: int, inter_size: int
) -> bool:
    return (
        _sm120_has_validated_shape(M, E, topk, hidden_size, inter_size)
        and M >= 1
        and M < _hybrid_v5_gather_min_m()
        and M <= 512
    )


def _hybrid_v5_uses_task30_small_m(
    M: int, E: int, topk: int, hidden_size: int, inter_size: int
) -> bool:
    return (
        _sm120_has_validated_shape(M, E, topk, hidden_size, inter_size)
        and M >= 1
        and M <= 512
    )


def _hybrid_v5_uses_task13_gather(
    M: int, E: int, topk: int, hidden_size: int, inter_size: int
) -> bool:
    return (
        _sm120_has_validated_shape(M, E, topk, hidden_size, inter_size)
        and M >= _hybrid_v5_gather_min_m()
    )


def nvfp4_fused_moe(
    hidden_states: Tensor,
    w1_fp4: Tensor,
    w2_fp4: Tensor,
    w1_blockscale: Tensor,
    w2_blockscale: Tensor,
    a1_global_scale: Optional[Tensor] = None,
    a2_global_scale: Optional[Tensor] = None,
    w1_global_scale: Optional[Tensor] = None,
    w2_global_scale: Optional[Tensor] = None,
    topk_ids: Optional[Tensor] = None,
    topk_weights: Optional[Tensor] = None,
    output: Optional[Tensor] = None,
    input_sf: Optional[Tensor] = None,
    pipeline: str = "hybrid_v5",
    gemm1_alpha: Optional[Tensor] = None,
    gemm2_alpha: Optional[Tensor] = None,
    output_preinitialized: bool = False,
    shared_event: Optional[int] = None,
) -> None:
    if pipeline not in {"hybrid_v3", "hybrid_v5"}:
        raise ValueError(
            "nvfp4_fused_moe_sm120 only supports pipeline='hybrid_v3' "
            "or pipeline='hybrid_v5'"
        )
    use_hybrid_v5 = pipeline == "hybrid_v5"

    a1_global_scale = _require_tensor(a1_global_scale, "a1_global_scale")
    a2_global_scale = _require_tensor(a2_global_scale, "a2_global_scale")
    topk_ids = _require_tensor(topk_ids, "topk_ids")
    topk_weights = _require_tensor(topk_weights, "topk_weights")
    output = _require_tensor(output, "output")

    use_external_alpha = gemm1_alpha is not None or gemm2_alpha is not None
    if use_external_alpha:
        gemm1_alpha = _require_tensor(gemm1_alpha, "gemm1_alpha")
        gemm2_alpha = _require_tensor(gemm2_alpha, "gemm2_alpha")
    else:
        w1_global_scale = _require_tensor(w1_global_scale, "w1_global_scale")
        w2_global_scale = _require_tensor(w2_global_scale, "w2_global_scale")

    device = hidden_states.device
    for name, tensor in (
        ("w1_fp4", w1_fp4),
        ("w2_fp4", w2_fp4),
        ("w1_blockscale", w1_blockscale),
        ("w2_blockscale", w2_blockscale),
        ("a1_global_scale", a1_global_scale),
        ("a2_global_scale", a2_global_scale),
        ("w1_global_scale", w1_global_scale),
        ("w2_global_scale", w2_global_scale),
        ("topk_ids", topk_ids),
        ("topk_weights", topk_weights),
        ("output", output),
        ("input_sf", input_sf),
        ("gemm1_alpha", gemm1_alpha),
        ("gemm2_alpha", gemm2_alpha),
    ):
        _check_device(name, tensor, device)

    for name, tensor in (
        ("hidden_states", hidden_states),
        ("w1_fp4", w1_fp4),
        ("w2_fp4", w2_fp4),
        ("w1_blockscale", w1_blockscale),
        ("w2_blockscale", w2_blockscale),
        ("a1_global_scale", a1_global_scale),
        ("a2_global_scale", a2_global_scale),
        ("topk_ids", topk_ids),
        ("topk_weights", topk_weights),
        ("output", output),
    ):
        _require_cuda_contiguous(tensor, name)

    if input_sf is not None:
        _require_cuda_contiguous(input_sf, "input_sf")
    if use_external_alpha:
        _require_cuda_contiguous(gemm1_alpha, "gemm1_alpha")
        _require_cuda_contiguous(gemm2_alpha, "gemm2_alpha")
    else:
        _require_cuda_contiguous(w1_global_scale, "w1_global_scale")
        _require_cuda_contiguous(w2_global_scale, "w2_global_scale")

    _require_dtype(w1_fp4, "w1_fp4", torch.uint8)
    _require_dtype(w2_fp4, "w2_fp4", torch.uint8)
    _require_dtype(a1_global_scale, "a1_global_scale", torch.float32)
    _require_dtype(a2_global_scale, "a2_global_scale", torch.float32)
    _require_dtype(topk_ids, "topk_ids", torch.int32)
    _require_dtype(topk_weights, "topk_weights", torch.float32)
    _require_dtype(output, "output", torch.bfloat16)

    if hidden_states.dtype == torch.uint8:
        if input_sf is None:
            raise ValueError("input_sf is required when hidden_states is uint8")
        hidden_size = hidden_states.shape[1] * 2
    elif hidden_states.dtype == torch.bfloat16:
        hidden_size = hidden_states.shape[1]
    else:
        raise ValueError(
            "hidden_states must be bfloat16 or uint8 NVFP4 packed, "
            f"got {hidden_states.dtype}"
        )

    if use_external_alpha:
        _require_dtype(gemm1_alpha, "gemm1_alpha", torch.float32)
        _require_dtype(gemm2_alpha, "gemm2_alpha", torch.float32)
    else:
        _require_dtype(w1_global_scale, "w1_global_scale", torch.float32)
        _require_dtype(w2_global_scale, "w2_global_scale", torch.float32)

    M = int(hidden_states.shape[0])
    E = int(w1_fp4.shape[0])
    topk = int(topk_ids.shape[1])
    inter_size = int(w2_fp4.shape[2] * 2)
    expanded = M * topk

    if topk_ids.numel() < expanded:
        raise ValueError("topk_ids shape is smaller than [M, topk]")
    if topk_weights.numel() < expanded:
        raise ValueError("topk_weights shape is smaller than [M, topk]")
    if output.numel() < M * hidden_size:
        raise ValueError("output shape is smaller than [M, hidden_size]")
    if w1_fp4.numel() < E * 2 * inter_size * hidden_size // 2:
        raise ValueError("w1_fp4 shape is smaller than [E, 2*inter, hidden/2]")
    if w2_fp4.numel() < E * hidden_size * inter_size // 2:
        raise ValueError("w2_fp4 shape is smaller than [E, hidden, inter/2]")
    if not _sm120_has_validated_shape(M, E, topk, hidden_size, inter_size):
        raise ValueError(
            "nvfp4_fused_moe_sm120 currently supports only validated shapes "
            "(E=256, topk=8, hidden_size=2048, inter_size=512), "
            "(E=256, topk=8, hidden_size=2048, inter_size=256), and "
            "(E=128, topk=8, hidden_size=2048, inter_size=768), and "
            "(E=128, topk=8, hidden_size=2048, inter_size=384), and "
            "(E=512, topk=10, hidden_size=2560, inter_size=320); "
            f"got pipeline={pipeline!r}, M={M}, E={E}, topk={topk}, "
            f"hidden_size={hidden_size}, inter_size={inter_size}"
        )

    shared_event_value = int(shared_event) if shared_event else 0
    if shared_event_value and not output_preinitialized:
        raise ValueError(
            "shared_event requires output_preinitialized=True because the "
            "event is used to wait for an existing output producer whose "
            "result must be accumulated."
        )

    mod = _build_and_load(device)

    w1_sf = w1_blockscale.view(torch.uint8)
    w2_sf = w2_blockscale.view(torch.uint8)

    if use_external_alpha:
        gemm1_alpha_t = gemm1_alpha
        gemm2_alpha_t = gemm2_alpha
    else:
        gemm1_alpha_t = torch.empty((E,), device=device, dtype=torch.float32)
        gemm2_alpha_t = torch.empty((E,), device=device, dtype=torch.float32)
        torch.mul(a1_global_scale, w1_global_scale, out=gemm1_alpha_t)
        torch.reciprocal(gemm1_alpha_t, out=gemm1_alpha_t)
        torch.mul(a2_global_scale, w2_global_scale, out=gemm2_alpha_t)
        torch.reciprocal(gemm2_alpha_t, out=gemm2_alpha_t)

    routing_ws = torch.empty(
        (int(mod.get_workspace_size_routing(M, E, topk)),),
        device=device,
        dtype=torch.uint8,
    )
    permuted_token_selected_experts = torch.empty(
        (expanded,), device=device, dtype=torch.int32)
    permuted_row = torch.empty((expanded,), device=device, dtype=torch.int32)
    unperm_map = torch.empty((expanded,), device=device, dtype=torch.int32)
    expert_offset = torch.empty((E + 1,), device=device, dtype=torch.int64)
    perm_scales = torch.empty((expanded,), device=device, dtype=torch.float32)

    def _wait_for_shared_event() -> None:
        if shared_event_value:
            mod.wait_for_shared_event(shared_event_value)
            _maybe_sync("wait_for_shared_event")

    use_phase6_small_m = use_hybrid_v5 and _hybrid_v5_uses_phase6_small_m(
        M, E, topk, hidden_size, inter_size)
    use_task30_small_m = use_hybrid_v5 and _hybrid_v5_uses_task30_small_m(
        M, E, topk, hidden_size, inter_size)
    use_task13_gather = use_hybrid_v5 and _hybrid_v5_uses_task13_gather(
        M, E, topk, hidden_size, inter_size)

    act_out = torch.empty(
        (expanded, inter_size // 2),
        device=device,
        dtype=torch.uint8,
    )

    is_e512_topk10 = _sm120_is_e512_topk10_shape(
        E, topk, hidden_size, inter_size)
    if is_e512_topk10:
        # e512_topk10 (E=512, topk=10, hidden=2560, inter=320) dedicated
        # additive path. Reuses the GREEN routing_sort (fused alpha +
        # completion-counter zeroing) and expand_input_rows (shared SF
        # staging); gemm1/gemm2 use the e512_topk10 task29/task30 kernels
        # plus the BF16-boundary activation. The dev topk=8 dispatch below is
        # never reached for this shape (both gemm2 sub-paths return).
        expand_out = torch.empty(
            (expanded, hidden_size // 2), device=device, dtype=torch.uint8)
        fc1_act_sf = torch.empty(
            (int(mod.get_fc1_act_sf_size(M, E, topk, hidden_size)),),
            device=device, dtype=torch.uint8)
        fc2_act_sf = torch.empty(
            (int(mod.get_fc2_act_sf_size(M, E, topk, inter_size)),),
            device=device, dtype=torch.uint8)

        use_shared_sf_staging = _hybrid_v5_uses_compact_small_m_sf(
            M, E, topk, hidden_size, inter_size)
        if not use_shared_sf_staging:
            fc2_act_sf.zero_()

        use_large_m_fused_path = M > _E512_TOPK10_TASK29_TASK30_MAX_M
        # Small-M always uses the in-kernel fixed-order FP32 finalize; the
        # atomic-finalize fallback is intentionally not ported (skill §1).
        use_task30_fixed_order_finalize = not use_large_m_fused_path
        fuse_alpha_in_routing = not use_external_alpha

        task30_fixed_expert_rows = None
        task30_completion_counters = None
        if use_task30_fixed_order_finalize:
            task30_fixed_expert_rows = torch.empty(
                (expanded, hidden_size), device=device, dtype=torch.bfloat16)
            task30_n_tiles = (hidden_size + 127) // 128
            task30_completion_counters = torch.empty(
                (M * task30_n_tiles,), device=device, dtype=torch.int32)

        # output_to_zero folds the BF16 output memset into the routing node.
        # Tokens whose topk_ids are all sentinel (-1) contribute no expert row,
        # so no GEMM2 CTA ever finalizes them; without this clear their output
        # rows would keep whatever the caller's buffer held. A preinitialized
        # output (shared-expert accumulation) must not be cleared here.
        mod.routing_sort(
            topk_ids,
            routing_ws,
            expert_offset,
            permuted_token_selected_experts,
            permuted_row,
            unperm_map,
            M,
            E,
            topk,
            None if output_preinitialized else output,
            a1_global_scale if fuse_alpha_in_routing else None,
            w1_global_scale if fuse_alpha_in_routing else None,
            gemm1_alpha_t if fuse_alpha_in_routing else None,
            a2_global_scale if fuse_alpha_in_routing else None,
            w2_global_scale if fuse_alpha_in_routing else None,
            gemm2_alpha_t if fuse_alpha_in_routing else None,
            task30_completion_counters,
        )
        _maybe_sync("routing_sort")

        mod.expand_input_rows(
            hidden_states,
            expand_out,
            topk_weights,
            perm_scales,
            permuted_row,
            a1_global_scale,
            expert_offset,
            fc1_act_sf,
            input_sf,
            permuted_token_selected_experts,
            M,
            E,
            topk,
            hidden_size,
            use_shared_sf_staging,
        )
        _maybe_sync("expand_input_rows")

        if M == 1:
            gemm1_out = torch.empty(
                (expanded, 2 * inter_size), device=device,
                dtype=torch.bfloat16)
            task29_ws = torch.empty(
                (int(mod.get_workspace_size_e512_topk10_task29_fused(
                    M, E, topk, hidden_size, inter_size)),),
                device=device, dtype=torch.uint8)
            mod.e512_topk10_task29_forward_fused(
                expand_out,
                w1_fp4,
                fc1_act_sf,
                w1_sf,
                gemm1_alpha_t,
                gemm1_out,
                expert_offset,
                task29_ws,
                M,
                E,
                topk,
                hidden_size,
                inter_size,
            )
            _maybe_sync("e512_topk10_task29_forward_fused")
            mod.e512_topk10_do_activation(
                act_out,
                gemm1_out,
                expert_offset,
                a2_global_scale,
                fc2_act_sf,
                permuted_token_selected_experts,
                M,
                E,
                topk,
                inter_size,
                use_shared_sf_staging,
                True,
            )
            _maybe_sync("e512_topk10_do_activation")
        elif not use_large_m_fused_path:
            task29_ws = torch.empty(
                (int(mod.get_workspace_size_e512_topk10_task29_grouped_m16_fused_act(
                    M, E, topk, hidden_size, inter_size)),),
                device=device, dtype=torch.uint8)
            mod.e512_topk10_task29_forward_grouped_m16_fused_act(
                expand_out,
                w1_fp4,
                fc1_act_sf,
                w1_sf,
                gemm1_alpha_t,
                act_out,
                a2_global_scale,
                fc2_act_sf,
                expert_offset,
                task29_ws,
                M,
                E,
                topk,
                hidden_size,
                inter_size,
                use_shared_sf_staging,
            )
            _maybe_sync("e512_topk10_task29_forward_grouped_m16_fused_act")
        else:
            # M > 1024: reuse the dev large-M cutlass fused-activation gemm1.
            gemm1_ws = torch.empty(
                (int(mod.get_workspace_size_gemm1(
                    M, E, topk, hidden_size, inter_size)),),
                device=device, dtype=torch.uint8)
            mod.gemm_forward_v20_fused_act(
                expand_out,
                w1_fp4,
                fc1_act_sf,
                w1_sf,
                gemm1_alpha_t,
                act_out,
                fc2_act_sf,
                a2_global_scale,
                expert_offset,
                gemm1_ws,
                M,
                E,
                topk,
                hidden_size,
                inter_size,
            )
            _maybe_sync("gemm_forward_v20_fused_act")

        if not use_large_m_fused_path:
            _wait_for_shared_event()
            task30_ws = torch.empty(
                (int(mod.get_workspace_size_e512_topk10_task30(
                    M, E, topk, hidden_size, inter_size)),),
                device=device, dtype=torch.uint8)
            mod.e512_topk10_task30_forward_fixed(
                act_out,
                w2_fp4,
                fc2_act_sf,
                w2_sf,
                gemm2_alpha_t,
                output,
                expert_offset,
                permuted_row,
                perm_scales,
                task30_ws,
                task30_fixed_expert_rows,
                topk_weights,
                topk_ids,
                task30_completion_counters,
                M,
                E,
                topk,
                hidden_size,
                inter_size,
                use_shared_sf_staging,
                bool(output_preinitialized),
            )
            _maybe_sync("e512_topk10_task30_forward_fixed")
            return

        # M > 1024: reuse the dev large-M cutlass fused-finalize gemm2. The
        # non-preinitialized output was already cleared by routing_sort above.
        _wait_for_shared_event()
        setup_ws = torch.empty(
            (int(mod.get_workspace_size_setup_fused_finalize_group_ptrs(E)),),
            device=device, dtype=torch.uint8)
        mod.setup_fused_finalize_group_ptrs(
            act_out,
            w2_fp4,
            fc2_act_sf,
            w2_sf,
            gemm2_alpha_t,
            expert_offset,
            setup_ws,
            perm_scales,
            permuted_row,
            M,
            E,
            topk,
            hidden_size,
            inter_size,
        )
        _maybe_sync("setup_fused_finalize_group_ptrs")
        cutlass_ws_bytes = int(mod.get_workspace_size_gemm2_fused_finalize(
            setup_ws, E, M, hidden_size, inter_size))
        cutlass_ws = torch.empty(
            (max(cutlass_ws_bytes, 1),), device=device, dtype=torch.uint8)
        mod.cutlass_gemm_fused_finalize_forward(
            act_out,
            w2_fp4,
            fc2_act_sf,
            w2_sf,
            gemm2_alpha_t,
            expert_offset,
            setup_ws,
            cutlass_ws,
            output,
            perm_scales,
            permuted_row,
            M,
            E,
            topk,
            hidden_size,
            inter_size,
        )
        _maybe_sync("cutlass_gemm_fused_finalize_forward")
        return

    if use_task13_gather:
        fc2_act_sf_bytes = max(
            int(mod.get_task13_fc2_act_sf_size(M, E, topk, inter_size)),
            int(mod.get_fc2_act_sf_size(M, E, topk, inter_size)),
        )
        fc2_act_sf = torch.empty(
            (fc2_act_sf_bytes,), device=device, dtype=torch.uint8)

        mod.routing_sort_with_scales(
            topk_ids,
            topk_weights,
            routing_ws,
            expert_offset,
            permuted_token_selected_experts,
            permuted_row,
            unperm_map,
            perm_scales,
            M,
            E,
            topk,
        )
        _maybe_sync("routing_sort_with_scales")

        task13_ws = torch.empty(
            (int(mod.get_workspace_size_task13_gather_fused_act(
                M, E, topk, hidden_size, inter_size)),),
            device=device,
            dtype=torch.uint8,
        )
        mod.task13_gemm1_gather_fused_act_forward(
            hidden_states,
            input_sf,
            w1_fp4,
            w1_sf,
            gemm1_alpha_t,
            act_out,
            a2_global_scale,
            fc2_act_sf,
            a1_global_scale,
            permuted_token_selected_experts,
            expert_offset,
            task13_ws,
            M,
            E,
            topk,
            hidden_size,
            inter_size,
        )
        _maybe_sync("task13_gemm1_gather_fused_act_forward")
    else:
        expand_out = torch.empty(
            (expanded, hidden_size // 2),
            device=device,
            dtype=torch.uint8,
        )
        fc1_act_sf = torch.empty(
            (int(mod.get_fc1_act_sf_size(M, E, topk, hidden_size)),),
            device=device,
            dtype=torch.uint8,
        )
        fc2_act_sf = torch.empty(
            (int(mod.get_fc2_act_sf_size(M, E, topk, inter_size)),),
            device=device,
            dtype=torch.uint8,
        )
        fc2_act_sf.zero_()

        mod.routing_sort(
            topk_ids,
            routing_ws,
            expert_offset,
            permuted_token_selected_experts,
            permuted_row,
            unperm_map,
            M,
            E,
            topk,
        )
        _maybe_sync("routing_sort")

        mod.expand_input_rows(
            hidden_states,
            expand_out,
            topk_weights,
            perm_scales,
            permuted_row,
            a1_global_scale,
            expert_offset,
            fc1_act_sf,
            input_sf,
            permuted_token_selected_experts,
            M,
            E,
            topk,
            hidden_size,
        )
        _maybe_sync("expand_input_rows")

        if use_phase6_small_m and M == 1:
            gemm1_out = torch.empty(
                (expanded, 2 * inter_size),
                device=device,
                dtype=torch.bfloat16)
            task29_ws = torch.empty(
                (int(mod.get_workspace_size_task29_gemm1_small_m_fused(
                    M, E, topk, hidden_size, inter_size)),),
                device=device,
                dtype=torch.uint8,
            )
            mod.task29_gemm1_small_m_forward_fused(
                expand_out,
                w1_fp4,
                fc1_act_sf,
                w1_sf,
                gemm1_alpha_t,
                gemm1_out,
                expert_offset,
                task29_ws,
                M,
                E,
                topk,
                hidden_size,
                inter_size,
            )
            _maybe_sync("task29_gemm1_small_m_forward_fused")
            mod.do_activation(
                act_out,
                gemm1_out,
                expert_offset,
                a2_global_scale,
                fc2_act_sf,
                permuted_token_selected_experts,
                M,
                E,
                topk,
                inter_size,
            )
            _maybe_sync("do_activation")
        elif use_phase6_small_m:
            task29_ws = torch.empty(
                (int(mod.get_workspace_size_task29_gemm1_small_m_grouped_m16_fused_act(
                    M, E, topk, hidden_size, inter_size)),),
                device=device,
                dtype=torch.uint8,
            )
            mod.task29_gemm1_small_m_forward_grouped_m16_fused_act(
                expand_out,
                w1_fp4,
                fc1_act_sf,
                w1_sf,
                gemm1_alpha_t,
                act_out,
                a2_global_scale,
                fc2_act_sf,
                expert_offset,
                task29_ws,
                M,
                E,
                topk,
                hidden_size,
                inter_size,
            )
            _maybe_sync("task29_gemm1_small_m_forward_grouped_m16_fused_act")
        elif M >= _FUSED_FINALIZE_M_THRESHOLD:
            gemm1_ws = torch.empty(
                (int(mod.get_workspace_size_gemm1(
                    M, E, topk, hidden_size, inter_size)),),
                device=device,
                dtype=torch.uint8,
            )
            mod.gemm_forward_v20_fused_act(
                expand_out,
                w1_fp4,
                fc1_act_sf,
                w1_sf,
                gemm1_alpha_t,
                act_out,
                fc2_act_sf,
                a2_global_scale,
                expert_offset,
                gemm1_ws,
                M,
                E,
                topk,
                hidden_size,
                inter_size,
            )
            _maybe_sync("gemm_forward_v20_fused_act")
        else:
            gemm1_ws = torch.empty(
                (int(mod.get_workspace_size_gemm1(
                    M, E, topk, hidden_size, inter_size)),),
                device=device,
                dtype=torch.uint8,
            )
            gemm1_out = torch.empty(
                (expanded, 2 * inter_size),
                device=device,
                dtype=torch.bfloat16)
            mod.gemm_forward_v20(
                expand_out,
                w1_fp4,
                fc1_act_sf,
                w1_sf,
                gemm1_alpha_t,
                gemm1_out,
                expert_offset,
                gemm1_ws,
                M,
                E,
                topk,
                hidden_size,
                inter_size,
            )
            _maybe_sync("gemm_forward_v20")

            mod.do_activation(
                act_out,
                gemm1_out,
                expert_offset,
                a2_global_scale,
                fc2_act_sf,
                permuted_token_selected_experts,
                M,
                E,
                topk,
                inter_size,
            )
            _maybe_sync("do_activation")

    if use_task30_small_m:
        _wait_for_shared_event()
        if not output_preinitialized:
            output.zero_()
        task30_ws = torch.empty(
            (int(mod.get_workspace_size_task30_gemm2_small_m(
                M, E, topk, hidden_size, inter_size)),),
            device=device,
            dtype=torch.uint8,
        )
        mod.task30_gemm2_small_m_forward(
            act_out,
            w2_fp4,
            fc2_act_sf,
            w2_sf,
            gemm2_alpha_t,
            output,
            expert_offset,
            permuted_row,
            perm_scales,
            task30_ws,
            M,
            E,
            topk,
            hidden_size,
            inter_size,
        )
        _maybe_sync("task30_gemm2_small_m_forward")
        return

    if M >= _FUSED_FINALIZE_M_THRESHOLD:
        _wait_for_shared_event()
        if not output_preinitialized:
            output.zero_()
        setup_ws = torch.empty(
            (int(mod.get_workspace_size_setup_fused_finalize_group_ptrs(E)),),
            device=device,
            dtype=torch.uint8,
        )
        mod.setup_fused_finalize_group_ptrs(
            act_out,
            w2_fp4,
            fc2_act_sf,
            w2_sf,
            gemm2_alpha_t,
            expert_offset,
            setup_ws,
            perm_scales,
            permuted_row,
            M,
            E,
            topk,
            hidden_size,
            inter_size,
        )
        _maybe_sync("setup_fused_finalize_group_ptrs")
        cutlass_ws_bytes = int(mod.get_workspace_size_gemm2_fused_finalize(
            setup_ws, E, M, hidden_size, inter_size))
        cutlass_ws = torch.empty(
            (max(cutlass_ws_bytes, 1),),
            device=device,
            dtype=torch.uint8,
        )
        mod.cutlass_gemm_fused_finalize_forward(
            act_out,
            w2_fp4,
            fc2_act_sf,
            w2_sf,
            gemm2_alpha_t,
            expert_offset,
            setup_ws,
            cutlass_ws,
            output,
            perm_scales,
            permuted_row,
            M,
            E,
            topk,
            hidden_size,
            inter_size,
        )
        _maybe_sync("cutlass_gemm_fused_finalize_forward")
        return

    gemm2_out = torch.empty(
        (expanded, hidden_size), device=device, dtype=torch.bfloat16)
    setup_ws = torch.empty(
        (int(mod.get_workspace_size_setup_gemm2_group_ptrs(E)),),
        device=device,
        dtype=torch.uint8,
    )
    mod.setup_gemm2_group_ptrs(
        act_out,
        w2_fp4,
        fc2_act_sf,
        w2_sf,
        gemm2_alpha_t,
        gemm2_out,
        expert_offset,
        setup_ws,
        M,
        E,
        topk,
        hidden_size,
        inter_size,
    )
    _maybe_sync("setup_gemm2_group_ptrs")
    cutlass_ws_bytes = int(mod.get_workspace_size_gemm2(
        setup_ws, E, M, hidden_size, inter_size))
    cutlass_ws = torch.empty(
        (max(cutlass_ws_bytes, 1),),
        device=device,
        dtype=torch.uint8,
    )
    mod.cutlass_gemm_forward(
        act_out,
        w2_fp4,
        fc2_act_sf,
        w2_sf,
        gemm2_alpha_t,
        gemm2_out,
        expert_offset,
        setup_ws,
        cutlass_ws,
        M,
        E,
        topk,
        hidden_size,
        inter_size,
    )
    _maybe_sync("cutlass_gemm_forward")

    _wait_for_shared_event()
    mod.finalize_moe_routing(
        gemm2_out,
        output,
        topk_weights,
        unperm_map,
        topk_ids,
        M,
        E,
        topk,
        hidden_size,
        bool(output_preinitialized),
    )
    _maybe_sync("finalize_moe_routing")


def nvfp4_fused_moe_sm120_build(device=None):
    return _build_and_load(device)
