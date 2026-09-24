"""CuTeDSL Chunk-GDN forward API for NVIDIA SM103 and SM120."""

import math
from typing import Optional, Tuple

import torch

from atrex.utils.device_target import detect_device_target

BT = 32
K_DIM = 128
V_DIM = 128
BV = 16
SM103_H = 4
SM103_HV = 32
SM103_MAX_SEQS = 16
SM103_MAX_TOKENS = 16384
SM103_SCALE = K_DIM ** -0.5


def _select_chunk_gdn_implementation(device=None):
    from atrex.src.nvidia.chunk_gdn import (
        select_chunk_gdn_implementation,
    )

    return select_chunk_gdn_implementation(device)


def _check_chunk_gdn_device(tensor: torch.Tensor):
    if tensor.device.type != "cuda":
        raise ValueError("CuTeDSL Chunk-GDN requires CUDA tensors")
    implementation = _select_chunk_gdn_implementation(tensor.device)
    if implementation is None:
        target = detect_device_target(tensor.device)
        raise RuntimeError(
            "CuTeDSL Chunk-GDN requires a supported NVIDIA target; "
            f"detected {target.family}/{target.arch}"
        )
    return implementation


def _check_chunk_gdn_current_device():
    implementation = _select_chunk_gdn_implementation()
    if implementation is None:
        target = detect_device_target()
        raise RuntimeError(
            "CuTeDSL Chunk-GDN requires a supported NVIDIA target; "
            f"detected {target.family}/{target.arch}"
        )
    return implementation


def _validate_inputs(ctx: dict, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                     g: torch.Tensor, beta: torch.Tensor):
    if q.ndim != 4:
        raise ValueError(f"q must be 4D [B, T, H, K], got shape {tuple(q.shape)}")
    b, t, h, k_dim = q.shape
    expected_t = ctx.get("T")
    if expected_t is not None and t != expected_t:
        raise ValueError(f"T shape mismatch: ctx was initialized for T={expected_t}, got T={t}")
    runtime_t = t if expected_t is None else expected_t
    expected_q = (ctx["B"], runtime_t, ctx["H"], ctx["K"])
    expected_v = (ctx["B"], runtime_t, ctx["HV"], ctx["V"])
    expected_gate = (ctx["B"], runtime_t, ctx["HV"])
    expected = {
        "q": expected_q,
        "k": expected_q,
        "v": expected_v,
        "g": expected_gate,
        "beta": expected_gate,
    }
    actual = {name: tensor.shape for name, tensor in (
        ("q", q), ("k", k), ("v", v), ("g", g), ("beta", beta)
    )}
    for name, shape in expected.items():
        if actual[name] != shape:
            raise ValueError(f"{name} shape mismatch: expected {shape}, got {actual[name]}")
    if (b, h, k_dim) != (ctx["B"], ctx["H"], ctx["K"]):
        raise ValueError(
            "q shape config mismatch: "
            f"expected B/H/K={(ctx['B'], ctx['H'], ctx['K'])}, got {(b, h, k_dim)}"
        )
    for name, tensor in (("q", q), ("k", k), ("v", v)):
        if tensor.dtype != torch.bfloat16:
            raise TypeError(f"{name} dtype mismatch: expected torch.bfloat16, got {tensor.dtype}")
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on the same CUDA device as q")
    for name, tensor in (("g", g), ("beta", beta)):
        if not _gate_dtype_supported(tensor):
            raise TypeError(
                f"{name} dtype mismatch: expected torch.bfloat16 or "
                f"torch.float32, got {tensor.dtype}")
        if tensor.device != q.device:
            raise ValueError(f"{name} must be on the same CUDA device as q")

    return _check_chunk_gdn_device(q)


def _gate_dtype_supported(tensor: torch.Tensor) -> bool:
    return tensor.dtype in (torch.bfloat16, torch.float32)


def _is_supported_sm120_fast_path(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    use_qk_l2norm_in_kernel: bool,
    cp_context,
    transpose_state_layout: bool,
) -> bool:
    if q.device.type != "cuda" or k.device != q.device or v.device != q.device:
        return False
    target = detect_device_target(q.device)
    if target.family != "nvidia" or target.arch != "sm120":
        return False
    if g.device != q.device or beta.device != q.device:
        return False
    if q.dtype != torch.bfloat16 or k.dtype != torch.bfloat16 or v.dtype != torch.bfloat16:
        return False
    if not _gate_dtype_supported(g) or not _gate_dtype_supported(beta):
        return False
    if q.shape != k.shape or q.ndim != 4 or v.ndim != 4:
        return False
    b, t, h, k_dim = q.shape
    bv, tv, hv, v_dim = v.shape
    # Accept the supported Qwen3.5 GDN head configurations.
    if (b, bv, tv, k_dim, v_dim) != (1, 1, t, 128, 128):
        return False
    if h not in (8, 16) or hv not in (32, 48, 64):
        return False
    if hv % h != 0:
        return False
    h_per_hv = hv // h
    if h_per_hv not in (2, 3, 4):
        return False
    if g.shape != (1, t, hv) or beta.shape != (1, t, hv):
        return False
    if cp_context is not None or transpose_state_layout:
        return False
    if not use_qk_l2norm_in_kernel:
        return False
    return True


def _cuda_runtime_major(version: Optional[str]) -> Optional[int]:
    if version is None:
        return None
    try:
        return int(version.split(".", 1)[0])
    except ValueError:
        return None


def _is_supported_sm103_fast_path(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float],
    initial_state: Optional[torch.Tensor],
    cu_seqlens: Optional[torch.Tensor],
    cu_seqlens_cpu: Optional[torch.Tensor],
    use_qk_l2norm_in_kernel: bool,
    cp_context,
    transpose_state_layout: bool,
    state_checkpoints: Optional[torch.Tensor],
    checkpoint_cu_starts: Optional[torch.Tensor],
    checkpoint_every_n_tokens: int,
) -> bool:
    """Return whether a call is inside the verified AKA M64 contract."""
    if q.device.type != "cuda":
        return False
    target = detect_device_target(q.device)
    if (
        target.family != "nvidia"
        or target.arch != "sm103"
        or (_cuda_runtime_major(target.runtime_version) or 0) < 13
    ):
        return False
    if any(tensor.device != q.device for tensor in (k, v, g, beta)):
        return False
    if q.dtype != torch.bfloat16 or k.dtype != q.dtype or v.dtype != q.dtype:
        return False
    if g.dtype != torch.float32 or beta.dtype != torch.float32:
        return False
    if not all(tensor.is_contiguous() for tensor in (q, k, v, g, beta)):
        return False
    if q.ndim != 4 or k.shape != q.shape or v.ndim != 4:
        return False
    physical_b, total_t, hq, dk = q.shape
    v_b, v_t, hv, dv = v.shape
    if (
        physical_b != 1
        or v_b != 1
        or v_t != total_t
        or total_t <= 0
        or total_t > SM103_MAX_TOKENS
        or hq != SM103_H
        or hv != SM103_HV
        or dk != K_DIM
        or dv != V_DIM
        or g.shape != (1, total_t, SM103_HV)
        or beta.shape != g.shape
    ):
        return False
    if use_qk_l2norm_in_kernel or cp_context is not None or transpose_state_layout:
        return False
    if (
        checkpoint_every_n_tokens != 0
        or state_checkpoints is not None
        or checkpoint_cu_starts is not None
    ):
        return False
    requested_scale = SM103_SCALE if scale is None else float(scale)
    if not math.isclose(requested_scale, SM103_SCALE, rel_tol=0.0, abs_tol=1e-12):
        return False

    if cu_seqlens is None:
        if cu_seqlens_cpu is not None:
            return False
        cu_values = [0, total_t]
    else:
        cu_values = _packed_cu_values(
            cu_seqlens,
            cu_seqlens_cpu,
            total_t,
            require_matching_cpu=True,
        )
    num_seqs = len(cu_values) - 1
    if num_seqs < 1 or num_seqs > SM103_MAX_SEQS:
        return False
    expected_state = (num_seqs, SM103_HV, V_DIM, K_DIM)
    return (
        initial_state is not None
        and initial_state.device == q.device
        and initial_state.dtype == torch.float32
        and tuple(initial_state.shape) == expected_state
        and initial_state.is_contiguous()
    )


def _is_supported_initial_state(
    initial_state: Optional[torch.Tensor],
    q: torch.Tensor,
    v: torch.Tensor,
    num_seqs: int,
) -> bool:
    if initial_state is None:
        return True
    expected = (num_seqs, int(v.shape[2]), int(v.shape[3]), int(q.shape[3]))
    return (
        initial_state.device == q.device
        and initial_state.dtype == torch.float32
        and tuple(initial_state.shape) == expected
    )


# Eligibility for single-sequence calls without explicit cu_seqlens.
def _can_use_direct_runtime_t(
    t: int,
    hv: int,
    output_final_state: bool,
) -> bool:
    if t <= 0:
        return False
    if output_final_state:
        # Tail final-state direct kernels have shown wider numerical drift on
        # real GDN-style inputs. Serving passes cu_seqlens and uses the varlen
        # prefill kernel for packed prompt batches.
        if t % BT != 0:
            return hv == 48 and t >= 4096 and t < 32768
        return t < 65536
    return t < 32768


def _packed_cu_values(
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: Optional[torch.Tensor],
    total_t: int,
    *,
    require_matching_cpu: bool = False,
) -> list[int]:
    if (
        cu_seqlens.ndim != 1
        or cu_seqlens.device.type != "cuda"
        or cu_seqlens.dtype not in (torch.int32, torch.int64)
        or not cu_seqlens.is_contiguous()
    ):
        raise ValueError(
            "cu_seqlens must be a contiguous 1D CUDA int32/int64 tensor"
        )
    if require_matching_cpu and cu_seqlens_cpu is None:
        raise ValueError(
            "SM103 AKA M64 requires cu_seqlens_cpu when cu_seqlens is supplied"
        )
    if cu_seqlens_cpu is not None:
        if (
            cu_seqlens_cpu.ndim != 1
            or cu_seqlens_cpu.device.type != "cpu"
            or cu_seqlens_cpu.dtype not in (torch.int32, torch.int64)
            or not cu_seqlens_cpu.is_contiguous()
        ):
            raise ValueError(
                "cu_seqlens_cpu must be a contiguous 1D CPU int32/int64 tensor"
            )
        if require_matching_cpu:
            if cu_seqlens_cpu.dtype != cu_seqlens.dtype:
                raise ValueError("cu_seqlens_cpu dtype must match cu_seqlens")
            if not torch.equal(cu_seqlens.detach().cpu(), cu_seqlens_cpu):
                raise ValueError("cu_seqlens_cpu must match cu_seqlens exactly")
        values = [int(x) for x in cu_seqlens_cpu.tolist()]
    elif int(cu_seqlens.numel()) == 2:
        # Single sequence: by contract cu_seqlens == [0, total_t]. Synthesize it
        # from shape metadata instead of a per-call D2H copy of the CUDA tensor.
        values = [0, int(total_t)]
    else:
        values = [int(x) for x in cu_seqlens.detach().cpu().tolist()]
    if len(values) != int(cu_seqlens.numel()):
        raise ValueError(
            "cu_seqlens_cpu size mismatch: "
            f"expected {cu_seqlens.numel()}, got {len(values)}")
    if len(values) < 2:
        raise ValueError("cu_seqlens must contain at least two entries")
    if values[0] != 0 or values[-1] != int(total_t):
        raise ValueError(
            "packed cu_seqlens must start at 0 and end at total sequence "
            f"length {total_t}, got {values[0]}..{values[-1]}")
    prev = values[0]
    for cur in values[1:]:
        if cur <= prev:
            raise ValueError("packed cu_seqlens must be strictly increasing")
        prev = cur
    return values


def _can_use_state_checkpoints(
    q: torch.Tensor,
    v: torch.Tensor,
    cu_values: list[int],
    state_checkpoints: Optional[torch.Tensor],
    checkpoint_cu_starts: Optional[torch.Tensor],
    checkpoint_every_n_tokens: int,
) -> bool:
    """Cheap metadata-only eligibility checks for the checkpoint fast path."""
    if not isinstance(checkpoint_every_n_tokens, int):
        return False
    if checkpoint_every_n_tokens == 0:
        return state_checkpoints is None and checkpoint_cu_starts is None
    if (
        checkpoint_every_n_tokens < 0
        or checkpoint_every_n_tokens % 64 != 0
        or state_checkpoints is None
        or checkpoint_cu_starts is None
    ):
        return False

    expected_tail = (int(v.shape[2]), int(v.shape[3]), int(q.shape[3]))
    expected_num_checkpoints = sum(
        (end - start) // checkpoint_every_n_tokens
        for start, end in zip(cu_values[:-1], cu_values[1:])
    )
    return (
        state_checkpoints.dtype == torch.float32
        and state_checkpoints.device == q.device
        and state_checkpoints.ndim == 4
        and tuple(state_checkpoints.shape[1:]) == expected_tail
        and int(state_checkpoints.shape[0]) == expected_num_checkpoints
        and state_checkpoints.is_contiguous()
        and checkpoint_cu_starts.dtype == torch.int64
        and checkpoint_cu_starts.device == q.device
        and checkpoint_cu_starts.ndim == 1
        and int(checkpoint_cu_starts.numel()) == len(cu_values)
        and checkpoint_cu_starts.is_contiguous()
    )


def _can_use_cu_seqlens_runtime(
    q: torch.Tensor,
    v: torch.Tensor,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: Optional[torch.Tensor],
    initial_state: Optional[torch.Tensor],
    state_checkpoints: Optional[torch.Tensor],
    checkpoint_cu_starts: Optional[torch.Tensor],
    checkpoint_every_n_tokens: int,
) -> bool:
    if q.shape[0] != 1:
        return False
    if q.shape[-1] != K_DIM or v.shape[-1] != V_DIM:
        return False
    if int(cu_seqlens.numel()) < 2:
        return False
    cu_values = _packed_cu_values(cu_seqlens, cu_seqlens_cpu, int(q.shape[1]))
    if not _is_supported_initial_state(
        initial_state, q, v, len(cu_values) - 1
    ):
        return False
    return _can_use_state_checkpoints(
        q,
        v,
        cu_values,
        state_checkpoints,
        checkpoint_cu_starts,
        checkpoint_every_n_tokens,
    )


def can_use_chunk_gdn_fwd_cutedsl(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    initial_state: Optional[torch.Tensor] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    head_first: bool = False,
    use_qk_l2norm_in_kernel: bool = False,
    cu_seqlens_cpu: Optional[torch.Tensor] = None,
    cp_context=None,
    transpose_state_layout: bool = False,
    allow_padding: bool = False,
    state_checkpoints: Optional[torch.Tensor] = None,
    checkpoint_cu_starts: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
    **kwargs,
) -> bool:
    """Return whether ATREX should handle this GDN chunk call."""
    # ``allow_padding`` does not change the mathematical contract. Tail
    # handling is validated by each architecture-specific implementation.
    del allow_padding
    try:
        if head_first or kwargs:
            return False
        if cu_seqlens is None and cu_seqlens_cpu is not None:
            return False
        target = detect_device_target(q.device)
        implementation = _select_chunk_gdn_implementation(q.device)
        if implementation is None:
            return False
        if target.family == "nvidia" and target.arch == "sm103":
            return _is_supported_sm103_fast_path(
                q,
                k,
                v,
                g,
                beta,
                scale,
                initial_state,
                cu_seqlens,
                cu_seqlens_cpu,
                use_qk_l2norm_in_kernel,
                cp_context,
                transpose_state_layout,
                state_checkpoints,
                checkpoint_cu_starts,
                checkpoint_every_n_tokens,
            )
        has_cu_seqlens = cu_seqlens is not None
        if not _is_supported_sm120_fast_path(
            q,
            k,
            v,
            g,
            beta,
            use_qk_l2norm_in_kernel,
            cp_context,
            transpose_state_layout,
        ):
            return False
        t = int(q.shape[1])
        hv = int(v.shape[2])
        if has_cu_seqlens:
            return _can_use_cu_seqlens_runtime(
                q,
                v,
                cu_seqlens,
                cu_seqlens_cpu,
                initial_state,
                state_checkpoints,
                checkpoint_cu_starts,
                checkpoint_every_n_tokens,
            )
        return (
            _is_supported_initial_state(initial_state, q, v, 1)
            and _can_use_direct_runtime_t(t, hv, output_final_state)
            and _can_use_state_checkpoints(
                q,
                v,
                [0, t],
                state_checkpoints,
                checkpoint_cu_starts,
                checkpoint_every_n_tokens,
            )
        )
    except (ImportError, RuntimeError, TypeError, ValueError):
        return False


def chunk_gdn_fwd_cutedsl_build(
    B: int = 1,
    H: int = 16,
    HV: int = 32,
    K: int = K_DIM,
    V: int = V_DIM,
    warmup: bool = False,
    seq_len: Optional[int] = None,
    output_final_state: bool = False,
    scale: Optional[float] = None,
) -> dict:
    """Build a target-specific CuTeDSL Chunk-GDN forward context.

    Called once during model initialization. The context is a plain metadata dict;
    all prefill runs on the varlen megakernel (compile-once), so sequence length
    stays dynamic and NO per-length kernel is precompiled here. ``seq_len`` (when
    given) is only recorded for shape validation. ``warmup``/``seq_len`` are kept for
    backward-compatible signatures; runtime compile is avoided via prewarm_buckets.
    """
    _check_chunk_gdn_current_device()
    target = detect_device_target()
    if B != 1:
        raise ValueError(f"only B=1 is supported, got B={B}")
    if target.family == "nvidia" and target.arch == "sm103":
        if (H, HV, K, V) != (SM103_H, SM103_HV, K_DIM, V_DIM):
            raise ValueError(
                "SM103 AKA M64 requires H/HV/K/V=(4, 32, 128, 128), "
                f"got {(H, HV, K, V)}"
            )
        requested_scale = SM103_SCALE if scale is None else float(scale)
        if not math.isclose(
            requested_scale, SM103_SCALE, rel_tol=0.0, abs_tol=1e-12
        ):
            raise ValueError(
                f"SM103 AKA M64 requires scale={SM103_SCALE}, got {requested_scale}"
            )
    if K != K_DIM or V != V_DIM:
        raise ValueError(f"only K={K_DIM}, V={V_DIM} are supported, got K={K}, V={V}")
    if HV % H != 0:
        raise ValueError(f"HV must be divisible by H, got HV={HV}, H={H}")
    if V % BV != 0:
        raise ValueError(f"V must be divisible by {BV}, got {V}")
    if seq_len is not None and seq_len <= 0:
        raise ValueError(f"seq_len must be positive, got {seq_len}")
    if warmup and seq_len is None:
        raise ValueError("warmup=True requires seq_len so kernels can be initialized statically")

    init_scale = float(scale) if scale is not None else 1.0 / math.sqrt(K)

    ctx = {
        "B": int(B),
        "H": int(H),
        "HV": int(HV),
        "K": int(K),
        "V": int(V),
        "BT": BT,
        "BV": BV,
        "T": int(seq_len) if seq_len is not None else None,
        "scale": init_scale if seq_len is not None else None,
        "output_final_state": output_final_state if seq_len is not None else None,
        "static_initialized": False,
        "target_arch": target.arch,
    }
    return ctx


def _dispatch_chunk_gdn_prefill(
    ctx: dict,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float,
    output_final_state: bool,
    cu_seqlens: Optional[torch.Tensor],
    cu_seqlens_cpu: Optional[torch.Tensor],
    qk_l2norm_already_applied: bool = False,
    gate_is_exp: bool = False,
    initial_state: Optional[torch.Tensor] = None,
    state_checkpoints: Optional[torch.Tensor] = None,
    checkpoint_cu_starts: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Dispatch single- and multi-sequence prefill to the selected kernel."""
    if cu_seqlens is None:
        cu_seqlens_cpu = torch.tensor(
            [0, int(q.shape[1])], dtype=torch.int32
        )
        cu_seqlens = torch.tensor(
            [0, int(q.shape[1])], device=q.device, dtype=torch.int32
        )
    return _chunk_gdn_fwd_cutedsl_cu_seqlens(
        ctx, q, k, v, g, beta,
        scale=scale,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        qk_l2norm_already_applied=qk_l2norm_already_applied,
        gate_is_exp=gate_is_exp,
        initial_state=initial_state,
        state_checkpoints=state_checkpoints,
        checkpoint_cu_starts=checkpoint_cu_starts,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
    )


def chunk_gdn_fwd_cutedsl(
    ctx: dict,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    scale: Optional[float] = None,
    output_final_state: bool = False,
    cu_seqlens: Optional[torch.Tensor] = None,
    cu_seqlens_cpu: Optional[torch.Tensor] = None,
    qk_l2norm_already_applied: bool = False,
    gate_is_exp: bool = False,
    initial_state: Optional[torch.Tensor] = None,
    state_checkpoints: Optional[torch.Tensor] = None,
    checkpoint_cu_starts: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Run the selected NVIDIA CuTeDSL Chunk-GDN forward implementation.

    Args:
        ctx: Context from :func:`chunk_gdn_fwd_cutedsl_build`.
        q: [B, T, H, K=128] bf16.
        k: [B, T, H, K=128] bf16.
        v: [B, T, HV, V=128] bf16.
        g: [B, T, HV] bf16/fp32 log-decay gate.
        beta: [B, T, HV] bf16/fp32 mixing weight.
        scale: Optional QK scale. Defaults to ``1 / sqrt(K)``.
        output_final_state: Whether to return the final recurrent state.
        state_checkpoints: Optional fp32 checkpoint output buffer with shape
            ``[num_checkpoints, HV, V, K]``.
        checkpoint_cu_starts: Contiguous per-sequence checkpoint offsets on
            CUDA. The caller must construct this together with
            ``state_checkpoints`` and reuse the same plan when consuming the
            checkpoint buffer. ATREX consumes this tensor directly.
        checkpoint_every_n_tokens: Checkpoint interval. Zero disables checkpointing.

    Returns:
        ``(o, final_state)``. ``o`` is [B, T, HV, V] bf16. ``final_state`` is
        [B, HV, V, K] fp32 when requested, otherwise ``None``.
    """
    requested_scale = scale
    if requested_scale is None:
        requested_scale = ctx["scale"] if ctx.get("scale") is not None else 1.0 / math.sqrt(ctx["K"])
    requested_scale = float(requested_scale)

    # Calls without cu_seqlens use a synthesized [0, t] for one sequence.
    return _dispatch_chunk_gdn_prefill(
        ctx,
        q,
        k,
        v,
        g,
        beta,
        scale=requested_scale,
        output_final_state=output_final_state,
        cu_seqlens=cu_seqlens,
        cu_seqlens_cpu=cu_seqlens_cpu,
        qk_l2norm_already_applied=qk_l2norm_already_applied,
        gate_is_exp=gate_is_exp,
        initial_state=initial_state,
        state_checkpoints=state_checkpoints,
        checkpoint_cu_starts=checkpoint_cu_starts,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
    )


@torch.no_grad()
def chunk_gdn_fwd_cutedsl_prewarm_buckets(
    *,
    B: int = 1,
    H: int = 16,
    HV: int = 32,
    K: int = 128,
    V: int = 128,
    output_final_state: bool = True,
    scale: Optional[float] = None,
    include_state_checkpoints: bool = False,
) -> None:
    """Precompile every varlen GDN prefill compile bucket at init => no runtime JIT.

    All prefill (single- AND multi-seq) now routes to the varlen megakernel. Its
    compile key reduces to {aligned_full_blocks} x {needs_init_state} x
    {cu_seqlens_dtype} and, optionally, {needs_checkpointing}. Sequence length and
    segment count are runtime, so one compile per bucket serves every shape. We
    trigger each with a tiny single-seq exemplar (aligned len 128 / tail len 97, x
    no-init/init, x int32/int64 cu_seqlens) plus the fused Q/K normalizer. SGLang
    enables checkpoint prewarming for radix-cache state tracking; other callers
    retain the non-checkpoint variants. B is fixed at one packed batch. The kernel
    always materializes its internal state workspace; the public API discards it
    when final state is not requested.
    """
    if B != 1:
        raise ValueError(f"only B=1 is supported, got B={B}")
    del output_final_state
    if K != K_DIM:
        raise ValueError(f"fused Q/K normalization requires K={K_DIM}, got {K}")
    implementation = _check_chunk_gdn_current_device()

    dev_idx = torch.cuda.current_device()
    dev = torch.device(f"cuda:{dev_idx}")
    sc = float(scale) if scale is not None else 1.0 / math.sqrt(K)
    target = detect_device_target(dev)
    if target.family == "nvidia" and target.arch == "sm103":
        if include_state_checkpoints:
            raise ValueError("SM103 AKA M64 does not support state checkpoints")
        if (H, HV, K, V) != (SM103_H, SM103_HV, K_DIM, V_DIM):
            raise ValueError(
                "SM103 AKA M64 requires H/HV/K/V=(4, 32, 128, 128), "
                f"got {(H, HV, K, V)}"
            )
        if not math.isclose(sc, SM103_SCALE, rel_tol=0.0, abs_tol=1e-12):
            raise ValueError(f"SM103 AKA M64 requires scale={SM103_SCALE}, got {sc}")
        ctx = {
            "B": 1, "H": SM103_H, "HV": SM103_HV,
            "K": K_DIM, "V": V_DIM, "BT": BT, "BV": BV,
            "T": None, "scale": None, "output_final_state": None,
            "static_initialized": False, "target_arch": "sm103",
        }
        for lengths in ((128,), (97,), (65, 63)):
            seqlen = sum(lengths)
            q = torch.randn(
                (1, seqlen, SM103_H, K_DIM), device=dev, dtype=torch.bfloat16
            )
            k = torch.randn_like(q)
            q = torch.nn.functional.normalize(q.float(), dim=-1).to(torch.bfloat16)
            k = torch.nn.functional.normalize(k.float(), dim=-1).to(torch.bfloat16)
            v = torch.randn(
                (1, seqlen, SM103_HV, V_DIM),
                device=dev,
                dtype=torch.bfloat16,
            )
            g = torch.nn.functional.logsigmoid(
                torch.randn((1, seqlen, SM103_HV), device=dev)
            )
            beta = torch.sigmoid(
                torch.randn((1, seqlen, SM103_HV), device=dev)
            )
            offsets = [0]
            for length in lengths:
                offsets.append(offsets[-1] + length)
            cu_cpu = torch.tensor(offsets, dtype=torch.int32)
            cu = cu_cpu.to(device=dev)
            init = torch.zeros(
                (len(lengths), SM103_HV, V_DIM, K_DIM),
                device=dev,
                dtype=torch.float32,
            )
            _chunk_gdn_fwd_cutedsl_cu_seqlens(
                ctx,
                q,
                k,
                v,
                g,
                beta,
                scale=sc,
                output_final_state=True,
                cu_seqlens=cu,
                cu_seqlens_cpu=cu_cpu,
                qk_l2norm_already_applied=True,
                initial_state=init,
            )
        torch.cuda.synchronize()
        return

    implementation.prewarm_fused_qk_l2_normalize_bf16(device=dev_idx)
    ctx = {
        "B": 1, "H": int(H), "HV": int(HV), "K": int(K), "V": int(V),
        "BT": BT, "BV": BV, "T": None, "scale": None,
        "output_final_state": None, "static_initialized": False,
        "target_arch": target.arch,
    }
    checkpoint_modes = (False, True) if include_state_checkpoints else (False,)
    for seqlen in (BT * 4, BT * 3 + 1):
        for with_init in (False, True):
            q = torch.randn((1, seqlen, int(H), int(K)), device=dev, dtype=torch.bfloat16)
            k = torch.randn_like(q)
            v = torch.randn((1, seqlen, int(HV), int(V)), device=dev, dtype=torch.bfloat16)
            g = torch.nn.functional.logsigmoid(
                torch.randn((1, seqlen, int(HV)), device=dev, dtype=torch.float32)
            ).to(torch.bfloat16)
            beta = torch.sigmoid(
                torch.randn((1, seqlen, int(HV)), device=dev, dtype=torch.float32)
            ).to(torch.bfloat16)
            init = (
                torch.zeros((1, int(HV), int(V), int(K)), device=dev, dtype=torch.float32)
                if with_init else None
            )
            for cu_dtype in (torch.int32, torch.int64):
                cu = torch.tensor([0, seqlen], device=dev, dtype=cu_dtype)
                cu_cpu = torch.tensor([0, seqlen], dtype=cu_dtype)
                for with_checkpoints in checkpoint_modes:
                    num_checkpoints = seqlen // 64 if with_checkpoints else 0
                    checkpoint_cu_starts = (
                        torch.tensor(
                            [0, num_checkpoints], device=dev, dtype=torch.int64
                        )
                        if with_checkpoints else None
                    )
                    state_checkpoints = (
                        torch.empty(
                            (num_checkpoints, int(HV), int(V), int(K)),
                            device=dev,
                            dtype=torch.float32,
                        )
                        if with_checkpoints else None
                    )
                    _chunk_gdn_fwd_cutedsl_cu_seqlens(
                        ctx, q, k, v, g, beta,
                        scale=sc, output_final_state=True,
                        cu_seqlens=cu, cu_seqlens_cpu=cu_cpu, initial_state=init,
                        state_checkpoints=state_checkpoints,
                        checkpoint_cu_starts=checkpoint_cu_starts,
                        checkpoint_every_n_tokens=64 if with_checkpoints else 0,
                    )
    torch.cuda.synchronize()


def _chunk_gdn_fwd_cutedsl_cu_seqlens(
    ctx: dict,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    g: torch.Tensor,
    beta: torch.Tensor,
    *,
    scale: float,
    output_final_state: bool,
    cu_seqlens: torch.Tensor,
    cu_seqlens_cpu: Optional[torch.Tensor],
    qk_l2norm_already_applied: bool = False,
    gate_is_exp: bool = False,
    initial_state: Optional[torch.Tensor] = None,
    state_checkpoints: Optional[torch.Tensor] = None,
    checkpoint_cu_starts: Optional[torch.Tensor] = None,
    checkpoint_every_n_tokens: int = 0,
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    implementation = _validate_inputs(
        {**ctx, "T": int(q.shape[1])},
        q,
        k,
        v,
        g,
        beta,
    )
    target = detect_device_target(q.device)
    ctx_target = ctx.get("target_arch")
    if ctx_target is not None and ctx_target != target.arch:
        raise ValueError(
            f"Chunk-GDN context targets {ctx_target}, but inputs are on {target.arch}"
        )
    if q.shape[0] != 1:
        raise ValueError(
            "ATREX cu_seqlens GDN prefill expects packed B=1 tensors, "
            f"got B={q.shape[0]}")
    cu_values = _packed_cu_values(
        cu_seqlens,
        cu_seqlens_cpu,
        int(q.shape[1]),
        require_matching_cpu=(
            target.family == "nvidia" and target.arch == "sm103"
        ),
    )
    if target.family == "nvidia" and target.arch == "sm103":
        if not _is_supported_sm103_fast_path(
            q,
            k,
            v,
            g,
            beta,
            scale,
            initial_state,
            cu_seqlens,
            cu_seqlens_cpu,
            not qk_l2norm_already_applied,
            None,
            False,
            state_checkpoints,
            checkpoint_cu_starts,
            checkpoint_every_n_tokens,
        ):
            raise ValueError(
                "call is outside the verified SM103 AKA M64 contract; Q/K "
                "must already be L2-normalized and all SM103 eligibility "
                "requirements must hold"
            )
        q_flat = q.squeeze(0)
        k_flat = k.squeeze(0)
        v_flat = v.squeeze(0)
        gate = g.squeeze(0)
        if not gate_is_exp:
            gate = torch.exp(gate)
        gate = gate.contiguous()
        beta_flat = beta.squeeze(0)
        cu_seqlens_i32 = (
            cu_seqlens
            if cu_seqlens.dtype == torch.int32
            else cu_seqlens.to(dtype=torch.int32)
        )
        num_seqs = len(cu_values) - 1
        output = torch.empty(
            (int(q.shape[1]), SM103_HV, V_DIM),
            dtype=torch.bfloat16,
            device=q.device,
        )
        final_state = torch.empty(
            (num_seqs, SM103_HV, V_DIM, K_DIM),
            dtype=torch.float32,
            device=q.device,
        )
        implementation.atrex_aka_chunk_gated_delta_rule_sm103_m64(
            q_flat,
            k_flat,
            v_flat,
            gate,
            beta_flat,
            output,
            cu_seqlens_i32,
            initial_state,
            final_state,
            float(scale),
        )
        return output.unsqueeze(0), final_state if output_final_state else None

    num_seqs = len(cu_values) - 1
    total_t = int(q.shape[1])
    hv = int(v.shape[2])
    k_dim = int(q.shape[3])
    v_dim = int(v.shape[3])
    aligned_full_blocks = all(
        ((end - start) % 64) == 0
        for start, end in zip(cu_values[:-1], cu_values[1:])
    )

    if not isinstance(checkpoint_every_n_tokens, int):
        raise TypeError("checkpoint_every_n_tokens must be an integer")
    needs_checkpointing = checkpoint_every_n_tokens > 0
    if checkpoint_every_n_tokens < 0:
        raise ValueError(
            "checkpoint_every_n_tokens must be non-negative, "
            f"got {checkpoint_every_n_tokens}"
        )
    if needs_checkpointing:
        if checkpoint_every_n_tokens % 64 != 0:
            raise ValueError(
                "checkpoint_every_n_tokens must be a multiple of 64, "
                f"got {checkpoint_every_n_tokens}"
            )
        if state_checkpoints is None or checkpoint_cu_starts is None:
            raise ValueError(
                "state_checkpoints and checkpoint_cu_starts are required "
                "when checkpoint_every_n_tokens > 0"
            )
        expected_tail = (hv, v_dim, k_dim)
        if (
            state_checkpoints.dtype != torch.float32
            or state_checkpoints.device != q.device
            or state_checkpoints.ndim != 4
            or tuple(state_checkpoints.shape[1:]) != expected_tail
            or not state_checkpoints.is_contiguous()
        ):
            raise ValueError(
                "state_checkpoints must be a contiguous CUDA fp32 tensor with shape "
                f"[num_checkpoints, {hv}, {v_dim}, {k_dim}], got "
                f"shape={tuple(state_checkpoints.shape)}, "
                f"dtype={state_checkpoints.dtype}, device={state_checkpoints.device}"
            )
        expected_num_checkpoints = sum(
            (end - start) // checkpoint_every_n_tokens
            for start, end in zip(cu_values[:-1], cu_values[1:])
        )
        if int(state_checkpoints.shape[0]) != expected_num_checkpoints:
            raise ValueError(
                "state_checkpoints row count must match the packed sequence "
                f"checkpoint plan: expected {expected_num_checkpoints}, got "
                f"{state_checkpoints.shape[0]}"
            )
        if (
            checkpoint_cu_starts.dtype != torch.int64
            or checkpoint_cu_starts.device != q.device
            or checkpoint_cu_starts.ndim != 1
            or checkpoint_cu_starts.numel() != num_seqs + 1
            or not checkpoint_cu_starts.is_contiguous()
        ):
            raise ValueError(
                "checkpoint_cu_starts must be a contiguous CUDA int64 tensor "
                "with shape "
                f"[{num_seqs + 1}], got shape={tuple(checkpoint_cu_starts.shape)}, "
                f"dtype={checkpoint_cu_starts.dtype}, "
                f"device={checkpoint_cu_starts.device}"
            )
    elif state_checkpoints is not None or checkpoint_cu_starts is not None:
        raise ValueError(
            "state_checkpoints and checkpoint_cu_starts must be None "
            "when checkpoint_every_n_tokens == 0"
        )
    q_flat = q.squeeze(0).contiguous()
    k_flat = k.squeeze(0).contiguous()
    if not qk_l2norm_already_applied:
        q_flat, k_flat = implementation.fused_qk_l2_normalize_bf16(
            q_flat, k_flat
        )
    v_flat = v.squeeze(0).contiguous()
    gate = g.squeeze(0).float()
    if not gate_is_exp:
        gate = torch.exp(gate).contiguous()
    beta_flat = beta.squeeze(0).float()
    # Carry-in state for chunked-prefill continuations. The varlen kernel reads
    # init_state with the SAME (num_seqs, HV, V, K) fp32 layout it uses to WRITE
    # final_state (kv_load/kv_store share the index convention), so final_state
    # feeds straight back with no transpose. Empirically verified: direct
    # rel_err ~3e-3, transposed ~1.4 (test_init_state.py --stage=noinit).
    init_state_arg = None
    if initial_state is not None:
        expected = (num_seqs, hv, v_dim, k_dim)
        if tuple(initial_state.shape) != expected:
            raise ValueError(
                "initial_state must have shape (num_seqs, HV, V, K)="
                f"{expected}, got {tuple(initial_state.shape)}")
        if initial_state.dtype != torch.float32:
            raise TypeError(
                "initial_state dtype mismatch: expected torch.float32, "
                f"got {initial_state.dtype}"
            )
        if initial_state.device != q.device:
            raise ValueError("initial_state must be on the same CUDA device as q")
        init_state_arg = initial_state.contiguous()

    output = torch.empty(
        (total_t, hv, v_dim),
        dtype=v.dtype,
        device=v.device,
    )
    final_state = torch.empty(
        (num_seqs, hv, v_dim, k_dim),
        dtype=torch.float32,
        device=v.device,
    )

    implementation.delta_rule_prefill_dsl_sm120(
        output,
        final_state,
        q_flat,
        k_flat,
        v_flat,
        init_state_arg,
        gate,
        beta_flat,
        cu_seqlens,
        float(scale),
        state_checkpoints=state_checkpoints,
        checkpoint_cu_starts=checkpoint_cu_starts,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        split_v_parts=2,
        aligned_full_blocks=aligned_full_blocks,
    )

    return output.unsqueeze(0), final_state if output_final_state else None
