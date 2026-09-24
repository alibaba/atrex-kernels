"""
Copyright (c) 2025 by FlashInfer team.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

  http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

Gated Delta Net Chunked Prefill - Blackwell SM103 AKA Adapter
==============================================================

Bridges ATREX's prepared-input Chunk-GDN API to the verified AKA M64 CuTe DSL
kernel for SM103 (Blackwell).

Follows the same compile-once-cache-and-replay pattern used by the decode
kernels in ``gdn_decode_pretranspose.py``.

State layout: ``[N, H, V, K]``.
"""

import functools
import threading
from typing import Optional

import torch

import cuda.bindings.driver as cuda
import cutlass
import cutlass.cute as cute
from cutlass.cute.runtime import from_dlpack

from .gated_delta_net_chunked import GatedDeltaNetChunkedKernel


# ---------------------------------------------------------------------------
# Compilation cache
# ---------------------------------------------------------------------------


# Keyed on static kernel configuration. Head counts (HQ, HV) are part of
# the key because the tile scheduler and GQA reshape logic bake them in.
@functools.cache
def _get_compiled_cache(
    device_index: int,
    io_dtype_str: str,
    state_dtype_str: str,
    HQ: int,
    HV: int,
    is_GQA: bool,
    use_initial_state: bool,
    store_final_state: bool,
    enable_checkpoints: bool,
    use_state_indices: bool,
    use_m64_rows: bool,
    use_static_tma_descriptors: bool,
    static_tma_full_chunks: bool,
    use_static_scale: bool,
    compact_cg1_events: bool,
):
    """Return a mutable dict that lazily stores the compiled kernel."""
    return {}


_compile_lock = threading.Lock()


def _get_num_sm(device: torch.device) -> int:
    return torch.cuda.get_device_properties(device).multi_processor_count


def _cutlass_io_dtype(torch_dtype: torch.dtype):
    if torch_dtype == torch.bfloat16:
        return cutlass.BFloat16
    elif torch_dtype == torch.float16:
        return cutlass.Float16
    else:
        raise ValueError(
            f"Unsupported dtype {torch_dtype}, expected bfloat16 or float16"
        )


def _cutlass_state_dtype(torch_dtype: torch.dtype):
    if torch_dtype == torch.float32:
        return cutlass.Float32
    elif torch_dtype == torch.bfloat16:
        return cutlass.BFloat16
    elif torch_dtype == torch.float16:
        return cutlass.Float16
    elif torch_dtype == torch.float8_e4m3fn:
        return cutlass.Float8E4M3FN
    elif torch_dtype == torch.float8_e5m2:
        return cutlass.Float8E5M2
    else:
        raise ValueError(
            f"Unsupported state dtype {torch_dtype}, expected float32, bfloat16, "
            "float16, float8_e4m3fn, or float8_e5m2"
        )


@functools.cache
def _device_capability(device_index: int) -> tuple[int, int]:
    return torch.cuda.get_device_capability(device_index)


def _device_index(q: torch.Tensor) -> int:
    return q.device.index if q.device.index is not None else torch.cuda.current_device()


def _get_gdn_launch_metadata(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    output: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    output_state: Optional[torch.Tensor],
    scale: float,
    checkpoint_every_n_tokens: int = 0,
    state_indices: Optional[torch.Tensor] = None,
    *,
    value_rows_override: int | None = None,
) -> dict[str, object]:
    if value_rows_override not in (None, 64, 128):
        raise ValueError("value_rows_override must be None, 64, or 128")
    if q.dtype not in (torch.float16, torch.bfloat16):
        raise ValueError("q must be float16 or bfloat16")
    if k.dtype != q.dtype or v.dtype != q.dtype or output.dtype != q.dtype:
        raise ValueError("k, v, and output must have the same dtype as q")
    if gate.dtype != torch.float32 or beta.dtype != torch.float32:
        raise ValueError("gate and beta must be float32")

    device_index = _device_index(q)
    capability = _device_capability(device_index)
    hq = q.size(1)
    hv = v.size(1)
    dk = q.size(2)
    batch = cu_seqlens.size(0) - 1
    enable_checkpoints = checkpoint_every_n_tokens > 0
    use_state_indices = state_indices is not None
    natural_m64_eligible = (
        capability == (10, 3)
        and dk == 128
        and k.size(2) == 128
        and v.size(2) == 128
        and hv >= hq
        and hv % hq == 0
        and initial_state is not None
        and output_state is not None
        and initial_state.dtype == torch.float32
        and output_state.dtype == torch.float32
        and not enable_checkpoints
        and not use_state_indices
    )
    if value_rows_override == 64 and not natural_m64_eligible:
        raise ValueError("value_rows_override=64 requires natural M64 eligibility")
    use_m64_rows = natural_m64_eligible and value_rows_override != 128
    use_static_tma_descriptors = use_m64_rows and batch == 1
    # The kernel schedules 64-token chunks in 128-token pairs.  A B=1 length
    # divisible by 64 but not 128 still executes one synthetic padding chunk,
    # whose gate must retain the predicated neutral value 1.0.
    static_tma_full_chunks = use_static_tma_descriptors and q.size(0) % 128 == 0
    use_static_scale = use_static_tma_descriptors and scale == dk**-0.5
    compact_cg1_events = use_static_tma_descriptors or (
        use_m64_rows and batch == 2 and hq == hv
    )
    return {
        "requested_value_rows": value_rows_override,
        "selected_value_rows": 64 if use_m64_rows else 128,
        "capability": list(capability),
        "dtypes": {
            "q": str(q.dtype),
            "k": str(k.dtype),
            "v": str(v.dtype),
            "gate": str(gate.dtype),
            "beta": str(beta.dtype),
            "output": str(output.dtype),
            "initial_state": str(initial_state.dtype) if initial_state is not None else None,
            "output_state": str(output_state.dtype) if output_state is not None else None,
        },
        "natural_m64_eligible": natural_m64_eligible,
        "use_m64_rows": use_m64_rows,
        "use_static_tma_descriptors": use_static_tma_descriptors,
        "static_tma_full_chunks": static_tma_full_chunks,
        "use_static_scale": use_static_scale,
        "compact_cg1_events": compact_cg1_events,
    }


def _mark_state_layout(s_cute, use_state_indices: bool, DK: int) -> None:
    """Mark the recurrent-state tensor layout for compilation.

    Packed mode (``use_state_indices=False``): the caller passes a compact
    ``[num_seqs, H, V, K]`` state, so we keep the original marking that both
    makes the layout dynamic (stride-1 K dim auto-deduced) and attaches a
    ``divisibility=DK`` hint on the K dim for wider vectorized state copies.

    Pool/indexed mode (``use_state_indices=True``): the caller passes its real
    SSM state pool ``[N_pool, H, V, K]`` whose dim-0 (slot) stride is padded by
    the mamba conv+ssm cache packing, i.e. ``stride[0] > H*V*K`` -> the layout is
    NON-COMPACT. ``mark_compact_shape_dynamic`` asserts a compact layout and
    raises ``RuntimeError: The stride_order is not consistent with the layout``.
    ``mark_layout_dynamic()`` alone pins the single stride-1 dim (mode 3 = K) to
    stride 1 and carries every other stride (including the padded dim-0) through
    as a dynamic runtime value. The kernel addresses the state purely via
    ``s.stride[...]`` (reshape at ~475-497 and ``mS_init/mS_out[..., state_row]``),
    so a padded dim-0 stride is handled correctly. cutlass-dsl offers no way to
    attach a divisibility hint without also requiring compactness, so this path
    drops the ``divisibility=DK`` hint; the stride-1 K dim is retained, so the
    128x128 state autovec copy still vectorizes (possibly a narrower vector).
    That copy is a negligible fraction of the kernel, so correctness is kept
    with no meaningful perf cost.
    """
    if use_state_indices:
        s_cute.mark_layout_dynamic()
    else:
        s_cute.mark_layout_dynamic().mark_compact_shape_dynamic(
            mode=3, stride_order=(0, 1, 2, 3), divisibility=DK
        )


# ---------------------------------------------------------------------------
# Internal launch machinery
# ---------------------------------------------------------------------------


def _launch_chunk_gated_delta_rule(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    output: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    output_state: Optional[torch.Tensor],
    scale: float,
    checkpoint_every_n_tokens: int = 0,
    cu_checkpoints: Optional[torch.Tensor] = None,
    output_checkpoints: Optional[torch.Tensor] = None,
    state_indices: Optional[torch.Tensor] = None,
    *,
    value_rows_override: int | None = None,
) -> None:
    """Internal host launcher shared with the original verified source.

    All tensors must be contiguous and on the same CUDA device.

    Args:
        q: ``(total_tokens, HQ, DK)`` float16/bfloat16
        k: ``(total_tokens, HK, DK)`` float16/bfloat16
        v: ``(total_tokens, HV, DK)`` float16/bfloat16
        gate: ``(total_tokens, HO)`` float32, forget gate
        beta: ``(total_tokens, HO)`` float32, update gate
        output: ``(total_tokens, HO, DK)`` float16/bfloat16, pre-allocated
        cu_seqlens: ``(num_seqs + 1,)`` int32
        initial_state: ``(num_seqs, HO, DK, DK)`` float32/bfloat16/float16/fp8, or None
        output_state: ``(num_seqs, HO, DK, DK)`` float32/bfloat16/float16/fp8, or None
        scale: attention scale factor (must not be 0)
        checkpoint_every_n_tokens: store intermediate state every N tokens (0 = disabled)
        cu_checkpoints: ``(num_seqs + 1,)`` int32, cumulative checkpoint counts
        output_checkpoints: ``(total_checkpoints, HO, DK, DK)`` float32/bfloat16/float16/fp8, or None
    """
    HQ = q.size(1)
    HV = v.size(1)
    DK = q.size(2)
    is_GQA = HQ >= HV
    use_initial_state = initial_state is not None
    store_final_state = output_state is not None
    enable_checkpoints = checkpoint_every_n_tokens > 0
    io_dtype = _cutlass_io_dtype(q.dtype)

    # Auto-detect state dtype from initial_state, default to float32
    if initial_state is not None:
        state_torch_dtype = initial_state.dtype
    elif output_state is not None:
        state_torch_dtype = output_state.dtype
    else:
        state_torch_dtype = torch.float32
    state_dtype = _cutlass_state_dtype(state_torch_dtype)

    _initial_state = initial_state if use_initial_state else None
    B = cu_seqlens.size(0) - 1
    _output_state = output_state if store_final_state else None
    use_state_indices = state_indices is not None
    _state_indices = state_indices if use_state_indices else None

    launch_metadata = _get_gdn_launch_metadata(
        q=q,
        k=k,
        v=v,
        gate=gate,
        beta=beta,
        output=output,
        cu_seqlens=cu_seqlens,
        initial_state=initial_state,
        output_state=output_state,
        scale=scale,
        checkpoint_every_n_tokens=checkpoint_every_n_tokens,
        state_indices=state_indices,
        value_rows_override=value_rows_override,
    )
    use_m64_rows = bool(launch_metadata["use_m64_rows"])
    use_static_tma_descriptors = bool(
        launch_metadata["use_static_tma_descriptors"]
    )
    static_tma_full_chunks = bool(launch_metadata["static_tma_full_chunks"])
    use_static_scale = bool(launch_metadata["use_static_scale"])
    compact_cg1_events = bool(launch_metadata["compact_cg1_events"])

    device_index = _device_index(q)
    cache = _get_compiled_cache(
        device_index,
        str(q.dtype),
        str(state_torch_dtype),
        HQ,
        HV,
        is_GQA,
        use_initial_state,
        store_final_state,
        enable_checkpoints,
        use_state_indices,
        use_m64_rows,
        use_static_tma_descriptors,
        static_tma_full_chunks,
        use_static_scale,
        compact_cg1_events,
    )

    if "compiled" not in cache:
        # CuTe compilation is process-global. Serialize the first compile for
        # one specialization while allowing subsequent launches to proceed
        # without holding the lock.
        with _compile_lock:
            if "compiled" not in cache:
                _compile_kernel(
                    cache=cache,
                    q=q,
                    k=k,
                    v=v,
                    gate=gate,
                    beta=beta,
                    output=output,
                    cu_seqlens=cu_seqlens,
                    initial_state=_initial_state,
                    output_state=_output_state,
                    state_indices=_state_indices,
                    output_checkpoints=output_checkpoints,
                    cu_checkpoints=cu_checkpoints,
                    checkpoint_every_n_tokens=checkpoint_every_n_tokens,
                    scale=scale,
                    io_dtype=io_dtype,
                    state_dtype=state_dtype,
                    HQ=HQ,
                    HV=HV,
                    DK=DK,
                    B=B,
                    is_GQA=is_GQA,
                    use_initial_state=use_initial_state,
                    store_final_state=store_final_state,
                    enable_checkpoints=enable_checkpoints,
                    use_state_indices=use_state_indices,
                    use_m64_rows=use_m64_rows,
                    use_static_tma_descriptors=use_static_tma_descriptors,
                    static_tma_full_chunks=static_tma_full_chunks,
                    use_static_scale=use_static_scale,
                    compact_cg1_events=compact_cg1_events,
                )

    # --- Execute ---
    compiled = cache["compiled"]
    num_sm = cache["num_sm"]

    workspace_size = GatedDeltaNetChunkedKernel.get_workspace_size(
        num_sm, B, HQ, HV, True
    )
    stream_value = torch.cuda.current_stream(device=q.device).cuda_stream
    # TMA descriptor workspaces are mutated by a launch. Keep one workspace
    # per CUDA stream so concurrent callers cannot race on descriptor updates.
    ws_key = ("workspace", device_index, int(stream_value))
    if ws_key not in cache or cache[ws_key].size(0) < workspace_size:
        cache[ws_key] = torch.empty(workspace_size, dtype=torch.int8, device=q.device)
    workspace = cache[ws_key]

    stream = cuda.CUstream(stream_value)
    compiled(
        q,
        k,
        v,
        gate,
        beta,
        output,
        cu_seqlens,
        _initial_state,
        _output_state,
        _state_indices,
        output_checkpoints,
        cu_checkpoints,
        checkpoint_every_n_tokens,
        scale,
        workspace,
        stream,
    )


def _compile_kernel(
    *,
    cache: dict,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    output: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_state: Optional[torch.Tensor],
    output_state: Optional[torch.Tensor],
    state_indices: Optional[torch.Tensor],
    output_checkpoints: Optional[torch.Tensor],
    cu_checkpoints: Optional[torch.Tensor],
    checkpoint_every_n_tokens: int,
    scale: float,
    io_dtype,
    state_dtype,
    HQ: int,
    HV: int,
    DK: int,
    B: int,
    is_GQA: bool,
    use_initial_state: bool,
    store_final_state: bool,
    enable_checkpoints: bool,
    use_state_indices: bool,
    use_m64_rows: bool,
    use_static_tma_descriptors: bool,
    static_tma_full_chunks: bool,
    use_static_scale: bool,
    compact_cg1_events: bool,
) -> None:
        # --- First call: compile the kernel ---
        num_sm = _get_num_sm(q.device)
        max_active_clusters = num_sm

        value_rows = 64 if use_m64_rows else 128
        gdn = GatedDeltaNetChunkedKernel(
            io_dtype=io_dtype,
            # The kernel requires the triangular-inverse dtype to match io_dtype
            # (asserted in its __init__); pass io_dtype so bf16 uses the same
            # validated path as fp16 instead of the default Float16.
            inverse_dtype=io_dtype,
            acc_dtype=cutlass.Float32,
            state_dtype=state_dtype,
            mma_tiler_qk=(64, 64, 128),
            mma_tiler_qs=(value_rows, 64, 128),
            mma_tiler_qkv=(value_rows, 64, 64),
            mma_tiler_kv=(value_rows, 128, 64),
            max_active_clusters=max_active_clusters,
            num_sm=num_sm,
            is_GQA=is_GQA,
            use_initial_state=use_initial_state,
            store_final_state=store_final_state,
            enable_checkpoints=enable_checkpoints,
            is_persistent=True,
            use_static_tma_descriptors=use_static_tma_descriptors,
            static_tma_full_chunks=static_tma_full_chunks,
            use_static_scale=use_static_scale,
            compact_cg1_events=compact_cg1_events,
        )
        gdn.kernel.set_name_prefix("atrex_aka")

        # Convert PyTorch tensors to CuTe tensors for compilation.
        # Token dimension (dim 0) must be dynamic to handle varying seq lengths.
        # Head and head_dim dimensions stay static (part of cache key).
        q_cute = from_dlpack(q, assumed_align=16)
        q_cute.mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1, 2), divisibility=1
        )
        k_cute = from_dlpack(k, assumed_align=16)
        k_cute.mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1, 2), divisibility=1
        )
        v_cute = from_dlpack(v, assumed_align=16)
        v_cute.mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1, 2), divisibility=1
        )
        gate_cute = from_dlpack(gate, assumed_align=16)
        gate_cute.mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1), divisibility=1
        )
        beta_cute = from_dlpack(beta, assumed_align=16)
        beta_cute.mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1), divisibility=1
        )
        o_cute = from_dlpack(output, assumed_align=16)
        o_cute.mark_compact_shape_dynamic(
            mode=0, stride_order=(0, 1, 2), divisibility=1
        )
        cu_seqlens_cute = from_dlpack(cu_seqlens, assumed_align=4).mark_layout_dynamic()

        s_in_cute = None
        if use_initial_state:
            s_in_cute = from_dlpack(initial_state, assumed_align=16)
            _mark_state_layout(s_in_cute, use_state_indices, DK)

        s_out_cute = None
        if store_final_state:
            s_out_cute = from_dlpack(output_state, assumed_align=16)
            _mark_state_layout(s_out_cute, use_state_indices, DK)

        s_indices_cute = None
        if use_state_indices:
            s_indices_cute = from_dlpack(
                state_indices, assumed_align=4
            ).mark_layout_dynamic()

        s_checkpoints_cute = None
        cu_checkpoints_cute = None
        if enable_checkpoints:
            s_checkpoints_cute = from_dlpack(output_checkpoints, assumed_align=16)
            s_checkpoints_cute.mark_layout_dynamic().mark_compact_shape_dynamic(
                mode=3, stride_order=(0, 1, 2, 3), divisibility=DK
            )
            cu_checkpoints_cute = from_dlpack(
                cu_checkpoints, assumed_align=4
            ).mark_layout_dynamic()

        workspace_size = GatedDeltaNetChunkedKernel.get_workspace_size(
            num_sm, B, HQ, HV, True
        )
        workspace = torch.empty(workspace_size, dtype=torch.int8, device=q.device)
        workspace_cute = from_dlpack(workspace, assumed_align=16)

        stream = cuda.CUstream(torch.cuda.current_stream(device=q.device).cuda_stream)

        compiled = cute.compile(
            gdn,
            q_cute,
            k_cute,
            v_cute,
            gate_cute,
            beta_cute,
            o_cute,
            cu_seqlens_cute,
            s_in_cute,
            s_out_cute,
            s_indices_cute,
            s_checkpoints_cute,
            cu_checkpoints_cute,
            checkpoint_every_n_tokens,
            scale,
            workspace_cute,
            stream,
            options="--enable-tvm-ffi --opt-level 3",
        )

        cache["compiled"] = compiled
        cache["num_sm"] = num_sm


def atrex_aka_chunk_gated_delta_rule_sm103_m64(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    gate: torch.Tensor,
    beta: torch.Tensor,
    output: torch.Tensor,
    cu_seqlens: torch.Tensor,
    initial_state: torch.Tensor,
    output_state: torch.Tensor,
    scale: float,
) -> None:
    """Launch the verified SM103 AKA implementation with M64 forced."""
    _launch_chunk_gated_delta_rule(
        q,
        k,
        v,
        gate,
        beta,
        output,
        cu_seqlens,
        initial_state,
        output_state,
        scale,
        value_rows_override=64,
    )
