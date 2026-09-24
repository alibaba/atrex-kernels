"""ATREX public operator API with lazy implementation imports."""

from __future__ import annotations

from collections.abc import Callable as _Callable
from importlib import import_module as _import_module
from typing import Any as _Any


__all__: tuple[str, ...] = ()


def _lazy_import_and_call(
    func_name: str,
    module_name: str,
    *,
    public_name: str | None = None,
) -> _Callable[..., _Any]:
    """Return a public API proxy that imports its implementation on first call.

    ``public_name`` is only needed when the exported ATREX API name differs
    from the implementation function name.
    """

    export_name = public_name or func_name

    def wrapper(*args: _Any, **kwargs: _Any) -> _Any:
        module = _import_module(module_name)
        func = getattr(module, func_name)
        if not callable(func):
            raise TypeError(f"{module_name}.{func_name} is not callable")

        globals()[export_name] = func
        return func(*args, **kwargs)

    wrapper.__name__ = export_name
    wrapper.__qualname__ = export_name
    wrapper.__module__ = __name__
    return wrapper


chunk_gdn_fwd_cutedsl_build = _lazy_import_and_call(
    "chunk_gdn_fwd_cutedsl_build",
    "atrex.api.chunk_gdn_cutedsl",
)
chunk_gdn_fwd_cutedsl = _lazy_import_and_call(
    "chunk_gdn_fwd_cutedsl",
    "atrex.api.chunk_gdn_cutedsl",
)
can_use_chunk_gdn_fwd_cutedsl = _lazy_import_and_call(
    "can_use_chunk_gdn_fwd_cutedsl",
    "atrex.api.chunk_gdn_cutedsl",
)
chunk_gdn_fwd_cutedsl_prewarm_buckets = _lazy_import_and_call(
    "chunk_gdn_fwd_cutedsl_prewarm_buckets",
    "atrex.api.chunk_gdn_cutedsl",
)
nvfp4_fused_moe = _lazy_import_and_call(
    "nvfp4_fused_moe",
    "atrex.api.nvfp4_fused_moe",
)
prepare_nvfp4_fused_moe_weights = _lazy_import_and_call(
    "prepare_nvfp4_fused_moe_weights",
    "atrex.api.nvfp4_fused_moe",
)
can_use_flash_attn_varlen_func = _lazy_import_and_call(
    "can_use_flash_attn_varlen_func",
    "atrex.api.flash_attn",
)
flash_attn_varlen_func = _lazy_import_and_call(
    "flash_attn_varlen_func",
    "atrex.api.flash_attn",
)

__all__ = (
    "can_use_flash_attn_varlen_func",
    "can_use_chunk_gdn_fwd_cutedsl",
    "chunk_gdn_fwd_cutedsl",
    "chunk_gdn_fwd_cutedsl_build",
    "chunk_gdn_fwd_cutedsl_prewarm_buckets",
    "flash_attn_varlen_func",
    "nvfp4_fused_moe",
    "prepare_nvfp4_fused_moe_weights",
)
