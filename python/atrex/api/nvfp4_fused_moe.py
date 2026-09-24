"""Public NVFP4 fused MoE API with runtime SM120 dispatch."""

from __future__ import annotations

import importlib
from typing import Any

from atrex.utils.device_target import detect_device_target


_IMPL_MODULES = {
    ("nvidia", "sm120"): "atrex.api.nvfp4_fused_moe_sm120",
}
_IMPL_CACHE = {}


def _first_tensor_device(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    for value in (*args, *kwargs.values()):
        device = getattr(value, "device", None)
        if device is not None:
            return device
    return None


def _load_impl(args: tuple[Any, ...], kwargs: dict[str, Any]):
    target = detect_device_target(_first_tensor_device(args, kwargs))
    module_name = _IMPL_MODULES.get((target.family, target.arch))
    if module_name is None:
        supported = ", ".join(
            f"{family}/{arch}" for family, arch in sorted(_IMPL_MODULES)
        )
        raise RuntimeError(
            "nvfp4_fused_moe supports only the validated NVIDIA SM120 "
            f"implementation; detected {target.family}/{target.arch}; "
            f"supported: {supported}"
        )

    module = _IMPL_CACHE.get(module_name)
    if module is None:
        module = importlib.import_module(module_name)
        _IMPL_CACHE[module_name] = module
    return module


def nvfp4_fused_moe(*args: Any, **kwargs: Any) -> None:
    impl = _load_impl(args, kwargs)
    return impl.nvfp4_fused_moe(*args, **kwargs)


def prepare_nvfp4_fused_moe_weights(*args: Any, **kwargs: Any):
    impl = _load_impl(args, kwargs)
    prepare = getattr(impl, "prepare_nvfp4_fused_moe_weights", None)
    if prepare is None:
        return None
    return prepare(*args, **kwargs)
