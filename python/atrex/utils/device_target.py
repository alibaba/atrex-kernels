"""Conservative Torch device classification for ATREX dispatch."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True, slots=True)
class DeviceTarget:
    """Observed runtime family and architecture for one logical device."""

    family: str
    arch: str
    runtime_version: str | None
    device_index: int


_PPU_PRODUCT_ARCH = {
    # Verified on Agate ZW-M890P. Extend only with exact observed product names.
    "ZW-M890P": "zwm890p",
}


def detect_device_target(device: Any = None) -> DeviceTarget:
    """Classify a Torch device without confusing ROCm or PPU with NVIDIA.

    Torch is imported lazily so importing :mod:`atrex` remains dependency-free.
    Runtime-family detection is separate from operator-specific eligibility.
    """

    import torch

    if device is None:
        if not torch.cuda.is_available():
            return DeviceTarget("unavailable", "unavailable", None, -1)
        index = torch.cuda.current_device()
    elif isinstance(device, int):
        index = device
    else:
        torch_device = torch.device(device)
        if torch_device.type != "cuda":
            return DeviceTarget(torch_device.type, torch_device.type, None, -1)
        index = torch_device.index
        if index is None:
            index = torch.cuda.current_device()

    properties = torch.cuda.get_device_properties(index)
    name = properties.name.strip()

    # ROCm still exposes device operations through torch.cuda. Check HIP first.
    if torch.version.hip is not None:
        gfx = getattr(properties, "gcnArchName", "").split(":", 1)[0]
        return DeviceTarget("amd", gfx or "unknown", torch.version.hip, index)

    # PPU exposes a CUDA compatibility runtime and capability (8, 9), so use
    # the exact observed product name before considering NVIDIA capability.
    ppu_arch = _PPU_PRODUCT_ARCH.get(name)
    if ppu_arch is not None:
        return DeviceTarget("alibaba_ppu", ppu_arch, torch.version.cuda, index)

    if torch.version.cuda is not None and name.upper().startswith("NVIDIA "):
        major, minor = torch.cuda.get_device_capability(index)
        return DeviceTarget("nvidia", f"sm{major}{minor}", torch.version.cuda, index)

    return DeviceTarget("unknown", "unknown", None, index)
