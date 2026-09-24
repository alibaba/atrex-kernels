"""NVIDIA SM120 NVFP4 fused MoE native implementation marker."""


def is_available() -> bool:
    return True


__all__ = ("is_available",)
