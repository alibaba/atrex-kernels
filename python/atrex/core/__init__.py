"""Compilation and runtime infrastructure shared by ATREX operators."""

from .compile_cu import cuda_kernel


__all__ = ("cuda_kernel",)
