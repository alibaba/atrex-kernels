"""Scoped CuTeDSL cache-version compatibility for installed ATREX kernels.

CuTeDSL 4.5.2 does not include every installed Python source in the cache
version used by ``cute.compile``.  ATREX therefore supplies a filesystem hash
while compiling its FlashAttention kernels.  The delegate is
installed once; each compilation selects its override through a context-local
value, so concurrent compilations are not serialized by a process-wide lock.
"""

from __future__ import annotations

import hashlib
import importlib
import sys
from contextlib import contextmanager
from contextvars import ContextVar
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import Callable, Iterator

from cutlass.cutlass_dsl import BaseDSL
import cutlass.cutlass_dsl.cutlass as _cutlass_dsl


_VersionProvider = Callable[[], object]
_VERSION_OVERRIDE: ContextVar[_VersionProvider | None] = ContextVar(
    "atrex_cutlass_dsl_version_override", default=None
)
_DELEGATE_INSTALL_LOCK = Lock()
_DELEGATE_DSL = None


@lru_cache(maxsize=1)
def _cutlass_dsl_content_hash():
    """Hash the installed CuTeDSL Python sources and native runtime."""
    package_root = Path(_cutlass_dsl.__file__).resolve().parents[1]
    source_files = list(package_root.rglob("*.py"))
    runtime_module_name = "cutlass._mlir._mlir_libs._cutlass_ir"
    runtime_module = sys.modules.get(runtime_module_name)
    if runtime_module is None:
        runtime_module = importlib.import_module(runtime_module_name)
    runtime_file_name = getattr(runtime_module, "__file__", None)
    if runtime_file_name is None:
        raise RuntimeError(
            "Unable to locate the CuTe DSL _cutlass_ir runtime library"
        )
    runtime_file = Path(runtime_file_name).resolve()

    version_hash = hashlib.sha256()
    for path in sorted(source_files + [runtime_file]):
        try:
            relative_path_str = path.relative_to(package_root).as_posix()
        except ValueError:
            relative_path_str = path.as_posix()
        relative_path = relative_path_str.encode()
        content = path.read_bytes()
        version_hash.update(len(relative_path).to_bytes(8, "little"))
        version_hash.update(relative_path)
        version_hash.update(len(content).to_bytes(8, "little"))
        version_hash.update(content)
    return version_hash


def _install_version_delegate():
    """Install one context-local delegate without holding a compile lock."""
    global _DELEGATE_DSL

    dsl = BaseDSL._get_dsl()
    if _DELEGATE_DSL is dsl:
        return dsl
    with _DELEGATE_INSTALL_LOCK:
        if _DELEGATE_DSL is dsl:
            return dsl
        upstream_get_version = dsl.get_version

        def delegated_get_version():
            provider = _VERSION_OVERRIDE.get()
            return (
                provider()
                if provider is not None
                else upstream_get_version()
            )

        dsl.get_version = delegated_get_version
        _DELEGATE_DSL = dsl
    return dsl


@contextmanager
def _use_cutlass_dsl_version(provider: _VersionProvider) -> Iterator[None]:
    """Use ``provider`` only in the current thread/context."""
    _install_version_delegate()
    token = _VERSION_OVERRIDE.set(provider)
    try:
        yield
    finally:
        _VERSION_OVERRIDE.reset(token)


@contextmanager
def use_filesystem_cutlass_dsl_version_hash() -> Iterator[None]:
    """Use the installed-files hash during one ATREX compilation."""
    with _use_cutlass_dsl_version(_cutlass_dsl_content_hash):
        yield


__all__ = ["use_filesystem_cutlass_dsl_version_hash"]
