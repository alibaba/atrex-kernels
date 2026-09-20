"""Compatibility helpers for compiling Atrex kernels with CuTeDSL 4.5.2."""

from __future__ import annotations

import hashlib
import importlib
import sys
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from threading import RLock, local

from cutlass.cutlass_dsl import BaseDSL
import cutlass.cutlass_dsl.cutlass as _cutlass_dsl

_CUTLASS_VERSION_OVERRIDE_LOCK = RLock()
_CUTLASS_VERSION_OVERRIDE_STATE = local()
_MISSING_INSTANCE_ATTRIBUTE = object()


@lru_cache(maxsize=1)
def _cutlass_dsl_content_hash():
    """Hash CuTeDSL sources without importing compatibility namespaces."""
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


@contextmanager
def use_filesystem_cutlass_dsl_version_hash():
    """Use the filesystem version hash only during an Atrex compilation."""
    dsl = BaseDSL._get_dsl()
    with _CUTLASS_VERSION_OVERRIDE_LOCK:
        previous = dsl.__dict__.get("get_version", _MISSING_INSTANCE_ATTRIBUTE)
        previous_bound = dsl.get_version
        previous_thread_override = getattr(
            _CUTLASS_VERSION_OVERRIDE_STATE,
            "get_version",
            _MISSING_INSTANCE_ATTRIBUTE,
        )

        def thread_local_get_version():
            override = getattr(
                _CUTLASS_VERSION_OVERRIDE_STATE, "get_version", None
            )
            return override() if override is not None else previous_bound()

        dsl.get_version = thread_local_get_version
        _CUTLASS_VERSION_OVERRIDE_STATE.get_version = _cutlass_dsl_content_hash
        try:
            yield
        finally:
            if previous_thread_override is _MISSING_INSTANCE_ATTRIBUTE:
                del _CUTLASS_VERSION_OVERRIDE_STATE.get_version
            else:
                _CUTLASS_VERSION_OVERRIDE_STATE.get_version = (
                    previous_thread_override
                )
            if previous is _MISSING_INSTANCE_ATTRIBUTE:
                del dsl.__dict__["get_version"]
            else:
                dsl.get_version = previous


__all__ = [
    "_cutlass_dsl_content_hash",
    "use_filesystem_cutlass_dsl_version_hash",
]
