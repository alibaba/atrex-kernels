"""Minimal lazy JIT support for ATREX CUDA-compatible extensions."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
from collections.abc import Callable, Iterator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache, wraps
from pathlib import Path
from types import ModuleType
from typing import Any

from atrex.utils.device_target import detect_device_target


@dataclass(frozen=True)
class _CudaKernelSpec:
    module_name: str
    function_name: str
    sources: tuple[str, ...]
    include_dirs: tuple[str, ...]
    external_include_dirs: tuple[str, ...]
    extra_cflags: tuple[str, ...]
    extra_cuda_cflags: tuple[str, ...]
    extra_ldflags: tuple[str, ...]
    cuda_arch_suffix: str


_MODULE_CACHE: dict[str, ModuleType] = {}
_FAST_MODULE_CACHE: dict[tuple[Any, ...], ModuleType] = {}
_MODULE_LOCKS: dict[str, threading.Lock] = {}
_CACHE_LOCK = threading.Lock()


def _atrex_source_root() -> Path:
    packaged = Path(__file__).resolve().parents[1] / "src"
    repository = Path(__file__).resolve().parents[3] / "src"
    for candidate in (packaged, repository):
        if candidate.is_dir():
            return candidate.resolve()
    raise RuntimeError("ATREX source tree is missing from the installation")


def _resolve_under_source_root(relative_paths: Sequence[str]) -> tuple[Path, ...]:
    source_root = _atrex_source_root()
    resolved = []
    for relative_path in relative_paths:
        path = (source_root / relative_path).resolve()
        try:
            path.relative_to(source_root)
        except ValueError as error:
            raise ValueError(
                f"JIT path escapes the ATREX source tree: {relative_path}"
            ) from error
        if not path.exists():
            raise FileNotFoundError(path)
        resolved.append(path)
    return tuple(resolved)


def _resolve_existing_paths(paths: Sequence[str]) -> tuple[Path, ...]:
    resolved = []
    for raw_path in paths:
        path = Path(raw_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(path)
        resolved.append(path)
    return tuple(resolved)


def _jit_cache_root() -> Path:
    configured = os.environ.get("ATREX_JIT_CACHE_DIR")
    if configured:
        return Path(configured).expanduser().resolve()
    xdg_cache = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache:
        return Path(xdg_cache).expanduser().resolve() / "atrex" / "jit"
    return Path.home() / ".cache" / "atrex" / "jit"


def _thread_lock(key: str) -> threading.Lock:
    with _CACHE_LOCK:
        return _MODULE_LOCKS.setdefault(key, threading.Lock())


@contextmanager
def _process_lock(path: Path) -> Iterator[None]:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(path, os.O_RDWR | os.O_CREAT, 0o666)
    try:
        try:
            import fcntl
        except ImportError:
            yield
        else:
            fcntl.flock(descriptor, fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(descriptor, fcntl.LOCK_UN)
    finally:
        os.close(descriptor)


def _call_device(args: tuple[Any, ...], kwargs: dict[str, Any]) -> Any:
    for value in (*args, *kwargs.values()):
        device = getattr(value, "device", None)
        if device is not None:
            return device
    return None


def _build_identity(
    spec: _CudaKernelSpec,
    sources: tuple[Path, ...],
    include_dirs: tuple[Path, ...],
    external_include_dirs: tuple[Path, ...],
    *,
    family: str,
    capability: tuple[int, int],
    torch_version: str,
    runtime_version: str | None,
    compiler_identity: dict[str, str | None],
) -> str:
    digest = hashlib.sha256()
    metadata = {
        "module_name": spec.module_name,
        "family": family,
        "capability": capability,
        "torch": torch_version,
        "runtime": runtime_version,
        "compiler": compiler_identity,
        "python_abi": sys.implementation.cache_tag,
        "extra_cflags": spec.extra_cflags,
        "extra_cuda_cflags": spec.extra_cuda_cflags,
        "extra_ldflags": spec.extra_ldflags,
        "cuda_arch_suffix": spec.cuda_arch_suffix,
        "external_include_dirs": spec.external_include_dirs,
        "resolved_external_include_dirs": tuple(
            str(path) for path in external_include_dirs
        ),
    }
    digest.update(json.dumps(metadata, sort_keys=True).encode())
    source_root = _atrex_source_root()
    for source in sources:
        digest.update(str(source.relative_to(source_root)).encode())
        digest.update(source.read_bytes())
    for include_dir in include_dirs:
        digest.update(str(include_dir.relative_to(source_root)).encode())
        for header in sorted(
            path for path in include_dir.rglob("*") if path.is_file()
        ):
            digest.update(str(header.relative_to(source_root)).encode())
            digest.update(header.read_bytes())
    for include_dir in external_include_dirs:
        digest.update(str(include_dir).encode())
        for header in sorted(
            path for path in include_dir.rglob("*") if path.is_file()
        ):
            digest.update(str(header.relative_to(include_dir)).encode())
            digest.update(header.read_bytes())
    return digest.hexdigest()[:16]


def _tool_version(executable: Path | str | None) -> str | None:
    if executable is None:
        return None
    try:
        result = subprocess.run(
            [str(executable), "--version"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            timeout=10,
        )
    except (OSError, subprocess.SubprocessError):
        return str(executable)
    return f"{executable}\n{result.stdout.strip()}"


@lru_cache(maxsize=None)
def _compiler_identity(cuda_home: str | None) -> dict[str, str | None]:
    nvcc = Path(cuda_home) / "bin" / "nvcc" if cuda_home else shutil.which("nvcc")
    cxx = os.environ.get("CXX") or shutil.which("c++") or shutil.which("g++")
    return {
        "cuda_home": cuda_home,
        "nvcc": _tool_version(nvcc),
        "cxx": _tool_version(cxx),
    }


def _load_cuda_extension(spec: _CudaKernelSpec, device: Any = None) -> ModuleType:
    import torch
    from torch.utils.cpp_extension import CUDA_HOME, load

    if not torch.cuda.is_available():
        raise RuntimeError("ATREX CUDA JIT requires an available CUDA-compatible device")

    if device is None:
        device_index = torch.cuda.current_device()
    else:
        torch_device = torch.device(device)
        if torch_device.type != "cuda":
            raise ValueError(f"ATREX CUDA JIT requires a CUDA tensor, got {torch_device}")
        device_index = torch_device.index
        if device_index is None:
            device_index = torch.cuda.current_device()

    target = detect_device_target(device_index)
    if target.family not in {"nvidia", "alibaba_ppu"}:
        raise RuntimeError(
            "ATREX CUDA JIT supports NVIDIA and Alibaba PPU CUDA-compatible "
            f"runtimes, got {target.family}/{target.arch}"
        )

    sources = _resolve_under_source_root(spec.sources)
    include_dirs = _resolve_under_source_root(spec.include_dirs)
    external_include_dirs = _resolve_existing_paths(spec.external_include_dirs)
    capability = tuple(torch.cuda.get_device_capability(device_index))
    compiler_identity = _compiler_identity(CUDA_HOME)
    identity = _build_identity(
        spec,
        sources,
        include_dirs,
        external_include_dirs,
        family=target.family,
        capability=capability,
        torch_version=torch.__version__,
        runtime_version=torch.version.cuda,
        compiler_identity=compiler_identity,
    )
    fast_key = (
        spec,
        target.family,
        capability,
        torch.__version__,
        torch.version.cuda,
        identity,
    )
    with _CACHE_LOCK:
        cached = _FAST_MODULE_CACHE.get(fast_key)
    if cached is not None:
        return cached

    safe_module_name = re.sub(r"[^A-Za-z0-9_]", "_", spec.module_name)
    build_name = f"{safe_module_name}_{identity}"

    with _CACHE_LOCK:
        cached = _MODULE_CACHE.get(build_name)
    if cached is not None:
        with _CACHE_LOCK:
            _FAST_MODULE_CACHE[fast_key] = cached
        return cached

    cache_root = _jit_cache_root()
    build_directory = cache_root / build_name
    lock_path = cache_root / f".{build_name}.lock"
    architecture = f"{capability[0]}{capability[1]}"
    architecture_flag = (
        f"-gencode=arch=compute_{architecture}{spec.cuda_arch_suffix},"
        f"code=sm_{architecture}{spec.cuda_arch_suffix}"
    )

    with _thread_lock(build_name), _process_lock(lock_path):
        with _CACHE_LOCK:
            cached = _MODULE_CACHE.get(build_name)
        if cached is not None:
            with _CACHE_LOCK:
                _FAST_MODULE_CACHE[fast_key] = cached
            return cached

        build_directory.mkdir(parents=True, exist_ok=True)
        with torch.cuda.device(device_index):
            module = load(
                name=build_name,
                sources=[str(path) for path in sources],
                extra_cflags=["-O3", *spec.extra_cflags],
                extra_cuda_cflags=[
                    "-O3",
                    architecture_flag,
                    *spec.extra_cuda_cflags,
                ],
                extra_ldflags=list(spec.extra_ldflags),
                extra_include_paths=[
                    *(str(path) for path in include_dirs),
                    *(str(path) for path in external_include_dirs),
                ],
                build_directory=str(build_directory),
                with_cuda=True,
                is_python_module=True,
                verbose=os.environ.get("ATREX_JIT_VERBOSE") == "1",
            )
        with _CACHE_LOCK:
            _MODULE_CACHE[build_name] = module
            _FAST_MODULE_CACHE[fast_key] = module
        return module


def cuda_kernel(
    *,
    sources: Sequence[str],
    module_name: str | None = None,
    function_name: str | None = None,
    include_dirs: Sequence[str] = (),
    external_include_dirs: Sequence[str] = (),
    extra_cflags: Sequence[str] = (),
    extra_cuda_cflags: Sequence[str] = (),
    extra_ldflags: Sequence[str] = (),
    cuda_arch_suffix: str = "",
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """Decorate a CUDA binding stub with lazy, cached extension compilation.

    Source and include paths are relative to the repository's ``src`` tree.
    The decorated function compiles on its first call. Its ``build`` attribute
    explicitly prewarms the same module for a selected device.
    """

    if not sources:
        raise ValueError("cuda_kernel requires at least one source file")

    def decorator(func: Callable[..., Any]) -> Callable[..., Any]:
        spec = _CudaKernelSpec(
            module_name=module_name or func.__name__,
            function_name=function_name or func.__name__,
            sources=tuple(sources),
            include_dirs=tuple(include_dirs),
            external_include_dirs=tuple(external_include_dirs),
            extra_cflags=tuple(extra_cflags),
            extra_cuda_cflags=tuple(extra_cuda_cflags),
            extra_ldflags=tuple(extra_ldflags),
            cuda_arch_suffix=cuda_arch_suffix,
        )

        @wraps(func)
        def wrapped(*args: Any, **kwargs: Any) -> Any:
            module = _load_cuda_extension(spec, _call_device(args, kwargs))
            compiled = getattr(module, spec.function_name)
            return compiled(*args, **kwargs)

        def build(device: Any = None) -> ModuleType:
            return _load_cuda_extension(spec, device)

        wrapped.build = build  # type: ignore[attr-defined]
        wrapped.jit_spec = spec  # type: ignore[attr-defined]
        return wrapped

    return decorator
