"""Compile cache for the SM120 Chunk-GDN CuTeDSL kernel."""

import os
import re
import shutil
import subprocess
import tempfile
import threading
import types
import typing

import cutlass.cute as cute
from cutlass.base_dsl.compiler import DumpDir


_in_mem_compile_cache = {}
_in_mem_compile_cache_lock = threading.Lock()


_PTXAS_UNSUPPORTED_VERSION_RE = re.compile(
    r"Unsupported \.version (?P<requested>\d+\.\d+); "
    r"current version is '(?P<supported>\d+\.\d+)'"
)
_PTX_VERSION_DIRECTIVE_RE = re.compile(
    r"(?m)^(?P<prefix>[ \t]*\.version[ \t]+)"
    r"(?P<version>\d+\.\d+)(?P<suffix>[ \t]*\r?)$"
)


_TMA_CLUSTER_LOAD = (
    "cp.async.bulk.tensor.3d.shared::cluster.global.tile."
    "mbarrier::complete_tx::bytes.L2::cache_hint"
)
_TMA_CTA_LOAD = (
    "cp.async.bulk.tensor.3d.shared::cta.global.tile."
    "mbarrier::complete_tx::bytes.L2::cache_hint"
)
_TMA_TILE_STORE = (
    "cp.async.bulk.tensor.3d.global.shared::cta.tile."
    "bulk_group.L2::cache_hint"
)
_TMA_CTA_STORE = "cp.async.bulk.tensor.3d.global.shared::cta.bulk_group"


def _as_options_tuple(options):
    if options is None:
        return ()
    if isinstance(options, tuple):
        return options
    return (options,)


class KeyedCompileMixin:
    def _get_compile_key(self):
        compile_key = getattr(self, "_KeyedCompileMixin_compile_key", None)
        if compile_key is None:
            collected_attrs = []
            for attr_name in sorted(dir(self)):
                attr_value = getattr(self, attr_name)
                if attr_name.startswith("__"):
                    continue
                if isinstance(
                    attr_value,
                    (
                        types.MethodType,
                        types.BuiltinMethodType,
                        types.MethodWrapperType,
                    ),
                ):
                    continue

                if isinstance(attr_value, typing.Hashable):
                    collected_attrs.append((attr_name, attr_value))

            # Kernel class identity affects emitted code even when two objects
            # happen to expose identical configuration attributes.
            compile_key = (str(type(self).__mro__),) + tuple(collected_attrs)
            setattr(self, "_KeyedCompileMixin_compile_key", compile_key)

        return compile_key


def _compile_options_key(options):
    if options is None:
        return None
    options = _as_options_tuple(options)
    return tuple((type(option), option.value) for option in options)


def _option_value(options, option_type):
    for option in _as_options_tuple(options):
        if isinstance(option, option_type):
            return option.value
    return None


def _needs_sm120a_tma_patch(options):
    return _option_value(options, cute.GPUArch) == "sm_120a"


def _patched_compile_options(options):
    options = _as_options_tuple(options)
    if not _needs_sm120a_tma_patch(options):
        return options

    has_keep_ptx = any(isinstance(option, cute.KeepPTX) for option in options)
    has_dump_dir = any(isinstance(option, DumpDir) for option in options)
    extras = []
    if not has_keep_ptx:
        extras.append(cute.KeepPTX(True))
    if has_dump_dir:
        dump_dir = _option_value(options, DumpDir)
        if dump_dir:
            os.makedirs(dump_dir, exist_ok=True)
    else:
        dump_dir = os.environ.get("ATREX_DSL_TMA_PATCH_DIR", "/tmp/atrex_dsl_tma_patch")
        os.makedirs(dump_dir, exist_ok=True)
        extras.append(DumpDir(dump_dir))
    return options + tuple(extras)


def _read_ptx_text(ptx_artifact):
    if isinstance(ptx_artifact, str) and os.path.exists(ptx_artifact):
        with open(ptx_artifact, "r", encoding="utf-8") as f:
            return f.read()
    return ptx_artifact


def _patch_sm120a_tma_ptx(ptx_text: str) -> str:
    patched = ptx_text.replace(_TMA_CLUSTER_LOAD, _TMA_CTA_LOAD)
    if _TMA_TILE_STORE in patched:
        lines = []
        for line in patched.splitlines(keepends=True):
            if _TMA_TILE_STORE not in line:
                lines.append(line)
                continue

            line_body = line.rstrip("\r\n")
            line_end = line[len(line_body):]
            line_body = line_body.replace(_TMA_TILE_STORE, _TMA_CTA_STORE)
            if line_body.endswith(";"):
                operands, separator, _cache_policy = line_body[:-1].rpartition(", %rd")
                if separator:
                    line_body = operands + ";"
            lines.append(line_body + line_end)
        patched = "".join(lines)

    if patched == ptx_text:
        return ptx_text
    return patched.replace("\x00", "")


def _find_ptxas() -> str:
    candidates = []
    for env_name in ("CUDA_PATH", "CUDA_HOME"):
        cuda_root = os.environ.get(env_name)
        if cuda_root:
            candidates.append(os.path.join(cuda_root, "bin", "ptxas"))

    path_ptxas = shutil.which("ptxas")
    if path_ptxas:
        candidates.append(path_ptxas)
    candidates.append("/usr/local/cuda/bin/ptxas")

    for candidate in candidates:
        if os.path.isfile(candidate) and os.access(candidate, os.X_OK):
            return candidate
    raise RuntimeError(
        "ptxas was not found; set CUDA_PATH/CUDA_HOME to the CUDA toolkit "
        "root or make ptxas available on PATH"
    )


def _downgrade_unsupported_ptx_version(
    ptx: str, ptxas_stderr: str
) -> typing.Optional[str]:
    """Downgrade a same-major PTX header when ptxas reports an older maximum.

    CuTe DSL may emit a newer PTX minor version than the CUDA toolkit bundled
    in the runtime image.  The SM120 TMA rewrite below only changes instructions
    that CUDA 13.0 already supports, so retrying with ptxas' reported maximum
    version is safe.  If the rewritten PTX uses a genuinely newer instruction,
    the retry still fails and its ptxas diagnostics are surfaced.
    """
    error_match = _PTXAS_UNSUPPORTED_VERSION_RE.search(ptxas_stderr)
    if error_match is None:
        return None

    requested = error_match.group("requested")
    supported = error_match.group("supported")
    requested_parts = tuple(int(part) for part in requested.split("."))
    supported_parts = tuple(int(part) for part in supported.split("."))
    if requested_parts[0] != supported_parts[0] or requested_parts <= supported_parts:
        return None

    directive_match = _PTX_VERSION_DIRECTIVE_RE.search(ptx)
    if (
        directive_match is None
        or directive_match.group("version") != requested
    ):
        return None

    version_start, version_end = directive_match.span("version")
    return ptx[:version_start] + supported + ptx[version_end:]


def _assemble_sm120a_cubin(ptx: str) -> bytes:
    with tempfile.TemporaryDirectory(prefix="atrex_dsl_tma_patch_") as tmp_dir:
        ptx_path = os.path.join(tmp_dir, "kernel.ptx")
        cubin_path = os.path.join(tmp_dir, "kernel.cubin")
        with open(ptx_path, "w", encoding="utf-8") as f:
            f.write(ptx)

        ptxas = _find_ptxas()
        cmd = [ptxas, "-arch=sm_120a", ptx_path, "-o", cubin_path]
        result = subprocess.run(cmd, check=False, text=True, capture_output=True)
        downgraded_ptx = None
        if result.returncode != 0:
            downgraded_ptx = _downgrade_unsupported_ptx_version(
                ptx, result.stderr
            )
        if downgraded_ptx is not None:
            ptx = downgraded_ptx
            with open(ptx_path, "w", encoding="utf-8") as f:
                f.write(ptx)
            result = subprocess.run(
                cmd, check=False, text=True, capture_output=True
            )
        if result.returncode != 0:
            raise RuntimeError(
                "failed to assemble patched sm120a TMA PTX\n"
                f"command: {' '.join(cmd)}\n"
                f"stdout:\n{result.stdout}\n"
                f"stderr:\n{result.stderr}"
            )

        with open(cubin_path, "rb") as f:
            return f.read()


def _install_patched_cubin_loader(compiled_fn, patched_cubin: bytes):
    import ctypes

    import cuda.bindings.runtime as cuda_runtime

    from cutlass.base_dsl.common import DSLRuntimeError
    from cutlass.base_dsl.runtime.cuda import checkCudaErrors

    def _load_cuda_library(self):
        if self.engine is None:
            raise DSLRuntimeError("CUDA JIT engine is not available")

        cuda_load_to_device = self.engine.raw_lookup("cuda_load_to_device")
        if cuda_load_to_device is None:
            raise DSLRuntimeError("cuda_load_to_device not found")
        cuda_load_to_device = ctypes.CFUNCTYPE(ctypes.c_int32, ctypes.c_void_p)(
            cuda_load_to_device
        )

        library_obj = checkCudaErrors(
            cuda_runtime.cudaLibraryLoadData(
                patched_cubin, None, None, 0, None, None, 0
            )
        )
        library = ctypes.c_void_p(int(library_obj))
        pointer_to_library = ctypes.pointer(library)
        pointer_to_pointer_to_library = ctypes.pointer(pointer_to_library)
        err = ctypes.c_int32(0)
        pointer_to_err = ctypes.pointer(err)
        device_id = ctypes.c_int32(0)
        pointer_to_device_id = ctypes.pointer(device_id)

        cuda_load_args = [
            pointer_to_pointer_to_library,
            pointer_to_device_id,
            pointer_to_err,
        ]
        packed_args = (ctypes.c_void_p * len(cuda_load_args))()
        for i, arg in enumerate(cuda_load_args):
            packed_args[i] = ctypes.cast(arg, ctypes.c_void_p)

        for dev in range(self.num_devices):
            device_id.value = dev
            cuda_load_to_device(packed_args)
            checkCudaErrors((cuda_runtime.cudaError_t(err.value),))

        return [library_obj]

    compiled_fn._flat_patched_cubin = patched_cubin
    compiled_fn._load_cuda_library = types.MethodType(_load_cuda_library, compiled_fn)
    if compiled_fn.artifacts is not None:
        compiled_fn.artifacts.CUBIN = patched_cubin


def _maybe_patch_sm120a_tma(compiled_fn, options):
    if not _needs_sm120a_tma_patch(options):
        return compiled_fn
    ptx = _read_ptx_text(getattr(compiled_fn, "__ptx__", None))
    if not ptx:
        return compiled_fn
    patched_ptx = _patch_sm120a_tma_ptx(ptx)
    if patched_ptx == ptx:
        return compiled_fn
    patched_cubin = _assemble_sm120a_cubin(patched_ptx)
    _install_patched_cubin_loader(compiled_fn, patched_cubin)
    return compiled_fn


def cached_compile(func, *args, compile_options=None, argument_key=(), **kwargs):
    # CuTeDSL specializes on argument types. Include caller-supplied argument
    # metadata that is not already represented by the kernel object itself.
    cache_key = (
        func._get_compile_key(),
        _compile_options_key(compile_options),
        tuple(argument_key),
    )
    with _in_mem_compile_cache_lock:
        compiled_fn = _in_mem_compile_cache.get(cache_key, None)
        if compiled_fn is None:
            compiler = cute.compile
            effective_compile_options = _patched_compile_options(compile_options)
            if effective_compile_options:
                compiler = cute.compile[effective_compile_options]
            compiled_fn = compiler(func, *args, **kwargs)
            compiled_fn = _maybe_patch_sm120a_tma(
                compiled_fn, effective_compile_options
            )
            _in_mem_compile_cache[cache_key] = compiled_fn

    return compiled_fn
