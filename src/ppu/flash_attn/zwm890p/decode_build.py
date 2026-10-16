"""Lazy HGCC builds of native PPU Q1/Q4 decode."""

import ctypes
from functools import lru_cache
import hashlib
import json
import os
from pathlib import Path
import subprocess
import threading

import torch

from atrex.core.compile_cu import _jit_cache_root, _process_lock


ROOT = Path(__file__).resolve().parent
LOCK = threading.Lock()
LIBRARIES = {}
DEFINES = {
    "q1": {"Q1_ONLY": 1},
    "q4": {"Q1_ONLY": 0},
}
BACKEND_FLAGS = (
    "-ppu-max-vreg-count=256", "-ppu-sink-matrix-addr=true",
    "-ppu-max-alloca-byte-size=1024", "-ppu-sink-async-addr=true",
    "-ppu-sink-load-addr=true", "-ppu-sink-store-addr=true",
    "-ppu-alloca-half-ldst-simplify=true", "-ppu-volatile-yield=false",
    "-sort-copy-before-coalesce=true",
)


@lru_cache(maxsize=None)
def _compiler_version(compiler):
    return subprocess.check_output([str(compiler), "--version"], timeout=30)


def _build(variant, device_index):
    compiler = Path(os.environ.get("PPU_SDK_HOME", "/usr/local/PPU_SDK")) / "bin/hgcc"
    if not compiler.is_file():
        raise RuntimeError(f"PPU decode requires HGCC: {compiler}")
    flags = [
        "-O3", "-std=c++17", "-arch=ppu_15", "--use_fast_math",
        "--resource-usage", "-lineinfo", "-DNDEBUG",
    ]
    flags += [f"-DDECODE_{name}={value}" for name, value in
              DEFINES[variant].items()]
    for flag in BACKEND_FLAGS:
        flags += ["-mllvm", flag]
    flags += ["-Xcompiler", "-fPIC", "-shared", str(ROOT / "native_decode.cu")]
    digest = hashlib.sha256(Path(__file__).read_bytes())
    digest.update(str(compiler.resolve()).encode() + _compiler_version(compiler))
    digest.update("\0".join(flags).encode())
    digest.update(json.dumps(dict(torch=torch.__version__, runtime=torch.version.cuda,
        device=device_index, capability=torch.cuda.get_device_capability(device_index))).encode())
    sources = [ROOT / "native_decode.cu"]
    sources += [source for source in ROOT.glob("decode_*")
                if source.is_file() and source.suffix in (".py", ".cu", ".h")]
    for source in sorted(sources):
        digest.update(str(source.relative_to(ROOT)).encode() + source.read_bytes())
    destination = _jit_cache_root() / f"ppu_decode_{variant}_{digest.hexdigest()[:24]}"
    library_path = destination / "libdecode.so"
    with _process_lock(destination / "build.lock"):
        if not library_path.is_file():
            temporary = destination / "libdecode.building.so"
            command = [str(compiler), *flags, "-o", str(temporary)]
            (destination / "command.json").write_text(json.dumps(command, indent=2) + "\n")
            with (destination / "compile.log").open("w") as log:
                result = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT)
            if result.returncode:
                raise RuntimeError(f"PPU decode compilation failed: {destination / 'compile.log'}")
            os.replace(temporary, library_path)
    library = ctypes.CDLL(str(library_path))
    library.atrex_ppu_decode.argtypes = [
        ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int64),
        ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_void_p,
    ]
    library.atrex_ppu_decode.restype = ctypes.c_int
    resource = library.atrex_ppu_decode_resources
    resource.argtypes = [ctypes.POINTER(ctypes.c_int)]
    resource.restype = ctypes.c_int
    return library


def load_library(variant, device):
    if variant not in DEFINES:
        raise ValueError(f"Unknown PPU decode variant: {variant}")
    device_index = torch.device(device).index
    if device_index is None:
        device_index = torch.cuda.current_device()
    key = (variant, device_index)
    with LOCK:
        if key not in LIBRARIES:
            if torch.cuda.is_current_stream_capturing():
                raise RuntimeError("Prewarm PPU decode before CUDA graph capture")
            with torch.cuda.device(device_index):
                LIBRARIES[key] = _build(variant, device_index)
        return LIBRARIES[key]
