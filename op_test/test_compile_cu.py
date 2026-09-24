from types import SimpleNamespace

import atrex.core.compile_cu as compile_cu


def test_cuda_kernel_decorator_is_lazy_and_exposes_prewarm(monkeypatch):
    loads = []
    module = SimpleNamespace(run=lambda value: value + 1)

    def fake_load(spec, device=None):
        loads.append((spec, device))
        return module

    monkeypatch.setattr(compile_cu, "_load_cuda_extension", fake_load)

    @compile_cu.cuda_kernel(
        sources=("cuda/example/example.cu",),
        module_name="atrex_example",
        function_name="run",
    )
    def example(value):
        raise AssertionError("the Python signature stub must not execute")

    assert loads == []
    assert example(2) == 3
    assert len(loads) == 1
    assert example.build("cuda:0") is module
    assert len(loads) == 2


def test_build_identity_changes_when_an_included_header_changes(
    monkeypatch, tmp_path
):
    source = tmp_path / "example.cu"
    include_dir = tmp_path / "include"
    header = include_dir / "example.h"
    include_dir.mkdir()
    source.write_text('#include "example.h"\n')
    header.write_text("#define VALUE 1\n")
    monkeypatch.setattr(compile_cu, "_atrex_source_root", lambda: tmp_path)

    spec = compile_cu._CudaKernelSpec(
        module_name="atrex_example",
        function_name="run",
        sources=("example.cu",),
        include_dirs=("include",),
        external_include_dirs=(),
        extra_cflags=(),
        extra_cuda_cflags=(),
        extra_ldflags=(),
        cuda_arch_suffix="",
    )
    arguments = {
        "family": "nvidia",
        "capability": (12, 0),
        "torch_version": "test",
        "runtime_version": "test",
        "compiler_identity": {"nvcc": "test"},
    }
    before = compile_cu._build_identity(
        spec, (source,), (include_dir,), (), **arguments
    )
    header.write_text("#define VALUE 2\n")
    after = compile_cu._build_identity(
        spec, (source,), (include_dir,), (), **arguments
    )

    assert before != after


def test_build_identity_changes_when_an_external_header_changes(
    monkeypatch, tmp_path
):
    source_root = tmp_path / "src"
    include_dir = source_root / "include"
    external_include_dir = tmp_path / "cutlass" / "include"
    source = source_root / "example.cu"
    internal_header = include_dir / "example.h"
    external_header = external_include_dir / "cutlass_like.h"

    include_dir.mkdir(parents=True)
    external_include_dir.mkdir(parents=True)
    source.write_text('#include "example.h"\n#include "cutlass_like.h"\n')
    internal_header.write_text("#define INTERNAL_VALUE 1\n")
    external_header.write_text("#define EXTERNAL_VALUE 1\n")
    monkeypatch.setattr(compile_cu, "_atrex_source_root", lambda: source_root)

    spec = compile_cu._CudaKernelSpec(
        module_name="atrex_example",
        function_name="run",
        sources=("example.cu",),
        include_dirs=("include",),
        external_include_dirs=(str(external_include_dir),),
        extra_cflags=(),
        extra_cuda_cflags=(),
        extra_ldflags=(),
        cuda_arch_suffix="",
    )
    arguments = {
        "family": "nvidia",
        "capability": (12, 0),
        "torch_version": "test",
        "runtime_version": "test",
        "compiler_identity": {"nvcc": "test"},
    }
    before = compile_cu._build_identity(
        spec, (source,), (include_dir,), (external_include_dir,), **arguments
    )
    external_header.write_text("#define EXTERNAL_VALUE 2\n")
    after = compile_cu._build_identity(
        spec, (source,), (include_dir,), (external_include_dir,), **arguments
    )

    assert before != after
