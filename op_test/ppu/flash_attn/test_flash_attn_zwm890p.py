"""PPU prefill and native direct-paged FP8 decode contracts."""

import pytest
import torch
import json
from functools import partial

from op_test.utils.flash_attn_contract import build_case, reference, profile_case
from op_test.utils.flash_attn_contract import check_eager_scratch_reuse
from op_test.utils.flash_attn_contract import error_metrics


@pytest.fixture(scope="module", autouse=True)
def require_target():
    from atrex.utils.device_target import detect_device_target

    target = detect_device_target()
    assert (target.family, target.arch) == ("alibaba_ppu", "zwm890p")


def _case(page_size, q_lens, kv_lens, *, contiguous=False):
    case = build_case(page_size, q_lens, kv_lens)
    for name in ("q_descale", "k_descale", "v_descale"):
        case[name] = None
    if contiguous:
        case["k"] = case["k"].contiguous()
        case["v"] = case["v"].contiguous()
    return case


def test_public_eligibility_and_fail_fast():
    from atrex import can_use_flash_attn_varlen_func, flash_attn_varlen_func

    case = _case(64, [65, 3], [257, 513])
    for version in (2, 3):
        assert can_use_flash_attn_varlen_func(**dict(case, fa_version=version))
    for update, match in (
        ({"fa_version": 4}, "fa_version"),
        ({"causal": False}, "causal"),
    ):
        kwargs = dict(case, **update)
        assert not can_use_flash_attn_varlen_func(**kwargs)
        with pytest.raises((NotImplementedError, ValueError), match=match):
            flash_attn_varlen_func(**kwargs)
    for bad_out in (
        torch.empty_like(case["q"]),
        torch.empty(
            (case["q"].shape[0] + 1, *case["q"].shape[1:]),
            dtype=torch.bfloat16,
            device="cuda",
        ),
        torch.empty(case["q"].shape, dtype=torch.bfloat16),
    ):
        kwargs = dict(case, out=bad_out)
        assert not can_use_flash_attn_varlen_func(**kwargs)
        with pytest.raises(ValueError, match="out"):
            flash_attn_varlen_func(**kwargs)


def _check(case, splits=0, graph=False):
    from atrex import flash_attn_varlen_func

    out = torch.empty_like(case["q"], dtype=torch.bfloat16)
    kwargs = dict(case, out=out, num_splits=splits)
    # Check that internal page remapping and packing never mutate framework inputs.
    saved = {name: case[name].clone() for name in
             ("k", "v", "block_table", "cu_seqlens_q", "seqused_k")}
    actual = flash_attn_varlen_func(**kwargs)
    assert actual.data_ptr() == out.data_ptr()
    expected, _ = reference(case)
    print("PPU_FP8_ERROR", json.dumps(dict(
        phase="eager", reference="fp32_same_fp8_inputs",
        q_shape=list(case["q"].shape), kv_shape=list(case["k"].shape),
        splits=splits, metrics=error_metrics(actual, expected),
    ), allow_nan=False))
    torch.testing.assert_close(actual.float(), expected, atol=.03, rtol=.05)
    for name, tensor in saved.items():
        assert torch.equal(case[name].view(torch.uint8), tensor.view(torch.uint8)), name
    if graph:
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(3):
                flash_attn_varlen_func(**kwargs)
        torch.cuda.current_stream().wait_stream(stream)
        captured = torch.cuda.CUDAGraph()
        with torch.cuda.graph(captured):
            graph_out = flash_attn_varlen_func(**kwargs)
        qlens = case["cu_seqlens_q"][1:] - case["cu_seqlens_q"][:-1]
        case["seqused_k"].copy_(torch.maximum(case["seqused_k"] - 1, qlens))
        captured.replay()
        torch.cuda.synchronize()
        expected, _ = reference(case)
        print("PPU_FP8_ERROR", json.dumps(dict(
            phase="graph_replay", reference="fp32_same_fp8_inputs",
            q_shape=list(case["q"].shape), kv_shape=list(case["k"].shape),
            splits=splits, metrics=error_metrics(graph_out, expected),
        ), allow_nan=False))
        torch.testing.assert_close(graph_out.float(), expected, atol=.03, rtol=.05)


@pytest.mark.parametrize("page_size", [64, 128, 256])
@pytest.mark.parametrize("contiguous", [False, True])
@pytest.mark.parametrize("splits", [0, 1, 2, 4])
@pytest.mark.parametrize("q_lens,kv_lens", [
    ([1, 1], [127, 513]),
    ([4, 2], [128, 511]),
    ([65, 3], [257, 513]),
])
def test_original_prep_kv_contract(page_size, contiguous, splits, q_lens, kv_lens):
    case = _case(page_size, q_lens, kv_lens, contiguous=contiguous)
    case["fa_version"] = 2
    _check(case, splits=splits)


@pytest.mark.parametrize("length", [63, 64, 65, 127, 128, 129, 255, 256, 257])
@pytest.mark.parametrize("page_size", [64, 256])
@pytest.mark.parametrize("heads", [(8, 1), (32, 2)])
@pytest.mark.parametrize("splits", [1, 4])
def test_prep_kv_masks_nan_v_padding(length, page_size, heads, splits):
    case = build_case(page_size, [17, 18], [length, length + 2],
                      hq=heads[0], hkv=heads[1])
    for request, kv_len in enumerate((length, length + 2)):
        tail = kv_len % page_size
        if tail:
            page = int(case["block_table"][request, kv_len // page_size])
            case["v"][page, tail:].view(torch.uint8).fill_(0x7f)
    expected, _ = reference(case)
    assert torch.isfinite(expected).all()
    _check(case, splits=splits)


@pytest.mark.parametrize("page_size", [64, 128])
@pytest.mark.parametrize("splits", [1, 4])
def test_graph_replay(page_size, splits):
    _check(_case(page_size, [65, 3], [257, 513]), splits=splits, graph=True)


def test_real_kernel_events():
    # No table remapping is necessary for this layout, making every custom
    # kernel directly attributable. Interleaved remapping is covered above.
    names = profile_case(_case(64, [65, 3], [257, 513], contiguous=True))
    assert any("atrex_ppu_prep_raw" in name for name in names)
    assert any("atrex_ppu_attn" in name for name in names)
    assert not any("atrex_ppu_attn_general" in name for name in names)


def test_prefill_fast_path_source_matches_main():
    import hashlib

    from atrex.core.compile_cu import _atrex_source_root
    from atrex.src.ppu.flash_attn.zwm890p import runtime

    root = _atrex_source_root() / "ppu/flash_attn/zwm890p"
    source = (root / "kernel.cu").read_text()
    expected = (
        ("template <int SPLIT, bool GROUPED = false, int QK_BUNDLE = 4>",
         "// ==================================================================================== KERNEL 3",
         "9a3f45b8c00c7c87bbed21c8a741e4b8dcfe2d3d24f454b6cdbab2d3cedcc8e5"),
        ("template <bool GROUPED>\nvoid launch_attn", "\nvoid combine(",
         "0b7459b26861ad2caae904ce40bc9d0884c0b8929e298b839b72f347aa83bf91"),
    )
    for first, last, digest in expected:
        start = source.index(first)
        segment = source[start:source.index(last, start)].rstrip() + "\n"
        assert hashlib.sha256(segment.encode()).hexdigest() == digest
    general = (root / "prefill_general.cu").read_text()
    assert "void atrex_ppu_attn_general(" in general
    assert "void atrex_ppu_attn(" not in general
    assert "bool GENERAL" not in source
    assert runtime._launch.jit_spec.sources == (
        "ppu/flash_attn/zwm890p/kernel.cu",
        "ppu/flash_attn/zwm890p/prefill_general.cu",
    )
    assert runtime._launch.jit_spec.include_dirs == ("ppu/flash_attn/zwm890p/include",)


def test_eager_scratch_not_retained_per_layer_or_shape():
    from atrex import flash_attn_varlen_func

    # Four live layer KV addresses, alternating two shapes and split modes.
    # No historical outputs are retained beyond these preallocated buffers.
    calls = []
    for _ in range(4):
        layer = []
        for q_lens, kv_lens, splits in (
            ([65, 3], [1025, 513], 1),
            ([4, 2], [513, 257], 4),
        ):
            case = _case(64, q_lens, kv_lens)
            out = torch.empty_like(case["q"], dtype=torch.bfloat16)
            layer.append(partial(flash_attn_varlen_func, **case,
                                 out=out, num_splits=splits))
        calls.append(layer)
    check_eager_scratch_reuse(calls)


@pytest.mark.parametrize("name", ["q_descale", "k_descale", "v_descale"])
def test_nonunit_descale_not_silently_ignored(name):
    from atrex import flash_attn_varlen_func

    case = _case(64, [65, 3], [257, 513])
    case[name] = torch.full((2, 1), .75, dtype=torch.float32, device="cuda")
    actual = flash_attn_varlen_func(**case)
    expected, _ = reference(case)
    torch.testing.assert_close(actual.float(), expected, atol=.03, rtol=.05)


@pytest.mark.parametrize("splits", [1, 4])
def test_unit_descales_and_graph(splits):
    case = _case(64, [65, 3], [257, 513])
    for name in ("q_descale", "k_descale", "v_descale"):
        case[name] = torch.ones((1, 1), dtype=torch.float32, device="cuda").expand(2, 1)
    _check(case, splits=splits, graph=True)


def test_prefill_scale_mutation_read_on_each_call():
    from atrex import flash_attn_varlen_func

    case = _case(64, [65, 3], [257, 513])
    case["q_descale"] = torch.ones((1, 1), dtype=torch.float32, device="cuda").expand(2, 1)
    flash_attn_varlen_func(**case)
    case["q_descale"]._base.fill_(.5)
    actual = flash_attn_varlen_func(**case)
    expected, _ = reference(case)
    torch.testing.assert_close(actual.float(), expected, atol=.03, rtol=.05)


def test_unversioned_inference_scale_is_not_cached():
    from atrex import flash_attn_varlen_func

    with torch.inference_mode():
        case = _case(64, [65, 3], [257, 513])
        case["q_descale"] = torch.ones((2, 1), dtype=torch.float32, device="cuda")
        flash_attn_varlen_func(**case)
        case["q_descale"].fill_(.5)
        actual = flash_attn_varlen_func(**case)
        expected, _ = reference(case)
        torch.testing.assert_close(actual.float(), expected, atol=.03, rtol=.05)


def test_lse_not_fabricated():
    from atrex import flash_attn_varlen_func

    with pytest.raises(NotImplementedError, match="LSE"):
        flash_attn_varlen_func(**_case(64, [4, 2], [257, 513]), return_softmax_lse=True)


@pytest.mark.parametrize("page_size", [16, 32, 48])
def test_unsupported_page_size(page_size):
    from atrex import flash_attn_varlen_func

    with pytest.raises(NotImplementedError, match="multiple of 64"):
        flash_attn_varlen_func(**_case(page_size, [4, 2], [257, 513]))


def test_unsupported_row_stride():
    from atrex import flash_attn_varlen_func

    case = build_case(64, [4, 2], [257, 513], layout="strided")
    case["fa_version"] = 2
    with pytest.raises(NotImplementedError, match="contiguous rows"):
        flash_attn_varlen_func(**case)


def _native_decode_reference(case, splits):
    import flash_attn_3._C

    out = torch.empty_like(case["q"], dtype=torch.bfloat16)
    native_out, *_ = torch.ops.flash_attn_3.fwd(
        case["q"], case["k"], case["v"], None, None, None, out,
        case["cu_seqlens_q"], None, None, None, case["seqused_k"],
        case["max_seqlen_q"], case["max_seqlen_k"], case["block_table"],
        None, None, None, None, None,
        case["q_descale"], case["k_descale"], case["v_descale"],
        case["softmax_scale"], True, -1, -1, 0, 0.0, True, None, splits, None, 0,
    )
    return native_out


@pytest.mark.parametrize("query,hq,hkv,path", [
    (1, 32, 2, "decode"), (4, 32, 2, "decode"),
    (1, 16, 1, "decode"), (4, 64, 4, "decode"),
    (1, 8, 1, "decode"), (2, 8, 1, "decode"), (5, 8, 1, "decode"),
    (2, 32, 2, "decode"), (3, 32, 2, "decode"), (5, 32, 2, "decode"),
    (4, 8, 1, "decode"), (65, 8, 1, "prefill"),
    (6, 32, 2, "prefill"), (256, 32, 2, "prefill"),
])
def test_prefill_decode_dispatch(query, hq, hkv, path, monkeypatch):
    from atrex import flash_attn_varlen_func
    from atrex.src.ppu.flash_attn.zwm890p import decode_runtime, launch, runtime

    case = build_case(64, [query, query], [1025, 2049], hq=hq, hkv=hkv)
    if path == "prefill":
        for name in ("q_descale", "k_descale", "v_descale"):
            case[name] = None
    calls = []
    attempted = []
    out = torch.empty_like(case["q"], dtype=torch.bfloat16)

    def decode(**kwargs):
        attempted.append("decode")
        decode_runtime.validate(launch._normalized_call(kwargs))
        calls.append("decode")
        return kwargs["out"], None

    def prefill(**kwargs):
        calls.append("prefill")
        return kwargs["out"], None

    monkeypatch.setattr(decode_runtime, "forward", decode)
    monkeypatch.setattr(runtime, "forward", prefill)
    assert flash_attn_varlen_func(**case, out=out).data_ptr() == out.data_ptr()
    assert calls == [path]
    assert attempted == (["decode"] if query <= 5 else [])


@pytest.mark.parametrize("error", [ValueError("invalid metadata"), RuntimeError("launch failed"),
                                  NotImplementedError("selected kernel failure")])
def test_decode_runtime_errors_do_not_fall_back(error, monkeypatch):
    from atrex import flash_attn_varlen_func
    from atrex.src.ppu.flash_attn.zwm890p import decode_runtime, launch, runtime

    case = _case(64, [4, 4], [1025, 2049])
    calls = []

    def decode(**kwargs):
        raise error

    def prefill(**kwargs):
        calls.append("prefill")
        raise AssertionError("Runtime failures must not use fallback")

    monkeypatch.setattr(decode_runtime, "forward", decode)
    monkeypatch.setattr(runtime, "forward", prefill)
    with pytest.raises(type(error), match=str(error)):
        flash_attn_varlen_func(**case)
    assert not calls


@pytest.mark.parametrize("query", range(1, 6))
@pytest.mark.parametrize("version", [None, 2, 3])
def test_fa_version_preserves_dispatch(query, version, monkeypatch):
    from atrex import can_use_flash_attn_varlen_func, flash_attn_varlen_func
    from atrex.src.ppu.flash_attn.zwm890p import decode_runtime, launch, runtime

    case = _case(64, [query, query], [1025, 2049])
    case["fa_version"] = version
    calls = []

    def decode(**kwargs):
        assert kwargs["fa_version"] == version
        decode_runtime.validate(launch._normalized_call(kwargs))
        calls.append("decode")
        return kwargs["out"], None

    def prefill(**kwargs):
        assert "fa_version" not in kwargs
        calls.append("prefill")
        return kwargs["out"], None

    monkeypatch.setattr(decode_runtime, "forward", decode)
    monkeypatch.setattr(runtime, "forward", prefill)
    assert can_use_flash_attn_varlen_func(**case)
    out = torch.empty_like(case["q"], dtype=torch.bfloat16)
    assert flash_attn_varlen_func(**case, out=out).data_ptr() == out.data_ptr()
    assert calls == ["prefill" if version == 2 else "decode"]


@pytest.mark.parametrize("query", range(1, 6))
def test_fa2_uses_original_prefill(query):
    case = _case(64, [query, query], [1025, 2049], contiguous=True)
    case["fa_version"] = 2
    _check(case, graph=True)
    names = profile_case(case)
    assert any("atrex_ppu_prep_raw" in name for name in names)
    assert not any("atrex_ppu_decode_kernel" in name or "atrex_ppu_native_decode_kernel" in name for name in names)


@pytest.mark.parametrize("query", range(1, 6))
@pytest.mark.parametrize("heads", [(8, 1), (16, 1), (24, 3), (32, 2), (48, 3), (64, 4)])
@pytest.mark.parametrize("page_size", [64, 128, 256])
def test_decode_runtime_heads_and_pages(query, heads, page_size):
    from atrex import can_use_flash_attn_varlen_func, flash_attn_varlen_func

    query_heads, kv_heads = heads
    q_lens = [query, 1, max(1, query - 1)]
    case = build_case(page_size, q_lens, [127, 1025, 2049], hq=query_heads, hkv=kv_heads)
    assert can_use_flash_attn_varlen_func(**case)
    for splits in (0, 1):
        expected = _native_decode_reference(case, splits)
        actual = flash_attn_varlen_func(**case, num_splits=splits)
        torch.testing.assert_close(actual.float(), expected.float(), atol=.01, rtol=.05)


@pytest.mark.parametrize("query", range(2, 6))
@pytest.mark.parametrize("heads", [(8, 1), (32, 2)])
@pytest.mark.parametrize("batch", [1, 3])
def test_short_q_auto_never_uses_prefill(query, heads, batch, monkeypatch):
    from atrex import can_use_flash_attn_varlen_func, flash_attn_varlen_func
    from atrex.src.ppu.flash_attn.zwm890p import runtime

    def unexpected(**kwargs):
        raise AssertionError("Supported short-Q attention must use decode")

    monkeypatch.setattr(runtime, "forward", unexpected)
    for length in (5, 31, 32, 33, 511, 1023, 1024):
        case = build_case(64, [query] * batch, [length] * batch, hq=heads[0], hkv=heads[1])
        assert can_use_flash_attn_varlen_func(**case)
        expected = _native_decode_reference(case, 0)
        actual = flash_attn_varlen_func(**case)
        torch.testing.assert_close(actual.float(), expected.float(), atol=.01, rtol=.05)


@pytest.mark.parametrize("query", range(1, 6))
@pytest.mark.parametrize("heads", [(8, 1), (32, 2)])
@pytest.mark.parametrize("amplitude", [1., 2., 4.])
def test_decode_softmax_dynamic_range(query, heads, amplitude):
    from atrex import flash_attn_varlen_func

    query_heads, kv_heads = heads
    case = build_case(64, [query] * 3, [127, 1025, 2049], hq=query_heads, hkv=kv_heads)
    for name in ("q", "k"):
        case[name] = (case[name].float() * amplitude).to(torch.float8_e4m3fn)
    for name in ("q_descale", "k_descale", "v_descale"):
        case[name].fill_(1)
    for splits in (0, 1):
        expected = _native_decode_reference(case, splits)
        actual = flash_attn_varlen_func(**case, num_splits=splits)
        torch.testing.assert_close(actual.float(), expected.float(), atol=.01, rtol=.05)


@pytest.mark.parametrize("query,heads,kv_heads", [(1, 32, 2), (4, 32, 4)])
def test_decode_large_work_queue(query, heads, kv_heads):
    from atrex import flash_attn_varlen_func

    case = build_case(64, [query] * 128, [31] * 128, hq=heads, hkv=kv_heads)
    table = torch.full((128, 512), -1, dtype=torch.int32, device="cuda")
    table[:, 0] = case["block_table"][:, 0]
    case["block_table"] = table
    case["max_seqlen_k"] = 32768
    for name in ("q_descale", "k_descale", "v_descale"):
        case[name].fill_(1)
    out = torch.full_like(case["q"], float("nan"), dtype=torch.bfloat16)
    expected = _native_decode_reference(case, 0)
    actual = flash_attn_varlen_func(**case, out=out)
    torch.testing.assert_close(actual.float(), expected.float(), atol=.01, rtol=.05)


def test_shared_decode_sources_and_independent_specializations():
    from atrex.src.ppu.flash_attn.zwm890p import decode_build

    for filename in ("decode_runtime.py", "native_decode.cu", "decode_build.py"):
        assert (decode_build.ROOT / filename).is_file()
    assert not (decode_build.ROOT / "decode/vendor").exists()
    source = (decode_build.ROOT / "native_decode.cu").read_text()
    assert "#include <cute/" not in source
    assert "#include <cutlass/" not in source
    for directory in ("q1", "q4", "common"):
        assert not (decode_build.ROOT / "decode" / directory).exists()
    assert set(decode_build.DEFINES) == {"q1", "q4"}
    assert decode_build.DEFINES["q1"]["Q1_ONLY"] == 1
    assert decode_build.DEFINES["q4"]["Q1_ONLY"] == 0
    assert decode_build.DEFINES["q1"] != decode_build.DEFINES["q4"]


@pytest.mark.parametrize("query", range(1, 6))
def test_decode_selects_compiled_query_variant(query, monkeypatch):
    from atrex import flash_attn_varlen_func
    from atrex.src.ppu.flash_attn.zwm890p import decode_build

    original_load = decode_build.load_library
    loaded = []

    def load(variant, device):
        loaded.append(variant)
        library = original_load(variant, device)
        assert len(library.atrex_ppu_decode.argtypes) == 5
        return library

    monkeypatch.setattr(decode_build, "load_library", load)
    case = build_case(64, [query, query], [127, 257], hq=32, hkv=2)
    actual = flash_attn_varlen_func(**case, num_splits=1)
    expected = _native_decode_reference(case, 1)
    torch.testing.assert_close(actual.float(), expected.float(), atol=.01, rtol=.05)
    assert loaded == ["q1" if query == 1 else "q4"]


def test_eligibility_does_not_compile_or_launch(monkeypatch):
    from atrex import can_use_flash_attn_varlen_func
    from atrex.src.ppu.flash_attn.zwm890p import launch
    from atrex.src.ppu.flash_attn.zwm890p import decode_build

    def unexpected(*args, **kwargs):
        raise AssertionError("Eligibility must not compile or execute kernels")

    monkeypatch.setattr(decode_build, "load_library", unexpected)
    assert can_use_flash_attn_varlen_func(**build_case(128, [4, 4], [1025, 2049], hq=64, hkv=4))
    assert can_use_flash_attn_varlen_func(**_case(64, [4, 4], [1025, 2049]))
    assert can_use_flash_attn_varlen_func(**build_case(64, [256, 1], [257, 127], hq=32, hkv=2))


@pytest.mark.parametrize("heads", [(4, 1), (8, 1), (16, 1), (32, 2), (48, 3), (64, 4)])
@pytest.mark.parametrize("page_size", [64, 128, 256])
def test_prefill_parameterized_heads_and_descales(heads, page_size):
    from atrex import can_use_flash_attn_varlen_func, flash_attn_varlen_func

    case = build_case(page_size, [65, 1, 17], [129, 33, 95], hq=heads[0], hkv=heads[1])
    case["softmax_scale"] *= .75
    assert can_use_flash_attn_varlen_func(**case)
    expected, _ = reference(case)
    for splits in (0, 1, 2, 4):
        actual = flash_attn_varlen_func(**case, num_splits=splits)
        torch.testing.assert_close(actual.float(), expected, atol=.03, rtol=.05)


@pytest.mark.parametrize("query", [6, 7, 15, 16, 17, 31, 32, 33, 255, 256, 257])
@pytest.mark.parametrize("heads", [(8, 1), (32, 2)])
def test_prefill_decode_boundary_queries(query, heads):
    from atrex import can_use_flash_attn_varlen_func, flash_attn_varlen_func

    case = build_case(64, [query, 1], [query + 63, 33], hq=heads[0], hkv=heads[1])
    assert can_use_flash_attn_varlen_func(**case)
    expected, _ = reference(case)
    for splits in (0, 4):
        actual = flash_attn_varlen_func(**case, num_splits=splits)
        torch.testing.assert_close(actual.float(), expected, atol=.03, rtol=.05)


@pytest.mark.parametrize("q_lens,kv_lens", [
    ([4] * 128, [4] * 128),
    ([1] * 256 + [256], [1] * 256 + [256]),
])
def test_deployment_warmup_shapes(q_lens, kv_lens):
    from atrex import can_use_flash_attn_varlen_func, flash_attn_varlen_func

    case = build_case(64, q_lens, kv_lens, hq=32, hkv=2)
    for name in ("q_descale", "k_descale", "v_descale"):
        case[name].fill_(1.)
    assert can_use_flash_attn_varlen_func(**case)
    out = torch.empty_like(case["q"], dtype=torch.bfloat16)
    expected, _ = reference(case)
    flash_attn_varlen_func(**case, out=out)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flash_attn_varlen_func(**case, out=out)
    out.fill_(float("nan"))
    graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(out.float(), expected, atol=.03, rtol=.05)


@pytest.mark.parametrize("heads", [(8, 1), (32, 2)])
@pytest.mark.parametrize("splits", [1, 4])
def test_general_prefill_graph_mutable_metadata(heads, splits):
    from atrex import flash_attn_varlen_func

    case = build_case(128, [256, 1, 4], [513, 257, 129], hq=heads[0], hkv=heads[1])
    out = torch.empty_like(case["q"], dtype=torch.bfloat16)
    flash_attn_varlen_func(**case, out=out, num_splits=splits)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flash_attn_varlen_func(**case, out=out, num_splits=splits)
    for _ in range(3):
        case["q_descale"].mul_(.75)
        case["k_descale"].mul_(.875)
        case["v_descale"].mul_(.5)
        case["seqused_k"].sub_(1)
        case["block_table"][:, :2].copy_(case["block_table"][:, :2].flip(1))
        expected, _ = reference(case)
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(out.float(), expected, atol=.03, rtol=.05)


def test_general_prefill_real_kernel_events():
    case = build_case(64, [256, 1, 4], [257, 33, 65], hq=32, hkv=2)
    case["k"] = case["k"].contiguous()
    case["v"] = case["v"].contiguous()
    names = profile_case(case)
    assert any("atrex_ppu_prep_raw" in name for name in names)
    assert any("atrex_ppu_attn_general" in name for name in names)
    assert not any("atrex_ppu_native_decode_kernel" in name for name in names)


@pytest.mark.parametrize("scale", [0., -.5, .75])
def test_prefill_scaled_logits_before_masking(scale):
    from atrex import flash_attn_varlen_func

    case = build_case(64, [65, 1], [129, 33], hq=32, hkv=2)
    case["q_descale"].fill_(scale)
    expected, _ = reference(case)
    for splits in (1, 4):
        actual = flash_attn_varlen_func(**case, num_splits=splits)
        torch.testing.assert_close(actual.float(), expected, atol=.03, rtol=.05)


def test_large_decode_batch_uses_parameterized_prefill():
    from atrex import can_use_flash_attn_varlen_func, flash_attn_varlen_func

    case = build_case(64, [1] * 993, [4] * 993, hq=32, hkv=2)
    for name in ("q_descale", "k_descale", "v_descale"):
        case[name].fill_(1.)
    assert can_use_flash_attn_varlen_func(**case)
    actual = flash_attn_varlen_func(**case)
    expected, _ = reference(case)
    torch.testing.assert_close(actual.float(), expected, atol=.03, rtol=.05)


@pytest.mark.parametrize("query", range(1, 6))
@pytest.mark.parametrize("splits", [0, 1, 4, 32])
@pytest.mark.parametrize("layout", ["interleaved", "contiguous", "strided"])
def test_direct_decode_contract_and_mutation(query, splits, layout):
    from atrex import can_use_flash_attn_varlen_func, flash_attn_varlen_func

    q_lens = [query, 1, max(1, query - 1)]
    case = build_case(64, q_lens, [127, 1025, 2049], hq=32, hkv=2,
                      layout="strided" if layout == "strided" else "interleaved")
    if layout == "contiguous":
        case["k"], case["v"] = case["k"].contiguous(), case["v"].contiguous()
    assert can_use_flash_attn_varlen_func(**case, num_splits=splits)
    out = torch.empty_like(case["q"], dtype=torch.bfloat16)
    saved = {name: case[name].clone() for name in
             ("q", "k", "v", "block_table", "cu_seqlens_q", "seqused_k")}
    for phase in range(4):
        if phase == 1:
            case["q"].copy_((case["q"].float() * .5).to(case["q"].dtype))
            case["v_descale"].mul_(.75)
        elif phase == 2:
            case["seqused_k"].sub_(1)
        elif phase == 3:
            case["block_table"][0, :2] = case["block_table"][0, :2].flip(0)
        expected = _native_decode_reference(case, splits)
        actual = flash_attn_varlen_func(**case, out=out, num_splits=splits)
        assert actual.data_ptr() == out.data_ptr()
        torch.testing.assert_close(actual.float(), expected.float(), atol=.01, rtol=.05)
        if phase == 0:
            for name, tensor in saved.items():
                assert torch.equal(case[name].view(torch.uint8), tensor.view(torch.uint8)), name


@pytest.mark.parametrize("query", range(1, 6))
def test_decode_implicit_descales(query):
    from atrex import flash_attn_varlen_func

    case = build_case(64, [query, query], [1025, 2049], hq=32, hkv=2)
    for name in ("q_descale", "k_descale", "v_descale"):
        case[name].fill_(1)
    expected = _native_decode_reference(case, 1)
    for name in ("q_descale", "k_descale", "v_descale"):
        case[name] = None
    actual = flash_attn_varlen_func(**case, num_splits=1)
    torch.testing.assert_close(actual.float(), expected.float(), atol=.01, rtol=.05)


@pytest.mark.parametrize("update", [
    {"fa_version": 4}, {"num_splits": -1},
    {"return_softmax_lse": True}, {"causal": False},
])
def test_decode_unsupported_options_fail_fast(update):
    from atrex import can_use_flash_attn_varlen_func, flash_attn_varlen_func

    case = build_case(64, [4, 4], [1025, 2049], hq=32, hkv=2)
    case.update(update)
    assert not can_use_flash_attn_varlen_func(**case)
    with pytest.raises((NotImplementedError, ValueError)):
        flash_attn_varlen_func(**case)


@pytest.mark.parametrize("query", range(1, 6))
@pytest.mark.parametrize("heads", [(8, 1), (32, 2)])
def test_decode_graph_replay(query, heads):
    from atrex import flash_attn_varlen_func

    case = build_case(64, [query, query], [4097, 6145], hq=heads[0], hkv=heads[1])
    case["seqused_k"].fill_(query)
    out = torch.empty_like(case["q"], dtype=torch.bfloat16)
    flash_attn_varlen_func(**case, out=out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flash_attn_varlen_func(**case, out=out)
    for length in (query, 31, 32, 33, 511, 1023, 1024, 1025, 4097):
        case["seqused_k"].copy_(torch.tensor([length, length + 1], dtype=torch.int32, device="cuda"))
        case["q_descale"].mul_(.5)
        case["block_table"][:, :2].copy_(case["block_table"][:, :2].flip(1))
        out.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        reference_case = dict(case)
        for name in ("seqused_k", "q_descale", "block_table"):
            reference_case[name] = case[name].clone()
        expected = _native_decode_reference(reference_case, 0)
        torch.testing.assert_close(out.float(), expected.float(), atol=.01, rtol=.05)


@pytest.mark.parametrize("query", range(1, 6))
@pytest.mark.parametrize("heads", [(8, 1), (32, 2)])
def test_decode_real_kernel_events(query, heads):
    case = build_case(64, [query, query], [1025, 2049], hq=heads[0], hkv=heads[1])
    names = profile_case(case)
    assert any("atrex_ppu_native_decode_kernel" in name for name in names)
    assert any("atrex_ppu_native_decode_prepare" in name for name in names)
    assert not any("atrex_ppu_prep_raw" in name for name in names)


def _benchmark_decode(output_path, head_pairs, queries=(1, 4)):
    import gc
    import os
    from pathlib import Path
    import statistics
    import atrex
    from atrex import flash_attn_varlen_func

    torch.set_num_threads(4)
    payload = dict(complete=False, atrex_path=atrex.__file__,
        revision=os.environ.get("ATREX_TEST_REVISION"),
        device=torch.cuda.get_device_name(),
        cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        scope="Full graph GPU forward, excluding CPU allocation, compilation and capture",
        peak_bytes_per_second=3276800000000, descales="unit", cases=[])
    for head_pair in head_pairs:
        heads = tuple(map(int, head_pair.split("/")))
        for batch in (64, 128):
            for query in queries:
                for regime in ("uniform", "mixed_full"):
                    lengths = [32768] * batch if regime == "uniform" else [10241, 16383, 32769, 65535] * (batch // 4)
                    case = build_case(64, [query] * batch, lengths, hq=heads[0], hkv=heads[1])
                    for name in ("q_descale", "k_descale", "v_descale"):
                        case[name] = torch.ones((batch, heads[1]), dtype=torch.float32, device="cuda")
                    expected = _native_decode_reference(case, 0)
                    actual = flash_attn_varlen_func(**case)
                    torch.testing.assert_close(actual.float(), expected.float(), atol=.01, rtol=.05)
                    maximum_error = (actual.float() - expected.float()).abs().max().item()
                    invoke = partial(flash_attn_varlen_func, **case)
                    for warmup in range(5):
                        invoke()
                    torch.cuda.synchronize()
                    graph = torch.cuda.CUDAGraph()
                    with torch.cuda.graph(graph):
                        for iteration in range(10):
                            invoke()
                    for warmup in range(20):
                        graph.replay()
                    torch.cuda.synchronize()
                    start = torch.cuda.Event(enable_timing=True)
                    end = torch.cuda.Event(enable_timing=True)
                    samples = []
                    for repeat in range(31):
                        start.record()
                        graph.replay()
                        end.record()
                        end.synchronize()
                        samples.append(start.elapsed_time(end) * 1000 / 10)
                    median = statistics.median(samples)
                    useful_bytes = sum(lengths) * heads[1] * 256 * 2
                    payload["cases"].append(dict(batch=batch, query=query, regime=regime,
                        heads=list(heads), lengths=lengths, page_size=64, median_us=median,
                        samples_us=samples, useful_kv_bytes=useful_bytes,
                        mbu=useful_bytes / (median * 1e-6) / payload["peak_bytes_per_second"],
                        max_abs_error=maximum_error, native_correct=True))
                    Path(output_path).write_text(json.dumps(payload, indent=2) + "\n")
                    print(batch, query, heads, regime, median, flush=True)
                    del graph, invoke, case, actual, expected
                    gc.collect()
                    torch.cuda.empty_cache()
    payload["complete"] = True
    Path(output_path).write_text(json.dumps(payload, indent=2) + "\n")


def _check_decode_parity(paths):
    from pathlib import Path
    import statistics

    runs = [json.loads(Path(path).read_text()) for path in paths]
    assert len(runs) == 4 and all(run["complete"] for run in runs)
    indexed = []
    for run in runs:
        assert run["device"] == "ZW-M890P" and run["descales"] == "unit"
        cases = {(tuple(case["heads"]), case["batch"], case["query"],
                  case["regime"], case["page_size"]): case for case in run["cases"]}
        assert cases and len(cases) == len(run["cases"])
        indexed.append(cases)
    assert all(cases.keys() == indexed[0].keys() for cases in indexed)
    ratios = []
    spreads = []
    for key in indexed[0]:
        cases = [group[key] for group in indexed]
        for case in cases:
            assert case["native_correct"] and len(case["samples_us"]) == 31
            assert all(sample > 0 for sample in case["samples_us"])
            assert case["median_us"] == statistics.median(case["samples_us"])
            assert case["lengths"] == cases[0]["lengths"]
            assert case["useful_kv_bytes"] == cases[0]["useful_kv_bytes"]
        a1, b1, b2, a2 = [case["median_us"] for case in cases]
        ratio = max(b1, b2) / min(a1, a2)
        spread = max(abs(a1 - a2) / statistics.mean((a1, a2)),
                     abs(b1 - b2) / statistics.mean((b1, b2)))
        assert ratio <= 1.05, (key, "performance regression", ratio)
        assert spread <= .015, (key, "unstable measurement", spread)
        ratios.append(ratio)
        spreads.append(spread)
    print(json.dumps(dict(cases=len(ratios), maximum_latency_ratio=max(ratios),
                          maximum_repeat_spread=max(spreads), passed=True)))


def _benchmark_revision(contract, output_path, splits=0, descales="none"):
    """Compare the migrated prefill path with the unchanged source revision.

    Synthetic model-dimension cases, not captured production tensors. The old
    public provider requires explicit prefill counts; only this API difference
    is adapted. Its distinct padded decode implementation is not this baseline.
    """
    import json
    from pathlib import Path
    import statistics
    import time
    import atrex
    from op_test.utils.flash_attn_contract import reference

    assert torch.cuda.get_device_name() == 'ZW-M890P'
    if contract == 'legacy':
        from atrex.api.ppu_flash_attn_fp8 import flash_attn_varlen_func
        from atrex.src.cuda.ppu_flash_attn_fp8 import runtime
    else:
        from atrex import flash_attn_varlen_func
    records = []
    for batch, nq, nk in [(1, 256, 256), (1, 4096, 4096),
                          (8, 32, 4096), (32, 32, 4096),
                          (64, 32, 4096), (128, 32, 4096)]:
        case = build_case(64, [nq]*batch, [nk]*batch)
        for name in ('q_descale', 'k_descale', 'v_descale'):
            case[name] = (torch.ones((batch, 1), dtype=torch.float32, device="cuda")
                          if descales == "identity" else None)
        out = torch.empty_like(case['q'], dtype=torch.bfloat16)
        kwargs = dict(case, out=out, return_softmax_lse=False, num_splits=splits)
        if contract == 'legacy':
            kwargs.update(num_prefill=batch, num_prefill_tokens=batch*nq)
        def invoke():
            return flash_attn_varlen_func(**kwargs)
        engaged = runtime._ENGAGED if contract == 'legacy' else None
        actual = invoke()
        if contract == 'legacy':
            assert runtime._ENGAGED == engaged + 1, 'Baseline fell back to vendor'
        expected, _ = reference(case)
        torch.testing.assert_close(actual.float(), expected, atol=.03, rtol=.05)
        del expected
        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(5):
                invoke()
        torch.cuda.current_stream().wait_stream(stream)
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(20):
                invoke()
        warm_until = time.monotonic() + .5
        while time.monotonic() < warm_until:
            graph.replay()
            torch.cuda.synchronize()
        samples = []
        for _ in range(31):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end)/20)
        record = dict(batch=batch, q=nq, kv=nk, page_size=64, requested_splits=splits, descales=descales,
                      median_ms=statistics.median(samples), samples_ms=samples)
        records.append(record)
        print({k:v for k,v in record.items() if k != 'samples_ms'}, flush=True)
    with Path(output_path).open('x') as handle:
        json.dump(dict(contract=contract, atrex_path=atrex.__file__,
                       device=torch.cuda.get_device_name(), torch_version=torch.__version__,
                       cases=records), handle, indent=2)


def _benchmark_prep_kv_tail(output_path):
    """Same-input full-page/tail A/B, including the public API and prep alone."""
    import gc
    import os
    from pathlib import Path
    import statistics
    import atrex
    from atrex import flash_attn_varlen_func
    from atrex.src.ppu.flash_attn.zwm890p import runtime

    def measure(call):
        for _ in range(5):
            call()
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph):
            for _ in range(10):
                call()
        for _ in range(10):
            graph.replay()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        samples = []
        for _ in range(31):
            start.record()
            graph.replay()
            end.record()
            end.synchronize()
            samples.append(start.elapsed_time(end) * 1000 / 10)
        return dict(median_us=statistics.median(samples), samples_us=samples)

    payload = dict(complete=False, atrex_path=atrex.__file__,
                   revision=os.environ.get("ATREX_TEST_REVISION"),
                   device=torch.cuda.get_device_name(), cases=[])
    for hq, hkv in ((8, 1), (32, 2)):
        for batch, nq, nk in ((1, 256, 256), (1, 4096, 4096),
                              (8, 32, 4096), (32, 32, 4096),
                              (1, 17, 65536), (8, 32, 65536)):
            for tail in (0, 1):
                case = build_case(64, [nq] * batch, [nk + tail] * batch,
                                  hq=hq, hkv=hkv)
                case["k"], case["v"] = case["k"].contiguous(), case["v"].contiguous()
                for name in ("q_descale", "k_descale", "v_descale"):
                    case[name].fill_(1.)
                out = torch.empty_like(case["q"], dtype=torch.bfloat16)
                invoke = partial(flash_attn_varlen_func, **case, out=out, num_splits=1)
                actual = invoke()
                expected, _ = reference(case)
                torch.testing.assert_close(actual.float(), expected, atol=.03, rtol=.05)
                pages = (nk + tail + 63) // 64
                packed_k = torch.empty(batch * hkv * pages * 16384,
                                       dtype=torch.uint8, device="cuda")
                packed_v = torch.empty_like(packed_k)
                kscale = torch.empty(batch * hkv * pages, device="cuda")
                vamax = torch.empty(batch * hkv * 256, device="cuda")
                prep = partial(runtime._launch.build(case["q"].device).prep_kv,
                               case["k"], case["v"], case["block_table"],
                               case["seqused_k"], packed_k, packed_v, kscale, vamax, pages)
                row = dict(batch=batch, query=nq, kv=nk + tail, heads=[hq, hkv],
                           prep=measure(prep), public_api=measure(invoke))
                payload["cases"].append(row)
                Path(output_path).write_text(json.dumps(payload, indent=2) + "\n")
                print(json.dumps({k: v for k, v in row.items()
                                  if k not in ("prep", "public_api")}), flush=True)
                del invoke, prep, case, out, actual, expected, packed_k, packed_v, kscale, vamax
                gc.collect()
                torch.cuda.empty_cache()
    payload["complete"] = True
    Path(output_path).write_text(json.dumps(payload, indent=2) + "\n")




@pytest.mark.parametrize("query", range(1, 6))
def test_decode_gqa8_expanded_unit_descales(query):
    from atrex import flash_attn_varlen_func

    q_lens = [query, 1, max(1, query - 1)]
    case = build_case(64, q_lens, [127, 1025, 2049], hq=8, hkv=1)
    for name in ("q_descale", "k_descale", "v_descale"):
        case[name] = torch.ones((1, 1), device="cuda", dtype=torch.float32).expand(3, 1)
    expected = _native_decode_reference(case, 0)
    actual = flash_attn_varlen_func(**case)
    torch.testing.assert_close(actual.float(), expected.float(), atol=.01, rtol=.05)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--benchmark-contract', choices=['legacy', 'current'])
    mode.add_argument('--decode-benchmark-output')
    mode.add_argument('--decode-parity-inputs', nargs=4, metavar=('A1', 'B1', 'B2', 'A2'))
    mode.add_argument('--prep-kv-tail-benchmark-output')
    parser.add_argument('--benchmark-output')
    parser.add_argument('--decode-heads', nargs='+', choices=['8/1', '32/2'], default=['8/1', '32/2'])
    parser.add_argument('--decode-queries', nargs='+', type=int, choices=range(1, 6), default=[1, 4])
    parser.add_argument('--benchmark-splits', type=int, default=0, choices=[0, 1, 2, 4, 8])
    parser.add_argument('--benchmark-descales', choices=['none', 'identity'], default='none')
    args = parser.parse_args()
    if args.decode_parity_inputs:
        _check_decode_parity(args.decode_parity_inputs)
    elif args.prep_kv_tail_benchmark_output:
        _benchmark_prep_kv_tail(args.prep_kv_tail_benchmark_output)
    elif args.decode_benchmark_output:
        _benchmark_decode(args.decode_benchmark_output, args.decode_heads, args.decode_queries)
    elif args.benchmark_contract and args.benchmark_output:
        _benchmark_revision(args.benchmark_contract, args.benchmark_output, args.benchmark_splits,
                            args.benchmark_descales)
    else:
        parser.error('Select a decode benchmark/parity mode or both --benchmark-contract and --benchmark-output')
