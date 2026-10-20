"""PPU prefill and native direct-paged FP8 decode contracts."""

import pytest
import torch
from functools import partial

from op_test.utils.flash_attn_contract import build_case, reference, profile_case
from op_test.utils.flash_attn_contract import check_eager_scratch_reuse


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
    assert torch.isfinite(expected).all()
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
    _check(case, splits=splits)


def _prep_layout_indices():
    token = torch.arange(64, dtype=torch.int64)[:, None]
    channel = torch.arange(256, dtype=torch.int64)[None, :]
    group = 2 * ((token & 15) >> 2) + (token & 1)
    local_key = 72 * group + 2 * ((channel & 31) >> 3) + ((channel >> 1) & 1)
    local_key += torch.where((channel & 4) != 0,
                             torch.where((group & 1) != 0, -8, 8), 0)
    local_key += ((token >> 1) & 1) * 512
    key_index = (2 * (token >> 5) + (channel >> 7)) * 2048
    key_index += ((token >> 4) & 1) * 1024
    key_index += local_key ^ (((channel >> 5) & 3) * 16)
    key_index = key_index * 2 + (channel & 1)
    slot = 32 * (token >> 5) + 8 * ((token >> 2) & 3) + 4 * ((token >> 4) & 1) + (token & 3)
    group = 2 * ((channel & 15) >> 2) + (channel & 1)
    local_value = 32 * group + 8 * (group >> 1) + 2 * ((slot & 31) >> 3) + ((slot >> 1) & 1)
    local_value += torch.where((slot & 4) != 0,
                               torch.where(((group >> 1) & 1) != 0, -8, 8), 0)
    local_value += ((channel >> 1) & 1) * 512
    value_index = ((channel >> 4) >> 1) * 1024 + ((channel >> 4) & 1) * 256
    value_index = (value_index + (local_value ^ ((slot >> 5) * 16))) * 2 + (slot & 1)
    return key_index.flatten(), value_index.flatten()


@pytest.mark.parametrize("kv_heads", [1, 2, 3])
@pytest.mark.parametrize("page_size", [64, 128, 256])
def test_prep_kv_bit_exact_and_graph_lengths(kv_heads, page_size):
    from atrex.src.ppu.flash_attn.zwm890p import runtime

    lengths = [0, 63, 65, 129, 511]
    case = build_case(page_size, [1] * len(lengths), lengths, hq=kv_heads * 8, hkv=kv_heads)
    shape = case["k"].shape
    bits = torch.arange(case["k"].numel(), dtype=torch.int64).reshape(shape)
    key_bits = ((bits * 13 + (bits // 256) * 17) & 255).to(torch.uint8)
    value_bits = ((bits * 29 + (bits // 256) * 31) & 255).to(torch.uint8)
    table = case["block_table"].cpu()
    for request, length in enumerate(lengths):
        if length % page_size:
            physical = int(table[request, length // page_size])
            value_bits[physical, length % page_size:].fill_(0x7f)
    case["k"].view(torch.uint8).copy_(key_bits.to("cuda"))
    case["v"].view(torch.uint8).copy_(value_bits.to("cuda"))
    key_view, virtual_stride, subpages = runtime._page_view(case["k"])
    value_view = runtime._page_view(case["v"])[0]
    remapped = runtime._remap_block_table(case["block_table"], virtual_stride, subpages)
    max_pages = 8
    packed_key = torch.full((len(lengths) * kv_heads * max_pages * 16384,), 0xa5,
                            dtype=torch.uint8, device="cuda")
    packed_value = torch.full_like(packed_key, 0xa5)
    key_scale = torch.empty(len(lengths) * kv_heads * max_pages, device="cuda")
    value_max = torch.empty(len(lengths) * kv_heads * 256, device="cuda")
    native = runtime._launch.build(case["q"].device)
    invoke = partial(native.prep_kv, key_view, value_view, remapped, case["seqused_k"],
                     packed_key, packed_value, key_scale, value_max, max_pages)
    invoke()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        invoke()
    key_index, value_index = _prep_layout_indices()
    assert torch.equal(key_index.sort().values, torch.arange(16384))
    assert torch.equal(value_index.sort().values, torch.arange(16384))
    for current in (lengths, [0, 64, 128, 130, 512]):
        case["seqused_k"].copy_(torch.tensor(current, dtype=torch.int32, device="cuda"))
        packed_key.fill_(0xa5)
        packed_value.fill_(0xa5)
        graph.replay()
        expected_key = torch.full((len(lengths), kv_heads, max_pages, 16384), 0xa5, dtype=torch.uint8)
        expected_value = torch.full_like(expected_key, 0xa5)
        for request, length in enumerate(current):
            for page in range((length + 63) // 64):
                physical = int(table[request, page // subpages])
                start = (page % subpages) * 64
                for head in range(kv_heads):
                    key_page = key_bits[physical, start:start + 64, head]
                    value_page = value_bits[physical, start:start + 64, head].clone()
                    value_page[max(0, length - page * 64):].zero_()
                    expected_key[request, head, page].index_copy_(0, key_index, key_page.flatten())
                    expected_value[request, head, page].index_copy_(0, value_index, value_page.flatten())
        assert torch.equal(packed_key.cpu(), expected_key.flatten())
        assert torch.equal(packed_value.cpu(), expected_value.flatten())
    assert torch.equal(case["k"].view(torch.uint8).cpu(), key_bits)
    assert torch.equal(case["v"].view(torch.uint8).cpu(), value_bits)


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
    expected, _ = reference(case)
    for splits in (0, 1):
        actual = flash_attn_varlen_func(**case, num_splits=splits)
        torch.testing.assert_close(actual.float(), expected, atol=.03, rtol=.05)


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
        expected, _ = reference(case)
        actual = flash_attn_varlen_func(**case)
        torch.testing.assert_close(actual.float(), expected, atol=.03, rtol=.05)


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
    expected, _ = reference(case)
    for splits in (0, 1):
        actual = flash_attn_varlen_func(**case, num_splits=splits)
        torch.testing.assert_close(actual.float(), expected, atol=.03, rtol=.05)


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
    expected, _ = reference(case)
    actual = flash_attn_varlen_func(**case, out=out)
    torch.testing.assert_close(actual.float(), expected, atol=.03, rtol=.05)


def test_decode_build_specializations():
    from atrex.src.ppu.flash_attn.zwm890p import decode_build

    source = (decode_build.ROOT / "native_decode.cu").read_text()
    assert "#include <cute/" not in source
    assert "#include <cutlass/" not in source
    assert decode_build.DEFINES["q1"]["Q1_ONLY"] == 1
    assert decode_build.DEFINES["q4"]["Q1_ONLY"] == 0


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
        original_launch = library.atrex_ppu_decode

        def invoke(pointers, *args):
            assert len(pointers) == 13
            return original_launch(pointers, *args)

        monkeypatch.setattr(library, "atrex_ppu_decode", invoke)
        return library

    monkeypatch.setattr(decode_build, "load_library", load)
    case = build_case(64, [query, query], [127, 257], hq=32, hkv=2)
    actual = flash_attn_varlen_func(**case, num_splits=1)
    expected, _ = reference(case)
    torch.testing.assert_close(actual.float(), expected, atol=.03, rtol=.05)
    assert loaded == ["q1" if query == 1 else "q4"]


def test_eligibility_does_not_compile_or_launch(monkeypatch):
    from atrex import can_use_flash_attn_varlen_func
    from atrex.src.ppu.flash_attn.zwm890p import launch
    from atrex.src.ppu.flash_attn.zwm890p import decode_build
    from atrex.src.ppu.flash_attn.zwm890p import runtime
    import atrex.api.flash_attn as api

    def unexpected(*args, **kwargs):
        raise AssertionError("Eligibility must not compile or execute kernels")

    monkeypatch.setattr(decode_build, "load_library", unexpected)
    monkeypatch.setattr(runtime._launch, "build", unexpected)
    monkeypatch.setattr(launch, "forward", unexpected)
    original_validate = api._validate_flash_attn_call

    def validate(call):
        assert "_eligibility_only" not in call
        return original_validate(call)

    monkeypatch.setattr(api, "_validate_flash_attn_call", validate)
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
        expected, _ = reference(case)
        actual = flash_attn_varlen_func(**case, out=out, num_splits=splits)
        assert actual.data_ptr() == out.data_ptr()
        torch.testing.assert_close(actual.float(), expected, atol=.03, rtol=.05)
        if phase == 0:
            for name, tensor in saved.items():
                assert torch.equal(case[name].view(torch.uint8), tensor.view(torch.uint8)), name


@pytest.mark.parametrize("query", range(1, 6))
def test_decode_implicit_descales(query):
    from atrex import flash_attn_varlen_func

    case = build_case(64, [query, query], [1025, 2049], hq=32, hkv=2)
    for name in ("q_descale", "k_descale", "v_descale"):
        case[name].fill_(1)
    expected, _ = reference(case)
    for name in ("q_descale", "k_descale", "v_descale"):
        case[name] = None
    actual = flash_attn_varlen_func(**case, num_splits=1)
    torch.testing.assert_close(actual.float(), expected, atol=.03, rtol=.05)


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
        expected, _ = reference(case)
        torch.testing.assert_close(out.float(), expected, atol=.03, rtol=.05)


@pytest.mark.parametrize("query", [1, 4, 5])
@pytest.mark.parametrize("heads", [(8, 1), (32, 2)])
def test_decode_real_kernel_events(query, heads):
    case = build_case(64, [query, query], [1025, 2049], hq=heads[0], hkv=heads[1])
    names = profile_case(case)
    assert any("atrex_ppu_native_decode_kernel" in name for name in names)
    assert any("atrex_ppu_native_decode_prepare" in name for name in names)
    assert not any("atrex_ppu_prep_raw" in name for name in names)


@pytest.mark.parametrize("query", range(1, 6))
def test_decode_gqa8_expanded_unit_descales(query):
    from atrex import flash_attn_varlen_func

    q_lens = [query, 1, max(1, query - 1)]
    case = build_case(64, q_lens, [127, 1025, 2049], hq=8, hkv=1)
    for name in ("q_descale", "k_descale", "v_descale"):
        case[name] = torch.ones((1, 1), device="cuda", dtype=torch.float32).expand(3, 1)
    expected, _ = reference(case)
    actual = flash_attn_varlen_func(**case)
    torch.testing.assert_close(actual.float(), expected, atol=.03, rtol=.05)
