"""Original PPU prep_kv/kernel: implicit/explicit unit scales, no LSE."""

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
    _check(_case(page_size, q_lens, kv_lens, contiguous=contiguous), splits=splits)


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
    with pytest.raises(NotImplementedError, match="descale"):
        flash_attn_varlen_func(**case)


@pytest.mark.parametrize("splits", [1, 4])
def test_unit_descales_and_graph(splits):
    case = _case(64, [65, 3], [257, 513])
    for name in ("q_descale", "k_descale", "v_descale"):
        case[name] = torch.ones((1, 1), dtype=torch.float32, device="cuda").expand(2, 1)
    _check(case, splits=splits, graph=True)


def test_unit_scale_mutation_invalidates_validation():
    from atrex import flash_attn_varlen_func

    case = _case(64, [65, 3], [257, 513])
    case["q_descale"] = torch.ones((1, 1), dtype=torch.float32, device="cuda").expand(2, 1)
    flash_attn_varlen_func(**case)
    case["q_descale"]._base.fill_(.5)
    with pytest.raises(NotImplementedError, match="unit descales"):
        flash_attn_varlen_func(**case)


def test_unversioned_inference_scale_is_not_cached():
    from atrex import flash_attn_varlen_func

    with torch.inference_mode():
        case = _case(64, [65, 3], [257, 513])
        case["q_descale"] = torch.ones((2, 1), dtype=torch.float32, device="cuda")
        flash_attn_varlen_func(**case)
        case["q_descale"].fill_(.5)
        with pytest.raises(NotImplementedError, match="unit descales"):
            flash_attn_varlen_func(**case)


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
    (1, 8, 1, "decode"), (2, 8, 1, "prefill"), (5, 8, 1, "prefill"),
    (4, 8, 1, "decode"), (65, 8, 1, "prefill"),
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
    monkeypatch.setattr(launch, "_ensure_prefill_jit", lambda call: None)
    assert flash_attn_varlen_func(**case, out=out).data_ptr() == out.data_ptr()
    assert calls == [path]
    assert attempted == (["decode"] if query <= 5 else [])


@pytest.mark.parametrize("error", [ValueError("invalid metadata"), RuntimeError("launch failed")])
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
    monkeypatch.setattr(launch, "_ensure_prefill_jit", lambda call: None)
    with pytest.raises(type(error), match=str(error)):
        flash_attn_varlen_func(**case)
    assert not calls


@pytest.mark.parametrize("query", [1, 4])
@pytest.mark.parametrize("heads", [(8, 1), (16, 1), (32, 2), (64, 4)])
@pytest.mark.parametrize("page_size", [64, 128, 256])
def test_decode_runtime_heads_and_pages(query, heads, page_size):
    from atrex import can_use_flash_attn_varlen_func, flash_attn_varlen_func

    query_heads, kv_heads = heads
    q_lens = [1, 1, 1] if query == 1 else [4, 1, 3]
    case = build_case(page_size, q_lens, [127, 1025, 2049], hq=query_heads, hkv=kv_heads)
    assert can_use_flash_attn_varlen_func(**case)
    for splits in (0, 1):
        expected = _native_decode_reference(case, splits)
        actual = flash_attn_varlen_func(**case, num_splits=splits)
        torch.testing.assert_close(actual.float(), expected.float(), atol=.01, rtol=.05)


@pytest.mark.parametrize("query", [2, 5])
def test_short_q_prefill_fallback_graph(query):
    _check(_case(128, [query, query], [1025, 2049]), graph=True)


def test_shared_decode_sources_and_independent_specializations():
    from atrex.src.ppu.flash_attn.zwm890p import decode_build

    for filename in ("decode_runtime.py", "decode_kernel.cu", "decode_scheduler.cu", "decode_resources.h"):
        assert (decode_build.ROOT / filename).is_file()
    assert (decode_build.ROOT / "decode/vendor/hopper/mainloop_fwd_sm80.hpp").is_file()
    for directory in ("q1", "q4", "common"):
        assert not (decode_build.ROOT / "decode" / directory).exists()
    assert set(decode_build.DEFINES) == {"q1", "q4"}
    assert decode_build.DEFINES["q1"]["Q1_ONLY"] == 1
    assert decode_build.DEFINES["q4"]["Q1_ONLY"] == 0
    assert decode_build.DEFINES["q1"]["QK_WINDOW"] != decode_build.DEFINES["q4"]["QK_WINDOW"]


def test_eligibility_does_not_compile_or_launch(monkeypatch):
    from atrex import can_use_flash_attn_varlen_func
    from atrex.src.ppu.flash_attn.zwm890p import launch
    from atrex.src.ppu.flash_attn.zwm890p import decode_build

    def unexpected(*args, **kwargs):
        raise AssertionError("Eligibility must not compile or execute kernels")

    monkeypatch.setattr(launch, "_ensure_prefill_jit", unexpected)
    monkeypatch.setattr(decode_build, "load_library", unexpected)
    assert can_use_flash_attn_varlen_func(**build_case(128, [4, 4], [1025, 2049], hq=64, hkv=4))
    assert can_use_flash_attn_varlen_func(**_case(64, [4, 4], [1025, 2049]))


@pytest.mark.parametrize("query,splits", [(1, 0), (1, 1), (1, 4), (1, 32), (4, 0), (4, 1)])
@pytest.mark.parametrize("layout", ["interleaved", "contiguous", "strided"])
def test_direct_decode_contract_and_mutation(query, splits, layout):
    from atrex import can_use_flash_attn_varlen_func, flash_attn_varlen_func

    q_lens = [query] * 3 if query == 1 else [4, 1, 3]
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


@pytest.mark.parametrize("query", [1, 4])
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
    {"fa_version": 2}, {"max_seqlen_q": 2}, {"num_splits": 4},
    {"num_splits": 33}, {"return_softmax_lse": True}, {"causal": False},
])
def test_decode_unsupported_options_fail_fast(update):
    from atrex import can_use_flash_attn_varlen_func, flash_attn_varlen_func

    case = build_case(64, [4, 4], [1025, 2049], hq=32, hkv=2)
    case.update(update)
    assert not can_use_flash_attn_varlen_func(**case)
    with pytest.raises((NotImplementedError, ValueError)):
        flash_attn_varlen_func(**case)


@pytest.mark.parametrize("query", [1, 4])
@pytest.mark.parametrize("heads", [(8, 1), (32, 2)])
def test_decode_graph_replay(query, heads):
    from atrex import flash_attn_varlen_func

    case = build_case(64, [query, query], [1025, 2049], hq=heads[0], hkv=heads[1])
    out = torch.empty_like(case["q"], dtype=torch.bfloat16)
    flash_attn_varlen_func(**case, out=out)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        flash_attn_varlen_func(**case, out=out)
    case["seqused_k"].sub_(1)
    case["q_descale"].mul_(.5)
    graph.replay()
    torch.cuda.synchronize()
    expected = _native_decode_reference(case, 0)
    torch.testing.assert_close(out.float(), expected.float(), atol=.01, rtol=.05)


@pytest.mark.parametrize("query", [1, 4])
@pytest.mark.parametrize("heads", [(8, 1), (32, 2)])
def test_decode_real_kernel_events(query, heads):
    case = build_case(64, [query, query], [1025, 2049], hq=heads[0], hkv=heads[1])
    names = profile_case(case)
    assert any("atrex_ppu_decode_prepare_scheduler" in name for name in names)
    assert any("atrex_ppu_decode_kernel" in name for name in names)
    assert not any("atrex_ppu_prep_raw" in name for name in names)


def _decode_deployment_case(batch, query, regime, seed):
    import random

    torch.manual_seed(seed)
    generator = random.Random(seed)
    lengths = [32768] * batch
    if regime != "uniform":
        lengths = ([10241, 16383, 32769, 65535] * ((batch + 3) // 4))[:batch]
        if regime == "mixed_sorted":
            lengths.sort()
        else:
            generator.shuffle(lengths)
    counts = [1 if regime == "mixed" and query == 4 and request % 5 == 0 else query
              for request in range(batch)]
    pages = [(length + 63) // 64 for length in lengths]
    capacity = sum(pages) + 7
    backing = torch.empty((capacity, 2, 64, 2, 256), dtype=torch.float8_e4m3fn, device="cuda")
    for start in range(0, capacity, 512):
        count = min(512, capacity - start)
        backing[start:start + count].copy_(torch.randn(
            (count, 2, 64, 2, 256), dtype=torch.bfloat16, device="cuda"
        ).clamp_(-448, 448).to(torch.float8_e4m3fn))
    physical = torch.randperm(capacity, dtype=torch.int64, device="cuda")
    table = torch.full((batch, max(pages) + 5), -1, dtype=torch.int32, device="cuda")
    cursor = 0
    for request, count in enumerate(pages):
        table[request, :count] = physical[cursor:cursor + count].int()
        cursor += count
    offsets = [0]
    for count in counts:
        offsets.append(offsets[-1] + count)
    q = torch.randn((offsets[-1], 32, 256), dtype=torch.bfloat16, device="cuda").clamp_(
        -448, 448).to(torch.float8_e4m3fn)
    return dict(
        q=q, k=backing[:, 0], v=backing[:, 1], out=torch.empty_like(q, dtype=torch.bfloat16),
        cu_seqlens_q=torch.tensor(offsets, dtype=torch.int32, device="cuda"),
        seqused_k=torch.tensor(lengths, dtype=torch.int32, device="cuda"),
        max_seqlen_q=max(counts), max_seqlen_k=max(lengths), softmax_scale=.0625,
        causal=True, window_size=(-1, -1), block_table=table, scheduler_metadata=None,
        fa_version=3, num_splits=0,
        q_descale=torch.ones((batch, 2), device="cuda", dtype=torch.float32),
        k_descale=torch.ones((batch, 2), device="cuda", dtype=torch.float32),
        v_descale=torch.ones((batch, 2), device="cuda", dtype=torch.float32),
    ), dict(batch=batch, query=query, regime=regime, seed=seed,
            kv_lengths=lengths, query_counts=counts, useful_bytes=sum(lengths) * 1024)


def _benchmark_decode_parity(output_path):
    import ast
    import ctypes
    import gc
    import hashlib
    import os
    from pathlib import Path
    import statistics
    import atrex
    from atrex import flash_attn_varlen_func
    from atrex.src.ppu.flash_attn.zwm890p.decode_build import load_library

    assert os.environ["CUDA_VISIBLE_DEVICES"] == "2"
    assert torch.cuda.device_count() == 1 and torch.cuda.get_device_name() == "ZW-M890P"
    torch.set_num_threads(4)
    source = Path(os.environ["V83_RUNTIME"])
    tree = ast.parse(source.read_text())
    tree.body = [node for node in tree.body if not isinstance(node, ast.ImportFrom) or node.level == 0]
    libraries = {}
    for query in (1, 4):
        library = ctypes.CDLL(str(Path(os.environ["V83_ROOT"]) / f"build/q{query}/actual/libtranspose.so"))
        library.native_decode.restype = ctypes.c_int
        library.native_decode.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.POINTER(ctypes.c_int64),
            ctypes.c_float, ctypes.c_int, ctypes.c_int, ctypes.c_void_p]
        libraries[f"q{query}"] = library
    namespace = {"load_library": libraries.__getitem__}
    exec(compile(tree, str(source), "exec"), namespace)
    baseline = namespace["run"]
    payload = dict(complete=False, atrex_path=atrex.__file__, physical_gpu=2,
        baseline_runtime_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),
        scope="Full eager forward, including invocation-local scratch/scheduler/combine",
        peak_bytes_per_second=3276800000000, libraries={}, cases=[])
    for query in (1, 4):
        library = load_library(f"q{query}", torch.device("cuda:0"))
        values = (ctypes.c_int * 4)()
        resource = library.atrex_ppu_decode_resources
        assert resource(values) == 0
        payload["libraries"][f"q{query}"] = dict(
            baseline_path=libraries[f"q{query}"]._name,
            baseline_sha256=hashlib.sha256(Path(libraries[f"q{query}"]._name).read_bytes()).hexdigest(),
            candidate_path=library._name,
            candidate_sha256=hashlib.sha256(Path(library._name).read_bytes()).hexdigest(),
            registers=values[0], shared_bytes=values[1], local_bytes=values[2], blocks_per_cu=values[3],
        )

    def measure(function, arguments):
        for warmup in range(5):
            function(**arguments)
        torch.cuda.synchronize()
        start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
        start.record()
        for iteration in range(20):
            function(**arguments)
        end.record()
        end.synchronize()
        return start.elapsed_time(end) * 1000 / 20

    for batch in (64, 128):
        for query in (1, 4):
            for regime in ("uniform", "mixed", "mixed_full", "mixed_sorted"):
                for seed in (20260925, 20260926):
                    arguments, identity = _decode_deployment_case(batch, query, regime, seed)
                    expected = _native_decode_reference(arguments, 0).clone()
                    before = baseline(**arguments).clone()
                    actual = flash_attn_varlen_func(**arguments)
                    torch.testing.assert_close(actual.float(), expected.float(), atol=.01, rtol=.05)
                    assert torch.equal(before, actual)
                    blocks = []
                    for repeat in range(3):
                        blocks.append([dict(label=label, us=measure(function, arguments))
                            for label, function in (("A", baseline), ("B", flash_attn_varlen_func),
                                                    ("B", flash_attn_varlen_func), ("A", baseline))])
                    summary = {}
                    for label in ("A", "B"):
                        pairs = [statistics.mean(sample["us"] for sample in block if sample["label"] == label)
                                 for block in blocks]
                        latency = statistics.mean(pairs)
                        summary[label] = dict(us=latency,
                            mbu=identity["useful_bytes"] / latency / 1e-6 / payload["peak_bytes_per_second"],
                            repeat_spread=(max(pairs) - min(pairs)) / latency)
                    record = dict(identity=identity, native_pass=True, bitwise_v83=True,
                        blocks=blocks, summary=summary, ratio=summary["B"]["us"] / summary["A"]["us"])
                    payload["cases"].append(record)
                    Path(output_path).write_text(json.dumps(payload, indent=2) + "\n")
                    assert torch.equal(baseline(**arguments), flash_attn_varlen_func(**arguments))
                    assert record["ratio"] <= 1.05, record
                    assert max(summary[label]["repeat_spread"] for label in ("A", "B")) <= .015, record
                    print("DECODE_PARITY", identity["batch"], query, regime, seed, summary, flush=True)
                    del arguments, expected, before, actual
                    gc.collect()
                    torch.cuda.empty_cache()
    payload["complete"] = True
    Path(output_path).write_text(json.dumps(payload, indent=2) + "\n")


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




@pytest.mark.parametrize("query", [1, 4])
def test_decode_gqa8_expanded_unit_descales(query):
    from atrex import flash_attn_varlen_func

    q_lens = [1, 1, 1] if query == 1 else [4, 1, 3]
    case = build_case(64, q_lens, [127, 1025, 2049], hq=8, hkv=1)
    for name in ("q_descale", "k_descale", "v_descale"):
        case[name] = torch.ones((1, 1), device="cuda", dtype=torch.float32).expand(3, 1)
    expected = _native_decode_reference(case, 0)
    actual = flash_attn_varlen_func(**case)
    torch.testing.assert_close(actual.float(), expected.float(), atol=.01, rtol=.05)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark-contract', choices=['legacy', 'current'])
    parser.add_argument('--benchmark-output')
    parser.add_argument('--decode-parity-output')
    parser.add_argument('--benchmark-splits', type=int, default=0, choices=[0, 1, 2, 4, 8])
    parser.add_argument('--benchmark-descales', choices=['none', 'identity'], default='none')
    args = parser.parse_args()
    if args.decode_parity_output:
        _benchmark_decode_parity(args.decode_parity_output)
    elif args.benchmark_contract and args.benchmark_output:
        _benchmark_revision(args.benchmark_contract, args.benchmark_output, args.benchmark_splits,
                            args.benchmark_descales)
    else:
        parser.error('Select --decode-parity-output or both --benchmark-contract and --benchmark-output')
