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
    assert not can_use_flash_attn_varlen_func(**dict(case, fa_version=4))
    with pytest.raises(NotImplementedError, match="fa_version"):
        flash_attn_varlen_func(**dict(case, fa_version=4))


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

    case = _case(64, [4, 2], [257, 513])
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

    case = _case(64, [4, 2], [257, 513])
    case["q_descale"] = torch.ones((1, 1), dtype=torch.float32, device="cuda").expand(2, 1)
    flash_attn_varlen_func(**case)
    case["q_descale"]._base.fill_(.5)
    with pytest.raises(NotImplementedError, match="unit descales"):
        flash_attn_varlen_func(**case)


def test_unversioned_inference_scale_is_not_cached():
    from atrex import flash_attn_varlen_func

    with torch.inference_mode():
        case = _case(64, [4, 2], [257, 513])
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


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark-contract', choices=['legacy', 'current'], required=True)
    parser.add_argument('--benchmark-output', required=True)
    parser.add_argument('--benchmark-splits', type=int, default=0, choices=[0, 1, 2, 4, 8])
    parser.add_argument('--benchmark-descales', choices=['none', 'identity'], default='none')
    args = parser.parse_args()
    _benchmark_revision(args.benchmark_contract, args.benchmark_output, args.benchmark_splits,
                        args.benchmark_descales)
