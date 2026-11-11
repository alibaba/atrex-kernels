"""Basic public QSA contract on sm103; requires an installed wheel."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qsa_contract as contract


@pytest.fixture
def api():
    return contract.installed_api("sm103")


@pytest.mark.parametrize(
    "page,ratio", [(16, 8), (128, 8), (256, 12), (1056, 12), (2112, 8)]
)
@pytest.mark.parametrize("dtype", ["bf16", "fp8"])
def test_prefill_reference(api, page, ratio, dtype):
    contract.boundaries_and_metadata_refresh(api, page, ratio, dtype)


@pytest.mark.parametrize(
    "queries,batch,ratio,hkv,page,dtype",
    [
        (1, 3, 8, 1, 128, "bf16"),
        (1, 8, 12, 2, 16, "fp8"),
        (1, 128, 8, 1, 2112, "fp8"),
        (4, 3, 8, 1, 128, "fp8"),
        (4, 23, 8, 2, 256, "bf16"),
        (4, 64, 8, 1, 256, "fp8"),
        (4, 40, 12, 1, 1056, "bf16"),
        (4, 128, 8, 2, 2112, "fp8"),
    ],
)
def test_decode_graph(api, queries, batch, ratio, hkv, page, dtype):
    contract.decode_graph_uniform_attention(
        api, queries, batch, ratio, hkv, page, dtype
    )


def test_invalid_pages(api):
    contract.negative_pages(api)


def test_negative_contract_and_capture(api):
    contract.negative_contract_and_capture(api)


def test_cache_releases_inputs(api):
    contract.cache_releases_inputs(api)


def test_warmed_contract_checks_live_metadata(api):
    contract.warmed_contract_checks_live_metadata(api)


def test_decode_independent_streams(api):
    contract.decode_same_layout_independent_streams(api)


def test_decode_cached_scale_validation(api):
    contract.decode_cached_scale_validation(api)


def test_strided_query_and_independent_streams(api):
    contract.strided_query_and_independent_streams(api)


@pytest.mark.parametrize("dtype", ["bf16", "fp8"])
def test_capture_guards_and_independent_graphs(api, dtype):
    contract.decode_capture_guards_and_independent_graphs(api, dtype)


def test_kernel_names(api, tmp_path):
    contract.kernel_names(__file__, tmp_path)


@pytest.mark.parametrize("batch,hkv,ratio,page", [(23, 2, 8, 256)])
def test_strided_mtp_graph_metadata(api, batch, hkv, ratio, page, monkeypatch):
    """All KV partitions must use the same union after graph input updates."""
    import torch
    from qsa_contract import oracle, check
    from atrex.src.nvidia.qsa.sm103 import runtime

    torch.manual_seed(20261109 + batch + page)
    rows = 4 * batch
    heads = ratio * hkv
    per_request = (8192 + page - 1) // page
    pages = batch * per_request
    length = per_request * page
    cache_shape = (pages, page, hkv, 256)
    cache_stride = (page * (hkv * 256 + 8), hkv * 256 + 8, 256, 1)
    k = torch.empty_strided(
        cache_shape, cache_stride, device="cuda", dtype=torch.bfloat16
    ).normal_()
    v = torch.empty_strided(
        cache_shape, cache_stride, device="cuda", dtype=torch.bfloat16
    ).normal_()
    ids = (
        torch.rand((rows, (length - 4) // 4), device="cuda")
        .argsort(dim=1)[:, :512]
        .int()
        .contiguous()
    )
    positions = (
        torch.full((batch, 1), length - 4, device="cuda", dtype=torch.int64)
        + torch.arange(4, device="cuda")[None, :]
    ).reshape(-1)
    q = torch.randn((rows, heads, 256), device="cuda", dtype=torch.bfloat16)
    data = dict(
        q=q,
        k_cache=k,
        v_cache=v,
        indexer_block_ids=ids,
        query_positions=positions,
        block_table=torch.randperm(pages, device="cuda")
        .int()
        .reshape(batch, per_request),
        token_to_req=torch.arange(
            batch, device="cuda", dtype=torch.int32
        ).repeat_interleave(4),
        use_prefill_config=False,
        output_gate=torch.randn_like(q),
        out=torch.empty_like(q),
    )
    compiled = []
    original_compile = runtime._compile

    def observed(function, args, specs):
        op = getattr(function, "__self__", None)
        if getattr(op, "deterministic_union", False):
            compiled.append((op.br, op.kv_splits))
        return original_compile(function, args, specs)

    monkeypatch.setattr(runtime, "_compile", observed)
    api(**data)
    torch.cuda.synchronize()
    assert (4, 3) in compiled
    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph, stream=stream):
        api(**data)
    for epoch in range(4):
        if epoch == 1:
            ids[:, ::17] = -1
            data["token_to_req"].copy_(data["token_to_req"].roll(4))
        elif epoch == 2:
            ids[::4] = -1
            positions[::4] = ((positions[::4] + 1) // 4) * 4 - 1
        elif epoch == 3:
            counts = torch.tensor(
                [0, 1, 7, 8, 9, 23, 24, 25, 63, 64, 65], device="cuda"
            )
            live = counts[torch.arange(rows, device="cuda") % len(counts)]
            ids.masked_fill_(
                torch.arange(512, device="cuda")[None, :] >= live[:, None], -1
            )
            positions.copy_(((positions + 1) // 4) * 4 - 1)
        expected = oracle(data, {})
        first = None
        for _ in range(2):
            data["out"].fill_(float("nan"))
            graph.replay()
            torch.cuda.synchronize()
            check(data["out"], expected)
            if first is None:
                first = data["out"].clone()
            else:
                assert torch.equal(data["out"], first)
    graph.reset()


if __name__ == "__main__":
    if sys.argv[1:] == ["--profile"]:
        contract.profile_workload("sm103")
    else:
        from qsa_regression import main

        main("sm103")
