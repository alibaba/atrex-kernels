"""Basic public QSA contract on sm100; requires an installed wheel."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import qsa_contract as contract


@pytest.fixture
def api():
    return contract.installed_api("sm100")


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


def test_sm103_dispatch_preserved(api, monkeypatch):
    contract.sm103_dispatch_preserved(api, monkeypatch)


if __name__ == "__main__":
    if sys.argv[1:] == ["--profile"]:
        contract.profile_workload("sm100")
    else:
        from qsa_regression import main

        main("sm100")
