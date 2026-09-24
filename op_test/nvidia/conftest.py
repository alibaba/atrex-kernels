"""Require the NVIDIA target named by each test module."""

import pytest

from atrex.utils.device_target import detect_device_target


@pytest.fixture(autouse=True)
def nvidia_target(request):
    try:
        target = detect_device_target()
    except ModuleNotFoundError as error:
        if error.name != "torch":
            raise
        pytest.skip("requires Torch and NVIDIA hardware")

    if target.family != "nvidia":
        pytest.skip(
            f"requires NVIDIA hardware, detected {target.family}/{target.arch}"
        )
    expected_arch = request.path.stem.rsplit("_", 1)[-1]
    if expected_arch.startswith("sm") and target.arch != expected_arch:
        pytest.skip(f"requires NVIDIA {expected_arch}, detected {target.arch}")
    return target
