# NVIDIA QSA tests

The public contract is documented in the [QSA guide](../../../src/nvidia/qsa/sm103/DESIGN.md).
Each target has one entry: `test_qsa_sm100.py` or `test_qsa_sm103.py`.
Tests require an installed ATREX wheel, CUDA 13+, PyTorch, Triton,
CuTeDSL (validated with 4.8.1), TVM-FFI, pytest and Nsight Systems.

```bash
python -m pip wheel --no-deps . -w /tmp/qsa-wheels
python -m pip install --no-deps /path/to/atrex.whl
python -m pytest -q -s op_test/nvidia/qsa/test_qsa_sm103.py
```

Replace `sm103` with `sm100` for that target. Source-tree imports are rejected.
The suite covers eager prefill, Q1/Q4 graph replay, metadata updates, empty
request slots, page boundaries, GQA 8/12, strided inputs, scales, alias checks,
prewarm requirements, stream independence, input lifetime and real kernel names.
It uses deterministic synthetic inputs and an independent FP32 reference.

## Numerical checks

BF16 and uniform-attention FP8 checks use `abs(error) <= .003 + .02*abs(ref)`.
Random FP8 fixtures retain the existing relative L2 limit of `.035`; their
strict element errors are also recorded. This fixture limit is not an accuracy
guarantee for all inputs. FP8 probability quantization can exceed the strict
element tolerance; existing failures must not be reported as strict passes.

## Paired performance regression

Install an accepted baseline and the candidate into separate directories.
Run the same eight cases against both wheels:

```bash
python op_test/nvidia/qsa/test_qsa_sm103.py --regression \
  --baseline-root /tmp/qsa-baseline --current-root /tmp/qsa-current \
  --output /tmp/qsa-regression.json
```

The matrix covers BF16/FP8, Q1/Q4, two KV heads and eager prefill, with
page16/128/1056/2112 and two metadata updates. Decode measures warm and cold-L2
CUDA Graph replay; prefill measures eager CUDA Events. Compilation and warmup
are excluded. Fifty samples alternate between wheels; cold-L2 replay clears
256 MiB outside the timed graph. Each latency must be within 5% of baseline.
Use `--cases 0,1` for a subset and `--reverse-capture` to investigate graph-order
sensitivity. The generated report contains source digests, dependency versions,
raw timings and accuracy metrics. Reports are generated outside the source tree.

Paired checks require finite outputs, strict new/old agreement, no new strict
reference failure and relative L2 growth no greater than 1% plus `1e-7`.
These are regression checks against an accepted baseline.
