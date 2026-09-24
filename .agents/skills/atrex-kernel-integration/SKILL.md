---
name: atrex-kernel-integration
description: Integrate a GPU operator into ATREX with minimal source, compatible lazy APIs, hardware dispatch, atrex-prefixed kernels, one functional test file, and target-GPU validation.
---

# ATREX kernel integration

Integrate the smallest proven scope. Preserve real callers. Never add unused
helpers, speculative abstractions, placeholder architectures, unrelated legacy
code, or tests outside the claimed hardware.

## 1. Inspect first

- Read repository instructions and preserve unrelated worktree changes.
- Identify the language from the launched kernel, not the filename.
- Identify the implementation target separately from its language, using the
  compiler target, architecture-specific instructions, and runtime guards.
- Record the signature, tensor contract, outputs/mutation, supported hardware,
  compile trigger/cache, reference, tolerances, and real downstream calls.
- Search existing APIs/tests. Treat an old PR as a file inventory unless the
  user explicitly requests that snapshot.
- Move only the implementation and its transitive runtime helpers.

## 2. Place source

Organize by implemented hardware target, never by DSL:

```text
src/<vendor>/<operator>/<arch>/
src/<vendor>/<operator>/common_utils/  # only proven cross-arch helpers
```

For example, NVIDIA SM120 Chunk-GDN belongs in `src/nvidia/chunk_gdn/sm120/`.
Use lowercase vendor names (`nvidia`, `amd`, `ppu`) and create no empty variants.
Keep language-specific compilation and kernel naming inside that target
implementation. Native CUDA/HIP must use native source and binding files, not
Python source strings. Add headers only when genuinely shared.

## 3. API and dispatch

- Exact semantic match: reuse the public API.
- Small compatible difference: propose argument/output/fallback mapping before
  coding the contract.
- No match: add `python/atrex/api/<operator>.py` and register only the agreed
  public functions in `python/atrex/__init__.py` via `_lazy_import_and_call`.

Preserve established import paths, names, positional behavior, defaults, and
returns, even when a legacy name mentions an implementation language. Keep
selectors, layouts, compiler helpers, and unused historical exports private.

Dispatch one semantic API in this order:

1. Detect runtime family and architecture from the input tensor's device.
2. Check dtype, shape, layout, options, and required runtime version.
3. Select the first eligible implementation in an explicit operator policy;
   otherwise use the documented fallback.

Probe Torch on the actual Agate image before changing detection. Check AMD via
`torch.version.hip`; check exact PPU product names such as `ZW-M890P` before
NVIDIA; classify NVIDIA only when CUDA is present and the product name starts
with `NVIDIA`, then use device capability for `smXY`. Otherwise return unknown.
`can_use_*` must not compile, allocate large buffers, switch devices, or mutate
tensors. Import only the selected implementation.

## 4. Compile and name kernels

`import atrex` must not compile kernels. Identify whether compilation happens on
the first API call, first specialization launch, or explicit prewarm. Cache by
every value that changes generated code or device context.

Every profiler-visible entry point, including helper kernels, starts with
`atrex_`:

| Language | Required control |
| --- | --- |
| CUDA/HIP | launched `__global__` name |
| Triton/Gluon | launched JIT function name |
| CuTeDSL | kernel name plus `set_name_prefix("atrex")` before `cute.compile` |
| FlyDSL/TileLang | compiled launch function name |
| CK | generated GPU entry symbol |

Profile the real public API on each claimed target using a dedicated workload
or profiler range. Use Torch profiler when it exposes raw launches; otherwise
use Nsight Systems (NVIDIA), rocprof (AMD), or ACU (PPU). Separate the call's
operator-owned GPU events from framework/third-party work, include helper
kernels, and fail on an empty event list or any name lacking `atrex_`. Record
the observed names and profiler evidence in the operator's test result. Do not
add a synthetic probe kernel for this gate.

## 5. One functional test file per operator target

Each implemented operator target has exactly one test module:

```text
op_test/<vendor>/<operator>/test_<operator>_<arch>.py
```

For example, `op_test/nvidia/chunk_gdn/test_chunk_gdn_sm120.py`. Keep
correctness, dispatch/fallback, negative eligibility, prewarm, profiler, and
bounded performance checks together. A profiler child process must invoke the
same file through a private CLI mode. Do not add package-smoke,
`benchmark_*.py`, or `profile_*.py` files for operators.

Generate only supported cases; do not skip known-invalid combinations. On a
claimed target, missing compilers, profilers, or references are failures. Use a
direct standalone reference, not vLLM/FLA application plumbing. Test collection
must not compile kernels. Build and install the wheel once per revision and
environment, then run all applicable operator tests against that installation;
tests must fail instead of importing from the source tree.

## 6. Validate by implementation target

Load `atrex-gpu-gateway` before Agate.

Never select hardware from the DSL or source language. Select the Agate
environment from the target implemented by this operator variant:

| Implemented target | Agate environment |
| --- | --- |
| NVIDIA `sm100` | L20A or L20C |
| NVIDIA `sm103` | L20B or L20D |
| NVIDIA `sm120` | L20N |
| AMD `gfx942` | MI308 |
| Alibaba PPU `zwm890p` | PPU ZWM890P |

For an unlisted target, probe and use matching hardware; never infer a fallback
environment from CUDA, Triton, Gluon, CuTeDSL, or any other implementation
technology.

Run only the applicable operator test files on every claimed target. Require
zero applicable skips, reference correctness, live dispatch, bounded
performance, and observed `atrex_` events. Install once per environment, never
once per test file, and do not add a separate package-smoke test.

For an operator migrated from another revision, also run a performance-parity
gate. In one target-GPU job, install the baseline revision (normally the latest
`master` before migration) and the candidate into isolated paths, then run the
candidate's same operator test benchmark against both packages through the same
public API. Keep inputs, dependencies, warmup, repetitions, clocks, and benchmark
order identical; exclude compilation and prewarm. Require every production case
to be no more than 5% slower than baseline, and report both revision IDs,
per-case median latency/ratio, and the job ID. If the baseline needs an
environment-only compatibility fix, isolate and disclose that patch; never
change its kernel math or dispatch. A missing or unreproducible baseline fails
the gate.

Before handoff, inspect the relevant diff and `git diff --check`. For review
feedback, refresh the latest revision, fix valid items, reply to every item with
the changed location or no-change evidence, and rerun only affected operator
gates. Report the contract, dispatch/fallback, compile trigger, target, exact
test result/job ID, profiler names, and limits. Commit, push, or open a review
only when requested.
