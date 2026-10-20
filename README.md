# ATREX

ATREX is a lightweight package for hardware-specific GPU operators. Operators
are integrated through one consistent review and validation workflow.

## Repository layout

```text
python/atrex/                              Public APIs and dispatch
src/<vendor>/<operator>/<arch>/            Target-specific implementations
src/<vendor>/<operator>/common_utils/      Proven cross-architecture helpers
op_test/<vendor>/<operator>/test_<operator>_<arch>.py
                                           Target-specific functional tests
.agents/skills/atrex-kernel-integration/   Integration workflow
```

Directory placement follows the implementation target, not its language.
Public APIs belong in `python/atrex/api/` and are lazily exported from
`python/atrex/__init__.py`. When implementations share an API, dispatch uses
the detected runtime, architecture, version, and call-specific eligibility.

## Test gate

An operator change is incomplete unless its matching target test is added or
updated in the same change. The current files are:

```text
op_test/nvidia/chunk_gdn/test_chunk_gdn_sm103.py
op_test/nvidia/chunk_gdn/test_chunk_gdn_sm120.py
op_test/nvidia/flash_attn/test_flash_attn_sm103.py
op_test/nvidia/flash_attn/test_flash_attn_sm120.py
op_test/ppu/flash_attn/test_flash_attn_zwm890p.py
```

The SM103 Chunk-GDN path integrates the AKA M64 implementation behind the
existing public API. Its adapter performs the caller-requested Q/K L2
normalization and supplies a zero initial state for first-chunk prefill without
changing the AKA kernel ABI. Strict eligibility checks guard the verified
shape and metadata domain. Its private launch interface and profiler-visible
kernels start with `atrex_aka_`. This SM103 prefill path does not support CUDA
Graph capture and rejects it before metadata synchronization or kernel launch.

FlashAttention exposes the lazy public APIs `atrex.flash_attn_varlen_func` and
`atrex.can_use_flash_attn_varlen_func` with the vLLM-compatible varlen/paged
contract. Dispatch follows the input tensor's device: NVIDIA SM103 uses the
FA4 implementation, NVIDIA SM120 uses FA3, and ZW-M890P uses its FP8 FA2/FA3
implementation. On SM103, eligible BF16 q4 decode calls use the AKA fast path;
other supported short-Q/decode and prefill calls use the corresponding Atrex
kernels. Callers can check eligibility with `can_use_flash_attn_varlen_func`;
unsupported options are rejected rather than silently ignored.

This is a breaking change for callers of the former lower-level SM103 FA4
signature: use the varlen/paged arguments (`block_table`, `window_size`,
`s_aux`, `return_softmax_lse`) and its conditional return shape instead of
`page_table`, separate window bounds, `learnable_sink`, and `return_lse`.
Existing callers should migrate with the vLLM integration; retaining the
previous ATREX revision is the rollback path. The NVIDIA package requires
`flash-attn-4==4.0.0b19`; target environments should not install the classic
`flash-attn` distribution alongside it because both provide `flash_attn`.

ZW-M890P prefill supports contiguous D256 FP8 Q, paged K/V with page sizes
divisible by 64, and GQA ratios divisible by four (including heads 8/1 and 32/2).
It consumes optional FP32 `[batch,num_kv_heads]` descales on every call/replay.
The original 8/1, implicit-unit-scale kernel signature, body, and launcher are
retained from main (`43798a6f`). Other heads/scales use the independent
`atrex_ppu_attn_general` kernel in a separate CUDA translation unit. Both share
the same local layout/instruction helpers; this adds no third-party dependency.
Prefill uses `prep_kv`:
each call allocates packed K and V buffers of
`requests * num_kv_heads * ceil(max_seqlen_k / 64) * 16384` bytes each, plus small scale and
optional Split-KV partial buffers. For 64 requests at 16K KV length, packed
K/V total 512 MiB for one KV head. Account for this invocation-local scratch when sizing the
serving KV cache; it is not a persistent per-layer cache.

ZW-M890P direct-paged decode uses the same API with FA3, FP8 E4M3 inputs,
BF16 output, causal full-window attention, head dimension 256, GQA 8/16,
page sizes 64/128/256, and maximum query lengths 1 through 5. FP32 descales have
shape `[batch,num_kv_heads]` (omitted means one); LSE is not supported.
All decode paths accept `num_splits=0..32`, without a minimum KV length for
automatic splitting. Ineligible short-Q calls use the native
prefill path only when its contract also matches; otherwise they fail explicitly.
Decode reads `block_table` directly with invocation-local scratch, without KV
packing or persistent input caches. HGCC lazily compiles `native_decode.cu`
using PPU SDK headers, without PPU CuTe/ACTLASS. Prewarm each required path
before CUDA Graph capture. Target tests include native-FA3 correctness,
dispatch, graph replay, real kernel events, and `--decode-benchmark-output`.
ABBA collection and checking require `--benchmark-identity identity.json`.
Pin `baseline.ref` to `main`, full baseline/candidate `revision` values,
distinct absolute installed `package` directories, and `sources` SHA256 maps
keyed by repository-relative `python/atrex/...` and `src/ppu/...` files.
Also pin the operator test file's `benchmark_sha256` and `benchmark` settings
(`head_pairs`, `queries`, `batches`, `include_prefill`, `descales`). Collection
uses `--benchmark-phase A1|B1|B2|A2` and verifies the imported installation and
source bytes before GPU work. The checker enforces A1/A2=main, B1/B2=candidate,
the pinned case coverage, and the 5% latency / 1.5% repeat-spread gates:

```bash
python op_test/ppu/flash_attn/test_flash_attn_zwm890p.py \
  --decode-parity-inputs A1.json B1.json B2.json A2.json \
  --benchmark-identity identity.json
```

For the common main-compatible suite, collect each phase with
`--decode-heads 8/1 --decode-queries 1 2 3 4 5 --decode-batches 8 16 32
--decode-include-prefill --decode-descales none`. This covers 30 short-Q and
11 prefill cases using implicit unit descales. Shapes that main rejects are
capability extensions, not valid main-latency comparisons.

All applicable tests must pass on their target hardware before the operator is
accepted. See the
[`atrex-kernel-integration`](.agents/skills/atrex-kernel-integration/SKILL.md)
skill for the full placement, API, dispatch, compilation, kernel-name, and
validation requirements.

## Build

```bash
python3 -m pip install -r requirements-dev.txt
./build.sh
```

`build.sh` writes the wheel to `dist/`, resolves and installs its declared
runtime dependencies, then force-reinstalls only the freshly built ATREX wheel.
Set `PYTHON=/path/to/python` to select a different environment.

## License

ATREX is licensed under the Apache License 2.0. See [LICENSE](LICENSE) and
[NOTICE](NOTICE).
