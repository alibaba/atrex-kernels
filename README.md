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

SM120 also supports HD128 BF16 and FP8 E4M3 noncausal paged attention. HD128
FP8 accepts page sizes 64/128/1664/3328 and BF16 output. Eligible HD256 E4M3
causal paged prefill uses TMA when the maximum query length exceeds four,
the page size is 64/128, and no LSE, window, sink, or softcap is requested.
The specialization accepts contiguous NHD and vLLM HND K/V views with FP32
`[batch,num_kv_heads]` descales, including broadcast views. Q1–4 decode keeps
the native packed-GQA/SplitKV implementation; other supported calls retain
the native compatibility path. Compilation remains lazy: prewarm each
specialization before CUDA Graph capture.

This is a breaking change for callers of the former lower-level SM103 FA4
signature: use the varlen/paged arguments (`block_table`, `window_size`,
`s_aux`, `return_softmax_lse`) and its conditional return shape instead of
`page_table`, separate window bounds, `learnable_sink`, and `return_lse`.
Existing callers should migrate with the vLLM integration; retaining the
previous ATREX revision is the rollback path. The wheel privately packages official
FlashAttention 4.0.0b20 helpers from commit
`73c992c8ca746935548df620ef3c1b6238fe6e68` (`fa4-v4.0.0.beta20`).
Builds initialize the pinned `third_party/flash-attention` submodule. Only the
wheel's copied imports use `atrex._vendor.flash_attn`; repository kernel sources
remain unchanged. ATREX keeps CuTeDSL 4.6.2 and Quack 0.6.4 and does not depend on
the public FA wheel, whose beta20 metadata pins a different compiler version.

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
before CUDA Graph capture. Target tests cover standalone FP32 correctness,
dispatch and fail-fast behavior, graph replay, scratch reuse, KV tail
sanitization, and real kernel events.

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
