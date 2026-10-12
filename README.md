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

ZW-M890P's existing 8-query-head / 1-KV-head path retains `prep_kv`: each call allocates packed K and V buffers of
`requests * ceil(max_seqlen_k / 64) * 16384` bytes each, plus small scale and
optional Split-KV partial buffers. For 64 requests at 16K KV length, packed
K/V total 512 MiB. Account for this invocation-local scratch when sizing the
serving KV cache; it is not a persistent per-layer cache.

ZW-M890P also supports direct-paged FP8 decode through the same public API.
Like the SM103 launch policy, short-Q calls (`1 <= max_seqlen_q <= 5`) first
try decode; only unsupported capabilities (`NotImplementedError`) fall back
to the original Atrex prefill implementation. Value/launch/compilation errors
are not swallowed. The dispatch layer does not encode a model's head count
or page size. Each implementation owns its side-effect-free eligibility.
The decode backend currently supports GQA=16, head dimension 256, page sizes
64/128/256, and maximum query length 1 or 4. Query/KV head counts and page size
are read from the caller's tensors and passed into the native entry; verified
head configurations include 16/1, 32/2, and 64/4. Other short-Q calls run the
existing 8/1-head prefill path when eligible; calls unsupported by both paths
fail explicitly. The existing prefill runtime and kernel source are unchanged.
Decode supports FA3, causal full-window attention, BF16 output, and per-request
`[batch,num_kv_heads]` FP32 descales (omitted scales mean one), but does not expose LSE.
Q1 accepts split counts 0 through 32; Q4 accepts automatic splitting for KV
lengths >=1024, or explicit `num_splits=1`. Unsupported decode calls fail
eligibility rather than being sent to the incompatible 8/1 prefill kernel.
Like the SM103 backend, PPU uses a shared decode runtime and shared kernel
sources, with independent Q1/Q4 compile-time configurations rather than
per-query source directories. `decode_build.py` compiles `decode_kernel.cu`
and `decode_scheduler.cu`; vendored dependencies live in `decode/vendor`.
The first eligible call lazily compiles its Q1 or Q4 specialization with HGCC.
Compatible short-Q calls also precompile the prefill fallback once per device;
prewarm each required path before CUDA Graph capture. Each invocation
owns and initializes its scheduling and partial-output scratch. KV pages are
read directly from `block_table`, without packing or a persistent input cache.

The PPU decode source derives from the verified `v83` experiment over Atrex
revision `c250c419db230fe282ef28ac770f762aeb359b15`. Only the compiler-resolved
transitive source dependencies are bundled, with shared headers deduplicated
and the original vendor notices/license retained. Profiler entry names are
prefixed with `atrex_ppu_decode_`; kernel math and native FP8/exp2 instructions
are preserved. The target test includes native-FA3 accuracy, metadata/input
mutation, dispatch, graph replay, actual profiler events, and an optional
`--decode-parity-output` migration gate against the frozen v83 binaries.

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
