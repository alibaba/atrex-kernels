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
