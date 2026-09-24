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
op_test/nvidia/chunk_gdn/test_chunk_gdn_sm120.py
op_test/nvidia/flash_attn/test_flash_attn_sm103.py
```

The SM103 FlashAttention path exposes the lower-level FA4 ABI used by vLLM and
currently dispatches only the AKA BF16 q4 specialization for Qwen3.7-Max TP4:
16 query heads, one KV head, head dimension 256, page size 128, and batch
sizes 16 through 28. Unsupported calls are rejected before launch so another
backend can be added as an explicit fallback without changing the public API.

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

`build.sh` writes the wheel to `dist/` and force-reinstalls it into the same
Python environment used for the build. Set `PYTHON=/path/to/python` to select a
different environment.

## License

ATREX is licensed under the Apache License 2.0. See [LICENSE](LICENSE) and
[NOTICE](NOTICE).
