# ATREX operator tests

Functional tests are organized by implemented hardware target and operator:

```text
op_test/<vendor>/<operator>/test_<operator>_<arch>.py
```

For example, `op_test/nvidia/chunk_gdn/test_chunk_gdn_sm120.py`. Keep
correctness, performance bounds, and kernel-name profiling in that one file.
The vendor fixture selects the target encoded in the filename; unsupported
cases must not be generated. Profile the real operator call rather than a
synthetic probe kernel.

Every operator integration must add or update its applicable tests in the same
change. All applicable tests must pass on each claimed target. Collection and
import must not compile a kernel; compile or prewarm after target selection.

Build and install the wheel once for each revision and test environment, then
run every applicable operator test against that installation in one pytest
session:

```bash
./build.sh
python3 -m pytest -q \
  op_test/nvidia/operator_a/test_operator_a_sm120.py \
  op_test/nvidia/operator_b/test_operator_b_sm120.py
```

Operator tests never install ATREX themselves and never fall back to the source
tree. Test collection fails when ATREX is missing or imported from
`python/atrex`.
