"""Public operator implementations and dispatch modules."""

import sys

# Backward-compatible alias for the pre-refactor module path: the semantic
# API refactor (e0b796bb) renamed ``atrex.api.chunk_gdn_cutedsl`` to
# ``atrex.api.chunk_gdn``. Both names must resolve to the same module for
# every import form — downstream integrations (e.g. the vLLM GDN prefill
# routing) probe the pre-refactor path via ``importlib.import_module``. The
# sys.modules entry covers import_module/import statements; the package
# attribute covers ``from atrex.api import chunk_gdn_cutedsl`` and attribute
# access.
try:
    from atrex.api import chunk_gdn as _chunk_gdn
except ImportError:
    pass  # torch-less environments: leave the alias unregistered.
else:
    sys.modules.setdefault("atrex.api.chunk_gdn_cutedsl", _chunk_gdn)
    chunk_gdn_cutedsl = _chunk_gdn
    del _chunk_gdn
