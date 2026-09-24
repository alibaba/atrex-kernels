"""Require operator tests to use an installed ATREX wheel."""

from __future__ import annotations

from pathlib import Path

import atrex


REPOSITORY_PYTHON = (Path(__file__).resolve().parents[1] / "python").resolve()
ATREX_PACKAGE = Path(atrex.__file__).resolve()

if ATREX_PACKAGE.is_relative_to(REPOSITORY_PYTHON):
    raise RuntimeError(
        "operator tests require an installed ATREX wheel, but atrex was "
        f"imported from the source tree: {ATREX_PACKAGE}"
    )
