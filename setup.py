from __future__ import annotations

import shutil
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


ROOT = Path(__file__).resolve().parent
SOURCE_IGNORE = shutil.ignore_patterns(
    ".DS_Store",
    "__pycache__",
    "*.pyc",
    "*.pyo",
    ".pytest_cache",
)
CUTLASS_HEADER_DIRS = (
    Path("include"),
    Path("tools") / "util" / "include",
)


class build_py(_build_py):
    """Copy the target-organized source tree into the wheel without compiling it."""

    def run(self) -> None:
        super().run()

        repository_source = ROOT / "src"
        packaged_source = Path(self.build_lib) / "atrex" / "src"
        if packaged_source.exists():
            shutil.rmtree(packaged_source)
        shutil.copytree(repository_source, packaged_source, ignore=SOURCE_IGNORE)

        cutlass_root = ROOT / "third_party" / "cutlass"
        packaged_cutlass = Path(self.build_lib) / "atrex" / "third_party" / "cutlass"
        missing = [
            str(cutlass_root / relative_dir)
            for relative_dir in CUTLASS_HEADER_DIRS
            if not (cutlass_root / relative_dir).is_dir()
        ]
        if missing:
            raise RuntimeError(
                "CUTLASS headers are required for the NVFP4 fused MoE wheel. "
                "Run `git submodule update --init --recursive`. Missing: "
                + ", ".join(missing)
            )
        if packaged_cutlass.exists():
            shutil.rmtree(packaged_cutlass)
        for relative_dir in CUTLASS_HEADER_DIRS:
            shutil.copytree(
                cutlass_root / relative_dir,
                packaged_cutlass / relative_dir,
                ignore=SOURCE_IGNORE,
            )


setup(cmdclass={"build_py": build_py})
