from __future__ import annotations

import ast
import io
import shutil
import tokenize
from pathlib import Path

from setuptools import setup
from setuptools.command.build_py import build_py as _build_py


ROOT = Path(__file__).resolve().parent
SOURCE_IGNORE = shutil.ignore_patterns(
    ".DS_Store",
    "._*",
    "__pycache__",
    "*.pyc",
    "*.pyo",
    ".pytest_cache",
    "pyproject.toml",
)
CUTLASS_HEADER_DIRS = (
    Path("include"),
    Path("tools") / "util" / "include",
)
FA_ROOT = ROOT / "third_party" / "flash-attention"
FA_NAMESPACE = "atrex._vendor.flash_attn"


def _private_fa_imports(source: str, filename: str) -> str:
    """Adapt static FA imports only in wheel staging, preserving source lines."""
    tree = ast.parse(source, filename=filename)
    targets = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom):
            if node.level == 0 and node.module and (
                node.module == "flash_attn.cute"
                or node.module.startswith("flash_attn.cute.")
            ):
                targets.append(node)
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name == "flash_attn.cute" or alias.name.startswith("flash_attn.cute."):
                    if alias.asname is None:
                        raise ValueError(f"{filename}:{node.lineno}: FA dotted import needs an alias")
                    targets.append(alias)
    if not targets:
        return source

    lines = source.splitlines(keepends=True)
    starts = [0]
    for line in lines:
        starts.append(starts[-1] + len(line))

    def offset(line, byte_column):
        return starts[line - 1] + len(lines[line - 1].encode("utf-8")[:byte_column].decode("utf-8"))

    roots = [
        starts[token.start[0] - 1] + token.start[1]
        for token in tokenize.generate_tokens(io.StringIO(source).readline)
        if token.type == tokenize.NAME and token.string == "flash_attn"
    ]
    replacements = []
    for node in targets:
        begin = offset(node.lineno, node.col_offset)
        end = offset(node.end_lineno, node.end_col_offset)
        replacements.append(next(position for position in roots if begin <= position < end))
    for position in sorted(replacements, reverse=True):
        source = source[:position] + FA_NAMESPACE + source[position + len("flash_attn"):]
    return source


def _copy_private_fa(build_lib: Path) -> None:
    """Ship the pinned beta20 helpers without installing FA's compiler pins."""
    if not (FA_ROOT / "flash_attn/cute").is_dir() or not (FA_ROOT / "LICENSE").is_file():
        raise RuntimeError(
            "FlashAttention helpers are required. Run "
            "`git submodule update --init -- third_party/flash-attention`."
        )
    vendor = build_lib / "atrex" / "_vendor"
    private_fa = vendor / "flash_attn"
    if private_fa.exists():
        shutil.rmtree(private_fa)
    private_fa.mkdir(parents=True)
    (vendor / "__init__.py").write_text('"""ATREX-private third-party helpers."""\n', encoding="utf-8")
    (private_fa / "__init__.py").write_text(
        '"""Private FlashAttention helpers; not a public compatibility API."""\n'
        '__version__ = "4.0.0b20"\n'
        '__source_commit__ = "73c992c8ca746935548df620ef3c1b6238fe6e68"\n',
        encoding="utf-8",
    )
    shutil.copytree(FA_ROOT / "flash_attn/cute", private_fa / "cute", ignore=SOURCE_IGNORE)
    for name in ("LICENSE", "NOTICE"):
        if (FA_ROOT / name).is_file():
            shutil.copy2(FA_ROOT / name, private_fa / name)
    for directory in (build_lib / "atrex/src", private_fa / "cute"):
        for path in sorted(directory.rglob("*.py")):
            original = path.read_bytes().decode("utf-8")
            adapted = _private_fa_imports(original, str(path))
            if adapted != original:
                path.write_bytes(adapted.encode("utf-8"))
            if directory == private_fa / "cute":
                for node in ast.walk(ast.parse(adapted, filename=str(path))):
                    if isinstance(node, ast.Import):
                        modules = [alias.name for alias in node.names]
                    elif isinstance(node, ast.ImportFrom) and node.level == 0:
                        modules = [node.module or ""]
                    else:
                        continue
                    if any(name == "flash_attn" or name.startswith("flash_attn.") for name in modules):
                        raise RuntimeError(f"Public FA import remains in private helper {path}:{node.lineno}")


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
                "Run `git submodule update --init -- third_party/cutlass`. Missing: "
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

        _copy_private_fa(Path(self.build_lib))

setup(cmdclass={"build_py": build_py})
