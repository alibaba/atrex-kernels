"""Eight-case paired wheel regression, invoked through the target test module."""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
from pathlib import Path
import statistics
import sys

import torch

from qsa_contract import (
    CASES,
    _inputs,
    _refresh,
    capture,
    error,
    launch_options,
    oracle,
)


class WheelAPI:
    """Isolate installed module namespaces; activate outside measured calls."""

    def __init__(self, root):
        self.root = Path(root).resolve()
        self.modules = {}

    def activate(self):
        for name in list(sys.modules):
            if name == "atrex" or name.startswith("atrex."):
                del sys.modules[name]
        sys.modules.update(self.modules)

    def __call__(self, **kwargs):
        self.activate()
        sys.path.insert(0, str(self.root))
        try:
            api = importlib.import_module("atrex")
            assert Path(api.__file__).resolve().is_relative_to(self.root), api.__file__
            return api.qsa_compact(**kwargs)
        finally:
            self.modules = {
                n: m
                for n, m in sys.modules.items()
                if n == "atrex" or n.startswith("atrex.")
            }
            sys.path.remove(str(self.root))


def identity(root):
    root = Path(root).resolve()
    paths = [
        root / "atrex/api/qsa_compact.py",
        *sorted((root / "atrex/src/nvidia/qsa").rglob("*.py")),
    ]
    assert len(paths) > 1 and all(p.is_file() for p in paths), root
    hashes = {
        str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in paths
    }
    return dict(
        root=str(root),
        source_digest=hashlib.sha256(
            json.dumps(hashes, sort_keys=True).encode()
        ).hexdigest(),
    )


def measure(apis, graphs, functions, queries, flush):
    timings = {n: {} for n in apis}
    for cache in ("warm", "cold_l2") if queries else ("eager",):
        samples = {n: [] for n in apis}
        for repeat in range(5):
            for sample in range(10):
                order = (
                    list(apis) if (repeat + sample) % 2 == 0 else list(reversed(apis))
                )
                for name in order:
                    if queries:
                        if cache == "cold_l2":
                            flush.zero_()
                            torch.cuda.synchronize()
                        graph, start, end = graphs[name]
                        graph.replay()
                        end.synchronize()
                    else:
                        apis[name].activate()
                        start, end = (
                            torch.cuda.Event(enable_timing=True) for _ in range(2)
                        )
                        start.record()
                        functions[name]()
                        end.record()
                        end.synchronize()
                    samples[name].append(start.elapsed_time(end) * 1000)
        for name in apis:
            timings[name][cache] = dict(
                median_us=statistics.median(samples[name]), samples_us=samples[name]
            )
    regression = {
        k: 100
        * (timings["current"][k]["median_us"] / timings["baseline"][k]["median_us"] - 1)
        for k in timings["current"]
    }
    return dict(timings=timings, regression_pct=regression)


def main(architecture):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--regression", action="store_true", required=True)
    parser.add_argument("--current-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--cases", default="all")
    parser.add_argument("--reverse-capture", action="store_true")
    args = parser.parse_args()
    ids = (
        list(range(len(CASES)))
        if args.cases == "all"
        else [int(x) for x in args.cases.split(",")]
    )
    if (
        not ids
        or len(set(ids)) != len(ids)
        or any(i < 0 or i >= len(CASES) for i in ids)
    ):
        parser.error("--cases must contain unique indices from 0 to 7")
    torch.set_grad_enabled(False)
    torch.backends.cuda.matmul.allow_tf32 = False
    assert (
        torch.cuda.get_device_capability()
        == {"sm100": (10, 0), "sm103": (10, 3)}[architecture]
    )
    roots = dict(baseline=args.baseline_root, current=args.current_root)
    report = dict(
        architecture=architecture,
        device=str(torch.cuda.get_device_properties(0)),
        packages={n: identity(root) for n, root in roots.items()},
        versions={
            n: importlib.metadata.version(n)
            for n in ("torch", "triton", "nvidia-cutlass-dsl", "apache-tvm-ffi")
        },
        measurement=dict(
            samples=50,
            order="per-sample alternating",
            cold_flush_bytes=256 * 1024 * 1024,
            reverse_capture=args.reverse_capture,
        ),
        cases=[],
        complete=False,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    failures = []
    flush = torch.empty(256 * 1024 * 1024, device="cuda", dtype=torch.uint8)
    save()
    for case in ids:
        page, dtype, mode = CASES[case]
        print("CASE_START", case, page, dtype, mode, flush=True)
        row = dict(
            case=case, page=page, dtype=dtype, mode=mode, accuracy=[], complete=False
        )
        report["cases"].append(row)
        graphs = {}
        try:
            data, scales, queries = _inputs(page, dtype, mode)
            _refresh(data, page, queries, 0)
            kwargs = launch_options(scales)
            apis = {n: WheelAPI(root) for n, root in roots.items()}
            outputs = {n: torch.empty_like(data["out"]) for n in apis}
            functions = {}
            for name in reversed(apis) if args.reverse_capture else apis:
                api = apis[name]
                call = dict(data, out=outputs[name], **kwargs)
                api(**call)
                public = api.modules["atrex"].qsa_compact
                fn = lambda public=public, call=call: public(**call)
                functions[name] = fn
                if queries:
                    graphs[name] = capture(fn)
                else:
                    for _ in range(3):
                        fn()
                torch.cuda.synchronize()
            for revision in range(2):
                _refresh(data, page, queries, revision)
                expected = oracle(data, scales)
                metrics = {}
                for name, api in apis.items():
                    if queries:
                        graphs[name][0].replay()
                    else:
                        api(**dict(data, out=outputs[name], **kwargs))
                    torch.cuda.synchronize()
                    metrics[name] = error(outputs[name], expected)
                metrics["new_vs_old"] = error(outputs["current"], outputs["baseline"])
                metrics["bitwise_equal"] = torch.equal(
                    outputs["current"], outputs["baseline"]
                )
                row["accuracy"].append(metrics)
                save()
                assert all(metrics[n]["finite"] for n in apis), metrics
                assert metrics["new_vs_old"]["passed"], metrics
                if metrics["baseline"]["passed"]:
                    assert metrics["current"]["passed"], metrics
                assert (
                    metrics["current"]["relative_l2"]
                    <= metrics["baseline"]["relative_l2"] * 1.01 + 1e-7
                ), metrics
            row.update(measure(apis, graphs, functions, queries, flush))
            failures.extend(
                (case, metric, value)
                for metric, value in row["regression_pct"].items()
                if value > 5
            )
            row["complete"] = True
            print("CASE_COMPLETE", case, row["regression_pct"], flush=True)
        except Exception as exc:
            row["error"] = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            for graph, _, _ in graphs.values():
                graph.reset()
            save()
    report.update(
        complete=True, performance_pass=not failures, regressions_over5pct=failures
    )
    save()
    assert not failures, failures
