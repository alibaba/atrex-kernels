from __future__ import annotations

from types import SimpleNamespace

import atrex


def test_lazy_api_loads_on_first_call_and_caches_real_function(monkeypatch):
    imports = []

    def implementation(value, *, increment=1):
        return value + increment

    module = SimpleNamespace(implementation=implementation)

    def fake_import_module(module_name):
        imports.append(module_name)
        return module

    monkeypatch.setattr(atrex, "_import_module", fake_import_module)
    proxy = atrex._lazy_import_and_call(
        "implementation",
        "atrex.api.example",
        public_name="example",
    )
    monkeypatch.setattr(atrex, "example", proxy, raising=False)

    assert imports == []
    assert proxy.__name__ == "example"
    assert atrex.example(3, increment=2) == 5
    assert imports == ["atrex.api.example"]
    assert atrex.example is implementation

    assert atrex.example(4) == 5
    assert imports == ["atrex.api.example"]


def test_lazy_api_does_not_cache_failed_resolution(monkeypatch):
    proxy = atrex._lazy_import_and_call("missing", "atrex.api.missing")
    monkeypatch.setattr(atrex, "missing", proxy, raising=False)

    def fail_import(module_name):
        raise ModuleNotFoundError(module_name)

    monkeypatch.setattr(atrex, "_import_module", fail_import)

    try:
        atrex.missing()
    except ModuleNotFoundError:
        pass
    else:
        raise AssertionError("missing implementation import unexpectedly succeeded")

    assert atrex.missing is proxy
