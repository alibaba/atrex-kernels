from types import SimpleNamespace

from atrex.utils.device_target import detect_device_target


class FakeCuda:
    def __init__(self, name, capability=(8, 9), gcn_arch_name=""):
        self._properties = SimpleNamespace(name=name, gcnArchName=gcn_arch_name)
        self._capability = capability

    def get_device_properties(self, index):
        return self._properties

    def get_device_capability(self, index):
        return self._capability


def install_fake_torch(monkeypatch, *, name, cuda=None, hip=None, gfx=""):
    fake_torch = SimpleNamespace(
        cuda=FakeCuda(name, gcn_arch_name=gfx),
        version=SimpleNamespace(cuda=cuda, hip=hip),
    )
    monkeypatch.setitem(__import__("sys").modules, "torch", fake_torch)


def test_rocm_is_classified_before_cuda_compatibility(monkeypatch):
    install_fake_torch(
        monkeypatch,
        name="AMD Instinct MI308X",
        hip="7.2.26015",
        gfx="gfx942:sramecc+:xnack-",
    )

    target = detect_device_target(0)

    assert (target.family, target.arch) == ("amd", "gfx942")


def test_ppu_sm89_compatibility_is_not_classified_as_nvidia(monkeypatch):
    install_fake_torch(monkeypatch, name="ZW-M890P", cuda="13.0")

    target = detect_device_target(0)

    assert (target.family, target.arch) == ("alibaba_ppu", "zwm890p")


def test_real_nvidia_sm89_remains_nvidia(monkeypatch):
    install_fake_torch(monkeypatch, name="NVIDIA L40S", cuda="12.8")

    target = detect_device_target(0)

    assert (target.family, target.arch) == ("nvidia", "sm89")
