import sys
from types import ModuleType

import pytest

from vllm_ascend.patch.platform.patch_acl_dvpp_shim import apply


def test_unavailable_dvpp_modules_allow_introspection(monkeypatch):
    class _DvppNotAvailable(ModuleType):
        def __getattr__(self, name):
            raise RuntimeError("DVPP feature is not supported on this platform.")

    acl = ModuleType("acl")
    acl.media = _DvppNotAvailable("acl.media")
    acl.himpi = _DvppNotAvailable("acl.himpi")
    monkeypatch.setitem(sys.modules, "acl", acl)
    monkeypatch.setitem(sys.modules, "acl.media", acl.media)
    monkeypatch.setitem(sys.modules, "acl.himpi", acl.himpi)

    apply()
    apply()

    assert acl.media.__file__ is None
    assert not hasattr(acl.himpi, "__missing_probe__")
    with pytest.raises(RuntimeError, match="DVPP feature is not supported"):
        acl.media.create_channel


def test_available_media_module_is_unchanged(monkeypatch):
    acl = ModuleType("acl")
    acl.media = ModuleType("acl.media")
    acl.himpi = ModuleType("acl.himpi")
    monkeypatch.setitem(sys.modules, "acl", acl)
    monkeypatch.setitem(sys.modules, "acl.media", acl.media)
    monkeypatch.setitem(sys.modules, "acl.himpi", acl.himpi)

    apply()

    assert "__file__" not in vars(acl.media)
    assert not hasattr(acl.media, "__missing_probe__")
