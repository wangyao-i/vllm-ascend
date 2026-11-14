"""CANN 9.3.0 / Ascend960 的 acl DVPP 占位模块兼容补丁。

CANN 9.3.0 的 ``acl/__init__.py`` 在 Ascend960 上这样写：

    if _soc_name.startswith("Ascend960"):
        media = _DvppNotAvailable("acl.media")
        himpi = _DvppNotAvailable("acl.himpi")
        sys.modules["acl.media"] = media

``_DvppNotAvailable.__getattr__`` 对**任意**属性都抛 RuntimeError。于是任何
模块扫描代码（例如 ``hasattr(module, "__file__")``，torch_npu._inductor 导入
时就有一处）都会直接把整个进程打挂：

    RuntimeError: DVPP feature is not supported on this platform.

本补丁把双下划线属性的访问改成抛 AttributeError，让 ``hasattr`` / ``dir`` 正常
返回；真实使用 DVPP 的代码仍会拿到原来那条清晰的 RuntimeError。
"""

from __future__ import annotations

import contextlib
import sys
import types

_FAKE_MODULES = ("acl.media", "acl.himpi")


def apply() -> None:
    """幂等；acl 未导入时先导入，保证占位模块已就位。"""
    try:
        import acl  # noqa: F401
    except Exception:
        return

    for name in _FAKE_MODULES:
        mod = sys.modules.get(name)
        if mod is None or not isinstance(mod, types.ModuleType):
            continue
        cls = type(mod)
        if cls.__name__ != "_DvppNotAvailable" or getattr(cls, "_a6_shimmed", False):
            continue

        original = cls.__getattr__

        def _shimmed_getattr(self, attr, _original=original):
            if attr.startswith("__"):
                raise AttributeError(attr)
            return _original(self, attr)

        cls.__getattr__ = _shimmed_getattr
        cls._a6_shimmed = True

        # 常见探测属性直接落进 __dict__，省掉一次 __getattr__
        for attr, value in (("__file__", None), ("__path__", []), ("__all__", [])):
            with contextlib.suppress(Exception):
                mod.__dict__[attr] = value
