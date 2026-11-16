# SPDX-License-Identifier: Apache-2.0
"""交付包算子解析（单点）。

9.2 的交付包叫 ``npu_ops_transformer``、9.3 叫 ``cann_ops_transformer``；同一算子在两边
的命名也不一致（9.2 的 ``kv_compress_epilog`` 不带前缀，``quant_lightning_indexer`` 带
``npu_``；9.3 全部不带）。调用点统一写不带前缀的名字，由这里按「包 → 前缀」各试一次。

DSv4 路径里只有少数算子走交付包（compressor、QLI、hc_pre、kv_compress_epilog、kpool 的
key_pool/pool_key_indexer），其余全是树内 ``_C_ascend``；所以这里只做解析，不放任何 shim。

A6 图模式补充（ACLGraph 捕获三禁之「不 import」）：交付包的首次 import 会触发
``ops/csrc`` JIT 编译，出现在 forward 里会被 Dynamo 判成 graph break（gb0007），
FULL_DECODE_ONLY 捕获随之失败。因此这里把 import 提到**模块导入期**，运行期只做
``torch.ops`` 属性查找 + 句柄缓存。
"""

from __future__ import annotations

import importlib
from typing import Any

import torch

_PACKAGES = ("cann_ops_transformer", "npu_ops_transformer")

# 模块导入期完成交付包 import（只做一次）
_IMPORT_OK: dict[str, bool] = {}
for _pkg in _PACKAGES:
    try:
        importlib.import_module(_pkg)
        _IMPORT_OK[_pkg] = True
    except Exception:  # 包不存在 / ABI 不配套
        _IMPORT_OK[_pkg] = False

# 算子句柄缓存：解析结果在整个进程内复用，避免捕获区里重复 import / getattr
_CACHE: dict[str, Any] = {}


def delivered_op_or_none(name: str):
    """交付包里 ``name`` 的算子；找不到返回 None（不抛）。"""
    cached = _CACHE.get(name)
    if cached is not None:
        return cached

    for package, ok in _IMPORT_OK.items():
        if not ok:
            continue
        namespace = getattr(torch.ops, package, None)
        if namespace is None:
            continue
        for candidate in (name, f"npu_{name}"):
            fn = getattr(namespace, candidate, None)
            if fn is not None:
                _CACHE[name] = fn
                return fn
    return None


def delivered_op(name: str):
    """交付包里 ``name`` 的算子；两边都找不到时抛 RuntimeError（带排查信息）。"""
    fn = delivered_op_or_none(name)
    if fn is not None:
        return fn
    tried = []
    for package, ok in _IMPORT_OK.items():
        tried.append(f"{package}: no {name} / npu_{name}" if ok else f"{package}: import failed")
    raise RuntimeError(f"delivered op {name!r} not found: {'; '.join(tried)}")


def prewarm(names: tuple[str, ...] | list[str] | None = None):
    """图捕获前预热缓存（可选）。"""
    for n in names or ():
        delivered_op_or_none(n)
