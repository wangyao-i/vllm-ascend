# SPDX-License-Identifier: Apache-2.0
"""交付包算子解析（单点）。

9.2 的交付包叫 ``npu_ops_transformer``、9.3 叫 ``cann_ops_transformer``；同一算子在两边
的命名也不一致（9.2 的 ``kv_compress_epilog`` 不带前缀，``quant_lightning_indexer`` 带
``npu_``；9.3 全部不带）。调用点统一写不带前缀的名字，由这里按「包 → 前缀」各试一次。

DSv4 路径里只有少数算子走交付包（compressor、QLI、hc_pre、kv_compress_epilog、kpool 的
key_pool/pool_key_indexer），其余全是树内 ``_C_ascend``；所以这里只做解析，不放任何 shim。
"""

from __future__ import annotations

import importlib

import torch

_PACKAGES = ("cann_ops_transformer", "npu_ops_transformer")


def delivered_op_or_none(name: str):
    """交付包里 ``name`` 的算子；找不到返回 None（不抛）。"""
    for package in _PACKAGES:
        try:
            importlib.import_module(package)
        except Exception:  # 包不存在 / ABI 不配套
            continue
        namespace = getattr(torch.ops, package, None)
        if namespace is None:
            continue
        for candidate in (name, f"npu_{name}"):
            fn = getattr(namespace, candidate, None)
            if fn is not None:
                return fn
    return None


def delivered_op(name: str):
    """交付包里 ``name`` 的算子；两边都找不到时抛 RuntimeError（带排查信息）。"""
    fn = delivered_op_or_none(name)
    if fn is not None:
        return fn
    tried = []
    for package in _PACKAGES:
        try:
            importlib.import_module(package)
            tried.append(f"{package}: no {name} / npu_{name}")
        except Exception as exc:
            tried.append(f"{package}: import failed ({type(exc).__name__}: {exc})")
    raise RuntimeError(f"delivered op {name!r} not found: {'; '.join(tried)}")
