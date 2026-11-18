# SPDX-License-Identifier: Apache-2.0
"""Exercise dispatch bodies on CPU without importing the NPU runtime.

The operator leaves are mocked. These tests cover routing and Python ABI
adaptation, while native kernels require separate hardware qualification.
"""

import __future__

import ast
import runpy
import sys
import types
from dataclasses import replace
from pathlib import Path
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]


def _function(path, name, namespace, owner=None):
    tree = ast.parse((ROOT / path).read_text(encoding="utf-8"))
    body = (
        tree.body
        if owner is None
        else next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner).body
    )
    function = next(node for node in body if isinstance(node, ast.FunctionDef) and node.name == name)
    function.decorator_list = []
    module = ast.Module(body=[function], type_ignores=[])
    exec(compile(module, path, "exec", flags=__future__.annotations.compiler_flag), namespace)
    return namespace[name]


@pytest.fixture
def profiles(monkeypatch):
    hardware = runpy.run_path(str(ROOT / "vllm_ascend/device/hardware.py"))
    path = ROOT / "vllm_ascend/device/hardware_profile.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    # Only runtime device discovery is omitted. All profile definitions run.
    tree.body = [
        node
        for node in tree.body
        if not (isinstance(node, ast.ImportFrom) and (node.module or "").startswith("vllm_ascend"))
        and not (
            isinstance(node, ast.Assign)
            and any(
                isinstance(target, ast.Name) and target.id == "_CURRENT_HARDWARE_PROFILE" for target in node.targets
            )
        )
    ]
    module = types.ModuleType("_a6_test_hardware_profile")
    monkeypatch.setitem(sys.modules, module.__name__, module)
    module.AscendDeviceType = hardware["AscendDeviceType"]
    exec(compile(tree, str(path), "exec"), module.__dict__)
    return module


@pytest.mark.parametrize("device", ["A2", "A3", "A5", "A6", "_310P"])
def test_complete_capability_matrix(profiles, device):
    namespace = dict(profiles.__dict__)
    namespace.update(pytest=pytest, torch=torch)
    tree = ast.parse((ROOT / "tests/ut/device/test_hardware_profile.py").read_text(encoding="utf-8"))
    tree.body = [node for node in tree.body if not isinstance(node, (ast.Import, ast.ImportFrom))]
    exec(compile(tree, "test_hardware_profile.py", "exec", flags=__future__.annotations.compiler_flag), namespace)
    namespace["test_hardware_profile_capability_matrix"](getattr(profiles.AscendDeviceType, device))


def _namespace(profiles, device):
    profile = profiles.get_hardware_profile(getattr(profiles.AscendDeviceType, device))
    return {
        "torch": types.SimpleNamespace(ops=types.SimpleNamespace(_C_ascend=Mock())),
        "torch_npu": Mock(),
        "HardwareCapability": profiles.HardwareCapability,
        "get_current_hardware_profile": lambda: profile,
        "delivered_op": Mock(),
        "_delivered_compressor": Mock(),
    }


@pytest.mark.parametrize("owner", ["BaseDeviceAdaptor", "A6DeviceAdaptor"])
def test_indexer_operator_and_metadata_contract(profiles, owner):
    namespace = _namespace(profiles, "A6" if owner == "A6DeviceAdaptor" else "A5")
    path = "vllm_ascend/device/device_op.py"
    compute = _function(path, "get_dsa_lightning_indexer_op", namespace, owner)
    metadata = _function(path, "get_dsa_lightning_indexer_metadata_op", namespace, owner)
    if owner == "BaseDeviceAdaptor":
        assert compute() is namespace["torch"].ops._C_ascend.npu_quant_lightning_indexer_v2
        metadata(device="npu", num_heads_q=64)
        namespace["torch"].ops._C_ascend.npu_quant_lightning_indexer_v2_metadata.assert_called_once_with(
            device="npu", num_heads_q=64
        )
    else:
        compute()
        namespace["delivered_op"].assert_called_with("quant_lightning_indexer")
        metadata(device="npu", num_heads_q=64)
        namespace["delivered_op"].assert_called_with("quant_lightning_indexer_metadata")
        namespace["delivered_op"].return_value.assert_called_once_with(num_heads_q=64)
        namespace["delivered_op"].side_effect = RuntimeError("delivered op missing")
        with pytest.raises(RuntimeError, match="delivered op missing"):
            compute()
        namespace["torch"].ops._C_ascend.npu_quant_lightning_indexer_v2.assert_not_called()
        namespace["torch"].ops._C_ascend.npu_quant_lightning_indexer_v2_metadata.assert_not_called()


@pytest.mark.parametrize("device", ["A2", "A3", "A5", "A6"])
@pytest.mark.parametrize("in_tree", [False, True])
def test_compressor_routes_by_profile(profiles, monkeypatch, device, in_tree):
    monkeypatch.delenv("A6_IN_TREE_COMPRESSOR", raising=False)
    if in_tree:
        monkeypatch.setenv("A6_IN_TREE_COMPRESSOR", "1")
    namespace = _namespace(profiles, device)
    operation = _function("vllm_ascend/models/deepseek_v4/compressor.py", "_compressor_op", namespace)
    inputs = [object() for _ in range(8)]
    operation(*inputs, coff=2, rotary_mode=2)
    native = namespace["torch"].ops._C_ascend.compressor
    delivered = namespace["_delivered_compressor"]
    if device == "A6" and not in_tree:
        delivered.assert_called_once_with(
            *inputs[:5], norm_weight=inputs[5], rope_sin=inputs[6], rope_cos=inputs[7], coff=2, rotary_mode=2
        )
        native.assert_not_called()
    else:
        native.assert_called_once_with(*inputs, coff=2, rotary_mode=2)
        delivered.assert_not_called()


@pytest.mark.parametrize("device", ["A2", "A3", "A5", "A6"])
def test_mhc_pre_post_shapes_and_epsilon_order(profiles, device):
    namespace = _namespace(profiles, device)
    path = "vllm_ascend/models/deepseek_v4/model.py"
    pre = _function(path, "hc_pre", namespace, "DeepseekV4DecoderLayer")
    post = _function(path, "hc_post", namespace, "DeepseekV4DecoderLayer")
    owner = types.SimpleNamespace(hc_mult=4, hc_sinkhorn_iters=20, norm_eps=1e-6, hc_eps=1e-5)
    residual, x = torch.zeros(2, 4, 8), torch.zeros(2, 8)
    weights = [object() for _ in range(3)]
    native = namespace["torch"].ops._C_ascend
    if device == "A6":
        namespace["delivered_op"].return_value.return_value = (
            x.unsqueeze(1),
            torch.zeros(2, 1, 4),
            torch.zeros(2, 1, 16),
        )
        hidden, gates, comb = pre(owner, residual, *weights)
        call = namespace["delivered_op"].return_value.call_args.args
        assert call[0].shape == (2, 1, 4, 8)
        assert call[1:] == (*weights, 4, 20, 1e-5, 1e-6, False)
        assert (hidden.shape, gates.shape, comb.shape) == ((2, 8), (2, 4), (2, 4, 4))
        post(owner, hidden, residual, gates, comb)
        namespace["torch_npu"].npu_mhc_post.assert_called_once_with(residual, comb, hidden, gates)
        native.npu_hc_pre_v2.assert_not_called()
        native.npu_hc_post.assert_not_called()
    else:
        result = pre(owner, residual, *weights)
        assert result is native.npu_hc_pre_v2.return_value
        native.npu_hc_pre_v2.assert_called_once_with(residual, *weights, 4, 20, 1e-6, 1e-5)
        gates, comb = torch.zeros(2, 4), torch.zeros(2, 4, 4)
        native.npu_hc_post.return_value = residual.unsqueeze(0)
        assert post(owner, x, residual, gates, comb).shape == residual.shape
        assert [value.shape for value in native.npu_hc_post.call_args.args] == [
            (1, 2, 8),
            (1, 2, 4, 8),
            (1, 2, 4),
            (1, 2, 4, 4),
        ]
        namespace["delivered_op"].assert_not_called()


def test_gdn_native_probe_is_independent_of_fla(profiles):
    namespace = _namespace(profiles, "A5")
    profile = profiles.get_hardware_profile(profiles.AscendDeviceType.A5)
    profile = replace(profile, capabilities=profile.capabilities - {profiles.HardwareCapability.FLA_GDN_PREFILL})
    namespace["get_current_hardware_profile"] = lambda: profile
    namespace["torch"].npu = Mock()
    namespace["torch"].zeros = Mock()
    namespace["torch"].full = Mock()
    namespace["torch"].tensor = Mock()
    namespace["torch"].bfloat16 = torch.bfloat16
    namespace["torch"].float32 = torch.float32
    namespace["torch"].int32 = torch.int32
    probe = _function("vllm_ascend/ops/gdn.py", "_probe_fused_chunk", namespace, "AscendGatedDeltaNetAttention")
    assert probe(types.SimpleNamespace(_fused_chunk_available=None))
    namespace["torch_npu"].npu_chunk_gated_delta_rule.assert_called_once()
    profile = profiles.get_hardware_profile(profiles.AscendDeviceType.A6)
    assert not probe(types.SimpleNamespace(_fused_chunk_available=None))
    assert namespace["torch_npu"].npu_chunk_gated_delta_rule.call_count == 1
