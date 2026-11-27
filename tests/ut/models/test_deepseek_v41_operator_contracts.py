# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import ast
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_deepseek_v41_model_call_sites_use_compiled_operator_layouts():
    """Keep framework call sites aligned with the operator PR contracts."""
    source = ast.parse((REPO_ROOT / "vllm_ascend/attention/dsa_v41.py").read_text())
    calls = [
        node
        for node in ast.walk(source)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Attribute)
        and node.func.attr in ("npu_sparse_flash_mla", "npu_sparse_flash_mla_metadata")
    ]
    assert len(calls) == 2
    for call in calls:
        layouts = {kw.arg: ast.literal_eval(kw.value) for kw in call.keywords if kw.arg in ("layout_q", "layout_kv")}
        assert layouts == {"layout_q": "TND", "layout_kv": "PA_BBND"}

    source = ast.parse((REPO_ROOT / "vllm_ascend/models/deepseek_v41/indexer.py").read_text())
    common = next(
        node.value
        for node in ast.walk(source)
        if isinstance(node, ast.Assign)
        and any(isinstance(target, ast.Name) and target.id == "common" for target in node.targets)
    )
    assert isinstance(common, ast.Call)
    layouts = {kw.arg: ast.literal_eval(kw.value) for kw in common.keywords if kw.arg in ("layout_q", "layout_k")}
    assert layouts == {"layout_q": "TND", "layout_k": "PA_BBND"}


def test_indexer_fp8_layout_has_one_decision_point():
    """The cache spec and the indexer write path must ask the same contract.

    A split decision would allocate one plane layout and write it in another;
    the scatter converts values numerically instead of copying bytes, so the
    only symptom is drifting index selection.
    """
    cache_config = ast.parse((REPO_ROOT / "vllm_ascend/models/deepseek_v41/cache_config.py").read_text())
    cache_functions = {node.name: node for node in cache_config.body if isinstance(node, ast.FunctionDef)}
    assert "indexer_fp8_cache_contract" in cache_functions
    assert "indexer_fp8_cache_contract()" in ast.unparse(cache_functions["make_index_cache_spec"])

    indexer = ast.parse((REPO_ROOT / "vllm_ascend/models/deepseek_v41/indexer.py").read_text())
    indexer_class = next(
        node for node in indexer.body if isinstance(node, ast.ClassDef) and node.name == "DeepseekV41Indexer"
    )
    init = next(node for node in indexer_class.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    value = None
    for node in ast.walk(init):
        if not isinstance(node, ast.Assign):
            continue
        for target in node.targets:
            if isinstance(target, ast.Attribute) and target.attr == "fp8_cache":
                value = ast.unparse(node.value)
    assert value is not None, "DeepseekV41Indexer must publish fp8_cache for the write path"
    assert "indexer_fp8_cache_contract()" in value
    # Never re-derive the layout from the runtime adaptor here: the spec is a
    # static profile fact and the two must not be able to disagree.
    assert "get_dsa_indexer_quant_mode" not in value


def test_compressor_call_sites_use_compiled_modes():
    for filename, count in (("models/deepseek_v4/compressor.py", 1),):
        source = ast.parse((REPO_ROOT / "vllm_ascend" / filename).read_text())
        calls = [
            node
            for node in ast.walk(source)
            if isinstance(node, ast.Call) and ast.unparse(node.func) == "torch.ops._C_ascend.compressor"
        ]
        assert len(calls) == count
        for call in calls:
            modes = {
                kw.arg: ast.literal_eval(kw.value) for kw in call.keywords if kw.arg in ("rotary_mode", "cache_mode")
            }
            assert modes == {"rotary_mode": 2, "cache_mode": 1}
