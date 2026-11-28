# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# ruff: noqa: E402
"""The PyTorch Engram gate must reproduce the gate it replaces on A6."""

import pytest

torch = pytest.importorskip("torch")

from vllm_ascend.models.deepseek_v41.engram.common import engram_gate
from vllm_ascend.ops.triton.engram_gate_torch import fused_engram_gate_torch

TOKENS = 3
BRANCHES = 4
DIM = 8
EPS = 1e-6


def _inputs():
    generator = torch.Generator().manual_seed(0)
    hidden = torch.randn(TOKENS, BRANCHES, DIM, dtype=torch.bfloat16, generator=generator)
    kv = torch.randn(TOKENS, (BRANCHES + 1) * DIM, dtype=torch.bfloat16, generator=generator)
    q_weight = torch.randn(BRANCHES, DIM, dtype=torch.bfloat16, generator=generator)
    k_weight = torch.randn(BRANCHES, DIM, dtype=torch.bfloat16, generator=generator)
    return hidden, kv, q_weight, k_weight


def test_matches_the_unfused_gate_without_rotation():
    hidden, kv, q_weight, k_weight = _inputs()
    mask = torch.ones(TOKENS, dtype=torch.bool)
    key = kv[:, : BRANCHES * DIM].view(TOKENS, BRANCHES, DIM)
    value = kv[:, BRANCHES * DIM :]

    expected = engram_gate(hidden, key, value, q_weight.float() * k_weight.float(), None, mask, EPS)
    actual = fused_engram_gate_torch(hidden, kv, q_weight, k_weight, mask, EPS)

    # Both round their FP32 result back to BF16, so only reduction-order noise
    # can differ.
    torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)


def test_inactive_tokens_keep_the_residual():
    hidden, kv, q_weight, k_weight = _inputs()
    mask = torch.zeros(TOKENS, dtype=torch.bool)
    mask[0] = True

    output = fused_engram_gate_torch(hidden, kv, q_weight, k_weight, mask, EPS)

    torch.testing.assert_close(output[1:], hidden[1:])
    assert not torch.equal(output[0], hidden[0])
