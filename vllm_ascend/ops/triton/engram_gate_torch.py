# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""PyTorch implementation of the fused Engram gate.

It replaces ``_engram_gate_kernel`` on stacks where the Triton kernel cannot be
compiled (A6's bishengir rejects it) while keeping the same formula and the same
FP32 operation order as far as PyTorch allows:

* ``hidden`` and ``key`` are converted to FP32 and normalized with
  ``sum(x * x) / DIM + eps`` before ``rsqrt``, then both factors multiply;
* ``dot`` is ``sum((h * (qw * kw)) * key) * rstd * DIM**-0.5``;
* ``magnitude`` is ``sqrt(max(|dot|, 1e-6))`` and the sign comes from the sign
  bit of ``dot``, so negative zero gates like the kernel;
* inactive tokens gate to exactly zero, and the value term is broadcast over
  the branches.
"""

import torch


def fused_engram_gate_torch(hidden, kv, q_weight, k_weight, token_mask, eps):
    """Return the gated hidden states for the layout ``fused_engram_gate`` takes."""
    tokens, branches, dim = hidden.shape
    h = hidden.float()
    k = kv[:, : branches * dim].reshape(tokens, branches, dim).float()
    value = kv[:, branches * dim : (branches + 1) * dim].float()
    qw = q_weight.float().reshape(branches, dim)
    kw = k_weight.float().reshape(branches, dim)
    rstd = torch.rsqrt((h * h).sum(-1) / dim + eps)
    rstd = rstd * torch.rsqrt((k * k).sum(-1) / dim + eps)
    dot = ((h * (qw * kw).unsqueeze(0)) * k).sum(-1) * rstd * (dim**-0.5)
    magnitude = torch.sqrt(dot.abs().clamp_min(1e-6))
    gate = torch.sigmoid(torch.where(torch.signbit(dot), -magnitude, magnitude))
    active = (token_mask.reshape(-1) != 0).unsqueeze(-1)
    gate = torch.where(active, gate, torch.zeros_like(gate))
    out = h + gate.unsqueeze(-1) * value.unsqueeze(1)
    return out.to(hidden.dtype)
