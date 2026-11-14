# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

# Adapted from
# https://github.com/huggingface/transformers/blob/v4.28.0/src/transformers/models/llama/modeling_llama.py
# Copyright 2023 The vLLM team.
# Copyright 2023 DeepSeek-AI and the HuggingFace Inc. team. All rights reserved.
#
# This code is based on EleutherAI's GPT-NeoX library and the GPT-NeoX
# and OPT implementations in this library. It has been modified from its
# original forms to accommodate minor architectural differences compared
# to GPT-NeoX and OPT used by the Meta AI team that trained the model.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
import typing
from dataclasses import dataclass

import torch
import torch_npu
from torch import nn
from transformers import DeepseekV2Config, DeepseekV3Config
from vllm.config import CacheConfig, VllmConfig
from vllm.model_executor.layers.layernorm import RMSNorm
from vllm.model_executor.layers.linear import ReplicatedLinear
from vllm.model_executor.layers.quantization import QuantizationConfig
from vllm.models.deepseek_v4.compressor import CompressorStateCache
from vllm.v1.kv_cache_interface import KVCacheSpec

from vllm_ascend.core.kv_cache_interface import AscendSlidingWindowMLASpec
from vllm_ascend.device.hardware_profile import HardwareCapability, get_current_hardware_profile
from vllm_ascend.utils import vllm_version_is
from vllm_ascend.worker.device_metadata import DeviceMetadataStage, wait_for_device_metadata

# vLLM #61392 switched DeepseekV4Config to the upstream Transformers class on
# main; the v0.32.0 lane still ships it under
# `vllm.transformers_utils.configs.deepseek_v4`.
if vllm_version_is("0.32.0"):
    from vllm.transformers_utils.configs.deepseek_v4 import DeepseekV4Config  # type: ignore[import-not-found]
else:
    from transformers import DeepseekV4Config


def _load_delivered_compressor():
    """交付包 ``npu_compressor`` 的可调用对象；优先直接加载已编译的 .so。

    扩展名为 ``npu_compressor``（pybind，PyInit_npu_compressor），模块里导出一个同名函数，
    参数与 aclnn 一致。直接 exec_module 可绕过 ``npu_ops_transformer.op_builder`` 的
    指纹校验与重编。
    """
    import glob
    import importlib.util
    import os

    pattern = os.path.expanduser("~/.cache/torch_extensions/py3*_cpu/npu_compressor/npu_compressor.so")
    for so in glob.glob(pattern):
        try:
            spec = importlib.util.spec_from_file_location("npu_compressor", so)
            if spec is None or spec.loader is None:
                continue
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            fn = getattr(module, "npu_compressor", None)
            if fn is not None:
                return fn
        except Exception:
            continue
    return None


_SYNCED_FIRST_CALL: set[str] = set()


def _delivered_op(name: str):
    """交付包里名为 ``name`` 的算子（9.2: ``npu_ops_transformer``，9.3: ``cann_ops_transformer``）。

    compressor 优先用直接加载的 pybind ``.so``（见 ``_load_delivered_compressor``），
    找不到才回退到包自己的 ``torch.ops`` 命名空间。
    """
    if name == "compressor":
        raw = _load_delivered_compressor()
        if raw is not None:

            def _positional(
                x,
                wkv,
                wgate,
                state_cache,
                ape,
                cmp_ratio,
                *,
                state_block_table=None,
                cu_seqlens=None,
                seqused=None,
                start_pos=None,
                coff=1,
                cache_mode=1,
            ):
                # pybind 版只接受 12 个位置参数
                return raw(
                    x,
                    wkv,
                    wgate,
                    state_cache,
                    ape,
                    cmp_ratio,
                    state_block_table,
                    cu_seqlens,
                    seqused,
                    start_pos,
                    coff,
                    cache_mode,
                )

            return _positional
    import importlib

    tried = []
    for pkg in ("cann_ops_transformer", "npu_ops_transformer"):
        try:
            importlib.import_module(pkg)
        except Exception as exc:  # 包不存在 / ABI 不配套
            tried.append(f"{pkg}: import failed ({type(exc).__name__}: {exc})")
            continue
        ns = getattr(torch.ops, pkg, None)
        if ns is None:
            tried.append(f"{pkg}: torch.ops.{pkg} missing")
            continue
        for cand in (name, f"npu_{name}"):
            fn = getattr(ns, cand, None)
            if fn is not None:
                return fn
        tried.append(f"{pkg}: no {name}")
    raise RuntimeError(f"delivered op {name!r} not found: {'; '.join(tried)}")


def _sync_first_call(op_key: str) -> None:
    """Synchronize once per op so asynchronously-reported aclnn failures surface at the call.

    Skipped -- and deliberately NOT marked as done -- while a stream is being captured: an aclnn
    op may not synchronize one (runtime 107027 "Not allow to synchronize captured-stream", the very
    failure ``dsv4_compressor_metadata`` was rewritten to avoid). The
    first call outside a capture then pays the sync, so attribution survives either ordering.
    """
    if op_key in _SYNCED_FIRST_CALL or _stream_is_capturing():
        return
    _SYNCED_FIRST_CALL.add(op_key)
    sync = getattr(torch_npu.npu, "synchronize", None)
    if sync is not None:
        sync()


def _stream_is_capturing() -> bool:
    """Whether the current NPU stream is being captured (ACLGraph).

    ``torch.compiler.is_compiling()`` cannot answer this: it tracks Dynamo / ``torch.compile``
    only and returns False throughout ``torch.npu.graph`` capture. The PTA API
    is the one the fork already uses in ``worker/v2/aclgraph_utils.py``. Probed per call rather
    than bound at import: it is consulted once per op per process, and a torch_npu build without
    ``torch_npu`` may not carry it at all.
    """
    probe = getattr(torch_npu.npu, "is_current_stream_capturing", None)
    if probe is None:
        return False
    try:
        return bool(probe())
    except Exception:  # noqa: BLE001 - never let this probe break an op call
        return False


def rms_norm_gamma(weight: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """Gamma in a dtype ``aclnnRmsNorm`` accepts beside ``x`` (see the dev fork's utils)."""
    if weight.dtype == torch.float32 or weight.dtype == x.dtype:
        return weight
    return weight.to(x.dtype)


@dataclass(frozen=True)
class RopeTables:
    """Partial-RoPE inputs of the compressor epilogue.

    ``head_dim`` is the rotated width (0 = no rotation) and ``sin``/``cos`` are the flattened
    ``[rows, head_dim]`` tables the fork hands over -- the same bytes the metadata op produces
    in ``[rows, 1, 1, head_dim]`` form.
    """

    sin: torch.Tensor | None = None
    cos: torch.Tensor | None = None
    head_dim: int = 0


def _compressor_norm_and_rope(
    compressed: torch.Tensor,
    norm_weight: torch.Tensor,
    rope: RopeTables,
    *,
    norm_eps: float,
) -> torch.Tensor:
    """Finish the compressor the delivered op stops short of: RMSNorm, then partial RoPE.

    The delivered ``npu_compressor`` documents five steps and names its output ``cmpKvOut =
    C_i^Comp``: matmul -> group +ape -> softmax -> Hadamard -> sum over the compression axis.
    This repo's in-tree compressor carries on with two more
    (``csrc/attention/compressor/README.md`` steps 6-7): ``RmsNorm(C^Comp) = norm_weight *
    C^Comp / RMS(C^Comp)``, then RoPE on the last ``rope_head_dim`` columns only
    (``op_kernel/arch35/compressor_block_vec.h``: "rope: 只对后RD进行rope"). The delivered
    arch35 kernel contains no norm and no rope at all -- its ``vf/`` directory has only
    ``vf_add`` / ``vf_mul`` / ``vf_softmax``, unlike the arch22 variant which still carries an
    ``RmsNormParam`` -- so for that entry the two steps belong to the caller.

    ``norm_weight`` is ``[D]`` and ``C^Comp`` is ``[..., D]``, so the norm is over the last dim,
    with eps inside the sqrt exactly as ``vf/vf_rms_norm.h`` computes it (sum of squares ->
    ``* 1/N`` -> ``+ eps`` -> ``sqrt`` -> divide -> multiply by gamma).
    """
    if compressed is None or compressed.numel() == 0 or norm_weight is None:
        return compressed

    # The delivered compressor produces C^Comp in fp32 while the checkpoint's gamma is bf16, and
    # aclnnRmsNorm has no (fp32 x, bf16 gamma) entry, so the gamma follows the rule in
    # ``rms_norm_gamma`` (shared with the Ascend RMSNorm):
    # Ascend RMSNorm: an fp32 gamma is the widest accepted form beside a 16-bit activation and
    # is kept, anything else follows the activation.
    norm_weight = rms_norm_gamma(norm_weight, compressed)
    normalized, _ = torch_npu.npu_rms_norm(compressed, norm_weight, norm_eps)
    if rope.head_dim > 0 and rope.cos is not None and rope.sin is not None:
        head_dim = normalized.shape[-1]
        # The delivered rope op requires x, cos and sin to be 4-D *together* and rotates the
        # ``partial_slice`` columns of the last dim; anything else is rejected before the
        # kernel runs (``EZ0013: ... x, cos and sin have incorrect shape dims``, see
        # inplace_rope_regbase_tiling_base.cpp:254). x is already [rows, D] here, and the fork
        # hands ``compressor`` flattened [rows, rope_head_dim] tables because the in-tree and
        # legacy entries take them that way (models/deepseek_v4/compressor.py) -- the bytes
        # are the same as the metadata op's [rows, 1, 1, dim] layout, so the unit axes are
        # restored here. ``unsqueeze`` (never ``reshape``) keeps x sharing this storage: the
        # op mutates its input in place, and that view is what carries the rotation back.
        x4 = normalized
        while x4.dim() < 4:
            x4 = x4.unsqueeze(-2)
        rows = x4.shape[0]
        inplace_partial_rotary_mul(
            x4,
            rope.cos.reshape(rows, 1, 1, rope.head_dim),
            rope.sin.reshape(rows, 1, 1, rope.head_dim),
            rotary_mode="interleave",
            partial_slice=[head_dim - rope.head_dim, head_dim],
        )
    return normalized


def inplace_partial_rotary_mul(x, r1, r2, *, rotary_mode="interleave", partial_slice=None):
    """Partial interleaved RoPE, mutating ``x`` in place.

    The in-tree ``_C_ascend`` op is preferred whenever this build registered it.
    Its aclnn entry carries the third attribute (``negate_sin``); the delivered
    package's entry passes only ``rotary_mode`` and ``partial_slice``, and on A6
    that shorter call lands in this repo's own tiling, which dereferences
    ``attrs->GetAttrPointer<bool>(2)`` without a null check -- so the 2-attribute
    call segfaults inside ``aclnnInplacePartialRotaryMulGetWorkspaceSize`` right
    after the shape checks pass. Callers already pass the negated sin where the
    sign must flip, so ``negate_sin=False`` reproduces their intent.
    """
    try:
        in_tree = torch.ops._C_ascend.inplace_partial_rotary_mul
    except AttributeError:  # build without the in-tree op registered
        in_tree = None
    if in_tree is not None:
        return in_tree(
            x,
            r1,
            r2,
            rotary_mode,
            list(partial_slice or [0, 0]),
            False,
        )
    fn = _delivered_op("inplace_partial_rotary_mul")
    return fn(x, r1, r2, rotary_mode=rotary_mode, partial_slice=partial_slice)


def _delivered_compressor(
    x,
    wkv,
    wgate,
    state_cache,
    ape,
    *,
    norm_weight=None,
    rope_sin=None,
    rope_cos=None,
    state_block_table=None,
    cu_seqlens=None,
    seqused=None,
    start_pos=None,
    rope_head_dim: int = 0,
    cmp_ratio: int = 4,
    coff: int = 1,
    norm_eps: float = 1e-6,
    rotary_mode: int = 0,  # noqa: ARG001 - in-tree only
    cache_mode: int = 1,
):
    """``_C_ascend.compressor`` (18-arg) -> the delivered compressor.

    CANN sig: ``compressor(x, wkv, wgate, state_cache, ape, cmp_ratio=4, *,
    state_block_table, cu_seqlens, seqused, start_pos, coff=1, cache_mode=1)``
    returning only ``cmp_kv``. It is resolved through the package's own interface
    (``npu_ops_transformer.ops.compressor``) because the registered ``torch.ops`` schema for
    this op cannot be called as documented -- see ``_OT_PYTHON_INTERFACE``.

    ``norm_weight`` / ``rope_sin`` / ``rope_cos`` / ``rope_head_dim`` / ``norm_eps`` /
    ``rotary_mode`` are keyword-only here (the five required tensors of the delivered op stay
    positional): the CANN op has **no** norm and no RoPE input, while this repo's in-tree
    compressor folds both in. Both missing steps are applied here, after the call
    (``_compressor_norm_and_rope``), and the result is what the caller expects.
    """
    # A6 runs the delivered 14-argument op: CANN 9.2 ships its bf16/fp16 kernel under
    # ``opp/built-in/.../ascend960dt/ops_transformer/compressor`` (attrs: cmp_ratio,
    # coff, cache_mode, state_cache_stride_dim0), and the package's own interface
    # ``npu_ops_transformer.ops.compressor.npu_compressor`` is the spelling that reaches
    # it. The in-tree ``_C_ascend.compressor`` folds RMSNorm and RoPE into the op (19
    # arguments) and needs a vendor bundle that would shadow the built-in A6 kernel, so it
    # must not be used here.
    # Validate before the call, not after: ``fn`` writes ``state_cache`` in place and the
    # first call also synchronizes, so a late rejection would leave a half-applied step and
    # surface a lazy state-cache shape guard instead of this message.
    if rotary_mode not in (0, 2):
        raise RuntimeError(
            f"compressor: rotary_mode={rotary_mode} is not supported by the delivered path "
            "(0 or 2 / interleave only); the in-tree op also accepts 1 (half)."
        )
    fn = _delivered_op("compressor")
    out = fn(
        x,
        wkv,
        wgate,
        state_cache,
        ape,
        cmp_ratio,
        state_block_table=state_block_table,
        cu_seqlens=cu_seqlens,
        seqused=seqused,
        start_pos=start_pos,
        coff=coff,
        cache_mode=cache_mode,
    )
    # One synchronize after the first call, like the two cache-writing ops: aclnn failures can
    # be reported only at the next sync, so without this the worker dies with the engine's error
    # long after the call that caused it.
    _sync_first_call("compressor")
    out = _compressor_norm_and_rope(
        out,
        norm_weight,
        RopeTables(sin=rope_sin, cos=rope_cos, head_dim=rope_head_dim),
        norm_eps=norm_eps,
    )
    # The delivered compressor yields C^Comp in fp32 and the norm/rope post-processing keeps it
    # there, while the in-tree contract lists ``cmp_kv`` as FLOAT16/BFLOAT16
    # (csrc/attention/compressor/README.md) and the result goes straight into the compressed
    # cache write. Casting to the caller's dtype was tried and then withdrawn: it changes the
    # payload the compressed-cache write interprets, and that change was never measured at
    # ``temperature=0``. Keep the pre-existing behaviour until that measurement exists.
    return out


def _compressor_op(
    x,
    wkv_weight,
    wgate_weight,
    state_cache,
    ape,
    norm_weight,
    rope_sin,
    rope_cos,
    **kwargs,
):
    """A6: the in-tree ``_C_ascend.compressor`` never returns on this silicon.

    Single-op evidence (2026-11-14): host launch returns in 0.25 s, then
    ``torch.npu.synchronize()`` never completes -- for T=4/8/22, coff=1/2 and
    both random and all-zero inputs, with no EZ error.  The delivered
    ``npu_compressor`` does the same five steps (matmul -> +ape -> softmax ->
    Hadamard -> sum) on the ascend960dt built-in kernel and is what PR50 runs;
    it stops before RMSNorm + RoPE, so the shim finishes those two steps.
    ``A6_IN_TREE_COMPRESSOR=1`` keeps the in-tree op for A/B comparison.
    """
    import os

    if os.environ.get("A6_IN_TREE_COMPRESSOR") == "1":
        return torch.ops._C_ascend.compressor(
            x,
            wkv_weight,
            wgate_weight,
            state_cache,
            ape,
            norm_weight,
            rope_sin,
            rope_cos,
            **kwargs,
        )
    # 交付包的 npu_compressor 是 pybind 扩展（PyInit_npu_compressor）：走
    # ``import npu_ops_transformer.ops.compressor`` 会触发 builder 校验/重编（本机实测反复
    # ninja 十分钟不返回）。这里直接加载已编译的 .so；它只做 5 步，
    # RMSNorm + RoPE 由 ``_delivered_compressor`` 补齐（见其 docstring）。
    return _delivered_compressor(
        x,
        wkv_weight,
        wgate_weight,
        state_cache,
        ape,
        norm_weight=norm_weight,
        rope_sin=rope_sin,
        rope_cos=rope_cos,
        **kwargs,
    )


class AscendCompressorStateCache(CompressorStateCache):
    def __init__(
        self,
        state_dim: int,
        dtype: torch.dtype,
        compress_ratio: int,
        block_size: int,
        prefix: str,
    ):
        super().__init__(state_dim, dtype, compress_ratio, prefix)
        self.compress_ratio = compress_ratio
        self.block_size = block_size

    def get_kv_cache_spec(self, vllm_config: VllmConfig) -> KVCacheSpec:
        from vllm_ascend.models.layer.attention.layer import dsv4_block_sizes

        pads = dsv4_block_sizes(vllm_config)[vllm_config.cache_config.block_size][1]
        page_size_padded = pads[0] if self.state_dim == 2 * 256 and self.compress_ratio == 4 else pads[1]

        return AscendSlidingWindowMLASpec(
            block_size=self.block_size,
            num_kv_heads=1,
            head_size=self.state_dim,
            dtype=self.dtype,
            sliding_window=self.sliding_window,
            alignment=None,
            page_size_padded=page_size_padded,
        )

    def forward(self): ...

    def get_attn_backend(self):
        # Keep these imports lazy to avoid a model-inspection circular import.
        if self.compress_ratio == 4:
            from vllm_ascend.attention.dsa_v1 import AscendDSAC4StateBackend

            return AscendDSAC4StateBackend
        if self.compress_ratio == 128:
            from vllm_ascend.attention.dsa_v1 import AscendDSAC128StateBackend

            return AscendDSAC128StateBackend
        raise ValueError(f"Unsupported DeepSeek V4 state-cache compression ratio: {self.compress_ratio}")


@dataclass(frozen=True)
class AscendCompressorMetadata:
    """Request metadata for the compressed KV and compressor state caches."""

    cache: typing.Any
    state: typing.Any


class Compressor(nn.Module):
    def __init__(
        self,
        vllm_config: VllmConfig,
        config: DeepseekV2Config | DeepseekV3Config | DeepseekV4Config,
        compress_ratio: int = 4,
        head_dim: int = 512,
        rotate: bool = False,
        *,
        cache_config: CacheConfig,
        quant_config: QuantizationConfig | None = None,
        prefix: str = "",
    ):
        super().__init__()
        from vllm_ascend.models.layer.attention.layer import DSV4_BLOCK_SIZES

        self.vllm_config = vllm_config
        self.config = config
        self.dim = config.hidden_size
        self.head_dim = head_dim
        self.rope_head_dim = config.qk_rope_head_dim
        self.nope_head_dim = head_dim - config.qk_rope_head_dim
        self.compress_ratio = compress_ratio
        self.overlap = compress_ratio == 4
        self.rotate = rotate
        self.norm_eps = config.rms_norm_eps
        self.coff = 1 + self.overlap

        self.ape = nn.Parameter(torch.empty(compress_ratio, self.coff * self.head_dim, dtype=torch.float32))
        self.wkv = ReplicatedLinear(
            self.dim,
            self.coff * self.head_dim,
            bias=False,
            quant_config=None
            if get_current_hardware_profile().supports(HardwareCapability.DSV4_COMPRESSED_CACHE)
            else quant_config,
            prefix=f"{prefix}.wkv",
            return_bias=False,
        )
        self.wgate = ReplicatedLinear(
            self.dim,
            self.coff * self.head_dim,
            bias=False,
            quant_config=None
            if get_current_hardware_profile().supports(HardwareCapability.DSV4_COMPRESSED_CACHE)
            else quant_config,
            prefix=f"{prefix}.wgate",
            return_bias=False,
        )

        # The custom compressor op consumes ND weights directly.
        self.wkv.skip_weight_nz_conversion = True
        self.wgate.skip_weight_nz_conversion = True

        # The DSV4 compressor kernel only accepts FP32 norm_weight.
        self.norm = RMSNorm(self.head_dim, config.rms_norm_eps, dtype=torch.float32)

        state_dtype = torch.float32
        # TODO(zyj): change following codes if block_size is configurable & refactor the magic numbers
        if compress_ratio == 4:
            self.state_cache = AscendCompressorStateCache(
                state_dim=2 * self.coff * self.head_dim,  # kv_state + score_state
                dtype=state_dtype,
                compress_ratio=compress_ratio,
                prefix=f"{prefix}.state_cache",
                block_size=DSV4_BLOCK_SIZES[cache_config.block_size][0][2],
            )
        elif compress_ratio == 128:
            self.state_cache = AscendCompressorStateCache(
                state_dim=2 * self.head_dim,  # kv_state + score_state
                dtype=state_dtype,
                compress_ratio=compress_ratio,
                prefix=f"{prefix}.state_cache",
                block_size=DSV4_BLOCK_SIZES[cache_config.block_size][0][3],
            )
        else:
            raise ValueError(
                f"Only support compress_ratio in [4, 128]. Got unsupported compress_ratio: {compress_ratio}"
            )

    def _compute_metadata(
        self,
        metadata: typing.Any,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Imported lazily to avoid a circular import at module load time.
        from vllm_ascend.attention.dsa_v1 import get_or_compute_compressor_metadata

        precomputed = getattr(metadata, "compressor_metadata", None)
        if precomputed is not None:
            group_id = metadata.compressor_metadata_group_id
            assert group_id is not None
            wait_for_device_metadata(DeviceMetadataStage.COMPRESSOR, group_id)
            return precomputed

        return get_or_compute_compressor_metadata(metadata, self.compress_ratio, self.vllm_config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        state_cache: torch.Tensor,
        metadata: AscendCompressorMetadata,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        compressor_metadata = metadata.cache.req_metadata
        state_metadata = metadata.state.req_metadata
        assert compressor_metadata is not None
        assert state_metadata is not None
        compress_cos, compress_sin, slot_mapping = self._compute_metadata(compressor_metadata)
        compressed_kv = _compressor_op(
            hidden_states,
            self.wkv.weight,
            self.wgate.weight,
            state_cache.squeeze(-2),
            self.ape,
            self.norm.weight,
            compress_sin.view(-1, compress_sin.shape[-1]),
            compress_cos.view(-1, compress_cos.shape[-1]),
            state_block_table=state_metadata.block_table,
            cu_seqlens=compressor_metadata.query_start_loc,
            seqused=None,
            start_pos=compressor_metadata.start_pos,
            rope_head_dim=self.rope_head_dim,
            cmp_ratio=self.compress_ratio,
            coff=2 if self.overlap else 1,
            norm_eps=self.norm_eps,
            rotary_mode=2,
            cache_mode=1,
        )
        return compressed_kv, slot_mapping
