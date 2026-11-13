# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Semantic profiles for supported Ascend hardware families.

This module is the boundary between hardware identity and shared business
logic. Detection maps a SoC to one immutable :class:`HardwareProfile`; code
outside the device package then asks that profile about a semantic contract
instead of branching on an A2, A3, A5, or 310P name.

The profile uses three kinds of fields:

* A :class:`HardwareCapability` is a binary, independently testable contract
  that says a code path, operator ABI, runtime facility, or kernel variant is
  available. It describes the supported software-and-hardware stack, not just
  a physical instruction in the silicon. A caller must still check its model,
  shape, dtype, user configuration, and runtime preconditions.
* A ``*Policy`` or ``*Mode`` selects behavior or a default. It answers "which
  rule applies?", not "is the device capable?".
* A ``*Family`` selects one implementation from several equivalent interfaces.

The absence of a capability means that its consumer must use its documented
fallback or reject that feature. It does not, by itself, mean that every
related operation is unsupported on the device.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, auto
from types import MappingProxyType

from vllm_ascend.device.device_config import get_device_config
from vllm_ascend.device.hardware import AscendDeviceType


class HardwareCapability(Enum):
    """Binary contracts consumed by shared business logic.

    ``profile.supports(X)`` answers only whether the exact contract documented
    for ``X`` is available. Capabilities are intentionally narrow so callers
    do not infer unrelated behavior from a device generation. New default or
    implementation-selection fields should normally be modeled as a policy,
    mode, or family rather than added here.

    A few historical members still describe a default or implementation
    selection. Their comments call this out explicitly; do not interpret them
    as broad claims about the underlying silicon.
    """

    # Legacy default: replace vLLM's custom-op list with ``["all"]`` during
    # platform configuration. Runtime extension loading is a separate contract.
    AUTO_ENABLE_CUSTOM_OPS = auto()
    # Register ATB extension entry points eagerly during worker construction.
    ATB_EXTENSIONS = auto()
    # Run the ATB matmul warmup that avoids first-request cache-write latency.
    ATB_WARMUP = auto()
    # Register fake/meta implementations for the custom BGMV and SGMV LoRA ops.
    BGMV_SGMV_META_REGISTRATION = auto()
    # SK scatter kernel for paged cache writes (A2/A3/A5 ABI).
    SCATTER_ND_CACHE_STORE = auto()
    # Extend SK cache writes to non-overlapping row-strided destinations.
    # A5 is excluded until arch35 handles their storage bounds correctly.
    SCATTER_ND_STRIDED_CACHE_STORE = auto()
    # Extend SK cache writes to FP8 E4M3FN/E5M2 (A5 ABI).
    SCATTER_ND_FP8_CACHE_STORE = auto()
    # Allow the CANN MegaMoe fused-MC2 path when its model, EP, and config checks pass.
    CANN_MEGAMOE = auto()
    # Allow A5 MegaMoe's MXFP-only path and its A5-specific calling conventions.
    CANN_MEGAMOE_MXFP = auto()
    # Split a mixed chunked-prefill batch into separate prefill and decode FIA calls.
    CHUNKED_PREFILL_PHASE_SPLIT = auto()
    # CANN device-side FlashAttn metadata with strided paged GQA K/V.
    DEVICE_LENGTH_FLASH_ATTN = auto()
    # Build CPU-affinity pools from the cluster-aware CPU topology.
    CLUSTER_CPU_TOPOLOGY = auto()
    # Legacy implementation selector: install the 310P-specific operator overrides.
    COMPATIBILITY_OP_IMPLEMENTATIONS = auto()
    # Install distributed-communication adaptations required by the compatibility path.
    DISTRIBUTED_COMMUNICATION_ADAPTATION = auto()
    # Use DSA C128-state kernel block sizes ``[4, 8, 16]`` instead of ``[8, 16, 32]``.
    DSA_C128_STATE_SMALL_BLOCK_SIZES = auto()
    # Use the DeepSeek-V4/DSA compressed-KV-cache layout and compressor/indexer flow.
    DSV4_COMPRESSED_CACHE = auto()
    # Use the DeepSeek-V4.1 mixed-quant cache and packaged QLI/QSMLA ABI.
    DSV41_PACKED_CACHE = auto()
    # Triton RMS/dot/gate/residual fusion for unrotated Engram inputs.
    ENGRAM_UNROTATED_GATE = auto()
    # Decode native FP8 codes and E8M0 group scales in Engram table lookups.
    ENGRAM_MXFP8 = auto()
    # Enable dynamic-MX norm fusion and the associated ``wo_a`` weight-layout contract.
    DYNAMIC_MX_QUANT_FUSION = auto()
    # Select DynamicMxQuantV3 ``scale_alg=1`` for model paths that require it.
    DYNAMIC_MX_QUANT_SCALE_ALG_ONE = auto()
    # Allow GDN prefill to use the external FLA NPU fused operator.
    FLA_GDN_PREFILL = auto()
    # Enable the FP8/C8 attention KV-cache ABI and matching attention preprocess paths.
    # This is not a general statement that every FP8 operation is supported.
    FP8_ATTENTION = auto()
    # CANN PoolKeyIndexer quant_mode=0 with E4M3 query/key and FP32 descales.
    POOL_KEY_INDEXER_FP8 = auto()
    # Prepare and reuse the fixed CANN GBSA schedule across supported M3 decode layers.
    # Callers must also check block size, dtype, and uniform query length.
    MINIMAX_M3_GBSA_METADATA_REUSE = auto()
    # Select the compatibility grouped-top-k router used by the fused-MoE path.
    FUSED_MOE_COMPATIBILITY = auto()
    # Use npu_rotary_mul with adjacent pairs (rotary_mode="interleave").
    FUSED_ROTARY_MUL_INTERLEAVE = auto()
    # Pass ``glu_alpha`` and ``glu_bias`` to the fused dequant-SwiGLU-quant operator.
    FUSED_SWIGLU_TUNING_ARGS = auto()
    # Select the compatibility GatedDeltaNet core and state-dtype implementation.
    GDN_COMPATIBILITY = auto()
    # Register the FX graph rewrite that fuses a supported matmul/reduce-scatter pattern.
    GRAPH_MM_REDUCE_SCATTER_FUSION = auto()
    # Register the FX graph rewrite that fuses the supported muls-plus-add pattern.
    GRAPH_MULS_ADD_FUSION = auto()
    # Register the FX graph rewrites for supported RMSNorm-plus-quant patterns.
    GRAPH_NORM_QUANT_FUSION = auto()
    # Register the FX graph rewrite that fuses q/k RMSNorm, RoPE and the weight-less
    # v RMSNorm into the SIMD ``split_qkv_rmsnorm_rope_vnorm`` Triton kernel.
    GRAPH_QKV_NORM_ROPE_FUSION = auto()
    # Let inplace_partial_rotary_mul negate sine internally; profiles without
    # this contract negate the sine input explicitly and pass negate_sin=False.
    INPLACE_PARTIAL_ROTARY_MUL_NEGATE_SIN = auto()
    # Reserve host CPUs for interrupt handling when constructing worker CPU pools.
    IRQ_CPU_RESERVATION = auto()
    # Create the local communication resource used by supported KV-transfer deployments.
    LOCAL_KV_COMM_RESOURCE = auto()
    # Use vLLM-Ascend's custom BGMV/SGMV LoRA kernels when rank constraints also pass.
    LORA_CUSTOM_OPS = auto()
    # Use the token-fused MLA cache protocol for eligible FlashMLA models.
    MLA_FLASH = auto()
    # Allow MLAPO with native floating-point projection weights, not only quantized weights.
    MLAPO_NATIVE_WEIGHTS = auto()
    # Accept ``fullmesh_v2`` as the MC2 communication algorithm.
    MC2_FULLMESH_V2_COMM = auto()
    # Accept hierarchical MC2 communication, subject to its expert-count constraints.
    MC2_HIERARCHY_COMM = auto()
    # Pass TP group/rank metadata required by the newer MoE dispatch/combine ABI.
    MOE_DISPATCH_EXTRA_ARGS = auto()
    # Pass shared-expert, expert-scale, quant-mode, and output-dtype metadata to MoE dispatch.
    MOE_DISPATCH_SHARED_EXPERT_ARGS = auto()
    # Route DeepSeek-V4 vision and hash rows through the fused
    # ``moe_gating_top_k_hash`` ABI with ``bias_vl`` and image sentinels.
    MOE_GATING_TOP_K_HASH_VISION = auto()
    # Allow the extended NPU graph backend; static-kernel mode depends on this contract.
    MM_REDUCE_SCATTER_AI_CPU_INFERENCE = auto()
    NPUGRAPH_EX = auto()
    # Use ``torch_npu.npu_top_k_top_p`` for sampling instead of pytorch fallback.
    NPU_TOP_K_TOP_P = auto()
    # Allow shared paged-attention decode when graph mode, shape, and model checks pass.
    PAGED_ATTENTION = auto()
    # Inspect PCIe topology to distinguish 310P Root-Complex and endpoint deployments.
    RC_DEVICE_DISCOVERY = auto()
    # Fused RMSNorm+cast is available for the A2/A3 (CANN 9.1) path.
    RMS_NORM_CAST = auto()
    # Fused Triton split+QKV RMSNorm+MRoPE kernel for Qwen3.5 attention.
    SPLIT_QKV_RMSNORM_MROPE = auto()
    # Import and register the compiled vLLM-Ascend custom-op library at runtime.
    # This is independent of whether custom ops are enabled by default.
    RUNTIME_CUSTOM_OPS = auto()
    # The fused Gemma RMSNorm custom op needs aclnnGemmaRmsNorm, which CANN does
    # not provide for A6 (no ascend960dt kernel).
    GEMMA_RMS_NORM_OP = auto()
    # The fused add+RMSNorm+bias custom op needs aclnnAddRmsNormBias, which
    # CANN does not provide for A6 (no libopapi symbol, no ascend960dt kernel).
    ADD_RMS_NORM_BIAS_OP = auto()
    # Legacy implementation selector: install the standard worker patch bundle.
    STANDARD_WORKER_PATCHES = auto()
    # Legacy implementation selector: install the standard Mamba platform patch.
    STANDARD_MAMBA_PATCH = auto()
    # Reserved contract for the A5 SwiGLU OAI MX-quant path; it currently has no consumer.
    SWIGLU_OAI_MX_QUANT = auto()
    # Use the Triton batch-memcpy kernel for Mamba state copies.
    TRITON_BATCH_MEMCPY = auto()
    # Native TurboQuant 4-bit non-causal MLA cache and SFA kernels.
    TURBOQUANT_4BIT_NC_CACHE = auto()
    # Honor MLAPO enablement on any pipeline role; other profiles limit it to decode consumers.
    UNRESTRICTED_MLAPO = auto()


class AttentionBackendFamily(Enum):
    """Attention backend implementation families selected by the platform."""

    STANDARD = auto()
    COMPATIBILITY = auto()


class CPUBindingMode(Enum):
    """CPU binding policies selected for worker processes."""

    TOPO_AFFINITY = "topo_affinity"
    GLOBAL_SLICE = "global_slice"


class DeviceAdaptorFamily(Enum):
    """Device operation adaptor implementation families."""

    STANDARD = auto()
    FP8_OPTIMIZED = auto()
    COMPATIBILITY = auto()
    A6 = auto()


class DeviceAddressingMode(Enum):
    """PCIe device addressing policies used by CPU binding."""

    DIRECT = auto()
    DUAL_CHIP_CARD = auto()


class MoECommPolicy(Enum):
    """MoE communication selection policies."""

    CAPACITY_AND_EXPERT_DENSITY = auto()
    FUSED_OR_CAPACITY = auto()
    CAPACITY_AND_WORLD_SIZE = auto()
    ALLGATHER = auto()


class QuantizationBackendFamily(Enum):
    """Quantization configuration implementation families."""

    STANDARD = auto()
    COMPATIBILITY = auto()


class WeightLayoutPolicy(Enum):
    """Weight layout selection policies for supported hardware families."""

    CONFIGURABLE = auto()
    FORCE_NZ = auto()


@dataclass(frozen=True, slots=True)
class HardwareProfile:
    """Immutable semantic contracts registered for one SoC family.

    ``_device_type`` is registration metadata and must not escape into business
    logic. Family, mode, and policy fields select implementations or behavior;
    ``capabilities`` records additive binary support contracts.
    """

    # Identity is private to detection and profile registration.
    _device_type: AscendDeviceType
    # Mutually exclusive implementation and behavior selections.
    attention_backend_family: AttentionBackendFamily
    cpu_binding_mode: CPUBindingMode
    default_worker_cls: str
    device_adaptor_family: DeviceAdaptorFamily
    device_addressing_mode: DeviceAddressingMode
    weight_layout_policy: WeightLayoutPolicy
    moe_comm_policy: MoECommPolicy
    quantization_backend_family: QuantizationBackendFamily
    # Independent, additive contracts queried with ``supports``.
    capabilities: frozenset[HardwareCapability]

    # NZ mm/addmm is supported, but batched matmul rejects NZ weights.
    requires_2d_nz_linear_input: bool = False
    # A5 uses mixed softmax for compatibility with CANN GBSA packages that
    # reject BF16 precision 0. A3 retains FP32 online softmax.
    minimax_m3_gbsa_bf16_softmax_precision: int = 0
    # Fixed GDN verification uses the strided recurrent kernel at/above this
    # query length to work around native operator numerical failures. None
    # retains native dispatch; this is a stack workaround, not a silicon limit.
    gdn_fixed_spec_recurrent_fallback_min_query_len: int | None = None

    def supports(self, capability: HardwareCapability) -> bool:
        """Return whether the exact documented ``capability`` contract is available."""

        return capability in self.capabilities


_STANDARD_CAPABILITIES = frozenset(
    {
        HardwareCapability.AUTO_ENABLE_CUSTOM_OPS,
        HardwareCapability.ATB_EXTENSIONS,
        HardwareCapability.ATB_WARMUP,
        HardwareCapability.ADD_RMS_NORM_BIAS_OP,
        HardwareCapability.BGMV_SGMV_META_REGISTRATION,
        HardwareCapability.FLA_GDN_PREFILL,
        HardwareCapability.FUSED_ROTARY_MUL_INTERLEAVE,
        HardwareCapability.FUSED_SWIGLU_TUNING_ARGS,
        HardwareCapability.GRAPH_MM_REDUCE_SCATTER_FUSION,
        HardwareCapability.GRAPH_MULS_ADD_FUSION,
        HardwareCapability.GRAPH_NORM_QUANT_FUSION,
        HardwareCapability.GRAPH_QKV_NORM_ROPE_FUSION,
        HardwareCapability.INPLACE_PARTIAL_ROTARY_MUL_NEGATE_SIN,
        HardwareCapability.IRQ_CPU_RESERVATION,
        HardwareCapability.LORA_CUSTOM_OPS,
        HardwareCapability.MC2_HIERARCHY_COMM,
        HardwareCapability.MOE_GATING_TOP_K_HASH_VISION,
        HardwareCapability.NPUGRAPH_EX,
        HardwareCapability.PAGED_ATTENTION,
        HardwareCapability.RUNTIME_CUSTOM_OPS,
        HardwareCapability.SCATTER_ND_CACHE_STORE,
        HardwareCapability.SCATTER_ND_STRIDED_CACHE_STORE,
        HardwareCapability.STANDARD_MAMBA_PATCH,
        HardwareCapability.STANDARD_WORKER_PATCHES,
        HardwareCapability.TRITON_BATCH_MEMCPY,
        HardwareCapability.TURBOQUANT_4BIT_NC_CACHE,
    }
)
_A3_CAPABILITIES = _STANDARD_CAPABILITIES | {
    HardwareCapability.MINIMAX_M3_GBSA_METADATA_REUSE,
    HardwareCapability.MC2_FULLMESH_V2_COMM,
    HardwareCapability.RMS_NORM_CAST,
}
# A6-ADAPT: A6 reuses A5's validated capability set (the silicon is close to
# A5 and both are whitelists that exclude ATB). A6-specific exclusions are
# applied in the A6 profile below.
_A5_CAPABILITIES = frozenset(
    {
        HardwareCapability.AUTO_ENABLE_CUSTOM_OPS,
        HardwareCapability.SCATTER_ND_CACHE_STORE,
        HardwareCapability.SCATTER_ND_FP8_CACHE_STORE,
        HardwareCapability.CANN_MEGAMOE,
        HardwareCapability.CANN_MEGAMOE_MXFP,
        HardwareCapability.DEVICE_LENGTH_FLASH_ATTN,
        HardwareCapability.CHUNKED_PREFILL_PHASE_SPLIT,
        HardwareCapability.CLUSTER_CPU_TOPOLOGY,
        HardwareCapability.DSA_C128_STATE_SMALL_BLOCK_SIZES,
        HardwareCapability.DSV4_COMPRESSED_CACHE,
        HardwareCapability.DSV41_PACKED_CACHE,
        HardwareCapability.ENGRAM_UNROTATED_GATE,
        HardwareCapability.ENGRAM_MXFP8,
        HardwareCapability.DYNAMIC_MX_QUANT_FUSION,
        HardwareCapability.DYNAMIC_MX_QUANT_SCALE_ALG_ONE,
        HardwareCapability.FLA_GDN_PREFILL,
        HardwareCapability.FP8_ATTENTION,
        HardwareCapability.POOL_KEY_INDEXER_FP8,
        HardwareCapability.FUSED_ROTARY_MUL_INTERLEAVE,
        HardwareCapability.GRAPH_MM_REDUCE_SCATTER_FUSION,
        HardwareCapability.GRAPH_MULS_ADD_FUSION,
        HardwareCapability.GRAPH_NORM_QUANT_FUSION,
        HardwareCapability.LOCAL_KV_COMM_RESOURCE,
        HardwareCapability.LORA_CUSTOM_OPS,
        HardwareCapability.MLA_FLASH,
        HardwareCapability.MLAPO_NATIVE_WEIGHTS,
        HardwareCapability.MINIMAX_M3_GBSA_METADATA_REUSE,
        HardwareCapability.MOE_DISPATCH_EXTRA_ARGS,
        HardwareCapability.MOE_DISPATCH_SHARED_EXPERT_ARGS,
        HardwareCapability.MM_REDUCE_SCATTER_AI_CPU_INFERENCE,
        HardwareCapability.MOE_GATING_TOP_K_HASH_VISION,
        HardwareCapability.NPUGRAPH_EX,
        HardwareCapability.RUNTIME_CUSTOM_OPS,
        HardwareCapability.STANDARD_MAMBA_PATCH,
        HardwareCapability.STANDARD_WORKER_PATCHES,
        HardwareCapability.SWIGLU_OAI_MX_QUANT,
        HardwareCapability.TRITON_BATCH_MEMCPY,
        HardwareCapability.UNRESTRICTED_MLAPO,
    }
)
# A6 does not support the ATB extension contracts: no nnal/libatb.so on A6 hosts.
# A6-ADAPT: no compiled extension on A6
_A6_UNSUPPORTED_CAPABILITIES = frozenset(
    {
        HardwareCapability.ATB_EXTENSIONS,
        HardwareCapability.ATB_WARMUP,
        # A6-ADAPT (2026-11-11, real box): the ASC-built kernels register fine, but
        # their aclnn entry points are not installed -- `aclnnHcPre` / `aclnnHcPost` /
        # `aclnnVllmAscend*` / `aclnnKvQuantSparseAttnSharedkv` are in none of this
        # CANN's opapi libraries (`aclnnVllmAscend*`: 0 symbols), so the first call
        # dies with "not in libopapi.so". Keep enable_custom_op() False (as the dev
        # fork does) until the vendor opapi package ships; the DSv4 path here goes
        # through the delivered cann_ops_transformer / npu_ops_transformer ops instead.
        HardwareCapability.RUNTIME_CUSTOM_OPS,
        # A6-ASC-FUSION: custom ops are available again. The kernel library
        # builds through the ASC CMake toolchain (find_package(ASC) plus
        # --npu-arch=dav-9201) instead of the legacy ascendc.cmake route, so
        # vllm_ascend_C is no longer missing on A6.
        # A6-ADAPT: no aclnnForeachAddListV2 on A6
        # The muls_add pass traces its pattern with torch._foreach_add, which
        # lowers to aclnnForeachAddListV2 and fails with 561103
        # (AclOpKernelInit failed) on A6's CANN 9.2.0. The in-place V1 variant
        # works; only the out-of-place V2 form is missing.
        HardwareCapability.GRAPH_MULS_ADD_FUSION,
        # A6-ADAPT: no TopKTopP kernel
        # libopapi.so exports aclnnApplyTopKTopP, but the ascend960dt (and
        # ascend910_93) op package registers no TopKTopP binary, so
        # torch_npu.npu_top_k_top_p fails with 561000
        # ("ApplyTopKTopPWithSorted does not has any binary"). Sampling keeps
        # the pytorch fallback from worker/v2/sample/apply_top_k_top_p.py.
        HardwareCapability.NPU_TOP_K_TOP_P,
        # A6-ADAPT: no aclnnGemmaRmsNorm kernel
        # libopapi.so exports the symbol, but CANN registers no ascend960dt
        # GemmaRmsNorm binary, so aclnnGemmaRmsNorm fails with EZ1009.
        # RMSNorm with (1 + weight) is the documented equivalent.
        HardwareCapability.GEMMA_RMS_NORM_OP,
        # A6-ADAPT: no aclnnAddRmsNormBias kernel
        # Same as GemmaRmsNorm: no ascend960dt binary, so RMSNorm with residual
        # falls back to npu_add_rms_norm plus a bias add.
        HardwareCapability.ADD_RMS_NORM_BIAS_OP,
        # A6-ADAPT: the fused Triton MRoPE kernel raises a vector core exception
        # on A6. Attention falls back to split + RMSNorm + rotary_emb.
        HardwareCapability.SPLIT_QKV_RMSNORM_MROPE,
        # A6-ADAPT: the delivered hash-gating shim accepts ``bias_vl`` but drops it
        # with a warning, so the fused vision hash cannot route image tokens the way
        # the in-tree path does. Withholding the capability makes the router take the
        # existing ``select_deepseek_v4_vision_experts`` fallback, which does honour
        # ``bias_vl`` -- silently degraded routing is worse than an unfused path.
        HardwareCapability.MOE_GATING_TOP_K_HASH_VISION,
    }
)
_A6_CAPABILITIES = _A5_CAPABILITIES - _A6_UNSUPPORTED_CAPABILITIES
_DEFAULT_WORKER_CLS = "vllm_ascend.worker.worker.NPUWorker"
_HARDWARE_PROFILES: Mapping[AscendDeviceType, HardwareProfile] = MappingProxyType(
    {
        AscendDeviceType.A2: HardwareProfile(
            _device_type=AscendDeviceType.A2,
            # A2 BF16 native verification produced NaNs with 8 drafts (9 queries).
            # Nine is the first observed failing configuration, not a proven
            # universal kernel bound. Remove after native recurrent parity is
            # validated on the replacement operator stack.
            gdn_fixed_spec_recurrent_fallback_min_query_len=9,
            attention_backend_family=AttentionBackendFamily.STANDARD,
            cpu_binding_mode=CPUBindingMode.TOPO_AFFINITY,
            default_worker_cls=_DEFAULT_WORKER_CLS,
            device_adaptor_family=DeviceAdaptorFamily.STANDARD,
            device_addressing_mode=DeviceAddressingMode.DIRECT,
            weight_layout_policy=WeightLayoutPolicy.CONFIGURABLE,
            moe_comm_policy=MoECommPolicy.CAPACITY_AND_EXPERT_DENSITY,
            quantization_backend_family=QuantizationBackendFamily.STANDARD,
            capabilities=_STANDARD_CAPABILITIES
            | {HardwareCapability.RMS_NORM_CAST, HardwareCapability.NPU_TOP_K_TOP_P},
        ),
        AscendDeviceType.A3: HardwareProfile(
            _device_type=AscendDeviceType.A3,
            attention_backend_family=AttentionBackendFamily.STANDARD,
            cpu_binding_mode=CPUBindingMode.GLOBAL_SLICE,
            default_worker_cls=_DEFAULT_WORKER_CLS,
            device_adaptor_family=DeviceAdaptorFamily.STANDARD,
            device_addressing_mode=DeviceAddressingMode.DUAL_CHIP_CARD,
            weight_layout_policy=WeightLayoutPolicy.CONFIGURABLE,
            moe_comm_policy=MoECommPolicy.FUSED_OR_CAPACITY,
            quantization_backend_family=QuantizationBackendFamily.STANDARD,
            capabilities=_A3_CAPABILITIES
            | {
                HardwareCapability.CANN_MEGAMOE,
                HardwareCapability.MOE_DISPATCH_EXTRA_ARGS,
                HardwareCapability.NPU_TOP_K_TOP_P,
            },
        ),
        AscendDeviceType._310P: HardwareProfile(
            _device_type=AscendDeviceType._310P,
            attention_backend_family=AttentionBackendFamily.COMPATIBILITY,
            cpu_binding_mode=CPUBindingMode.TOPO_AFFINITY,
            default_worker_cls="vllm_ascend._310p.worker_310p.NPUWorker310",
            device_adaptor_family=DeviceAdaptorFamily.COMPATIBILITY,
            device_addressing_mode=DeviceAddressingMode.DIRECT,
            weight_layout_policy=WeightLayoutPolicy.FORCE_NZ,
            moe_comm_policy=MoECommPolicy.ALLGATHER,
            quantization_backend_family=QuantizationBackendFamily.COMPATIBILITY,
            capabilities=frozenset(
                {
                    HardwareCapability.COMPATIBILITY_OP_IMPLEMENTATIONS,
                    HardwareCapability.DISTRIBUTED_COMMUNICATION_ADAPTATION,
                    HardwareCapability.FUSED_MOE_COMPATIBILITY,
                    HardwareCapability.FUSED_SWIGLU_TUNING_ARGS,
                    HardwareCapability.GDN_COMPATIBILITY,
                    HardwareCapability.IRQ_CPU_RESERVATION,
                    HardwareCapability.RC_DEVICE_DISCOVERY,
                    HardwareCapability.RUNTIME_CUSTOM_OPS,
                }
            ),
        ),
        AscendDeviceType.A5: HardwareProfile(
            _device_type=AscendDeviceType.A5,
            requires_2d_nz_linear_input=True,
            minimax_m3_gbsa_bf16_softmax_precision=1,
            attention_backend_family=AttentionBackendFamily.STANDARD,
            cpu_binding_mode=CPUBindingMode.TOPO_AFFINITY,
            default_worker_cls=_DEFAULT_WORKER_CLS,
            device_adaptor_family=DeviceAdaptorFamily.FP8_OPTIMIZED,
            device_addressing_mode=DeviceAddressingMode.DIRECT,
            weight_layout_policy=WeightLayoutPolicy.CONFIGURABLE,
            moe_comm_policy=MoECommPolicy.CAPACITY_AND_WORLD_SIZE,
            quantization_backend_family=QuantizationBackendFamily.STANDARD,
            capabilities=frozenset(
                {
                    HardwareCapability.AUTO_ENABLE_CUSTOM_OPS,
                    HardwareCapability.SCATTER_ND_CACHE_STORE,
                    HardwareCapability.SCATTER_ND_FP8_CACHE_STORE,
                    HardwareCapability.CANN_MEGAMOE,
                    HardwareCapability.CANN_MEGAMOE_MXFP,
                    HardwareCapability.DEVICE_LENGTH_FLASH_ATTN,
                    HardwareCapability.CHUNKED_PREFILL_PHASE_SPLIT,
                    HardwareCapability.CLUSTER_CPU_TOPOLOGY,
                    HardwareCapability.DSA_C128_STATE_SMALL_BLOCK_SIZES,
                    HardwareCapability.DSV4_COMPRESSED_CACHE,
                    HardwareCapability.DSV41_PACKED_CACHE,
                    HardwareCapability.ENGRAM_UNROTATED_GATE,
                    HardwareCapability.ENGRAM_MXFP8,
                    HardwareCapability.DYNAMIC_MX_QUANT_FUSION,
                    HardwareCapability.DYNAMIC_MX_QUANT_SCALE_ALG_ONE,
                    HardwareCapability.FLA_GDN_PREFILL,
                    HardwareCapability.FP8_ATTENTION,
                    HardwareCapability.POOL_KEY_INDEXER_FP8,
                    HardwareCapability.FUSED_ROTARY_MUL_INTERLEAVE,
                    HardwareCapability.GRAPH_MM_REDUCE_SCATTER_FUSION,
                    HardwareCapability.GRAPH_MULS_ADD_FUSION,
                    HardwareCapability.GRAPH_NORM_QUANT_FUSION,
                    HardwareCapability.LOCAL_KV_COMM_RESOURCE,
                    HardwareCapability.LORA_CUSTOM_OPS,
                    HardwareCapability.MLA_FLASH,
                    HardwareCapability.MLAPO_NATIVE_WEIGHTS,
                    HardwareCapability.MINIMAX_M3_GBSA_METADATA_REUSE,
                    HardwareCapability.MOE_DISPATCH_EXTRA_ARGS,
                    HardwareCapability.MOE_DISPATCH_SHARED_EXPERT_ARGS,
                    HardwareCapability.MM_REDUCE_SCATTER_AI_CPU_INFERENCE,
                    HardwareCapability.MOE_GATING_TOP_K_HASH_VISION,
                    HardwareCapability.NPUGRAPH_EX,
                    HardwareCapability.RUNTIME_CUSTOM_OPS,
                    HardwareCapability.STANDARD_MAMBA_PATCH,
                    HardwareCapability.STANDARD_WORKER_PATCHES,
                    HardwareCapability.SWIGLU_OAI_MX_QUANT,
                    HardwareCapability.TRITON_BATCH_MEMCPY,
                    HardwareCapability.UNRESTRICTED_MLAPO,
                    HardwareCapability.NPU_TOP_K_TOP_P,
                }
            ),
        ),
        # Ascend960 (A6): silicon is close to A5. Reuse A5's validated
        # capability set minus the ATB contracts A6 does not support.
        AscendDeviceType.A6: HardwareProfile(
            _device_type=AscendDeviceType.A6,
            attention_backend_family=AttentionBackendFamily.STANDARD,
            cpu_binding_mode=CPUBindingMode.TOPO_AFFINITY,
            default_worker_cls=_DEFAULT_WORKER_CLS,
            device_adaptor_family=DeviceAdaptorFamily.A6,
            device_addressing_mode=DeviceAddressingMode.DIRECT,
            weight_layout_policy=WeightLayoutPolicy.CONFIGURABLE,
            moe_comm_policy=MoECommPolicy.CAPACITY_AND_EXPERT_DENSITY,
            quantization_backend_family=QuantizationBackendFamily.STANDARD,
            capabilities=_A6_CAPABILITIES,
        ),
    }
)


def get_hardware_profile(device_type: AscendDeviceType) -> HardwareProfile:
    """Return the immutable profile registered for ``device_type``."""

    try:
        return _HARDWARE_PROFILES[device_type]
    except KeyError as exc:
        raise RuntimeError(f"No hardware profile is registered for device type: {device_type}.") from exc


_CURRENT_HARDWARE_PROFILE = get_hardware_profile(get_device_config()._device_type)


def get_current_hardware_profile() -> HardwareProfile:
    """Return the profile selected by the current device configuration."""

    return _CURRENT_HARDWARE_PROFILE
