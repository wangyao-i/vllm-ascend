#
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.
# This file is a part of the vllm-ascend project.
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


from typing import Any, Callable, Dict, Optional, Tuple, Union, TypeVar

import torch
import torch_npu
from vllm.config import CompilationMode, get_current_vllm_config
from vllm.distributed import get_ep_group, get_tensor_model_parallel_rank, get_tensor_model_parallel_world_size
from vllm.forward_context import get_forward_context, ForwardContext

from vllm_ascend.ascend_config import get_ascend_config
from vllm_ascend.distributed.parallel_state import get_mc2_group
from vllm_ascend.ops.fused_moe.experts_selector import select_experts
from vllm_ascend.attention.mla_v1 import AscendMLAMetadata
from vllm_ascend.models.layers.mla import AscendMLAModules
from vllm_ascend.ops.weight_prefetch import maybe_npu_prefetch
from vllm_ascend.attention.attention_v1 import AscendAttentionState
from vllm_ascend.compilation.acl_graph import (get_graph_params,
                                               update_graph_params_workspaces)


GROUP_SIZE = 32


class AscendW8A8MXFP8DynamicLinearMethod:
    """Linear method for Ascend W8A8_DYNAMIC.
    """
    model_dtype = None


    def __init__(self):
        self.transpose_weight = True


    @staticmethod
    def get_weight(input_size: int, output_size: int,
                   params_dtype: torch.dtype) -> Dict[str, Any]:
        params_dict = {
            "weight": torch.empty(output_size, input_size, dtype=torch.float8_e4m3fn)
        }
        return params_dict

    @staticmethod
    def get_pertensor_param(params_dtype: torch.dtype) -> Dict[str, Any]:
        return {}

    @staticmethod
    def get_perchannel_param(
            output_size: int,
            params_dtype: torch.dtype,
    ) -> Dict[str, Any]:
        return {}

    def get_pergroup_param(self, input_size: int, output_size: int,
                           params_dtype: torch.dtype, layer_type: Optional[str] = None) -> Dict[str, Any]:
        params_dict = {}
        params_dict["weight_scale"] = torch.empty(
            output_size, input_size // GROUP_SIZE, dtype=torch.uint8)
        return params_dict

    @staticmethod
    def apply(
            layer: torch.nn.Module,
            x: Union[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]],
            bias: Optional[torch.Tensor] = None,
            tp_rank: Optional[int] = 0,
    ) -> torch.Tensor:

        quantized_x, dynamic_scale = torch_npu.npu_dynamic_mx_quant(x, dst_type=torch.float8_e4m3fn)
        pertoken_scale = dynamic_scale
        output_dtype = x.dtype

        output = torch_npu.npu_quant_matmul(
            quantized_x,
            layer.weight,
            layer.weight_scale,
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale=pertoken_scale,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            bias=bias,
            output_dtype=output_dtype,
            group_sizes=[1, 1, GROUP_SIZE]
        )

        return output

    def process_weights_after_loading(self, layer):
        n_dim, k_dim = layer.weight_scale.data.shape
        layer.weight_scale.data = layer.weight_scale.data.reshape(n_dim, k_dim//2, 2)
        if self.transpose_weight:
            layer.weight.data = layer.weight.data.transpose(0, 1)
            layer.weight_scale.data = layer.weight_scale.data.transpose(0, 1)


class AscendW8A8MXFP8DynamicFusedMoEMethod:
    """FusedMoe method for Ascend W8A8_DYNAMIC.
    """
    model_dtype = None

    def __init__(self):
        self.transpose_weight = True

        self.ep_group = get_ep_group()

        vllm_config = get_current_vllm_config()
        ascend_config = get_ascend_config()
        self.use_aclgraph = (
                vllm_config.compilation_config.level == CompilationMode.VLLM_COMPILE
                and not vllm_config.model_config.enforce_eager
                and not ascend_config.torchair_graph_config.enabled)
        self.dynamic_eplb = ascend_config.dynamic_eplb or ascend_config.expert_map_record_path

    @staticmethod
    def get_weight(num_experts: int, intermediate_size_per_partition: int,
                   hidden_sizes: int,
                   params_dtype: torch.dtype) -> Dict[str, Any]:
        param_dict = {}
        param_dict["w13_weight"] = torch.empty(num_experts,
                                               2 * intermediate_size_per_partition,
                                               hidden_sizes,
                                               dtype=torch.float8_e4m3fn)
        param_dict["w2_weight"] = torch.empty(num_experts,
                                              hidden_sizes,
                                              intermediate_size_per_partition,
                                              dtype=torch.float8_e4m3fn)
        return param_dict

    @staticmethod
    def get_dynamic_quant_param(num_experts: int,
                                intermediate_size_per_partition: int,
                                hidden_sizes: int,
                                params_dtype: torch.dtype) -> Dict[str, Any]:
        param_dict = {}
        param_dict["w13_weight_scale"] = torch.empty(
            num_experts,
            2 * intermediate_size_per_partition,
            hidden_sizes // GROUP_SIZE,
            dtype=torch.uint8)

        param_dict["w2_weight_scale"] = torch.empty(num_experts,
                                                    hidden_sizes,
                                                    intermediate_size_per_partition // GROUP_SIZE,
                                                    dtype=torch.uint8)
        return param_dict

    def apply(
            self,
            layer: torch.nn.Module,
            x: torch.Tensor,
            router_logits: torch.Tensor,
            top_k: int,
            renormalize: bool,
            use_grouped_topk: bool = False,
            global_num_experts: int = -1,
            expert_map: Optional[torch.Tensor] = None,
            topk_group: Optional[int] = None,
            num_expert_group: Optional[int] = None,
            custom_routing_function: Optional[Callable] = None,
            scoring_func: str = "softmax",
            e_score_correction_bias: Optional[torch.Tensor] = None,
            is_prefill: bool = True,
            enable_force_load_balance: bool = True,
            log2phy: torch.Tensor = None,
            global_redundant_expert_num: int = 0,
            shared_experts: Optional[Any] = None,
            quantized_x_for_share: Optional[Any] = None,
            dynamic_scale_for_share: Optional[Any] = None,
            **kwargs,
    ) -> torch.Tensor:
        assert router_logits.shape[
                   1] == global_num_experts - global_redundant_expert_num, "Number of global experts mismatch (excluding redundancy)"
        topk_weights, topk_ids = select_experts(
            hidden_states=x,
            router_logits=router_logits,
            top_k=top_k,
            use_grouped_topk=use_grouped_topk,
            renormalize=renormalize,
            topk_group=topk_group,
            num_expert_group=num_expert_group,
            custom_routing_function=custom_routing_function,
            scoring_func=scoring_func,
            e_score_correction_bias=e_score_correction_bias,
            global_num_experts=global_num_experts)

        # this is a naive implementation for experts load balance so as
        # to avoid accumulating too much tokens on a single rank.
        # currently it is only activated when doing profile runs.
        if enable_force_load_balance:
            topk_ids = torch.randint_like(topk_ids, 0, global_num_experts)

        topk_weights = topk_weights.to(x.dtype)

        moe_comm_method = get_forward_context().moe_comm_method
        return moe_comm_method.fused_experts(
            hidden_states=x,
            w1=layer.w13_weight,
            w1_scale=layer.w13_weight_scale,
            w2=layer.w2_weight,
            w2_scale=layer.w2_weight_scale,
            topk_weights=topk_weights,
            topk_ids=topk_ids,
            use_int8_w8a8=False,
            expert_map=expert_map,
            log2phy=log2phy,
            global_redundant_expert_num=global_redundant_expert_num,
            shared_experts=shared_experts,
            quantized_x_for_share=quantized_x_for_share,
            dynamic_scale_for_share=dynamic_scale_for_share,
            dynamic_eplb=self.dynamic_eplb,
            mc2_mask=kwargs.get("mc2_mask", None),
            use_A5_quant=True,
            use_fp8_comm=True,
            act_quant_type=torch.float8_e4m3fn,
            weight_quant_type=torch.float8_e4m3fn,
            scale_type=torch_npu.float8_e8m0fnu,
            per_token_scale_type=torch_npu.float8_e8m0fnu,
            comm_quant_mode=4)

    def process_weights_after_loading(self, layer):
        g_num, n_size, k_size = layer.w13_weight_scale.shape
        layer.w13_weight_scale.data = layer.w13_weight_scale.data.reshape(g_num, n_size, k_size//2, 2)
        g_num, n_size, k_size = layer.w2_weight_scale.shape
        layer.w2_weight_scale.data = layer.w2_weight_scale.data.reshape(g_num, n_size, k_size//2, 2)
        if self.transpose_weight:
            # FIXME(linfeng): currently we have to force contiguous here for weight and weight_scale of GMM.
            # Have to investigate performance impact and root cause.
            layer.w13_weight.data = layer.w13_weight.data.transpose(
                1, 2)
            layer.w2_weight.data = layer.w2_weight.data.transpose(
                1, 2)
            layer.w13_weight_scale.data = layer.w13_weight_scale.data.transpose(1, 2)
            layer.w2_weight_scale.data = layer.w2_weight_scale.data.transpose(1, 2)

M = TypeVar("M", bound=AscendMLAMetadata)

class AscendFAQuantAttentionMethod:

    def __init__(self):
        self.transpose_weight = True
        self.printFlag = False

    def create_weights(self, layer: torch.nn.Module) -> None:
        extra_module_names = ["fa_q", "fa_k", "fa_v"]
        for name in extra_module_names:
            setattr(layer, name, torch.nn.Module())

        params_dict = {}
        dtype = torch.get_default_dtype()

        params_dict["fa_q.scale"] = torch.empty((layer.num_heads, 1),
                                                dtype=torch.float32)
        params_dict["fa_k.scale"] = torch.empty((layer.num_kv_heads, 1),
                                                dtype=torch.float32)
        params_dict["fa_v.scale"] = torch.empty((layer.num_kv_heads, 1),
                                                dtype=torch.float32)
        params_dict["fa_q.offset"] = torch.empty((layer.num_heads, 1),
                                                 dtype=torch.int8)
        params_dict["fa_k.offset"] = torch.empty((layer.num_kv_heads, 1),
                                                 dtype=torch.int8)
        params_dict["fa_v.offset"] = torch.empty((layer.num_kv_heads, 1),
                                                 dtype=torch.int8)
        for name, weight in params_dict.items():
            module_name, weight_name = name.split('.')
            module = getattr(layer, module_name)
            weight_param = torch.nn.Parameter(weight, requires_grad=False)
            module.register_parameter(weight_name, weight_param)
            setattr(weight_param, "weight_loader", weight_loader)

    def process_weights_after_loading(self, layer: torch.nn.Module) -> None:
        if not hasattr(layer,"fa_quant_layer") or layer.fa_quant_layer is False:
            return
        fa_qscale = layer.fa_q.scale
        fa_kscale = layer.fa_k.scale
        fa_vscale = layer.fa_v.scale
        head_size = 1 if layer.use_mla else layer.head_size
        
        layer.fa_qscale = 1.0 / torch.nn.Parameter(layer.fa_q.scale,
                                                   requires_grad=False)
        layer.fa_qscale = layer.fa_qscale.transpose(-2, -1).contiguous().unsqueeze(0)
        layer.quant_qscale = torch.nn.Parameter(layer.fa_q.scale,
                                                   requires_grad=False)

        repeated_fa_kscale = torch.squeeze(layer.fa_k.scale).unsqueeze(0)
        layer.fa_kscale = torch.nn.Parameter(repeated_fa_kscale.to(torch.float),
                                             requires_grad=False)
        layer.quant_kscale = torch.nn.Parameter(repeated_fa_kscale.to(torch.bfloat16),
                                             requires_grad=False)
        repeated_fa_kscale_perdim = repeated_fa_kscale.repeat(512)
        layer.quant_kscale_perdim = torch.nn.Parameter(repeated_fa_kscale_perdim.to(torch.float),
                                             requires_grad=False)

        repeated_fa_vscale = torch.squeeze(layer.fa_v.scale).unsqueeze(0)
        layer.fa_vscale = torch.nn.Parameter(repeated_fa_vscale.to(torch.float),
                                             requires_grad=False)
        repeated_fa_vscale_perdim = repeated_fa_vscale.repeat(64)
        layer.quant_vscale_perdim = torch.nn.Parameter(repeated_fa_vscale_perdim.to(torch.float),
                                             requires_grad=False)

        repeated_query_offset = layer.fa_q.offset.repeat(1, head_size)
        layer.fa_qoffset = torch.nn.Parameter(repeated_query_offset.to(torch.float),
                                              requires_grad=False)
        repeated_fa_koffset = torch.squeeze(layer.fa_k.offset).unsqueeze(0)
        layer.fa_kvoffset = torch.nn.Parameter(repeated_fa_koffset.to(torch.float),
                                               requires_grad=False)

        if fa_kscale.shape[0] <= 0:
            raise ValueError(
                "Expected size of fa_kscale in dimension 0 should be greater than 0"
                f"but got {fa_kscale.shape[0]}.")
        gqa_size = fa_qscale.shape[0] // fa_kscale.shape[0]
        fa3_k_scale, fa3_v_scale = fa_kscale.repeat(1, gqa_size).view(
            -1, 1), fa_vscale.repeat(1, gqa_size).view(-1, 1)
        layer.qk_scale = torch.nn.Parameter(torch.squeeze(
            fa_qscale * fa3_k_scale).to(torch.float),
                                            requires_grad=False)
        layer.fa3_k_scale = torch.nn.Parameter(
            torch.squeeze(fa3_k_scale).contiguous().to(torch.float),
            requires_grad=False)
        layer.fa3_v_scale = torch.nn.Parameter(
            torch.squeeze(fa3_v_scale).contiguous().to(torch.float),
            requires_grad=False)
        

    def apply(self, layer: torch.nn.Module, hidden_states: torch.Tensor,
              kv_cache: Tuple[torch.Tensor], attn_metadata: M, mla_module: AscendMLAModules, need_gather_q_kv: bool = False,
        
        output: Optional[torch.Tensor] = None,) -> torch.Tensor:
        forward_context = get_forward_context()
        if not forward_context.with_prefill:
            self.mla_decode_apply_aclnn(layer, hidden_states, kv_cache, attn_metadata, mla_module, need_gather_q_kv, output)
            return output
        layer_impl = None if getattr(layer, "fa_quant_layer", False) == False else layer
        layer.impl.forward(layer.layer_name, hidden_states, kv_cache, attn_metadata,
                            need_gather_q_kv, output, layer_impl)
        return output

    def exec_kv_decode(
        self,
        impl,
        kv_no_split: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        kv_cache: Tuple,
        slots: torch.Tensor,
        layer
    ):
        B = kv_no_split.shape[0]
        N = impl.num_kv_heads
        S = 1
        kv_no_split = kv_no_split.view(
            B, N, S, impl.kv_lora_rank + impl.qk_rope_head_dim)
        k_scale = None if layer is None else layer.quant_kscale_perdim
        k_nope,k_pe = torch.split(kv_no_split, [512, 64], dim=-1)
        k_nope, _ = torch_npu.npu_rms_norm(k_nope, impl.kv_a_layernorm.weight, impl.kv_a_layernorm.variance_epsilon)
        k_pe = torch_npu.npu_interleave_rope(k_pe,cos,sin)
        quant_k_nope = torch_npu.npu_quantize(input=k_nope,scales=k_scale,zero_points=None,dtype=kv_cache[0].dtype,axis=-1)
        torch_npu.npu_scatter_pa_kv_cache(key=quant_k_nope.squeeze(1), value=quant_k_nope.squeeze(1), slot_mapping=slots.to(torch.int64), key_cache=kv_cache[0],value_cache= kv_cache[0])
        torch_npu.npu_scatter_pa_kv_cache(key=k_pe.squeeze(1), value=k_pe.squeeze(1), slot_mapping=slots.to(torch.int64), key_cache=kv_cache[1],value_cache= kv_cache[1])

        return kv_cache[1], kv_cache[0]


    def _mla_preprocess(self, impl, hidden_states, kv_cache,
                        attn_metadata, need_gather_q_kv, layer = None):
        # MLA Preprocess:
        # 1. Perform q_a_proj and q_a_layernorm to obtain q_c
        # 2. Perform kv_a_proj_with_mqa to obtain kv_no_split
        # 3. If need_gather_q_kv, perform all_gather.
        # 4. Preprocess decode tokens, write kv cache and get:
        # decode_ql_nope, decode_q_pe, decode_k_pe, decode_k_nope
        # 5. Preprocess prefill tokens, write kv cache and get:
        num_decode_tokens = attn_metadata.num_decode_tokens
        num_actual_tokens = attn_metadata.num_actual_tokens
        if impl.fused_qkv_a_proj is not None:
            maybe_npu_prefetch(inputs=impl.fused_qkv_a_proj.weight,
                               dependency=hidden_states,
                               enabled=impl.enable_prefetch)
            qkv_lora = impl.fused_qkv_a_proj(hidden_states)[0]
            q_c, kv_no_split = qkv_lora.split(
                [impl.q_lora_rank, impl.kv_lora_rank + impl.qk_rope_head_dim],
                dim=-1,
            )
            q_c = impl.q_a_layernorm(q_c)
        else:
            q_c = hidden_states
            kv_no_split = impl.kv_a_proj_with_mqa(hidden_states)[0]
        # Process for Flash Comm V1
        q_c = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            q_c, need_gather_q_kv)
        kv_no_split = torch.ops.vllm.maybe_all_gather_and_maybe_unpad(
            kv_no_split, need_gather_q_kv)
        # Preprocess for decode tokens
        decode_q_c = q_c[:num_decode_tokens]
        cos = attn_metadata.decode.cos
        sin = attn_metadata.decode.sin
        decode_ql_nope, decode_q_pe = \
            impl._q_proj_and_k_up_proj(decode_q_c)
        decode_q_pe = impl.rope_single(decode_q_pe, cos, sin)
        decode_slots = attn_metadata.slot_mapping[:num_decode_tokens]
        decode_kv_no_split = kv_no_split[:num_decode_tokens]
        decode_k_pe, decode_k_nope = self.exec_kv_decode(impl,
            decode_kv_no_split, cos, sin, kv_cache, decode_slots,layer)
        return decode_ql_nope, decode_q_pe, decode_k_nope, decode_k_pe
    
    
    def _mla_decode(self, impl, layer,
        q_nope: torch.Tensor,
        q_pe: torch.Tensor,
        k_nope: torch.Tensor,
        k_pe: torch.Tensor,
        block_size: int,
        attn_metadata: AscendMLAMetadata,
    ) -> torch.Tensor:
        decode_meta = attn_metadata.decode
        assert decode_meta is not None
        num_tokens = q_nope.size(0)
        # shape of knope/k_pe for npu graph mode should be:
        # [num_blocks, num_kv_heads, block_size, self.kv_lora_rank/self.qk_rope_head_dim]
        actual_seq_lengths = None
        if impl.enable_kv_nz:
            k_nope = k_nope.view(-1, impl.num_kv_heads,
                                 impl.kv_lora_rank // 32, block_size, 32)
            k_pe = k_pe.view(-1, impl.num_kv_heads,
                             impl.qk_rope_head_dim // 16, block_size, 16)
            input_layout = "BSND"
        else:
            k_nope = k_nope.view(-1, impl.num_kv_heads, block_size,
                                 impl.kv_lora_rank)
            k_pe = k_pe.view(-1, impl.num_kv_heads, block_size,
                             impl.qk_rope_head_dim)
            input_layout = "BNSD"

        if attn_metadata.attn_state in [
                AscendAttentionState.SpecDecoding,
                AscendAttentionState.ChunkedPrefill,
                AscendAttentionState.DecodeOnly,
        ] and impl.speculative_config is not None:
            # Use TND layout for pure SpecDecoding and SpecDecoding in ChunkedPrefill
            input_layout = "TND"
            # [bs * q_seq_len, num_heads_per_rank, dim]
            # TODO: If the driver is upgraded later, the contiguous function can be deleted.
            q_nope = q_nope.view(num_tokens, impl.num_heads, -1).contiguous()
            q_pe = q_pe.view(num_tokens, impl.num_heads, -1)
            sparse_mode = 3
            spec_attn_mask = attn_metadata.decode.attn_mask  # type:ignore
            actual_seq_lengths = decode_meta.actual_seq_lengths_q
        else:
            if impl.enable_kv_nz:
                q_nope = q_nope.view(num_tokens, 1, impl.num_heads,
                                     -1).contiguous()
                q_pe = q_pe.view(num_tokens, 1, impl.num_heads, -1)
            else:
                q_nope = q_nope.view(num_tokens, impl.num_heads, 1,
                                     -1).contiguous()
                q_pe = q_pe.view(num_tokens, impl.num_heads, 1, -1)
            sparse_mode = 0
            spec_attn_mask = None
            # [bs * q_seq_len, num_heads_per_rank, dim]
            # TODO: If the driver is upgraded later, the contiguous function can be deleted.
        common_kwargs = {
            'query_rope': q_pe,
            'key_rope': k_pe,
            'num_heads': impl.num_heads,
            'num_key_value_heads': impl.num_kv_heads,
            'input_layout': input_layout,
            'atten_mask': spec_attn_mask,
            'sparse_mode': sparse_mode,
            'scale': impl.scale,
            'antiquant_mode': 0,
            'antiquant_scale': None,
            'block_table': decode_meta.block_table,
            'block_size': block_size,
            "actual_seq_lengths": actual_seq_lengths,
            "actual_seq_lengths_kv": decode_meta.seq_lens_list,
        }
        graph_params = get_graph_params()
        forward_context: ForwardContext = get_forward_context()
        if forward_context.capturing:
            stream = torch_npu.npu.current_stream()

            event = torch.npu.ExternalEvent()
            event.wait(stream)
            event.reset(stream)
            graph_params.events[num_tokens].append(event)

            workspace = graph_params.workspaces.get(num_tokens)
            if workspace is None:
                workspace = torch_npu._npu_fused_infer_attention_score_get_max_workspace(
                    q_nope, k_nope, k_nope, **common_kwargs)
                update_graph_params_workspaces(num_tokens,
                                               weak_ref_tensors(workspace))

            attn_output = torch.empty_like(q_nope)
            softmax_lse = torch.empty(num_tokens,
                                      dtype=q_nope.dtype,
                                      device=q_nope.device)

            graph_params.attn_params[num_tokens].append(
                (weak_ref_tensors(q_nope), weak_ref_tensors(k_nope),
                 weak_ref_tensors(q_pe), weak_ref_tensors(k_pe),
                 impl.num_heads, impl.num_kv_heads, input_layout,
                 weak_ref_tensors(spec_attn_mask) if spec_attn_mask is not None
                 else None, sparse_mode, impl.scale, decode_meta.block_table,
                 block_size, decode_meta.seq_lens_list, actual_seq_lengths,
                 weak_ref_tensors(attn_output), weak_ref_tensors(softmax_lse)))

            torch.npu.graph_task_group_begin(stream)
            torch_npu.npu_fused_infer_attention_score.out(
                q_nope,
                k_nope,
                k_nope,
                **common_kwargs,
                workspace=workspace,
                out=[attn_output, softmax_lse])
            handle = torch.npu.graph_task_group_end(stream)
            graph_params.handles[num_tokens].append(handle)
        else:
            q_nope, pertoken_scale = torch_npu.npu_dynamic_quant(q_nope,dst_type=torch.float8_e4m3fn)
            common_kwargs_v2 = {
            'query_rope': q_pe,
            'key_rope': k_pe,
            'num_query_heads': impl.num_heads,
            'num_key_value_heads': impl.num_kv_heads,
            'input_layout': input_layout,
            'atten_mask': spec_attn_mask,
            'sparse_mode': sparse_mode,
            'softmax_scale': impl.scale,
            'query_quant_mode': 3,
            'key_quant_mode': 0,
            'value_quant_mode': 0,
            'dequant_scale_query': pertoken_scale,
            'dequant_scale_key': layer.fa_kscale,
            'dequant_scale_value': layer.fa_vscale,
            'block_table': decode_meta.block_table,
            'block_size': block_size,
            "actual_seq_qlen": actual_seq_lengths,
            "actual_seq_kvlen": decode_meta.seq_lens_list,
        }
            decode_meta.seq_lens_list[1] = decode_meta.seq_lens_list[0]
            attn_output, _ = torch_npu.npu_fused_infer_attention_score_v2(
                q_nope, k_nope, k_nope, **common_kwargs_v2)

        return impl._v_up_proj(attn_output)

    def mla_decode_apply_aclnn(self, layer: torch.nn.Module, hidden_states: torch.Tensor,
              kv_cache: Tuple[torch.Tensor], attn_metadata: M, mla_module: AscendMLAModules, need_gather_q_kv: bool = False,
                output: Optional[torch.Tensor] = None,) -> torch.Tensor: # 面向 DeepseekV2DecoderLayer
        impl = layer.impl
        num_actual_tokens = attn_metadata.num_actual_tokens
        num_decode_tokens = attn_metadata.num_decode_tokens
        output_padded = output
        o_proj_input_shape = (get_forward_context().num_tokens,
                              impl.num_heads * impl.v_head_dim)
        o_proj_input = torch.empty(o_proj_input_shape,
                                   dtype=hidden_states.dtype,
                                   device=hidden_states.device)
        q_nope, q_pe, k_nope, k_pe = self._mla_preprocess(
                impl, hidden_states, kv_cache, attn_metadata,
                need_gather_q_kv,layer)


        o_proj_input[:num_decode_tokens] = self._mla_decode(impl, layer, q_nope, q_pe, k_nope,
                                                        k_pe, kv_cache[0].shape[1], attn_metadata)

        MAX_O_PROJ_PREFETCH_SIZE = 16 * 1024 * 1024
        maybe_npu_prefetch(inputs=impl.o_proj.weight,
                           dependency=o_proj_input,
                           max_size=MAX_O_PROJ_PREFETCH_SIZE,
                           enabled=impl.enable_prefetch)

        output[...] = impl.o_proj(o_proj_input,
                                  is_prefill=False)[0]
        del o_proj_input

        return output_padded
        
        

