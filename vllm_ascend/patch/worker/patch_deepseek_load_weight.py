import vllm
import typing
from typing import Optional, Union, Any, cast
from collections.abc import Callable, Iterable

import torch
from collections.abc import Callable, Iterable
from vllm.model_executor.models.deepseek_v2 import DeepseekV2Model, DeepseekV2ForCausalLM, get_spec_layer_idx_from_weight_name
from vllm.model_executor.models.utils import is_pp_missing_parameter
from vllm.model_executor.model_loader.weight_utils import default_weight_loader, maybe_remap_kv_scale_name
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm_ascend.patch.worker.patch_num_hidden_layers import patch_num_hidden_layers
from vllm.attention import Attention
from vllm.model_executor.layers.fused_moe import FusedMoE
from vllm.model_executor.model_loader.weight_utils import (
    default_weight_loader, maybe_remap_kv_scale_name)


def remap_C8_kv_scale_name(name: str, params_dict: dict) -> Optional[str]:
    replace_scale_names = [
        "fa_q.scale", "fa_k.scale", "fa_v.scale", "fa_q.offset",
        "fa_k.offset", "fa_v.offset"
    ]
    for scale_name in replace_scale_names:
        if name.endswith(scale_name):
            remap_name = name.replace(scale_name, f"mla_attn.mla_attn.{scale_name}")
            if remap_name in params_dict:
                return remap_name
            else:
                return remap_name.replace("mla_attn", "attn")
    return name



def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
    stacked_params_mapping = [
        ("gate_up_proj", "gate_proj", 0),
        ("gate_up_proj", "up_proj", 1),
        ("fused_qkv_a_proj", "q_a_proj", 0),
        ("fused_qkv_a_proj", "kv_a_proj_with_mqa", 1),
    ]
    # Params for weights, fp8 weight scales, fp8 activation scales
    expert_params_mapping = FusedMoE.make_expert_params_mapping(
        ckpt_gate_proj_name="gate_proj",
        ckpt_down_proj_name="down_proj",
        ckpt_up_proj_name="up_proj",
        num_experts=self.config.n_routed_experts,
        num_redundant_experts=self.num_redundant_experts)
    
    params_dict = dict(self.named_parameters())
    # print(f"params_dict:{params_dict}")
    loaded_params: set[str] = set()
    for name, loaded_weight in weights:
        if "rotary_emb.inv_freq" in name:
            continue
        spec_layer = get_spec_layer_idx_from_weight_name(self.config, name)
        
        if spec_layer is not None:
            continue  # skip spec decode layers for main model

        for (param_name, weight_name, shard_id) in stacked_params_mapping:
            # Skip non-stacked layers and experts (experts handled below).
            if weight_name not in name:
                continue
            # We have mlp.experts[0].gate_proj in the checkpoint.
            # Since we handle the experts below in expert_params_mapping,
            # we need to skip here BEFORE we update the name, otherwise
            # name will be updated to mlp.experts[0].gate_up_proj, which
            # will then be updated below in expert_params_mapping
            # for mlp.experts[0].gate_gate_up_proj, which breaks load.
            if (("mlp.experts." in name) and name not in params_dict):
                continue
            name_mapped = name.replace(weight_name, param_name)
            # QKV fusion is optional, fall back to normal
            # weight loading if it's not enabled
            # if go with fusion option, then update name
            if ((param_name == "fused_qkv_a_proj")
                    and name_mapped not in params_dict):
                continue
            else:
                name = name_mapped
            # Skip loading extra bias for GPTQ models.
            if name.endswith(".bias") and name not in params_dict:
                continue

            if is_pp_missing_parameter(name, self):
                continue

            param = params_dict[name]
            weight_loader = param.weight_loader
            weight_loader(param, loaded_weight, shard_id)
            break
        else:
            is_expert_weight = False
            for mapping in expert_params_mapping:
                param_name, weight_name, expert_id, shard_id = mapping
                if weight_name not in name:
                    continue

                # Anyway, this is an expert weight and should not be
                # attempted to load as other weights later
                is_expert_weight = True

                # Do not modify `name` since the loop may continue here
                # Instead, create a new variable
                name_mapped = name.replace(weight_name, param_name)

                if is_pp_missing_parameter(name_mapped, self):
                    continue

                param = params_dict[name_mapped]
                # We should ask the weight loader to return success or not
                # here since otherwise we may skip experts with other
                # available replicas.
                weight_loader = typing.cast(Callable[..., bool],
                                            param.weight_loader)
                success = weight_loader(param,
                                        loaded_weight,
                                        name_mapped,
                                        shard_id=shard_id,
                                        expert_id=expert_id,
                                        return_success=True)
                if success:
                    name = name_mapped
                    break
            else:
                if is_expert_weight:
                    # We've checked that this is an expert weight
                    # However it's not mapped locally to this rank
                    # So we simply skip it
                    continue

                # Skip loading extra bias for GPTQ models.
                if name.endswith(".bias") and name not in params_dict:
                    continue

                # Remapping the name of FP8 kv-scale.
                name = maybe_remap_kv_scale_name(name, params_dict)
                name = remap_C8_kv_scale_name(name, params_dict)
                if name is None:
                    continue

                if is_pp_missing_parameter(name, self):
                    continue

                param = params_dict[name]
                weight_loader = getattr(param, "weight_loader",
                                        default_weight_loader)
                weight_loader(param, loaded_weight)
        loaded_params.add(name)
    if self.quant_config.is_enable_fa_quant():
        # param.split(".fa_q.scale")
        fa_quant_layers = {
            param.split(".fa_q.scale")[0]
            for param in loaded_params if "fa_q.scale" in param
        }
        modules_dict = dict(self.named_modules())
        for module_name, module in modules_dict.items():
            if isinstance(module, Attention):
                if module_name in fa_quant_layers:
                    module.dtype = torch.float8_e4m3fn
                    # Due to the existence of the fallback layer, new attributes are added to distinguish
                    setattr(module, "fa_quant_layer", True)
                else:
                    setattr(module, "fa_quant_layer", False)

    return loaded_params

DeepseekV2ForCausalLM.load_weights = load_weights
DeepseekV2Model.__init__ = patch_num_hidden_layers(DeepseekV2Model, NUM_HIDDEN_LAYERS)