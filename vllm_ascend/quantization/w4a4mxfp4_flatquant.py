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
import math
from typing import Any, Dict, Optional, Tuple

import torch
import torch_npu

GROUP_SIZE = 32


def get_decompose_dim(n):
    a = int(math.sqrt(n))
    if a * a < n:
        a += 1
    while True:
        tmp = a * a - n
        b = int(math.sqrt(tmp))
        if b * b == tmp:
            break
        a += 1
    return a - b, a + b


class AscendW4A4FpFlatQuantDynamicLinearMethod:
    """Linear method for Ascend W4A4_MXFP4_FLATQUANT_DYNAMIC."""
    input_size = 0

    def __init__(self):
        self.transpose_weight = True
        self.sym = True

    @staticmethod
    def get_weight(input_size: int, output_size: int, params_dtype: torch.dtype) -> Dict[str, Any]:
        if input_size % 2 != 0:
            raise ValueError(f"input_size ({input_size}) must be divisible by 2 for int4 packing")
        AscendW4A4FpFlatQuantDynamicLinearMethod.input_size = input_size
        params_dict = {
            "weight": torch.empty(output_size, input_size, dtype=torch.int8)
        }
        return params_dict

    @staticmethod
    def get_pertensor_param(params_dtype: torch.dtype) -> Dict[str, Any]:
        params_dict = {}
        left_trans_dim, right_trans_dim = get_decompose_dim(
            AscendW4A4FpFlatQuantDynamicLinearMethod.input_size)
        params_dict["left_trans"] = torch.empty(left_trans_dim,
                                                left_trans_dim,
                                                dtype=params_dtype)
        params_dict["right_trans"] = torch.empty(right_trans_dim,
                                                 right_trans_dim,
                                                 dtype=params_dtype)
        params_dict["clip_ratio"] = torch.empty(1, dtype=torch.float32)
        return params_dict

    @staticmethod
    def get_perchannel_param(output_size: int, params_dtype: torch.dtype) -> Dict[str, Any]:
        return {}

    def get_pergroup_param(self,
                           input_size: int,
                           output_size: int,
                           params_dtype: torch.dtype,
                           layer_type: Optional[str] = None) -> Dict[str, Any]:
        if layer_type == "row":
            raise ValueError("Tensor Parallelism is not supported for rotated weight tensors"
                             "of 'o_proj' or 'down_proj' layers.")
        params_dict = {}
        params_dict["weight_scale"] = torch.empty(output_size,
                                                  input_size // GROUP_SIZE,
                                                  dtype=torch.uint8)
        return params_dict

    @staticmethod
    def apply(
        layer: torch.nn.Module,
        x: torch.Tensor,
        bias: Optional[torch.Tensor] = None,
        tp_rank: Optional[int] = 0,
    ) -> torch.Tensor:
        original_dtype = x.dtype
        input_shape = x.shape
        in_features = input_shape[-1]
        left_dim = layer.left_trans.shape[0]
        right_dim = layer.right_trans.shape[0]
        if left_dim * right_dim != in_features:
            raise ValueError(
                f"FlatQuant transform matrices dimension mismatch: "
                f"left_dim({left_dim}) * right_dim({right_dim}) != in_features({in_features})"
            )
        x_reshaped = x.view(-1, left_dim, right_dim)

        clip_ratio = 1.0
        x_quantized_fp4, pertoken_scale = torch_npu.npu_kronecker_quant(
            x_reshaped, layer.left_trans, layer.right_trans, clip_ratio, dst_dtype=torch_npu.float4_e2m1fn_x2
        )

        output = torch_npu.npu_quant_matmul(
            x_quantized_fp4,
            layer.weight,
            layer.weight_scale,
            scale_dtype=torch_npu.float8_e8m0fnu,
            pertoken_scale=pertoken_scale,
            pertoken_scale_dtype=torch_npu.float8_e8m0fnu,
            bias=bias,
            output_dtype=original_dtype,
            x1_dtype=torch_npu.float4_e2m1fn_x2,
            x2_dtype=torch_npu.float4_e2m1fn_x2,
            group_sizes=[1, 1, GROUP_SIZE]
        )
        output = output.view(*input_shape[:-1], -1)
        return output

    def process_weights_after_loading(self, layer):
        layer.weight.data = torch_npu.npu_dtype_cast(layer.weight.data, torch_npu.float4_e2m1fn_x2)
        layer.weight_scale.data = layer.weight_scale.data.view(-1, layer.weight_scale.shape[-1] // 2, 2)
        if self.transpose_weight:
            layer.weight.data = layer.weight.data.transpose(0, 1)
            layer.weight_scale.data = layer.weight_scale.data.transpose(0, 1)

        layer.left_trans = torch.nn.Parameter(layer.left_trans.data.t().contiguous())
        layer.right_trans = torch.nn.Parameter(layer.right_trans.data)
