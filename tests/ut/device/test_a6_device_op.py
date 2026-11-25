# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

from types import SimpleNamespace
from unittest import mock

import pytest
import torch

from vllm_ascend.device import device_op
from vllm_ascend.device.device_op import A6DeviceAdaptor
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.device.hardware_profile import get_hardware_profile


def test_a6_moe_unpermute_preserves_invalid_routes():
    tokens = torch.zeros(2, 4, dtype=torch.bfloat16)
    indices = torch.tensor([0, -1], dtype=torch.int32)
    probs = torch.tensor([[1, 0]], dtype=torch.bfloat16)

    with mock.patch.object(device_op.torch_npu, "npu_moe_token_unpermute") as unpermute:
        result = A6DeviceAdaptor.npu_moe_token_unpermute(tokens, indices, probs)

    unpermute.assert_called_once()
    assert unpermute.call_args.kwargs["sorted_indices"] is indices
    assert result is unpermute.return_value


@pytest.mark.parametrize("row_stride", [32, 64])
@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16, torch.float32])
def test_a6_scatter_preserves_original_views(row_stride, dtype):
    storage = torch.zeros(8, row_stride, dtype=dtype)
    cache = storage[:, :32]
    indices = torch.tensor([[1], [3]], dtype=torch.int32)
    updates = torch.ones(2, 32, dtype=dtype)
    with (
        mock.patch.object(
            device_op, "get_current_hardware_profile", return_value=get_hardware_profile(AscendDeviceType.A6)
        ),
        mock.patch.object(device_op.torch_npu, "npu_scatter_nd_update_", create=True) as generic,
        mock.patch.object(torch.ops._C_ascend, "npu_scatter_nd_update_sk", create=True) as sk,
    ):
        A6DeviceAdaptor.scatter_cache(cache, indices, updates)
    sk.assert_not_called()
    generic.assert_called_once()
    assert all(actual is expected for actual, expected in zip(generic.call_args.args, (cache, indices, updates)))
    assert cache.stride() == (row_stride, 1)
    assert cache.data_ptr() == storage.data_ptr()


@pytest.mark.parametrize("page_step", [1, 2])
def test_a6_indexer_adapts_v2_offsets_and_weights(page_step):
    query = torch.zeros(3, 32, 128, dtype=torch.bfloat16)
    key = torch.zeros(2 * page_step, 128, 1, 128, dtype=torch.bfloat16)[::page_step]
    key_stride = key.stride()
    weights = torch.zeros(3, 32, dtype=torch.bfloat16)
    cumulative_query = torch.tensor([1, 3], dtype=torch.int64)
    lengths = torch.tensor([128, 256], dtype=torch.int64)
    block_table = torch.tensor([[0, 0], [0, 1]], dtype=torch.int32)
    expected = torch.zeros(3, 1, 2048, dtype=torch.int32)
    with (
        mock.patch.object(
            torch.ops._C_ascend, "npu_lightning_indexer_v2", create=True, return_value=(expected, None)
        ) as indexer,
        mock.patch.object(torch.ops._C_ascend, "npu_lightning_indexer_v2_metadata", create=True) as metadata,
    ):
        result = A6DeviceAdaptor.indexer_select_post_process(
            query,
            None,
            None,
            weights,
            [key],
            0,
            None,
            SimpleNamespace(block_table=block_table),
            cumulative_query,
            lengths,
            False,
            False,
            False,
        )
    assert result is expected
    indexer.assert_called_once()
    metadata.assert_called_once()
    call = indexer.call_args
    assert call.args[0] is query and call.args[1] is key
    assert key.stride() == key_stride
    assert call.args[3] == 2048
    assert metadata.call_args.args == (32, 1, 128, 2048)
    assert call.args[2].dtype == torch.float32
    assert call.kwargs["block_table"] is block_table
    assert call.kwargs["layout_k"] == "PA_BBND"
    assert call.kwargs["layout_q"] == "TND"
    assert call.kwargs["metadata"] is metadata.return_value
    torch.testing.assert_close(call.kwargs["cu_seqlens_q"], torch.tensor([0, 1, 3], dtype=torch.int32))
    torch.testing.assert_close(call.kwargs["seqused_k"], lengths.to(torch.int32))
    torch.testing.assert_close(cumulative_query, torch.tensor([1, 3], dtype=torch.int64))


def test_a6_adaptor_and_capabilities():
    profile = get_hardware_profile(AscendDeviceType.A6)
    with mock.patch.object(device_op, "get_current_hardware_profile", return_value=profile):
        assert device_op.get_device_adaptor() is A6DeviceAdaptor
    assert profile.supports(device_op.HardwareCapability.RUNTIME_CUSTOM_OPS)
    assert profile.supports(device_op.HardwareCapability.DYNAMIC_MX_QUANT_FUSION)
    assert not profile.supports(device_op.HardwareCapability.SCATTER_ND_CACHE_STORE)
    assert not profile.supports(device_op.HardwareCapability.GRAPH_MULS_ADD_FUSION)
    assert not profile.supports(device_op.HardwareCapability.ADD_RMS_NORM_BIAS_OP)
    assert not profile.supports(device_op.HardwareCapability.GEMMA_RMS_NORM_OP)
    assert not profile.supports(device_op.HardwareCapability.SPLIT_QKV_RMSNORM_MROPE)


def test_a6_fp8_indexer_keeps_delivered_operator(monkeypatch):
    monkeypatch.delenv("A6_IN_TREE_INDEXER", raising=False)
    monkeypatch.delenv("A6_QLI_DBG", raising=False)
    query = torch.zeros(3, 32, 128, dtype=torch.float8_e4m3fn)
    key = torch.zeros(2, 128, 1, 128, dtype=torch.float8_e4m3fn)
    query_scale = torch.ones(3, 32)
    key_scale = torch.ones(2, 128, 1, 1)
    weights = torch.zeros(3, 32, dtype=torch.bfloat16)
    cumulative_query = torch.tensor([1, 3], dtype=torch.int32)
    lengths = torch.tensor([128, 256], dtype=torch.int32)
    block_table = torch.tensor([[0, 0], [0, 1]], dtype=torch.int32)
    expected = torch.zeros(3, 1, 2048, dtype=torch.int32)
    operator = mock.Mock(return_value=(expected, None))
    metadata = mock.Mock()
    with (
        mock.patch.object(A6DeviceAdaptor, "get_dsa_lightning_indexer_op", return_value=operator),
        mock.patch.object(A6DeviceAdaptor, "get_dsa_lightning_indexer_metadata_op", metadata),
        mock.patch.object(device_op, "lightning_indexer_v2") as bf16_ops,
    ):
        result = A6DeviceAdaptor.indexer_select_post_process(
            query,
            query_scale,
            query.shape,
            weights,
            [key, key_scale],
            0,
            1,
            SimpleNamespace(block_table=block_table),
            cumulative_query,
            lengths,
            True,
            False,
            False,
        )
    assert result is expected
    bf16_ops.assert_not_called()
    operator.assert_called_once()
    metadata.assert_called_once()
    call = operator.call_args.kwargs
    assert call["key"] is key
    assert call["weights"].dtype == torch.float32
    assert call["key_dequant_scale"].shape == key.shape[:-1]
    assert call["layout_k"] == "PA_BSND"
    assert call["quant_mode"] == 1
    assert call["metadata"] is metadata.return_value
    torch.testing.assert_close(call["cu_seqlens_q"], torch.tensor([0, 1, 3], dtype=torch.int32))
