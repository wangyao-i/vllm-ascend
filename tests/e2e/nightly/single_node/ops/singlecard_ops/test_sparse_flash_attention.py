# SPDX-License-Identifier: Apache-2.0

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.device.device_config import get_ascend_device_type
from vllm_ascend.device.hardware import AscendDeviceType
from vllm_ascend.utils import enable_custom_op


@pytest.mark.skipif(
    get_ascend_device_type() != AscendDeviceType.A5,
    reason="Regression for the A5 SFA kernel",
)
@pytest.mark.parametrize("selected_count", [0, 1, 127, 128, 129, 257])
def test_sparse_flash_attention_padded_indices_lse(selected_count):
    enable_custom_op()
    torch.manual_seed(928)
    heads, kv_length, capacity = 64, 512, 2048
    query = torch.randn(1, heads, 512, dtype=torch.bfloat16)
    query_rope = torch.randn(1, heads, 64, dtype=torch.bfloat16)
    key = torch.randn(kv_length, 512, dtype=torch.bfloat16)
    key_rope = torch.randn(kv_length, 64, dtype=torch.bfloat16)
    selected = torch.randperm(kv_length)[:selected_count].sort().values
    indices = torch.full((1, 1, capacity), -1, dtype=torch.int32)
    indices[0, 0, :selected_count] = selected.int()
    pages = key.reshape(4, 128, 1, 512).npu()
    rope_pages = key_rope.reshape(4, 128, 1, 64).npu()
    output, maximum, total = torch.ops._C_ascend.npu_sparse_flash_attention(
        query=query.npu(),
        key=pages,
        value=pages,
        sparse_indices=indices.npu(),
        scale_value=1 / 24,
        sparse_block_size=1,
        block_table=torch.arange(4, dtype=torch.int32).reshape(1, -1).npu(),
        actual_seq_lengths_query=torch.tensor([1], dtype=torch.int32).npu(),
        actual_seq_lengths_kv=torch.tensor([kv_length], dtype=torch.int32).npu(),
        query_rope=query_rope.npu(),
        key_rope=rope_pages,
        layout_query="TND",
        layout_kv="PA_BSND",
        sparse_mode=0,
        attention_mode=2,
        return_softmax_lse=True,
    )
    output = output.cpu().float().reshape(heads, 512)
    lse = (maximum.cpu().float() + total.cpu().float().log()).reshape(heads)
    if selected_count == 0:
        assert torch.count_nonzero(output) == 0
        assert torch.isneginf(lse).all()
        return
    logits = (query[0].float() @ key[selected].float().T + query_rope[0].float() @ key_rope[selected].float().T) / 24
    expected = logits.softmax(-1) @ key[selected].float()
    torch.testing.assert_close(output, expected, atol=0.03, rtol=0.01)
    torch.testing.assert_close(lse, logits.logsumexp(-1), atol=0.005, rtol=0.001)


@pytest.mark.skipif(get_ascend_device_type() != AscendDeviceType.A6, reason="Regression for the A6 SFA kernel")
def test_sparse_flash_attention_tnd_output_padding():
    enable_custom_op()
    # Two requests each attend to a different constant KV page. Only their
    # 236 real queries have an attention result; T can include extra rows.
    heads, page_size = 64, 128
    first_end, valid_rows = 120, 236
    height = valid_rows + 40
    dtype = torch.bfloat16
    query = torch.zeros((height, heads, 512), dtype=dtype, device="npu")
    query_rope = query.new_zeros((height, heads, 64))
    pages = query.new_ones((2, page_size, 1, 512))
    pages[1].fill_(2)
    rope_pages = pages.new_zeros((2, page_size, 1, 64))
    indices = torch.full((height, 1, 2048), -1, dtype=torch.int32, device="npu")
    indices[:valid_rows, 0, :page_size] = torch.arange(page_size, dtype=torch.int32, device="npu")
    block_table = indices.new_tensor([[0], [1]])
    query_lengths = indices.new_tensor([first_end, valid_rows])
    kv_lengths = indices.new_tensor([page_size, page_size])

    def attention(rows):
        output, _, _ = torch.ops._C_ascend.npu_sparse_flash_attention(
            query=query[:rows],
            key=pages,
            value=pages,
            sparse_indices=indices[:rows],
            scale_value=1 / 24,
            sparse_block_size=1,
            block_table=block_table,
            actual_seq_lengths_query=query_lengths,
            actual_seq_lengths_kv=kv_lengths,
            query_rope=query_rope[:rows],
            key_rope=rope_pages,
            layout_query="TND",
            layout_kv="PA_BSND",
            sparse_mode=3,
            attention_mode=2,
            return_softmax_lse=False,
        )
        return output

    reference = attention(valid_rows)
    assert (reference[:first_end] == 1).all()
    assert (reference[first_end:] == 2).all()
    for padding in (1, 40):
        output = attention(valid_rows + padding)
        assert output.shape == (valid_rows + padding, heads, 512)
        torch.testing.assert_close(output[:valid_rows], reference, rtol=0, atol=0)
        assert torch.count_nonzero(output[valid_rows:]) == 0

    # Replay into a known nonzero buffer so fresh allocator pages cannot hide missing stores.
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        output = attention(height)
    output.fill_(float("nan"))
    graph.replay()
    torch.testing.assert_close(output[:valid_rows], reference, rtol=0, atol=0)
    assert torch.count_nonzero(output[valid_rows:]) == 0
