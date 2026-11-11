# SPDX-License-Identifier: Apache-2.0
"""A3 eager bring-up: independent union oracle and strided payload checks."""

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch
import torch_npu  # noqa: F401

from vllm_ascend.hisparse.a3_eager import gather, resolve_cpu
from vllm_ascend.hisparse.cache_ops import allocate_mapped_host


@pytest.mark.parametrize("repeat", [False, True])
def test_union_protects_later_query_and_deduplicates_refill(repeat):
    # Token 4 is selected only by the later query. Its cached slot is oldest,
    # so resolving the first row alone would wrongly evict it.
    args = dict(
        global_indices=torch.tensor([[0, 1, -1, -1], [4, 1, -1, -1]], dtype=torch.int32),
        request_ids=torch.tensor([0, 0], dtype=torch.int32),
        request_state_indices=torch.tensor([0], dtype=torch.int32),
        source_block_table=torch.tensor([[1, 2]], dtype=torch.int32),
        resident_block_table=torch.zeros((1, 2), dtype=torch.int32),
        resident_null_block=0,
        source_block_size=4,
        resident_block_size=4,
        hot_block_table=torch.tensor([[1]], dtype=torch.int32),
        device_global_indices=torch.tensor([[8, -1, -1, -1]], dtype=torch.int32),
        lru_slots=torch.tensor([[0, 1, 2, 3]], dtype=torch.int16),
        num_valid_rows=None,
        max_union_rows=4,
        attention_block_stride=4,
        host_cache=torch.empty(16, 8),
        hot_cache=torch.empty(2, 4, 8),
    )
    result = resolve_cpu(args)
    assert result["hot_indices"].tolist() == [[5, 6, -1, -1], [4, 6, -1, -1]]
    assert result["swap_counts"].tolist() == [2, 0]
    assert result["swap_host_physical_rows"][0, :2].tolist() == [4, 5]
    assert result["device_global_indices"].tolist() == [[8, 4, 5, -1]]
    assert result["lru_slots"].tolist() == [[3, 1, 2, 0]]
    assert result["stats"].tolist() == [2, 2]
    if repeat:
        args.update({k: result[k] for k in ("device_global_indices", "lru_slots")})
        again = resolve_cpu(args)
        assert again["swap_counts"].tolist() == [0, 0]
        torch.testing.assert_close(again["hot_indices"], result["hot_indices"])


def test_refill_preserves_shared_slab_gaps():
    torch.npu.set_device(0)
    host = allocate_mapped_host((16, 8), torch.bfloat16)
    host.copy_(torch.arange(128).reshape(16, 8))
    backing = torch.full((3, 2, 4, 8), -7, dtype=torch.bfloat16, device="npu")
    hot = backing[:, 1]
    src = torch.tensor([[4, 5, -1, -1]], dtype=torch.int32, device="npu")
    dst = torch.tensor([[5, 8, -1, -1]], dtype=torch.int32, device="npu")
    counts = torch.tensor([2], dtype=torch.int32, device="npu")
    gather(host, hot, src, dst, counts, None, None, None, 0)
    expected = torch.full_like(backing, -7).cpu()
    expected[1, 1, 1] = host[4]
    expected[2, 1, 0] = host[5]
    torch.testing.assert_close(backing.cpu(), expected)
    host._hisparse_registration.close()


@pytest.mark.parametrize("active", [0, 1, 2])
def test_padding_and_resident_rows_do_not_allocate_hot_slots(active):
    args = dict(
        global_indices=torch.tensor([[0, 1, -1, 2], [0, 3, -1, -1]], dtype=torch.int32),
        request_ids=torch.tensor([0, 0], dtype=torch.int32),
        request_state_indices=torch.tensor([0], dtype=torch.int32),
        source_block_table=torch.tensor([[1]], dtype=torch.int32),
        resident_block_table=torch.tensor([[2]], dtype=torch.int32),
        resident_null_block=0,
        source_block_size=4,
        resident_block_size=4,
        hot_block_table=torch.tensor([[1]], dtype=torch.int32),
        device_global_indices=torch.full((1, 4), -1, dtype=torch.int32),
        lru_slots=torch.arange(4, dtype=torch.int16).view(1, 4),
        num_valid_rows=torch.tensor([active], dtype=torch.int32),
        max_union_rows=4,
        attention_block_stride=8,
        host_cache=torch.empty(16, 8),
        hot_cache=torch.empty(3, 4, 8),
    )
    result = resolve_cpu(args)
    expected = torch.tensor([[8, 9, -1, 10], [8, 11, -1, -1]], dtype=torch.int32)
    expected[active:] = -1
    torch.testing.assert_close(result["hot_indices"], expected)
    assert result["swap_counts"].sum() == 0
    torch.testing.assert_close(result["device_global_indices"], args["device_global_indices"])
    torch.testing.assert_close(result["lru_slots"], args["lru_slots"])


@pytest.mark.parametrize("seed", range(10))
def test_resident_fast_path_matches_general_resolver(seed):
    torch.manual_seed(seed)
    source = torch.arange(1, 17, dtype=torch.int32).reshape(2, 8)
    args = dict(
        global_indices=torch.randint(-4, 36, (4, 16), dtype=torch.int32),
        request_ids=torch.tensor([0, 0, 1, -1], dtype=torch.int32),
        request_state_indices=torch.tensor([1, 0], dtype=torch.int32),
        source_block_table=source,
        resident_block_table=source.flip(0),
        resident_null_block=0,
        source_block_size=4,
        resident_block_size=4,
        hot_block_table=torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]], dtype=torch.int32),
        device_global_indices=torch.full((2, 16), -1, dtype=torch.int32),
        lru_slots=torch.arange(16, dtype=torch.int16)[None].repeat(2, 1),
        num_valid_rows=torch.tensor([seed % 5], dtype=torch.int32),
        max_union_rows=16,
        attention_block_stride=8,
        host_cache=torch.empty(68, 8),
        hot_cache=torch.empty(17, 4, 8),
    )
    fast = resolve_cpu(args)
    with patch("vllm_ascend.hisparse.a3_eager._resident_only", return_value=None):
        reference = resolve_cpu(args)
    for name, expected in reference.items():
        torch.testing.assert_close(fast[name], expected, msg=name)


@pytest.mark.parametrize("dtype,kernel_type", [(torch.float16, 1), (torch.bfloat16, 2), (torch.float32, 3)])
def test_missing_rmsnorm_bias_operator_preserves_a3_rounding(dtype, kernel_type):
    from tests.e2e.nightly.single_node.ops.singlecard_ops.test_add_rms_norm_bias import npu_add_rms_norm_bias_golden
    from vllm_ascend.ops.layernorm import _add_rms_norm_bias_fallback

    torch.manual_seed(129)
    x, residual = [torch.randn(7, 7168).to(dtype) for _ in range(2)]
    weight, bias = [torch.randn(7168).to(dtype) for _ in range(2)]
    expected, _, expected_residual = npu_add_rms_norm_bias_golden(x, residual, weight, bias, kernel_type)
    actual, actual_residual = _add_rms_norm_bias_fallback(x.npu(), residual.npu(), weight.npu(), bias.npu(), 1e-6)
    torch.testing.assert_close(actual.cpu(), expected, atol=0.0001, rtol=0.01)
    torch.testing.assert_close(actual_residual.cpu(), expected_residual, atol=0, rtol=0)


@pytest.mark.parametrize("valid_rows", [1, 3, 4])
def test_bf16_prolog_preserves_outputs_and_packs_only_valid_rows(valid_rows):
    from vllm_ascend.attention.sfa_hisparse import AscendHiSparseImpl
    from vllm_ascend.attention.sfa_v1 import AscendSFAImpl

    torch.npu.set_device(0)
    backing = torch.full((3, 2, 4, 12), -7, device="npu", dtype=torch.bfloat16)
    destination = backing[:, 1]
    mapping = torch.tensor([1, 5, 8, 9], device="npu", dtype=torch.int32)
    latent = torch.empty((1, 4, 1, 8), device="npu", dtype=torch.bfloat16)
    rope = torch.empty((1, 4, 1, 4), device="npu", dtype=torch.bfloat16)
    slots = torch.arange(4, device="npu", dtype=torch.int64)
    cache = SimpleNamespace(write_target=lambda rows, count: (destination, mapping, count))
    metadata = SimpleNamespace(query_start_loc=torch.tensor([0, valid_rows], device="npu", dtype=torch.int32))
    impl = object.__new__(AscendHiSparseImpl)
    impl._kv_staging_capacity = 4
    impl.kv_lora_rank = 8
    impl.qk_rope_head_dim = 4
    impl.enable_sparse_sfa_c8 = False
    impl.hisparse_cache = cache
    impl._kv_workspace = lambda dtype: (latent, rope, slots)
    native_result = (object(), object(), object(), object())
    values = torch.arange(32, dtype=torch.bfloat16).reshape(4, 8)
    rotations = torch.arange(100, 116, dtype=torch.bfloat16).reshape(4, 4)

    def prolog(_self, hidden, caches, cos, sin, local_slots, *, attn_metadata):
        assert caches[0] is latent and caches[1] is rope
        assert local_slots.dtype == torch.int64
        torch.testing.assert_close(local_slots.cpu(), torch.arange(4))
        latent.copy_(values.reshape_as(latent))
        rope.copy_(rotations.reshape_as(rope))
        return native_result

    expected = torch.full_like(backing, -7).cpu()
    for row, physical in enumerate([1, 5, 8, 9][:valid_rows]):
        expected[physical // 4, 1, physical % 4] = torch.cat((values[row], rotations[row]))

    def finish(actual_cache, actual_metadata):
        assert actual_cache is cache and actual_metadata is metadata
        torch.testing.assert_close(backing.cpu(), expected, atol=0, rtol=0)

    with (
        patch.object(AscendSFAImpl, "_sfa_preprocess_prolog_v3", prolog),
        patch("vllm_ascend.attention.sfa_hisparse.finish_hisparse_kv_update", side_effect=finish) as publish,
    ):
        result = impl._sfa_preprocess_prolog_v3(
            torch.empty((4, 16), device="npu", dtype=torch.bfloat16),
            (),
            None,
            None,
            mapping,
            attn_metadata=metadata,
        )
    assert result is native_result
    publish.assert_called_once()
