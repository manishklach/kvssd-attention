from pathlib import Path

import pytest
import torch

from kvssd.attention import merge_attention_chunks, packed_attention, reference_attention
from kvssd.format import CacheSpec
from kvssd.store import KVCacheStore


@pytest.mark.parametrize("bits", [2, 4])
def test_packed_attention_matches_dequantized_reference(tmp_path: Path, bits: int) -> None:
    gen = torch.Generator().manual_seed(5)
    keys = torch.randn(1, 13, 2, 32, generator=gen)
    values = torch.randn(1, 13, 2, 32, generator=gen)
    query = torch.randn(4, 32, generator=gen)
    spec = CacheSpec(1, 2, 32, block_tokens=8, bits=bits, group_size=16)
    store = KVCacheStore.create(tmp_path, keys, values, spec)
    blocks = [store.read(0, i) for i in store.blocks_for_layer(0)]
    actual = packed_attention(query, blocks)
    dequant_k = torch.cat(
        [
            __import__("kvssd.quant", fromlist=["dequantize_tensor"]).dequantize_tensor(x.key)[
                : x.valid_tokens
            ]
            for x in blocks
        ]
    )
    dequant_v = torch.cat(
        [
            __import__("kvssd.quant", fromlist=["dequantize_tensor"]).dequantize_tensor(x.value)[
                : x.valid_tokens
            ]
            for x in blocks
        ]
    )
    expected = reference_attention(query, dequant_k, dequant_v)
    torch.testing.assert_close(actual, expected)


def test_gqa_validation() -> None:
    with pytest.raises(ValueError, match="divisible"):
        reference_attention(torch.randn(3, 8), torch.randn(4, 2, 8), torch.randn(4, 2, 8))


def test_chunk_softmax_merge_matches_concatenation() -> None:
    gen = torch.Generator().manual_seed(17)
    query = torch.randn(4, 16, generator=gen)
    keys = torch.randn(11, 2, 16, generator=gen)
    values = torch.randn(11, 2, 16, generator=gen)
    mapping = torch.arange(4) // 2
    scores = torch.einsum("hd,thd->ht", query, keys[:, mapping]) / 4

    def chunk(start: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        local = scores[:, start:end]
        lse = torch.logsumexp(local, dim=-1)
        out = torch.einsum("ht,thd->hd", torch.softmax(local, -1), values[start:end, mapping])
        return out, lse

    left, left_lse = chunk(0, 5)
    right, right_lse = chunk(5, 11)
    merged, _ = merge_attention_chunks(left, left_lse, right, right_lse)
    expected = reference_attention(query, keys, values)
    torch.testing.assert_close(merged, expected, atol=1e-6, rtol=1e-5)
