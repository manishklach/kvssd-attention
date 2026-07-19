import importlib.util
import os

import pytest
import torch

from kvssd.attention import merge_attention_chunks
from kvssd.backends import PackedBatch, resolve_attention_backend
from kvssd.backends.reference import ReferenceAttentionBackend
from kvssd.quant import quantize_tensor


def make_batch(
    bits: int = 4, query_heads: int = 4, kv_heads: int = 2, head_dim: int = 32
) -> tuple[torch.Tensor, PackedBatch]:
    generator = torch.Generator().manual_seed(41)
    query = torch.randn(query_heads, head_dim, generator=generator)
    keys = torch.randn(2, 8, kv_heads, head_dim, generator=generator)
    values = torch.randn(2, 8, kv_heads, head_dim, generator=generator)
    key_q = quantize_tensor(keys, bits, 16)
    value_q = quantize_tensor(values, bits, 16)
    batch = PackedBatch(
        key_q.packed,
        value_q.packed,
        key_q.scales,
        value_q.scales,
        torch.tensor([8, 5], dtype=torch.int32),
        bits,
        16,
    )
    return query, batch


def test_triton_probe_is_explicit_when_unavailable() -> None:
    if importlib.util.find_spec("triton") is not None:
        pytest.skip("Triton is installed in this environment")
    with pytest.raises(RuntimeError, match="Triton is not installed"):
        resolve_attention_backend("triton", "cpu")


@pytest.mark.triton_interpreter
@pytest.mark.parametrize("bits", [2, 4])
@pytest.mark.parametrize(
    ("query_heads", "kv_heads", "head_dim"),
    [(1, 1, 32), (4, 1, 64), (4, 2, 128)],
)
def test_triton_interpreter_matches_reference(
    bits: int, query_heads: int, kv_heads: int, head_dim: int
) -> None:
    if importlib.util.find_spec("triton") is None or os.getenv("TRITON_INTERPRET") != "1":
        pytest.skip("requires Triton with TRITON_INTERPRET=1")
    query, batch = make_batch(bits, query_heads, kv_heads, head_dim)
    triton_backend = resolve_attention_backend("triton", "cpu")
    reference = ReferenceAttentionBackend()
    actual, actual_lse = triton_backend.run(query, batch, 1 / query.shape[-1] ** 0.5)
    expected, expected_lse = reference.run(query, batch, 1 / query.shape[-1] ** 0.5)
    torch.testing.assert_close(actual, expected, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(actual_lse, expected_lse, atol=2e-5, rtol=2e-5)


@pytest.mark.triton_interpreter
def test_triton_chunk_merge_matches_single_invocation() -> None:
    if importlib.util.find_spec("triton") is None or os.getenv("TRITON_INTERPRET") != "1":
        pytest.skip("requires Triton with TRITON_INTERPRET=1")
    query, batch = make_batch(bits=4)
    backend = resolve_attention_backend("triton", "cpu")
    scale = 1 / query.shape[-1] ** 0.5
    complete_out, complete_lse = backend.run(query, batch, scale)
    chunks = []
    for index in range(2):
        chunk = PackedBatch(
            batch.packed_k[index : index + 1],
            batch.packed_v[index : index + 1],
            batch.scales_k[index : index + 1],
            batch.scales_v[index : index + 1],
            batch.valid_tokens[index : index + 1],
            batch.bits,
            batch.group_size,
        )
        chunks.append(backend.run(query, chunk, scale))
    merged_out, merged_lse = merge_attention_chunks(
        chunks[0][0], chunks[0][1], chunks[1][0], chunks[1][1]
    )
    torch.testing.assert_close(merged_out, complete_out, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(merged_lse, complete_lse, atol=2e-5, rtol=2e-5)


@pytest.mark.gpu_hardware
@pytest.mark.parametrize("bits", [2, 4])
def test_triton_physical_gpu_matches_reference(bits: int) -> None:
    if os.getenv("KVSSD_RUN_GPU_TESTS") != "1" or not torch.cuda.is_available():
        pytest.skip("requires an opted-in CUDA or ROCm hardware runner")
    query, cpu_batch = make_batch(bits, query_heads=4, kv_heads=2, head_dim=128)
    device = torch.device("cuda")
    batch = PackedBatch(
        cpu_batch.packed_k.to(device),
        cpu_batch.packed_v.to(device),
        cpu_batch.scales_k.to(device),
        cpu_batch.scales_v.to(device),
        cpu_batch.valid_tokens.to(device),
        bits,
        cpu_batch.group_size,
    )
    query = query.to(device)
    backend = resolve_attention_backend("triton", device)
    actual, actual_lse = backend.run(query, batch, 1 / query.shape[-1] ** 0.5)
    expected, expected_lse = ReferenceAttentionBackend().run(
        query, batch, 1 / query.shape[-1] ** 0.5
    )
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
    torch.testing.assert_close(actual_lse, expected_lse, atol=2e-4, rtol=2e-4)
