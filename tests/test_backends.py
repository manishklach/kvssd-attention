from pathlib import Path

import pytest
import torch

from kvssd.attention import packed_attention
from kvssd.backends import available_attention_backends, resolve_attention_backend
from kvssd.format import CacheSpec
from kvssd.pipeline import KVSSDPipeline
from kvssd.store import KVCacheStore


def make_blocks(path: Path):
    gen = torch.Generator().manual_seed(29)
    shape = (1, 11, 2, 32)
    store = KVCacheStore.create(
        path,
        torch.randn(shape, generator=gen),
        torch.randn(shape, generator=gen),
        CacheSpec(1, 2, 32, block_tokens=8, bits=4, group_size=16),
    )
    return store, [store.read(0, block) for block in store.blocks_for_layer(0)]


def test_cpu_auto_resolves_to_reference() -> None:
    backend = resolve_attention_backend("auto", "cpu")
    assert backend.name == "reference"
    availability = available_attention_backends("cpu")
    assert {item.name for item in availability} >= {"cuda", "reference"}
    assert next(item for item in availability if item.name == "reference").available


def test_explicit_cuda_fails_closed_without_extension() -> None:
    if torch.cuda.is_available():
        pytest.skip("CPU-only failure contract")
    with pytest.raises(RuntimeError, match="CUDA extension requires"):
        resolve_attention_backend("cuda", "cpu")


def test_explicit_reference_matches_legacy_reference(tmp_path: Path) -> None:
    store, blocks = make_blocks(tmp_path)
    query = torch.randn(4, 32, generator=torch.Generator().manual_seed(31))
    explicit = packed_attention(query, blocks, attention_backend="reference")
    automatic = packed_attention(query, blocks)
    torch.testing.assert_close(explicit, automatic)
    store.close()


def test_pipeline_exposes_selected_backend(tmp_path: Path) -> None:
    store, _ = make_blocks(tmp_path)
    with KVSSDPipeline(store, device="cpu", attention_backend="reference") as pipeline:
        assert pipeline.attention_backend.name == "reference"
    store.close()


def test_unknown_backend_reports_choices() -> None:
    with pytest.raises(ValueError, match="unknown attention backend"):
        resolve_attention_backend("imaginary", "cpu")
