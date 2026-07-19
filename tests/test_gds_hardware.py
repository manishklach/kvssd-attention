from __future__ import annotations

import os
from pathlib import Path

import pytest
import torch

from kvssd.format import CacheSpec, decode_record
from kvssd.pipeline import KVSSDPipeline
from kvssd.storage import resolve_storage_backend
from kvssd.store import KVCacheStore


@pytest.mark.gds_hardware
def test_gds_reads_crc_valid_record_directly_to_cuda(tmp_path: Path) -> None:
    if os.getenv("KVSSD_RUN_GDS_TESTS") != "1" or not torch.cuda.is_available():
        pytest.skip("requires an opted-in NVIDIA GDS hardware runner")
    shape = (1, 8, 1, 32)
    store = KVCacheStore.create(
        tmp_path,
        torch.randn(shape),
        torch.randn(shape),
        CacheSpec(1, 1, 32, block_tokens=8, bits=4, group_size=16),
        storage_backend="buffered",
    )
    entry = store.entry(0, 0)
    backend = resolve_storage_backend("gds", tmp_path / "blocks.kvssd")
    destination = backend.allocate_buffer(entry.length)
    backend.read_into(entry, destination)
    raw = destination.cpu()
    block = decode_record(raw, store.spec, verify_crc=True)
    assert block.layer == 0 and block.block == 0 and block.valid_tokens == 8
    backend.close()
    store.close()


@pytest.mark.gds_hardware
def test_gds_to_fused_attention_pipeline_matches_reference(tmp_path: Path) -> None:
    if os.getenv("KVSSD_RUN_GDS_TESTS") != "1" or not torch.cuda.is_available():
        pytest.skip("requires an opted-in NVIDIA GDS hardware runner")
    shape = (1, 19, 2, 64)
    generator = torch.Generator().manual_seed(101)
    keys = torch.randn(shape, generator=generator)
    values = torch.randn(shape, generator=generator)
    spec = CacheSpec(1, 2, 64, block_tokens=8, bits=4, group_size=16)
    writer = KVCacheStore.create(tmp_path, keys, values, spec, storage_backend="buffered")
    writer.close()
    query = torch.randn(4, 64, generator=generator)
    with KVCacheStore.open(tmp_path, storage_backend="buffered") as reference_store:
        with KVSSDPipeline(
            reference_store, device="cpu", attention_backend="reference", io_depth=2
        ) as pipeline:
            expected = pipeline.decode(query, 0)
    with KVCacheStore.open(tmp_path, storage_backend="gds") as gds_store:
        with KVSSDPipeline(
            gds_store, device="cuda", attention_backend="triton", io_depth=2, stage_blocks=3
        ) as pipeline:
            actual = pipeline.decode(query.cuda(), 0).cpu()
    torch.testing.assert_close(actual, expected, atol=2e-4, rtol=2e-4)
