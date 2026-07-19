from pathlib import Path
from dataclasses import replace

import pytest
import torch

from kvssd.format import CacheSpec
from kvssd.storage import available_storage_backends, resolve_storage_backend
from kvssd.storage.direct import DirectIOStorageBackend
from kvssd.storage.gds import GDSStorageBackend
from kvssd.store import KVCacheStore


def make_store(path: Path) -> KVCacheStore:
    generator = torch.Generator().manual_seed(37)
    shape = (1, 16, 1, 32)
    return KVCacheStore.create(
        path,
        torch.randn(shape, generator=generator),
        torch.randn(shape, generator=generator),
        CacheSpec(1, 1, 32, block_tokens=8, bits=4, group_size=16),
        storage_backend="buffered",
    )


def test_auto_storage_resolves_to_best_available_backend(tmp_path: Path) -> None:
    created = make_store(tmp_path)
    created.close()
    store = KVCacheStore.open(tmp_path, storage_backend="auto")
    assert store.storage_backend.name in {"direct", "buffered"}
    availability = available_storage_backends(tmp_path / "blocks.kvssd")
    assert [item.name for item in availability] == ["gds", "direct", "buffered"]
    assert availability[-1].available
    store.close()


def test_explicit_missing_file_fails_closed(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="does not exist"):
        resolve_storage_backend("buffered", tmp_path / "missing.kvssd")


def test_unknown_storage_backend_reports_choices(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="unknown storage backend"):
        resolve_storage_backend("imaginary", tmp_path / "cache.kvssd")


def test_buffered_rejects_device_or_layout_mismatch(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    entry = store.entry(0, 0)
    with pytest.raises(ValueError, match="sufficiently large"):
        store.read_into(entry, torch.empty(entry.length - 1, dtype=torch.uint8))
    store.close()


def test_direct_allocator_is_record_aligned(tmp_path: Path) -> None:
    buffer = DirectIOStorageBackend(tmp_path / "unused").allocate_buffer(4096)
    assert buffer.is_contiguous()
    assert buffer.numel() == 4096
    assert buffer.data_ptr() % 4096 == 0


def test_gds_capability_probe_is_safe_without_hardware(tmp_path: Path) -> None:
    backend = GDSStorageBackend(tmp_path / "missing.kvssd")
    availability = backend.probe()
    assert availability.name == "gds"
    assert not availability.available
    assert availability.direct
    assert availability.reason


def test_explicit_gds_fails_closed_without_capability(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="storage backend 'gds' unavailable"):
        resolve_storage_backend("gds", tmp_path / "missing.kvssd")


def test_linux_direct_read_and_alignment_contract(tmp_path: Path) -> None:
    store = make_store(tmp_path)
    store.close()
    path = tmp_path / "blocks.kvssd"
    backend = DirectIOStorageBackend(path)
    if not backend.probe().available:
        pytest.skip("requires a Linux filesystem accepting O_DIRECT")
    opened = KVCacheStore.open(tmp_path, storage_backend="buffered")
    entry = opened.entry(0, 0)
    destination = backend.allocate_buffer(entry.length)
    backend.read_into(entry, destination)
    assert any(destination.tolist())
    with pytest.raises(ValueError, match="unaligned direct read"):
        backend.read_into(replace(entry, offset=entry.offset + 1), destination)
    backend.close()
    opened.close()
