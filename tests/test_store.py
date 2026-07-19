from pathlib import Path

import pytest
import torch

from kvssd.format import CacheSpec
from kvssd.loader import AsyncBlockLoader
from kvssd.quant import dequantize_tensor
from kvssd.store import KVCacheStore


def make_store(path: Path, bits: int = 4) -> tuple[KVCacheStore, torch.Tensor, torch.Tensor]:
    shape = (2, 19, 2, 32)
    gen = torch.Generator().manual_seed(3)
    keys, values = torch.randn(shape, generator=gen), torch.randn(shape, generator=gen)
    spec = CacheSpec(2, 2, 32, block_tokens=8, bits=bits, group_size=16)
    return KVCacheStore.create(path, keys, values, spec), keys, values


def test_store_round_trip_and_padding(tmp_path: Path) -> None:
    store, keys, _ = make_store(tmp_path)
    block = store.read(1, 2)
    assert block.valid_tokens == 3
    restored = dequantize_tensor(block.key)
    assert torch.mean(torch.abs(restored[:3] - keys[1, 16:])).item() < 0.15
    assert torch.count_nonzero(restored[3:]) == 0
    assert all(entry.offset % 4096 == 0 for entry in store.manifest.entries)


def test_crc_detects_corruption(tmp_path: Path) -> None:
    store, _, _ = make_store(tmp_path)
    entry = store.entry(0, 0)
    data_path = tmp_path / "blocks.kvssd"
    with data_path.open("r+b") as f:
        f.seek(entry.offset + 80)
        old = f.read(1)
        f.seek(entry.offset + 80)
        f.write(bytes([old[0] ^ 0xFF]))
    with pytest.raises(IOError, match="CRC"):
        store.read(0, 0)


def test_async_loader_does_not_alias_recycled_slots(tmp_path: Path) -> None:
    store, _, _ = make_store(tmp_path)
    with AsyncBlockLoader(store, depth=2, pinned=False) as loader:
        blocks = list(loader.iter_blocks(0, [0, 1, 2]))
    snapshots = [x.key.packed.clone() for x in blocks]
    assert [x.block for x in blocks] == [0, 1, 2]
    assert all(torch.equal(x.key.packed, expected) for x, expected in zip(blocks, snapshots))


def test_parallel_reader_closes_its_single_descriptor(tmp_path: Path) -> None:
    store, _, _ = make_store(tmp_path)
    with AsyncBlockLoader(store, depth=3, pinned=False) as loader:
        list(loader.iter_blocks(0, [0, 1, 2]))
    store.close()
    # This specifically catches concurrent lazy-open races on Windows, where a leaked duplicate
    # descriptor prevents deletion. It is also a useful lifecycle assertion on POSIX.
    (tmp_path / "blocks.kvssd").unlink()
