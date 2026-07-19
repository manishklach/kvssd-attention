from pathlib import Path

import torch

from kvssd.format import CacheSpec
from kvssd.pipeline import KVSSDPipeline
from kvssd.selectors import RecentSinkBlocks
from kvssd.store import KVCacheStore


def test_cpu_end_to_end_with_selection(tmp_path: Path) -> None:
    gen = torch.Generator().manual_seed(11)
    shape = (1, 40, 2, 32)
    store = KVCacheStore.create(
        tmp_path,
        torch.randn(shape, generator=gen),
        torch.randn(shape, generator=gen),
        CacheSpec(1, 2, 32, block_tokens=8, bits=4, group_size=16),
    )
    with KVSSDPipeline(store, device="cpu", io_depth=2) as pipeline:
        output = pipeline.decode(torch.randn(4, 32, generator=gen), 0, RecentSinkBlocks(1, 2))
    assert output.shape == (4, 32)
    assert torch.isfinite(output).all()
