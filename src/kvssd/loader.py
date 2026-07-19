from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from queue import Queue
from typing import Iterator

import torch

from .format import PackedKVBlock, decode_record
from .quant import QuantizedTensor
from .store import KVCacheStore


@dataclass
class _Slot:
    raw: torch.Tensor
    future: Future[None] | None = None


class AsyncBlockLoader:
    """Bounded pinned-memory prefetcher; slots provide backpressure by construction."""

    def __init__(self, store: KVCacheStore, depth: int = 2, pinned: bool = True):
        if depth < 1:
            raise ValueError("depth must be positive")
        self.store = store
        self.executor = ThreadPoolExecutor(max_workers=depth, thread_name_prefix="kvssd-io")
        self.free: Queue[_Slot] = Queue()
        use_pinned = pinned and torch.cuda.is_available()
        for _ in range(depth):
            self.free.put(_Slot(torch.empty(store.spec.record_bytes, dtype=torch.uint8, pin_memory=use_pinned)))

    def iter_blocks(self, layer: int, block_ids: list[int]) -> Iterator[PackedKVBlock]:
        pending: list[tuple[_Slot, Future[None]]] = []
        source = iter(block_ids)

        def submit(block_id: int) -> None:
            slot = self.free.get()
            entry = self.store.entry(layer, block_id)
            future = self.executor.submit(self.store.read_into, entry, slot.raw)
            slot.future = future
            pending.append((slot, future))

        for _ in range(min(self.free.qsize(), len(block_ids))):
            submit(next(source))
        while pending:
            slot, future = pending.pop(0)
            future.result()
            viewed = decode_record(slot.raw, self.store.spec)
            # Records are views into a bounded reusable I/O slot. Detach the four payloads
            # before returning the slot; this prevents a later pread from corrupting an
            # in-flight consumer. The pipeline subsequently coalesces these into one H2D copy.
            block = PackedKVBlock(
                viewed.layer,
                viewed.block,
                viewed.valid_tokens,
                QuantizedTensor(
                    viewed.key.packed.clone(), viewed.key.scales.clone(), viewed.key.original_shape,
                    viewed.key.bits, viewed.key.group_size,
                ),
                QuantizedTensor(
                    viewed.value.packed.clone(), viewed.value.scales.clone(),
                    viewed.value.original_shape, viewed.value.bits, viewed.value.group_size,
                ),
            )
            try:
                next_id = next(source)
            except StopIteration:
                next_id = None
            yield block
            self.free.put(slot)
            if next_id is not None:
                submit(next_id)

    def close(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=False)

    def __enter__(self) -> "AsyncBlockLoader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
