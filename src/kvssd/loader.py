from __future__ import annotations

from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass
from queue import Queue
from typing import Iterator

import torch

from .format import BlockEntry, PackedKVBlock, decode_record
from .quant import QuantizedTensor
from .store import KVCacheStore


@dataclass
class _Slot:
    raw: torch.Tensor
    future: Future[None] | None = None
    reuse_event: torch.cuda.Event | None = None


class AsyncBlockLoader:
    """Bounded pinned-memory prefetcher; slots provide backpressure by construction."""

    def __init__(
        self,
        store: KVCacheStore,
        depth: int = 2,
        pinned: bool = True,
        device: torch.device | None = None,
    ):
        if depth < 1:
            raise ValueError("depth must be positive")
        self.store = store
        self.executor = ThreadPoolExecutor(max_workers=depth, thread_name_prefix="kvssd-io")
        self.free: Queue[_Slot] = Queue()
        use_pinned = pinned and torch.cuda.is_available()
        for _ in range(depth):
            self.free.put(
                _Slot(
                    store.storage_backend.allocate_buffer(
                        store.spec.record_bytes, pinned=use_pinned, device=device
                    )
                )
            )

    def iter_blocks(self, layer: int, block_ids: list[int]) -> Iterator[PackedKVBlock]:
        pending: list[tuple[_Slot, Future[None], BlockEntry]] = []
        source = iter(block_ids)

        def submit(block_id: int) -> None:
            slot = self.free.get()
            if slot.reuse_event is not None:
                slot.reuse_event.synchronize()
                slot.reuse_event = None
            entry = self.store.entry(layer, block_id)
            future = self.executor.submit(self.store.read_into, entry, slot.raw)
            slot.future = future
            pending.append((slot, future, entry))

        for _ in range(min(self.free.qsize(), len(block_ids))):
            submit(next(source))
        while pending:
            slot, future, entry = pending.pop(0)
            future.result()
            if slot.raw.device.type == "cpu":
                viewed = decode_record(slot.raw, self.store.spec)
            else:
                viewed = self._decode_device_record(slot.raw, entry)
            # Records are views into a bounded reusable I/O slot. Detach the four payloads
            # before returning the slot; this prevents a later read from corrupting an
            # in-flight consumer. The pipeline subsequently coalesces the payloads.
            block = PackedKVBlock(
                viewed.layer,
                viewed.block,
                viewed.valid_tokens,
                QuantizedTensor(
                    viewed.key.packed.clone(),
                    viewed.key.scales.clone(),
                    viewed.key.original_shape,
                    viewed.key.bits,
                    viewed.key.group_size,
                ),
                QuantizedTensor(
                    viewed.value.packed.clone(),
                    viewed.value.scales.clone(),
                    viewed.value.original_shape,
                    viewed.value.bits,
                    viewed.value.group_size,
                ),
            )
            if slot.raw.device.type == "cuda":
                slot.reuse_event = torch.cuda.Event()
                slot.reuse_event.record(torch.cuda.current_stream(slot.raw.device))
            try:
                next_id = next(source)
            except StopIteration:
                next_id = None
            yield block
            self.free.put(slot)
            if next_id is not None:
                submit(next_id)

    def _decode_device_record(self, raw: torch.Tensor, entry: BlockEntry) -> PackedKVBlock:
        """Build GPU views using the trusted manifest; the on-disk header remains in-place."""
        from .format import HEADER

        spec = self.store.spec
        cursor = HEADER.size
        scales_bytes = spec.scale_count * 2
        packed_bytes = spec.packed_count
        scale_shape = (spec.block_tokens, spec.kv_heads, spec.head_dim // spec.group_size)
        packed_shape = (spec.block_tokens, spec.kv_heads, spec.packed_dim)
        ks = raw[cursor : cursor + scales_bytes].view(torch.float16).reshape(scale_shape)
        cursor += scales_bytes
        vs = raw[cursor : cursor + scales_bytes].view(torch.float16).reshape(scale_shape)
        cursor += scales_bytes
        kp = raw[cursor : cursor + packed_bytes].reshape(packed_shape)
        cursor += packed_bytes
        vp = raw[cursor : cursor + packed_bytes].reshape(packed_shape)
        shape = (spec.block_tokens, spec.kv_heads, spec.head_dim)
        return PackedKVBlock(
            entry.layer,
            entry.block,
            entry.valid_tokens,
            QuantizedTensor(kp, ks, shape, spec.bits, spec.group_size),
            QuantizedTensor(vp, vs, shape, spec.bits, spec.group_size),
        )

    def close(self) -> None:
        self.executor.shutdown(wait=True, cancel_futures=False)

    def __enter__(self) -> "AsyncBlockLoader":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
