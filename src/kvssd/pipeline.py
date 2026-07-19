from __future__ import annotations

import math
from collections.abc import Iterator, Sequence

import torch

from .attention import (
    cuda_extension_available,
    fused_attention_tensors,
    merge_attention_chunks,
    packed_attention,
)
from .format import PackedKVBlock
from .loader import AsyncBlockLoader
from .selectors import AllBlocks, BlockSelector
from .store import KVCacheStore


class KVSSDPipeline:
    """End-to-end selection, SSD prefetch, pinned staging, and packed attention pipeline."""

    def __init__(
        self,
        store: KVCacheStore,
        device: str | torch.device = "cuda",
        io_depth: int = 2,
        stage_blocks: int = 4,
        require_cuda_kernel: bool = False,
    ):
        self.store = store
        self.device = torch.device(device)
        if self.device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA device requested but CUDA is unavailable")
        self.loader = AsyncBlockLoader(store, depth=io_depth, pinned=self.device.type == "cuda")
        if stage_blocks < 1:
            raise ValueError("stage_blocks must be positive")
        self.stage_blocks = stage_blocks
        self.require_cuda_kernel = require_cuda_kernel
        self.transfer_stream = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None
        self.compute_stream = torch.cuda.Stream(device=self.device) if self.device.type == "cuda" else None

    @staticmethod
    def _chunks(source: Iterator[PackedKVBlock], size: int) -> Iterator[list[PackedKVBlock]]:
        chunk: list[PackedKVBlock] = []
        for block in source:
            chunk.append(block)
            if len(chunk) == size:
                yield chunk
                chunk = []
        if chunk:
            yield chunk

    def _stage(self, blocks: Sequence[PackedKVBlock]) -> tuple[tuple[torch.Tensor, ...], torch.cuda.Event]:
        """Coalesce a chunk in pinned memory and enqueue one non-blocking H2D transfer."""
        assert self.transfer_stream is not None
        first = blocks[0]
        count = len(blocks)
        spec = self.store.spec
        shapes = (
            (count, spec.block_tokens, spec.kv_heads, spec.packed_dim),
            (count, spec.block_tokens, spec.kv_heads, spec.packed_dim),
            (count, spec.block_tokens, spec.kv_heads, spec.head_dim // spec.group_size),
            (count, spec.block_tokens, spec.kv_heads, spec.head_dim // spec.group_size),
        )
        host = [
            torch.empty(shapes[0], dtype=torch.uint8, pin_memory=True),
            torch.empty(shapes[1], dtype=torch.uint8, pin_memory=True),
            torch.empty(shapes[2], dtype=torch.float16, pin_memory=True),
            torch.empty(shapes[3], dtype=torch.float16, pin_memory=True),
        ]
        for index, block in enumerate(blocks):
            host[0][index].copy_(block.key.packed)
            host[1][index].copy_(block.value.packed)
            host[2][index].copy_(block.key.scales)
            host[3][index].copy_(block.value.scales)
        valid_host = torch.tensor([x.valid_tokens for x in blocks], dtype=torch.int32).pin_memory()
        with torch.cuda.stream(self.transfer_stream):
            device_tensors = tuple(x.to(self.device, non_blocking=True) for x in host)
            valid = valid_host.to(self.device, non_blocking=True)
            event = torch.cuda.Event()
            event.record(self.transfer_stream)
        _ = first  # documents that all blocks share one layout, enforced by the store
        return (*device_tensors, valid), event

    def _decode_cuda_streamed(
        self, query: torch.Tensor, layer: int, selected: list[int]
    ) -> torch.Tensor:
        assert self.compute_stream is not None and self.transfer_stream is not None
        source = self.loader.iter_blocks(layer, selected)
        chunks = self._chunks(source, self.stage_blocks)
        try:
            current = self._stage(next(chunks))
        except StopIteration as exc:
            raise ValueError("selector returned no blocks") from exc

        merged_out: torch.Tensor | None = None
        merged_lse: torch.Tensor | None = None
        scale = 1.0 / math.sqrt(query.shape[-1])
        caller_ready = torch.cuda.Event()
        caller_ready.record(torch.cuda.current_stream(self.device))
        self.compute_stream.wait_event(caller_ready)
        query.record_stream(self.compute_stream)
        while current is not None:
            tensors, ready = current
            with torch.cuda.stream(self.compute_stream):
                self.compute_stream.wait_event(ready)
                out, lse = fused_attention_tensors(
                    query, *tensors, self.store.spec.bits, self.store.spec.group_size, scale
                )
                for tensor in tensors:
                    tensor.record_stream(self.compute_stream)
                if merged_out is None:
                    merged_out, merged_lse = out, lse
                else:
                    assert merged_lse is not None
                    merged_out, merged_lse = merge_attention_chunks(
                        merged_out, merged_lse, out, lse
                    )
            # Reading and staging the next chunk occurs while the preceding kernel is executing.
            try:
                current = self._stage(next(chunks))
            except StopIteration:
                current = None
        assert merged_out is not None
        caller = torch.cuda.current_stream(self.device)
        done = torch.cuda.Event()
        done.record(self.compute_stream)
        caller.wait_event(done)
        merged_out.record_stream(caller)
        return merged_out.to(query.dtype)

    def decode(
        self,
        query: torch.Tensor,
        layer: int,
        selector: BlockSelector | None = None,
    ) -> torch.Tensor:
        selector = selector or AllBlocks()
        selected = selector.select(self.store.blocks_for_layer(layer))
        if not selected:
            raise ValueError("selector returned no blocks")
        query = query.to(self.device)
        if self.device.type == "cuda" and cuda_extension_available():
            return self._decode_cuda_streamed(query, layer, selected)
        blocks = list(self.loader.iter_blocks(layer, selected))
        return packed_attention(
            query,
            blocks,
            sm_scale=1.0 / math.sqrt(query.shape[-1]),
            require_cuda_kernel=self.require_cuda_kernel,
        )

    def close(self) -> None:
        self.loader.close()

    def __enter__(self) -> "KVSSDPipeline":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()
