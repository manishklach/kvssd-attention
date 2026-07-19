from __future__ import annotations

from pathlib import Path

import torch

from .format import BlockEntry, CacheSpec, Manifest, PackedKVBlock, decode_record, encode_record
from .quant import quantize_tensor
from .storage import StorageBackend, resolve_storage_backend


class KVCacheStore:
    """Append-only fixed-record KV cache store designed for large sequential SSD reads."""

    def __init__(
        self,
        root: str | Path,
        manifest: Manifest,
        storage_backend: str | StorageBackend | None = None,
    ):
        self.root = Path(root)
        self.manifest = manifest
        self.spec = manifest.spec
        self._entries = {(x.layer, x.block): x for x in manifest.entries}
        self.storage_backend = resolve_storage_backend(storage_backend, self.root / "blocks.kvssd")

    @classmethod
    def create(
        cls,
        root: str | Path,
        keys: torch.Tensor,
        values: torch.Tensor,
        spec: CacheSpec,
        storage_backend: str | StorageBackend | None = None,
    ) -> "KVCacheStore":
        """Create a store from [layers, tokens, kv_heads, head_dim] tensors."""
        spec.validate()
        if keys.shape != values.shape or keys.ndim != 4:
            raise ValueError("keys and values must share [layers,tokens,kv_heads,head_dim]")
        if keys.shape[0] != spec.layers or tuple(keys.shape[2:]) != (spec.kv_heads, spec.head_dim):
            raise ValueError("input tensors do not match CacheSpec")
        root = Path(root)
        root.mkdir(parents=True, exist_ok=True)
        entries: list[BlockEntry] = []
        with (root / "blocks.kvssd").open("wb") as output:
            for layer in range(spec.layers):
                for block_id, start in enumerate(range(0, keys.shape[1], spec.block_tokens)):
                    valid = min(spec.block_tokens, keys.shape[1] - start)
                    shape = (spec.block_tokens, spec.kv_heads, spec.head_dim)
                    k = torch.zeros(shape, dtype=keys.dtype)
                    v = torch.zeros(shape, dtype=values.dtype)
                    k[:valid].copy_(keys[layer, start : start + valid].cpu())
                    v[:valid].copy_(values[layer, start : start + valid].cpu())
                    packed = PackedKVBlock(
                        layer,
                        block_id,
                        valid,
                        quantize_tensor(k, spec.bits, spec.group_size),
                        quantize_tensor(v, spec.bits, spec.group_size),
                    )
                    record, crc = encode_record(packed, spec)
                    offset = output.tell()
                    output.write(record)
                    entries.append(BlockEntry(layer, block_id, offset, len(record), valid, crc))
        manifest = Manifest(spec=spec, entries=entries)
        manifest.save(root / "manifest.json")
        return cls(root, manifest, storage_backend)

    @classmethod
    def open(
        cls,
        root: str | Path,
        storage_backend: str | StorageBackend | None = None,
    ) -> "KVCacheStore":
        root = Path(root)
        manifest = Manifest.load(root / "manifest.json")
        manifest.spec.validate()
        return cls(root, manifest, storage_backend)

    def entry(self, layer: int, block: int) -> BlockEntry:
        try:
            return self._entries[(layer, block)]
        except KeyError as exc:
            raise KeyError(f"no KV block layer={layer}, block={block}") from exc

    def blocks_for_layer(self, layer: int) -> list[int]:
        return sorted(x.block for x in self.manifest.entries if x.layer == layer)

    def read_into(self, entry: BlockEntry, destination: torch.Tensor) -> None:
        self.storage_backend.read_into(entry, destination)

    def read(self, layer: int, block: int, verify_crc: bool = True) -> PackedKVBlock:
        entry = self.entry(layer, block)
        raw = self.storage_backend.allocate_buffer(entry.length)
        self.read_into(entry, raw)
        if raw.device.type != "cpu":
            raw = raw.cpu()
        return decode_record(raw, self.spec, verify_crc)

    def close(self) -> None:
        self.storage_backend.close()

    def __enter__(self) -> "KVCacheStore":
        return self

    def __exit__(self, *_: object) -> None:
        self.close()

    def __del__(self) -> None:
        try:
            self.close()
        except Exception:
            pass
