from __future__ import annotations

import json
import struct
import zlib
from dataclasses import asdict, dataclass
from pathlib import Path

import torch

from .quant import QuantizedTensor


ALIGNMENT = 4096
MAGIC = b"KVSSD001"
HEADER = struct.Struct("<8sIIIIIIIIIII")


def align_up(value: int, alignment: int = ALIGNMENT) -> int:
    return (value + alignment - 1) // alignment * alignment


@dataclass(frozen=True)
class CacheSpec:
    layers: int
    kv_heads: int
    head_dim: int
    block_tokens: int = 128
    bits: int = 4
    group_size: int = 32

    def validate(self) -> None:
        if min(self.layers, self.kv_heads, self.head_dim, self.block_tokens) <= 0:
            raise ValueError("all cache dimensions must be positive")
        if self.bits not in (2, 4):
            raise ValueError("bits must be 2 or 4")
        if self.head_dim % self.group_size:
            raise ValueError("head_dim must be divisible by group_size")

    @property
    def packed_dim(self) -> int:
        return self.head_dim * self.bits // 8

    @property
    def scale_count(self) -> int:
        return self.block_tokens * self.kv_heads * (self.head_dim // self.group_size)

    @property
    def packed_count(self) -> int:
        return self.block_tokens * self.kv_heads * self.packed_dim

    @property
    def payload_bytes(self) -> int:
        return 2 * self.scale_count * 2 + 2 * self.packed_count

    @property
    def record_bytes(self) -> int:
        return align_up(HEADER.size + self.payload_bytes)


@dataclass(frozen=True)
class BlockEntry:
    layer: int
    block: int
    offset: int
    length: int
    valid_tokens: int
    crc32: int


@dataclass
class Manifest:
    spec: CacheSpec
    entries: list[BlockEntry]
    version: int = 1

    def save(self, path: Path) -> None:
        body = {
            "version": self.version,
            "spec": asdict(self.spec),
            "entries": [asdict(x) for x in self.entries],
        }
        path.write_text(json.dumps(body, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "Manifest":
        body = json.loads(path.read_text(encoding="utf-8"))
        return cls(
            spec=CacheSpec(**body["spec"]),
            entries=[BlockEntry(**x) for x in body["entries"]],
            version=body["version"],
        )


@dataclass(frozen=True)
class PackedKVBlock:
    layer: int
    block: int
    valid_tokens: int
    key: QuantizedTensor
    value: QuantizedTensor


def encode_record(block: PackedKVBlock, spec: CacheSpec) -> tuple[bytes, int]:
    expected = (spec.block_tokens, spec.kv_heads, spec.head_dim)
    if block.key.original_shape != expected or block.value.original_shape != expected:
        raise ValueError(f"padded KV block must have shape {expected}")
    chunks = (
        block.key.scales.cpu().contiguous().view(torch.uint8).numpy().tobytes(),
        block.value.scales.cpu().contiguous().view(torch.uint8).numpy().tobytes(),
        block.key.packed.cpu().contiguous().numpy().tobytes(),
        block.value.packed.cpu().contiguous().numpy().tobytes(),
    )
    payload = b"".join(chunks)
    crc = zlib.crc32(payload)
    header = HEADER.pack(
        MAGIC, 1, block.layer, block.block, block.valid_tokens, spec.bits,
        spec.block_tokens, spec.kv_heads, spec.head_dim, spec.group_size, len(payload), crc,
    )
    record = header + payload
    return record + bytes(spec.record_bytes - len(record)), crc


def decode_record(raw: torch.Tensor, spec: CacheSpec, verify_crc: bool = True) -> PackedKVBlock:
    if raw.dtype != torch.uint8 or raw.device.type != "cpu" or not raw.is_contiguous():
        raise TypeError("raw record must be a contiguous CPU uint8 tensor")
    view = memoryview(raw.numpy())
    fields = HEADER.unpack(bytes(view[: HEADER.size]))
    magic, version, layer, block, valid, bits, bt, heads, dim, group, payload_len, crc = fields
    if magic != MAGIC or version != 1:
        raise ValueError("invalid KVSSD record header")
    if (bits, bt, heads, dim, group) != (
        spec.bits, spec.block_tokens, spec.kv_heads, spec.head_dim, spec.group_size
    ):
        raise ValueError("record layout does not match manifest")
    payload = view[HEADER.size : HEADER.size + payload_len]
    if verify_crc and zlib.crc32(payload) != crc:
        raise IOError(f"CRC mismatch for layer={layer}, block={block}")

    scales_bytes = spec.scale_count * 2
    packed_bytes = spec.packed_count
    cursor = HEADER.size

    def slice_tensor(count: int) -> torch.Tensor:
        nonlocal cursor
        out = raw[cursor : cursor + count]
        cursor += count
        return out

    scale_shape = (spec.block_tokens, spec.kv_heads, spec.head_dim // spec.group_size)
    packed_shape = (spec.block_tokens, spec.kv_heads, spec.packed_dim)
    ks = slice_tensor(scales_bytes).view(torch.float16).reshape(scale_shape)
    vs = slice_tensor(scales_bytes).view(torch.float16).reshape(scale_shape)
    kp = slice_tensor(packed_bytes).reshape(packed_shape)
    vp = slice_tensor(packed_bytes).reshape(packed_shape)
    shape = (spec.block_tokens, spec.kv_heads, spec.head_dim)
    return PackedKVBlock(
        layer=layer,
        block=block,
        valid_tokens=valid,
        key=QuantizedTensor(kp, ks, shape, spec.bits, spec.group_size),
        value=QuantizedTensor(vp, vs, shape, spec.bits, spec.group_size),
    )

