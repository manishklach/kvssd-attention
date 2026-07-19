from __future__ import annotations

import hashlib
import json
import os
import struct
import threading
import zlib
from dataclasses import dataclass
from pathlib import Path

import torch

from ..format import ALIGNMENT, align_up
from ..quant import QuantizedTensor, dequantize_tensor, quantize_tensor


MAGIC = b"KVVLLM02"
HEADER = struct.Struct("<8sBBBBIIQI")
_DTYPE_TO_CODE = {torch.float16: 1, torch.bfloat16: 2, torch.float32: 3}
_CODE_TO_DTYPE = {value: key for key, value in _DTYPE_TO_CODE.items()}


@dataclass(frozen=True)
class VLLMRecordInfo:
    bits: int
    group_size: int
    dtype: torch.dtype
    elements: int
    payload_bytes: int


def _flat_tensor(view: memoryview, dtype: torch.dtype) -> torch.Tensor:
    byte_view = view.cast("B")
    item_size = torch.empty((), dtype=dtype).element_size()
    if len(byte_view) % item_size:
        raise ValueError(f"block byte length {len(byte_view)} is not divisible by {dtype}")
    return torch.frombuffer(byte_view, dtype=dtype)


def encode_vllm_block(
    view: memoryview,
    *,
    bits: int,
    group_size: int,
    dtype: torch.dtype,
) -> bytes:
    values = _flat_tensor(view, dtype).clone()
    if values.numel() % group_size:
        raise ValueError(
            f"vLLM block has {values.numel()} values, not divisible by group_size={group_size}"
        )
    quantized = quantize_tensor(values, bits, group_size)
    scales = quantized.scales.view(torch.uint8).numpy().tobytes()
    packed = quantized.packed.numpy().tobytes()
    payload = scales + packed
    crc = zlib.crc32(payload)
    header = HEADER.pack(
        MAGIC,
        1,
        bits,
        _DTYPE_TO_CODE[dtype],
        0,
        group_size,
        values.numel(),
        len(payload),
        crc,
    )
    record = header + payload
    return record + bytes(align_up(len(record), ALIGNMENT) - len(record))


def decode_vllm_block(raw: bytes, destination: memoryview) -> VLLMRecordInfo:
    if len(raw) < HEADER.size:
        raise ValueError("truncated KVSSD vLLM record")
    magic, version, bits, dtype_code, _, group_size, elements, payload_bytes, crc = (
        HEADER.unpack_from(raw)
    )
    if magic != MAGIC or version != 1:
        raise ValueError("invalid KVSSD vLLM record header")
    try:
        dtype = _CODE_TO_DTYPE[dtype_code]
    except KeyError as exc:
        raise ValueError(f"unsupported KVSSD vLLM dtype code {dtype_code}") from exc
    payload = raw[HEADER.size : HEADER.size + payload_bytes]
    if len(payload) != payload_bytes:
        raise ValueError("truncated KVSSD vLLM payload")
    if zlib.crc32(payload) != crc:
        raise IOError("KVSSD vLLM block CRC mismatch")
    scale_count = elements // group_size
    scale_bytes = scale_count * 2
    packed_bytes = elements * bits // 8
    if scale_bytes + packed_bytes != payload_bytes:
        raise ValueError("KVSSD vLLM payload layout is inconsistent")
    scales = torch.frombuffer(bytearray(payload[:scale_bytes]), dtype=torch.float16)
    packed = torch.frombuffer(bytearray(payload[scale_bytes:]), dtype=torch.uint8)
    quantized = QuantizedTensor(packed, scales, (elements,), bits, group_size)
    target = _flat_tensor(destination, dtype)
    if target.numel() != elements:
        raise ValueError(f"destination has {target.numel()} values, record requires {elements}")
    target.copy_(dequantize_tensor(quantized, dtype=dtype))
    return VLLMRecordInfo(bits, group_size, dtype, elements, payload_bytes)


def record_path(root: str | Path, key: bytes, namespace: str) -> Path:
    if len(key) < 5:
        raise ValueError("vLLM offload key must include a hash and four-byte group index")
    block_hash = key[:-4].hex()
    group = int.from_bytes(key[-4:], "big", signed=False)
    return (
        Path(root)
        / namespace
        / block_hash[:3]
        / f"{block_hash[3:5]}_g{group}"
        / (block_hash + ".kvssd")
    )


def configuration_namespace(configuration: dict[str, object]) -> str:
    canonical = json.dumps(configuration, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(canonical.encode()).hexdigest()[:16]
    model = str(configuration.get("model", "model")).replace("/", "_").replace("\\", "_")
    return f"{model}_{digest}"


def store_record_atomic(path: Path, record: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        return
    temporary = path.with_name(f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp")
    try:
        with temporary.open("xb") as output:
            output.write(record)
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary, path)
    finally:
        try:
            temporary.unlink()
        except FileNotFoundError:
            pass
