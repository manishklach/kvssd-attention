from __future__ import annotations

import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from enum import Enum, auto
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from kvssd.integrations import vllm as integration
from kvssd.integrations.vllm_codec import (
    decode_vllm_block,
    encode_vllm_block,
    record_path,
)


@pytest.mark.parametrize("bits", [2, 4])
def test_vllm_block_codec_round_trip(bits: int) -> None:
    source = torch.linspace(-2, 2, 64, dtype=torch.float16)
    source_bytes = bytearray(source.view(torch.uint8).numpy().tobytes())
    record = encode_vllm_block(
        memoryview(source_bytes), bits=bits, group_size=16, dtype=torch.float16
    )
    destination = bytearray(len(source_bytes))
    info = decode_vllm_block(record, memoryview(destination))
    restored = torch.frombuffer(destination, dtype=torch.float16)
    tolerance = 1.1 if bits == 2 else 0.2
    torch.testing.assert_close(restored, source, atol=tolerance, rtol=0)
    assert len(record) % 4096 == 0
    assert info.bits == bits


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_vllm_block_codec_preserves_declared_dtype(dtype: torch.dtype) -> None:
    source = torch.linspace(-1, 1, 64, dtype=dtype)
    source_bytes = bytearray(source.view(torch.uint8).numpy().tobytes())
    record = encode_vllm_block(memoryview(source_bytes), bits=4, group_size=16, dtype=dtype)
    destination = bytearray(len(source_bytes))
    info = decode_vllm_block(record, memoryview(destination))
    restored = torch.frombuffer(destination, dtype=dtype)
    torch.testing.assert_close(restored.float(), source.float(), atol=0.1, rtol=0)
    assert info.dtype == dtype


def test_vllm_codec_detects_corruption() -> None:
    source = bytearray(torch.ones(32, dtype=torch.float16).view(torch.uint8).numpy().tobytes())
    record = bytearray(
        encode_vllm_block(memoryview(source), bits=4, group_size=16, dtype=torch.float16)
    )
    record[40] ^= 1
    with pytest.raises(IOError, match="CRC"):
        decode_vllm_block(bytes(record), memoryview(bytearray(len(source))))


def test_vllm_record_mapping_includes_group(tmp_path: Path) -> None:
    block_hash = bytes.fromhex("ab12cd34")
    key = block_hash + (7).to_bytes(4, "big")
    path = record_path(tmp_path, key, "model_digest")
    assert path.name == "ab12cd34.kvssd"
    assert path.parent.name.endswith("_g7")
    assert path.is_relative_to(tmp_path / "model_digest")


class _LookupResult(Enum):
    MISS = auto()
    HIT = auto()
    RETRY = auto()


@dataclass
class _JobResult:
    job_id: int
    success: bool


def _fake_manager(tmp_path: Path):
    values = np.linspace(-1, 1, 128, dtype=np.float16).reshape(2, 64)
    manager = object.__new__(integration.KVSSDSecondaryTierManager)
    manager._primary_kv_view = memoryview(values)
    manager._block_bytes = values.strides[0]
    manager._root = tmp_path
    manager._bits = 4
    manager._group_size = 16
    manager._dtype = torch.float16
    manager.namespace = "fake_model_contract"
    manager._pool = ThreadPoolExecutor(max_workers=2)
    manager._futures = {}
    manager._lookup_futures = {}
    manager._inflight_keys = Counter()
    manager._lock = threading.Lock()
    return manager, values


def test_fake_vllm_runtime_store_evict_reload_contract(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(integration, "LookupResult", _LookupResult)
    monkeypatch.setattr(integration, "JobResult", _JobResult)
    manager, values = _fake_manager(tmp_path)
    original = values[0].copy()
    key = b"hash-for-prefix" + (0).to_bytes(4, "big")
    request = SimpleNamespace(req_id="request-1")
    job = SimpleNamespace(job_id=11, keys=[key], block_ids=np.array([0]))

    assert manager.lookup(key, request) is _LookupResult.RETRY
    for future in tuple(manager._lookup_futures.values()):
        future.result()
    assert manager.lookup(key, request) is _LookupResult.MISS
    manager.submit_store(job)
    assert manager.lookup(key, request) is _LookupResult.RETRY
    for future in tuple(manager._futures):
        future.result()
    assert list(manager.get_finished_jobs()) == [_JobResult(11, True)]
    assert manager.lookup(key, request) is _LookupResult.RETRY
    for future in tuple(manager._lookup_futures.values()):
        future.result()
    assert manager.lookup(key, request) is _LookupResult.HIT

    values[0].fill(0)
    manager.submit_load(SimpleNamespace(job_id=12, keys=[key], block_ids=np.array([0])))
    for future in tuple(manager._futures):
        future.result()
    assert list(manager.get_finished_jobs()) == [_JobResult(12, True)]
    np.testing.assert_allclose(values[0], original, atol=0.15, rtol=0)
    manager.shutdown()


def test_vllm_optional_import_has_diagnostic() -> None:
    available, reason = integration.vllm_available()
    assert isinstance(available, bool)
    assert reason
