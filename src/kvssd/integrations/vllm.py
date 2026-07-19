"""vLLM 0.25 custom offloading spec and INT2/INT4 secondary tier.

Use ``kvssd.integrations.vllm`` as vLLM's ``spec_module_path`` and
``KVSSDOffloadingSpec`` as ``spec_name``. Importing this module without vLLM is safe;
constructing the spec is not.
"""

from __future__ import annotations

import threading
from collections import Counter
from collections.abc import Iterable
from concurrent.futures import Future, ThreadPoolExecutor, wait
from pathlib import Path
from typing import Any

import torch

from .vllm_codec import (
    configuration_namespace,
    decode_vllm_block,
    encode_vllm_block,
    record_path,
    store_record_atomic,
)

try:
    from vllm.v1.kv_offload.base import LookupResult, RequestOffloadingContext
    from vllm.v1.kv_offload.tiering.base import JobResult, SecondaryTierManager
    from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory
    from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec
except ImportError as exc:
    _VLLM_IMPORT_ERROR: ImportError | None = exc
    LookupResult = RequestOffloadingContext = JobResult = None
    SecondaryTierManager = TieringOffloadingSpec = object
    SecondaryTierFactory = None
else:
    _VLLM_IMPORT_ERROR = None


def _dtype(name: str) -> torch.dtype:
    choices = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    try:
        return choices[name]
    except KeyError as exc:
        raise ValueError(f"kvssd_dtype must be one of {', '.join(choices)}") from exc


class KVSSDSecondaryTierManager(SecondaryTierManager):
    """Asynchronous vLLM CPU-tier ↔ packed-KVSSD-file bridge."""

    def __init__(
        self,
        offloading_spec,
        primary_kv_view: memoryview,
        tier_type: str,
        root_dir: str,
        bits: int = 4,
        group_size: int = 32,
        kvssd_dtype: str = "float16",
        n_io_threads: int = 8,
    ) -> None:
        if _VLLM_IMPORT_ERROR is not None:
            raise RuntimeError(
                "vLLM 0.25 is required for KVSSDOffloadingSpec"
            ) from _VLLM_IMPORT_ERROR
        super().__init__(offloading_spec, primary_kv_view, tier_type)
        if bits not in (2, 4):
            raise ValueError("bits must be 2 or 4")
        if group_size <= 0 or group_size % (8 // bits):
            raise ValueError("group_size must be positive and pack-aligned")
        if n_io_threads < 1:
            raise ValueError("n_io_threads must be positive")
        assert primary_kv_view.strides is not None
        self._block_bytes = primary_kv_view.strides[0]
        self._root = Path(root_dir)
        self._bits = bits
        self._group_size = group_size
        self._dtype = _dtype(kvssd_dtype)
        config = offloading_spec.vllm_config
        model_config = config.model_config
        parallel = config.parallel_config
        identity = {
            "model": getattr(model_config, "model", "model"),
            "dtype": kvssd_dtype,
            "bits": bits,
            "group_size": group_size,
            "block_bytes": self._block_bytes,
            "tensor_parallel": parallel.tensor_parallel_size,
            "pipeline_parallel": parallel.pipeline_parallel_size,
        }
        self.namespace = configuration_namespace(identity)
        self._pool = ThreadPoolExecutor(max_workers=n_io_threads, thread_name_prefix="kvssd-vllm")
        self._futures: dict[Future[None], tuple[int, tuple[bytes, ...]]] = {}
        self._lookup_futures: dict[tuple[str, bytes], Future[bool]] = {}
        self._inflight_keys: Counter[bytes] = Counter()
        self._lock = threading.Lock()

    def _block_view(self, block_id: int) -> memoryview:
        start = block_id * self._block_bytes
        return self._primary_kv_view.cast("B")[start : start + self._block_bytes]

    def _path(self, key: bytes) -> Path:
        return record_path(self._root, bytes(key), self.namespace)

    def lookup(self, key, req_context):
        raw_key = bytes(key)
        with self._lock:
            if self._inflight_keys[raw_key] > 0:
                return LookupResult.RETRY
            lookup_key = (req_context.req_id, raw_key)
            future = self._lookup_futures.get(lookup_key)
            if future is None:
                self._lookup_futures[lookup_key] = self._pool.submit(self._path(raw_key).is_file)
                return LookupResult.RETRY
            if not future.done():
                return LookupResult.RETRY
            del self._lookup_futures[lookup_key]
        return LookupResult.HIT if future.result() else LookupResult.MISS

    def _store_job(self, keys, block_ids) -> None:
        for key, block_id in zip(keys, block_ids):
            path = self._path(key)
            record = encode_vllm_block(
                self._block_view(int(block_id)),
                bits=self._bits,
                group_size=self._group_size,
                dtype=self._dtype,
            )
            store_record_atomic(path, record)

    def _load_job(self, keys, block_ids) -> None:
        for key, block_id in zip(keys, block_ids):
            decode_vllm_block(self._path(key).read_bytes(), self._block_view(int(block_id)))

    def _submit(self, job_metadata, function) -> None:
        keys = tuple(bytes(key) for key in job_metadata.keys)
        block_ids = tuple(int(block_id) for block_id in job_metadata.block_ids)
        with self._lock:
            self._inflight_keys.update(keys)
            future = self._pool.submit(function, keys, block_ids)
            self._futures[future] = (job_metadata.job_id, keys)

    def submit_store(self, job_metadata) -> None:
        self._submit(job_metadata, self._store_job)

    def submit_load(self, job_metadata) -> None:
        self._submit(job_metadata, self._load_job)

    def get_finished_jobs(self) -> Iterable[Any]:
        finished = []
        with self._lock:
            for future, (job_id, keys) in list(self._futures.items()):
                if not future.done():
                    continue
                del self._futures[future]
                self._inflight_keys.subtract(keys)
                self._inflight_keys += Counter()
                finished.append(JobResult(job_id=job_id, success=future.exception() is None))
        return finished

    def has_pending_work(self) -> bool:
        with self._lock:
            return bool(self._futures) or any(
                not future.done() for future in self._lookup_futures.values()
            )

    def on_new_request(self, req_context):
        del req_context
        return RequestOffloadingContext()

    def on_request_finished(self, req_context) -> None:
        with self._lock:
            stale = [key for key in self._lookup_futures if key[0] == req_context.req_id]
            for key in stale:
                del self._lookup_futures[key]

    def drain_jobs(self) -> None:
        while True:
            with self._lock:
                futures = tuple(self._futures)
                lookups = tuple(self._lookup_futures.values())
            if not futures and not lookups:
                return
            wait((*futures, *lookups))
            self.get_finished_jobs()
            with self._lock:
                self._lookup_futures.clear()

    def shutdown(self) -> None:
        self._pool.shutdown(wait=True, cancel_futures=False)


class KVSSDOffloadingSpec(TieringOffloadingSpec):
    """vLLM spec that selects the built-in CPU primary tier plus KVSSD storage."""

    def __init__(self, vllm_config, kv_cache_config):
        if _VLLM_IMPORT_ERROR is not None:
            raise RuntimeError(
                "vLLM 0.25 is required for KVSSDOffloadingSpec"
            ) from _VLLM_IMPORT_ERROR
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config
        tiers = extra.get("secondary_tiers", [])
        if not any(tier.get("type") == "kvssd" for tier in tiers):
            raise ValueError("KVSSDOffloadingSpec requires a secondary tier with type='kvssd'")
        super().__init__(vllm_config, kv_cache_config)


if SecondaryTierFactory is not None and "kvssd" not in SecondaryTierFactory._registry:
    SecondaryTierFactory.register_tier(
        "kvssd", "kvssd.integrations.vllm", "KVSSDSecondaryTierManager"
    )


def vllm_available() -> tuple[bool, str]:
    if _VLLM_IMPORT_ERROR is not None:
        return False, f"vLLM integration unavailable: {_VLLM_IMPORT_ERROR}"
    return True, "vLLM 0.25 offloading interfaces imported"
