"""Exercise KVSSD against the exact vLLM 0.25 offloading abstract classes.

GPU-specific vLLM components are stubbed. The base, secondary-tier, factory, and tiering-spec
definitions are loaded from the supplied vLLM source checkout and used at runtime.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import logging
import sys
import tempfile
import time
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np


def _package(name: str) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__path__ = []  # type: ignore[attr-defined]
    sys.modules[name] = module
    return module


def _module(name: str, **members: object) -> types.ModuleType:
    module = types.ModuleType(name)
    vars(module).update(members)
    sys.modules[name] = module
    return module


def _load(name: str, path: Path) -> types.ModuleType:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot load {name} from {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def _install_vllm_contract(source: Path) -> tuple[types.ModuleType, types.ModuleType, type]:
    for name in (
        "vllm",
        "vllm.v1",
        "vllm.v1.core",
        "vllm.v1.kv_offload",
        "vllm.v1.kv_offload.cpu",
        "vllm.v1.kv_offload.tiering",
    ):
        _package(name)

    _module("vllm.logger", init_logger=logging.getLogger)

    class VllmConfig:
        pass

    class KVCacheConfig:
        pass

    _module("vllm.config", VllmConfig=VllmConfig)
    _module("vllm.v1.kv_cache_interface", KVCacheConfig=KVCacheConfig)
    _module(
        "vllm.v1.core.kv_cache_utils",
        resolve_kv_cache_block_sizes=lambda *_: ((16,), 16),
    )

    root = source / "vllm" / "v1" / "kv_offload"
    base = _load("vllm.v1.kv_offload.base", root / "base.py")
    tiering_base = _load("vllm.v1.kv_offload.tiering.base", root / "tiering" / "base.py")
    _load("vllm.v1.kv_offload.tiering.factory", root / "tiering" / "factory.py")

    class CPUOffloadingWorker:
        pass

    class SharedOffloadRegion:
        BLOCK_SIZE_ALIGNMENT = 4096

    class CPUOffloadingSpec:
        @classmethod
        def build_metric_definitions(cls, extra_config):
            del cls, extra_config
            return {}

        def __init__(self, vllm_config, kv_cache_config):
            self.vllm_config = vllm_config
            self.kv_cache_config = kv_cache_config
            self.extra_config = vllm_config.kv_transfer_config.kv_connector_extra_config
            self.kv_events_config = SimpleNamespace(
                enable_kv_cache_events=False,
                self_describing_kv_events=False,
            )
            self.num_blocks = 2
            self.kv_bytes_per_offloaded_block = 4096
            self.cpu_page_size_per_worker = 4096
            self.eviction_policy = "lru"
            self.block_size_factor = 1

    class CPUPrimaryTierOffloadingManager:
        pass

    class TieringOffloadingManager:
        pass

    _module("vllm.v1.kv_offload.cpu.gpu_worker", CPUOffloadingWorker=CPUOffloadingWorker)
    _module(
        "vllm.v1.kv_offload.cpu.shared_offload_region",
        SharedOffloadRegion=SharedOffloadRegion,
    )
    _module("vllm.v1.kv_offload.cpu.spec", CPUOffloadingSpec=CPUOffloadingSpec)
    _module(
        "vllm.v1.kv_offload.tiering.manager",
        CPUPrimaryTierOffloadingManager=CPUPrimaryTierOffloadingManager,
        TieringOffloadingManager=TieringOffloadingManager,
    )
    tiering_spec = _load("vllm.v1.kv_offload.tiering.spec", root / "tiering" / "spec.py")
    return base, tiering_base, tiering_spec.TieringOffloadingSpec


def _poll_lookup(manager, key, context, retry, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while True:
        result = manager.lookup(key, context)
        if result is not retry:
            return result
        if time.monotonic() >= deadline:
            raise TimeoutError("vLLM lookup contract did not complete")
        time.sleep(0.005)


def _poll_job(manager, job_id: int, timeout: float = 5.0):
    deadline = time.monotonic() + timeout
    while True:
        for result in manager.get_finished_jobs():
            if result.job_id == job_id:
                return result
        if time.monotonic() >= deadline:
            raise TimeoutError(f"vLLM transfer job {job_id} did not complete")
        time.sleep(0.005)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("vllm_source", type=Path)
    args = parser.parse_args()
    base, tiering_base, tiering_spec = _install_vllm_contract(args.vllm_source)

    sys.modules.pop("kvssd.integrations.vllm", None)
    integration = importlib.import_module("kvssd.integrations.vllm")
    assert issubclass(integration.KVSSDOffloadingSpec, tiering_spec)
    assert not getattr(integration.KVSSDSecondaryTierManager, "__abstractmethods__", set())
    assert (
        "kvssd" in sys.modules["vllm.v1.kv_offload.tiering.factory"].SecondaryTierFactory._registry
    )

    values = np.linspace(-1, 1, 128, dtype=np.float16).reshape(2, 64)
    original = values[0].copy()
    config = SimpleNamespace(
        model_config=SimpleNamespace(model="runtime-contract-model"),
        parallel_config=SimpleNamespace(tensor_parallel_size=1, pipeline_parallel_size=1),
    )
    offloading_spec = SimpleNamespace(vllm_config=config)
    with tempfile.TemporaryDirectory() as root:
        manager = integration.KVSSDSecondaryTierManager(
            offloading_spec,
            memoryview(values),
            "kvssd",
            root,
            bits=4,
            group_size=16,
            kvssd_dtype="float16",
            n_io_threads=2,
        )
        key = base.make_offload_key(b"runtime-contract-hash", 0)
        context = base.ReqContext("request-1")
        assert (
            _poll_lookup(manager, key, context, base.LookupResult.RETRY) is base.LookupResult.MISS
        )

        store = tiering_base.JobMetadata(1, [key], np.array([0]), False, context)
        manager.submit_store(store)
        assert _poll_job(manager, 1).success
        assert _poll_lookup(manager, key, context, base.LookupResult.RETRY) is base.LookupResult.HIT

        values[0].fill(0)
        load = tiering_base.JobMetadata(2, [key], np.array([0]), True, context)
        manager.submit_load(load)
        assert _poll_job(manager, 2).success
        np.testing.assert_allclose(values[0], original, atol=0.15, rtol=0)
        manager.on_request_finished(context)
        manager.drain_jobs()
        manager.shutdown()

    print("vLLM 0.25 source-backed runtime contract is compatible")


if __name__ == "__main__":
    main()
