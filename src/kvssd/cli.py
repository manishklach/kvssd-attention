from __future__ import annotations

import argparse
import importlib.metadata
import json
import platform
import tempfile
import time
from dataclasses import asdict
from pathlib import Path

import torch

from .format import CacheSpec
from .pipeline import KVSSDPipeline
from .store import KVCacheStore


def _runtime_metadata(device: str) -> dict[str, object]:
    try:
        triton_version = importlib.metadata.version("triton")
    except importlib.metadata.PackageNotFoundError:
        triton_version = None
    gpu = torch.cuda.get_device_name() if device == "cuda" else None
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "triton": triton_version,
        "cuda": torch.version.cuda,
        "rocm": torch.version.hip,
        "gpu": gpu,
    }


def _create_demo(args: argparse.Namespace) -> None:
    generator = torch.Generator().manual_seed(args.seed)
    shape = (args.layers, args.tokens, args.kv_heads, args.head_dim)
    keys = torch.randn(shape, generator=generator, dtype=torch.float16)
    values = torch.randn(shape, generator=generator, dtype=torch.float16)
    spec = CacheSpec(
        args.layers, args.kv_heads, args.head_dim, args.block_tokens, args.bits, args.group_size
    )
    store = KVCacheStore.create(
        args.output, keys, values, spec, storage_backend=args.storage_backend
    )
    logical = 2 * keys.numel() * keys.element_size()
    physical = (Path(args.output) / "blocks.kvssd").stat().st_size
    print(
        json.dumps(
            {
                "path": str(Path(args.output).resolve()),
                "logical_bytes": logical,
                "physical_bytes": physical,
                "effective_ratio": logical / physical,
                "records": len(store.manifest.entries),
                "storage_backend": asdict(store.storage_backend.probe()),
            },
            indent=2,
        )
    )


def _inspect(args: argparse.Namespace) -> None:
    with KVCacheStore.open(args.path, storage_backend=args.storage_backend) as store:
        print(
            json.dumps(
                {
                    "spec": store.manifest.spec.__dict__,
                    "records": len(store.manifest.entries),
                    "data_bytes": (Path(args.path) / "blocks.kvssd").stat().st_size,
                    "storage_backend": asdict(store.storage_backend.probe()),
                },
                indent=2,
            )
        )


def _benchmark(args: argparse.Namespace) -> None:
    device = "cuda" if torch.cuda.is_available() and not args.cpu else "cpu"
    work_root = Path(args.work_dir).expanduser().resolve() if args.work_dir else None
    if work_root is not None:
        work_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=work_root) as tmp:
        spec = CacheSpec(
            1, args.kv_heads, args.head_dim, args.block_tokens, args.bits, args.group_size
        )
        shape = (1, args.tokens, args.kv_heads, args.head_dim)
        keys, values = (
            torch.randn(shape, dtype=torch.float16),
            torch.randn(shape, dtype=torch.float16),
        )
        store = KVCacheStore.create(tmp, keys, values, spec, storage_backend=args.storage_backend)
        query = torch.randn(args.query_heads, args.head_dim, dtype=torch.float16)
        with KVSSDPipeline(
            store,
            device=device,
            require_cuda_kernel=args.require_cuda,
            attention_backend=args.attention_backend,
        ) as pipeline:
            attention_availability = pipeline.attention_backend.probe(pipeline.device)
            storage_availability = store.storage_backend.probe()
            for _ in range(args.warmup):
                pipeline.decode(query, 0)
            if device == "cuda":
                torch.cuda.synchronize()
            start = time.perf_counter()
            for _ in range(args.iterations):
                pipeline.decode(query, 0)
            if device == "cuda":
                torch.cuda.synchronize()
            elapsed = time.perf_counter() - start
        store.close()
        size = (Path(tmp) / "blocks.kvssd").stat().st_size
        print(
            json.dumps(
                {
                    "device": device,
                    "work_root": str(work_root) if work_root is not None else None,
                    "runtime": _runtime_metadata(device),
                    "shape": {
                        "tokens": args.tokens,
                        "query_heads": args.query_heads,
                        "kv_heads": args.kv_heads,
                        "head_dim": args.head_dim,
                        "block_tokens": args.block_tokens,
                        "bits": args.bits,
                        "group_size": args.group_size,
                    },
                    "attention_backend": asdict(attention_availability),
                    "storage_backend": asdict(storage_availability),
                    "iterations": args.iterations,
                    "mean_ms": elapsed * 1e3 / args.iterations,
                    "ssd_payload_mib": size / 2**20,
                    "effective_read_gib_s": size * args.iterations / elapsed / 2**30,
                },
                indent=2,
            )
        )


def main() -> None:
    parser = argparse.ArgumentParser(prog="kvssd")
    sub = parser.add_subparsers(required=True)
    create = sub.add_parser("create-demo", help="write a deterministic synthetic KV store")
    create.add_argument("output")
    create.add_argument("--layers", type=int, default=2)
    create.add_argument("--tokens", type=int, default=1024)
    create.add_argument("--kv-heads", type=int, default=8)
    create.add_argument("--head-dim", type=int, default=128)
    create.add_argument("--block-tokens", type=int, default=128)
    create.add_argument("--bits", type=int, choices=(2, 4), default=4)
    create.add_argument("--group-size", type=int, default=32)
    create.add_argument("--seed", type=int, default=7)
    create.add_argument("--storage-backend", default="auto")
    create.set_defaults(func=_create_demo)
    inspect = sub.add_parser("inspect", help="inspect a store manifest")
    inspect.add_argument("path")
    inspect.add_argument("--storage-backend", default="auto")
    inspect.set_defaults(func=_inspect)
    bench = sub.add_parser("benchmark", help="run the complete store-to-attention path")
    bench.add_argument("--tokens", type=int, default=8192)
    bench.add_argument("--kv-heads", type=int, default=8)
    bench.add_argument("--query-heads", type=int, default=32)
    bench.add_argument("--head-dim", type=int, default=128)
    bench.add_argument("--block-tokens", type=int, default=128)
    bench.add_argument("--bits", type=int, choices=(2, 4), default=4)
    bench.add_argument("--group-size", type=int, default=32)
    bench.add_argument("--warmup", type=int, default=2)
    bench.add_argument("--iterations", type=int, default=10)
    bench.add_argument("--cpu", action="store_true")
    bench.add_argument("--require-cuda", action="store_true")
    bench.add_argument("--attention-backend", default="auto")
    bench.add_argument("--storage-backend", default="auto")
    bench.add_argument(
        "--work-dir",
        help="parent directory for benchmark stores (use the qualified NVMe mount for GDS)",
    )
    bench.set_defaults(func=_benchmark)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
