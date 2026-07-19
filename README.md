# KVSSD Attention

[![CI](https://github.com/manishklach/kvssd-attention/actions/workflows/ci.yml/badge.svg)](https://github.com/manishklach/kvssd-attention/actions/workflows/ci.yml)
[![Release](https://img.shields.io/github/v/release/manishklach/kvssd-attention)](https://github.com/manishklach/kvssd-attention/releases)
[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)

KVSSD Attention is a reference implementation of this serving path:

```text
2-bit or 4-bit KV blocks on NVMe
  → bounded asynchronous reads
  → pinned host staging
  → non-blocking CUDA-stream transfers
  → fused register-level dequantization and decode attention
```

The project is deliberately narrow. It implements single-query-token decode attention, including
grouped-query attention (GQA), rather than presenting an entire LLM server. The Python API is small
enough to integrate into a serving runtime, while the storage format, scheduler boundary, and CUDA
kernel are independently testable.

## Why this exists

Long-context decoding repeatedly reads a growing KV cache. Quantization reduces those bytes, but
capacity alone is not enough: a useful storage-tiered design must coordinate SSD layout, read queue
depth, pinned-memory ownership, PCIe transfers, CUDA streams, and attention numerics. KVSSD Attention
puts those pieces in one auditable repository instead of measuring an isolated codec or kernel.

The central rule is that a decompressed cache is never materialized in HBM. Packed blocks move from
storage to the GPU, and the CUDA kernel reconstructs individual values only when they are consumed.

## Project status

`v0.1.1` is an engineering reference release. The portable correctness suite, storage lifecycle,
packaging, and CPU end-to-end path are verified. The CUDA implementation is included and fails closed
when unavailable, but its numerical and performance validation requires a real NVIDIA GPU. No GPU
throughput claims are made from the AMD-only development host.

## What is implemented

- Symmetric groupwise INT2 and INT4 quantization with FP16 scales.
- Fixed-size, 4 KiB-aligned, append-only SSD records with per-record CRC32.
- Pluggable block selection (`AllBlocks` and sink-plus-recent are included).
- Bounded threaded `preadv` loading, with a portable Windows fallback.
- Pinned, coalesced staging buffers and dedicated CUDA transfer/compute streams.
- Chunked H2D/compute overlap with numerically correct online-softmax result merging.
- A fused CUDA kernel that unpacks and dequantizes K/V in registers and never writes a full
  decompressed KV cache to HBM.
- CPU reference implementation, CLI, tests, benchmark command, and CI.

## Quick start

CPU/reference installation and validation:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e '.[dev]'
pytest

kvssd create-demo /tmp/demo-kv --tokens 4096 --bits 4
kvssd inspect /tmp/demo-kv
kvssd benchmark --cpu --tokens 8192 --bits 4
```

CUDA fast-path installation requires a CUDA-enabled PyTorch installation, the CUDA toolkit, and a
working C++ compiler:

```bash
pip install torch --index-url https://download.pytorch.org/whl/cu128
KVSSD_BUILD_CUDA=1 pip install -e . --no-build-isolation
kvssd benchmark --tokens 131072 --bits 4 --require-cuda
```

Use the PyTorch CUDA wheel/index appropriate for your driver; `cu128` above is only an example.

## Python example

```python
import torch
from kvssd import CacheSpec, KVCacheStore, KVSSDPipeline, RecentSinkBlocks

# K/V layout: [layers, tokens, kv_heads, head_dim]
keys = torch.randn(32, 65_536, 8, 128, dtype=torch.float16)
values = torch.randn_like(keys)
spec = CacheSpec(
    layers=32,
    kv_heads=8,
    head_dim=128,
    block_tokens=128,
    bits=4,
    group_size=32,
)
store = KVCacheStore.create("./cache", keys, values, spec)

# Query layout: [query_heads, head_dim]; 32 query heads / 8 KV heads is GQA(4).
query = torch.randn(32, 128, device="cuda", dtype=torch.float16)
with KVSSDPipeline(store, stage_blocks=4, require_cuda_kernel=True) as pipeline:
    output = pipeline.decode(query, layer=0, selector=RecentSinkBlocks(sink=1, recent=32))
```

`KVCacheStore.create` is an offline/reference writer. A production integration should append blocks
directly as prefill produces them rather than assembling a full in-memory tensor first.

## Data path

```mermaid
flowchart LR
    S["NVMe: aligned packed records"] -->|"preadv workers"| R["Reusable read slots"]
    R -->|"validate CRC + select"| P["Pinned coalescing buffer"]
    P -->|"cudaMemcpyAsync"| G["Packed KV in HBM"]
    G --> K["Fused INT2/INT4 decode kernel"]
    Q["Query heads"] --> K
    K --> M["Online-softmax chunk merge"]
    M --> O["Attention output"]
```

Reads for later blocks are issued while earlier reads are consumed. On CUDA, packed chunks are
transferred on one stream and processed on another. Each chunk returns its normalized attention
output and log-sum-exp; those statistics allow exact softmax merging across chunks without retaining
the attention matrix.

## Supported fast-path layout

| Property | Current contract |
|---|---|
| Decode query length | 1 |
| KV bits | 2 or 4 |
| Scale type | FP16 |
| Quantization | Symmetric, per token/head/group |
| Head dimension | Up to 256, divisible by group size |
| Attention | MHA, MQA, or GQA |
| Record alignment | 4096 bytes |
| CUDA query dtype | FP16, BF16, or FP32 |

## Performance interpretation

The CPU fallback verifies semantics; it is not a performance implementation. GPU measurements should
separate these quantities:

1. storage read bandwidth and queue depth;
2. pinned staging and H2D bandwidth;
3. fused-kernel time;
4. end-to-end time, including selection and synchronization;
5. quantization-induced task-quality change.

The included `effective_read_gib_s` metric is end-to-end payload throughput, not the SSD's raw media
bandwidth. See [docs/benchmarking.md](docs/benchmarking.md) for the benchmark protocol.

## Deliberate limitations

- The CUDA kernel favors clarity and auditability over architecture-specific tensor-core tuning.
- The included selectors do not implement semantic top-k retrieval. The selector interface is the
  intended integration point for block centroids, retrieval heads, or an external scheduler.
- The default backend uses buffered `preadv`; direct I/O and GPUDirect Storage are future backends.
- Quantization quality must be evaluated on the target model and task. INT2 is not automatically safe.
- Windows can run the CPU tests, but the production CUDA/SSD target is Linux.

These constraints are explicit so benchmark results cannot silently compare the reference fallback
against the fused path. `require_cuda_kernel=True` and `--require-cuda` fail closed when the extension
is absent.

## License

Apache-2.0.

See [CHANGELOG.md](CHANGELOG.md) for release history and
[CONTRIBUTING.md](CONTRIBUTING.md) for validation requirements.
