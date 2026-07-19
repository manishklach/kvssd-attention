# vLLM 0.25 proof-of-concept integration

KVSSD integrates at vLLM's public V1 offloading boundary. `KVSSDOffloadingSpec` subclasses
`TieringOffloadingSpec`, retaining vLLM's pinned CPU primary tier and registering a `kvssd` secondary
tier. Completed prefix blocks are quantized asynchronously to INT2 or INT4 records and promoted back
through the primary tier on a hit.

This proves persistent packed-prefix lifecycle integration. It does **not** inject KVSSD's fused
decode kernel into PagedAttention: vLLM 0.25 explicitly requires secondary tiers to stage through its
CPU primary tier. The standalone `KVSSDPipeline` is where GDS can currently feed fused attention
directly.

## Install and serve

Install a supported vLLM release and KVSSD in the same Linux environment:

```bash
pip install -e '.[vllm]'
```

Example configuration:

```bash
vllm serve meta-llama/Llama-3.1-8B-Instruct \
  --enable-prefix-caching \
  --kv-transfer-config '{
    "kv_connector": "OffloadingConnector",
    "kv_role": "kv_both",
    "kv_connector_extra_config": {
      "spec_name": "KVSSDOffloadingSpec",
      "spec_module_path": "kvssd.integrations.vllm",
      "cpu_bytes_to_use": 4294967296,
      "block_size": 16,
      "secondary_tiers": [{
        "type": "kvssd",
        "root_dir": "/mnt/nvme/vllm-kvssd",
        "bits": 4,
        "group_size": 32,
        "kvssd_dtype": "float16",
        "n_io_threads": 16
      }]
    }
  }'
```

`kvssd_dtype` must match the byte representation in vLLM's CPU primary cache (`float16`,
`bfloat16`, or `float32`). The adapter validates bit width, group alignment, block byte size, and the
presence of a `kvssd` tier. A configuration fingerprint includes model, dtype, quantization, block
bytes, and tensor/pipeline parallel sizes, preventing incompatible runs from sharing records.

## Record identity and lifecycle

Each vLLM `OffloadKey` contains a content hash plus a four-byte cache-group index. KVSSD maps it to:

```text
<root>/<model>_<config-digest>/<hash[0:3]>/<hash[3:5]>_g<group>/<hash>.kvssd
```

Writes use a temporary file, `fsync`, and atomic replace. Reads verify the header and CRC before
dequantizing into the primary slot. Lookups return `RETRY` while a matching transfer is in flight;
job completion is reported through vLLM's normal scheduler polling API.

Because quantization is lossy, validate model/task quality—especially at INT2—before production use.

## Validation and prefix-reuse benchmark

Portable tests cover codec corruption, hash/group mapping, fake-runtime store/evict/reload, and the
streaming benchmark client. CI loads the actual base, secondary-tier, factory, and tiering-spec
classes from the exact `v0.25.0` source tag, instantiates KVSSD against those abstract classes, and
executes the lifecycle.

The physical NVIDIA workflow runs the serving qualification automatically. It launches vLLM 0.25,
sends a cold and warm target prompt plus distinct prompts that exercise CPU-primary eviction pressure,
waits for stable secondary-tier `.kvssd` records, restarts the server, and sends the target again.
The gate requires both `prompt_tokens_details.cached_tokens > 0` and an increase in vLLM's
`external_prefix_cache_hits` counter after restart. This distinguishes persistent secondary-tier reuse
from an in-process local prefix hit. The benchmark explicitly sets vLLM's KV-load policy to `fail`, so
a corrupt or failed reload cannot silently recompute and pass. Raw JSON, Prometheus snapshots, and
both server logs are retained.
vLLM is installed in a dedicated virtual environment so its pinned PyTorch/Triton dependencies cannot
replace the environment used to build and qualify the CUDA and cuFile extensions.

Run the same qualification manually with a public model ID or local model path:

```bash
python benchmarks/vllm_prefix_reuse.py \
  --model facebook/opt-125m \
  --root /mnt/nvme/vllm-kvssd-qualification \
  --output results/vllm-prefix-reuse.json \
  --artifact-dir results/vllm-logs \
  --cpu-bytes 67108864
```

Use a new empty root for every qualification. For a comparative performance study, run the same
prompt sequence against vLLM's built-in `fs` tier with identical CPU bytes and thread counts; the
release gate proves lifecycle correctness and persistent reuse, not a universal speedup.

Do not describe the result as direct GDS unless the standalone GDS/fused pipeline—not this staged
vLLM adapter—was actually measured.
