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

Portable tests cover codec corruption, hash/group mapping, and fake-runtime store/evict/reload. CI
checks the integration contract against the exact `v0.25.0` vLLM source tag. A real serving benchmark
still requires a supported GPU host:

1. start the server with a clean KVSSD root;
2. send the same long prompt twice with deterministic decoding;
3. record first-token latency, cache-hit metrics, record bytes, and task output for each request;
4. restart the server and repeat to demonstrate persistent reuse;
5. compare against vLLM's built-in `fs` secondary tier using identical CPU bytes and thread counts.

Do not describe the result as direct GDS unless the standalone GDS/fused pipeline—not this staged
vLLM adapter—was actually measured.
