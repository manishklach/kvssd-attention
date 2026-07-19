# Benchmarking protocol

Run on a Linux host with the target NVMe topology and CUDA GPU. Record the GPU, driver, CUDA, PyTorch,
filesystem, mount options, SSD model, RAID layout, CPU NUMA node, and PCIe topology.

Before measuring:

```bash
KVSSD_BUILD_CUDA=1 pip install -e . --no-build-isolation
pytest
kvssd create-demo /mnt/nvme/kvssd-demo --tokens 131072 --bits 4
kvssd benchmark --tokens 131072 --bits 4 --warmup 5 --iterations 50 --require-cuda
```

For a backend/bit-width matrix, write one self-describing JSON object per line:

```bash
python benchmarks/run_matrix.py \
  --output results/$(hostname)-$(date +%F).jsonl \
  --storage auto gds direct buffered \
  --attention triton cuda \
  --bits 2 4
```

Explicit unavailable combinations fail instead of being relabeled as fallback results. Each result
records Python, platform, PyTorch, Triton, CUDA/ROCm, GPU name, tensor layout, and structured backend
capability reports alongside timing.

For media measurements, control page-cache state explicitly and report whether the run is cold,
warm, or direct-I/O. Do not use cache-dropping commands on a shared machine. Use `iostat -x 1`, Nsight
Systems, and PyTorch profiler traces to answer three separate questions:

- Are reads large enough and deep enough to saturate the SSD?
- Does H2D transfer overlap the preceding fused kernel?
- Is fused attention limited by packed-KV HBM bandwidth or arithmetic?

Compare at least FP16, INT8/FP8 where available, INT4, and INT2. Report model-task quality alongside
throughput; a faster configuration with unacceptable retrieval or generation degradation is not a win.

The manually dispatched hardware workflow is a qualification gate, not a performance benchmark. Use
the matrix runner on the actual NVMe mount after qualification, retain the raw JSONL, and state whether
GDS dynamic routing or the kernel driver handled the I/O. Never infer media bandwidth from a run whose
selected storage backend is `buffered`.
