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

For media measurements, control page-cache state explicitly and report whether the run is cold,
warm, or direct-I/O. Do not use cache-dropping commands on a shared machine. Use `iostat -x 1`, Nsight
Systems, and PyTorch profiler traces to answer three separate questions:

- Are reads large enough and deep enough to saturate the SSD?
- Does H2D transfer overlap the preceding fused kernel?
- Is fused attention limited by packed-KV HBM bandwidth or arithmetic?

Compare at least FP16, INT8/FP8 where available, INT4, and INT2. Report model-task quality alongside
throughput; a faster configuration with unacceptable retrieval or generation degradation is not a win.

