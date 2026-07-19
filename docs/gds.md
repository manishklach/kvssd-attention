# GPUDirect Storage backend

The optional `gds` backend issues synchronous `cuFileRead` calls from the bounded asynchronous loader
workers into 4 KiB-aligned CUDA tensors. Multiple selected blocks can therefore be in flight while a
previous packed chunk is consumed by fused attention. The call is synchronous per worker; concurrency
comes from the configured I/O depth, not from an undocumented CUDA-stream assumption.

## Requirements

- Linux and an NVIDIA CUDA GPU;
- CUDA/GPUDirect Storage with `cufile.h` and `libcufile` installed;
- the `nvidia-fs` driver or a supported GDS dynamic-routing configuration;
- a filesystem/file descriptor accepted by `cuFileHandleRegister`;
- 4 KiB-aligned KVSSD record offsets, lengths, and destination addresses.

Build against the CUDA-enabled PyTorch environment that will run KVSSD:

```bash
KVSSD_BUILD_GDS=1 pip install -e . --no-build-isolation
```

`KVSSD_BUILD_CUDA=1` and `KVSSD_BUILD_GDS=1` may be set together. They create independent
`kvssd._C` and `kvssd._gds` extensions, so the Triton attention path does not require the hand-written
CUDA extension.

## Selection and fallback

`storage_backend="auto"` probes in this order:

```text
gds → direct → buffered
```

The selected backend appears in `kvssd inspect` and benchmark JSON. Explicit `gds` selection never
silently falls back: missing Linux/CUDA support, an omitted extension, a failed cuFile driver/file
registration, bad alignment, a short read, or a cuFile error raises with record offset and block
identity. Use `auto` when fallback is desired.

## Validation

Portable CI tests probe capability and alignment without NVIDIA hardware. The manually dispatched
`hardware-validation` workflow requires a self-hosted runner labeled `nvidia-gds`; it builds both
extensions, reads a complete record directly into CUDA memory, copies it back only for the test, and
verifies the record CRC. A queued or absent hardware job is not evidence of GDS support.

For host diagnostics, first run NVIDIA's `gdscheck -p`, then:

```bash
kvssd inspect /mnt/gds/cache --storage-backend gds
KVSSD_RUN_GDS_TESTS=1 pytest -m gds_hardware -q
```

The current record reader trusts manifest layout on the hot path. CRC verification is performed by
portable reads and by the hardware qualification test; adding a GPU CRC kernel is future work.
