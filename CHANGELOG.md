# Changelog

All notable project changes are documented here. The project follows Semantic Versioning.

## [0.1.1] - 2026-07-19

### Fixed

- Declared NumPy as a runtime dependency. The SSD codec uses NumPy-backed tensor buffer views for
  serialization and direct reads; clean Linux installations previously failed with
  `RuntimeError: Numpy is not available`.
- Limited push-triggered CI runs to `main`, avoiding redundant workflow executions for release tags.

### Validation

- 12 portable tests pass with the corrected dependency set.
- Ruff and wheel packaging pass.

## [0.1.0] - 2026-07-19

### Added

- Symmetric, groupwise INT2 and INT4 KV-cache quantization with packed unsigned codes and FP16 scales.
- Versioned, append-only, 4 KiB-aligned SSD record format with manifests and per-record CRC32.
- Portable asynchronous block loader using bounded worker and staging queues.
- Explicit block-selection interface with full-cache and sink-plus-recent policies.
- Pinned host coalescing and distinct CUDA transfer and compute streams.
- Chunked fused decode attention with exact log-sum-exp merging across independently staged chunks.
- CUDA kernel supporting FP16, BF16, and FP32 queries; MHA, MQA, and GQA; head dimensions up to 256.
- CPU reference implementation and fail-closed enforcement for deployments requiring the CUDA kernel.
- CLI commands for synthetic store creation, inspection, and end-to-end benchmarking.
- Correctness, corruption, padding, buffer-ownership, descriptor-lifecycle, GQA, and softmax-merge tests.
- Linux CI, Python packaging, source-distribution manifests, design notes, and benchmark guidance.

### Validation

- 12 portable tests pass.
- Ruff static checks pass.
- CPU end-to-end CLI execution passes on Windows.
- Python wheel and source distribution build successfully.
- The CUDA runtime path remains to be validated on physical NVIDIA hardware.

[0.1.1]: https://github.com/manishklach/kvssd-attention/releases/tag/v0.1.1
[0.1.0]: https://github.com/manishklach/kvssd-attention/releases/tag/v0.1.0
