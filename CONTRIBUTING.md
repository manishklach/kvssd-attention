# Contributing

Run `ruff check src tests` and `pytest` before submitting changes. CUDA changes should additionally be
built with `KVSSD_BUILD_CUDA=1` and compared against the dequantized reference for both 2-bit and 4-bit
caches, multiple valid-token tails, MHA, MQA, and GQA.

Format changes require a version bump, backward-compatible reader tests, and documentation of every
new metadata field. Performance claims should include hardware, software versions, workload shapes,
warm/cold cache state, and raw benchmark output.

Triton changes must pass `TRITON_INTERPRET=1 pytest -m triton_interpreter` and the appropriate
opt-in CUDA or ROCm job. GDS changes must keep capability probes hardware-free, preserve explicit
fail-closed behavior, and pass the `nvidia-gds` hardware job before release. vLLM changes must pass the
tagged source-contract check and the fake store/evict/reload lifecycle test.
