# Contributing

Run `ruff check src tests` and `pytest` before submitting changes. CUDA changes should additionally be
built with `KVSSD_BUILD_CUDA=1` and compared against the dequantized reference for both 2-bit and 4-bit
caches, multiple valid-token tails, MHA, MQA, and GQA.

Format changes require a version bump, backward-compatible reader tests, and documentation of every
new metadata field. Performance claims should include hardware, software versions, workload shapes,
warm/cold cache state, and raw benchmark output.

