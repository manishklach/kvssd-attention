# Design notes

## Record format

`blocks.kvssd` is an append-only sequence of fixed-size records. Each record represents one layer and
one token block. The header contains the layout, valid-token count, payload length, and CRC32. The
payload order is:

1. key FP16 scales;
2. value FP16 scales;
3. packed key codes;
4. packed value codes;
5. zero padding to the next 4096-byte boundary.

The final partial block is zero-padded; `valid_tokens` prevents padded positions from entering the
softmax. Fixed records make offsets deterministic and turn selected adjacent blocks into predictable
large reads.

## Quantization

For each group, KVSSD stores a scale `s = max(abs(x)) / (2^(bits-1)-1)` and an unsigned code centered
at `2^(bits-1)`. The kernel reconstructs a value as `(code - midpoint) * s`. Codes are packed low bits
first within each byte. There are no zero-point loads on the hot path.

This layout is intentionally kernel-friendly. More accurate key-per-channel or outlier-preserving
schemes can be added as new format versions, but must include matching fused-kernel implementations.

## Streaming softmax

Every staged chunk computes an output `O_i` and log-normalizer `L_i`. Two chunks are merged using:

```text
L = logaddexp(L_a, L_b)
O = O_a * exp(L_a - L) + O_b * exp(L_b - L)
```

This is algebraically equivalent to attention over the concatenated tokens. It allows bounded HBM
staging and compute/I/O overlap without approximating softmax across chunks.

## Safety invariants

- A manifest layout mismatch is rejected before tensor interpretation.
- CRC failures identify the layer and block.
- Read slots are detached before reuse, preventing in-flight consumers from observing overwritten
  storage.
- Partial-block padding is masked by `valid_tokens` in both reference and CUDA paths.
- The production option fails closed if the fused extension was not loaded.

