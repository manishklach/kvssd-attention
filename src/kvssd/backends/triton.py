from __future__ import annotations

import importlib.util
import os

import torch

from .base import BackendAvailability, PackedBatch

try:
    import triton
    import triton.language as tl
except ImportError:
    triton = None
    tl = None


if triton is not None:

    @triton.jit
    def _fused_decode_kernel(
        query_ptr,
        packed_k_ptr,
        packed_v_ptr,
        scales_k_ptr,
        scales_v_ptr,
        valid_ptr,
        output_ptr,
        lse_ptr,
        sm_scale,
        BITS: tl.constexpr,
        GROUP_SIZE: tl.constexpr,
        NUM_BLOCKS: tl.constexpr,
        BLOCK_TOKENS: tl.constexpr,
        QUERY_HEADS: tl.constexpr,
        KV_HEADS: tl.constexpr,
        HEAD_DIM: tl.constexpr,
        PACKED_DIM: tl.constexpr,
        NUM_GROUPS: tl.constexpr,
        BLOCK_D: tl.constexpr,
    ):
        query_head = tl.program_id(0)
        offsets_d = tl.arange(0, BLOCK_D)
        dim_mask = offsets_d < HEAD_DIM
        query = tl.load(query_ptr + query_head * HEAD_DIM + offsets_d, mask=dim_mask, other=0.0)
        query = query.to(tl.float32)
        kv_head = query_head // (QUERY_HEADS // KV_HEADS)
        per_byte: tl.constexpr = 8 // BITS
        code_mask: tl.constexpr = (1 << BITS) - 1
        midpoint: tl.constexpr = 1 << (BITS - 1)

        running_max = -float("inf")
        running_sum = 0.0
        accumulator = tl.zeros((BLOCK_D,), dtype=tl.float32)

        for block in range(NUM_BLOCKS):
            valid = tl.load(valid_ptr + block)
            for token in range(BLOCK_TOKENS):
                active = token < valid
                packed_base = ((block * BLOCK_TOKENS + token) * KV_HEADS + kv_head) * PACKED_DIM
                byte_offsets = offsets_d // per_byte
                shifts = (offsets_d % per_byte) * BITS
                key_bytes = tl.load(
                    packed_k_ptr + packed_base + byte_offsets,
                    mask=dim_mask,
                    other=0,
                )
                key_codes = (key_bytes >> shifts) & code_mask
                scale_base = ((block * BLOCK_TOKENS + token) * KV_HEADS + kv_head) * NUM_GROUPS
                key_scales = tl.load(
                    scales_k_ptr + scale_base + offsets_d // GROUP_SIZE,
                    mask=dim_mask,
                    other=0.0,
                )
                key = (key_codes.to(tl.float32) - midpoint) * key_scales
                score = tl.sum(query * key, axis=0) * sm_scale
                score = tl.where(active, score, -float("inf"))

                next_max = tl.maximum(running_max, score)
                old_weight = tl.exp(running_max - next_max)
                new_weight = tl.exp(score - next_max)
                running_sum = running_sum * old_weight + new_weight
                running_max = next_max

                value_bytes = tl.load(
                    packed_v_ptr + packed_base + byte_offsets,
                    mask=dim_mask,
                    other=0,
                )
                value_codes = (value_bytes >> shifts) & code_mask
                value_scales = tl.load(
                    scales_v_ptr + scale_base + offsets_d // GROUP_SIZE,
                    mask=dim_mask,
                    other=0.0,
                )
                value = (value_codes.to(tl.float32) - midpoint) * value_scales
                accumulator = accumulator * old_weight + value * new_weight

        output = accumulator / running_sum
        tl.store(
            output_ptr + query_head * HEAD_DIM + offsets_d,
            output,
            mask=dim_mask,
        )
        tl.store(lse_ptr + query_head, running_max + tl.log(running_sum))

else:
    _fused_decode_kernel = None


class TritonAttentionBackend:
    name = "triton"

    def probe(self, device: torch.device) -> BackendAvailability:
        interpreter = os.getenv("TRITON_INTERPRET") == "1"
        if importlib.util.find_spec("triton") is None or _fused_decode_kernel is None:
            return BackendAvailability(
                self.name,
                False,
                True,
                False,
                "Triton is not installed; install kvssd-attention[triton] on Linux",
                device.type,
            )
        if interpreter:
            return BackendAvailability(
                self.name,
                True,
                False,
                False,
                "Triton interpreter mode; functional validation only",
                device.type,
            )
        if device.type != "cuda" or not torch.cuda.is_available():
            return BackendAvailability(
                self.name,
                False,
                True,
                False,
                "Triton GPU backend requires CUDA or ROCm PyTorch on Linux",
                device.type,
            )
        if torch.version.hip is None:
            major, minor = torch.cuda.get_device_capability(device)
            if (major, minor) < (8, 0):
                return BackendAvailability(
                    self.name,
                    False,
                    True,
                    False,
                    f"NVIDIA compute capability {major}.{minor} is below Triton's 8.0 target",
                    device.type,
                )
            platform = f"NVIDIA compute capability {major}.{minor}"
        else:
            platform = f"AMD ROCm {torch.version.hip}"
        return BackendAvailability(
            self.name, True, True, False, f"Triton backend on {platform}", device.type
        )

    def run(
        self, query: torch.Tensor, batch: PackedBatch, sm_scale: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        availability = self.probe(query.device)
        if not availability.available:
            raise RuntimeError(f"Triton backend unavailable: {availability.reason}")
        batch.validate(query)
        if query.shape[-1] > 256:
            raise ValueError("Triton backend supports head_dim <= 256")
        if any(valid <= 0 for valid in batch.valid_tokens.tolist()):
            raise ValueError("every staged block must contain at least one valid token")
        block_d = triton.next_power_of_2(query.shape[-1])
        output = torch.empty(
            (query.shape[0], query.shape[1]), device=query.device, dtype=torch.float32
        )
        lse = torch.empty((query.shape[0],), device=query.device, dtype=torch.float32)
        assert _fused_decode_kernel is not None
        _fused_decode_kernel[(query.shape[0],)](
            query.contiguous(),
            batch.packed_k.contiguous(),
            batch.packed_v.contiguous(),
            batch.scales_k.contiguous(),
            batch.scales_v.contiguous(),
            batch.valid_tokens.contiguous(),
            output,
            lse,
            sm_scale,
            BITS=batch.bits,
            GROUP_SIZE=batch.group_size,
            NUM_BLOCKS=batch.packed_k.shape[0],
            BLOCK_TOKENS=batch.packed_k.shape[1],
            QUERY_HEADS=query.shape[0],
            KV_HEADS=batch.packed_k.shape[2],
            HEAD_DIM=query.shape[1],
            PACKED_DIM=batch.packed_k.shape[3],
            NUM_GROUPS=batch.scales_k.shape[3],
            BLOCK_D=block_d,
            num_warps=4,
        )
        return output, lse
