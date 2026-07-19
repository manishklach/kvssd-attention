from __future__ import annotations

import torch

from ..quant import unpack_codes
from .base import BackendAvailability, PackedBatch


class ReferenceAttentionBackend:
    name = "reference"

    def probe(self, device: torch.device) -> BackendAvailability:
        return BackendAvailability(
            self.name,
            True,
            False,
            False,
            "portable PyTorch dequantize-then-attend reference",
            device.type,
        )

    def run(
        self, query: torch.Tensor, batch: PackedBatch, sm_scale: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch.validate(query)
        head_dim = query.shape[-1]
        midpoint = 1 << (batch.bits - 1)

        def dequantize(packed: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
            codes = unpack_codes(packed, batch.bits, head_dim).float()
            expanded_scales = scales.float().repeat_interleave(batch.group_size, dim=-1)
            return (codes - midpoint) * expanded_scales

        keys = dequantize(batch.packed_k, batch.scales_k)
        values = dequantize(batch.packed_v, batch.scales_v)
        key_chunks = []
        value_chunks = []
        for index, valid in enumerate(batch.valid_tokens.tolist()):
            key_chunks.append(keys[index, :valid])
            value_chunks.append(values[index, :valid])
        keys_flat = torch.cat(key_chunks, dim=0)
        values_flat = torch.cat(value_chunks, dim=0)
        q_heads = query.shape[0]
        kv_heads = keys_flat.shape[1]
        mapping = torch.arange(q_heads, device=query.device) // (q_heads // kv_heads)
        scores = torch.einsum("hd,thd->ht", query.float(), keys_flat[:, mapping]) * sm_scale
        lse = torch.logsumexp(scores, dim=-1)
        weights = torch.softmax(scores, dim=-1)
        output = torch.einsum("ht,thd->hd", weights, values_flat[:, mapping])
        return output, lse
