from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch


@dataclass(frozen=True)
class BackendAvailability:
    name: str
    available: bool
    accelerated: bool
    direct: bool
    reason: str
    device_type: str


@dataclass(frozen=True)
class PackedBatch:
    packed_k: torch.Tensor
    packed_v: torch.Tensor
    scales_k: torch.Tensor
    scales_v: torch.Tensor
    valid_tokens: torch.Tensor
    bits: int
    group_size: int

    def validate(self, query: torch.Tensor) -> None:
        if query.ndim != 2:
            raise ValueError("query must be [query_heads, head_dim]")
        if self.packed_k.ndim != 4 or self.packed_k.shape != self.packed_v.shape:
            raise ValueError("packed KV must match [blocks,tokens,kv_heads,packed_dim]")
        if self.scales_k.ndim != 4 or self.scales_k.shape != self.scales_v.shape:
            raise ValueError("KV scales must match [blocks,tokens,kv_heads,groups]")
        if self.valid_tokens.shape != (self.packed_k.shape[0],):
            raise ValueError("valid_tokens must contain one count per block")
        if self.bits not in (2, 4):
            raise ValueError("bits must be 2 or 4")
        head_dim = query.shape[-1]
        if head_dim % self.group_size:
            raise ValueError("group_size must divide head_dim")
        if self.packed_k.shape[-1] != head_dim * self.bits // 8:
            raise ValueError("packed dimension does not match query head dimension")
        if query.shape[0] % self.packed_k.shape[2]:
            raise ValueError("query heads must be divisible by KV heads")
        tensors = (
            self.packed_k,
            self.packed_v,
            self.scales_k,
            self.scales_v,
            self.valid_tokens,
        )
        if any(tensor.device != query.device for tensor in tensors):
            raise ValueError("query and packed batch must share a device")


@runtime_checkable
class AttentionBackend(Protocol):
    name: str

    def probe(self, device: torch.device) -> BackendAvailability: ...

    def run(
        self,
        query: torch.Tensor,
        batch: PackedBatch,
        sm_scale: float,
    ) -> tuple[torch.Tensor, torch.Tensor]: ...
