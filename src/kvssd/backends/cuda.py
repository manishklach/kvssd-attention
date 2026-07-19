from __future__ import annotations

import importlib

import torch

from .base import BackendAvailability, PackedBatch


def _extension():
    try:
        return importlib.import_module("kvssd._C")
    except ImportError:
        return None


class CudaExtensionBackend:
    name = "cuda"

    def probe(self, device: torch.device) -> BackendAvailability:
        if device.type != "cuda":
            reason = f"CUDA extension requires a CUDA device, got {device.type}"
            return BackendAvailability(self.name, False, True, False, reason, device.type)
        if _extension() is None:
            reason = "extension not installed; rebuild with KVSSD_BUILD_CUDA=1"
            return BackendAvailability(self.name, False, True, False, reason, device.type)
        return BackendAvailability(
            self.name, True, True, False, "CUDA extension is available", device.type
        )

    def run(
        self, query: torch.Tensor, batch: PackedBatch, sm_scale: float
    ) -> tuple[torch.Tensor, torch.Tensor]:
        batch.validate(query)
        extension = _extension()
        if extension is None:
            raise RuntimeError("CUDA extension unavailable; rebuild with KVSSD_BUILD_CUDA=1")
        return extension.fused_decode(
            query,
            batch.packed_k,
            batch.packed_v,
            batch.scales_k,
            batch.scales_v,
            batch.valid_tokens,
            batch.bits,
            batch.group_size,
            sm_scale,
        )
