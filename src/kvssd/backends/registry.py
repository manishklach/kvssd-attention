from __future__ import annotations

from collections.abc import Callable

import torch

from .base import AttentionBackend, BackendAvailability
from .cuda import CudaExtensionBackend
from .reference import ReferenceAttentionBackend
from .triton import TritonAttentionBackend


BackendFactory = Callable[[], AttentionBackend]


_BUILTINS: dict[str, BackendFactory] = {
    "cuda": CudaExtensionBackend,
    "reference": ReferenceAttentionBackend,
    "triton": TritonAttentionBackend,
}


def register_attention_backend(name: str, factory: BackendFactory) -> None:
    normalized = name.strip().lower()
    if not normalized or normalized == "auto":
        raise ValueError("backend name must be non-empty and cannot be 'auto'")
    _BUILTINS[normalized] = factory


def available_attention_backends(device: str | torch.device) -> list[BackendAvailability]:
    target = torch.device(device)
    return [factory().probe(target) for factory in _BUILTINS.values()]


def resolve_attention_backend(
    requested: str | AttentionBackend | None,
    device: str | torch.device,
) -> AttentionBackend:
    target = torch.device(device)
    if requested is not None and not isinstance(requested, str):
        availability = requested.probe(target)
        if not availability.available:
            raise RuntimeError(
                f"attention backend '{availability.name}' unavailable: {availability.reason}"
            )
        return requested

    name = (requested or "auto").strip().lower()
    if name == "auto":
        order = ("cuda", "triton", "reference") if target.type == "cuda" else ("reference",)
        for candidate in order:
            factory = _BUILTINS.get(candidate)
            if factory is None:
                continue
            backend = factory()
            if backend.probe(target).available:
                return backend
        raise RuntimeError(f"no attention backend is available for {target}")

    try:
        backend = _BUILTINS[name]()
    except KeyError as exc:
        choices = ", ".join(["auto", *_BUILTINS])
        raise ValueError(f"unknown attention backend '{name}'; choose one of: {choices}") from exc
    availability = backend.probe(target)
    if not availability.available:
        raise RuntimeError(f"attention backend '{name}' unavailable: {availability.reason}")
    return backend
