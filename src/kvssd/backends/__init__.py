"""Attention backend contracts and built-in implementations."""

from .base import AttentionBackend, BackendAvailability, PackedBatch
from .registry import available_attention_backends, resolve_attention_backend

__all__ = [
    "AttentionBackend",
    "BackendAvailability",
    "PackedBatch",
    "available_attention_backends",
    "resolve_attention_backend",
]
