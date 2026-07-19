"""Storage backend contracts and built-in implementations."""

from .base import StorageAvailability, StorageBackend
from .registry import available_storage_backends, resolve_storage_backend

__all__ = [
    "StorageAvailability",
    "StorageBackend",
    "available_storage_backends",
    "resolve_storage_backend",
]
