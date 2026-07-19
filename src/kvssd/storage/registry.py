from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from .base import StorageAvailability, StorageBackend
from .buffered import BufferedStorageBackend
from .direct import DirectIOStorageBackend
from .gds import GDSStorageBackend


StorageFactory = Callable[[Path], StorageBackend]


_BUILTINS: dict[str, StorageFactory] = {
    "gds": GDSStorageBackend,
    "direct": DirectIOStorageBackend,
    "buffered": BufferedStorageBackend,
}


def register_storage_backend(name: str, factory: StorageFactory) -> None:
    normalized = name.strip().lower()
    if not normalized or normalized == "auto":
        raise ValueError("backend name must be non-empty and cannot be 'auto'")
    _BUILTINS[normalized] = factory


def available_storage_backends(path: str | Path) -> list[StorageAvailability]:
    target = Path(path)
    return [factory(target).probe() for factory in _BUILTINS.values()]


def resolve_storage_backend(
    requested: str | StorageBackend | None,
    path: str | Path,
) -> StorageBackend:
    target = Path(path)
    if requested is not None and not isinstance(requested, str):
        availability = requested.probe()
        if not availability.available:
            raise RuntimeError(
                f"storage backend '{availability.name}' unavailable: {availability.reason}"
            )
        return requested

    name = (requested or "auto").strip().lower()
    if name == "auto":
        for candidate in ("gds", "direct", "buffered"):
            factory = _BUILTINS.get(candidate)
            if factory is None:
                continue
            backend = factory(target)
            if backend.probe().available:
                return backend
        raise RuntimeError(f"no storage backend is available for {target}")

    try:
        backend = _BUILTINS[name](target)
    except KeyError as exc:
        choices = ", ".join(["auto", *_BUILTINS])
        raise ValueError(f"unknown storage backend '{name}'; choose one of: {choices}") from exc
    availability = backend.probe()
    if not availability.available:
        raise RuntimeError(f"storage backend '{name}' unavailable: {availability.reason}")
    return backend
