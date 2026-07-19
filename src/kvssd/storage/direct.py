from __future__ import annotations

import os
import sys
import threading
from pathlib import Path

import torch

from ..format import ALIGNMENT, BlockEntry
from .base import StorageAvailability


def _aligned_uint8(size: int, *, pinned: bool = False) -> torch.Tensor:
    owner = torch.empty(
        size + ALIGNMENT - 1,
        dtype=torch.uint8,
        pin_memory=pinned and torch.cuda.is_available(),
    )
    shift = (-owner.data_ptr()) % ALIGNMENT
    return owner[shift : shift + size]


class DirectIOStorageBackend:
    """Linux O_DIRECT reads into explicitly aligned host buffers."""

    name = "direct"

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._fd: int | None = None
        self._lock = threading.Lock()

    def probe(self) -> StorageAvailability:
        if sys.platform != "linux" or not hasattr(os, "O_DIRECT"):
            return StorageAvailability(self.name, False, True, True, "O_DIRECT requires Linux")
        if not self.path.is_file():
            return StorageAvailability(
                self.name, False, True, True, f"data file does not exist: {self.path}"
            )
        try:
            fd = os.open(self.path, os.O_RDONLY | os.O_DIRECT)
            os.close(fd)
        except OSError as exc:
            return StorageAvailability(self.name, False, True, True, f"O_DIRECT open failed: {exc}")
        return StorageAvailability(self.name, True, True, True, "Linux O_DIRECT positioned reads")

    def _file_descriptor(self) -> int:
        with self._lock:
            if self._fd is None:
                self._fd = os.open(self.path, os.O_RDONLY | os.O_DIRECT)
            return self._fd

    def allocate_buffer(
        self,
        size: int,
        *,
        pinned: bool = False,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        del device
        return _aligned_uint8(size, pinned=pinned)

    def read_into(self, entry: BlockEntry, destination: torch.Tensor) -> None:
        if destination.device.type != "cpu" or destination.dtype != torch.uint8:
            raise ValueError("direct backend requires a CPU uint8 destination")
        if destination.numel() < entry.length:
            raise ValueError("destination is smaller than the requested record")
        if entry.offset % ALIGNMENT or entry.length % ALIGNMENT:
            raise ValueError(
                f"unaligned direct read for layer={entry.layer}, block={entry.block}: "
                f"offset={entry.offset}, length={entry.length}, alignment={ALIGNMENT}"
            )
        if destination.data_ptr() % ALIGNMENT:
            raise ValueError(
                f"direct destination address must be {ALIGNMENT}-byte aligned; "
                f"got address=0x{destination.data_ptr():x}"
            )
        total = os.preadv(
            self._file_descriptor(),
            [memoryview(destination[: entry.length].numpy())],
            entry.offset,
        )
        if total != entry.length:
            raise EOFError(
                f"short direct read for layer={entry.layer}, block={entry.block}: "
                f"offset={entry.offset}, expected={entry.length}, got={total}"
            )

    def close(self) -> None:
        with self._lock:
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None
