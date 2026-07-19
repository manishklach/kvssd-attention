from __future__ import annotations

import os
import threading
from pathlib import Path

import torch

from ..format import BlockEntry
from .base import StorageAvailability


class BufferedStorageBackend:
    """Portable positioned reads into CPU tensors, using preadv where available."""

    name = "buffered"

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._fd: int | None = None
        self._fd_lock = threading.Lock()
        self._fallback_lock = threading.Lock()

    def probe(self) -> StorageAvailability:
        if not self.path.is_file():
            return StorageAvailability(
                self.name, False, False, True, f"data file does not exist: {self.path}"
            )
        return StorageAvailability(
            self.name,
            True,
            False,
            True,
            "portable buffered positioned reads",
        )

    def _file_descriptor(self) -> int:
        with self._fd_lock:
            if self._fd is None:
                self._fd = os.open(self.path, os.O_RDONLY | getattr(os, "O_BINARY", 0))
            return self._fd

    def allocate_buffer(
        self,
        size: int,
        *,
        pinned: bool = False,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        del device
        return torch.empty(size, dtype=torch.uint8, pin_memory=pinned and torch.cuda.is_available())

    def read_into(self, entry: BlockEntry, destination: torch.Tensor) -> None:
        if destination.device.type != "cpu":
            raise ValueError("buffered backend requires a CPU destination tensor")
        if destination.dtype != torch.uint8 or destination.numel() < entry.length:
            raise ValueError("destination must be a sufficiently large uint8 tensor")
        target = memoryview(destination[: entry.length].numpy())
        fd = self._file_descriptor()
        if hasattr(os, "preadv"):
            total = os.preadv(fd, [target], entry.offset)
        else:
            with self._fallback_lock:
                os.lseek(fd, entry.offset, os.SEEK_SET)
                total = 0
                while total < entry.length:
                    chunk = os.read(fd, entry.length - total)
                    if not chunk:
                        break
                    target[total : total + len(chunk)] = chunk
                    total += len(chunk)
        if total != entry.length:
            raise EOFError(
                f"short read for layer={entry.layer}, block={entry.block}: "
                f"expected {entry.length}, got {total}"
            )

    def close(self) -> None:
        with self._fd_lock:
            if self._fd is not None:
                os.close(self._fd)
                self._fd = None
