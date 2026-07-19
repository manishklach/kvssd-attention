from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import torch

from ..format import ALIGNMENT, BlockEntry
from .base import StorageAvailability


class GDSStorageBackend:
    """Optional cuFile backend that reads aligned records directly into CUDA memory."""

    name = "gds"

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self._handle = None

    def probe(self) -> StorageAvailability:
        if sys.platform != "linux":
            return StorageAvailability(self.name, False, True, True, "cuFile requires Linux")
        if not self.path.is_file():
            return StorageAvailability(
                self.name, False, True, True, f"data file does not exist: {self.path}"
            )
        if not torch.cuda.is_available() or torch.version.hip is not None:
            return StorageAvailability(
                self.name, False, True, True, "cuFile requires an NVIDIA CUDA device"
            )
        if importlib.util.find_spec("kvssd._gds") is None:
            return StorageAvailability(
                self.name,
                False,
                True,
                True,
                "kvssd._gds is not built; install with KVSSD_BUILD_GDS=1",
            )
        try:
            self._ensure_handle()
        except Exception as exc:
            return StorageAvailability(self.name, False, True, True, f"cuFile probe failed: {exc}")
        return StorageAvailability(
            self.name, True, True, True, "cuFile direct reads into aligned CUDA buffers"
        )

    def _ensure_handle(self):
        if self._handle is None:
            from kvssd import _gds

            self._handle = _gds.GDSFile(str(self.path))
        return self._handle

    def allocate_buffer(
        self,
        size: int,
        *,
        pinned: bool = False,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        del pinned
        owner = torch.empty(size + ALIGNMENT - 1, dtype=torch.uint8, device=device or "cuda")
        shift = (-owner.data_ptr()) % ALIGNMENT
        return owner[shift : shift + size]

    def read_into(self, entry: BlockEntry, destination: torch.Tensor) -> None:
        if destination.device.type != "cuda" or destination.dtype != torch.uint8:
            raise ValueError("GDS backend requires a CUDA uint8 destination")
        if not destination.is_contiguous() or destination.numel() < entry.length:
            raise ValueError("GDS destination must be contiguous and large enough")
        if entry.offset % ALIGNMENT or entry.length % ALIGNMENT:
            raise ValueError(
                f"unaligned GDS read for layer={entry.layer}, block={entry.block}: "
                f"offset={entry.offset}, length={entry.length}, alignment={ALIGNMENT}"
            )
        if destination.data_ptr() % ALIGNMENT:
            raise ValueError(
                f"GDS destination address must be {ALIGNMENT}-byte aligned; "
                f"got address=0x{destination.data_ptr():x}"
            )
        try:
            total = self._ensure_handle().read(destination, entry.offset, entry.length)
        except Exception as exc:
            raise OSError(
                f"cuFile read failed for layer={entry.layer}, block={entry.block}, "
                f"offset={entry.offset}, bytes={entry.length}: {exc}"
            ) from exc
        if total != entry.length:
            raise EOFError(
                f"short cuFile read for layer={entry.layer}, block={entry.block}: "
                f"offset={entry.offset}, expected={entry.length}, got={total}"
            )

    def close(self) -> None:
        self._handle = None
