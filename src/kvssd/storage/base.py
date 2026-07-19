from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import torch

from ..format import BlockEntry


@dataclass(frozen=True)
class StorageAvailability:
    name: str
    available: bool
    direct: bool
    asynchronous: bool
    reason: str


@runtime_checkable
class StorageBackend(Protocol):
    name: str

    def probe(self) -> StorageAvailability: ...

    def allocate_buffer(
        self,
        size: int,
        *,
        pinned: bool = False,
        device: torch.device | None = None,
    ) -> torch.Tensor: ...

    def read_into(self, entry: BlockEntry, destination: torch.Tensor) -> None: ...

    def close(self) -> None: ...
