from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


class BlockSelector(Protocol):
    def select(self, available: list[int]) -> list[int]: ...


@dataclass(frozen=True)
class AllBlocks:
    def select(self, available: list[int]) -> list[int]:
        return list(available)


@dataclass(frozen=True)
class RecentSinkBlocks:
    """Keep initial sink blocks and the most recent blocks, without duplicates."""

    sink: int = 1
    recent: int = 8

    def select(self, available: list[int]) -> list[int]:
        if self.sink < 0 or self.recent < 0:
            raise ValueError("sink and recent counts must be non-negative")
        chosen = available[: self.sink] + (available[-self.recent :] if self.recent else [])
        return list(dict.fromkeys(chosen))

