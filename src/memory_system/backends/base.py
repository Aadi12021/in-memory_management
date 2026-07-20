"""Storage backend abstraction."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from ..events import MemoryEvent, MemoryTier, RetrievalResult


class MemoryBackend(ABC):
    """Anything that can store and query MemoryEvents implements this."""

    @abstractmethod
    def add(self, event: MemoryEvent) -> None:
        ...

    @abstractmethod
    def get_all(self, tier: Optional[MemoryTier] = None) -> list[MemoryEvent]:
        ...

    @abstractmethod
    def query(
        self, query: str, top_k: int = 5, tier: Optional[MemoryTier] = None
    ) -> list[RetrievalResult]:
        ...

    @abstractmethod
    def update_tier(self, event_id: str, new_tier: MemoryTier) -> None:
        ...

    @abstractmethod
    def remove(self, event_id: str) -> None:
        ...
