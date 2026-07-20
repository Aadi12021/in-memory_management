"""Core data types shared across the package."""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional


class MemoryTier(Enum):
    SENSORY = "sensory"
    WORKING = "working"
    LONG_TERM = "long_term"


@dataclass
class MemoryEvent:
    """A single unit of memory moving through the system."""

    content: Any
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    tier: MemoryTier = MemoryTier.SENSORY
    salience: float = 0.0
    metadata: dict = field(default_factory=dict)
    last_reinforced: Optional[datetime] = None

    def reinforce(self, when: Optional[datetime] = None) -> None:
        """Mark this event as accessed/reinforced, resetting decay clock."""
        self.last_reinforced = when or datetime.now(timezone.utc)


@dataclass
class RetrievalResult:
    event: MemoryEvent
    score: float
    tier: MemoryTier
