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


@dataclass
class ConsolidationReport:
    """Result of an offline-consolidation pass (deduplicate/compress/
    strengthen_connections/offline_consolidate). Replaces a bare int
    count -- these methods do several different kinds of things, so a
    count alone would hide what actually happened.

    The third element of each `merged` tuple and the second element of
    each `compressed` tuple is `None` under dry_run=True: a dry run
    never calls add(), so there is no real MemoryEvent.id to report.
    """

    merged: list[tuple[str, str, Optional[str]]] = field(default_factory=list)
    compressed: list[tuple[list[str], Optional[str]]] = field(default_factory=list)
    strengthened: list[tuple[str, str]] = field(default_factory=list)
