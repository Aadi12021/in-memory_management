"""Policies deciding how a memory's retrievability fades over time."""

from __future__ import annotations

import math
from abc import ABC, abstractmethod
from datetime import datetime, timezone

from ..events import MemoryEvent


class DecayPolicy(ABC):
    @abstractmethod
    def current_strength(self, event: MemoryEvent, now: datetime | None = None) -> float:
        """Returns a value in [0, 1] representing how strong/retrievable
        this memory currently is. 1.0 = fully fresh, 0.0 = fully forgotten.
        """
        ...

    def should_forget(self, event: MemoryEvent, now: datetime | None = None, floor: float = 0.05) -> bool:
        return self.current_strength(event, now) <= floor


class ForgettingCurveDecay(DecayPolicy):
    """Classic Ebbinghaus-style exponential decay: strength halves every
    `half_life_days`. Long-term memories decay slower than working memories
    by applying a tier multiplier -- this is the one real behavioral
    difference between tiers in the decay model.
    """

    def __init__(self, half_life_days: float = 14.0, long_term_multiplier: float = 4.0):
        self.half_life_days = half_life_days
        self.long_term_multiplier = long_term_multiplier

    def current_strength(self, event: MemoryEvent, now: datetime | None = None) -> float:
        now = now or datetime.now(timezone.utc)
        anchor = event.last_reinforced or event.timestamp

        elapsed_days = (now - anchor).total_seconds() / 86400
        if elapsed_days <= 0:
            return 1.0

        half_life = self.half_life_days
        if event.tier.value == "long_term":
            half_life *= self.long_term_multiplier

        decay_constant = math.log(2) / half_life
        return math.exp(-decay_constant * elapsed_days)


class NoDecay(DecayPolicy):
    """Baseline: memories never fade. Useful for testing or for backends
    where you want an external process to manage eviction instead.
    """

    def current_strength(self, event: MemoryEvent, now: datetime | None = None) -> float:
        return 1.0
