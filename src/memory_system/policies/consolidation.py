"""Policies deciding whether a memory graduates from working -> long-term."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..events import MemoryEvent


class ConsolidationPolicy(ABC):
    @abstractmethod
    def should_consolidate(self, event: MemoryEvent) -> bool:
        ...


class SurpriseBasedConsolidation(ConsolidationPolicy):
    """Consolidates events whose salience score crosses a threshold.

    Salience is expected to be set upstream (e.g. by a SalienceScorer, or
    by an external signal like a surprise/novelty score from a perception
    pipeline). This policy just applies the threshold rule.
    """

    def __init__(self, threshold: float = 0.7):
        self.threshold = threshold

    def should_consolidate(self, event: MemoryEvent) -> bool:
        return event.salience >= self.threshold


class AlwaysConsolidate(ConsolidationPolicy):
    """Naive baseline: everything gets promoted. Useful for testing or
    for users who don't want salience-gating at all.
    """

    def should_consolidate(self, event: MemoryEvent) -> bool:
        return True


class RepetitionBasedConsolidation(ConsolidationPolicy):
    """Consolidates events that have been reinforced (retrieved/accessed)
    at least `min_reinforcements` times -- mirrors spaced-repetition intuition.
    """

    def __init__(self, min_reinforcements: int = 2):
        self.min_reinforcements = min_reinforcements
        self._counts: dict[str, int] = {}

    def note_access(self, event: MemoryEvent) -> None:
        self._counts[event.id] = self._counts.get(event.id, 0) + 1

    def should_consolidate(self, event: MemoryEvent) -> bool:
        return self._counts.get(event.id, 0) >= self.min_reinforcements
