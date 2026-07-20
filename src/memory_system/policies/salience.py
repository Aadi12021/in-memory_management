"""Assigns an importance/surprise score to incoming events.

This is the natural extension point for PERCEPT-1-style surprise scoring:
a user could plug in a scorer backed by an embedding-distance novelty
signal, an LLM judgment call, or a simple heuristic like the one below.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Optional

from ..events import MemoryEvent


class SalienceScorer(ABC):
    @abstractmethod
    def score(self, event: MemoryEvent, context: Optional[Any] = None) -> float:
        """Returns a salience score, conventionally in [0, 1]."""
        ...


class ConstantSalience(SalienceScorer):
    """Baseline: every event gets the same score. Useful default so the
    system works out of the box before a user wires in something smarter.
    """

    def __init__(self, value: float = 0.5):
        self.value = value

    def score(self, event: MemoryEvent, context: Optional[Any] = None) -> float:
        return self.value


class LengthHeuristicSalience(SalienceScorer):
    """Toy heuristic: longer content is treated as more salient, capped
    at 1.0. Mostly here as a template for writing a real scorer, not
    meant to be taken seriously as a production heuristic.
    """

    def __init__(self, saturation_length: int = 500):
        self.saturation_length = saturation_length

    def score(self, event: MemoryEvent, context: Optional[Any] = None) -> float:
        length = len(str(event.content))
        return min(length / self.saturation_length, 1.0)
