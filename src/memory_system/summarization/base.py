"""Abstract boundary for turning a group of related MemoryEvents into
one dense summary string. Mirrors EntityExtractor's shape
(backends/graph.py) -- the interface just needs to accept events and
return something usable, the hard problem (a good summarization
prompt/model) is decided separately by whichever concrete
implementation you pick.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..events import MemoryEvent


class MemorySummarizer(ABC):
    @abstractmethod
    def summarize(self, events: list[MemoryEvent]) -> str:
        ...
