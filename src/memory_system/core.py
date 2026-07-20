"""The public-facing entry point: TieredMemory."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from .backends.base import MemoryBackend
from .events import MemoryEvent, MemoryTier, RetrievalResult
from .policies.consolidation import ConsolidationPolicy
from .policies.decay import DecayPolicy
from .policies.salience import SalienceScorer


class TieredMemory:
    """Composes a backend with consolidation/decay/salience policies.

    Typical usage:
        memory = TieredMemory(
            backend=InMemoryBackend(),
            consolidation_policy=SurpriseBasedConsolidation(threshold=0.7),
            decay_policy=ForgettingCurveDecay(half_life_days=14),
        )
        memory.store("User is allergic to peanuts.")
        memory.consolidate()
        results = memory.retrieve("dietary restrictions")
    """

    def __init__(
        self,
        backend: MemoryBackend,
        consolidation_policy: ConsolidationPolicy,
        decay_policy: DecayPolicy,
        salience_scorer: Optional[SalienceScorer] = None,
        forget_floor: float = 0.05,
    ):
        self.backend = backend
        self.consolidation_policy = consolidation_policy
        self.decay_policy = decay_policy
        self.salience_scorer = salience_scorer
        self.forget_floor = forget_floor

    def store(self, content: Any, metadata: Optional[dict] = None) -> MemoryEvent:
        """Ingest new content at the working tier."""
        event = MemoryEvent(
            content=content,
            tier=MemoryTier.WORKING,
            metadata=metadata or {},
        )
        if self.salience_scorer is not None:
            event.salience = self.salience_scorer.score(event)
        self.backend.add(event)
        return event

    def consolidate(self) -> int:
        """Promote eligible working-tier memories to long-term. Returns
        the number of events promoted.
        """
        promoted = 0
        for event in self.backend.get_all(tier=MemoryTier.WORKING):
            if self.consolidation_policy.should_consolidate(event):
                self.backend.update_tier(event.id, MemoryTier.LONG_TERM)
                promoted += 1
        return promoted

    def decay(self, now: Optional[datetime] = None) -> int:
        """Run a decay pass, removing events whose strength has fallen
        below the forget floor. Returns the number of events forgotten.
        """
        now = now or datetime.now(timezone.utc)
        forgotten = 0
        for event in self.backend.get_all():
            if self.decay_policy.should_forget(event, now, floor=self.forget_floor):
                self.backend.remove(event.id)
                forgotten += 1
        return forgotten

    def retrieve(
        self, query: str, top_k: int = 5, tier: Optional[MemoryTier] = None
    ) -> list[RetrievalResult]:
        """Query memory, weighting relevance by current decay strength."""
        raw_results = self.backend.query(query, top_k=top_k * 2, tier=tier)

        weighted: list[RetrievalResult] = []
        for result in raw_results:
            strength = self.decay_policy.current_strength(result.event)
            weighted.append(
                RetrievalResult(
                    event=result.event,
                    score=result.score * strength,
                    tier=result.tier,
                )
            )
            result.event.reinforce()  # accessing a memory reinforces it

        weighted.sort(key=lambda r: r.score, reverse=True)
        return weighted[:top_k]
