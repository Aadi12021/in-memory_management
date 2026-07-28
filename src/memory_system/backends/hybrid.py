"""Combines a lexical MemoryBackend (e.g. InMemoryBackend) and a
semantic MemoryBackend (e.g. ChromaBackend) into one ranked result via
Reciprocal Rank Fusion (RRF). See
docs/superpowers/specs/2026-07-28-hybrid-retrieval-design.md for the
full design rationale -- in short, the two backends' raw scores aren't
on comparable scales (TF-IDF cosine similarity vs. embedding-distance-
derived similarity), so RRF combines by rank position within each
backend's own ordering instead of by raw score magnitude.
"""

from __future__ import annotations

from typing import Optional

from ..events import MemoryEvent, MemoryTier, RetrievalResult
from .base import MemoryBackend


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievalResult]],
    k: int = 60,
) -> list[RetrievalResult]:
    """Fuses multiple ranked RetrievalResult lists into one list, summing
    1 / (k + rank) for each event across every list it appears in (rank
    is 1-indexed within that list). Events are matched across lists by
    event.id. An event absent from a list simply contributes no term for
    that list -- no special-casing needed for empty or partial lists.
    k=60 is the standard RRF damping constant (also the default used by
    Elasticsearch/Weaviate/Azure AI Search hybrid search).

    Returns events sorted descending by summed score. The returned
    RetrievalResult reuses the event/tier from wherever that event.id
    was first encountered across the input lists.
    """
    scores: dict[str, float] = {}
    first_seen: dict[str, RetrievalResult] = {}

    for result_list in result_lists:
        for rank, result in enumerate(result_list, start=1):
            event_id = result.event.id
            scores[event_id] = scores.get(event_id, 0.0) + 1.0 / (k + rank)
            if event_id not in first_seen:
                first_seen[event_id] = result

    fused = [
        RetrievalResult(
            event=first_seen[event_id].event,
            score=score,
            tier=first_seen[event_id].tier,
        )
        for event_id, score in scores.items()
    ]
    fused.sort(key=lambda r: r.score, reverse=True)
    return fused


class HybridBackendSyncError(Exception):
    """Raised when a HybridBackend mutating call (add/update_tier/
    remove) succeeds on one internal backend but fails on the other,
    leaving them out of sync. The successful side already has state
    the other doesn't -- check `succeeded`/`failed` before deciding
    how to recover.
    """

    def __init__(self, method: str, event_id: str, succeeded: str, failed: str, original: Exception):
        self.method = method
        self.event_id = event_id
        self.succeeded = succeeded
        self.failed = failed
        self.original = original
        super().__init__(
            f"HybridBackend.{method}(event_id={event_id!r}) succeeded on "
            f"{succeeded}_backend but failed on {failed}_backend: {original!r}. "
            "The two backends are now out of sync."
        )


class HybridBackend(MemoryBackend):
    """Combines a lexical backend (e.g. InMemoryBackend) and a semantic
    backend (e.g. ChromaBackend) via Reciprocal Rank Fusion. No new
    optional extra required -- whatever dependency semantic_backend
    needs (e.g. chromadb) is its own concern, not HybridBackend's.

    Mirroring invariant: add/update_tier/remove write to lexical_backend
    first, then semantic_backend. If lexical_backend's write fails,
    nothing has diverged (semantic_backend was never touched) and the
    original exception propagates as-is. If lexical_backend succeeds
    but semantic_backend's write fails, the two backends are now out of
    sync -- raises HybridBackendSyncError wrapping the original
    exception rather than silently leaving them divergent, since RRF's
    correctness depends on matching event.id across both stores.
    """

    def __init__(
        self,
        lexical_backend: MemoryBackend,
        semantic_backend: MemoryBackend,
        rrf_k: int = 60,
        fetch_k_multiplier: int = 5,
    ):
        self.lexical_backend = lexical_backend
        self.semantic_backend = semantic_backend
        self.rrf_k = rrf_k
        self.fetch_k_multiplier = fetch_k_multiplier

    def add(self, event: MemoryEvent) -> None:
        self.lexical_backend.add(event)
        try:
            self.semantic_backend.add(event)
        except Exception as exc:
            raise HybridBackendSyncError("add", event.id, "lexical", "semantic", exc) from exc

    def get_all(self, tier: Optional[MemoryTier] = None) -> list[MemoryEvent]:
        """Delegates to lexical_backend only. Assumes the mirroring
        invariant held (see add/update_tier/remove) -- if it didn't,
        this won't reflect semantic_backend's state.
        """
        return self.lexical_backend.get_all(tier=tier)

    def query(
        self, query: str, top_k: int = 5, tier: Optional[MemoryTier] = None
    ) -> list[RetrievalResult]:
        fetch_k = top_k * self.fetch_k_multiplier
        lexical_results = self.lexical_backend.query(query, top_k=fetch_k, tier=tier)
        semantic_results = self.semantic_backend.query(query, top_k=fetch_k, tier=tier)
        fused = reciprocal_rank_fusion([lexical_results, semantic_results], k=self.rrf_k)
        return fused[:top_k]

    def update_tier(self, event_id: str, new_tier: MemoryTier) -> None:
        self.lexical_backend.update_tier(event_id, new_tier)
        try:
            self.semantic_backend.update_tier(event_id, new_tier)
        except Exception as exc:
            raise HybridBackendSyncError("update_tier", event_id, "lexical", "semantic", exc) from exc

    def remove(self, event_id: str) -> None:
        self.lexical_backend.remove(event_id)
        try:
            self.semantic_backend.remove(event_id)
        except Exception as exc:
            raise HybridBackendSyncError("remove", event_id, "lexical", "semantic", exc) from exc
