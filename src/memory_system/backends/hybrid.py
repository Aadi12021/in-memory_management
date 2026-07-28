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

from ..events import RetrievalResult


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
