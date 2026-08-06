from __future__ import annotations

from memory_system.backends.memory import InMemoryBackend
from memory_system.core import _find_similar_pairs
from memory_system.events import MemoryEvent, MemoryTier


def make_long_term_event(content):
    return MemoryEvent(content=content, tier=MemoryTier.LONG_TERM)


def test_finds_near_identical_pair_above_threshold():
    backend = InMemoryBackend()
    a = make_long_term_event("User is severely allergic to peanuts.")
    b = make_long_term_event("User is severely allergic to peanuts and tree nuts.")
    backend.add(a)
    backend.add(b)

    pairs = _find_similar_pairs(backend, MemoryTier.LONG_TERM, threshold=0.3)

    pair_ids = {(p[0].id, p[1].id) for p in pairs} | {(p[1].id, p[0].id) for p in pairs}
    assert (a.id, b.id) in pair_ids


def test_excludes_unrelated_pair_below_threshold():
    backend = InMemoryBackend()
    a = make_long_term_event("User is severely allergic to peanuts.")
    b = make_long_term_event("The weather forecast for tomorrow is sunny.")
    backend.add(a)
    backend.add(b)

    pairs = _find_similar_pairs(backend, MemoryTier.LONG_TERM, threshold=0.5)

    assert pairs == []


def test_never_matches_event_with_itself():
    backend = InMemoryBackend()
    a = make_long_term_event("User is severely allergic to peanuts.")
    backend.add(a)

    # threshold=0.0 would trivially accept a self-match if the exclusion
    # didn't exist -- a single event querying its own content scores 1.0
    # against itself (verified empirically: InMemoryBackend.query(a.content)
    # on a single-event corpus returns a itself at score=1.0).
    pairs = _find_similar_pairs(backend, MemoryTier.LONG_TERM, threshold=0.0)

    assert pairs == []


def test_deduplicates_symmetric_pair_reporting():
    backend = InMemoryBackend()
    a = make_long_term_event("User is severely allergic to peanuts.")
    b = make_long_term_event("User is severely allergic to peanuts and tree nuts.")
    backend.add(a)
    backend.add(b)

    pairs = _find_similar_pairs(backend, MemoryTier.LONG_TERM, threshold=0.3)

    # (a, b) and (b, a) are the same pair -- must be reported once, not twice
    assert len(pairs) == 1
