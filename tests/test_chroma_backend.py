"""Integration tests for ChromaBackend against a real (in-memory,
non-persistent) chromadb instance. Requires the `chroma` extra:
    pip install -e ".[chroma]"
Skipped automatically if chromadb isn't installed.

Each test uses its own randomly-named collection: chromadb.Client()
shares its underlying in-memory system across ChromaBackend instances
constructed with the same collection_name in the same process, so
reusing a fixed name would leak documents between tests.
"""

import uuid

import pytest

pytest.importorskip("chromadb")

from memory_system.backends.chroma import ChromaBackend
from memory_system.events import MemoryEvent, MemoryTier


def make_backend():
    return ChromaBackend(collection_name=f"test_{uuid.uuid4().hex}")


def test_add_stores_event_retrievable_via_get_all():
    backend = make_backend()
    event = MemoryEvent(content="The user is allergic to peanuts.")
    backend.add(event)

    all_events = backend.get_all()

    assert len(all_events) == 1
    assert all_events[0].id == event.id
    assert str(all_events[0].content) == "The user is allergic to peanuts."


def test_query_ranks_more_relevant_document_higher():
    backend = make_backend()
    allergic = MemoryEvent(content="The user is severely allergic to peanuts and tree nuts.")
    hiking = MemoryEvent(content="The user enjoys hiking on weekends.")
    weather = MemoryEvent(content="The weather forecast for tomorrow is sunny.")
    for event in (allergic, hiking, weather):
        backend.add(event)

    results = backend.query("peanuts nuts allergy", top_k=3)

    assert len(results) == 3
    assert results[0].event.id == allergic.id
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_query_respects_top_k():
    backend = make_backend()
    for i in range(5):
        backend.add(MemoryEvent(content=f"document number {i} about cats"))

    results = backend.query("cats", top_k=2)

    assert len(results) == 2


def test_query_filters_by_tier():
    backend = make_backend()
    working = MemoryEvent(content="alpha document about cats", tier=MemoryTier.WORKING)
    long_term = MemoryEvent(content="beta document about cats", tier=MemoryTier.LONG_TERM)
    backend.add(working)
    backend.add(long_term)

    results = backend.query("cats", top_k=5, tier=MemoryTier.LONG_TERM)

    assert len(results) == 1
    assert results[0].event.id == long_term.id


def test_update_tier_changes_tier_and_is_queryable_by_new_tier():
    backend = make_backend()
    event = MemoryEvent(content="a document about dogs", tier=MemoryTier.WORKING)
    backend.add(event)

    backend.update_tier(event.id, MemoryTier.LONG_TERM)

    assert backend.get_all()[0].tier == MemoryTier.LONG_TERM
    assert backend.query("dogs", top_k=5, tier=MemoryTier.WORKING) == []
    long_term_results = backend.query("dogs", top_k=5, tier=MemoryTier.LONG_TERM)
    assert len(long_term_results) == 1
    assert long_term_results[0].event.id == event.id


def test_remove_deletes_event_from_get_all_and_query():
    backend = make_backend()
    keep = MemoryEvent(content="a document about birds")
    drop = MemoryEvent(content="a document about fish")
    backend.add(keep)
    backend.add(drop)

    backend.remove(drop.id)

    assert {e.id for e in backend.get_all()} == {keep.id}
    result_ids = {r.event.id for r in backend.query("birds fish", top_k=5)}
    assert result_ids == {keep.id}
