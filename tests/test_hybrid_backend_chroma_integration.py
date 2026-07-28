"""Integration tests for HybridBackend against a real ChromaBackend
(semantic side) and InMemoryBackend (lexical side). Requires the
`chroma` extra:
    pip install -e ".[chroma]"
Skipped automatically if chromadb isn't installed.

Each test uses its own randomly-named collection, same reasoning as
test_chroma_backend.py: chromadb.Client()'s non-persistent client
shares its underlying store across ChromaBackend instances built with
the same collection_name in one process.
"""

from __future__ import annotations

import uuid

import pytest

pytest.importorskip("chromadb")

from memory_system.backends.chroma import ChromaBackend
from memory_system.backends.hybrid import HybridBackend
from memory_system.backends.memory import InMemoryBackend
from memory_system.events import MemoryEvent


def make_hybrid_backend():
    lexical = InMemoryBackend()
    semantic = ChromaBackend(collection_name=f"test_hybrid_{uuid.uuid4().hex}")
    return HybridBackend(lexical_backend=lexical, semantic_backend=semantic)


def test_add_mirrors_to_both_real_backends():
    hybrid = make_hybrid_backend()
    event = MemoryEvent(content="a fact for both backends")

    hybrid.add(event)

    assert hybrid.lexical_backend.get_all()[0].id == event.id
    assert hybrid.semantic_backend.get_all()[0].id == event.id


def test_query_ranks_more_relevant_document_higher():
    hybrid = make_hybrid_backend()
    allergic = MemoryEvent(content="The user is severely allergic to peanuts and tree nuts.")
    hiking = MemoryEvent(content="The user enjoys hiking on weekends.")
    weather = MemoryEvent(content="The weather forecast for tomorrow is sunny.")
    for event in (allergic, hiking, weather):
        hybrid.add(event)

    results = hybrid.query("peanuts nuts allergy", top_k=3)

    assert results[0].event.id == allergic.id


def test_query_finds_semantically_related_content_with_no_lexical_overlap():
    hybrid = make_hybrid_backend()
    hybrid.add(MemoryEvent(content="User is allergic to peanuts."))
    hybrid.add(MemoryEvent(content="The weather forecast for tomorrow is sunny."))

    results = hybrid.query("food the user cannot safely eat", top_k=2)

    assert len(results) >= 1
    assert any("peanut" in str(r.event.content).lower() for r in results)
