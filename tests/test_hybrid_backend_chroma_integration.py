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
    legume_fact = MemoryEvent(content="Severe reaction to legumes in childhood.")
    sales_report = MemoryEvent(content="The quarterly sales report is due Friday.")
    hybrid.add(legume_fact)
    hybrid.add(sales_report)
    hybrid.add(MemoryEvent(content="Traffic on the highway was heavy this morning."))
    hybrid.add(MemoryEvent(content="A new coffee shop opened downtown."))

    # Rank the full corpus (untruncated) so the assertion lands on the real,
    # well-separated relevance margin between the semantically related fact
    # and an unrelated one (roughly 0.42 vs. 0.34 in practice), rather than on
    # whichever distractor happens to land right at a top-k truncation
    # boundary -- the three distractors cluster close enough there that a
    # membership-in-top-2 assertion would be one embedding-model update away
    # from flipping for reasons unrelated to HybridBackend's own correctness.
    all_results = hybrid.query("dietary restrictions from an allergy", top_k=4)
    result_ids = [r.event.id for r in all_results]

    assert result_ids.index(legume_fact.id) < result_ids.index(sales_report.id)

    # Truncation is a separate concern from ranking: 4 documents, top_k=2 -> exactly 2.
    truncated_results = hybrid.query("dietary restrictions from an allergy", top_k=2)
    assert len(truncated_results) == 2
