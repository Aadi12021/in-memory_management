"""Tests for reciprocal_rank_fusion() and HybridBackend."""

from __future__ import annotations

import pytest
from unittest.mock import patch

from memory_system.backends.base import MemoryBackend
from memory_system.backends.hybrid import (
    HybridBackend,
    HybridBackendSyncError,
    reciprocal_rank_fusion,
)
from memory_system.backends.memory import InMemoryBackend
from memory_system.events import MemoryEvent, MemoryTier, RetrievalResult


def make_result(content, score, event_id=None, tier=MemoryTier.WORKING):
    event = MemoryEvent(content=content, tier=tier)
    if event_id is not None:
        event.id = event_id
    return RetrievalResult(event=event, score=score, tier=tier)


def test_rrf_empty_lists_returns_empty():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_single_list_uses_1_over_k_plus_rank():
    a = make_result("alpha", score=0.9, event_id="a")
    b = make_result("beta", score=0.5, event_id="b")

    fused = reciprocal_rank_fusion([[a, b]], k=60)

    assert [r.event.id for r in fused] == ["a", "b"]
    assert fused[0].score == 1.0 / 61
    assert fused[1].score == 1.0 / 62


def test_rrf_custom_k_changes_damping():
    a = make_result("alpha", score=0.9, event_id="a")

    fused = reciprocal_rank_fusion([[a]], k=1)

    assert fused[0].score == 1.0 / 2


def test_rrf_document_found_by_both_lists_scores_higher_than_by_one():
    shared = make_result("shared doc", score=0.9, event_id="shared")
    only_in_first = make_result("first-only doc", score=0.9, event_id="first_only")

    fused = reciprocal_rank_fusion([[shared, only_in_first], [shared]])
    fused_by_id = {r.event.id: r.score for r in fused}

    assert fused_by_id["shared"] > fused_by_id["first_only"]


def test_rrf_document_found_by_only_one_list_still_appears():
    lexical_only = make_result("lexical match", score=0.8, event_id="lexical_only")
    semantic_only = make_result("semantic match", score=0.6, event_id="semantic_only")

    fused = reciprocal_rank_fusion([[lexical_only], [semantic_only]])

    assert {r.event.id for r in fused} == {"lexical_only", "semantic_only"}


def test_rrf_sorts_descending_by_summed_score_across_lists():
    a = make_result("alpha", score=0.9, event_id="a")
    b = make_result("beta", score=0.5, event_id="b")
    list_a = [a, b]  # a rank1, b rank2
    list_b = [b]     # b rank1

    fused = reciprocal_rank_fusion([list_a, list_b])

    # b: 1/61 + 1/62 (two contributions) vs a: 1/61 (one contribution) -> b wins
    # even though a ranked #1 in list_a and b only ranked #2 there.
    assert [r.event.id for r in fused] == ["b", "a"]


def test_rrf_preserves_event_and_tier_from_source_result():
    result = make_result("some content", score=0.9, event_id="a", tier=MemoryTier.LONG_TERM)

    fused = reciprocal_rank_fusion([[result]])

    assert fused[0].event is result.event
    assert fused[0].tier == MemoryTier.LONG_TERM


class RaisingBackend(MemoryBackend):
    """Test double: a MemoryBackend whose configured methods raise
    instead of executing, used to force HybridBackend's mirroring
    failure path deterministically.
    """

    def __init__(self, fail_on=frozenset()):
        self.fail_on = fail_on
        self.events: dict[str, MemoryEvent] = {}

    def add(self, event):
        if "add" in self.fail_on:
            raise RuntimeError("simulated add failure")
        self.events[event.id] = event

    def get_all(self, tier=None):
        events = list(self.events.values())
        if tier is not None:
            events = [e for e in events if e.tier == tier]
        return events

    def query(self, query, top_k=5, tier=None):
        return []

    def update_tier(self, event_id, new_tier):
        if "update_tier" in self.fail_on:
            raise RuntimeError("simulated update_tier failure")
        if event_id in self.events:
            self.events[event_id].tier = new_tier

    def remove(self, event_id):
        if "remove" in self.fail_on:
            raise RuntimeError("simulated remove failure")
        self.events.pop(event_id, None)


def test_add_mirrors_event_to_both_backends():
    lexical = InMemoryBackend()
    semantic = InMemoryBackend()
    hybrid = HybridBackend(lexical_backend=lexical, semantic_backend=semantic)
    event = MemoryEvent(content="a shared fact")

    hybrid.add(event)

    assert lexical.get_all()[0].id == event.id
    assert semantic.get_all()[0].id == event.id


def test_add_raises_sync_error_when_semantic_backend_fails():
    lexical = InMemoryBackend()
    semantic = RaisingBackend(fail_on={"add"})
    hybrid = HybridBackend(lexical_backend=lexical, semantic_backend=semantic)
    event = MemoryEvent(content="a fact")

    with pytest.raises(HybridBackendSyncError) as exc_info:
        hybrid.add(event)

    exc = exc_info.value
    assert exc.method == "add"
    assert exc.event_id == event.id
    assert exc.succeeded == "lexical"
    assert exc.failed == "semantic"
    assert isinstance(exc.__cause__, RuntimeError)
    # lexical already has it -- that IS the divergence being reported, not rolled back
    assert lexical.get_all()[0].id == event.id


def test_add_propagates_original_exception_when_lexical_backend_fails():
    lexical = RaisingBackend(fail_on={"add"})
    semantic = InMemoryBackend()
    hybrid = HybridBackend(lexical_backend=lexical, semantic_backend=semantic)
    event = MemoryEvent(content="a fact")

    with pytest.raises(RuntimeError) as exc_info:
        hybrid.add(event)

    assert not isinstance(exc_info.value, HybridBackendSyncError)
    assert semantic.get_all() == []  # never touched -- nothing diverged


def test_update_tier_mirrors_to_both_backends():
    lexical = InMemoryBackend()
    semantic = InMemoryBackend()
    hybrid = HybridBackend(lexical_backend=lexical, semantic_backend=semantic)
    event = MemoryEvent(content="a fact", tier=MemoryTier.WORKING)
    hybrid.add(event)

    # semantic and lexical share the same MemoryEvent instance (HybridBackend.add
    # hands one object to both), so InMemoryBackend's in-place mutation would make
    # asserting on semantic.get_all()[0].tier pass even if semantic_backend.update_tier
    # were never called. Spy on the call itself instead -- this is what actually
    # matters for a real backend like ChromaBackend, where update_tier() has an
    # external side effect (syncing Chroma's own metadata) beyond the shared
    # Python attribute.
    with patch.object(semantic, "update_tier", wraps=semantic.update_tier) as semantic_spy:
        hybrid.update_tier(event.id, MemoryTier.LONG_TERM)

    assert lexical.get_all()[0].tier == MemoryTier.LONG_TERM
    semantic_spy.assert_called_once_with(event.id, MemoryTier.LONG_TERM)


def test_update_tier_raises_sync_error_when_semantic_backend_fails():
    lexical = InMemoryBackend()
    semantic = RaisingBackend(fail_on={"update_tier"})
    hybrid = HybridBackend(lexical_backend=lexical, semantic_backend=semantic)
    event = MemoryEvent(content="a fact")
    hybrid.add(event)

    with pytest.raises(HybridBackendSyncError) as exc_info:
        hybrid.update_tier(event.id, MemoryTier.LONG_TERM)

    assert exc_info.value.method == "update_tier"
    assert exc_info.value.succeeded == "lexical"
    assert exc_info.value.failed == "semantic"


def test_remove_mirrors_to_both_backends():
    lexical = InMemoryBackend()
    semantic = InMemoryBackend()
    hybrid = HybridBackend(lexical_backend=lexical, semantic_backend=semantic)
    event = MemoryEvent(content="a fact")
    hybrid.add(event)

    hybrid.remove(event.id)

    assert lexical.get_all() == []
    assert semantic.get_all() == []


def test_remove_raises_sync_error_when_semantic_backend_fails():
    lexical = InMemoryBackend()
    semantic = RaisingBackend(fail_on={"remove"})
    hybrid = HybridBackend(lexical_backend=lexical, semantic_backend=semantic)
    event = MemoryEvent(content="a fact")
    hybrid.add(event)

    with pytest.raises(HybridBackendSyncError) as exc_info:
        hybrid.remove(event.id)

    assert exc_info.value.method == "remove"
    assert exc_info.value.succeeded == "lexical"
    assert exc_info.value.failed == "semantic"


def test_get_all_delegates_to_lexical_backend_only():
    lexical = InMemoryBackend()
    semantic = InMemoryBackend()
    hybrid = HybridBackend(lexical_backend=lexical, semantic_backend=semantic)
    lexical_only_event = MemoryEvent(content="only in lexical")
    lexical.add(lexical_only_event)  # bypasses HybridBackend.add on purpose

    result = hybrid.get_all()

    assert [e.id for e in result] == [lexical_only_event.id]


def test_get_all_filters_by_tier():
    lexical = InMemoryBackend()
    semantic = InMemoryBackend()
    hybrid = HybridBackend(lexical_backend=lexical, semantic_backend=semantic)
    working = MemoryEvent(content="working fact", tier=MemoryTier.WORKING)
    long_term = MemoryEvent(content="long term fact", tier=MemoryTier.LONG_TERM)
    hybrid.add(working)
    hybrid.add(long_term)

    result = hybrid.get_all(tier=MemoryTier.LONG_TERM)

    assert [e.id for e in result] == [long_term.id]


def test_query_fuses_results_from_two_real_backends():
    lexical = InMemoryBackend()
    semantic = InMemoryBackend()
    hybrid = HybridBackend(lexical_backend=lexical, semantic_backend=semantic)
    allergic = MemoryEvent(content="The user is severely allergic to peanuts and tree nuts.")
    hiking = MemoryEvent(content="The user enjoys hiking on weekends with peanut butter snacks.")
    for event in (allergic, hiking):
        hybrid.add(event)

    results = hybrid.query("peanuts nuts allergy", top_k=2)

    assert len(results) == 2
    assert results[0].event.id == allergic.id
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


class StubBackend(MemoryBackend):
    """Test double: a MemoryBackend whose query() always returns a
    fixed, pre-set list of results regardless of the query string --
    used to simulate "the semantic backend found something the
    lexical backend didn't" deterministically, without needing a real
    embedding model.
    """

    def __init__(self, results):
        self.results = results
        self.events: dict[str, MemoryEvent] = {r.event.id: r.event for r in results}

    def add(self, event):
        self.events[event.id] = event

    def get_all(self, tier=None):
        events = list(self.events.values())
        if tier is not None:
            events = [e for e in events if e.tier == tier]
        return events

    def query(self, query, top_k=5, tier=None):
        return self.results[:top_k]

    def update_tier(self, event_id, new_tier):
        if event_id in self.events:
            self.events[event_id].tier = new_tier

    def remove(self, event_id):
        self.events.pop(event_id, None)


def test_query_requests_fetch_k_not_top_k_from_each_backend():
    lexical = InMemoryBackend()
    semantic = InMemoryBackend()
    hybrid = HybridBackend(lexical_backend=lexical, semantic_backend=semantic, fetch_k_multiplier=5)
    for i in range(10):
        hybrid.add(MemoryEvent(content=f"document number {i} about cats and kittens"))

    with patch.object(lexical, "query", wraps=lexical.query) as lexical_spy, \
         patch.object(semantic, "query", wraps=semantic.query) as semantic_spy:
        hybrid.query("cats", top_k=2)

    lexical_spy.assert_called_once_with("cats", top_k=10, tier=None)
    semantic_spy.assert_called_once_with("cats", top_k=10, tier=None)


def test_query_passes_tier_through_to_both_backends():
    lexical = InMemoryBackend()
    semantic = InMemoryBackend()
    hybrid = HybridBackend(lexical_backend=lexical, semantic_backend=semantic)

    with patch.object(lexical, "query", wraps=lexical.query) as lexical_spy, \
         patch.object(semantic, "query", wraps=semantic.query) as semantic_spy:
        hybrid.query("cats", top_k=3, tier=MemoryTier.LONG_TERM)

    lexical_spy.assert_called_once_with("cats", top_k=15, tier=MemoryTier.LONG_TERM)
    semantic_spy.assert_called_once_with("cats", top_k=15, tier=MemoryTier.LONG_TERM)


def test_query_returns_document_found_by_only_one_backend():
    lexical = InMemoryBackend()  # empty -- simulates zero lexical overlap
    semantic_only_event = MemoryEvent(content="a fact only the semantic backend would find")
    semantic = StubBackend(
        results=[RetrievalResult(event=semantic_only_event, score=0.9, tier=semantic_only_event.tier)]
    )
    hybrid = HybridBackend(lexical_backend=lexical, semantic_backend=semantic)

    results = hybrid.query("food I can't eat", top_k=5)

    result_ids = {r.event.id for r in results}
    assert semantic_only_event.id in result_ids


def test_query_truncates_fused_results_to_top_k():
    lexical = InMemoryBackend()
    semantic = InMemoryBackend()
    hybrid = HybridBackend(lexical_backend=lexical, semantic_backend=semantic)
    for i in range(10):
        event = MemoryEvent(content=f"document {i} about cats")
        lexical.add(event)
        semantic.add(event)

    results = hybrid.query("cats", top_k=3)

    assert len(results) == 3
