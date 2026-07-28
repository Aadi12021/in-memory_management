"""Tests for reciprocal_rank_fusion() and HybridBackend."""

from __future__ import annotations

from memory_system.backends.hybrid import reciprocal_rank_fusion
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
