from __future__ import annotations

import sys
from datetime import datetime, timezone
from pathlib import Path

# Import ScriptedExtractor from test_graph_backend
sys.path.insert(0, str(Path(__file__).parent))
from test_graph_backend import ScriptedExtractor

from memory_system import AlwaysConsolidate, NoDecay, TieredMemory
from memory_system.backends.graph import GraphBackend
from memory_system.backends.hybrid import HybridBackend
from memory_system.backends.memory import InMemoryBackend
from memory_system.core import _find_similar_pairs, _find_graph_backend
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


def make_long_term_memory():
    return TieredMemory(
        backend=InMemoryBackend(),
        consolidation_policy=AlwaysConsolidate(),
        decay_policy=NoDecay(),
    )


def add_long_term(memory, content, salience=0.5, timestamp=None, metadata=None):
    event = MemoryEvent(
        content=content,
        tier=MemoryTier.LONG_TERM,
        salience=salience,
        metadata=metadata or {},
    )
    if timestamp is not None:
        event.timestamp = timestamp
    memory.backend.add(event)
    return event


def test_deduplicate_merges_near_identical_pair():
    memory = make_long_term_memory()
    a = add_long_term(memory, "User is severely allergic to peanuts.")
    b = add_long_term(memory, "User is severely allergic to peanuts and tree nuts.")

    report = memory.deduplicate(threshold=0.3)

    assert len(report.merged) == 1
    assert set(report.merged[0][:2]) == {a.id, b.id}
    remaining = memory.backend.get_all(tier=MemoryTier.LONG_TERM)
    assert len(remaining) == 1
    assert remaining[0].id == report.merged[0][2]


def test_deduplicate_keeps_higher_salience_content():
    memory = make_long_term_memory()
    a = add_long_term(memory, "User is severely allergic to peanuts.", salience=0.9)
    add_long_term(memory, "User is severely allergic to peanuts and tree nuts.", salience=0.2)

    memory.deduplicate(threshold=0.3)

    remaining = memory.backend.get_all(tier=MemoryTier.LONG_TERM)
    assert remaining[0].content == a.content


def test_deduplicate_salience_is_max_of_both():
    memory = make_long_term_memory()
    add_long_term(memory, "User is severely allergic to peanuts.", salience=0.3)
    add_long_term(memory, "User is severely allergic to peanuts and tree nuts.", salience=0.9)

    memory.deduplicate(threshold=0.3)

    remaining = memory.backend.get_all(tier=MemoryTier.LONG_TERM)
    assert remaining[0].salience == 0.9


def test_deduplicate_timestamp_is_earliest_source():
    memory = make_long_term_memory()
    early = datetime(2020, 1, 1, tzinfo=timezone.utc)
    late = datetime(2024, 1, 1, tzinfo=timezone.utc)
    add_long_term(memory, "User is severely allergic to peanuts.", timestamp=late)
    add_long_term(memory, "User is severely allergic to peanuts and tree nuts.", timestamp=early)

    memory.deduplicate(threshold=0.3)

    remaining = memory.backend.get_all(tier=MemoryTier.LONG_TERM)
    assert remaining[0].timestamp == early


def test_deduplicate_last_reinforced_is_set_to_now():
    memory = make_long_term_memory()
    add_long_term(memory, "User is severely allergic to peanuts.")
    add_long_term(memory, "User is severely allergic to peanuts and tree nuts.")
    before = datetime.now(timezone.utc)

    memory.deduplicate(threshold=0.3)

    remaining = memory.backend.get_all(tier=MemoryTier.LONG_TERM)
    assert remaining[0].last_reinforced >= before


def test_deduplicate_records_merged_from_provenance():
    memory = make_long_term_memory()
    a = add_long_term(memory, "User is severely allergic to peanuts.")
    b = add_long_term(memory, "User is severely allergic to peanuts and tree nuts.")

    memory.deduplicate(threshold=0.3)

    remaining = memory.backend.get_all(tier=MemoryTier.LONG_TERM)
    assert set(remaining[0].metadata["merged_from"]) == {a.id, b.id}


def test_deduplicate_leaves_unrelated_events_untouched():
    memory = make_long_term_memory()
    add_long_term(memory, "User is severely allergic to peanuts.")
    add_long_term(memory, "The weather forecast for tomorrow is sunny.")

    report = memory.deduplicate(threshold=0.5)

    assert report.merged == []
    assert len(memory.backend.get_all(tier=MemoryTier.LONG_TERM)) == 2


def test_deduplicate_dry_run_does_not_mutate_backend():
    memory = make_long_term_memory()
    a = add_long_term(memory, "User is severely allergic to peanuts.")
    b = add_long_term(memory, "User is severely allergic to peanuts and tree nuts.")

    report = memory.deduplicate(threshold=0.3, dry_run=True)

    assert len(report.merged) == 1
    assert set(report.merged[0][:2]) == {a.id, b.id}
    assert report.merged[0][2] is None
    remaining_ids = {e.id for e in memory.backend.get_all(tier=MemoryTier.LONG_TERM)}
    assert remaining_ids == {a.id, b.id}


def test_find_graph_backend_returns_graph_backend_directly():
    graph = GraphBackend(extractor=ScriptedExtractor())
    assert _find_graph_backend(graph) is graph


def test_find_graph_backend_finds_it_inside_hybrid_backend():
    lexical = InMemoryBackend()
    graph = GraphBackend(extractor=ScriptedExtractor())
    hybrid = HybridBackend(lexical_backend=lexical, semantic_backend=graph)
    assert _find_graph_backend(hybrid) is graph


def test_find_graph_backend_returns_none_for_plain_backend():
    assert _find_graph_backend(InMemoryBackend()) is None


def test_deduplicate_on_graph_backend_preserves_entities_unique_to_each_source():
    graph = GraphBackend(extractor=ScriptedExtractor())
    memory = TieredMemory(backend=graph, consolidation_policy=AlwaysConsolidate(), decay_policy=NoDecay())
    a = MemoryEvent(
        content={"entities": [], "edges": [("user", "peanut", "ALLERGIC_TO")]},
        tier=MemoryTier.LONG_TERM, salience=0.9,
    )
    b = MemoryEvent(
        content={"entities": [], "edges": [("user", "hiking", "ENJOYS")]},
        tier=MemoryTier.LONG_TERM, salience=0.2,
    )
    memory.backend.add(a)
    memory.backend.add(b)

    # a and b share only the "user" entity, so GraphBackend's entity-overlap
    # query() scores this cross-pair at 1.0 (verified empirically) --
    # threshold=1.0 finds it, self-matches score 2.0 (both of a's own
    # entities) and are excluded by _find_similar_pairs regardless.
    report = memory.deduplicate(threshold=1.0)

    assert len(report.merged) == 1
    merged_id = report.merged[0][2]
    # a had higher salience, so its content (only mentioning "peanut")
    # survives verbatim -- but b's ENJOYS/hiking relationship must still be
    # preserved via reassign_relationships, not lost just because b's
    # content didn't survive.
    result = graph.related_to("user", max_hops=1)
    assert {e.id for e in result} == {"peanut", "hiking"}
    remaining = memory.backend.get_all(tier=MemoryTier.LONG_TERM)
    assert len(remaining) == 1
    assert remaining[0].id == merged_id
    assert remaining[0].content == a.content
