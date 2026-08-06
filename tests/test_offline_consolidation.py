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
from memory_system.events import ConsolidationReport, MemoryEvent, MemoryTier
from memory_system.summarization.base import MemorySummarizer


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
    # related_to() alone is NOT sufficient to catch reassign_relationships()
    # being broken: GraphBackend.remove() prunes self._edges by
    # source_event_id but never prunes self._adjacency, so a stale
    # adjacency entry left behind by a discarded event would still make
    # related_to() traverse to "hiking" even if reassign_relationships()
    # did nothing at all. Assert directly that the surviving relationship
    # is actually attached to the merged event, not a discarded original --
    # entities_for_event() is driven by self._edges (source_event_id), not
    # the adjacency index, so it can't be fooled by a stale entry.
    assert graph.entities_for_event(merged_id) == {"user", "peanut", "hiking"}
    remaining = memory.backend.get_all(tier=MemoryTier.LONG_TERM)
    assert len(remaining) == 1
    assert remaining[0].id == merged_id
    assert remaining[0].content == a.content


class FakeSummarizer(MemorySummarizer):
    """Test double: returns fixed text without calling any API,
    records what it was called with -- matches test_hybrid_backend.py's
    StubBackend pattern for testing orchestration logic in isolation
    from a real dependency.
    """

    def __init__(self, summary_text="a summary"):
        self.summary_text = summary_text
        self.calls: list[list[MemoryEvent]] = []

    def summarize(self, events):
        self.calls.append(events)
        return self.summary_text


def test_compress_groups_and_summarizes_related_events():
    memory = make_long_term_memory()
    a = add_long_term(memory, "User is severely allergic to peanuts.")
    b = add_long_term(memory, "User is severely allergic to peanuts and tree nuts.")
    summarizer = FakeSummarizer("User has peanut and tree nut allergies.")

    report = memory.compress(threshold=0.3, summarizer=summarizer)

    assert len(report.compressed) == 1
    source_ids, new_id = report.compressed[0]
    assert set(source_ids) == {a.id, b.id}
    remaining = memory.backend.get_all(tier=MemoryTier.LONG_TERM)
    assert len(remaining) == 1
    assert remaining[0].id == new_id
    assert remaining[0].content == "User has peanut and tree nut allergies."


def test_compress_salience_is_max_of_group():
    memory = make_long_term_memory()
    add_long_term(memory, "User is severely allergic to peanuts.", salience=0.3)
    add_long_term(memory, "User is severely allergic to peanuts and tree nuts.", salience=0.9)

    memory.compress(threshold=0.3, summarizer=FakeSummarizer())

    remaining = memory.backend.get_all(tier=MemoryTier.LONG_TERM)
    assert remaining[0].salience == 0.9


def test_compress_records_summarized_from_provenance():
    memory = make_long_term_memory()
    a = add_long_term(memory, "User is severely allergic to peanuts.")
    b = add_long_term(memory, "User is severely allergic to peanuts and tree nuts.")

    memory.compress(threshold=0.3, summarizer=FakeSummarizer())

    remaining = memory.backend.get_all(tier=MemoryTier.LONG_TERM)
    assert set(remaining[0].metadata["summarized_from"]) == {a.id, b.id}


def test_compress_dry_run_does_not_mutate_backend_or_call_summarizer():
    memory = make_long_term_memory()
    a = add_long_term(memory, "User is severely allergic to peanuts.")
    b = add_long_term(memory, "User is severely allergic to peanuts and tree nuts.")
    summarizer = FakeSummarizer()

    report = memory.compress(threshold=0.3, summarizer=summarizer, dry_run=True)

    assert len(report.compressed) == 1
    assert report.compressed[0][1] is None
    assert summarizer.calls == []
    remaining_ids = {e.id for e in memory.backend.get_all(tier=MemoryTier.LONG_TERM)}
    assert remaining_ids == {a.id, b.id}


def test_compress_skips_group_when_summarizer_raises():
    memory = make_long_term_memory()
    a = add_long_term(memory, "User is severely allergic to peanuts.")
    b = add_long_term(memory, "User is severely allergic to peanuts and tree nuts.")

    class BrokenSummarizer(MemorySummarizer):
        def summarize(self, events):
            raise RuntimeError("API unavailable")

    report = memory.compress(threshold=0.3, summarizer=BrokenSummarizer())

    assert report.compressed == []
    remaining_ids = {e.id for e in memory.backend.get_all(tier=MemoryTier.LONG_TERM)}
    assert remaining_ids == {a.id, b.id}  # nothing removed, group was skipped


def test_compress_on_graph_backend_preserves_entities_the_summary_omits():
    graph = GraphBackend(extractor=ScriptedExtractor())
    memory = TieredMemory(backend=graph, consolidation_policy=AlwaysConsolidate(), decay_policy=NoDecay())
    a = MemoryEvent(content={"entities": [], "edges": [("user", "peanut", "ALLERGIC_TO")]}, tier=MemoryTier.LONG_TERM)
    b = MemoryEvent(content={"entities": [], "edges": [("user", "hiking", "ENJOYS")]}, tier=MemoryTier.LONG_TERM)
    c = MemoryEvent(content={"entities": [], "edges": [("user", "quinoa", "ENJOYS")]}, tier=MemoryTier.LONG_TERM)
    for event in (a, b, c):
        memory.backend.add(event)

    # all three share only the "user" entity, so GraphBackend's
    # entity-overlap query() cross-scores every pair at 1.0 (verified
    # empirically) -- threshold=1.0 groups all three into one component.
    # A fake summary that only mentions "peanut" simulates exactly what a
    # real LLM summary would plausibly do with a 3-item group.
    summarizer = FakeSummarizer("User has a peanut allergy.")
    report = memory.compress(threshold=1.0, summarizer=summarizer)

    assert len(report.compressed) == 1
    new_id = report.compressed[0][1]
    result = graph.related_to("user", max_hops=1)
    assert {e.id for e in result} == {"peanut", "hiking", "quinoa"}
    # related_to() alone can't tell a genuinely-preserved relationship
    # apart from a stale self._adjacency entry left behind by
    # GraphBackend.remove() (which prunes self._edges by source_event_id
    # but never touches self._adjacency) -- see the analogous dedup test
    # above. Confirm directly, via entities_for_event() (driven by
    # self._edges, not adjacency), that every relationship is actually
    # attached to the summary event, not a discarded original. The
    # summary's own content is a plain string ("User has a peanut
    # allergy."), which ScriptedExtractor doesn't parse into any edges at
    # all, so every entity here can only have arrived via
    # reassign_relationships().
    assert graph.entities_for_event(new_id) == {"user", "peanut", "hiking", "quinoa"}


def test_strengthen_connections_bumps_edge_between_entities_co_associated_by_merge():
    graph = GraphBackend(extractor=ScriptedExtractor())
    memory = TieredMemory(backend=graph, consolidation_policy=AlwaysConsolidate(), decay_policy=NoDecay())

    # a pre-existing, unrelated fact linking peanut and protein
    unrelated = MemoryEvent(content={"entities": [], "edges": [("peanut", "protein", "CONTAINS")]})
    graph.add(unrelated)
    graph.find_edge("peanut", "protein").strength = 0.5

    # two similar long-term events that will merge, co-associating
    # "peanut" and "protein" with the same surviving event
    a = MemoryEvent(content={"entities": [], "edges": [("user", "peanut", "ALLERGIC_TO")]}, tier=MemoryTier.LONG_TERM)
    b = MemoryEvent(content={"entities": [], "edges": [("user", "protein", "ALLERGIC_TO")]}, tier=MemoryTier.LONG_TERM)
    memory.backend.add(a)
    memory.backend.add(b)
    merge_report = memory.deduplicate(threshold=1.0)
    assert len(merge_report.merged) == 1

    report = memory.strengthen_connections(merge_report=merge_report)

    assert graph.find_edge("peanut", "protein").strength == 0.6
    assert {frozenset(pair) for pair in report.strengthened} == {frozenset({"peanut", "protein"})}


def test_strengthen_connections_skips_edges_sourced_from_the_consolidation_event_itself():
    # No pre-existing "bystander" edge here at all -- every edge that
    # ends up touching the merged event's entities is either freshly
    # extracted from the surviving event's own content or reassigned
    # onto it by reassign_relationships(). Both are "part of this
    # consolidation event" by definition (source_event_id == new_id),
    # so strengthen_connections() must skip all of them. If the
    # source_event_id == event_id guard were removed, this would find
    # the (user, peanut) and (user, protein) edges the merge itself
    # just created and strengthen them, which is exactly the bug this
    # guard exists to prevent (see review discussion in
    # docs/superpowers/specs/2026-08-06-offline-consolidation.md,
    # Mechanism 3).
    graph = GraphBackend(extractor=ScriptedExtractor())
    memory = TieredMemory(backend=graph, consolidation_policy=AlwaysConsolidate(), decay_policy=NoDecay())

    a = MemoryEvent(content={"entities": [], "edges": [("user", "peanut", "ALLERGIC_TO")]}, tier=MemoryTier.LONG_TERM)
    b = MemoryEvent(content={"entities": [], "edges": [("user", "protein", "ALLERGIC_TO")]}, tier=MemoryTier.LONG_TERM)
    memory.backend.add(a)
    memory.backend.add(b)
    merge_report = memory.deduplicate(threshold=1.0)
    assert len(merge_report.merged) == 1
    new_id = merge_report.merged[0][2]

    # Sanity check on the premise: entities_for_event(new_id) really
    # does include entities from both the surviving event's own edge
    # and the reassigned edge from the discarded event -- so there IS
    # a pairwise combination for the guard to have to skip.
    assert graph.entities_for_event(new_id) >= {"user", "peanut", "protein"}

    report = memory.strengthen_connections(merge_report=merge_report)

    assert report.strengthened == []


def test_strengthen_connections_strengthens_bystander_edge_even_when_a_consolidation_sourced_edge_exists_for_the_same_pair():
    """find_edge() (used internally before this fix) only returns the
    FIRST matching relationship for a pair, ignoring relation_type --
    but the graph model permits parallel typed edges between the same
    two entities (reassign_relationships() groups by (source_id,
    target_id, relation_type), so two different relation types both
    survive). If a pair has BOTH a consolidation-sourced edge and a
    genuine pre-existing bystander edge, strengthen_connections() must
    not let an arbitrary single-edge lookup decide the outcome -- it
    must check every relationship between the pair (find_edges()) and
    strengthen the bystander regardless of which one happens to come
    first in GraphBackend._edges.

    a's own content directly relates peanut and protein
    (INGREDIENT_OF); since a has the higher salience, its content (and
    that edge) survives the merge, so INGREDIENT_OF ends up sourced
    from the merged event -- a bona fide "part of this consolidation"
    edge. The CONTAINS edge from `unrelated` is added AFTER a and b, so
    it lands later in GraphBackend._edges than INGREDIENT_OF -- the
    exact ordering that made the old find_edge()-only implementation
    return the consolidation-sourced edge first, see it skipped by the
    self-source guard, and never even look at the bystander.
    """
    graph = GraphBackend(extractor=ScriptedExtractor())
    memory = TieredMemory(backend=graph, consolidation_policy=AlwaysConsolidate(), decay_policy=NoDecay())

    a = MemoryEvent(
        content={
            "entities": [],
            "edges": [("user", "peanut", "ALLERGIC_TO"), ("peanut", "protein", "INGREDIENT_OF")],
        },
        tier=MemoryTier.LONG_TERM, salience=0.9,
    )
    b = MemoryEvent(
        content={"entities": [], "edges": [("user", "protein", "ALLERGIC_TO")]},
        tier=MemoryTier.LONG_TERM, salience=0.1,
    )
    memory.backend.add(a)
    memory.backend.add(b)

    # genuine, unrelated bystander fact -- added after a/b so it lands
    # later in _edges than the surviving INGREDIENT_OF edge
    unrelated = MemoryEvent(content={"entities": [], "edges": [("peanut", "protein", "CONTAINS")]})
    graph.add(unrelated)
    for edge in graph.find_edges("peanut", "protein"):
        if edge.relation_type == "CONTAINS":
            edge.strength = 0.5

    merge_report = memory.deduplicate(threshold=1.0)
    assert len(merge_report.merged) == 1
    merged_id = merge_report.merged[0][2]

    # Sanity check on the premise: both a consolidation-sourced edge AND
    # a bystander edge now connect (peanut, protein).
    edges = graph.find_edges("peanut", "protein")
    assert len(edges) == 2
    consolidation_edge = next(e for e in edges if e.relation_type == "INGREDIENT_OF")
    bystander_edge = next(e for e in edges if e.relation_type == "CONTAINS")
    assert consolidation_edge.source_event_id == merged_id
    assert bystander_edge.source_event_id == unrelated.id

    report = memory.strengthen_connections(merge_report=merge_report)

    # the bystander gets strengthened...
    assert bystander_edge.strength == 0.6
    # ...but the consolidation's own new edge is untouched, per the
    # skip-self-sourced-edges contract.
    assert consolidation_edge.strength == 1.0
    assert {frozenset(pair) for pair in report.strengthened} == {frozenset({"peanut", "protein"})}


def test_strengthen_connections_caps_at_one():
    graph = GraphBackend(extractor=ScriptedExtractor())
    memory = TieredMemory(backend=graph, consolidation_policy=AlwaysConsolidate(), decay_policy=NoDecay())
    unrelated = MemoryEvent(content={"entities": [], "edges": [("peanut", "protein", "CONTAINS")]})
    graph.add(unrelated)
    graph.find_edge("peanut", "protein").strength = 0.95
    a = MemoryEvent(content={"entities": [], "edges": [("user", "peanut", "ALLERGIC_TO")]}, tier=MemoryTier.LONG_TERM)
    b = MemoryEvent(content={"entities": [], "edges": [("user", "protein", "ALLERGIC_TO")]}, tier=MemoryTier.LONG_TERM)
    memory.backend.add(a)
    memory.backend.add(b)
    merge_report = memory.deduplicate(threshold=1.0)

    memory.strengthen_connections(merge_report=merge_report)

    assert graph.find_edge("peanut", "protein").strength == 1.0


def test_strengthen_connections_dry_run_does_not_mutate_strength():
    graph = GraphBackend(extractor=ScriptedExtractor())
    memory = TieredMemory(backend=graph, consolidation_policy=AlwaysConsolidate(), decay_policy=NoDecay())
    unrelated = MemoryEvent(content={"entities": [], "edges": [("peanut", "protein", "CONTAINS")]})
    graph.add(unrelated)
    a = MemoryEvent(content={"entities": [], "edges": [("user", "peanut", "ALLERGIC_TO")]}, tier=MemoryTier.LONG_TERM)
    b = MemoryEvent(content={"entities": [], "edges": [("user", "protein", "ALLERGIC_TO")]}, tier=MemoryTier.LONG_TERM)
    memory.backend.add(a)
    memory.backend.add(b)
    merge_report = memory.deduplicate(threshold=1.0)

    memory.strengthen_connections(merge_report=merge_report, dry_run=True)

    assert graph.find_edge("peanut", "protein").strength == 1.0  # default, untouched


def test_strengthen_connections_with_no_reports_returns_empty():
    graph = GraphBackend(extractor=ScriptedExtractor())
    memory = TieredMemory(backend=graph, consolidation_policy=AlwaysConsolidate(), decay_policy=NoDecay())
    graph.add(MemoryEvent(content={"entities": [], "edges": [("peanut", "protein", "CONTAINS")]}))

    report = memory.strengthen_connections()

    assert report.strengthened == []


def test_strengthen_connections_on_non_graph_backend_returns_empty():
    memory = make_long_term_memory()  # plain InMemoryBackend
    add_long_term(memory, "User is severely allergic to peanuts.")

    report = memory.strengthen_connections()

    assert report.strengthened == []


def test_offline_consolidate_runs_dedup_then_compress():
    memory = make_long_term_memory()
    a = add_long_term(memory, "User is severely allergic to peanuts.")
    b = add_long_term(memory, "User is severely allergic to peanuts and tree nuts.")

    report = memory.offline_consolidate(merge_threshold=0.3, group_threshold=0.3)

    assert len(report.merged) == 1
    assert set(report.merged[0][:2]) == {a.id, b.id}
    remaining = memory.backend.get_all(tier=MemoryTier.LONG_TERM)
    assert len(remaining) == 1


def test_offline_consolidate_skips_compress_when_no_summarizer_given():
    """Regression note: the original version of this test used a
    single similar pair with merge_threshold == group_threshold, so
    deduplicate() already merged the only two similar events, leaving
    nothing for compress() to group -- report.compressed == [] held
    regardless of whether the `if summarizer is not None` guard in
    offline_consolidate() existed at all. A version of the wrapper
    that always called compress() (dropping the guard) would have
    passed that test unchanged (verified empirically by calling
    memory.compress(0.3, None, dry_run=False) directly on the
    post-dedup backend state and getting compressed: [] either way).

    This version fixes the "nothing left to compress" half of the gap
    by using a merge_threshold (0.8) stricter than the pair's actual
    similarity (~0.75, verified empirically), so deduplicate() does
    NOT merge it and the pair survives untouched with group_threshold
    (0.3) loose enough that compress() would group it if it ran.

    But that alone still isn't discriminating in non-dry_run mode:
    compress()'s `except Exception: continue` around
    `summarizer.summarize(group_events)` is fail-soft by design, so
    even a guardless offline_consolidate() calling
    compress(group_threshold, None, dry_run=False) would hit
    `None.summarize(...)`, swallow the resulting AttributeError, and
    still return compressed == [] -- verified empirically by patching
    the guard out and rerunning this exact scenario. Non-dry_run mode
    genuinely cannot tell "guard skipped compress" apart from "compress
    ran, summarizer blew up, group was skipped".

    dry_run=True is what actually discriminates: with dry_run, compress()
    skips the `summarizer.summarize()` call entirely (it's inside the
    `if not dry_run` block) and unconditionally records each found group
    in report.compressed with new_id=None. So a guardless wrapper calling
    compress(group_threshold, None, dry_run=True) would return a
    *non-empty* report.compressed for this same pair (verified
    empirically), while the guarded wrapper never calls compress() at
    all and returns compressed == []. That's the assertion that would
    actually fail if the guard were removed.
    """
    memory = make_long_term_memory()
    a = add_long_term(memory, "User is severely allergic to peanuts.")
    b = add_long_term(memory, "User is severely allergic to peanuts and tree nuts.")

    report = memory.offline_consolidate(merge_threshold=0.8, group_threshold=0.3)

    assert report.merged == []
    assert report.compressed == []
    remaining = memory.backend.get_all(tier=MemoryTier.LONG_TERM)
    assert len(remaining) == 2

    # Prove the premise: compress() really would have grouped this
    # pair had the guard not skipped it -- i.e. compressed == [] above
    # reflects the guard, not an empty tier.
    would_be_report = memory.compress(0.3, FakeSummarizer(), dry_run=True)
    assert len(would_be_report.compressed) == 1
    assert set(would_be_report.compressed[0][0]) == {a.id, b.id}

    # The actually-discriminating check: under dry_run, compress()
    # records a group without ever touching the summarizer, so a
    # guardless wrapper would populate report.compressed here even
    # with summarizer=None. Only the guard keeps this empty.
    dry_report = memory.offline_consolidate(merge_threshold=0.8, group_threshold=0.3, dry_run=True)
    assert dry_report.compressed == []


def test_offline_consolidate_calls_compress_when_summarizer_given():
    """No other Task 8 test ever calls offline_consolidate() with a
    real summarizer, so the `self.compress(group_threshold, summarizer,
    dry_run=dry_run)` call itself -- its argument wiring, and its
    running at all -- was never exercised through the wrapper.

    Four events, two topics. merge_threshold=0.77 sits between the two
    pairs' actual similarity scores (hiking ~0.79, allergy ~0.75, both
    verified empirically): deduplicate() merges the hiking pair but
    leaves the allergy pair untouched. group_threshold=0.3 then lets
    compress() group and summarize the surviving allergy pair through
    the wrapper. Asserting FakeSummarizer.calls is non-empty proves
    offline_consolidate() actually invoked compress() (and thus the
    summarizer), not merely that no error occurred.
    """
    memory = make_long_term_memory()
    a1 = add_long_term(memory, "User is severely allergic to peanuts.")
    a2 = add_long_term(memory, "User is severely allergic to peanuts and tree nuts.")
    add_long_term(memory, "User enjoys hiking on weekends.")
    add_long_term(memory, "User enjoys hiking and camping on weekends.")
    summarizer = FakeSummarizer("User has peanut and tree nut allergies.")

    report = memory.offline_consolidate(
        merge_threshold=0.77, group_threshold=0.3, summarizer=summarizer
    )

    assert len(report.merged) == 1  # the hiking pair, merged by dedup
    assert len(report.compressed) == 1  # the allergy pair, compressed
    assert set(report.compressed[0][0]) == {a1.id, a2.id}
    assert len(summarizer.calls) == 1  # wrapper really invoked compress()
    remaining = memory.backend.get_all(tier=MemoryTier.LONG_TERM)
    assert len(remaining) == 2  # hiking survivor + allergy summary


def test_offline_consolidate_dry_run_mutates_nothing():
    memory = make_long_term_memory()
    a = add_long_term(memory, "User is severely allergic to peanuts.")
    b = add_long_term(memory, "User is severely allergic to peanuts and tree nuts.")

    report = memory.offline_consolidate(merge_threshold=0.3, group_threshold=0.9, dry_run=True)

    assert len(report.merged) == 1
    assert report.merged[0][2] is None
    remaining_ids = {e.id for e in memory.backend.get_all(tier=MemoryTier.LONG_TERM)}
    assert remaining_ids == {a.id, b.id}


def test_offline_consolidate_dry_run_propagates_to_compress_and_strengthen():
    """The dry_run test above only ever exercises dedup: it uses
    InMemoryBackend with no summarizer, so strengthen_connections()
    short-circuits to empty immediately (non-graph backend) and
    compress() is never called at all (no summarizer) -- neither
    stage's dry_run handling is checked through the wrapper.

    This test uses a GraphBackend (so strengthen_connections() has a
    pre-existing bystander edge it could touch) and a FakeSummarizer
    (so compress() has a real group it could summarize), then calls
    offline_consolidate(..., dry_run=True) and checks all three
    stages left everything untouched: the summarizer was never
    invoked, the bystander edge's strength is unchanged, and the two
    long-term events are still present under their original ids.
    """
    graph = GraphBackend(extractor=ScriptedExtractor())
    memory = TieredMemory(backend=graph, consolidation_policy=AlwaysConsolidate(), decay_policy=NoDecay())
    unrelated = MemoryEvent(content={"entities": [], "edges": [("peanut", "protein", "CONTAINS")]})
    graph.add(unrelated)
    graph.find_edge("peanut", "protein").strength = 0.5
    a = MemoryEvent(content={"entities": [], "edges": [("user", "peanut", "ALLERGIC_TO")]}, tier=MemoryTier.LONG_TERM)
    b = MemoryEvent(content={"entities": [], "edges": [("user", "protein", "ALLERGIC_TO")]}, tier=MemoryTier.LONG_TERM)
    memory.backend.add(a)
    memory.backend.add(b)
    summarizer = FakeSummarizer()

    report = memory.offline_consolidate(
        merge_threshold=1.0, group_threshold=1.0, summarizer=summarizer, dry_run=True
    )

    # Sanity check on the premise: there really was a mergeable pair
    # and a compressible group here, so the assertions below reflect
    # dry_run suppressing real work, not an absence of work.
    assert len(report.merged) == 1
    assert report.merged[0][2] is None
    assert len(report.compressed) == 1
    assert report.compressed[0][1] is None

    # compress()'s summarizer is never invoked under dry_run.
    assert summarizer.calls == []
    # strengthen_connections() never mutates the bystander edge.
    assert graph.find_edge("peanut", "protein").strength == 0.5
    # deduplicate() never mutates the backend.
    remaining_ids = {e.id for e in graph.get_all(tier=MemoryTier.LONG_TERM)}
    assert remaining_ids == {a.id, b.id}


def test_offline_consolidate_end_to_end_on_hybrid_backend_reaches_through_to_graph_backend():
    """Every other GraphBackend-flavored test in this module exercises
    TieredMemory against a bare GraphBackend. _find_graph_backend()'s
    reach-through path for a GraphBackend nested inside a HybridBackend
    is otherwise only unit-tested at the helper-function level
    (test_find_graph_backend_finds_it_inside_hybrid_backend) -- nothing
    builds a real TieredMemory over a HybridBackend and runs
    offline_consolidate() end-to-end to prove the whole pipeline
    actually reaches the nested GraphBackend, not just the helper.

    HybridBackend mirrors every write to both of its backends
    (InMemoryBackend here, matching how it's constructed elsewhere in
    the suite -- see tests/test_hybrid_backend.py) and fuses query()
    results via Reciprocal Rank Fusion, so `threshold` is on a very
    different scale (~0.03, not ~1.0 the way GraphBackend's raw
    entity-overlap score is) -- verified empirically: a and b's
    RRF-fused score here is 1/(60+2) + 1/(60+2) ~= 0.0323 (rank 2 in
    both the lexical and semantic result lists, RRF k=60), so 0.03 finds it.
    """
    lexical = InMemoryBackend()
    graph = GraphBackend(extractor=ScriptedExtractor())
    hybrid = HybridBackend(lexical_backend=lexical, semantic_backend=graph)
    memory = TieredMemory(backend=hybrid, consolidation_policy=AlwaysConsolidate(), decay_policy=NoDecay())

    # a genuine, unrelated bystander graph fact, pre-dating the merge
    unrelated = MemoryEvent(content={"entities": [], "edges": [("peanut", "protein", "CONTAINS")]})
    memory.backend.add(unrelated)
    graph.find_edge("peanut", "protein").strength = 0.5

    # two long-term events that will dedup-merge, co-associating
    # "peanut" and "protein" with the same surviving event -- and each
    # contributing a relationship (the "protein" one) the other doesn't
    # mention, so preservation via reassign_relationships matters too
    a = MemoryEvent(
        content={"entities": [], "edges": [("user", "peanut", "ALLERGIC_TO")]},
        tier=MemoryTier.LONG_TERM, salience=0.9,
    )
    b = MemoryEvent(
        content={"entities": [], "edges": [("user", "protein", "ALLERGIC_TO")]},
        tier=MemoryTier.LONG_TERM, salience=0.2,
    )
    memory.backend.add(a)
    memory.backend.add(b)

    report = memory.offline_consolidate(merge_threshold=0.03, group_threshold=0.03)

    # dedup ran through the HybridBackend and reassigned relationships
    # on the nested GraphBackend
    assert len(report.merged) == 1
    merged_id = report.merged[0][2]
    assert graph.entities_for_event(merged_id) == {"user", "peanut", "protein"}

    # strengthen_connections() reached the nested GraphBackend and
    # bumped the pre-existing bystander edge -- not the merge's own
    # (there is no direct peanut-protein edge from the merge itself
    # here, only the CONTAINS bystander, so this also confirms the
    # bystander, not a phantom, was what got strengthened)
    assert report.strengthened == [("peanut", "protein")]
    assert graph.find_edge("peanut", "protein").strength == 0.6

    # the merge actually happened on both halves of the HybridBackend
    remaining = memory.backend.get_all(tier=MemoryTier.LONG_TERM)
    assert len(remaining) == 1
    assert remaining[0].id == merged_id


def test_offline_consolidate_order_prevents_strengthening_edges_about_to_be_removed():
    """If strengthen_connections() ran BEFORE deduplicate() (the wrong
    order), it would strengthen edges belonging to events dedup is
    about to remove -- and since GraphBackend.remove() prunes edges by
    source_event_id, those edges would then be deleted moments after
    being strengthened, wasted work at best. Running dedup first means
    that by the time strengthen_connections() runs, source_event_id on
    every surviving edge already points at the merged event, not a
    doomed original -- so this test isn't just checking dedup ran
    first, it's checking the edges strengthen_connections() sees are
    the actual final, post-cleanup ones, not ones about to vanish.
    """
    graph = GraphBackend(extractor=ScriptedExtractor())
    memory = TieredMemory(backend=graph, consolidation_policy=AlwaysConsolidate(), decay_policy=NoDecay())
    unrelated = MemoryEvent(content={"entities": [], "edges": [("peanut", "protein", "CONTAINS")]})
    graph.add(unrelated)
    graph.find_edge("peanut", "protein").strength = 0.5
    a = MemoryEvent(content={"entities": [], "edges": [("user", "peanut", "ALLERGIC_TO")]}, tier=MemoryTier.LONG_TERM)
    b = MemoryEvent(content={"entities": [], "edges": [("user", "protein", "ALLERGIC_TO")]}, tier=MemoryTier.LONG_TERM)
    memory.backend.add(a)
    memory.backend.add(b)

    report = memory.offline_consolidate(merge_threshold=1.0, group_threshold=1.0)

    assert graph.find_edge("peanut", "protein").strength == 0.6
    merged_id = report.merged[0][2]
    assert {"user", "peanut", "protein"} <= graph.entities_for_event(merged_id)
    remaining_ids = {e.id for e in graph.get_all()}
    assert a.id not in remaining_ids
    assert b.id not in remaining_ids
