"""Integration tests for GraphBackend against the real
RuleBasedEntityExtractor, as opposed to test_graph_backend.py's
ScriptedExtractor test double. The point here is specifically to prove
related_to()/consolidation_signal()/query()/explain_path() don't secretly
assume any fixed entity name, relation-type string, or fixture-specific
graph shape: every test below is paired with a same-shaped test in a
structurally different domain (allergy/food vs. geography vs. work), and
the assertions would fail if the implementation were hardcoded to any one
of them.
"""

from memory_system.backends.graph import GraphBackend
from memory_system.events import MemoryEvent
from memory_system.extraction.rules_based import RuleBasedEntityExtractor


def build_graph(sentences):
    graph = GraphBackend(extractor=RuleBasedEntityExtractor())
    events = [MemoryEvent(content=text) for text in sentences]
    for event in events:
        graph.add(event)
    return graph, events


# --- related_to: same traversal, two unrelated domains -----------------------


def test_related_to_multihop_allergy_domain():
    # user -ALLERGIC_TO-> peanut -CONTAINS-> protein
    graph, _ = build_graph([
        "User is allergic to peanuts.",
        "Peanuts contains protein.",
    ])
    related = graph.related_to("user", max_hops=2)
    assert {e.id for e in related} == {"peanut", "protein"}


def test_related_to_multihop_geography_domain():
    # user -LIVES_IN-> boston -CONTAINS-> fenway park
    # Different entity names, different relation types than the allergy
    # test above -- if related_to() were hardcoded to "peanut" or
    # "ALLERGIC_TO" this would fail while the allergy test passed.
    graph, _ = build_graph([
        "User lives in Boston.",
        "Boston contains Fenway Park.",
    ])
    related = graph.related_to("user", max_hops=2)
    assert {e.id for e in related} == {"boston", "fenway park"}


def test_related_to_filters_by_real_relation_type():
    graph, _ = build_graph([
        "User is allergic to peanuts.",
        "User enjoys hiking.",
    ])
    related = graph.related_to("user", relation_type="ALLERGIC_TO")
    assert [e.id for e in related] == ["peanut"]


# --- consolidation_signal: same formula, two unrelated domains ---------------


def test_consolidation_signal_from_real_extraction_allergy_domain():
    graph, _ = build_graph([
        "User is allergic to peanuts.",
        "User enjoys hiking.",
        "User enjoys quinoa.",
    ])
    # "user" is the source of 3 real extracted edges; "hiking" of 1.
    assert graph.consolidation_signal("user") == 3 / 4
    assert graph.consolidation_signal("hiking") == 1 / 2


def test_consolidation_signal_from_real_extraction_work_domain():
    # Different entities/relation types than the allergy test -- proves
    # the degree count isn't tied to any specific extracted id.
    graph, _ = build_graph([
        "User works at Anthropic.",
        "User lives in Boston.",
    ])
    assert graph.consolidation_signal("user") == 2 / 3
    assert graph.consolidation_signal("anthropic") == 1 / 2


# --- query: entity-overlap matching against real extracted entities ----------


def test_query_matches_real_extracted_entity_mentioned_in_text():
    graph, events = build_graph([
        "User is allergic to peanuts.",
        "User enjoys hiking.",
    ])
    results = graph.query("peanuts")
    assert len(results) == 1
    assert results[0].event.id == events[0].id


def test_query_ranks_by_overlap_across_unrelated_domains():
    # One allergy-domain event, one geography-domain event in the same
    # graph. If query() secretly assumed a fixed set of entities, mixing
    # domains like this would break the ranking.
    graph, events = build_graph([
        "User is allergic to peanuts.",
        "User lives in Boston.",
    ])
    results = graph.query("user peanuts")
    assert results[0].event.id == events[0].id
    assert results[0].score > results[1].score


# --- explain_path: same traversal, two unrelated domains ---------------------


def test_explain_path_real_multihop_chain_work_domain():
    # user -WORKS_AT-> anthropic -CONTAINS-> research lab
    graph, _ = build_graph([
        "User works at Anthropic.",
        "Anthropic contains research labs.",
    ])
    path = graph.explain_path("user", "research lab")
    assert [r.relation_type for r in path] == ["WORKS_AT", "CONTAINS"]


def test_explain_path_real_multihop_chain_allergy_domain():
    # Different domain/relation types than the work-chain test above --
    # proves the traversal isn't tied to any specific relation string.
    graph, _ = build_graph([
        "User is allergic to peanuts.",
        "Peanuts contains protein.",
    ])
    path = graph.explain_path("user", "protein")
    assert [r.relation_type for r in path] == ["ALLERGIC_TO", "CONTAINS"]


def test_explain_path_none_when_no_real_path_exists():
    graph, _ = build_graph([
        "User is allergic to peanuts.",
        "User enjoys hiking.",
    ])
    assert graph.explain_path("peanut", "hiking") is None
