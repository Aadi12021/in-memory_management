from memory_system.backends.graph import Entity, EntityExtractor, GraphBackend, Relationship
from memory_system.events import MemoryEvent, MemoryTier


class ScriptedExtractor(EntityExtractor):
    """Test double: bypasses the extraction problem entirely so graph
    traversal/structure logic can be tested against a graph shape the
    test controls exactly. event.content is a dict of the form
    {"entities": [id, ...], "edges": [(source_id, target_id, relation_type), ...]}.
    Content that isn't dict-shaped (e.g. a plain summary string from
    compress()) yields no entities/relationships -- same fail-soft
    convention real extractors use for content they can't parse.
    """

    def extract(self, event: MemoryEvent) -> tuple[list[Entity], list[Relationship]]:
        spec = event.content
        if not isinstance(spec, dict):
            return [], []
        entities: dict[str, Entity] = {eid: Entity(id=eid, label=eid) for eid in spec.get("entities", [])}
        relationships: list[Relationship] = []
        for source_id, target_id, relation_type in spec.get("edges", []):
            entities.setdefault(source_id, Entity(id=source_id, label=source_id))
            entities.setdefault(target_id, Entity(id=target_id, label=target_id))
            relationships.append(
                Relationship(
                    source_id=source_id,
                    target_id=target_id,
                    relation_type=relation_type,
                    source_event_id=event.id,
                )
            )
        return list(entities.values()), relationships


def make_graph(edges=(), entities=()):
    graph = GraphBackend(extractor=ScriptedExtractor())
    graph.add(MemoryEvent(content={"entities": list(entities), "edges": list(edges)}))
    return graph


def make_graph_from_events(specs, tiers=None):
    """specs: one {"entities": [...], "edges": [...]} dict per MemoryEvent.
    tiers, if given, is a same-length list of MemoryTier to assign each event.
    """
    graph = GraphBackend(extractor=ScriptedExtractor())
    events = []
    for i, spec in enumerate(specs):
        kwargs = {"tier": tiers[i]} if tiers else {}
        event = MemoryEvent(content=spec, **kwargs)
        graph.add(event)
        events.append(event)
    return graph, events


# --- related_to ------------------------------------------------------------


def test_related_to_returns_direct_neighbor():
    graph = make_graph(edges=[("user", "peanut", "ALLERGIC_TO")])
    result = graph.related_to("user")
    assert [e.id for e in result] == ["peanut"]


def test_related_to_filters_by_relation_type():
    graph = make_graph(edges=[
        ("user", "peanut", "ALLERGIC_TO"),
        ("user", "hiking", "ENJOYS"),
    ])
    result = graph.related_to("user", relation_type="ALLERGIC_TO")
    assert [e.id for e in result] == ["peanut"]


def test_related_to_traverses_multiple_hops():
    graph = make_graph(edges=[
        ("user", "peanut", "ALLERGIC_TO"),
        ("peanut", "peanut_butter_cake", "INGREDIENT_OF"),
    ])
    result = graph.related_to("user", max_hops=2)
    assert {e.id for e in result} == {"peanut", "peanut_butter_cake"}


def test_related_to_respects_max_hops_limit():
    graph = make_graph(edges=[
        ("user", "peanut", "ALLERGIC_TO"),
        ("peanut", "peanut_butter_cake", "INGREDIENT_OF"),
    ])
    result = graph.related_to("user", max_hops=1)
    assert [e.id for e in result] == ["peanut"]


def test_related_to_avoids_infinite_loop_on_cycles():
    graph = make_graph(edges=[
        ("user", "peanut", "ALLERGIC_TO"),
        ("peanut", "user", "ALLERGIC_TO"),
    ])
    result = graph.related_to("user", max_hops=5)
    assert [e.id for e in result] == ["peanut"]


def test_related_to_isolated_entity_returns_empty():
    graph = make_graph(entities=["hiking"])
    assert graph.related_to("hiking") == []


def test_related_to_unknown_entity_returns_empty():
    graph = make_graph(edges=[("user", "peanut", "ALLERGIC_TO")])
    assert graph.related_to("nonexistent") == []


# --- consolidation_signal ----------------------------------------------------


def test_consolidation_signal_zero_for_isolated_entity():
    graph = make_graph(entities=["hiking"])
    assert graph.consolidation_signal("hiking") == 0.0


def test_consolidation_signal_zero_for_unknown_entity():
    graph = make_graph(edges=[("user", "peanut", "ALLERGIC_TO")])
    assert graph.consolidation_signal("nonexistent") == 0.0


def test_consolidation_signal_counts_edges_regardless_of_direction():
    graph = make_graph(edges=[
        ("peanut_butter_cake", "peanut", "CONTAINS"),
        ("user", "peanut", "ALLERGIC_TO"),
    ])
    # "peanut" has degree 2 (target of both edges); "user" has degree 1.
    assert graph.consolidation_signal("peanut") == 2 / 3
    assert graph.consolidation_signal("user") == 1 / 2


def test_consolidation_signal_increases_with_more_connections():
    graph = make_graph(edges=[
        ("user", "peanut", "ALLERGIC_TO"),
        ("user", "hiking", "ENJOYS"),
        ("user", "quinoa", "ENJOYS"),
    ])
    signal_user = graph.consolidation_signal("user")
    signal_peanut = graph.consolidation_signal("peanut")
    assert 0.0 < signal_peanut < signal_user < 1.0


def test_consolidation_signal_is_bounded_below_one():
    graph = make_graph(edges=[
        ("user", "a", "ENJOYS"),
        ("user", "b", "ENJOYS"),
        ("user", "c", "ENJOYS"),
        ("user", "d", "ENJOYS"),
        ("user", "e", "ENJOYS"),
    ])
    assert graph.consolidation_signal("user") < 1.0


# --- query -------------------------------------------------------------------


def test_query_returns_event_whose_entity_is_mentioned_in_query_text():
    graph, events = make_graph_from_events([
        {"edges": [("user", "peanut", "ALLERGIC_TO")]},
        {"edges": [("user", "hiking", "ENJOYS")]},
    ])
    results = graph.query("peanut")
    assert len(results) == 1
    assert results[0].event.id == events[0].id


def test_query_ranks_higher_entity_overlap_first():
    graph, events = make_graph_from_events([
        {"edges": [
            ("user", "peanut", "ALLERGIC_TO"),
            ("peanut", "peanut_butter_cake", "INGREDIENT_OF"),
        ]},
        {"edges": [("user", "hiking", "ENJOYS")]},
    ])
    results = graph.query("user peanut peanut_butter_cake")
    assert results[0].event.id == events[0].id
    assert results[0].score > results[1].score


def test_query_respects_top_k():
    graph, events = make_graph_from_events([
        {"edges": [("user", "peanut", "ALLERGIC_TO")]},
        {"edges": [("user", "hiking", "ENJOYS")]},
        {"edges": [("user", "quinoa", "ENJOYS")]},
    ])
    results = graph.query("user", top_k=2)
    assert len(results) == 2


def test_query_filters_by_tier():
    graph, events = make_graph_from_events(
        [
            {"edges": [("user", "peanut", "ALLERGIC_TO")]},
            {"edges": [("user", "peanut", "ALLERGIC_TO")]},
        ],
        tiers=[MemoryTier.WORKING, MemoryTier.LONG_TERM],
    )
    results = graph.query("peanut", tier=MemoryTier.LONG_TERM)
    assert len(results) == 1
    assert results[0].event.id == events[1].id


def test_query_no_matching_entities_returns_empty():
    graph, _ = make_graph_from_events([{"edges": [("user", "peanut", "ALLERGIC_TO")]}])
    assert graph.query("weather forecast tomorrow") == []


def test_query_empty_graph_returns_empty():
    graph = GraphBackend(extractor=ScriptedExtractor())
    assert graph.query("anything") == []


# --- explain_path --------------------------------------------------------------


def test_explain_path_returns_direct_relationship():
    graph = make_graph(edges=[("user", "peanut", "ALLERGIC_TO")])
    path = graph.explain_path("user", "peanut")
    assert [r.relation_type for r in path] == ["ALLERGIC_TO"]


def test_explain_path_returns_multi_hop_chain():
    graph = make_graph(edges=[
        ("user", "peanut", "ALLERGIC_TO"),
        ("peanut", "peanut_butter_cake", "INGREDIENT_OF"),
    ])
    path = graph.explain_path("user", "peanut_butter_cake")
    assert [r.relation_type for r in path] == ["ALLERGIC_TO", "INGREDIENT_OF"]


def test_explain_path_finds_shortest_path_when_multiple_exist():
    graph = make_graph(edges=[
        ("user", "peanut", "ALLERGIC_TO"),
        ("peanut", "cake", "INGREDIENT_OF"),
        ("user", "cake", "AVOID"),
    ])
    path = graph.explain_path("user", "cake")
    assert [r.relation_type for r in path] == ["AVOID"]


def test_explain_path_returns_none_when_no_path_exists():
    graph = make_graph(edges=[("user", "peanut", "ALLERGIC_TO")], entities=["hiking"])
    assert graph.explain_path("user", "hiking") is None


def test_explain_path_returns_empty_list_for_same_entity():
    graph = make_graph(entities=["user"])
    assert graph.explain_path("user", "user") == []


def test_explain_path_returns_none_for_unknown_entities():
    graph = make_graph(edges=[("user", "peanut", "ALLERGIC_TO")])
    assert graph.explain_path("user", "nonexistent") is None
    assert graph.explain_path("nonexistent", "peanut") is None


def test_relationship_strength_defaults_to_one():
    rel = Relationship(
        source_id="user", target_id="peanut", relation_type="ALLERGIC_TO", source_event_id="evt1"
    )
    assert rel.strength == 1.0


# --- reassign_relationships ---------------------------------------------------


def test_reassign_relationships_retargets_source_event_id():
    graph, events = make_graph_from_events([
        {"edges": [("user", "peanut", "ALLERGIC_TO")]},
    ])
    merged_event = MemoryEvent(content={"entities": [], "edges": []})
    graph.add(merged_event)

    survivors = graph.reassign_relationships([events[0].id], merged_event.id)

    assert len(survivors) == 1
    assert survivors[0].source_event_id == merged_event.id
    assert survivors[0].relation_type == "ALLERGIC_TO"


def test_reassign_relationships_preserves_relationships_unique_to_each_source():
    graph, events = make_graph_from_events([
        {"edges": [("user", "peanut", "ALLERGIC_TO")]},
        {"edges": [("user", "hiking", "ENJOYS")]},
    ])
    merged_event = MemoryEvent(content={"entities": [], "edges": []})
    graph.add(merged_event)

    survivors = graph.reassign_relationships([events[0].id, events[1].id], merged_event.id)

    relation_types = {(r.target_id, r.relation_type) for r in survivors}
    assert relation_types == {("peanut", "ALLERGIC_TO"), ("hiking", "ENJOYS")}
    assert all(r.source_event_id == merged_event.id for r in survivors)
    result = graph.related_to("user", max_hops=1)
    assert {e.id for e in result} == {"peanut", "hiking"}


def test_reassign_relationships_collapses_identical_relationships_keeping_max():
    graph, events = make_graph_from_events([
        {"edges": [("user", "peanut", "ALLERGIC_TO")]},
        {"edges": [("user", "peanut", "ALLERGIC_TO")]},
    ])
    graph._edges[0].confidence = 0.6
    graph._edges[0].strength = 0.4
    graph._edges[1].confidence = 0.9
    graph._edges[1].strength = 0.7
    merged_event = MemoryEvent(content={"entities": [], "edges": []})
    graph.add(merged_event)

    survivors = graph.reassign_relationships([events[0].id, events[1].id], merged_event.id)

    assert len(survivors) == 1
    assert survivors[0].confidence == 0.9
    assert survivors[0].strength == 0.7
    assert survivors[0].source_event_id == merged_event.id


def test_reassign_relationships_then_remove_does_not_delete_reassigned_edges():
    graph, events = make_graph_from_events([
        {"edges": [("user", "peanut", "ALLERGIC_TO")]},
        {"edges": [("user", "hiking", "ENJOYS")]},
    ])
    merged_event = MemoryEvent(content={"entities": [], "edges": []})
    graph.add(merged_event)
    graph.reassign_relationships([events[0].id, events[1].id], merged_event.id)

    graph.remove(events[0].id)
    graph.remove(events[1].id)

    result = graph.related_to("user", max_hops=1)
    assert {e.id for e in result} == {"peanut", "hiking"}


def test_reassign_relationships_collapses_with_pre_existing_on_new_event():
    """Bug fix: reassign_relationships must fold in pre-existing relationships
    already attached to new_event_id (e.g., from add() extraction) into the
    same collapsing pass, not create duplicates.

    Real scenario: add(merged_event) runs extraction which creates a fresh
    relationship on merged_event.id with one confidence/strength, then
    reassign_relationships() retargets an old relationship with different
    confidence/strength -- they should collapse into one with max values.
    """
    # Create old event with a relationship
    graph, events = make_graph_from_events([
        {"edges": [("user", "peanut", "ALLERGIC_TO")]},
    ])
    graph._edges[0].confidence = 0.6
    graph._edges[0].strength = 0.4
    old_event_id = events[0].id

    # Create merged event whose extraction ALSO creates the same relationship
    merged_event = MemoryEvent(content={"edges": [("user", "peanut", "ALLERGIC_TO")]})
    graph.add(merged_event)
    # Extraction created a fresh relationship on merged_event with default confidence/strength
    graph._edges[-1].confidence = 0.8
    graph._edges[-1].strength = 0.6

    # Now reassign the old event's relationship to the merged event
    survivors = graph.reassign_relationships([old_event_id], merged_event.id)

    # Should have exactly ONE relationship for (user -> peanut ALLERGIC_TO)
    # with max confidence/strength from both sources
    matching_rels = [r for r in graph._edges
                     if r.source_id == "user" and r.target_id == "peanut"
                     and r.relation_type == "ALLERGIC_TO"
                     and r.source_event_id == merged_event.id]
    assert len(matching_rels) == 1
    assert matching_rels[0].confidence == 0.8  # max(0.6, 0.8)
    assert matching_rels[0].strength == 0.6    # max(0.4, 0.6)


def test_entities_for_event_returns_source_and_target_ids():
    graph, events = make_graph_from_events([
        {"edges": [("user", "peanut", "ALLERGIC_TO")]},
    ])
    assert graph.entities_for_event(events[0].id) == {"user", "peanut"}


def test_entities_for_event_returns_empty_for_unknown_event():
    graph = make_graph(edges=[("user", "peanut", "ALLERGIC_TO")])
    assert graph.entities_for_event("nonexistent") == set()


def test_find_edge_returns_relationship_in_either_direction():
    graph = make_graph(edges=[("user", "peanut", "ALLERGIC_TO")])
    assert graph.find_edge("user", "peanut") is not None
    assert graph.find_edge("peanut", "user") is not None


def test_find_edge_returns_none_when_no_edge_exists():
    graph = make_graph(edges=[("user", "peanut", "ALLERGIC_TO")], entities=["hiking"])
    assert graph.find_edge("user", "hiking") is None


def test_find_edges_returns_all_parallel_relationships_between_a_pair():
    # the graph model permits parallel typed edges between the same two
    # entities -- reassign_relationships() groups by (source_id,
    # target_id, relation_type), so two different relation types both
    # survive between the same pair. find_edges() must surface both,
    # unlike find_edge() which only returns one of them.
    graph, events = make_graph_from_events([
        {"edges": [("peanut", "protein", "CONTAINS")]},
        {"edges": [("peanut", "protein", "INGREDIENT_OF")]},
    ])
    edges = graph.find_edges("peanut", "protein")
    assert len(edges) == 2
    assert {e.relation_type for e in edges} == {"CONTAINS", "INGREDIENT_OF"}
    # either direction still finds both
    assert len(graph.find_edges("protein", "peanut")) == 2


def test_find_edges_returns_empty_list_when_no_edge_exists():
    graph = make_graph(edges=[("user", "peanut", "ALLERGIC_TO")], entities=["hiking"])
    assert graph.find_edges("user", "hiking") == []


def test_find_edge_is_a_thin_wrapper_over_find_edges():
    graph, events = make_graph_from_events([
        {"edges": [("peanut", "protein", "CONTAINS")]},
        {"edges": [("peanut", "protein", "INGREDIENT_OF")]},
    ])
    assert graph.find_edge("peanut", "protein") in graph.find_edges("peanut", "protein")
