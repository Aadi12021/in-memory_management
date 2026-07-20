from memory_system.backends.graph import Entity, EntityExtractor, GraphBackend, Relationship
from memory_system.events import MemoryEvent, MemoryTier


class ScriptedExtractor(EntityExtractor):
    """Test double: bypasses the extraction problem entirely so graph
    traversal/structure logic can be tested against a graph shape the
    test controls exactly. event.content is a dict of the form
    {"entities": [id, ...], "edges": [(source_id, target_id, relation_type), ...]}.
    """

    def extract(self, event: MemoryEvent) -> tuple[list[Entity], list[Relationship]]:
        spec = event.content
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
