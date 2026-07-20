from memory_system.backends.graph_sketch import Entity, EntityExtractor, GraphBackend, Relationship
from memory_system.events import MemoryEvent


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
