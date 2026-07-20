"""
Interface sketch: GraphBackend

DESIGN SKETCH -- not implementation. Goal: nail the interface before
committing to how entity/relationship extraction actually works
(that's the harder problem, tackled after this).

Core idea: instead of (or alongside) storing memories as flat text,
extract entities and relationships from each MemoryEvent and store
them as a graph. Enables multi-hop queries that pure similarity
search can't answer: "what should the user avoid at the party?"
requires traversing User -> ALLERGIC_TO -> Peanuts -> INGREDIENT_OF
-> Peanut Butter Cake, which is not a single similarity match.

This backend implements the existing MemoryBackend interface (so it's
swappable like any other backend) AND adds graph-specific query
methods on top, since "traverse relationships" and "find similar
text" are genuinely different query shapes.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Optional

from ..events import MemoryEvent, MemoryTier, RetrievalResult
from .base import MemoryBackend


# ---------------------------------------------------------------------------
# Graph primitives
# ---------------------------------------------------------------------------

@dataclass
class Entity:
    """A node in the memory graph. E.g. 'User', 'Peanuts', 'Hiking'."""
    id: str                    # normalized/canonical name, e.g. "peanuts"
    label: str                 # display name, e.g. "Peanuts"
    entity_type: str = "unknown"   # e.g. "food", "person", "activity"


@dataclass
class Relationship:
    """A directed, typed edge between two entities, sourced from a
    specific memory event so we can trace provenance and apply decay
    at the relationship level, not just the memory level.
    """
    source_id: str             # Entity.id
    target_id: str             # Entity.id
    relation_type: str         # e.g. "ALLERGIC_TO", "ENJOYS", "INGREDIENT_OF"
    source_event_id: str       # which MemoryEvent this came from
    confidence: float = 1.0    # extraction confidence, see extraction problem
    metadata: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------
# The extractor boundary -- deliberately abstract, decided separately
# ---------------------------------------------------------------------------

class EntityExtractor(ABC):
    """Turns raw memory content into entities + relationships.
    This is THE hard problem (tackled next) -- the interface just
    needs to accept an event and return a graph fragment.
    """

    @abstractmethod
    def extract(self, event: MemoryEvent) -> tuple[list[Entity], list[Relationship]]:
        ...


# ---------------------------------------------------------------------------
# The backend itself
# ---------------------------------------------------------------------------

class GraphBackend(MemoryBackend):
    """
    Dependency-free adjacency-list graph, no neo4j/graph DB required
    for v0.1. Implements the standard MemoryBackend interface (so
    store/consolidate/decay/retrieve all still work as usual) and adds
    graph-native traversal methods on top.
    """

    def __init__(self, extractor: EntityExtractor):
        self.extractor = extractor
        self._events: dict[str, MemoryEvent] = {}
        self._entities: dict[str, Entity] = {}
        self._edges: list[Relationship] = []
        # adjacency index for fast traversal: entity_id -> [Relationship, ...]
        self._adjacency: dict[str, list[Relationship]] = {}

    # --- standard MemoryBackend interface -----------------------------

    def add(self, event: MemoryEvent) -> None:
        """Stores the event AND runs extraction to update the graph."""
        self._events[event.id] = event
        entities, relationships = self.extractor.extract(event)
        for entity in entities:
            self._entities[entity.id] = entity
        for rel in relationships:
            self._edges.append(rel)
            self._adjacency.setdefault(rel.source_id, []).append(rel)

    def get_all(self, tier: Optional[MemoryTier] = None) -> list[MemoryEvent]:
        events = list(self._events.values())
        if tier is not None:
            events = [e for e in events if e.tier == tier]
        return events

    def query(
        self, query: str, top_k: int = 5, tier: Optional[MemoryTier] = None
    ) -> list[RetrievalResult]:
        """Falls back to matching events whose linked entities overlap
        with entities mentioned in the query. Real text similarity
        still belongs to InMemoryBackend/ChromaBackend -- this is
        entity-overlap matching, a different (complementary) signal.
        """
        ...

    def update_tier(self, event_id: str, new_tier: MemoryTier) -> None:
        if event_id in self._events:
            self._events[event_id].tier = new_tier

    def remove(self, event_id: str) -> None:
        self._events.pop(event_id, None)
        self._edges[:] = [e for e in self._edges if e.source_event_id != event_id]

    # --- graph-native methods, the actual point of this backend -------

    def related_to(
        self, entity_id: str, relation_type: Optional[str] = None, max_hops: int = 1
    ) -> list[Entity]:
        """Traverse the graph from an entity, optionally filtered by
        relationship type. This is what answers "what foods should the
        user avoid" -- a query no similarity search can answer directly.

        BFS over outgoing edges via the adjacency index. `relation_type`,
        when given, restricts which edges are followed at every hop (not
        just the first). Cycles are handled by never revisiting a node,
        so `max_hops` bounds the search even on cyclic graphs.
        """
        if entity_id not in self._entities:
            return []

        visited = {entity_id}
        frontier = {entity_id}
        found_ids: list[str] = []

        for _ in range(max_hops):
            next_frontier: set[str] = set()
            for node_id in frontier:
                for rel in self._adjacency.get(node_id, []):
                    if relation_type is not None and rel.relation_type != relation_type:
                        continue
                    if rel.target_id in visited:
                        continue
                    visited.add(rel.target_id)
                    next_frontier.add(rel.target_id)
                    found_ids.append(rel.target_id)
            frontier = next_frontier
            if not frontier:
                break

        return [self._entities[eid] for eid in found_ids if eid in self._entities]

    def explain_path(self, source_id: str, target_id: str) -> Optional[list[Relationship]]:
        """Returns the chain of relationships connecting two entities,
        if any -- useful for showing *why* something was surfaced,
        not just that it was.
        """
        ...

    def consolidation_signal(self, entity_id: str) -> float:
        """Structural signal for consolidation policies: entities with
        many connections are arguably more worth promoting to
        long-term memory than isolated ones. A GraphAwareConsolidation
        policy could use this alongside/instead of salience scores.

        Degree (edges touching the entity, either as source or target)
        mapped through degree / (degree + 1) so the signal is bounded
        to [0, 1) and comparable to salience scores -- 0 for isolated
        entities, asymptotically approaching 1 as connections grow,
        with no need to know the graph's max degree up front.
        """
        degree = sum(
            1 for rel in self._edges if rel.source_id == entity_id or rel.target_id == entity_id
        )
        return degree / (degree + 1) if degree else 0.0


# ---------------------------------------------------------------------------
# What using this should feel like
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    graph = GraphBackend(extractor=None)  # e.g. LLMEntityExtractor(...)

    graph.add(MemoryEvent(content="User is allergic to peanuts."))
    graph.add(MemoryEvent(content="Mom's birthday cake has peanut butter frosting."))

    # multi-hop: this is the query a flat vector/lexical store can't do
    risky_foods = graph.related_to("user", relation_type="ALLERGIC_TO", max_hops=2)
