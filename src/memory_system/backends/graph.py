"""
GraphBackend: adjacency-list graph storage for MemoryEvents.

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
from collections import deque
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
    strength: float = 1.0      # consolidation-reinforced importance -- distinct from
                                # confidence, which is about extraction certainty, not
                                # how reinforced this connection has become over time
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

        "Mentioned in the query" means a known entity id appears as a
        substring of the (lowercased) query text -- simple and
        dependency-free, matching this backend's v0.1 scope. Score is
        the size of the overlap between those entities and the ones
        linked to each event (via source_event_id on its edges).
        """
        normalized_query = query.lower()
        matched_entity_ids = {eid for eid in self._entities if eid in normalized_query}
        if not matched_entity_ids:
            return []

        event_entity_ids: dict[str, set[str]] = {}
        for rel in self._edges:
            ids = event_entity_ids.setdefault(rel.source_event_id, set())
            ids.add(rel.source_id)
            ids.add(rel.target_id)

        results: list[RetrievalResult] = []
        for event_id, entity_ids in event_entity_ids.items():
            event = self._events.get(event_id)
            if event is None:
                continue
            if tier is not None and event.tier != tier:
                continue
            overlap = matched_entity_ids & entity_ids
            if not overlap:
                continue
            results.append(RetrievalResult(event=event, score=float(len(overlap)), tier=event.tier))

        results.sort(key=lambda r: r.score, reverse=True)
        return results[:top_k]

    def update_tier(self, event_id: str, new_tier: MemoryTier) -> None:
        if event_id in self._events:
            self._events[event_id].tier = new_tier

    def remove(self, event_id: str) -> None:
        self._events.pop(event_id, None)
        self._edges[:] = [e for e in self._edges if e.source_event_id != event_id]

    def reassign_relationships(self, old_event_ids: list[str], new_event_id: str) -> list[Relationship]:
        """Retargets every relationship whose source_event_id is in
        old_event_ids to point at new_event_id instead, preserving the
        relationships themselves rather than re-deriving them via
        extraction from a single surviving content string, which would
        silently drop anything only present in a discarded event's
        original phrasing (see docs/superpowers/specs/
        2026-08-06-offline-consolidation.md).

        Also folds in any pre-existing relationships already attached to
        new_event_id into the same collapsing pass, so that if add()
        extraction re-creates a matching relationship on new_event_id
        before this call, it participates in max()-collapsing rather than
        creating a duplicate.

        Relationships that become identical after reassignment (same
        source_id, target_id, relation_type, now all pointing at
        new_event_id) collapse into one, keeping max(confidence) and
        max(strength) across the collapsed set. Mutates matching
        Relationship objects in place (self._edges and self._adjacency
        hold references to the same objects, so no separate adjacency
        update is needed for survivors); removed duplicates are
        filtered out of both by object identity, not value equality,
        since dataclass equality could otherwise match the wrong
        object when two relationships happen to share all field
        values.

        Returns the relationships now attached to new_event_id.
        """
        old_ids = set(old_event_ids)
        affected = [
            rel for rel in self._edges
            if rel.source_event_id in old_ids or rel.source_event_id == new_event_id
        ]

        groups: dict[tuple[str, str, str], list[Relationship]] = {}
        for rel in affected:
            key = (rel.source_id, rel.target_id, rel.relation_type)
            groups.setdefault(key, []).append(rel)

        redundant_object_ids: set[int] = set()
        survivors: list[Relationship] = []
        for group in groups.values():
            survivor = group[0]
            survivor.source_event_id = new_event_id
            if len(group) > 1:
                survivor.confidence = max(rel.confidence for rel in group)
                survivor.strength = max(rel.strength for rel in group)
                redundant_object_ids.update(id(rel) for rel in group[1:])
            survivors.append(survivor)

        if redundant_object_ids:
            self._edges[:] = [rel for rel in self._edges if id(rel) not in redundant_object_ids]
            for entity_id in self._adjacency:
                self._adjacency[entity_id] = [
                    rel for rel in self._adjacency[entity_id] if id(rel) not in redundant_object_ids
                ]

        return survivors

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

        BFS over the adjacency index (same traversal shape as
        related_to) so the result is the shortest chain when more than
        one path exists. Returns [] when source and target are the
        same known entity (trivially connected, zero hops needed), and
        None when either entity is unknown or no path connects them.
        """
        if source_id not in self._entities or target_id not in self._entities:
            return None
        if source_id == target_id:
            return []

        visited = {source_id}
        queue = deque([source_id])
        incoming: dict[str, Relationship] = {}

        while queue:
            node_id = queue.popleft()
            for rel in self._adjacency.get(node_id, []):
                if rel.target_id in visited:
                    continue
                visited.add(rel.target_id)
                incoming[rel.target_id] = rel
                if rel.target_id == target_id:
                    queue.clear()
                    break
                queue.append(rel.target_id)

        if target_id not in incoming:
            return None

        path: list[Relationship] = []
        node_id = target_id
        while node_id != source_id:
            rel = incoming[node_id]
            path.append(rel)
            node_id = rel.source_id
        path.reverse()
        return path

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
    from ..extraction.rules_based import RuleBasedEntityExtractor

    graph = GraphBackend(extractor=RuleBasedEntityExtractor())

    graph.add(MemoryEvent(content="User is allergic to peanuts."))
    graph.add(MemoryEvent(content="Peanuts contains protein."))

    # multi-hop: this is the query a flat vector/lexical store can't do.
    # No relation_type filter here, since the real chain crosses types:
    # user -ALLERGIC_TO-> peanut -CONTAINS-> protein.
    related = graph.related_to("user", max_hops=2)
    print([entity.id for entity in related])
