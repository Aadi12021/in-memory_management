"""The public-facing entry point: TieredMemory."""

from __future__ import annotations

import logging
from datetime import datetime, timezone
from typing import Any, Optional

from .backends.base import MemoryBackend
from .backends.graph import GraphBackend
from .backends.hybrid import HybridBackend
from .summarization.base import MemorySummarizer
from .events import ConsolidationReport, MemoryEvent, MemoryTier, RetrievalResult
from .policies.consolidation import ConsolidationPolicy
from .policies.decay import DecayPolicy
from .policies.salience import SalienceScorer

logger = logging.getLogger(__name__)


def _find_similar_pairs(
    backend: MemoryBackend, tier: MemoryTier, threshold: float
) -> list[tuple[MemoryEvent, MemoryEvent, float]]:
    """Finds pairs of events in `tier` whose similarity (via the
    backend's own query()) is at or above `threshold`. Excludes
    self-matches explicitly: every backend's query() will return the
    querying event's own content back to itself, usually at rank 1
    with a near-perfect score, since none of them exclude the query
    source. Without this check, every event would trivially "match
    itself" above any threshold.
    """
    candidates = backend.get_all(tier=tier)
    seen_pairs: set[frozenset[str]] = set()
    pairs: list[tuple[MemoryEvent, MemoryEvent, float]] = []
    for event in candidates:
        results = backend.query(str(event.content), top_k=len(candidates), tier=tier)
        for result in results:
            if result.event.id == event.id:
                continue
            if result.score < threshold:
                continue
            pair_key = frozenset({event.id, result.event.id})
            if pair_key in seen_pairs:
                continue
            seen_pairs.add(pair_key)
            pairs.append((event, result.event, result.score))
    return pairs


def _find_graph_backend(backend: MemoryBackend) -> Optional[GraphBackend]:
    """Finds the GraphBackend relevant to graph-specific consolidation
    steps: `backend` itself if it is one, or whichever of a
    HybridBackend's two composed backends is one. Returns None for any
    other backend, which callers treat as "nothing to do here," not an
    error -- most backends have no graph structure at all.
    """
    if isinstance(backend, GraphBackend):
        return backend
    if isinstance(backend, HybridBackend):
        for sub in (backend.lexical_backend, backend.semantic_backend):
            if isinstance(sub, GraphBackend):
                return sub
    return None


def _connected_components(
    pairs: list[tuple[MemoryEvent, MemoryEvent, float]]
) -> list[list[MemoryEvent]]:
    """Groups events into connected components over the similarity
    pairs found by _find_similar_pairs -- e.g. if (a, b) and (b, c) are
    both above threshold, a/b/c end up in one group even though a and
    c were never compared directly.
    """
    adjacency: dict[str, set[str]] = {}
    events_by_id: dict[str, MemoryEvent] = {}
    for a, b, _score in pairs:
        events_by_id[a.id] = a
        events_by_id[b.id] = b
        adjacency.setdefault(a.id, set()).add(b.id)
        adjacency.setdefault(b.id, set()).add(a.id)

    visited: set[str] = set()
    components: list[list[MemoryEvent]] = []
    for event_id in adjacency:
        if event_id in visited:
            continue
        component_ids: list[str] = []
        frontier = [event_id]
        visited.add(event_id)
        while frontier:
            current = frontier.pop()
            component_ids.append(current)
            for neighbor in adjacency.get(current, ()):
                if neighbor not in visited:
                    visited.add(neighbor)
                    frontier.append(neighbor)
        components.append([events_by_id[eid] for eid in component_ids])

    return components


class TieredMemory:
    """Composes a backend with consolidation/decay/salience policies.

    Typical usage:
        memory = TieredMemory(
            backend=InMemoryBackend(),
            consolidation_policy=SurpriseBasedConsolidation(threshold=0.7),
            decay_policy=ForgettingCurveDecay(half_life_days=14),
        )
        memory.store("User is allergic to peanuts.")
        memory.consolidate()
        results = memory.retrieve("dietary restrictions")
    """

    def __init__(
        self,
        backend: MemoryBackend,
        consolidation_policy: ConsolidationPolicy,
        decay_policy: DecayPolicy,
        salience_scorer: Optional[SalienceScorer] = None,
        forget_floor: float = 0.05,
    ):
        self.backend = backend
        self.consolidation_policy = consolidation_policy
        self.decay_policy = decay_policy
        self.salience_scorer = salience_scorer
        self.forget_floor = forget_floor

    def store(self, content: Any, metadata: Optional[dict] = None) -> MemoryEvent:
        """Ingest new content at the working tier."""
        event = MemoryEvent(
            content=content,
            tier=MemoryTier.WORKING,
            metadata=metadata or {},
        )
        if self.salience_scorer is not None:
            event.salience = self.salience_scorer.score(event)
        self.backend.add(event)
        return event

    def consolidate(self) -> int:
        """Promote eligible working-tier memories to long-term. Returns
        the number of events promoted.
        """
        promoted = 0
        for event in self.backend.get_all(tier=MemoryTier.WORKING):
            if self.consolidation_policy.should_consolidate(event):
                self.backend.update_tier(event.id, MemoryTier.LONG_TERM)
                promoted += 1
        return promoted

    def deduplicate(self, threshold: float, dry_run: bool = False) -> ConsolidationReport:
        """Merges near-duplicate long-term memories. `threshold` is
        required, not defaulted: InMemoryBackend's TF-IDF cosine,
        ChromaBackend's 1/(1+distance), HybridBackend's RRF-fused
        scores, and GraphBackend's unbounded entity-overlap counts are
        all on different scales, so no single default threshold is
        meaningful across backends -- see docs/superpowers/specs/
        2026-08-06-offline-consolidation.md.
        """
        pairs = _find_similar_pairs(self.backend, MemoryTier.LONG_TERM, threshold)
        graph_backend = _find_graph_backend(self.backend)
        now = datetime.now(timezone.utc)

        report = ConsolidationReport()
        merged_event_ids: set[str] = set()

        for a, b, _score in pairs:
            if a.id in merged_event_ids or b.id in merged_event_ids:
                continue  # already absorbed by an earlier pair in this pass

            new_id = None
            if not dry_run:
                keep, discard = (a, b) if a.salience >= b.salience else (b, a)
                merged_event = MemoryEvent(
                    content=keep.content,
                    timestamp=min(a.timestamp, b.timestamp),
                    tier=MemoryTier.LONG_TERM,
                    salience=max(a.salience, b.salience),
                    metadata={**discard.metadata, **keep.metadata, "merged_from": [a.id, b.id]},
                    last_reinforced=now,
                )
                self.backend.add(merged_event)
                if graph_backend is not None:
                    graph_backend.reassign_relationships([a.id, b.id], merged_event.id)
                self.backend.remove(a.id)
                self.backend.remove(b.id)
                new_id = merged_event.id

            report.merged.append((a.id, b.id, new_id))
            merged_event_ids.add(a.id)
            merged_event_ids.add(b.id)

        return report

    def compress(
        self, threshold: float, summarizer: MemorySummarizer, dry_run: bool = False
    ) -> ConsolidationReport:
        """Groups related long-term memories and replaces each group
        with one LLM-generated summary. `threshold` is required for
        the same reason deduplicate()'s is -- no universal default
        across backends with incompatible score scales.
        """
        pairs = _find_similar_pairs(self.backend, MemoryTier.LONG_TERM, threshold)
        graph_backend = _find_graph_backend(self.backend)
        now = datetime.now(timezone.utc)

        groups = _connected_components(pairs)

        report = ConsolidationReport()
        for group_events in groups:
            if len(group_events) < 2:
                continue

            new_id = None
            if not dry_run:
                try:
                    summary_text = summarizer.summarize(group_events)
                except Exception:
                    # fail soft: a broken/unavailable LLM call shouldn't
                    # abort the whole pass, it should just mean this
                    # group is skipped -- but log it, so a real failure
                    # is visible instead of indistinguishable from "no
                    # group was ever eligible here."
                    logger.warning(
                        "compress(): summarizer.summarize() failed for group %s, "
                        "skipping this group",
                        [e.id for e in group_events],
                        exc_info=True,
                    )
                    continue
                summary_event = MemoryEvent(
                    content=summary_text,
                    timestamp=min(e.timestamp for e in group_events),
                    tier=MemoryTier.LONG_TERM,
                    salience=max(e.salience for e in group_events),
                    metadata={"summarized_from": [e.id for e in group_events]},
                    last_reinforced=now,
                )
                self.backend.add(summary_event)
                if graph_backend is not None:
                    graph_backend.reassign_relationships(
                        [e.id for e in group_events], summary_event.id
                    )
                for event in group_events:
                    self.backend.remove(event.id)
                new_id = summary_event.id

            report.compressed.append(([e.id for e in group_events], new_id))

        return report

    def strengthen_connections(
        self,
        merge_report: Optional[ConsolidationReport] = None,
        compress_report: Optional[ConsolidationReport] = None,
        dry_run: bool = False,
    ) -> ConsolidationReport:
        """Strengthens PRE-EXISTING "bystander" graph edges between
        entities that a deduplicate()/compress() pass just associated
        with the same surviving event -- NOT the connections that pass
        itself created.

        Concretely: for every merged/summary event id in the given
        reports, this looks at all entities touched by that event
        (`entities_for_event`) and, for every pair of them that has an
        existing relationship (`find_edges`), bumps that relationship's
        `strength` by 0.1 (capped at 1.0) -- UNLESS that relationship's
        `source_event_id` is the merge/summary event itself, in which
        case it is explicitly skipped. So if this pass is what created
        or reassigned the (entity_a, entity_b) edge in the first place,
        that edge's own strength/confidence is left alone here (it was
        already handled by reassign_relationships()'s max()-collapsing
        during the same pass); only an independently pre-existing edge
        between the two entities -- one whose source_event_id points at
        some other, unrelated event -- gets strengthened. A caller
        reading only this docstring should not expect the pass's own
        new connections to come out of it strengthened; they won't.

        Pass the ConsolidationReports deduplicate()/compress() returned
        (from the same pass); called with no reports (the default),
        there is nothing to strengthen and an empty ConsolidationReport
        is returned.

        Only meaningful for GraphBackend (or a HybridBackend composed
        with one) -- returns an empty ConsolidationReport immediately
        for any other backend, which is not an error, just "nothing
        to strengthen here."
        """
        graph_backend = _find_graph_backend(self.backend)
        if graph_backend is None:
            return ConsolidationReport()

        new_ids: list[str] = []
        if merge_report is not None:
            new_ids += [new_id for _a, _b, new_id in merge_report.merged if new_id is not None]
        if compress_report is not None:
            new_ids += [new_id for _sources, new_id in compress_report.compressed if new_id is not None]

        report = ConsolidationReport()
        for event_id in new_ids:
            entities = sorted(graph_backend.entities_for_event(event_id))
            for i, entity_a in enumerate(entities):
                for entity_b in entities[i + 1:]:
                    # A pair of entities can have more than one
                    # relationship between them (different relation
                    # types), and find_edge() would arbitrarily return
                    # just one -- iterate all of them so a genuine
                    # bystander edge isn't skipped just because a
                    # consolidation-sourced edge for the same pair
                    # happens to come first.
                    for edge in graph_backend.find_edges(entity_a, entity_b):
                        # Only strengthen edges that are NOT part of this consolidation event itself
                        # (those edges' confidence/strength are already handled by the merge/compress).
                        # Strengthen only "external" edges that happen to connect co-mentioned entities.
                        if edge.source_event_id == event_id:
                            continue
                        if not dry_run:
                            edge.strength = min(1.0, edge.strength + 0.1)
                        report.strengthened.append((entity_a, entity_b))

        return report

    def offline_consolidate(
        self,
        merge_threshold: float,
        group_threshold: float,
        summarizer: Optional[MemorySummarizer] = None,
        dry_run: bool = False,
    ) -> ConsolidationReport:
        """Runs deduplicate() -> compress() -> strengthen_connections(),
        in that fixed order, not configurable. GraphBackend.remove()
        already prunes edges for removed events as a side effect, so
        running strengthen_connections() last is the only ordering
        that doesn't waste work against code that already exists --
        see docs/superpowers/specs/2026-08-06-offline-consolidation.md.
        """
        merge_report = self.deduplicate(merge_threshold, dry_run=dry_run)
        compress_report = (
            self.compress(group_threshold, summarizer, dry_run=dry_run)
            if summarizer is not None
            else ConsolidationReport()
        )
        strengthen_report = self.strengthen_connections(merge_report, compress_report, dry_run=dry_run)
        return ConsolidationReport(
            merged=merge_report.merged,
            compressed=compress_report.compressed,
            strengthened=strengthen_report.strengthened,
        )

    def decay(self, now: Optional[datetime] = None) -> int:
        """Run a decay pass, removing events whose strength has fallen
        below the forget floor. Returns the number of events forgotten.
        """
        now = now or datetime.now(timezone.utc)
        forgotten = 0
        for event in self.backend.get_all():
            if self.decay_policy.should_forget(event, now, floor=self.forget_floor):
                self.backend.remove(event.id)
                forgotten += 1
        return forgotten

    def retrieve(
        self, query: str, top_k: int = 5, tier: Optional[MemoryTier] = None
    ) -> list[RetrievalResult]:
        """Query memory, weighting relevance by current decay strength."""
        raw_results = self.backend.query(query, top_k=top_k * 2, tier=tier)

        weighted: list[RetrievalResult] = []
        for result in raw_results:
            strength = self.decay_policy.current_strength(result.event)
            weighted.append(
                RetrievalResult(
                    event=result.event,
                    score=result.score * strength,
                    tier=result.tier,
                )
            )
            result.event.reinforce()  # accessing a memory reinforces it

        weighted.sort(key=lambda r: r.score, reverse=True)
        return weighted[:top_k]
