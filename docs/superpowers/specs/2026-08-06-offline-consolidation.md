# Offline consolidation design

Date: 2026-08-06
Status: approved, not yet implemented

## Problem

`TieredMemory.consolidate()` is one-way and one-time: it promotes
eligible working-tier events to long-term, once, and never revisits
long-term memory again. Nothing in tiered-memory today reorganizes
long-term memory after the fact — near-duplicate facts accumulate
side by side forever, related memories never get condensed, and
`GraphBackend`'s relationships never strengthen based on what turned
out to matter. This adds `offline_consolidate()` (plus three
independently-callable sub-methods it composes): a manual,
caller-triggered batch pass over long-term memory that merges
near-duplicates, compresses related groups into summaries, and
strengthens graph connections discovered during the pass -- modeled
on sleep-dependent memory consolidation, but implemented as an
ordinary method, not a scheduled or automatic process.

## Grounded against the actual current code, not assumed

Read `core.py`, `events.py`, `backends/base.py`, `backends/graph.py`
directly before designing anything. Three facts from that reading
shape every decision below:

- **`MemoryBackend` has no content-update method** -- only
  `add`/`get_all`/`query`/`update_tier`/`remove`. Merge and compress
  can only be built from `remove()` + `add()`. This turns out to be
  an advantage: that composition is backend-agnostic and requires
  zero new interface methods, working identically across all four
  backends.
- **No backend's `query()` excludes the query source from its own
  results.** `InMemoryBackend`, `ChromaBackend`, and `GraphBackend`
  all rank the full corpus against the query text with no
  "don't return what I'm searching with" logic. Querying with an
  event's own content will return that event itself, usually at rank
  1 with a near-perfect score.
- **`GraphBackend.remove(event_id)` already deletes every edge whose
  `source_event_id` matches** (`self._edges[:] = [e for e in
  self._edges if e.source_event_id != event_id]`). This is existing
  behavior, not something this design adds -- and it directly
  determines the required execution order (see below).
- **`MemoryEvent` has no co-retrieval tracking.** `retrieve()` calls
  `event.reinforce()` per event but never records which events were
  retrieved *together*. "Strengthen connections between frequently
  co-retrieved memories," taken literally, needs tracking state that
  doesn't exist anywhere in tiered-memory today.
- **`GraphBackend.Relationship.confidence`** already exists, but
  means "how sure was the extractor this relationship is real" --
  a different concept from "how reinforced is this connection by
  consolidation."

## Interface

Four new public methods on `TieredMemory`, added to `core.py`. Three
independently callable (matching how `consolidate()`/`decay()` are
already separate, composable primitives, not bundled into one
mega-method), plus a convenience wrapper that runs all three in a
fixed order:

```python
def deduplicate(self, threshold: float, dry_run: bool = False) -> ConsolidationReport: ...
def compress(self, threshold: float, summarizer: MemorySummarizer, dry_run: bool = False) -> ConsolidationReport: ...
def strengthen_connections(self, dry_run: bool = False) -> ConsolidationReport: ...

def offline_consolidate(
    self,
    merge_threshold: float,
    group_threshold: float,
    summarizer: Optional[MemorySummarizer] = None,
    dry_run: bool = False,
) -> ConsolidationReport:
    """Runs deduplicate() -> compress() -> strengthen_connections(),
    in that fixed order, not configurable. See "Execution order" below
    for why this order is load-bearing, not a style choice.
    """
    merge_report = self.deduplicate(merge_threshold, dry_run=dry_run)
    compress_report = (
        self.compress(group_threshold, summarizer, dry_run=dry_run)
        if summarizer is not None
        else ConsolidationReport()
    )
    strengthen_report = self.strengthen_connections(dry_run=dry_run)
    return ConsolidationReport(
        merged=merge_report.merged,
        compressed=compress_report.compressed,
        strengthened=strengthen_report.strengthened,
    )
```

**Naming:** `offline_consolidate()`, not `sleep()`. Both were
considered. `store`/`consolidate`/`decay`/`retrieve` are all literal,
mechanism-naming verbs; `sleep()` would be the only metaphorical name
on the class, and risks reading as "pause execution" (echoing
`time.sleep`) rather than "reorganize memory." `offline_consolidate()`
keeps the same descriptive register as its siblings.

**`ConsolidationReport`** (new dataclass, `events.py`, alongside
`RetrievalResult`) replaces the bare `int` that `consolidate()`/
`decay()` return -- this pass does three qualitatively different
things, so a count alone hides what actually happened:

```python
@dataclass
class ConsolidationReport:
    merged: list[tuple[str, str, str]] = field(default_factory=list)       # (source_a_id, source_b_id, new_id)
    compressed: list[tuple[list[str], str]] = field(default_factory=list)  # (source_ids, new_summary_id)
    strengthened: list[tuple[str, str]] = field(default_factory=list)      # (source_id, target_id) entity pairs
```

Every method returns this same shape whether `dry_run=True` or not --
a dry run answers "what would happen," a real run answers "what did
happen," same fields either way.

## Mechanism 1: Deduplication / merging

**Similarity via the backend's own `query()`** -- no new backend
methods. For each long-term event, query with its own content and
look at what comes back. `deduplicate()` and `compress()` always call
this with `tier=MemoryTier.LONG_TERM` hardcoded -- `tier` is a
parameter on the helper for testability (so a test can pass a smaller
synthetic tier), not something `offline_consolidate()` exposes to
callers; this whole feature is scoped to reorganizing long-term
memory, matching the "revisits long-term memory" framing throughout:

```python
def _find_similar_pairs(backend, tier, threshold) -> list[tuple[MemoryEvent, MemoryEvent, float]]:
    candidates = backend.get_all(tier=tier)
    seen_pairs: set[frozenset[str]] = set()
    pairs = []
    for event in candidates:
        results = backend.query(str(event.content), top_k=len(candidates), tier=tier)
        for result in results:
            # Self-matching: every backend's query() will return the
            # querying event's own content back to itself, usually at
            # rank 1 with a near-perfect score, since none of them
            # exclude the query source. Without this check, every
            # event trivially "matches itself" above any threshold
            # and deduplicate() would try to merge every event with
            # itself first.
            if result.event.id == event.id:
                continue
            if result.score < threshold:
                continue
            pair_key = frozenset({event.id, result.event.id})
            if pair_key in seen_pairs:  # (a, b) and (b, a) are the same pair
                continue
            seen_pairs.add(pair_key)
            pairs.append((event, result.event, result.score))
    return pairs
```

**No universal default `threshold`.** TF-IDF cosine (`InMemoryBackend`),
`1/(1+distance)` (`ChromaBackend`), tiny RRF-fused scores
(`HybridBackend`), and unbounded entity-overlap counts (`GraphBackend`)
are on different scales -- the same incompatibility already
established in the hybrid-retrieval design. `threshold` is therefore
a **required** argument on `deduplicate()`/`offline_consolidate()`,
never defaulted, forcing a conscious per-backend choice instead of a
silently-meaningless default.

**Merge action**, for each pair above threshold: `remove()` both
originals, `add()` one new event.

- **`content`**: the higher-salience source's content, unmodified --
  not synthesized. Merge is content-*preserving* (pick a
  representative); compression is content-*transforming*
  (synthesize). Conflating them would make merge silently depend on
  an LLM it doesn't need.
- **`salience`**: `max(a.salience, b.salience)` -- preserves "at
  least as important as its most important source." Sum would double
  count; average would let a strong fact get diluted by a weak
  duplicate.
- **`last_reinforced`**: set to `now` (the consolidation pass time).
  Consolidation is itself a reinforcement event -- this resets the
  decay clock, consistent with `ForgettingCurveDecay` treating
  `last_reinforced` as "last touched."
- **`timestamp`**: the *earliest* source's original timestamp --
  preserves "when this was first learned" rather than making a fact
  look newly-discovered.
- **`metadata`**: shallow union of both sources' metadata (survivor's
  values win on key conflicts), plus `metadata["merged_from"] =
  [a.id, b.id]` -- the only provenance trail v1 keeps (see Safety).

**`GraphBackend`-specific consequence, worth stating plainly.**
`GraphBackend.add()` re-runs entity extraction on whatever content
it's given. Since a merged event keeps only one source's content
verbatim (not a combination), any entities/relationships that were
unique to the *discarded* source's text are not re-extracted and
effectively drop out of the graph -- they were never in the surviving
content for the extractor to find again. This is an accepted
consequence of "merge picks a representative, it doesn't synthesize"
(re-affirmed above), not an oversight, but it's the kind of thing that
looks like a bug if it's only discovered during implementation rather
than named here.

## Mechanism 2: Compression / summarization

**Requires an LLM** -- new `MemorySummarizer` ABC, mirroring
`EntityExtractor`'s exact shape:

```python
class MemorySummarizer(ABC):
    @abstractmethod
    def summarize(self, events: list[MemoryEvent]) -> str: ...
```

Concrete `LLMSummarizer` reuses the **existing** `llm` extra (same
`anthropic` dependency `LLMEntityExtractor` already needs -- no new
optional-dependency permutation). Same resilience posture as
`LLMEntityExtractor`: fail soft, not hard -- an API error skips that
group (logged, not raised) rather than aborting the whole pass.
`summarizer` is `Optional` on `offline_consolidate()`; when `None`,
`compress()` is skipped entirely (mirrors `TieredMemory.__init__`'s
existing `salience_scorer: Optional[SalienceScorer] = None` pattern
of "absent optional component means skip that feature").

**Grouping** reuses the same `_find_similar_pairs()` helper as dedup,
at a separate, looser `group_threshold` (compression groups don't
need to be near-duplicates, just related enough to be worth
condensing). Groups are connected components over the
above-threshold pairs (simple BFS/union-find over the pair graph).
Runs strictly after `deduplicate()` (see Execution order) so
compression never has to re-handle near-identical pairs dedup already
resolved -- it only ever sees the already-deduplicated remainder.

**Compress action**, for each group (size >= 2): `remove()` every
source, `add()` one new summary event.

- `content` = `summarizer.summarize(group_events)`.
- `salience` = `max(salience for event in group)`.
- `last_reinforced` = `now`; `timestamp` = earliest in group -- same
  reasoning as merge.
- `metadata["summarized_from"] = [event.id for event in group]`.

## Mechanism 3: Connection strengthening

**Scoped exactly to where it's meaningful, not applied universally.**
Only `GraphBackend` has connections to strengthen at all;
`InMemoryBackend`/`ChromaBackend` have no relationship structure.
`HybridBackend` is in scope only when one of its two composed
backends happens to be a `GraphBackend`:

```python
def strengthen_connections(self, dry_run: bool = False) -> ConsolidationReport:
    graph_backend = self._find_graph_backend(self.backend)
    if graph_backend is None:
        return ConsolidationReport()  # not an error -- this backend has nothing to strengthen
    ...

def _find_graph_backend(backend: MemoryBackend) -> Optional[GraphBackend]:
    if isinstance(backend, GraphBackend):
        return backend
    if isinstance(backend, HybridBackend):
        for sub in (backend.lexical_backend, backend.semantic_backend):
            if isinstance(sub, GraphBackend):
                return sub
    return None
```

This `isinstance`-based reach-through isn't a new pattern -- the
README already documents that `GraphBackend`'s graph-native methods
are "only reachable by calling the backend directly," i.e.
graph-specific behavior already requires backend-type-aware code
elsewhere in the library. This is the same shape of exception,
applied consistently.

**New field, not overloaded reuse.** `Relationship.confidence` means
extraction confidence; strengthening means something else
(consolidation-reinforced importance). Reusing `confidence` would
conflate two different concepts in one number. Adds
`Relationship.strength: float = 1.0` to the `Relationship` dataclass
in `backends/graph.py` -- backward compatible (new field, has a
default, no existing construction site breaks).

**What "strengthen" means in v1, and what's explicitly deferred.**
"Frequently co-retrieved" as literally stated needs retrieval
co-occurrence tracking that doesn't exist anywhere in tiered-memory
today (`retrieve()` reinforces events individually, never records
which events came back *together*). Building that tracking is a
separate feature in its own right -- a new question of where the
state lives (`TieredMemory`? a running counter on `MemoryEvent`?) and
its own design pass, not a detail of this one. **v1 strengthens
connections between entities that were found related by *this same
consolidation pass*** -- i.e., entities mentioned in events that
`deduplicate()` or `compress()` just merged or grouped together. For
every merged/compressed group, for every pair of entities in
`graph_backend._entities` mentioned across that group's source
events: if an edge already connects them, bump its `strength` by a
fixed increment (`+0.1`), capped at `1.0` (matching `confidence`'s
own typical range, so the two fields stay comparable even though they
mean different things). **Pairs with no existing edge are skipped,
not used to create a new one** -- consolidation has no principled way
to assign a `relation_type` to a brand-new edge (that's
`EntityExtractor`'s job, working from the original text, which
consolidation doesn't re-run), so inventing untyped edges from bare
co-occurrence is out of scope for v1. True co-retrieval-frequency
strengthening is a deferred v2, not part of this spec.

## Execution order: dedup -> compress -> strengthen, hardcoded

Not a style preference -- the only ordering compatible with code that
already exists. `GraphBackend.remove(event_id)` already deletes every
edge whose `source_event_id` matches the removed event, as a side
effect of `remove()` itself (`backends/graph.py`, current behavior,
not something this design adds). `deduplicate()` and `compress()`
both call `remove()` on their source events. So by the time
`strengthen_connections()` runs, if it runs *last*, the graph has
already had every edge belonging to merged-away or summarized-away
events pruned automatically -- `strengthen_connections()` is
mechanically guaranteed to only ever see edges belonging to events
that survived this pass. Running it first, or interleaved, would let
it strengthen edges moments before `remove()` deletes them -- wasted
work at best, strengthening a relationship that's about to vanish.
`offline_consolidate()` calls the three methods in this fixed order
in its body; the order is not a parameter and is not configurable.

## Safety: this is the first content-rewriting method in the library

Every other method in `TieredMemory` -- `store`, `consolidate`,
`decay`, `retrieve` -- only adds, moves tier, removes-if-forgotten, or
reads. None of them rewrite what a stored event *means*.
`deduplicate()` and `compress()` are qualitatively different: they
destroy original events and replace them with a synthesized one. This
needs to be documented loudly, not treated as "just another method,"
in both the docstring and the eventual README section.

Mitigations for v1:

- **`dry_run: bool = False` on every method**, including the
  `offline_consolidate()` wrapper. A dry run computes and returns the
  same `ConsolidationReport` shape -- which pairs would merge, which
  groups would compress, which edges would strengthen -- without
  calling `remove()`/`add()`/mutating anything. Always run this
  first is the documented recommendation, not enforced by the API.
- **Provenance metadata** (`merged_from`, `summarized_from`) on every
  surviving synthesized event -- the *fact that* a merge/compression
  happened and *which ids* were involved is always recoverable from
  the surviving event, even without a report.
- **No true undo in v1, stated plainly, not hidden.** Originals are
  hard-removed via `backend.remove()`, not soft-deleted. The
  provenance metadata tells you *which* events were merged away, not
  their original *content* -- if the caller didn't separately log
  the `ConsolidationReport` (or the events themselves) before running
  a real pass, that content is gone. Real reversibility (soft-delete,
  versioning) is a bigger feature, out of scope here.

## File structure

- `core.py`: `deduplicate`, `compress`, `strengthen_connections`,
  `offline_consolidate` added to `TieredMemory`; `_find_similar_pairs`
  and `_find_graph_backend` as module-private helpers.
- `events.py`: `ConsolidationReport` dataclass.
- `src/memory_system/summarization/` (new package, mirrors
  `extraction/`'s existing structure and naming exactly):
  - `base.py`: `MemorySummarizer` ABC.
  - `llm_based.py`: `LLMSummarizer`, using the existing `llm` extra.
- `backends/graph.py`: add `strength: float = 1.0` to `Relationship`.

## Testing

- `deduplicate()`: `InMemoryBackend` (deterministic TF-IDF), two
  near-identical synthetic facts above threshold, one clearly
  unrelated fact below threshold. Explicit test that a single event
  never merges with itself (the self-matching exclusion is a named
  test case, not incidentally covered).
- `compress()`: a fake `MemorySummarizer` test double for the default
  suite (matching how `LLMEntityExtractor`'s tests mock the anthropic
  client rather than calling it), plus one real-LLM live test gated
  on `ANTHROPIC_API_KEY` (matching `test_llm_extractor_live.py`'s
  existing skip-if-absent pattern exactly).
- `strengthen_connections()`: `GraphBackend` directly, and a
  `HybridBackend` wrapping a `GraphBackend` as one of its two
  composed backends, to exercise `_find_graph_backend`'s reach-through
  path. Also: `InMemoryBackend`/`ChromaBackend` alone return an empty
  `ConsolidationReport`, not an error.
- Execution order: a test that constructs a scenario where, if
  `strengthen_connections()` ran before `deduplicate()`, it would
  strengthen an edge belonging to an event that dedup is about to
  remove -- asserting the edge is absent (not merely unstrengthened)
  after a real `offline_consolidate()` pass confirms the order is
  actually enforced, not just documented.
- `dry_run=True`: for each method, assert the backend's state
  (`get_all()`, and for `GraphBackend`, `_edges`) is byte-identical
  before and after, while the returned `ConsolidationReport` is
  non-empty and matches what a real run would have done.

## Non-goals

- Scheduling or automatic triggering -- manual call only, exactly
  matching `consolidate()`/`decay()`.
- True co-retrieval-frequency tracking -- v1's "strengthen" is
  scoped to same-pass relatedness only; frequency-based strengthening
  needs new tracking infrastructure and is a separate design.
- Reversibility/undo beyond provenance metadata -- no soft-delete or
  versioning in v1.
- Changing `consolidate()` or `decay()`'s existing behavior or tests
  -- `offline_consolidate()` is fully additive, a new set of methods,
  not a modification of the existing ones.
- A pluggable duplicate-detection policy abstraction (mirroring
  `SalienceScorer`/`ConsolidationPolicy`) -- v1 uses a single required
  `threshold` float via the backend's own `query()`; a swappable
  detection strategy is a plausible v2 if the fixed-threshold
  approach proves too rigid in practice, not built preemptively here.
