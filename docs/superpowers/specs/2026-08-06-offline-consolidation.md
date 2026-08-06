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

def strengthen_connections(
    self,
    merge_report: Optional[ConsolidationReport] = None,
    compress_report: Optional[ConsolidationReport] = None,
    dry_run: bool = False,
) -> ConsolidationReport: ...

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
    strengthen_report = self.strengthen_connections(merge_report, compress_report, dry_run=dry_run)
    return ConsolidationReport(
        merged=merge_report.merged,
        compressed=compress_report.compressed,
        strengthened=strengthen_report.strengthened,
    )
```

**Correction made while writing the implementation plan, not caught during
design review:** the original version of this signature was
`strengthen_connections(self, dry_run: bool = False)` -- no way to
tell it which events `deduplicate()`/`compress()` had just touched.
"Entities related by this same pass" (see Mechanism 3) is
undefined without that information; the method literally could not
have been implemented as originally specified. Writing real,
executable code for `offline_consolidate()`'s wrapper body is what
surfaced this -- prose review didn't catch it because the pseudocode
never had to actually run. `strengthen_connections()` now takes the
`merge_report`/`compress_report` from the same pass (both optional,
both `None` by default -- called with neither, correctly, there is
nothing to strengthen and it returns an empty `ConsolidationReport`).
`offline_consolidate()`'s wrapper is updated to pass them through.

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
    merged: list[tuple[str, str, Optional[str]]] = field(default_factory=list)       # (source_a_id, source_b_id, new_id)
    compressed: list[tuple[list[str], Optional[str]]] = field(default_factory=list)  # (source_ids, new_summary_id)
    strengthened: list[tuple[str, str]] = field(default_factory=list)                # (source_id, target_id) entity pairs
```

Every method returns this same shape whether `dry_run=True` or not --
a dry run answers "what would happen," a real run answers "what did
happen," same fields either way, with one exception: `new_id`/
`new_summary_id` is `None` under `dry_run=True`. A dry run never calls
`add()`, so there is no real `MemoryEvent.id` to report -- inventing
one just for display would imply an event exists that doesn't. The
source ids (always real, since dry runs still identify *which*
existing events would merge or group) are what a dry run is actually
for; the synthesized id only exists once a real pass creates it.

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

**`GraphBackend`-specific fix: reassign existing relationships, don't
rely on re-extraction.** `GraphBackend.add()` re-runs entity
extraction on whatever content it's given. Since a merged event keeps
only one source's content verbatim (not a combination), relying on
`add(merged_event)` alone would let entities/relationships that were
unique to the *discarded* source's text quietly vanish -- not because
they were judged redundant, but because the extractor never saw that
text again. Deduplication is supposed to mean "these are the same
fact, keep one," not "discard information" -- silent, unmeasured
information loss inside an operation whose whole premise is
losslessness is a real defect, not an accepted tradeoff of "merge
doesn't synthesize." (Entity *nodes* themselves are never actually at
risk -- `self._entities` is a global dict keyed by canonical entity
id, not scoped to any one event, so an entity record isn't deleted by
removing the event that introduced it. What's actually at risk is the
*edges*: `GraphBackend.remove(event_id)` deletes every `Relationship`
whose `source_event_id` matches, and if a discarded source's edges
aren't preserved before that removal runs, every relationship it
contributed disappears. An entity left with zero surviving edges is
still technically present in `_entities`, but unreachable from
`related_to()`/`explain_path()` traversal -- functionally gone from
the graph even though its bare record isn't physically deleted.)

The fix: **reuse the extraction that already happened for both
sources**, instead of re-deriving from whichever single text string
survives. New public method on `GraphBackend`:

```python
def reassign_relationships(self, old_event_ids: list[str], new_event_id: str) -> list[Relationship]:
    """Retargets every relationship whose source_event_id is in
    old_event_ids to point at new_event_id instead, preserving the
    relationships themselves (entities, relation_type, confidence,
    strength) rather than re-deriving them from a single surviving
    content string, which would silently drop anything only present
    in a discarded event's original phrasing.

    Collapses relationships that become identical after reassignment
    (same source_id, target_id, relation_type now all pointing at
    new_event_id) into one, keeping max(confidence) and max(strength)
    across the collapsed set -- same reasoning as salience's max() on
    event merge: two sources independently asserting the same fact
    should end up as one edge at least as strong as either alone, not
    two parallel duplicates and not a diluted average. Does NOT apply
    strengthen_connections()'s separate +0.1 reinforcement bump --
    that stays strengthen_connections()'s job alone (see Mechanism 3
    and Execution order), so a collapsed edge isn't double-boosted by
    both mechanisms in the same pass.

    Returns the relationships now attached to new_event_id.
    """
```

`deduplicate()`'s merge action, for a `GraphBackend` (or a
`HybridBackend` composed with one, via the same `_find_graph_backend`
helper `strengthen_connections()` uses), becomes, per merged pair:

1. `backend.add(merged_event)` -- stores the merged event; its own
   `add()` re-extracts from the surviving content as usual, so
   entities/relationships still present in that text are captured
   the normal way.
2. `graph_backend.reassign_relationships([a.id, b.id], merged_event.id)`
   -- retargets *every* relationship either original source
   contributed (including ones step 1's re-extraction already
   re-derived, which collapse into the same edge rather than
   duplicating) onto the merged event.
3. `backend.remove(a.id)`, `backend.remove(b.id)` -- removes the
   original events. By this point every relationship they contributed
   already points at `merged_event.id`, not at `a.id`/`b.id`, so
   `remove()`'s existing `source_event_id`-matching prune (see
   Execution order) does not touch them. Only `a`/`b`'s entries in
   `_events` are removed.

This order matters as much as the execution order at the
`offline_consolidate()` level does, for the identical underlying
reason: `remove()` prunes by `source_event_id`, so anything that
needs to survive a `remove()` call must already point somewhere else
*before* that call runs, not after.

**Why `source_event_id` has to be reassigned, not just left alone.**
Leaving it pointing at the now-deleted original event id would not
just be stale bookkeeping -- it would break the field's actual
contract and the parts of this spec that depend on it:

- `Relationship`'s own docstring says `source_event_id` exists so
  callers "can trace provenance ... at the relationship level" -- a
  dangling reference to an event that no longer exists breaks that
  for good, since `self._events.get(source_event_id)` returns `None`
  forever after.
- The Execution order section's entire argument for running
  `strengthen_connections()` last rests on `remove()` correctly
  pruning edges for removed events. An edge with a stale
  `source_event_id` would never be pruned by any *future* `remove()`
  call either (its `source_event_id` matches nothing live), so it
  would sit in `_edges` as a permanent orphan, accumulating silently
  across every subsequent consolidation pass.
- After reassignment, the merged event *is* the current, correct
  "specific memory event" this relationship should be attributed to
  -- it's the surviving representation of that fact. Reassigning
  `source_event_id` to it isn't a workaround, it's making the field
  say something true.

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

**Compress action**, for each group (size >= 2): `add()` one new
summary event, `remove()` every source.

- `content` = `summarizer.summarize(group_events)`.
- `salience` = `max(salience for event in group)`.
- `last_reinforced` = `now`; `timestamp` = earliest in group -- same
  reasoning as merge.
- `metadata["summarized_from"] = [event.id for event in group]`.

**Same `GraphBackend` fix as merge, same reason -- if anything, a
sharper case for it.** `summarizer.summarize()` produces LLM-
synthesized text, which is *less* likely than merge's "pick one
source's original text" to happen to mention every entity each
individual source event extracted, especially once a group has 3+
sources condensed into one paragraph. This is the identical failure
mode Mechanism 1 has, not a new one, so it gets the identical fix:
`reassign_relationships()` already takes `old_event_ids: list[str]`
-- built for a pair, but a list works for a group of any size with no
changes -- so `compress()`'s action for `GraphBackend` (or
`HybridBackend` composed with one) is: `add(summary_event)` ->
`graph_backend.reassign_relationships([e.id for e in group], summary_event.id)`
-> `remove()` each source. Same collapse-duplicates-with-max()
behavior, same "not strengthen_connections()'s job" boundary, same
reasoning for why `source_event_id` must point at the summary event
afterward, not any of the removed sources -- all as specified in
Mechanism 1, reused here rather than restated.

## Mechanism 3: Connection strengthening

**Scoped exactly to where it's meaningful, not applied universally.**
Only `GraphBackend` has connections to strengthen at all;
`InMemoryBackend`/`ChromaBackend` have no relationship structure.
`HybridBackend` is in scope only when one of its two composed
backends happens to be a `GraphBackend`:

```python
def strengthen_connections(
    self,
    merge_report: Optional[ConsolidationReport] = None,
    compress_report: Optional[ConsolidationReport] = None,
    dry_run: bool = False,
) -> ConsolidationReport:
    graph_backend = _find_graph_backend(self.backend)
    if graph_backend is None:
        return ConsolidationReport()  # not an error -- this backend has nothing to strengthen

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
                edge = graph_backend.find_edge(entity_a, entity_b)
                if edge is None:
                    continue
                # Skip edges that are themselves part of this
                # consolidation event -- see the correction note below.
                if edge.source_event_id == event_id:
                    continue
                if not dry_run:
                    edge.strength = min(1.0, edge.strength + 0.1)
                report.strengthened.append((entity_a, entity_b))
    return report

def _find_graph_backend(backend: MemoryBackend) -> Optional[GraphBackend]:
    if isinstance(backend, GraphBackend):
        return backend
    if isinstance(backend, HybridBackend):
        for sub in (backend.lexical_backend, backend.semantic_backend):
            if isinstance(sub, GraphBackend):
                return sub
    return None
```

**Correction made while writing the implementation plan, not caught during
design review:** the algorithm above, exactly as originally specified, is
inconsistent with the test this same spec mandates below (a merge
co-associating two entities via a pre-existing, unrelated edge between
them, asserting only that edge gets strengthened). By the time
`strengthen_connections()` runs, `deduplicate()`/`compress()` have
already given the merged/summary event its *own* relationships two
ways: `add()` re-runs entity extraction against the surviving event's
content, and `reassign_relationships()` retargets the discarded
events' relationships onto the same new event id. `entities_for_event(new_id)`
therefore returns entities from both sources indiscriminately -- it
cannot tell "an edge that happens to connect two entities this merge
brought together" apart from "an edge that IS this merge." Without an
exclusion, the literal algorithm strengthens both: run against the
mandated test's exact scenario, it produces 3 strengthened pairs where
the test asserts exactly 1 (verified empirically during implementation).
The fix, now reflected in the code block above: skip any edge whose
`source_event_id` equals the `event_id` currently being processed --
those edges' confidence/strength are already handled by the merge/
compress that just created them; only genuinely external "bystander"
edges between co-associated entities are eligible for strengthening.

**Where the entity pairs actually come from -- resolved concretely,
not left as "for every pair of entities mentioned across that group,"
which begged the question of how those entities get found once the
original source events are gone.** By the time `strengthen_connections()`
runs, `deduplicate()`/`compress()` have already removed every
original source event -- their `merge_report`/`compress_report` only
retain ids, not the events themselves. But that's sufficient: after
`reassign_relationships()` ran (Mechanisms 1 and 2), every
relationship either source contributed now has `source_event_id`
pointing at the *new* merged/summary event. So "which entities were
related by this pass" is answerable entirely from the new event's id
-- no need to remember what the old, now-deleted events were
individually connected to. Two small new public methods on
`GraphBackend` make this queryable without `strengthen_connections()`
reaching into `_edges`/`_adjacency` directly:

```python
def entities_for_event(self, event_id: str) -> set[str]:
    """Entity ids touched by relationships sourced from this event,
    as either source or target."""
    return {
        eid
        for rel in self._edges
        if rel.source_event_id == event_id
        for eid in (rel.source_id, rel.target_id)
    }

def find_edge(self, entity_a: str, entity_b: str) -> Optional[Relationship]:
    """The relationship connecting these two entities, in either
    direction, or None if none exists."""
    for rel in self._edges:
        if {rel.source_id, rel.target_id} == {entity_a, entity_b}:
            return rel
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
`Relationship.strength: float = 0.5` to the `Relationship` dataclass
in `backends/graph.py` -- backward compatible (new field, has a
default, no existing construction site breaks).

**Correction made in post-ship exploratory testing, not caught during
design review or implementation review:** this field originally
defaulted to `1.0`, the same value as the `+0.1` cap
`strengthen_connections()` enforces. Since no `EntityExtractor`
(`RuleBasedEntityExtractor`, `LLMEntityExtractor`) or any other code
path in the library ever writes to `.strength` except
`strengthen_connections()`'s own bump and `reassign_relationships()`'s
`max()`-collapse (which can only ever produce `<= 1.0` too), every
real, extractor-produced `Relationship` was born already at the
ceiling: `min(1.0, 1.0 + 0.1)` is a mathematical no-op.
`strengthen_connections()` was therefore a complete no-op on any
genuinely extractor-populated graph -- every unit test that observed a
strength change manually set `.strength = <below-cap value>` on an
edge before calling the method, which masked the defect, since no real
code path does that on its own. Fixed by lowering the default to
`0.5`: a neutral midpoint, analogous to `MemoryEvent.salience`
defaulting to `0.0` rather than an already-maxed value, leaving room
for real reinforcement to have an observable effect (five `+0.1`
rounds to reach the cap). This was found by an exploratory testing
pass run against real `store()`/`consolidate()`/`offline_consolidate()`
flows with the real `RuleBasedEntityExtractor` -- not by the automated
test suite, which had no test exercising a real extractor's output
without manually overriding `.strength` first.

**What "strengthen" means in v1, and what's explicitly deferred.**
"Frequently co-retrieved" as literally stated needs retrieval
co-occurrence tracking that doesn't exist anywhere in tiered-memory
today (`retrieve()` reinforces events individually, never records
which events came back *together*). Building that tracking is a
separate feature in its own right -- a new question of where the
state lives (`TieredMemory`? a running counter on `MemoryEvent`?) and
its own design pass, not a detail of this one. **v1 strengthens
connections between entities that ended up associated with the same
surviving event after this same consolidation pass** -- concretely,
for each `new_id` in the `merge_report`/`compress_report` passed in,
every pairwise combination of entities `entities_for_event(new_id)`
returns: if `find_edge()` says a relationship already connects them,
bump its `strength` by a fixed increment (`+0.1`), capped at `1.0`
(matching `confidence`'s own typical range, so the two fields stay
comparable even though they mean different things). **Pairs with no
existing edge are skipped,
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
- `backends/graph.py`: add `strength: float = 1.0` to `Relationship`;
  add three new public methods on `GraphBackend`:
  `reassign_relationships(old_event_ids, new_event_id)`,
  `entities_for_event(event_id)`, `find_edge(entity_a, entity_b)` --
  the latter two exist so `strengthen_connections()` never has to
  reach into `_edges`/`_adjacency` directly, consistent with how
  `reassign_relationships()` was added instead of doing the same from
  `core.py`.

## Testing

- `deduplicate()`: `InMemoryBackend` (deterministic TF-IDF), two
  near-identical synthetic facts above threshold, one clearly
  unrelated fact below threshold. Explicit test that a single event
  never merges with itself (the self-matching exclusion is a named
  test case, not incidentally covered).
- `deduplicate()` on `GraphBackend`: two events whose text differs
  enough that each extracts at least one entity/relationship the
  other doesn't (the exact scenario that silently lost information
  before this fix) -- merge them, then assert every relationship
  either original event contributed is still traversable from the
  merged event via `related_to()`, and that `Relationship.source_event_id`
  for each is the merged event's id, not either original's (a stale
  reference would be the bug re-appearing in a different form). Also:
  two events asserting the identical relationship (same source_id/
  target_id/relation_type) collapse into one edge after merge, not
  two, with `confidence`/`strength` equal to the max of the two
  originals.
- `compress()`: a fake `MemorySummarizer` test double for the default
  suite (matching how `LLMEntityExtractor`'s tests mock the anthropic
  client rather than calling it), plus one real-LLM live test gated
  on `ANTHROPIC_API_KEY` (matching `test_llm_extractor_live.py`'s
  existing skip-if-absent pattern exactly).
- `compress()` on `GraphBackend`: a fake `MemorySummarizer` whose
  `summarize()` returns text that deliberately omits an entity one of
  the 3+ source events contributed (simulating exactly what a real LLM
  summary would plausibly do) -- assert that entity's relationship is
  still traversable from the summary event afterward, same shape of
  assertion as the `deduplicate()` `GraphBackend` test above.
- `strengthen_connections()`: `GraphBackend` directly, and a
  `HybridBackend` wrapping a `GraphBackend` as one of its two
  composed backends, to exercise `_find_graph_backend`'s reach-through
  path. Also: `InMemoryBackend`/`ChromaBackend` alone return an empty
  `ConsolidationReport`, not an error. Also: called with no
  `merge_report`/`compress_report` (the default), returns an empty
  `ConsolidationReport` even against a real `GraphBackend` -- there is
  nothing to strengthen without knowing which pass produced what.
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
