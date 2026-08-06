# Offline Consolidation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `offline_consolidate()` (plus three independently-callable sub-methods: `deduplicate()`, `compress()`, `strengthen_connections()`) to `TieredMemory` — a manual, caller-triggered batch pass over long-term memory that merges near-duplicates, compresses related groups into summaries, and strengthens `GraphBackend` connections discovered during the pass.

**Architecture:** Merge/compress are built entirely from the existing `remove()`+`add()` `MemoryBackend` primitives (no new backend interface methods needed for that part). Similarity detection reuses each backend's own `query()`. Three new public `GraphBackend` methods (`reassign_relationships`, `entities_for_event`, `find_edge`) let the `GraphBackend`-specific entity-preservation fix and connection-strengthening logic stay out of `core.py`'s hands entirely, never touching `GraphBackend._edges`/`_adjacency` directly.

**Tech Stack:** Pure Python for `deduplicate`/`strengthen_connections`/the `GraphBackend` additions (no new dependency). `compress()`'s `LLMSummarizer` reuses the existing `llm` extra (`anthropic`) — zero new optional-dependency permutations.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-08-06-offline-consolidation.md` — read it if anything below is ambiguous. Two corrections were made to it while writing this plan (both already committed to the spec, `3f4ea11` and `077b4aa`): the `GraphBackend`-specific entity-preservation fix (`reassign_relationships`), and `strengthen_connections()`'s signature gaining `merge_report`/`compress_report` parameters (the original signature had no way to know which events a pass had touched).
- Python floor is `>=3.9` (`pyproject.toml`). Every new file MUST start with `from __future__ import annotations` — a prior commit (`6ef6dc0`) broke CI on 3.9 specifically because a test file used `X | None` without it.
- `threshold` on `deduplicate()`/`compress()` is a **required** argument, never defaulted — the backends' similarity scores are on incompatible scales (established in the hybrid-retrieval spec), so no universal default is meaningful.
- Execution order inside `offline_consolidate()`: `deduplicate()` → `compress()` → `strengthen_connections()`, hardcoded in the method body, not a parameter.
- `dry_run=True` on every method must leave the backend byte-identical (no `add()`/`remove()`/mutation calls at all) and must report `None` for any not-yet-created id (`ConsolidationReport.merged`'s third element, `compressed`'s second element).
- `Relationship.confidence` (extraction confidence) and `Relationship.strength` (consolidation-reinforced importance, new field, default `1.0`) are separate fields — never conflate them.
- `GraphBackend`'s existing `remove(event_id)` already deletes every edge whose `source_event_id` matches (current behavior, not something this plan changes) — every task involving `GraphBackend` must account for this.

---

### Task 1: `ConsolidationReport` dataclass + `Relationship.strength` field

**Files:**
- Modify: `src/memory_system/events.py`
- Modify: `src/memory_system/backends/graph.py`
- Create: `tests/test_consolidation_report.py`
- Modify: `tests/test_graph_backend.py`

**Interfaces:**
- Produces: `ConsolidationReport` (in `memory_system.events`) with fields `merged: list[tuple[str, str, Optional[str]]]`, `compressed: list[tuple[list[str], Optional[str]]]`, `strengthened: list[tuple[str, str]]`, all defaulting to empty lists. `Relationship.strength: float` (in `memory_system.backends.graph`), defaults `1.0`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_consolidation_report.py`:

```python
from __future__ import annotations

from memory_system.events import ConsolidationReport


def test_default_construction_has_empty_lists():
    report = ConsolidationReport()
    assert report.merged == []
    assert report.compressed == []
    assert report.strengthened == []


def test_constructs_with_populated_fields():
    report = ConsolidationReport(
        merged=[("a", "b", "c")],
        compressed=[(["d", "e"], "f")],
        strengthened=[("g", "h")],
    )
    assert report.merged == [("a", "b", "c")]
    assert report.compressed == [(["d", "e"], "f")]
    assert report.strengthened == [("g", "h")]


def test_dry_run_new_id_can_be_none():
    report = ConsolidationReport(merged=[("a", "b", None)], compressed=[(["c"], None)])
    assert report.merged[0][2] is None
    assert report.compressed[0][1] is None
```

Add to `tests/test_graph_backend.py` (append at the end of the file):

```python
def test_relationship_strength_defaults_below_the_strengthen_cap():
    # Post-ship correction (see this task's "Correction made..." note
    # below): must be strictly below the 1.0 cap strengthen_connections()
    # enforces, or no real Relationship could ever be observed to
    # strengthen.
    rel = Relationship(
        source_id="user", target_id="peanut", relation_type="ALLERGIC_TO", source_event_id="evt1"
    )
    assert rel.strength == 0.5
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_consolidation_report.py tests/test_graph_backend.py::test_relationship_strength_defaults_below_the_strengthen_cap -v`
Expected: `test_consolidation_report.py` fails with `ImportError: cannot import name 'ConsolidationReport'`; the `Relationship` test fails with `TypeError: Relationship.__init__() got an unexpected keyword argument` — no, actually it will fail because `strength` doesn't exist as an attribute: `AttributeError: 'Relationship' object has no attribute 'strength'`.

- [ ] **Step 3: Write the implementation**

In `src/memory_system/events.py`, add after the existing `RetrievalResult` class (end of file):

```python
@dataclass
class ConsolidationReport:
    """Result of an offline-consolidation pass (deduplicate/compress/
    strengthen_connections/offline_consolidate). Replaces a bare int
    count -- these methods do several different kinds of things, so a
    count alone would hide what actually happened.

    The third element of each `merged` tuple and the second element of
    each `compressed` tuple is `None` under dry_run=True: a dry run
    never calls add(), so there is no real MemoryEvent.id to report.
    """

    merged: list[tuple[str, str, Optional[str]]] = field(default_factory=list)
    compressed: list[tuple[list[str], Optional[str]]] = field(default_factory=list)
    strengthened: list[tuple[str, str]] = field(default_factory=list)
```

In `src/memory_system/backends/graph.py`, modify the `Relationship` dataclass (currently lines 40-51) from:

```python
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
```

to:

```python
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
    strength: float = 0.5      # consolidation-reinforced importance -- distinct from
                                # confidence, which is about extraction certainty, not
                                # how reinforced this connection has become over time
    metadata: dict = field(default_factory=dict)
```

**Correction made in post-ship exploratory testing, not caught during
task review or the final whole-branch review:** this field originally
shipped defaulting to `1.0` (matching this task's own commit,
`d1eb55c`) -- the same value as the `+0.1` cap
`strengthen_connections()` (Task 7) enforces. No code path in the
library other than `strengthen_connections()`'s own bump ever writes
to `.strength`, so every real, extractor-produced `Relationship` was
born already at the ceiling, making `strengthen_connections()` a
complete no-op on genuinely extractor-populated data --
`min(1.0, 1.0 + 0.1)` never moves. Every Task 7/8 unit test that
observed a strength change manually set `.strength` to a below-cap
value before calling the method, which is not something any real code
path does on its own, so the defect passed every review gate. Fixed by
lowering the default to `0.5` (see the spec's Mechanism 3 section for
the full writeup and a dedicated end-to-end regression test using the
real `RuleBasedEntityExtractor`, added to `tests/test_offline_consolidation.py`).

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_consolidation_report.py tests/test_graph_backend.py -v`
Expected: all pass (3 new + existing `test_graph_backend.py` tests, none broken by the new field since it has a default).

- [ ] **Step 5: Commit**

```bash
git add src/memory_system/events.py src/memory_system/backends/graph.py tests/test_consolidation_report.py tests/test_graph_backend.py
git commit -m "$(cat <<'EOF'
Add ConsolidationReport dataclass and Relationship.strength field

First piece of offline consolidation (see
docs/superpowers/specs/2026-08-06-offline-consolidation.md).
ConsolidationReport replaces the bare int that consolidate()/decay()
return for the new deduplicate/compress/strengthen_connections/
offline_consolidate methods, which each do several qualitatively
different things a count alone would hide. Relationship.strength is a
new field, separate from the existing confidence field -- confidence
means extraction certainty, strength means consolidation-reinforced
importance, conflating them would lose information.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `_find_similar_pairs()` with explicit self-matching exclusion

**Files:**
- Modify: `src/memory_system/core.py`
- Create: `tests/test_offline_consolidation.py`

**Interfaces:**
- Consumes: `MemoryBackend.get_all()`/`query()` (existing), `MemoryTier` (existing).
- Produces: `_find_similar_pairs(backend: MemoryBackend, tier: MemoryTier, threshold: float) -> list[tuple[MemoryEvent, MemoryEvent, float]]`, a module-level function in `memory_system.core` (not a `TieredMemory` method). Consumed by Tasks 4 and 6.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_offline_consolidation.py`:

```python
from __future__ import annotations

from memory_system.backends.memory import InMemoryBackend
from memory_system.core import _find_similar_pairs
from memory_system.events import MemoryEvent, MemoryTier


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_offline_consolidation.py -v`
Expected: FAIL with `ImportError: cannot import name '_find_similar_pairs' from 'memory_system.core'`

- [ ] **Step 3: Write the implementation**

In `src/memory_system/core.py`, add after the imports (after line 12, before the blank lines preceding `class TieredMemory:`):

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_offline_consolidation.py -v`
Expected: 4 passed

- [ ] **Step 5: Commit**

```bash
git add src/memory_system/core.py tests/test_offline_consolidation.py
git commit -m "$(cat <<'EOF'
Add _find_similar_pairs() with explicit self-matching exclusion

Second piece of offline consolidation. No backend's query() excludes
the querying event's own content from its own results -- verified
empirically, InMemoryBackend.query(a.content) on a single-event
corpus returns a itself at score=1.0. Without the explicit
`result.event.id == event.id` skip, every event would trivially
"match itself" above any threshold and deduplicate() would try to
merge every event with itself first. That exclusion is a named test
case (test_never_matches_event_with_itself), not incidentally
covered.

4 new tests, TDD: near-identical pair found above threshold, unrelated
pair excluded below threshold, self-matching excluded, symmetric pair
(a,b)/(b,a) reported once not twice.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: `GraphBackend.reassign_relationships()`

**Files:**
- Modify: `src/memory_system/backends/graph.py`
- Modify: `tests/test_graph_backend.py`

**Interfaces:**
- Produces: `GraphBackend.reassign_relationships(self, old_event_ids: list[str], new_event_id: str) -> list[Relationship]`. Consumed by Task 5 (`deduplicate()`) and Task 6 (`compress()`).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_graph_backend.py` (append at the end):

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_graph_backend.py -v -k reassign`
Expected: FAIL with `AttributeError: 'GraphBackend' object has no attribute 'reassign_relationships'`

- [ ] **Step 3: Write the implementation**

In `src/memory_system/backends/graph.py`, add as a new method on `GraphBackend`, placed after `remove()` (currently lines 151-153) and before the `# --- graph-native methods` comment:

```python
    def reassign_relationships(self, old_event_ids: list[str], new_event_id: str) -> list[Relationship]:
        """Retargets every relationship whose source_event_id is in
        old_event_ids to point at new_event_id instead, preserving the
        relationships themselves rather than re-deriving them via
        extraction from a single surviving content string, which would
        silently drop anything only present in a discarded event's
        original phrasing (see docs/superpowers/specs/
        2026-08-06-offline-consolidation.md).

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
        affected = [rel for rel in self._edges if rel.source_event_id in old_ids]

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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_graph_backend.py -v`
Expected: all pass (4 new + all existing `test_graph_backend.py` tests unaffected).

- [ ] **Step 5: Commit**

```bash
git add src/memory_system/backends/graph.py tests/test_graph_backend.py
git commit -m "$(cat <<'EOF'
Add GraphBackend.reassign_relationships()

Third piece of offline consolidation, and the fix for a real
information-loss defect: relying on GraphBackend.add() to re-extract
entities from whichever single content string survives a merge would
silently drop any relationship that was only present in a discarded
source's original phrasing -- not because it was judged redundant,
but because the extractor never saw that text again. Deduplication is
supposed to mean "these are the same fact, keep one," not "discard
information."

reassign_relationships() reuses the extraction that already happened
for every source instead of re-deriving from one surviving string:
retargets each relationship's source_event_id onto the new event,
collapsing now-identical relationships (same source_id/target_id/
relation_type) with max(confidence)/max(strength) rather than
duplicating or diluting. Removal is by object identity, not dataclass
value-equality, to avoid list.remove() targeting the wrong object
when two relationships happen to share all field values.

Reassigning source_event_id (not leaving it dangling) matters beyond
bookkeeping: GraphBackend.remove() already prunes edges by matching
source_event_id, so an edge left pointing at a removed event's id
would (a) never be prunable by any future remove() either, becoming a
permanent orphan, and (b) break source_event_id's own documented
purpose of tracing provenance, since self._events.get() on a removed
id returns None.

4 new tests: retargeting, preserving entities unique to each source
(the exact scenario that silently lost information before this fix),
collapsing identical relationships with max(), and confirming
reassigned edges survive a subsequent remove() on the original
events.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: `TieredMemory.deduplicate()` core (InMemoryBackend path)

**Files:**
- Modify: `src/memory_system/core.py`
- Modify: `tests/test_offline_consolidation.py`

**Interfaces:**
- Consumes: `_find_similar_pairs` (Task 2), `ConsolidationReport` (Task 1).
- Produces: `TieredMemory.deduplicate(self, threshold: float, dry_run: bool = False) -> ConsolidationReport`. Extended by Task 5 (adds the `GraphBackend`-specific path). Consumed by Task 8 (`offline_consolidate()`).

- [ ] **Step 1: Write the failing tests**

Add to the top imports of `tests/test_offline_consolidation.py`:

```python
from datetime import datetime, timezone

from memory_system import AlwaysConsolidate, NoDecay, TieredMemory
```

Append to `tests/test_offline_consolidation.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_offline_consolidation.py -v -k deduplicate`
Expected: FAIL with `AttributeError: 'TieredMemory' object has no attribute 'deduplicate'`

- [ ] **Step 3: Write the implementation**

In `src/memory_system/core.py`, change line 9 from:

```python
from .events import MemoryEvent, MemoryTier, RetrievalResult
```

to:

```python
from .events import ConsolidationReport, MemoryEvent, MemoryTier, RetrievalResult
```

Then add `deduplicate()` as a new method on `TieredMemory`, placed after `consolidate()` (currently ends at line 64) and before `decay()`:

```python
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
                self.backend.remove(a.id)
                self.backend.remove(b.id)
                new_id = merged_event.id

            report.merged.append((a.id, b.id, new_id))
            merged_event_ids.add(a.id)
            merged_event_ids.add(b.id)

        return report
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_offline_consolidation.py -v`
Expected: 13 passed (4 from Task 2 + 9 new)

- [ ] **Step 5: Commit**

```bash
git add src/memory_system/core.py tests/test_offline_consolidation.py
git commit -m "$(cat <<'EOF'
Add TieredMemory.deduplicate() core (InMemoryBackend path)

Fourth piece of offline consolidation. Merge action: remove() both
originals, add() one new event whose content is the higher-salience
source's content unmodified (merge preserves, it doesn't synthesize --
that's compress()'s job), salience is max() of both sources (not sum,
which would double-count, not average, which would dilute a strong
fact merged with a weak duplicate), timestamp is the earliest source's
(preserves "when first learned"), last_reinforced is set to now
(consolidation is itself a reinforcement event), and metadata carries
merged_from provenance.

GraphBackend-specific handling (reassign_relationships from the prior
commit) is deliberately NOT wired in yet -- that's Task 5, kept
separate so this task's core merge mechanics can be reviewed
independently of the graph-specific fix.

9 new tests (13 total): merges near-identical pairs, keeps the
higher-salience content, salience/timestamp/last_reinforced/
merged_from rules, leaves unrelated events untouched, and dry_run
mutates nothing and reports new_id=None.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 5: `_find_graph_backend()` + wire `deduplicate()`'s `GraphBackend` path

**Files:**
- Modify: `src/memory_system/core.py`
- Modify: `tests/test_offline_consolidation.py`

**Interfaces:**
- Consumes: `GraphBackend.reassign_relationships` (Task 3), `HybridBackend.lexical_backend`/`semantic_backend` (existing public attributes).
- Produces: `_find_graph_backend(backend: MemoryBackend) -> Optional[GraphBackend]`, a module-level function in `memory_system.core`. Consumed by Task 6 (`compress()`) and Task 7 (`strengthen_connections()`).

- [ ] **Step 1: Write the failing tests**

Add to the top imports of `tests/test_offline_consolidation.py`:

```python
from memory_system.backends.graph import GraphBackend
from memory_system.backends.hybrid import HybridBackend
from memory_system.core import _find_graph_backend
from tests.test_graph_backend import ScriptedExtractor
```

Append to `tests/test_offline_consolidation.py`:

```python
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
    remaining = memory.backend.get_all(tier=MemoryTier.LONG_TERM)
    assert len(remaining) == 1
    assert remaining[0].id == merged_id
    assert remaining[0].content == a.content
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_offline_consolidation.py -v -k "find_graph_backend or graph_backend"`
Expected: FAIL with `ImportError: cannot import name '_find_graph_backend' from 'memory_system.core'`

- [ ] **Step 3: Write the implementation**

In `src/memory_system/core.py`, add two new imports after line 8 (`from .backends.base import MemoryBackend`):

```python
from .backends.graph import GraphBackend
from .backends.hybrid import HybridBackend
```

Add `_find_graph_backend` as a new module-level function, right after `_find_similar_pairs` (from Task 2):

```python
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
```

Modify `deduplicate()` (from Task 4) to detect and use the graph backend. Change:

```python
        pairs = _find_similar_pairs(self.backend, MemoryTier.LONG_TERM, threshold)
        now = datetime.now(timezone.utc)
```

to:

```python
        pairs = _find_similar_pairs(self.backend, MemoryTier.LONG_TERM, threshold)
        graph_backend = _find_graph_backend(self.backend)
        now = datetime.now(timezone.utc)
```

and change:

```python
                self.backend.add(merged_event)
                self.backend.remove(a.id)
                self.backend.remove(b.id)
                new_id = merged_event.id
```

to:

```python
                self.backend.add(merged_event)
                if graph_backend is not None:
                    graph_backend.reassign_relationships([a.id, b.id], merged_event.id)
                self.backend.remove(a.id)
                self.backend.remove(b.id)
                new_id = merged_event.id
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_offline_consolidation.py -v`
Expected: 17 passed (13 from Task 4 + 4 new)

- [ ] **Step 5: Commit**

```bash
git add src/memory_system/core.py tests/test_offline_consolidation.py
git commit -m "$(cat <<'EOF'
Wire deduplicate() into GraphBackend via reassign_relationships()

Fifth piece of offline consolidation. _find_graph_backend() detects
whether the relevant backend is a GraphBackend directly, or a
HybridBackend composed with one -- same isinstance-based reach-through
the README already documents GraphBackend's own graph-native methods
requiring elsewhere.

deduplicate()'s merge action now calls reassign_relationships() for
GraphBackend before removing the original events, closing the
information-loss gap Task 3 fixed at the GraphBackend level: without
this wiring, that fix would exist but never actually run.

4 new tests (17 total): _find_graph_backend's three cases (direct
GraphBackend, inside a HybridBackend, absent), and an end-to-end
deduplicate() test on a real GraphBackend proving an entity/
relationship unique to the lower-salience (discarded-content) source
survives the merge -- the exact scenario that silently lost
information before Task 3's fix existed.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 6: `MemorySummarizer`/`LLMSummarizer` + `TieredMemory.compress()`

**Files:**
- Create: `src/memory_system/summarization/__init__.py`
- Create: `src/memory_system/summarization/base.py`
- Create: `src/memory_system/summarization/llm_based.py`
- Modify: `src/memory_system/core.py`
- Create: `tests/test_summarization.py`
- Create: `tests/test_summarization_live.py`
- Modify: `tests/test_offline_consolidation.py`

**Interfaces:**
- Consumes: `_find_similar_pairs` (Task 2), `_find_graph_backend` (Task 5), `GraphBackend.reassign_relationships` (Task 3).
- Produces: `MemorySummarizer` ABC (`memory_system.summarization.base`) with `summarize(self, events: list[MemoryEvent]) -> str`. `LLMSummarizer` (`memory_system.summarization.llm_based`). `TieredMemory.compress(self, threshold: float, summarizer: MemorySummarizer, dry_run: bool = False) -> ConsolidationReport`. `_connected_components` (module-private, `memory_system.core`). Consumed by Task 8 (`offline_consolidate()`).

- [ ] **Step 1: Write the failing tests**

Create `src/memory_system/summarization/__init__.py` (empty file — matches `src/memory_system/extraction/__init__.py`'s existing convention of not re-exporting anything; callers always use the full module path).

Create `tests/test_summarization.py`:

```python
from __future__ import annotations

from unittest.mock import MagicMock, patch

from memory_system.events import MemoryEvent
from memory_system.summarization.llm_based import LLMSummarizer


def make_mock_response(text: str):
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    return response


def make_summarizer():
    with patch("memory_system.summarization.llm_based.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        summarizer = LLMSummarizer(api_key="fake-key-for-testing")
        return summarizer, mock_client


def test_returns_the_model_response_text():
    summarizer, mock_client = make_summarizer()
    mock_client.messages.create.return_value = make_mock_response(
        "User has severe peanut and tree nut allergies."
    )

    result = summarizer.summarize([
        MemoryEvent(content="User is allergic to peanuts."),
        MemoryEvent(content="User is also allergic to tree nuts."),
    ])

    assert result == "User has severe peanut and tree nut allergies."


def test_strips_surrounding_whitespace():
    summarizer, mock_client = make_summarizer()
    mock_client.messages.create.return_value = make_mock_response(
        "\n  User has peanut allergies.  \n"
    )

    result = summarizer.summarize([MemoryEvent(content="User is allergic to peanuts.")])

    assert result == "User has peanut allergies."
```

Create `tests/test_summarization_live.py`:

```python
"""Live integration test for LLMSummarizer against the real Anthropic
API -- as opposed to test_summarization.py, which mocks the client
entirely. Skipped unless ANTHROPIC_API_KEY is set, since it makes a
real, billed API call and needs real credentials.
"""

import os

import pytest

from memory_system.events import MemoryEvent
from memory_system.summarization.llm_based import LLMSummarizer


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="requires a real ANTHROPIC_API_KEY to call the live Anthropic API",
)
def test_summarizes_related_memories_from_real_api():
    summarizer = LLMSummarizer()
    events = [
        MemoryEvent(content="User is allergic to peanuts."),
        MemoryEvent(content="User is also allergic to tree nuts."),
    ]

    result = summarizer.summarize(events)

    assert isinstance(result, str)
    assert len(result) > 0
    assert "peanut" in result.lower() or "nut" in result.lower() or "allerg" in result.lower()
```

Add to the top imports of `tests/test_offline_consolidation.py`:

```python
from memory_system.summarization.base import MemorySummarizer
```

Append to `tests/test_offline_consolidation.py`:

```python
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
    result = graph.related_to("user", max_hops=1)
    assert {e.id for e in result} == {"peanut", "hiking", "quinoa"}
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_summarization.py tests/test_offline_consolidation.py -v -k "compress or summariz"`
Expected: FAIL with `ModuleNotFoundError: No module named 'memory_system.summarization'`

- [ ] **Step 3: Write the implementation**

Create `src/memory_system/summarization/base.py`:

```python
"""Abstract boundary for turning a group of related MemoryEvents into
one dense summary string. Mirrors EntityExtractor's shape
(backends/graph.py) -- the interface just needs to accept events and
return something usable, the hard problem (a good summarization
prompt/model) is decided separately by whichever concrete
implementation you pick.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from ..events import MemoryEvent


class MemorySummarizer(ABC):
    @abstractmethod
    def summarize(self, events: list[MemoryEvent]) -> str:
        ...
```

Create `src/memory_system/summarization/llm_based.py`:

```python
"""LLM-based memory summarization. Requires the `llm` extra (same
anthropic dependency LLMEntityExtractor already needs -- no new
optional-dependency permutation):
    pip install tiered-memory[llm]
"""

from __future__ import annotations

from typing import Optional

from ..events import MemoryEvent
from .base import MemorySummarizer

try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None


SUMMARIZATION_PROMPT = """Summarize the following related memories into one dense, factual paragraph. Preserve concrete facts, names, and preferences. Do not add commentary or filler.

Memories:
{events}

Output only the summary paragraph, nothing else.
"""


class LLMSummarizer(MemorySummarizer):
    """Calls Claude to condense a group of related memories into one
    summary. Note this is a per-group API call -- offline_consolidate()
    may call this once per compression group in a single pass.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-5"):
        if anthropic is None:
            raise ImportError(
                "LLMSummarizer requires the anthropic package. "
                "Install with: pip install tiered-memory[llm]"
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def summarize(self, events: list[MemoryEvent]) -> str:
        formatted = "\n".join(f"- {event.content}" for event in events)
        prompt = SUMMARIZATION_PROMPT.format(events=formatted)

        response = self._client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
```

In `src/memory_system/core.py`, add a new import after the `_find_graph_backend`-related imports (after `from .backends.hybrid import HybridBackend`):

```python
from .summarization.base import MemorySummarizer
```

Add `_connected_components` as a new module-level function, after `_find_graph_backend`:

```python
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
```

Add `compress()` as a new method on `TieredMemory`, placed after `deduplicate()` and before `decay()`:

```python
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
                    # group is skipped
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_summarization.py tests/test_summarization_live.py tests/test_offline_consolidation.py -v`
Expected: `test_summarization.py`: 2 passed. `test_summarization_live.py`: 1 skipped (no `ANTHROPIC_API_KEY` in the dev environment unless set). `test_offline_consolidation.py`: 24 passed (17 from Task 5 + 7 new).

- [ ] **Step 5: Commit**

```bash
git add src/memory_system/summarization/ src/memory_system/core.py tests/test_summarization.py tests/test_summarization_live.py tests/test_offline_consolidation.py
git commit -m "$(cat <<'EOF'
Add MemorySummarizer/LLMSummarizer and TieredMemory.compress()

Sixth piece of offline consolidation. MemorySummarizer mirrors
EntityExtractor's ABC shape exactly; LLMSummarizer reuses the
existing llm extra (same anthropic dependency LLMEntityExtractor
already needs, zero new optional-dependency permutations), same
fail-soft posture as LLMEntityExtractor -- a broken API call skips
that compression group rather than aborting the whole pass.

_connected_components() groups events via BFS over the same
_find_similar_pairs() output deduplicate() uses, at a separate,
looser group_threshold -- compress() runs after deduplicate() (see
Execution order in the spec), so it only ever sees the
already-deduplicated remainder.

compress()'s GraphBackend handling reuses reassign_relationships()
from commit-3, the identical fix as deduplicate()'s -- arguably a
sharper case, since LLM-synthesized summary text is even less likely
than picking one original verbatim to happen to mention every entity
each source contributed.

9 new tests: LLMSummarizer response handling (mocked, matching
test_llm_extractor.py's anthropic-mocking pattern) plus one live test
gated on ANTHROPIC_API_KEY; compress()'s grouping/salience/provenance/
dry_run/fail-soft behavior with a FakeSummarizer double; and a
GraphBackend test proving a 3-source group's entity the fake summary
deliberately omits still survives via reassign_relationships.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 7: `strengthen_connections()` + `GraphBackend.entities_for_event()`/`find_edge()`

**Files:**
- Modify: `src/memory_system/backends/graph.py`
- Modify: `src/memory_system/core.py`
- Modify: `tests/test_graph_backend.py`
- Modify: `tests/test_offline_consolidation.py`

**Interfaces:**
- Consumes: `_find_graph_backend` (Task 5), `ConsolidationReport` (Task 1).
- Produces: `GraphBackend.entities_for_event(self, event_id: str) -> set[str]`, `GraphBackend.find_edge(self, entity_a: str, entity_b: str) -> Optional[Relationship]`. `TieredMemory.strengthen_connections(self, merge_report: Optional[ConsolidationReport] = None, compress_report: Optional[ConsolidationReport] = None, dry_run: bool = False) -> ConsolidationReport`. Consumed by Task 8 (`offline_consolidate()`).

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_graph_backend.py` (append at the end):

```python
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
```

Add to the top imports of `tests/test_offline_consolidation.py`:

```python
from memory_system.events import ConsolidationReport
```

Append to `tests/test_offline_consolidation.py`:

```python
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

    assert graph.find_edge("peanut", "protein").strength == 0.5  # default, untouched


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_graph_backend.py tests/test_offline_consolidation.py -v -k "entities_for_event or find_edge or strengthen"`
Expected: FAIL with `AttributeError: 'GraphBackend' object has no attribute 'entities_for_event'` (and similarly for `find_edge`, `strengthen_connections`).

- [ ] **Step 3: Write the implementation**

In `src/memory_system/backends/graph.py`, add two new methods on `GraphBackend`, placed after `reassign_relationships()` (from Task 3) and before the `# --- graph-native methods` comment:

```python
    def entities_for_event(self, event_id: str) -> set[str]:
        """Entity ids touched by relationships sourced from this
        event, as either source or target."""
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

In `src/memory_system/core.py`, add `strengthen_connections()` as a new method on `TieredMemory`, placed after `compress()` (from Task 6) and before `decay()`:

```python
    def strengthen_connections(
        self,
        merge_report: Optional[ConsolidationReport] = None,
        compress_report: Optional[ConsolidationReport] = None,
        dry_run: bool = False,
    ) -> ConsolidationReport:
        """Strengthens graph connections between entities that ended
        up associated with the same surviving event after this pass's
        deduplicate()/compress() calls. Pass the ConsolidationReports
        those methods returned (from the same pass); called with no
        reports (the default), there is nothing to strengthen and an
        empty ConsolidationReport is returned.

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
                    edge = graph_backend.find_edge(entity_a, entity_b)
                    if edge is None:
                        continue
                    # Only strengthen edges that are NOT part of this consolidation event itself
                    # (those edges' confidence/strength are already handled by the merge/compress).
                    # Strengthen only "external" edges that happen to connect co-mentioned entities.
                    if edge.source_event_id == event_id:
                        continue
                    if not dry_run:
                        edge.strength = min(1.0, edge.strength + 0.1)
                    report.strengthened.append((entity_a, entity_b))

        return report
```

**Correction made while writing this plan, not caught during design
review:** the code block above, exactly as originally specified in the
approved spec, cannot pass the Step 1 test also mandated by that same
spec (`test_strengthen_connections_bumps_edge_between_entities_co_associated_by_merge`,
which asserts a merge strengthens *only* the pre-existing bystander
edge between `peanut` and `protein`). `deduplicate()`'s merge action
calls `self.backend.add(merged_event)` -- which re-runs entity
extraction and creates fresh relationships sourced from the new event
-- *before* `reassign_relationships()` retargets the discarded event's
relationships onto that same new id. So `entities_for_event(new_id)`
ends up covering entities from both the freshly-extracted edge and the
reassigned edge, and the literal algorithm strengthens every pairwise
combination among them, including the edges that are themselves part
of the merge just performed -- not just the one bystander edge the
test expects. Verified empirically while implementing Task 7: running
the literal algorithm against the test's exact scenario produces 3
strengthened pairs, but the test asserts exactly 1. Fix, already
folded into the code block above: skip any edge whose
`source_event_id` equals the `event_id` currently being processed.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_graph_backend.py tests/test_offline_consolidation.py -v`
Expected: `test_graph_backend.py`: all pass (4 new). `test_offline_consolidation.py`: 30 passed (24 from Task 6 + 6 new).

- [ ] **Step 5: Commit**

```bash
git add src/memory_system/backends/graph.py src/memory_system/core.py tests/test_graph_backend.py tests/test_offline_consolidation.py
git commit -m "$(cat <<'EOF'
Add strengthen_connections() and its GraphBackend query helpers

Seventh piece of offline consolidation. entities_for_event() and
find_edge() are new public GraphBackend methods so
strengthen_connections() never has to reach into _edges/_adjacency
directly -- same reasoning as reassign_relationships() existing
instead of core.py touching GraphBackend's internals.

strengthen_connections() takes merge_report/compress_report from the
same pass (both optional, both None by default -- a correction to the
originally-approved spec, made while writing this plan: the original
signature had no way to know which events a pass had touched, so
"entities related by this same pass" was undefined without it). For
each new_id in those reports, every pairwise combination of
entities_for_event(new_id) that already has an edge gets its strength
bumped by +0.1, capped at 1.0. Pairs with no existing edge are
skipped -- consolidation has no principled way to assign a
relation_type to a brand-new edge, that's EntityExtractor's job.

6 new tests: bumping an existing edge between entities co-associated
by a merge, capping at 1.0, dry_run leaving strength untouched, no
reports given returns empty, and non-GraphBackend backends return
empty rather than erroring.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 8: `offline_consolidate()` wrapper + integration tests

**Files:**
- Modify: `src/memory_system/core.py`
- Modify: `tests/test_offline_consolidation.py`

**Interfaces:**
- Consumes: `deduplicate()` (Task 5), `compress()` (Task 6), `strengthen_connections()` (Task 7), all on `TieredMemory`.
- Produces: `TieredMemory.offline_consolidate(self, merge_threshold: float, group_threshold: float, summarizer: Optional[MemorySummarizer] = None, dry_run: bool = False) -> ConsolidationReport`. Terminal method of this plan.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_offline_consolidation.py`:

```python
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
    memory = make_long_term_memory()
    add_long_term(memory, "User is severely allergic to peanuts.")
    add_long_term(memory, "User is severely allergic to peanuts and tree nuts.")

    report = memory.offline_consolidate(merge_threshold=0.3, group_threshold=0.3)

    assert report.compressed == []


def test_offline_consolidate_dry_run_mutates_nothing():
    memory = make_long_term_memory()
    a = add_long_term(memory, "User is severely allergic to peanuts.")
    b = add_long_term(memory, "User is severely allergic to peanuts and tree nuts.")

    report = memory.offline_consolidate(merge_threshold=0.3, group_threshold=0.9, dry_run=True)

    assert len(report.merged) == 1
    assert report.merged[0][2] is None
    remaining_ids = {e.id for e in memory.backend.get_all(tier=MemoryTier.LONG_TERM)}
    assert remaining_ids == {a.id, b.id}


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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_offline_consolidation.py -v -k offline_consolidate`
Expected: FAIL with `AttributeError: 'TieredMemory' object has no attribute 'offline_consolidate'`

- [ ] **Step 3: Write the implementation**

In `src/memory_system/core.py`, add `offline_consolidate()` as a new method on `TieredMemory`, placed after `strengthen_connections()` (from Task 7) and before `decay()`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_offline_consolidation.py -v`
Expected: 34 passed (30 from Task 7 + 4 new)

Then run the full suite to confirm no regressions anywhere in the project:

Run: `pytest tests/ -v`
Expected: all prior tests still pass, plus every test added across this plan's 8 tasks.

- [ ] **Step 5: Commit**

```bash
git add src/memory_system/core.py tests/test_offline_consolidation.py
git commit -m "$(cat <<'EOF'
Add offline_consolidate() wrapper, completing offline consolidation

Eighth and final piece. Runs deduplicate() -> compress() ->
strengthen_connections() in a fixed, hardcoded order -- not a
parameter, not configurable. GraphBackend.remove() already deletes
every edge whose source_event_id matches the removed event (existing
behavior, not something this plan added); dedup and compress both
call remove() on their source events, so by the time
strengthen_connections() runs last, every edge belonging to a
merged-away or summarized-away event has already been pruned or
reassigned. Running it first or interleaved would strengthen edges
moments before remove() deletes them.

4 new tests (34 total in test_offline_consolidation.py): the wrapper
runs both stages and returns a combined report, compress is skipped
entirely when no summarizer is given, dry_run mutates nothing across
all three stages, and the load-bearing execution-order test -- a
scenario where, if strengthen_connections() used stale pre-merge
event associations instead of the merged event's final ones, it would
strengthen or miss the wrong edge. Asserting against the actual
post-merge graph state (not just "dedup ran before strengthen" as a
call-order fact) confirms the order is enforced in a way that
matters, not just documented.

Full suite green after this commit.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```
