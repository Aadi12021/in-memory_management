# Hybrid Retrieval Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `HybridBackend`, a `MemoryBackend` that combines an `InMemoryBackend`-style lexical backend and a `ChromaBackend`-style semantic backend into one ranked result via Reciprocal Rank Fusion (RRF).

**Architecture:** A pure `reciprocal_rank_fusion()` function (no backend dependency, fully unit-testable with synthetic data) does the actual fusion math. `HybridBackend(MemoryBackend)` composes two `MemoryBackend` instances, mirrors all writes to both with fail-loud sync-error handling, and calls `reciprocal_rank_fusion()` inside `query()` after oversampling each backend.

**Tech Stack:** Pure Python (stdlib only for `HybridBackend`/`reciprocal_rank_fusion` themselves — no new dependency). Tests use `InMemoryBackend` (zero dependencies) for the bulk of coverage, plus a final real-`ChromaBackend` integration pass gated on the existing `chroma` extra.

## Global Constraints

- Spec: `docs/superpowers/specs/2026-07-28-hybrid-retrieval-design.md` — read it if anything below is ambiguous.
- Python floor is `>=3.9` (`pyproject.toml`). Every new file MUST start with `from __future__ import annotations` — a prior commit (`6ef6dc0`) broke CI on 3.9 specifically because a *test* file used `X | None` without it. This applies to test files too, not just `src/`.
- RRF damping constant: `k=60` (spec-mandated default, matches Elasticsearch/Weaviate/Azure AI Search defaults).
- Oversampling: `fetch_k = top_k * fetch_k_multiplier`, `fetch_k_multiplier` defaults to `5`.
- Mirroring order for `add`/`update_tier`/`remove`: `lexical_backend` first, then `semantic_backend`.
- Fail-loud rule: if `lexical_backend`'s write throws, propagate the original exception unwrapped (nothing diverged yet). If `semantic_backend`'s write throws after `lexical_backend` succeeded, raise `HybridBackendSyncError` wrapping the original via `raise ... from`.
- `get_all()` delegates to `lexical_backend` only, with an explicit docstring caveat stating that and stating the assumption that mirroring held.
- `HybridBackend` itself takes on no new optional dependency and needs no new `pyproject.toml` extra — whatever `semantic_backend` needs (e.g. `chromadb`) is that instance's own concern.
- New file: `src/memory_system/backends/hybrid.py` (one file per backend, matching `memory.py`/`chroma.py`/`graph.py`).

---

### Task 1: `reciprocal_rank_fusion()` pure function

**Files:**
- Create: `src/memory_system/backends/hybrid.py`
- Test: `tests/test_hybrid_backend.py`

**Interfaces:**
- Produces: `reciprocal_rank_fusion(result_lists: list[list[RetrievalResult]], k: int = 60) -> list[RetrievalResult]` — used by Task 2's `HybridBackend.query()`.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_hybrid_backend.py`:

```python
"""Tests for reciprocal_rank_fusion() and HybridBackend."""

from __future__ import annotations

from memory_system.backends.hybrid import reciprocal_rank_fusion
from memory_system.events import MemoryEvent, MemoryTier, RetrievalResult


def make_result(content, score, event_id=None, tier=MemoryTier.WORKING):
    event = MemoryEvent(content=content, tier=tier)
    if event_id is not None:
        event.id = event_id
    return RetrievalResult(event=event, score=score, tier=tier)


def test_rrf_empty_lists_returns_empty():
    assert reciprocal_rank_fusion([]) == []
    assert reciprocal_rank_fusion([[], []]) == []


def test_rrf_single_list_uses_1_over_k_plus_rank():
    a = make_result("alpha", score=0.9, event_id="a")
    b = make_result("beta", score=0.5, event_id="b")

    fused = reciprocal_rank_fusion([[a, b]], k=60)

    assert [r.event.id for r in fused] == ["a", "b"]
    assert fused[0].score == 1.0 / 61
    assert fused[1].score == 1.0 / 62


def test_rrf_custom_k_changes_damping():
    a = make_result("alpha", score=0.9, event_id="a")

    fused = reciprocal_rank_fusion([[a]], k=1)

    assert fused[0].score == 1.0 / 2


def test_rrf_document_found_by_both_lists_scores_higher_than_by_one():
    shared = make_result("shared doc", score=0.9, event_id="shared")
    only_in_first = make_result("first-only doc", score=0.9, event_id="first_only")

    fused = reciprocal_rank_fusion([[shared, only_in_first], [shared]])
    fused_by_id = {r.event.id: r.score for r in fused}

    assert fused_by_id["shared"] > fused_by_id["first_only"]


def test_rrf_document_found_by_only_one_list_still_appears():
    lexical_only = make_result("lexical match", score=0.8, event_id="lexical_only")
    semantic_only = make_result("semantic match", score=0.6, event_id="semantic_only")

    fused = reciprocal_rank_fusion([[lexical_only], [semantic_only]])

    assert {r.event.id for r in fused} == {"lexical_only", "semantic_only"}


def test_rrf_sorts_descending_by_summed_score_across_lists():
    a = make_result("alpha", score=0.9, event_id="a")
    b = make_result("beta", score=0.5, event_id="b")
    list_a = [a, b]  # a rank1, b rank2
    list_b = [b]     # b rank1

    fused = reciprocal_rank_fusion([list_a, list_b])

    # b: 1/61 + 1/62 (two contributions) vs a: 1/61 (one contribution) -> b wins
    # even though a ranked #1 in list_a and b only ranked #2 there.
    assert [r.event.id for r in fused] == ["b", "a"]


def test_rrf_preserves_event_and_tier_from_source_result():
    result = make_result("some content", score=0.9, event_id="a", tier=MemoryTier.LONG_TERM)

    fused = reciprocal_rank_fusion([[result]])

    assert fused[0].event is result.event
    assert fused[0].tier == MemoryTier.LONG_TERM
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_hybrid_backend.py -v`
Expected: FAIL / ERROR with `ModuleNotFoundError: No module named 'memory_system.backends.hybrid'`

- [ ] **Step 3: Write the implementation**

Create `src/memory_system/backends/hybrid.py`:

```python
"""Combines a lexical MemoryBackend (e.g. InMemoryBackend) and a
semantic MemoryBackend (e.g. ChromaBackend) into one ranked result via
Reciprocal Rank Fusion (RRF). See
docs/superpowers/specs/2026-07-28-hybrid-retrieval-design.md for the
full design rationale -- in short, the two backends' raw scores aren't
on comparable scales (TF-IDF cosine similarity vs. embedding-distance-
derived similarity), so RRF combines by rank position within each
backend's own ordering instead of by raw score magnitude.
"""

from __future__ import annotations

from ..events import RetrievalResult


def reciprocal_rank_fusion(
    result_lists: list[list[RetrievalResult]],
    k: int = 60,
) -> list[RetrievalResult]:
    """Fuses multiple ranked RetrievalResult lists into one list, summing
    1 / (k + rank) for each event across every list it appears in (rank
    is 1-indexed within that list). Events are matched across lists by
    event.id. An event absent from a list simply contributes no term for
    that list -- no special-casing needed for empty or partial lists.
    k=60 is the standard RRF damping constant (also the default used by
    Elasticsearch/Weaviate/Azure AI Search hybrid search).

    Returns events sorted descending by summed score. The returned
    RetrievalResult reuses the event/tier from wherever that event.id
    was first encountered across the input lists.
    """
    scores: dict[str, float] = {}
    first_seen: dict[str, RetrievalResult] = {}

    for result_list in result_lists:
        for rank, result in enumerate(result_list, start=1):
            event_id = result.event.id
            scores[event_id] = scores.get(event_id, 0.0) + 1.0 / (k + rank)
            if event_id not in first_seen:
                first_seen[event_id] = result

    fused = [
        RetrievalResult(
            event=first_seen[event_id].event,
            score=score,
            tier=first_seen[event_id].tier,
        )
        for event_id, score in scores.items()
    ]
    fused.sort(key=lambda r: r.score, reverse=True)
    return fused
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_hybrid_backend.py -v`
Expected: 7 passed

- [ ] **Step 5: Commit**

```bash
git add src/memory_system/backends/hybrid.py tests/test_hybrid_backend.py
git commit -m "$(cat <<'EOF'
Add reciprocal_rank_fusion(): pure RRF fusion for hybrid retrieval

First piece of HybridBackend (see
docs/superpowers/specs/2026-07-28-hybrid-retrieval-design.md). Pure
function, no backend dependency, so the fusion math is fully unit
tested with synthetic RetrievalResult lists -- no InMemoryBackend or
ChromaBackend construction needed. k=60 default matches the standard
RRF damping constant used by Elasticsearch/Weaviate/Azure AI Search.

7 tests, TDD: empty lists, single-list rank->score formula, custom k,
agreement across lists scoring higher than a single-list hit, a
document found by only one list still surfacing, fused sort order
overriding a single list's own order when agreement changes the
outcome, and event/tier passthrough.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 2: `HybridBackend` class — constructor, mirrored writes, fail-loud sync errors, `get_all`, `query`

**Files:**
- Modify: `src/memory_system/backends/hybrid.py`
- Modify: `tests/test_hybrid_backend.py`

**Interfaces:**
- Consumes: `reciprocal_rank_fusion(result_lists, k)` from Task 1.
- Produces: `HybridBackend(lexical_backend, semantic_backend, rrf_k=60, fetch_k_multiplier=5)` implementing `MemoryBackend`; `HybridBackendSyncError(method, event_id, succeeded, failed, original)` with those four public attributes plus `.original`, raised on partial write failure. Both consumed by Task 3 and Task 4.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_hybrid_backend.py` (below the existing RRF tests), updating the top imports first:

```python
from __future__ import annotations

import pytest

from memory_system.backends.base import MemoryBackend
from memory_system.backends.hybrid import (
    HybridBackend,
    HybridBackendSyncError,
    reciprocal_rank_fusion,
)
from memory_system.backends.memory import InMemoryBackend
from memory_system.events import MemoryEvent, MemoryTier, RetrievalResult
```

Then append:

```python
class RaisingBackend(MemoryBackend):
    """Test double: a MemoryBackend whose configured methods raise
    instead of executing, used to force HybridBackend's mirroring
    failure path deterministically.
    """

    def __init__(self, fail_on=frozenset()):
        self.fail_on = fail_on
        self.events: dict[str, MemoryEvent] = {}

    def add(self, event):
        if "add" in self.fail_on:
            raise RuntimeError("simulated add failure")
        self.events[event.id] = event

    def get_all(self, tier=None):
        events = list(self.events.values())
        if tier is not None:
            events = [e for e in events if e.tier == tier]
        return events

    def query(self, query, top_k=5, tier=None):
        return []

    def update_tier(self, event_id, new_tier):
        if "update_tier" in self.fail_on:
            raise RuntimeError("simulated update_tier failure")
        if event_id in self.events:
            self.events[event_id].tier = new_tier

    def remove(self, event_id):
        if "remove" in self.fail_on:
            raise RuntimeError("simulated remove failure")
        self.events.pop(event_id, None)


def test_add_mirrors_event_to_both_backends():
    lexical = InMemoryBackend()
    semantic = InMemoryBackend()
    hybrid = HybridBackend(lexical_backend=lexical, semantic_backend=semantic)
    event = MemoryEvent(content="a shared fact")

    hybrid.add(event)

    assert lexical.get_all()[0].id == event.id
    assert semantic.get_all()[0].id == event.id


def test_add_raises_sync_error_when_semantic_backend_fails():
    lexical = InMemoryBackend()
    semantic = RaisingBackend(fail_on={"add"})
    hybrid = HybridBackend(lexical_backend=lexical, semantic_backend=semantic)
    event = MemoryEvent(content="a fact")

    with pytest.raises(HybridBackendSyncError) as exc_info:
        hybrid.add(event)

    exc = exc_info.value
    assert exc.method == "add"
    assert exc.event_id == event.id
    assert exc.succeeded == "lexical"
    assert exc.failed == "semantic"
    assert isinstance(exc.__cause__, RuntimeError)
    # lexical already has it -- that IS the divergence being reported, not rolled back
    assert lexical.get_all()[0].id == event.id


def test_add_propagates_original_exception_when_lexical_backend_fails():
    lexical = RaisingBackend(fail_on={"add"})
    semantic = InMemoryBackend()
    hybrid = HybridBackend(lexical_backend=lexical, semantic_backend=semantic)
    event = MemoryEvent(content="a fact")

    with pytest.raises(RuntimeError) as exc_info:
        hybrid.add(event)

    assert not isinstance(exc_info.value, HybridBackendSyncError)
    assert semantic.get_all() == []  # never touched -- nothing diverged


def test_update_tier_mirrors_to_both_backends():
    lexical = InMemoryBackend()
    semantic = InMemoryBackend()
    hybrid = HybridBackend(lexical_backend=lexical, semantic_backend=semantic)
    event = MemoryEvent(content="a fact", tier=MemoryTier.WORKING)
    hybrid.add(event)

    hybrid.update_tier(event.id, MemoryTier.LONG_TERM)

    assert lexical.get_all()[0].tier == MemoryTier.LONG_TERM
    assert semantic.get_all()[0].tier == MemoryTier.LONG_TERM


def test_update_tier_raises_sync_error_when_semantic_backend_fails():
    lexical = InMemoryBackend()
    semantic = RaisingBackend(fail_on={"update_tier"})
    hybrid = HybridBackend(lexical_backend=lexical, semantic_backend=semantic)
    event = MemoryEvent(content="a fact")
    hybrid.add(event)

    with pytest.raises(HybridBackendSyncError) as exc_info:
        hybrid.update_tier(event.id, MemoryTier.LONG_TERM)

    assert exc_info.value.method == "update_tier"
    assert exc_info.value.succeeded == "lexical"
    assert exc_info.value.failed == "semantic"


def test_remove_mirrors_to_both_backends():
    lexical = InMemoryBackend()
    semantic = InMemoryBackend()
    hybrid = HybridBackend(lexical_backend=lexical, semantic_backend=semantic)
    event = MemoryEvent(content="a fact")
    hybrid.add(event)

    hybrid.remove(event.id)

    assert lexical.get_all() == []
    assert semantic.get_all() == []


def test_remove_raises_sync_error_when_semantic_backend_fails():
    lexical = InMemoryBackend()
    semantic = RaisingBackend(fail_on={"remove"})
    hybrid = HybridBackend(lexical_backend=lexical, semantic_backend=semantic)
    event = MemoryEvent(content="a fact")
    hybrid.add(event)

    with pytest.raises(HybridBackendSyncError) as exc_info:
        hybrid.remove(event.id)

    assert exc_info.value.method == "remove"
    assert exc_info.value.succeeded == "lexical"
    assert exc_info.value.failed == "semantic"


def test_get_all_delegates_to_lexical_backend_only():
    lexical = InMemoryBackend()
    semantic = InMemoryBackend()
    hybrid = HybridBackend(lexical_backend=lexical, semantic_backend=semantic)
    lexical_only_event = MemoryEvent(content="only in lexical")
    lexical.add(lexical_only_event)  # bypasses HybridBackend.add on purpose

    result = hybrid.get_all()

    assert [e.id for e in result] == [lexical_only_event.id]


def test_get_all_filters_by_tier():
    lexical = InMemoryBackend()
    semantic = InMemoryBackend()
    hybrid = HybridBackend(lexical_backend=lexical, semantic_backend=semantic)
    working = MemoryEvent(content="working fact", tier=MemoryTier.WORKING)
    long_term = MemoryEvent(content="long term fact", tier=MemoryTier.LONG_TERM)
    hybrid.add(working)
    hybrid.add(long_term)

    result = hybrid.get_all(tier=MemoryTier.LONG_TERM)

    assert [e.id for e in result] == [long_term.id]


def test_query_fuses_results_from_two_real_backends():
    lexical = InMemoryBackend()
    semantic = InMemoryBackend()
    hybrid = HybridBackend(lexical_backend=lexical, semantic_backend=semantic)
    allergic = MemoryEvent(content="The user is severely allergic to peanuts and tree nuts.")
    hiking = MemoryEvent(content="The user enjoys hiking on weekends with peanut butter snacks.")
    for event in (allergic, hiking):
        hybrid.add(event)

    results = hybrid.query("peanuts nuts allergy", top_k=2)

    assert len(results) == 2
    assert results[0].event.id == allergic.id
    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_hybrid_backend.py -v`
Expected: the 7 Task 1 tests still pass; the new tests FAIL/ERROR with `ImportError: cannot import name 'HybridBackend' from 'memory_system.backends.hybrid'`

- [ ] **Step 3: Write the implementation**

Append to `src/memory_system/backends/hybrid.py` (add these imports to the top, alongside the existing `from ..events import RetrievalResult` line -- replace that line with the fuller one below):

```python
from __future__ import annotations

from typing import Optional

from ..events import MemoryEvent, MemoryTier, RetrievalResult
from .base import MemoryBackend
```

Then append the class and exception:

```python
class HybridBackendSyncError(Exception):
    """Raised when a HybridBackend mutating call (add/update_tier/
    remove) succeeds on one internal backend but fails on the other,
    leaving them out of sync. The successful side already has state
    the other doesn't -- check `succeeded`/`failed` before deciding
    how to recover.
    """

    def __init__(self, method: str, event_id: str, succeeded: str, failed: str, original: Exception):
        self.method = method
        self.event_id = event_id
        self.succeeded = succeeded
        self.failed = failed
        self.original = original
        super().__init__(
            f"HybridBackend.{method}(event_id={event_id!r}) succeeded on "
            f"{succeeded}_backend but failed on {failed}_backend: {original!r}. "
            "The two backends are now out of sync."
        )


class HybridBackend(MemoryBackend):
    """Combines a lexical backend (e.g. InMemoryBackend) and a semantic
    backend (e.g. ChromaBackend) via Reciprocal Rank Fusion. No new
    optional extra required -- whatever dependency semantic_backend
    needs (e.g. chromadb) is its own concern, not HybridBackend's.

    Mirroring invariant: add/update_tier/remove write to lexical_backend
    first, then semantic_backend. If lexical_backend's write fails,
    nothing has diverged (semantic_backend was never touched) and the
    original exception propagates as-is. If lexical_backend succeeds
    but semantic_backend's write fails, the two backends are now out of
    sync -- raises HybridBackendSyncError wrapping the original
    exception rather than silently leaving them divergent, since RRF's
    correctness depends on matching event.id across both stores.
    """

    def __init__(
        self,
        lexical_backend: MemoryBackend,
        semantic_backend: MemoryBackend,
        rrf_k: int = 60,
        fetch_k_multiplier: int = 5,
    ):
        self.lexical_backend = lexical_backend
        self.semantic_backend = semantic_backend
        self.rrf_k = rrf_k
        self.fetch_k_multiplier = fetch_k_multiplier

    def add(self, event: MemoryEvent) -> None:
        self.lexical_backend.add(event)
        try:
            self.semantic_backend.add(event)
        except Exception as exc:
            raise HybridBackendSyncError("add", event.id, "lexical", "semantic", exc) from exc

    def get_all(self, tier: Optional[MemoryTier] = None) -> list[MemoryEvent]:
        """Delegates to lexical_backend only. Assumes the mirroring
        invariant held (see add/update_tier/remove) -- if it didn't,
        this won't reflect semantic_backend's state.
        """
        return self.lexical_backend.get_all(tier=tier)

    def query(
        self, query: str, top_k: int = 5, tier: Optional[MemoryTier] = None
    ) -> list[RetrievalResult]:
        fetch_k = top_k * self.fetch_k_multiplier
        lexical_results = self.lexical_backend.query(query, top_k=fetch_k, tier=tier)
        semantic_results = self.semantic_backend.query(query, top_k=fetch_k, tier=tier)
        fused = reciprocal_rank_fusion([lexical_results, semantic_results], k=self.rrf_k)
        return fused[:top_k]

    def update_tier(self, event_id: str, new_tier: MemoryTier) -> None:
        self.lexical_backend.update_tier(event_id, new_tier)
        try:
            self.semantic_backend.update_tier(event_id, new_tier)
        except Exception as exc:
            raise HybridBackendSyncError("update_tier", event_id, "lexical", "semantic", exc) from exc

    def remove(self, event_id: str) -> None:
        self.lexical_backend.remove(event_id)
        try:
            self.semantic_backend.remove(event_id)
        except Exception as exc:
            raise HybridBackendSyncError("remove", event_id, "lexical", "semantic", exc) from exc
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_hybrid_backend.py -v`
Expected: 17 passed (7 from Task 1 + 10 new)

- [ ] **Step 5: Commit**

```bash
git add src/memory_system/backends/hybrid.py tests/test_hybrid_backend.py
git commit -m "$(cat <<'EOF'
Add HybridBackend: composed MemoryBackend with fail-loud mirroring

Implements the class half of hybrid retrieval (see
docs/superpowers/specs/2026-07-28-hybrid-retrieval-design.md): wraps
a lexical_backend + semantic_backend pair, implements the full
MemoryBackend interface, and uses reciprocal_rank_fusion() (added in
the prior commit) inside query().

Mirroring invariant: add/update_tier/remove write lexical_backend
first, semantic_backend second. A first-write failure propagates
unwrapped (nothing diverged). A second-write failure raises
HybridBackendSyncError, wrapping the original exception via `raise
... from`, stating which backend succeeded and which failed --
silent divergence would otherwise be worse than a caller having to
handle an exception, since RRF's correctness depends on matching
event.id across both stores. get_all() delegates to lexical_backend
only, documented explicitly in its docstring.

10 new tests (17 total): mirroring on all three mutators, the
fail-loud/propagate-unwrapped split on both sides of the write order,
get_all()'s delegation and tier filtering, and a real-backend (two
InMemoryBackend instances) query() smoke test proving the fusion
wiring produces a sorted, deduplicated result.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 3: `query()` oversampling, tier passthrough, and single-backend rescue coverage

**Files:**
- Modify: `tests/test_hybrid_backend.py`

**Interfaces:**
- Consumes: `HybridBackend`, `HybridBackendSyncError` from Task 2. No new production code -- this task only adds test coverage for `HybridBackend.query()` behavior that Task 2's single smoke test didn't isolate (exact `fetch_k` sizing, `tier` passthrough, and the single-backend-only match case that's the core reason hybrid retrieval exists).

- [ ] **Step 1: Write the failing tests**

Add to the top imports of `tests/test_hybrid_backend.py`:

```python
from unittest.mock import patch
```

Append to `tests/test_hybrid_backend.py`:

```python
class StubBackend(MemoryBackend):
    """Test double: a MemoryBackend whose query() always returns a
    fixed, pre-set list of results regardless of the query string --
    used to simulate "the semantic backend found something the
    lexical backend didn't" deterministically, without needing a real
    embedding model.
    """

    def __init__(self, results):
        self.results = results
        self.events: dict[str, MemoryEvent] = {r.event.id: r.event for r in results}

    def add(self, event):
        self.events[event.id] = event

    def get_all(self, tier=None):
        events = list(self.events.values())
        if tier is not None:
            events = [e for e in events if e.tier == tier]
        return events

    def query(self, query, top_k=5, tier=None):
        return self.results[:top_k]

    def update_tier(self, event_id, new_tier):
        if event_id in self.events:
            self.events[event_id].tier = new_tier

    def remove(self, event_id):
        self.events.pop(event_id, None)


def test_query_requests_fetch_k_not_top_k_from_each_backend():
    lexical = InMemoryBackend()
    semantic = InMemoryBackend()
    hybrid = HybridBackend(lexical_backend=lexical, semantic_backend=semantic, fetch_k_multiplier=5)
    for i in range(10):
        hybrid.add(MemoryEvent(content=f"document number {i} about cats and kittens"))

    with patch.object(lexical, "query", wraps=lexical.query) as lexical_spy, \
         patch.object(semantic, "query", wraps=semantic.query) as semantic_spy:
        hybrid.query("cats", top_k=2)

    lexical_spy.assert_called_once_with("cats", top_k=10, tier=None)
    semantic_spy.assert_called_once_with("cats", top_k=10, tier=None)


def test_query_passes_tier_through_to_both_backends():
    lexical = InMemoryBackend()
    semantic = InMemoryBackend()
    hybrid = HybridBackend(lexical_backend=lexical, semantic_backend=semantic)

    with patch.object(lexical, "query", wraps=lexical.query) as lexical_spy, \
         patch.object(semantic, "query", wraps=semantic.query) as semantic_spy:
        hybrid.query("cats", top_k=3, tier=MemoryTier.LONG_TERM)

    lexical_spy.assert_called_once_with("cats", top_k=15, tier=MemoryTier.LONG_TERM)
    semantic_spy.assert_called_once_with("cats", top_k=15, tier=MemoryTier.LONG_TERM)


def test_query_returns_document_found_by_only_one_backend():
    lexical = InMemoryBackend()  # empty -- simulates zero lexical overlap
    semantic_only_event = MemoryEvent(content="a fact only the semantic backend would find")
    semantic = StubBackend(
        results=[RetrievalResult(event=semantic_only_event, score=0.9, tier=semantic_only_event.tier)]
    )
    hybrid = HybridBackend(lexical_backend=lexical, semantic_backend=semantic)

    results = hybrid.query("food I can't eat", top_k=5)

    result_ids = {r.event.id for r in results}
    assert semantic_only_event.id in result_ids


def test_query_truncates_fused_results_to_top_k():
    lexical = InMemoryBackend()
    semantic = InMemoryBackend()
    hybrid = HybridBackend(lexical_backend=lexical, semantic_backend=semantic)
    for i in range(10):
        event = MemoryEvent(content=f"document {i} about cats")
        lexical.add(event)
        semantic.add(event)

    results = hybrid.query("cats", top_k=3)

    assert len(results) == 3
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `pytest tests/test_hybrid_backend.py -v`
Expected: the 17 existing tests still pass; `StubBackend`-dependent and spy-based tests FAIL if there's a naming/import typo -- if `HybridBackend.query()` already oversamples correctly per Task 2's implementation, these should actually PASS immediately since no new production code is needed here. Confirm they pass for the right reason by temporarily checking: if `test_query_requests_fetch_k_not_top_k_from_each_backend` fails, it means Task 2's `query()` implementation doesn't multiply by `fetch_k_multiplier` correctly -- re-check that step before proceeding.

- [ ] **Step 3: No implementation changes needed**

Task 2's `query()` implementation already does the oversampling and fusion this task tests. This task exists to add coverage, not new code.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_hybrid_backend.py -v`
Expected: 21 passed (17 from Tasks 1-2 + 4 new)

- [ ] **Step 5: Commit**

```bash
git add tests/test_hybrid_backend.py
git commit -m "$(cat <<'EOF'
Add HybridBackend.query() coverage: fetch_k sizing, tier passthrough, single-backend rescue

Task 2 landed a working query() with one smoke test; this fills in
the specific behaviors the spec calls out as the actual point of
hybrid retrieval:

- fetch_k_multiplier is applied (top_k=2 -> each backend queried with
  top_k=10), verified via wrapping spies on both backends' query().
- tier filters pass through to both backends unchanged.
- a document found by only one backend (simulating zero lexical
  overlap, via a StubBackend standing in for a semantic hit the
  lexical side would never surface) still appears in the fused
  result -- the exact "food I can't eat" / "allergic to peanuts"
  scenario from the design discussion.
- fused results truncate to top_k after fusion, not before.

4 new tests (21 total). No production code changes -- Task 2's
query() already does this; this task is coverage only.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```

---

### Task 4: real `ChromaBackend` integration test + package export

**Files:**
- Create: `tests/test_hybrid_backend_chroma_integration.py`
- Modify: `src/memory_system/backends/__init__.py`

**Interfaces:**
- Consumes: `HybridBackend` from Task 2, `ChromaBackend`/`InMemoryBackend` (existing).
- Produces: `HybridBackend`, `HybridBackendSyncError` importable from `memory_system.backends` (public package path), matching how `InMemoryBackend`/`MemoryBackend` are already exposed there.

- [ ] **Step 1: Write the failing test**

Create `tests/test_hybrid_backend_chroma_integration.py`:

```python
"""Integration tests for HybridBackend against a real ChromaBackend
(semantic side) and InMemoryBackend (lexical side). Requires the
`chroma` extra:
    pip install -e ".[chroma]"
Skipped automatically if chromadb isn't installed.

Each test uses its own randomly-named collection, same reasoning as
test_chroma_backend.py: chromadb.Client()'s non-persistent client
shares its underlying store across ChromaBackend instances built with
the same collection_name in one process.
"""

from __future__ import annotations

import uuid

import pytest

pytest.importorskip("chromadb")

from memory_system.backends.chroma import ChromaBackend
from memory_system.backends.hybrid import HybridBackend
from memory_system.backends.memory import InMemoryBackend
from memory_system.events import MemoryEvent


def make_hybrid_backend():
    lexical = InMemoryBackend()
    semantic = ChromaBackend(collection_name=f"test_hybrid_{uuid.uuid4().hex}")
    return HybridBackend(lexical_backend=lexical, semantic_backend=semantic)


def test_add_mirrors_to_both_real_backends():
    hybrid = make_hybrid_backend()
    event = MemoryEvent(content="a fact for both backends")

    hybrid.add(event)

    assert hybrid.lexical_backend.get_all()[0].id == event.id
    assert hybrid.semantic_backend.get_all()[0].id == event.id


def test_query_ranks_more_relevant_document_higher():
    hybrid = make_hybrid_backend()
    allergic = MemoryEvent(content="The user is severely allergic to peanuts and tree nuts.")
    hiking = MemoryEvent(content="The user enjoys hiking on weekends.")
    weather = MemoryEvent(content="The weather forecast for tomorrow is sunny.")
    for event in (allergic, hiking, weather):
        hybrid.add(event)

    results = hybrid.query("peanuts nuts allergy", top_k=3)

    assert results[0].event.id == allergic.id


def test_query_finds_semantically_related_content_with_no_lexical_overlap():
    hybrid = make_hybrid_backend()
    hybrid.add(MemoryEvent(content="User is allergic to peanuts."))
    hybrid.add(MemoryEvent(content="The weather forecast for tomorrow is sunny."))

    results = hybrid.query("food the user cannot safely eat", top_k=2)

    assert len(results) >= 1
    assert any("peanut" in str(r.event.content).lower() for r in results)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_hybrid_backend_chroma_integration.py -v`
Expected: if `chromadb` is installed, FAIL with `ModuleNotFoundError: No module named 'memory_system.backends.hybrid'` only if Task 2 somehow didn't land -- otherwise this should already pass on the `hybrid.py` side. If `chromadb` isn't installed, expect `3 skipped`.

- [ ] **Step 3: Wire `HybridBackend` into the package's public exports**

Modify `src/memory_system/backends/__init__.py`:

```python
from .base import MemoryBackend
from .hybrid import HybridBackend, HybridBackendSyncError
from .memory import InMemoryBackend

__all__ = ["MemoryBackend", "InMemoryBackend", "HybridBackend", "HybridBackendSyncError"]

try:
    from .chroma import ChromaBackend  # noqa: F401
    __all__.append("ChromaBackend")
except ImportError:
    pass
```

This is unconditional (not wrapped in `try/except ImportError`) because `hybrid.py` itself has no dependency beyond `..events` and `.base` -- unlike `ChromaBackend`, which needs the conditional guard because it directly imports `chromadb`.

- [ ] **Step 4: Run tests to verify they pass**

Run: `pytest tests/test_hybrid_backend_chroma_integration.py -v`
Expected (if `chromadb` installed via `pip install -e ".[dev,chroma]"`): 3 passed
Expected (if not installed): 3 skipped

Then run the full suite to confirm no regressions:

Run: `pytest tests/ -v`
Expected: all prior tests still pass, plus the 21 from `test_hybrid_backend.py` and the 3 (or skipped) from `test_hybrid_backend_chroma_integration.py`.

- [ ] **Step 5: Commit**

```bash
git add tests/test_hybrid_backend_chroma_integration.py src/memory_system/backends/__init__.py
git commit -m "$(cat <<'EOF'
Wire HybridBackend into package exports; add real ChromaBackend integration test

Exposes HybridBackend/HybridBackendSyncError at
memory_system.backends, matching how InMemoryBackend/MemoryBackend
are already exported there. Unconditional import, not guarded by
try/except ImportError like ChromaBackend's -- hybrid.py itself has
no dependency beyond ..events and .base.

New integration test file (mirrors test_chroma_backend.py's
pytest.importorskip("chromadb") pattern, skipped if chromadb isn't
installed) proves the actual pairing the design targets -- a real
InMemoryBackend + a real ChromaBackend -- works end to end: mirrored
add(), TF-IDF-favored ranking still comes through on lexical matches,
and a query with no lexical overlap at all still surfaces the
semantically related fact via the real embedding-based backend.

Full suite green after this: see test output for exact count.

Co-Authored-By: Claude Sonnet 5 <noreply@anthropic.com>
EOF
)"
```
