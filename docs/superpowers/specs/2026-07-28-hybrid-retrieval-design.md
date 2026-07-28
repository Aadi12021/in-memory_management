# Hybrid retrieval design

Date: 2026-07-28
Status: approved, not yet implemented

## Problem

`InMemoryBackend` (TF-IDF + cosine similarity) and `ChromaBackend`
(embedding-based semantic search) each miss what the other catches.
TF-IDF misses "food I can't eat" matching a stored "User is allergic
to peanuts." fact (zero lexical overlap); embeddings can miss exact
rare-term matches TF-IDF's IDF weighting would nail. `tiered-memory`
currently makes users pick one `MemoryBackend` per `TieredMemory`
instance. This adds a third backend, `HybridBackend`, that combines
both signals into one ranked result, similar in spirit to how mem0
fuses multiple retrieval signals in parallel.

## Interface: `HybridBackend(MemoryBackend)`

`TieredMemory` holds exactly one `backend: MemoryBackend`
(`core.py:82` calls `self.backend.query(...)`). A composed class that
itself implements `MemoryBackend` fits that shape without changing
`TieredMemory` at all — the same pattern `GraphBackend` already uses,
composing an `EntityExtractor` internally while exposing the plain
`MemoryBackend` interface externally. `HybridBackend` composes two
backend instances instead of an extractor.

```python
class HybridBackend(MemoryBackend):
    def __init__(
        self,
        lexical_backend: MemoryBackend,
        semantic_backend: MemoryBackend,
        rrf_k: int = 60,
        fetch_k_multiplier: int = 5,
    ):
        ...

    def add(self, event: MemoryEvent) -> None: ...
    def get_all(self, tier: Optional[MemoryTier] = None) -> list[MemoryEvent]: ...
    def query(self, query: str, top_k: int = 5, tier: Optional[MemoryTier] = None) -> list[RetrievalResult]: ...
    def update_tier(self, event_id: str, new_tier: MemoryTier) -> None: ...
    def remove(self, event_id: str) -> None: ...
```

Design decisions:

- **Constructor takes two `MemoryBackend`-typed instances**, named by
  role (`lexical_backend`, `semantic_backend`) rather than hardcoded
  to `InMemoryBackend`/`ChromaBackend` specifically. Consistent with
  the "each piece is swappable" architecture already documented in
  the README; costs nothing, and in practice these will almost always
  be an `InMemoryBackend` and a `ChromaBackend`.
- **No new optional extra.** `HybridBackend` has no direct dependency
  on `chromadb` — that dependency lives entirely in whichever
  `semantic_backend` instance the caller constructs and passes in.
  Same reasoning the README already gives for why `GraphBackend`
  itself has no extra dependencies.
- **`get_all()` delegates to `lexical_backend` only.** Both stores
  should hold identical events if the mirroring invariant (below)
  holds, so there's nothing to fuse for an unranked listing. The
  docstring states this explicitly and states the assumption plainly,
  the same way `ChromaBackend`'s docstring already calls out its own
  collection-sharing caveat rather than leaving it to be inferred:

  ```python
  def get_all(self, tier: Optional[MemoryTier] = None) -> list[MemoryEvent]:
      """Delegates to lexical_backend only. Assumes the mirroring
      invariant held (see add/update_tier/remove) -- if it didn't,
      this won't reflect semantic_backend's state.
      """
  ```
- **Lives at `src/memory_system/backends/hybrid.py`**, one file per
  backend, matching `memory.py` / `chroma.py` / `graph.py`.
- **Out of scope for this design**: generalizing to N backends
  (RRF's formula generalizes trivially, but nothing here asks for
  more than two), and per-backend weighting beyond `rrf_k`. Both are
  easy to add later without breaking this interface if needed.

## Fusion algorithm: Reciprocal Rank Fusion (RRF)

### Why not weighted-average or max-of-both

The two backends' scores are not on comparable scales, and the
incompatibility isn't superficial -- it's a shape mismatch, not just
a units mismatch:

- `InMemoryBackend.query()` scores via TF-IDF cosine similarity,
  recomputed per call from *that call's* candidate set (IDF depends
  on `n_docs` for that query). Bounded `[0, 1]`, but sparse: any
  document with zero token overlap scores exactly 0 and is dropped
  from the result list entirely (`memory.py`: `if score > 0`).
- `ChromaBackend.query()` scores via `1.0 / (1.0 + distance)` on
  embedding cosine distance. Also bounded `(0, 1]`, but distributed
  completely differently: embeddings for "related but not identical"
  text cluster in a narrow mid-range, rarely near-0 distance, so its
  scores rarely spread as wide as TF-IDF's do.

Both happen to land in "roughly `[0, 1]`" by coincidence, not because
they measure the same thing. A weighted average or a max() comparison
both directly compare these magnitudes against each other -- whichever
backend's distribution happens to run "hotter" for a given query would
systematically dominate the fused ranking regardless of the weight
chosen, unless a per-query normalization step (e.g. min-max) is added
first. That's an extra moving part, and min-max over a small result
set (which `top_k` often is) is itself unstable.

max-of-both has a second, independent problem: it doesn't reward
agreement between the two signals. A document ranked #1 by only one
backend and absent from the other scores identically to a document
ranked #1 by *both* -- which throws away the main reason to do hybrid
retrieval in the first place (agreement across differently-biased
scorers is a stronger relevance signal than either alone).

### RRF

```python
def reciprocal_rank_fusion(
    result_lists: list[list[RetrievalResult]],
    k: int = 60,
) -> list[RetrievalResult]:
    """score(event) = sum over lists containing it of 1 / (k + rank),
    rank is 1-indexed position within that list. Events matched
    across lists by event.id -- see HybridBackend's mirroring
    invariant for why that id is trustworthy across both backends.
    Lists an event doesn't appear in simply contribute nothing;
    empty lists contribute nothing. Returns events sorted descending
    by summed score.
    """
```

RRF never touches raw score magnitude -- only rank *position* within
each backend's own internally-consistent ordering, so the
incomparable-magnitudes problem never comes up. `k=60` is the
standard damping constant from the original RRF literature (also
what Elasticsearch/Weaviate/Azure AI Search default to for hybrid
search), not a bespoke choice -- it's the one tunable knob, versus
weighted-average needing two backend weights justified. It rewards
agreement naturally: a document both backends rank highly accumulates
both `1/(k+rank)` terms.

**The "zero lexical overlap, strong semantic relevance" case** (your
original concern) isn't a special case under RRF -- it's the
expected, designed-for behavior. If `lexical_backend` drops a
document entirely (score 0, filtered out) and `semantic_backend`
ranks it #1, that document gets exactly one contribution term,
`1/(k+1)`, instead of two. No branch needed for "a backend returned
nothing" -- an empty or partial result list just contributes fewer
terms and the sum degrades gracefully. Both backends returning `[]`
fuses to `[]`.

### `fetch_k` oversampling

`HybridBackend.query(query, top_k=5, ...)` must request more than
`top_k` from each internal backend before fusing -- a document ranked
#8 lexically but #1 semantically could belong in the final top 5
post-fusion, but would never be seen if only 5 raw results were
pulled per backend. `fetch_k = top_k * fetch_k_multiplier` (default
multiplier `5`) is requested from each backend, fused, then truncated
to `top_k`. This mirrors `TieredMemory.retrieve()`'s own existing
oversampling (`top_k=top_k * 2` before its decay-reweighting pass,
`core.py:82`) -- same shape of problem, oversample before a second
ranking pass trims the pool.

## Mirroring invariant and fail-loud sync errors

RRF's correctness depends on matching documents across the two
backends' result lists by `event.id`. That only works if both
backends actually hold the same events under the same ids -- so every
mutating `HybridBackend` method (`add`, `update_tier`, `remove`) must
write to both `lexical_backend` and `semantic_backend`.

**Partial write failure is fail-loud, not silent.** If the first
backend's write succeeds and the second's throws, the two backends
are now out of sync -- and since RRF has no way to detect or correct
for silent divergence, that's a worse failure mode than surfacing an
exception. Each mutating method:

1. Writes to `lexical_backend` first.
2. If that succeeds, writes to `semantic_backend`.
3. If step 2 throws, raises `HybridBackendSyncError` -- wrapping the
   original exception (`raise ... from original`), stating which
   backend succeeded, which failed, which method and `event_id` were
   involved.
4. If step 1 throws, nothing has diverged yet (`semantic_backend` was
   never touched) -- the original exception propagates unwrapped,
   since this isn't a sync failure, just a normal single-backend
   failure.

```python
class HybridBackendSyncError(Exception):
    """Raised when a HybridBackend mutating call (add/update_tier/
    remove) succeeds on one internal backend but fails on the other,
    leaving them out of sync. Wraps the original exception via
    `raise ... from`. Always check which backend succeeded before
    deciding how to recover -- the successful side already has state
    the other doesn't.
    """
```

This applies identically to `add()`, `update_tier()`, and `remove()`
-- all three have the same mirroring risk, not just `add()`.

## Testing

- `reciprocal_rank_fusion()` is a pure function: unit tested directly
  with synthetic `RetrievalResult` lists, no backend construction
  needed at all. Same isolation-testing pattern as
  `build_semantic_profile()` / `FakeDecayPolicy` in the
  `PerceptSalienceScorer` work -- ranking/tie-break/empty-list logic
  tested without any real backend in the loop.
- `HybridBackend` itself is tested against real backend instances
  (`InMemoryBackend` + `ChromaBackend`), matching the existing
  `test_chroma_backend.py` convention of testing against a real
  ChromaDB instance rather than a mock. Covers: the mirroring
  invariant (a failing second backend actually raises
  `HybridBackendSyncError` with correct attribution), `fetch_k`
  oversampling actually changing results, and the "found by only one
  backend" case actually surfacing in fused results.

## Non-goals

- Generalizing beyond two backends.
- Per-backend weighting beyond `rrf_k`.
- Changing `TieredMemory` itself -- `HybridBackend` is a drop-in
  `MemoryBackend`, nothing upstream needs to change.
