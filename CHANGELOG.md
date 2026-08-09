# Changelog

## 0.3.0

### Added

- `HybridBackend` (`memory_system.backends.hybrid`) — combines a lexical
  backend (e.g. `InMemoryBackend`) and a semantic backend (e.g.
  `ChromaBackend`) into one ranked result via Reciprocal Rank Fusion
  (RRF), rather than requiring a choice between retrieval strategies.
  TF-IDF cosine similarity and embedding-distance similarity aren't on
  comparable scales, so RRF combines by each backend's own rank order
  instead of raw score. Mirrors every write to both backends and raises
  `HybridBackendSyncError` if a write to the second backend fails after
  the first succeeded, rather than leaving the two silently out of
  sync. Requires no extra of its own — whatever the two backends you
  pass it need is their concern. See the "Choosing a backend" section
  of the README and
  [docs/superpowers/specs/2026-07-28-hybrid-retrieval-design.md](https://github.com/Aadi12021/in-memory_management/blob/main/docs/superpowers/specs/2026-07-28-hybrid-retrieval-design.md)
  for the full design rationale. *(Note: this shipped in `src/` prior
  to this release without its own version bump or changelog entry —
  documented here retroactively.)*
- Offline consolidation: `TieredMemory.deduplicate()`, `.compress()`,
  `.strengthen_connections()`, and `.offline_consolidate()`
  (`memory_system.core`) — periodic cleanup of the long-term tier, run
  separately from the online `store()`/`consolidate()`/`decay()` loop.
  `deduplicate()` merges near-duplicate long-term events found via the
  backend's own `query()` at or above a required `threshold` (no
  default -- backend score scales, e.g. `InMemoryBackend`'s TF-IDF
  cosine vs. `HybridBackend`'s RRF-fused scores, are not comparable).
  `compress()` groups related events into connected components and
  replaces each group with one LLM-generated summary via a
  `MemorySummarizer`, failing soft (logged via `logging.warning`, not
  raised) if the summarizer errors on a group. `strengthen_connections()`
  is `GraphBackend`-only (also reached through a `HybridBackend`
  composed with one) and reinforces pre-existing "bystander"
  relationships between entities a merge/compress pass just
  co-associated, explicitly skipping the relationships that pass itself
  created. `offline_consolidate()` runs all three in a fixed order.
  This is the library's first destructive operation (`deduplicate()`
  and `compress()` remove their source events once merged/summarized);
  every method accepts `dry_run=True` to preview the effect with no
  mutation. See the "Offline consolidation" section of the README.
- `ConsolidationReport` (`memory_system.events`, exported from the
  top-level package) — the structured return type shared by
  `deduplicate()`, `compress()`, `strengthen_connections()`, and
  `offline_consolidate()`, replacing a bare count with the actual
  merged/compressed/strengthened pairs and ids (`None` in place of a
  new event id under `dry_run`).
- `memory_system.summarization` package: `MemorySummarizer` (abstract
  base) and `LLMSummarizer` (Claude-based, `pip install
  tiered-memory[llm]`) — the summarization dependency `compress()`
  needs, mirroring the `EntityExtractor`/`LLMEntityExtractor` shape
  already used for graph extraction.

### Fixed

- `Relationship.strength` defaulted to `1.0`, the same value as the
  `+0.1` cap `strengthen_connections()` enforces. Since nothing else in
  the library ever wrote to `.strength`, every real,
  extractor-produced relationship was born already at the ceiling,
  making `strengthen_connections()` a silent no-op on any genuinely
  populated graph. Now defaults to `0.5`.
- `compress()` didn't validate a `MemorySummarizer`'s return type — a
  summarizer violating its own `str` contract had its output silently
  stored as corrupted `MemoryEvent` content. Now treated the same as a
  raised exception: logged, group skipped, pass continues.

Both found in a post-merge exploratory-testing pass across all four
backends, which also closed a real coverage gap (`ChromaBackend` had
zero test coverage with any offline-consolidation method) and added a
threshold-floor guidance callout to the README for `ChromaBackend` and
`HybridBackend`.

## 0.2.0

### Added

- `PerceptSalienceScorer` (`memory_system.policies.percept_salience`) — a
  real, embedding-based novelty/surprise `SalienceScorer`, ported from
  the predictive-coding stage of PERCEPT-1 (Cognitive Digital Twin
  series). Embeds incoming content against a live "prior" summary built
  from the long-term tier and scores salience as the cosine distance
  between them, normalized to `[0, 1]`. Local and synchronous — no LLM
  calls — so it's safe to run inline in `TieredMemory.store()`'s hot
  path.
- `build_semantic_profile()` (`memory_system.policies.percept_bridge`) —
  the bridge function that turns a `TieredMemory` backend's long-term
  tier into the profile shape `PerceptSalienceScorer` embeds as its
  "prior," ranked by `DecayPolicy.current_strength()` rather than raw
  ingest-time salience.
- New `percept` optional extra (`pip install tiered-memory[percept]`):
  `sentence-transformers`, `numpy`. Deliberately does not depend on
  `anthropic` — the LLM-based half of PERCEPT-1's pipeline
  (`_generate_prior_diff`, `MultimodalBinder`'s vision path) is not part
  of this scorer.
- `examples/percept_salience.py` — seeds a long-term profile, then shows
  a related memory scoring lower salience than an unrelated one, and
  only the unrelated one crossing `SurpriseBasedConsolidation`'s
  threshold.

## 0.1.0

Initial release: `TieredMemory` core loop (`store` → `consolidate` →
`decay` → `retrieve`), `InMemoryBackend` (TF-IDF + cosine + stemming,
zero dependencies), `ChromaBackend`, `GraphBackend` with
`RuleBasedEntityExtractor` and `LLMEntityExtractor`, consolidation/decay
policies, and toy `SalienceScorer` defaults (`ConstantSalience`,
`LengthHeuristicSalience`).
