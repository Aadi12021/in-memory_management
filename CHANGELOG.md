# Changelog

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
