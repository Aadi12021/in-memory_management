# tiered-memory

A cognitive-science-grounded memory layer for AI agents.

Most memory libraries for LLM agents are flat vector-store wrappers with
recency scoring bolted on. `tiered-memory` instead models memory the way
human cognition does: information moves through **tiers** (working →
long-term), gets **consolidated** based on salience, and **decays** over
time unless reinforced.

- **Tiered** — working memory and long-term memory behave differently by design
- **Consolidation-aware** — plug in your own policy for what's worth remembering long-term
- **Decay-aware** — memories fade on a forgetting curve instead of living forever
- **Backend-agnostic** — ships with a zero-dependency TF-IDF in-memory backend and a ChromaDB backend; bring your own

## Install

```bash
pip install tiered-memory
# with ChromaDB support:
pip install tiered-memory[chroma]
```

## Quickstart

```python
from memory_system import (
    TieredMemory,
    InMemoryBackend,
    SurpriseBasedConsolidation,
    ForgettingCurveDecay,
    ConstantSalience,
)

memory = TieredMemory(
    backend=InMemoryBackend(),
    consolidation_policy=SurpriseBasedConsolidation(threshold=0.7),
    decay_policy=ForgettingCurveDecay(half_life_days=14),
    salience_scorer=ConstantSalience(value=0.9),
)

memory.store("User is allergic to peanuts.")
memory.consolidate()  # promotes high-salience events to long-term

results = memory.retrieve("peanut allergy", top_k=3)
for r in results:
    print(r.event.content, r.score)
```

## Choosing a backend

- **`InMemoryBackend`** (default, zero dependencies) — real TF-IDF +
  cosine similarity, the same family of technique classic search
  engines used before embeddings. Includes lightweight stemming, so
  "peanuts" matches "peanut" and "hiking" matches "hike". It does
  *not* do semantic/concept matching — "peanuts" won't match "dietary
  restrictions" the way an embedding model would. Good for small to
  medium memory sizes, testing, and CI.
- **`ChromaBackend`** (`pip install tiered-memory[chroma]`) — real
  embedding-based semantic search via ChromaDB. Use this when you need
  concept-level matching or are scaling past what an in-process index
  comfortably handles.

## Why tiers?

Flat vector search treats every memory as equally durable. In practice,
most of what an agent hears is transient context that shouldn't stick
around, and only a small fraction is worth carrying forward. Modeling
that distinction explicitly, rather than relying on similarity search
alone, gives you a place to hang consolidation rules, decay behavior, and
salience scoring that a flat store has no natural home for.

## Architecture

```
TieredMemory
 ├── MemoryBackend        (storage: InMemoryBackend, ChromaBackend, or your own)
 ├── ConsolidationPolicy  (working -> long-term promotion rules)
 ├── DecayPolicy          (retrievability over time)
 └── SalienceScorer       (importance/surprise scoring on ingest)
```

Each piece is swappable. Bring your own backend by implementing
`MemoryBackend`; bring your own consolidation logic by implementing
`ConsolidationPolicy`; same for decay and salience.

## Status

Early alpha (v0.1). Core loop (`store` → `consolidate` → `decay` →
`retrieve`) works end-to-end with the in-memory and ChromaDB backends.
API may still shift before v1.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md).

## License

MIT
