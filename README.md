# tiered-memory

A cognitive-science-grounded memory layer for AI agents.

Most memory libraries for LLM agents are flat vector-store wrappers with
recency scoring bolted on. `tiered-memory` does two things a flat store
doesn't:

1. Models memory the way human cognition does: information moves through
   **tiers** (working → long-term), gets **consolidated** based on
   salience, and **decays** over time unless reinforced.
2. Can store facts as a **graph of entities and relationships**, not just
   text — which means it can answer multi-hop questions a similarity
   search fundamentally cannot, because the answer was never phrased as a
   single sentence in the first place.

## The thing a flat vector store can't do

Say you've stored these two facts, and nothing else: "User is allergic
to peanuts." and "Peanuts contains protein." Ask a similarity search "is
the user's allergy connected to protein?" and it has nothing to work
with — no stored sentence mentions "user" and "protein" together.
`GraphBackend` answers it anyway, by extracting entities and
relationships from each fact and traversing the chain:

```python
from memory_system.backends.graph import GraphBackend
from memory_system.events import MemoryEvent
from memory_system.extraction.rules_based import RuleBasedEntityExtractor

graph = GraphBackend(extractor=RuleBasedEntityExtractor())

graph.add(MemoryEvent(content="User is allergic to peanuts."))
graph.add(MemoryEvent(content="Peanuts contains protein."))

# user -ALLERGIC_TO-> peanut -CONTAINS-> protein: two hops, not one similarity match
related = graph.related_to("user", max_hops=2)
print([entity.id for entity in related])
# ['peanut', 'protein']

path = graph.explain_path("user", "protein")
print([(rel.source_id, rel.relation_type, rel.target_id) for rel in path])
# [('user', 'ALLERGIC_TO', 'peanut'), ('peanut', 'CONTAINS', 'protein')]
```

`related_to()` finds what's reachable; `explain_path()` shows *why* — the
actual chain of relationships, for when you need to justify a surfaced
memory rather than just return it. See
[examples/graph_backend.py](https://github.com/Aadi12021/in-memory_management/blob/main/examples/graph_backend.py) for a version with
an unrelated distractor fact included, to show the traversal doesn't just
return everything.

## Install

```bash
pip install tiered-memory

# with ChromaDB (real embedding-based semantic search):
pip install tiered-memory[chroma]

# with Claude-based entity/relationship extraction for GraphBackend:
pip install tiered-memory[llm]

# with a real, embedding-based novelty/surprise SalienceScorer:
pip install tiered-memory[percept]
```

`GraphBackend` itself has no extra dependencies — it's the *extractor*
you plug into it (`RuleBasedEntityExtractor` is zero-dependency;
`LLMEntityExtractor` needs the `llm` extra) that determines what you
need installed.

## Quickstart

The core tiered-memory loop — store, consolidate, decay, retrieve —
works the same way regardless of which backend you choose:

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
  engines used before embeddings. Includes a lightweight suffix-stripping
  stemmer, so regular plurals and verb endings collapse together
  (`peanut`/`peanuts`, `walk`/`walking`, `allergy`/`allergies`), including
  short silent-`e` words whose `e` is dropped by `-es`/`-ed`/`-ing` and
  restored on stemming (`hike`/`hiking`, `live`/`lives`, `like`/`likes`,
  `love`/`loves`). It does *not* do semantic/concept matching
  (`"peanuts"` won't match `"dietary restrictions"`), and the stemmer is
  genuinely minimal: it doesn't connect irregular derivations
  (`allergic`/`allergy` stay separate tokens), and it doesn't handle
  consonant-doubling before `-ing`/`-ed` (`run`/`running`,
  `swim`/`swimming` stay separate too). See
  [benchmark/retrieval_benchmark.py](https://github.com/Aadi12021/in-memory_management/blob/main/benchmark/retrieval_benchmark.py)
  and [tests/test_stemmer.py](https://github.com/Aadi12021/in-memory_management/blob/main/tests/test_stemmer.py) for what's verified
  to work and what isn't. Good for small to medium memory sizes,
  testing, and CI.
- **`ChromaBackend`** (`pip install tiered-memory[chroma]`) — real
  embedding-based semantic search via ChromaDB. Use this when you need
  concept-level matching or are scaling past what an in-process index
  comfortably handles. Note: the non-persistent (in-memory) client
  shares its underlying store across `ChromaBackend` instances built
  with the same `collection_name` in one process — use distinct names
  for isolated stores (e.g. per test, per session).
- **`GraphBackend`** — stores entities and relationships extracted from
  each memory instead of (or alongside) flat text, and exposes
  graph-native queries on top of the usual store/retrieve interface:
  - `related_to(entity_id, relation_type=None, max_hops=1)` — BFS
    traversal, optionally filtered to one relationship type
  - `explain_path(source_id, target_id)` — the shortest relationship
    chain connecting two entities, or `None`
  - `consolidation_signal(entity_id)` — a structural signal (bounded to
    `[0, 1)`, based on how connected an entity is) that a consolidation
    policy could use alongside or instead of salience
  - `query(text, top_k, tier)` — the standard `MemoryBackend` interface,
    implemented here as entity-overlap matching rather than text
    similarity, so `GraphBackend` is swappable anywhere a backend is
    expected
  Requires an `EntityExtractor` (see below) to turn stored text into
  graph structure.

## Choosing an extractor

`GraphBackend` needs something to turn raw memory content into entities
and relationships. Two are included:

- **`RuleBasedEntityExtractor`** (zero dependencies) — regex patterns
  over a fixed set of phrasings (`"allergic to X"`, `"enjoys X"`,
  `"works at X"`, `"X contains Y"`, etc.). Fast, free, deterministic, and
  narrow: it only catches phrasing it has a pattern for, and its object
  capture caps out at two words, so compound nouns like "peanut butter
  cake" can lose their leading word. Good for structured, predictable
  input.
- **`LLMEntityExtractor`** (`pip install tiered-memory[llm]`) — calls
  Claude to extract entities/relationships as structured JSON. Much
  higher recall and robust to phrasing the rule-based extractor can't
  cover, at the cost of a per-memory API call. Needs `ANTHROPIC_API_KEY`
  set (or pass `api_key=` directly).

```python
from memory_system.extraction.llm_based import LLMEntityExtractor

extractor = LLMEntityExtractor()  # reads ANTHROPIC_API_KEY from the environment
```

## Choosing a salience scorer

`TieredMemory` calls `salience_scorer.score(event)` on every `store()` to
decide how important/surprising an incoming memory is; consolidation
policies like `SurpriseBasedConsolidation` act on that score.

- **`ConstantSalience`** / **`LengthHeuristicSalience`** (zero
  dependencies) — toy defaults so the system works out of the box.
  Every event gets the same score, or a score proportional to content
  length. Fine for testing and for exploring the rest of the pipeline;
  not meant to reflect real importance.
- **`PerceptSalienceScorer`** (`pip install tiered-memory[percept]`) —
  a real novelty/surprise signal, ported from the predictive-coding
  stage of PERCEPT-1 (a perception daemon from the Cognitive Digital
  Twin series). Embeds the incoming content and a live "prior" summary
  built from what's currently in the long-term tier, and scores
  salience as the cosine distance between them (normalized to
  `[0, 1]`) — content that's novel relative to what the system already
  knows scores higher than repetition of established facts. Local and
  synchronous (no LLM calls, no network access at score time), so it's
  safe to run inline in `store()`'s hot path; it does re-fetch and
  re-embed the whole long-term tier on every call, which is cheap for
  `InMemoryBackend` but worth budgeting for on a large `ChromaBackend`
  collection. See
  [examples/percept_salience.py](https://github.com/Aadi12021/in-memory_management/blob/main/examples/percept_salience.py).

## Retrieval quality: InMemoryBackend vs. naive keyword matching

[benchmark/retrieval_benchmark.py](https://github.com/Aadi12021/in-memory_management/blob/main/benchmark/retrieval_benchmark.py) runs
26 synthetic facts and 24 queries (ground truth judged by hand before
running either method) through `InMemoryBackend` and a naive
substring/exact-match baseline (lowercase + tokenize, no stemming, no
stopword weighting — "no match" if nothing overlaps at all, rather than
defaulting to an arbitrary result).

```
InMemoryBackend (TF-IDF + stemming): 24/24 correct top-1
naive substring/exact-match:         22/24 correct top-1
```

Both naive misses are real, not cherry-picked: one query only has a
plural form of the relevant word in the stored fact (naive finds zero
overlap and gives up; stemming connects them), and one ties on the
common words shared with an unrelated fact (naive has no way to weight
the one rare, decisive word higher; TF-IDF's IDF weighting does). Run it
yourself: `python benchmark/retrieval_benchmark.py`.

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
 ├── MemoryBackend          (storage + retrieval; swappable)
 │    ├── InMemoryBackend   TF-IDF + cosine + stemming, zero dependencies
 │    ├── ChromaBackend     real embeddings via ChromaDB
 │    └── GraphBackend      entities + relationships, multi-hop traversal
 │         └── EntityExtractor            (text -> entities/relationships)
 │              ├── RuleBasedEntityExtractor   regex, zero dependencies
 │              └── LLMEntityExtractor         Claude-based, higher recall
 ├── ConsolidationPolicy    (working -> long-term promotion rules)
 ├── DecayPolicy            (retrievability over time)
 └── SalienceScorer         (importance/surprise scoring on ingest)
      ├── ConstantSalience / LengthHeuristicSalience   toy defaults, zero dependencies
      └── PerceptSalienceScorer                        real novelty signal, ported from PERCEPT-1
```

Each piece is swappable. Bring your own backend by implementing
`MemoryBackend`; bring your own consolidation logic by implementing
`ConsolidationPolicy`; same for decay, salience, and entity extraction
(`EntityExtractor`).

Note: `GraphBackend`'s graph-native methods (`related_to`,
`explain_path`, `consolidation_signal`) are only reachable by calling the
backend directly (`memory.backend.related_to(...)`) — `TieredMemory`'s
own `store`/`retrieve` interface only knows about the generic
`MemoryBackend` methods every backend implements.

## Status

Early alpha (v0.2). Core loop (`store` → `consolidate` → `decay` →
`retrieve`) works end-to-end with all three backends. `GraphBackend`'s
full method set (`add`/`get_all`/`query`/`update_tier`/`remove`/
`related_to`/`explain_path`/`consolidation_signal`) is implemented and
tested, including integration tests against a real `RuleBasedEntityExtractor`
and a real ChromaDB instance. `PerceptSalienceScorer` adds a real,
embedding-based salience signal alongside the existing toy scorers. API
may still shift before v1.

## Contributing

See [CONTRIBUTING.md](https://github.com/Aadi12021/in-memory_management/blob/main/CONTRIBUTING.md).

## License

MIT
