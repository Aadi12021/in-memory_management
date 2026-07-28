# LongMemEval benchmark design

Date: 2026-07-28
Status: approved, not yet implemented

## Problem

`benchmark/retrieval_benchmark.py` is homemade: 26 synthetic facts, 24
hand-judged queries, written by us. It's useful for regression-testing
`InMemoryBackend`'s stemmer, but it can't answer "is tiered-memory's
retrieval actually good," because we wrote both the questions and the
answer key. LongMemEval (ICLR 2025) is a real, external, standardized
long-term-memory benchmark with independently authored questions and
ground truth. This adds a second benchmark against it, so results are
citable against something we didn't design ourselves.

## The dataset (verified against primary sources, not summarized secondhand)

Source: paper [arXiv:2410.10813](https://arxiv.org/abs/2410.10813),
code at [github.com/xiaowu0162/LongMemEval](https://github.com/xiaowu0162/LongMemEval),
data at [huggingface.co/datasets/xiaowu0162/longmemeval-cleaned](https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned)
(public, ungated, MIT license). Verified by reading the actual README,
the actual `evaluate_qa.py` source, and by downloading and parsing the
real `longmemeval_oracle.json` file directly.

**Structure.** 500 QA instances, shared across all three official
variants (they differ only in haystack size, not in questions). Each
instance:

```
question_id, question_type, question, answer, question_date,
haystack_session_ids, haystack_dates,
haystack_sessions: [ [ {role: user/assistant, content: str, has_answer?: true}, ... ], ... ],
answer_session_ids
```

`haystack_session_ids`/`haystack_dates` are sorted by timestamp in the
S and M variants (not in Oracle). `has_answer: true` marks the
specific turns containing evidence; `answer_session_ids` marks the
evidence sessions.

**Variant used: `longmemeval_s_cleaned.json`** (277 MB, ~40 sessions
per question, ~115k tokens). Not Oracle (15.4 MB, evidence-only
haystack — removes the actual "needle in haystack" retrieval problem
entirely, so it doesn't test retrieval quality, just QA-given-perfect-
context). Not M (2.74 GB, ~500 sessions per question — real local
embedding compute for that volume is realistically hours on a laptop,
impractical for a solo run).

**Question categories** (real counts, computed by parsing the actual
file, not the paper's rounded description):

| Category | Total | Abstention (`_abs`) |
|---|---|---|
| temporal-reasoning | 133 | 6 |
| multi-session | 133 | 12 |
| knowledge-update | 78 | 6 |
| single-session-user | 70 | 6 |
| single-session-assistant | 56 | 0 |
| single-session-preference | 30 | 0 |
| **Total** | **500** | **30** |

Abstention questions (`question_id` ending `_abs`) have no real
evidence in the haystack; the correct behavior is declining to answer.

**The official metric is LLM-as-judge QA accuracy, not exact match, and
not what this benchmark measures (see below).** Read directly from
`evaluate_qa.py`: a judge model (GPT-4o family, or a locally-served
`llama-3.1-70b-instruct` — no Claude option) is given the question,
correct answer, and a model's generated response, and asked a
category-specific yes/no correctness prompt (temporal-reasoning
forgives off-by-one date errors; knowledge-update allows outdated info
alongside the correct updated answer; single-session-preference's
"answer" field is actually a rubric, not a literal string). This
requires a *reader* LLM to generate an answer from retrieved context
in the first place — tiered-memory has no reader/generation step at
all, it only returns ranked `RetrievalResult`s. Reproducing this
literal metric is out of scope for this spec; see Non-goals and the
v3 staging note below.

## What this benchmark actually measures: recall@k

Since tiered-memory owns retrieval, not reading or generation, this
benchmark measures **recall@k against LongMemEval's own ground-truth
labels** (`has_answer`, `answer_session_ids`) — not by running the
upstream repo's `print_retrieval_metrics.py` script (that expects
their own retrieval-log schema from `run_retrieval.sh`), but by
computing the same underlying concept directly against our own
`memory.retrieve()` output, since we control what metadata rides along
with each stored `MemoryEvent`.

For a question with `top_k=k`:

- **Session-level recall@k** (primary metric): `1` if at least one of
  the retrieved events' `session_id` (carried in `MemoryEvent.metadata`,
  set at ingestion) is in `answer_session_ids`, else `0`. Averaged
  across questions -> the headline number, directly analogous to
  "hit rate" in standard retrieval-benchmark usage.
- **Turn-level recall@k** (secondary, stricter): of all turns marked
  `has_answer: true` for this question, the fraction actually present
  among the `top_k` retrieved events. More literal "recall" (fraction
  of relevant items retrieved) than the binary hit/miss above.

Both are computed directly from fields already present on each
returned `RetrievalResult.event.metadata` — no separate lookup table,
no LLM call, no external API, zero cost beyond local compute.

## Adapter design: mapping LongMemEval onto tiered-memory's API

Lives at `benchmark/longmemeval_benchmark.py`, alongside the existing
`benchmark/retrieval_benchmark.py` -- same directory convention, a
second, independent benchmark rather than a variant of the first
(different dataset, different metric, different dependency footprint:
this one needs `chroma` installed and a local download of
`longmemeval_s_cleaned.json`, neither of which the homemade benchmark
requires).

**One fresh `TieredMemory` per question.** Each question's
`haystack_sessions` is its own independent context — question 2's
haystack must not leak into question 1's retrieval. Per question:

```python
lexical = InMemoryBackend()
semantic = ChromaBackend(collection_name=f"longmemeval_{question_id}")
memory = TieredMemory(
    backend=HybridBackend(lexical_backend=lexical, semantic_backend=semantic),
    consolidation_policy=AlwaysConsolidate(),  # required by the constructor, never invoked
    decay_policy=NoDecay(),                    # deliberate -- see "What this does not measure"
)
```

`consolidation_policy` is a required constructor argument with no
default; `AlwaysConsolidate()` is the simplest harmless choice since
`consolidate()` is never called in this harness (everything is queried
straight out of the working tier — `retrieve()` searches across all
tiers by default when `tier=None`, so nothing needs promoting).
`decay_policy=NoDecay()` is a deliberate choice, not a placeholder --
see below.

**Ingestion: one `store()` call per turn, not per session.** Turn
granularity matches `MemoryEvent`'s atomic unit, matches `has_answer`'s
own granularity (enabling turn-level recall without extra bookkeeping),
and mirrors LongMemEval's own baseline retrievers, which support both
`turn` and `session` granularity and default comparisons at turn level.
Sessions are ingested in `haystack_session_ids` order (already
timestamp-sorted for the S variant); turns within a session in their
original list order:

```python
for session_id, session in zip(example["haystack_session_ids"], example["haystack_sessions"]):
    for turn in session:
        memory.store(
            f"{turn['role']}: {turn['content']}",
            metadata={
                "session_id": session_id,
                "role": turn["role"],
                "has_answer": turn.get("has_answer", False),
            },
        )
```

**Querying: one `retrieve()` call per question.**

```python
results = memory.retrieve(example["question"], top_k=k)
```

Score session-level and turn-level recall directly from
`results[i].event.metadata["session_id"]` / `["has_answer"]` against
`example["answer_session_ids"]`.

**Backend(s) under evaluation.** `HybridBackend(InMemoryBackend(),
ChromaBackend())` is the flagship number, per your framing ("current
best"). But the harness is nearly free to run against `InMemoryBackend`
alone and `ChromaBackend` alone too (same code path, swap the backend
constructor) -- and that comparison is the whole point of having built
`HybridBackend` in the first place. v1 runs all three configs against
the same subset, so the result answers not just "how good is
tiered-memory" but "does fusion actually help over either component
alone" -- a hybrid feature whose own benchmark never shows it beating
its components would itself be an important, honest finding.

## What this benchmark does NOT measure (by design, for v1)

- **Reading/reasoning/generation.** Whether a downstream reader could
  correctly answer the question given the retrieved memories --
  including resolving `knowledge-update` contradictions or reasoning
  about `temporal-reasoning` date math -- is untested. Retrieval
  surfacing the right raw turns is necessary but not sufficient for
  the official QA-accuracy metric; this benchmark only measures the
  necessary part.
- **Decay and temporal freshness.** `MemoryEvent.timestamp` isn't
  settable through `store()` (defaults to `datetime.now()`), so
  ingesting and querying happen at ~the same wall-clock instant
  regardless of the dataset's real historical dates
  (`haystack_dates`/`question_date`). Rather than bypassing `store()`
  to backdate events -- which would mean testing a code path real
  callers don't use -- v1 uses `NoDecay()` explicitly, so the
  simplification is a stated design choice, not an accidental artifact
  of `current_strength()` coincidentally evaluating to ~1.0 for
  everything.
- **Consolidation policy behavior.** `consolidate()` is never called;
  this benchmark tests backend retrieval quality in isolation from the
  working/long-term lifecycle, matching how LongMemEval's own baseline
  retrievers are flat "index everything, then retrieve" systems with
  no analogous lifecycle stage.
- **Literal comparability to published baselines.** Recall@k against
  ground-truth labels is not the same number as the paper's or any
  other system's reported QA accuracy. A result here should always be
  reported as "recall@k on tiered-memory's own retrieval-only metric,"
  never framed as beating or losing to a published accuracy figure.

## Subset methodology: 100-question stratified sample

Stratified by `question_type` (proportional allocation, largest-
remainder rounding to land on exactly 100 -- the two categories with
the largest remainder ties, `temporal-reasoning` and `multi-session`,
both round up since they're also the two largest categories by raw
count):

| Category | Population | Sample (20%) |
|---|---|---|
| temporal-reasoning | 133 | 27 |
| multi-session | 133 | 27 |
| knowledge-update | 78 | 15 |
| single-session-user | 70 | 14 |
| single-session-assistant | 56 | 11 |
| single-session-preference | 30 | 6 |
| **Total** | **500** | **100** |

Sampling is uniform at random *within* each category stratum, with a
fixed seed committed alongside the results for reproducibility.
Abstention questions are not separately stratified as a second axis --
given the per-category abstention rates above, a random draw within
each stratum should land roughly 5-6 abstention questions in the
100-question sample (proportional to their ~6% population rate), which
is reported as an observed count after sampling, not enforced as a
quota. A doubly-stratified (category x abstention) design would add
real complexity for a v1 whose main goal is validating the adapter
works correctly, not producing a publication-grade sample.

## v1 / v2 / v3 staging

- **v1 (this spec's scope):** retrieval-only recall@k (session-level +
  turn-level), 100-question stratified subset of `longmemeval_s_cleaned.json`,
  three backend configs (`InMemoryBackend`, `ChromaBackend`,
  `HybridBackend`). Zero external API cost -- both TF-IDF and Chroma's
  default embedding function run fully locally. Citable as "recall@k
  on a 100-question stratified subset of LongMemEval_S." Goal: validate
  the adapter is correct and produce a first honest number.
- **v2:** same metric, same three backends, full 500-question S set.
  Still zero external API cost. More statistically robust; still not
  comparable to published QA-accuracy numbers, just a bigger N of the
  same retrieval-only metric. Only worth doing once v1 confirms the
  adapter and harness are correct and the per-question cost (mostly
  local embedding time) is acceptable at 5x the volume.
- **v3 (optional, real cost, not currently scoped):** end-to-end QA
  accuracy -- add a reader LLM (Claude, since this project is
  Anthropic-native) that generates an answer from retrieved memories,
  then judge via LongMemEval's own prompt templates ported to run
  against Claude instead of GPT-4o. Must be loudly and permanently
  caveated everywhere it's reported: this would not be the official
  metric (different judge model than the published baselines used),
  so it is not a valid apples-to-apples comparison to any number in
  the paper or on a leaderboard. Real API cost: one reader call and
  one judge call per question. Not part of this spec's scope; a
  separate decision if ever pursued.

## Non-goals

- Reproducing `evaluate_qa.py`'s literal metric or its exact published
  numbers.
- Testing `longmemeval_m_cleaned.json` (impractical solo-scale compute).
- Testing `longmemeval_oracle.json` as the primary variant (removes the
  retrieval problem the benchmark exists to test; may be worth a
  separate, explicitly-labeled sanity check later, not part of v1).
- Modeling decay/consolidation lifecycle behavior against this dataset.
- Any change to `tiered-memory`'s own source (`src/memory_system/`) --
  this is a benchmark harness consuming the public API, not a feature.
