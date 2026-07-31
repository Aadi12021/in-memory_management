# LongMemEval_S results: 100-question stratified subset

Run date: 2026-07-29. Raw log: [`2026-07-29-full-run.log`](2026-07-29-full-run.log).

Design and methodology: [`docs/superpowers/specs/2026-07-28-longmemeval-benchmark.md`](../../docs/superpowers/specs/2026-07-28-longmemeval-benchmark.md).
Harness: [`benchmark/longmemeval_benchmark.py`](../longmemeval_benchmark.py).

**This is recall@k against LongMemEval's own ground-truth labels
(`has_answer`, `answer_session_ids`), not the official LLM-judged
QA-accuracy metric** -- `tiered-memory` has no answer-generation step,
so this measures whether `retrieve()` surfaces the evidence, not
whether a downstream reader could correctly answer from it. Not
comparable to any published LongMemEval baseline number.

## Reproducing this run

```bash
python benchmark/longmemeval_benchmark.py --n 100 --seed 42 --top-k 10 --out results.json
```

### Provenance

| | |
|---|---|
| Dataset | `longmemeval_s_cleaned.json`, sha256 `d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442` |
| Dataset source | https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned |
| chromadb version | 1.5.9 |
| tiered-memory version | 0.2.0 |
| git commit | `9b1811daeb41e2e9ccced447b56b9d9d234a0ee4` |
| Seed | 42 |
| top_k | 10 |
| Sample size | 100 (stratified, 27/27/15/14/11/6 across the 6 question_type categories, 95 non-abstention + 5 abstention) |
| Total compute time | ~1.93 hours summed across 300 backend-question calls |

The dataset hash matters more than the URL here: `DATA_URL` points at
an unpinned HF branch ref, so a future fetch of the same URL is not
guaranteed to return the same bytes (the upstream README's own
changelog shows the "cleaned" dataset has already moved once, in
2025/09). If you can't reproduce this hash, you're not running against
the same data.

## Results

### Session-level recall@10

| Backend | Recall | n |
|---|---|---|
| InMemoryBackend | 0.9368 | 95 |
| **ChromaBackend** | **0.9684** | 95 |
| HybridBackend | 0.9579 | 95 |

### Turn-level recall@10

| Backend | Recall | n |
|---|---|---|
| InMemoryBackend | 0.7400 | 95 |
| ChromaBackend | 0.6967 | 95 |
| **HybridBackend** | **0.7861** | 95 |

**No single backend wins on both metrics.** `ChromaBackend` alone has
the best session-level recall; `HybridBackend` has the best turn-level
recall, beating even `InMemoryBackend` here (at the 20-question pilot
scale, `HybridBackend` had sat *between* its two components on this
metric -- that reversed at full scale).

### Per-category session-level recall

| Category | InMemory | Chroma | Hybrid | n |
|---|---|---|---|---|
| knowledge-update | 1.0000 | 0.9167 | 1.0000 | 12 |
| multi-session | 1.0000 | 1.0000 | **0.9600** | 25 |
| single-session-assistant | 1.0000 | 1.0000 | 1.0000 | 11 |
| single-session-preference | **0.6667** | 1.0000 | 0.8333 | 6 |
| single-session-user | 1.0000 | 0.9286 | 1.0000 | 14 |
| temporal-reasoning | **0.8519** | 0.9630 | 0.9259 | 27 |

`InMemoryBackend` is weakest on `single-session-preference` (paraphrase-heavy
questions, where lexical matching alone struggles as expected) and
`temporal-reasoning`; semantic matching helps both. `multi-session` is
the one category where `HybridBackend` underperforms *both* of its own
components (0.96 vs 1.0/1.0) -- a real instance of RRF's re-ranking
tradeoff (see below), visible at category scale here rather than just
single-question scale.

## Why fusion doesn't uniformly win: a dilution investigation

Before scaling from a 20-question pilot to this full run, we
investigated why `HybridBackend`'s turn-level recall sat between its
two components in the pilot (it doesn't, at full scale -- see above --
but the mechanism is the same and still explains `multi-session`'s
regression here). Deep-diving 4 representative questions from the
pilot (comparing each backend's own top-10 against `HybridBackend`'s
fused top-10, matched by `(session_id, content)` since `MemoryEvent.id`
is a fresh UUID per independent ingestion and can't be used for
cross-backend identity comparison):

- In 2 of 4 cases, `HybridBackend` was *worse* than `InMemoryBackend`
  alone -- not because fusion introduced new noise, but because
  Reciprocal Rank Fusion's own re-ranking demoted a true positive that
  `InMemoryBackend` alone would have kept in its top-10. RRF rewards
  items *both* backends rank moderately (consensus) over an item one
  backend ranks highly and the other doesn't rank at all -- a real,
  known tradeoff (consensus robustness vs. single-backend peak
  precision), not an implementation defect.

This is the same mechanism behind `multi-session`'s 0.96 in the table
above.

## Known nuances (not defects, disclosed for anyone citing this)

- **`HybridBackend`'s candidate pool is deeper than the standalone
  runs.** It queries each of its two internal backends at
  `top_k * fetch_k_multiplier` (50, at `top_k=10`, `fetch_k_multiplier=5`)
  before fusing down to `top_k=10`, while the standalone `InMemoryBackend`/
  `ChromaBackend` runs only ever see `top_k=10`. This doesn't inflate
  `HybridBackend`'s score artificially (each standalone backend's own
  top-10 is already score-determined, independent of how deep a
  *different* run might have searched) -- but it means `HybridBackend`
  structurally has more opportunity to find cross-list agreement,
  which is intrinsic to how RRF works, not a fairness issue.
- **4 of the 100 sampled questions have a duplicated session** in their
  own haystack (`1e043500`, `001be529`, `gpt4_4929293b`, `gpt4_76048e76`)
  -- verified byte-identical duplicates, a property of the upstream
  dataset, not a harness bug. None of the duplicated sessions is an
  answer session in this sample, so it didn't produce a false-positive
  hit here, but a duplicated *answer* session could let a hit on the
  wrong copy count, and duplicated turns consume top-k slots.
- **Max relevant turns per question tops out at 6** across this
  sample, well under `top_k=10` -- turn-level recall is never
  artificially capped by `k` in this run.
- **The telemetry-disable fix did not resolve the latency outliers.**
  `ANONYMIZED_TELEMETRY=False` is applied (chromadb defaults it to
  `True`), a real, verified, zero-downside condition worth disabling
  regardless -- but in this run, with it already disabled, ChromaBackend's
  median call was 29.0s / p90 32.8s (n=100, computed directly from the
  raw log) while one call still took 930.8s. The hypothesis was
  reasonable but the fix didn't eliminate the spike behavior; the
  actual cause of these single-question latency outliers remains
  unexplained.
