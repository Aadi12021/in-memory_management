"""LongMemEval benchmark: recall@k for InMemoryBackend, ChromaBackend, and
HybridBackend against a 100-question stratified subset of LongMemEval_S
(ICLR 2025, https://github.com/xiaowu0162/LongMemEval).

Not the official metric -- LongMemEval's own evaluate_qa.py measures
LLM-judged QA accuracy of a full reader+generation pipeline, which
tiered-memory doesn't have (it only returns ranked RetrievalResults, no
answer generation). This measures recall@k against the dataset's own
ground-truth labels (has_answer, answer_session_ids) instead: does
retrieve() surface the evidence session/turns, not whether a downstream
reader could correctly answer from them. See
docs/superpowers/specs/2026-07-28-longmemeval-benchmark.md for the full
design rationale.

Abstention questions (question_id ending in "_abs") are excluded from the
recall computation, matching LongMemEval's own stated practice ("we
always skip the 30 abstention instances" for retrieval evaluation, per
the upstream README) -- their answer_session_ids point to a topically
related but insufficient session (there is no real evidence location, by
design), so "recall" isn't a coherent concept for them. How many land in
the sample is reported separately, not silently dropped.

Run: python benchmark/longmemeval_benchmark.py
Requires: pip install -e ".[chroma]"
Downloads ~277MB to benchmark/data/ on first run (gitignored, cached
after).
"""

from __future__ import annotations

import json
import os
import random
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

# Must be set before chromadb is imported. Disables chromadb's default
# posthog-based usage telemetry -- verified to default to True in this
# environment, a plausible contributor to occasional network-dependent
# latency spikes observed in early pilot runs (one question took 31
# minutes vs. a ~2 minute typical case, with no corresponding increase in
# haystack size). Not proven as the sole cause, but zero-downside to
# disable in a benchmark meant to be fully local.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import chromadb  # noqa: E402

from memory_system import AlwaysConsolidate, InMemoryBackend, NoDecay, TieredMemory  # noqa: E402
from memory_system.backends.chroma import ChromaBackend  # noqa: E402
from memory_system.backends.hybrid import HybridBackend  # noqa: E402

DATA_URL = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"
DATA_PATH = Path(__file__).resolve().parent / "data" / "longmemeval_s_cleaned.json"

# Fixed seed for a reproducible draw -- see docs/superpowers/specs/2026-07-28-longmemeval-benchmark.md
SEED = 42
TOP_K = 10

# Real question_type population sizes out of the full 500 -- verified by
# parsing the actual dataset file, not the paper's rounded description.
POPULATION_COUNTS = {
    "temporal-reasoning": 133,
    "multi-session": 133,
    "knowledge-update": 78,
    "single-session-user": 70,
    "single-session-assistant": 56,
    "single-session-preference": 30,
}

DEFAULT_SAMPLE_SIZE = 100

BACKEND_NAMES = ["InMemoryBackend", "ChromaBackend", "HybridBackend"]


def compute_stratified_sizes(population_counts: dict[str, int], n_total: int) -> dict[str, int]:
    """Proportional allocation of an n_total-question sample across
    population_counts, largest-remainder rounding to land on exactly
    n_total. For n_total=100 this reproduces the spec's approved
    27/27/15/14/11/6 sizes exactly.
    """
    total_population = sum(population_counts.values())
    exact = {cat: count * n_total / total_population for cat, count in population_counts.items()}
    floors = {cat: int(v) for cat, v in exact.items()}
    remainder = n_total - sum(floors.values())
    by_remainder = sorted(
        population_counts.keys(),
        key=lambda c: (-(exact[c] - floors[c]), -population_counts[c]),
    )
    for cat in by_remainder[:remainder]:
        floors[cat] += 1
    return floors


def download_dataset() -> None:
    if DATA_PATH.exists():
        return
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {DATA_URL} to {DATA_PATH} (~277MB, one-time)...")
    urllib.request.urlretrieve(DATA_URL, DATA_PATH)


def load_dataset() -> list[dict]:
    download_dataset()
    return json.loads(DATA_PATH.read_text())


def sample_subset(all_examples: list[dict], seed: int = SEED, n_total: int = DEFAULT_SAMPLE_SIZE) -> list[dict]:
    """Stratified sample by question_type, uniform within each stratum,
    deterministic given the fixed seed and category iteration order.
    n_total=100 matches the approved spec exactly; smaller n_total values
    (e.g. for a faster pilot run) use the same proportional method, not a
    subset of the 100-question draw.
    """
    by_category: dict[str, list[dict]] = defaultdict(list)
    for ex in all_examples:
        by_category[ex["question_type"]].append(ex)

    sizes = compute_stratified_sizes(POPULATION_COUNTS, n_total)
    rng = random.Random(seed)
    sampled = []
    for category, n in sizes.items():
        pool = by_category[category]
        sampled.extend(rng.sample(pool, n))
    return sampled


def build_memory(backend_name: str, question_id: str) -> TieredMemory:
    if backend_name == "InMemoryBackend":
        backend = InMemoryBackend()
    elif backend_name == "ChromaBackend":
        backend = ChromaBackend(collection_name=f"lme_{question_id}_chroma")
    elif backend_name == "HybridBackend":
        backend = HybridBackend(
            lexical_backend=InMemoryBackend(),
            semantic_backend=ChromaBackend(collection_name=f"lme_{question_id}_hybrid"),
        )
    else:
        raise ValueError(backend_name)

    return TieredMemory(
        backend=backend,
        consolidation_policy=AlwaysConsolidate(),  # required by the constructor, never invoked
        decay_policy=NoDecay(),  # deliberate -- see spec's "what this does not measure"
    )


def cleanup_chroma_collections(question_id: str) -> None:
    """chromadb.Client() shares one in-process system across every
    ChromaBackend instance regardless of Python object lifetime -- a
    collection created by a ChromaBackend that's since been garbage
    collected stays resident until explicitly deleted. Without this,
    100 sequential questions would accumulate ~100 collections' worth of
    embedded vectors for the whole run. Uses a fresh chromadb.Client()
    call, not ChromaBackend itself (which doesn't expose the underlying
    client) -- verified empirically that this client shares the same
    process-wide system, so it can delete collections it didn't create.
    """
    client = chromadb.Client()
    for name in (f"lme_{question_id}_chroma", f"lme_{question_id}_hybrid"):
        try:
            client.delete_collection(name)
        except Exception:
            pass


def ingest_question(memory: TieredMemory, example: dict) -> None:
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


def score_question(memory: TieredMemory, example: dict, top_k: int) -> tuple[int, float | None]:
    results = memory.retrieve(example["question"], top_k=top_k)
    answer_sessions = set(example["answer_session_ids"])
    retrieved_sessions = {r.event.metadata["session_id"] for r in results}
    session_hit = 1 if retrieved_sessions & answer_sessions else 0

    total_relevant_turns = sum(
        1
        for session in example["haystack_sessions"]
        for turn in session
        if turn.get("has_answer", False)
    )
    retrieved_relevant_turns = sum(1 for r in results if r.event.metadata.get("has_answer", False))
    turn_recall = (retrieved_relevant_turns / total_relevant_turns) if total_relevant_turns else None

    return session_hit, turn_recall


def run_benchmark(
    examples: list[dict], backend_names: list[str] = BACKEND_NAMES, top_k: int = TOP_K, verbose: bool = True
) -> dict:
    results = {
        name: {"session_hits": [], "turn_recalls": [], "per_category": defaultdict(list)}
        for name in backend_names
    }
    abstention_ids = [ex["question_id"] for ex in examples if ex["question_id"].endswith("_abs")]

    for i, example in enumerate(examples):
        qid = example["question_id"]
        is_abstention = qid.endswith("_abs")
        for backend_name in backend_names:
            t0 = time.time()
            memory = build_memory(backend_name, qid)
            ingest_question(memory, example)
            session_hit, turn_recall = score_question(memory, example, top_k)
            elapsed = time.time() - t0
            cleanup_chroma_collections(qid)

            if not is_abstention:
                results[backend_name]["session_hits"].append(session_hit)
                if turn_recall is not None:
                    results[backend_name]["turn_recalls"].append(turn_recall)
                results[backend_name]["per_category"][example["question_type"]].append(session_hit)

            if verbose:
                marker = " [ABSTENTION -- excluded from recall]" if is_abstention else ""
                print(
                    f"[{i + 1}/{len(examples)}] {backend_name:16s} {qid:28s} "
                    f"session_hit={session_hit} turn_recall={turn_recall} ({elapsed:.2f}s){marker}"
                )

    results["_abstention_ids"] = abstention_ids
    return results


def summarize(results: dict, backend_names: list[str] = BACKEND_NAMES, seed: int = SEED) -> None:
    print("\n" + "=" * 72)
    print(f"RESULTS (top_k={TOP_K}, seed={seed})")
    print("=" * 72)
    for name in backend_names:
        r = results[name]
        n = len(r["session_hits"])
        session_recall = sum(r["session_hits"]) / n if n else float("nan")
        turn_recalls = r["turn_recalls"]
        turn_recall = sum(turn_recalls) / len(turn_recalls) if turn_recalls else float("nan")
        print(f"\n{name}")
        print(f"  session-level recall@{TOP_K}: {session_recall:.4f}  (n={n})")
        print(f"  turn-level recall@{TOP_K}:    {turn_recall:.4f}  (n={len(turn_recalls)})")
        print("  per-category session-level recall:")
        for category, hits in sorted(r["per_category"].items()):
            print(f"    {category:28s} {sum(hits) / len(hits):.4f}  (n={len(hits)})")

    print(f"\nAbstention questions in sample (excluded from recall, n={len(results['_abstention_ids'])}):")
    for qid in results["_abstention_ids"]:
        print(f"  {qid}")


def main() -> None:
    # Line-buffer stdout regardless of how this is invoked. Without this,
    # piping to a file (nohup ... > log.log &) fully buffers stdout and
    # nothing is written until the process exits -- observed firsthand
    # during the pilot run: ~2 hours with zero visible output despite the
    # process actively computing. `tail -f` on the output file only works
    # as a progress check if this is set.
    sys.stdout.reconfigure(line_buffering=True)

    n_total = DEFAULT_SAMPLE_SIZE
    seed = SEED
    args = sys.argv[1:]
    i = 0
    while i < len(args):
        if args[i] == "--n":
            n_total = int(args[i + 1])
            i += 2
        elif args[i] == "--seed":
            seed = int(args[i + 1])
            i += 2
        else:
            raise SystemExit(f"Unknown argument: {args[i]} (supported: --n N, --seed N)")

    all_examples = load_dataset()
    subset = sample_subset(all_examples, seed=seed, n_total=n_total)

    print(f"Sampled {len(subset)} questions (seed={seed}, n_total={n_total}):")
    for ex in subset:
        print(f"  {ex['question_id']:28s} {ex['question_type']}")

    results = run_benchmark(subset)
    summarize(results, seed=seed)


if __name__ == "__main__":
    main()
