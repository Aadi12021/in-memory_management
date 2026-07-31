"""LongMemEval benchmark: recall@k for InMemoryBackend, ChromaBackend, and
HybridBackend against a stratified subset of LongMemEval_S (ICLR 2025,
https://github.com/xiaowu0162/LongMemEval).

Not the official metric -- LongMemEval's own evaluate_qa.py measures
LLM-judged QA accuracy of a full reader+generation pipeline, which
tiered-memory doesn't have (it only returns ranked RetrievalResults, no
answer generation). This measures recall@k against the dataset's own
ground-truth labels (has_answer, answer_session_ids) instead: does
retrieve() surface the evidence session/turns, not whether a downstream
reader could correctly answer from them. See
docs/superpowers/specs/2026-07-28-longmemeval-benchmark.md for the full
design rationale.

Abstention questions (question_id ending in "_abs") are skipped entirely
(not ingested, not queried) rather than scored and discarded, matching
LongMemEval's own stated practice ("we always skip the 30 abstention
instances" for retrieval evaluation, per the upstream README). Their
answer_session_ids point to a topically related session, not necessarily
one containing real evidence -- 9 of the 30 abstention questions in the
full dataset do carry has_answer turns, 21 don't, so "there is no
evidence location" is not a blanket description, just the reason
"recall" isn't treated as a coherent concept for them here. How many
land in a given sample is reported separately, not silently dropped.

Run: python benchmark/longmemeval_benchmark.py [--n N] [--seed N]
     [--top-k N] [--out results.json]
Requires: pip install -e ".[chroma]"
Downloads ~277MB to benchmark/data/ on first run (gitignored, cached and
integrity-checked against a pinned sha256 on every run after).
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

# Must be set before chromadb is imported. Disables chromadb's default
# posthog-based usage telemetry -- verified to default to True in this
# environment. Plausible but NOT confirmed as a fix for the latency
# spikes observed during piloting: in the full 100-question run with
# telemetry already disabled, ChromaBackend's median call was 28.9s /
# p90 32.8s, but one call still took 930.8s. Disabling telemetry is kept
# regardless, since it's zero-downside and correct for a benchmark
# that's supposed to be fully local -- but it did not eliminate the
# outlier behavior, and that remains unexplained.
os.environ.setdefault("ANONYMIZED_TELEMETRY", "False")

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import chromadb  # noqa: E402

from memory_system import AlwaysConsolidate, InMemoryBackend, NoDecay, TieredMemory  # noqa: E402
from memory_system.backends.chroma import ChromaBackend  # noqa: E402
from memory_system.backends.hybrid import HybridBackend  # noqa: E402

DATA_URL = "https://huggingface.co/datasets/xiaowu0162/longmemeval-cleaned/resolve/main/longmemeval_s_cleaned.json"
DATA_PATH = Path(__file__).resolve().parent / "data" / "longmemeval_s_cleaned.json"

# Pinned hash of the file as fetched 2026-07-29 -- see docs/superpowers/specs/
# 2026-07-28-longmemeval-benchmark.md. Guards against a truncated/corrupted
# download being silently treated as valid on a later run, and against
# unnoticed upstream changes to the "cleaned" dataset breaking
# reproducibility (see the 2025/09 "further cleaned up" changelog entry
# in the upstream README -- this file has moved before).
EXPECTED_DATA_SHA256 = "d6f21ea9d60a0d56f34a05b609c79c88a451d2ae03597821ea3d5a9678c3a442"

# Fixed default seed for a reproducible draw -- see docs/superpowers/specs/2026-07-28-longmemeval-benchmark.md
SEED = 42
TOP_K = 10
DEFAULT_SAMPLE_SIZE = 100

# Real question_type population sizes out of the full 500 -- verified by
# parsing the actual dataset file, not the paper's rounded description.
# validate_population_counts() checks this against the loaded file on
# every run, so a silent mismatch (e.g. upstream changes the dataset)
# fails loudly instead of sampling from the wrong proportions.
POPULATION_COUNTS = {
    "temporal-reasoning": 133,
    "multi-session": 133,
    "knowledge-update": 78,
    "single-session-user": 70,
    "single-session-assistant": 56,
    "single-session-preference": 30,
}

BACKEND_NAMES = ["InMemoryBackend", "ChromaBackend", "HybridBackend"]


def compute_stratified_sizes(population_counts: dict[str, int], n_total: int) -> dict[str, int]:
    """Proportional allocation of an n_total-question sample across
    population_counts, largest-remainder rounding to land on exactly
    n_total. For n_total=100 this reproduces the spec's approved
    27/27/15/14/11/6 sizes exactly.

    Uses exact integer arithmetic (floor division + integer remainder),
    not floating point -- 133*100/500 and 78*100/500 are both
    mathematically exactly .6, but as floats they come out as
    0.6000000000000014 vs 0.5999999999999996, so a float-based remainder
    comparison gets the right answer for this specific case by luck, not
    by construction. Integer arithmetic makes the same tie-break
    (population_counts[c] as the secondary sort key) actually decide it.
    """
    total_population = sum(population_counts.values())
    numerators = {cat: count * n_total for cat, count in population_counts.items()}
    floors = {cat: num // total_population for cat, num in numerators.items()}
    remainder = n_total - sum(floors.values())
    by_remainder = sorted(
        population_counts.keys(),
        key=lambda c: (-(numerators[c] % total_population), -population_counts[c]),
    )
    for cat in by_remainder[:remainder]:
        floors[cat] += 1
    return floors


def sha256_of(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_dataset() -> None:
    """Downloads to a .part file and verifies its hash before the final
    rename, so an interrupted download can never leave a truncated file
    at DATA_PATH masquerading as a valid cache. Re-verifies the hash of
    an already-cached file on every call too (~1-2s for 277MB) rather
    than trusting DATA_PATH.exists() alone -- a corrupted cache from a
    prior run would otherwise persist silently until manually deleted.
    """
    DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    if DATA_PATH.exists():
        if sha256_of(DATA_PATH) == EXPECTED_DATA_SHA256:
            return
        print(f"Cached file at {DATA_PATH} failed the integrity check -- re-downloading.")

    tmp_path = DATA_PATH.with_suffix(".json.part")
    print(f"Downloading {DATA_URL} to {DATA_PATH} (~277MB, one-time)...")
    urllib.request.urlretrieve(DATA_URL, tmp_path)

    actual = sha256_of(tmp_path)
    if actual != EXPECTED_DATA_SHA256:
        tmp_path.unlink()
        raise RuntimeError(
            f"Downloaded file hash mismatch: expected {EXPECTED_DATA_SHA256}, got {actual}. "
            "The download may have been interrupted, or the upstream dataset may have changed "
            "(update EXPECTED_DATA_SHA256 if so, after verifying the new file by hand). "
            "Deleted the partial download; re-run to retry."
        )
    tmp_path.replace(DATA_PATH)


def load_dataset() -> list[dict]:
    download_dataset()
    with DATA_PATH.open("r", encoding="utf-8") as f:
        return json.load(f)


def validate_population_counts(all_examples: list[dict]) -> None:
    actual: dict[str, int] = defaultdict(int)
    for ex in all_examples:
        actual[ex["question_type"]] += 1
    if dict(actual) != POPULATION_COUNTS:
        raise RuntimeError(
            f"POPULATION_COUNTS is stale: expected {POPULATION_COUNTS}, "
            f"the loaded dataset actually has {dict(actual)}. The dataset file may have "
            "changed upstream -- update POPULATION_COUNTS (and re-derive the spec's "
            "stratified sample sizes) before trusting a sample drawn from it."
        )


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

    Checks list_collections() first rather than try/except around
    delete_collection() -- chromadb raises chromadb.errors.NotFoundError
    on a missing collection (verified on 1.5.9), but that exception class
    doesn't exist on chromadb<1.0 (this project's declared floor is
    >=0.4.0, which raised plain ValueError instead), so catching it by
    name isn't version-safe, and a bare `except Exception: pass` would
    silently swallow a genuine failure with zero visibility. Checking
    membership first means delete_collection() is only ever called when
    the collection is known to exist, so any exception it raises is a
    real problem and is allowed to propagate.
    """
    client = chromadb.Client()
    existing = {getattr(c, "name", c) for c in client.list_collections()}
    for name in (f"lme_{question_id}_chroma", f"lme_{question_id}_hybrid"):
        if name in existing:
            client.delete_collection(name)


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
    backends_out = {
        name: {"session_hits": [], "turn_recalls": [], "per_category": defaultdict(list)}
        for name in backend_names
    }
    abstention_ids = [ex["question_id"] for ex in examples if ex["question_id"].endswith("_abs")]

    for i, example in enumerate(examples):
        qid = example["question_id"]
        if qid.endswith("_abs"):
            if verbose:
                print(f"[{i + 1}/{len(examples)}] SKIPPED (abstention, excluded from recall): {qid}")
            continue

        for backend_name in backend_names:
            t0 = time.time()
            memory = build_memory(backend_name, qid)
            ingest_question(memory, example)
            session_hit, turn_recall = score_question(memory, example, top_k)
            elapsed = time.time() - t0
            cleanup_chroma_collections(qid)

            backends_out[backend_name]["session_hits"].append(session_hit)
            if turn_recall is not None:
                backends_out[backend_name]["turn_recalls"].append(turn_recall)
            backends_out[backend_name]["per_category"][example["question_type"]].append(session_hit)

            if verbose:
                print(
                    f"[{i + 1}/{len(examples)}] {backend_name:16s} {qid:28s} "
                    f"session_hit={session_hit} turn_recall={turn_recall} ({elapsed:.2f}s)"
                )

    for name in backends_out:
        backends_out[name]["per_category"] = dict(backends_out[name]["per_category"])

    return {"backends": backends_out, "abstention_ids": abstention_ids}


def summarize(results: dict, backend_names: list[str] = BACKEND_NAMES, seed: int = SEED, top_k: int = TOP_K) -> dict:
    """Prints the human-readable report and returns a structured summary
    dict (JSON-serializable) covering the same numbers, for --out.
    """
    backends = results["backends"]
    summary = {"top_k": top_k, "seed": seed, "backends": {}, "abstention_ids": results["abstention_ids"]}

    print("\n" + "=" * 72)
    print(f"RESULTS (top_k={top_k}, seed={seed})")
    print("=" * 72)
    for name in backend_names:
        r = backends[name]
        n = len(r["session_hits"])
        session_recall = sum(r["session_hits"]) / n if n else float("nan")
        turn_recalls = r["turn_recalls"]
        turn_recall = sum(turn_recalls) / len(turn_recalls) if turn_recalls else float("nan")
        per_category = {cat: sum(hits) / len(hits) for cat, hits in sorted(r["per_category"].items())}

        summary["backends"][name] = {
            "session_recall": session_recall,
            "session_n": n,
            "turn_recall": turn_recall,
            "turn_n": len(turn_recalls),
            "per_category_session_recall": per_category,
        }

        print(f"\n{name}")
        print(f"  session-level recall@{top_k}: {session_recall:.4f}  (n={n})")
        print(f"  turn-level recall@{top_k}:    {turn_recall:.4f}  (n={len(turn_recalls)})")
        print("  per-category session-level recall:")
        for category, value in per_category.items():
            print(f"    {category:28s} {value:.4f}  (n={len(r['per_category'][category])})")

    print(f"\nAbstention questions in sample (excluded from recall, n={len(results['abstention_ids'])}):")
    for qid in results["abstention_ids"]:
        print(f"  {qid}")

    return summary


def get_git_commit() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=Path(__file__).resolve().parent.parent,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except Exception:
        return None


def compute_provenance() -> dict:
    """Records what a third party would need to reproduce this run:
    the exact dataset (by hash, since the upstream URL is an unpinned
    branch ref), the exact chromadb version (its default embedding
    function is version-dependent, so this affects ChromaBackend's and
    HybridBackend's numbers), the package version, and the git commit
    this benchmark script itself was run from.
    """
    return {
        "dataset_url": DATA_URL,
        "dataset_sha256": sha256_of(DATA_PATH),
        "chromadb_version": chromadb.__version__,
        "package_version": __import__("memory_system").__version__,
        "git_commit": get_git_commit(),
    }


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=DEFAULT_SAMPLE_SIZE, dest="n_total", help="stratified sample size (default: 100)")
    parser.add_argument("--seed", type=int, default=SEED, help="random seed for sampling (default: 42)")
    parser.add_argument("--top-k", type=int, default=TOP_K, dest="top_k", help="top_k for retrieve() calls (default: 10)")
    parser.add_argument("--out", type=Path, default=None, help="optional path to write structured JSON results")
    return parser.parse_args(argv)


def main() -> None:
    # Line-buffer stdout regardless of how this is invoked. Without this,
    # piping to a file (nohup ... > log.log &) fully buffers stdout and
    # nothing is written until the process exits -- observed firsthand
    # during the pilot run: ~2 hours with zero visible output despite the
    # process actively computing. `tail -f` on the output file only works
    # as a progress check if this is set.
    sys.stdout.reconfigure(line_buffering=True)

    args = parse_args(sys.argv[1:])
    total_population = sum(POPULATION_COUNTS.values())
    if args.n_total <= 0:
        raise SystemExit(f"--n must be positive, got {args.n_total}")
    if args.n_total > total_population:
        raise SystemExit(f"--n {args.n_total} exceeds the total population ({total_population})")

    all_examples = load_dataset()
    validate_population_counts(all_examples)
    subset = sample_subset(all_examples, seed=args.seed, n_total=args.n_total)

    print(f"Sampled {len(subset)} questions (seed={args.seed}, n_total={args.n_total}):")
    for ex in subset:
        print(f"  {ex['question_id']:28s} {ex['question_type']}")

    provenance = compute_provenance()
    print("\nProvenance:")
    for key, value in provenance.items():
        print(f"  {key}: {value}")

    results = run_benchmark(subset, top_k=args.top_k)
    summary = summarize(results, seed=args.seed, top_k=args.top_k)

    if args.out:
        args.out.write_text(json.dumps({"provenance": provenance, "summary": summary}, indent=2))
        print(f"\nWrote structured results to {args.out}")


if __name__ == "__main__":
    main()
