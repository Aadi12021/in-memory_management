"""Retrieval quality benchmark: InMemoryBackend (TF-IDF + cosine +
stemming) vs. a naive substring/exact-match baseline, on a synthetic
set of facts and queries.

Not a rigorous IR benchmark (no held-out set, no cross-validation, a
single human-judged relevant fact per query) -- it's meant to produce
one honest, reproducible number for the README, not a research result.

Ground truth for each query was chosen by hand *before* running either
algorithm, based on which fact a person would say the query is
"about." Results are reported as-is, including any query where
either backend gets it wrong -- nothing here was tuned after the fact
to make either method look better.

The naive baseline: lowercase + `\\w+` tokenize (no stemming, no
stopword removal), score a fact by how many distinct query tokens
appear in it, and take the argmax (ties broken by earliest-added
fact). This is what "search" means in a lot of hand-rolled agent
memory: literal word matching, nothing else. It's a real, common
baseline, not a strawman -- see README for how this differs from the
InMemoryBackend the library actually ships.

Run: python benchmark/retrieval_benchmark.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from memory_system.backends.memory import InMemoryBackend  # noqa: E402
from memory_system.events import MemoryEvent  # noqa: E402

# --- synthetic fact set ------------------------------------------------------
# Deliberately includes: regular plurals (peanut/peanuts, meal/meals,
# sport/sports, cat/cats, product/products), regular -ed/-ing verb forms
# (play/played), and facts that share a common word ("user", "the") so a
# naive matcher has to fall back on something to break ties -- exactly the
# case InMemoryBackend's IDF weighting is for.

FACTS = [
    "The user is severely allergic to peanuts and tree nuts.",                    # 0
    "The user once mentioned peanuts in passing during a story about school lunches.",  # 1
    "The user is allergic to shellfish, especially shrimp and crab.",             # 2
    "The user's cousin has a severe allergy to bee stings.",                      # 3
    "The user enjoys long walks in the park every morning.",                      # 4
    "The user's dog loves walking through the neighborhood at dusk.",             # 5
    "The user likes quinoa salad for lunch most days.",                           # 6
    "The user likes rice with almost every meal.",                                # 7
    "The user likes pasta on Friday nights.",                                     # 8
    "The user works at a nonprofit focused on ocean conservation.",               # 9
    "The user's sister works at a hospital as a nurse.",                          # 10
    "The user lives in Boston near the Charles River.",                          # 11
    "The user previously lived in Chicago for five years.",                       # 12
    "The user dislikes cold weather and avoids winter sports.",                   # 13
    "The user's favorite color is teal.",                                        # 14
    "The user's least favorite color is orange.",                                # 15
    "The user plays the guitar every evening after dinner.",                      # 16
    "The user used to play the violin as a child.",                              # 17
    "The user is training for a marathon next spring.",                          # 18
    "The user finished a half marathon last year in the rain.",                   # 19
    "The user drinks coffee every morning without fail.",                         # 20
    "The user avoids caffeine after three in the afternoon.",                     # 21
    "The user has two cats named Biscuit and Waffles.",                          # 22
    "The user adopted a rescue dog named Juniper.",                              # 23
    "The user is vegetarian but eats eggs and dairy.",                            # 24
    "The user follows a strict vegan diet with no animal products.",              # 25
]

# (query, expected_fact_index, what it's testing)
QUERIES = [
    ("peanut nut", 0, "singular query vs. plural fact (peanut/peanuts, nut/nuts); no exact-token overlap exists at all"),
    ("tree nut", 0, "singular query vs. plural fact (nut/nuts)"),
    ("shellfish shrimp crab", 2, "control: exact words present"),
    ("bee sting allergy", 3, "control: exact words present"),
    ("walking the dog", 5, "ranking: 'dog' + 'walk' both point to fact 5, not 4 or 23"),
    ("quinoa", 6, "control: rare word, exact match"),
    ("rice meals", 7, "plural query vs. singular fact (meal/meals)"),
    ("pasta Friday", 8, "control: exact words present"),
    ("ocean conservation nonprofit", 9, "control: rare words, exact match"),
    ("nurse hospital", 10, "control: exact words present"),
    ("what does the user drink each morning", 20, "common-word tie ('the'/'user'/'morning' shared with fact 4) needs IDF to break correctly"),
    ("lived in Chicago", 12, "control: exact words present"),
    ("cold weather winter sport", 13, "singular query vs. plural fact (sport/sports)"),
    ("favorite color teal", 14, "ranking: disambiguate from fact 15 (also 'favorite color')"),
    ("least favorite color orange", 15, "ranking: disambiguate from fact 14"),
    ("guitar evening", 16, "control: exact words present"),
    ("played violin as a child", 17, "verb tense: played/play"),
    ("training for a marathon", 18, "ranking: disambiguate from fact 19 (also mentions marathon)"),
    ("finished a half marathon", 19, "ranking: disambiguate from fact 18"),
    ("morning coffee", 20, "control: exact words present"),
    ("caffeine afternoon", 21, "control: rare words, exact match"),
    ("cats named", 22, "plural query vs. plural fact, but 'cat' stem needed to rank above other facts"),
    ("rescue dog Juniper", 23, "control: rare proper noun, exact match"),
    ("vegan diet animal products", 25, "plural query vs. plural fact (product/products)"),
]


def naive_top1(facts: list[str], query: str) -> int:
    """argmax of distinct-token overlap, no stemming, no stopword removal.
    Ties among nonzero scores go to the earliest-added fact (deterministic,
    and if anything generous to naive -- a real system would tie-break
    arbitrarily). Zero overlap across every fact returns -1 ("no match"),
    not an arbitrary index -- crediting naive with a correct answer it
    reached by tie-break alone, on a query where it found literally
    nothing in common with any fact, would overstate it.
    """
    query_tokens = set(re.findall(r"\w+", query.lower()))
    best_index = -1
    best_score = 0
    for i, fact in enumerate(facts):
        fact_tokens = set(re.findall(r"\w+", fact.lower()))
        score = len(query_tokens & fact_tokens)
        if score > best_score:
            best_score = score
            best_index = i
    return best_index


def inmemory_top1(backend: InMemoryBackend, event_by_fact: dict[int, MemoryEvent], query: str) -> int:
    results = backend.query(query, top_k=1)
    if not results:
        return -1
    winning_id = results[0].event.id
    for index, event in event_by_fact.items():
        if event.id == winning_id:
            return index
    return -1


def main() -> None:
    backend = InMemoryBackend()
    event_by_fact: dict[int, MemoryEvent] = {}
    for i, fact in enumerate(FACTS):
        event = MemoryEvent(content=fact)
        backend.add(event)
        event_by_fact[i] = event

    naive_correct = 0
    inmemory_correct = 0
    rows = []

    for query, expected, note in QUERIES:
        naive_guess = naive_top1(FACTS, query)
        inmemory_guess = inmemory_top1(backend, event_by_fact, query)

        naive_ok = naive_guess == expected
        inmemory_ok = inmemory_guess == expected
        naive_correct += naive_ok
        inmemory_correct += inmemory_ok

        rows.append((query, expected, naive_guess, naive_ok, inmemory_guess, inmemory_ok, note))

    total = len(QUERIES)

    print(f"{'query':<38}{'expected':>9}{'naive':>8}{'ok':>4}   {'inmem':>6}{'ok':>4}   note")
    print("-" * 120)
    for query, expected, naive_guess, naive_ok, inmemory_guess, inmemory_ok, note in rows:
        print(
            f"{query:<38}{expected:>9}{naive_guess:>8}{'Y' if naive_ok else 'N':>4}   "
            f"{inmemory_guess:>6}{'Y' if inmemory_ok else 'N':>4}   {note}"
        )

    print("-" * 120)
    print(f"naive substring/exact-match: {naive_correct}/{total} correct top-1")
    print(f"InMemoryBackend (TF-IDF + stemming): {inmemory_correct}/{total} correct top-1")


if __name__ == "__main__":
    main()
