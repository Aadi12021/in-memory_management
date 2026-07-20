"""A dependency-free, TF-IDF-based in-memory backend.

This is a real lexical information-retrieval implementation (term
frequency-inverse document frequency + cosine similarity) -- the same
family of technique classic search engines used before embeddings.
It has no external dependencies (pure stdlib), so `tiered-memory`
works out of the box with zero setup.

What it is NOT: a semantic/embedding-based matcher. A lightweight
stemmer collapses plurals and common verb endings (peanut/peanuts,
hike/hiking), but it won't connect irregular derivations (allergy/
allergic) or synonyms/concepts (peanuts/dietary restrictions) the way
an embedding model would. For that, use ChromaBackend or an
embedding-based backend. This backend is for projects that want good,
dependency-free lexical search, and for testing/CI.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from typing import Optional

from ..events import MemoryEvent, MemoryTier, RetrievalResult
from .base import MemoryBackend


_VOWELS = set("aeiou")


def _dropped_silent_e(stem: str) -> bool:
    """True if `stem` ends in consonant-vowel-consonant (e.g. "hik",
    "liv") -- the pattern left behind when a short silent-e word
    (hike, live, like, love) has its final 'e' dropped before a suffix
    like -ing/-ed/-es is stripped back off.
    """
    if len(stem) < 3:
        return False
    a, b, c = stem[-3], stem[-2], stem[-1]
    return a not in _VOWELS and b in _VOWELS and c not in _VOWELS


def _stem(word: str) -> str:
    """Minimal suffix-stripping stemmer (not a full Porter stemmer, but
    enough to collapse common inflections like peanut/peanuts,
    hike/hiking, allergy/allergies) so exact-token TF-IDF doesn't miss
    obvious matches due to plurals or verb endings.

    After stripping "es"/"ed"/"ing", restores a dropped silent 'e' when
    the resulting stem is consonant-vowel-consonant (hik -> hike,
    liv -> live, lik -> like, lov -> love) -- the common case for
    short, regular silent-e verbs. This is a targeted heuristic, not a
    full Porter-style analysis: it won't catch every silent-e word
    (e.g. longer words with a vowel digraph before the final consonant,
    like believe/believes), and it doesn't address the separate
    consonant-doubling case (run/running, swim/swimming still don't
    collapse).
    """
    for suffix in ("ies", "es", "ing", "ed", "ly", "s"):
        if word.endswith(suffix) and len(word) - len(suffix) >= 3:
            if suffix == "ies":
                return word[: -len(suffix)] + "y"
            stem = word[: -len(suffix)]
            if suffix in ("es", "ed", "ing") and _dropped_silent_e(stem):
                return stem + "e"
            return stem
    return word


def _tokenize(text: str) -> list[str]:
    return [_stem(w) for w in re.findall(r"\w+", str(text).lower())]


class InMemoryBackend(MemoryBackend):
    """TF-IDF + cosine similarity search, computed in pure Python.

    Vocabulary and document frequencies are (re)computed at query time
    over whatever's currently in the tier scope being searched. That's
    O(n) per query rather than maintaining a live index -- the right
    tradeoff for the memory sizes this backend is meant for (hundreds to
    low thousands of events, not millions). If you need to scale past
    that, use ChromaBackend or another vector-store-backed option.
    """

    def __init__(self) -> None:
        self._store: dict[str, MemoryEvent] = {}

    def add(self, event: MemoryEvent) -> None:
        self._store[event.id] = event

    def get_all(self, tier: Optional[MemoryTier] = None) -> list[MemoryEvent]:
        events = list(self._store.values())
        if tier is not None:
            events = [e for e in events if e.tier == tier]
        return events

    def query(
        self, query: str, top_k: int = 5, tier: Optional[MemoryTier] = None
    ) -> list[RetrievalResult]:
        candidates = self.get_all(tier=tier)
        if not candidates:
            return []

        doc_tokens = {e.id: _tokenize(e.content) for e in candidates}
        query_tokens = _tokenize(query)
        if not query_tokens:
            return []

        n_docs = len(candidates)
        doc_freq: Counter[str] = Counter()
        for tokens in doc_tokens.values():
            doc_freq.update(set(tokens))

        def idf(term: str) -> float:
            # smoothed idf: avoids divide-by-zero and dampens rare-term dominance
            return math.log((1 + n_docs) / (1 + doc_freq.get(term, 0))) + 1

        def tfidf_vector(tokens: list[str]) -> dict[str, float]:
            counts = Counter(tokens)
            total = len(tokens) or 1
            return {term: (count / total) * idf(term) for term, count in counts.items()}

        def cosine_sim(vec_a: dict[str, float], vec_b: dict[str, float]) -> float:
            shared = set(vec_a) & set(vec_b)
            if not shared:
                return 0.0
            dot = sum(vec_a[t] * vec_b[t] for t in shared)
            norm_a = math.sqrt(sum(v * v for v in vec_a.values()))
            norm_b = math.sqrt(sum(v * v for v in vec_b.values()))
            if norm_a == 0 or norm_b == 0:
                return 0.0
            return dot / (norm_a * norm_b)

        query_vec = tfidf_vector(query_tokens)

        scored: list[RetrievalResult] = []
        for event in candidates:
            doc_vec = tfidf_vector(doc_tokens[event.id])
            score = cosine_sim(query_vec, doc_vec)
            if score > 0:
                scored.append(RetrievalResult(event=event, score=score, tier=event.tier))

        scored.sort(key=lambda r: r.score, reverse=True)
        return scored[:top_k]

    def update_tier(self, event_id: str, new_tier: MemoryTier) -> None:
        if event_id in self._store:
            self._store[event_id].tier = new_tier

    def remove(self, event_id: str) -> None:
        self._store.pop(event_id, None)
