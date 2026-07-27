"""SalienceScorer backed by PERCEPT-1's predictive-coding surprise
signal (see ~/predictive_coder.py in the Cognitive Digital Twin
series for the original).

Ports only the embedding + cosine-distance half of PredictiveCoder,
deliberately skipping:
  - MultimodalBinder's vision path (Claude API calls for images)
  - _generate_prior_diff, an unconditional Claude API call on every
    perceive() -- this scorer needs to stay cheap enough to sit in
    TieredMemory.store()'s hot path, so it makes no LLM calls at all.

Where PERCEPT-1 reads its "prior" from a static semantic_profile.json
file (MEMEX-1's Tier 3 profile), this scorer derives the prior live
from TieredMemory's own long-term tier via build_semantic_profile.

Requires the `percept` extra: sentence-transformers, numpy.
    pip install tiered-memory[percept]
"""

from __future__ import annotations

import json
from typing import Any, Optional

try:
    import numpy as np
    from sentence_transformers import SentenceTransformer
except ImportError:  # pragma: no cover
    np = None
    SentenceTransformer = None

from ..backends.base import MemoryBackend
from ..events import MemoryEvent, MemoryTier
from .decay import DecayPolicy
from .percept_bridge import build_semantic_profile
from .salience import SalienceScorer


def _summarize_prior(profile: dict[str, Any]) -> str:
    """Ported from PredictiveCoder._summarize_prior: turns the profile
    dict into the natural-language "prior" sentence that gets embedded.
    """
    if not profile or all(not v for v in profile.values()):
        return "No prior context established. All inputs treated as maximally novel."
    pruned = {k: v for k, v in profile.items() if v}
    return f"Known global context: {json.dumps(pruned)}"


def _cosine_distance(vec_a: "np.ndarray", vec_b: "np.ndarray") -> float:
    """Ported from PredictiveCoder._cosine_distance. Cosine distance in
    [0, 2]: 0 = identical, 2 = opposite.
    """
    dot = np.dot(vec_a, vec_b)
    norm = np.linalg.norm(vec_a) * np.linalg.norm(vec_b)
    if norm == 0:
        return 1.0
    return float(1.0 - dot / norm)


class PerceptSalienceScorer(SalienceScorer):
    """Real novelty/surprise SalienceScorer, ported from PERCEPT-1's
    PredictiveCoder: embeds the incoming event content and a "prior"
    summary of what the system already stably knows, then scores
    salience as how far the new content sits from that prior (cosine
    distance, normalized from its native [0, 2] range to [0, 1] by
    halving).

    Local and synchronous only -- no Anthropic calls -- so it's safe
    to run inline in TieredMemory.store()'s hot path. It calls
    backend.get_all(tier=MemoryTier.LONG_TERM) fresh on every score()
    to build the prior: cheap for InMemoryBackend, but a real,
    non-trivial cost worth knowing about for large ChromaBackend
    collections, since it re-fetches (and re-embeds) the whole
    long-term tier on every single store() call.
    """

    def __init__(
        self,
        backend: MemoryBackend,
        decay_policy: DecayPolicy,
        embedding_model: str = "all-MiniLM-L6-v2",
        max_events: int = 25,
        max_chars_per_event: int = 300,
    ):
        self.backend = backend
        self.decay_policy = decay_policy
        self.max_events = max_events
        self.max_chars_per_event = max_chars_per_event
        self.encoder = SentenceTransformer(embedding_model)

    def score(self, event: MemoryEvent, context: Optional[Any] = None) -> float:
        long_term_events = self.backend.get_all(tier=MemoryTier.LONG_TERM)
        profile = build_semantic_profile(
            long_term_events,
            self.decay_policy,
            max_events=self.max_events,
            max_chars_per_event=self.max_chars_per_event,
        )
        prior_summary = _summarize_prior(profile)

        prior_vec = self.encoder.encode(prior_summary, normalize_embeddings=True)
        actual_vec = self.encoder.encode(str(event.content), normalize_embeddings=True)

        surprise_score = _cosine_distance(prior_vec, actual_vec)
        return min(max(surprise_score / 2.0, 0.0), 1.0)
