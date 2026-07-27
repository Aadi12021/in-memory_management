"""Bridges TieredMemory's long-term tier into the profile shape
PERCEPT-1's PredictiveCoder expects (see percept_salience.py for the
SalienceScorer that consumes this).

PredictiveCoder treats its semantic_profile.json as a bare
Dict[str, Any]: falsy values get pruned, then the whole thing is
json.dumps'd into one sentence used as the embedding "prior". There's
no fixed field schema -- only that shape -- so the long-term tier is
represented here as a single list of known-fact strings.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from ..events import MemoryEvent
from .decay import DecayPolicy


def build_semantic_profile(
    long_term_events: list[MemoryEvent],
    decay_policy: DecayPolicy,
    max_events: int = 25,
    max_chars_per_event: int = 300,
) -> dict[str, Any]:
    """Builds a PredictiveCoder-consumable profile dict from the
    long-term tier.

    Ranked by decay_policy.current_strength(event) descending -- not
    event.salience, which is a snapshot of novelty at ingestion time.
    Strength reflects what the system stably knows *now*: a memory
    that was salient once but never reinforced isn't necessarily part
    of stable knowledge, while one that's held up over time is.

    Capped at max_events, since all-MiniLM-L6-v2 silently truncates at
    ~256 tokens -- concatenating the entire long-term tier would just
    lose the tail of it unnoticed. Each fact is also truncated to
    max_chars_per_event for the same reason.

    Empty long-term tier -> {} (mirrors PERCEPT-1's own "no profile
    file" case, which _summarize_prior treats as "no prior, everything
    novel").
    """
    if not long_term_events:
        return {}

    now = datetime.now(timezone.utc)
    ranked = sorted(
        long_term_events,
        key=lambda event: decay_policy.current_strength(event, now),
        reverse=True,
    )[:max_events]

    facts = []
    for event in ranked:
        text = str(event.content)
        if len(text) > max_chars_per_event:
            text = text[:max_chars_per_event] + "...[truncated]"
        facts.append(text)

    return {"long_term_memories": facts}
