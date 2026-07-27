"""Tests for PerceptSalienceScorer, PERCEPT-1's ported cosine-distance
surprise signal wired up as a real SalienceScorer.

Uses a real (small, local) sentence-transformers model rather than a
mock -- there's no network call involved (embedding is local), so this
stays fast, and the whole point of these tests is to verify the actual
cosine-distance direction survived the port correctly. Skipped if the
`percept` extra isn't installed.
"""

from unittest.mock import patch

import pytest

pytest.importorskip("sentence_transformers")

from memory_system.events import MemoryEvent, MemoryTier
from memory_system.backends.memory import InMemoryBackend
from memory_system.policies.decay import NoDecay
from memory_system.policies.percept_salience import PerceptSalienceScorer


@pytest.fixture(scope="module")
def scorer():
    backend = InMemoryBackend()
    return PerceptSalienceScorer(backend=backend, decay_policy=NoDecay())


def make_backend_with_long_term_facts(facts: list[str]) -> InMemoryBackend:
    backend = InMemoryBackend()
    for fact in facts:
        event = MemoryEvent(content=fact, tier=MemoryTier.LONG_TERM)
        backend.add(event)
    return backend


def test_score_is_normalized_to_unit_interval(scorer):
    scorer.backend = make_backend_with_long_term_facts(
        ["The user is a software engineer who writes Python."]
    )
    score = scorer.score(MemoryEvent(content="The user writes Python code."))
    assert 0.0 <= score <= 1.0


def test_novel_content_scores_higher_than_related_content(scorer):
    scorer.backend = make_backend_with_long_term_facts(
        ["The user is a software engineer who writes Python code daily."]
    )

    related_score = scorer.score(
        MemoryEvent(content="The user wrote some Python code today.")
    )
    unrelated_score = scorer.score(
        MemoryEvent(content="The stock market crashed today amid inflation fears.")
    )

    assert unrelated_score > related_score


def test_empty_long_term_tier_still_returns_valid_score(scorer):
    scorer.backend = InMemoryBackend()
    score = scorer.score(MemoryEvent(content="Anything at all."))
    assert 0.0 <= score <= 1.0


def test_working_tier_events_do_not_affect_the_prior(scorer):
    empty_backend = InMemoryBackend()

    working_junk_backend = InMemoryBackend()
    working_junk_backend.add(
        MemoryEvent(content="Irrelevant working-memory note.", tier=MemoryTier.WORKING)
    )

    event = MemoryEvent(content="Some new content to score.")

    scorer.backend = empty_backend
    score_with_empty_tier = scorer.score(event)

    scorer.backend = working_junk_backend
    score_with_working_junk = scorer.score(event)

    assert score_with_empty_tier == pytest.approx(score_with_working_junk)


def test_score_calls_build_semantic_profile_with_long_term_events_and_decay_policy(scorer):
    backend = make_backend_with_long_term_facts(["fact one", "fact two"])
    scorer.backend = backend
    long_term_events = backend.get_all(tier=MemoryTier.LONG_TERM)

    with patch(
        "memory_system.policies.percept_salience.build_semantic_profile",
        return_value={},
    ) as mock_build:
        scorer.score(MemoryEvent(content="new content"))

    assert mock_build.call_count == 1
    args, kwargs = mock_build.call_args
    called_events = args[0]
    called_decay_policy = args[1]
    assert {e.id for e in called_events} == {e.id for e in long_term_events}
    assert called_decay_policy is scorer.decay_policy
