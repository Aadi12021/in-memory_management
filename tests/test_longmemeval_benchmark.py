"""Tests for the pure, non-network functions in
benchmark/longmemeval_benchmark.py: the stratified sampling arithmetic
and the recall@k scoring logic. Doesn't touch the real dataset download,
chromadb, or TieredMemory -- those are exercised by actually running the
benchmark, not by this suite.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "benchmark"))

from longmemeval_benchmark import compute_stratified_sizes, score_question


# --- compute_stratified_sizes ------------------------------------------

REAL_POPULATION = {
    "temporal-reasoning": 133,
    "multi-session": 133,
    "knowledge-update": 78,
    "single-session-user": 70,
    "single-session-assistant": 56,
    "single-session-preference": 30,
}


def test_reproduces_spec_approved_100_question_sizes():
    sizes = compute_stratified_sizes(REAL_POPULATION, 100)

    assert sizes == {
        "temporal-reasoning": 27,
        "multi-session": 27,
        "knowledge-update": 15,
        "single-session-user": 14,
        "single-session-assistant": 11,
        "single-session-preference": 6,
    }


def test_reproduces_20_question_pilot_sizes():
    sizes = compute_stratified_sizes(REAL_POPULATION, 20)

    assert sizes == {
        "temporal-reasoning": 6,
        "multi-session": 5,
        "knowledge-update": 3,
        "single-session-user": 3,
        "single-session-assistant": 2,
        "single-session-preference": 1,
    }


def test_sizes_always_sum_to_n_total():
    for n in (1, 7, 17, 50, 100, 250, 500):
        sizes = compute_stratified_sizes(REAL_POPULATION, n)
        assert sum(sizes.values()) == n


def test_tie_break_uses_exact_integer_remainder_not_float_rounding():
    # 133*10/500 and 78*10/500 are both mathematically fractional in a
    # way that floating point represents imprecisely (133*100/500 and
    # 78*100/500 are *exactly* equal fractional parts at n=100, which is
    # the case that silently depended on float representation error
    # before this function used integer arithmetic). This exact
    # breakdown (temporal-reasoning's larger remainder and larger raw
    # count both point the same way) was verified against a real run of
    # the function, not hand-derived.
    population = {"temporal-reasoning": 133, "knowledge-update": 78, "filler": 289}

    sizes = compute_stratified_sizes(population, 10)

    assert sizes == {"temporal-reasoning": 3, "knowledge-update": 1, "filler": 6}
    assert sum(sizes.values()) == 10


def test_single_category_gets_everything():
    assert compute_stratified_sizes({"only": 500}, 42) == {"only": 42}


def test_zero_n_total_gives_all_zeros():
    assert compute_stratified_sizes({"a": 100, "b": 200}, 0) == {"a": 0, "b": 0}


# --- score_question ------------------------------------------------------


class FakeEvent:
    def __init__(self, metadata):
        self.metadata = metadata


class FakeResult:
    def __init__(self, metadata):
        self.event = FakeEvent(metadata)


class FakeMemory:
    """Test double: retrieve() returns a pre-set list of results
    regardless of the query, so score_question's own logic can be tested
    without a real TieredMemory/backend.
    """

    def __init__(self, results):
        self._results = results

    def retrieve(self, query, top_k):
        return self._results[:top_k]


def make_example(haystack_sessions, answer_session_ids):
    return {
        "question": "irrelevant for this test",
        "haystack_sessions": haystack_sessions,
        "answer_session_ids": answer_session_ids,
    }


def test_session_hit_when_a_retrieved_event_is_in_an_answer_session():
    memory = FakeMemory([FakeResult({"session_id": "s1", "has_answer": True})])
    example = make_example(
        haystack_sessions=[[{"has_answer": True}]],
        answer_session_ids=["s1"],
    )

    session_hit, turn_recall = score_question(memory, example, top_k=5)

    assert session_hit == 1
    assert turn_recall == 1.0


def test_session_miss_when_no_retrieved_event_is_in_an_answer_session():
    memory = FakeMemory([FakeResult({"session_id": "unrelated", "has_answer": False})])
    example = make_example(
        haystack_sessions=[[{"has_answer": True}]],
        answer_session_ids=["s1"],
    )

    session_hit, turn_recall = score_question(memory, example, top_k=5)

    assert session_hit == 0
    assert turn_recall == 0.0


def test_turn_recall_is_fraction_of_relevant_turns_retrieved():
    # 3 relevant turns exist in the haystack; only 1 is retrieved.
    memory = FakeMemory([FakeResult({"session_id": "s1", "has_answer": True})])
    example = make_example(
        haystack_sessions=[[{"has_answer": True}, {"has_answer": True}, {"has_answer": True}]],
        answer_session_ids=["s1"],
    )

    _, turn_recall = score_question(memory, example, top_k=5)

    assert turn_recall == 1 / 3


def test_turn_recall_is_none_when_haystack_has_no_relevant_turns():
    memory = FakeMemory([FakeResult({"session_id": "s1", "has_answer": False})])
    example = make_example(
        haystack_sessions=[[{"has_answer": False}]],
        answer_session_ids=["s1"],
    )

    _, turn_recall = score_question(memory, example, top_k=5)

    assert turn_recall is None


def test_only_top_k_results_count_toward_scoring():
    # 2 relevant turns exist in the haystack; retrieve() would return
    # both, but top_k=1 means only the first should count.
    memory = FakeMemory(
        [
            FakeResult({"session_id": "s1", "has_answer": True}),
            FakeResult({"session_id": "s1", "has_answer": True}),
        ]
    )
    example = make_example(
        haystack_sessions=[[{"has_answer": True}, {"has_answer": True}]],
        answer_session_ids=["s1"],
    )

    _, turn_recall = score_question(memory, example, top_k=1)

    assert turn_recall == 0.5
