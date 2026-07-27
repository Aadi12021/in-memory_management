from datetime import datetime, timezone

from memory_system.events import MemoryEvent, MemoryTier
from memory_system.policies.decay import DecayPolicy
from memory_system.policies.percept_bridge import build_semantic_profile


class FakeDecayPolicy(DecayPolicy):
    """Returns a caller-controlled strength per event id, so ranking
    order can be asserted without depending on real elapsed-time math.
    """

    def __init__(self, strengths: dict[str, float], default: float = 0.0):
        self.strengths = strengths
        self.default = default

    def current_strength(self, event: MemoryEvent, now: datetime | None = None) -> float:
        return self.strengths.get(event.id, self.default)


def make_event(content, event_id=None) -> MemoryEvent:
    event = MemoryEvent(content=content, tier=MemoryTier.LONG_TERM)
    if event_id is not None:
        event.id = event_id
    return event


def test_empty_long_term_tier_returns_empty_profile():
    profile = build_semantic_profile([], FakeDecayPolicy({}))
    assert profile == {}


def test_ranks_by_decay_strength_descending():
    a = make_event("A", event_id="a")
    b = make_event("B", event_id="b")
    c = make_event("C", event_id="c")
    policy = FakeDecayPolicy({"a": 0.5, "b": 0.9, "c": 0.1})

    profile = build_semantic_profile([a, b, c], policy)

    assert profile["long_term_memories"] == ["B", "A", "C"]


def test_caps_at_max_events():
    events = [make_event(f"fact-{i}", event_id=str(i)) for i in range(5)]
    # strengths equal to index, so higher index = higher strength = kept
    policy = FakeDecayPolicy({str(i): float(i) for i in range(5)})

    profile = build_semantic_profile(events, policy, max_events=2)

    assert profile["long_term_memories"] == ["fact-4", "fact-3"]


def test_truncates_long_event_content():
    long_content = "x" * 1000
    event = make_event(long_content, event_id="only")
    policy = FakeDecayPolicy({"only": 1.0})

    profile = build_semantic_profile([event], policy, max_chars_per_event=300)

    fact = profile["long_term_memories"][0]
    assert fact.startswith("x" * 300)
    assert fact.endswith("...[truncated]")
    assert len(fact) == 300 + len("...[truncated]")


def test_short_event_content_is_not_truncated():
    event = make_event("short fact", event_id="only")
    policy = FakeDecayPolicy({"only": 1.0})

    profile = build_semantic_profile([event], policy, max_chars_per_event=300)

    assert profile["long_term_memories"] == ["short fact"]


def test_non_string_event_content_is_stringified():
    event = make_event({"fact": "user likes tea"}, event_id="only")
    policy = FakeDecayPolicy({"only": 1.0})

    profile = build_semantic_profile([event], policy)

    assert profile["long_term_memories"] == [str({"fact": "user likes tea"})]
