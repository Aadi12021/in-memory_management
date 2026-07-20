from datetime import datetime, timedelta, timezone

from memory_system import (
    ConstantSalience,
    ForgettingCurveDecay,
    InMemoryBackend,
    MemoryTier,
    SurpriseBasedConsolidation,
    TieredMemory,
)


def make_memory(threshold=0.7, half_life_days=14.0, salience=0.9):
    return TieredMemory(
        backend=InMemoryBackend(),
        consolidation_policy=SurpriseBasedConsolidation(threshold=threshold),
        decay_policy=ForgettingCurveDecay(half_life_days=half_life_days),
        salience_scorer=ConstantSalience(value=salience),
    )


def test_store_puts_event_in_working_tier():
    memory = make_memory()
    event = memory.store("User is allergic to peanuts.")
    assert event.tier == MemoryTier.WORKING
    assert event.salience == 0.9


def test_consolidate_promotes_high_salience_events():
    memory = make_memory(threshold=0.7, salience=0.9)
    event = memory.store("Important fact.")
    promoted = memory.consolidate()
    assert promoted == 1
    stored = memory.backend.get_all(tier=MemoryTier.LONG_TERM)
    assert len(stored) == 1
    assert stored[0].id == event.id


def test_consolidate_skips_low_salience_events():
    memory = make_memory(threshold=0.7, salience=0.3)
    memory.store("Trivial fact.")
    promoted = memory.consolidate()
    assert promoted == 0
    assert len(memory.backend.get_all(tier=MemoryTier.LONG_TERM)) == 0


def test_retrieve_finds_relevant_events():
    memory = make_memory()
    memory.store("User is allergic to peanuts.")
    memory.store("User enjoys hiking on weekends.")
    results = memory.retrieve("peanuts allergy", top_k=1)
    assert len(results) == 1
    assert "peanuts" in str(results[0].event.content).lower()


def test_decay_removes_stale_events():
    memory = make_memory(half_life_days=1.0)
    event = memory.store("Ephemeral working-memory note.")
    far_future = datetime.now(timezone.utc) + timedelta(days=30)
    forgotten = memory.decay(now=far_future)
    assert forgotten == 1
    assert memory.backend.get_all() == []


def test_long_term_decays_slower_than_working():
    memory = make_memory(half_life_days=1.0)
    working_event = memory.store("Working memory note.")
    long_term_event = memory.store("Important note.")
    memory.backend.update_tier(long_term_event.id, MemoryTier.LONG_TERM)

    check_time = datetime.now(timezone.utc) + timedelta(days=1)
    working_strength = memory.decay_policy.current_strength(working_event, check_time)
    long_term_strength = memory.decay_policy.current_strength(long_term_event, check_time)

    assert long_term_strength > working_strength
