"""Shows how to bring your own consolidation policy.

This is the main extension point for domain-specific memory rules --
e.g. always consolidate anything tagged as a user preference, regardless
of salience score.
"""

from memory_system import (
    ForgettingCurveDecay,
    InMemoryBackend,
    MemoryEvent,
    TieredMemory,
)
from memory_system.policies.consolidation import ConsolidationPolicy


class TagBasedConsolidation(ConsolidationPolicy):
    """Consolidates anything whose metadata has consolidate=True,
    ignoring salience entirely.
    """

    def should_consolidate(self, event: MemoryEvent) -> bool:
        return bool(event.metadata.get("consolidate"))


memory = TieredMemory(
    backend=InMemoryBackend(),
    consolidation_policy=TagBasedConsolidation(),
    decay_policy=ForgettingCurveDecay(half_life_days=14),
)

memory.store("User is allergic to peanuts.", metadata={"consolidate": True})
memory.store("It's raining today.", metadata={"consolidate": False})

promoted = memory.consolidate()
print(f"Promoted {promoted} event(s) to long-term memory.")
