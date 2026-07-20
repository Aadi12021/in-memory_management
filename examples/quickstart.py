"""Minimal end-to-end example: store, consolidate, retrieve."""

from memory_system import (
    ConstantSalience,
    ForgettingCurveDecay,
    InMemoryBackend,
    SurpriseBasedConsolidation,
    TieredMemory,
)

memory = TieredMemory(
    backend=InMemoryBackend(),
    consolidation_policy=SurpriseBasedConsolidation(threshold=0.7),
    decay_policy=ForgettingCurveDecay(half_life_days=14),
    salience_scorer=ConstantSalience(value=0.9),
)

memory.store("User is allergic to peanuts.")
memory.store("User enjoys hiking on weekends.")
memory.store("User's favorite color is teal.")

promoted = memory.consolidate()
print(f"Promoted {promoted} events to long-term memory.")

results = memory.retrieve("peanut allergy", top_k=2)
print("\nRetrieval results for 'peanut allergy':")
for r in results:
    print(f"  [{r.tier.value}] score={r.score:.2f} -> {r.event.content}")
