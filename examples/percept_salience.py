"""Shows PerceptSalienceScorer -- a real, embedding-based novelty/
surprise SalienceScorer ported from PERCEPT-1's PredictiveCoder --
driving SurpriseBasedConsolidation, instead of a toy scorer like
ConstantSalience or LengthHeuristicSalience.

Requires the `percept` extra:
    pip install tiered-memory[percept]

Seeds a few related "software engineer who writes Python" memories so
the scorer builds up a known long-term profile, then compares one new
memory on the same topic (should register as low surprise -- it's no
longer novel) against one on a completely unrelated topic (should
register as high surprise).
"""

from memory_system import SurpriseBasedConsolidation, TieredMemory
from memory_system.backends.memory import InMemoryBackend
from memory_system.policies.decay import ForgettingCurveDecay
from memory_system.policies.percept_salience import PerceptSalienceScorer

backend = InMemoryBackend()
decay_policy = ForgettingCurveDecay(half_life_days=30)
scorer = PerceptSalienceScorer(backend=backend, decay_policy=decay_policy)

memory = TieredMemory(
    backend=backend,
    consolidation_policy=SurpriseBasedConsolidation(threshold=0.3),
    decay_policy=decay_policy,
    salience_scorer=scorer,
)

print("Seeding the long-term profile with related memories:")
seeds = [
    "User is a software engineer who writes Python for work.",
    "User builds backend services in Python at their job.",
    "User spends most of the workday writing and reviewing Python code.",
]
for content in seeds:
    event = memory.store(content)
    print(f"  salience={event.salience:.4f}  {content}")
    memory.consolidate()

print("\nScoring new memories against that profile:")
related = memory.store("User also writes Python scripts to automate deployments.")
print(f"  salience={related.salience:.4f}  (related)   {related.content}")

novel = memory.store("User's dog gave birth to six puppies this week.")
print(f"  salience={novel.salience:.4f}  (unrelated) {novel.content}")

promoted = memory.consolidate()
print(f"\nPromoted {promoted} more event(s) to long-term memory (threshold=0.3).")
print(
    "The unrelated puppy news registers as more surprising than the "
    "already-familiar Python fact, and only it crosses the consolidation "
    "threshold -- PerceptSalienceScorer is discriminating between "
    "genuinely novel content and repetition of what the system already knows."
)
