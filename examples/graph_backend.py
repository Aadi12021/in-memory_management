"""Shows GraphBackend answering a multi-hop query a flat vector or
keyword store can't: nothing in the *text* of any single stored fact
mentions "protein," but the relationship chain
    user -ALLERGIC_TO-> peanut -CONTAINS-> protein
connects the user's allergy to it in two hops. A similarity search
over these sentences has no way to surface "protein" unless the query
happens to be textually/semantically close to one of the stored
sentences -- GraphBackend finds it by traversing relationships instead.
"""

from memory_system.backends.graph import GraphBackend
from memory_system.events import MemoryEvent
from memory_system.extraction.rules_based import RuleBasedEntityExtractor

graph = GraphBackend(extractor=RuleBasedEntityExtractor())

graph.add(MemoryEvent(content="User is allergic to peanuts."))
graph.add(MemoryEvent(content="Peanuts contains protein."))
graph.add(MemoryEvent(content="User enjoys hiking."))  # unrelated, to show traversal doesn't just return everything

print("Direct allergies (1 hop, filtered to ALLERGIC_TO):")
print([e.id for e in graph.related_to("user", relation_type="ALLERGIC_TO", max_hops=1)])

print("\nEverything reachable within 2 hops (no relation_type filter):")
print([e.id for e in graph.related_to("user", max_hops=2)])

print("\nWhy is 'protein' connected to the user? explain_path shows the chain:")
path = graph.explain_path("user", "protein")
for rel in path:
    print(f"  {rel.source_id} -{rel.relation_type}-> {rel.target_id}")

print(
    "\nNote: no stored sentence mentions 'protein' and 'user' together -- "
    "a similarity search over this text has nothing to match on. GraphBackend "
    "found it by following ALLERGIC_TO then CONTAINS, not by comparing text."
)
