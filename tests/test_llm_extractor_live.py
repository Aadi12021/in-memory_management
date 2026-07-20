"""Live integration test for LLMEntityExtractor against the real
Anthropic API -- as opposed to test_llm_extractor.py, which mocks the
client entirely. Skipped unless ANTHROPIC_API_KEY is set, since it
makes a real, billed API call and needs real credentials.

Asserts on the *structure* of what comes back (entity ids present,
relation type, provenance, confidence range), not exact wording --
live model output for open-ended fields like entity labels isn't
perfectly deterministic between calls.
"""

import os

import pytest

from memory_system.events import MemoryEvent
from memory_system.extraction.llm_based import LLMEntityExtractor


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="requires a real ANTHROPIC_API_KEY to call the live Anthropic API",
)
def test_extracts_allergy_relationship_from_real_api():
    extractor = LLMEntityExtractor()
    event = MemoryEvent(content="The user is severely allergic to peanuts.")

    entities, relationships = extractor.extract(event)

    entity_ids = {e.id for e in entities}
    assert "user" in entity_ids
    assert len(entities) >= 2  # at least "user" and something for peanuts

    allergy_rels = [r for r in relationships if r.relation_type == "ALLERGIC_TO"]
    assert len(allergy_rels) == 1

    rel = allergy_rels[0]
    assert rel.source_id == "user"
    assert rel.target_id in entity_ids
    assert rel.source_event_id == event.id
    assert 0.0 <= rel.confidence <= 1.0
