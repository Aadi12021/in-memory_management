"""LLM-based entity/relationship extraction.

Trades the rule-based extractor's zero-dependency simplicity for much
higher recall and robustness to phrasing: "can't eat X", "X makes them
break out in hives", and "Peanut butter cake" (full compound noun,
unlike the rule-based extractor's 2-word cap) all work here without
writing a new regex pattern for each.

Requires the `anthropic` package and an API key -- this is the
"upgrade path" extractor, same relationship RuleBasedEntityExtractor
has to InMemoryBackend vs ChromaBackend: free/local default, paid/
smarter option for when you need it.
"""

from __future__ import annotations

import json
from typing import Optional

from ..events import MemoryEvent
from ..backends.graph import Entity, EntityExtractor, Relationship

try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None


EXTRACTION_PROMPT = """Extract entities and relationships from the text below.

Return ONLY valid JSON, no other text, in this exact shape:
{{
  "entities": [{{"id": "snake_case_id", "label": "Display Name", "entity_type": "person|food|place|activity|other"}}],
  "relationships": [{{"source_id": "...", "target_id": "...", "relation_type": "SCREAMING_SNAKE_CASE", "confidence": 0.0-1.0}}]
}}

Rules:
- entity ids must be snake_case and reused consistently (same entity = same id)
- if the text is about "the user"/"I"/"my", use entity id "user"
- relation_type should be a general reusable category (e.g. ALLERGIC_TO, ENJOYS, WORKS_AT), not a one-off description
- confidence reflects how explicitly the text states the relationship, not how important it is
- if no clear entities/relationships exist, return empty lists

Text: "{text}"
"""


class LLMEntityExtractor(EntityExtractor):
    """Calls Claude to extract entities/relationships as structured JSON.

    Note this is a per-memory API call -- for high-volume ingestion,
    consider batching events and extracting several at once, or falling
    back to RuleBasedEntityExtractor for low-stakes/high-volume content
    and reserving this for content worth the cost.
    """

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "claude-sonnet-5",
        min_confidence: float = 0.3,
    ):
        if anthropic is None:
            raise ImportError(
                "LLMEntityExtractor requires the anthropic package. "
                "Install with: pip install tiered-memory[llm]"
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model
        self.min_confidence = min_confidence

    def extract(self, event: MemoryEvent) -> tuple[list[Entity], list[Relationship]]:
        prompt = EXTRACTION_PROMPT.format(text=str(event.content))

        response = self._client.messages.create(
            model=self.model,
            max_tokens=1000,
            messages=[{"role": "user", "content": prompt}],
        )

        raw_text = "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
        raw_text = raw_text.removeprefix("```json").removeprefix("```").removesuffix("```").strip()

        try:
            parsed = json.loads(raw_text)
        except json.JSONDecodeError:
            # fail soft: a malformed extraction shouldn't crash ingestion,
            # it should just mean this event contributes nothing to the graph
            return [], []

        entities = [
            Entity(id=e["id"], label=e["label"], entity_type=e.get("entity_type", "unknown"))
            for e in parsed.get("entities", [])
        ]

        relationships = [
            Relationship(
                source_id=r["source_id"],
                target_id=r["target_id"],
                relation_type=r["relation_type"],
                source_event_id=event.id,
                confidence=r.get("confidence", 0.5),
            )
            for r in parsed.get("relationships", [])
            if r.get("confidence", 0.5) >= self.min_confidence
        ]

        return entities, relationships
