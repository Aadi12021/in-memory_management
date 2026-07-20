"""Rules-based entity/relationship extraction. Zero dependencies,
deterministic, fast. Ceiling is real: it only catches phrasings it has
patterns for. See LLMEntityExtractor for the higher-recall alternative.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from ..events import MemoryEvent
from ..backends.graph import Entity, EntityExtractor, Relationship


def _normalize(text: str) -> str:
    """Canonical entity id: lowercased, singular-ish, no punctuation."""
    text = text.strip().lower()
    text = re.sub(r"[^\w\s]", "", text)
    if text.endswith("s") and len(text) > 3:
        text = text[:-1]
    return text.strip()


# Common trailing words that a greedy capture picks up but that aren't
# part of the entity itself ("hiking on weekends" -> "hiking"). This is
# a truncation heuristic, not real parsing -- it helps the common case
# and does nothing for more complex phrasing.
_TRAILING_STOPWORDS = {
    "on", "at", "in", "for", "with", "during", "while", "and", "the", "a", "an"
}


def _truncate_at_stopword(phrase: str) -> str:
    words = phrase.split()
    for i, word in enumerate(words):
        if word in _TRAILING_STOPWORDS:
            return " ".join(words[:i]) if i > 0 else phrase
    return phrase


@dataclass
class RelationPattern:
    """One extraction rule: a regex with two capture groups (subject,
    object) plus the relation type it implies. Patterns are matched
    against lowercased text.
    """
    pattern: str
    relation_type: str
    confidence: float = 0.8   # rules are never fully certain about intent


# The actual pattern library. This is the part that grows over time as
# real usage surfaces phrasings it misses -- that's expected and fine,
# it's a lookup table, not a black box.
DEFAULT_PATTERNS: list[RelationPattern] = [
    RelationPattern(r"\ballergic to (\w+(?:\s\w+)?)", "ALLERGIC_TO", confidence=0.9),
    RelationPattern(r"\ballergy to (\w+(?:\s\w+)?)", "ALLERGIC_TO", confidence=0.9),
    RelationPattern(r"\bdislikes? (\w+(?:\s\w+)?)", "DISLIKES", confidence=0.7),
    RelationPattern(r"\bhates? (\w+(?:\s\w+)?)", "DISLIKES", confidence=0.7),
    RelationPattern(r"\benjoys? (\w+(?:\s\w+)?)", "ENJOYS", confidence=0.7),
    RelationPattern(r"(?<!dis)\blikes? (\w+(?:\s\w+)?)", "ENJOYS", confidence=0.7),
    RelationPattern(r"\bworks? at (\w+(?:\s\w+)?)", "WORKS_AT", confidence=0.85),
    RelationPattern(r"\blives? in (\w+(?:\s\w+)?)", "LIVES_IN", confidence=0.85),
    RelationPattern(r"(\w+(?:\s\w+)?) \bcontains? (\w+(?:\s\w+)?)", "CONTAINS", confidence=0.6),
]


class RuleBasedEntityExtractor(EntityExtractor):
    """Regex-pattern extraction. Assumes a fixed subject (default:
    "user") unless a pattern explicitly captures its own subject
    (like the CONTAINS pattern, which has two capture groups).

    This is deliberately simple. It will miss anything not covered by
    a pattern -- "allergic to" works, "can't eat X" or "X makes them
    break out in hives" won't, unless you add patterns for them. The
    honest ceiling here is: good for a known, narrow set of phrasings
    (like structured user-preference statements), not for open-ended
    natural language.
    """

    def __init__(self, patterns: list[RelationPattern] | None = None, default_subject: str = "user"):
        self.patterns = patterns or DEFAULT_PATTERNS
        self.default_subject = default_subject

    def extract(self, event: MemoryEvent) -> tuple[list[Entity], list[Relationship]]:
        text = str(event.content).lower()
        entities: dict[str, Entity] = {}
        relationships: list[Relationship] = []

        subject_id = _normalize(self.default_subject)
        entities[subject_id] = Entity(id=subject_id, label=self.default_subject.title(), entity_type="person")

        for rule in self.patterns:
            for match in re.finditer(rule.pattern, text):
                groups = match.groups()
                if len(groups) == 2:
                    # pattern captures its own subject and object (e.g. CONTAINS)
                    subj_raw, obj_raw = groups
                    subj_raw = _truncate_at_stopword(subj_raw.strip())
                    subj_id = _normalize(subj_raw)
                    entities.setdefault(subj_id, Entity(id=subj_id, label=subj_raw.strip().title()))
                else:
                    # pattern captures only an object; subject is the default (e.g. "user")
                    obj_raw = groups[0]
                    subj_id = subject_id

                obj_raw = _truncate_at_stopword(obj_raw.strip())
                obj_id = _normalize(obj_raw)
                entities.setdefault(obj_id, Entity(id=obj_id, label=obj_raw.strip().title()))

                relationships.append(
                    Relationship(
                        source_id=subj_id,
                        target_id=obj_id,
                        relation_type=rule.relation_type,
                        source_event_id=event.id,
                        confidence=rule.confidence,
                    )
                )

        return list(entities.values()), relationships
