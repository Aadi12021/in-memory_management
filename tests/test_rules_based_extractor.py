from memory_system.events import MemoryEvent
from memory_system.extraction.rules_based import RuleBasedEntityExtractor


def extract(text):
    extractor = RuleBasedEntityExtractor()
    event = MemoryEvent(content=text)
    return extractor.extract(event)


def test_extracts_allergic_to_relationship():
    entities, rels = extract("User is allergic to peanuts.")
    assert any(r.relation_type == "ALLERGIC_TO" and r.target_id == "peanut" for r in rels)


def test_dislikes_does_not_false_match_enjoys_pattern():
    """Regression test: 'dislikes' contains the substring 'likes', which
    without word boundaries incorrectly matched the ENJOYS pattern too.
    """
    entities, rels = extract("User dislikes cold weather.")
    relation_types = [r.relation_type for r in rels]
    assert "DISLIKES" in relation_types
    assert "ENJOYS" not in relation_types


def test_object_capture_stops_at_trailing_preposition():
    """Regression test: 'enjoys hiking on weekends' should extract
    'hiking', not 'hiking on'.
    """
    entities, rels = extract("The user enjoys hiking on weekends.")
    enjoy_rels = [r for r in rels if r.relation_type == "ENJOYS"]
    assert len(enjoy_rels) == 1
    assert enjoy_rels[0].target_id == "hiking"


def test_known_limitation_three_word_compound_subject_gets_truncated():
    """KNOWN LIMITATION, not a target behavior: the CONTAINS pattern
    captures at most 2 words for its subject, so 'Peanut butter cake'
    loses the leading word and becomes 'butter cake'. This is the kind
    of gap that motivates using an LLM-based extractor for anything
    beyond short, simple phrasing. If this test starts failing because
    someone fixed the truncation, that's a welcome regression -- update
    the assertion rather than treating it as broken.
    """
    entities, rels = extract("Peanut butter cake contains peanuts.")
    contains_rels = [r for r in rels if r.relation_type == "CONTAINS"]
    assert len(contains_rels) == 1
    assert contains_rels[0].source_id == "butter cake"  # NOT "peanut butter cake"


def test_confidence_scores_reflect_pattern_reliability():
    _, rels = extract("User is allergic to peanuts.")
    assert rels[0].confidence == 0.9

    _, rels = extract("Peanut butter cake contains peanuts.")
    assert rels[0].confidence == 0.6  # CONTAINS is the least reliable pattern


def test_no_match_returns_only_default_subject_entity():
    entities, rels = extract("It's raining outside today.")
    assert rels == []
    assert len(entities) == 1
    assert entities[0].id == "user"
