import json
from unittest.mock import MagicMock, patch

from memory_system.events import MemoryEvent
from memory_system.extraction.llm_based import LLMEntityExtractor


def make_mock_response(json_payload: dict, wrap_in_markdown: bool = False):
    """Builds a fake anthropic Message response with the given JSON as
    the text content, optionally wrapped in ```json fences the way the
    model sometimes does despite being told not to.
    """
    text = json.dumps(json_payload)
    if wrap_in_markdown:
        text = f"```json\n{text}\n```"
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    return response


def make_extractor():
    with patch("memory_system.extraction.llm_based.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        extractor = LLMEntityExtractor(api_key="fake-key-for-testing")
        return extractor, mock_client


def test_parses_well_formed_json_response():
    extractor, mock_client = make_extractor()
    mock_client.messages.create.return_value = make_mock_response({
        "entities": [
            {"id": "user", "label": "User", "entity_type": "person"},
            {"id": "peanuts", "label": "Peanuts", "entity_type": "food"},
        ],
        "relationships": [
            {"source_id": "user", "target_id": "peanuts", "relation_type": "ALLERGIC_TO", "confidence": 0.95}
        ],
    })

    entities, rels = extractor.extract(MemoryEvent(content="User is allergic to peanuts."))

    assert len(entities) == 2
    assert len(rels) == 1
    assert rels[0].relation_type == "ALLERGIC_TO"
    assert rels[0].confidence == 0.95


def test_strips_markdown_code_fences():
    """The prompt asks for raw JSON, but models sometimes wrap it in
    ```json fences anyway -- this should be handled, not crash.
    """
    extractor, mock_client = make_extractor()
    mock_client.messages.create.return_value = make_mock_response(
        {"entities": [], "relationships": []}, wrap_in_markdown=True
    )

    entities, rels = extractor.extract(MemoryEvent(content="Something."))

    assert entities == []
    assert rels == []


def test_malformed_json_fails_soft_not_hard():
    """If the model returns unparseable output, extraction should
    return empty results, not raise and crash the whole ingestion pipeline.
    """
    extractor, mock_client = make_extractor()
    block = MagicMock()
    block.type = "text"
    block.text = "this is not valid json at all {{{"
    response = MagicMock()
    response.content = [block]
    mock_client.messages.create.return_value = response

    entities, rels = extractor.extract(MemoryEvent(content="Something."))

    assert entities == []
    assert rels == []


def test_low_confidence_relationships_are_filtered_out():
    extractor, mock_client = make_extractor()
    extractor.min_confidence = 0.5
    mock_client.messages.create.return_value = make_mock_response({
        "entities": [{"id": "user", "label": "User", "entity_type": "person"}],
        "relationships": [
            {"source_id": "user", "target_id": "x", "relation_type": "MAYBE_RELATED", "confidence": 0.2},
            {"source_id": "user", "target_id": "y", "relation_type": "CLEARLY_STATED", "confidence": 0.9},
        ],
    })

    entities, rels = extractor.extract(MemoryEvent(content="Something."))

    assert len(rels) == 1
    assert rels[0].relation_type == "CLEARLY_STATED"


def test_missing_confidence_defaults_to_middle_value():
    extractor, mock_client = make_extractor()
    mock_client.messages.create.return_value = make_mock_response({
        "entities": [{"id": "user", "label": "User", "entity_type": "person"}],
        "relationships": [
            {"source_id": "user", "target_id": "x", "relation_type": "SOME_RELATION"}
        ],
    })

    entities, rels = extractor.extract(MemoryEvent(content="Something."))

    assert rels[0].confidence == 0.5
