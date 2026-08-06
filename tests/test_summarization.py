from __future__ import annotations

from unittest.mock import MagicMock, patch

from memory_system.events import MemoryEvent
from memory_system.summarization.llm_based import LLMSummarizer


def make_mock_response(text: str):
    block = MagicMock()
    block.type = "text"
    block.text = text
    response = MagicMock()
    response.content = [block]
    return response


def make_summarizer():
    with patch("memory_system.summarization.llm_based.anthropic") as mock_anthropic:
        mock_client = MagicMock()
        mock_anthropic.Anthropic.return_value = mock_client
        summarizer = LLMSummarizer(api_key="fake-key-for-testing")
        return summarizer, mock_client


def test_returns_the_model_response_text():
    summarizer, mock_client = make_summarizer()
    mock_client.messages.create.return_value = make_mock_response(
        "User has severe peanut and tree nut allergies."
    )

    result = summarizer.summarize([
        MemoryEvent(content="User is allergic to peanuts."),
        MemoryEvent(content="User is also allergic to tree nuts."),
    ])

    assert result == "User has severe peanut and tree nut allergies."


def test_strips_surrounding_whitespace():
    summarizer, mock_client = make_summarizer()
    mock_client.messages.create.return_value = make_mock_response(
        "\n  User has peanut allergies.  \n"
    )

    result = summarizer.summarize([MemoryEvent(content="User is allergic to peanuts.")])

    assert result == "User has peanut allergies."
