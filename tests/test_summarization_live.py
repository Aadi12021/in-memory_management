"""Live integration test for LLMSummarizer against the real Anthropic
API -- as opposed to test_summarization.py, which mocks the client
entirely. Skipped unless ANTHROPIC_API_KEY is set, since it makes a
real, billed API call and needs real credentials.
"""

import os

import pytest

from memory_system.events import MemoryEvent
from memory_system.summarization.llm_based import LLMSummarizer


@pytest.mark.skipif(
    not os.environ.get("ANTHROPIC_API_KEY"),
    reason="requires a real ANTHROPIC_API_KEY to call the live Anthropic API",
)
def test_summarizes_related_memories_from_real_api():
    summarizer = LLMSummarizer()
    events = [
        MemoryEvent(content="User is allergic to peanuts."),
        MemoryEvent(content="User is also allergic to tree nuts."),
    ]

    result = summarizer.summarize(events)

    assert isinstance(result, str)
    assert len(result) > 0
    assert "peanut" in result.lower() or "nut" in result.lower() or "allerg" in result.lower()
