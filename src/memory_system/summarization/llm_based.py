"""LLM-based memory summarization. Requires the `llm` extra (same
anthropic dependency LLMEntityExtractor already needs -- no new
optional-dependency permutation):
    pip install tiered-memory[llm]
"""

from __future__ import annotations

from typing import Optional

from ..events import MemoryEvent
from .base import MemorySummarizer

try:
    import anthropic
except ImportError:  # pragma: no cover
    anthropic = None


SUMMARIZATION_PROMPT = """Summarize the following related memories into one dense, factual paragraph. Preserve concrete facts, names, and preferences. Do not add commentary or filler.

Memories:
{events}

Output only the summary paragraph, nothing else.
"""


class LLMSummarizer(MemorySummarizer):
    """Calls Claude to condense a group of related memories into one
    summary. Note this is a per-group API call -- offline_consolidate()
    may call this once per compression group in a single pass.
    """

    def __init__(self, api_key: Optional[str] = None, model: str = "claude-sonnet-5"):
        if anthropic is None:
            raise ImportError(
                "LLMSummarizer requires the anthropic package. "
                "Install with: pip install tiered-memory[llm]"
            )
        self._client = anthropic.Anthropic(api_key=api_key)
        self.model = model

    def summarize(self, events: list[MemoryEvent]) -> str:
        formatted = "\n".join(f"- {event.content}" for event in events)
        prompt = SUMMARIZATION_PROMPT.format(events=formatted)

        response = self._client.messages.create(
            model=self.model,
            max_tokens=500,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(
            block.text for block in response.content if block.type == "text"
        ).strip()
