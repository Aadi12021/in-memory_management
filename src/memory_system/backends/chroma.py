"""ChromaDB-backed implementation. Requires the `chroma` extra:
    pip install tiered-memory[chroma]
"""

from __future__ import annotations

from typing import Optional

from ..events import MemoryEvent, MemoryTier, RetrievalResult
from .base import MemoryBackend

try:
    import chromadb
except ImportError:  # pragma: no cover
    chromadb = None


class ChromaBackend(MemoryBackend):
    def __init__(self, collection_name: str = "tiered_memory", persist_path: Optional[str] = None):
        if chromadb is None:
            raise ImportError(
                "ChromaBackend requires chromadb. Install with: pip install tiered-memory[chroma]"
            )
        client = (
            chromadb.PersistentClient(path=persist_path)
            if persist_path
            else chromadb.Client()
        )
        self._collection = client.get_or_create_collection(collection_name)
        # id -> full MemoryEvent, since Chroma only stores text/metadata/embeddings
        self._events: dict[str, MemoryEvent] = {}

    def add(self, event: MemoryEvent) -> None:
        self._events[event.id] = event
        self._collection.upsert(
            ids=[event.id],
            documents=[str(event.content)],
            metadatas=[{"tier": event.tier.value, "salience": event.salience}],
        )

    def get_all(self, tier: Optional[MemoryTier] = None) -> list[MemoryEvent]:
        events = list(self._events.values())
        if tier is not None:
            events = [e for e in events if e.tier == tier]
        return events

    def query(
        self, query: str, top_k: int = 5, tier: Optional[MemoryTier] = None
    ) -> list[RetrievalResult]:
        where = {"tier": tier.value} if tier is not None else None
        results = self._collection.query(query_texts=[query], n_results=top_k, where=where)
        out: list[RetrievalResult] = []
        for event_id, distance in zip(results["ids"][0], results["distances"][0]):
            event = self._events.get(event_id)
            if event is None:
                continue
            score = 1.0 / (1.0 + distance)  # convert distance to a similarity-ish score
            out.append(RetrievalResult(event=event, score=score, tier=event.tier))
        return out

    def update_tier(self, event_id: str, new_tier: MemoryTier) -> None:
        if event_id in self._events:
            self._events[event_id].tier = new_tier
            self._collection.update(ids=[event_id], metadatas=[{"tier": new_tier.value}])

    def remove(self, event_id: str) -> None:
        self._events.pop(event_id, None)
        self._collection.delete(ids=[event_id])
