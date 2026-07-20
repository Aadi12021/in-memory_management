from memory_system.backends.memory import InMemoryBackend
from memory_system.events import MemoryEvent


def add(backend, content):
    event = MemoryEvent(content=content)
    backend.add(event)
    return event


def test_ranks_more_relevant_document_higher():
    backend = InMemoryBackend()
    add(backend, "The user is severely allergic to peanuts and tree nuts.")
    add(backend, "The user enjoys hiking on weekends.")
    add(backend, "The user mentioned peanuts once in passing.")

    # "peanuts"/"nuts" stem-match "peanut"/"nut" in doc 1 on multiple terms;
    # doc 3 only matches on "peanut". Doc 1 should rank first.
    results = backend.query("peanuts nuts", top_k=3)

    assert len(results) >= 2
    top_content = str(results[0].event.content)
    assert "allergic" in top_content


def test_rare_terms_weighted_higher_than_common_terms():
    backend = InMemoryBackend()
    # "user" appears in every doc (low idf); "quinoa" is rare (high idf)
    add(backend, "The user likes quinoa salad.")
    add(backend, "The user likes rice.")
    add(backend, "The user likes pasta.")

    results = backend.query("quinoa", top_k=3)

    assert len(results) == 1
    assert "quinoa" in str(results[0].event.content)


def test_no_matching_terms_returns_empty():
    backend = InMemoryBackend()
    add(backend, "The user is allergic to peanuts.")

    results = backend.query("weather forecast tomorrow", top_k=3)

    assert results == []


def test_empty_backend_returns_empty():
    backend = InMemoryBackend()
    results = backend.query("anything", top_k=3)
    assert results == []
