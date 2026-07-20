# Contributing

Thanks for considering a contribution!

## Setup

```bash
git clone https://github.com/YOUR_USERNAME/tiered-memory.git
cd tiered-memory
pip install -e ".[dev]"
```

## Running tests

```bash
pytest tests/ -v
```

## Adding a new policy or backend

This project is built around swappable pieces:
- New `ConsolidationPolicy` → add to `src/memory_system/policies/consolidation.py`
- New `DecayPolicy` → add to `src/memory_system/policies/decay.py`
- New `SalienceScorer` → add to `src/memory_system/policies/salience.py`
- New `MemoryBackend` → add a file under `src/memory_system/backends/`

Please include tests for any new policy or backend, and a short example
in `examples/` if it introduces a new usage pattern.

## Pull requests

- Keep PRs focused on one change
- Make sure `pytest tests/` passes before opening
- Describe the "why," not just the "what," in your PR description
