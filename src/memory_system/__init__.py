from .backends import InMemoryBackend, MemoryBackend
from .core import TieredMemory
from .events import MemoryEvent, MemoryTier, RetrievalResult
from .policies import (
    AlwaysConsolidate,
    ConsolidationPolicy,
    ConstantSalience,
    DecayPolicy,
    ForgettingCurveDecay,
    LengthHeuristicSalience,
    NoDecay,
    RepetitionBasedConsolidation,
    SalienceScorer,
    SurpriseBasedConsolidation,
)

__version__ = "0.2.0"

__all__ = [
    "TieredMemory",
    "MemoryEvent",
    "MemoryTier",
    "RetrievalResult",
    "MemoryBackend",
    "InMemoryBackend",
    "ConsolidationPolicy",
    "AlwaysConsolidate",
    "SurpriseBasedConsolidation",
    "RepetitionBasedConsolidation",
    "DecayPolicy",
    "ForgettingCurveDecay",
    "NoDecay",
    "SalienceScorer",
    "ConstantSalience",
    "LengthHeuristicSalience",
]
