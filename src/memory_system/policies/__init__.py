from .consolidation import (
    AlwaysConsolidate,
    ConsolidationPolicy,
    RepetitionBasedConsolidation,
    SurpriseBasedConsolidation,
)
from .decay import DecayPolicy, ForgettingCurveDecay, NoDecay
from .percept_bridge import build_semantic_profile
from .salience import ConstantSalience, LengthHeuristicSalience, SalienceScorer

__all__ = [
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
    "build_semantic_profile",
]

try:
    from .percept_salience import PerceptSalienceScorer  # noqa: F401
    __all__.append("PerceptSalienceScorer")
except ImportError:
    pass
