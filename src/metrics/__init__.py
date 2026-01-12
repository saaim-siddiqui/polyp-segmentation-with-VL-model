"""
Metrics package for Vision-Language Segmentation.

Contains:
- Uncertainty-Text Correlation metrics (URS, SAS, Conditioned-ECE)
- Segmentation metrics (Dice, IoU)
"""

from .uncertainty_metrics import (
    SpatialUncertaintyAggregator,
    AttributeUncertaintyCorrelation,
    UncertaintyReductionScore,
    SemanticAlignmentScore,
    AttributeConditionedECE,
    UncertaintyMetricsCalculator,
)

__all__ = [
    "SpatialUncertaintyAggregator",
    "AttributeUncertaintyCorrelation",
    "UncertaintyReductionScore",
    "SemanticAlignmentScore",
    "AttributeConditionedECE",
    "UncertaintyMetricsCalculator",
]