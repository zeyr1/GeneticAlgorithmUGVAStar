"""
Surrogate Module
================

Surrogate models for fast path evaluation.
"""

from .model import (
    SurrogateModel,
    LocalSurrogateEnsemble,
    SurrogateFeatureExtractor,
    SurrogateSample,
)

__all__ = [
    'SurrogateModel',
    'LocalSurrogateEnsemble',
    'SurrogateFeatureExtractor',
    'SurrogateSample',
]
