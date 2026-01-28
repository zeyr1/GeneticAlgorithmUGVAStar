"""
UGV Navigation System - Modular Architecture
=============================================

Surrogate-Assisted Receding-Horizon Planning Under Field-of-View Constraints.

This package implements energy-time-distance trade-off optimization for 
unmanned ground vehicle navigation with limited sensing range.

Key Features:
- SOLID-compliant modular architecture
- Adaptive FoV for dynamic sensing range adjustment
- Multi-strategy recovery system for dead-end handling
- Hierarchical planning (global sketch + local refinement)
- Local surrogate ensemble for improved prediction
- Real-time visualization and debugging

Author: UGV Navigation Team
Version: 2.0.0
"""

__version__ = "2.0.0"
__author__ = "UGV Navigation Team"

from .config import Config, GASettings
from .terrain import TerrainType
from .environment import Environment, LocalEnvironment
from .energy import EnergyModel
from .planning import AStarPlanner, RecedingHorizonController
from .optimization import LocalGASolver, LocalSurrogateEnsemble
from .recovery import RecoveryManager, AdaptiveFoV
from .metrics import PathMetrics, RunClassifier
from .visualization import LiveMonitor, TerrainVisualizer
from .pipeline import ExperimentRunner

__all__ = [
    'Config', 'GASettings',
    'TerrainType',
    'Environment', 'LocalEnvironment', 
    'EnergyModel',
    'AStarPlanner', 'RecedingHorizonController',
    'LocalGASolver', 'LocalSurrogateEnsemble',
    'RecoveryManager', 'AdaptiveFoV',
    'PathMetrics', 'RunClassifier',
    'LiveMonitor', 'TerrainVisualizer',
    'ExperimentRunner',
]
