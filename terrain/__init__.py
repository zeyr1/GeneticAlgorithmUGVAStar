"""
Terrain Module
==============

Terrain types, properties, and procedural map generation.
"""

from .types import TerrainType, TerrainProperties
from .generator import MapGenerator, GeneratedMap, generate_start_goal

__all__ = [
    'TerrainType',
    'TerrainProperties',
    'MapGenerator',
    'GeneratedMap',
    'generate_start_goal',
]
