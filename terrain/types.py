"""
Terrain Types Module
====================

Defines terrain type enumeration and utilities.
"""

from enum import IntEnum
from typing import Dict, Any


class TerrainType(IntEnum):
    """
    Terrain type enumeration.
    
    Values are integers for efficient numpy array storage.
    """
    ASPHALT = 0
    GRASS = 1
    MUD = 2
    SAND = 3
    WALL = 4
    
    @classmethod
    def from_name(cls, name: str) -> 'TerrainType':
        """Get terrain type from string name"""
        return cls[name.upper()]
    
    @property
    def name_lower(self) -> str:
        """Get lowercase name"""
        return self.name.lower()
    
    def is_traversable(self) -> bool:
        """Check if terrain is traversable"""
        return self != TerrainType.WALL
    
    @classmethod
    def traversable_types(cls) -> tuple:
        """Get all traversable terrain types"""
        return (cls.ASPHALT, cls.GRASS, cls.MUD, cls.SAND)
    
    @classmethod
    def get_display_colors(cls) -> Dict['TerrainType', str]:
        """Get visualization colors for each terrain type"""
        return {
            cls.ASPHALT: 'darkgray',
            cls.GRASS: 'lightgreen',
            cls.MUD: 'saddlebrown',
            cls.SAND: 'gold',
            cls.WALL: 'black'
        }


class TerrainProperties:
    """
    Static terrain properties lookup.
    
    Provides fast access to terrain-specific parameters.
    """
    
    # Friction coefficients (rolling resistance)
    FRICTION = {
        TerrainType.ASPHALT: 0.015,
        TerrainType.GRASS: 0.08,
        TerrainType.MUD: 0.25,
        TerrainType.SAND: 0.15,
        TerrainType.WALL: float('inf')
    }
    
    # Maximum safe velocities (m/s)
    MAX_VELOCITY = {
        TerrainType.ASPHALT: 15.0,
        TerrainType.GRASS: 8.0,
        TerrainType.MUD: 3.0,
        TerrainType.SAND: 5.0,
        TerrainType.WALL: 0.0
    }
    
    # Base risk scores
    BASE_RISK = {
        TerrainType.ASPHALT: 0.05,
        TerrainType.GRASS: 0.15,
        TerrainType.MUD: 0.45,
        TerrainType.SAND: 0.25,
        TerrainType.WALL: 1.0
    }
    
    # Terramechanics sinkage multiplier
    SINKAGE_MULTIPLIER = {
        TerrainType.ASPHALT: 0.0,
        TerrainType.GRASS: 0.1,
        TerrainType.MUD: 1.0,
        TerrainType.SAND: 0.4,
        TerrainType.WALL: 0.0
    }
    
    @classmethod
    def get_friction(cls, terrain_type: TerrainType) -> float:
        """Get friction coefficient for terrain type"""
        return cls.FRICTION.get(terrain_type, 0.15)
    
    @classmethod
    def get_max_velocity(cls, terrain_type: TerrainType) -> float:
        """Get maximum velocity for terrain type"""
        return cls.MAX_VELOCITY.get(terrain_type, 5.0)
    
    @classmethod
    def get_base_risk(cls, terrain_type: TerrainType) -> float:
        """Get base risk score for terrain type"""
        return cls.BASE_RISK.get(terrain_type, 0.5)
    
    @classmethod
    def get_sinkage_multiplier(cls, terrain_type: TerrainType) -> float:
        """Get terramechanics sinkage multiplier"""
        return cls.SINKAGE_MULTIPLIER.get(terrain_type, 0.0)
