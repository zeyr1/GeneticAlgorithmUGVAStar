"""
Environment World Module
========================

Main environment class representing the complete world state.
"""

import numpy as np
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

from ..config import Config
from ..terrain import TerrainType, TerrainProperties, MapGenerator, GeneratedMap, generate_start_goal


class Environment:
    """
    Complete world environment for UGV navigation.
    
    Contains:
    - Terrain type map
    - Elevation map
    - Risk map
    - Uncertainty map
    - Start and goal positions
    
    Provides:
    - Cell validity checking
    - Terrain property queries
    - Map statistics
    """
    
    def __init__(self, config: Config, seed: Optional[int] = None,
                 generated_map: Optional[GeneratedMap] = None):
        """
        Initialize environment.
        
        Args:
            config: Configuration object
            seed: Random seed for generation
            generated_map: Pre-generated map (optional)
        """
        self.config = config
        self.seed = seed
        self.size = config.map.grid_size
        self.cell_size = config.map.cell_size
        
        # Generate or use provided map
        if generated_map is not None:
            self._map = generated_map
        else:
            generator = MapGenerator(config.map, seed)
            self._map = generator.generate()
        
        # Generate start/goal
        self.start, self.goal = generate_start_goal(self.size, seed)
        
        # Validate start/goal
        self._validate_and_fix_positions()
        
        # Statistics cache
        self._stats: Optional[Dict] = None
    
    def _validate_and_fix_positions(self):
        """Ensure start and goal are on traversable terrain"""
        # Fix start if needed
        if not self.is_valid(*self.start):
            self.start = self._find_nearest_valid(self.start)
        
        # Fix goal if needed
        if not self.is_valid(*self.goal):
            self.goal = self._find_nearest_valid(self.goal)
    
    def _find_nearest_valid(self, pos: Tuple[int, int], max_search: int = 50) -> Tuple[int, int]:
        """Find nearest valid cell to given position"""
        x, y = pos
        for radius in range(1, max_search):
            for dx in range(-radius, radius + 1):
                for dy in range(-radius, radius + 1):
                    nx, ny = x + dx, y + dy
                    if self.is_valid(nx, ny):
                        return (nx, ny)
        # Fallback
        return (100, 100)
    
    # ==================== Property Access ====================
    
    @property
    def terrain(self) -> np.ndarray:
        """Terrain type map"""
        return self._map.terrain
    
    @property
    def elevation(self) -> np.ndarray:
        """Elevation map (meters)"""
        return self._map.elevation
    
    @property
    def risk_map(self) -> np.ndarray:
        """Risk score map (0-1)"""
        return self._map.risk
    
    @property
    def uncertainty_map(self) -> np.ndarray:
        """Localization uncertainty map (0-1)"""
        return self._map.uncertainty
    
    # ==================== Cell Queries ====================
    
    def is_valid(self, x: int, y: int) -> bool:
        """Check if cell is valid (in bounds and traversable)"""
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return False
        return self.terrain[x, y] != TerrainType.WALL
    
    def is_in_bounds(self, x: int, y: int) -> bool:
        """Check if cell is within map bounds"""
        return 0 <= x < self.size and 0 <= y < self.size
    
    def get_terrain_type(self, x: int, y: int) -> TerrainType:
        """Get terrain type at cell"""
        if not self.is_in_bounds(x, y):
            return TerrainType.WALL
        return TerrainType(self.terrain[x, y])
    
    def get_terrain_name(self, x: int, y: int) -> str:
        """Get terrain name at cell"""
        return self.get_terrain_type(x, y).name_lower
    
    def get_friction_coeff(self, x: int, y: int) -> float:
        """Get rolling resistance coefficient at cell"""
        terrain_type = self.get_terrain_type(x, y)
        return TerrainProperties.get_friction(terrain_type)
    
    def get_max_velocity(self, x: int, y: int) -> float:
        """Get maximum safe velocity at cell"""
        terrain_type = self.get_terrain_type(x, y)
        return TerrainProperties.get_max_velocity(terrain_type)
    
    def get_elevation(self, x: int, y: int) -> float:
        """Get elevation at cell (meters)"""
        if not self.is_in_bounds(x, y):
            return 40.0  # Default for out of bounds
        return float(self.elevation[x, y])
    
    def get_risk(self, x: int, y: int) -> float:
        """Get risk score at cell"""
        if not self.is_in_bounds(x, y):
            return 1.0
        return float(self.risk_map[x, y])
    
    def get_uncertainty(self, x: int, y: int) -> float:
        """Get localization uncertainty at cell"""
        if not self.is_in_bounds(x, y):
            return 1.0
        return float(self.uncertainty_map[x, y])
    
    # ==================== Statistics ====================
    
    def get_stats(self) -> Dict:
        """Get map statistics (cached)"""
        if self._stats is None:
            self._stats = self._compute_stats()
        return self._stats
    
    def _compute_stats(self) -> Dict:
        """Compute map statistics"""
        total_cells = self.size * self.size
        
        terrain_counts = {}
        for t in TerrainType:
            count = np.sum(self.terrain == t)
            terrain_counts[t.name_lower] = {
                'count': int(count),
                'percentage': float(count / total_cells * 100)
            }
        
        return {
            'size': self.size,
            'cell_size_m': self.cell_size,
            'total_cells': total_cells,
            'terrain_distribution': terrain_counts,
            'elevation': {
                'min': float(self.elevation.min()),
                'max': float(self.elevation.max()),
                'mean': float(self.elevation.mean()),
                'std': float(self.elevation.std())
            },
            'risk': {
                'min': float(self.risk_map.min()),
                'max': float(self.risk_map.max()),
                'mean': float(self.risk_map.mean())
            },
            'start': self.start,
            'goal': self.goal,
            'straight_line_distance_m': float(
                np.sqrt((self.goal[0] - self.start[0])**2 + 
                       (self.goal[1] - self.start[1])**2) * self.cell_size
            )
        }
    
    # ==================== Utilities ====================
    
    def distance_to_goal(self, pos: Tuple[int, int]) -> float:
        """Euclidean distance from position to goal (in meters)"""
        return float(np.sqrt(
            (pos[0] - self.goal[0])**2 + (pos[1] - self.goal[1])**2
        ) * self.cell_size)
    
    def is_goal_reached(self, pos: Tuple[int, int], tolerance: int = 2) -> bool:
        """Check if position is at or near goal"""
        return (abs(pos[0] - self.goal[0]) <= tolerance and 
                abs(pos[1] - self.goal[1]) <= tolerance)
    
    def get_neighbors(self, x: int, y: int, include_diagonal: bool = True) -> list:
        """Get valid neighbor cells"""
        if include_diagonal:
            directions = [
                (-1, -1), (-1, 0), (-1, 1),
                (0, -1),          (0, 1),
                (1, -1),  (1, 0),  (1, 1)
            ]
        else:
            directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        
        neighbors = []
        for dx, dy in directions:
            nx, ny = x + dx, y + dy
            if self.is_valid(nx, ny):
                neighbors.append((nx, ny))
        return neighbors
    
    def save_to_npz(self, filepath: str):
        """Save environment to NPZ file"""
        np.savez_compressed(
            filepath,
            terrain=self.terrain,
            elevation=self.elevation,
            risk=self.risk_map,
            uncertainty=self.uncertainty_map,
            start=np.array(self.start),
            goal=np.array(self.goal),
            size=self.size,
            cell_size=self.cell_size,
            seed=self.seed if self.seed is not None else -1
        )
    
    @classmethod
    def load_from_npz(cls, filepath: str, config: Config) -> 'Environment':
        """Load environment from NPZ file"""
        data = np.load(filepath)
        
        generated_map = GeneratedMap(
            terrain=data['terrain'],
            elevation=data['elevation'],
            risk=data['risk'],
            uncertainty=data['uncertainty']
        )
        
        seed = int(data['seed']) if data['seed'] >= 0 else None
        env = cls(config, seed=seed, generated_map=generated_map)
        env.start = tuple(data['start'])
        env.goal = tuple(data['goal'])
        
        return env
