"""
Local Environment Module
========================

Field-of-View (FoV) wrapper around the global environment.
Simulates limited sensing by treating cells outside the FoV window
as "unknown" with configurable default properties.
"""

import numpy as np
from typing import Tuple, Optional, Dict, Set
from dataclasses import dataclass, field

from .world import Environment
from ..config import Config, UnknownTerrainConfig
from ..terrain import TerrainType, TerrainProperties


@dataclass
class FoVBounds:
    """Field of View boundary coordinates"""
    xmin: int
    xmax: int
    ymin: int
    ymax: int
    
    @property
    def width(self) -> int:
        return self.xmax - self.xmin + 1
    
    @property
    def height(self) -> int:
        return self.ymax - self.ymin + 1
    
    @property
    def area(self) -> int:
        return self.width * self.height
    
    def contains(self, x: int, y: int) -> bool:
        """Check if position is within bounds"""
        return self.xmin <= x <= self.xmax and self.ymin <= y <= self.ymax
    
    def as_dict(self) -> Dict[str, int]:
        return {
            'xmin': self.xmin,
            'xmax': self.xmax,
            'ymin': self.ymin,
            'ymax': self.ymax
        }
    
    @classmethod
    def from_position(cls, pos: Tuple[int, int], radius: int, grid_size: int) -> 'FoVBounds':
        """Create FoV bounds centered at position"""
        x, y = pos
        return cls(
            xmin=max(0, x - radius),
            xmax=min(grid_size - 1, x + radius),
            ymin=max(0, y - radius),
            ymax=min(grid_size - 1, y + radius)
        )


class LocalEnvironment:
    """
    FoV-constrained view of the global environment.
    
    This wrapper simulates limited sensing range:
    - Cells inside FoV window: true values from global environment
    - Cells outside FoV window: conservative default values
    
    Features:
    - Adaptive unknown terrain modeling
    - Global memory (visited cells, failed cells)
    - Terrain statistics for adaptive defaults
    """
    
    def __init__(self, 
                 global_env: Environment,
                 bounds: FoVBounds,
                 unknown_config: Optional[UnknownTerrainConfig] = None,
                 visited_counts: Optional[Dict[Tuple[int, int], int]] = None,
                 failed_cells: Optional[Set[Tuple[int, int]]] = None):
        """
        Initialize local environment.
        
        Args:
            global_env: Reference to full global environment
            bounds: FoV boundary coordinates
            unknown_config: Configuration for unknown terrain handling
            visited_counts: Global visitation counts (shared memory)
            failed_cells: Set of cells where planning failed (shared memory)
        """
        self._global_env = global_env
        self._bounds = bounds
        self._unknown_config = unknown_config or UnknownTerrainConfig()
        
        # Shared global memory
        self.visited_counts = visited_counts or {}
        self.failed_cells = failed_cells or set()
        
        # Compute adaptive defaults from observed terrain
        self._adaptive_defaults = self._compute_adaptive_defaults()
        
        # Cache for properties
        self.config = global_env.config
        self.size = global_env.size
    
    def _compute_adaptive_defaults(self) -> Dict[str, float]:
        """Compute adaptive unknown terrain defaults from observed region"""
        if self._unknown_config.mode != 'adaptive':
            return self._unknown_config.get_defaults()
        
        # Sample terrain in visible region
        terrain_in_view = self._global_env.terrain[
            self._bounds.xmin:self._bounds.xmax+1,
            self._bounds.ymin:self._bounds.ymax+1
        ]
        
        elev_in_view = self._global_env.elevation[
            self._bounds.xmin:self._bounds.xmax+1,
            self._bounds.ymin:self._bounds.ymax+1
        ]
        
        # Compute statistics
        traversable_mask = terrain_in_view != TerrainType.WALL
        if not np.any(traversable_mask):
            return self._unknown_config.balanced
        
        # Average friction of visible terrain
        frictions = []
        for t in TerrainType.traversable_types():
            mask = terrain_in_view == t
            if np.any(mask):
                frictions.extend([TerrainProperties.get_friction(t)] * np.sum(mask))
        
        avg_friction = np.mean(frictions) if frictions else 0.15
        avg_elevation = np.mean(elev_in_view[traversable_mask])
        
        # Blend between observed and slightly pessimistic
        return {
            'friction': float(avg_friction * 1.2),  # Slightly pessimistic
            'risk': 0.4,
            'elevation': float(avg_elevation)
        }
    
    # ==================== FoV Queries ====================
    
    def in_view(self, x: int, y: int) -> bool:
        """Check if cell is within current FoV"""
        return self._bounds.contains(x, y)
    
    @property
    def bounds(self) -> FoVBounds:
        """Get current FoV bounds"""
        return self._bounds
    
    def update_bounds(self, new_bounds: FoVBounds):
        """Update FoV bounds (for adaptive FoV)"""
        self._bounds = new_bounds
        self._adaptive_defaults = self._compute_adaptive_defaults()
    
    # ==================== Cell Queries ====================
    
    def is_valid(self, x: int, y: int) -> bool:
        """Check if cell is valid (handles unknown regions)"""
        # Out of global bounds: invalid
        if x < 0 or x >= self.size or y < 0 or y >= self.size:
            return False
        
        # Walls are always invalid (we assume we can detect them)
        if self._global_env.terrain[x, y] == TerrainType.WALL:
            return False
        
        # Failed cells are treated as high-risk but not invalid
        # (allows recovery attempts)
        
        return True
    
    def get_terrain_type(self, x: int, y: int) -> TerrainType:
        """Get terrain type (unknown outside FoV returns GRASS as default)"""
        if not self._global_env.is_in_bounds(x, y):
            return TerrainType.WALL
        
        # Inside FoV: true value
        if self.in_view(x, y):
            return TerrainType(self._global_env.terrain[x, y])
        
        # Outside FoV: assume traversable (but penalized)
        return TerrainType.GRASS
    
    def get_terrain_name(self, x: int, y: int) -> str:
        """Get terrain name at cell"""
        return self.get_terrain_type(x, y).name_lower
    
    def get_friction_coeff(self, x: int, y: int) -> float:
        """Get friction coefficient (conservative default outside FoV)"""
        if self.in_view(x, y):
            return self._global_env.get_friction_coeff(x, y)
        return self._adaptive_defaults['friction']
    
    def get_max_velocity(self, x: int, y: int) -> float:
        """Get max velocity (conservative outside FoV)"""
        if self.in_view(x, y):
            return self._global_env.get_max_velocity(x, y)
        # Conservative speed in unknown terrain
        return min(5.0, self.config.terrain.max_velocity.get('grass', 8.0))
    
    def get_elevation(self, x: int, y: int) -> float:
        """Get elevation (default for unknown)"""
        if self.in_view(x, y):
            return self._global_env.get_elevation(x, y)
        return self._adaptive_defaults['elevation']
    
    def get_risk(self, x: int, y: int) -> float:
        """Get risk score (high for unknown and failed cells)"""
        # Failed cells get extra risk penalty
        if (x, y) in self.failed_cells:
            return 0.9
        
        if self.in_view(x, y):
            return self._global_env.get_risk(x, y)
        return self._adaptive_defaults['risk']
    
    def get_uncertainty(self, x: int, y: int) -> float:
        """Get uncertainty (high outside FoV)"""
        if self.in_view(x, y):
            return self._global_env.get_uncertainty(x, y)
        return 0.8  # High uncertainty for unknown
    
    # ==================== Memory Queries ====================
    
    def get_visited_penalty(self, x: int, y: int) -> float:
        """Get penalty based on visitation count"""
        count = self.visited_counts.get((x, y), 0)
        # Exponential penalty for repeated visits
        return float(count * 0.1)
    
    def is_failed_cell(self, x: int, y: int) -> bool:
        """Check if cell is marked as failed"""
        return (x, y) in self.failed_cells
    
    def mark_failed(self, x: int, y: int):
        """Mark cell as failed (dead-end discovered)"""
        self.failed_cells.add((x, y))
    
    def increment_visit(self, x: int, y: int):
        """Increment visit count for cell"""
        key = (x, y)
        self.visited_counts[key] = self.visited_counts.get(key, 0) + 1
    
    # ==================== Global Environment Access ====================
    
    @property
    def terrain(self) -> np.ndarray:
        """Direct access to terrain array (for compatibility)"""
        return self._global_env.terrain
    
    @property
    def elevation(self) -> np.ndarray:
        """Direct access to elevation array"""
        return self._global_env.elevation
    
    @property
    def risk_map(self) -> np.ndarray:
        """Direct access to risk map"""
        return self._global_env.risk_map
    
    @property
    def uncertainty_map(self) -> np.ndarray:
        """Direct access to uncertainty map"""
        return self._global_env.uncertainty_map
    
    @property
    def start(self) -> Tuple[int, int]:
        return self._global_env.start
    
    @property
    def goal(self) -> Tuple[int, int]:
        return self._global_env.goal
    
    def get_global_env(self) -> Environment:
        """Get reference to global environment"""
        return self._global_env


def extract_local_bounds(pos: Tuple[int, int], radius: int, grid_size: int) -> FoVBounds:
    """
    Extract FoV bounds centered at position.
    
    Args:
        pos: Center position (x, y)
        radius: FoV radius in cells
        grid_size: Total grid size
    
    Returns:
        FoVBounds object
    """
    return FoVBounds.from_position(pos, radius, grid_size)


def create_local_environment(global_env: Environment,
                            pos: Tuple[int, int],
                            radius: int,
                            unknown_config: Optional[UnknownTerrainConfig] = None,
                            visited_counts: Optional[Dict] = None,
                            failed_cells: Optional[Set] = None) -> LocalEnvironment:
    """
    Factory function to create LocalEnvironment.
    
    Args:
        global_env: Global environment reference
        pos: Current robot position
        radius: FoV radius in cells
        unknown_config: Unknown terrain configuration
        visited_counts: Shared visited counts dict
        failed_cells: Shared failed cells set
    
    Returns:
        LocalEnvironment instance
    """
    bounds = extract_local_bounds(pos, radius, global_env.size)
    return LocalEnvironment(
        global_env=global_env,
        bounds=bounds,
        unknown_config=unknown_config,
        visited_counts=visited_counts,
        failed_cells=failed_cells
    )
