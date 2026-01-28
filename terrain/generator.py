"""
Terrain Generator Module
========================

Procedural generation of realistic terrain maps with diverse features.
Single Responsibility: Only generates terrain, elevation, risk, and uncertainty maps.
"""

import numpy as np
from scipy.ndimage import gaussian_filter, distance_transform_edt
from typing import Tuple, Optional, Dict
from dataclasses import dataclass

from .types import TerrainType, TerrainProperties
from ..config import MapConfig


@dataclass
class GeneratedMap:
    """Container for all generated map layers"""
    terrain: np.ndarray  # TerrainType values
    elevation: np.ndarray  # meters
    risk: np.ndarray  # 0-1 scale
    uncertainty: np.ndarray  # 0-1 scale
    
    @property
    def size(self) -> int:
        return self.terrain.shape[0]
    
    def get_slope(self) -> np.ndarray:
        """Compute slope magnitude from elevation"""
        # Assume 2m cell size for gradient calculation
        dh_dx, dh_dy = np.gradient(self.elevation, 2.0)
        return np.hypot(dh_dx, dh_dy)


class MapGenerator:
    """
    Procedural map generator for UGV navigation scenarios.
    
    Creates diverse terrain with:
    - Road networks (asphalt)
    - Natural terrain (grass, sand, mud)
    - Urban zones with buildings (walls)
    - Realistic elevation profiles
    - Risk and uncertainty maps
    """
    
    def __init__(self, config: Optional[MapConfig] = None, seed: Optional[int] = None):
        self.config = config or MapConfig()
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.size = self.config.grid_size
    
    def generate(self) -> GeneratedMap:
        """Generate complete map with all layers"""
        terrain = self._generate_terrain()
        elevation = self._generate_elevation()
        risk = self._compute_risk(terrain, elevation)
        uncertainty = self._generate_uncertainty(terrain)
        
        return GeneratedMap(
            terrain=terrain,
            elevation=elevation,
            risk=risk,
            uncertainty=uncertainty
        )
    
    def _generate_terrain(self) -> np.ndarray:
        """Generate terrain type map"""
        terrain = np.ones((self.size, self.size), dtype=np.int32) * TerrainType.GRASS
        
        # 1. Road network
        self._add_road_network(terrain)
        
        # 2. Diagonal highway
        self._add_diagonal_highway(terrain)
        
        # 3. Sand regions
        self._add_elliptical_regions(terrain, TerrainType.SAND, 
                                     self.config.num_sand_regions, (80, 180))
        
        # 4. Mud regions
        self._add_elliptical_regions(terrain, TerrainType.MUD,
                                     self.config.num_mud_regions, (60, 120))
        
        # 5. Urban zones
        self._add_urban_zones(terrain)
        
        # 6. Scattered obstacles
        self._add_scattered_obstacles(terrain)
        
        # 7. Ensure start/goal corridors are clear
        self._ensure_corridors(terrain)
        
        return terrain
    
    def _add_road_network(self, terrain: np.ndarray):
        """Add horizontal and vertical roads"""
        num_h = self.rng.integers(*self.config.num_h_roads)
        num_v = self.rng.integers(*self.config.num_v_roads)
        
        # Horizontal roads - spread across map
        h_positions = self.rng.choice(
            range(100, self.size - 100, 80), 
            min(num_h, (self.size - 200) // 80), 
            replace=False
        )
        for pos in h_positions:
            width = self.rng.integers(12, 20)
            terrain[pos:pos+width, :] = TerrainType.ASPHALT
        
        # Vertical roads
        v_positions = self.rng.choice(
            range(100, self.size - 100, 80),
            min(num_v, (self.size - 200) // 80),
            replace=False
        )
        for pos in v_positions:
            width = self.rng.integers(12, 20)
            terrain[:, pos:pos+width] = TerrainType.ASPHALT
    
    def _add_diagonal_highway(self, terrain: np.ndarray):
        """Add curved diagonal highway connecting map corners"""
        offset = self.rng.integers(-100, 100)
        for i in range(self.size):
            curve = int(np.sin(i / 200) * 50)
            road_center = int(i * 0.9 + offset + curve)
            if 0 <= road_center < self.size:
                width = self.rng.integers(14, 22)
                start = max(0, road_center - width // 2)
                end = min(self.size, road_center + width // 2)
                terrain[i, start:end] = TerrainType.ASPHALT
    
    def _add_elliptical_regions(self, terrain: np.ndarray, 
                                terrain_type: TerrainType,
                                count_range: Tuple[int, int],
                                radius_range: Tuple[int, int]):
        """Add elliptical terrain regions"""
        num_regions = self.rng.integers(*count_range)
        
        for _ in range(num_regions):
            cx = self.rng.integers(150, self.size - 150)
            cy = self.rng.integers(150, self.size - 150)
            rx = self.rng.integers(*radius_range)
            ry = self.rng.integers(*radius_range)
            
            # Use meshgrid for efficient ellipse mask
            y, x = np.ogrid[max(0, cx-rx):min(self.size, cx+rx),
                           max(0, cy-ry):min(self.size, cy+ry)]
            
            # Offset for local coordinates
            lx = max(0, cx-rx)
            ly = max(0, cy-ry)
            
            mask = ((x + ly - cy) / ry) ** 2 + ((y + lx - cx) / rx) ** 2 < 1.0
            
            # Apply to terrain
            terrain[max(0, cx-rx):min(self.size, cx+rx),
                   max(0, cy-ry):min(self.size, cy+ry)][mask] = terrain_type
    
    def _add_urban_zones(self, terrain: np.ndarray):
        """Add urban zones with building clusters"""
        num_urban = self.rng.integers(*self.config.num_urban_zones)
        
        for _ in range(num_urban):
            ux = self.rng.integers(200, self.size - 200)
            uy = self.rng.integers(200, self.size - 200)
            zone_size = self.rng.integers(100, 180)
            
            # Create building clusters
            num_buildings = self.rng.integers(15, 30)
            for _ in range(num_buildings):
                bx = ux + self.rng.integers(-zone_size//2, zone_size//2)
                by = uy + self.rng.integers(-zone_size//2, zone_size//2)
                bw = self.rng.integers(12, 30)
                bh = self.rng.integers(12, 30)
                
                if 50 < bx < self.size - 50 and 50 < by < self.size - 50:
                    terrain[bx:min(self.size, bx+bw), 
                           by:min(self.size, by+bh)] = TerrainType.WALL
    
    def _add_scattered_obstacles(self, terrain: np.ndarray):
        """Add randomly scattered obstacles"""
        num_obstacles = self.rng.integers(*self.config.num_obstacles)
        
        for _ in range(num_obstacles):
            ox = self.rng.integers(100, self.size - 100)
            oy = self.rng.integers(100, self.size - 100)
            size = self.rng.integers(8, 25)
            
            terrain[ox:min(self.size, ox+size),
                   oy:min(self.size, oy+size)] = TerrainType.WALL
    
    def _ensure_corridors(self, terrain: np.ndarray):
        """Ensure clear corridors near typical start/goal regions"""
        # Clear start region (bottom-left quadrant)
        for i in range(50, 200):
            for j in range(50, 200):
                if terrain[i, j] == TerrainType.WALL:
                    terrain[i, j] = TerrainType.GRASS
        
        # Clear goal region (top-right quadrant)
        for i in range(self.size - 200, self.size - 50):
            for j in range(self.size - 200, self.size - 50):
                if terrain[i, j] == TerrainType.WALL:
                    terrain[i, j] = TerrainType.GRASS
        
        # Add connecting roads near edges
        # Bottom edge road
        terrain[30:45, :] = TerrainType.ASPHALT
        # Top edge road
        terrain[self.size-45:self.size-30, :] = TerrainType.ASPHALT
        # Left edge road
        terrain[:, 30:45] = TerrainType.ASPHALT
        # Right edge road
        terrain[:, self.size-45:self.size-30] = TerrainType.ASPHALT
    
    def _generate_elevation(self) -> np.ndarray:
        """Generate realistic elevation profile"""
        x = np.linspace(0, 4*np.pi, self.size)
        y = np.linspace(0, 4*np.pi, self.size)
        X, Y = np.meshgrid(x, y)
        
        # Multiple sine waves for natural terrain
        elevation = (20 * np.sin(X) * np.cos(Y) + 
                    15 * np.sin(2*X + 1) * np.sin(2*Y) +
                    25 * np.cos(X/2) * np.sin(Y/2))
        
        # Add noise
        noise = self.rng.standard_normal((self.size, self.size)) * 3
        elevation += noise
        
        # Smooth
        elevation = gaussian_filter(elevation, sigma=15.0)
        
        # Normalize to 0-80m range
        elevation = (elevation - elevation.min()) / (elevation.max() - elevation.min() + 1e-9) * 80
        
        # Add steep hills
        hill_centers = [
            (self.size // 5, self.size // 5),
            (4 * self.size // 5, 4 * self.size // 5),
            (2 * self.size // 5, 3 * self.size // 5),
            (3 * self.size // 5, 2 * self.size // 5)
        ]
        
        for cx, cy in hill_centers:
            r = self.rng.integers(80, 120)
            height = self.rng.integers(25, 40)
            
            y_grid, x_grid = np.ogrid[:self.size, :self.size]
            dist = np.sqrt((y_grid - cx)**2 + (x_grid - cy)**2)
            mask = dist < r
            elevation[mask] += height * np.exp(-(dist[mask]**2) / (2 * (r/2)**2))
        
        return elevation.astype(np.float32)
    
    def _compute_risk(self, terrain: np.ndarray, elevation: np.ndarray) -> np.ndarray:
        """Compute risk map from terrain and slope"""
        # Slope magnitude
        dh_dx, dh_dy = np.gradient(elevation, self.config.cell_size)
        slope = np.hypot(dh_dx, dh_dy)
        
        # Base terrain risk lookup
        risk = np.zeros_like(terrain, dtype=np.float32)
        for t_type in TerrainType:
            mask = terrain == t_type
            risk[mask] = TerrainProperties.get_base_risk(t_type)
        
        # Combine with slope
        risk = np.minimum(1.0, slope * 3.0 + risk)
        
        # Proximity risk near walls
        wall_mask = terrain == TerrainType.WALL
        if np.any(wall_mask):
            dist_cells = distance_transform_edt(~wall_mask)
            within_5 = dist_cells <= 5.0
            prox_risk = np.zeros_like(risk)
            prox_risk[within_5] = 0.3 * np.exp(-(dist_cells[within_5] ** 2) / 10.0)
            risk = np.minimum(1.0, risk + prox_risk)
        
        return risk
    
    def _generate_uncertainty(self, terrain: np.ndarray) -> np.ndarray:
        """Generate localization uncertainty map"""
        uncertainty = self.rng.random((self.size, self.size)).astype(np.float32) * 0.2
        
        # Higher near walls
        wall_mask = terrain == TerrainType.WALL
        if np.any(wall_mask):
            dist_cells = distance_transform_edt(~wall_mask)
            within_8 = dist_cells <= 8.0
            uncertainty[within_8] = np.minimum(1.0, uncertainty[within_8] + 0.4)
        
        # Lower on roads
        uncertainty[terrain == TerrainType.ASPHALT] *= 0.3
        
        return uncertainty


def generate_start_goal(size: int, seed: Optional[int] = None) -> Tuple[Tuple[int, int], Tuple[int, int]]:
    """
    Generate valid start and goal positions.
    
    Start: Bottom-left region [50, 200]
    Goal: Top-right region [size-200, size-50]
    """
    rng = np.random.default_rng(seed)
    
    start = (
        rng.integers(80, 180),
        rng.integers(80, 180)
    )
    
    goal = (
        rng.integers(size - 180, size - 80),
        rng.integers(size - 180, size - 80)
    )
    
    return start, goal
