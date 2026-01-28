"""
Energy Model Module
===================

Comprehensive energy consumption model for UGV navigation.
Includes terrain effects, slope work, aerodynamic drag, 
turning losses, and regenerative braking.
"""

import numpy as np
from typing import Tuple, Dict, Optional, Union
from dataclasses import dataclass, field

from ..config import Config
from ..terrain import TerrainType, TerrainProperties


@dataclass
class EnergyBreakdown:
    """Detailed energy breakdown for a path segment"""
    slope: float = 0.0       # Gravitational potential energy
    friction: float = 0.0    # Rolling resistance
    aero: float = 0.0        # Aerodynamic drag
    turn: float = 0.0        # Turning energy
    accel: float = 0.0       # Acceleration energy
    regen: float = 0.0       # Regenerative braking (negative)
    terramech: float = 0.0   # Terramechanics (sinkage)
    risk: float = 0.0        # Risk penalty
    uncertainty: float = 0.0  # Localization uncertainty penalty
    
    @property
    def total(self) -> float:
        """Total energy consumption"""
        return (self.slope + self.friction + self.aero + self.turn + 
                self.accel + self.regen + self.terramech + 
                self.risk + self.uncertainty)
    
    def to_dict(self) -> Dict[str, float]:
        return {
            'slope': self.slope,
            'friction': self.friction,
            'aero': self.aero,
            'turn': self.turn,
            'accel': self.accel,
            'regen': self.regen,
            'terramech': self.terramech,
            'risk': self.risk,
            'uncertainty': self.uncertainty,
            'total': self.total
        }
    
    def __add__(self, other: 'EnergyBreakdown') -> 'EnergyBreakdown':
        return EnergyBreakdown(
            slope=self.slope + other.slope,
            friction=self.friction + other.friction,
            aero=self.aero + other.aero,
            turn=self.turn + other.turn,
            accel=self.accel + other.accel,
            regen=self.regen + other.regen,
            terramech=self.terramech + other.terramech,
            risk=self.risk + other.risk,
            uncertainty=self.uncertainty + other.uncertainty
        )


class EnergyModel:
    """
    Physics-based energy consumption model.
    
    Computes total energy for path segments considering:
    - Slope work (uphill/downhill)
    - Rolling resistance (terrain-dependent)
    - Aerodynamic drag
    - Turning losses
    - Acceleration/deceleration
    - Regenerative braking recovery
    - Terramechanics (soft terrain sinkage)
    - Risk and uncertainty penalties
    """
    
    def __init__(self, env, config: Config):
        """
        Initialize energy model.
        
        Args:
            env: Environment (global or local)
            config: Configuration object
        """
        self.env = env
        self.config = config
        
        # Vehicle parameters
        self.mass = config.vehicle.mass
        self.gravity = config.energy.gravity
        self.drive_efficiency = config.vehicle.drive_efficiency
        self.drag_coeff = config.vehicle.drag_coefficient
        self.frontal_area = config.vehicle.frontal_area
        self.air_density = config.energy.air_density
        
        # Energy coefficients
        self.k_turn = config.energy.k_turn
        self.regen_factor = config.energy.regen_factor
        self.c_sinkage = config.energy.c_sinkage
        self.risk_weight = config.energy.risk_weight
        self.uncertainty_weight = config.energy.uncertainty_weight
        
        # Cell size
        self.cell_size = config.map.cell_size
        
        # Cache for repeated calculations
        self._cache: Dict[Tuple, Tuple[float, EnergyBreakdown]] = {}
        self._cache_hits = 0
        self._cache_misses = 0
    
    def calculate_segment_energy(self, 
                                 x1: int, y1: int, v1: float,
                                 x2: int, y2: int, v2: float,
                                 use_cache: bool = True) -> Tuple[float, EnergyBreakdown]:
        """
        Calculate total energy for traversing a segment.
        
        Args:
            x1, y1: Start position
            v1: Start velocity (m/s)
            x2, y2: End position
            v2: End velocity (m/s)
            use_cache: Whether to use result cache
        
        Returns:
            Tuple of (total_energy, breakdown)
        """
        # Invalid destination
        if not self.env.is_valid(x2, y2):
            return float('inf'), EnergyBreakdown()
        
        # Check cache
        if use_cache:
            cache_key = (x1, y1, int(v1*100), x2, y2, int(v2*100))
            if cache_key in self._cache:
                self._cache_hits += 1
                return self._cache[cache_key]
            self._cache_misses += 1
        
        # Calculate distance
        distance = np.sqrt((x2 - x1)**2 + (y2 - y1)**2) * self.cell_size
        
        if distance < 1e-6:
            result = (0.0, EnergyBreakdown())
            if use_cache:
                self._cache[cache_key] = result
            return result
        
        # Average velocity
        v_avg = (v1 + v2) / 2
        
        # 1. Slope energy (gravitational potential)
        delta_h = self.env.elevation[x2, y2] - self.env.elevation[x1, y1]
        if delta_h > 0:
            # Uphill: energy required
            e_slope = self.mass * self.gravity * delta_h / self.drive_efficiency
        else:
            # Downhill: some energy still needed for control
            e_slope = self.mass * self.gravity * delta_h * 0.3
        
        # 2. Rolling resistance (friction)
        friction_coeff = self.env.get_friction_coeff(x2, y2)
        e_friction = friction_coeff * self.mass * self.gravity * distance
        
        # 3. Aerodynamic drag
        e_aero = 0.5 * self.drag_coeff * self.frontal_area * self.air_density * (v_avg ** 2) * distance
        
        # 4. Turning energy
        e_turn = self._calculate_turn_energy(x1, y1, x2, y2, v_avg)
        
        # 5. Acceleration energy
        if v2 > v1:
            e_accel = 0.5 * self.mass * (v2**2 - v1**2) / self.drive_efficiency
        else:
            # Deceleration: some energy lost as heat
            e_accel = 0.5 * self.mass * (v2**2 - v1**2) * 0.2
        
        # 6. Regenerative braking
        e_regen = 0.0
        if delta_h < 0:
            potential_recovery = abs(self.mass * self.gravity * delta_h)
            e_regen += self.regen_factor * potential_recovery
        if v2 < v1:
            kinetic_recovery = 0.5 * self.mass * (v1**2 - v2**2)
            e_regen += self.regen_factor * kinetic_recovery * 0.6
        
        # 7. Terramechanics (soft terrain sinkage)
        terrain_type = self.env.get_terrain_type(x2, y2)
        sinkage_mult = TerrainProperties.get_sinkage_multiplier(terrain_type)
        e_terramech = self.c_sinkage * distance * sinkage_mult
        
        # 8. Risk penalty
        risk = self.env.get_risk(x2, y2)
        e_risk = risk * self.risk_weight
        
        # 9. Uncertainty penalty
        uncertainty = self.env.get_uncertainty(x2, y2)
        e_uncertainty = uncertainty * self.uncertainty_weight
        
        # Create breakdown
        breakdown = EnergyBreakdown(
            slope=e_slope,
            friction=e_friction,
            aero=e_aero,
            turn=e_turn,
            accel=e_accel,
            regen=e_regen,
            terramech=e_terramech,
            risk=e_risk,
            uncertainty=e_uncertainty
        )
        
        total = max(0.1, breakdown.total)  # Minimum positive energy
        
        # Cache result
        if use_cache:
            self._cache[cache_key] = (total, breakdown)
        
        return total, breakdown
    
    def _calculate_turn_energy(self, x1: int, y1: int, 
                               x2: int, y2: int, velocity: float) -> float:
        """Calculate turning energy penalty"""
        if x2 == x1 and y2 == y1:
            return 0.0
        
        # Angle of movement
        angle = abs(np.arctan2(y2 - y1, x2 - x1))
        
        # Turning penalty increases with speed
        turn_penalty = self.k_turn * angle * (1 + (velocity / 10.0)**2)
        return turn_penalty
    
    def calculate_path_energy(self, 
                              path: list,
                              velocities: Optional[list] = None,
                              default_velocity: float = 5.0) -> Tuple[float, EnergyBreakdown]:
        """
        Calculate total energy for a complete path.
        
        Args:
            path: List of (x, y) positions
            velocities: Optional list of velocities (len = len(path))
            default_velocity: Default velocity if not specified
        
        Returns:
            Tuple of (total_energy, cumulative_breakdown)
        """
        if len(path) < 2:
            return 0.0, EnergyBreakdown()
        
        # Default velocities
        if velocities is None:
            velocities = [default_velocity] * len(path)
        
        total_energy = 0.0
        total_breakdown = EnergyBreakdown()
        
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            v1 = velocities[i]
            v2 = velocities[i + 1]
            
            energy, breakdown = self.calculate_segment_energy(x1, y1, v1, x2, y2, v2)
            
            if energy == float('inf'):
                return float('inf'), EnergyBreakdown()
            
            total_energy += energy
            total_breakdown = total_breakdown + breakdown
        
        return total_energy, total_breakdown
    
    def get_cache_stats(self) -> Dict[str, int]:
        """Get cache hit/miss statistics"""
        return {
            'hits': self._cache_hits,
            'misses': self._cache_misses,
            'size': len(self._cache)
        }
    
    def clear_cache(self):
        """Clear the energy calculation cache"""
        self._cache.clear()
        self._cache_hits = 0
        self._cache_misses = 0


class VelocityProfiler:
    """
    Computes velocity profiles for paths with kinodynamic constraints.
    
    Enforces:
    - Terrain-based max velocities
    - Acceleration limits (forward pass)
    - Deceleration limits (backward pass)
    """
    
    def __init__(self, env, config: Config, mode: str = 'energy'):
        """
        Initialize velocity profiler.
        
        Args:
            env: Environment (global or local)
            config: Configuration object
            mode: 'time' for aggressive, 'energy' for conservative
        """
        self.env = env
        self.config = config
        self.mode = mode
        
        self.a_max = config.vehicle.a_max
        self.a_min = config.vehicle.a_min
        self.cell_size = config.map.cell_size
        
        # Get velocity profile for mode
        self.velocity_profile = config.get_velocity_profile(mode)
    
    def compute_velocities(self, path: list) -> np.ndarray:
        """
        Compute kinodynamically feasible velocities for path.
        
        Args:
            path: List of (x, y) positions
        
        Returns:
            Array of velocities for each path node
        """
        n = len(path)
        if n == 0:
            return np.array([])
        
        # Initialize with terrain-based targets
        velocities = np.zeros(n)
        for i, (x, y) in enumerate(path):
            terrain_name = self.env.get_terrain_name(x, y)
            max_v = self.env.get_max_velocity(x, y)
            target_v = self.velocity_profile.get(terrain_name, 5.0)
            velocities[i] = min(target_v, max_v)
        
        # Ensure start and end are slower
        velocities[0] = min(velocities[0], 3.0)
        velocities[-1] = min(velocities[-1], 2.0)
        
        # Forward pass: acceleration limits
        for i in range(n - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2) * self.cell_size
            
            v1 = velocities[i]
            max_reachable = np.sqrt(max(0.0, v1**2 + 2.0 * self.a_max * dist))
            
            if velocities[i + 1] > max_reachable:
                velocities[i + 1] = max_reachable
        
        # Backward pass: deceleration limits
        brake = abs(self.a_min)
        for i in range(n - 1, 0, -1):
            x1, y1 = path[i - 1]
            x2, y2 = path[i]
            dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2) * self.cell_size
            
            v2 = velocities[i]
            max_reachable = np.sqrt(max(0.0, v2**2 + 2.0 * brake * dist))
            
            if velocities[i - 1] > max_reachable:
                velocities[i - 1] = max_reachable
        
        return velocities
    
    def get_target_velocity(self, terrain_name: str) -> float:
        """Get target velocity for terrain type"""
        return self.velocity_profile.get(terrain_name, 5.0)
