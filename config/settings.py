"""
Configuration Settings Module
==============================

Dataclass-based configuration with validation and defaults.
Follows Single Responsibility Principle - only handles configuration.
"""

from dataclasses import dataclass, field
from typing import Dict, Optional, Any
import numpy as np


@dataclass
class VehicleConfig:
    """Vehicle physical parameters"""
    mass: float = 1800.0  # kg
    drag_coefficient: float = 0.6
    frontal_area: float = 2.5  # m²
    drive_efficiency: float = 0.88
    
    # Kinodynamic constraints
    a_max: float = 2.5  # m/s² max acceleration
    a_min: float = -4.0  # m/s² max deceleration
    max_turn_angle: float = np.pi / 6  # 30 degrees per step


@dataclass
class TerrainConfig:
    """Terrain-related parameters"""
    # Friction coefficients (rolling resistance)
    friction: Dict[str, float] = field(default_factory=lambda: {
        'asphalt': 0.015,
        'grass': 0.08,
        'mud': 0.25,
        'sand': 0.15,
        'wall': float('inf')
    })
    
    # Maximum velocities (m/s)
    max_velocity: Dict[str, float] = field(default_factory=lambda: {
        'asphalt': 15.0,  # 54 km/h
        'grass': 8.0,     # 29 km/h
        'mud': 3.0,       # 11 km/h
        'sand': 5.0,      # 18 km/h
        'wall': 0.0
    })


@dataclass
class EnergyConfig:
    """Energy model parameters"""
    gravity: float = 9.81  # m/s²
    air_density: float = 1.225  # kg/m³
    
    # Energy coefficients
    k_turn: float = 800.0  # Turning energy (J/radian)
    regen_factor: float = -0.65  # Regenerative braking efficiency
    c_sinkage: float = 450.0  # Terramechanics sinkage (J/m)
    
    # Penalty weights
    risk_weight: float = 500.0  # J
    uncertainty_weight: float = 200.0  # J


@dataclass
class FoVConfig:
    """Field of View configuration with adaptive parameters"""
    base_radius_cells: int = 25  # ~50m at 2m/cell
    min_radius_cells: int = 15
    max_radius_cells: int = 50  # Allow expansion up to 100m
    
    # Adaptive scaling
    expansion_factor: float = 1.5  # Multiplier when stuck
    contraction_rate: int = 5  # Cells to reduce when succeeding
    
    # Execute steps
    base_execute_steps: int = 12
    min_execute_steps: int = 3
    max_execute_steps: int = 25


@dataclass
class UnknownTerrainConfig:
    """Configuration for unknown (outside FoV) terrain modeling"""
    mode: str = 'adaptive'  # 'optimistic', 'balanced', 'pessimistic', 'adaptive'
    
    # Pessimistic defaults (original)
    pessimistic: Dict[str, float] = field(default_factory=lambda: {
        'friction': 0.5,
        'risk': 1.0,
        'elevation': 40.0
    })
    
    # Balanced defaults
    balanced: Dict[str, float] = field(default_factory=lambda: {
        'friction': 0.15,
        'risk': 0.5,
        'elevation': 35.0
    })
    
    # Optimistic defaults
    optimistic: Dict[str, float] = field(default_factory=lambda: {
        'friction': 0.08,
        'risk': 0.2,
        'elevation': 30.0
    })
    
    def get_defaults(self) -> Dict[str, float]:
        """Get current mode's defaults"""
        if self.mode == 'pessimistic':
            return self.pessimistic
        elif self.mode == 'optimistic':
            return self.optimistic
        else:
            return self.balanced


@dataclass
class RecoveryConfig:
    """Dead-end recovery configuration"""
    enabled: bool = True
    max_recovery_attempts: int = 5
    backtrack_threshold: float = 0.25
    
    # Strategy priorities (tried in order)
    strategies: tuple = ('expand_fov', 'backtrack', 'random_escape', 'global_replan')
    
    # Backtrack settings
    max_backtrack_steps: int = 50
    backtrack_penalty_factor: float = 1.5
    
    # Global replan settings
    global_replan_resolution: int = 10  # Coarse planning resolution


@dataclass 
class GAConfig:
    """Genetic Algorithm configuration"""
    pop_size: int = 50
    generations: int = 35
    elite_frac: float = 0.10
    crossover_rate: float = 0.85
    mutation_rate: float = 0.25
    tournament_k: int = 4
    
    # Genome structure
    n_via: int = 4
    max_via_jitter: int = 20
    
    # Objective weights: J = wE*E + wT*T + wS*S
    weights: Dict[str, float] = field(default_factory=lambda: {
        'energy': 0.4,
        'time': 0.3,
        'safety': 0.0
    })


@dataclass
class SurrogateConfig:
    """Surrogate model configuration"""
    enabled: bool = True
    warmup_generations: int = 5
    surrogate_fraction: float = 0.80
    top_true_fraction: float = 0.25
    retrain_interval: int = 3
    max_mape_threshold: float = 0.20
    
    # Local surrogate ensemble
    use_local_ensemble: bool = True
    grid_divisions: int = 4  # 4x4 = 16 local models
    min_samples_per_region: int = 10


@dataclass
class MapConfig:
    """Map generation configuration"""
    size_meters: float = 2000.0  # 2km x 2km
    cell_size: float = 2.0  # 2m per cell
    
    @property
    def grid_size(self) -> int:
        return int(self.size_meters / self.cell_size)
    
    # Generation parameters
    num_h_roads: tuple = (3, 6)  # min, max
    num_v_roads: tuple = (3, 6)
    num_sand_regions: tuple = (5, 9)
    num_mud_regions: tuple = (4, 8)
    num_urban_zones: tuple = (2, 5)
    num_obstacles: tuple = (20, 40)


@dataclass
class VisualizationConfig:
    """Visualization and debugging configuration"""
    enabled: bool = True
    live_monitor: bool = False  # Real-time visualization
    save_frames: bool = False
    frame_interval: float = 0.1  # seconds
    
    # Colors
    terrain_colors: tuple = ('darkgray', 'lightgreen', 'saddlebrown', 'gold', 'black')
    path_colors: Dict[str, str] = field(default_factory=lambda: {
        'full_map_time': 'blue',
        'full_map_energy': 'orange', 
        'fov_time': 'green',
        'fov_energy': 'red',
        'fov_ga': 'purple',
        'fov_ga_surrogate': 'brown',
        'recovery': 'cyan'
    })


@dataclass
class Config:
    """
    Master configuration class combining all sub-configurations.
    
    Usage:
        config = Config()
        config = Config(fov=FoVConfig(base_radius_cells=30))
    """
    vehicle: VehicleConfig = field(default_factory=VehicleConfig)
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    energy: EnergyConfig = field(default_factory=EnergyConfig)
    fov: FoVConfig = field(default_factory=FoVConfig)
    unknown: UnknownTerrainConfig = field(default_factory=UnknownTerrainConfig)
    recovery: RecoveryConfig = field(default_factory=RecoveryConfig)
    ga: GAConfig = field(default_factory=GAConfig)
    surrogate: SurrogateConfig = field(default_factory=SurrogateConfig)
    map: MapConfig = field(default_factory=MapConfig)
    visualization: VisualizationConfig = field(default_factory=VisualizationConfig)
    
    # Global settings
    random_seed: Optional[int] = None
    verbose: bool = False
    max_iterations: int = 20000
    max_total_seconds: float = 1200.0  # 20 minutes
    max_replan_seconds: float = 60.0
    
    # Velocity profiles (computed from terrain config)
    _v_time_profile: Dict[str, float] = field(default_factory=dict, repr=False)
    _v_energy_profile: Dict[str, float] = field(default_factory=dict, repr=False)
    
    def __post_init__(self):
        """Compute derived values after initialization"""
        self._compute_velocity_profiles()
    
    def _compute_velocity_profiles(self):
        """Compute mode-specific velocity profiles"""
        # Time-optimal: 85-95% of max speeds
        self._v_time_profile = {
            terrain: v * 0.90 
            for terrain, v in self.terrain.max_velocity.items()
        }
        self._v_time_profile['mud'] = self.terrain.max_velocity['mud'] * 0.85
        
        # Energy-optimal: 40-72% of max speeds
        self._v_energy_profile = {
            'asphalt': self.terrain.max_velocity['asphalt'] * 0.72,
            'grass': self.terrain.max_velocity['grass'] * 0.72,
            'mud': self.terrain.max_velocity['mud'] * 0.62,
            'sand': self.terrain.max_velocity['sand'] * 0.67,
            'wall': 0.0
        }
    
    def get_velocity_profile(self, mode: str) -> Dict[str, float]:
        """Get velocity profile for planning mode"""
        if mode == 'time':
            return self._v_time_profile
        else:
            return self._v_energy_profile
    
    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Config':
        """Create Config from dictionary"""
        config = cls()
        for key, value in d.items():
            if hasattr(config, key):
                setattr(config, key, value)
        config._compute_velocity_profiles()
        return config
    
    def to_dict(self) -> Dict[str, Any]:
        """Export config to dictionary"""
        from dataclasses import asdict
        return asdict(self)


# Convenience aliases
GASettings = GAConfig
