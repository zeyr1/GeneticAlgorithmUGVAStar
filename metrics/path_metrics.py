"""
Path Metrics Module
===================

Comprehensive metrics tracking for path evaluation.
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
import math

from ..terrain import TerrainType


@dataclass
class PathMetrics:
    """
    Comprehensive metrics for a planned/executed path.
    
    Tracks:
    - Distance, time, energy
    - Velocity statistics
    - Terrain breakdown
    - Energy component breakdown
    """
    
    total_distance: float = 0.0  # meters
    total_time: float = 0.0      # seconds
    total_energy: float = 0.0    # Joules
    
    # Velocity stats
    velocities: List[float] = field(default_factory=list)
    
    # Per-terrain stats
    terrain_distance: Dict[str, float] = field(default_factory=dict)
    terrain_time: Dict[str, float] = field(default_factory=dict)
    terrain_energy: Dict[str, float] = field(default_factory=dict)
    
    # Energy breakdown
    energy_breakdown: Dict[str, float] = field(default_factory=lambda: {
        'slope': 0.0,
        'friction': 0.0,
        'aero': 0.0,
        'turn': 0.0,
        'accel': 0.0,
        'regen': 0.0,
        'terramech': 0.0,
        'risk': 0.0,
        'uncertainty': 0.0
    })
    
    # Segment count
    num_segments: int = 0
    
    def add_segment(self, distance: float, time: float, energy: float,
                    velocity: float, breakdown: Dict[str, float],
                    terrain_type: int):
        """Add a segment's metrics"""
        self.total_distance += distance
        self.total_time += time
        self.total_energy += energy
        self.velocities.append(velocity)
        self.num_segments += 1
        
        # Terrain breakdown
        terrain_name = TerrainType(terrain_type).name_lower
        self.terrain_distance[terrain_name] = self.terrain_distance.get(terrain_name, 0.0) + distance
        self.terrain_time[terrain_name] = self.terrain_time.get(terrain_name, 0.0) + time
        self.terrain_energy[terrain_name] = self.terrain_energy.get(terrain_name, 0.0) + energy
        
        # Energy breakdown
        for key, value in breakdown.items():
            if key in self.energy_breakdown:
                self.energy_breakdown[key] += value
    
    @property
    def avg_velocity(self) -> float:
        """Average velocity (m/s)"""
        return np.mean(self.velocities) if self.velocities else 0.0
    
    @property
    def max_velocity(self) -> float:
        """Maximum velocity (m/s)"""
        return max(self.velocities) if self.velocities else 0.0
    
    @property
    def min_velocity(self) -> float:
        """Minimum velocity (m/s)"""
        return min(self.velocities) if self.velocities else 0.0
    
    @property
    def energy_efficiency(self) -> float:
        """Energy efficiency (J/m)"""
        if self.total_distance > 0:
            return self.total_energy / self.total_distance
        return float('inf')
    
    @property
    def total_distance_km(self) -> float:
        return self.total_distance / 1000.0
    
    @property
    def total_time_min(self) -> float:
        return self.total_time / 60.0
    
    @property
    def total_energy_kJ(self) -> float:
        return self.total_energy / 1000.0
    
    @property
    def avg_speed_kmh(self) -> float:
        return self.avg_velocity * 3.6
    
    @property
    def max_speed_kmh(self) -> float:
        return self.max_velocity * 3.6
    
    def to_dict(self) -> Dict:
        """Convert to dictionary for serialization"""
        return {
            'total_distance_m': self.total_distance,
            'total_distance_km': self.total_distance_km,
            'total_time_s': self.total_time,
            'total_time_min': self.total_time_min,
            'total_energy_J': self.total_energy,
            'total_energy_kJ': self.total_energy_kJ,
            'avg_velocity_ms': self.avg_velocity,
            'max_velocity_ms': self.max_velocity,
            'avg_speed_kmh': self.avg_speed_kmh,
            'max_speed_kmh': self.max_speed_kmh,
            'energy_efficiency_Jm': self.energy_efficiency,
            'num_segments': self.num_segments,
            'terrain_distance': self.terrain_distance,
            'terrain_time': self.terrain_time,
            'terrain_energy': self.terrain_energy,
            'energy_breakdown': self.energy_breakdown
        }


@dataclass
class BacktrackingStats:
    """Statistics about path backtracking (moving away from goal)"""
    backtracking_distance_m: float = 0.0
    total_distance_m: float = 0.0
    backtrack_ratio: float = 0.0
    min_dist_to_goal_m: float = float('inf')
    max_dist_to_goal_m: float = 0.0
    distance_progress_ratio: float = 0.0  # How much closer we got vs distance traveled


def compute_backtracking_stats(path: List[Tuple[int, int]], 
                               goal: Tuple[int, int],
                               cell_size: float = 2.0) -> BacktrackingStats:
    """
    Compute backtracking statistics for a path.
    
    Backtracking = distance traveled while moving away from goal.
    
    Args:
        path: List of (x, y) positions
        goal: Goal position
        cell_size: Cell size in meters
    
    Returns:
        BacktrackingStats object
    """
    if not path or len(path) < 2:
        return BacktrackingStats()
    
    gx, gy = goal
    
    def dist_to_goal(p):
        return math.hypot(p[0] - gx, p[1] - gy) * cell_size
    
    total_m = 0.0
    back_m = 0.0
    prev = path[0]
    prev_d = dist_to_goal(prev)
    min_d = prev_d
    max_d = prev_d
    start_d = prev_d
    
    for cur in path[1:]:
        seg = math.hypot(cur[0] - prev[0], cur[1] - prev[1]) * cell_size
        total_m += seg
        cur_d = dist_to_goal(cur)
        
        if cur_d > prev_d + 1e-9:  # Moving away from goal
            back_m += seg
        
        min_d = min(min_d, cur_d)
        max_d = max(max_d, cur_d)
        prev, prev_d = cur, cur_d
    
    ratio = back_m / total_m if total_m > 1e-9 else 0.0
    
    # Progress ratio: how much closer we got vs total distance
    end_d = dist_to_goal(path[-1])
    progress = start_d - end_d
    progress_ratio = progress / total_m if total_m > 1e-9 else 0.0
    
    return BacktrackingStats(
        backtracking_distance_m=back_m,
        total_distance_m=total_m,
        backtrack_ratio=ratio,
        min_dist_to_goal_m=min_d,
        max_dist_to_goal_m=max_d,
        distance_progress_ratio=progress_ratio
    )


class RunStatus:
    """Enumeration of run status types"""
    SUCCESS = 'success'
    COLLISION = 'collision'
    DEAD_END = 'dead_end'
    TIMEOUT = 'timeout'
    BACKTRACKING = 'backtracking'
    UNKNOWN = 'unknown'


@dataclass
class RunResult:
    """Complete result of a planning run"""
    status: str
    failure_type: Optional[str] = None
    
    # Path data
    path: List[Tuple[int, int]] = field(default_factory=list)
    modes: List[str] = field(default_factory=list)
    
    # Metrics
    metrics: Optional[PathMetrics] = None
    backtracking: Optional[BacktrackingStats] = None
    
    # Timing
    total_time_s: float = 0.0
    replans: int = 0
    
    # Recovery stats
    recovery_attempts: int = 0
    recovery_successes: int = 0
    
    # Additional info
    info: Dict = field(default_factory=dict)
    
    @property
    def is_success(self) -> bool:
        return self.status == RunStatus.SUCCESS
    
    def to_dict(self) -> Dict:
        return {
            'status': self.status,
            'failure_type': self.failure_type,
            'path_length': len(self.path),
            'metrics': self.metrics.to_dict() if self.metrics else None,
            'backtracking': {
                'backtrack_ratio': self.backtracking.backtrack_ratio,
                'backtracking_distance_m': self.backtracking.backtracking_distance_m,
                'min_dist_to_goal_m': self.backtracking.min_dist_to_goal_m,
            } if self.backtracking else None,
            'total_time_s': self.total_time_s,
            'replans': self.replans,
            'recovery_attempts': self.recovery_attempts,
            'recovery_successes': self.recovery_successes,
            'info': self.info
        }


class RunClassifier:
    """
    Classifies run outcomes based on path analysis.
    
    Categories:
    - success: Reached goal
    - collision: Path entered invalid cell
    - dead_end: No feasible continuation found
    - timeout: Exceeded time/iteration budget
    - backtracking: Excessive backtracking detected
    """
    
    def __init__(self, backtrack_threshold: float = 0.25):
        self.backtrack_threshold = backtrack_threshold
    
    def classify(self, 
                 path: List[Tuple[int, int]],
                 goal: Tuple[int, int],
                 env,
                 dead_end: bool = False,
                 timeout: bool = False,
                 backtracking_stats: Optional[BacktrackingStats] = None) -> Tuple[str, Optional[str]]:
        """
        Classify run outcome.
        
        Args:
            path: Executed path
            goal: Goal position
            env: Environment for validity checking
            dead_end: Dead-end flag from planner
            timeout: Timeout flag from planner
            backtracking_stats: Pre-computed backtracking stats
        
        Returns:
            Tuple of (status, failure_type)
        """
        # Check success first
        if path and tuple(path[-1]) == tuple(goal):
            return RunStatus.SUCCESS, None
        
        # Check collision
        if path:
            for p in path:
                if not env.is_valid(int(p[0]), int(p[1])):
                    return RunStatus.COLLISION, 'collision'
        
        # Check timeout
        if timeout:
            return RunStatus.TIMEOUT, 'timeout'
        
        # Check dead end
        if dead_end:
            return RunStatus.DEAD_END, 'dead_end'
        
        # Check excessive backtracking
        if backtracking_stats is not None:
            if backtracking_stats.backtrack_ratio >= self.backtrack_threshold:
                return RunStatus.BACKTRACKING, 'backtracking'
        
        # Default to dead end if not at goal
        return RunStatus.DEAD_END, 'dead_end'
