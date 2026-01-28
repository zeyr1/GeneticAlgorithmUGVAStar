"""
A* Planner Module
=================

A* pathfinding with mode-specific costs and kinodynamic awareness.
"""

import heapq
import math
import numpy as np
from typing import Tuple, List, Optional, Dict, Set
from dataclasses import dataclass

from ..config import Config
from ..energy import EnergyModel


@dataclass
class PlannerStats:
    """Statistics from a planning run"""
    iterations: int = 0
    nodes_expanded: int = 0
    path_length: int = 0
    success: bool = False
    reason: str = ''


class AStarPlanner:
    """
    A* planner with multiple optimization modes.
    
    Modes:
    - 'energy': Minimize energy consumption (terrain + slope + risk)
    - 'time': Minimize travel time (distance / speed)
    - 'distance': Minimize geometric distance
    
    Features:
    - Forward cone filtering to reduce backward exploration
    - Road preference bonuses
    - Visited cell penalties
    - Dynamic epsilon inflation for long searches
    - Configurable expansion limits
    """
    
    def __init__(self, 
                 env,
                 energy_model: EnergyModel,
                 config: Config,
                 mode: str = 'energy',
                 epsilon: float = 1.0,
                 max_expansions: Optional[int] = None,
                 cone_angle: float = math.pi * 1.5):
        """
        Initialize A* planner.
        
        Args:
            env: Environment (global or local)
            energy_model: Energy model for cost computation
            config: Configuration object
            mode: Planning mode ('energy', 'time', 'distance')
            epsilon: Heuristic inflation factor
            max_expansions: Maximum node expansions (None for unlimited)
            cone_angle: Forward cone angle for neighbor filtering (radians)
        """
        self.env = env
        self.energy_model = energy_model
        self.config = config
        self.mode = mode
        self.epsilon = epsilon
        self.max_expansions = max_expansions
        self.cone_angle = cone_angle
        
        # Road preference weights
        self.road_bonus = 0.20
        self.road_exit_penalty = 0.15
        
        # Direction bias
        self.direction_bias_weight = 0.15
        
        # Last planning stats
        self.last_stats: Optional[PlannerStats] = None
    
    def plan(self, 
             start: Tuple[int, int],
             goal: Tuple[int, int],
             local_window: Optional[Dict[str, int]] = None) -> List[Tuple[int, int]]:
        """
        Find path from start to goal.
        
        Args:
            start: Start position (x, y)
            goal: Goal position (x, y)
            local_window: Optional FoV window bounds {'xmin', 'xmax', 'ymin', 'ymax'}
        
        Returns:
            Path as list of (x, y) positions, or empty list if no path found
        """
        stats = PlannerStats()
        
        # Validate start and goal
        if not self.env.is_valid(start[0], start[1]):
            stats.reason = 'invalid_start'
            self.last_stats = stats
            return []
        
        if not self.env.is_valid(goal[0], goal[1]):
            stats.reason = 'invalid_goal'
            self.last_stats = stats
            return []
        
        # Setup window bounds
        if local_window is not None:
            xmin, xmax = int(local_window['xmin']), int(local_window['xmax'])
            ymin, ymax = int(local_window['ymin']), int(local_window['ymax'])
            
            def in_window(p):
                return xmin <= p[0] <= xmax and ymin <= p[1] <= ymax
            
            if not in_window(start) or not in_window(goal):
                stats.reason = 'start_or_goal_outside_window'
                self.last_stats = stats
                return []
            
            # Set max iterations based on window size
            window_area = (xmax - xmin + 1) * (ymax - ymin + 1)
            max_iters = min(500000, int(80 * window_area))
        else:
            def in_window(p):
                return True
            max_iters = 3000000
        
        # Override with explicit max_expansions if set
        if self.max_expansions is not None:
            max_iters = min(max_iters, self.max_expansions)
        
        # A* initialization
        open_set = []
        heapq.heappush(open_set, (0.0, start))
        came_from = {}
        g_score = {start: 0.0}
        closed = set()
        
        iterations = 0
        
        while open_set and iterations < max_iters:
            iterations += 1
            
            current_f, current = heapq.heappop(open_set)
            
            if current in closed:
                continue
            closed.add(current)
            stats.nodes_expanded += 1
            
            # Goal check
            if current == goal:
                path = self._reconstruct_path(came_from, current)
                stats.iterations = iterations
                stats.path_length = len(path)
                stats.success = True
                stats.reason = 'success'
                self.last_stats = stats
                return path
            
            # Expand neighbors
            for neighbor in self._get_neighbors(current):
                if not in_window(neighbor):
                    continue
                if not self.env.is_valid(neighbor[0], neighbor[1]):
                    continue
                
                # Forward cone filter
                if not self._passes_cone_filter(current, neighbor, goal):
                    continue
                
                # Compute edge cost
                edge_cost = self._edge_cost(current, neighbor)
                
                # Road preference
                edge_cost = self._apply_road_preference(current, neighbor, edge_cost)
                
                # Visited penalty (if local environment)
                if hasattr(self.env, 'get_visited_penalty'):
                    edge_cost += self.env.get_visited_penalty(neighbor[0], neighbor[1])
                
                tentative_g = g_score[current] + edge_cost
                
                if neighbor not in g_score or tentative_g < g_score[neighbor]:
                    came_from[neighbor] = current
                    g_score[neighbor] = tentative_g
                    
                    # Dynamic epsilon inflation for long searches
                    inflation = 1.0 + min(3.0, iterations / 50000.0)
                    eff_epsilon = self.epsilon * inflation
                    
                    # Heuristic with direction bias
                    h = self._heuristic(neighbor, goal)
                    dir_penalty = self._direction_penalty(current, neighbor, goal)
                    
                    f_score = tentative_g + eff_epsilon * h + dir_penalty
                    heapq.heappush(open_set, (f_score, neighbor))
        
        # No path found
        stats.iterations = iterations
        stats.reason = 'no_path_found' if iterations < max_iters else 'max_iterations'
        self.last_stats = stats
        return []
    
    def _get_neighbors(self, pos: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Get 8-connected neighbors"""
        x, y = pos
        neighbors = []
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if dx == 0 and dy == 0:
                    continue
                neighbors.append((x + dx, y + dy))
        return neighbors
    
    def _edge_cost(self, node1: Tuple[int, int], node2: Tuple[int, int]) -> float:
        """Compute edge cost based on planning mode"""
        distance = math.hypot(node2[0] - node1[0], node2[1] - node1[1])
        
        if self.mode == 'distance':
            return distance
        
        if self.mode == 'time':
            max_speed = self.env.get_max_velocity(node2[0], node2[1])
            return distance / max(max_speed, 0.1)
        
        # Energy mode (default)
        friction = self.env.get_friction_coeff(node2[0], node2[1])
        
        # Elevation change
        delta_h = self.env.elevation[node2[0], node2[1]] - self.env.elevation[node1[0], node1[1]]
        
        # Base cost
        base_cost = distance * friction * 100
        
        # Elevation cost
        if delta_h > 0:
            elevation_cost = delta_h * 150
        else:
            elevation_cost = delta_h * 50
        
        # Risk penalty
        risk_penalty = self.env.risk_map[node2[0], node2[1]] * 50
        
        return base_cost + elevation_cost + risk_penalty
    
    def _heuristic(self, pos: Tuple[int, int], goal: Tuple[int, int]) -> float:
        """Admissible heuristic (Euclidean distance)"""
        return math.hypot(pos[0] - goal[0], pos[1] - goal[1])
    
    def _passes_cone_filter(self, current: Tuple[int, int], 
                            neighbor: Tuple[int, int],
                            goal: Tuple[int, int]) -> bool:
        """Check if neighbor is within forward cone toward goal"""
        dx = neighbor[0] - current[0]
        dy = neighbor[1] - current[1]
        gx = goal[0] - current[0]
        gy = goal[1] - current[1]
        
        # Normalize
        d_len = math.sqrt(dx*dx + dy*dy)
        g_len = math.sqrt(gx*gx + gy*gy)
        
        if d_len < 1e-9 or g_len < 1e-9:
            return True
        
        # Cosine of angle between directions
        cos_angle = (dx * gx + dy * gy) / (d_len * g_len)
        
        # Check against cone threshold
        cone_threshold = math.cos(self.cone_angle / 2.0)
        return cos_angle >= cone_threshold
    
    def _direction_penalty(self, current: Tuple[int, int],
                          neighbor: Tuple[int, int],
                          goal: Tuple[int, int]) -> float:
        """Compute direction alignment penalty"""
        dx = neighbor[0] - current[0]
        dy = neighbor[1] - current[1]
        gx = goal[0] - current[0]
        gy = goal[1] - current[1]
        
        d_len = math.sqrt(dx*dx + dy*dy)
        g_len = math.sqrt(gx*gx + gy*gy)
        
        if d_len < 1e-9 or g_len < 1e-9:
            return 0.0
        
        # Alignment in [-1, 1], 1 = perfectly aligned
        alignment = (dx * gx + dy * gy) / (d_len * g_len)
        
        # Penalty for misalignment
        return self.direction_bias_weight * (1.0 - alignment)
    
    def _apply_road_preference(self, current: Tuple[int, int],
                               neighbor: Tuple[int, int],
                               cost: float) -> float:
        """Apply road preference bonus/penalty"""
        from ..terrain import TerrainType
        
        try:
            neighbor_terrain = self.env.terrain[neighbor[0], neighbor[1]]
            
            # Bonus for staying on roads
            if neighbor_terrain == TerrainType.ASPHALT:
                cost *= (1.0 - self.road_bonus)
            
            # Penalty for leaving roads
            current_terrain = self.env.terrain[current[0], current[1]]
            if current_terrain == TerrainType.ASPHALT and neighbor_terrain != TerrainType.ASPHALT:
                cost += self.road_exit_penalty
        except (IndexError, AttributeError):
            pass
        
        return cost
    
    def _reconstruct_path(self, came_from: Dict, current: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Reconstruct path from came_from dict"""
        path = [current]
        while current in came_from:
            current = came_from[current]
            path.append(current)
        path.reverse()
        return path


def choose_local_goal(current: Tuple[int, int],
                      global_goal: Tuple[int, int],
                      bounds) -> Tuple[int, int]:
    """
    Choose local goal within FoV bounds.
    
    If global goal is inside bounds, return it.
    Otherwise, find intersection of ray toward goal with bounds.
    
    Args:
        current: Current position
        global_goal: Global goal position
        bounds: FoVBounds object or dict with xmin, xmax, ymin, ymax
    
    Returns:
        Local goal position
    """
    # Handle both FoVBounds object and dict
    if hasattr(bounds, 'xmin'):
        xmin, xmax = bounds.xmin, bounds.xmax
        ymin, ymax = bounds.ymin, bounds.ymax
    else:
        xmin, xmax = bounds['xmin'], bounds['xmax']
        ymin, ymax = bounds['ymin'], bounds['ymax']
    
    gx, gy = global_goal
    cx, cy = current
    
    # Check if goal is in bounds
    if xmin <= gx <= xmax and ymin <= gy <= ymax:
        return global_goal
    
    # Compute direction to goal
    dx = gx - cx
    dy = gy - cy
    
    if dx == 0 and dy == 0:
        return current
    
    # Find intersection with bounds
    candidates = []
    
    if dx != 0:
        t1 = (xmin - cx) / dx
        t2 = (xmax - cx) / dx
        if t1 > 0:
            candidates.append(t1)
        if t2 > 0:
            candidates.append(t2)
    
    if dy != 0:
        t3 = (ymin - cy) / dy
        t4 = (ymax - cy) / dy
        if t3 > 0:
            candidates.append(t3)
        if t4 > 0:
            candidates.append(t4)
    
    # Find closest valid intersection
    best_t = None
    for t in candidates:
        x = cx + t * dx
        y = cy + t * dy
        
        # Check if intersection is on bounds
        if xmin - 0.5 <= x <= xmax + 0.5 and ymin - 0.5 <= y <= ymax + 0.5:
            if best_t is None or t < best_t:
                best_t = t
    
    if best_t is not None:
        x = int(round(cx + best_t * dx))
        y = int(round(cy + best_t * dy))
        # Clamp to bounds
        x = max(xmin, min(xmax, x))
        y = max(ymin, min(ymax, y))
        return (x, y)
    
    # Fallback: clamp goal to bounds
    return (
        max(xmin, min(xmax, gx)),
        max(ymin, min(ymax, gy))
    )
