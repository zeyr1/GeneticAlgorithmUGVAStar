"""
Recovery Manager Module
=======================

Multi-strategy recovery system for dead-end situations.
Implements various escape strategies when local planning fails.
"""

import math
import numpy as np
from typing import Tuple, List, Optional, Dict, Set
from dataclasses import dataclass
from enum import Enum

from ..config import Config, RecoveryConfig
from ..environment import Environment, create_local_environment, FoVBounds
from ..energy import EnergyModel
from ..planning.astar import AStarPlanner, choose_local_goal


class RecoveryStrategy(Enum):
    """Available recovery strategies"""
    EXPAND_FOV = 'expand_fov'
    BACKTRACK = 'backtrack'
    RANDOM_ESCAPE = 'random_escape'
    GLOBAL_REPLAN = 'global_replan'
    WALL_FOLLOW = 'wall_follow'


@dataclass
class RecoveryResult:
    """Result of a recovery attempt"""
    success: bool
    strategy_used: str
    new_position: Optional[Tuple[int, int]] = None
    escape_path: Optional[List[Tuple[int, int]]] = None
    message: str = ''


class RecoveryManager:
    """
    Multi-strategy recovery manager for dead-end situations.
    
    Strategies (tried in priority order):
    1. expand_fov: Expand FoV radius and replan
    2. backtrack: Move back along executed path
    3. random_escape: Random walk to escape local minimum
    4. global_replan: Use coarse global planning to find new route
    5. wall_follow: Follow walls to escape corridors
    
    The manager tries strategies in order until one succeeds
    or all strategies are exhausted.
    """
    
    def __init__(self, 
                 config: Config,
                 recovery_config: Optional[RecoveryConfig] = None,
                 seed: int = 42):
        """
        Initialize recovery manager.
        
        Args:
            config: Main configuration
            recovery_config: Recovery-specific config (default from main config)
            seed: Random seed for stochastic strategies
        """
        self.config = config
        self.recovery_config = recovery_config or config.recovery
        self.rng = np.random.default_rng(seed)
        
        # Strategy implementations
        self._strategies = {
            RecoveryStrategy.EXPAND_FOV: self._expand_fov_strategy,
            RecoveryStrategy.BACKTRACK: self._backtrack_strategy,
            RecoveryStrategy.RANDOM_ESCAPE: self._random_escape_strategy,
            RecoveryStrategy.GLOBAL_REPLAN: self._global_replan_strategy,
            RecoveryStrategy.WALL_FOLLOW: self._wall_follow_strategy,
        }
        
        # Statistics
        self.attempts = 0
        self.successes = 0
        self.strategy_stats = {s.value: {'attempts': 0, 'successes': 0} 
                              for s in RecoveryStrategy}
    
    def attempt_recovery(self, 
                        controller_state,
                        env: Environment,
                        goal: Tuple[int, int],
                        mode: str = 'energy') -> bool:
        """
        Attempt recovery from dead-end situation.
        
        Args:
            controller_state: Current controller state
            env: Global environment
            goal: Goal position
            mode: Planning mode
        
        Returns:
            True if recovery succeeded, False otherwise
        """
        if not self.recovery_config.enabled:
            return False
        
        self.attempts += 1
        
        # Get strategy order from config
        strategy_order = [RecoveryStrategy(s) for s in self.recovery_config.strategies
                         if s in [rs.value for rs in RecoveryStrategy]]
        
        # Try each strategy in order
        for strategy in strategy_order:
            self.strategy_stats[strategy.value]['attempts'] += 1
            
            try:
                result = self._strategies[strategy](
                    controller_state, env, goal, mode
                )
                
                if result.success:
                    self.successes += 1
                    self.strategy_stats[strategy.value]['successes'] += 1
                    
                    # Apply recovery result to state
                    self._apply_recovery(controller_state, result)
                    
                    return True
                    
            except Exception as e:
                if self.config.verbose:
                    print(f"Recovery strategy {strategy.value} failed: {e}")
                continue
        
        return False
    
    def _apply_recovery(self, state, result: RecoveryResult):
        """Apply recovery result to controller state"""
        if result.new_position is not None:
            state.current_position = result.new_position
        
        if result.escape_path is not None:
            # Add escape path to executed path
            for pos in result.escape_path[1:]:  # Skip first (current position)
                state.executed_path.append(tuple(pos))
                state.executed_modes.append('escape')
            
            if result.escape_path:
                state.current_position = tuple(result.escape_path[-1])
    
    # ==================== Strategy Implementations ====================
    
    def _expand_fov_strategy(self, state, env: Environment,
                             goal: Tuple[int, int], mode: str) -> RecoveryResult:
        """
        Expand FoV and try to find alternative path.
        
        Double the FoV radius and attempt to plan around obstacles.
        """
        # Compute expanded radius
        current_radius = state.current_fov_radius
        expanded_radius = min(
            self.config.fov.max_radius_cells,
            current_radius * 2
        )
        
        if expanded_radius <= current_radius:
            return RecoveryResult(False, 'expand_fov', message='Already at max FoV')
        
        # Create expanded local environment
        local_env = create_local_environment(
            global_env=env,
            pos=state.current_position,
            radius=expanded_radius,
            unknown_config=self.config.unknown,
            visited_counts=state.visited_counts,
            failed_cells=state.failed_cells
        )
        
        bounds = local_env.bounds
        local_goal = choose_local_goal(state.current_position, goal, bounds)
        
        # Plan with expanded view and distance mode (more aggressive)
        planner = AStarPlanner(
            env=local_env,
            energy_model=EnergyModel(local_env, self.config),
            config=self.config,
            mode='distance',  # Use distance for escape
            epsilon=1.5,
            max_expansions=int(bounds.area * 15),
            cone_angle=math.pi * 2.0  # Full 360 degrees
        )
        
        path = planner.plan(
            state.current_position,
            local_goal,
            local_window=bounds.as_dict()
        )
        
        if path and len(path) >= 2:
            # Update state's FoV radius
            state.current_fov_radius = expanded_radius
            return RecoveryResult(
                True, 'expand_fov',
                new_position=tuple(path[1]),  # Move one step
                escape_path=path[:min(5, len(path))],  # Execute first few steps
                message=f'Expanded FoV to {expanded_radius}'
            )
        
        return RecoveryResult(False, 'expand_fov', message='No path with expanded FoV')
    
    def _backtrack_strategy(self, state, env: Environment,
                           goal: Tuple[int, int], mode: str) -> RecoveryResult:
        """
        Backtrack along executed path to escape local minimum.
        
        Mark current position as failed and move back to previous position.
        """
        if len(state.executed_path) < 3:
            return RecoveryResult(False, 'backtrack', message='Not enough history')
        
        # Mark current and nearby cells as failed
        cx, cy = state.current_position
        state.failed_cells.add((cx, cy))
        for dx in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                if env.terrain[cx+dx, cy+dy] == 4:  # WALL
                    continue
                # Only mark if high visit count
                if state.visited_counts.get((cx+dx, cy+dy), 0) > 2:
                    state.failed_cells.add((cx+dx, cy+dy))
        
        # Find backtrack target - go back to where we weren't stuck
        backtrack_steps = min(
            self.recovery_config.max_backtrack_steps,
            len(state.executed_path) - 1
        )
        
        # Look for a position with low visit count
        for i in range(1, backtrack_steps + 1):
            idx = len(state.executed_path) - i - 1
            if idx < 0:
                break
            
            candidate = state.executed_path[idx]
            visit_count = state.visited_counts.get(candidate, 0)
            
            if visit_count <= 2 and candidate not in state.failed_cells:
                return RecoveryResult(
                    True, 'backtrack',
                    new_position=candidate,
                    message=f'Backtracked {i} steps'
                )
        
        # Fallback: just go back a few steps
        idx = max(0, len(state.executed_path) - 5)
        return RecoveryResult(
            True, 'backtrack',
            new_position=state.executed_path[idx],
            message='Backtracked to earlier position'
        )
    
    def _random_escape_strategy(self, state, env: Environment,
                                goal: Tuple[int, int], mode: str) -> RecoveryResult:
        """
        Random walk to escape local minimum.
        
        Take random valid steps to escape the current region.
        """
        current = state.current_position
        escape_path = [current]
        
        max_steps = 20
        
        for _ in range(max_steps):
            # Get valid neighbors
            neighbors = []
            cx, cy = escape_path[-1]
            
            for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
                nx, ny = cx + dx, cy + dy
                if env.is_valid(nx, ny):
                    # Prefer cells not in failed set and with low visit count
                    if (nx, ny) not in state.failed_cells:
                        neighbors.append((nx, ny))
            
            if not neighbors:
                break
            
            # Bias toward goal direction
            gx, gy = goal
            scored = []
            for (nx, ny) in neighbors:
                # Score = distance reduction + randomness
                old_dist = math.hypot(cx - gx, cy - gy)
                new_dist = math.hypot(nx - gx, ny - gy)
                progress = old_dist - new_dist
                
                visit_penalty = state.visited_counts.get((nx, ny), 0) * 0.5
                random_bonus = self.rng.random() * 2
                
                score = progress + random_bonus - visit_penalty
                scored.append((score, (nx, ny)))
            
            # Select probabilistically based on scores
            scored.sort(reverse=True)
            
            # Take best with some probability, else random
            if self.rng.random() < 0.7:
                next_pos = scored[0][1]
            else:
                next_pos = self.rng.choice([s[1] for s in scored])
            
            escape_path.append(next_pos)
            
            # Check if we've escaped (different region, closer to goal)
            if len(escape_path) >= 5:
                gx, gy = goal
                start_dist = math.hypot(current[0] - gx, current[1] - gy)
                curr_dist = math.hypot(next_pos[0] - gx, next_pos[1] - gy)
                
                if curr_dist < start_dist * 0.9:  # Made progress
                    return RecoveryResult(
                        True, 'random_escape',
                        escape_path=escape_path,
                        message=f'Random escape: {len(escape_path)} steps'
                    )
        
        if len(escape_path) > 1:
            return RecoveryResult(
                True, 'random_escape',
                escape_path=escape_path,
                message=f'Partial random escape: {len(escape_path)} steps'
            )
        
        return RecoveryResult(False, 'random_escape', message='Random escape failed')
    
    def _global_replan_strategy(self, state, env: Environment,
                                goal: Tuple[int, int], mode: str) -> RecoveryResult:
        """
        Use coarse global planning to find alternative route.
        
        Plans on a downsampled grid to find a global path,
        then uses first waypoint as intermediate goal.
        """
        resolution = self.recovery_config.global_replan_resolution
        
        # Create downsampled environment for global planning
        # (We'll use full environment but with very high epsilon)
        
        planner = AStarPlanner(
            env=env,
            energy_model=EnergyModel(env, self.config),
            config=self.config,
            mode='distance',
            epsilon=3.0,  # Very inflated heuristic for speed
            max_expansions=100000,
            cone_angle=math.pi * 2.0
        )
        
        global_path = planner.plan(state.current_position, goal)
        
        if global_path and len(global_path) >= 2:
            # Find first waypoint that's outside failed region
            for i, waypoint in enumerate(global_path[1:], 1):
                if waypoint not in state.failed_cells:
                    # Plan locally to this waypoint
                    local_radius = min(
                        self.config.fov.max_radius_cells,
                        int(math.hypot(waypoint[0] - state.current_position[0],
                                      waypoint[1] - state.current_position[1])) + 10
                    )
                    
                    local_env = create_local_environment(
                        global_env=env,
                        pos=state.current_position,
                        radius=local_radius,
                        visited_counts=state.visited_counts,
                        failed_cells=state.failed_cells
                    )
                    
                    local_planner = AStarPlanner(
                        env=local_env,
                        energy_model=EnergyModel(local_env, self.config),
                        config=self.config,
                        mode='distance',
                        epsilon=1.5,
                        cone_angle=math.pi * 2.0
                    )
                    
                    local_path = local_planner.plan(
                        state.current_position,
                        waypoint,
                        local_window=local_env.bounds.as_dict()
                    )
                    
                    if local_path and len(local_path) >= 2:
                        # Update FoV radius for the waypoint distance
                        state.current_fov_radius = local_radius
                        
                        return RecoveryResult(
                            True, 'global_replan',
                            escape_path=local_path[:min(10, len(local_path))],
                            message=f'Global replan via waypoint {i}'
                        )
        
        return RecoveryResult(False, 'global_replan', message='Global replan failed')
    
    def _wall_follow_strategy(self, state, env: Environment,
                              goal: Tuple[int, int], mode: str) -> RecoveryResult:
        """
        Follow walls to escape corridors.
        
        When stuck near walls, follow them until finding an opening.
        """
        current = state.current_position
        cx, cy = current
        
        # Check if we're near a wall
        wall_neighbors = []
        for dx in [-2, -1, 0, 1, 2]:
            for dy in [-2, -1, 0, 1, 2]:
                if env.terrain[cx+dx, cy+dy] == 4:  # WALL
                    wall_neighbors.append((cx+dx, cy+dy))
        
        if not wall_neighbors:
            return RecoveryResult(False, 'wall_follow', message='Not near walls')
        
        # Find direction along wall
        escape_path = [current]
        
        # Determine goal direction
        gx, gy = goal
        goal_angle = math.atan2(gy - cy, gx - cx)
        
        # Try both clockwise and counter-clockwise
        for direction in [1, -1]:
            path = [current]
            
            for step in range(30):
                px, py = path[-1]
                
                # Find next step along wall
                best_next = None
                best_score = -float('inf')
                
                for dx, dy in [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]:
                    nx, ny = px + dx, py + dy
                    
                    if not env.is_valid(nx, ny):
                        continue
                    if (nx, ny) in state.failed_cells:
                        continue
                    
                    # Score based on: staying near wall + progress toward goal
                    wall_dist = min(
                        math.hypot(nx - wx, ny - wy) 
                        for wx, wy in wall_neighbors
                    ) if wall_neighbors else 10
                    
                    goal_dist = math.hypot(nx - gx, ny - gy)
                    
                    # Prefer staying near wall while making progress
                    score = -goal_dist - abs(wall_dist - 2) * 5
                    score += self.rng.random()  # Tie-breaker
                    
                    if score > best_score:
                        best_score = score
                        best_next = (nx, ny)
                
                if best_next is None:
                    break
                
                path.append(best_next)
                
                # Check if we've found an opening (away from walls, toward goal)
                px, py = best_next
                goal_progress = math.hypot(cx - gx, cy - gy) - math.hypot(px - gx, py - gy)
                
                if goal_progress > 5 * self.config.map.cell_size:
                    return RecoveryResult(
                        True, 'wall_follow',
                        escape_path=path,
                        message=f'Wall follow: {len(path)} steps'
                    )
            
            # Try this path if it made some progress
            if len(path) > 3:
                escape_path = path
                break
        
        if len(escape_path) > 1:
            return RecoveryResult(
                True, 'wall_follow',
                escape_path=escape_path,
                message=f'Partial wall follow: {len(escape_path)} steps'
            )
        
        return RecoveryResult(False, 'wall_follow', message='Wall follow failed')
    
    # ==================== Statistics ====================
    
    def get_stats(self) -> Dict:
        """Get recovery statistics"""
        return {
            'total_attempts': self.attempts,
            'total_successes': self.successes,
            'success_rate': self.successes / max(1, self.attempts),
            'strategy_stats': self.strategy_stats
        }
    
    def reset_stats(self):
        """Reset statistics"""
        self.attempts = 0
        self.successes = 0
        for s in self.strategy_stats:
            self.strategy_stats[s] = {'attempts': 0, 'successes': 0}
