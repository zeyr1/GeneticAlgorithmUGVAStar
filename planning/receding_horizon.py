"""
Receding Horizon Controller Module
===================================

Main planning loop with FoV-constrained receding horizon control.
Integrates recovery strategies for dead-end handling.
"""

import time
import math
from typing import Tuple, List, Dict, Optional, Set, Callable
from dataclasses import dataclass, field

from ..config import Config
from ..environment import Environment, LocalEnvironment, FoVBounds, create_local_environment
from ..energy import EnergyModel, VelocityProfiler
from ..metrics import PathMetrics, compute_backtracking_stats, RunClassifier, RunResult, RunStatus
from .astar import AStarPlanner, choose_local_goal


@dataclass
class ControllerState:
    """Internal state of the receding horizon controller"""
    current_position: Tuple[int, int]
    executed_path: List[Tuple[int, int]] = field(default_factory=list)
    executed_modes: List[str] = field(default_factory=list)
    
    # Memory
    visited_counts: Dict[Tuple[int, int], int] = field(default_factory=dict)
    failed_cells: Set[Tuple[int, int]] = field(default_factory=set)
    
    # Adaptive parameters
    current_fov_radius: int = 25
    current_execute_steps: int = 12
    
    # Counters
    iteration: int = 0
    replans: int = 0
    recovery_attempts: int = 0
    recovery_successes: int = 0
    
    # Timing
    start_time: float = 0.0
    total_planning_time: float = 0.0
    
    # Flags
    dead_end: bool = False
    timeout: bool = False
    in_recovery: bool = False


class RecedingHorizonController:
    """
    Receding horizon controller for FoV-constrained navigation.
    
    Main loop:
    1. Extract local FoV window
    2. Create local environment with memory
    3. Plan to local goal using A*
    4. Optional: Refine with GA
    5. Execute first K steps
    6. If stuck: trigger recovery
    7. Repeat until goal reached or failure
    
    Features:
    - Adaptive FoV radius (expands when stuck)
    - Multi-strategy recovery system
    - Backtracking detection and handling
    - Real-time progress callbacks
    """
    
    def __init__(self, 
                 env: Environment,
                 config: Config,
                 recovery_manager=None,
                 adaptive_fov=None,
                 progress_callback: Optional[Callable] = None):
        """
        Initialize controller.
        
        Args:
            env: Global environment
            config: Configuration object
            recovery_manager: RecoveryManager instance (optional)
            adaptive_fov: AdaptiveFoV instance (optional)
            progress_callback: Callback for progress updates fn(state, info)
        """
        self.env = env
        self.config = config
        self.recovery_manager = recovery_manager
        self.adaptive_fov = adaptive_fov
        self.progress_callback = progress_callback
        
        # Create global energy model
        self.energy_model = EnergyModel(env, config)
        
        # Run classifier
        self.classifier = RunClassifier(config.recovery.backtrack_threshold)
        
        # State
        self.state: Optional[ControllerState] = None
    
    def run(self,
            start: Optional[Tuple[int, int]] = None,
            goal: Optional[Tuple[int, int]] = None,
            mode: str = 'energy',
            use_ga: bool = False,
            ga_solver=None,
            max_iterations: Optional[int] = None,
            max_time: Optional[float] = None) -> RunResult:
        """
        Run receding horizon planning from start to goal.
        
        Args:
            start: Start position (default: env.start)
            goal: Goal position (default: env.goal)
            mode: Planning mode ('energy', 'time', 'distance')
            use_ga: Whether to use GA refinement
            ga_solver: GA solver instance (required if use_ga=True)
            max_iterations: Maximum iterations (default from config)
            max_time: Maximum time in seconds (default from config)
        
        Returns:
            RunResult with path, metrics, and status
        """
        # Initialize
        start = start or self.env.start
        goal = goal or self.env.goal
        max_iterations = max_iterations or self.config.max_iterations
        max_time = max_time or self.config.max_total_seconds
        
        # Initialize state
        self.state = ControllerState(
            current_position=tuple(start),
            executed_path=[tuple(start)],
            current_fov_radius=self.config.fov.base_radius_cells,
            current_execute_steps=self.config.fov.base_execute_steps,
            start_time=time.perf_counter()
        )
        
        # Initialize adaptive FoV if provided
        if self.adaptive_fov is not None:
            self.adaptive_fov.reset()
            self.state.current_fov_radius = self.adaptive_fov.current_radius
        
        failures = []
        
        # Main loop
        while not self._is_goal_reached(goal):
            # Check termination conditions
            if self.state.iteration >= max_iterations:
                self.state.timeout = True
                failures.append({'iter': self.state.iteration, 'type': 'max_iterations'})
                break
            
            elapsed = time.perf_counter() - self.state.start_time
            if elapsed > max_time:
                self.state.timeout = True
                failures.append({'iter': self.state.iteration, 'type': 'timeout'})
                break
            
            self.state.iteration += 1
            
            # Update visit count
            self.state.visited_counts[self.state.current_position] = \
                self.state.visited_counts.get(self.state.current_position, 0) + 1
            
            # Check for oscillation/stuck condition
            if self._is_stuck():
                success = self._handle_stuck(goal, mode)
                if not success:
                    self.state.dead_end = True
                    failures.append({'iter': self.state.iteration, 'type': 'stuck_no_recovery'})
                    break
                continue
            
            # Extract local environment
            local_env = self._create_local_env()
            bounds = local_env.bounds
            
            # Choose local goal
            local_goal = choose_local_goal(
                self.state.current_position, goal, bounds
            )
            
            # Plan with A*
            planner = AStarPlanner(
                env=local_env,
                energy_model=EnergyModel(local_env, self.config),
                config=self.config,
                mode=mode,
                epsilon=1.3,
                max_expansions=self._compute_expansion_cap(bounds),
                cone_angle=math.pi * 1.2
            )
            
            path = planner.plan(
                self.state.current_position,
                local_goal,
                local_window=bounds.as_dict()
            )
            
            # Handle planning failure
            if not path or len(path) < 2:
                # Try recovery
                success = self._attempt_recovery(goal, mode, 'astar_failed')
                if not success:
                    self.state.dead_end = True
                    failures.append({'iter': self.state.iteration, 'type': 'dead_end'})
                    break
                continue
            
            self.state.replans += 1
            
            # Optional GA refinement
            if use_ga and ga_solver is not None:
                path, modes = self._refine_with_ga(
                    ga_solver, path, local_env, bounds, local_goal, mode
                )
            else:
                modes = [mode] * (len(path) - 1)
            
            # Execute steps
            executed = self._execute_steps(path, modes, goal)
            
            if not executed:
                # Execution failed (collision or stuck)
                success = self._attempt_recovery(goal, mode, 'execution_failed')
                if not success:
                    self.state.dead_end = True
                    failures.append({'iter': self.state.iteration, 'type': 'execution_failed'})
                    break
                continue
            
            # Update adaptive FoV on success
            if self.adaptive_fov is not None:
                bt = compute_backtracking_stats(
                    self.state.executed_path[-20:], goal, self.config.map.cell_size
                )
                self.adaptive_fov.update(success=True, backtrack_ratio=bt.backtrack_ratio)
                self.state.current_fov_radius = self.adaptive_fov.current_radius
            
            # Progress callback
            if self.progress_callback is not None:
                self._call_progress_callback()
        
        # Compute final metrics
        return self._build_result(goal, failures)
    
    def _is_goal_reached(self, goal: Tuple[int, int]) -> bool:
        """Check if current position is at goal"""
        return self.state.current_position == tuple(goal)
    
    def _is_stuck(self) -> bool:
        """Detect if robot is stuck (oscillating or high visit count)"""
        visit_count = self.state.visited_counts.get(self.state.current_position, 0)
        
        # High visit count indicates loop
        if visit_count > 4:
            return True
        
        # Check recent backtracking
        if len(self.state.executed_path) >= 10:
            recent = self.state.executed_path[-10:]
            unique = set(recent)
            if len(unique) < len(recent) * 0.5:  # More than 50% revisits
                return True
        
        return False
    
    def _handle_stuck(self, goal: Tuple[int, int], mode: str) -> bool:
        """Handle stuck condition"""
        self.state.in_recovery = True
        
        # Expand FoV
        if self.adaptive_fov is not None:
            self.adaptive_fov.update(success=False, backtrack_ratio=0.5)
            self.state.current_fov_radius = self.adaptive_fov.current_radius
        else:
            # Manual expansion
            self.state.current_fov_radius = min(
                self.config.fov.max_radius_cells,
                int(self.state.current_fov_radius * 1.5)
            )
        
        # Reduce execution steps
        self.state.current_execute_steps = max(
            self.config.fov.min_execute_steps,
            self.state.current_execute_steps // 2
        )
        
        # Use recovery manager if available
        if self.recovery_manager is not None:
            success = self.recovery_manager.attempt_recovery(
                self.state, self.env, goal, mode
            )
            self.state.recovery_attempts += 1
            if success:
                self.state.recovery_successes += 1
                self.state.in_recovery = False
                return True
        
        # Simple recovery: backtrack
        if len(self.state.executed_path) > 5:
            # Mark current as failed
            self.state.failed_cells.add(self.state.current_position)
            
            # Backtrack a few steps
            backtrack_steps = min(5, len(self.state.executed_path) - 1)
            self.state.current_position = self.state.executed_path[-(backtrack_steps + 1)]
            
            self.state.in_recovery = False
            return True
        
        self.state.in_recovery = False
        return False
    
    def _attempt_recovery(self, goal: Tuple[int, int], mode: str, reason: str) -> bool:
        """Attempt recovery from failure"""
        self.state.recovery_attempts += 1
        
        # Mark current cell as problematic
        self.state.failed_cells.add(self.state.current_position)
        
        # Use recovery manager
        if self.recovery_manager is not None:
            success = self.recovery_manager.attempt_recovery(
                self.state, self.env, goal, mode
            )
            if success:
                self.state.recovery_successes += 1
                return True
        
        # Expand FoV and try with larger view
        old_radius = self.state.current_fov_radius
        self.state.current_fov_radius = min(
            self.config.fov.max_radius_cells,
            int(self.state.current_fov_radius * 2)
        )
        
        if self.state.current_fov_radius > old_radius:
            # Try planning with expanded FoV
            local_env = self._create_local_env()
            bounds = local_env.bounds
            local_goal = choose_local_goal(self.state.current_position, goal, bounds)
            
            # Use distance mode for escape
            planner = AStarPlanner(
                env=local_env,
                energy_model=EnergyModel(local_env, self.config),
                config=self.config,
                mode='distance',
                epsilon=1.5,
                max_expansions=self._compute_expansion_cap(bounds) * 2,
                cone_angle=math.pi * 2.0  # Full circle
            )
            
            path = planner.plan(
                self.state.current_position,
                local_goal,
                local_window=bounds.as_dict()
            )
            
            if path and len(path) >= 2:
                self.state.recovery_successes += 1
                return True
        
        return False
    
    def _create_local_env(self) -> LocalEnvironment:
        """Create local environment at current position"""
        return create_local_environment(
            global_env=self.env,
            pos=self.state.current_position,
            radius=self.state.current_fov_radius,
            unknown_config=self.config.unknown,
            visited_counts=self.state.visited_counts,
            failed_cells=self.state.failed_cells
        )
    
    def _compute_expansion_cap(self, bounds: FoVBounds) -> int:
        """Compute A* expansion cap based on window size"""
        area = bounds.area
        return int(max(10000, min(150000, area * 10)))
    
    def _refine_with_ga(self, ga_solver, seed_path: List, local_env: LocalEnvironment,
                        bounds: FoVBounds, local_goal: Tuple[int, int], 
                        mode: str) -> Tuple[List, List]:
        """Refine path with GA solver"""
        try:
            best_path, best_modes, _, _ = ga_solver.solve(
                start=self.state.current_position,
                goal=local_goal,
                seed_path=seed_path,
                window_bounds=(bounds.xmin, bounds.xmax, bounds.ymin, bounds.ymax),
                local_env=local_env
            )
            
            if best_path and len(best_path) >= 2:
                return best_path, best_modes
        except Exception as e:
            if self.config.verbose:
                print(f"GA refinement failed: {e}")
        
        # Fallback to seed path
        return seed_path, [mode] * (len(seed_path) - 1)
    
    def _execute_steps(self, path: List[Tuple[int, int]], 
                       modes: List[str],
                       goal: Tuple[int, int]) -> bool:
        """Execute steps from planned path"""
        if len(path) < 2:
            return False
        
        steps_to_execute = path[1:self.state.current_execute_steps + 1]
        
        for i, step in enumerate(steps_to_execute):
            step = tuple(step)
            
            # Validate step
            if not self.env.is_valid(step[0], step[1]):
                # Collision - mark as failed
                self.state.failed_cells.add(step)
                return False
            
            # Execute
            self.state.executed_path.append(step)
            if i < len(modes):
                self.state.executed_modes.append(modes[i])
            
            self.state.current_position = step
            
            # Check goal
            if step == tuple(goal):
                return True
        
        return True
    
    def _call_progress_callback(self):
        """Call progress callback with current state"""
        if self.progress_callback is None:
            return
        
        info = {
            'iteration': self.state.iteration,
            'position': self.state.current_position,
            'path_length': len(self.state.executed_path),
            'fov_radius': self.state.current_fov_radius,
            'in_recovery': self.state.in_recovery,
            'elapsed_time': time.perf_counter() - self.state.start_time
        }
        
        try:
            self.progress_callback(self.state, info)
        except Exception:
            pass
    
    def _build_result(self, goal: Tuple[int, int], failures: List) -> RunResult:
        """Build final result"""
        elapsed = time.perf_counter() - self.state.start_time
        
        # Compute backtracking stats
        backtracking = compute_backtracking_stats(
            self.state.executed_path, goal, self.config.map.cell_size
        )
        
        # Classify run
        status, failure_type = self.classifier.classify(
            path=self.state.executed_path,
            goal=goal,
            env=self.env,
            dead_end=self.state.dead_end,
            timeout=self.state.timeout,
            backtracking_stats=backtracking
        )
        
        # Compute path metrics
        metrics = self._compute_metrics()
        
        return RunResult(
            status=status,
            failure_type=failure_type,
            path=self.state.executed_path,
            modes=self.state.executed_modes,
            metrics=metrics,
            backtracking=backtracking,
            total_time_s=elapsed,
            replans=self.state.replans,
            recovery_attempts=self.state.recovery_attempts,
            recovery_successes=self.state.recovery_successes,
            info={
                'final_fov_radius': self.state.current_fov_radius,
                'iterations': self.state.iteration,
                'failures': failures
            }
        )
    
    def _compute_metrics(self) -> PathMetrics:
        """Compute path metrics for executed path"""
        metrics = PathMetrics()
        path = self.state.executed_path
        modes = self.state.executed_modes
        
        if len(path) < 2:
            return metrics
        
        # Compute velocities
        mode = modes[0] if modes else 'energy'
        profiler = VelocityProfiler(self.env, self.config, mode)
        velocities = profiler.compute_velocities(path)
        
        # Evaluate each segment
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            v1 = velocities[i] if i < len(velocities) else 5.0
            v2 = velocities[i + 1] if i + 1 < len(velocities) else 5.0
            
            energy, breakdown = self.energy_model.calculate_segment_energy(
                x1, y1, v1, x2, y2, v2
            )
            
            distance = math.sqrt((x2-x1)**2 + (y2-y1)**2) * self.config.map.cell_size
            time_seg = distance / max((v1 + v2) / 2, 0.1)
            
            terrain_type = self.env.terrain[x2, y2]
            metrics.add_segment(distance, time_seg, energy, (v1+v2)/2, 
                              breakdown.to_dict(), terrain_type)
        
        return metrics
