"""
GA Solver Module
================

Local GA solver for path refinement in receding-horizon planning.
"""

import numpy as np
import hashlib
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field

from .individual import GAIndividual, GeneticOperators
from ...config import Config, GAConfig
from ...environment import LocalEnvironment
from ...energy import EnergyModel, VelocityProfiler
from ...metrics import PathMetrics
from ...planning.astar import AStarPlanner


@dataclass
class GAResult:
    """Result from GA optimization"""
    best_path: List[Tuple[int, int]]
    best_modes: List[str]
    best_fitness: float
    metrics: Optional[PathMetrics] = None
    generations_run: int = 0
    true_evals: int = 0
    surrogate_evals: int = 0
    surrogate_mape: Optional[float] = None


class LocalGASolver:
    """
    Local GA solver for path refinement.
    
    Used within receding-horizon loop to refine A* seed paths.
    
    Features:
    - Via-point representation with mode selection
    - A* repair operator for path feasibility
    - Optional surrogate model for faster evaluation
    - Caching of A* paths and evaluations
    """
    
    def __init__(self,
                 env,  # Global environment
                 local_env: LocalEnvironment,
                 energy_model: EnergyModel,
                 config: Config,
                 ga_config: Optional[GAConfig] = None,
                 surrogate=None,
                 seed: int = 42):
        """
        Initialize GA solver.
        
        Args:
            env: Global environment
            local_env: Local (FoV) environment
            energy_model: Energy model for evaluation
            config: Main configuration
            ga_config: GA-specific configuration
            surrogate: Optional surrogate model
            seed: Random seed
        """
        self.env = env
        self.local_env = local_env
        self.energy_model = energy_model
        self.config = config
        self.ga_config = ga_config or config.ga
        self.surrogate = surrogate
        
        # Genetic operators
        self.operators = GeneticOperators(
            mutation_rate=self.ga_config.mutation_rate,
            crossover_rate=self.ga_config.crossover_rate,
            tournament_k=self.ga_config.tournament_k,
            max_via_jitter=self.ga_config.max_via_jitter,
            seed=seed
        )
        
        self.rng = np.random.default_rng(seed)
        
        # Objective weights
        self.weights = self.ga_config.weights
        
        # Caches
        self._astar_cache: Dict[Tuple, List] = {}
        self._eval_cache: Dict[str, float] = {}
        
        # Stats
        self._true_evals = 0
        self._surrogate_evals = 0
    
    def solve(self,
              start: Tuple[int, int],
              goal: Tuple[int, int],
              seed_path: List[Tuple[int, int]],
              window_bounds: Tuple[int, int, int, int],
              enable_surrogate: bool = True) -> GAResult:
        """
        Run GA optimization.
        
        Args:
            start: Start position
            goal: Goal position (local goal)
            seed_path: Seed path from A*
            window_bounds: (xmin, xmax, ymin, ymax)
            enable_surrogate: Whether to use surrogate model
        
        Returns:
            GAResult with best path and metrics
        """
        # Clear caches for this solve
        self._astar_cache.clear()
        self._eval_cache.clear()
        self._true_evals = 0
        self._surrogate_evals = 0
        
        xmin, xmax, ymin, ymax = window_bounds
        bounds = (xmin, xmax, ymin, ymax)
        
        # Initialize population
        population = self._initialize_population(seed_path, start, goal, bounds)
        
        best_individual = None
        best_path = []
        best_modes = []
        
        # Evolution loop
        for gen in range(self.ga_config.generations):
            # Determine if using surrogate this generation
            use_surr = (enable_surrogate and 
                       self.surrogate is not None and
                       gen >= self.config.surrogate.warmup_generations)
            
            # Evaluate population
            self._evaluate_population(
                population, start, goal, bounds, use_surrogate=use_surr
            )
            
            # Sort by fitness
            population.sort(key=lambda x: x.fitness)
            
            # Track best
            if population[0].fitness < float('inf'):
                if best_individual is None or population[0].fitness < best_individual.fitness:
                    best_individual = population[0].copy()
            
            # Elitism
            elite_n = max(1, int(self.ga_config.elite_frac * len(population)))
            new_pop = [population[i].copy() for i in range(elite_n)]
            
            # Create offspring
            while len(new_pop) < self.ga_config.pop_size:
                p1 = self.operators.tournament_select(population)
                p2 = self.operators.tournament_select(population)
                c1, c2 = self.operators.crossover(p1, p2)
                c1 = self.operators.mutate(c1, bounds)
                c2 = self.operators.mutate(c2, bounds)
                new_pop.append(c1)
                if len(new_pop) < self.ga_config.pop_size:
                    new_pop.append(c2)
            
            population = new_pop
            
            # Retrain surrogate periodically
            if (self.surrogate is not None and 
                gen % self.config.surrogate.retrain_interval == 0):
                self.surrogate.fit()
        
        # Get best path
        if best_individual is not None:
            best_path, best_modes = self._repair_to_path(
                start, goal, best_individual, bounds
            )
        
        if not best_path:
            # Fallback to seed path
            best_path = seed_path
            best_modes = ['energy'] * (len(seed_path) - 1)
        
        # Compute final metrics
        metrics = self._compute_metrics(best_path, best_modes)
        
        return GAResult(
            best_path=best_path,
            best_modes=best_modes,
            best_fitness=best_individual.fitness if best_individual else float('inf'),
            metrics=metrics,
            generations_run=self.ga_config.generations,
            true_evals=self._true_evals,
            surrogate_evals=self._surrogate_evals,
            surrogate_mape=self.surrogate.last_mape if self.surrogate else None
        )
    
    def _initialize_population(self,
                               seed_path: List[Tuple[int, int]],
                               start: Tuple[int, int],
                               goal: Tuple[int, int],
                               bounds: Tuple[int, int, int, int]) -> List[GAIndividual]:
        """Initialize GA population"""
        population = []
        n_via = self.ga_config.n_via
        
        # First individual: exact seed path (no jitter)
        if seed_path and len(seed_path) >= 3:
            ind = self.operators.create_from_seed_path(
                seed_path, n_via, bounds, jitter=False
            )
            ind.seg_modes = ['energy'] * (n_via + 1)
            population.append(ind)
        
        # Rest: jittered seed paths and random
        while len(population) < self.ga_config.pop_size:
            if seed_path and self.rng.random() < 0.7:
                # Based on seed
                ind = self.operators.create_from_seed_path(
                    seed_path, n_via, bounds, jitter=True
                )
            else:
                # Random
                ind = self.operators.create_random_individual(n_via, bounds)
            population.append(ind)
        
        return population
    
    def _evaluate_population(self,
                            population: List[GAIndividual],
                            start: Tuple[int, int],
                            goal: Tuple[int, int],
                            bounds: Tuple[int, int, int, int],
                            use_surrogate: bool):
        """Evaluate all individuals in population"""
        
        for ind in population:
            if ind.fitness < float('inf') and ind.is_true_eval:
                continue  # Already evaluated
            
            # Repair to feasible path
            path, modes = self._repair_to_path(start, goal, ind, bounds)
            
            if not path or len(path) < 2:
                ind.fitness = float('inf')
                continue
            
            # Compute hash for caching
            path_hash = self._hash_path(path, modes)
            
            if path_hash in self._eval_cache:
                ind.fitness = self._eval_cache[path_hash]
                continue
            
            # Evaluate
            if use_surrogate and self.surrogate is not None and self.surrogate.can_predict():
                # Use surrogate
                fitness = self.surrogate.predict(path, modes, self.local_env)
                ind.is_true_eval = False
                self._surrogate_evals += 1
            else:
                # True evaluation
                fitness = self._true_evaluate(path, modes)
                ind.is_true_eval = True
                self._true_evals += 1
                
                # Add to surrogate training data
                if self.surrogate is not None:
                    self.surrogate.add_sample(path, modes, fitness, self.local_env)
            
            ind.fitness = fitness
            self._eval_cache[path_hash] = fitness
    
    def _repair_to_path(self,
                        start: Tuple[int, int],
                        goal: Tuple[int, int],
                        ind: GAIndividual,
                        bounds: Tuple[int, int, int, int]) -> Tuple[List, List]:
        """
        Repair individual to feasible path using A* segments.
        
        Connects: start -> via1 -> via2 -> ... -> goal
        """
        control_points = [start] + ind.via + [goal]
        full_path = []
        full_modes = []
        
        local_window = {
            'xmin': bounds[0], 'xmax': bounds[1],
            'ymin': bounds[2], 'ymax': bounds[3]
        }
        
        for i in range(len(control_points) - 1):
            s = control_points[i]
            g = control_points[i + 1]
            mode = ind.seg_modes[i] if i < len(ind.seg_modes) else 'energy'
            
            # Check cache
            cache_key = (s, g, mode)
            if cache_key in self._astar_cache:
                sub_path = self._astar_cache[cache_key]
            else:
                # Run A*
                planner = AStarPlanner(
                    env=self.local_env,
                    energy_model=self.energy_model,
                    config=self.config,
                    mode=mode,
                    epsilon=1.3,
                    max_expansions=50000,
                    cone_angle=np.pi * 1.5
                )
                sub_path = planner.plan(s, g, local_window=local_window)
                self._astar_cache[cache_key] = sub_path
            
            if not sub_path or len(sub_path) < 2:
                return [], []
            
            # Merge paths
            if not full_path:
                full_path = list(sub_path)
            else:
                full_path.extend(sub_path[1:])  # Skip duplicate point
            
            # Add modes for segments
            full_modes.extend([mode] * (len(sub_path) - 1))
        
        return full_path, full_modes
    
    def _true_evaluate(self, path: List[Tuple[int, int]], 
                       modes: List[str]) -> float:
        """True evaluation using energy model"""
        if len(path) < 2:
            return float('inf')
        
        # Compute velocities
        profiler = VelocityProfiler(
            self.local_env, self.config, 
            mode=modes[0] if modes else 'energy'
        )
        velocities = profiler.compute_velocities(path)
        
        total_energy = 0.0
        total_time = 0.0
        total_risk = 0.0
        
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            v1 = velocities[i] if i < len(velocities) else 5.0
            v2 = velocities[i + 1] if i + 1 < len(velocities) else 5.0
            
            energy, breakdown = self.energy_model.calculate_segment_energy(
                x1, y1, v1, x2, y2, v2
            )
            
            if energy == float('inf'):
                return float('inf')
            
            distance = np.sqrt((x2-x1)**2 + (y2-y1)**2) * self.config.map.cell_size
            time_seg = distance / max((v1 + v2) / 2, 0.1)
            
            total_energy += energy
            total_time += time_seg
            total_risk += breakdown.risk + breakdown.uncertainty
        
        # Weighted objective
        w = self.weights
        objective = (w.get('energy', 0.4) * total_energy +
                    w.get('time', 0.3) * total_time +
                    w.get('safety', 0.0) * total_risk)
        
        return objective
    
    def _compute_metrics(self, path: List, modes: List) -> PathMetrics:
        """Compute path metrics"""
        from ...metrics import PathMetrics
        
        metrics = PathMetrics()
        
        if len(path) < 2:
            return metrics
        
        profiler = VelocityProfiler(
            self.local_env, self.config,
            mode=modes[0] if modes else 'energy'
        )
        velocities = profiler.compute_velocities(path)
        
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            v1 = velocities[i] if i < len(velocities) else 5.0
            v2 = velocities[i + 1] if i + 1 < len(velocities) else 5.0
            
            energy, breakdown = self.energy_model.calculate_segment_energy(
                x1, y1, v1, x2, y2, v2
            )
            
            distance = np.sqrt((x2-x1)**2 + (y2-y1)**2) * self.config.map.cell_size
            time_seg = distance / max((v1 + v2) / 2, 0.1)
            
            terrain_type = self.local_env.terrain[x2, y2]
            metrics.add_segment(
                distance, time_seg, energy, (v1+v2)/2,
                breakdown.to_dict(), terrain_type
            )
        
        return metrics
    
    def _hash_path(self, path: List, modes: List) -> str:
        """Create hash for path caching"""
        h = hashlib.sha1()
        h.update(np.array(path, dtype=np.int16).tobytes())
        h.update('|'.join(modes).encode())
        return h.hexdigest()[:16]
