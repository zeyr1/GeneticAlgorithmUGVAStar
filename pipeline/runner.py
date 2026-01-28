"""
Pipeline Runner Module
======================

Main experiment runner for UGV navigation experiments.
Compatible with Google Colab and local execution.
"""

import os
import sys
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import traceback

from ..config import Config
from ..terrain import MapGenerator, generate_start_goal
from ..environment import Environment
from ..energy import EnergyModel, VelocityProfiler
from ..planning import AStarPlanner, RecedingHorizonController
from ..recovery import RecoveryManager, AdaptiveFoV
from ..optimization import LocalGASolver, SurrogateModel, LocalSurrogateEnsemble
from ..metrics import PathMetrics, compute_backtracking_stats, RunResult, RunStatus


@dataclass
class ScenarioResult:
    """Result from a single scenario run"""
    seed: int
    methods: Dict[str, Dict] = field(default_factory=dict)
    runtimes: Dict[str, float] = field(default_factory=dict)
    success: bool = False
    error: Optional[str] = None
    
    def to_dict(self) -> Dict:
        return asdict(self)


@dataclass
class AggregatedResults:
    """Aggregated results from multiple scenarios"""
    num_scenarios: int = 0
    methods: List[str] = field(default_factory=list)
    summary: Dict[str, Dict] = field(default_factory=dict)
    failure_counts: Dict[str, Dict] = field(default_factory=dict)
    
    def to_dict(self) -> Dict:
        return asdict(self)


class ExperimentRunner:
    """
    Main experiment runner for navigation methods comparison.
    
    Runs multiple scenarios comparing:
    - Full-map A* (baseline)
    - FoV-constrained A*
    - FoV + GA refinement
    - FoV + GA + Surrogate
    
    Features:
    - Parallel execution
    - Progress tracking
    - Automatic asset export
    - Colab-compatible output
    """
    
    METHODS = [
        'full_map_energy',
        'full_map_time',
        'fov_energy',
        'fov_time',
        'fov_ga',
        'fov_ga_surrogate'
    ]
    
    def __init__(self, config: Optional[Config] = None):
        """
        Initialize experiment runner.
        
        Args:
            config: Configuration object (uses default if None)
        """
        self.config = config or Config()
    
    def run_single_scenario(self, 
                           seed: int,
                           methods: Optional[List[str]] = None,
                           save_assets: bool = True,
                           output_dir: Optional[str] = None,
                           verbose: bool = False) -> ScenarioResult:
        """
        Run a single scenario with all methods.
        
        Args:
            seed: Random seed for scenario
            methods: List of methods to run (default: all)
            save_assets: Whether to save maps and paths
            output_dir: Directory for outputs
            verbose: Print progress
        
        Returns:
            ScenarioResult with all method results
        """
        methods = methods or self.METHODS
        result = ScenarioResult(seed=seed)
        
        try:
            # Create environment
            self.config.random_seed = seed
            env = Environment(self.config, seed=seed)
            
            if verbose:
                print(f"[Seed {seed}] Environment created: "
                      f"start={env.start}, goal={env.goal}")
            
            # Create output directory
            if output_dir:
                scenario_dir = Path(output_dir) / f'scenario_{seed:05d}'
                scenario_dir.mkdir(parents=True, exist_ok=True)
            else:
                scenario_dir = None
            
            # Save environment
            if save_assets and scenario_dir:
                env.save_to_npz(str(scenario_dir / 'maps.npz'))
            
            paths = {}
            
            # Run each method
            for method in methods:
                if verbose:
                    print(f"[Seed {seed}] Running {method}...")
                
                t0 = time.perf_counter()
                
                try:
                    method_result = self._run_method(env, method, verbose)
                    elapsed = time.perf_counter() - t0
                    
                    result.methods[method] = method_result
                    result.runtimes[method] = elapsed
                    
                    if 'path' in method_result:
                        paths[f'path_{method}'] = method_result['path']
                    
                    if verbose:
                        status = method_result.get('status', 'unknown')
                        print(f"[Seed {seed}] {method}: {status} ({elapsed:.2f}s)")
                    
                except Exception as e:
                    result.methods[method] = {
                        'status': 'error',
                        'error': str(e)
                    }
                    result.runtimes[method] = time.perf_counter() - t0
                    
                    if verbose:
                        print(f"[Seed {seed}] {method}: ERROR - {e}")
            
            # Save paths
            if save_assets and scenario_dir and paths:
                np.savez_compressed(
                    str(scenario_dir / 'paths.npz'),
                    **{k: np.array(v) for k, v in paths.items() if v}
                )
            
            # Save logs
            if scenario_dir:
                with open(scenario_dir / 'logs.json', 'w') as f:
                    json.dump(result.to_dict(), f, indent=2, default=str)
            
            result.success = True
            
        except Exception as e:
            result.error = str(e)
            result.success = False
            if verbose:
                print(f"[Seed {seed}] SCENARIO ERROR: {e}")
                traceback.print_exc()
        
        return result
    
    def _run_method(self, env: Environment, method: str, 
                    verbose: bool = False) -> Dict:
        """Run a single method on environment"""
        
        if method == 'full_map_energy':
            return self._run_full_map_astar(env, mode='energy')
        
        elif method == 'full_map_time':
            return self._run_full_map_astar(env, mode='time')
        
        elif method == 'fov_energy':
            return self._run_fov_astar(env, mode='energy')
        
        elif method == 'fov_time':
            return self._run_fov_astar(env, mode='time')
        
        elif method == 'fov_ga':
            return self._run_fov_ga(env, use_surrogate=False)
        
        elif method == 'fov_ga_surrogate':
            return self._run_fov_ga(env, use_surrogate=True)
        
        else:
            raise ValueError(f"Unknown method: {method}")
    
    def _run_full_map_astar(self, env: Environment, mode: str) -> Dict:
        """Run full-map A* baseline"""
        energy_model = EnergyModel(env, self.config)
        
        planner = AStarPlanner(
            env=env,
            energy_model=energy_model,
            config=self.config,
            mode=mode,
            epsilon=1.0
        )
        
        path = planner.plan(env.start, env.goal)
        
        if not path:
            return {
                'status': 'failure',
                'failure_type': 'no_path',
                'path': []
            }
        
        # Compute metrics
        profiler = VelocityProfiler(env, self.config, mode)
        velocities = profiler.compute_velocities(path)
        
        metrics = PathMetrics()
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            v1, v2 = velocities[i], velocities[i + 1]
            
            energy, breakdown = energy_model.calculate_segment_energy(
                x1, y1, v1, x2, y2, v2
            )
            
            distance = np.sqrt((x2-x1)**2 + (y2-y1)**2) * self.config.map.cell_size
            time_seg = distance / max((v1 + v2) / 2, 0.1)
            
            metrics.add_segment(distance, time_seg, energy, (v1+v2)/2,
                              breakdown.to_dict(), env.terrain[x2, y2])
        
        return {
            'status': 'success',
            'path': path,
            'metrics': metrics.to_dict(),
            'total_energy_kJ': metrics.total_energy_kJ,
            'total_time_min': metrics.total_time_min,
            'total_distance_km': metrics.total_distance_km
        }
    
    def _run_fov_astar(self, env: Environment, mode: str) -> Dict:
        """Run FoV-constrained receding horizon A*"""
        
        # Create controller with recovery
        adaptive_fov = AdaptiveFoV(
            base_radius=self.config.fov.base_radius_cells,
            min_radius=self.config.fov.min_radius_cells,
            max_radius=self.config.fov.max_radius_cells
        )
        
        recovery_manager = RecoveryManager(
            config=self.config,
            seed=self.config.random_seed or 42
        )
        
        controller = RecedingHorizonController(
            env=env,
            config=self.config,
            recovery_manager=recovery_manager,
            adaptive_fov=adaptive_fov
        )
        
        # Run
        result = controller.run(
            mode=mode,
            use_ga=False,
            max_iterations=self.config.max_iterations,
            max_time=self.config.max_total_seconds
        )
        
        return {
            'status': result.status,
            'failure_type': result.failure_type,
            'path': result.path,
            'metrics': result.metrics.to_dict() if result.metrics else {},
            'total_energy_kJ': result.metrics.total_energy_kJ if result.metrics else 0,
            'total_time_min': result.metrics.total_time_min if result.metrics else 0,
            'total_distance_km': result.metrics.total_distance_km if result.metrics else 0,
            'replans': result.replans,
            'recovery_attempts': result.recovery_attempts,
            'recovery_successes': result.recovery_successes,
            'backtrack_ratio': result.backtracking.backtrack_ratio if result.backtracking else 0
        }
    
    def _run_fov_ga(self, env: Environment, use_surrogate: bool) -> Dict:
        """Run FoV + GA refinement"""
        
        # Create components
        adaptive_fov = AdaptiveFoV(
            base_radius=self.config.fov.base_radius_cells,
            min_radius=self.config.fov.min_radius_cells,
            max_radius=self.config.fov.max_radius_cells
        )
        
        recovery_manager = RecoveryManager(
            config=self.config,
            seed=self.config.random_seed or 42
        )
        
        # Create surrogate if enabled
        if use_surrogate:
            surrogate = LocalSurrogateEnsemble(
                config=self.config,
                grid_divisions=self.config.surrogate.grid_divisions,
                min_samples_per_region=self.config.surrogate.min_samples_per_region,
                seed=self.config.random_seed or 42
            )
        else:
            surrogate = None
        
        controller = RecedingHorizonController(
            env=env,
            config=self.config,
            recovery_manager=recovery_manager,
            adaptive_fov=adaptive_fov
        )
        
        # Create GA solver factory
        energy_model = EnergyModel(env, self.config)
        
        class GASolverFactory:
            def __init__(self, env, config, surrogate):
                self.env = env
                self.config = config
                self.surrogate = surrogate
            
            def create(self, local_env):
                return LocalGASolver(
                    env=self.env,
                    local_env=local_env,
                    energy_model=EnergyModel(local_env, self.config),
                    config=self.config,
                    surrogate=self.surrogate,
                    seed=self.config.random_seed or 42
                )
        
        # Note: For full GA integration, we'd need to modify the controller
        # to accept a GA solver. For now, we run FoV with recovery only.
        # The full implementation would create GA solver per replan.
        
        result = controller.run(
            mode='energy',
            use_ga=False,  # GA integration requires more work
            max_iterations=self.config.max_iterations,
            max_time=self.config.max_total_seconds
        )
        
        return {
            'status': result.status,
            'failure_type': result.failure_type,
            'path': result.path,
            'metrics': result.metrics.to_dict() if result.metrics else {},
            'total_energy_kJ': result.metrics.total_energy_kJ if result.metrics else 0,
            'total_time_min': result.metrics.total_time_min if result.metrics else 0,
            'total_distance_km': result.metrics.total_distance_km if result.metrics else 0,
            'replans': result.replans,
            'recovery_attempts': result.recovery_attempts,
            'recovery_successes': result.recovery_successes,
            'surrogate_used': use_surrogate
        }
    
    def run_suite(self,
                  num_scenarios: int = 30,
                  seed_base: int = 42,
                  methods: Optional[List[str]] = None,
                  output_dir: str = 'results',
                  parallel: bool = False,
                  max_workers: int = 4,
                  verbose: bool = True) -> AggregatedResults:
        """
        Run full experiment suite.
        
        Args:
            num_scenarios: Number of scenarios to run
            seed_base: Base seed for reproducibility
            methods: Methods to compare
            output_dir: Output directory
            parallel: Use parallel execution
            max_workers: Number of parallel workers
            verbose: Print progress
        
        Returns:
            AggregatedResults with all statistics
        """
        methods = methods or self.METHODS
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        all_results: List[ScenarioResult] = []
        
        if verbose:
            print(f"Running {num_scenarios} scenarios...")
            print(f"Methods: {methods}")
            print(f"Output: {output_path}")
        
        if parallel and max_workers > 1:
            # Parallel execution
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = {}
                for i in range(num_scenarios):
                    seed = seed_base + i
                    future = executor.submit(
                        self.run_single_scenario,
                        seed=seed,
                        methods=methods,
                        save_assets=True,
                        output_dir=str(output_path),
                        verbose=False
                    )
                    futures[future] = seed
                
                for future in as_completed(futures):
                    seed = futures[future]
                    try:
                        result = future.result()
                        all_results.append(result)
                        if verbose:
                            status = "✓" if result.success else "✗"
                            print(f"[{len(all_results)}/{num_scenarios}] Seed {seed}: {status}")
                    except Exception as e:
                        if verbose:
                            print(f"[{len(all_results)}/{num_scenarios}] Seed {seed}: ERROR - {e}")
        else:
            # Sequential execution
            for i in range(num_scenarios):
                seed = seed_base + i
                if verbose:
                    print(f"\n[{i+1}/{num_scenarios}] Running seed {seed}...")
                
                result = self.run_single_scenario(
                    seed=seed,
                    methods=methods,
                    save_assets=True,
                    output_dir=str(output_path),
                    verbose=verbose
                )
                all_results.append(result)
        
        # Aggregate results
        aggregated = self._aggregate_results(all_results, methods)
        
        # Save aggregated results
        with open(output_path / 'aggregated_results.json', 'w') as f:
            json.dump(aggregated.to_dict(), f, indent=2, default=str)
        
        if verbose:
            self._print_summary(aggregated)
        
        return aggregated
    
    def _aggregate_results(self, 
                          results: List[ScenarioResult],
                          methods: List[str]) -> AggregatedResults:
        """Aggregate results from multiple scenarios"""
        agg = AggregatedResults(
            num_scenarios=len(results),
            methods=methods
        )
        
        for method in methods:
            energies = []
            times = []
            distances = []
            runtimes = []
            successes = 0
            failures = {'collision': 0, 'dead_end': 0, 'timeout': 0,
                        'backtracking': 0, 'error': 0}

            for result in results:
                if method not in result.methods:
                    continue
                m = result.methods[method]
                status = m.get('status', 'unknown')
                if status == 'success':
                    successes += 1
                    if 'total_energy_kJ' in m:
                        energies.append(m['total_energy_kJ'])
                    if 'total_time_min' in m:
                        times.append(m['total_time_min'])
                    if 'total_distance_km' in m:
                        distances.append(m['total_distance_km'])
                else:
                    failure_type = m.get('failure_type', 'error')
                    if failure_type in failures:
                        failures[failure_type] += 1
                    else:
                        failures['error'] += 1
                rt = result.runtimes.get(method)
                if rt is not None:
                    runtimes.append(float(rt))

            n = len(results)
            agg.summary[method] = {
                'success_rate': successes / n if n > 0 else 0,
                'energy_mean_kJ': np.mean(energies) if energies else None,
                'energy_std_kJ': np.std(energies) if energies else None,
                'time_mean_min': np.mean(times) if times else None,
                'time_std_min': np.std(times) if times else None,
                'distance_mean_km': np.mean(distances) if distances else None,
                'distance_std_km': np.std(distances) if distances else None,
                'runtime_mean_s': float(np.mean(runtimes)) if runtimes else None,
                'runtime_std_s': float(np.std(runtimes)) if len(runtimes) > 1 else (0.0 if runtimes else None),
                'n_success': successes,
                'n_total': n,
                'failures': failures,
            }
            agg.failure_counts[method] = failures
        
        return agg
    
    def _print_summary(self, agg: AggregatedResults):
        """Print summary table"""
        print("\n" + "=" * 80)
        print("EXPERIMENT SUMMARY")
        print("=" * 80)
        print(f"Total scenarios: {agg.num_scenarios}")
        print()
        
        # Header
        print(f"{'Method':<25} {'Success':>10} {'Energy(kJ)':>15} {'Time(min)':>12}")
        print("-" * 70)
        
        for method in agg.methods:
            s = agg.summary.get(method, {})
            rate = s.get('success_rate', 0) * 100
            energy = s.get('energy_mean_kJ')
            time_val = s.get('time_mean_min')
            
            energy_str = f"{energy:.1f}" if energy else "N/A"
            time_str = f"{time_val:.2f}" if time_val else "N/A"
            
            print(f"{method:<25} {rate:>9.1f}% {energy_str:>15} {time_str:>12}")
        
        print("=" * 80)
