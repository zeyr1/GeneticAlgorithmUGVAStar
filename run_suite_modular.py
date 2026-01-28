#!/usr/bin/env python3
"""
Colab Pipeline Adapter
======================

Provides backward compatibility with the original pipeline_colab.py interface.
Wraps the new modular ugv_nav package for seamless integration.

Usage in Colab:
    from run_suite_modular import run_experiment_suite
    
    results = run_experiment_suite(
        num_scenarios=30,
        seed_base=42,
        output_dir='results',
        methods=['full_map_energy', 'fov_energy', 'fov_ga']
    )
"""

import os
import sys
import json
import numpy as np
from pathlib import Path
from typing import List, Optional, Dict, Any

# Add package to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ugv_nav import Config, ExperimentRunner
from ugv_nav.pipeline import AggregatedResults


def run_experiment_suite(
    num_scenarios: int = 30,
    seed_base: int = 42,
    output_dir: str = 'results',
    methods: Optional[List[str]] = None,
    fov_radius: int = 25,
    parallel: bool = False,
    max_workers: int = 4,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Run experiment suite compatible with original pipeline.
    
    Args:
        num_scenarios: Number of scenarios to run
        seed_base: Base seed for reproducibility
        output_dir: Output directory for results
        methods: List of methods to run (default: all)
        fov_radius: FoV radius in cells
        parallel: Use parallel execution
        max_workers: Number of parallel workers
        verbose: Print progress
    
    Returns:
        Dictionary with aggregated results
    """
    # Create config
    config = Config()
    config.fov.base_radius_cells = fov_radius
    
    # Create runner
    runner = ExperimentRunner(config)
    
    # Run suite
    results = runner.run_suite(
        num_scenarios=num_scenarios,
        seed_base=seed_base,
        methods=methods,
        output_dir=output_dir,
        parallel=parallel,
        max_workers=max_workers,
        verbose=verbose
    )
    
    return results.to_dict()


def generate_latex_table(results: Dict[str, Any], output_file: str = 'results_table.tex'):
    """
    Generate LaTeX table from results.
    
    Args:
        results: Aggregated results dictionary
        output_file: Output file path
    """
    summary = results.get('summary', {})
    methods = results.get('methods', list(summary.keys()))
    
    lines = [
        r'\begin{table}[h]',
        r'\centering',
        r'\caption{UGV Navigation Results}',
        r'\label{tab:results}',
        r'\begin{tabular}{lccc}',
        r'\toprule',
        r'Method & Success Rate & Energy (kJ) & Time (min) \\',
        r'\midrule',
    ]
    
    for method in methods:
        s = summary.get(method, {})
        rate = s.get('success_rate', 0) * 100
        energy = s.get('energy_mean_kJ')
        energy_std = s.get('energy_std_kJ')
        time_val = s.get('time_mean_min')
        time_std = s.get('time_std_min')
        
        # Format method name
        method_name = method.replace('_', r'\_')
        
        # Format values
        if energy is not None and energy_std is not None:
            energy_str = f'{energy:.1f} $\\pm$ {energy_std:.1f}'
        elif energy is not None:
            energy_str = f'{energy:.1f}'
        else:
            energy_str = '--'
        
        if time_val is not None and time_std is not None:
            time_str = f'{time_val:.2f} $\\pm$ {time_std:.2f}'
        elif time_val is not None:
            time_str = f'{time_val:.2f}'
        else:
            time_str = '--'
        
        lines.append(f'{method_name} & {rate:.1f}\\% & {energy_str} & {time_str} \\\\')
    
    lines.extend([
        r'\bottomrule',
        r'\end{tabular}',
        r'\end{table}',
    ])
    
    with open(output_file, 'w') as f:
        f.write('\n'.join(lines))
    
    print(f'LaTeX table saved to {output_file}')


def aggregate_existing_results(input_dir: str, output_file: str = 'aggregated.json') -> Dict:
    """
    Aggregate results from existing scenario runs.
    
    Args:
        input_dir: Directory containing scenario results
        output_file: Output file for aggregated results
    
    Returns:
        Aggregated results dictionary
    """
    input_path = Path(input_dir)
    
    # Find all log files
    log_files = list(input_path.glob('**/logs.json'))
    
    if not log_files:
        print(f'No log files found in {input_dir}')
        return {}
    
    print(f'Found {len(log_files)} log files')
    
    # Load all results
    all_results = []
    for log_file in log_files:
        try:
            with open(log_file) as f:
                data = json.load(f)
                all_results.append(data)
        except Exception as e:
            print(f'Error reading {log_file}: {e}')
    
    # Aggregate
    methods = set()
    for r in all_results:
        methods.update(r.get('methods', {}).keys())
    
    methods = sorted(methods)
    
    summary = {}
    for method in methods:
        energies = []
        times = []
        distances = []
        successes = 0
        failures = {'collision': 0, 'dead_end': 0, 'timeout': 0, 'error': 0}
        
        for r in all_results:
            if method not in r.get('methods', {}):
                continue
            
            m = r['methods'][method]
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
        
        n = len(all_results)
        summary[method] = {
            'success_rate': successes / n if n > 0 else 0,
            'n_success': successes,
            'n_total': n,
            'energy_mean_kJ': float(np.mean(energies)) if energies else None,
            'energy_std_kJ': float(np.std(energies)) if energies else None,
            'time_mean_min': float(np.mean(times)) if times else None,
            'time_std_min': float(np.std(times)) if times else None,
            'distance_mean_km': float(np.mean(distances)) if distances else None,
            'distance_std_km': float(np.std(distances)) if distances else None,
            'failures': failures
        }
    
    output = {
        'num_scenarios': len(all_results),
        'methods': methods,
        'summary': summary
    }
    
    # Save
    with open(output_file, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f'Aggregated results saved to {output_file}')
    
    return output


def print_summary(results: Dict[str, Any]):
    """Print formatted summary table"""
    summary = results.get('summary', {})
    methods = results.get('methods', list(summary.keys()))
    
    print('\n' + '=' * 80)
    print('EXPERIMENT SUMMARY')
    print('=' * 80)
    print(f"Total scenarios: {results.get('num_scenarios', 'N/A')}")
    print()
    
    # Header
    print(f"{'Method':<25} {'Success':>10} {'Energy(kJ)':>15} {'Time(min)':>12}")
    print('-' * 70)
    
    for method in methods:
        s = summary.get(method, {})
        rate = s.get('success_rate', 0) * 100
        energy = s.get('energy_mean_kJ')
        time_val = s.get('time_mean_min')
        
        energy_str = f'{energy:.1f}' if energy else 'N/A'
        time_str = f'{time_val:.2f}' if time_val else 'N/A'
        
        print(f'{method:<25} {rate:>9.1f}% {energy_str:>15} {time_str:>12}')
    
    print('=' * 80)


# Main entry point for direct execution
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='UGV Navigation Experiment Runner')
    parser.add_argument('--num_scenarios', type=int, default=30)
    parser.add_argument('--seed_base', type=int, default=42)
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--fov_radius', type=int, default=25)
    parser.add_argument('--parallel', action='store_true')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--methods', type=str, help='Comma-separated methods')
    
    args = parser.parse_args()
    
    methods = args.methods.split(',') if args.methods else None
    
    results = run_experiment_suite(
        num_scenarios=args.num_scenarios,
        seed_base=args.seed_base,
        output_dir=args.output_dir,
        methods=methods,
        fov_radius=args.fov_radius,
        parallel=args.parallel,
        max_workers=args.workers,
        verbose=True
    )
    
    print_summary(results)
    generate_latex_table(results, f'{args.output_dir}/results_table.tex')
