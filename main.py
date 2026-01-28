#!/usr/bin/env python3
"""
UGV Navigation System - Main Entry Point
=========================================

Usage:
    # Single scenario test
    python main.py test --seed 42 --verbose
    
    # Run full experiment suite
    python main.py suite --num_scenarios 30 --seed_base 42 --output results/
    
    # Run with live visualization (debug mode)
    python main.py debug --seed 42
    
    # Aggregate existing results
    python main.py aggregate --input results/ --output aggregated.json

For Google Colab:
    from ugv_nav import ExperimentRunner, Config
    
    config = Config()
    runner = ExperimentRunner(config)
    results = runner.run_suite(num_scenarios=30)
"""

import argparse
import sys
import json
from pathlib import Path


def run_test(args):
    """Run a single scenario test"""
    from ugv_nav import Config, ExperimentRunner
    
    config = Config()
    config.random_seed = args.seed
    config.verbose = args.verbose
    
    # Adjust FoV if specified
    if args.fov_radius:
        config.fov.base_radius_cells = args.fov_radius
    
    runner = ExperimentRunner(config)
    
    methods = args.methods.split(',') if args.methods else None
    
    result = runner.run_single_scenario(
        seed=args.seed,
        methods=methods,
        save_assets=not args.no_save,
        output_dir=args.output,
        verbose=args.verbose
    )
    
    # Print results
    print("\n" + "=" * 60)
    print(f"SCENARIO RESULT (seed={args.seed})")
    print("=" * 60)
    
    for method, data in result.methods.items():
        status = data.get('status', 'unknown')
        energy = data.get('total_energy_kJ', 'N/A')
        time_val = data.get('total_time_min', 'N/A')
        
        status_icon = "✓" if status == 'success' else "✗"
        print(f"{status_icon} {method:25s}: {status:12s} E={energy:>8} kJ  T={time_val:>6} min")
    
    print("=" * 60)
    
    return 0 if result.success else 1


def run_suite(args):
    """Run full experiment suite"""
    from ugv_nav import Config, ExperimentRunner
    
    config = Config()
    
    if args.fov_radius:
        config.fov.base_radius_cells = args.fov_radius
    
    runner = ExperimentRunner(config)
    
    methods = args.methods.split(',') if args.methods else None
    
    results = runner.run_suite(
        num_scenarios=args.num_scenarios,
        seed_base=args.seed_base,
        methods=methods,
        output_dir=args.output,
        parallel=args.parallel,
        max_workers=args.workers,
        verbose=args.verbose
    )
    
    print(f"\nResults saved to: {args.output}")
    
    return 0


def run_debug(args):
    """Run with live visualization for debugging"""
    from ugv_nav import Config, Environment, RecedingHorizonController
    from ugv_nav.recovery import RecoveryManager, AdaptiveFoV
    from ugv_nav.visualization import LiveMonitor, create_debug_callback
    
    config = Config()
    config.random_seed = args.seed
    config.verbose = True
    
    if args.fov_radius:
        config.fov.base_radius_cells = args.fov_radius
    
    print(f"Creating environment with seed {args.seed}...")
    env = Environment(config, seed=args.seed)
    
    print(f"Start: {env.start}, Goal: {env.goal}")
    print(f"Distance: {env.distance_to_goal(env.start):.0f}m")
    
    # Create monitor
    monitor = LiveMonitor(env)
    callback = create_debug_callback(monitor)
    
    # Create controller with recovery
    adaptive_fov = AdaptiveFoV(
        base_radius=config.fov.base_radius_cells,
        max_radius=config.fov.max_radius_cells
    )
    
    recovery_manager = RecoveryManager(config, seed=args.seed)
    
    controller = RecedingHorizonController(
        env=env,
        config=config,
        recovery_manager=recovery_manager,
        adaptive_fov=adaptive_fov,
        progress_callback=callback
    )
    
    print("\nStarting navigation with live visualization...")
    print("Close the window to stop.\n")
    
    try:
        result = controller.run(
            mode='energy',
            max_iterations=args.max_iters
        )
        
        print("\n" + "=" * 60)
        print(f"RESULT: {result.status}")
        if result.failure_type:
            print(f"Failure type: {result.failure_type}")
        print(f"Path length: {len(result.path)}")
        print(f"Replans: {result.replans}")
        print(f"Recoveries: {result.recovery_attempts} ({result.recovery_successes} successful)")
        
        if result.metrics:
            print(f"Energy: {result.metrics.total_energy_kJ:.1f} kJ")
            print(f"Time: {result.metrics.total_time_min:.2f} min")
            print(f"Distance: {result.metrics.total_distance_km:.2f} km")
        
        print("=" * 60)
        
        # Show analysis
        monitor.show_analysis()
        
    except KeyboardInterrupt:
        print("\nStopped by user.")
    finally:
        monitor.close()
    
    return 0


def run_aggregate(args):
    """Aggregate results from multiple runs"""
    import numpy as np
    
    input_path = Path(args.input)
    
    if not input_path.exists():
        print(f"Error: Input path does not exist: {input_path}")
        return 1
    
    # Find all scenario logs
    log_files = list(input_path.glob('**/logs.json'))
    
    if not log_files:
        print(f"No log files found in {input_path}")
        return 1
    
    print(f"Found {len(log_files)} log files")
    
    # Aggregate
    all_results = []
    for log_file in log_files:
        try:
            with open(log_file) as f:
                data = json.load(f)
                all_results.append(data)
        except Exception as e:
            print(f"Error reading {log_file}: {e}")
    
    if not all_results:
        print("No valid results found")
        return 1
    
    # Compute statistics
    methods = set()
    for r in all_results:
        methods.update(r.get('methods', {}).keys())
    
    methods = sorted(methods)
    
    summary = {}
    for method in methods:
        energies = []
        times = []
        successes = 0
        
        for r in all_results:
            if method in r.get('methods', {}):
                m = r['methods'][method]
                if m.get('status') == 'success':
                    successes += 1
                    if 'total_energy_kJ' in m:
                        energies.append(m['total_energy_kJ'])
                    if 'total_time_min' in m:
                        times.append(m['total_time_min'])
        
        n = len(all_results)
        summary[method] = {
            'success_rate': successes / n if n > 0 else 0,
            'n_success': successes,
            'n_total': n,
            'energy_mean_kJ': float(np.mean(energies)) if energies else None,
            'energy_std_kJ': float(np.std(energies)) if energies else None,
            'time_mean_min': float(np.mean(times)) if times else None,
            'time_std_min': float(np.std(times)) if times else None,
        }
    
    output = {
        'num_scenarios': len(all_results),
        'methods': methods,
        'summary': summary
    }
    
    # Save
    output_path = Path(args.output)
    with open(output_path, 'w') as f:
        json.dump(output, f, indent=2)
    
    print(f"\nAggregated results saved to: {output_path}")
    
    # Print summary
    print("\n" + "=" * 70)
    print("AGGREGATED SUMMARY")
    print("=" * 70)
    print(f"{'Method':<25} {'Success':>10} {'Energy(kJ)':>15} {'Time(min)':>12}")
    print("-" * 70)
    
    for method in methods:
        s = summary[method]
        rate = s['success_rate'] * 100
        energy = s.get('energy_mean_kJ')
        time_val = s.get('time_mean_min')
        
        energy_str = f"{energy:.1f}" if energy else "N/A"
        time_str = f"{time_val:.2f}" if time_val else "N/A"
        
        print(f"{method:<25} {rate:>9.1f}% {energy_str:>15} {time_str:>12}")
    
    print("=" * 70)
    
    return 0


def main():
    parser = argparse.ArgumentParser(
        description='UGV Navigation System',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Run single scenario test')
    test_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    test_parser.add_argument('--fov_radius', type=int, help='FoV radius in cells')
    test_parser.add_argument('--methods', type=str, help='Comma-separated methods')
    test_parser.add_argument('--output', type=str, default='test_output', help='Output directory')
    test_parser.add_argument('--no_save', action='store_true', help='Do not save assets')
    test_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # Suite command
    suite_parser = subparsers.add_parser('suite', help='Run full experiment suite')
    suite_parser.add_argument('--num_scenarios', type=int, default=30, help='Number of scenarios')
    suite_parser.add_argument('--seed_base', type=int, default=42, help='Base seed')
    suite_parser.add_argument('--fov_radius', type=int, help='FoV radius in cells')
    suite_parser.add_argument('--methods', type=str, help='Comma-separated methods')
    suite_parser.add_argument('--output', type=str, default='results', help='Output directory')
    suite_parser.add_argument('--parallel', action='store_true', help='Use parallel execution')
    suite_parser.add_argument('--workers', type=int, default=4, help='Number of workers')
    suite_parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    
    # Debug command
    debug_parser = subparsers.add_parser('debug', help='Run with live visualization')
    debug_parser.add_argument('--seed', type=int, default=42, help='Random seed')
    debug_parser.add_argument('--fov_radius', type=int, help='FoV radius in cells')
    debug_parser.add_argument('--max_iters', type=int, default=5000, help='Max iterations')
    
    # Aggregate command
    agg_parser = subparsers.add_parser('aggregate', help='Aggregate results')
    agg_parser.add_argument('--input', type=str, required=True, help='Input directory')
    agg_parser.add_argument('--output', type=str, default='aggregated.json', help='Output file')
    
    args = parser.parse_args()
    
    if args.command is None:
        parser.print_help()
        return 1
    
    if args.command == 'test':
        return run_test(args)
    elif args.command == 'suite':
        return run_suite(args)
    elif args.command == 'debug':
        return run_debug(args)
    elif args.command == 'aggregate':
        return run_aggregate(args)
    else:
        parser.print_help()
        return 1


if __name__ == '__main__':
    sys.exit(main())
