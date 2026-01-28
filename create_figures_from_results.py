#!/usr/bin/env python3
"""
Generate Figures and Tables from Existing Results
==================================================

This script reads your existing experiment results (logs.json, maps.npz, paths.npz)
and generates publication-ready figures and LaTeX tables.

Usage:
    python create_figures_from_results.py --input paper_results --output figures

Input structure expected:
    paper_results/
    ├── aggregated_results.json
    └── scenario_00042/
        ├── logs.json
        ├── maps.npz
        └── paths.npz
"""

import sys
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from pathlib import Path
import csv
from typing import Dict, List, Any

# Fix path
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


class ResultsVisualizer:
    """Generate publication figures from existing experiment results."""
    
    METHOD_LABELS = {
        'full_map_time': 'Full-map time',
        'full_map_energy': 'Full-map energy',
        'fov_time': 'FoV time',
        'fov_energy': 'FoV energy',
        'fov_ga': 'FoV+GA',
        'fov_ga_surrogate': 'FoV+GA+Surrogate',
    }
    
    METHOD_COLORS = {
        'full_map_time': '#1f77b4',
        'full_map_energy': '#ff7f0e',
        'fov_time': '#2ca02c',
        'fov_energy': '#d62728',
        'fov_ga': '#9467bd',
        'fov_ga_surrogate': '#8c564b',
    }
    
    TERRAIN_CMAP = ListedColormap(['darkgray', 'lightgreen', 'saddlebrown', 'gold', 'black'])
    TERRAIN_NAMES = ['Asphalt', 'Grass', 'Mud', 'Sand', 'Wall']
    
    def __init__(self, input_dir: str, output_dir: str = 'figures'):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        
        # Create output directories
        self.figures_dir = self.output_dir / 'figures'
        self.tables_dir = self.output_dir / 'tables'
        self.maps_dir = self.output_dir / 'maps'
        
        for d in [self.figures_dir, self.tables_dir, self.maps_dir]:
            d.mkdir(parents=True, exist_ok=True)
        
        # Load data
        self.aggregated = None
        self.scenario_logs = {}
        self.scenario_maps = {}
        self.scenario_paths = {}
    
    def load_all_data(self):
        """Load all data from input directory."""
        print("Loading data...")
        
        # Load aggregated results
        agg_file = self.input_dir / 'aggregated_results.json'
        if agg_file.exists():
            with open(agg_file) as f:
                self.aggregated = json.load(f)
            print(f"  [ok] Loaded aggregated_results.json")
        
        # Find all scenario directories
        scenario_dirs = sorted(self.input_dir.glob('scenario_*'))
        print(f"  Found {len(scenario_dirs)} scenario directories")
        
        for scenario_dir in scenario_dirs:
            seed = scenario_dir.name.replace('scenario_', '')
            
            # Load logs.json
            logs_file = scenario_dir / 'logs.json'
            if logs_file.exists():
                with open(logs_file) as f:
                    self.scenario_logs[seed] = json.load(f)
            
            # Load maps.npz
            maps_file = scenario_dir / 'maps.npz'
            if maps_file.exists():
                self.scenario_maps[seed] = dict(np.load(maps_file))
            
            # Load paths.npz
            paths_file = scenario_dir / 'paths.npz'
            if paths_file.exists():
                self.scenario_paths[seed] = dict(np.load(paths_file))
        
        print(f"  [ok] Loaded {len(self.scenario_logs)} scenario logs")
        print(f"  [ok] Loaded {len(self.scenario_maps)} scenario maps")
        print(f"  [ok] Loaded {len(self.scenario_paths)} scenario paths")
    
    def compute_statistics(self) -> Dict:
        """Compute statistics from scenario logs."""
        if self.aggregated and 'summary' in self.aggregated:
            summary = {k: dict(v) for k, v in self.aggregated['summary'].items()}
        else:
            summary = {}
            methods = set()
            for log in self.scenario_logs.values():
                if 'methods' in log:
                    methods.update(log['methods'].keys())
            for method in methods:
                energies = []
                times = []
                distances = []
                successes = 0
                failures = {'collision': 0, 'dead_end': 0, 'timeout': 0,
                            'backtracking': 0, 'no_path': 0, 'error': 0}
                for log in self.scenario_logs.values():
                    if 'methods' not in log or method not in log['methods']:
                        continue
                    m = log['methods'][method]
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
                        ft = m.get('failure_type', 'error')
                        if ft in failures:
                            failures[ft] += 1
                        else:
                            failures['error'] += 1
                n = len(self.scenario_logs)
                summary[method] = {
                    'success_rate': successes / n if n > 0 else 0,
                    'n_success': successes,
                    'n_total': n,
                    'energy_mean_kJ': np.mean(energies) if energies else None,
                    'energy_std_kJ': np.std(energies) if energies else None,
                    'time_mean_min': np.mean(times) if times else None,
                    'time_std_min': np.std(times) if times else None,
                    'distance_mean_km': np.mean(distances) if distances else None,
                    'distance_std_km': np.std(distances) if distances else None,
                    'raw_energies': energies,
                    'raw_times': times,
                    'raw_distances': distances,
                    'failures': failures,
                }

        # Augment with runtime from scenario_logs (for both aggregated and computed)
        for method in list(summary.keys()):
            if summary[method].get('runtime_mean_s') is not None:
                continue
            runtimes = []
            for log in self.scenario_logs.values():
                r = log.get('runtimes', {}).get(method)
                if r is not None:
                    runtimes.append(float(r))
            if runtimes:
                summary[method]['runtime_mean_s'] = float(np.mean(runtimes))
                summary[method]['runtime_std_s'] = float(np.std(runtimes)) if len(runtimes) > 1 else 0.0

        # Merge failures from aggregated failure_counts if missing
        if self.aggregated and 'failure_counts' in self.aggregated:
            for method, fc in self.aggregated['failure_counts'].items():
                if method in summary and 'failures' not in summary[method]:
                    summary[method]['failures'] = dict(fc)
        return summary
    
    def generate_all(self):
        """Generate all figures and tables."""
        print("\n" + "="*60)
        print("GENERATING FIGURES AND TABLES")
        print("="*60)
        
        summary = self.compute_statistics()

        # 1. Table I (key parameters)
        self.generate_table_parameters()
        print("[ok] Table I (parameters) saved")

        # 2. LaTeX Table (Table II)
        self.generate_latex_table(summary)
        print("[ok] LaTeX table (Table II) saved")

        # 3. Table III (GA planning stats)
        self.generate_table_ga_stats()
        print("[ok] Table III (GA stats) saved")
        
        # 4. Failure taxonomy (Figure 2)
        self.plot_failure_taxonomy(summary)
        print("[ok] Failure taxonomy plot saved")

        # 5. Pareto plot (Figure 3)
        self.plot_pareto_energy_time()
        print("[ok] Pareto energy-time plot saved")

        # 6. Distance comparison (Figure 4)
        self.plot_distance_comparison(summary)
        print("[ok] Distance comparison plot saved")

        # 7. Runtime comparison (Figure 5)
        self.plot_runtime_comparison()
        print("[ok] Runtime comparison plot saved")

        # 8. Success rate bar chart
        self.plot_success_rate(summary)
        print("[ok] Success rate plot saved")

        # 9. Scenario maps with trajectories (Figure 1)
        self.plot_scenario_maps()
        print("[ok] Scenario maps saved")

        # 10. CSV summary
        self.save_csv_summary(summary)
        print("[ok] CSV summary saved")
        
        print("\n" + "="*60)
        print(f"ALL OUTPUTS SAVED TO: {self.output_dir}")
        print("="*60)
    
    def generate_latex_table(self, summary: Dict):
        """Generate LaTeX table like Table II."""
        
        # Compute 95% CI from std
        def ci95(std, n):
            if std is None or n == 0:
                return None
            return 1.96 * std / np.sqrt(n)
        
        lines = [
            r'\begin{table*}[t]',
            r'\centering',
            r'\caption{Method-level performance summary over N scenarios. Mean$\pm$95\% CI are computed over successful trials only; methods with zero successes show "---" for those metrics.}',
            r'\label{tab:results}',
            r'\begin{tabular}{lccccc}',
            r'\toprule',
            r'Method & Success & Time (min) & Energy (kJ) & Distance (km) & Runtime (s) \\',
            r'\midrule',
        ]
        
        method_order = ['full_map_time', 'full_map_energy', 'fov_time', 'fov_energy', 'fov_ga', 'fov_ga_surrogate']
        
        for method in method_order:
            if method not in summary:
                continue
            
            s = summary[method]
            label = self.METHOD_LABELS.get(method, method).replace('_', r'\_')
            
            n_success = s.get('n_success', 0)
            n_total = s.get('n_total', 0)
            
            # Success rate
            rate = s.get('success_rate', 0)
            if isinstance(rate, float) and rate <= 1:
                rate_pct = rate * 100
            else:
                rate_pct = rate
            success_str = f"{rate_pct:.1f}\\% ({n_success}/{n_total})"
            
            # Time
            t_mean = s.get('time_mean_min')
            t_std = s.get('time_std_min')
            if t_mean is not None:
                t_ci = ci95(t_std, n_success)
                time_str = f"{t_mean:.2f} $\\pm$ {t_ci:.2f}" if t_ci else f"{t_mean:.2f}"
            else:
                time_str = "---"
            
            # Energy
            e_mean = s.get('energy_mean_kJ')
            e_std = s.get('energy_std_kJ')
            if e_mean is not None:
                e_ci = ci95(e_std, n_success)
                energy_str = f"{e_mean:.1f} $\\pm$ {e_ci:.1f}" if e_ci else f"{e_mean:.1f}"
            else:
                energy_str = "---"
            
            # Distance
            d_mean = s.get('distance_mean_km')
            d_std = s.get('distance_std_km')
            if d_mean is not None:
                d_ci = ci95(d_std, n_success)
                dist_str = f"{d_mean:.2f} $\\pm$ {d_ci:.2f}" if d_ci else f"{d_mean:.2f}"
            else:
                dist_str = "---"

            # Runtime (from scenario runtimes, over all trials with a runtime)
            rt_mean = s.get('runtime_mean_s')
            rt_std = s.get('runtime_std_s')
            n_rt = sum(1 for log in self.scenario_logs.values() if log.get('runtimes', {}).get(method) is not None)
            if rt_mean is not None and n_rt > 0:
                rt_ci = ci95(rt_std, n_rt)
                runtime_str = f"{rt_mean:.2f} $\\pm$ {rt_ci:.2f}" if rt_ci else f"{rt_mean:.2f}"
            else:
                runtime_str = "---"

            lines.append(f'{label} & {success_str} & {time_str} & {energy_str} & {dist_str} & {runtime_str} \\\\')
        
        lines.extend([
            r'\bottomrule',
            r'\end{tabular}',
            r'\end{table*}',
        ])
        
        with open(self.tables_dir / 'table_results.tex', 'w') as f:
            f.write('\n'.join(lines))
        
        # Also save a simple text version
        with open(self.tables_dir / 'table_results.txt', 'w') as f:
            f.write("="*100 + "\n")
            f.write("TABLE II: Method-level Performance Summary\n")
            f.write("="*100 + "\n")
            f.write(f"{'Method':<22} {'Success':>12} {'Energy (kJ)':>14} {'Time (min)':>12} {'Distance (km)':>14} {'Runtime (s)':>12}\n")
            f.write("-"*100 + "\n")

            for method in method_order:
                if method not in summary:
                    continue
                s = summary[method]
                rate = s.get('success_rate', 0)
                if isinstance(rate, float) and rate <= 1:
                    rate = rate * 100
                n_success = s.get('n_success', 0)
                n_total = s.get('n_total', 0)
                e_mean = s.get('energy_mean_kJ')
                t_mean = s.get('time_mean_min')
                d_mean = s.get('distance_mean_km')
                rt_mean = s.get('runtime_mean_s')
                f.write(f"{method:<22} {rate:>6.1f}% ({n_success}/{n_total}) ")
                f.write(f"{e_mean:>14.1f}" if e_mean else "            ---")
                f.write(f"{t_mean:>12.2f}" if t_mean else "          ---")
                f.write(f"{d_mean:>14.2f}" if d_mean else "            ---")
                f.write(f"{rt_mean:>12.2f}" if rt_mean is not None else "          ---")
                f.write("\n")
            f.write("="*100 + "\n")

    def generate_table_parameters(self):
        """Generate Table I: key parameters (paper format)."""
        # Paper Table I values; override from aggregated config if present
        params = [
            ("Map size / resolution", "1000 $\\times$ 1000 cells, $c = 2$ m"),
            ("FoV radius", "$R = 15$ cells ($\\approx 30$ m)"),
            ("Execute steps", "$K = 12$"),
            ("Speed limits (m/s)", "asphalt 15, grass 8, mud 3, sand 5"),
            ("Friction coeffs", "asphalt 0.015, grass 0.08, mud 0.25, sand 0.15"),
            ("Accel bounds", "$a_{max} = 2.5$, $a_{min} = -4.0$ m/s$^2$"),
            ("Max turn angle", "$\\pi/6$ rad (30$^\\circ$)"),
            ("Unknown (outside FoV)", "friction 0.5, risk 1.0, elevation 40 m"),
            ("GA weights $(w_E, w_T, w_S)$", "(0.4, 0.3, 0.0)"),
            ("Scenarios", "$N = 30$"),
            ("Hardware / software", "i5-9400F, 16GB RAM; Python 3.10.2"),
        ]
        lines = [
            r'\begin{table}[h]',
            r'\centering',
            r'\caption{Key parameters used in experiments (from code).}',
            r'\label{tab:parameters}',
            r'\begin{tabular}{ll}',
            r'\toprule',
            r'Parameter & Value \\',
            r'\midrule',
        ]
        for name, val in params:
            lines.append(f"{name} & {val} \\\\")
        lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}'])
        with open(self.tables_dir / 'table_parameters.tex', 'w') as f:
            f.write('\n'.join(lines))
        with open(self.tables_dir / 'table_parameters.txt', 'w', encoding='utf-8') as f:
            f.write("TABLE I: Key Parameters\n" + "=" * 60 + "\n")
            for name, val in params:
                v = val.replace("$", "").replace("\\", "").replace("approx", "~").replace("pi", "pi")
                f.write(f"  {name:<35} {v}\n")
            f.write("=" * 60 + "\n")

    def generate_table_ga_stats(self):
        """Generate Table III: GA planning statistics (replans, replan time, true evals/replan)."""
        ga_methods = ['fov_ga', 'fov_ga_surrogate']
        rows = []
        for method in ga_methods:
            replans_list = []
            replan_times = []
            true_evals_list = []
            for log in self.scenario_logs.values():
                m = (log.get('methods') or {}).get(method, {})
                rp = m.get('replans')
                if rp is not None:
                    replans_list.append(int(rp))
                rt = (log.get('runtimes') or {}).get(method)
                if rt is not None and rp is not None and rp > 0:
                    replan_times.append(float(rt) / int(rp))
                ev = m.get('true_evals_per_replan') or m.get('true_evals')
                if ev is not None:
                    true_evals_list.append(float(ev))
            replans = f"{np.mean(replans_list):.2f}" if replans_list else "---"
            replan_time = f"{np.mean(replan_times):.2f}" if replan_times else "---"
            true_evals = f"{np.mean(true_evals_list):.1f}" if true_evals_list else "---"
            rows.append((self.METHOD_LABELS.get(method, method), replans, replan_time, true_evals))

        lines = [
            r'\begin{table}[h]',
            r'\centering',
            r'\caption{GA planning statistics (means over scenarios; may include failures depending on logging).}',
            r'\label{tab:ga_stats}',
            r'\begin{tabular}{lccc}',
            r'\toprule',
            r'Method & replans & replan time (s) & true evals/replan \\',
            r'\midrule',
        ]
        for label, rp, rt, te in rows:
            lab = label.replace('_', r'\_')
            lines.append(f"{lab} & {rp} & {rt} & {te} \\\\")
        lines.extend([r'\bottomrule', r'\end{tabular}', r'\end{table}'])
        with open(self.tables_dir / 'table_ga_stats.tex', 'w') as f:
            f.write('\n'.join(lines))
        with open(self.tables_dir / 'table_ga_stats.txt', 'w') as f:
            f.write("TABLE III: GA Planning Statistics\n" + "=" * 70 + "\n")
            f.write(f"{'Method':<25} {'replans':>12} {'replan time (s)':>18} {'true evals/replan':>18}\n")
            f.write("-" * 70 + "\n")
            for (label, rp, rt, te) in rows:
                f.write(f"{label:<25} {rp:>12} {rt:>18} {te:>18}\n")
            f.write("=" * 70 + "\n")

    def plot_failure_taxonomy(self, summary: Dict):
        """Plot failure taxonomy bar chart (Figure 2)."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = [m for m in ['full_map_time', 'full_map_energy', 'fov_time', 'fov_energy', 'fov_ga', 'fov_ga_surrogate'] 
                   if m in summary]
        
        if not methods:
            print("  Warning: No methods found for failure taxonomy")
            return
        
        labels = [self.METHOD_LABELS.get(m, m) for m in methods]
        
        # Get failure counts
        categories = ['success', 'collision', 'dead_end', 'timeout', 'backtracking', 'error']
        colors = ['#2ecc71', '#e74c3c', '#3498db', '#f39c12', '#9b59b6', '#95a5a6']
        
        bottoms = np.zeros(len(methods))
        
        for cat, color in zip(categories, colors):
            values = []
            for m in methods:
                s = summary[m]
                if cat == 'success':
                    values.append(s.get('n_success', 0))
                elif 'failures' in s:
                    values.append(s['failures'].get(cat, 0))
                else:
                    # Try to get from aggregated failure_counts
                    if self.aggregated and 'failure_counts' in self.aggregated:
                        fc = self.aggregated['failure_counts'].get(m, {})
                        values.append(fc.get(cat, 0))
                    else:
                        values.append(0)
            
            if sum(values) > 0:  # Only plot if there are values
                ax.bar(labels, values, bottom=bottoms, label=cat.replace('_', ' ').title(), 
                       color=color, edgecolor='white', linewidth=0.5)
                bottoms += np.array(values)
        
        ax.set_ylabel('Count', fontsize=12)
        ax.set_title('Failure Taxonomy (counts)', fontsize=14)
        ax.legend(loc='upper right')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'failure_taxonomy.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / 'failure_taxonomy.pdf', bbox_inches='tight')
        plt.close()
    
    def plot_pareto_energy_time(self):
        """Plot energy vs time Pareto frontier (Figure 3)."""
        fig, ax = plt.subplots(figsize=(10, 7))
        
        # Collect data from individual scenarios
        method_data = {}
        
        for seed, log in self.scenario_logs.items():
            if 'methods' not in log:
                continue
            
            for method, data in log['methods'].items():
                if data.get('status') != 'success':
                    continue
                
                energy = data.get('total_energy_kJ')
                time_val = data.get('total_time_min')
                
                if energy is not None and time_val is not None:
                    if method not in method_data:
                        method_data[method] = {'energies': [], 'times': []}
                    method_data[method]['energies'].append(energy)
                    method_data[method]['times'].append(time_val)
        
        # Plot each method (including FoV+GA+Surrogate)
        for method in ['full_map_time', 'full_map_energy', 'fov_time', 'fov_energy', 'fov_ga', 'fov_ga_surrogate']:
            if method not in method_data:
                continue
            energies = method_data[method]['energies']
            times = method_data[method]['times']
            ax.scatter(energies, times,
                      c=self.METHOD_COLORS.get(method, 'gray'),
                      label=self.METHOD_LABELS.get(method, method),
                      alpha=0.7, s=80, edgecolors='white', linewidth=0.5)
        
        ax.set_xlabel('Energy (kJ)', fontsize=12)
        ax.set_ylabel('Time (min)', fontsize=12)
        ax.set_title('Pareto: Energy vs Time (successful runs)', fontsize=14)
        ax.legend(fontsize=10)
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'pareto_energy_time.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / 'pareto_energy_time.pdf', bbox_inches='tight')
        plt.close()
    
    def plot_distance_comparison(self, summary: Dict):
        """Plot distance comparison bar chart (Figure 4)."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = [m for m in ['full_map_time', 'full_map_energy', 'fov_time', 'fov_energy', 'fov_ga', 'fov_ga_surrogate']
                   if m in summary and summary[m].get('distance_mean_km') is not None]
        
        if not methods:
            print("  Warning: No distance data found")
            return
        
        labels = [self.METHOD_LABELS.get(m, m) for m in methods]
        means = [summary[m]['distance_mean_km'] for m in methods]
        
        # Compute 95% CI
        cis = []
        for m in methods:
            s = summary[m]
            std = s.get('distance_std_km')
            n = s.get('n_success', 1)
            if std and n > 0:
                cis.append(1.96 * std / np.sqrt(n))
            else:
                cis.append(0)
        
        colors = [self.METHOD_COLORS.get(m, 'gray') for m in methods]
        
        bars = ax.bar(labels, means, yerr=cis, capsize=5, color=colors, 
                     edgecolor='white', linewidth=0.5)
        
        ax.set_ylabel('Distance (km)', fontsize=12)
        ax.set_title('Distance (success-only mean ± CI95)', fontsize=14)
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'distance_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / 'distance_comparison.pdf', bbox_inches='tight')
        plt.close()
    
    def plot_runtime_comparison(self):
        """Plot runtime comparison (Figure 5)."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Collect runtime from logs
        method_runtimes = {}
        
        for seed, log in self.scenario_logs.items():
            if 'runtimes' in log:
                for method, runtime in log['runtimes'].items():
                    if method not in method_runtimes:
                        method_runtimes[method] = []
                    method_runtimes[method].append(runtime)
        
        if not method_runtimes:
            print("  Warning: No runtime data found in logs")
            # Create placeholder
            ax.text(0.5, 0.5, 'No runtime data available', ha='center', va='center', 
                   transform=ax.transAxes, fontsize=14)
        else:
            methods = [m for m in ['full_map_time', 'full_map_energy', 'fov_time', 'fov_energy', 'fov_ga', 'fov_ga_surrogate']
                       if m in method_runtimes]
            
            labels = [self.METHOD_LABELS.get(m, m) for m in methods]
            means = [np.mean(method_runtimes[m]) for m in methods]
            cis = [1.96 * np.std(method_runtimes[m]) / np.sqrt(len(method_runtimes[m])) 
                   for m in methods]
            colors = [self.METHOD_COLORS.get(m, 'gray') for m in methods]
            
            bars = ax.bar(labels, means, yerr=cis, capsize=5, color=colors,
                         edgecolor='white', linewidth=0.5)
            
            ax.set_ylabel('Runtime (s)', fontsize=12)
        
        ax.set_title('Runtime (mean ± CI95)', fontsize=14)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'runtime_comparison.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / 'runtime_comparison.pdf', bbox_inches='tight')
        plt.close()
    
    def plot_success_rate(self, summary: Dict):
        """Plot success rate comparison."""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        methods = [m for m in ['full_map_time', 'full_map_energy', 'fov_time', 'fov_energy', 'fov_ga', 'fov_ga_surrogate'] 
                   if m in summary]
        
        labels = [self.METHOD_LABELS.get(m, m) for m in methods]
        
        rates = []
        for m in methods:
            rate = summary[m].get('success_rate', 0)
            if isinstance(rate, float) and rate <= 1:
                rate = rate * 100
            rates.append(rate)
        
        colors = [self.METHOD_COLORS.get(m, 'gray') for m in methods]
        
        bars = ax.bar(labels, rates, color=colors, edgecolor='white', linewidth=0.5)
        
        # Add value labels on bars
        for bar, rate in zip(bars, rates):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                   f'{rate:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        ax.set_ylabel('Success Rate (%)', fontsize=12)
        ax.set_title('Success Rate Comparison', fontsize=14)
        ax.set_ylim(0, 110)
        
        # Add horizontal line at paper's original rate
        ax.axhline(y=23.3, color='red', linestyle='--', alpha=0.7, label='Original paper (23.3%)')
        ax.legend()
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.figures_dir / 'success_rate.png', dpi=300, bbox_inches='tight')
        plt.savefig(self.figures_dir / 'success_rate.pdf', bbox_inches='tight')
        plt.close()
    
    def plot_scenario_maps(self, max_scenarios: int = 5):
        """Plot terrain maps with trajectories (Figure 1 style)."""
        available = [s for s in self.scenario_maps.keys() if s in self.scenario_paths]
        if not available:
            print("  Warning: No scenario maps with paths found")
            return
        sorted_seeds = sorted(available)
        # Paper's representative scenario is seed 42; dirs are scenario_00042
        representative_seed = '00042' if '00042' in sorted_seeds else (sorted_seeds[0] if sorted_seeds else None)

        for seed in sorted_seeds[:max_scenarios]:
            maps = self.scenario_maps[seed]
            paths = self.scenario_paths[seed]
            
            # Get log for start/goal
            log = self.scenario_logs.get(seed, {})
            
            fig, axes = plt.subplots(1, 3, figsize=(15, 5))
            
            # Panel 1: Terrain with trajectories
            ax1 = axes[0]
            terrain = maps.get('terrain', maps.get('terrain_map'))
            if terrain is not None:
                ax1.imshow(terrain.T, cmap=self.TERRAIN_CMAP, origin='lower', alpha=0.8)
            
            # Plot paths
            for key, path in paths.items():
                if path is not None and len(path) > 0:
                    # Extract method name from key (e.g., 'path_fov_energy' -> 'fov_energy')
                    method = key.replace('path_', '')
                    color = self.METHOD_COLORS.get(method, 'blue')
                    label = self.METHOD_LABELS.get(method, method)
                    
                    ax1.plot(path[:, 0], path[:, 1], 
                            color=color, label=label, linewidth=1.5, alpha=0.8)
            
            # Start and goal markers
            start = maps.get('start')
            goal = maps.get('goal')
            if start is not None:
                ax1.plot(start[0], start[1], 'go', markersize=12, 
                        markeredgecolor='white', markeredgewidth=2, label='Start', zorder=10)
            if goal is not None:
                ax1.plot(goal[0], goal[1], 'r*', markersize=15,
                        markeredgecolor='white', markeredgewidth=2, label='Goal', zorder=10)
            
            ax1.set_title(f'Terrain (seed {seed})', fontsize=12)
            ax1.legend(loc='upper left', fontsize=8)
            ax1.set_xlabel('X (cells)')
            ax1.set_ylabel('Y (cells)')
            
            # Panel 2: Elevation
            ax2 = axes[1]
            elevation = maps.get('elevation', maps.get('elevation_map'))
            if elevation is not None:
                im2 = ax2.imshow(elevation.T, cmap='terrain', origin='lower')
                plt.colorbar(im2, ax=ax2, label='Elevation (m)')
            ax2.set_title('Elevation', fontsize=12)
            ax2.set_xlabel('X (cells)')
            ax2.set_ylabel('Y (cells)')
            
            # Panel 3: Risk or Slope
            ax3 = axes[2]
            risk = maps.get('risk_map', maps.get('risk'))
            if risk is not None:
                im3 = ax3.imshow(risk.T, cmap='Reds', origin='lower')
                plt.colorbar(im3, ax=ax3, label='Risk')
                ax3.set_title('Risk Map', fontsize=12)
            elif elevation is not None:
                # Compute slope magnitude
                gy, gx = np.gradient(elevation)
                slope = np.sqrt(gx**2 + gy**2)
                im3 = ax3.imshow(slope.T, cmap='Reds', origin='lower')
                plt.colorbar(im3, ax=ax3, label='Slope magnitude')
                ax3.set_title('Slope Magnitude', fontsize=12)
            ax3.set_xlabel('X (cells)')
            ax3.set_ylabel('Y (cells)')
            
            plt.suptitle(f'Scenario Seed {seed}', fontsize=14, fontweight='bold')
            plt.tight_layout()

            plt.savefig(self.maps_dir / f'terrain_elevation_slope_seed{seed}.png',
                       dpi=300, bbox_inches='tight')
            plt.savefig(self.maps_dir / f'terrain_elevation_slope_seed{seed}.pdf',
                       bbox_inches='tight')
            # Canonical Figure 1 for paper (Appendix B: figures/terrain_elevation_slope.png)
            if representative_seed and seed == representative_seed:
                plt.savefig(self.figures_dir / 'terrain_elevation_slope.png', dpi=300, bbox_inches='tight')
                plt.savefig(self.figures_dir / 'terrain_elevation_slope.pdf', bbox_inches='tight')
            plt.close()
    
    def save_csv_summary(self, summary: Dict):
        """Save summary as CSV."""
        with open(self.output_dir / 'summary.csv', 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Method', 'Success_Rate_%', 'N_Success', 'N_Total',
                           'Energy_Mean_kJ', 'Energy_Std', 'Time_Mean_min', 'Time_Std',
                           'Distance_Mean_km', 'Distance_Std', 'Runtime_Mean_s', 'Runtime_Std'])
            for method, s in summary.items():
                rate = s.get('success_rate', 0)
                if isinstance(rate, float) and rate <= 1:
                    rate = rate * 100
                writer.writerow([
                    method,
                    f"{rate:.1f}",
                    s.get('n_success', 0),
                    s.get('n_total', 0),
                    f"{s['energy_mean_kJ']:.1f}" if s.get('energy_mean_kJ') else '',
                    f"{s['energy_std_kJ']:.1f}" if s.get('energy_std_kJ') else '',
                    f"{s['time_mean_min']:.2f}" if s.get('time_mean_min') else '',
                    f"{s['time_std_min']:.2f}" if s.get('time_std_min') else '',
                    f"{s['distance_mean_km']:.2f}" if s.get('distance_mean_km') else '',
                    f"{s['distance_std_km']:.2f}" if s.get('distance_std_km') else '',
                    f"{s['runtime_mean_s']:.2f}" if s.get('runtime_mean_s') is not None else '',
                    f"{s['runtime_std_s']:.2f}" if s.get('runtime_std_s') is not None else '',
                ])
    
    def print_comparison_with_paper(self, summary: Dict):
        """Print comparison with original paper results."""
        print("\n" + "="*70)
        print("COMPARISON WITH ORIGINAL PAPER")
        print("="*70)
        
        # Original paper results (from Table II)
        paper_results = {
            'full_map_energy': {'success': 100.0, 'energy': 5033.6, 'time': 7.45},
            'fov_energy': {'success': 23.3, 'energy': 6547.8, 'time': 10.28},
            'fov_ga': {'success': 30.0, 'energy': 7000.4, 'time': 10.97},
        }
        
        print(f"{'Method':<20} {'Paper Success':>15} {'New Success':>15} {'Improvement':>15}")
        print("-"*70)
        
        for method in ['full_map_energy', 'fov_energy', 'fov_ga']:
            if method not in summary:
                continue
            
            paper = paper_results.get(method, {})
            paper_success = paper.get('success', 0)
            
            new_rate = summary[method].get('success_rate', 0)
            if isinstance(new_rate, float) and new_rate <= 1:
                new_rate = new_rate * 100
            
            improvement = new_rate - paper_success
            
            print(f"{method:<20} {paper_success:>14.1f}% {new_rate:>14.1f}% {improvement:>+14.1f}%")
        
        print("="*70)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Generate figures from existing results')
    parser.add_argument('--input', type=str, default='paper_results', 
                       help='Input directory with scenario results')
    parser.add_argument('--output', type=str, default='publication_figures',
                       help='Output directory for figures')
    
    args = parser.parse_args()
    
    visualizer = ResultsVisualizer(args.input, args.output)
    visualizer.load_all_data()
    visualizer.generate_all()
    
    # Print comparison with paper
    summary = visualizer.compute_statistics()
    visualizer.print_comparison_with_paper(summary)


if __name__ == '__main__':
    main()
