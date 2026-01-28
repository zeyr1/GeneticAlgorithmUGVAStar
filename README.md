# UGV Navigation System - Modular Architecture v2.0

**Surrogate-Assisted Receding-Horizon Planning Under Field-of-View Constraints**

A comprehensive system for energy-time-distance trade-off optimization in unmanned ground vehicle (UGV) navigation with limited sensing range. This implementation supports multiple planning strategies, adaptive field-of-view (FoV) management, and surrogate-assisted optimization for scalable evaluation.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Usage Examples](#usage-examples)
- [Generating Paper Results](#generating-paper-results)
- [Project Structure](#project-structure)
- [Configuration](#configuration)
- [Results Examples](#results-examples)
- [Troubleshooting](#troubleshooting)
- [Citation](#citation)

---

## 🎯 Overview

This system implements receding-horizon planning for UGV navigation under explicit FoV constraints. It compares multiple methods:

- **Full-map baselines** (complete environmental knowledge)
- **FoV-constrained methods** (realistic partial observability)
- **GA-refined methods** (local optimization)
- **Surrogate-assisted methods** (scalable evaluation)

The system generates publication-ready figures and tables for research papers, including performance comparisons, failure taxonomies, and Pareto frontiers.

---

## ✨ Key Features

### 🎯 Core Capabilities

- **Adaptive FoV**: Dynamic sensing range adjustment (15-150 cells, ~30-300m)
- **Multi-Strategy Recovery**: Dead-end handling with 5 recovery strategies
- **Global Memory**: Visited cell tracking and loop prevention
- **Modular Architecture**: SOLID-compliant, easily extensible
- **Real-time Visualization**: Live monitoring and debugging tools
- **Publication-Ready Outputs**: Automatic figure and table generation

### 📊 Planning Methods

| Method | Description | Use Case |
|--------|-------------|----------|
| `full_map_energy` | Full-map A* (energy-optimal) | Baseline comparison |
| `full_map_time` | Full-map A* (time-optimal) | Baseline comparison |
| `fov_energy` | FoV-constrained + Recovery | Realistic navigation |
| `fov_time` | FoV-constrained (time mode) | Fast traversal |
| `fov_ga` | FoV + GA refinement | Local optimization |
| `fov_ga_surrogate` | FoV + GA + Surrogate | Scalable evaluation |

---

## 📦 Installation

### Requirements

- Python 3.10+
- NumPy
- SciPy
- Matplotlib
- scikit-learn (for surrogate models)

### Setup

```bash
# Clone or download the repository
cd ugv_nav_modular_v2

# Install dependencies (if using a virtual environment)
pip install numpy scipy matplotlib scikit-learn

# Verify installation
python -c "from ugv_nav import Config, ExperimentRunner; print('Installation successful!')"
```

---

## 🚀 Quick Start

### 1. Test Single Scenario

```bash
python -m ugv_nav.main test --seed 42 --verbose
```

**Expected Output:**
```
[Seed 42] Environment created: start=(150, 150), goal=(850, 850)
[Seed 42] Running full_map_energy...
[Seed 42] full_map_energy: success (0.23s)
[Seed 42] Running fov_energy...
[Seed 42] fov_energy: success (5.81s)
```

### 2. Run Small Test Suite (3 scenarios)

```bash
python -m ugv_nav.main suite --num_scenarios 3 --seed_base 42 --fov_radius 15 --output test_results
```

### 3. Generate Figures from Results

```bash
python -m ugv_nav.create_figures_from_results --input test_results --output test_figures
```

**Output Structure:**
```
test_figures/
├── figures/          # Publication-ready figures (PNG + PDF)
├── tables/           # LaTeX tables (.tex) and text versions (.txt)
├── maps/             # Scenario visualizations
└── summary.csv       # CSV summary
```

---

## 📖 Usage Examples

### Example 1: Run Full Experiment Suite

Generate complete results for 30 scenarios (as in the paper):

```bash
python -m ugv_nav.main suite \
    --num_scenarios 30 \
    --seed_base 42 \
    --fov_radius 15 \
    --output paper_results
```

**What happens:**
- Creates `paper_results/` directory
- Runs 6 methods × 30 scenarios = 180 runs
- Saves logs, maps, and paths for each scenario
- Generates `aggregated_results.json` with summary statistics

**Time:** ~30-60 minutes (depending on hardware)

### Example 2: Generate Publication Figures

After running experiments, create all figures and tables:

```bash
python -m ugv_nav.create_figures_from_results \
    --input paper_results \
    --output my_figures
```

**Generated Outputs:**

#### Figures (`my_figures/figures/`)
- `terrain_elevation_slope.png` - Representative scenario (Figure 1)
- `failure_taxonomy.png` - Success/failure breakdown (Figure 2)
- `pareto_energy_time.png` - Energy-time trade-offs (Figure 3)
- `distance_comparison.png` - Path distance comparison (Figure 4)
- `runtime_comparison.png` - Computational cost (Figure 5)
- `success_rate.png` - Success rate comparison

#### Tables (`my_figures/tables/`)
- `table_parameters.tex` - Key parameters (Table I)
- `table_results.tex` - Performance summary (Table II)
- `table_ga_stats.tex` - GA statistics (Table III)

### Example 3: Custom Configuration

```python
from ugv_nav import Config, ExperimentRunner

# Create custom configuration
config = Config()
config.fov.base_radius_cells = 20  # Larger FoV
config.recovery.enabled = True
config.recovery.max_recovery_attempts = 3

# Run experiments
runner = ExperimentRunner(config)
results = runner.run_suite(
    num_scenarios=10,
    seed_base=100,
    output_dir='custom_results'
)

# Print summary
print(f"Success rates: {results.summary}")
```

### Example 4: Debug Mode with Visualization

```bash
python -m ugv_nav.main debug --seed 42 --fov_radius 15
```

Opens real-time visualization showing:
- Terrain map with trajectories
- Current FoV window
- Energy consumption over time
- Recovery attempts

---

## 📊 Generating Paper Results

### Complete Workflow

**Step 1: Run Experiments**
```bash
python -m ugv_nav.main suite \
    --num_scenarios 30 \
    --seed_base 42 \
    --fov_radius 15 \
    --output paper_results \
    --verbose
```

**Step 2: Generate Figures and Tables**
```bash
python -m ugv_nav.create_figures_from_results \
    --input paper_results \
    --output my_figures
```

**Step 3: Check Outputs**
```bash
# View summary table
type my_figures\tables\table_results.txt

# View parameters
type my_figures\tables\table_parameters.txt

# Check figures
dir my_figures\figures
```

### Parallel Execution (Faster)

For multi-core systems:

```bash
python -m ugv_nav.main suite \
    --num_scenarios 30 \
    --seed_base 42 \
    --fov_radius 15 \
    --output paper_results \
    --parallel \
    --workers 4
```

**Speedup:** ~3-4x on 4-core systems

---

## 📁 Project Structure

```
ugv_nav_modular_v2/
├── ugv_nav/                    # Main package
│   ├── config/                 # Configuration management
│   │   └── settings.py         # All parameters (FoV, recovery, GA, etc.)
│   ├── terrain/                # Terrain modeling
│   │   ├── types.py            # TerrainType enum
│   │   └── generator.py        # Procedural map generation
│   ├── environment/            # Environment representation
│   │   ├── world.py            # Global Environment
│   │   └── local_view.py       # FoV-constrained LocalEnvironment
│   ├── energy/                 # Energy model
│   │   └── model.py            # Physics-based energy calculation
│   ├── planning/               # Planning algorithms
│   │   ├── astar.py            # A* planner
│   │   └── receding_horizon.py # Main control loop
│   ├── recovery/               # Recovery system
│   │   ├── adaptive_fov.py     # Adaptive FoV management
│   │   └── strategies.py       # Recovery strategies
│   ├── optimization/           # GA and Surrogate
│   │   ├── ga/                 # Genetic algorithm
│   │   │   ├── solver.py       # GA solver
│   │   │   └── individual.py   # Genome representation
│   │   └── surrogate/          # Surrogate models
│   │       └── model.py        # Random forest surrogate
│   ├── metrics/                # Metrics and classification
│   │   └── path_metrics.py     # Path evaluation
│   ├── visualization/          # Live monitoring
│   │   └── monitor.py          # Debug visualization
│   ├── pipeline/               # Experiment management
│   │   └── runner.py           # ExperimentRunner
│   ├── main.py                 # CLI entry point
│   ├── create_figures_from_results.py  # Figure generation
│   └── test_system.py          # Test script
├── paper_results/              # Experiment outputs (after running)
│   ├── aggregated_results.json
│   └── scenario_00042/
│       ├── logs.json
│       ├── maps.npz
│       └── paths.npz
├── my_figures/                 # Generated figures (after create_figures)
│   ├── figures/                # Publication figures
│   ├── tables/                 # LaTeX tables
│   └── maps/                   # Scenario visualizations
└── README.md                   # This file
```

---

## ⚙️ Configuration

### Key Parameters

```python
from ugv_nav import Config

config = Config()

# FoV settings
config.fov.base_radius_cells = 15      # ~30m at 2m/cell
config.fov.min_radius_cells = 10
config.fov.max_radius_cells = 150      # Expansion limit
config.fov.base_execute_steps = 12     # Steps per replan

# Recovery settings
config.recovery.enabled = True
config.recovery.max_recovery_attempts = 5
config.recovery.strategies = (
    'expand_fov',      # Expand sensing range
    'backtrack',       # Go back along path
    'random_escape',   # Random exploration
    'global_replan',   # Coarse global replan
    'wall_follow'      # Follow obstacles
)

# Unknown terrain model
config.unknown.mode = 'adaptive'  # 'optimistic', 'balanced', 'pessimistic'

# GA settings
config.ga.pop_size = 50
config.ga.generations = 35
config.ga.weights = {'energy': 0.4, 'time': 0.3, 'safety': 0.0}

# Surrogate settings
config.surrogate.enabled = True
config.surrogate.warmup_generations = 5
config.surrogate.surrogate_fraction = 0.80
```

### Default Values (Paper Settings)

| Parameter | Value | Description |
|-----------|-------|-------------|
| Map size | 1000×1000 cells (2km×2km) | Grid resolution |
| Cell size | 2 m | Spatial resolution |
| FoV radius | 15 cells (~30m) | Sensing range |
| Execute steps | K=12 | Steps per replan |
| Speed limits | asphalt: 15, grass: 8, mud: 3, sand: 5 m/s | Terrain-dependent |
| Friction coeffs | asphalt: 0.015, grass: 0.08, mud: 0.25, sand: 0.15 | Rolling resistance |
| Accel bounds | ±2.5 m/s² (max), -4.0 m/s² (min) | Kinodynamic limits |
| Max turn angle | 30° per step | Steering constraint |
| Unknown terrain | friction: 0.5, risk: 1.0, elevation: 40m | Conservative defaults |
| GA weights | (w_E, w_T, w_S) = (0.4, 0.3, 0.0) | Objective weights |
| Scenarios | N=30 | Number of test scenarios |

---

## 📈 Results Examples

### Example Results from 30 Scenarios

Based on actual runs with seed_base=42, fov_radius=15:

#### Performance Summary (Table II)

```
Method                      Success    Energy (kJ)   Time (min)  Distance (km)  Runtime (s)
----------------------------------------------------------------------------------------------------
full_map_time           100.0% (30/30)         5334.5        4.77          2.17        0.23
full_map_energy         100.0% (30/30)         3894.9        4.76          2.51      198.52
fov_time                 76.7% (23/30)         5334.7        4.76          2.18        1.59
fov_energy               73.3% (22/30)         4852.2        5.67          2.22        5.81
fov_ga                   73.3% (22/30)         4852.2        5.67          2.22        6.04
fov_ga_surrogate         73.3% (22/30)         4852.2        5.67          2.22        6.33
```

**Key Observations:**
- Full-map methods achieve **100% success** (complete information)
- FoV methods show **73-77% success** (partial observability challenge)
- Successful FoV trajectories achieve **comparable distance** (2.18-2.22 km vs 2.17-2.51 km)
- Runtime overhead: FoV methods ~1.6-6.3s vs full-map 0.2-198s

#### Failure Taxonomy

From `failure_taxonomy.png`:
- **Full-map**: 100% success, 0% failures
- **FoV methods**: ~23-27% failures, primarily:
  - **Dead-end** (most common): Limited sensing causes commitment to impassable paths
  - **Timeout**: Exceeded maximum planning time
  - **Collision**: Obstacle encounters
  - **Backtracking**: Excessive backtracking detected

#### Energy-Time Trade-offs

From `pareto_energy_time.png`:
- **Full-map energy**: Lowest energy (3895 kJ) but longer time (4.76 min)
- **Full-map time**: Fastest (4.77 min) but higher energy (5335 kJ)
- **FoV methods**: Clustered in middle range, showing compressed trade-off space

#### Distance Comparison

From `distance_comparison.png`:
- Successful FoV trajectories achieve **comparable or better** geometric efficiency
- This reflects selection bias: only easier scenarios succeed under limited observability

#### Runtime Analysis

From `runtime_comparison.png`:
- **Full-map time**: Fastest (0.23s) - simple A* search
- **Full-map energy**: Slower (198.5s) - more complex cost evaluation
- **FoV methods**: Moderate overhead (1.6-6.3s) - replanning loop cost

### CSV Summary

The `summary.csv` file contains all metrics:

```csv
Method,Success_Rate_%,N_Success,N_Total,Energy_Mean_kJ,Energy_Std,Time_Mean_min,Time_Std,Distance_Mean_km,Distance_Std,Runtime_Mean_s,Runtime_Std
full_map_energy,100.0,30,30,3894.9,513.8,4.76,0.79,2.51,0.14,198.52,81.54
full_map_time,100.0,30,30,5334.5,351.9,4.77,0.41,2.17,0.09,0.23,0.08
fov_energy,73.3,22,30,4852.2,452.2,5.67,0.72,2.22,0.10,5.81,4.42
...
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. No Output in Terminal

**Problem:** Terminal shows nothing while running.

**Solutions:**
```bash
# Use unbuffered mode
python -u -m ugv_nav.main suite --num_scenarios 30 --output paper_results

# Or set environment variable
$env:PYTHONUNBUFFERED=1
python -m ugv_nav.main suite --num_scenarios 30 --output paper_results

# Check if files are being created
dir paper_results
```

#### 2. Missing Runtime in Table II

**Problem:** Runtime column shows "---" in LaTeX table.

**Solution:** Ensure `logs.json` files contain `"runtimes"` field. The pipeline runner automatically adds this. If using old results, re-run experiments.

#### 3. Empty Table III (GA Stats)

**Problem:** `true_evals/replan` shows "---".

**Solution:** This is expected if GA integration is not fully active. The table structure is ready; it will populate when GA logging is implemented.

#### 4. Unicode Errors on Windows

**Problem:** `UnicodeEncodeError` when printing.

**Solution:** The code uses ASCII-safe markers (`[ok]` instead of `✓`). If issues persist:
```bash
$env:PYTHONIOENCODING='utf-8'
python -m ugv_nav.create_figures_from_results --input paper_results --output my_figures
```

#### 5. Long Runtime

**Problem:** Experiments take too long.

**Solutions:**
```bash
# Use parallel execution
python -m ugv_nav.main suite --num_scenarios 30 --parallel --workers 4

# Reduce scenarios for testing
python -m ugv_nav.main suite --num_scenarios 5 --output test_results
```

#### 6. Missing Figures

**Problem:** Some figures not generated.

**Check:**
- Ensure `paper_results/` contains `scenario_*/` folders
- Verify `logs.json`, `maps.npz`, `paths.npz` exist in each scenario folder
- Check for errors in terminal output

---

## 📝 Citation

If you use this code in your research, please cite:

```bibtex
@article{ugv_nav_2025,
  title={Surrogate-Assisted Receding-Horizon Planning Under Field-of-View Constraints: Energy--Time--Distance Trade-offs for Unmanned Ground Vehicle Navigation},
  author={Oğurlu, Ahmet Berke and Arcan, Umut İkbal},
  journal={[Journal Name]},
  year={2025}
}
```

---

## 📧 Contact

For questions, issues, or contributions:

- **Email**: berkeogurlu@gmail.com
- **Repository**: [GitHub URL]

---

## 📄 License

MIT License - See LICENSE file for details.

---

## 🙏 Acknowledgments

- Yağız Berkutay Ayhan for optimization assistance
- Contributors to the UGV Navigation project

---

**Last Updated:** January 2025  
**Version:** 2.0.0
