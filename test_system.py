#!/usr/bin/env python3
"""
Quick Test Script for UGV Navigation System
============================================

Tests all modules can be imported and basic functionality works.
"""

import sys
import os
import traceback

# Fix path - add parent directory so 'ugv_nav' package is found
script_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(script_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


def test_imports():
    """Test all module imports"""
    print("Testing imports...")
    
    try:
        from ugv_nav.config import Config, GAConfig, FoVConfig
        print("  ✓ config")
    except Exception as e:
        print(f"  ✗ config: {e}")
        return False
    
    try:
        from ugv_nav.terrain import TerrainType, MapGenerator, generate_start_goal
        print("  ✓ terrain")
    except Exception as e:
        print(f"  ✗ terrain: {e}")
        return False
    
    try:
        from ugv_nav.environment import Environment, LocalEnvironment, create_local_environment
        print("  ✓ environment")
    except Exception as e:
        print(f"  ✗ environment: {e}")
        return False
    
    try:
        from ugv_nav.energy import EnergyModel, VelocityProfiler
        print("  ✓ energy")
    except Exception as e:
        print(f"  ✗ energy: {e}")
        return False
    
    try:
        from ugv_nav.metrics import PathMetrics, RunClassifier, RunResult
        print("  ✓ metrics")
    except Exception as e:
        print(f"  ✗ metrics: {e}")
        return False
    
    try:
        from ugv_nav.planning import AStarPlanner, RecedingHorizonController
        print("  ✓ planning")
    except Exception as e:
        print(f"  ✗ planning: {e}")
        return False
    
    try:
        from ugv_nav.recovery import RecoveryManager, AdaptiveFoV
        print("  ✓ recovery")
    except Exception as e:
        print(f"  ✗ recovery: {e}")
        return False
    
    try:
        from ugv_nav.optimization import LocalGASolver, SurrogateModel
        print("  ✓ optimization")
    except Exception as e:
        print(f"  ✗ optimization: {e}")
        return False
    
    try:
        from ugv_nav.pipeline import ExperimentRunner
        print("  ✓ pipeline")
    except Exception as e:
        print(f"  ✗ pipeline: {e}")
        return False
    
    print("All imports successful!\n")
    return True


def test_config():
    """Test configuration"""
    print("Testing configuration...")
    
    from ugv_nav.config import Config
    
    config = Config()
    
    assert config.map.grid_size == 1000, "Grid size should be 1000"
    assert config.fov.base_radius_cells == 25, "Base FoV radius should be 25"
    assert config.vehicle.mass == 1800, "Vehicle mass should be 1800"
    
    print(f"  Map size: {config.map.size_meters}m ({config.map.grid_size} cells)")
    print(f"  FoV radius: {config.fov.base_radius_cells} cells")
    print(f"  Vehicle mass: {config.vehicle.mass} kg")
    
    print("Configuration test passed!\n")
    return True


def test_terrain_generation():
    """Test terrain generation"""
    print("Testing terrain generation...")
    
    from ugv_nav.terrain import MapGenerator, TerrainType
    from ugv_nav.config import Config
    
    config = Config()
    generator = MapGenerator(config.map, seed=42)
    
    generated = generator.generate()
    
    assert generated.terrain.shape == (1000, 1000), "Terrain shape mismatch"
    assert generated.elevation.shape == (1000, 1000), "Elevation shape mismatch"
    
    # Count terrain types
    terrain_counts = {}
    for t in TerrainType:
        count = (generated.terrain == t).sum()
        terrain_counts[t.name] = count
    
    print(f"  Terrain distribution:")
    for name, count in terrain_counts.items():
        pct = count / (1000*1000) * 100
        print(f"    {name}: {pct:.1f}%")
    
    print("Terrain generation test passed!\n")
    return True


def test_environment():
    """Test environment creation"""
    print("Testing environment...")
    
    from ugv_nav.config import Config
    from ugv_nav.environment import Environment
    
    config = Config()
    env = Environment(config, seed=42)
    
    print(f"  Start: {env.start}")
    print(f"  Goal: {env.goal}")
    print(f"  Distance: {env.distance_to_goal(env.start):.0f}m")
    
    assert env.is_valid(*env.start), "Start should be valid"
    assert env.is_valid(*env.goal), "Goal should be valid"
    
    print("Environment test passed!\n")
    return True


def test_astar():
    """Test A* pathfinding"""
    print("Testing A* pathfinding...")
    
    from ugv_nav.config import Config
    from ugv_nav.environment import Environment
    from ugv_nav.energy import EnergyModel
    from ugv_nav.planning import AStarPlanner
    
    config = Config()
    env = Environment(config, seed=42)
    energy_model = EnergyModel(env, config)
    
    planner = AStarPlanner(
        env=env,
        energy_model=energy_model,
        config=config,
        mode='energy',
        epsilon=1.0
    )
    
    path = planner.plan(env.start, env.goal)
    
    if path:
        print(f"  ✓ Path found: {len(path)} nodes")
        print(f"  Start: {path[0]}, End: {path[-1]}")
        
        # Verify path
        assert path[0] == env.start, "Path should start at start"
        assert path[-1] == env.goal, "Path should end at goal"
        
        # Check path continuity
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            dist = max(abs(x2 - x1), abs(y2 - y1))
            assert dist <= 1, f"Path discontinuity at {i}"
        
        print("A* pathfinding test passed!\n")
        return True
    else:
        print("  ✗ No path found!")
        return False


def test_recovery():
    """Test recovery system"""
    print("Testing recovery system...")
    
    from ugv_nav.config import Config
    from ugv_nav.recovery import AdaptiveFoV, RecoveryManager
    
    config = Config()
    
    # Test adaptive FoV
    fov = AdaptiveFoV(base_radius=25, max_radius=150)
    
    assert fov.current_radius == 25, "Initial radius should be 25"
    
    # Simulate stuck
    fov.update(success=False, backtrack_ratio=0.3)
    assert fov.current_radius > 25, "Radius should increase when stuck"
    
    print(f"  FoV after stuck: {fov.current_radius}")
    
    # Test recovery manager creation
    recovery = RecoveryManager(config, seed=42)
    assert recovery is not None
    
    print("Recovery system test passed!\n")
    return True


def test_receding_horizon():
    """Test receding horizon controller"""
    print("Testing receding horizon controller...")
    
    from ugv_nav.config import Config
    from ugv_nav.environment import Environment
    from ugv_nav.planning import RecedingHorizonController
    from ugv_nav.recovery import RecoveryManager, AdaptiveFoV
    
    config = Config()
    config.max_iterations = 500  # Limit for test
    config.max_total_seconds = 30
    
    env = Environment(config, seed=42)
    
    adaptive_fov = AdaptiveFoV(
        base_radius=config.fov.base_radius_cells,
        max_radius=config.fov.max_radius_cells
    )
    
    recovery_manager = RecoveryManager(config, seed=42)
    
    controller = RecedingHorizonController(
        env=env,
        config=config,
        recovery_manager=recovery_manager,
        adaptive_fov=adaptive_fov
    )
    
    print("  Running limited iteration test...")
    
    result = controller.run(
        mode='energy',
        max_iterations=100,  # Very limited for quick test
        max_time=10
    )
    
    print(f"  Status: {result.status}")
    print(f"  Path length: {len(result.path)}")
    print(f"  Replans: {result.replans}")
    print(f"  Recovery attempts: {result.recovery_attempts}")
    
    print("Receding horizon test passed!\n")
    return True


def run_all_tests():
    """Run all tests"""
    print("=" * 60)
    print("UGV NAVIGATION SYSTEM - MODULE TESTS")
    print("=" * 60 + "\n")
    
    tests = [
        ("Imports", test_imports),
        ("Configuration", test_config),
        ("Terrain Generation", test_terrain_generation),
        ("Environment", test_environment),
        ("A* Pathfinding", test_astar),
        ("Recovery System", test_recovery),
        ("Receding Horizon", test_receding_horizon),
    ]
    
    results = []
    
    for name, test_func in tests:
        try:
            success = test_func()
            results.append((name, success, None))
        except Exception as e:
            print(f"  ✗ EXCEPTION: {e}")
            traceback.print_exc()
            results.append((name, False, str(e)))
    
    # Summary
    print("=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, s, _ in results if s)
    total = len(results)
    
    for name, success, error in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status}: {name}")
        if error:
            print(f"         Error: {error}")
    
    print(f"\nTotal: {passed}/{total} tests passed")
    print("=" * 60)
    
    return passed == total


if __name__ == '__main__':
    success = run_all_tests()
    sys.exit(0 if success else 1)
