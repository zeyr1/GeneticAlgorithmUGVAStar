"""
Visualization Module
====================

Real-time visualization, debugging, and analysis tools.
Helps understand where and why the robot gets stuck.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Circle, FancyArrowPatch
import matplotlib.animation as animation
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass
import time
import os


@dataclass
class VisualizationConfig:
    """Configuration for visualization"""
    terrain_colors: tuple = ('darkgray', 'lightgreen', 'saddlebrown', 'gold', 'black')
    path_colors: Dict[str, str] = None
    figure_size: Tuple[int, int] = (12, 10)
    dpi: int = 100
    save_frames: bool = False
    frame_dir: str = 'frames'
    
    def __post_init__(self):
        if self.path_colors is None:
            self.path_colors = {
                'planned': 'blue',
                'executed': 'red',
                'recovery': 'cyan',
                'failed': 'orange',
                'fov': 'yellow'
            }


class TerrainVisualizer:
    """
    Static terrain visualization.
    
    Creates publication-ready terrain maps with paths overlaid.
    """
    
    def __init__(self, env, config: Optional[VisualizationConfig] = None):
        """
        Initialize visualizer.
        
        Args:
            env: Environment object
            config: Visualization configuration
        """
        self.env = env
        self.config = config or VisualizationConfig()
        self.cmap = ListedColormap(self.config.terrain_colors)
    
    def plot_terrain(self, ax=None, show_elevation: bool = False) -> plt.Axes:
        """
        Plot terrain map.
        
        Args:
            ax: Matplotlib axes (creates new if None)
            show_elevation: Overlay elevation contours
        
        Returns:
            Matplotlib axes
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        # Plot terrain
        ax.imshow(
            self.env.terrain.T,
            cmap=self.cmap,
            origin='lower',
            alpha=0.8
        )
        
        # Elevation contours
        if show_elevation:
            contours = ax.contour(
                self.env.elevation.T,
                levels=10,
                colors='white',
                alpha=0.3,
                linewidths=0.5
            )
        
        # Start and goal markers
        ax.plot(self.env.start[0], self.env.start[1], 
               'go', markersize=15, markeredgecolor='white', 
               markeredgewidth=2, label='Start')
        ax.plot(self.env.goal[0], self.env.goal[1], 
               'r*', markersize=20, markeredgecolor='white',
               markeredgewidth=2, label='Goal')
        
        ax.set_xlabel('X (cells)')
        ax.set_ylabel('Y (cells)')
        ax.legend(loc='upper right')
        
        return ax
    
    def plot_path(self, ax, path: List[Tuple[int, int]], 
                  color: str = 'blue', label: str = None,
                  linewidth: float = 2.0, alpha: float = 0.8):
        """Plot a path on existing axes"""
        if not path or len(path) < 2:
            return
        
        path_arr = np.array(path)
        ax.plot(path_arr[:, 0], path_arr[:, 1], 
               color=color, linewidth=linewidth, 
               alpha=alpha, label=label)
    
    def plot_fov(self, ax, position: Tuple[int, int], radius: int,
                 color: str = 'yellow', alpha: float = 0.2):
        """Plot FoV circle"""
        circle = Circle(position, radius, 
                       fill=True, color=color, alpha=alpha)
        ax.add_patch(circle)
    
    def plot_failed_cells(self, ax, failed_cells: set,
                          color: str = 'red', marker: str = 'x'):
        """Plot failed/blocked cells"""
        if not failed_cells:
            return
        
        cells = np.array(list(failed_cells))
        ax.scatter(cells[:, 0], cells[:, 1], 
                  c=color, marker=marker, s=50, alpha=0.7,
                  label='Failed cells')
    
    def create_comparison_figure(self, paths: Dict[str, List],
                                metrics: Dict[str, Dict] = None,
                                title: str = 'Path Comparison') -> plt.Figure:
        """
        Create multi-panel comparison figure.
        
        Args:
            paths: Dict of method_name -> path
            metrics: Dict of method_name -> metrics dict
            title: Figure title
        
        Returns:
            Matplotlib figure
        """
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.3)
        
        # Main terrain map with all paths
        ax_map = fig.add_subplot(gs[0, 0])
        self.plot_terrain(ax_map, show_elevation=True)
        
        colors = ['blue', 'red', 'green', 'purple', 'orange', 'cyan']
        for i, (name, path) in enumerate(paths.items()):
            if path:
                self.plot_path(ax_map, path, 
                             color=colors[i % len(colors)],
                             label=name, linewidth=2)
        
        ax_map.legend(loc='lower left')
        ax_map.set_title(f'{title} - Paths')
        
        # Elevation profile
        ax_elev = fig.add_subplot(gs[0, 1])
        ax_elev.imshow(self.env.elevation.T, cmap='terrain', origin='lower')
        ax_elev.set_title('Elevation Map')
        ax_elev.set_xlabel('X (cells)')
        ax_elev.set_ylabel('Y (cells)')
        
        # Metrics comparison (if provided)
        if metrics:
            ax_metrics = fig.add_subplot(gs[1, 0])
            self._plot_metrics_bars(ax_metrics, metrics)
        
        # Risk map
        ax_risk = fig.add_subplot(gs[1, 1])
        im = ax_risk.imshow(self.env.risk_map.T, cmap='Reds', origin='lower')
        plt.colorbar(im, ax=ax_risk, label='Risk')
        ax_risk.set_title('Risk Map')
        
        plt.suptitle(title, fontsize=14, fontweight='bold')
        
        return fig
    
    def _plot_metrics_bars(self, ax, metrics: Dict[str, Dict]):
        """Plot metrics comparison as bar chart"""
        methods = list(metrics.keys())
        
        # Extract common metrics
        metric_names = ['total_energy_kJ', 'total_time_min', 'total_distance_km']
        display_names = ['Energy (kJ)', 'Time (min)', 'Distance (km)']
        
        x = np.arange(len(methods))
        width = 0.25
        
        for i, (metric, display) in enumerate(zip(metric_names, display_names)):
            values = []
            for method in methods:
                m = metrics[method]
                if metric in m:
                    values.append(m[metric])
                elif hasattr(m, metric.replace('_', '')):
                    values.append(getattr(m, metric.replace('_', '')))
                else:
                    values.append(0)
            
            ax.bar(x + i * width, values, width, label=display)
        
        ax.set_xlabel('Method')
        ax.set_ylabel('Value')
        ax.set_xticks(x + width)
        ax.set_xticklabels(methods, rotation=45, ha='right')
        ax.legend()
        ax.set_title('Metrics Comparison')
    
    def save_figure(self, fig: plt.Figure, filename: str, dpi: int = None):
        """Save figure to file"""
        fig.savefig(filename, dpi=dpi or self.config.dpi, 
                   bbox_inches='tight', facecolor='white')


class LiveMonitor:
    """
    Real-time visualization for debugging navigation.
    
    Shows:
    - Current robot position
    - FoV boundary
    - Planned vs executed path
    - Failed cells
    - Recovery attempts
    - Statistics overlay
    
    CRITICAL: This helps identify where and why the robot gets stuck!
    """
    
    def __init__(self, env, config: Optional[VisualizationConfig] = None):
        """
        Initialize live monitor.
        
        Args:
            env: Environment object
            config: Visualization configuration
        """
        self.env = env
        self.config = config or VisualizationConfig()
        
        self.fig = None
        self.ax = None
        self.cmap = ListedColormap(self.config.terrain_colors)
        
        # Plot elements (updated dynamically)
        self._robot_marker = None
        self._fov_circle = None
        self._executed_line = None
        self._planned_line = None
        self._failed_scatter = None
        self._stats_text = None
        self._title_text = None
        
        # History for analysis
        self.position_history: List[Tuple[int, int]] = []
        self.fov_history: List[int] = []
        self.stuck_positions: List[Tuple[int, int]] = []
        
        # Frame saving
        self._frame_count = 0
        
        self._initialized = False
    
    def initialize(self, interactive: bool = True):
        """
        Initialize the plot window.
        
        Args:
            interactive: Enable interactive mode for live updates
        """
        if self._initialized:
            return
        
        if interactive:
            plt.ion()
        
        self.fig, self.ax = plt.subplots(figsize=self.config.figure_size)
        
        # Plot terrain
        self.ax.imshow(
            self.env.terrain.T,
            cmap=self.cmap,
            origin='lower',
            alpha=0.7
        )
        
        # Start and goal
        self.ax.plot(self.env.start[0], self.env.start[1],
                    'go', markersize=12, markeredgecolor='white',
                    markeredgewidth=2, label='Start', zorder=10)
        self.ax.plot(self.env.goal[0], self.env.goal[1],
                    'r*', markersize=15, markeredgecolor='white',
                    markeredgewidth=2, label='Goal', zorder=10)
        
        # Initialize dynamic elements
        self._robot_marker, = self.ax.plot([], [], 'bo', markersize=10,
                                          markeredgecolor='white',
                                          markeredgewidth=2, zorder=15,
                                          label='Robot')
        
        self._fov_circle = Circle((0, 0), 25, fill=True, 
                                  color='yellow', alpha=0.15, zorder=5)
        self.ax.add_patch(self._fov_circle)
        
        self._executed_line, = self.ax.plot([], [], 'r-', linewidth=2,
                                           alpha=0.8, label='Executed', zorder=8)
        
        self._planned_line, = self.ax.plot([], [], 'b--', linewidth=1.5,
                                          alpha=0.6, label='Planned', zorder=7)
        
        self._failed_scatter = self.ax.scatter([], [], c='orange', marker='x',
                                              s=50, alpha=0.7, zorder=9,
                                              label='Failed cells')
        
        # Stats text box
        self._stats_text = self.ax.text(
            0.02, 0.98, '', transform=self.ax.transAxes,
            verticalalignment='top', fontsize=9,
            bbox=dict(boxstyle='round', facecolor='white', alpha=0.8),
            zorder=20
        )
        
        # Title
        self._title_text = self.ax.set_title('Live Navigation Monitor')
        
        self.ax.legend(loc='upper right')
        self.ax.set_xlabel('X (cells)')
        self.ax.set_ylabel('Y (cells)')
        
        self.fig.tight_layout()
        
        self._initialized = True
        
        if interactive:
            plt.show(block=False)
            plt.pause(0.1)
    
    def update(self, 
               position: Tuple[int, int],
               fov_radius: int,
               executed_path: List[Tuple[int, int]],
               planned_path: Optional[List[Tuple[int, int]]] = None,
               failed_cells: Optional[set] = None,
               stats: Optional[Dict] = None,
               title: Optional[str] = None,
               is_stuck: bool = False,
               in_recovery: bool = False):
        """
        Update the live visualization.
        
        Args:
            position: Current robot position
            fov_radius: Current FoV radius
            executed_path: Path executed so far
            planned_path: Currently planned path (optional)
            failed_cells: Set of failed cell coordinates
            stats: Statistics dictionary to display
            title: Custom title
            is_stuck: Whether robot is currently stuck
            in_recovery: Whether robot is in recovery mode
        """
        if not self._initialized:
            self.initialize()
        
        # Track history
        self.position_history.append(position)
        self.fov_history.append(fov_radius)
        if is_stuck:
            self.stuck_positions.append(position)
        
        # Update robot position
        self._robot_marker.set_data([position[0]], [position[1]])
        
        # Change robot color based on state
        if is_stuck:
            self._robot_marker.set_color('red')
            self._robot_marker.set_markersize(15)
        elif in_recovery:
            self._robot_marker.set_color('orange')
            self._robot_marker.set_markersize(12)
        else:
            self._robot_marker.set_color('blue')
            self._robot_marker.set_markersize(10)
        
        # Update FoV circle
        self._fov_circle.center = position
        self._fov_circle.radius = fov_radius
        
        # Color FoV based on state
        if is_stuck:
            self._fov_circle.set_color('red')
            self._fov_circle.set_alpha(0.2)
        elif in_recovery:
            self._fov_circle.set_color('orange')
            self._fov_circle.set_alpha(0.2)
        else:
            self._fov_circle.set_color('yellow')
            self._fov_circle.set_alpha(0.15)
        
        # Update executed path
        if executed_path:
            path_arr = np.array(executed_path)
            self._executed_line.set_data(path_arr[:, 0], path_arr[:, 1])
        
        # Update planned path
        if planned_path:
            path_arr = np.array(planned_path)
            self._planned_line.set_data(path_arr[:, 0], path_arr[:, 1])
        else:
            self._planned_line.set_data([], [])
        
        # Update failed cells
        if failed_cells:
            cells = np.array(list(failed_cells))
            self._failed_scatter.set_offsets(cells)
        else:
            self._failed_scatter.set_offsets(np.empty((0, 2)))
        
        # Update stats
        if stats:
            stats_str = self._format_stats(stats, is_stuck, in_recovery)
            self._stats_text.set_text(stats_str)
        
        # Update title
        if title:
            status = ""
            if is_stuck:
                status = " [STUCK!]"
            elif in_recovery:
                status = " [RECOVERING...]"
            self.ax.set_title(f'{title}{status}')
        
        # Refresh display
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
        
        # Save frame if configured
        if self.config.save_frames:
            self._save_frame()
        
        # Small pause for visibility
        plt.pause(0.01)
    
    def _format_stats(self, stats: Dict, is_stuck: bool, in_recovery: bool) -> str:
        """Format statistics for display"""
        lines = []
        
        if 'iteration' in stats:
            lines.append(f"Iteration: {stats['iteration']}")
        
        if 'position' in stats:
            pos = stats['position']
            lines.append(f"Position: ({pos[0]}, {pos[1]})")
        
        if 'fov_radius' in stats:
            lines.append(f"FoV Radius: {stats['fov_radius']} cells")
        
        if 'path_length' in stats:
            lines.append(f"Path Length: {stats['path_length']}")
        
        if 'elapsed_time' in stats:
            lines.append(f"Time: {stats['elapsed_time']:.1f}s")
        
        if 'distance_to_goal' in stats:
            lines.append(f"To Goal: {stats['distance_to_goal']:.0f}m")
        
        if 'recovery_attempts' in stats:
            lines.append(f"Recoveries: {stats['recovery_attempts']}")
        
        # Status indicator
        if is_stuck:
            lines.append("\n⚠️ STUCK - Need recovery!")
        elif in_recovery:
            lines.append("\n🔄 Recovery in progress...")
        
        return '\n'.join(lines)
    
    def _save_frame(self):
        """Save current frame to file"""
        os.makedirs(self.config.frame_dir, exist_ok=True)
        filename = os.path.join(self.config.frame_dir, 
                               f'frame_{self._frame_count:05d}.png')
        self.fig.savefig(filename, dpi=100, bbox_inches='tight')
        self._frame_count += 1
    
    def mark_stuck(self, position: Tuple[int, int], reason: str = ''):
        """Mark a position where robot got stuck"""
        self.stuck_positions.append(position)
        
        # Add annotation
        self.ax.annotate(
            f'STUCK\n{reason}' if reason else 'STUCK',
            xy=position,
            xytext=(position[0] + 20, position[1] + 20),
            fontsize=8,
            color='red',
            arrowprops=dict(arrowstyle='->', color='red'),
            zorder=25
        )
    
    def show_analysis(self):
        """Show analysis of stuck positions"""
        if not self.stuck_positions:
            print("No stuck positions recorded!")
            return
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Map with stuck positions
        ax1 = axes[0]
        ax1.imshow(self.env.terrain.T, cmap=self.cmap, 
                  origin='lower', alpha=0.7)
        
        # Plot full path history
        if self.position_history:
            path_arr = np.array(self.position_history)
            ax1.plot(path_arr[:, 0], path_arr[:, 1], 
                    'b-', alpha=0.3, linewidth=1)
        
        # Highlight stuck positions
        stuck_arr = np.array(self.stuck_positions)
        ax1.scatter(stuck_arr[:, 0], stuck_arr[:, 1],
                   c='red', s=100, marker='X', 
                   label=f'Stuck ({len(self.stuck_positions)} times)')
        
        ax1.plot(self.env.start[0], self.env.start[1], 'go', 
                markersize=12, label='Start')
        ax1.plot(self.env.goal[0], self.env.goal[1], 'r*', 
                markersize=15, label='Goal')
        
        ax1.legend()
        ax1.set_title('Stuck Position Analysis')
        
        # FoV radius over time
        ax2 = axes[1]
        ax2.plot(self.fov_history, 'b-', linewidth=1)
        ax2.axhline(y=25, color='g', linestyle='--', 
                   label='Base FoV (25)')
        
        # Mark stuck moments
        for i, pos in enumerate(self.position_history):
            if pos in self.stuck_positions:
                ax2.axvline(x=i, color='red', alpha=0.3)
        
        ax2.set_xlabel('Iteration')
        ax2.set_ylabel('FoV Radius (cells)')
        ax2.set_title('FoV Radius Over Time (red = stuck)')
        ax2.legend()
        
        plt.tight_layout()
        plt.show()
        
        # Print summary
        print(f"\n{'='*50}")
        print("STUCK POSITION ANALYSIS")
        print(f"{'='*50}")
        print(f"Total stuck occurrences: {len(self.stuck_positions)}")
        print(f"Unique stuck positions: {len(set(self.stuck_positions))}")
        
        # Analyze terrain at stuck positions
        terrain_at_stuck = {}
        for pos in self.stuck_positions:
            t = ['asphalt', 'grass', 'mud', 'sand', 'wall'][
                self.env.terrain[pos[0], pos[1]]
            ]
            terrain_at_stuck[t] = terrain_at_stuck.get(t, 0) + 1
        
        print("\nTerrain at stuck positions:")
        for terrain, count in sorted(terrain_at_stuck.items(), 
                                    key=lambda x: x[1], reverse=True):
            print(f"  {terrain}: {count}")
    
    def close(self):
        """Close the monitor window"""
        if self.fig is not None:
            plt.close(self.fig)
            self._initialized = False


def create_debug_callback(monitor: LiveMonitor):
    """
    Create a progress callback function for RecedingHorizonController.
    
    Usage:
        monitor = LiveMonitor(env)
        callback = create_debug_callback(monitor)
        controller = RecedingHorizonController(env, config, progress_callback=callback)
    """
    def callback(state, info):
        # Compute distance to goal
        dist_to_goal = np.sqrt(
            (state.current_position[0] - monitor.env.goal[0])**2 +
            (state.current_position[1] - monitor.env.goal[1])**2
        ) * 2.0  # Assuming 2m cell size
        
        stats = {
            'iteration': state.iteration,
            'position': state.current_position,
            'fov_radius': state.current_fov_radius,
            'path_length': len(state.executed_path),
            'elapsed_time': info.get('elapsed_time', 0),
            'distance_to_goal': dist_to_goal,
            'recovery_attempts': state.recovery_attempts
        }
        
        # Detect if stuck
        is_stuck = state.visited_counts.get(state.current_position, 0) > 3
        
        monitor.update(
            position=state.current_position,
            fov_radius=state.current_fov_radius,
            executed_path=state.executed_path,
            failed_cells=state.failed_cells,
            stats=stats,
            title=f'Navigation - Iteration {state.iteration}',
            is_stuck=is_stuck,
            in_recovery=state.in_recovery
        )
    
    return callback
