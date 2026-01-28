"""
Surrogate Model Module
======================

Surrogate models for fast path evaluation in GA.
Includes both global and local (region-based) ensemble models.
"""

import numpy as np
from typing import List, Tuple, Dict, Optional, Any
from dataclasses import dataclass, field
import math

try:
    from sklearn.ensemble import RandomForestRegressor
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False


@dataclass
class SurrogateSample:
    """Training sample for surrogate model"""
    features: np.ndarray
    target: float
    region: Optional[Tuple[int, int]] = None


class SurrogateFeatureExtractor:
    """
    Extract features from path for surrogate prediction.
    
    Features:
    - Path statistics (length, segment count)
    - Terrain distribution
    - Elevation statistics
    - Mode distribution
    - Geometric features
    """
    
    TERRAIN_NAMES = ['asphalt', 'grass', 'mud', 'sand']
    MODE_NAMES = ['time', 'energy', 'safe']
    
    def __init__(self, config):
        self.config = config
        self.cell_size = config.map.cell_size
    
    def extract(self, 
                path: List[Tuple[int, int]], 
                modes: List[str],
                env) -> np.ndarray:
        """
        Extract feature vector from path.
        
        Args:
            path: Path as list of (x, y)
            modes: Modes for each segment
            env: Environment for terrain lookup
        
        Returns:
            Feature vector (1D numpy array)
        """
        if len(path) < 2:
            return np.zeros(20)
        
        features = []
        
        # 1. Path length statistics
        total_dist = 0
        for i in range(len(path) - 1):
            x1, y1 = path[i]
            x2, y2 = path[i + 1]
            total_dist += math.sqrt((x2-x1)**2 + (y2-y1)**2) * self.cell_size
        
        features.append(total_dist / 1000.0)  # Normalized distance (km)
        features.append(len(path))  # Segment count
        
        # 2. Terrain distribution
        terrain_counts = {t: 0 for t in self.TERRAIN_NAMES}
        for x, y in path:
            try:
                t_type = int(env.terrain[x, y])
                t_name = ['asphalt', 'grass', 'mud', 'sand', 'wall'][t_type]
                if t_name in terrain_counts:
                    terrain_counts[t_name] += 1
            except:
                terrain_counts['grass'] += 1
        
        total = max(1, sum(terrain_counts.values()))
        for t in self.TERRAIN_NAMES:
            features.append(terrain_counts[t] / total)
        
        # 3. Elevation statistics
        elevations = []
        for x, y in path:
            try:
                elevations.append(float(env.elevation[x, y]))
            except:
                elevations.append(40.0)
        
        elevations = np.array(elevations)
        features.append(np.mean(elevations) / 80.0)  # Normalized mean
        features.append(np.std(elevations) / 20.0)  # Normalized std
        
        # Elevation change
        if len(elevations) > 1:
            total_climb = sum(max(0, elevations[i+1] - elevations[i]) 
                            for i in range(len(elevations)-1))
            total_descent = sum(max(0, elevations[i] - elevations[i+1]) 
                              for i in range(len(elevations)-1))
            features.append(total_climb / 100.0)
            features.append(total_descent / 100.0)
        else:
            features.extend([0, 0])
        
        # 4. Mode distribution
        mode_counts = {m: 0 for m in self.MODE_NAMES}
        for m in modes:
            if m in mode_counts:
                mode_counts[m] += 1
        
        total_modes = max(1, len(modes))
        for m in self.MODE_NAMES:
            features.append(mode_counts[m] / total_modes)
        
        # 5. Geometric features
        if len(path) >= 2:
            # Start-goal distance
            start = path[0]
            end = path[-1]
            direct_dist = math.sqrt((end[0]-start[0])**2 + (end[1]-start[1])**2) * self.cell_size
            features.append(direct_dist / 1000.0)
            
            # Path efficiency (ratio of direct to actual distance)
            efficiency = direct_dist / max(1, total_dist)
            features.append(efficiency)
            
            # Turn count (approximate)
            turns = 0
            for i in range(1, len(path) - 1):
                x0, y0 = path[i-1]
                x1, y1 = path[i]
                x2, y2 = path[i+1]
                
                v1 = (x1 - x0, y1 - y0)
                v2 = (x2 - x1, y2 - y1)
                
                if v1 != (0, 0) and v2 != (0, 0):
                    dot = v1[0]*v2[0] + v1[1]*v2[1]
                    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
                    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
                    cos_angle = dot / (mag1 * mag2 + 1e-9)
                    
                    if cos_angle < 0.7:  # Significant turn
                        turns += 1
            
            features.append(turns / max(1, len(path)))
        else:
            features.extend([0, 0, 0])
        
        # Pad to fixed size
        while len(features) < 20:
            features.append(0)
        
        return np.array(features[:20], dtype=np.float32)


class SurrogateModel:
    """
    Random Forest surrogate model for path evaluation.
    
    Learns mapping from path features to objective value.
    """
    
    def __init__(self, 
                 config,
                 enabled: bool = True,
                 n_estimators: int = 50,
                 max_depth: int = 10,
                 seed: int = 42):
        """
        Initialize surrogate model.
        
        Args:
            config: Configuration object
            enabled: Whether surrogate is enabled
            n_estimators: Number of trees in forest
            max_depth: Maximum tree depth
            seed: Random seed
        """
        self.config = config
        self.enabled = enabled and SKLEARN_AVAILABLE
        self.seed = seed
        
        self.feature_extractor = SurrogateFeatureExtractor(config)
        
        # Training data
        self.X: List[np.ndarray] = []
        self.y: List[float] = []
        
        # Model
        self._model = None
        self._is_fitted = False
        
        if self.enabled:
            self._model = RandomForestRegressor(
                n_estimators=n_estimators,
                max_depth=max_depth,
                random_state=seed,
                n_jobs=-1
            )
        
        # Statistics
        self.last_mape: Optional[float] = None
        self._predictions = 0
    
    def add_sample(self, 
                   path: List[Tuple[int, int]], 
                   modes: List[str],
                   target: float,
                   env):
        """Add training sample"""
        if not self.enabled or target == float('inf'):
            return
        
        features = self.feature_extractor.extract(path, modes, env)
        self.X.append(features)
        self.y.append(target)
    
    def fit(self) -> bool:
        """Fit model to collected samples"""
        if not self.enabled or len(self.X) < 10:
            return False
        
        X = np.array(self.X)
        y = np.array(self.y)
        
        try:
            self._model.fit(X, y)
            self._is_fitted = True
            
            # Compute MAPE on training data (simple validation)
            predictions = self._model.predict(X)
            errors = np.abs(predictions - y) / (np.abs(y) + 1e-9)
            self.last_mape = float(np.mean(errors))
            
            return True
        except Exception as e:
            self._is_fitted = False
            return False
    
    def can_predict(self) -> bool:
        """Check if model can make predictions"""
        return self.enabled and self._is_fitted
    
    def predict(self, 
                path: List[Tuple[int, int]], 
                modes: List[str],
                env) -> float:
        """Predict objective value for path"""
        if not self.can_predict():
            return float('inf')
        
        features = self.feature_extractor.extract(path, modes, env)
        
        try:
            prediction = self._model.predict(features.reshape(1, -1))[0]
            self._predictions += 1
            return float(prediction)
        except:
            return float('inf')
    
    def get_stats(self) -> Dict:
        """Get model statistics"""
        return {
            'enabled': self.enabled,
            'is_fitted': self._is_fitted,
            'n_samples': len(self.X),
            'predictions': self._predictions,
            'last_mape': self.last_mape
        }
    
    def reset(self):
        """Reset model state"""
        self.X.clear()
        self.y.clear()
        self._is_fitted = False
        self.last_mape = None
        self._predictions = 0


class LocalSurrogateEnsemble:
    """
    Ensemble of local surrogate models for different map regions.
    
    Divides map into grid of regions, each with its own surrogate.
    This improves prediction accuracy by capturing local patterns.
    """
    
    def __init__(self,
                 config,
                 grid_divisions: int = 4,
                 min_samples_per_region: int = 10,
                 seed: int = 42):
        """
        Initialize ensemble.
        
        Args:
            config: Configuration object
            grid_divisions: Number of divisions per axis (4 = 16 regions)
            min_samples_per_region: Minimum samples before fitting region model
            seed: Random seed
        """
        self.config = config
        self.grid_divisions = grid_divisions
        self.min_samples = min_samples_per_region
        self.seed = seed
        
        self.feature_extractor = SurrogateFeatureExtractor(config)
        
        # Region models: (region_x, region_y) -> model_data
        self.region_data: Dict[Tuple[int, int], Dict] = {}
        
        # Global fallback model
        self.global_model = SurrogateModel(config, enabled=True, seed=seed)
        
        self.enabled = SKLEARN_AVAILABLE
        self.last_mape: Optional[float] = None
    
    def _get_region(self, path: List[Tuple[int, int]]) -> Tuple[int, int]:
        """Determine region for path based on centroid"""
        if not path:
            return (0, 0)
        
        # Compute centroid
        cx = sum(p[0] for p in path) / len(path)
        cy = sum(p[1] for p in path) / len(path)
        
        # Map to region
        grid_size = self.config.map.grid_size
        region_x = int(cx / grid_size * self.grid_divisions)
        region_y = int(cy / grid_size * self.grid_divisions)
        
        region_x = max(0, min(self.grid_divisions - 1, region_x))
        region_y = max(0, min(self.grid_divisions - 1, region_y))
        
        return (region_x, region_y)
    
    def add_sample(self,
                   path: List[Tuple[int, int]],
                   modes: List[str],
                   target: float,
                   env):
        """Add training sample to appropriate region"""
        if not self.enabled or target == float('inf'):
            return
        
        region = self._get_region(path)
        features = self.feature_extractor.extract(path, modes, env)
        
        if region not in self.region_data:
            self.region_data[region] = {
                'X': [],
                'y': [],
                'model': None,
                'is_fitted': False
            }
        
        self.region_data[region]['X'].append(features)
        self.region_data[region]['y'].append(target)
        
        # Also add to global model
        self.global_model.add_sample(path, modes, target, env)
    
    def fit(self):
        """Fit all region models with sufficient data"""
        if not self.enabled:
            return
        
        mapes = []
        
        for region, data in self.region_data.items():
            if len(data['X']) >= self.min_samples:
                try:
                    X = np.array(data['X'])
                    y = np.array(data['y'])
                    
                    model = RandomForestRegressor(
                        n_estimators=30,
                        max_depth=8,
                        random_state=self.seed,
                        n_jobs=-1
                    )
                    model.fit(X, y)
                    
                    data['model'] = model
                    data['is_fitted'] = True
                    
                    # Compute MAPE
                    preds = model.predict(X)
                    mape = np.mean(np.abs(preds - y) / (np.abs(y) + 1e-9))
                    mapes.append(mape)
                    
                except Exception:
                    data['is_fitted'] = False
        
        # Fit global model as fallback
        self.global_model.fit()
        
        if mapes:
            self.last_mape = float(np.mean(mapes))
    
    def can_predict(self) -> bool:
        """Check if any model can predict"""
        if not self.enabled:
            return False
        
        for data in self.region_data.values():
            if data['is_fitted']:
                return True
        
        return self.global_model.can_predict()
    
    def predict(self,
                path: List[Tuple[int, int]],
                modes: List[str],
                env) -> float:
        """Predict using appropriate region model or global fallback"""
        if not self.enabled:
            return float('inf')
        
        region = self._get_region(path)
        features = self.feature_extractor.extract(path, modes, env)
        
        # Try region model first
        if region in self.region_data and self.region_data[region]['is_fitted']:
            try:
                model = self.region_data[region]['model']
                return float(model.predict(features.reshape(1, -1))[0])
            except:
                pass
        
        # Fallback to global
        return self.global_model.predict(path, modes, env)
    
    def get_stats(self) -> Dict:
        """Get ensemble statistics"""
        region_stats = {}
        for region, data in self.region_data.items():
            region_stats[f"{region}"] = {
                'samples': len(data['X']),
                'fitted': data['is_fitted']
            }
        
        return {
            'enabled': self.enabled,
            'n_regions': len(self.region_data),
            'fitted_regions': sum(1 for d in self.region_data.values() if d['is_fitted']),
            'last_mape': self.last_mape,
            'region_stats': region_stats,
            'global_stats': self.global_model.get_stats()
        }
    
    def reset(self):
        """Reset all models"""
        self.region_data.clear()
        self.global_model.reset()
        self.last_mape = None
