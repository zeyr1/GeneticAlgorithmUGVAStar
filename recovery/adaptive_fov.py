"""
Adaptive Field-of-View Module
==============================

Dynamic FoV radius adjustment based on navigation success/failure.
"""

from dataclasses import dataclass
from typing import Optional


@dataclass
class FoVState:
    """Current state of adaptive FoV"""
    base_radius: int
    current_radius: int
    min_radius: int
    max_radius: int
    stuck_count: int = 0
    success_count: int = 0
    expansion_factor: float = 1.5
    contraction_rate: int = 5


class AdaptiveFoV:
    """
    Adaptive Field-of-View radius controller.
    
    Strategy:
    - Expand FoV when stuck or backtracking
    - Contract FoV when consistently successful
    - Use exponential expansion for severe stuck conditions
    
    This helps robots "see further" when encountering dead-ends,
    enabling better global awareness without full-map planning.
    """
    
    def __init__(self,
                 base_radius: int = 25,
                 min_radius: int = 15,
                 max_radius: int = 150,
                 expansion_factor: float = 1.5,
                 contraction_rate: int = 5):
        """
        Initialize adaptive FoV controller.
        
        Args:
            base_radius: Default/starting radius (cells)
            min_radius: Minimum allowed radius
            max_radius: Maximum allowed radius
            expansion_factor: Multiplier when stuck
            contraction_rate: Cells to reduce when succeeding
        """
        self.state = FoVState(
            base_radius=base_radius,
            current_radius=base_radius,
            min_radius=min_radius,
            max_radius=max_radius,
            expansion_factor=expansion_factor,
            contraction_rate=contraction_rate
        )
    
    @property
    def current_radius(self) -> int:
        """Get current FoV radius"""
        return self.state.current_radius
    
    def reset(self):
        """Reset to base radius"""
        self.state.current_radius = self.state.base_radius
        self.state.stuck_count = 0
        self.state.success_count = 0
    
    def update(self, success: bool, backtrack_ratio: float = 0.0) -> int:
        """
        Update FoV based on navigation outcome.
        
        Args:
            success: Whether last planning/execution succeeded
            backtrack_ratio: Ratio of distance spent backtracking (0-1)
        
        Returns:
            Updated FoV radius
        """
        if not success or backtrack_ratio > 0.2:
            # Failure or significant backtracking: expand
            self.state.stuck_count += 1
            self.state.success_count = 0
            
            # Exponential expansion
            new_radius = int(self.state.base_radius * 
                           (self.state.expansion_factor ** self.state.stuck_count))
            
            self.state.current_radius = min(self.state.max_radius, new_radius)
            
        else:
            # Success: potentially contract
            self.state.success_count += 1
            self.state.stuck_count = max(0, self.state.stuck_count - 1)
            
            # Gradual contraction after multiple successes
            if self.state.success_count >= 3:
                new_radius = self.state.current_radius - self.state.contraction_rate
                self.state.current_radius = max(self.state.min_radius, new_radius)
                self.state.success_count = 0
        
        return self.state.current_radius
    
    def force_expand(self, factor: float = 2.0) -> int:
        """Force immediate expansion (for recovery)"""
        self.state.stuck_count += 2
        new_radius = int(self.state.current_radius * factor)
        self.state.current_radius = min(self.state.max_radius, new_radius)
        return self.state.current_radius
    
    def get_state(self) -> dict:
        """Get current state as dictionary"""
        return {
            'current_radius': self.state.current_radius,
            'base_radius': self.state.base_radius,
            'stuck_count': self.state.stuck_count,
            'success_count': self.state.success_count,
            'expansion_ratio': self.state.current_radius / self.state.base_radius
        }
