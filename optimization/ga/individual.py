"""
GA Individual and Operators Module
===================================

Genetic algorithm individual representation and genetic operators.
"""

import numpy as np
from typing import List, Tuple, Optional
from dataclasses import dataclass, field
from copy import deepcopy


@dataclass
class GAIndividual:
    """
    Individual in GA population.
    
    Genome:
    - via: List of via-points between start and goal
    - seg_modes: List of modes for each segment ('time', 'energy', 'safe')
    
    Fitness:
    - Lower is better (minimization)
    - inf = not evaluated
    """
    via: List[Tuple[int, int]] = field(default_factory=list)
    seg_modes: List[str] = field(default_factory=list)
    fitness: float = float('inf')
    is_true_eval: bool = False  # True if evaluated with real energy model
    
    def copy(self) -> 'GAIndividual':
        """Create a deep copy"""
        return GAIndividual(
            via=list(self.via),
            seg_modes=list(self.seg_modes),
            fitness=self.fitness,
            is_true_eval=self.is_true_eval
        )
    
    def __repr__(self) -> str:
        return f"GAIndividual(via={len(self.via)}, modes={self.seg_modes}, fitness={self.fitness:.2f})"


class GeneticOperators:
    """
    Genetic operators for path optimization.
    
    Operators:
    - Mutation: Jitter via-points, change modes
    - Crossover: Single-point crossover for both via-points and modes
    - Tournament selection
    """
    
    MODES = ['time', 'energy', 'safe']
    
    def __init__(self, 
                 mutation_rate: float = 0.25,
                 crossover_rate: float = 0.85,
                 tournament_k: int = 4,
                 max_via_jitter: int = 20,
                 seed: int = 42):
        """
        Initialize genetic operators.
        
        Args:
            mutation_rate: Probability of mutating each gene
            crossover_rate: Probability of crossover
            tournament_k: Tournament selection size
            max_via_jitter: Maximum jitter for via-point mutation
            seed: Random seed
        """
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.tournament_k = tournament_k
        self.max_via_jitter = max_via_jitter
        self.rng = np.random.default_rng(seed)
    
    def mutate(self, ind: GAIndividual, 
               bounds: Tuple[int, int, int, int]) -> GAIndividual:
        """
        Mutate an individual.
        
        Args:
            ind: Individual to mutate
            bounds: (xmin, xmax, ymin, ymax) bounds for via-points
        
        Returns:
            Mutated individual (new object)
        """
        xmin, xmax, ymin, ymax = bounds
        mutant = ind.copy()
        mutated = False
        
        # Mutate via-points
        for i in range(len(mutant.via)):
            if self.rng.random() < self.mutation_rate:
                x, y = mutant.via[i]
                
                # 20% chance of large mutation (anywhere in window)
                if self.rng.random() < 0.2:
                    x = int(self.rng.integers(xmin, xmax + 1))
                    y = int(self.rng.integers(ymin, ymax + 1))
                else:
                    # Small jitter
                    jitter = self.max_via_jitter
                    x = int(np.clip(
                        x + self.rng.integers(-jitter, jitter + 1),
                        xmin, xmax
                    ))
                    y = int(np.clip(
                        y + self.rng.integers(-jitter, jitter + 1),
                        ymin, ymax
                    ))
                
                mutant.via[i] = (x, y)
                mutated = True
        
        # Mutate modes (lower rate)
        for i in range(len(mutant.seg_modes)):
            if self.rng.random() < self.mutation_rate * 0.7:
                mutant.seg_modes[i] = self.rng.choice(self.MODES)
                mutated = True
        
        if mutated:
            mutant.fitness = float('inf')
            mutant.is_true_eval = False
        
        return mutant
    
    def crossover(self, parent1: GAIndividual, 
                  parent2: GAIndividual) -> Tuple[GAIndividual, GAIndividual]:
        """
        Single-point crossover.
        
        Args:
            parent1: First parent
            parent2: Second parent
        
        Returns:
            Tuple of two children
        """
        if self.rng.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1 = parent1.copy()
        child2 = parent2.copy()
        
        # Via-point crossover
        n_via = len(parent1.via)
        if n_via > 1:
            cut = int(self.rng.integers(1, n_via))
            child1.via = parent1.via[:cut] + parent2.via[cut:]
            child2.via = parent2.via[:cut] + parent1.via[cut:]
        
        # Mode crossover
        n_modes = len(parent1.seg_modes)
        if n_modes > 1:
            mcut = int(self.rng.integers(1, n_modes))
            child1.seg_modes = parent1.seg_modes[:mcut] + parent2.seg_modes[mcut:]
            child2.seg_modes = parent2.seg_modes[:mcut] + parent1.seg_modes[mcut:]
        
        # Reset fitness
        child1.fitness = float('inf')
        child1.is_true_eval = False
        child2.fitness = float('inf')
        child2.is_true_eval = False
        
        return child1, child2
    
    def tournament_select(self, population: List[GAIndividual]) -> GAIndividual:
        """
        Tournament selection.
        
        Args:
            population: Population to select from
        
        Returns:
            Selected individual
        """
        k = min(self.tournament_k, len(population))
        candidates = self.rng.choice(population, size=k, replace=False)
        return min(candidates, key=lambda x: x.fitness)
    
    def create_random_individual(self, 
                                 n_via: int,
                                 bounds: Tuple[int, int, int, int]) -> GAIndividual:
        """
        Create random individual within bounds.
        
        Args:
            n_via: Number of via-points
            bounds: (xmin, xmax, ymin, ymax)
        
        Returns:
            New random individual
        """
        xmin, xmax, ymin, ymax = bounds
        
        via = [
            (int(self.rng.integers(xmin, xmax + 1)),
             int(self.rng.integers(ymin, ymax + 1)))
            for _ in range(n_via)
        ]
        
        seg_modes = [self.rng.choice(self.MODES) for _ in range(n_via + 1)]
        
        return GAIndividual(via=via, seg_modes=seg_modes)
    
    def create_from_seed_path(self,
                              seed_path: List[Tuple[int, int]],
                              n_via: int,
                              bounds: Tuple[int, int, int, int],
                              jitter: bool = True) -> GAIndividual:
        """
        Create individual from a seed path.
        
        Args:
            seed_path: Seed path to base individual on
            n_via: Number of via-points
            bounds: (xmin, xmax, ymin, ymax)
            jitter: Whether to add random jitter
        
        Returns:
            New individual based on seed path
        """
        xmin, xmax, ymin, ymax = bounds
        
        if len(seed_path) < n_via + 2:
            # Pad path if too short
            seed_path = list(seed_path) + [seed_path[-1]] * (n_via + 2 - len(seed_path))
        
        # Sample via-points from seed path
        indices = np.linspace(1, len(seed_path) - 2, n_via, dtype=int)
        
        via = []
        for idx in indices:
            x, y = seed_path[idx]
            
            if jitter:
                jit = self.max_via_jitter
                x = int(np.clip(
                    x + self.rng.integers(-jit, jit + 1),
                    xmin, xmax
                ))
                y = int(np.clip(
                    y + self.rng.integers(-jit, jit + 1),
                    ymin, ymax
                ))
            
            via.append((x, y))
        
        seg_modes = [self.rng.choice(self.MODES) for _ in range(n_via + 1)]
        
        return GAIndividual(via=via, seg_modes=seg_modes)
