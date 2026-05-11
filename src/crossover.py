import numpy as np
from copy import deepcopy
from .solution import Solution

def uniform_crossover(parent1: Solution, parent2: Solution, crossover_rate: float) -> tuple[Solution, Solution]:
    
    if parent1.n_triangles != parent2.n_triangles:
        raise Exception("Parents did not have the same number of triangles.") 
    
    child1 = deepcopy(parent1)
    child2 = deepcopy(parent2)
    
    for i_triangle in range(parent1.n_triangles):
        if np.random.random() < crossover_rate:
            child2.representation[i_triangle] = parent1.representation[i_triangle]
            child1.representation[i_triangle] = parent2.representation[i_triangle]
            
    return child1, child2

# We need to other crossover operators
