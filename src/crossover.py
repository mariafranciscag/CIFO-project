import numpy as np
from copy import deepcopy
from .solution import Solution

def uniform_crossover(parent1: Solution, parent2: Solution, crossover_rate: float) -> tuple[Solution, Solution]:
    """
    Uniform crossover swaps each of the triangles between the parents with the same probability
    """
    
    if parent1.n_triangles != parent2.n_triangles:
        raise Exception("Parents did not have the same number of triangles.")
    
    child1 = deepcopy(parent1)
    child2 = deepcopy(parent2)
    
    for i_triangle in range(parent1.n_triangles):
        if np.random.random() < crossover_rate:
            child2.representation[i_triangle] = parent1.representation[i_triangle]
            child1.representation[i_triangle] = parent2.representation[i_triangle]
            
    return child1, child2

def one_point_row_crossover(parent1: Solution, parent2: Solution, crossover_rate: float) -> tuple[Solution, Solution]:
    """
    One-point row crossover chooses randomly a split point, and the triangles are exchanged between the parents 
    before the split and kept the same after the split point. It is only applied with crossover_rate probability.
    """
    
    if parent1.n_triangles != parent2.n_triangles:
        raise Exception("Parents did not have the same number of triangles.")

    child1 = deepcopy(parent1)
    child2 = deepcopy(parent2)
    
    if np.random.random() < crossover_rate:
        crossover_point = np.random.choice(parent1.n_triangles)
        
        for i_triangle in range(parent1.n_triangles):
            if i_triangle < crossover_point:
                child2.representation[i_triangle] = parent1.representation[i_triangle]
                child1.representation[i_triangle] = parent2.representation[i_triangle]
    
    return child1, child2

def two_point_row_crossover(parent1: Solution, parent2: Solution, crossover_rate: float) -> tuple[Solution, Solution]:
    """
    Two-point row crossover chooses randomly two split points, and the triangles are exchanged between the parents 
    between the two splits and kept the same outside the split points. It is only applied with crossover_rate probability.
    """
    
    if parent1.n_triangles != parent2.n_triangles:
        raise Exception("Parents did not have the same number of triangles.")

    child1 = deepcopy(parent1)
    child2 = deepcopy(parent2)
    
    if np.random.random() < crossover_rate:
        crossover_point1 = np.random.choice(parent1.n_triangles)
        crossover_point2 = np.random.choice(parent1.n_triangles)
        
        for i_triangle in range(parent1.n_triangles):
            if min(crossover_point1, crossover_point2) < i_triangle < max(crossover_point1, crossover_point2):
                child2.representation[i_triangle] = parent1.representation[i_triangle]
                child1.representation[i_triangle] = parent2.representation[i_triangle]
    
    return child1, child2

def geometry_color_separated_uniform_crossover(parent1: Solution, parent2: Solution, crossover_rate: float) -> tuple[Solution, Solution]:
    """
    For each selected triangle, take coordinates from one parent and color/alpha from the other. 
    This is useful because good shapes and good colors may evolve separately.
    It is basically uniform crossover, but we don't swap the whole triangle.
    """
    
    if parent1.n_triangles != parent2.n_triangles:
        raise Exception("Parents did not have the same number of triangles.")
    
    child1 = deepcopy(parent1)
    child2 = deepcopy(parent2)
    
    for i_triangle in range(parent1.n_triangles):
        if np.random.random() < crossover_rate:
            child2.representation[i_triangle][:6] = parent1.representation[i_triangle][:6]
            child1.representation[i_triangle][7:] = parent2.representation[i_triangle][7:]
            
    return child1, child2

def arithmetic_blend_crossover(parent1: Solution, parent2: Solution, crossover_rate: float, alpha: float = 0.5) -> tuple[Solution, Solution]:
    """
    For numeric genes: child = alpha * parent1 + (1 - alpha) * parent2, then round and clip values. 
    This is more “real-coded GA” style and may work well because coordinates/colors are continuous-ish integers.
    """
    
    if parent1.n_triangles != parent2.n_triangles:
        raise Exception("Parents did not have the same number of triangles.")

    triangle_len = len(parent1.representation[0])
    
    child1 = deepcopy(parent1)
    child2 = deepcopy(parent2)
    
    for i_triangle in range(parent1.n_triangles):
        if np.random.random() < crossover_rate:
            for i_gene in range(triangle_len):
                max_value = 255
                if i_gene in [0, 2, 4]:
                    max_value = 300
                elif i_gene in [1, 3, 5]:
                    max_value = 400
                elif parent1.use_self_adaptive_mutation_rate and i_gene == triangle_len - 1:
                    max_value = 1
                
                child2.representation[i_triangle][i_gene] = min(max_value, max(0, \
                    alpha * parent1.representation[i_triangle][i_gene] + \
                    (1 - alpha) * parent2.representation[i_triangle][i_gene]
                ))
                
                child1.representation[i_triangle][i_gene] = min(max_value, max(0, \
                    alpha * parent2.representation[i_triangle][i_gene] + \
                    (1 - alpha) * parent1.representation[i_triangle][i_gene]
                ))
    
    return child1, child2
                
                