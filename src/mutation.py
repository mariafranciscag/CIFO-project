import numpy as np
from .solution import Solution

def random_per_gene_reset_mutation(individual: Solution, mutation_rate: float) -> Solution:
    rep = individual.representation
    n = individual.n_triangles
    triangle_len = len(rep[0])

    # boolean mask: True where a gene should mutate
    mask = np.random.random((n, triangle_len)) < mutation_rate

    # generate all candidate new values upfront
    candidates = np.empty((n, triangle_len), dtype=int)
    candidates[:, [0, 2, 4]] = np.random.randint(0, individual.width, size=(n, 3))
    candidates[:, [1, 3, 5]] = np.random.randint(0, individual.height, size=(n, 3))
    candidates[:, 6:] = np.random.randint(0, 255, size=(n, triangle_len - 6))

    # apply mutations in one vectorized step
    rep[:] = np.where(mask, candidates, rep)

    return individual

def gaussian_creep_mutation(individual: Solution, mutation_rate: float, gaussian_std_dev: int = 5) -> Solution:
    rep = individual.representation
    n = individual.n_triangles
    triangle_len = len(rep[0])

    for i_triangle in range(n):
        for i_gene in range(triangle_len):
            if np.random.random() < mutation_rate:
                
                mutation = 0
                max_value = 255
                
                if i_gene in [0, 2, 4]:
                    # x position gene
                    max_value = 300
                    mutation = np.random.normal(scale=gaussian_std_dev) # sample from a normal/gaussian distribution
                    
                elif i_gene in [1, 3, 5]:
                    # y position gene
                    max_value = 400
                    mutation = np.random.normal(scale=gaussian_std_dev)
                    
                else:
                    # color gene
                    max_value = 255
                    mutation = np.random.normal(scale=gaussian_std_dev)
                
                rep[i_triangle, i_gene] = np.clip(rep[i_triangle, i_gene] + mutation, 0, max_value)

    return individual

def whole_triangle_mutation(individual: Solution, mutation_rate: float) -> Solution:
    rep = individual.representation
    n = individual.n_triangles
    triangle_len = len(rep[0])
    
    for i_triangle in range(n):
        if np.random.random() < mutation_rate:
            rep[i_triangle, [0, 2, 4]] = np.random.randint(0, individual.width, size=(3,))
            rep[i_triangle, [1, 3, 5]] = np.random.randint(0, individual.height, size=(3,))
            rep[i_triangle, 6:] = np.random.randint(0, 255, size=(triangle_len - 6,))
    
    return individual

def layer_order_mutation(individual: Solution, mutation_rate: float) -> Solution:
    rep = individual.representation
    n = individual.n_triangles
    triangle_len = len(rep[0])
    
    if np.random.random() < mutation_rate:
        i_triangles_to_swap = np.random.choice(n, 2, replace=False)
        triangle1, triangle2 = rep[i_triangles_to_swap]
        rep[i_triangles_to_swap] = triangle2, triangle1
    
    return individual
        