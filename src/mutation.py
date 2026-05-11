import numpy as np
from .solution import Solution

def random_per_gene_mutation(individual: Solution, mutation_rate: float) -> Solution:
    rep = individual.representation
    n = individual.n_triangles
    triangle_rep_lenght = 10 if individual.use_transparency else 9

    # boolean mask: True where a gene should mutate
    mask = np.random.random((n, triangle_rep_lenght)) < mutation_rate

    # generate all candidate new values upfront
    candidates = np.empty((n, triangle_rep_lenght), dtype=int)
    candidates[:, [0, 2, 4]] = np.random.randint(0, individual.width, size=(n, 3))
    candidates[:, [1, 3, 5]] = np.random.randint(0, individual.height, size=(n, 3))
    candidates[:, 6:] = np.random.randint(0, 255, size=(n, triangle_rep_lenght - 6))

    # apply mutations in one vectorized step
    rep[:] = np.where(mask, candidates, rep)

    return individual
