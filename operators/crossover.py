import random
import numpy as np
from copy import deepcopy
from scripts.triangle_sol import TriangleSolution
from scripts.ga_utils import invalidate_cache


def one_point_crossover(parent1, parent2, crossover_rate=0.9):
    """
    1-point whole-triangle crossover.

    The crossover point is chosen at a triangle boundary (multiple of 10),
    so genes belonging to the same triangle are never split across parents.
    If crossover does not fire, both parents are returned as deep copies
    with their cached fitness preserved.
    """
    if random.random() > crossover_rate:
        c1, c2 = deepcopy(parent1), deepcopy(parent2)
        c1._cached_fitness = getattr(parent1, '_cached_fitness', None)
        c2._cached_fitness = getattr(parent2, '_cached_fitness', None)
        return c1, c2

    n_tri = len(parent1.representation) // 10
    point = random.randint(1, n_tri - 1) * 10

    r1 = np.concatenate([parent1.representation[:point], parent2.representation[point:]])
    r2 = np.concatenate([parent2.representation[:point], parent1.representation[point:]])

    c1, c2 = TriangleSolution(r1), TriangleSolution(r2)
    invalidate_cache(c1)
    invalidate_cache(c2)
    return c1, c2


def two_point_crossover(parent1, parent2, crossover_rate=0.9):
    """
    Two-point whole-triangle crossover.

    Both cut points are placed at triangle boundaries (multiples of 10).
    The segment between the two points is swapped between parents.
    """
    if random.random() > crossover_rate:
        c1, c2 = deepcopy(parent1), deepcopy(parent2)
        c1._cached_fitness = getattr(parent1, '_cached_fitness', None)
        c2._cached_fitness = getattr(parent2, '_cached_fitness', None)
        return c1, c2

    n_tri = len(parent1.representation) // 10
    p1, p2 = sorted(random.sample(range(1, n_tri), 2))
    p1 *= 10
    p2 *= 10

    r1 = np.concatenate([
        parent1.representation[:p1],
        parent2.representation[p1:p2],
        parent1.representation[p2:]
    ])
    r2 = np.concatenate([
        parent2.representation[:p1],
        parent1.representation[p1:p2],
        parent2.representation[p2:]
    ])

    c1, c2 = TriangleSolution(r1), TriangleSolution(r2)
    invalidate_cache(c1)
    invalidate_cache(c2)
    return c1, c2


def uniform_crossover(parent1, parent2, crossover_rate=0.9, swap_prob=0.5):
    """
    Uniform whole-triangle crossover.

    For each triangle, independently assigns it to child1 or child2
    with probability `swap_prob`. Triangles are never split mid-gene.
    """
    if random.random() > crossover_rate:
        c1, c2 = deepcopy(parent1), deepcopy(parent2)
        c1._cached_fitness = getattr(parent1, '_cached_fitness', None)
        c2._cached_fitness = getattr(parent2, '_cached_fitness', None)
        return c1, c2

    n_tri = len(parent1.representation) // 10
    r1, r2 = [], []

    for t in range(n_tri):
        s = t * 10
        e = s + 10
        if random.random() < swap_prob:
            r1.append(parent2.representation[s:e])
            r2.append(parent1.representation[s:e])
        else:
            r1.append(parent1.representation[s:e])
            r2.append(parent2.representation[s:e])

    c1 = TriangleSolution(np.concatenate(r1))
    c2 = TriangleSolution(np.concatenate(r2))
    invalidate_cache(c1)
    invalidate_cache(c2)
    return c1, c2
