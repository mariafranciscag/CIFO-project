import numpy as np
from copy import deepcopy
from scripts.ga_utils import invalidate_cache


def one_point_crossover(parent1, parent2, crossover_rate=0.9):
    """
    1-point whole-triangle crossover.

    Works with both TriangleSolution (10 genes) and AdaptiveTriangleSolution (11 genes).
    The crossover point is chosen at a triangle boundary, so genes belonging to the 
    same triangle are never split across parents.
    If crossover does not fire, both parents are returned as deep copies.
    """
    if np.random.random() > crossover_rate:
        c1, c2 = deepcopy(parent1), deepcopy(parent2)
        c1._cached_fitness = getattr(parent1, '_cached_fitness', None)
        c2._cached_fitness = getattr(parent2, '_cached_fitness', None)
        return c1, c2

    triangle_len = parent1.triangle_len
    n_tri = len(parent1.representation) // triangle_len
    point = np.random.randint(1, n_tri - 1) * triangle_len

    r1 = np.concatenate([parent1.representation[:point], parent2.representation[point:]])
    r2 = np.concatenate([parent2.representation[:point], parent1.representation[point:]])

    c1, c2 = type(parent1)(r1), type(parent2)(r2)
    invalidate_cache(c1)
    invalidate_cache(c2)
    return c1, c2


def two_point_crossover(parent1, parent2, crossover_rate=0.9):
    """
    Two-point whole-triangle crossover.

    Works with both TriangleSolution (10 genes) and AdaptiveTriangleSolution (11 genes).
    Both cut points are placed at triangle boundaries.
    The segment between the two points is swapped between parents.
    """
    if np.random.random() > crossover_rate:
        c1, c2 = deepcopy(parent1), deepcopy(parent2)
        c1._cached_fitness = getattr(parent1, '_cached_fitness', None)
        c2._cached_fitness = getattr(parent2, '_cached_fitness', None)
        return c1, c2

    triangle_len = parent1.triangle_len
    n_tri = len(parent1.representation) // triangle_len
    p1, p2 = sorted(np.random.choice(range(1, n_tri), 2, replace=False))
    p1 *= triangle_len
    p2 *= triangle_len

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

    c1, c2 = type(parent1)(r1), type(parent2)(r2)
    invalidate_cache(c1)
    invalidate_cache(c2)
    return c1, c2


def uniform_crossover(parent1, parent2, crossover_rate=0.9, swap_prob=0.5):
    """
    Uniform whole-triangle crossover.

    Works with both TriangleSolution (10 genes) and AdaptiveTriangleSolution (11 genes).
    For each triangle, independently assigns it to child1 or child2
    with probability `swap_prob`. Triangles are never split mid-gene.
    """
    if np.random.random() > crossover_rate:
        c1, c2 = deepcopy(parent1), deepcopy(parent2)
        c1._cached_fitness = getattr(parent1, '_cached_fitness', None)
        c2._cached_fitness = getattr(parent2, '_cached_fitness', None)
        return c1, c2

    triangle_len = parent1.triangle_len
    n_tri = len(parent1.representation) // triangle_len
    r1, r2 = [], []

    for t in range(n_tri):
        s = t * triangle_len
        e = s + triangle_len
        if np.random.random() < swap_prob:
            r1.append(parent2.representation[s:e])
            r2.append(parent1.representation[s:e])
        else:
            r1.append(parent1.representation[s:e])
            r2.append(parent2.representation[s:e])

    c1 = type(parent1)(np.concatenate(r1))
    c2 = type(parent2)(np.concatenate(r2))
    invalidate_cache(c1)
    invalidate_cache(c2)
    return c1, c2


def geometric_crossover(parent1, parent2, crossover_rate=0.9):
    """
    Geometric crossover for real-valued triangle representations.

    Works with both TriangleSolution (10 genes) and
    AdaptiveMutationRateTriangleSolution (11 genes). For adaptive solutions,
    the mutation-rate gene is averaged like the other continuous genes and
    clipped to [0, 1].

    Returns a single offspring, since a convex combination of the two parents
    already lies on the segment between them.
    """
    if np.random.random() > crossover_rate:
        parent = parent1 if np.random.random() < 0.5 else parent2
        child = deepcopy(parent)
        child._cached_fitness = getattr(parent, "_cached_fitness", None)
        return child

    triangle_len = parent1.triangle_len
    if triangle_len != parent2.triangle_len:
        raise ValueError("Parents must use the same triangle representation length.")

    p1 = parent1.representation.astype(np.float32)
    p2 = parent2.representation.astype(np.float32)

    if p1.shape != p2.shape:
        raise ValueError("Parents must have representations with the same shape.")

    weights = np.random.random(size=p1.shape).astype(np.float32)

    child_representation = weights * p1 + (1 - weights) * p2

    lower_per_triangle = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    upper_per_triangle = [
        parent1.img_width, parent1.img_height,
        parent1.img_width, parent1.img_height,
        parent1.img_width, parent1.img_height,
        255, 255, 255, 255
    ]

    if triangle_len == 11:
        lower_per_triangle.append(0)
        upper_per_triangle.append(1)
    elif triangle_len != 10:
        raise ValueError(f"Unsupported triangle representation length: {triangle_len}")

    lower = np.tile(lower_per_triangle, parent1.n_triangles)
    upper = np.tile(upper_per_triangle, parent1.n_triangles)

    child_representation = np.clip(child_representation, lower, upper).astype(np.float32)

    child = type(parent1)(child_representation)
    invalidate_cache(child)
    return child
    
