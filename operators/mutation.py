import random
import numpy as np
from ga_utils import invalidate_cache


def triangle_gaussian_mutation(individual, mutation_rate=0.01,
                               sigma_vertex=0.05, sigma_colour=10):
    """
    Gaussian mutation on full triangle (vertices + RGBA).

    Each triangle is mutated independently with probability `mutation_rate`.
    Vertex genes are perturbed with `sigma_vertex`; colour genes with `sigma_colour`.
    All values are clipped to their valid domains after perturbation.
    """
    rep   = individual.representation[:]
    n_tri = individual.n_triangles

    for t in range(n_tri):
        if random.random() < mutation_rate:
            base = t * 10
            for i in range(6):
                rep[base + i] = float(np.clip(
                    rep[base + i] + random.gauss(0, sigma_vertex), 0.0, 1.0
                ))
            for i in range(6, 10):
                rep[base + i] = float(np.clip(
                    rep[base + i] + random.gauss(0, sigma_colour), 0, 255
                ))

    individual.representation = rep
    invalidate_cache(individual)
    return individual


def triangle_color_mutation(individual, mutation_rate=0.01, sigma_colour=15):
    """
    Colour-only mutation.

    Perturbs only the RGBA channels (genes 6–9) of each triangle.
    Useful for fine-tuning colour without disturbing geometry.
    """
    rep   = individual.representation[:]
    n_tri = individual.n_triangles

    for t in range(n_tri):
        if random.random() < mutation_rate:
            base = t * 10
            for i in range(6, 10):
                rep[base + i] = float(np.clip(
                    rep[base + i] + random.gauss(0, sigma_colour), 0, 255
                ))

    individual.representation = rep
    invalidate_cache(individual)
    return individual


def triangle_vertex_mutation(individual, mutation_rate=0.01, sigma_vertex=0.05):
    """
    Vertex-only mutation.

    Perturbs only the coordinate genes (0–5) of each triangle.
    Useful for refining shape without disturbing colour.
    """
    rep   = individual.representation[:]
    n_tri = individual.n_triangles

    for t in range(n_tri):
        if random.random() < mutation_rate:
            base = t * 10
            for i in range(6):
                rep[base + i] = float(np.clip(
                    rep[base + i] + random.gauss(0, sigma_vertex), 0.0, 1.0
                ))

    individual.representation = rep
    invalidate_cache(individual)
    return individual


def triangle_level_mutation(individual,
                            mutation_rate=0.01,
                            sigma_vertex=0.05,
                            sigma_colour=10,
                            mode="mixed"):
    """
    Dispatcher for triangle-level mutation.

    Parameters
    ----------
    mode : 'gaussian' | 'color' | 'vertex' | 'mixed'
        'mixed' randomly selects one of the three operators per call.
    """
    if mode == "gaussian":
        return triangle_gaussian_mutation(individual, mutation_rate,
                                          sigma_vertex, sigma_colour)
    elif mode == "color":
        return triangle_color_mutation(individual, mutation_rate, sigma_colour)
    elif mode == "vertex":
        return triangle_vertex_mutation(individual, mutation_rate, sigma_vertex)
    elif mode == "mixed":
        choice = random.choice(["gaussian", "color", "vertex"])
        return triangle_level_mutation(individual, mutation_rate,
                                       sigma_vertex, sigma_colour, mode=choice)
    return individual
