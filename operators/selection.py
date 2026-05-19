import random
import numpy as np
from scripts.ga_utils import get_fitness


def roulette_wheel_selection(population):
    """
    Fitness-proportionate selection (roulette wheel).

    Because this is a minimisation problem, fitness values are inverted before
    computing selection probabilities: individuals with lower RMSE receive a
    proportionally higher chance of being selected.
    """
    fitnesses = np.array([get_fitness(ind) for ind in population])
    inverted  = 1.0 / (fitnesses + 1e-8)
    probs     = inverted / inverted.sum()
    idx       = np.random.choice(len(population), p=probs)
    return population[idx]


def rank_based_selection(population):
    """
    Rank-based selection.

    Individuals are ranked by fitness (rank 1 = worst, rank N = best).
    Selection probability is proportional to rank, which reduces the
    dominance of very fit individuals and preserves diversity.
    """
    n          = len(population)
    sorted_pop = sorted(population, key=lambda ind: get_fitness(ind), reverse=True)
    ranks      = np.arange(1, n + 1, dtype=float)
    probs      = ranks / ranks.sum()
    idx        = np.random.choice(n, p=probs)
    return sorted_pop[idx]


def tournament_selection(population, tournament_size=2):
    """
    Tournament selection.

    Randomly samples `tournament_size` individuals and returns the one
    with the lowest RMSE (minimisation problem).
    """
    contestants = random.sample(population, tournament_size)
    return min(contestants, key=lambda ind: get_fitness(ind))
