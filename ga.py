import random
import numpy as np
from copy import deepcopy

from triangle_sol import TriangleSolution
from ga_utils import get_fitness, evaluate_population
from operators.selection import rank_based_selection, roulette_wheel_selection
from operators.crossover import one_point_crossover
from operators.mutation import triangle_level_mutation


def genetic_algorithm(
    population_size     = 50,
    n_generations       = 500,
    selection_method    = 'rank',     # 'rank' | 'roulette' | 'tournament' | callable
    crossover_fn        = None,       # callable(p1, p2, rate) -> (c1, c2)
    mutation_fn         = None,       # callable(ind, rate, **kw) -> ind
    crossover_rate      = 0.9,
    mutation_rate       = 0.01,
    mutation_rate_decay = 0.0,        # per-generation decay fraction
    sigma_vertex        = 0.05,
    sigma_colour        = 10,
    elitism_n           = 2,
    n_jobs              = -1,
    verbose             = True,
    verbose_every       = 50,
):
    """
    Generational GA for the triangle image approximation problem.

    Parameters
    ----------
    selection_method    : 'rank', 'roulette', 'tournament', or a callable(pop) -> individual
    crossover_fn        : callable(p1, p2, crossover_rate) -> (child1, child2)
    mutation_fn         : callable(ind, rate, **sigma_kw) -> ind
    mutation_rate_decay : rate /= (1 + decay) each generation

    Returns
    -------
    best        : TriangleSolution  — best individual found
    history     : list[float]       — best RMSE per generation
    avg_history : list[float]       — average RMSE per generation
    """
    # resolve selection operator
    if callable(selection_method):
        select = selection_method
    elif selection_method == 'rank':
        select = rank_based_selection
    elif selection_method == 'tournament':
        from operators.selection import tournament_selection
        select = tournament_selection
    else:
        select = roulette_wheel_selection

    _crossover = crossover_fn if crossover_fn is not None else one_point_crossover
    _mutate    = mutation_fn  if mutation_fn  is not None else triangle_level_mutation

    # initialise population
    population            = [TriangleSolution() for _ in range(population_size)]
    population            = evaluate_population(population, n_jobs=n_jobs)
    current_mutation_rate = mutation_rate

    history     = []
    avg_history = []

    for gen in range(n_generations):
        population.sort(key=lambda ind: get_fitness(ind))

        best_fitness = get_fitness(population[0])
        avg_fitness  = float(np.mean([get_fitness(ind) for ind in population]))
        history.append(best_fitness)
        avg_history.append(avg_fitness)

        if verbose and (gen % verbose_every == 0 or gen == n_generations - 1):
            print(f"Gen {gen:>4d} | best RMSE: {best_fitness:.4f} | "
                  f"avg RMSE: {avg_fitness:.4f} | "
                  f"mut rate: {current_mutation_rate:.5f}")

        # elitism
        next_gen = []
        for ind in population[:elitism_n]:
            elite = deepcopy(ind)
            elite._cached_fitness = get_fitness(ind)
            next_gen.append(elite)

        # build offspring
        offspring = []
        while len(next_gen) + len(offspring) < population_size:
            p1 = select(population)
            p2 = select(population)

            o1, o2 = _crossover(p1, p2, crossover_rate)

            _mutate(o1, current_mutation_rate,
                    sigma_vertex=sigma_vertex, sigma_colour=sigma_colour)
            _mutate(o2, current_mutation_rate,
                    sigma_vertex=sigma_vertex, sigma_colour=sigma_colour)

            offspring.append(o1)
            if len(next_gen) + len(offspring) < population_size:
                offspring.append(o2)

        offspring  = evaluate_population(offspring, n_jobs=n_jobs)
        population = next_gen + offspring

        if mutation_rate_decay > 0:
            current_mutation_rate /= (1 + mutation_rate_decay)

    population.sort(key=lambda ind: get_fitness(ind))
    best = population[0]

    if verbose:
        print(f"\nFinished. Best RMSE: {get_fitness(best):.4f}")

    return best, history, avg_history
