import numpy as np
from copy import deepcopy
from typing import Callable

from .triangle_sol import TriangleSolution
from .ga_utils import get_fitness, evaluate_population
from operators.selection import tournament_selection
from operators.crossover import one_point_crossover
from operators.mutation import triangle_gaussian_mutation


def genetic_algorithm(
    solution_class: Callable = TriangleSolution,
    population_size: int = 50,
    n_generations: int = 200,
    selection_function: Callable = tournament_selection, # callable(pop) -> individual
    crossover_function: Callable = one_point_crossover, # callable(p1, p2, rate) -> (c1, c2)
    mutation_function: Callable = triangle_gaussian_mutation, # callable(ind, rate, **kw) -> ind
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.01,
    mutation_rate_decay: float | None = None, # per-generation decay fraction
    elitism_n = 2,
    n_jobs = -1,
    verbose = True,
    verbose_every = 50,
    generation_callback: Callable | None = None,
):
    """
    Generational GA for the triangle image approximation problem.

    Parameters
    ----------
    selection_function : callable(pop) -> individual
    crossover_function : callable(p1, p2, crossover_rate) -> (child1, child2)
    mutation_function : callable(ind, rate, **sigma_kw) -> ind
    mutation_rate_decay : rate /= (1 + decay) each generation or None
    generation_callback : callable(gen, best_fitness, avg_fitness) -> None or None
        Optional hook called after each generation is completed.

    Returns
    -------
    best        : solution_class    — best individual found
    history     : list[float]       — best RMSE per generation
    avg_history : list[float]       — average RMSE per generation
    """

    # initialise population
    population = [solution_class() for _ in range(population_size)]
    population = evaluate_population(population, n_jobs=n_jobs)
    current_mutation_rate = mutation_rate

    history = []
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
            p1 = selection_function(population)
            p2 = selection_function(population)

            o1, o2 = crossover_function(p1, p2, crossover_rate)

            mutation_function(o1, current_mutation_rate)
            mutation_function(o2, current_mutation_rate)

            offspring.append(o1)
            if len(next_gen) + len(offspring) < population_size:
                offspring.append(o2)

        offspring = evaluate_population(offspring, n_jobs=n_jobs)
        population = next_gen + offspring

        if mutation_rate_decay is not None and mutation_rate_decay > 0:
            current_mutation_rate /= (1 + mutation_rate_decay)

        if generation_callback is not None:
            generation_callback(gen, best_fitness, avg_fitness)

    population.sort(key=lambda ind: get_fitness(ind))
    best = population[0]

    if verbose:
        print(f"\nFinished. Best RMSE: {get_fitness(best):.4f}")

    return best, history, avg_history
