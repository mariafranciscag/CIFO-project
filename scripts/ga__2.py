import numpy as np
from copy import deepcopy
from typing import Callable

# Imports internos do teu próprio projeto
from .triangle_sol import TriangleSolution
from .ga_utils import get_fitness, evaluate_population
from operators.fitness_sharing import apply_fitness_sharing
from operators.selection import tournament_selection
from operators.crossover import one_point_crossover
from operators.mutation import triangle_gaussian_mutation

def genetic_algorithm_vermeer(
    population_size: int = 50,
    n_generations: int = 500,
    selection_function: Callable = None,
    crossover_function: Callable = None,
    mutation_function: Callable = None,
    structural_mutation_function: Callable = None, # Nova mutação de ordem de camadas
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.02,
    structural_mutation_rate: float = 0.05,
    mutation_rate_decay: float = 0.0,
    elitism_n = 2,
    n_jobs = -1,
    verbose = True,
    verbose_every = 50,
):
    # Inicialização padrão do teu projeto
    population = [TriangleSolution() for _ in range(population_size)]
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
            print(f"Gen {gen:>4d} | Best RMSE: {best_fitness:.4f} | Avg RMSE: {avg_fitness:.4f} | Mut Rate: {current_mutation_rate:.5f}")

        # Elitismo protegido
        next_gen = []
        for ind in population[:elitism_n]:
            elite = deepcopy(ind)
            elite._cached_fitness = get_fitness(ind)
            next_gen.append(elite)

        # Construção da nova geração
        offspring = []
        while len(next_gen) + len(offspring) < population_size:
            p1 = selection_function(population)
            p2 = selection_function(population)

            o1, o2 = crossover_function(p1, p2, crossover_rate)

            # CORREÇÃO CRÍTICA: Capturar obrigatoriamente o retorno da mutação (In-place protection)
            o1 = mutation_function(o1, current_mutation_rate)
            o2 = mutation_function(o2, current_mutation_rate)

            # Injeção da Mutação Estrutural (Troca de Camadas) para evitar estagnação de clones
            if structural_mutation_function is not None:
                o1 = structural_mutation_function(o1, structural_mutation_rate)
                o2 = structural_mutation_function(o2, structural_mutation_rate)

            offspring.append(o1)
            if len(next_gen) + len(offspring) < population_size:
                offspring.append(o2)

        offspring = evaluate_population(offspring, n_jobs=n_jobs)
        population = next_gen + offspring

        if mutation_rate_decay > 0:
            current_mutation_rate /= (1 + mutation_rate_decay)

    population.sort(key=lambda ind: get_fitness(ind))
    return population[0], history, avg_history