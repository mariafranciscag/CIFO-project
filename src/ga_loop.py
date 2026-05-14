import numpy as np
import datetime
from typing import Callable

from src.solution import Solution

def base_ga_loop(
    solution_class: Callable,
    population_size: int,
    n_iterations: int,
    selection_function: Callable,
    crossover_function: Callable,
    crossover_rate: float,
    mutation_function: Callable,
    mutation_rate: float,
    progress_callback: Callable | None = None,
    verbose: bool = True,
) -> tuple[Solution, dict]:
    
    # create population with given size
    population = [ solution_class() for _ in range(population_size) ]
    for individual in population:
        individual.calculate_fitness()
    
    best_solution = population[0]
    best_fitness = population[0].fitness
    
    history = {
        "avg_fitness": [],
        "best_fitness": []
    }
    
    iteration_start_time = datetime.datetime.now()
    
    for i in range(n_iterations):
        
        # Selection phase
        parents = selection_function(population)
        
        offspring: list[Solution] = []
        # Reproduction with variation phase
        for j in range(0, len(parents), 2):
            parent1 = parents[j]
            parent2 = parents[j + 1] 
            
            child1, child2 = crossover_function(parent1, parent2, crossover_rate)
            
            child1 = mutation_function(child1, mutation_rate)
            child2 = mutation_function(child2, mutation_rate)
            
            offspring.extend([child1, child2])
        
        # Update the population and best solution
        population = offspring
        population_fitness = [ individual.calculate_fitness() for individual in population ]
        average_fitness = np.average(population_fitness)
        best_fitness_index = np.argmin(population_fitness)
        
        if population_fitness[best_fitness_index] <= best_fitness:
            best_fitness = population_fitness[best_fitness_index]
            best_solution = population[best_fitness_index]
        
        iteration_running_time = datetime.datetime.now() - iteration_start_time
        iteration_start_time = datetime.datetime.now()
        
        history["avg_fitness"].append(average_fitness)
        history["best_fitness"].append(best_fitness)
        
        if progress_callback is not None:
            progress_callback()
        
        if verbose and ((i + 1) % 10 == 0 or i == 0):
            print(f"Iteration {i + 1} - Fitness: Avg: {average_fitness:.2f}, Best: {best_fitness:.2f}; Mutation rate: {mutation_rate:.3f}; {iteration_running_time.total_seconds():.2f}s/iteration")
        
    return best_solution, history


def mutation_weight_decay_ga_loop(
    solution_class: Callable,
    population_size: int,
    n_iterations: int,
    selection_function: Callable,
    crossover_function: Callable,
    crossover_rate: float,
    mutation_function: Callable,
    mutation_rate: float,
    mutation_rate_decay: float | None = None,
    progress_callback: Callable | None = None,
    verbose: bool = True,
) -> tuple[Solution, dict]:
    
    # create population with given size
    population = [ solution_class() for _ in range(population_size) ]
    for individual in population:
        individual.calculate_fitness()
    
    best_solution = population[0]
    best_fitness = population[0].fitness
    
    history = {
        "avg_fitness": [],
        "best_fitness": []
    }
    
    iteration_start_time = datetime.datetime.now()
    
    for i in range(n_iterations):
        
        # Selection phase
        parents = selection_function(population)
        
        offspring: list[Solution] = []
        # Reproduction with variation phase
        for j in range(0, len(parents), 2):
            parent1 = parents[j]
            parent2 = parents[j + 1] 
            
            child1, child2 = crossover_function(parent1, parent2, crossover_rate)
            
            child1 = mutation_function(child1, mutation_rate)
            child2 = mutation_function(child2, mutation_rate)
            
            offspring.extend([child1, child2])
        
        # Update the population and best solution
        population = offspring
        population_fitness = [ individual.calculate_fitness() for individual in population ]
        average_fitness = np.average(population_fitness)
        best_fitness_index = np.argmin(population_fitness)
        
        if population_fitness[best_fitness_index] <= best_fitness:
            best_fitness = population_fitness[best_fitness_index]
            best_solution = population[best_fitness_index]
        
        iteration_running_time = datetime.datetime.now() - iteration_start_time
        iteration_start_time = datetime.datetime.now()
        
        history["avg_fitness"].append(average_fitness)
        history["best_fitness"].append(best_fitness)
        
        if progress_callback is not None:
            progress_callback()
        
        if verbose and ((i + 1) % 10 == 0 or i == 0):
            print(f"Iteration {i + 1} - Fitness: Avg: {average_fitness:.2f}, Best: {best_fitness:.2f}; Mutation rate: {mutation_rate:.3f}; {iteration_running_time.total_seconds():.2f}s/iteration")
        
        if mutation_rate_decay:
            mutation_rate /= 1 + mutation_rate_decay
        
    return best_solution, history
