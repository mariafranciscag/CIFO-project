import numpy as np
from .solution import Solution

def tournament_selection(population: list[Solution], tournament_size) -> list[Solution]:
    
    selected = []
    
    for _ in range(len(population)):
        
        # Draw K individuals from the population with uniform probability and with replacement
        tournament: np.ndarray = np.random.choice(population, size=tournament_size, replace=True)
        
        # Select the best fitness inside the tournament
        winner = min(tournament, key=lambda individual: individual.fitness)
        selected.append(winner)
    
    return selected
