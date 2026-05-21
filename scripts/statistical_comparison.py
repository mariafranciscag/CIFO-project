from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
from multiprocessing import Manager
from queue import Empty
from threading import Thread

from operators.crossover import two_point_crossover
from operators.mutation import triangle_gaussian_mutation
from operators.selection import tournament_selection
from scripts.ga import genetic_algorithm
from scripts.triangle_sol import TriangleSolution


def _track_generation_progress(progress_queue, progress_bar, total_generations: int):
    """Consume worker generation updates and advance one shared progress bar."""
    completed = 0
    while completed < total_generations:
        try:
            update = progress_queue.get(timeout=0.1)
        except Empty:
            continue

        if update is None:
            break

        progress_bar.update(update)
        completed += update

def _run_single_ga(
    solution_class,
    population_size: int,
    n_generations: int,
    selection_function,
    crossover_function,
    mutation_function,
    crossover_rate: float,
    mutation_rate: float,
    mutation_rate_decay: float | None,
    elitism_n: int,
    target_array,  # Pass target array to workers
    progress_queue=None,
):
    """Helper function for parallel execution of a single GA run."""
    # Re-initialize target array in worker process (required for loky backend)
    TriangleSolution.target_array = target_array

    def report_generation(*_):
        if progress_queue is not None:
            progress_queue.put(1)

    _, hist, _ = genetic_algorithm(
        solution_class=solution_class,
        population_size=population_size,
        n_generations=n_generations,
        selection_function=selection_function,
        crossover_function=crossover_function,
        mutation_function=mutation_function,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        mutation_rate_decay=mutation_rate_decay,
        elitism_n=elitism_n,
        n_jobs=1,  # Single-threaded fitness eval to avoid nesting
        verbose=False,
        generation_callback=report_generation,
    )
    return hist


def run_ga_statistical_comparison(
    parameter_to_test: str,
    parameter_values: list,
    n_runs: int = 30,
    n_jobs: int = -1,
    n_generations: int = 200
):
    """
    Run GA multiple times with different parameter values and compute average fitness and std dev.
    
    Args:
        parameter_to_test: Name of parameter to vary
        parameter_values: List of values to test
        n_runs: Number of runs per parameter value
        n_jobs: Number of parallel jobs (-1 = all cores)
    
    Returns:
        Dictionary mapping parameter_value -> (avg_best_fitness, std_best_fitness)
    """
    results = {}
    
    # Collect all tasks: (parameter_value, run_index)
    tasks = [
        (parameter_value, i_run)
        for parameter_value in parameter_values
        for i_run in range(n_runs)
    ]
    
    task_generations = [
        parameter_value if parameter_to_test == "n_generations" else n_generations
        for parameter_value, _ in tasks
    ]
    total_generations = sum(task_generations)
    

    # Run all GA instances in parallel with progress bar
    print(
        f"\nStarting statistical comparison: {len(tasks)} GA runs, "
        f"{total_generations:,} total generations\n"
    )
    
    with Manager() as manager:
        progress_queue = manager.Queue()
        with tqdm(total=total_generations, desc="Progress", unit=" gen") as progress_bar:
            progress_thread = Thread(
                target=_track_generation_progress,
                args=(progress_queue, progress_bar, total_generations),
                daemon=True,
            )
            progress_thread.start()

            try:
                histories = Parallel(n_jobs=n_jobs, backend='loky')(
                    delayed(_run_single_ga)(
                        solution_class=parameter_value if parameter_to_test == "solution_class" else TriangleSolution,
                        population_size=parameter_value if parameter_to_test == "population_size" else 50,
                        n_generations=parameter_value if parameter_to_test == "n_generations" else n_generations,
                        selection_function=parameter_value if parameter_to_test == "selection_function" else tournament_selection,
                        crossover_function=parameter_value if parameter_to_test == "crossover_function" else two_point_crossover,
                        mutation_function=parameter_value if parameter_to_test == "mutation_function" else triangle_gaussian_mutation,
                        crossover_rate=parameter_value if parameter_to_test == "crossover_rate" else 0.9,
                        mutation_rate=parameter_value if parameter_to_test == "mutation_rate" else 0.02,
                        mutation_rate_decay=parameter_value if parameter_to_test == "mutation_rate_decay" else None,
                        elitism_n=parameter_value if parameter_to_test == "elitism_n" else 2,
                        target_array=TriangleSolution.target_array,
                        progress_queue=progress_queue,
                    )
                    for parameter_value, i_run in tasks
                )
            finally:
                progress_queue.put(None)
                progress_thread.join()
    
    # Reorganize results by parameter value
    task_idx = 0
    for parameter_value in parameter_values:
        histories_for_param = []
        for i_run in range(n_runs):
            histories_for_param.append(histories[task_idx])
            task_idx += 1
        
        # Convert to numpy array and compute mean and std dev
        histories_array = np.array(histories_for_param)
        avg_best_fitness = np.mean(histories_array, axis=0)
        std_best_fitness = np.std(histories_array, axis=0)
        results[parameter_value] = (avg_best_fitness, std_best_fitness)
    
    return results

def plot_statistical_comparison(
    results: dict,
    labels: list | None = None,
    title: str = "Statistical Comparison",
    ylabel: str = "Average Best RMSE (Lower is Better)",
    figsize: tuple = (12, 6),
):
    """
    Plot GA statistical results with standard deviation bands.
    
    Args:
        results: Dictionary mapping parameter_value -> (avg_fitness, std_fitness)
        labels: List of labels for each parameter value (defaults to sequential numbering)
        title: Plot title
        ylabel: Y-axis label
        figsize: Figure size as (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if labels is None:
        labels = [f"Parameter {i+1}" for i in range(len(results))]
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
    
    for idx, (label, (avg_fitness, std_fitness)) in zip(range(len(results)), results.items()):
        generations = np.arange(len(avg_fitness))
        ax.plot(generations, avg_fitness, label=labels[idx], linewidth=2.5, 
                marker='o', markersize=3, color=colors[idx])
        ax.fill_between(generations, avg_fitness - std_fitness, avg_fitness + std_fitness, 
                         alpha=0.2, color=colors[idx])
    
    ax.set_xlabel("Generation", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
