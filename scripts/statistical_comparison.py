from joblib import Parallel, delayed
from matplotlib import pyplot as plt
from tqdm import tqdm
import numpy as np
from multiprocessing import Manager
from queue import Empty
from threading import Thread
from scipy import stats
import pickle
from pathlib import Path
import plotly.graph_objects as go
from plotly.colors import qualitative

from operators.crossover import one_point_crossover
from operators.mutation import triangle_gaussian_mutation
from operators.selection import tournament_selection
from scripts.ga import genetic_algorithm
from scripts.triangle_sol import TriangleSolution


def _track_evaluation_progress(progress_queue, progress_bar, total_evaluations: int):
    """Consume worker evaluation updates and advance one shared progress bar."""
    completed = 0
    while completed < total_evaluations:
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
            progress_queue.put(population_size)

    best, hist, _ = genetic_algorithm(
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
    return hist, best.fitness()


def _calculate_population_size(parameter_value, parameter_to_test, eval_budget):
    if parameter_to_test == "population_size":
        return parameter_value
    elif parameter_to_test == "n_generations":
        return eval_budget // parameter_value
    else:
        return 50

def _calculate_n_generations(parameter_value, parameter_to_test, eval_budget):
    if parameter_to_test == "n_generations":
        return parameter_value
    elif parameter_to_test == "population_size":
        return eval_budget // parameter_value
    else:
        return eval_budget // 50 # add more generations if eval budget is increased

def run_ga_statistical_comparison(
    parameter_to_test: str,
    parameter_values: list,
    n_runs: int = 30,
    n_jobs: int = -1,
    # n_generations: int = 200,
    # population_size = 50,
    eval_budget: int = 10_000, # 50 individuals x 200 generations = 10 000 evaluations
    progress_smoothing: float = 0.0,
    progress_mininterval: float = 1.0,
):
    """
    Run GA multiple times with different parameter values and compute average fitness and std dev.
    
    Args:
        parameter_to_test: Name of parameter to vary
        parameter_values: List of values to test
        n_runs: Number of runs per parameter value
        n_jobs: Number of parallel jobs (-1 = all cores)
        progress_smoothing: tqdm smoothing factor. 0.0 reports the whole-run
            average speed/ETA, which is less jumpy for parallel GA runs.
        progress_mininterval: Minimum seconds between progress bar refreshes.
    
    Returns:
        Dictionary mapping each parameter value to mean/std histories and final
        best fitness values for each run.
    """
    results = {}
    
    # Collect all tasks: (parameter_value, run_index)
    tasks = [
        (parameter_value, i_run)
        for parameter_value in parameter_values
        for i_run in range(n_runs)
    ]
    
    task_configs = [
        (
            _calculate_population_size(parameter_value, parameter_to_test, eval_budget),
            _calculate_n_generations(parameter_value, parameter_to_test, eval_budget),
        )
        for parameter_value, _ in tasks
    ]
    task_evaluations = [
        population_size * n_generations
        for population_size, n_generations in task_configs
    ]
    total_evaluations = sum(task_evaluations)
    

    # Run all GA instances in parallel with progress bar
    print(
        f"\nStarting statistical comparison: {len(tasks)} GA runs, "
        f"{total_evaluations:,} total evaluations\n"
    )
    
    with Manager() as manager:
        progress_queue = manager.Queue()
        with tqdm(
            total=total_evaluations,
            desc="Progress",
            unit=" evaluation",
            smoothing=progress_smoothing,
            mininterval=progress_mininterval,
        ) as progress_bar:
            progress_thread = Thread(
                target=_track_evaluation_progress,
                args=(progress_queue, progress_bar, total_evaluations),
                daemon=True,
            )
            progress_thread.start()

            try:
                run_outputs = Parallel(n_jobs=n_jobs, backend='loky')(
                    delayed(_run_single_ga)(
                        solution_class=parameter_value if parameter_to_test == "solution_class" else TriangleSolution,
                        population_size=_calculate_population_size(parameter_value, parameter_to_test, eval_budget),
                        n_generations=_calculate_n_generations(parameter_value, parameter_to_test, eval_budget),
                        selection_function=parameter_value if parameter_to_test == "selection_function" else tournament_selection,
                        crossover_function=parameter_value if parameter_to_test == "crossover_function" else one_point_crossover,
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
    for i_param, parameter_value in enumerate(parameter_values):
        histories_for_param = []
        final_values_for_param = []
        population_size = _calculate_population_size(parameter_value, parameter_to_test, eval_budget)
        n_generations = _calculate_n_generations(parameter_value, parameter_to_test, eval_budget)
        
        for i_run in range(n_runs):
            history, final_best = run_outputs[i_param * n_runs + i_run]
            histories_for_param.append(history)
            final_values_for_param.append(final_best)
        
        # Convert to numpy array and compute mean and std dev
        histories_array = np.array(histories_for_param)
        final_values_array = np.array(final_values_for_param)
        evaluation_axis = np.arange(1, histories_array.shape[1] + 1) * population_size
        
        results[parameter_value] = {
            "mean_history": np.mean(histories_array, axis=0),
            "median_history": np.median(histories_array, axis=0),
            "std_history": np.std(histories_array, axis=0),
            "best_per_run": final_values_array,
            "evaluation_axis": evaluation_axis,
            "population_size": population_size,
            "n_generations": n_generations,
            "evaluation_budget": population_size * n_generations,
        }
    
    return results


def plot_statistical_comparison(
    results: dict,
    labels: list | None = None,
    title: str = "Statistical Comparison",
    ylabel: str = "Average Best RMSE (Lower is Better)",
    figsize: tuple = (12, 6),
    x_axis: str = "evaluations",
    statistic: str = "mean",
):
    """
    Plot GA statistical results with standard deviation bands.
    
    Args:
        results: Dictionary returned by run_ga_statistical_comparison
        labels: List of labels for each parameter value (defaults to sequential numbering)
        title: Plot title
        ylabel: Y-axis label
        figsize: Figure size as (width, height)
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    if labels is None:
        labels = [f"Parameter {i+1}" for i in range(len(results))]
    
    colors = plt.cm.Set1(np.linspace(0, 1, len(results)))
    
    for idx, (parameter_value, stats) in enumerate(results.items()):
        avg_fitness = stats["median_history"] if statistic == "median" else stats["mean_history"]
        std_fitness = stats["std_history"]
        x_values = _get_plot_x_values(stats, x_axis=x_axis)
        ax.plot(x_values, avg_fitness, label=labels[idx], linewidth=2.5, 
                marker='o', markersize=3, color=colors[idx])
        if statistic != "median":
            ax.fill_between(x_values, avg_fitness - std_fitness, avg_fitness + std_fitness, 
                             alpha=0.2, color=colors[idx])
    
    ax.set_xlabel(_get_x_axis_label(x_axis), fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(fontsize=11, loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def _get_plot_x_values(result_stats: dict, x_axis: str = "evaluations"):
    """Return the x-axis values for one result series."""
    if x_axis == "generations":
        return np.arange(len(result_stats["mean_history"]))

    if "evaluation_axis" in result_stats:
        return np.asarray(result_stats["evaluation_axis"])

    population_size = result_stats.get("population_size", 1)
    return np.arange(1, len(result_stats["mean_history"]) + 1) * population_size


def _get_x_axis_label(x_axis: str = "evaluations"):
    if x_axis == "generations":
        return "Generation"
    return "Cumulative fitness evaluations"


def run_statistical_test(values_a, values_b, name_a="A", name_b="B", paired=False, alpha=0.05):
    values_a = np.asarray(values_a)
    values_b = np.asarray(values_b)

    # 1. Normality tests
    shapiro_a = stats.shapiro(values_a)
    shapiro_b = stats.shapiro(values_b)

    print(f"{name_a} normality p-value: {shapiro_a.pvalue:.4f}")
    print(f"{name_b} normality p-value: {shapiro_b.pvalue:.4f}")

    normal_a = shapiro_a.pvalue >= alpha
    normal_b = shapiro_b.pvalue >= alpha

    # 2. Choose comparison test
    if normal_a and normal_b:
        test = stats.ttest_ind(values_a, values_b, equal_var=False)
        test_name = "Welch t-test"
    else:
        if paired:
            test = stats.wilcoxon(values_a, values_b)
            test_name = "Wilcoxon signed-rank test"
        else:
            test = stats.mannwhitneyu(values_a, values_b, alternative="two-sided")
            test_name = "Mann-Whitney U / Wilcoxon rank-sum test"

    print(f"\n{test_name}")
    print(f"p-value: {test.pvalue:.4f}")

    print(f"\n{name_a} mean final RMSE: {values_a.mean():.4f}")
    print(f"{name_b} mean final RMSE: {values_b.mean():.4f}")

    if test.pvalue < alpha:
        print("Difference is statistically significant.")
    else:
        print("No statistically significant difference detected.")


def save_statistical_results(results: dict, filepath: str):
    """
    Save results returned by run_ga_statistical_comparison to a pickle file.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "wb") as f:
        pickle.dump(results, f)


def load_statistical_results(filepath: str) -> dict:
    """
    Load saved statistical comparison results.
    """
    with open(filepath, "rb") as f:
        return pickle.load(f)


def plot_statistical_comparison_interactive(
    results: dict,
    labels: list | None = None,
    title: str = "Statistical Comparison",
    ylabel: str = "Average Best RMSE (Lower is Better)",
    width: int = 1000,
    height: int = 550,
    x_axis: str = "evaluations",
    statistic: str = "mean",
):
    """
    Interactive GA statistical results plot with standard deviation bands.
    Click legend items to hide/show each plotted result.

    Args:
        results: Dictionary returned by run_ga_statistical_comparison
        labels: List of labels for each parameter value
        title: Plot title
        ylabel: Y-axis label
        width: Figure width in pixels
        height: Figure height in pixels
    """
    if labels is None:
        labels = [f"Parameter {i + 1}" for i in range(len(results))]

    colors = qualitative.Set1
    fig = go.Figure()

    for idx, (parameter_value, stats) in enumerate(results.items()):
        avg_fitness = np.asarray(stats["median_history"] if statistic == "median" else stats["mean_history"])
        std_fitness = np.asarray(stats["std_history"])
        x_values = _get_plot_x_values(stats, x_axis=x_axis)

        label = labels[idx]
        color = colors[idx % len(colors)]

        upper = avg_fitness + std_fitness
        lower = avg_fitness - std_fitness

        # Upper boundary of std band
        if statistic != "median":
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=upper,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    hoverinfo="skip",
                    legendgroup=label,
                    name=f"{label} std upper",
                )
            )

            # Lower boundary, filled to previous upper trace
            fig.add_trace(
                go.Scatter(
                    x=x_values,
                    y=lower,
                    mode="lines",
                    line=dict(width=0),
                    fill="tonexty",
                    fillcolor=color.replace("rgb", "rgba").replace(")", ", 0.20)")
                    if color.startswith("rgb")
                    else color,
                    showlegend=False,
                    hoverinfo="skip",
                    legendgroup=label,
                    name=f"{label} std band",
                )
            )

        # Mean line
        fig.add_trace(
            go.Scatter(
                x=x_values,
                y=avg_fitness,
                mode="lines+markers",
                name=label,
                legendgroup=label,
                line=dict(color=color, width=2.5),
                marker=dict(size=5),
                customdata=np.stack([std_fitness], axis=-1),
                hovertemplate=(
                    f"{_get_x_axis_label(x_axis)}: %{{x}}<br>"
                    "Mean: %{y:.5f}<br>"
                    "Std: %{customdata[0]:.5f}"
                    "<extra></extra>"
                ),
            )
        )

    fig.update_layout(
        title=title,
        width=width,
        height=height,
        xaxis_title=_get_x_axis_label(x_axis),
        yaxis_title=ylabel,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            title="Click to hide/show",
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
        margin=dict(l=70, r=30, t=70, b=60),
    )

    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.12)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.12)")

    fig.show()
