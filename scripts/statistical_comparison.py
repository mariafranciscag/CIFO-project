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
from functools import partial
import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.subplots import make_subplots

from operators.crossover import one_point_crossover
from operators.mutation import triangle_gaussian_mutation
from operators.selection import tournament_selection
from operators.new_fitness_sharing import population_diversity_report
from scripts.ga import genetic_algorithm
from scripts.triangle_sol import TriangleSolution


def _is_pickleable(value) -> bool:
    try:
        pickle.dumps(value)
    except Exception:
        return False
    return True


def _safe_pickle_label(value) -> str:
    if isinstance(value, str):
        return value
    if callable(value):
        module = getattr(value, "__module__", None)
        qualname = getattr(value, "__qualname__", None)
        if module and qualname and "<lambda>" not in qualname:
            return f"{module}.{qualname}"
    return repr(value)



def _make_pickleable(value):
    if isinstance(value, dict):
        safe_dict = {}
        for key, item in value.items():
            safe_key = _safe_pickle_label(key)
            safe_item = _make_pickleable(item)

            if safe_key in safe_dict:
                base_key = str(safe_key)
                suffix = 2
                while f"{base_key} [{suffix}]" in safe_dict:
                    suffix += 1
                safe_key = f"{base_key} [{suffix}]"

            safe_dict[safe_key] = safe_item
        return safe_dict

    if _is_pickleable(value):
        return value

    if isinstance(value, list):
        return [_make_pickleable(item) for item in value]

    if isinstance(value, tuple):
        return tuple(_make_pickleable(item) for item in value)

    return _safe_pickle_label(value)


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
    seed: int | None = None,
    return_representation: bool = False,
    return_diversity: bool = False,
    diversity_interval: int = 10,
):
    """Helper function for parallel execution of a single GA run."""
    # Re-initialize target array in worker process (required for loky backend)
    TriangleSolution.target_array = target_array
    diversity_log = []

    def report_generation(*_):
        if hasattr(selection_function, 'next_generation'):
            selection_function.next_generation()
        if progress_queue is not None:
            progress_queue.put(population_size)

    def diversity_callback(gen, population):
        if gen % diversity_interval == 0:
            diversity_log.append({'gen': gen, **population_diversity_report(population)})

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
        population_callback=diversity_callback if return_diversity else None,
        seed=seed,
    )
    result = [hist, best.fitness()]
    if return_representation:
        result.append(np.asarray(best.representation))
    if return_diversity:
        result.append(diversity_log)
    return tuple(result)


def _calculate_population_size(parameter_value, parameter_to_test, eval_budget, default_population_size=50):
    if parameter_to_test == "population_size":
        return parameter_value
    elif parameter_to_test == "n_generations":
        return eval_budget // parameter_value
    else:
        return default_population_size

def _calculate_n_generations(parameter_value, parameter_to_test, eval_budget, default_population_size=50):
    if parameter_to_test == "n_generations":
        return parameter_value
    elif parameter_to_test == "population_size":
        return eval_budget // parameter_value
    else:
        return eval_budget // default_population_size # add more generations if eval budget is increased

def _select_mutation_function(parameter_value, parameter_to_test):
    if parameter_to_test == "mutation_function":
        return parameter_value
    if parameter_to_test == "sigma_vertex":
        return partial(triangle_gaussian_mutation, sigma_vertex=parameter_value)
    if parameter_to_test in ("sigma_colour", "sigma_color"):
        return partial(triangle_gaussian_mutation, sigma_colour=parameter_value)
    return triangle_gaussian_mutation

def _param_value(parameter_name, default_value, parameter_value, parameter_to_test, fixed_params):
    if parameter_to_test == parameter_name:
        return parameter_value
    return fixed_params.get(parameter_name, default_value)

def _build_run_kwargs(
    parameter_value,
    parameter_to_test,
    eval_budget,
    default_population_size,
    fixed_params,
    target_array,
    progress_queue,
    seed: int | None = None,
):
    population_size = _calculate_population_size(
        parameter_value,
        parameter_to_test,
        eval_budget,
        default_population_size,
    )
    n_generations = _calculate_n_generations(
        parameter_value,
        parameter_to_test,
        eval_budget,
        default_population_size,
    )

    if parameter_to_test != "population_size":
        population_size = fixed_params.get("population_size", population_size)
    if parameter_to_test != "n_generations":
        n_generations = fixed_params.get("n_generations", n_generations)

    mutation_function = _select_mutation_function(parameter_value, parameter_to_test)
    if parameter_to_test not in ("mutation_function", "sigma_vertex", "sigma_colour", "sigma_color"):
        mutation_function = fixed_params.get("mutation_function", mutation_function)

    return {
        "solution_class": _param_value("solution_class", TriangleSolution, parameter_value, parameter_to_test, fixed_params),
        "population_size": population_size,
        "n_generations": n_generations,
        "selection_function": _param_value("selection_function", tournament_selection, parameter_value, parameter_to_test, fixed_params),
        "crossover_function": _param_value("crossover_function", one_point_crossover, parameter_value, parameter_to_test, fixed_params),
        "mutation_function": mutation_function,
        "crossover_rate": _param_value("crossover_rate", 0.9, parameter_value, parameter_to_test, fixed_params),
        "mutation_rate": _param_value("mutation_rate", 0.02, parameter_value, parameter_to_test, fixed_params),
        "mutation_rate_decay": _param_value("mutation_rate_decay", None, parameter_value, parameter_to_test, fixed_params),
        "elitism_n": _param_value("elitism_n", 2, parameter_value, parameter_to_test, fixed_params),
        "target_array": target_array,
        "progress_queue": progress_queue,
        "seed": seed,
    }

def run_ga_statistical_comparison(
    parameter_to_test: str,
    parameter_values: list,
    n_runs: int = 30,
    n_jobs: int = -1,
    n_generations: int | None = None,
    population_size: int = 50,
    eval_budget: int = 10_000, # 50 individuals x 200 generations = 10 000 evaluations
    progress_smoothing: float = 0.0,
    progress_mininterval: float = 1.0,
    fixed_params: dict | None = None,
    base_seed: int | None = None,
    return_representations: bool = False,
    return_diversity: bool = False,
    diversity_interval: int = 10,
):
    """
    Run GA multiple times with different parameter values and compute average fitness and std dev.

    Args:
        parameter_to_test: Name of parameter to vary
        parameter_values: List of values to test
        n_runs: Number of runs per parameter value
        n_jobs: Number of parallel jobs (-1 = all cores)
        n_generations: Optional fixed generation count for non-population-size
            comparisons. Kept for backward compatibility with older notebook cells.
        population_size: Population size for non-population-size comparisons.
        eval_budget: Fitness-evaluation budget used when n_generations is not
            supplied, and always used for population-size comparisons.
        progress_smoothing: tqdm smoothing factor. 0.0 reports the whole-run
            average speed/ETA, which is less jumpy for parallel GA runs.
        progress_mininterval: Minimum seconds between progress bar refreshes.
        fixed_params: Optional GA parameter overrides used for every run unless
            that same parameter is the one being tested. Example:
            {"crossover_function": uniform_crossover, "mutation_rate": 0.01}.
        base_seed: Optional base seed. When set, run i_run receives
            seed = base_seed + i_run, making the experiment reproducible and
            pairing equivalent runs across parameter values.
    
    Returns:
        Dictionary mapping each parameter value to mean/std histories and final
        best fitness values for each run.
    """
    results = {}
    fixed_params = fixed_params or {}
    if n_generations is not None and parameter_to_test != "population_size":
        eval_budget = population_size * n_generations
    
    # Collect all tasks: (parameter_value, run_index)
    tasks = [
        (parameter_value, i_run)
        for parameter_value in parameter_values
        for i_run in range(n_runs)
    ]
    
    task_configs = [
        (
            _build_run_kwargs(
                parameter_value=parameter_value,
                parameter_to_test=parameter_to_test,
                eval_budget=eval_budget,
                default_population_size=population_size,
                fixed_params=fixed_params,
                target_array=TriangleSolution.target_array,
                progress_queue=None,
            )["population_size"],
            _build_run_kwargs(
                parameter_value=parameter_value,
                parameter_to_test=parameter_to_test,
                eval_budget=eval_budget,
                default_population_size=population_size,
                fixed_params=fixed_params,
                target_array=TriangleSolution.target_array,
                progress_queue=None,
            )["n_generations"],
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
                        **_build_run_kwargs(
                            parameter_value=parameter_value,
                            parameter_to_test=parameter_to_test,
                            eval_budget=eval_budget,
                            default_population_size=population_size,
                            fixed_params=fixed_params,
                            target_array=TriangleSolution.target_array,
                            progress_queue=progress_queue,
                            seed=(base_seed + i_run) if base_seed is not None else None,
                        ),
                        return_representation=return_representations,
                        return_diversity=return_diversity,
                        diversity_interval=diversity_interval,
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
        representations_for_param = []
        run_kwargs = _build_run_kwargs(
            parameter_value=parameter_value,
            parameter_to_test=parameter_to_test,
            eval_budget=eval_budget,
            default_population_size=population_size,
            fixed_params=fixed_params,
            target_array=TriangleSolution.target_array,
            progress_queue=None,
        )
        actual_population_size = run_kwargs["population_size"]
        actual_n_generations = run_kwargs["n_generations"]
        
        diversity_logs_for_param = []
        for i_run in range(n_runs):
            output = list(run_outputs[i_param * n_runs + i_run])
            history = output.pop(0)
            final_best = output.pop(0)
            if return_representations:
                representations_for_param.append(output.pop(0))
            if return_diversity:
                diversity_logs_for_param.append(output.pop(0))
                
            histories_for_param.append(history)
            final_values_for_param.append(final_best)
        
        # Convert to numpy array and compute mean and std dev
        histories_array = np.array(histories_for_param)
        final_values_array = np.array(final_values_for_param)
        evaluation_axis = np.arange(1, histories_array.shape[1] + 1) * actual_population_size
        
        result_entry = {
            "mean_history": np.mean(histories_array, axis=0),
            "median_history": np.median(histories_array, axis=0),
            "std_history": np.std(histories_array, axis=0),
            "best_per_run": final_values_array,
            "evaluation_axis": evaluation_axis,
            "population_size": actual_population_size,
            "n_generations": actual_n_generations,
            "evaluation_budget": actual_population_size * actual_n_generations,
        }
        if return_representations:
            result_entry["representations"] = representations_for_param
        if return_diversity and len(diversity_logs_for_param) > 0:
            mean_div_log = []
            num_points = len(diversity_logs_for_param[0])
            metrics = [k for k in diversity_logs_for_param[0][0].keys() if k != 'gen']
            for pt in range(num_points):
                point_dict = {'gen': diversity_logs_for_param[0][pt]['gen']}
                for key in metrics:
                    point_dict[key] = float(np.mean([d[pt][key] for d in diversity_logs_for_param]))
                mean_div_log.append(point_dict)
            result_entry["mean_diversity_history"] = mean_div_log
            
        results[parameter_value] = result_entry
    
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
    
    for idx, (parameter_value, stat_vals) in enumerate(results.items()):
        avg_fitness = stat_vals["median_history"] if statistic == "median" else stat_vals["mean_history"]
        std_fitness = stat_vals["std_history"]
        x_values = _get_plot_x_values(stat_vals, x_axis=x_axis)
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
    if paired:
        if normal_a and normal_b:
            test = stats.ttest_rel(values_a, values_b)
            test_name = "Paired t-test"
        else:
            test = stats.wilcoxon(values_a, values_b)
            test_name = "Wilcoxon signed-rank test"
    else:
        if normal_a and normal_b:
            test = stats.ttest_ind(values_a, values_b, equal_var=False)
            test_name = "Welch t-test"
        else:
            test = stats.mannwhitneyu(values_a, values_b, alternative="two-sided")
            test_name = "Mann-Whitney U test"

    print(f"\n{test_name}")
    print(f"p-value: {test.pvalue:.6g}")

    print(f"\n{name_a} mean final score: {values_a.mean():.4f}")
    print(f"{name_b} mean final score: {values_b.mean():.4f}")

    if test.pvalue < alpha:
        print("Difference is statistically significant.")
    else:
        print("No statistically significant difference detected.")


def run_multi_group_statistical_test(groups, names=None, alpha=0.05, paired=False):
    """
    Perform statistical comparison across multiple groups (arbitrary number of groups).
    Supports:
    1. A dictionary returned by `run_ga_statistical_comparison` (extracts 'best_per_run')
    2. A dictionary of {group_name: [values]}
    3. A list of lists/arrays of values, with an optional list of group names.
    
    Checks normality for each group (Shapiro-Wilk) and performs:
    - Omnibus test: One-way ANOVA (if all normal) or Kruskal-Wallis (if any non-normal).
      (Or Friedman test if paired=True).
    - Post-hoc pairwise tests with Holm-Bonferroni correction if omnibus is significant.
    """
    # 1. Parse inputs
    group_names = []
    group_data = []

    if isinstance(groups, dict):
        if names is not None:
            if len(names) != len(groups):
                raise ValueError("Length of names must match length of groups list.")
        for i, (key, val) in enumerate(groups.items()):
            if names is not None:
                group_names.append(str(names[i]))
            else:
                group_names.append(str(key))
            if isinstance(val, dict) and "best_per_run" in val:
                group_data.append(np.asarray(val["best_per_run"]))
            else:
                group_data.append(np.asarray(val))
    elif isinstance(groups, list):
        if names is not None:
            if len(names) != len(groups):
                raise ValueError("Length of names must match length of groups list.")
            group_names = [str(n) for n in names]
        else:
            group_names = [f"Group {i+1}" for i in range(len(groups))]
        group_data = [np.asarray(g) for g in groups]
    else:
        raise TypeError("groups must be a dictionary or a list of arrays/lists.")

    n_groups = len(group_data)
    if n_groups < 2:
        print("At least 2 groups are required for statistical testing.")
        return

    # 2. Normality tests (Shapiro-Wilk)
    normality_results = []
    all_normal = True
    print("=== Normality Checks (Shapiro-Wilk) ===")
    for name, data in zip(group_names, group_data):
        shapiro_res = stats.shapiro(data)
        p_val = shapiro_res.pvalue
        is_normal = p_val >= alpha
        normality_results.append(is_normal)
        if not is_normal:
            all_normal = False
        print(f"  - {name:<30} p-value: {p_val:.4f} (Normal: {is_normal})")
    print()

    # 3. Omnibus test
    print("=== Omnibus Statistical Test ===")
    if paired:
        lengths = [len(g) for g in group_data]
        if len(set(lengths)) > 1:
            raise ValueError("All groups must have the same number of samples for paired tests.")
            
        print("Running paired omnibus test (Friedman test)...")
        res = stats.friedmanchisquare(*group_data)
        test_name = "Friedman test"
    else:
        if all_normal:
            print("All groups are approximately normal. Running One-way ANOVA...")
            res = stats.f_oneway(*group_data)
            test_name = "One-way ANOVA"
        else:
            print("At least one group is non-normal. Running Kruskal-Wallis H-test...")
            res = stats.kruskal(*group_data)
            test_name = "Kruskal-Wallis H-test"

    omnibus_p = res.pvalue
    print(f"Omnibus test used: {test_name}")
    print(f"p-value: {omnibus_p:.6f}")
    
    # Print summary statistics
    print("\n=== Summary Statistics ===")
    for name, data in zip(group_names, group_data):
        print(f"  - {name:<30} Mean: {data.mean():.4f} | Median: {np.median(data):.4f} | Std: {data.std():.4f}")
    print()

    test_results = {
        "omnibus_test": test_name,
        "omnibus_p": omnibus_p,
        "pairwise_comparisons": []
    }

    if omnibus_p >= alpha:
        print("Omnibus test is NOT statistically significant (p >= alpha).")
        print("No post-hoc pairwise comparisons will be performed.")
        return

    print("Omnibus test is statistically significant (p < alpha)!")
    print("Performing post-hoc pairwise comparisons with Holm-Bonferroni correction...")
    print()
    
    comparisons = []
    for i in range(n_groups):
        for j in range(i + 1, n_groups):
            name_a, data_a = group_names[i], group_data[i]
            name_b, data_b = group_names[j], group_data[j]
            
            is_normal_a = normality_results[i]
            is_normal_b = normality_results[j]
            
            if paired:
                if is_normal_a and is_normal_b:
                    test_res = stats.ttest_rel(data_a, data_b)
                    pairwise_test_name = "Paired t-test"
                else:
                    test_res = stats.wilcoxon(data_a, data_b)
                    pairwise_test_name = "Wilcoxon signed-rank"
            else:
                if is_normal_a and is_normal_b:
                    test_res = stats.ttest_ind(data_a, data_b, equal_var=False)
                    pairwise_test_name = "Welch t-test"
                else:
                    test_res = stats.mannwhitneyu(data_a, data_b, alternative="two-sided")
                    pairwise_test_name = "Mann-Whitney U"
                    
            comparisons.append({
                "pair": (name_a, name_b),
                "raw_p": test_res.pvalue,
                "test_used": pairwise_test_name,
                "mean_diff": data_a.mean() - data_b.mean(),
            })
            
    comparisons.sort(key=lambda x: x["raw_p"])
    m = len(comparisons)
    
    print(f"{'Comparison':<50} | {'Test Used':<22} | {'Raw p-value':<12} | {'Adj p-value':<12} | Significant?")
    print("-" * 115)
    
    last_sig = True
    for k, comp in enumerate(comparisons, start=1):
        raw_p = comp["raw_p"]
        adj_p = raw_p * (m - k + 1)
        if k > 1:
            prev_adj_p = comparisons[k-2]["adj_p"]
            adj_p = max(adj_p, prev_adj_p)
        adj_p = min(1.0, adj_p)
        comp["adj_p"] = adj_p
        
        is_sig = adj_p < alpha and last_sig
        if not is_sig:
            last_sig = False
            
        signif_str = "YES" if is_sig else "NO"
        comp["significant"] = is_sig
        pair_str = f"{comp['pair'][0]} vs {comp['pair'][1]}"
        print(f"{pair_str:<50} | {comp['test_used']:<22} | {raw_p:<12.6f} | {adj_p:<12.6f} | {signif_str}")
        
    print()
    test_results["pairwise_comparisons"] = comparisons


def save_statistical_results(results: dict, filepath: str):
    """
    Save results returned by run_ga_statistical_comparison to a pickle file.

    Notebook-defined lambdas cannot be pickled, so unpickleable keys/values are
    converted to readable string labels before saving.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    serializable_results = _make_pickleable(results)

    with open(filepath, "wb") as f:
        pickle.dump(serializable_results, f)


def _unquote_result_key(key):
    if isinstance(key, str) and len(key) >= 2 and key[0] == key[-1] == "'":
        return key[1:-1]
    return key


def _normalize_loaded_result_keys(value):
    if isinstance(value, dict):
        return {
            _unquote_result_key(key): _normalize_loaded_result_keys(item)
            for key, item in value.items()
        }

    if isinstance(value, list):
        return [_normalize_loaded_result_keys(item) for item in value]

    if isinstance(value, tuple):
        return tuple(_normalize_loaded_result_keys(item) for item in value)

    return value


def load_statistical_results(filepath: str) -> dict:
    """
    Load saved statistical comparison results.

    Some older result files were saved after applying key sanitization
    recursively, which left inner result keys quoted as strings such as
    ``'mean_history'``. Normalize them on load so old notebooks remain usable.
    """
    with open(filepath, "rb") as f:
        return _normalize_loaded_result_keys(pickle.load(f))


def plot_statistical_comparison_interactive(
    results: dict,
    labels: list | None = None,
    title: str = "Statistical Comparison",
    ylabel: str | list = "Average Best RMSE (Lower is Better)",
    width: int = 1000,
    height: int = 550,
    x_axis: str = "evaluations",
    statistic: str = "mean",
    subplots: bool = False,
):
    """
    Interactive GA statistical results plot with standard deviation bands.
    Click legend items to hide/show each plotted result.

    Args:
        results: Dictionary returned by run_ga_statistical_comparison
        labels: List of labels for each parameter value
        title: Plot title
        ylabel: Y-axis label (str) or list of labels (one per result).
            A list is mainly useful with ``subplots=True`` when each series
            has its own scale (e.g. RMSE vs ΔE2000).
        width: Figure width in pixels
        height: Figure height in pixels
        subplots: When True, each result gets its own subplot row with an
            independent y-axis.  Useful when different results have
            incompatible y-scales (e.g. RMSE vs ΔE2000).
    """
    if labels is None:
        labels = [f"Parameter {i + 1}" for i in range(len(results))]

    n_results = len(results)
    colors = qualitative.Set1

    if subplots:
        fig = make_subplots(
            rows=1, cols=n_results,
            shared_yaxes=False,
            subplot_titles=labels,
            horizontal_spacing=0.08,
        )
        ylabels = ylabel if isinstance(ylabel, list) else [ylabel] * n_results
    else:
        fig = go.Figure()

    for idx, (parameter_value, stat_vals) in enumerate(results.items()):
        avg_fitness = np.asarray(stat_vals["median_history"] if statistic == "median" else stat_vals["mean_history"])
        std_fitness = np.asarray(stat_vals["std_history"])
        x_values = _get_plot_x_values(stat_vals, x_axis=x_axis)

        label = labels[idx]
        color = colors[idx % len(colors)]
        row_kwargs = dict(row=1, col=idx + 1) if subplots else {}

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
                ),
                **row_kwargs,
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
                ),
                **row_kwargs,
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
            ),
            **row_kwargs,
        )

    fig.update_layout(
        title=title,
        width=width * (1.2 if subplots else 1),
        height=height,
        template="plotly_white",
        hovermode="x unified",
        legend=dict(
            title="Click to hide/show",
            itemclick="toggle",
            itemdoubleclick="toggleothers",
        ),
        margin=dict(l=70, r=30, t=70, b=60),
    )

    if subplots:
        for idx in range(n_results):
            fig.update_yaxes(title_text=ylabels[idx], row=1, col=idx + 1)
            fig.update_xaxes(title_text=_get_x_axis_label(x_axis), row=1, col=idx + 1)
    else:
        fig.update_layout(
            xaxis_title=_get_x_axis_label(x_axis),
            yaxis_title=ylabel if isinstance(ylabel, str) else ylabel[0],
        )

    fig.update_xaxes(showgrid=True, gridcolor="rgba(0,0,0,0.12)")
    fig.update_yaxes(showgrid=True, gridcolor="rgba(0,0,0,0.12)")

    fig.show()


def plot_diversity_history(diversity_history, figsize=(14, 8)):
    """
    Plot all diversity metrics over generations from a list of report dicts.
    One subplot per metric, so you can visually correlate diversity collapse
    with fitness stagnation.
    """
    metrics = list(diversity_history[0].keys())
    gens = np.arange(len(diversity_history))
    fig, axes = plt.subplots(len(metrics), 1, figsize=figsize, sharex=True)
    
    for ax, metric in zip(axes, metrics):
        values = [d[metric] for d in diversity_history]
        ax.plot(gens, values, linewidth=1.5)
        ax.set_ylabel(metric.replace('_', ' '), fontsize=9)
        ax.grid(True, alpha=0.3)
    
    axes[-1].set_xlabel('Generation')
    axes[0].set_title('Population diversity over generations')
    plt.tight_layout()
    plt.show()


# ── Cross-evaluation and image comparison ─────────────────────────────────────
#
# Generalized versions of helpers originally in perceptual_comparison.py.
# They operate on any run_ga_statistical_comparison results that contain
# 'representations' (produced with return_representations=True).


def cross_evaluate_results(
    results: dict,
    score_functions: dict,
    solution_classes: dict | list | None = None,
):
    """
    Score every stored best-solution representation on multiple metrics.

    Parameters
    ----------
    results : dict
        Output of ``run_ga_statistical_comparison`` (must have been run
        with ``return_representations=True``).
    score_functions : dict[str, callable]
        Mapping of metric name → scoring function.  Each function takes a
        single ``TriangleSolution`` (or subclass) instance and returns a
        float.  Example::

            {"RMSE": rmse_score, "ΔE2000": perceptual_score}

    solution_classes : dict | list | None
        Maps each result key to the solution class used to reconstruct the
        individual from its representation.

        * dict  – ``{result_key: SolutionClass, ...}``
        * list  – positionally matched to ``results.keys()``
        * None  – defaults to ``TriangleSolution`` for every key.

    Returns
    -------
    dict  – ``{result_key: {metric_name: np.ndarray(n_runs,), ...}, ...}``
    """
    keys = list(results.keys())
    if solution_classes is None:
        cls_map = {k: TriangleSolution for k in keys}
    elif isinstance(solution_classes, list):
        cls_map = dict(zip(keys, solution_classes))
    else:
        cls_map = solution_classes

    cross_eval = {}
    for key, data in results.items():
        if "representations" not in data:
            raise ValueError(
                f"Result key {key!r} has no 'representations'. "
                "Re-run with return_representations=True."
            )
        cls = cls_map.get(key, TriangleSolution)
        scores = {metric: [] for metric in score_functions}
        for rep in data["representations"]:
            sol = cls(representation=rep)
            for metric, fn in score_functions.items():
                scores[metric].append(fn(sol))
        cross_eval[key] = {m: np.array(v) for m, v in scores.items()}
    return cross_eval


def cross_eval_table(cross_eval: dict, labels: list | None = None):
    """
    Print a cross-evaluation table (mean ± std) and return the raw dict.

    Parameters
    ----------
    cross_eval : dict
        Output of ``cross_evaluate_results``.
    labels : list[str] | None
        Human-readable row labels (one per result key).  Defaults to the
        string representation of each key.

    Returns
    -------
    dict  – the same ``cross_eval`` dict, unchanged, for chaining.
    """
    keys = list(cross_eval.keys())
    if labels is None:
        labels = [str(k) for k in keys]
    metrics = list(next(iter(cross_eval.values())).keys())

    # Header
    header = f"{'GA (optimised for)':<26}"
    for m in metrics:
        header += f" {m:>22}"
    print(f"\n{header}")
    print("-" * len(header))

    for (key, data), label in zip(cross_eval.items(), labels):
        row = f"{label:<26}"
        for m in metrics:
            vals = data[m]
            row += f" {vals.mean():>9.3f} ± {vals.std():<6.3f}"
        print(row)
    print("-" * len(header))


def plot_image_comparison(
    results: dict,
    solution_classes: dict | list | None = None,
    score_functions: dict | None = None,
    labels: list | None = None,
    figsize: tuple = (13, 6),
):
    """
    Show the target image alongside each GA's representative best solution.

    The representative is the run whose final fitness is closest to the
    median across runs (the same heuristic used by
    ``perceptual_comparison.plot_image_comparison``).

    Parameters
    ----------
    results : dict
        Output of ``run_ga_statistical_comparison`` with representations.
    solution_classes : dict | list | None
        Maps result key → solution class (see ``cross_evaluate_results``).
    score_functions : dict | None
        If given, each panel's subtitle includes scores on all metrics.
    labels : list[str] | None
        Human-readable panel titles (one per result key).
    figsize : tuple
        Matplotlib figure size.
    """
    keys = list(results.keys())
    n = len(keys)

    if solution_classes is None:
        cls_map = {k: TriangleSolution for k in keys}
    elif isinstance(solution_classes, list):
        cls_map = dict(zip(keys, solution_classes))
    else:
        cls_map = solution_classes

    if labels is None:
        labels = [str(k) for k in keys]

    target = np.asarray(TriangleSolution.target_array)

    fig, axes = plt.subplots(1, n + 1, figsize=figsize)

    # Target panel
    target_disp = target[:, :, :3]
    if target_disp.max() > 1:
        target_disp = target_disp / 255.0
    axes[0].imshow(target_disp)
    axes[0].set_title("Target")
    axes[0].axis("off")

    for i, key in enumerate(keys):
        data = results[key]
        if "representations" not in data:
            continue

        # Pick median-run representative
        scores = data["best_per_run"]
        median_idx = int(np.argmin(np.abs(scores - np.median(scores))))

        cls = cls_map.get(key, TriangleSolution)
        sol = cls(representation=data["representations"][median_idx])
        img = np.asarray(sol.get_image())[:, :, :3]

        panel_title = labels[i]
        if score_functions:
            parts = [f"{name}={fn(sol):.2f}" for name, fn in score_functions.items()]
            panel_title += "\n" + "  ".join(parts)

        axes[i + 1].imshow(img)
        axes[i + 1].set_title(panel_title)
        axes[i + 1].axis("off")

    fig.suptitle("Representative reconstruction — median run of each GA")
    plt.tight_layout()
    plt.show()


def plot_diversity_statistical_comparison_interactive(
    results: dict,
    labels: list | None = None,
    title: str = "Statistical Comparison: Diversity Metrics"
):
    """
    Plot interactive subplots for each diversity metric across compared parameters.
    Requires that the results dictionary contains 'mean_diversity_history'.
    """
    keys = list(results.keys())
    if labels is None:
        labels = [str(k) for k in keys]

    # Filter to only results that have diversity tracking
    valid_keys = [k for k in keys if "mean_diversity_history" in results[k] and len(results[k]["mean_diversity_history"]) > 0]
    if not valid_keys:
        print("No diversity logs found in the provided results.")
        return

    # Extract metrics from the first valid result
    first_history = results[valid_keys[0]]["mean_diversity_history"]
    metrics = [k for k in first_history[0].keys() if k != 'gen']
    n_metrics = len(metrics)

    # Calculate grid dimensions (up to 2 columns)
    cols = 2
    rows = (n_metrics + 1) // 2

    fig = make_subplots(
        rows=rows, 
        cols=cols, 
        subplot_titles=[m.replace("_", " ").title() for m in metrics]
    )

    colors = qualitative.Plotly

    for i_metric, metric in enumerate(metrics):
        r = (i_metric // cols) + 1
        c = (i_metric % cols) + 1

        for i_k, key in enumerate(valid_keys):
            data = results[key]["mean_diversity_history"]
            gens = [d["gen"] for d in data]
            vals = [d[metric] for d in data]

            show_legend = (i_metric == 0) # Only show legend once per label
            
            fig.add_trace(
                go.Scatter(
                    x=gens,
                    y=vals,
                    mode="lines",
                    name=labels[keys.index(key)],
                    legendgroup=labels[keys.index(key)],
                    showlegend=show_legend,
                    line=dict(color=colors[i_k % len(colors)], width=2),
                ),
                row=r,
                col=c
            )
            
            fig.update_xaxes(title_text="Generation", row=r, col=c)

    fig.update_layout(
        height=300 * rows,
        title_text=title,
        template="plotly_white",
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1
        )
    )
    
    fig.show()
