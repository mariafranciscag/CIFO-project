import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from tqdm import tqdm

from ga import genetic_algorithm
from ga_utils import get_fitness


# ── Grid search ───────────────────────────────────────────────────────────────

def run_grid_search(grid, gs_generations, gs_repeats, fixed_params, n_jobs_ga=1):
    """
    Exhaustive grid search over GA hyperparameters.

    For each combination of parameters in `grid`, the GA is run
    `gs_repeats` times. The mean and std of the best RMSE across
    repeats are recorded. Results are returned sorted by mean RMSE
    (ascending — lower is better).

    Parameters
    ----------
    grid           : dict[str, list]  — parameter name -> list of values to try
    gs_generations : int              — number of generations per run
    gs_repeats     : int              — independent runs per combination
    fixed_params   : dict             — parameters held constant
    n_jobs_ga      : int              — parallel workers inside each GA run

    Returns
    -------
    results : list[dict] — each entry has 'params', 'mean_rmse', 'std_rmse', 'all_rmse'
    """
    keys   = list(grid.keys())
    values = list(grid.values())
    combos = list(product(*values))

    print(f"Grid search: {len(combos)} combinations x {gs_repeats} repeats "
          f"= {len(combos) * gs_repeats} total runs\n")

    results = []

    for combo in tqdm(combos, desc='Grid search'):
        params = dict(zip(keys, combo))

        run_rmses = []
        for _ in range(gs_repeats):
            best_sol, _, _ = genetic_algorithm(
                n_generations=gs_generations,
                n_jobs=n_jobs_ga,
                **params,
                **fixed_params,
            )
            run_rmses.append(get_fitness(best_sol))

        results.append({
            'params'    : params,
            'mean_rmse' : float(np.mean(run_rmses)),
            'std_rmse'  : float(np.std(run_rmses)),
            'all_rmse'  : run_rmses,
        })

    results.sort(key=lambda x: x['mean_rmse'])
    return results


# ── Display helpers ───────────────────────────────────────────────────────────

def print_grid_search_results(results, top_n=10):
    """Prints a ranked table of the top_n combinations."""
    print(f"{'Rank':<5} {'Mean RMSE':>10} {'Std':>8}   Parameters")
    print("-" * 70)
    for rank, r in enumerate(results[:top_n], 1):
        param_str = ", ".join(f"{k}={v}" for k, v in r['params'].items())
        print(f"{rank:<5} {r['mean_rmse']:>10.4f} {r['std_rmse']:>8.4f}   {param_str}")


def plot_grid_search_results(results, top_n=15):
    """Bar chart of mean RMSE for the top_n combinations."""
    top    = results[:top_n]
    labels = ["\n".join(f"{k}={v}" for k, v in r['params'].items()) for r in top]
    means  = [r['mean_rmse'] for r in top]
    stds   = [r['std_rmse']  for r in top]

    fig, ax = plt.subplots(figsize=(max(10, top_n * 0.9), 5))
    bars = ax.bar(range(top_n), means, yerr=stds, capsize=4,
                  color='#89AC76', edgecolor='black', linewidth=0.6)
    ax.set_xticks(range(top_n))
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylabel('Mean RMSE')
    ax.set_title(f'Grid Search — Top {top_n} Combinations')
    ax.bar_label(bars, fmt='%.3f', fontsize=7, padding=3)
    plt.tight_layout()
    plt.show()


def plot_fitness_history(history, avg_history=None, title='GA — fitness over generations'):
    """Plots best (and optionally average) RMSE per generation."""
    fig, ax = plt.subplots(figsize=(9, 4))
    ax.plot(history, label='Best RMSE', linewidth=1.5)
    if avg_history:
        ax.plot(avg_history, label='Avg RMSE', linewidth=1, linestyle='--', alpha=0.7)
    ax.set_xlabel('Generation')
    ax.set_ylabel('RMSE')
    ax.set_title(title)
    ax.legend()
    plt.tight_layout()
    plt.show()
