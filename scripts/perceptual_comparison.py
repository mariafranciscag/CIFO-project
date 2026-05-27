"""RGB-RMSE vs CIEDE2000 paired comparison (Challenge 1).

Both GAs run with the same config, only the fitness function differs.
Each pair shares a seed so the initial population matches, then every best
solution is scored on both metrics for a fair cross-comparison.
"""

import numpy as np
from joblib import Parallel, delayed
from scipy.stats import wilcoxon
import matplotlib.pyplot as plt

from .triangle_sol import TriangleSolution
from .ga import genetic_algorithm
from .perceptual_fitness import (
    PerceptualTriangleSolution, rmse_score, perceptual_score,
)
from operators.selection import tournament_selection
from operators.crossover import one_point_crossover
from operators.mutation import triangle_gaussian_mutation


def _run_one(solution_class, seed, ga_kwargs, target_array):
    # Top-level so joblib/loky can pickle it. target_array is re-bound
    # because each worker is a fresh process.
    TriangleSolution.target_array = target_array

    best, history, _ = genetic_algorithm(
        solution_class=solution_class,
        seed=seed,
        n_jobs=1,
        verbose=False,
        **ga_kwargs,
    )
    return {
        'representation': np.asarray(best.representation),
        'history': list(history),
        'rmse': rmse_score(best),
        'deltaE': perceptual_score(best),
    }


def run_paired_comparison(
    n_runs: int = 30,
    base_seed: int = 1000,
    population_size: int = 50,
    n_generations: int = 200,
    selection_function=tournament_selection,
    crossover_function=one_point_crossover,
    mutation_function=triangle_gaussian_mutation,
    crossover_rate: float = 0.9,
    mutation_rate: float = 0.02,
    mutation_rate_decay=None,
    elitism_n: int = 2,
    n_jobs: int = -1,
):
    """Run n_runs paired GA runs for each fitness function in parallel."""
    ga_kwargs = dict(
        population_size=population_size,
        n_generations=n_generations,
        selection_function=selection_function,
        crossover_function=crossover_function,
        mutation_function=mutation_function,
        crossover_rate=crossover_rate,
        mutation_rate=mutation_rate,
        mutation_rate_decay=mutation_rate_decay,
        elitism_n=elitism_n,
    )
    seeds = [base_seed + i for i in range(n_runs)]
    target_array = TriangleSolution.target_array
    if target_array is None:
        raise Exception("No target set. Call set_perceptual_target() first.")

    labelled_tasks = (
        [('rgb', TriangleSolution, s) for s in seeds]
        + [('perceptual', PerceptualTriangleSolution, s) for s in seeds]
    )
    print(f"Paired comparison: {len(labelled_tasks)} GA runs "
          f"({n_runs} per fitness function), {n_generations} generations each.")

    outcomes = Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_run_one)(cls, s, ga_kwargs, target_array)
        for _, cls, s in labelled_tasks
    )

    results = {'seeds': seeds, 'meta': ga_kwargs}
    for group in ('rgb', 'perceptual'):
        group_outcomes = [o for (g, _, _), o in zip(labelled_tasks, outcomes)
                          if g == group]
        results[group] = {
            'rmse':            np.array([o['rmse'] for o in group_outcomes]),
            'deltaE':          np.array([o['deltaE'] for o in group_outcomes]),
            'history':         np.array([o['history'] for o in group_outcomes]),
            'representations': [o['representation'] for o in group_outcomes],
        }
    return results


def cross_eval_table(results: dict):
    """Print the 2x2 cross-evaluation table (mean +/- std). Returns a dict."""
    rows = {}
    print(f"\n{'GA (optimised for)':<26} {'RMSE (RGB)':>18} {'ΔE2000 (perceptual)':>22}")
    print("-" * 68)
    for group, label in (('rgb', 'RGB RMSE'), ('perceptual', 'CIEDE2000')):
        rmse = results[group]['rmse']
        de = results[group]['deltaE']
        rows[group] = {
            'rmse_mean': float(rmse.mean()), 'rmse_std': float(rmse.std()),
            'deltaE_mean': float(de.mean()), 'deltaE_std': float(de.std()),
        }
        print(f"{label:<26} "
              f"{rmse.mean():>9.3f} +/- {rmse.std():<6.3f} "
              f"{de.mean():>13.3f} +/- {de.std():<6.3f}")
    print("-" * 68)
    return rows


def wilcoxon_tests(results: dict, alpha: float = 0.05):
    """Paired Wilcoxon signed-rank tests on RMSE and ΔE2000."""
    print("\nWilcoxon signed-rank tests (paired by seed):")
    out = {}
    for metric in ('rmse', 'deltaE'):
        rgb = results['rgb'][metric]
        perc = results['perceptual'][metric]
        stat, p = wilcoxon(rgb, perc)
        better = 'RGB-GA' if rgb.mean() < perc.mean() else 'perceptual-GA'
        verdict = ('significant' if p < alpha else 'not significant')
        out[metric] = {'statistic': float(stat), 'p_value': float(p),
                       'better': better, 'significant': p < alpha}
        name = 'RMSE' if metric == 'rmse' else 'ΔE2000'
        print(f"  {name:<8} W={stat:>10.2f}  p={p:.2e}  "
              f"-> {better} lower ({verdict} at alpha={alpha})")
    return out


def plot_convergence(results: dict, figsize=(13, 5)):
    """Each GA's convergence on its own metric, mean +/- std band."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    panels = [
        ('rgb', 'RGB-RMSE GA', 'Best RMSE', '#C44E52'),
        ('perceptual', 'Perceptual CIEDE2000 GA', 'Best ΔE2000', '#4C72B0'),
    ]
    for ax, (group, title, ylabel, colour) in zip(axes, panels):
        hist = results[group]['history']
        mean, std = hist.mean(axis=0), hist.std(axis=0)
        gens = np.arange(hist.shape[1])
        ax.plot(gens, mean, color=colour, linewidth=2, label='mean')
        ax.fill_between(gens, mean - std, mean + std, color=colour, alpha=0.2,
                        label='+/- 1 std')
        ax.set_title(title)
        ax.set_xlabel('Generation')
        ax.set_ylabel(ylabel)
        ax.grid(True, alpha=0.3)
        ax.legend()
    fig.suptitle(f"Convergence over {len(results['seeds'])} paired runs")
    plt.tight_layout()
    plt.show()


def plot_metric_distributions(results: dict, figsize=(11, 4.5)):
    """Boxplots of final scores per metric, both GAs side by side."""
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    for ax, metric, name in ((axes[0], 'rmse', 'Final RMSE (RGB)'),
                             (axes[1], 'deltaE', 'Final ΔE2000 (perceptual)')):
        ax.boxplot([results['rgb'][metric], results['perceptual'][metric]],
                   tick_labels=['RGB-RMSE GA', 'Perceptual GA'])
        ax.set_ylabel(name)
        ax.set_title(name)
        ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


def _representative_index(results: dict, group: str, metric: str):
    scores = results[group][metric]
    return int(np.argmin(np.abs(scores - np.median(scores))))


def plot_image_comparison(results: dict, figsize=(13, 6)):
    """Target vs each GA's median-fitness reconstruction."""
    target = np.asarray(TriangleSolution.target_array)

    rgb_idx = _representative_index(results, 'rgb', 'rmse')
    perc_idx = _representative_index(results, 'perceptual', 'deltaE')

    rgb_best = TriangleSolution(
        representation=results['rgb']['representations'][rgb_idx])
    perc_best = PerceptualTriangleSolution(
        representation=results['perceptual']['representations'][perc_idx])

    panels = [
        (target[:, :, :3] / 255.0, 'Target', None),
        (np.asarray(rgb_best.get_image())[:, :, :3],
         f"RGB-RMSE GA\nRMSE={rmse_score(rgb_best):.2f}  "
         f"ΔE2000={perceptual_score(rgb_best):.2f}", None),
        (np.asarray(perc_best.get_image())[:, :, :3],
         f"Perceptual GA\nRMSE={rmse_score(perc_best):.2f}  "
         f"ΔE2000={perceptual_score(perc_best):.2f}", None),
    ]
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    for ax, (img, title, _) in zip(axes, panels):
        ax.imshow(img)
        ax.set_title(title)
        ax.axis('off')
    fig.suptitle('Representative reconstruction (median run of each GA)')
    plt.tight_layout()
    plt.show()
