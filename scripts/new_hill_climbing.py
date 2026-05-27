"""
HILL CLIMBING LOCAL SEARCH FOR THE TRIANGLE IMAGE GA (post-processing step)

    After the GA converges, this module polishes the best solution by iterating
    small perturbations and keeping any that improve fitness. The result is a
    tighter approximation without touching the GA loop at all.

Design:
    - Stochastic hill climbing - each iteration randomly perturbs one
    randomly chosen triangle (vertices and/or colours) and accepts the move
    only if RMSE strictly decreases.
    - Adaptive step sizes - 'delta_vertex' and 'elta_colour' shrink by
    'decay_factor' every time a full pass over all triangles produces no
    improvement, mimicking a simple line-search cool-down.
    . Side-effect free - the input solution is deep-copied; the caller's
    object is never modified.

API:
    hill_climbing(solution, ...)  →  (best_solution, log, n_evaluations)
"""

import numpy as np
from copy import deepcopy
from typing import Optional

from scripts.triangle_sol import TriangleSolution
from scripts.ga_utils import get_fitness


# Core perturbation helpers 

def _perturb_triangle(rep: np.ndarray, triangle_idx: int, triangle_len: int,
                       delta_vertex: float, delta_colour: float,
                       img_width: int, img_height: int) -> np.ndarray:
    """
    Return a *copy* of `rep` with one triangle perturbed in-place.

    Vertices are shifted by U(-delta, +delta); colour channels by the same
    window.  Values are clipped to their valid domains.
    """
    rep = rep.copy()
    base = triangle_idx * triangle_len

    # Vertex x-coords (genes 0, 2, 4)
    for i in [0, 2, 4]:
        rep[base + i] = np.clip(
            rep[base + i] + np.random.uniform(-delta_vertex, delta_vertex),
            0, img_width,
        )
    # Vertex y-coords (genes 1, 3, 5)
    for i in [1, 3, 5]:
        rep[base + i] = np.clip(
            rep[base + i] + np.random.uniform(-delta_vertex, delta_vertex),
            0, img_height,
        )
    # RGBA colour genes (6–9)
    for i in range(6, 10):
        rep[base + i] = np.clip(
            rep[base + i] + np.random.uniform(-delta_colour, delta_colour),
            0, 255,
        )
    return rep


# Hill climbing driver 

def hill_climbing(
    solution: TriangleSolution,
    max_iterations: int = 30,
    delta_vertex: float = 12.0,
    delta_colour: float = 10.0,
    decay_factor: float = 0.6,
    min_delta: float = 0.1,
    trials_per_iteration: int = 1,
    verbose: bool = True,
) -> tuple:
    """
    Stochastic hill climbing applied to a single GA solution.

    Each iteration randomly picks one triangle and perturbs its vertices and
    colour genes.  The move is accepted if the rendered RMSE decreases.

    One *iteration* = one full pass over all triangles (each tried
    'trials_per_iteration' times).  If the pass produces zero improvements,
    the step sizes are multiplied by 'decay_factor'.  This continues until
    'max_iterations' passes have been completed or both deltas fall below
    'min_delta'.

    PARAMETERS:
    solution : TriangleSolution
        Starting point.  A deep copy is made, the original is not mutated.
    max_iterations : int
        Maximum number of full passes over all triangles.
    delta_vertex : float
        Initial maximum vertex perturbation in pixels.
    delta_colour : float
        Initial maximum colour perturbation (0–255 units).
    decay_factor : float
        Multiplicative shrinkage applied to both deltas when a pass fails.
        Typical values: 0.5–0.8.
    min_delta : float
        Stop early once *both* deltas fall below this threshold.
    trials_per_iteration : int
        Number of random perturbations tried per triangle per pass.
    verbose : bool
        Print progress every iteration.

    RETURNS:
    best_solution : TriangleSolution
        Polished solution (deep copy of input, then improved).
    log : list[dict]
        Per-iteration records: {'iteration', 'rmse', 'improvements',
        'delta_vertex', 'delta_colour'}.
    n_evaluations : int
        Total number of fitness evaluations performed.
    """
    # Work on a copy so the caller's solution is never touched
    best = deepcopy(solution)
    best_fitness = get_fitness(best)

    n_triangles = best.n_triangles
    triangle_len = best.triangle_len
    img_width = best.img_width
    img_height = best.img_height

    log = []
    n_evaluations = 0

    if verbose:
        print(f"Hill Climbing start — RMSE: {best_fitness:.4f} | "
              f"delta_v={delta_vertex:.2f}, delta_c={delta_colour:.2f}")

    for iteration in range(max_iterations):
        # Early stop: steps are too small to matter
        if delta_vertex < min_delta and delta_colour < min_delta:
            if verbose:
                print(f"  Early stop: deltas below min_delta={min_delta}")
            break

        improvements = 0
        # Randomise the order we visit triangles each pass
        triangle_order = np.random.permutation(n_triangles)

        for tri_idx in triangle_order:
            for _ in range(trials_per_iteration):
                # Build a candidate neighbour
                new_rep = _perturb_triangle(
                    best.representation, tri_idx, triangle_len,
                    delta_vertex, delta_colour, img_width, img_height,
                )
                # Evaluate — create a throw-away solution sharing the same class
                candidate = type(best)(representation=new_rep)
                n_evaluations += 1
                candidate_fitness = get_fitness(candidate)

                if candidate_fitness < best_fitness:
                    best = candidate
                    best_fitness = candidate_fitness
                    improvements += 1
                    # Short-circuit: already found a better rep, move on
                    break

        log.append({
            'iteration': iteration,
            'rmse': best_fitness,
            'improvements': improvements,
            'delta_vertex': delta_vertex,
            'delta_colour': delta_colour,
        })

        if verbose:
            print(f"  Iter {iteration + 1:>3d} | RMSE: {best_fitness:.4f} | "
                  f"improvements: {improvements:>4d} | "
                  f"delta_v={delta_vertex:.3f}, delta_c={delta_colour:.3f}")

        # If no improvement this pass, shrink the step sizes
        if improvements == 0:
            delta_vertex *= decay_factor
            delta_colour *= decay_factor

    if verbose:
        print(f"\nHill Climbing done — final RMSE: {best_fitness:.4f} | "
              f"evaluations: {n_evaluations:,}")

    return best, log, n_evaluations


# Plotting helper 

def plot_hill_climbing_log(log: list, ga_final_rmse: Optional[float] = None,
                            figsize: tuple = (10, 4)):
    """
    Plot RMSE vs iteration from the HC log.

    PARAMETERS:
    log : list[dict] - returned by hill_climbing()
    ga_final_rmse : float, optional
        If provided, draws a horizontal dashed line at the GA's pre-HC RMSE
        so the improvement is immediately visible.
    """
    import matplotlib.pyplot as plt

    iterations = [r['iteration'] + 1 for r in log]
    rmse_values = [r['rmse'] for r in log]
    improvements = [r['improvements'] for r in log]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

    # RMSE convergence
    ax1.plot(iterations, rmse_values, color='#4C72B0', linewidth=2, marker='o',
             markersize=4, label='HC RMSE')
    if ga_final_rmse is not None:
        ax1.axhline(ga_final_rmse, color='#C44E52', linestyle='--', linewidth=1.5,
                    label=f'GA final ({ga_final_rmse:.4f})')
    ax1.set_xlabel('Hill Climbing Iteration')
    ax1.set_ylabel('RMSE (lower is better)')
    ax1.set_title('Hill Climbing — RMSE convergence')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Improvements per iteration (bar chart)
    ax2.bar(iterations, improvements, color='#55A868', alpha=0.8)
    ax2.set_xlabel('Hill Climbing Iteration')
    ax2.set_ylabel('# Accepted moves')
    ax2.set_title('Hill Climbing — accepted moves per iteration')
    ax2.grid(True, alpha=0.3, axis='y')

    plt.tight_layout()
    plt.show()
