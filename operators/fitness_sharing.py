"""
Premature convergence happens when the population clusters around a local
optimum. Fitness sharing penalises individuals with many close neighbours,
forcing selection pressure to spread across multiple niches.


The shared fitness replaces raw RMSE during selection:

    shared_fitness(i) = raw_rmse(i) * niche_count(i)

    niche_count(i) = sum_j  sh( dist(i, j) )

    sh(d) = 1 - (d / sigma_share)^alpha  ,  if d < sigma_share
    sh(d) = 0  ,  otherwise

    
Because raw_fitness is RMSE (lower = better), multiplying by niche_count
(>= 1.0) makes crowded individuals appear worse to selection, driving
exploration into less-visited regions of the search space.

"""

import random
import numpy as np
from scripts.ga_utils import get_fitness          # project-standard fitness accessor


# Internal helpers

def representation_vectors(population):
    """
    Stack individual representations into a single (N, D) float64 array.

    TriangleSolution stores its genome as a flat 1-D numpy array of length
    n_triangles * 10  (6 vertex coords + 4 RGBA channels per triangle,
    all already in a single row).  We simply cast to float64 and stack.

    Note: ``individual.representation`` is a 1-D array in this project
    (not 2-D), so no ``.flatten()`` call is required — but it is safe to
    include one for robustness.
    """
    return np.stack(
        [ind.representation.flatten().astype(np.float64) for ind in population]
    )  # (N, D)


# Sharing function

def sharing_function(d, sigma_share, alpha=1.0):
    """
    Triangular sharing function.

        sh(d) = 1 - (d / sigma_share)^alpha    if d < sigma_share
        sh(d) = 0                               otherwise

    Parameters
    ----------
    d           : float or np.ndarray
    sigma_share : float
        Niche radius.  The genome is a vector of ~1 000 genes; most genes
        live in [0, 1] (vertices) or [0, 255] (RGBA).  A reasonable
        starting value is sigma_share=50, which is roughly 1.5 % of the
        maximum possible L2 distance.  Tune up if the population stays
        too spread out, down if it still converges prematurely.
    alpha       : float
        Shape parameter.  1.0 (linear, default) is standard.  Values > 1
        make the function concave — only very close individuals penalise
        each other.

    Returns
    -------
    float or np.ndarray in [0, 1].
    """
    return np.where(d < sigma_share,
                    1.0 - (d / sigma_share) ** alpha,
                    0.0)


# Computation of counts

def compute_counts(population, sigma_share, alpha=1.0):
    """
    Compute the niche count for every individual (vectorised, O(N²·D)).

    niche_count(i) = sum_j sh( ||rep_i - rep_j||_2 )

    The self-distance sh(0) = 1.0 is included — standard formulation.

    Parameters
    ----------
    population  : list[TriangleSolution]
    sigma_share : float
    alpha       : float

    Returns
    -------
    np.ndarray of shape (N,), dtype float64.
    """
    vecs  = representation_vectors(population)                       # (N, D)
    diff  = vecs[:, np.newaxis, :] - vecs[np.newaxis, :, :]          # (N, N, D)
    dists = np.sqrt((diff ** 2).sum(axis=2))                          # (N, N)
    sh    = sharing_function(dists, sigma_share, alpha)               # (N, N)
    return sh.sum(axis=1)                                             # (N,)


# Apply sharing — call once per generation before selection

def apply_fitness_sharing(population, sigma_share, alpha=1.0):
    """
    Compute and cache shared fitness on each individual.

    Uses ``get_fitness(ind)`` (from ga_utils) to retrieve raw RMSE —
    consistent with the rest of the project (caches are respected).

    Sets ``ind._shared_fitness = raw_rmse * niche_count`` for every
    individual.  Call this **once per generation**, right before the
    selection step.

    Parameters
    ----------
    population  : list[TriangleSolution]
        Must already have valid (possibly cached) fitness values.
    sigma_share : float
    alpha       : float
    """
    niche_counts = compute_counts(population, sigma_share, alpha)
    for ind, nc in zip(population, niche_counts):
        ind._shared_fitness = get_fitness(ind) * nc   # ← uses ga_utils, not calculate_fitness()


# Selection wrapper

def fitness_sharing_selection(population, method="tournament",
                               sigma_share=50.0, alpha=1.0,
                               tournament_size=3):
    """
    Selection operator that ranks individuals by shared fitness.

    Reads ``ind._shared_fitness`` (written by apply_fitness_sharing).
    Falls back to ``get_fitness(ind)`` — via ga_utils — if sharing has
    not been applied yet, so the function is safe to call at any time.

    The interface mirrors the existing selection functions in selection.py:
    it accepts a population list and returns a single individual (no copy).

    Parameters
    ----------
    population     : list[TriangleSolution]
    method         : 'tournament' | 'rank' | 'roulette'
    sigma_share    : float — passed to apply_fitness_sharing when needed.
    alpha          : float — sharing function shape.
    tournament_size: int   — used only when method='tournament'.

    Returns
    -------
    TriangleSolution — one selected individual (not a copy).
    """
    def _sf(ind):
        # Prefer cached shared fitness; fall back to raw RMSE via ga_utils
        return getattr(ind, "_shared_fitness", get_fitness(ind))

    n = len(population)

    if method == "tournament":
        # sample without replacement — same as tournament_selection in selection.py
        contestants = random.sample(population, min(tournament_size, n))
        return min(contestants, key=_sf)   # lower shared_fitness = better (RMSE)

    elif method == "rank":
        # worst-to-best so rank index 0 = worst, index N-1 = best
        sorted_pop = sorted(population, key=_sf, reverse=True)
        ranks  = np.arange(1, n + 1, dtype=float)
        probs  = ranks / ranks.sum()
        return sorted_pop[np.random.choice(n, p=probs)]

    else:  # roulette — invert because lower = better
        sf    = np.array([_sf(ind) for ind in population])
        inv   = 1.0 / (sf + 1e-8)
        probs = inv / inv.sum()
        return population[np.random.choice(n, p=probs)]


# Diversity metric (diagnostic / logging only)

def measure_diversity(population, sample_n=20):
    """
    Estimate average pairwise L2 distance in the population.

    Computing all N² pairs is expensive for large populations; we sample
    up to ``sample_n`` individuals and average over the upper triangle of
    the distance matrix (excluding self-distances).

    Intended for logging / plotting alongside ``avg_history`` in the GA —
    not used internally by the selection operators.

    Parameters
    ----------
    population : list[TriangleSolution]
    sample_n   : int — max individuals to sample (default 20).

    Returns
    -------
    float — estimated average pairwise L2 distance.
    """
    n        = len(population)
    sample_n = min(n, sample_n)
    idx      = np.random.choice(n, sample_n, replace=False)
    vecs     = np.stack([
        population[i].representation.flatten().astype(np.float64)
        for i in idx
    ])                                                         # (sample_n, D)
    diff     = vecs[:, np.newaxis, :] - vecs[np.newaxis, :, :]
    dists    = np.sqrt((diff ** 2).sum(axis=2))
    mask     = np.triu(np.ones((sample_n, sample_n), dtype=bool), k=1)
    return float(dists[mask].mean()) if mask.sum() > 0 else 0.0
