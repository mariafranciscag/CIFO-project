"""
FITNESS SHARING 

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
from scripts.ga_utils import get_fitness
from operators.selection import tournament_selection


# Internal helpers 

def representation_vectors(population):
    """
    --> Stack individual representations into a single (N, D) float64 array.
    """
    return np.stack(
        [ind.representation.flatten().astype(np.float64) for ind in population]
    )


# Sharing function 

def sharing_function(d, sigma_share, alpha=1.0):
    """
    --> Triangular sharing function.

        sh(d) = 1 - (d / sigma_share)^alpha  ,  if d < sigma_share
        sh(d) = 0  ,  otherwise
    """
    return np.where(d < sigma_share,
                    1.0 - (d / sigma_share) ** alpha,
                    0.0)


# Niche count computation 

def compute_counts(population, sigma_share, alpha=1.0):
    """
    --> Compute the niche count for every individual (vectorised, O(N²·D)).

    niche_count(i) = sum_j sh( ||rep_i - rep_j||_2 )

    The self-distance sh(0) = 1.0 is included — standard formulation.
    """
    vecs = representation_vectors(population)
    diff = vecs[:, np.newaxis, :] - vecs[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    sh = sharing_function(dists, sigma_share, alpha)
    return sh.sum(axis=1)


# Apply sharing 

def apply_fitness_sharing(population, sigma_share, alpha=1.0):
    """
    --> Compute and cache shared fitness on each individual.

    Sets ind._shared_fitness = raw_rmse * niche_count for every individual.
    Call once per generation, right before the selection step.
    """
    niche_counts = compute_counts(population, sigma_share, alpha)
    for ind, nc in zip(population, niche_counts):
        ind._shared_fitness = get_fitness(ind) * nc


# Selection with shared fitness 

def fitness_sharing_selection(population, method="tournament",
                               sigma_share=50.0, alpha=1.0,
                               tournament_size=3):
    """
    --> Selection operator that ranks individuals by shared fitness.

    Reads 'ind._shared_fitness' (written by apply_fitness_sharing).
    Falls back to raw fitness if sharing has not been applied yet.
    """
    def _sf(ind):
        return getattr(ind, "_shared_fitness", get_fitness(ind))

    n = len(population)

    if method == "tournament":
        contestants = random.sample(population, min(tournament_size, n))
        return min(contestants, key=_sf)

    elif method == "rank":
        sorted_pop = sorted(population, key=_sf, reverse=True)
        ranks = np.arange(1, n + 1, dtype=float)
        probs = ranks / ranks.sum()
        return sorted_pop[np.random.choice(n, p=probs)]

    else:  # roulette — invert because lower = better
        sf = np.array([_sf(ind) for ind in population])
        inv = 1.0 / (sf + 1e-8)
        probs = inv / inv.sum()
        return population[np.random.choice(n, p=probs)]


# Generation counter callback 

class GenerationCounterCallback:
    """
    Pass this as 'generation_callback' to 'genetic_algorithm' so that
    'AutomaticFitnessSharingSelection' knows when a new generation starts.

    How to use:

        selector = AutomaticFitnessSharingSelection(...)
        callback = GenerationCounterCallback(selector)

        best, hist, avg = genetic_algorithm(
            selection_function=selector,
            generation_callback=callback,
            ...
        )
    """
    def __init__(self, *selectors):
        self.selectors = selectors

    def __call__(self, gen, best_fitness, avg_fitness):
        for sel in self.selectors:
            if hasattr(sel, 'next_generation'):
                sel.next_generation()


# AutomaticFitnessSharingSelection

class AutomaticFitnessSharingSelection:
    """
    --> Wrapper that applies fitness sharing automatically before each selection.

    Uses an explicit generation counter incremented via
    'next_generation()'.  It is paired with 'GenerationCounterCallback' so that the
    GA advances the counter automatically.

    How to use:

        selector = AutomaticFitnessSharingSelection(sigma_share=80)
        callback = GenerationCounterCallback(selector)

        best, hist, avg = genetic_algorithm(
            selection_function=selector,
            generation_callback=callback,
            ...
        )
    """

    def __init__(self, sigma_share: float = 50.0, alpha: float = 1.0,
                 selection_function=None):
        self.sigma_share = sigma_share
        self.alpha = alpha
        # default to tournament if nothing passed
        self.selection_function = selection_function or tournament_selection
        self._current_gen = 0         
        self._last_applied_gen = -1  

    def next_generation(self):
        """
        --> Advance the internal generation counter (call once per generation).~
        """
        self._current_gen += 1

    def _ensure_sharing_applied(self, population):
        """
        --> Recompute sharing if we have not done so for this generation.
        """
        if self._last_applied_gen != self._current_gen:
            apply_fitness_sharing(population,
                                  sigma_share=self.sigma_share,
                                  alpha=self.alpha)
            self._last_applied_gen = self._current_gen

    def __call__(self, population):
        self._ensure_sharing_applied(population)
        return self.selection_function(population)


# AdvancedFitnessSharingSelection 

class AdvancedFitnessSharingSelection:
    """
    --> Fitness sharing with normalised gene distances.

    Without normalisation, vertex genes (range ≈ 0–400) completely dominate
    the distance calculation over colour genes (0–255), so individuals that
    differ only in colour are treated as identical — negating the diversity
    benefit for colour exploration.

    This variant divides each gene by its maximum possible value before
    computing L2 distances, so vertices and colours contribute proportionally.


    PARAMETERS:
        sigma_share : float
            Niche radius expressed as a **fraction of the theoretical maximum
            normalised distance** (sqrt(D) where D = n_genes).
            Typical starting value: 0.15 (15 %).
    """

    def __init__(self, sigma_share: float = 0.15, alpha: float = 1.0,
                 tournament_size: int = 5):
        self.sigma_share = sigma_share
        self.alpha = alpha
        self.tournament_size = tournament_size
        self._current_gen = 0
        self._last_applied_gen = -1

    def next_generation(self):
        """ 
        --> Advance the internal generation counter.
        """
        self._current_gen += 1

    def _apply_normalized_sharing(self, population):
        """ 
        --> Compute sharing after normalising gene ranges to [0, 1].
        """
        vecs = np.stack(
            [ind.representation.flatten().astype(np.float64) for ind in population]
        )
        n_genes = vecs.shape[1]
        # Build the normalisation mask for the 10-gene-per-triangle layout
        norm_mask = np.tile(
            [300.0, 400.0, 300.0, 400.0, 300.0, 400.0, 255.0, 255.0, 255.0, 255.0],
            n_genes // 10,
        )
        # Handle adaptive solutions (11 genes: append mutation rate range [0,1])
        if n_genes % 10 != 0:
            extra = n_genes - len(norm_mask)
            norm_mask = np.concatenate([norm_mask, np.ones(extra)])

        vecs_norm = vecs / norm_mask

        diff = vecs_norm[:, np.newaxis, :] - vecs_norm[np.newaxis, :, :]
        dists = np.sqrt((diff ** 2).sum(axis=2))

        max_dist = np.sqrt(n_genes)          # theoretical max in normalised space
        eff_sigma = self.sigma_share * max_dist

        sh = np.where(dists < eff_sigma,
                          1.0 - (dists / eff_sigma) ** self.alpha,
                          0.0)
        counts = sh.sum(axis=1)

        for ind, nc in zip(population, counts):
            ind._shared_fitness = get_fitness(ind) * nc

    def _ensure_sharing_applied(self, population):
        if self._last_applied_gen != self._current_gen:
            self._apply_normalized_sharing(population)
            self._last_applied_gen = self._current_gen

    def __call__(self, population):
        self._ensure_sharing_applied(population)
        return fitness_sharing_selection(
            population,
            method="tournament",
            tournament_size=self.tournament_size,
        )


# Diversity monitoring system 

def measure_diversity(population, sample_n=20):
    """
    --> Estimate average pairwise L2 distance in the population (diagnostic only).
    """
    n = len(population)
    sample_n = min(n, sample_n)
    idx = np.random.choice(n, sample_n, replace=False)
    vecs = np.stack([
        population[i].representation.flatten().astype(np.float64)
        for i in idx
    ])
    diff = vecs[:, np.newaxis, :] - vecs[np.newaxis, :, :]
    dists = np.sqrt((diff ** 2).sum(axis=2))
    mask = np.triu(np.ones((sample_n, sample_n), dtype=bool), k=1)
    return float(dists[mask].mean()) if mask.sum() > 0 else 0.0


def genotypic_variance(population):
    """
    Mean per-gene variance across the population.

    A scalar summary of how spread out the raw genomes are.
    Collapses toward 0 when all individuals converge to the same representation.
    """
    vecs = representation_vectors(population)   # (N, D)
    return float(vecs.var(axis=0).mean())


def genotypic_entropy(population, n_bins=20):
    """
    Mean per-gene Shannon entropy (bits), estimated via histogram binning.

    Unlike variance, this captures bimodal or heavy-tailed gene distributions
    that arise when the population splits into sub-niches.

    Parameters
    ----------
    n_bins : int
        Histogram resolution. 20 bins is reasonable for genes in [0, 255];
        increase if you want finer resolution on vertex coordinates.

    Returns
    -------
    float — mean entropy in bits across all genes.
    """
    vecs = representation_vectors(population)   # (N, D)
    entropies = []
    for gene_col in vecs.T:                     # iterate over D genes
        counts, _ = np.histogram(gene_col, bins=n_bins)
        probs = counts / counts.sum()
        # Shannon entropy; ignore zero-probability bins to avoid log(0)
        probs = probs[probs > 0]
        entropies.append(-float(np.sum(probs * np.log2(probs))))
    return float(np.mean(entropies))


def fitness_variance(population):
    """
    Variance of raw fitness values across the population.

    Cheap phenotypic diversity proxy — no rendering required.
    Collapsing fitness variance is often the earliest warning of premature
    convergence, even before genotypic measures show it.
    """
    fitnesses = np.array([get_fitness(ind) for ind in population])
    return float(fitnesses.var())


def fitness_entropy(population, n_bins=20):
    """
    Shannon entropy (bits) of the fitness distribution.

    Bins the population's fitness values and computes entropy over that
    histogram. Low entropy = most individuals clustered at one fitness level.
    """
    fitnesses = np.array([get_fitness(ind) for ind in population])
    counts, _ = np.histogram(fitnesses, bins=n_bins)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    return float(-np.sum(probs * np.log2(probs)))


def population_diversity_report(population, sample_n=20, n_bins=20):
    """
    Compute all diversity metrics in one call and return a summary dict.

    Combines the cheap metrics (computed on the full population) with the
    expensive pairwise L2 (sampled). Designed to be called once per
    generation and appended to a history list for later plotting.

    Returns
    -------
    dict with keys:
        'genotypic_variance'  : mean per-gene variance
        'genotypic_entropy'   : mean per-gene Shannon entropy (bits)
        'mean_pairwise_l2'    : sampled mean pairwise L2 distance (existing metric)
        'fitness_variance'    : variance of raw fitness values
        'fitness_entropy'     : Shannon entropy of fitness distribution
    """
    return {
        'genotypic_variance': genotypic_variance(population),
        'genotypic_entropy':  genotypic_entropy(population, n_bins=n_bins),
        'mean_pairwise_l2':   measure_diversity(population, sample_n=sample_n),
        'fitness_variance':   fitness_variance(population),
        'fitness_entropy':    fitness_entropy(population, n_bins=n_bins),
    }
