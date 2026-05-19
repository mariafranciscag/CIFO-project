from joblib import Parallel, delayed
from .triangle_sol import TriangleSolution


# ── Fitness cache ─────────────────────────────────────────────────────────────

def get_fitness(ind):
    """Return cached fitness, computing it only on the first call."""
    if not hasattr(ind, '_cached_fitness') or ind._cached_fitness is None:
        ind._cached_fitness = ind.fitness()
    return ind._cached_fitness


def invalidate_cache(ind):
    """Call after any mutation/crossover that changes the representation."""
    ind._cached_fitness = None


# ── Parallel evaluation ───────────────────────────────────────────────────────

def _eval_one(ind: TriangleSolution) -> TriangleSolution:
    """Top-level function required for joblib pickling."""
    ind._cached_fitness = ind.fitness()
    return ind


def evaluate_population(population: list[TriangleSolution], n_jobs: int = -1) -> list[TriangleSolution]:
    """Compute (or refresh) fitness for every individual in parallel."""
    return Parallel(n_jobs=n_jobs, backend='loky')(
        delayed(_eval_one)(ind) for ind in population
    )
