"""
Perceptual fitness for the triangle-image GA — Challenge 1.

The baseline `TriangleSolution.fitness()` scores a candidate with pixel-wise RGB
RMSE. RMSE measures distance in *device* colour space, where equal numeric
differences do **not** correspond to equal *perceived* differences — the human
eye is unevenly sensitive across the colour gamut.

This module replaces that metric with **CIEDE2000 (ΔE2000)**, the standard
perceptual colour-difference formula:

    1. sRGB pixels are converted to the CIELAB space, which is designed to be
       approximately perceptually uniform (L* = lightness, a* = green↔red,
       b* = blue↔yellow).
    2. CIEDE2000 then measures the difference between two Lab colours, applying
       chroma/hue-dependent weighting and an interaction term that correct
       CIELAB's residual non-uniformities.

Fitness = mean per-pixel ΔE2000 between the rendered image and the target.
Like RMSE this is a *minimisation* objective (lower = more similar), so every
existing GA component (selection, elitism, caching) works unchanged.

Design note — this is implemented as a *subclass* of `TriangleSolution` so it
can be passed straight to `genetic_algorithm(solution_class=...)` without
touching the GA loop. The conversion of the (fixed) target image is cached on
the class so it is computed only once per process.
"""

import numpy as np
from skimage.color import rgb2lab, deltaE_ciede2000

from .triangle_sol import TriangleSolution


# ── Colour-space conversion ───────────────────────────────────────────────────

def rgb_to_lab(rgb_arr):
    """
    Convert an 8-bit RGB image to CIELAB.

    Parameters
    ----------
    rgb_arr : array-like, shape (H, W, 3+) — values in [0, 255]
        Extra channels (e.g. alpha) are dropped; ΔE is a colour-only metric.

    Returns
    -------
    np.ndarray, shape (H, W, 3), dtype float64 — the image in CIELAB space.
    """
    rgb = np.asarray(rgb_arr, dtype=np.float64)[:, :, :3]
    # skimage expects sRGB in [0, 1]; it applies the sRGB gamma internally.
    return rgb2lab(rgb / 255.0)


# ── Target handling ───────────────────────────────────────────────────────────

def set_perceptual_target(target_rgb):
    """
    Register the target image for perceptual fitness.

    Wires the target into `TriangleSolution.target_array` (so plain instances
    still work) and pre-computes its CIELAB form once, caching it on
    `PerceptualTriangleSolution.target_lab`. Call this once, before running a
    GA with `solution_class=PerceptualTriangleSolution`.
    """
    TriangleSolution.target_array = np.asarray(target_rgb)
    PerceptualTriangleSolution.target_lab = rgb_to_lab(target_rgb)


# ── Perceptual solution ───────────────────────────────────────────────────────

class PerceptualTriangleSolution(TriangleSolution):
    """
    A `TriangleSolution` whose fitness is the mean per-pixel CIEDE2000 distance
    to the target image, instead of RGB RMSE.

    Everything else (representation, rendering, mutation, crossover, the
    `_fitness` cache invalidated by `__setattr__`) is inherited unchanged.
    """

    # CIELAB form of the target, shared across instances. Filled lazily on the
    # first fitness() call (or eagerly by set_perceptual_target). Caching here
    # means each worker process converts the target at most once.
    target_lab = None

    def _get_target_lab(self):
        """Return the cached target-in-Lab, computing it on first use.

        Lazy computation is what keeps the joblib/loky workers correct: a
        worker is a fresh process where the class attribute is reset to None,
        but each individual still carries its own `self.target_array`, so the
        Lab target can always be rebuilt from it.
        """
        if PerceptualTriangleSolution.target_lab is None:
            if self.target_array is None:
                raise Exception(
                    "No target set — call set_perceptual_target() first."
                )
            PerceptualTriangleSolution.target_lab = rgb_to_lab(self.target_array)
        return PerceptualTriangleSolution.target_lab

    def fitness(self):
        """Mean per-pixel CIEDE2000 distance to the target (lower is better)."""
        if self._fitness is not None:
            return self._fitness

        rendered_lab = rgb_to_lab(np.asarray(self._render()))
        delta_e = deltaE_ciede2000(self._get_target_lab(), rendered_lab)
        self._fitness = float(np.mean(delta_e))
        return self._fitness

    def __repr__(self):
        return (f"PerceptualTriangleSolution(deltaE2000={self.fitness():.4f}, "
                f"n_triangles={self.n_triangles})")


# ── Cross-evaluation helpers ──────────────────────────────────────────────────
#
# For a *fair* comparison the two GAs must be judged on a common scale: each
# run's best solution is scored on BOTH metrics. These helpers compute a metric
# for any solution directly from its rendered image — they never read the
# `_fitness` cache, which holds whichever metric that solution's class uses.

def rmse_score(solution):
    """Pixel-wise RMSE of `solution` against the target (the baseline metric).

    Matches `TriangleSolution.fitness()`; channels are intersected so it is
    correct whether the target is stored as RGB or RGBA.
    """
    rendered = np.asarray(solution.get_image(), dtype=np.float32)
    target = np.asarray(solution.target_array, dtype=np.float32)
    c = min(rendered.shape[-1], target.shape[-1])
    diff = rendered[:, :, :c] - target[:, :, :c]
    return float(np.sqrt(np.mean(diff ** 2)))


def perceptual_score(solution):
    """Mean CIEDE2000 distance of `solution` to the target (the perceptual metric)."""
    target_lab = (PerceptualTriangleSolution.target_lab
                  if PerceptualTriangleSolution.target_lab is not None
                  else rgb_to_lab(solution.target_array))
    rendered_lab = rgb_to_lab(np.asarray(solution.get_image()))
    return float(np.mean(deltaE_ciede2000(target_lab, rendered_lab)))
