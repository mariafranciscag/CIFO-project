"""Perceptual fitness for Challenge 1.

Swaps the GA's RGB-RMSE for mean per-pixel CIEDE2000 distance in CIELAB.
Same minimisation objective, so nothing else in the GA needs to change.
"""

import numpy as np
from skimage.color import rgb2lab, deltaE_ciede2000

from .triangle_sol import TriangleSolution


def rgb_to_lab(rgb_arr):
    """Convert an 8-bit RGB(A) image to CIELAB. Alpha is dropped."""
    rgb = np.asarray(rgb_arr, dtype=np.float32)[:, :, :3]
    return rgb2lab(rgb / 255.0).astype(np.float32, copy=False)


def set_perceptual_target(target_rgb):
    """Set the target on both the base class and the perceptual subclass."""
    TriangleSolution.target_array = np.asarray(target_rgb)
    PerceptualTriangleSolution.target_lab = rgb_to_lab(target_rgb)


class PerceptualTriangleSolution(TriangleSolution):
    """TriangleSolution scored by mean CIEDE2000 instead of RGB RMSE."""

    # Cached CIELAB target. Filled lazily so loky workers (fresh process,
    # class attrs gone) can rebuild it from self.target_array.
    target_lab = None

    def _get_target_lab(self):
        if PerceptualTriangleSolution.target_lab is None:
            if self.target_array is None:
                raise Exception(
                    "No target set. Call set_perceptual_target() first."
                )
            PerceptualTriangleSolution.target_lab = rgb_to_lab(self.target_array)
        return PerceptualTriangleSolution.target_lab

    def fitness(self):
        if self._fitness is not None:
            return self._fitness
        rendered_rgb = np.asarray(self._render(), dtype=np.float32)[:, :, :3]
        rendered_lab = rgb_to_lab(rendered_rgb)
        delta_e = deltaE_ciede2000(self._get_target_lab(), rendered_lab)
        self._fitness = float(np.mean(delta_e))
        return self._fitness

    def __repr__(self):
        return (f"PerceptualTriangleSolution(deltaE2000={self.fitness():.4f}, "
                f"n_triangles={self.n_triangles})")


def rmse_score(solution):
    """RMSE of the rendered solution vs the target."""
    rendered = np.asarray(solution.get_image(), dtype=np.float32)
    target = np.asarray(solution.target_array, dtype=np.float32)
    c = min(rendered.shape[-1], target.shape[-1])
    diff = rendered[:, :, :c] - target[:, :, :c]
    return float(np.sqrt(np.mean(diff ** 2)))


def perceptual_score(solution):
    """Mean CIEDE2000 of the rendered solution vs the target."""
    target_lab = (PerceptualTriangleSolution.target_lab
                  if PerceptualTriangleSolution.target_lab is not None
                  else rgb_to_lab(solution.target_array))
    rendered_rgb = np.asarray(solution.get_image(), dtype=np.float32)[:, :, :3]
    rendered_lab = rgb_to_lab(rendered_rgb)
    return float(np.mean(deltaE_ciede2000(target_lab, rendered_lab)))
