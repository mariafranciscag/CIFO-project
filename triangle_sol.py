import random
import numpy as np
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


class TriangleSolution:
    """
    Represents a candidate image composed of n_triangles triangles.
    """
    target_array = None 
    def __init__(self, representation=None, 
                 n_triangles=100,
                 img_width=300,
                 img_height=400,
                 target_array=None):

        self.n_triangles = n_triangles
        self.img_width = img_width
        self.img_height = img_height

        # Set instance target_array if is defined
        if target_array is not None:
            self.target_array = target_array

        if representation is not None:
            self.representation = representation
        else:
            self.representation = self.random_initial_representation()

        self._fitness = None


    # Random initial genotype
    def random_initial_representation(self):
        genes = []
        for _ in range(self.n_triangles):
            vertices = [random.random() for _ in range(6)]
            colour   = [random.randint(0, 255) for _ in range(4)]
            genes.extend(vertices + colour)
        return genes


    # Fitness
    def fitness(self):
        if self._fitness is not None:
            return self._fitness

        rendered     = self._render()
        rendered_arr = np.array(rendered, dtype=np.float32)

        # Fall back to class-level target if instance-level is None
        target = self.target_array if self.target_array is not None else TriangleSolution.target_array

        diff = rendered_arr - target
        self._fitness = float(np.sqrt(np.mean(diff ** 2)))
        return self._fitness


    # Rendering
    def _render(self):
        canvas = Image.new('RGBA', (self.img_width, self.img_height), (0, 0, 0, 255))
        draw   = ImageDraw.Draw(canvas, 'RGBA')

        for i in range(self.n_triangles):
            base = i * 10
            triangle = self.representation[base: base + 10]

            x1 = triangle[0] * self.img_width
            y1 = triangle[1] * self.img_height
            x2 = triangle[2] * self.img_width
            y2 = triangle[3] * self.img_height
            x3 = triangle[4] * self.img_width
            y3 = triangle[5] * self.img_height

            r = int(np.clip(triangle[6], 0, 255))
            g = int(np.clip(triangle[7], 0, 255))
            b = int(np.clip(triangle[8], 0, 255))
            a = int(np.clip(triangle[9], 0, 255))

            draw.polygon([(x1, y1), (x2, y2), (x3, y3)], fill=(r, g, b, a))

        return canvas


    # Utilities
    def __repr__(self):
        return f"TriangleSolution(fitness={self.fitness():.4f}, n_triangles={self.n_triangles})"

    def __lt__(self, other):
        return self.fitness() < other.fitness()

    def __setattr__(self, name, value):
        if name == 'representation':
            super().__setattr__('_fitness', None)
        super().__setattr__(name, value)

    def get_image(self):
        return self._render()

    def show(self):
        fig, axes = plt.subplots(1, 2, figsize=(8, 5))
        axes[0].imshow(self.target_array.astype(np.uint8))
        axes[0].set_title('Target')
        axes[0].axis('off')

        axes[1].imshow(np.array(self._render()))
        axes[1].set_title(f'Candidate (RMSE = {self.fitness():.2f})')
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()


# ---------------------------------------------------------------------------
# Hill Climbing — local search refinement for TriangleSolution
# ---------------------------------------------------------------------------

def hill_climbing(
    solution,
    max_iterations=50,
    delta_vertex=0.02,
    delta_color=5.0,
    min_delta_vertex=0.001,
    min_delta_color=1.0,
    decay_factor=0.5,
    max_evaluations=None,
    verbose=True,
):
    """
    First-improvement hill climbing over individual genes.

    For each pass (iteration), sweep through every triangle's 10 genes.
    For each gene, try +delta and -delta. Accept the first perturbation
    that lowers RMSE and move on to the next gene. After a full pass with
    no improvement, shrink deltas by *decay_factor*. Stop when deltas fall
    below their minimum thresholds, *max_iterations* passes are done, or
    the fitness-evaluation budget *max_evaluations* is exhausted.

    Parameters
    ----------
    solution        : TriangleSolution — modified in-place and returned.
    max_iterations  : int   — max full passes over all 1000 genes.
    delta_vertex    : float — initial perturbation for vertex genes.
    delta_color     : float — initial perturbation for colour genes.
    min_delta_vertex: float — stop threshold for vertex perturbation.
    min_delta_color : float — stop threshold for colour perturbation.
    decay_factor    : float — multiplier applied to deltas after a
                              pass with no improvement.
    max_evaluations : int or None — hard cap on fitness evaluations.
                              None means unlimited.
    verbose         : bool  — print progress after each pass.

    Returns
    -------
    solution   : TriangleSolution — the improved individual.
    rmse_log   : list[float]      — best RMSE recorded after each pass.
    eval_count : int              — total fitness evaluations consumed.
    """
    current_fitness = solution.fitness()
    eval_count = 0
    rmse_log = [current_fitness]
    n_genes = len(solution.representation)  # 1000

    for iteration in range(max_iterations):
        improved_this_pass = False

        for gene_idx in range(n_genes):
            if max_evaluations is not None and eval_count >= max_evaluations:
                if verbose:
                    print(f"  Budget exhausted at {eval_count} evaluations.")
                return solution, rmse_log, eval_count

            # choose perturbation size based on gene type
            is_vertex = (gene_idx % 10) < 6
            delta = delta_vertex if is_vertex else delta_color
            lo, hi = (0.0, 1.0) if is_vertex else (0, 255)

            original_value = solution.representation[gene_idx]

            # try both +delta and -delta
            for sign in (+1, -1):
                candidate_value = original_value + sign * delta
                candidate_value = float(np.clip(candidate_value, lo, hi))

                if candidate_value == original_value:
                    continue  # clipping made it a no-op

                # apply perturbation — triggers fitness cache invalidation
                rep = solution.representation[:]
                rep[gene_idx] = candidate_value
                solution.representation = rep

                new_fitness = solution.fitness()
                eval_count += 1

                if new_fitness < current_fitness:
                    # accept improvement, move to next gene
                    current_fitness = new_fitness
                    improved_this_pass = True
                    break  # stop trying directions for this gene
                else:
                    # revert
                    rep = solution.representation[:]
                    rep[gene_idx] = original_value
                    solution.representation = rep

        rmse_log.append(current_fitness)

        if verbose:
            print(f"  HC pass {iteration + 1:>3d} | RMSE: {current_fitness:.4f} "
                  f"| evals: {eval_count} | delta_v: {delta_vertex:.4f} "
                  f"| delta_c: {delta_color:.2f}")

        if not improved_this_pass:
            # shrink perturbation sizes
            delta_vertex *= decay_factor
            delta_color *= decay_factor

            if delta_vertex < min_delta_vertex and delta_color < min_delta_color:
                if verbose:
                    print("  Deltas below minimum thresholds — stopping.")
                break

    if verbose:
        print(f"  HC finished. Final RMSE: {current_fitness:.4f} | "
              f"Total evals: {eval_count}")

    return solution, rmse_log, eval_count