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