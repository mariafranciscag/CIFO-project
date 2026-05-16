import random
import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt


class TriangleSolution:
    """
    Represents a candidate image composed of n_triangles triangles.
    """
    target_array = None
    def __init__(
        self,
        representation: NDArray[np.int16] | None = None,
        n_triangles: int = 100,
        img_width: int = 300,
        img_height: int = 400,
        target_array: NDArray[np.int16] | None = None,
    ):

        self.n_triangles = n_triangles
        self.img_width = img_width
        self.img_height = img_height

        # Set instance target_array if is defined
        if target_array is not None:
            self.target_array = target_array
        else:
            self.target_array = TriangleSolution.target_array

        if representation is not None:
            self.representation = representation
        else:
            self.representation = self.random_initial_representation()

        self._fitness = None


    # Random initial genotype
    def random_initial_representation(self):
        genes = np.empty((self.n_triangles * 10,), dtype=np.int16)
        
        for i in range(self.n_triangles):
            
            x1 = random.randint(0, self.img_width)
            y1 = random.randint(0, self.img_height)
            x2 = random.randint(0, self.img_width)
            y2 = random.randint(0, self.img_height)
            x3 = random.randint(0, self.img_width)
            y3 = random.randint(0, self.img_height)
            
            vertices = [x1, y1, x2, y2, x3, y3]
            colour = [random.randint(0, 255) for _ in range(4)]
            
            genes[i*10:(i*10)+10] = vertices + colour
        
        return genes


    # Fitness
    def fitness(self):
        if self._fitness is not None:
            return self._fitness

        rendered = self._render()
        rendered_arr = np.array(rendered, dtype=np.float32)

        target = self.target_array.astype(np.float32)

        diff = rendered_arr - target
        self._fitness = float(np.sqrt(np.mean(diff ** 2)))
        
        return self._fitness


    # Rendering
    def _render(self):
        canvas = Image.new('RGB', (self.img_width, self.img_height), (0, 0, 0))
        draw = ImageDraw.Draw(canvas, 'RGBA')

        for i in range(self.n_triangles):
            base = i * 10
            triangle = self.representation[base: base + 10]

            x1 = triangle[0]
            y1 = triangle[1]
            x2 = triangle[2]
            y2 = triangle[3]
            x3 = triangle[4]
            y3 = triangle[5]

            r = triangle[6]
            g = triangle[7]
            b = triangle[8]
            a = triangle[9]

            draw.polygon([(x1, y1), (x2, y2), (x3, y3)], fill=(r, g, b, a))

        # fixes bug where the triangles were drawn without transparency
        return canvas.convert('RGBA')


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
        axes[0].imshow(self.target_array)
        axes[0].set_title('Target')
        axes[0].axis('off')

        axes[1].imshow(np.array(self._render()))
        axes[1].set_title(f'Candidate (RMSE = {self.fitness():.2f})')
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()