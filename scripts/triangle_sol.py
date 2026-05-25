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
        representation: NDArray[np.float32] | None = None,
        n_triangles: int = 100,
        img_width: int = 300,
        img_height: int = 400,
        target_array: NDArray[np.float32] | None = None,
    ):

        self.n_triangles = n_triangles
        self.img_width = img_width
        self.img_height = img_height
        
        # Base: 10 genes per triangle [x1, y1, x2, y2, x3, y3, R, G, B, A]
        self.triangle_len = 10

        # Set instance target_array if is defined
        if target_array is not None:
            self.target_array = target_array
        elif TriangleSolution.target_array is not None:
            self.target_array = TriangleSolution.target_array
        else:
            raise Exception("No valid target array for solution.")

        if representation is not None:
            self.representation = representation
        else:
            self.representation = self.random_initial_representation()

        self._fitness = None


    # Random initial genotype
    def random_initial_representation(self):
        genes = np.empty((self.n_triangles * self.triangle_len,), dtype=np.float32)
        
        for i in range(self.n_triangles):
            base = i * self.triangle_len
            
            # Vertices (genes 0-5)
            genes[base + 0] = np.random.randint(0, self.img_width)      # x1
            genes[base + 1] = np.random.randint(0, self.img_height)     # y1
            genes[base + 2] = np.random.randint(0, self.img_width)      # x2
            genes[base + 3] = np.random.randint(0, self.img_height)     # y2
            genes[base + 4] = np.random.randint(0, self.img_width)      # x3
            genes[base + 5] = np.random.randint(0, self.img_height)     # y3
            
            # Colour (genes 6-9)
            genes[base + 6] = np.random.randint(0, 255)                 # R
            genes[base + 7] = np.random.randint(0, 255)                 # G
            genes[base + 8] = np.random.randint(0, 255)                 # B
            genes[base + 9] = np.random.randint(0, 255)                 # A
        
        return genes


    # Fitness
    def fitness(self):
        if self._fitness is not None:
            return self._fitness

        rendered = self._render()
        rendered_arr = np.array(rendered, dtype=np.float32)

        target = self.target_array

        diff = rendered_arr - target # type: ignore
        self._fitness = float(np.sqrt(np.mean(diff ** 2)))
        
        return self._fitness


    # Rendering
    def _render(self):
        canvas = Image.new('RGB', (self.img_width, self.img_height), (0, 0, 0))
        draw = ImageDraw.Draw(canvas, 'RGBA')

        for i in range(self.n_triangles):
            base = i * self.triangle_len
            triangle = self.representation[base: base + self.triangle_len]

            x1 = int(triangle[0])
            y1 = int(triangle[1])
            x2 = int(triangle[2])
            y2 = int(triangle[3])
            x3 = int(triangle[4])
            y3 = int(triangle[5])

            r = int(triangle[6])
            g = int(triangle[7])
            b = int(triangle[8])
            a = int(triangle[9])

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
        axes[0].imshow(self.target_array / 255) # type: ignore
        axes[0].set_title('Target')
        axes[0].axis('off')

        axes[1].imshow(np.array(self._render()))
        axes[1].set_title(f'Candidate (RMSE = {self.fitness():.2f})')
        axes[1].axis('off')

        plt.tight_layout()
        plt.show()


class AdaptiveMutationRateTriangleSolution(TriangleSolution):
    """
    Extends TriangleSolution with per-triangle self-adaptive mutation rates.
    
    Each triangle has 11 genes instead of 10:
    [x1, y1, x2, y2, x3, y3, R, G, B, A, mutation_rate]
    
    The 11th gene (mutation_rate) is evolved and controls how aggressively
    that triangle is mutated during evolution.
    """
    initial_mutation_rate = 0.01
    mutation_rate_penalty_weight = 10.0
    
    def __init__(
        self,
        representation: NDArray[np.float32] | None = None,
        n_triangles: int = 100,
        img_width: int = 300,
        img_height: int = 400,
        target_array: NDArray[np.float32] | None = None,
    ):
        self.n_triangles = n_triangles
        self.img_width = img_width
        self.img_height = img_height
        self.triangle_len = 11

        if target_array is not None:
            self.target_array = target_array
        else:
            self.target_array = TriangleSolution.target_array

        if representation is not None:
            self.representation = np.asarray(representation, dtype=np.float64)
        else:
            self.representation = self.random_initial_representation()

        self._fitness = None

    def random_initial_representation(self):
        """Generate 11-gene representation with evolved mutation rates."""
        genes = np.empty((self.n_triangles * self.triangle_len,), dtype=np.float32)
        
        for i in range(self.n_triangles):
            base = i * self.triangle_len
            
            # Vertices (genes 0-5)
            genes[base + 0] = np.random.randint(0, self.img_width)      # x1
            genes[base + 1] = np.random.randint(0, self.img_height)     # y1
            genes[base + 2] = np.random.randint(0, self.img_width)      # x2
            genes[base + 3] = np.random.randint(0, self.img_height)     # y2
            genes[base + 4] = np.random.randint(0, self.img_width)      # x3
            genes[base + 5] = np.random.randint(0, self.img_height)     # y3
            
            # Colour (genes 6-9)
            genes[base + 6] = np.random.randint(0, 255)                 # R
            genes[base + 7] = np.random.randint(0, 255)                 # G
            genes[base + 8] = np.random.randint(0, 255)                 # B
            genes[base + 9] = np.random.randint(0, 255)                 # A
            
            # Self-adaptive mutation rate (gene 10)
            genes[base + 10] = np.clip(
                np.random.lognormal(np.log(self.initial_mutation_rate), 0.5), 0, 1
            )
        
        return genes

    # Fitness
    def fitness(self):
        if self._fitness is not None:
            return self._fitness

        rendered = self._render()
        rendered_arr = np.array(rendered, dtype=np.float32)

        target = self.target_array
        
        mean_mutation_rate = float(np.mean([
            rate
            for rate in self.representation[10::self.triangle_len]
        ]))

        diff = rendered_arr - target # type: ignore
        self._fitness = float(
            np.sqrt(np.mean(diff ** 2))
            + self.mutation_rate_penalty_weight * mean_mutation_rate
        )
        
        return self._fitness
    
    def __repr__(self):
        return f"AdaptiveTriangleSolution(fitness={self.fitness():.4f}, n_triangles={self.n_triangles}, adaptive_rates=True)"
