import numpy as np
from numpy.typing import NDArray
from PIL import Image, ImageDraw

class Solution:
    
    # ADD ELITISM
    def __init__(
        self,
        width: int,
        height: int,
        n_triangles: int,
        target_image: NDArray[np.uint8],
        use_transparency: bool = True,
        use_self_adaptive_mutation_rate: bool = False,
        representation: NDArray | None = None
    ) -> None:
        
        self.width = width
        self.height = height
        self.n_triangles = n_triangles
        self.use_transparency = use_transparency
        self.use_self_adaptive_mutation_rate = use_self_adaptive_mutation_rate
        self.fitness = None
        
        if representation == None:
            self.representation = self.random_representation()
        else:
            self.representation = representation

        if use_transparency and target_image.shape != (400, 300, 4):
            raise Exception("Invalid target image shape using transparency.")
        
        if not use_transparency and target_image.shape != (400, 300, 3):
            raise Exception("Invalid target image shape not using transparency.")
        
        self.target_image = target_image
    
    def random_representation(self):
        # format of our individual triangle is:
        # [x1, y1, x2, y2, x3, y3, r, g, b, alpha, mutation_rate]
        # or
        # [x1, y1, x2, y2, x3, y3, r, g, b, mutation_rate]
        # or
        # [x1, y1, x2, y2, x3, y3, r, g, b, alpha]
        # or
        # [x1, y1, x2, y2, x3, y3, r, g, b]

        n_genes = 9
        
        if (self.use_transparency):
            n_genes += 1
        
        if (self.use_self_adaptive_mutation_rate):
            n_genes += 1

        representation = np.empty((self.n_triangles, n_genes), dtype=float)
        
        # fill x coordinates (columns 0, 2, 4)
        representation[:, [0, 2, 4]] = np.random.randint(0, self.width, size=(self.n_triangles, 3))

        # fill y coordinates (columns 1, 3, 5)
        representation[:, [1, 3, 5]] = np.random.randint(0, self.height, size=(self.n_triangles, 3))

        # fill color (columns 6, 7, 8)
        representation[:, [6, 7, 8]] = np.random.randint(0, 255, size=(self.n_triangles, 3))
        
        if self.use_transparency and self.use_self_adaptive_mutation_rate:
            
            # fill alpha and mutation rate (column 9 and 10)
            representation[:, 9] = np.random.randint(0, 255, size=(self.n_triangles,))
            representation[:, 10] = np.random.rand(self.n_triangles)
            
        elif self.use_transparency:
            
            # fill alpha (column 9)
            representation[:, 9] = np.random.randint(0, 255, size=(self.n_triangles,))
            
        elif self.use_self_adaptive_mutation_rate:
            
            # fill mutation rate (column 9)
            representation[:, 9] = np.random.rand(self.n_triangles)

        return representation

    def render(self):
        image = Image.new("RGB", (self.width, self.height))
        draw = ImageDraw.Draw(image, "RGBA" if self.use_transparency else "RGB")

        for tri in self.representation:
            if self.use_transparency:
                draw.polygon(
                    [(int(tri[0]), int(tri[1])), (int(tri[2]), int(tri[3])), (int(tri[4]), int(tri[5]))],
                    fill=(int(tri[6]), int(tri[7]), int(tri[8]), int(tri[9]))
                )
            else:
                draw.polygon(
                    [(int(tri[0]), int(tri[1])), (int(tri[2]), int(tri[3])), (int(tri[4]), int(tri[5]))],
                    fill=(int(tri[6]), int(tri[7]), int(tri[8]))
                )

        return image.convert("RGBA" if self.use_transparency else "RGB")

    def calculate_fitness(self):
        solution_image = np.asarray(self.render(), dtype=np.uint8)

        # we need to convert to a bigger type because the difference in uint8 
        # will wrap around and give us incorrect results
        error = solution_image.astype(np.float32) - self.target_image.astype(np.float32)
        self.fitness = np.sqrt(np.mean(error ** 2))
        
        return self.fitness