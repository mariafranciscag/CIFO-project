import numpy as np
from PIL import Image, ImageDraw

class Solution:
    
    # ADD ELITISM
    def __init__(
        self,
        width: int,
        height: int,
        n_triangles: int,
        target_image: np.ndarray,
        use_transparency: bool,
        representation: np.ndarray | None = None
    ) -> None:
        
        self.width = width
        self.height = height
        self.n_triangles = n_triangles
        self.use_transparency = use_transparency
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
        # [x1, y1, x2, y2, x3, y3, r, g, b, alpha]
        # or
        # [x1, y1, x2, y2, x3, y3, r, g, b]

        representation = np.empty((self.n_triangles, 9), dtype=int)
        
        if (self.use_transparency):
            representation = np.empty((self.n_triangles, 10), dtype=int)

        # fill x coordinates (columns 0, 2, 4)
        representation[:, [0, 2, 4]] = np.random.randint(0, self.width, size=(self.n_triangles, 3))

        # fill y coordinates (columns 1, 3, 5)
        representation[:, [1, 3, 5]] = np.random.randint(0, self.height, size=(self.n_triangles, 3))

        # fill color (columns 6, 7, 8)
        representation[:, [6, 7, 8]] = np.random.randint(0, 255, size=(self.n_triangles, 3))
        
        if (self.use_transparency):
            # fill alpha (column 9)
            representation[:, 9] = np.random.randint(0, 255, size=(self.n_triangles,))

        return representation

    def render(self):
        image = Image.new("RGB", (self.width, self.height))
        draw = ImageDraw.Draw(image, "RGBA" if self.use_transparency else "RGB")

        for tri in self.representation:
            if self.use_transparency:
                draw.polygon(
                    [(tri[0], tri[1]), (tri[2], tri[3]), (tri[4], tri[5])],
                    fill=(tri[6], tri[7], tri[8], tri[9])
                )
            else:
                draw.polygon(
                    [(tri[0], tri[1]), (tri[2], tri[3]), (tri[4], tri[5])],
                    fill=(tri[6], tri[7], tri[8])
                )

        return image.convert("RGBA" if self.use_transparency else "RGB")

    def calculate_fitness(self):
        solution_image = np.asarray(self.render(), dtype=np.uint8)

        # we need to convert to a bigger type because the difference in uint8 
        # will wrap around and give us incorrect results
        error = solution_image.astype(np.float16) - self.target_image.astype(np.float16)
        self.fitness = np.sqrt(np.mean(error ** 2))
        
        return self.fitness