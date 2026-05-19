import numpy as np
from scripts.ga_utils import invalidate_cache
from scripts.triangle_sol import TriangleSolution


def triangle_gaussian_mutation(individual: TriangleSolution, mutation_rate: float = 0.01, sigma_vertex: float = 5, sigma_colour: float = 10) -> TriangleSolution:
    """
    Gaussian mutation on full triangle (vertices + RGBA).

    Each triangle is mutated independently with probability `mutation_rate`.
    Vertex genes are perturbed with `sigma_vertex` (pixels); colour genes with `sigma_colour` (0-255 units).
    All values are clipped to their valid domains after perturbation.
    """
    rep = individual.representation[:]
    n_triangles = individual.n_triangles

    for t in range(n_triangles):
        if np.random.random() < mutation_rate:
            base = t * 10
            for i in [0, 2, 4]:
                rep[base + i] = np.clip(
                    rep[base + i] + np.random.normal(0, sigma_vertex), 0, 300
                )
            for i in [1, 3, 5]:
                rep[base + i] = np.clip(
                    rep[base + i] + np.random.normal(0, sigma_vertex), 0, 400
                )
            for i in range(6, 10):
                rep[base + i] = np.clip(
                    rep[base + i] + np.random.normal(0, sigma_colour), 0, 255
                )

    individual.representation = rep
    invalidate_cache(individual)
    return individual


def triangle_gaussian_color_mutation(individual: TriangleSolution, mutation_rate: float = 0.01, sigma_colour: float = 15) -> TriangleSolution:
    """
    Gaussian creep mutation on color genes only.

    Perturbs only the RGBA channels (genes 6–9) of each triangle with Gaussian noise.
    Useful for fine-tuning colour without disturbing geometry.
    """
    rep = individual.representation[:]
    n_triangles = individual.n_triangles

    for t in range(n_triangles):
        if np.random.random() < mutation_rate:
            base = t * 10
            for i in range(6, 10):
                rep[base + i] = np.clip(
                    rep[base + i] + np.random.normal(0, sigma_colour), 0, 255
                )

    individual.representation = rep
    invalidate_cache(individual)
    return individual


def triangle_gaussian_vertex_mutation(individual: TriangleSolution, mutation_rate: float = 0.01, sigma_vertex: float = 5) -> TriangleSolution:
    """
    Gaussian creep mutation on vertex genes only.

    Perturbs only the coordinate genes (0–5) of each triangle with Gaussian noise (sigma in pixels).
    Useful for refining shape without disturbing colour.
    """
    rep = individual.representation[:]
    n_triangles = individual.n_triangles

    for t in range(n_triangles):
        if np.random.random() < mutation_rate:
            base = t * 10
            for i in [0, 2, 4]:
                rep[base + i] = np.clip(
                    rep[base + i] + np.random.normal(0, sigma_vertex), 0, 300
                )
            for i in [1, 3, 5]:
                rep[base + i] = np.clip(
                    rep[base + i] + np.random.normal(0, sigma_vertex), 0, 400
                )

    individual.representation = rep
    invalidate_cache(individual)
    return individual

def triangle_reset_gene_mutation(individual: TriangleSolution, mutation_rate: float = 0.01) -> TriangleSolution:
    """
    Reset mutation with per-gene granularity.
    
    Each gene independently has probability `mutation_rate` of being reset to a random value.
    Provides fine-grained stochastic replacement of genetic material.
    """
    rep = individual.representation[:]
    n_triangles = individual.n_triangles

    # Create mutation mask for all genes
    mask = np.random.random(rep.shape) < mutation_rate

    # Generate candidate new values
    candidates = np.empty(rep.shape, dtype=float)
    for t in range(n_triangles):
        base = t * 10
        # Vertex genes (x coordinates)
        candidates[base:base+10][0] = np.random.randint(0, individual.img_width)
        candidates[base:base+10][2] = np.random.randint(0, individual.img_width)
        candidates[base:base+10][4] = np.random.randint(0, individual.img_width)
        # Vertex genes (y coordinates)
        candidates[base:base+10][1] = np.random.randint(0, individual.img_height)
        candidates[base:base+10][3] = np.random.randint(0, individual.img_height)
        candidates[base:base+10][5] = np.random.randint(0, individual.img_height)
        # Color genes (RGBA)
        candidates[base+6:base+10] = np.random.randint(0, 255, size=4)

    # Apply mutations
    rep[:] = np.where(mask, candidates, rep)
    individual.representation = rep
    invalidate_cache(individual)
    return individual

def triangle_reset_mutation(individual: TriangleSolution, mutation_rate: float = 0.01) -> TriangleSolution:
    """
    Reset mutation with per-triangle granularity.
    
    Each triangle independently has probability `mutation_rate` of having all its genes 
    (vertices and colors) reset to random values. Useful for introducing large-scale 
    structural changes.
    """
    rep = individual.representation[:]
    n_triangles = individual.n_triangles
    
    for t in range(n_triangles):
        if np.random.random() < mutation_rate:
            base = t * 10
            # Vertex coordinates (x)
            rep[base] = np.random.randint(0, individual.img_width)
            rep[base+2] = np.random.randint(0, individual.img_width)
            rep[base+4] = np.random.randint(0, individual.img_width)
            # Vertex coordinates (y)
            rep[base+1] = np.random.randint(0, individual.img_height)
            rep[base+3] = np.random.randint(0, individual.img_height)
            rep[base+5] = np.random.randint(0, individual.img_height)
            # Color genes (RGBA)
            rep[base+6:base+10] = np.random.randint(0, 255, size=4)
    
    individual.representation = rep
    invalidate_cache(individual)
    return individual

def triangle_layer_swap_mutation(individual: TriangleSolution, mutation_rate: float = 0.01) -> TriangleSolution:
    """
    Structural mutation: swap the layering order of two triangles.
    
    With probability `mutation_rate`, two random triangles exchange their positions in the 
    drawing order. This changes which triangle occludes which without modifying geometry 
    or color, enabling exploration of occlusion patterns.
    """
    rep = individual.representation[:]
    n_triangles = individual.n_triangles
    
    if np.random.random() < mutation_rate:
        # Select two random triangles to swap
        idx1, idx2 = np.random.choice(n_triangles, 2, replace=False)
        base1 = idx1 * 10
        base2 = idx2 * 10
        
        # Swap the 10-gene blocks
        rep[base1:base1+10], rep[base2:base2+10] = rep[base2:base2+10].copy(), rep[base1:base1+10].copy()
    
    individual.representation = rep
    invalidate_cache(individual)
    return individual