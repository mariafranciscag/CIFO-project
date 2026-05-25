import numpy as np
from scripts.ga_utils import invalidate_cache
from scripts.triangle_sol import TriangleSolution


def triangle_gaussian_mutation(individual: TriangleSolution, mutation_rate: float = 0.01, sigma_vertex: float = 5, sigma_colour: float = 10, sigma_mutation_rate: float = 0.25) -> TriangleSolution:
    """
    Gaussian mutation on full triangle (vertices + RGBA).

    Works with both TriangleSolution (10 genes, uses mutation_rate parameter) 
    and AdaptiveTriangleSolution (11 genes, uses evolved per-triangle rates).
    
    Each triangle is mutated independently with probability given by:
    - mutation_rate parameter (for base TriangleSolution)
    - evolved gene 10 (for AdaptiveTriangleSolution)
    
    Vertex genes are perturbed with `sigma_vertex` (pixels); colour genes with `sigma_colour` (0-255 units).
    All values are clipped to their valid domains after perturbation.
    """
    rep = individual.representation[:]
    n_triangles = individual.n_triangles
    triangle_len = individual.triangle_len
    is_adaptive = (triangle_len == 11)

    for t in range(n_triangles):
        base = t * triangle_len
        
        # Get mutation rate (either parameter or evolved gene)
        if is_adaptive:
            triangle_mutation_rate = rep[base + 10]
        else:
            triangle_mutation_rate = mutation_rate
        
        if np.random.random() < triangle_mutation_rate:
            # Mutate vertices (genes 0-5)
            for i in [0, 2, 4]:
                rep[base + i] = np.clip(
                    rep[base + i] + np.random.normal(0, sigma_vertex), 0, 300
                )
            for i in [1, 3, 5]:
                rep[base + i] = np.clip(
                    rep[base + i] + np.random.normal(0, sigma_vertex), 0, 400
                )
            # Mutate colours (genes 6-9)
            for i in range(6, 10):
                rep[base + i] = np.clip(
                    rep[base + i] + np.random.normal(0, sigma_colour), 0, 255
                )
        
        # If adaptive, evolve the mutation rate itself
        if is_adaptive:
            rep[base + 10] = np.clip(
                rep[base + 10] + np.random.normal(0, sigma_mutation_rate), 0, 1
            )

    individual.representation = rep
    invalidate_cache(individual)
    return individual


def triangle_gaussian_color_mutation(individual: TriangleSolution, mutation_rate: float = 0.01, sigma_colour: float = 15) -> TriangleSolution:
    """
    Gaussian creep mutation on color genes only.

    Works with both TriangleSolution (10 genes) and AdaptiveTriangleSolution (11 genes).
    Perturbs only the RGBA channels (genes 6–9) of each triangle with Gaussian noise.
    Useful for fine-tuning colour without disturbing geometry.
    """
    rep = individual.representation[:]
    n_triangles = individual.n_triangles
    triangle_len = individual.triangle_len
    is_adaptive = (triangle_len == 11)

    for t in range(n_triangles):
        base = t * triangle_len
        
        # Get mutation rate (either parameter or evolved gene)
        if is_adaptive:
            triangle_mutation_rate = rep[base + 10]
        else:
            triangle_mutation_rate = mutation_rate
        
        if np.random.random() < triangle_mutation_rate:
            for i in range(6, 10):
                rep[base + i] = np.clip(
                    rep[base + i] + np.random.normal(0, sigma_colour), 0, 255
                )
        
        # If adaptive, evolve the mutation rate itself
        if is_adaptive:
            rep[base + 10] = np.clip(
                rep[base + 10] + np.random.normal(0, 1), 0, 1
            )

    individual.representation = rep
    invalidate_cache(individual)
    return individual


def triangle_gaussian_vertex_mutation(individual: TriangleSolution, mutation_rate: float = 0.01, sigma_vertex: float = 5) -> TriangleSolution:
    """
    Gaussian creep mutation on vertex genes only.

    Works with both TriangleSolution (10 genes) and AdaptiveTriangleSolution (11 genes).
    Perturbs only the coordinate genes (0–5) of each triangle with Gaussian noise (sigma in pixels).
    Useful for refining shape without disturbing colour.
    """
    rep = individual.representation[:]
    n_triangles = individual.n_triangles
    triangle_len = individual.triangle_len
    is_adaptive = (triangle_len == 11)

    for t in range(n_triangles):
        base = t * triangle_len
        
        # Get mutation rate (either parameter or evolved gene)
        if is_adaptive:
            triangle_mutation_rate = rep[base + 10]
        else:
            triangle_mutation_rate = mutation_rate
        
        if np.random.random() < triangle_mutation_rate:
            for i in [0, 2, 4]:
                rep[base + i] = np.clip(
                    rep[base + i] + np.random.normal(0, sigma_vertex), 0, 300
                )
            for i in [1, 3, 5]:
                rep[base + i] = np.clip(
                    rep[base + i] + np.random.normal(0, sigma_vertex), 0, 400
                )
        
        # If adaptive, evolve the mutation rate itself
        if is_adaptive:
            rep[base + 10] = np.clip(
                rep[base + 10] + np.random.normal(0, 1), 0, 1
            )

    individual.representation = rep
    invalidate_cache(individual)
    return individual

def triangle_reset_gene_mutation(individual: TriangleSolution, mutation_rate: float = 0.01) -> TriangleSolution:
    """
    Reset mutation with per-gene granularity.
    
    Works with both TriangleSolution (10 genes) and AdaptiveTriangleSolution (11 genes).
    Each gene independently has probability of being reset to a random value.
    For adaptive solutions, also evolves the mutation rate gene.
    Provides fine-grained stochastic replacement of genetic material.
    """
    rep = individual.representation[:]
    n_triangles = individual.n_triangles
    triangle_len = individual.triangle_len
    is_adaptive = (triangle_len == 11)

    for t in range(n_triangles):
        base = t * triangle_len
        
        # Get mutation rate (either parameter or evolved gene)
        if is_adaptive:
            triangle_mutation_rate = rep[base + 10]
        else:
            triangle_mutation_rate = mutation_rate
        
        # Mutate genes 0-9 (not the rate, which is always at gene 10 if adaptive)
        for gene_idx in range(10):
            if np.random.random() < triangle_mutation_rate:
                if gene_idx in [0, 2, 4]:  # x coordinates
                    rep[base + gene_idx] = np.random.randint(0, individual.img_width)
                elif gene_idx in [1, 3, 5]:  # y coordinates
                    rep[base + gene_idx] = np.random.randint(0, individual.img_height)
                else:  # colors (6-9)
                    rep[base + gene_idx] = np.random.randint(0, 255)
        
        # If adaptive, evolve the mutation rate itself (gene 10)
        if is_adaptive:
            rep[base + 10] = np.random.uniform(0, 1)
        
    individual.representation = rep
    invalidate_cache(individual)
    return individual

def triangle_reset_mutation(individual: TriangleSolution, mutation_rate: float = 0.01) -> TriangleSolution:
    """
    Reset mutation with per-triangle granularity.
    
    Works with both TriangleSolution (10 genes) and AdaptiveTriangleSolution (11 genes).
    Each triangle independently has probability of having all its geometry and color genes reset.
    For adaptive solutions, also evolves the mutation rate gene.
    Useful for introducing large-scale structural changes.
    """
    rep = individual.representation[:]
    n_triangles = individual.n_triangles
    triangle_len = individual.triangle_len
    is_adaptive = (triangle_len == 11)
    
    for t in range(n_triangles):
        base = t * triangle_len
        
        # Get mutation rate (either parameter or evolved gene)
        if is_adaptive:
            triangle_mutation_rate = rep[base + 10]
        else:
            triangle_mutation_rate = mutation_rate
        
        if np.random.random() < triangle_mutation_rate:
            # Reset all geometry and color genes (0-9)
            # Vertex coordinates (x)
            rep[base + 0] = np.random.randint(0, individual.img_width)
            rep[base + 2] = np.random.randint(0, individual.img_width)
            rep[base + 4] = np.random.randint(0, individual.img_width)
            # Vertex coordinates (y)
            rep[base + 1] = np.random.randint(0, individual.img_height)
            rep[base + 3] = np.random.randint(0, individual.img_height)
            rep[base + 5] = np.random.randint(0, individual.img_height)
            # Color genes (RGBA)
            rep[base + 6:base + 10] = np.random.randint(0, 255, size=4)
        
            # If adaptive, evolve the mutation rate itself (gene 10)
            if is_adaptive:
                rep[base + 10] = np.clip(np.random.lognormal(-2, 1), 0, 1)
    
    individual.representation = rep
    invalidate_cache(individual)
    return individual

def triangle_layer_swap_mutation(individual: TriangleSolution, mutation_rate: float = 0.01) -> TriangleSolution:
    """
    Structural mutation: swap the layering order of two triangles.
    
    Works with both TriangleSolution (10 genes) and AdaptiveTriangleSolution (11 genes).
    With probability equal to the mutation_rate parameter (or average of evolved rates for adaptive),
    two random triangles exchange their positions in the drawing order.
    For adaptive solutions, both triangles' rates continue to evolve.
    Enables exploration of occlusion patterns without modifying geometry or color.
    """
    rep = individual.representation[:]
    n_triangles = individual.n_triangles
    triangle_len = individual.triangle_len
    is_adaptive = (triangle_len == 11)
    
    # Select two random triangles
    idx1, idx2 = np.random.choice(n_triangles, 2, replace=False)
    base1 = idx1 * triangle_len
    base2 = idx2 * triangle_len
    
    # Determine swap probability
    if is_adaptive:
        # Use average of the two triangles' mutation rates
        avg_swap_rate = (rep[base1 + 10] + rep[base2 + 10]) / 2.0
    else:
        # Use the function parameter
        avg_swap_rate = mutation_rate
    
    if np.random.random() < avg_swap_rate:
        # Swap the triangle blocks
        rep[base1:base1+triangle_len], rep[base2:base2+triangle_len] = \
            rep[base2:base2+triangle_len].copy(), rep[base1:base1+triangle_len].copy()
    
    # # If adaptive, evolve mutation rates of both triangles
    # if is_adaptive:
    #     rep[base1 + 10] = np.clip(
    #         rep[base1 + 10] + np.random.normal(0, 1), 0, 1
    #     )
    #     rep[base2 + 10] = np.clip(
    #         rep[base2 + 10] + np.random.normal(0, 1), 0, 1
    #     )
    
    individual.representation = rep
    invalidate_cache(individual)
    return individual