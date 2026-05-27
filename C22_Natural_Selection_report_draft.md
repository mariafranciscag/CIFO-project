# C22_Natural_Selection

## Vermeer's *Girl with a Pearl Earring*: 100 Triangles Challenge

**Computational Intelligence for Optimization, 2025/2026**

**Group members:** Afonso Hermenegildo (20221958), Andre Ferreira (20250398), Marco Martins (20221951), Maria Francisca Goncalves (20221942)

---

## 1. Introduction

The goal of this project is to approximate the provided 300 x 400 image of Vermeer's *Girl with a Pearl Earring* using exactly 100 colored triangles. This is an image-approximation optimization problem: each candidate solution defines a rendered image, and the algorithm searches for triangle configurations that make the rendered image as close as possible to the target.

The main project objective follows the statement requirement: minimize the pixel-wise RMSE between the rendered candidate and the original image. The selected additional challenge is **Challenge 1: using an alternative fitness function**. For that challenge, we keep the same GA structure and representation, but replace RMSE with a perceptual color-difference metric based on CIEDE2000.

The report is structured as follows. First, we define the optimization problem and representation. Then, we describe the implemented genetic operators. Next, we present the experimental methodology used to choose operators and hyperparameters. Finally, we report the current final GA configuration, the perceptual-fitness challenge, and additional experiments such as fitness sharing and hill climbing.

Exact RMSE values, p-values, and final plots should be refreshed after the last notebook rerun. The conclusions in this draft reflect the current state of the project, while numerical placeholders marked `TODO` should be replaced before export.

## 2. Problem Definition

### 2.1 Formal Optimization Problem

The optimization problem is to find a sequence of 100 semi-transparent triangles that minimizes the difference between the rendered candidate image and the target image. Let `x` be a candidate genotype and `render(x)` be the image produced by drawing its triangles in order. The main objective is:

```text
minimize RMSE(render(x), target)
```

This is a continuous, bounded optimization problem with stochastic search. Coordinates, colors, and alpha values are numerical genes, and small perturbations can produce meaningful local changes in the image. At the same time, the rendering order creates strong interactions between genes because later triangles can cover earlier ones.

### 2.2 Solution Representation

Each individual is an ordered list of 100 triangles. A triangle is encoded by 10 genes:

```text
[x1, y1, x2, y2, x3, y3, R, G, B, A]
```

The six coordinate genes are stored in image-pixel units and clipped to the canvas bounds. The color genes store RGBA values in `[0, 255]`. The order of triangles is part of the genotype because the image is rendered sequentially: later triangles can occlude earlier triangles.

An adaptive representation was also implemented with an eleventh gene per triangle:

```text
[x1, y1, x2, y2, x3, y3, R, G, B, A, mutation_rate]
```

This representation was used to test whether mutation rates should evolve inside the genotype. It remains an experimental variant; the current final GA uses the simpler fixed-rate representation because it is easier to control and performed better in the current comparison.

### 2.3 Search Space

The search space is extremely large. Even ignoring the continuous nature of coordinates and colors, each candidate contains 100 triangles and each triangle has 10 bounded genes. With integer pixel coordinates and 8-bit color channels, the number of possible candidates is far beyond exhaustive search:

```text
(300 * 400)^3 * 256^4 possibilities per triangle
```

raised to 100 ordered triangles. The rendering order makes the space even more difficult because moving or recoloring one triangle can change the visible contribution of many others through occlusion. This scale motivates the use of a Genetic Algorithm rather than exhaustive search or random sampling.

### 2.4 RMSE Fitness Function

The main project metric is the pixel-wise RMSE between the rendered candidate and the target:

```text
RMSE = sqrt(mean((candidate - target)^2))
```

Lower RMSE means a better solution. This metric is computationally simple and matches the project statement, but it measures numerical pixel similarity rather than perceived visual similarity. Each individual caches its fitness value, and populations are evaluated in parallel using `joblib`.

## 3. Genetic Operators

The notebook separates the definition of genetic operators from the statistical tests used to choose among them. This follows a clearer report structure: first define the design space, then justify the final choices empirically.

### 3.1 Selection

Selection chooses parents based on fitness. Because this is a minimization problem, lower RMSE corresponds to better fitness.

- **[Roulette-wheel selection](operators/selection.py#L5):** uses inverted RMSE values so lower-error individuals have higher selection probability.
- **[Rank-based selection](operators/selection.py#L20):** assigns probabilities according to sorted rank rather than raw RMSE differences.
- **[Tournament selection](operators/selection.py#L36):** samples a subset of individuals and selects the best one; tournament size controls selection pressure.
- **[Fitness sharing](operators/new_fitness_sharing.py#L86):** adjusts fitness during selection by penalizing individuals in crowded regions of genotype space.

Tournament selection is the current final choice because its pressure is easy to tune. The notebook compares several tournament sizes and currently keeps tournament size 10.

### 3.2 Crossover

Crossover recombines parent genotypes. Since each group of genes defines a complete triangle, the discrete crossover operators preserve triangle boundaries instead of cutting inside a triangle.

- **[One-point crossover](operators/crossover.py#L7):** swaps the suffix of two parents after a randomly chosen triangle boundary.
- **[Two-point crossover](operators/crossover.py#L35):** swaps a middle segment between two triangle-boundary cut points.
- **[Uniform crossover](operators/crossover.py#L72):** independently chooses, for each triangle position, which parent contributes that triangle.
- **[Geometric crossover](operators/crossover.py#L107):** creates one real-valued offspring by taking a weighted combination of parent genes and clipping values to valid coordinate and color ranges.

Uniform whole-triangle crossover is the current final choice. One-point and two-point crossover preserve larger blocks of drawing order, but useful triangles are not guaranteed to be adjacent in the genotype. Uniform crossover mixes partial solutions more freely while still respecting triangle boundaries.

Geometric crossover is conceptually appropriate for continuous representations, but in this image problem averaging two triangles can also create an intermediate triangle that is less visually useful than either parent. It is kept as an experimental operator.

### 3.3 Mutation

Mutation introduces new variation by modifying existing individuals. Since the representation is bounded and mostly continuous, mutation operators either perturb genes locally or reset parts of the representation.

| Mutation operator | Intended contribution |
|---|---|
| [Full Gaussian mutation](operators/mutation.py#L6) | Jointly refines geometry and RGBA values. |
| [Color-only Gaussian mutation](operators/mutation.py#L61) | Changes RGBA values without moving triangle vertices. |
| [Vertex-only Gaussian mutation](operators/mutation.py#L100) | Changes triangle geometry without changing color. |
| [Per-gene reset mutation](operators/mutation.py#L142) | Randomly resets individual genes. |
| [Per-triangle reset mutation](operators/mutation.py#L183) | Replaces complete triangles for stronger exploration. |
| [Layer-swap mutation](operators/mutation.py#L227) | Swaps triangle drawing order, changing occlusion without changing shape or color. |
| [Geometric mutation](operators/mutation.py#L274) | Applies bounded uniform perturbations following the box-mutation idea for continuous optimization. |

The current final choice is full Gaussian mutation with tuned vertex and color sigmas. It provides local refinement for both triangle geometry and color, which is important once the main visual structure has emerged.

## 4. Performance Analysis

### 4.1 Experimental Methodology

The notebook compares operators and hyperparameters using repeated independent GA runs. Since GA performance depends on random initialization and stochastic operators, decisions are based on distributions of results rather than on a single execution.

For each configuration, the helper records best-fitness histories and final best RMSE values. Plots can show either average best fitness (ABF) or median best fitness (MBF), with standard-deviation bands. When comparing different population sizes, performance is plotted against cumulative fitness evaluations rather than generations, because larger populations perform more evaluations per generation.

When two configurations are close, the notebook performs statistical tests on the final best RMSE values. It first applies a normality test. If both samples are approximately normal, Welch's t-test is used. Otherwise, independent samples are compared with the Mann-Whitney U test, while paired samples use the Wilcoxon signed-rank test. The significance threshold is `alpha = 0.05`.

### 4.2 Operator Selection

The operator comparison covers selection, crossover, and mutation:

| Component | Values compared |
|---|---|
| Selection | Roulette wheel, rank-based, tournament with several tournament sizes |
| Crossover | One-point, two-point, uniform, geometric |
| Mutation operator | Full Gaussian, color Gaussian, vertex Gaussian, gene reset, triangle reset, layer swap, geometric |

The current conclusion is that tournament selection, uniform whole-triangle crossover, and full Gaussian mutation form the strongest and most stable base configuration. Tournament selection gives direct control over selection pressure, uniform crossover recombines triangles without splitting them, and Gaussian mutation performs useful local refinement in the continuous genotype.

### 4.3 Hyperparameter Tuning

After selecting the main operator families, the notebook tunes key hyperparameters:

| Hyperparameter | Values compared |
|---|---|
| Crossover rate | 0.90, 0.80, 0.70, 0.50, 0.30 |
| Mutation rate | 0.50, 0.20, 0.10, 0.05, 0.02, 0.01 |
| Gaussian sigma | `sigma_vertex` and `sigma_colour` grids |
| Mutation adaptation | Fixed rate, decay schedules, self-adaptive mutation-rate gene |
| Elitism | Several elite sizes |

The current final settings are:

- Crossover rate 0.9.
- `sigma_vertex = 5` and `sigma_colour = 20`.
- Mutation rate 0.05.
- Elite size 2.
- No mutation-rate decay.

Exact p-values and final mean/median RMSE values should be regenerated before the final PDF.

### 4.4 Population Size and Evaluation Budget

Population size must be compared carefully. A run with a larger population performs more fitness evaluations per generation, so plotting only by generation gives larger populations an unfair computational budget. The notebook was updated so population-size comparisons use equal or comparable fitness-evaluation budgets and plot ABF/MBF against cumulative evaluations.

The current population-size comparison considers 200, 400, 800, and 1500 individuals. The current final configuration uses population size 200 for more generations, giving it the intended evaluation budget while keeping each generation relatively cheap.

## 5. Final GA Result

The current final-run cell uses:

```text
population_size = 200
n_generations = 8000
selection = tournament_selection(tournament_size=10)
crossover = uniform_crossover
crossover_rate = 0.9
mutation = triangle_gaussian_mutation(sigma_vertex=5, sigma_colour=20)
mutation_rate = 0.05
elitism_n = 2
mutation_rate_decay = None
```

This configuration follows from the statistical comparison while using the intended total evaluation budget. The final RMSE, convergence plot, and saved image should be updated after the last full rerun.

**TODO before final PDF:** insert the final RMSE, final convergence plot, and final `best_result.png` generated by the rerun.

![Final 100-triangle approximation](best_result.png)

## 6. Challenge 1: Perceptual Fitness

The selected challenge is the alternative-fitness challenge. The implemented perceptual solution replaces RMSE with mean per-pixel CIEDE2000 color distance in CIELAB color space:

```text
fitness = mean(DeltaE2000(Lab(candidate), Lab(target)))
```

This is implemented in `PerceptualTriangleSolution`, a subclass of `TriangleSolution` that only changes the `fitness()` method. The GA loop, operators, representation, and caching behavior are unchanged.

The motivation is that equal numeric differences in RGB are not equally visible to humans. CIEDE2000 is more aligned with perceptual color differences, so it can reward images that look closer to a human observer even when their RMSE is not the lowest.

### 6.1 Paired Comparison Design

The experiment is designed as a paired comparison between the RMSE GA and the perceptual GA. For run `i`, both algorithms use the same seed and the same GA configuration, so the initial population and stochastic sequence are comparable. The only intended difference is the fitness function.

Because RMSE and Delta E2000 are on different scales, the comparison cross-evaluates every best solution on both metrics:

| Optimized fitness | Evaluated with RMSE | Evaluated with Delta E2000 |
|---|---:|---:|
| RMSE GA | TODO | TODO |
| Perceptual GA | TODO | TODO |

The paired Wilcoxon signed-rank test is then used to test whether one method consistently produces lower scores on each metric. The expected interpretation is that each GA may perform best on the metric it directly optimizes, while the report should also discuss which result appears visually closer to a human observer.

**TODO before final PDF:** run the paired perceptual comparison, insert the cross-evaluation table, Wilcoxon p-values, convergence plots, metric distributions, and visual comparison.

## 7. Additional Experiments

### 7.1 Fitness Sharing

Fitness sharing was implemented to encourage population diversity by penalizing individuals in crowded regions of the search space. The later implementation uses normalized representation distances and includes diversity-report helpers, which makes niche counts more meaningful than using raw coordinate and color distances directly.

Fitness sharing is useful as an exploration mechanism, but it adds computational overhead and is not currently part of the final GA configuration. It is reported as an additional diversity experiment rather than as the final selection method.

### 7.2 Hill Climbing for Local Search

The notebook also includes a hill-climbing polishing step applied after the GA. This step perturbs the best GA solution locally with decaying step sizes and accepts improvements. It is useful because the GA discovers a strong global structure, while local search can make small refinements to triangle vertices and colors.

If the final submitted image uses hill climbing, the report should count it as an extra post-processing phase and report both values:

```text
GA final RMSE: TODO
After hill climbing: TODO
Additional evaluations: TODO
```

If the submitted image uses only the GA result, this section should remain an additional experiment rather than part of the final result.

## 8. Conclusion

The implementation solves the assigned image-approximation problem with a fixed-length, real-valued representation of 100 semi-transparent triangles. The main GA uses the required RMSE objective, while the selected challenge extends the project with a perceptual CIEDE2000 fitness function.

The operator study supports the final design. Tournament selection was kept because it provides direct control over selection pressure. Uniform whole-triangle crossover was kept because it recombines partial image structures without breaking triangles. Full Gaussian mutation was kept because the genotype is continuous and local perturbations are meaningful for both geometry and color. The population-size comparison is now framed by fitness-evaluation budget, which makes the comparison fairer than plotting only by generation.

More complex additions, including fitness sharing, self-adaptive mutation rates, layer-swap mutation, geometric crossover, geometric mutation, and hill climbing, were implemented and considered. The current final GA keeps the simpler configuration because it is easier to justify statistically and has produced the best current convergence behavior, while the additional methods remain useful for analysis and future improvement.

## 9. References

**TODO before final PDF:** format references consistently. Suggested references:

1. Group project statement, Computational Intelligence for Optimization, 2025/2026.
2. L. Vanneschi and S. Silva, Genetic Algorithms chapter from the course material.
3. G. Sharma, W. Wu, and E. N. Dalal, "The CIEDE2000 Color-Difference Formula: Implementation Notes, Supplementary Test Data, and Mathematical Observations."
4. `scikit-image` documentation for `rgb2lab` and `deltaE_ciede2000`, if the implementation reference is included.

## Appendix: Reproducibility Notes

| File | Purpose |
|---|---|
| `scripts/triangle_sol.py` | Solution representation, rendering, adaptive representation, and RMSE fitness |
| `scripts/ga.py` | Main generational GA with elitism, one-child/two-child crossover support, and optional mutation-rate decay |
| `scripts/ga_utils.py` | Fitness caching and parallel population evaluation |
| `operators/selection.py` | Roulette, rank-based, and tournament selection |
| `operators/crossover.py` | One-point, two-point, uniform, and geometric crossover |
| `operators/mutation.py` | Gaussian, reset, layer-swap, and geometric mutation operators |
| `operators/new_fitness_sharing.py` | Normalized fitness-sharing selection and diversity metrics |
| `scripts/new_hill_climbing.py` | Local-search polishing after the GA |
| `scripts/perceptual_fitness.py` | CIEDE2000 perceptual fitness and cross-evaluation helpers |
| `scripts/perceptual_comparison.py` | Paired RMSE-vs-perceptual experiment, Wilcoxon tests, and plots |
| `scripts/statistical_comparison.py` | Repeated-run comparison, ABF/MBF plotting, statistical testing, and result saving |
| `main_nb.ipynb` | Full notebook workflow, plots, final run, challenge section, and saved result |

The final run is reproduced by executing the final GA section in `main_nb.ipynb` after loading `project_info/girl_pearl_earing.png` into `TriangleSolution.target_array`. Before exporting the final report, rerun the statistical and final-result cells, refresh `best_result.png`, and replace all `TODO` placeholders with the final values.

## Appendix: Figures and Tables to Add

The body of the report should keep only the most important plots. Extra plots can be moved here:

- Operator comparison plots for selection, crossover, and mutation.
- Hyperparameter comparison plots for mutation rate, sigma values, elitism, and crossover rate.
- Population-size comparison plotted against cumulative evaluations.
- Perceptual-fitness convergence and cross-evaluation plots.
- Final candidate image and any hill-climbing improvement curve.
