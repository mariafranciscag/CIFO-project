# C22_Natural_Selection

## Vermeer's *Girl with a Pearl Earring*: 100 Triangles Challenge

**Computational Intelligence for Optimization, 2025/2026**

**Group members:** Afonso Hermenegildo (20221958), Andre Ferreira (20250398), Marco Martins (20221951), Maria Francisca Goncalves (20221942)

---

## 1. Introduction

The goal of this project is to approximate the provided 300 x 400 image of Vermeer's *Girl with a Pearl Earring* using exactly 100 colored triangles. This is an image-approximation optimization problem: each candidate solution defines a rendered image, and the algorithm searches for triangle configurations that make the rendered image as close as possible to the target.

The main project objective follows the statement requirement: minimize the pixel-wise RMSE between the rendered candidate and the original image. The selected additional challenge is **Challenge 1: using an alternative fitness function**. For that challenge, we keep the same GA structure and representation, but replace RMSE with a perceptual color-difference metric based on CIEDE2000.

The report is structured as follows. First, we define the optimization problem and representation. Then, we describe the implemented genetic operators. Next, we present the experimental methodology used to choose operators and hyperparameters. Finally, we report the perceptual-fitness challenge, the final result, and the main conclusions.

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

This representation was used to test whether mutation rates should evolve inside the genotype. It remains an experimental variant; the final GA uses the simpler fixed-rate representation because it is easier to control and performed better in the comparison.

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

Tournament selection is the final choice because its pressure is easy to tune. The notebook compares several tournament sizes and keeps tournament size 10.

### 3.2 Crossover

Crossover recombines parent genotypes. Since each group of genes defines a complete triangle, the discrete crossover operators preserve triangle boundaries instead of cutting inside a triangle.

- **[One-point crossover](operators/crossover.py#L7):** swaps the suffix of two parents after a randomly chosen triangle boundary.
- **[Two-point crossover](operators/crossover.py#L35):** swaps a middle segment between two triangle-boundary cut points.
- **[Uniform crossover](operators/crossover.py#L72):** independently chooses, for each triangle position, which parent contributes that triangle.
- **[Geometric crossover](operators/crossover.py#L107):** creates one real-valued offspring by taking a weighted combination of parent genes and clipping values to valid coordinate and color ranges.

Uniform whole-triangle crossover is the final choice. One-point and two-point crossover preserve larger blocks of drawing order, but useful triangles are not guaranteed to be adjacent in the genotype. Uniform crossover mixes partial solutions more freely while still respecting triangle boundaries.

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

The final choice is full Gaussian mutation with tuned vertex and color sigmas. It provides local refinement for both triangle geometry and color, which is important once the main visual structure has emerged.

### 3.4 Diversity Preservation

Fitness sharing was implemented to encourage population diversity by penalizing individuals in crowded regions of the search space. The later implementation uses normalized representation distances and includes diversity-report helpers, which makes niche counts more meaningful than using raw coordinate and color distances directly.

Fitness sharing is useful as an exploration mechanism, but it adds computational overhead and is not part of the selected final GA configuration. It is therefore reported as a diversity-preservation experiment rather than as the final selection method.

## 4. Performance Analysis

### 4.1 Experimental Methodology

The notebook compares operators and hyperparameters using repeated independent GA runs, usually 30 runs per configuration. Since GA performance depends on random initialization and stochastic operators, decisions are based on result distributions instead of single runs.

For each configuration, the helper records best-RMSE histories and final best RMSE values. Convergence plots show average best RMSE over time with standard-deviation bands. When population size is varied, plots use cumulative fitness evaluations instead of generations, because larger populations perform more evaluations per generation.

Statistical testing follows the same rule throughout the notebook. A normality check is applied first. For independent samples, Welch's t-test is used when both samples are approximately normal and the Mann-Whitney U test is used otherwise. For paired samples, a paired t-test is used for approximately normal data and the Wilcoxon signed-rank test is used otherwise. The significance threshold is `alpha = 0.05`.

### 4.2 Selection

The notebook first compares roulette-wheel selection, rank-based selection, and tournament selection with tournament sizes 2, 3, 5, and 10. Since tournament size 10 performed best among the initial settings, a follow-up test compared higher tournament sizes 15, 20, and 30.

Higher tournament sizes improved early convergence by increasing selection pressure, but the larger tournament sizes produced very similar final RMSE distributions. The differences among sizes 10, 15, 20, and 30 were not statistically significant. We therefore selected tournament size 10 as a balanced choice: it gives strong selection pressure without pushing diversity loss further than necessary.

### 4.3 Crossover

Four crossover operators were compared: one-point, two-point, uniform, and geometric crossover. The discrete operators preserve triangle boundaries, while geometric crossover returns one real-valued offspring by averaging parent genes.

Uniform crossover gave the best results. It significantly outperformed both one-point and two-point crossover in the notebook comparisons:

| Comparison | Test | Mean RMSE |
|---|---|---|
| One-point vs. uniform | Welch t-test, `p < 0.0001` | 39.6114 vs. 35.4099 |
| Two-point vs. uniform | Welch t-test, `p < 0.0001` | 38.7940 vs. 35.4099 |

This supports the decision to recombine whole triangles independently rather than preserving large contiguous blocks of drawing order. The crossover-rate sweep then showed that 0.90 was the strongest tested setting; it significantly improved over 0.80 (`p = 0.0032`, mean RMSE 34.9475 vs. 35.8176).

### 4.4 Mutation

The mutation comparison tested seven operators: full Gaussian mutation, color-only Gaussian mutation, vertex-only Gaussian mutation, per-gene reset, per-triangle reset, layer swap, and geometric mutation. Full Gaussian mutation was selected because it can refine both geometry and color in the same operator.

The statistical comparison supports this choice. Full Gaussian mutation significantly outperformed per-triangle reset mutation (`p < 0.0001`, mean RMSE 39.2494 vs. 40.8370). This is consistent with the representation: once useful image structure appears, small continuous changes are more useful than repeatedly replacing whole triangles.

The mutation-rate sweep compared 0.01, 0.02, 0.05, 0.10, 0.20, and 0.50 under a smaller repeated-run budget. These results were used to identify a reasonable mutation range, not to replicate the full 8000-generation final run. The final long run uses a lower initial mutation rate of 0.01 because long runs give mutation many more opportunities to refine the image, so high mutation rates can disrupt late-stage improvements.

### 4.5 Gaussian Step Sizes and Mutation Adaptation

The Gaussian mutation step sizes were tuned separately for vertices and colours. For `sigma_vertex`, the tested values were statistically equivalent, so a small value was preferred to keep geometric changes controlled. For `sigma_colour`, `sigma_colour = 20` and `sigma_colour = 50` were not significantly different (`p = 0.0778`), but `sigma_colour = 20` had a slightly better mean RMSE and a more stable late convergence trend.

The selected values are therefore `sigma_vertex = 5` and `sigma_colour = 20`.

Two mutation-adaptation strategies were also tested: global mutation-rate decay and a self-adaptive representation with a mutation-rate gene. In the shorter repeated-run tests, mutation decay did not show a consistent advantage over a fixed rate, and the self-adaptive representation did not outperform the static representation. However, decay is more relevant in long runs, where exploration early and refinement late are both important. The final 8000-generation run therefore uses a mild decay of 0.0001 as a conservative long-run refinement choice.

### 4.6 Elitism

The elitism comparison tested 2, 5, 10, and 20 preserved elites. Most settings produced similar convergence curves and final RMSE distributions, while 20 elites risked preserving too much of the population unchanged.

The final choice is 2 elites. This is enough to prevent losing the best solution between generations while leaving most of the population available for selection, crossover, and mutation.

### 4.7 Population Size and Evaluation Budget

The population-size comparison tested 200, 400, 800, and 1500 individuals. This comparison is evaluated against cumulative fitness evaluations rather than generations, since larger populations perform more evaluations per generation. A generation-based comparison would unfairly favour larger populations.

Larger populations can improve diversity, but they are much more expensive. Population size 200 offered the best quality-to-cost trade-off for the final long run because it allowed many generations within the available computation budget.

### 4.8 Selected Configuration

The final standard GA configuration used in the notebook is:

| Component | Selected setting |
|---|---|
| Selection | Tournament selection, tournament size 10 |
| Crossover | Uniform whole-triangle crossover |
| Crossover rate | 0.9 |
| Mutation | Full Gaussian mutation |
| Gaussian sigma | `sigma_vertex = 5`, `sigma_colour = 20` |
| Mutation rate | Initial rate 0.01 |
| Mutation-rate decay | 0.0001 |
| Elitism | 2 elites |
| Population size | 200 |

## 5. Perceptual Fitness (Challenge 1)

### 5.1 Motivation

The selected challenge is the alternative-fitness challenge. The baseline RMSE fitness measures numerical differences in RGB space, but equal RGB differences are not always equally visible to humans. This means that an image can have a good RMSE while still looking perceptually less natural in some regions.

To address this, we implemented a perceptual fitness function based on CIEDE2000 color distance in CIELAB space. CIEDE2000 is designed to better reflect perceived differences in lightness, chroma, and hue.

### 5.2 Implementation

The perceptual solution replaces RMSE with mean per-pixel CIEDE2000 color distance in CIELAB color space:

```text
fitness = mean(DeltaE2000(Lab(candidate), Lab(target)))
```

This is implemented in `PerceptualTriangleSolution`, a subclass of `TriangleSolution` that only changes the `fitness()` method. The GA loop, operators, representation, and caching behavior are unchanged. The target image is converted to Lab once and reused, while each rendered candidate is converted before computing the mean Delta E2000 score.

### 5.3 Experimental Setup

The experiment is designed as a paired comparison between the RMSE GA and the perceptual GA. For run `i`, both algorithms use the same seed and the same GA configuration, so the initial population and stochastic sequence are comparable. The only intended difference is the fitness function.

Because RMSE and Delta E2000 are on different scales, the comparison cross-evaluates every best solution on both metrics. This allows each final image to be evaluated both as a pixel-matching solution and as a perceptual-color solution.

### 5.4 Results

| Optimized fitness | Evaluated with RMSE | Evaluated with Delta E2000 |
|---|---:|---:|
| RGB-RMSE GA | 39.406 +/- 1.268 | 19.231 +/- 0.750 |
| Perceptual GA | 44.718 +/- 1.618 | 17.298 +/- 0.660 |

Each GA performs best on the metric it directly optimizes. The RMSE GA achieves lower RMSE, while the perceptual GA achieves lower Delta E2000. Since normality tests passed and runs were paired by seed, paired t-tests were used. The RMSE comparison produced `p = 6.38e-17`, and the Delta E2000 comparison produced `p = 2.30e-12`, indicating statistically significant differences.

### 5.5 Visual Comparison

The notebook also compares representative images from the RMSE-optimized and perceptual-optimized runs. The RMSE solution is better on exact pixel error, while the perceptual solution tends to look closer in colour balance. This is consistent with the cross-evaluation table: CIEDE2000 rewards perceptual colour agreement even when the pixel-wise RMSE is worse.

A single longer perceptual run was also executed with population size 200 and 2000 generations. It used the tuned GA configuration, but fewer generations than the final standard GA because CIEDE2000 fitness is much more expensive to evaluate.

### 5.6 Discussion

The perceptual GA sacrifices pixel-level accuracy in exchange for better perceptual color agreement. This is visible in smoother color regions such as skin tones and the dark background, where CIEDE2000 rewards color differences closer to how humans perceive them. The trade-off is that RMSE remains better for matching exact pixels and sharper edges.

## 6. Final Results

### 6.1 Final Standard GA

The final standard GA uses:

```text
population_size = 200
n_generations = 8000
selection = tournament_selection(tournament_size=10)
crossover = uniform_crossover
crossover_rate = 0.9
mutation = triangle_gaussian_mutation(sigma_vertex=5, sigma_colour=20)
mutation_rate = 0.01
elitism_n = 2
mutation_rate_decay = 0.0001
```

This configuration follows from the smaller-budget statistical comparisons and a final long-run refinement setting with mild mutation-rate decay. The final GA-only RMSE achieved was **16.8944**.

### 6.2 Hill-Climbing Polish

After the GA, a hill-climbing polishing step was applied to the best solution. This step perturbs triangle vertices and colors locally, accepts only RMSE-improving moves, and gradually reduces the step sizes. It is a post-processing phase rather than a replacement for the GA.

```text
GA final RMSE: 16.8944
After hill climbing: 16.3021
Additional evaluations: 50,232
```

The submitted image is the post-polish result:

![Final 100-triangle approximation](best_result.png)

### 6.3 Fitness-Sharing Result

A final-scale fitness-sharing run was also tested to check whether the diversity mechanism remained useful in a longer run. It is reported separately from the submitted result. Fitness sharing helped analyse population diversity, but the simpler tournament-selection GA remains easier to justify as the main configuration because it avoids the additional diversity-computation overhead.

## 7. Conclusion

The implementation solves the assigned image-approximation problem with a fixed-length, real-valued representation of 100 semi-transparent triangles. The main GA uses the required RMSE objective, while the selected challenge extends the project with a perceptual CIEDE2000 fitness function.

The operator study supports the final design. Tournament selection was kept because it provides direct control over selection pressure. Uniform whole-triangle crossover was kept because it recombines partial image structures without breaking triangles. Full Gaussian mutation was kept because the genotype is continuous and local perturbations are meaningful for both geometry and color. The population-size comparison is now framed by fitness-evaluation budget, which makes the comparison fairer than plotting only by generation.

More complex additions, including fitness sharing, self-adaptive mutation rates, layer-swap mutation, geometric crossover, geometric mutation, and hill climbing, were implemented and considered. The selected final GA keeps the simpler tournament-selection configuration because it is easier to justify statistically and avoids unnecessary overhead. Hill climbing is used only as a final post-processing polish and is reported separately from the GA-only RMSE.

## 8. References

1. Group project statement, Computational Intelligence for Optimization, 2025/2026.
2. L. Vanneschi and S. Silva, Genetic Algorithms chapter from the course material.
3. G. Sharma, W. Wu, and E. N. Dalal, "The CIEDE2000 Color-Difference Formula: Implementation Notes, Supplementary Test Data, and Mathematical Observations."
4. `scikit-image` documentation for `rgb2lab` and `deltaE_ciede2000`.

## 9. Annex

### 9.1 Reproducibility Notes

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
| `scripts/perceptual_comparison.py` | Paired RMSE-vs-perceptual experiment, cross-evaluation helpers, and plots |
| `scripts/statistical_comparison.py` | Repeated-run comparison, ABF/MBF plotting, statistical testing, and result saving |
| `main_nb.ipynb` | Full notebook workflow, plots, final run, challenge section, and saved result |

The final run is reproduced by executing the final GA section in `main_nb.ipynb` after loading `project_info/girl_pearl_earing.png` into `TriangleSolution.target_array`. The submitted `best_result.png` is produced by applying the hill-climbing section to the saved final GA solution.

### 9.2 Supplementary Figures and Tables

Additional supporting plots include:

- Operator comparison plots for selection, crossover, and mutation.
- Hyperparameter comparison plots for mutation rate, sigma values, elitism, and crossover rate.
- Population-size comparison plotted against cumulative evaluations.
- Perceptual-fitness convergence and cross-evaluation plots.
- Final candidate image and any hill-climbing improvement curve.
