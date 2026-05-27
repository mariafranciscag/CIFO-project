# AGENTS.md

This file is a handoff for future Codex chats working on this repository.

## Project Overview

- Course/project: Computational Intelligence for Optimization, 2025/2026.
- Project name: `C22_Natural_Selection`.
- Main task: approximate Vermeer's *Girl with a Pearl Earring* using exactly 100 colored triangles.
- Target image: `project_info/girl_pearl_earing.png`.
- Main notebook: `main_nb.ipynb`.
- Report draft: `C22_Natural_Selection_report_draft.md`.
- Main objective from the statement: minimize pixel-wise RMSE between rendered candidate and target image.
- Selected additional challenge: Challenge 1, alternative fitness function.
- The alternative fitness is perceptual color distance using CIEDE2000 in CIELAB space.

## User Preferences and Workflow

- If the user asks for a markdown draft, do not edit files unless they explicitly say editing is allowed.
- The user allowed editing the notebook/report in several recent requests, but this preference still matters for future markdown drafts.
- Keep report prose concise and project-focused.
- Exact experimental values have now been filled in from the final rerun.
- The notebook and report have undergone a final cleanup pass, including the removal of unused imports, unused variables, and old test cells.

## Important Files

### Core GA

- `scripts/triangle_sol.py`
  - `TriangleSolution`: 100 triangles, 10 genes per triangle.
  - `AdaptiveMutationRateTriangleSolution`: 11 genes per triangle, with mutation rate as an extra gene.
  - Rendering uses PIL and draws triangles sequentially, so later triangles occlude earlier triangles.
  - Fitness is cached in `_fitness`.

- `scripts/ga.py`
  - Main generational GA.
  - Supports elitism, mutation-rate decay, callbacks, seeded runs, and both one-child and two-child crossover outputs.
  - Important fix already made: duplicate `generation_callback` call was removed because it caused progress bars to reach 100% too early and report double speed.

- `scripts/ga_utils.py`
  - Fitness caching and parallel population evaluation helpers.

### Operators

- `operators/selection.py`
  - `roulette_wheel_selection`
  - `rank_based_selection`
  - `tournament_selection`

- `operators/crossover.py`
  - `one_point_crossover`
  - `two_point_crossover`
  - `uniform_crossover`
  - `geometric_crossover`
  - Important decision: `geometric_crossover` now returns one offspring, not two.
  - The GA loop in `scripts/ga.py` was updated to accept either one child or a tuple of two children.

- `operators/mutation.py`
  - `triangle_gaussian_mutation`
  - `triangle_gaussian_color_mutation`
  - `triangle_gaussian_vertex_mutation`
  - `triangle_reset_gene_mutation`
  - `triangle_reset_mutation`
  - `triangle_layer_swap_mutation`
  - `geometric_mutation`
  - `geometric_mutation` applies bounded uniform perturbations and clips to valid coordinate/color ranges.

- `operators/new_fitness_sharing.py`
  - Newer fitness-sharing implementation.
  - Includes normalized representation distances, niche counts, selection wrappers, and diversity metrics.
  - Prefer this over the older `operators/fitness_sharing.py` when discussing the final notebook.

### Experiments and Reporting Helpers

- `scripts/statistical_comparison.py`
  - Runs repeated GA comparisons.
  - Supports fixed GA parameters via `fixed_params`, so the varied parameter can be tested while all other parameters are held constant.
  - Supports equal evaluation budgets for population-size experiments.
  - Plots against cumulative evaluations, not generations, when needed.
  - Stores mean, median, std histories, final best values, population size, generation count, and evaluation axis.
  - Progress smoothing was added because tqdm ETA/speed jumped around.
  - Saving results now converts dictionary keys to strings so notebook lambdas/functions do not break pickle.

- `scripts/perceptual_fitness.py`
  - `PerceptualTriangleSolution`: subclass of `TriangleSolution` that replaces RMSE with mean per-pixel CIEDE2000.
  - `set_perceptual_target`, `rmse_score`, `perceptual_score`.
  - Recent speed-minded change: Lab conversion now uses `float32` arrays and strips to RGB channels before converting.
  - Perceptual fitness is still expected to be significantly slower than RMSE because `rgb2lab` and `deltaE_ciede2000` run for every evaluated individual.

- `scripts/perceptual_comparison.py`
  - Paired comparison between RMSE GA and perceptual GA.
  - Uses paired seeds and cross-evaluates every best solution on both metrics.
  - Uses Wilcoxon signed-rank tests for paired comparison.

- `scripts/new_hill_climbing.py`
  - Local search / hill-climbing polish after GA.

## Notebook Structure

Current `main_nb.ipynb` structure:

1. Imports
2. Target Image Loading and Inspection
3. Solution Representation
4. Baseline GA Run
5. Genetic Operators
6. Statistical Tests
   - Selection
   - Crossover
   - Mutation
   - Elitism
   - Population size
7. Final GA Run
8. Fitness Sharing
9. Hill Climbing for Local Search
10. Challenge 1, Perceptual Fitness (CIEDE2000)
11. Save Best Result

*(A fully working and anchored Table of Contents has been automatically injected into the notebook itself).*

The Genetic Operators section contains clickable links to operator definitions in `.py` files.

## Current Report Draft Structure

`C22_Natural_Selection_report_draft.md` was reorganized based on a previous year's report:

1. Introduction
2. Problem Definition
   - Formal Optimization Problem
   - Solution Representation
   - Search Space
   - RMSE Fitness Function
3. Genetic Operators
4. Performance Analysis
   - Experimental Methodology
   - Operator Selection
   - Hyperparameter Tuning
   - Population Size and Evaluation Budget
5. Final GA Result
6. Challenge 1: Perceptual Fitness
7. Additional Experiments
   - Fitness Sharing
   - Hill Climbing for Local Search
8. Conclusion
9. References
10. Appendix: Reproducibility Notes
11. Appendix: Supplementary Figures and Tables

The report has been populated with the final values from the `main_nb.ipynb` runs. The exact RMSE, corrected paired-test p-values for the perceptual comparison, and perceptual-fitness cross-evaluations are filled in.

## Main Technical Decisions

### Representation

- Individual = ordered sequence of 100 triangles.
- Base triangle has 10 genes:
  - `[x1, y1, x2, y2, x3, y3, R, G, B, A]`.
- Coordinates are in image-pixel units, not normalized.
- Color uses RGBA in `[0, 255]`.
- Triangle order matters due to occlusion.
- Adaptive mutation-rate representation was implemented but is not the current final choice.

### Fitness

- Main required metric: RMSE, minimization.
- Alternative challenge metric: CIEDE2000 perceptual color distance.
- Perceptual metric is a drop-in `TriangleSolution` subclass so the GA loop and operators stay unchanged.
- Cross-evaluation is needed because RMSE and Delta E2000 are on different scales.

### Selection

- Implemented roulette-wheel, rank-based, tournament, and fitness-sharing selection.
- Current final choice: tournament selection, tournament size 10.
- Rationale: easy control of selection pressure, robust in current comparisons.

### Crossover

- Implemented one-point, two-point, uniform, geometric.
- Discrete crossover operators preserve whole-triangle boundaries.
- Current final choice: uniform whole-triangle crossover.
- Rationale: recombines useful triangles across the genotype without splitting triangle genes.
- Geometric crossover is experimental and returns one child.

### Mutation

- Implemented full Gaussian, color-only Gaussian, vertex-only Gaussian, reset mutations, layer-swap, and geometric mutation.
- Current final choice: full Gaussian mutation.
- Current final sigma values in the notebook/report draft:
  - `sigma_vertex = 5`
  - `sigma_colour = 20`
- Current final mutation rate: `0.05`.

### Statistical Methodology

- Avoid drawing conclusions from single stochastic runs.
- Use repeated independent runs, generally 30 runs where feasible.
- Use final best RMSE distributions for statistical comparison.
- Test normality first.
- If both samples are approximately normal, use Welch's t-test.
- Otherwise:
  - independent runs: Mann-Whitney U / Wilcoxon rank-sum
  - paired runs: Wilcoxon signed-rank
- Use `alpha = 0.05`.
- Population-size experiments should be compared by equal or comparable fitness-evaluation budgets, not only by generation count.

### Current Final GA Configuration

From the notebook/report draft:

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

The final long run used mild mutation-rate decay because full 8000-generation replicated tests for every mutation-rate setting were computationally expensive. The smaller statistical tests should be described as tuning evidence, not exact replication of the final long-run budget.

## Remaining Checks

- Ensure the final PDF report includes `best_result.png` and convergence plots matching the filled text values.
- Hill climbing is reported as the post-processing step that produces the submitted `best_result.png`; keep the GA-only and post-polish RMSE values distinct.
- `load_statistical_results()` normalizes older pickle files whose saved result keys were quoted strings.
- After saving statistical results with the new helper, plot labels should usually be passed explicitly.
- The perceptual GA is naturally slower; only low-risk micro-optimizations should be made unless the user asks for deeper algorithmic changes.

## Useful Commands

Compile touched Python files:

```powershell
uv run python -m py_compile scripts\ga.py scripts\statistical_comparison.py scripts\perceptual_fitness.py scripts\perceptual_comparison.py operators\crossover.py operators\mutation.py
```

Notebook module reload pattern:

```python
%load_ext autoreload
%autoreload 2
```

or:

```python
import importlib
import scripts.statistical_comparison as sc
import scripts.ga as ga
importlib.reload(sc)
importlib.reload(ga)
```

## Cautions for Future Agents

- Do not revert user changes.
- This repository may have uncommitted changes and git may complain about dubious ownership.
- Use `apply_patch` for manual edits.
- For notebook edits, validate JSON after patching.
- Keep report and notebook consistent with each other and with the project statement.
- Do not reframe the selected challenge as operator evaluation; the selected challenge is alternative fitness.
