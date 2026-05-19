import numpy as np
from scripts.ga_utils import invalidate_cache

def hill_climbing(
    solution,
    max_iterations=50,
    delta_vertex=10.0,      # Começa com passos reais em pixeis (ex: 10 pixeis)
    delta_color=10.0,       # Começa com passos reais de cor (ex: 10 unidades)
    min_delta_vertex=0.5,   # Para quando o ajuste for menor que meio pixel
    min_delta_color=1.0,    # Para quando o ajuste de cor for irrelevante
    decay_factor=0.7,       # Decaimento ligeiramente mais suave para não prender cedo
    max_evaluations=None,
    verbose=True,
):
    """
    First-improvement hill climbing optimizado para o Vermeer Engine.
    Atua diretamente sobre a representação in-place para máxima eficiência.
    """
    current_fitness = solution.fitness()
    eval_count = 0
    rmse_log = [current_fitness]
    n_genes = len(solution.representation)  # 1000

    # Extrair os limites reais do objeto individual para evitar quebras de encapsulamento
    img_w = getattr(solution, "img_width", 300)
    img_h = getattr(solution, "img_height", 400)

    for iteration in range(max_iterations):
        improved_this_pass = False

        for gene_idx in range(n_genes):
            if max_evaluations is not None and eval_count >= max_evaluations:
                if verbose:
                    print(f"   Orçamento esgotado com {eval_count} avaliações.")
                return solution, rmse_log, eval_count

            # 1. Definir limites dinâmicos com base no gene exato do bloco de 10
            mod_10 = gene_idx % 10
            if mod_10 in [0, 2, 4]:    # Coordenadas X
                delta = delta_vertex
                lo, hi = 0.0, float(img_w)
            elif mod_10 in [1, 3, 5]:  # Coordenadas Y
                delta = delta_vertex
                lo, hi = 0.0, float(img_h)
            else:                      # Cores RGBA (6, 7, 8, 9)
                delta = delta_color
                lo, hi = 0.0, 255.0

            original_value = solution.representation[gene_idx]

            # 2. Tentar as duas direções (+delta e -delta)
            for sign in (+1, -1):
                candidate_value = original_value + sign * delta
                candidate_value = float(np.clip(candidate_value, lo, hi))

                if candidate_value == original_value:
                    continue  # O clipping neutralizou o movimento

                # Alteração cirúrgica DIRECTA in-place no array original
                solution.representation[gene_idx] = candidate_value
                invalidate_cache(solution) # Força o recálculo do RMSE no passo seguinte
                
                new_fitness = solution.fitness()
                eval_count += 1

                if new_fitness < current_fitness:
                    # Sucesso! Mantém a alteração e avança para o próximo gene
                    current_fitness = new_fitness
                    improved_this_pass = True
                    break
                else:
                    # Falhou: Desfaz a alteração in-place sem alocar novos arrays
                    solution.representation[gene_idx] = original_value
                    invalidate_cache(solution)

        rmse_log.append(current_fitness)

        if verbose:
            print(f"   HC pass {iteration + 1:>2d} | RMSE Atual: {current_fitness:.4f} "
                  f"| Evals: {eval_count} | delta_v: {delta_vertex:.2f} "
                  f"| delta_c: {delta_color:.2f}")

        # Se passámos por todos os 1000 genes e nenhum melhorou o RMSE, encolhemos o passo
        if not improved_this_pass:
            delta_vertex *= decay_factor
            delta_color *= decay_factor

            if delta_vertex < min_delta_vertex and delta_color < min_delta_color:
                if verbose:
                    print("   Deltas atingiram o limite mínimo de ajuste fino — Parando.")
                break

    if verbose:
        print(f"   -> Local Search concluído. RMSE Inicial: {rmse_log[0]:.4f} | Final: {current_fitness:.4f}")

    return solution, rmse_log, eval_count