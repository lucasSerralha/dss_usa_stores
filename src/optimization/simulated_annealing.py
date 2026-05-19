"""
simulated_annealing.py — Otimização por Simulated Annealing para O1

Estratégia:
  Começa numa solução aleatória e itera perturbando-a.
  Aceita soluções piores com probabilidade exp(-(delta)/T), onde T decresce
  geometricamente (arrefecimento). Isto permite escapar de ótimos locais.

Parâmetros principais:
  T_init     : Temperatura inicial (controla exploração inicial)
  T_min      : Temperatura mínima (critério de paragem)
  alpha      : Fator de arrefecimento  T_k+1 = alpha × T_k  (0 < alpha < 1)
  iterations : Número de perturbações por temperatura (inner loop)

Variáveis de decisão (21 por solução, 3 por dia × 7 dias):
  pr    ∈ [0.00, 0.30]   — desconto aplicado
  hr_x  ∈ [0, 15]        — número de peritos (staff expert)
  hr_j  ∈ [0, 15]        — número de juniores (staff junior)
"""

import numpy as np
import random
import math
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(SRC_DIR)

from utils.profit_logic import optimize_weekly_wrapper


def generate_initial_solution():
    """
    Gera uma solução inicial aleatória uniforme dentro dos limites.

    Retorna um array numpy de forma (21,).
    """
    solution = []
    for _ in range(7):
        solution.extend([
            random.uniform(0.0, 0.30),   # desconto pr
            float(random.randint(0, 15)),  # peritos hr_x
            float(random.randint(0, 15)),  # juniores hr_j
        ])
    return np.array(solution, dtype=float)


def generate_neighbor(solution, step_pr: float = 0.05, step_staff: int = 2):
    """
    Gera uma solução vizinha perturbando UMA variável aleatória.

    - Variáveis contínuas (pr):  perturbação gaussiana truncada a ±step_pr
    - Variáveis inteiras (staff): perturbação de ±1 ou ±step_staff
    """
    neighbor = solution.copy()
    index = random.randint(0, len(solution) - 1)

    if index % 3 == 0:
        # Desconto: perturbação suave gaussiana
        delta = random.gauss(0, step_pr / 2)
        neighbor[index] = max(0.0, min(0.30, neighbor[index] + delta))
    else:
        # Staff: salto inteiro
        delta = random.choice([-step_staff, -1, 1, step_staff])
        neighbor[index] = max(0.0, min(15.0, neighbor[index] + delta))

    return neighbor


def evaluate_profit(solution, store, forecast_customers, forecast_is_weekend):
    """Avalia o lucro invertido (f1) de uma solução."""
    f1, _, _ = optimize_weekly_wrapper(solution, store, forecast_customers, forecast_is_weekend)
    return f1  # Negativo do lucro real — minimizar == maximizar lucro


def simulated_annealing(
    store,
    forecast_customers,
    forecast_is_weekend,
    T_init: float = 5000.0,
    T_min: float = 1.0,
    alpha: float = 0.995,
    iter_per_temp: int = 50,
    n_restarts: int = 5,
    seed: int = 42,
):
    """
    Simulated Annealing com múltiplos restarts para maximizar lucro (O1).

    Parâmetros
    ----------
    store              : Nome da loja
    forecast_customers : Previsão de clientes (7 dias)
    forecast_is_weekend: Flags de fim-de-semana (7 dias)
    T_init             : Temperatura inicial
    T_min              : Temperatura mínima (critério de paragem do arrefecimento)
    alpha              : Taxa de arrefecimento geométrico (ex: 0.995)
    iter_per_temp      : Iterações por nível de temperatura
    n_restarts         : Número de restarts independentes
    seed               : Semente para reprodutibilidade

    Retorna
    -------
    best_solution  : array (21,) — melhor vetor de decisão global
    best_score     : float       — lucro invertido do melhor global
    best_history   : list[float] — histórico do melhor restart (por iteração)
    all_histories  : list[list]  — históricos de todos os restarts
    """
    random.seed(seed)
    np.random.seed(seed)

    global_best_solution = None
    global_best_score = float("inf")
    all_histories = []

    for restart in range(n_restarts):
        # --- Estado inicial ---
        current_solution = generate_initial_solution()
        current_score = evaluate_profit(current_solution, store, forecast_customers, forecast_is_weekend)

        best_solution = current_solution.copy()
        best_score = current_score

        T = T_init
        history = [best_score]

        # --- Loop de arrefecimento ---
        while T > T_min:
            for _ in range(iter_per_temp):
                neighbor = generate_neighbor(current_solution)
                neighbor_score = evaluate_profit(neighbor, store, forecast_customers, forecast_is_weekend)

                delta = neighbor_score - current_score  # delta < 0 → melhoria

                # Critério de Metropolis
                if delta < 0:
                    # Aceita sempre se for melhoria
                    current_solution = neighbor
                    current_score = neighbor_score
                else:
                    # Aceita com probabilidade exp(-delta / T)
                    prob = math.exp(-delta / T)
                    if random.random() < prob:
                        current_solution = neighbor
                        current_score = neighbor_score

                # Atualiza o melhor do restart
                if current_score < best_score:
                    best_score = current_score
                    best_solution = current_solution.copy()

            history.append(best_score)
            T *= alpha  # Arrefecimento geométrico

        all_histories.append(history)

        # Atualiza o melhor global entre restarts
        if best_score < global_best_score:
            global_best_score = best_score
            global_best_solution = best_solution.copy()

    # Seleciona o histórico do melhor restart para visualização
    best_restart_idx = int(np.argmin([h[-1] for h in all_histories]))
    best_history = all_histories[best_restart_idx]

    return global_best_solution, global_best_score, best_history, all_histories
