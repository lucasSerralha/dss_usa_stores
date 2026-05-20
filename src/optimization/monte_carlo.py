"""
monte_carlo.py — Otimização por Monte Carlo para O1 (Maximizar Lucro por Loja)

Estratégia:
  Gera N soluções completamente aleatórias dentro do espaço de decisão,
  avalia o lucro de cada uma e retém a melhor.
  Serve como baseline estocástico e para análise da distribuição do espaço.

Variáveis de decisão (21 por solução, 3 por dia × 7 dias):
  pr    ∈ [0.00, 0.30]   — desconto aplicado
  hr_x  ∈ [0, 15]        — número de peritos (staff expert)
  hr_j  ∈ [0, 15]        — número de juniores (staff junior)
"""

import numpy as np
import random
import sys
import os

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(SRC_DIR)

from utils.profit_logic import optimize_weekly_wrapper


def generate_random_solution(rng: np.random.Generator | None = None):
    """
    Gera uma solução aleatória uniforme dentro dos limites do espaço de decisão.

    Retorna um array numpy de forma (21,):
      [pr_0, hr_x_0, hr_j_0, pr_1, hr_x_1, hr_j_1, ..., pr_6, hr_x_6, hr_j_6]
    """
    solution = []
    for _ in range(7):
        pr = random.uniform(0.0, 0.30)
        hr_x = random.randint(0, 15)
        hr_j = random.randint(0, 15)
        solution.extend([pr, hr_x, hr_j])
    return np.array(solution, dtype=float)


def evaluate_profit(solution, store, forecast_customers, forecast_is_weekend):
    """Avalia o lucro invertido (f1) de uma solução."""
    f1, _, _ = optimize_weekly_wrapper(solution, store, forecast_customers, forecast_is_weekend)
    return f1  # Negativo do lucro real — minimizar == maximizar lucro


def monte_carlo(
    store,
    forecast_customers,
    forecast_is_weekend,
    n_samples: int = 20_000,
    seed: int = 42,
):
    """
    Otimização por Monte Carlo puro.

    Parâmetros
    ----------
    store              : Nome da loja ('baltimore', 'lancaster', etc.)
    forecast_customers : Lista com 7 previsões diárias de clientes
    forecast_is_weekend: Lista de 7 booleanos (True = fim-de-semana)
    n_samples          : Número de soluções aleatórias a amostrar
    seed               : Semente para reprodutibilidade

    Retorna
    -------
    best_solution : array (21,) — melhor vetor de decisão encontrado
    best_score    : float       — lucro invertido da melhor solução (negativo)
    all_scores    : list[float] — lucro invertido de todas as amostras
    history       : list[float] — evolução do melhor lucro invertido ao longo das amostras
    """
    random.seed(seed)
    np.random.seed(seed)

    best_solution = None
    best_score = float("inf")  # Minimizar (lucro invertido)
    all_scores = []
    history = []  # Melhor acumulado a cada amostra

    for i in range(n_samples):
        solution = generate_random_solution()
        score = evaluate_profit(solution, store, forecast_customers, forecast_is_weekend)
        all_scores.append(score)

        if score < best_score:
            best_score = score
            best_solution = solution.copy()

        history.append(best_score)

    return best_solution, best_score, all_scores, history
