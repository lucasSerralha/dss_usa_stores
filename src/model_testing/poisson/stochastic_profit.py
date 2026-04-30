"""
stochastic_profit.py — Cálculo de Lucro Estocástico via Distribuição de Poisson
Este módulo implementa a lógica de lucro esperado, integrando a incerteza
na chegada de clientes através de uma distribuição probabilística.
"""

import numpy as np
import pandas as pd
from scipy.stats import poisson
import logging
import os
import sys

# Adicionar o root ao sys.path para importar módulos locais se necessário
# SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "../../../"))
# if PROJECT_ROOT not in sys.path:
#     sys.path.append(PROJECT_ROOT)

from src.utils.profit_logic import (
    calculate_daily_metrics, 
    STORE_PARAMS, 
    PROFIT_SCALE,
    ELASTICITY_K
)

log = logging.getLogger("TIAPOSE.StochasticProfit")

def calculate_expected_daily_profit(store, is_weekend, lambda_param, pr, hr_x, hr_j):
    """
    Calcula o lucro esperado diário integrando sobre a distribuição de Poisson.
    
    E[Profit] = sum_{k=0}^{inf} Profit(k) * P(X=k)
    
    Onde X ~ Poisson(lambda_param).
    """
    # Truncagem da soma: até onde a probabilidade acumulada é 99.9%
    # ou pelo menos um valor razoável para evitar loops infinitos
    max_k = int(poisson.ppf(0.999, lambda_param)) if lambda_param > 0 else 0
    
    expected_sales = 0
    expected_costs = 0 # Na verdade custos de RH são fixos para o dia dado hr_x, hr_j
    
    # Probabilidades de cada k
    k_values = np.arange(0, max_k + 1)
    probs = poisson.pmf(k_values, lambda_param)
    
    # Custos de RH são independentes do número de clientes (decisão a priori)
    # Mas calculate_daily_metrics retorna-os, vamos calcular uma vez
    sample_metrics = calculate_daily_metrics(store, is_weekend, 0, pr, hr_x, hr_j)
    daily_hr_cost = sample_metrics['cost_x'] + sample_metrics['cost_j']
    
    for k, p in zip(k_values, probs):
        metrics = calculate_daily_metrics(store, is_weekend, k, pr, hr_x, hr_j)
        daily_sales = metrics['sales_x'] + metrics['sales_j']
        expected_sales += daily_sales * p
        
    return expected_sales - daily_hr_cost

def calculate_stochastic_weekly_profit(store, weekly_plan_with_lambda):
    """
    Calcula o lucro semanal esperado.
    weekly_plan_with_lambda: lista de dicionários contendo 'lambda_param' em vez de 'customers'
    """
    total_expected_operating_profit = 0
    
    for day_data in weekly_plan_with_lambda:
        # Aplicar elasticidade ao lambda (a taxa média de chegada aumenta com o desconto)
        effective_lambda = day_data['lambda_param'] * (1 + ELASTICITY_K * day_data['pr'])
        
        daily_expected_op_profit = calculate_expected_daily_profit(
            store=store,
            is_weekend=day_data['is_weekend'],
            lambda_param=effective_lambda,
            pr=day_data['pr'],
            hr_x=day_data['hr_x'],
            hr_j=day_data['hr_j']
        )
        total_expected_operating_profit += daily_expected_op_profit
        
    fixed_cost = STORE_PARAMS[store.lower()]['W_s']
    final_expected_profit = (total_expected_operating_profit - fixed_cost) * PROFIT_SCALE
    
    return final_expected_profit

def risk_of_understaffing(lambda_param, hr_x, hr_j):
    """
    Calcula a probabilidade de haver mais clientes do que a capacidade de atendimento.
    Capacidade X = hr_x * 7
    Capacidade J = hr_j * 6
    """
    capacity = (hr_x * 7) + (hr_j * 6)
    # P(X > capacity) = 1 - P(X <= capacity)
    prob_under = 1 - poisson.cdf(capacity, lambda_param)
    return prob_under

if __name__ == "__main__":
    # Teste Simples
    logging.basicConfig(level=logging.INFO)
    
    store = 'baltimore'
    is_weekend = False
    base_lambda = 80.0
    pr = 0.10
    hr_x = 6
    hr_j = 4
    
    # Teste de elasticidade no lambda
    eff_lambda = base_lambda * (1 + ELASTICITY_K * pr)
    
    exp_profit = calculate_expected_daily_profit(store, is_weekend, eff_lambda, pr, hr_x, hr_j)
    risk = risk_of_understaffing(eff_lambda, hr_x, hr_j)
    
    print(f"--- Teste Estocástico ({store}) ---")
    print(f"Lambda Base: {base_lambda}")
    print(f"Lambda com Desconto ({pr*100}%): {eff_lambda:.2f}")
    print(f"Staff: X={hr_x}, J={hr_j} (Capacidade: {(hr_x*7)+(hr_j*6)})")
    print(f"Lucro Operacional Esperado (Diário): ${exp_profit:.2f}")
    print(f"Risco de Understaffing: {risk*100:.2f}%")
    
    # Comparação com Determinístico
    from src.utils.profit_logic import calculate_daily_metrics
    det_metrics = calculate_daily_metrics(store, is_weekend, int(round(eff_lambda)), pr, hr_x, hr_j)
    det_profit = (det_metrics['sales_x'] + det_metrics['sales_j']) - (det_metrics['cost_x'] + det_metrics['cost_j'])
    print(f"Lucro Operacional Determinístico (no ponto médio): ${det_profit:.2f}")
    print(f"Diferença (Incerteza): ${exp_profit - det_profit:.2f}")
