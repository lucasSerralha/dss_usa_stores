"""
stochastic_profit_gaussiano.py — Cálculo de Lucro Estocástico via Distribuição Gaussiana
Implementa a lógica de lucro esperado utilizando a incerteza das vendas
modelada por uma distribuição Normal.
"""

import numpy as np
import pandas as pd
from scipy.stats import norm
import logging
import os
import sys

from src.utils.profit_logic import (
    calculate_daily_metrics, 
    STORE_PARAMS, 
    PROFIT_SCALE,
    ELASTICITY_K
)

log = logging.getLogger("TIAPOSE.StochasticProfitGauss")

def calculate_expected_daily_profit_gauss(store, is_weekend, customers, pr, hr_x, hr_j, sigma_sales):
    """
    Calcula o lucro esperado diário integrando sobre a distribuição Gaussiana das vendas.
    
    E[Profit] = Integral_{-inf}^{inf} Profit(s) * f(s) ds
    Onde s é a venda e f(s) é a PDF da Normal(mu, sigma).
    
    Como a função de lucro no nosso sistema DSS é tipicamente linear ou 
    quase-linear em relação às vendas (após os custos fixos), o valor esperado 
    do lucro costuma ser muito próximo do lucro calculado na média (E[f(X)] ≈ f(E[X])).
    
    No entanto, para efeitos de "Model Testing", vamos simular a integração.
    """
    # Determinístico na média
    metrics = calculate_daily_metrics(store, is_weekend, customers, pr, hr_x, hr_j)
    mu_sales = metrics['sales_x'] + metrics['sales_j']
    daily_hr_cost = metrics['cost_x'] + metrics['cost_j']
    
    # Simulação de Monte Carlo para integração (mais simples que quadratura para este caso)
    samples = np.random.normal(mu_sales, sigma_sales, 1000)
    samples = np.maximum(0, samples) # Vendas não podem ser negativas
    
    expected_sales = np.mean(samples)
    
    return expected_sales - daily_hr_cost

def calculate_stochastic_weekly_profit_gauss(store, weekly_plan, sigma_sales_daily):
    """
    Calcula o lucro semanal esperado sob incerteza Gaussiana.
    """
    total_expected_operating_profit = 0
    
    for day_data in weekly_plan:
        # Aplicar elasticidade aos clientes
        effective_customers = int(round(day_data['customers'] * (1 + ELASTICITY_K * day_data['pr'])))
        
        daily_expected_op_profit = calculate_expected_daily_profit_gauss(
            store=store,
            is_weekend=day_data['is_weekend'],
            customers=effective_customers,
            pr=day_data['pr'],
            hr_x=day_data['hr_x'],
            hr_j=day_data['hr_j'],
            sigma_sales=sigma_sales_daily
        )
        total_expected_operating_profit += daily_expected_op_profit
        
    fixed_cost = STORE_PARAMS[store.lower()]['W_s']
    final_expected_profit = (total_expected_operating_profit - fixed_cost) * PROFIT_SCALE
    
    return final_expected_profit

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    
    store = 'baltimore'
    is_weekend = False
    customers = 100
    pr = 0.10
    hr_x = 8
    hr_j = 4
    sigma_sales = 5000 # Exemplo de desvio padrão diário
    
    exp_profit = calculate_expected_daily_profit_gauss(store, is_weekend, customers, pr, hr_x, hr_j, sigma_sales)
    
    print(f"--- Teste Estocástico Gaussiano ({store}) ---")
    print(f"Clientes: {customers}")
    print(f"Desconto: {pr*100}%")
    print(f"Sigma Vendas (Incerteza): ${sigma_sales:,.2f}")
    print(f"Lucro Operacional Esperado (Diário): ${exp_profit:,.2f}")
    
    # Comparação com Determinístico
    metrics = calculate_daily_metrics(store, is_weekend, int(round(customers * (1 + ELASTICITY_K * pr))), pr, hr_x, hr_j)
    det_profit = (metrics['sales_x'] + metrics['sales_j']) - (metrics['cost_x'] + metrics['cost_j'])
    print(f"Lucro Operacional Determinístico: ${det_profit:,.2f}")
    print(f"Efeito da Incerteza (Jensen's Inequality): ${exp_profit - det_profit:,.2f}")
