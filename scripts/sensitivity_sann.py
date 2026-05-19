"""
sensitivity_sann.py — Estudo de sensibilidade de hiperparâmetros (Simulated Annealing)

Este script avalia o impacto da Temperatura Inicial (T_init) e da Taxa de Arrefecimento (alpha)
na convergência e no lucro final obtido pelo algoritmo de Simulated Annealing nas 4 lojas.
Gera um relatório CSV e um heatmap (gráfico) consolidados.
"""

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from time import time

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(ROOT_DIR)

from src.optimization.simulated_annealing import simulated_annealing

def get_latest_forecast(store: str):
    """Lê os últimos 7 dias das previsões do baseline (assumido disponível)."""
    # Para o propósito deste estudo, podemos usar as últimas 7 linhas do dataset real como se fossem previsões,
    # se o CSV de previsões não estiver facilmente disponível.
    # Mas o correto é usar a ground truth ou as previsões criadas
    df = pd.read_csv(os.path.join(ROOT_DIR, "data", "raw", f"{store.lower()}.csv"))
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    last_week = df.tail(7).copy()
    
    forecast_customers = last_week['Num_Customers'].values
    
    # Derivar is_weekend a partir do Date
    is_weekend = (last_week['Date'].dt.dayofweek >= 5).astype(int).values
    
    return forecast_customers, is_weekend

def run_sensitivity():
    stores = ["Baltimore", "Lancaster", "Philadelphia", "Richmond"]
    t_inits = [1000, 5000, 10000]
    alphas = [0.85, 0.90, 0.95, 0.99]
    n_restarts = 3 # Manter pequeno para correr num tempo razoável
    
    out_dir = os.path.join(ROOT_DIR, "results", "05_Sensitivity_Analysis")
    os.makedirs(out_dir, exist_ok=True)
    
    results = []
    
    print("Iniciando estudo de sensibilidade (SANN)...")
    
    for store in stores:
        print(f"\\n--- Loja: {store} ---")
        forecast_customers, is_weekend = get_latest_forecast(store)
        
        for t_init in t_inits:
            for alpha in alphas:
                print(f"  A testar T_init={t_init}, alpha={alpha}...", end=" ")
                start_time = time()
                
                _, best_score, _, _ = simulated_annealing(
                    store=store,
                    forecast_customers=forecast_customers,
                    forecast_is_weekend=is_weekend,
                    T_init=t_init,
                    T_min=1.0,
                    alpha=alpha,
                    iter_per_temp=30, # ligeiramente reduzido para acelerar o grid search
                    n_restarts=n_restarts,
                    seed=42
                )
                
                exec_time = time() - start_time
                profit = -best_score # best_score é o negativo do lucro
                
                print(f"Profit={profit:.2f} (Time={exec_time:.2f}s)")
                
                results.append({
                    "Store": store,
                    "T_init": t_init,
                    "alpha": alpha,
                    "Profit": profit,
                    "Exec_Time_sec": round(exec_time, 2)
                })
                
    # Salvar resultados
    df_res = pd.DataFrame(results)
    csv_path = os.path.join(out_dir, "sann_sensitivity.csv")
    df_res.to_csv(csv_path, index=False)
    print(f"\\nResultados gravados em {csv_path}")
    
    # Gerar gráficos (heatmap) por loja
    sns.set_theme(style="whitegrid")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Sensibilidade SANN: Profit vs Hiperparâmetros", fontsize=16, fontweight='bold')
    
    for i, store in enumerate(stores):
        ax = axes[i//2, i%2]
        df_store = df_res[df_res["Store"] == store]
        pivot = df_store.pivot(index="T_init", columns="alpha", values="Profit")
        
        sns.heatmap(pivot, annot=True, fmt=".0f", cmap="YlGnBu", ax=ax, cbar=True)
        ax.set_title(f"Loja: {store}")
        ax.set_ylabel("T_init")
        ax.set_xlabel("alpha (cooling rate)")
        
    plt.tight_layout()
    img_path = os.path.join(out_dir, "sann_sensitivity_heatmap.png")
    plt.savefig(img_path, dpi=150)
    print(f"Heatmaps gravados em {img_path}")

if __name__ == "__main__":
    run_sensitivity()
