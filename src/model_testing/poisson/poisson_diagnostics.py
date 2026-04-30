"""
poisson_diagnostics.py — Diagnóstico de Aderência à Distribuição de Poisson
Valida se os dados históricos de chegada de clientes seguem uma distribuição
de Poisson (Média ≈ Variância) e realiza testes de aderência.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import os
import glob

def run_diagnostics(file_path):
    store_name = os.path.basename(file_path).replace('.csv', '')
    print(f"\n{'='*60}")
    print(f"DIAGNÓSTICO: {store_name.upper()}")
    print(f"{'='*60}")
    
    df = pd.read_csv(file_path)
    customers = df['Num_Customers'].dropna()
    
    # 1. Estatísticas Descritivas
    mean_val = customers.mean()
    var_val = customers.var()
    ratio = var_val / mean_val
    
    print(f"Média de Clientes: {mean_val:.2f}")
    print(f"Variância:         {var_val:.2f}")
    print(f"Razão Var/Média:   {ratio:.2f} (Ideal para Poisson ≈ 1.0)")
    
    if ratio > 1.2:
        print("Aviso: Detetada Sobredispersão (Overdispersion).")
    elif ratio < 0.8:
        print("Aviso: Detetada Subdispersão.")
    else:
        print("Info: A dispersão está alinhada com uma distribuição de Poisson.")
        
    # 2. Teste de Aderência (Qui-Quadrado simplificado)
    # Comparamos as frequências observadas com as esperadas de uma Poisson(mean_val)
    observed, bins = np.histogram(customers, bins='auto')
    bin_centers = (bins[:-1] + bins[1:]) / 2
    expected = stats.poisson.pmf(bin_centers.astype(int), mean_val) * len(customers)
    
    # Normalizar esperado para que a soma seja igual ao observado (para o teste)
    expected = expected * (observed.sum() / expected.sum())
    
    # Teste de Kolmogorov-Smirnov (mais robusto para contínuos/grandes amostras)
    ks_stat, p_val = stats.kstest(customers, 'poisson', args=(mean_val,))
    print(f"Teste KS (p-value): {p_val:.4f}")
    
    if p_val < 0.05:
        print("Resultado: A distribuição observada difere significativamente de uma Poisson teórica.")
    else:
        print("Resultado: Não há evidência para rejeitar a hipótese de distribuição de Poisson.")

    # 3. Visualização (Opcional - salva se o diretório existir)
    output_dir = "results/04_Model_Testing/Poisson"
    output_plot = os.path.join(output_dir, f"poisson_diagnostic_{store_name}.png")
    os.makedirs(output_dir, exist_ok=True)
    
    plt.figure(figsize=(10, 6))
    sns.histplot(customers, kde=False, stat='density', label='Observado', color='skyblue')
    
    # Linha Poisson Teórica
    x = np.arange(customers.min(), customers.max())
    plt.plot(x, stats.poisson.pmf(x.astype(int), mean_val), 'r-', lw=2, label=fr'Poisson teórica ($\lambda$={mean_val:.1f})')
    
    plt.title(f"Distribuição de Chegada de Clientes - {store_name.capitalize()}")
    plt.xlabel("Número de Clientes")
    plt.ylabel("Densidade")
    plt.legend()
    plt.grid(alpha=0.3)
    
    plt.savefig(output_plot)
    print(f"Gráfico guardado em: {output_plot}")

if __name__ == "__main__":
    raw_files = glob.glob("data/raw/*.csv")
    if not raw_files:
        print("Erro: Nenhum ficheiro encontrado em data/raw/")
    else:
        for f in raw_files:
            run_diagnostics(f)
