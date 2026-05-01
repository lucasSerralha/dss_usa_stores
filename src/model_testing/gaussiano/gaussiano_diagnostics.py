"""
gaussiano_diagnostics.py — Diagnóstico de Aderência à Distribuição Gaussiana (Normal)
Valida se os dados históricos de vendas (Sales) seguem uma distribuição
Normal e realiza testes de normalidade.
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
    print(f"DIAGNÓSTICO GAUSSIANO: {store_name.upper()}")
    print(f"{'='*60}")
    
    df = pd.read_csv(file_path)
    sales = df['Sales'].dropna()
    
    # 1. Estatísticas Descritivas
    mean_val = sales.mean()
    std_val = sales.std()
    skewness = stats.skew(sales)
    kurtosis = stats.kurtosis(sales)
    
    print(f"Média de Vendas:    ${mean_val:,.2f}")
    print(f"Desvio Padrão:      ${std_val:,.2f}")
    print(f"Coef. Assimetria:   {skewness:.4f} (Ideal ≈ 0)")
    print(f"Curtose:            {kurtosis:.4f} (Ideal ≈ 0)")
    
    # 2. Testes de Normalidade
    # Teste de Shapiro-Wilk (indicado para n < 5000)
    shapiro_stat, shapiro_p = stats.shapiro(sales)
    print(f"Teste Shapiro-Wilk (p-value): {shapiro_p:.4f}")
    
    # Teste de D'Agostino e Pearson
    k2_stat, p_val = stats.normaltest(sales)
    print(f"Teste de Normalidade (p-value): {p_val:.4f}")
    
    if p_val < 0.05:
        print("Resultado: A distribuição de vendas foge da Normal teórica (p < 0.05).")
    else:
        print("Resultado: Não há evidência para rejeitar a hipótese de distribuição Normal.")

    # 3. Visualização
    output_dir = "results/04_Model_Testing/Gaussiano"
    output_plot = os.path.join(output_dir, f"gaussiano_diagnostic_{store_name}.png")
    os.makedirs(output_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Histograma + Curva Normal
    sns.histplot(sales, kde=True, stat='density', label='Observado', color='salmon', ax=ax1)
    x = np.linspace(sales.min(), sales.max(), 100)
    ax1.plot(x, stats.norm.pdf(x, mean_val, std_val), 'r--', lw=2, label='Normal Teórica')
    ax1.set_title(f"Distribuição de Vendas - {store_name.capitalize()}")
    ax1.set_xlabel("Vendas (u.m.)")
    ax1.legend()
    
    # Q-Q Plot
    stats.probplot(sales, dist="norm", plot=ax2)
    ax2.set_title(f"Q-Q Plot - {store_name.capitalize()}")
    
    plt.tight_layout()
    plt.savefig(output_plot)
    plt.close()
    print(f"Gráfico guardado em: {output_plot}")

if __name__ == "__main__":
    raw_files = glob.glob("data/raw/*.csv")
    if not raw_files:
        print("Erro: Nenhum ficheiro encontrado em data/raw/")
    else:
        for f in raw_files:
            run_diagnostics(f)
