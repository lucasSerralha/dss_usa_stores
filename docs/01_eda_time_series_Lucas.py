import pandas as pd
import os
import matplotlib.pyplot as plt
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller

def run_eda(file_path):
    print("A iniciar Análise Exploratória de Séries Temporais...\n")
    df = pd.read_csv(file_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Vamos focar na loja de maior volume (Lancaster) como amostra
    loja = 'Lancaster'
    df_loja = df[df['Store'] == loja].sort_values('Date').set_index('Date')
    vendas = df_loja['Sales'].dropna()

    # 1. Teste de Dickey-Fuller Aumentado (Estacionaridade)
    # Se p-value < 0.05, a série é estacionária (não tem tendência forte de longo prazo)
    print(f"--- Teste de Estacionaridade (ADF) para {loja} ---")
    result = adfuller(vendas)
    print(f'Estatística ADF: {result[0]:.4f}')
    print(f'p-value: {result[1]:.4f}')
    if result[1] <= 0.05:
        print("-> Conclusão: A série é ESTACIONÁRIA. Excelente para modelos preditivos.\n")
    else:
        print("-> Conclusão: A série NÃO É ESTACIONÁRIA. Precisará de diferenciação (d=1 no ARIMA).\n")

    # 2. Decomposição Sazonal (Tendência, Sazonalidade e Ruído)
    # Period=7 porque os dados são diários e a sazonalidade óbvia é semanal
    print("A gerar gráficos de decomposição sazonal...")
    decomposicao = seasonal_decompose(vendas, model='additive', period=7)
    
    fig = decomposicao.plot()
    fig.set_size_inches(12, 8)
    fig.suptitle(f'Decomposição Sazonal de Vendas - {loja}', fontsize=14)
    plt.tight_layout()
    
    # Guardar o gráfico
    output_dir = os.path.join(os.path.dirname(file_path), "../../results/")
    os.makedirs(output_dir, exist_ok=True)
    plot_path = os.path.join(output_dir, f"eda_decomposicao_{loja.lower()}.png")
    plt.savefig(plot_path)
    print(f"Gráfico guardado em: {plot_path}")

if __name__ == "__main__":
    # Apontar para os dados já processados (a Fonte Única da Verdade)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_FILE = os.path.abspath(os.path.join(SCRIPT_DIR, "../data/processed/features_stores_merged.csv"))
    
    # Executa a função
    try:
        run_eda(INPUT_FILE)
    except ModuleNotFoundError:
        print("Por favor, instala a biblioteca statsmodels executando: pip install statsmodels")