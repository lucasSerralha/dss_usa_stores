import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import os

def generate_acf_plots():
    print("A carregar dados para análise estatística...")
    # Caminho absoluto
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    data_path = os.path.join(base_dir, 'data', 'processed', 'baltimore_processed.csv')
    
    if not os.path.exists(data_path):
        print(f"Erro: Ficheiro não encontrado em {data_path}")
        return

    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')
    
    # Criar pasta de resultados se não existir
    out_dir = os.path.join(base_dir, 'results', '01_EDA_Gallery')
    os.makedirs(out_dir, exist_ok=True)

    # Plot ACF (Autocorrelação) - Mostra picos a cada 7 dias
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_acf(df['Sales'].dropna(), lags=21, ax=ax, title="Autocorrelação das Vendas (Baltimore) - Picos em 7, 14, 21")
    plt.savefig(os.path.join(out_dir, 'acf_plot.png'))
    plt.close()

    # Plot PACF (Autocorrelação Parcial)
    fig, ax = plt.subplots(figsize=(10, 5))
    plot_pacf(df['Sales'].dropna(), lags=21, ax=ax, title="Autocorrelação Parcial (Baltimore)")
    plt.savefig(os.path.join(out_dir, 'pacf_plot.png'))
    plt.close()

    print(f"✅ SUCESSO: Gráficos ACF e PACF guardados em {out_dir}")

if __name__ == "__main__":
    generate_acf_plots()