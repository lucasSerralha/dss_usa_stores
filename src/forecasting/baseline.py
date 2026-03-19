import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error

def run_baseline_model(input_path, results_dir):
    print("A iniciar a avaliação do Modelo Baseline (Seasonal Naive)...")
    
    if not os.path.exists(input_path):
        print(f"Erro: Ficheiro não encontrado em {input_path}")
        return

    # 1. Garantir que as pastas de resultados existem
    os.makedirs(results_dir, exist_ok=True)
    
    # 2. Carregar os dados enriquecidos (A nossa "Fonte Única da Verdade")
    df = pd.read_csv(input_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Remover linhas vazias geradas pelos primeiros 7 dias (onde não há histórico para o Lag)
    df_clean = df.dropna(subset=['Sales_Lag7', 'Sales'])
    
    metrics_list = []
    
    print("\nResultados do Erro de Previsão (Lag 7 Dias):")
    print("-" * 50)
    
    # 3. Calcular o erro (RMSE e MAE) para cada loja individualmente
    for store in df_clean['Store'].unique():
        store_data = df_clean[df_clean['Store'] == store]
        
        y_true = store_data['Sales']
        y_pred = store_data['Sales_Lag7'] # A nossa "previsão" Naive
        
        mae = mean_absolute_error(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        
        metrics_list.append({
            'Store': store,
            'Model': 'Seasonal Naive',
            'MAE': round(mae, 2),
            'RMSE': round(rmse, 2)
        })
        
        print(f"Loja: {store:<12} | MAE: ${mae:<8.2f} | RMSE: ${rmse:.2f}")
        
    # 4. Guardar as métricas em formato CSV na pasta de relatórios
    metrics_df = pd.DataFrame(metrics_list)
    metrics_file = os.path.join(results_dir, "baseline_metrics.csv")
    metrics_df.to_csv(metrics_file, index=False)
    print("-" * 50)
    print(f"Métricas oficiais guardadas em: {metrics_file}")
    
   # 5. GERAR GRÁFICOS PARA TODAS AS LOJAS
    print("\nA gerar gráficos comparativos para todas as lojas (últimos 30 dias)...")
    
    for store in df_clean['Store'].unique():
        store_data = df_clean[df_clean['Store'] == store].tail(30)
        
        plt.figure(figsize=(12, 6))
        plt.plot(store_data['Date'], store_data['Sales'], label='Vendas Reais', marker='o', color='#1f77b4')
        plt.plot(store_data['Date'], store_data['Sales_Lag7'], label='Previsão (Baseline)', linestyle='--', marker='x', color='#ff7f0e')
        
        plt.title(f'Baseline (Seasonal Naive) - {store}: Vendas Reais vs Previsão')
        plt.xlabel('Data')
        plt.ylabel('Vendas ($)')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        plot_file = os.path.join(results_dir, f"{store.lower()}_baseline_plot.png")
        plt.savefig(plot_file, bbox_inches='tight')
        plt.close()
        print(f"Gráfico gerado: {plot_file}")
    
if __name__ == "__main__":
    # Navegação absoluta de caminhos (A partir de src/forecasting/)
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_FILE = os.path.abspath(os.path.join(SCRIPT_DIR, "../../data/processed/features_stores_merged.csv"))
    
    # Apontar para a nossa pasta de resultados na raiz do projeto
    RESULTS_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "../../results/forecasting"))
    
    run_baseline_model(INPUT_FILE, RESULTS_DIR)