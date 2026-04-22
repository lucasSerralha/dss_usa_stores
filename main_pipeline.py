import os
import sys
import logging
import pandas as pd

# Bloqueio de criação de pastas __pycache__ (Manter repositório minimalista)
sys.dont_write_bytecode = True
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# Importação dos módulos de lógica de negócio
from src.data.preparation import run_full_preparation
from src.forecasting.trainer import train_and_evaluate_all

# Configuração do LOG para monitorização do estado da execução
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_summary_plot(df, output_path):
    """
    Gera um gráfico comparativo para mostrar qual a experiência mais fidedigna.
    Compara o RMSE entre diferentes conjuntos de variáveis.
    """
    plt.figure(figsize=(15, 8))
    sns.set_theme(style="whitegrid")
    
    try:
        # Focamos no RMSE pois é a métrica que melhor pune grandes desvios (Fidedignidade)
        ax = sns.barplot(data=df, x='Store', y='RMSE', hue='Experiment', palette='magma')
        
        plt.title('Comparação de Erro (RMSE) por Conjunto de Variáveis', fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Localização da Loja', fontsize=14)
        plt.ylabel('RMSE ($) - Menor é Melhor', fontsize=14)
        plt.legend(title='Cenário de Variáveis', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Gráfico de experiência gerado com sucesso: {output_path}")
    except Exception as e:
        logger.error(f"Erro ao gerar gráfico de experiência: {e}")

def main():
    """
    Função principal que orquestra todo o fluxo de dados: Pré-processamento, Treino e Relatórios.
    """
    print("\n" + "="*60)
    print("USA STORES FORECASTING - PIPELINE PRINCIPAL (EDIÇÃO SENIOR 2026)")
    print("="*60)
    
    # 0. CONFIGURAÇÃO DA ESTRUTURA DE DIRETÓRIOS (Arquitetura de Resultados)
    results_base = 'results'
    subdirs = ['00_Master_Summary', '01_EDA_Gallery', '02_Forecasting_Report']
    for sd in subdirs:
        os.makedirs(os.path.join(results_base, sd), exist_ok=True)

    # 1. LIMPEZA E ENGENHARIA DE ATRIBUTOS (Feature Engineering)
    logger.info("Passo 1: Preparação de Dados e Engenharia de Atributos Profissional")
    run_full_preparation(input_dir='data/raw/', output_dir='data/processed/')
    
    # --- FASE 2: BATERIA DE EXPERIMENTAÇÃO (W4.5 - Fidedignidade) ---
    logger.info("Passo 2: Execução de Experiências com Diferentes Conjuntos de Variáveis")
    processed_files = glob.glob('data/processed/*_processed.csv')
    processed_files = [f for f in processed_files if 'all_stores' not in f]
    
    if not processed_files:
        logger.error("Nenhum ficheiro processado foi encontrado. Interrompendo pipeline.")
        return

    # Definição dos conjuntos de variáveis (Apostas Técnicas do António e do Professor)
    # NOTA: 'Num_Employees' está banido de todas as experiências por questões de causalidade.
    feature_sets = {
        "A_Temporal_Base": ['day_of_week', 'is_weekend', 'month', 'season_num', 'sales_lag_7', 'sales_lag_28'],
        "B_Sales_Dynamics": ['day_of_week', 'month', 'sales_lag_1', 'sales_lag_2', 'sales_lag_3', 'sales_lag_4', 'sales_lag_5', 'sales_lag_6', 'sales_lag_7', 'sales_roll_mean_7', 'sales_roll_std_7'],
        "C_Context_Expert": ['Num_Customers', 'Pct_On_Sale', 'TouristEvent', 'is_holiday', 'days_to_next_holiday', 'day_of_week', 'sales_lag_1', 'sales_lag_2', 'sales_lag_3', 'sales_lag_4', 'sales_lag_5', 'sales_lag_6', 'sales_lag_7', 'sales_roll_mean_7']
    }

    master_results_list = []
    
    # Loop de experiências por loja e por conjunto de variáveis
    for f in processed_files:
        for set_name, features_list in feature_sets.items():
            res_dict = train_and_evaluate_all(f, output_dir=results_base, custom_features=features_list, experiment_name=set_name)
            
            store_name = list(res_dict.keys())[0]
            for row in res_dict[store_name]:
                row['Store'] = store_name.capitalize()
                row['Experiment'] = set_name
                master_results_list.append(row)
    
    # --- FASE 3: RELATÓRIO MESTRE E RANKING DE FIDEDIGNIDADE ---
    logger.info("Passo 3: Consolidação de Resultados e Identificação dos Melhores 'Setups'")
    master_report_df = pd.DataFrame(master_results_list)
    
    # Ordenamos por RMSE para identificar o setup mais fidedigno
    cols = ['Store', 'Experiment', 'Model', 'MAE', 'RMSE', 'MAPE']
    master_report_df = master_report_df[cols].sort_values(['Store', 'RMSE'])
    
    # Exportação do Report Master
    report_path = os.path.join(results_base, '00_Master_Summary', 'fidelity_experimentation_report.csv')
    master_report_df.to_csv(report_path, index=False)
    logger.info(f"Relatório de experiências guardado em: {report_path}")
    
    # Geração do Gráfico Resumo Geral
    summary_plot_path = os.path.join(results_base, '00_Master_Summary', 'model_comparison_summary.png')
    generate_summary_plot(master_report_df, summary_plot_path)
    
    # Recomendação Automática do Melhor Modelo por Loja
    print("\n" + "="*80)
    print("RANKING DE FIDEDIGNIDADE (MELHOR CONFIGURAÇÃO POR LOJA)")
    print("="*80 + "\n")
    best_setups = master_report_df.groupby('Store').first().reset_index()
    print(best_setups.to_string(index=False))
    print("\n" + "="*80 + "\n")
    
    logger.info("Pipeline Mestre Concluído com Sucesso.")
    logger.info(f"Resultados e Galeria Visual disponíveis em: {os.path.abspath(results_base)}")

if __name__ == "__main__":
    main()
