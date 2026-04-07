import os
import sys
import logging
import pandas as pd
import glob
import matplotlib.pyplot as plt
import seaborn as sns

# Adicionar a pasta 'src' ao PATH para permitir importações dos módulos centralizados
sys.path.append(os.path.join(os.getcwd(), 'src'))

# Importação dos módulos de lógica de negócio (Estrutura W4/W5)
from data.preparation import run_full_preparation
from forecasting.trainer import train_and_evaluate_all

# Configuração do LOG para monitorização do estado da execução
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def generate_summary_plot(df, output_path):
    """
    Gera um gráfico de barras profissional comparando o MAE entre modelos e lojas.
    Utilizado para a síntese final do relatório mestre (00_Master_Summary).
    """
    plt.figure(figsize=(15, 8))
    sns.set_theme(style="whitegrid")
    
    try:
        # Criação do gráfico de barras agrupado por Loja e Modelo
        ax = sns.barplot(data=df, x='Store', y='MAE', hue='Model', palette='viridis')
        
        plt.title('Erro Médio Absoluto (MAE) por Modelo e Loja', fontsize=18, fontweight='bold', pad=20)
        plt.xlabel('Localização da Loja', fontsize=14)
        plt.ylabel('MAE ($)', fontsize=14)
        plt.legend(title='Modelo de Previsão', bbox_to_anchor=(1.05, 1), loc='upper left')
        
        # Inserção de rótulos de dados sobre as barras para análise rápida
        for p in ax.patches:
            if p.get_height() > 0:
                ax.annotate(format(p.get_height(), '.0f'), 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha = 'center', va = 'center', 
                            xytext = (0, 9), 
                            textcoords = 'offset points',
                            fontsize=8, rotation=45)

        plt.tight_layout()
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Gráfico resumo gerado com sucesso: {output_path}")
    except Exception as e:
        logger.error(f"Erro ao gerar gráfico resumo: {e}")

def main():
    """
    Função principal que orquestra todo o fluxo de dados: Pré-processamento, Treino e Relatórios.
    """
    print("\n" + "="*60)
    print("USA STORES FORECASTING - PIPELINE MESTRE (EDICAO SENIOR 2026)")
    print("="*60)
    
    # 0. CONFIGURAÇÃO DA ESTRUTURA DE DIRETÓRIOS (Arquitetura de Resultados)
    results_base = 'results'
    subdirs = ['00_Master_Summary', '01_EDA_Gallery', '02_Forecasting_Report']
    for sd in subdirs:
        os.makedirs(os.path.join(results_base, sd), exist_ok=True)

    # 1. LIMPEZA E ENGENHARIA DE ATRIBUTOS (Feature Engineering)
    logger.info("Passo 1: Preparação de Dados e Engenharia de Atributos Profissional")
    run_full_preparation(input_dir='data/raw/', output_dir='data/processed/')
    
    # 2. AVALIAÇÃO DE MODELOS OBRIGATÓRIOS (W4 - Forecasting II)
    logger.info("Passo 2: Comparação de Performance Multi-Modelo (5 Modelos Integrados)")
    processed_files = glob.glob('data/processed/*_processed.csv')
    processed_files = [f for f in processed_files if 'all_stores' not in f]
    
    if not processed_files:
        logger.error("Nenhum ficheiro processado foi encontrado. Interrompendo pipeline.")
        return

    master_results_list = []
    
    # Iteração por cada loja para treino e validação individual
    for f in processed_files:
        # Treino e avaliação, direcionando os outputs para results/02_Forecasting_Report
        res_dict = train_and_evaluate_all(f, output_dir=results_base)
        
        store_name = list(res_dict.keys())[0]
        for row in res_dict[store_name]:
            row['Store'] = store_name.capitalize()
            master_results_list.append(row)
    
    # 3. GERAÇÃO DO RELATÓRIO MESTRE CONSOLIDADO
    logger.info("Passo 3: Geração de Relatórios e Quadros Comparativos Finais")
    master_report_df = pd.DataFrame(master_results_list)
    
    # Filtragem e ordenação para facilitar a tomada de decisão (melhor MAE primeiro)
    cols = ['Store', 'Model', 'MAE', 'RMSE', 'MAPE']
    master_report_df = master_report_df[cols].sort_values(['Store', 'MAE'])
    
    # Exportação do Report Master em CSV
    report_path = os.path.join(results_base, '00_Master_Summary', 'final_performance_report.csv')
    master_report_df.to_csv(report_path, index=False)
    logger.info(f"Relatório de performance consolidado guardado em: {report_path}")
    
    # Geração do Gráfico Resumo Geral
    summary_plot_path = os.path.join(results_base, '00_Master_Summary', 'model_comparison_summary.png')
    generate_summary_plot(master_report_df, summary_plot_path)
    
    # 4. EXIBIÇÃO DO SUMÁRIO DE PERFORMANCE NO TERMINAL
    print("\n" + "="*60)
    print("RELATÓRIO FINAL DE PERFORMANCE (ORDENADO POR MELHOR MAE)")
    print("="*60 + "\n")
    print(master_report_df.to_string(index=False))
    
    # Recomendação Automática do Melhor Modelo por Loja
    print("\n" + "="*60)
    print("MODELOS RECOMENDADOS POR LOJA")
    print("="*60 + "\n")
    best_models = master_report_df.groupby('Store').first().reset_index()
    print(best_models.to_string(index=False))
    print("\n" + "="*60 + "\n")
    
    logger.info("Pipeline Mestre Concluído com Sucesso.")
    logger.info(f"Resultados e Galeria Visual disponíveis em: {os.path.abspath(results_base)}")

if __name__ == "__main__":
    main()
