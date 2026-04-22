import pandas as pd
import numpy as np
import holidays
import os
import glob
import logging

# Configuração do LOG profissional para rastreamento de pipeline
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Inicialização de feriados dos EUA (contexto do dataset)
us_holidays = holidays.US()

def get_season(date):
    """Retorna a estação do ano baseada no mês da data fornecida."""
    month = date.month
    if month in [12, 1, 2]: return 'Winter'
    elif month in [3, 4, 5]: return 'Spring'
    elif month in [6, 7, 8]: return 'Summer'
    else: return 'Autumn'

def prepare_store_data(file_path, output_dir='data/processed/'):
    """
    Executa a preparação completa de dados para um ficheiro de loja individual.
    Inclui limpeza técnica, tratamento de anomalias e engenharia de atributos (Feature Engineering).
    """
    store_name = os.path.basename(file_path).replace('.csv', '')
    logger.info(f"A preparar dados para a loja: {store_name}")
    
    # 1. Carregamento e Ordenação Temporal
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # 2. Tratamento de Valores em Falta (Interpolação Linear)
    # Pct_On_Sale: Variável crítica para impacto promocional
    df['Pct_On_Sale'] = df['Pct_On_Sale'].interpolate(method='linear').fillna(method='bfill')
    
    # 3. Tratamento de Anomalias (Dados Sujos)
    # Caso: Clientes = 0 mas Vendas > 0 (Erro de integridade de dados)
    mask_dirty = (df['Num_Customers'] == 0) & (df['Sales'] > 0)
    if mask_dirty.any():
        logger.warning(f"  [{store_name}] Corrigidos {mask_dirty.sum()} pontos de dados inconsistentes (Vendas sem Clientes)")
        df.loc[mask_dirty, 'Num_Customers'] = np.nan
        df['Num_Customers'] = df['Num_Customers'].interpolate(method='linear')
    
    # 4. Tratamento de Outliers (Capping 1% - 99%)
    # Garante que valores extremos não causem distorções no aprendizado do modelo
    for col in ['Num_Customers', 'Num_Employees', 'Sales']:
        if col in df.columns:
            q_low = df[col].quantile(0.01)
            q_high = df[col].quantile(0.99)
            df[col] = df[col].clip(q_low, q_high)
            
    # 5. Codificação de Variáveis Categóricas
    if df['TouristEvent'].dtype == 'object':
        df['TouristEvent'] = df['TouristEvent'].map({'No': 0, 'Yes': 1}).fillna(0)
    
    # 6. Feature Engineering - Atributos Temporais
    df['is_holiday'] = df['Date'].apply(lambda x: 1 if x in us_holidays else 0)
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['IsWeekend'] = (df['day_of_week'] >= 5).astype(int)
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['season_num'] = df['Date'].apply(get_season).map({'Winter': 0, 'Spring': 1, 'Summer': 2, 'Autumn': 3})
    
    # 7. Atributos de Forecasting - Atrasos Temporais (Lags de 7, 14, 21, 28 dias)
    # Permite ao modelo entender padrões cíclicos passados
    for lag in [7, 14, 21, 28]:
        df[f'sales_lag_{lag}'] = df['Sales'].shift(lag)
        df[f'customers_lag_{lag}'] = df['Num_Customers'].shift(lag)
        
    # 8. Atributos de Forecasting - Médias Móveis (7 dias)
    # Captura a dinâmica recente das vendas e a sua volatilidade
    df['sales_roll_mean_7'] = df['Sales'].shift(1).rolling(window=7).mean()
    df['sales_roll_std_7'] = df['Sales'].shift(1).rolling(window=7).std()
    
    # 9. Compatibilidade com Prophet (Nomenclatura exigida pela biblioteca)
    df['ds'] = df['Date'] # Data
    df['y'] = df['Sales'] # Alvo (Target)
    
    # 10. Limpeza Final (Remoção de registos com NaN resultantes dos lags)
    df = df.dropna().reset_index(drop=True)
    
    # 11. Exportação do Dataset Processado
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, f"{store_name}_processed.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"  [{store_name}] Ficheiro guardado com sucesso: {output_path}")
    
    return df

def run_full_preparation(input_dir='data/raw/', output_dir='data/processed/'):
    """Processa todos os ficheiros brutos de lojas detetados no diretório de entrada."""
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not csv_files:
        logger.error(f"Nenhum ficheiro CSV encontrado em {input_dir}")
        return
        
    for f in csv_files:
        prepare_store_data(f, output_dir)
    
    # Consolidação de um dataset mestre para análises globais (W5 Optimization)
    create_merged_dataset(output_dir)
    
    logger.info("Preparação de dados concluída para todas as lojas.")

def create_merged_dataset(processed_dir='data/processed/'):
    """Combina os CSVs individuais das lojas num único dataset consolidado."""
    logger.info("A criar dataset mestre consolidado (all_stores_processed.csv)")
    processed_files = glob.glob(os.path.join(processed_dir, "*_processed.csv"))
    
    # Filtro para evitar a auto-inclusão do próprio ficheiro consolidado
    processed_files = [f for f in processed_files if 'all_stores' not in f]
    
    all_data = []
    for f in processed_files:
        temp_df = pd.read_csv(f)
        store_name = os.path.basename(f).replace('_processed.csv', '')
        temp_df['store_id'] = store_name # Identificação da origem dos dados
        all_data.append(temp_df)
    
    if all_data:
        merged_df = pd.concat(all_data, ignore_index=True)
        output_path = os.path.join(processed_dir, "all_stores_processed.csv")
        merged_df.to_csv(output_path, index=False)
        logger.info(f"  Dataset consolidado guardado em: {output_path}")

if __name__ == "__main__":
    run_full_preparation()
