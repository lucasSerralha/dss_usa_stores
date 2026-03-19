import pandas as pd
import os
import numpy as np

def create_features(input_path, output_path):
    print(f"A carregar dados unificados de: {input_path}")
    
    if not os.path.exists(input_path):
        print("Erro: Ficheiro all_stores_merged.csv não encontrado. Executa o data_loader.py primeiro.")
        return

    # 1. Carregar dados e garantir que a data é do tipo correto
    df = pd.read_csv(input_path)
    df['Date'] = pd.to_datetime(df['Date'])
    
    # Ordenar por Loja e por Data (Crucial para Séries Temporais)
    df = df.sort_values(by=['Store', 'Date']).reset_index(drop=True)

    print("A extrair variáveis de calendário...")
    # 2. Variáveis de Calendário
    df['DayOfWeek'] = df['Date'].dt.dayofweek # 0=Segunda, 6=Domingo
    df['Month'] = df['Date'].dt.month
    df['DayOfMonth'] = df['Date'].dt.day
    df['IsWeekend'] = df['DayOfWeek'].apply(lambda x: 1 if x >= 5 else 0)

    print("A normalizar a variável de descontos (Pct_On_Sale para pr)...")
    # 3. Limpeza sugerida pelo Professor: Normalizar Descontos (0 a 1)
    # Assumimos que Pct_On_Sale vem em percentagem (ex: 10.5). Passamos para decimal (0.105)
    df['pr'] = df['Pct_On_Sale'] / 100.0
    # Garantir que não há valores negativos por erro nos dados originais
    df['pr'] = df['pr'].clip(lower=0.0)

    # Transformar a variável categórica TouristEvent em binária (0 ou 1)
    df['TouristEvent_Bin'] = df['TouristEvent'].apply(lambda x: 1 if x.lower() == 'yes' else 0)

    print("A criar Lags temporais (Vendas passadas)...")
    # 4. Feature Engineering: Lags (O segredo para o Forecasting)
    # Criamos uma coluna com as vendas que ocorreram exatamente há 1 semana atrás (Lag de 7 dias)
    # O groupby('Store') garante que não misturamos o passado de Baltimore com o de Lancaster
    df['Sales_Lag7'] = df.groupby('Store')['Sales'].shift(7)
    
    # Criar um Lag do número de clientes (também há 7 dias)
    df['Customers_Lag7'] = df.groupby('Store')['Num_Customers'].shift(7)

    # Como os primeiros 7 dias de cada loja não têm passado para preencher o Lag, geram-se NaNs.
    # Vamos preencher esses valores nulos com a média da loja ou simplesmente remover essas 7 linhas.
    # Neste caso, vamos usar backfill (copiar o valor seguinte mais próximo) para não perdermos dias.
    df['Sales_Lag7'] = df.groupby('Store')['Sales_Lag7'].bfill()
    df['Customers_Lag7'] = df.groupby('Store')['Customers_Lag7'].bfill()

    # 5. Guardar o novo dataset enriquecido
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)
    
    print(f"Sucesso! Dataset enriquecido com {len(df.columns)} colunas.")
    print(f"Guardado em: {output_path}")
    
    # Mostrar uma pequena amostra das novas colunas
    print("\nAmostra das novas variáveis (Store, Date, IsWeekend, pr, Sales_Lag7):")
    print(df[['Store', 'Date', 'IsWeekend', 'pr', 'Sales_Lag7']].head(3))

if __name__ == "__main__":
    # Configuração de caminhos absolutos
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    INPUT_FILE = os.path.abspath(os.path.join(SCRIPT_DIR, "../../data/processed/all_stores_merged.csv"))
    OUTPUT_FILE = os.path.abspath(os.path.join(SCRIPT_DIR, "../../data/processed/features_stores_merged.csv"))
    
    create_features(INPUT_FILE, OUTPUT_FILE)