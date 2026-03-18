import pandas as pd
import os

def load_and_merge_data(raw_data_path, processed_data_path):
    """
    Lê os ficheiros CSV das 4 lojas, adiciona uma coluna identificadora,
    concatena tudo num único DataFrame e guarda na pasta processed.
    """
    stores = ['baltimore', 'lancaster', 'philadelphia', 'richmond']
    all_data = []

    for store in stores:
        # Constrói o caminho para cada ficheiro
        file_path = os.path.join(raw_data_path, f"{store}.csv")
        
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            
            # Adiciona a coluna com o nome da loja para não perdermos a rastreabilidade
            df['Store'] = store.capitalize()
            
            all_data.append(df)
        else:
            print(f"Aviso: Ficheiro não encontrado - {file_path}")

    # Concatena todos os DataFrames da lista num só grande DataFrame
    if all_data:
        merged_df = pd.concat(all_data, ignore_index=True)
        
        # --- PONTO DE INTEGRAÇÃO ---
        # importar e chamar aqui a função de limpeza:
        # merged_df = clean_data(merged_df)
        
        # Guarda o resultado unificado
        # Cria a pasta caso não exista
        os.makedirs(processed_data_path, exist_ok=True) 
        output_file = os.path.join(processed_data_path, "all_stores_merged.csv")
        
        merged_df.to_csv(output_file, index=False)
        print(f"Sucesso! {len(merged_df)} linhas guardadas em: {output_file}")
        
        return merged_df
    else:
        print("Erro: Nenhum dado foi carregado.")
        return None

if __name__ == "__main__":
    # Caminhos relativos à localização do script
    RAW_PATH = "../../data/raw"
    PROCESSED_PATH = "../../data/processed"
    
    load_and_merge_data(RAW_PATH, PROCESSED_PATH)