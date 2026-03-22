import pandas as pd
import os

def train_test_split_time_series(df, test_size=0.2):
    
    train_list = []
    test_list = []

    for store in df['Store'].unique():
        
        df_store = df[df['Store'] == store].sort_values('Date')
        
        split_index = int(len(df_store) * (1 - test_size))
        
        train = df_store.iloc[:split_index]
        test = df_store.iloc[split_index:]
        
        train_list.append(train)
        test_list.append(test)

    train_df = pd.concat(train_list)
    test_df = pd.concat(test_list)

    return train_df, test_df


if __name__ == "__main__":
    
    # Caminhos
    INPUT_PATH = "../../data/processed/all_stores_merged.csv"
    OUTPUT_PATH = "../../data/processed"

    # Carregar dados
    df = pd.read_csv(INPUT_PATH)
    df['Date'] = pd.to_datetime(df['Date'])

    # Split
    train_df, test_df = train_test_split_time_series(df)

    # Criar pasta
    os.makedirs(OUTPUT_PATH, exist_ok=True)

    # Guardar
    train_df.to_csv(f"{OUTPUT_PATH}/train.csv", index=False)
    test_df.to_csv(f"{OUTPUT_PATH}/test.csv", index=False)

    print("Split concluído com sucesso!")