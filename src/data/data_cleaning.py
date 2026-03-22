import pandas as pd

def clean_data(df):
    
    df = df.copy()

    # 1. Datas
    df['Date'] = pd.to_datetime(df['Date'])
    df = df.sort_values('Date')

    # 2. Valores em falta
    df['Pct_On_Sale'] = df['Pct_On_Sale'].interpolate()
    df = df.ffill()

    # 3. Outliers (clipping)
    for col in ['Num_Customers', 'Num_Employees', 'Sales']:
        q_low = df[col].quantile(0.01)
        q_high = df[col].quantile(0.99)
        df[col] = df[col].clip(q_low, q_high)

    # 4. Variáveis categóricas
    df['TouristEvent'] = df['TouristEvent'].map({'Yes': 1, 'No': 0})

    return df