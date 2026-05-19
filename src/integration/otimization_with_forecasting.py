import pandas as pd
import numpy as np
import os
import random

from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor


#IMPORTAR HILL CLIMBING

import sys

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(SRC_DIR)

from optimization.hill_climbing import hill_climbing


#CARREGAR DADOS

BASE_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

file_path = os.path.join(
    BASE_DIR,
    "data",
    "processed",
    "all_stores_processed.csv"
)

data = pd.read_csv(file_path)


#PREPARAÇÃO DOS DADOS

data["TouristEvent"] = data["TouristEvent"].astype(str).str.lower()

data["TouristEvent"] = data["TouristEvent"].map({
    "no": 0,
    "yes": 1,
    "0": 0,
    "1": 1,
    "false": 0,
    "true": 1
})

df_model = data[[
    "Num_Employees",
    "Num_Customers",
    "Pct_On_Sale",
    "TouristEvent",
    "sales_roll_mean_7",
    "sales_roll_std_7",
    "Sales"
]].copy()

df_model = df_model.replace([np.inf, -np.inf], np.nan)
df_model = df_model.dropna()


#TREINAR XGBOOST

X = df_model.drop(columns=["Sales"])
y = df_model["Sales"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)

model = XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)


#GERAR PREVISÕES FUTURAS


future_data = X_test.head(7)

forecast_sales = model.predict(future_data)

print("\nPrevisão de vendas para os próximos dias:\n")

for i, value in enumerate(forecast_sales):
    print(f"Dia {i+1}: {value:.2f}")


#CONVERTER PARA CLIENTES

forecast_customers = []

for sale in forecast_sales:
    estimated_customers = int(sale / 300)
    forecast_customers.append(estimated_customers)

print("\nClientes previstos:\n")
print(forecast_customers)


#PREVISÃO FIM DE SEMANA

forecast_is_weekend = [
    False,
    False,
    False,
    False,
    False,
    True,
    True
]


#EXECUTAR HILL CLIMBING


solution, score, history = hill_climbing(
    'baltimore',
    forecast_customers,
    forecast_is_weekend,
    iterations=1000
)


#MOSTRAR RESULTADOS

days = [
    "Segunda",
    "Terça",
    "Quarta",
    "Quinta",
    "Sexta",
    "Sábado",
    "Domingo"
]

print("\nMelhor estratégia semanal encontrada:\n")

for i in range(7):

    pr = solution[i*3]
    hr_x = int(solution[i*3 + 1])
    hr_j = int(solution[i*3 + 2])

    print(f"{days[i]}:")
    print(f"  Desconto: {pr*100:.2f}%")
    print(f"  Funcionários Experientes: {hr_x}")
    print(f"  Funcionários Juniores: {hr_j}")
    print()

print(f"Lucro Semanal Estimado: {-score:.2f}")