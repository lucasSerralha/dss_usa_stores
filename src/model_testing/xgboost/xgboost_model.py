import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error
from xgboost import XGBRegressor


#CARREGAR DADOS

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
file_path = os.path.join(BASE_DIR, "data", "processed", "all_stores_processed.csv")

data = pd.read_csv(file_path)

print("Preview dos dados:")
print(data.head())


#PREPARAÇÃO DOS DADOS

# Converter TouristEvent para numérico
data["TouristEvent"] = data["TouristEvent"].astype(str).str.lower()
data["TouristEvent"] = data["TouristEvent"].map({
    "no": 0,
    "yes": 1,
    "0": 0,
    "1": 1,
    "false": 0,
    "true": 1
})

#Selecionar variáveis relevantes
df_model = data[[
    "Num_Employees",
    "Num_Customers",
    "Pct_On_Sale",
    "TouristEvent",
    "sales_roll_mean_7",
    "sales_roll_std_7",
    "Sales"
]].copy()

#Limpeza de dados
df_model = df_model.replace([np.inf, -np.inf], np.nan)
df_model = df_model.dropna()

print("\nNúmero de linhas após limpeza:", len(df_model))


#SEPARAR X e y

X = df_model.drop(columns=["Sales"])
y = df_model["Sales"]


#TESTE

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


#MODELO XGBOOST (MELHORADO)

model = XGBRegressor(
    n_estimators=300,
    max_depth=5,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

model.fit(X_train, y_train)

#PREVISÕES

y_pred = model.predict(X_test)


#AVALIAÇÃO

mae = mean_absolute_error(y_test, y_pred)

print("\nRESULTADOS DO MODELO XGBOOST\n")


print(f"Erro Médio Absoluto (MAE): {mae:.2f}")


#EXEMPLOS REAL vs PREVISTO

print("\nExemplos de previsão:\n")

for i in range(5):
    print(f"Real: {y_test.iloc[i]:.2f} | Previsto: {y_pred[i]:.2f}")


#GRÁFICO REAL vs PREVISTO

plt.figure(figsize=(8,6))
plt.scatter(y_test, y_pred)
plt.xlabel("Valores Reais")
plt.ylabel("Valores Previstos")
plt.title("Real vs Previsto - XGBoost")
plt.grid(True)

#Linha perfeita
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         color='red')

plt.show()


#IMPORTÂNCIA DAS VARIÁVEIS

importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(8,6))
plt.barh(features, importances)
plt.title("Importância das Variáveis")
plt.xlabel("Importância")
plt.show()