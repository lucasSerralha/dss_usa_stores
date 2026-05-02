import pandas as pd
import numpy as np
import statsmodels.api as sm
import os


#Carregar dados

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../.."))
file_path = os.path.join(BASE_DIR, "data", "processed", "all_stores_processed.csv")

data = pd.read_csv(file_path)

print("Preview dos dados:")
print(data.head())

#Preparação dos dados

#Ver valores da variável TouristEvent
print("\nValores únicos de TouristEvent:")
print(data["TouristEvent"].unique())

#Converter TouristEvent para formato numérico
data["TouristEvent"] = data["TouristEvent"].astype(str).str.lower()
data["TouristEvent"] = data["TouristEvent"].map({
    "no": 0,
    "yes": 1,
    "0": 0,
    "1": 1,
    "false": 0,
    "true": 1
})

#Selecionar apenas as variáveis relevantes
df_model = data[[
    "Num_Employees",
    "Num_Customers",
    "Pct_On_Sale",
    "TouristEvent",
    "Sales"
]].copy()

#Tratar valores inválidos
df_model = df_model.replace([np.inf, -np.inf], np.nan)
df_model = df_model.fillna(0)

print("\nNúmero de linhas depois da limpeza:", len(df_model))

#Separar variáveis independentes e dependente
X = df_model.drop(columns=["Sales"])
y = df_model["Sales"]


#Modelo Binomial Negativo

X = sm.add_constant(X)

model = sm.GLM(
    y,
    X,
    family=sm.families.NegativeBinomial()
)

results = model.fit()

#Resultados

print("\nResumo do modelo:\n")
print(results.summary())

#Interpretação simples


print("\nInterpretação dos coeficientes:\n")

coef = results.params
pvalues = results.pvalues

for var in coef.index:
    print(f"{var}:")
    print(f"  Coeficiente = {coef[var]:.4f}")
    print(f"  p-value = {pvalues[var]:.4f}")

    if pvalues[var] < 0.05:
        print("  Resultado significativo")
    else:
        print("  Resultado não significativo")

    print()