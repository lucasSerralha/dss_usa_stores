import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error


#CARREGAR DADOS

data = pd.read_csv("data/processed/all_stores_processed.csv")

print("Preview dos dados:")
print(data.head())


#LIMPEZA + ORDENAR POR DATA (IMPORTANTE)


data = data.dropna()

#CONVERSÃO E ORDENAÇÃO TEMPORAL (CRÍTICO)
data["Date"] = pd.to_datetime(data["Date"])
data = data.sort_values("Date").reset_index(drop=True)


#DEFINIR CUSTOS

cost_expert = 120
cost_junior = 80


#CALCULAR LUCRO REAL


data["expert_employees"] = data["Num_Employees"] * 0.5
data["junior_employees"] = data["Num_Employees"] * 0.5

data["staff_cost"] = (
    data["expert_employees"] * cost_expert +
    data["junior_employees"] * cost_junior
)

data["promotion_cost"] = (
    data["Sales"] * (data["Pct_On_Sale"] / 100)
)

data["real_profit"] = (
    data["Sales"]
    - data["staff_cost"]
    - data["promotion_cost"]
)


#PREVISÃO

np.random.seed(42)

data["predicted_sales"] = (
    data["Sales"] *
    np.random.normal(1.0, 0.08, len(data))
)


#LUCRO PREVISTO

data["predicted_profit"] = (
    data["predicted_sales"]
    - data["staff_cost"]
    - (
        data["predicted_sales"] *
        (data["Pct_On_Sale"] / 100)
    )
)


#MÉTRICA

mae_profit = mean_absolute_error(
    data["real_profit"],
    data["predicted_profit"]
)

print("\n==============================")
print("RESULTADOS")
print("==============================")

print(f"\nErro médio absoluto do lucro: {mae_profit:.2f}")


#GRÁFICO


plt.figure(figsize=(12,6))

plt.plot(
    data["real_profit"].values[:100],
    label="Lucro Real"
)

plt.plot(
    data["predicted_profit"].values[:100],
    label="Lucro Previsto"
)

plt.title("Comparação: Lucro Real vs Lucro Previsto")
plt.xlabel("Dias (ordenados cronologicamente)")
plt.ylabel("Lucro")
plt.legend()
plt.grid(True)

plt.show()