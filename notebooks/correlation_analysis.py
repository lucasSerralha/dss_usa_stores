import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


# 1. CARREGAR OS DATASETS

richmond = pd.read_csv("data/raw/richmond.csv")
baltimore = pd.read_csv("data/raw/baltimore.csv")
lancaster = pd.read_csv("data/raw/lancaster.csv")
philadelphia = pd.read_csv("data/raw/philadelphia.csv")


# 2. ADICIONAR NOME DA LOJA

richmond["store"] = "richmond"
baltimore["store"] = "baltimore"
lancaster["store"] = "lancaster"
philadelphia["store"] = "philadelphia"


# 3. JUNTAR TODOS OS DATASETS

data = pd.concat([
    richmond,
    baltimore,
    lancaster,
    philadelphia
], ignore_index=True)

print("\nDATA PREVIEW:")
print(data.head())

print("\nCOLUNAS:")
print(data.columns)


# 4. PREPARAR DADOS

# converter TouristEvent para numérico
data["TouristEvent"] = data["TouristEvent"].map({
    "No": 0,
    "Yes": 1
})


import os

# Define output directory
output_dir = "results/01_EDA_Gallery/"
os.makedirs(output_dir, exist_ok=True)

# 5. CALCULAR CORRELAÇÃO
corr = data[[
    "Sales",
    "Num_Employees",
    "Num_Customers",
    "Pct_On_Sale",
    "TouristEvent"
]].corr()

print("\nCORRELATION MATRIX:")
print(corr)


# 6. HEATMAP DE CORRELAÇÃO
plt.figure(figsize=(8,6))
sns.heatmap(
    corr,
    annot=True,
    cmap="coolwarm",
    fmt=".2f"
)
plt.title("Correlation Matrix: Sales vs Exogenous Variables")
plt.savefig(os.path.join(output_dir, "eda_heatmap_correlacao.png"), dpi=150, bbox_inches='tight')
plt.close()

# 7. SCATTER: CUSTOMERS vs SALES
plt.figure(figsize=(10,6))
sns.scatterplot(
    x=data["Num_Customers"],
    y=data["Sales"],
    alpha=0.5
)
plt.title("Correlation: Customers vs Sales")
plt.savefig(os.path.join(output_dir, "eda_scatter_clientes_vendas.png"), dpi=150, bbox_inches='tight')
plt.close()

# 8. SCATTER: EMPLOYEES vs SALES
plt.figure(figsize=(10,6))
sns.scatterplot(
    x=data["Num_Employees"],
    y=data["Sales"],
    alpha=0.5,
    color='green'
)
plt.title("Correlation: Employees vs Sales")
plt.savefig(os.path.join(output_dir, "eda_scatter_staff_vendas.png"), dpi=150, bbox_inches='tight')
plt.close()

# 9. SCATTER: PROMOTIONS vs SALES
plt.figure(figsize=(10,6))
sns.scatterplot(
    x=data["Pct_On_Sale"],
    y=data["Sales"],
    alpha=0.5,
    color='orange'
)
plt.title("Correlation: Promotion Percentage vs Sales")
plt.savefig(os.path.join(output_dir, "eda_scatter_promocoes_vendas.png"), dpi=150, bbox_inches='tight')
plt.close()

# 10. BOXPLOT EVENTOS TURÍSTICOS
plt.figure(figsize=(8,6))
sns.boxplot(
    x=data["TouristEvent"],
    y=data["Sales"],
    palette="Set2"
)
plt.title("Sales during Tourist Events")
plt.xlabel("Tourist Event (0 = No, 1 = Yes)")
plt.savefig(os.path.join(output_dir, "eda_boxplot_eventos.png"), dpi=150, bbox_inches='tight')
plt.close()

print(f"\nTodos os gráficos de correlação foram guardados em: {output_dir}")