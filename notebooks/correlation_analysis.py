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
])

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

plt.title("Correlation between Sales and Exogenous Variables")

plt.show()


# 7. SCATTER: CUSTOMERS vs SALES

plt.figure()

sns.scatterplot(
    x=data["Num_Customers"],
    y=data["Sales"]
)

plt.title("Customers vs Sales")

plt.show()


# 8. SCATTER: EMPLOYEES vs SALES

plt.figure()

sns.scatterplot(
    x=data["Num_Employees"],
    y=data["Sales"]
)

plt.title("Employees vs Sales")

plt.show()


# 9. SCATTER: PROMOTIONS vs SALES

plt.figure()

sns.scatterplot(
    x=data["Pct_On_Sale"],
    y=data["Sales"]
)

plt.title("Promotion Percentage vs Sales")

plt.show()


# 10. BOXPLOT EVENTOS TURÍSTICOS

plt.figure()

sns.boxplot(
    x=data["TouristEvent"],
    y=data["Sales"]
)

plt.title("Sales during Tourist Events")

plt.xlabel("Tourist Event (0 = No, 1 = Yes)")

plt.show()