import numpy as np
import pandas as pd
import os
import sys

# =====================================================
# CONFIGURAÇÃO DO PROJETO
# =====================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(SCRIPT_DIR, "..", ".."))

sys.path.append(os.path.join(PROJECT_ROOT, "src"))

from utils.profit_logic import optimize_weekly_wrapper


# =====================================================
# CARREGAR PREVISÃO REAL
# =====================================================

def load_real_forecast(store="baltimore"):

    file_path = os.path.join(
        PROJECT_ROOT,
        "data",
        "processed",
        "all_stores_processed.csv"
    )

    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Ficheiro não encontrado: {file_path}")

    df = pd.read_csv(file_path)

    df = df[df["store_id"] == store].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date")

    customers = df["Num_Customers"].tail(7).values
    weekend = (df["Date"].dt.dayofweek >= 5).tail(7).values

    return customers, weekend


# =====================================================
# O3 SCALARIZATION (VERSÃO CORRIGIDA)
# =====================================================

def o3_scalarization(solution, store="baltimore", debug=False):

    # previsões reais
    customers, weekend = load_real_forecast(store)

    # otimização base (O1, O2, O3 vêm daqui)
    f1, f2, f3 = optimize_weekly_wrapper(
        decision_vars=solution,
        store=store,
        forecast_customers=customers,
        forecast_is_weekend=weekend
    )

    profit = -f1
    staff = f2

    # =====================================================
    # O3 CORRETO (SEM BASELINE ARTIFICIAL)
    # =====================================================

    staff_safe = max(staff, 1)

    # eficiência operacional real
    o3 = profit / staff_safe

    # escalarização pedida (w = 0.7)
    score = 0.7 * o3

    # =====================================================
    # OUTPUT LIMPO E CLARO
    # =====================================================

    if debug:

        print("\n" + "=" * 60)
        print("           RESULTADOS DO MODELO O3")
        print("=" * 60)

        print(f"Lucro total estimado:        {profit:,.2f} €")
        print(f"Número de funcionários:      {staff}")
        print(f"Eficiência (O3):             {o3:,.4f}")

        print("-" * 60)
        print(f"Score final (peso 0.7):      {score:,.4f}")

        if profit < 0:
            print("\nEstado: operação com prejuízo")
        elif o3 > 0:
            print("\nEstado: operação eficiente")
        else:
            print("\nEstado: operação ineficiente")

        print("=" * 60)

    return score


# =====================================================
# TESTE LOCAL
# =====================================================

if __name__ == "__main__":

    print("A executar O3 (versão final corrigida)")

    solution = np.random.rand(21)

    score = o3_scalarization(solution, "baltimore", debug=True)

    print("\nResultado final O3:", round(score, 2))