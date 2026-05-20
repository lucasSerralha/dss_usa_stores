"""
run_simulated_annealing_o1.py — Tarefa O1: Otimizacao Individual por Loja via Simulated Annealing

Algoritmo: Simulated Annealing com multiplos restarts
Objetivo unico: Maximizar o lucro semanal (f1) para cada loja individualmente.

O SA aceita solucos piores com probabilidade exp(-delta/T), permitindo escapar
de otimos locais. A temperatura decresce geometricamente (T *= alpha).

Resultados guardados em: results/03_Optimization/individual/simulated_annealing/
"""

import sys
import os
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.dont_write_bytecode = True
sys.path.insert(0, "src")

from optimization.simulated_annealing import simulated_annealing
from utils.profit_logic import optimize_weekly_wrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("O1.SimAnneal")

# ---------------------------------------------------------------------------
# Previsoes — ultima semana observada (2014-06-08 Dom → 2014-06-14 Sab)
# ---------------------------------------------------------------------------
FORECAST_INPUTS = {
    "baltimore": {
        "customers":  [97, 61, 65, 71, 65, 89, 125],
        "is_weekend": [True, False, False, False, False, False, True],
    },
    "lancaster": {
        "customers":  [116, 72, 77, 84, 77, 106, 149],
        "is_weekend": [True, False, False, False, False, False, True],
    },
    "philadelphia": {
        "customers":  [230, 144, 154, 168, 154, 211, 298],
        "is_weekend": [True, False, False, False, False, False, True],
    },
    "richmond": {
        "customers":  [64, 40, 42, 46, 42, 58, 82],
        "is_weekend": [True, False, False, False, False, False, True],
    },
}

DAY_LABELS = ["Dom", "Seg", "Ter", "Qua", "Qui", "Sex", "Sab"]
STORES = list(FORECAST_INPUTS.keys())
OUT_DIR = os.path.join("results", "03_Optimization", "individual", "simulated_annealing")
os.makedirs(OUT_DIR, exist_ok=True)

# --- Hiperparametros do SA ---
SA_PARAMS = dict(
    T_init=5000.0,    # Temperatura inicial alta (explora amplamente)
    T_min=1.0,        # Temperatura minima (criterio de paragem)
    alpha=0.995,      # Taxa de arrefecimento geometrico (~2001 passos de temperatura)
    iter_per_temp=50, # Perturbacoes por nivel de temperatura
    n_restarts=5,     # Restarts independentes para robustez
    seed=42,
)


# ===========================================================================
# Decodificacao da solucao
# ===========================================================================

def decode_solution(solution, forecast_customers, forecast_is_weekend):
    """Descodifica o vetor de 21 variaveis num plano semanal legivel."""
    plan = []
    for i in range(7):
        pr = max(0.0, min(0.30, solution[i * 3]))
        hr_x = max(0, int(round(solution[i * 3 + 1])))
        hr_j = max(0, int(round(solution[i * 3 + 2])))
        plan.append({
            "day":         i,
            "day_label":   DAY_LABELS[i],
            "pr":          round(pr, 4),
            "hr_x":        hr_x,
            "hr_j":        hr_j,
            "total_staff": hr_x + hr_j,
            "customers":   forecast_customers[i],
            "is_weekend":  forecast_is_weekend[i],
        })
    return plan


# ===========================================================================
# Graficos
# ===========================================================================

def plot_convergence(store, best_history, all_histories):
    """Grafico de convergencia do SA: todos os restarts + melhor."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for h in all_histories:
        lucro_h = [-x for x in h]
        ax.plot(lucro_h, color="#FFCC80", alpha=0.5, linewidth=0.8)

    best_lucro = [-x for x in best_history]
    ax.plot(best_lucro, color="#FF6F00", linewidth=2.0, label="Melhor Restart")

    ax.set_xlabel("Passos de temperatura", fontsize=11)
    ax.set_ylabel("Lucro ($)", fontsize=11)
    ax.set_title(
        f"Convergencia Simulated Annealing — {store.capitalize()} (O1: Max Lucro)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"{store}_sa_convergence.png")
    plt.savefig(path, dpi=150)
    plt.close()
    log.info("Grafico de convergencia guardado -> %s", path)


def plot_temperature_schedule(store):
    """Visualizacao do esquema de arrefecimento geometrico."""
    T = SA_PARAMS["T_init"]
    temps = []
    while T > SA_PARAMS["T_min"]:
        temps.append(T)
        T *= SA_PARAMS["alpha"]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(temps, color="#FF6F00", linewidth=1.5)
    ax.set_xlabel("Passos de temperatura", fontsize=11)
    ax.set_ylabel("Temperatura T", fontsize=11)
    ax.set_title(
        f"Esquema de Arrefecimento SA — alpha={SA_PARAMS['alpha']} "
        f"({len(temps)} passos)",
        fontsize=12, fontweight="bold",
    )
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"{store}_sa_temperature_schedule.png")
    plt.savefig(path, dpi=150)
    plt.close()
    log.info("Esquema de temperatura guardado -> %s", path)


def plot_best_plan(store, plan, lucro):
    """Grafico do plano semanal otimo encontrado pelo SA."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    days = [d["day_label"] for d in plan]
    hr_x = [d["hr_x"] for d in plan]
    hr_j = [d["hr_j"] for d in plan]
    pr = [d["pr"] for d in plan]
    total = [d["total_staff"] for d in plan]
    x = np.arange(len(days))
    w = 0.35

    ax = axes[0]
    ax.bar(x - w / 2, hr_x, w, label="Peritos", color="#FF6F00", alpha=0.85)
    ax.bar(x + w / 2, hr_j, w, label="Juniores", color="#FFCC02", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(days)
    ax.set_title("Composicao de Staff"); ax.set_ylabel("Funcionarios")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(x, [p * 100 for p in pr], color="#E65100", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(days)
    ax.set_title("Desconto Diario (%)"); ax.set_ylabel("Desconto (%)")
    ax.set_ylim(0, 35); ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    colors = ["#F44336" if t > 8 else "#4CAF50" for t in total]
    ax.bar(x, total, color=colors, alpha=0.85)
    ax.axhline(8, color="black", ls="--", lw=1.2, label="Limite util (8)")
    ax.set_xticks(x); ax.set_xticklabels(days)
    ax.set_title("Staff Total por Dia"); ax.set_ylabel("Total")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"{store.capitalize()} [Simulated Annealing] — O1: Lucro Maximo ${lucro:,.0f}",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"{store}_sa_best_plan.png")
    plt.savefig(path, dpi=150)
    plt.close()
    log.info("Grafico do plano otimo guardado -> %s", path)


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("\n" + "=" * 65)
    print("  O1 — Otimizacao Individual por Loja (SIMULATED ANNEALING)")
    print(f"  T_init={SA_PARAMS['T_init']} | alpha={SA_PARAMS['alpha']} | "
          f"iter/T={SA_PARAMS['iter_per_temp']} | restarts={SA_PARAMS['n_restarts']}")
    print("=" * 65 + "\n")

    all_summaries = []

    for store in STORES:
        inputs = FORECAST_INPUTS[store]
        log.info("=== A otimizar: %s ===", store.upper())

        best_solution, best_score, best_history, all_histories = simulated_annealing(
            store=store,
            forecast_customers=inputs["customers"],
            forecast_is_weekend=inputs["is_weekend"],
            **SA_PARAMS,
        )

        lucro_real = -best_score
        plan = decode_solution(best_solution, inputs["customers"], inputs["is_weekend"])

        total_staff = sum(d["total_staff"] for d in plan)
        avg_discount = float(np.mean([d["pr"] for d in plan]))

        # Numero total de iteracoes (aprox)
        n_temp_steps = len(best_history)
        total_iters = n_temp_steps * SA_PARAMS["iter_per_temp"] * SA_PARAMS["n_restarts"]

        summary = {
            "store":          store,
            "algoritmo":      "Simulated Annealing",
            "lucro_maximo":   round(lucro_real, 2),
            "total_staff":    total_staff,
            "avg_discount":   round(avg_discount * 100, 2),
            "T_init":         SA_PARAMS["T_init"],
            "alpha":          SA_PARAMS["alpha"],
            "n_restarts":     SA_PARAMS["n_restarts"],
            "n_temp_steps":   n_temp_steps,
            "total_iters":    total_iters,
        }
        all_summaries.append(summary)

        log.info("  Lucro Maximo: $%.0f", lucro_real)

        print(f"\n{'-' * 60}")
        print(f"  {store.upper()} — Lucro Maximo: ${lucro_real:,.2f}")
        print(f"  Staff Total Semanal: {total_staff}  |  Desconto Medio: {avg_discount * 100:.1f}%")
        print(f"  Passos de temperatura: {n_temp_steps}  |  Iters totais: {total_iters:,}")
        print(f"\n  Plano Semanal Otimo (Simulated Annealing):")
        print(f"  {'Dia':<5} {'Desconto':>9} {'Peritos':>8} {'Juniores':>9} {'Total':>6} {'Clientes':>9} {'FdS':>5}")
        print(f"  {'-' * 56}")
        for d in plan:
            fds = "Sim" if d["is_weekend"] else "Nao"
            print(
                f"  {d['day_label']:<5} {d['pr'] * 100:>8.1f}%"
                f" {d['hr_x']:>8} {d['hr_j']:>9} {d['total_staff']:>6}"
                f" {d['customers']:>9} {fds:>5}"
            )

        # --- Guardar CSVs ---
        plan_df = pd.DataFrame(plan)
        plan_df.to_csv(os.path.join(OUT_DIR, f"{store}_sa_best_plan.csv"), index=False)

        conv_df = pd.DataFrame({
            "temp_step": range(len(best_history)),
            "best_lucro": [-x for x in best_history],
        })
        conv_df.to_csv(os.path.join(OUT_DIR, f"{store}_sa_convergence.csv"), index=False)

        # --- Graficos ---
        plot_convergence(store, best_history, all_histories)
        plot_temperature_schedule(store)
        plot_best_plan(store, plan, lucro_real)

    # --- Resumo geral ---
    summary_df = pd.DataFrame(all_summaries)
    summary_path = os.path.join(OUT_DIR, "simulated_annealing_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 65)
    print("  RESUMO — O1: SIMULATED ANNEALING")
    print("=" * 65)
    print(summary_df[["store", "lucro_maximo", "avg_discount", "total_staff"]].to_string(index=False))
    print(f"\n  Resultados guardados em: {os.path.abspath(OUT_DIR)}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
