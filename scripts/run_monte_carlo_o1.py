"""
run_monte_carlo_o1.py — Tarefa O1: Otimização Individual por Loja via Monte Carlo

Algoritmo: Monte Carlo (amostragem aleatória pura)
Objetivo único: Maximizar o lucro semanal (f1) para cada loja individualmente.

Funciona como baseline estocástico: avalia N soluções aleatórias e retém a melhor.
Produz também a distribuição do espaço de lucro (histograma) para análise.

Resultados guardados em: results/03_Optimization/individual/monte_carlo/
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

from optimization.monte_carlo import monte_carlo
from utils.profit_logic import optimize_weekly_wrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("O1.MonteCarlo")

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
OUT_DIR = os.path.join("results_v2", "03_Optimization", "individual", "monte_carlo")
os.makedirs(OUT_DIR, exist_ok=True)

N_SAMPLES = 20_000  # Numero de amostras Monte Carlo


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

def plot_convergence(store, history):
    """Evolucao do melhor lucro acumulado ao longo das amostras Monte Carlo."""
    fig, ax = plt.subplots(figsize=(10, 5))
    lucro_history = [-x for x in history]
    ax.plot(lucro_history, color="#9C27B0", linewidth=1.5, label="Melhor acumulado")
    ax.set_xlabel("Amostras avaliadas", fontsize=11)
    ax.set_ylabel("Lucro ($)", fontsize=11)
    ax.set_title(
        f"Convergencia Monte Carlo — {store.capitalize()} (O1: Max Lucro)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"{store}_mc_convergence.png")
    plt.savefig(path, dpi=150)
    plt.close()
    log.info("Grafico de convergencia guardado -> %s", path)


def plot_distribution(store, all_scores, best_score):
    """Histograma da distribuicao de lucros de todas as amostras."""
    fig, ax = plt.subplots(figsize=(10, 5))
    lucros = [-x for x in all_scores]
    ax.hist(lucros, bins=80, color="#9C27B0", alpha=0.7, edgecolor="white", linewidth=0.3)
    ax.axvline(-best_score, color="#F44336", linewidth=2.0, linestyle="--",
               label=f"Melhor: ${-best_score:,.0f}")
    ax.axvline(float(np.mean(lucros)), color="#FF9800", linewidth=1.5, linestyle=":",
               label=f"Media: ${np.mean(lucros):,.0f}")
    ax.set_xlabel("Lucro ($)", fontsize=11)
    ax.set_ylabel("Frequencia", fontsize=11)
    ax.set_title(
        f"Distribuicao Monte Carlo — {store.capitalize()} ({N_SAMPLES:,} amostras)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"{store}_mc_distribution.png")
    plt.savefig(path, dpi=150)
    plt.close()
    log.info("Histograma de distribuicao guardado -> %s", path)


def plot_best_plan(store, plan, lucro):
    """Grafico do plano semanal otimo encontrado pelo Monte Carlo."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    days = [d["day_label"] for d in plan]
    hr_x = [d["hr_x"] for d in plan]
    hr_j = [d["hr_j"] for d in plan]
    pr = [d["pr"] for d in plan]
    total = [d["total_staff"] for d in plan]
    x = np.arange(len(days))
    w = 0.35

    ax = axes[0]
    ax.bar(x - w / 2, hr_x, w, label="Peritos", color="#9C27B0", alpha=0.85)
    ax.bar(x + w / 2, hr_j, w, label="Juniores", color="#CE93D8", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(days)
    ax.set_title("Composicao de Staff"); ax.set_ylabel("Funcionarios")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(x, [p * 100 for p in pr], color="#7B1FA2", alpha=0.85)
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
        f"{store.capitalize()} [Monte Carlo] — O1: Lucro Maximo ${lucro:,.0f}",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"{store}_mc_best_plan.png")
    plt.savefig(path, dpi=150)
    plt.close()
    log.info("Grafico do plano otimo guardado -> %s", path)


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("\n" + "=" * 65)
    print("  O1 — Otimizacao Individual por Loja (MONTE CARLO)")
    print(f"  Amostras: {N_SAMPLES:,} | Seed: 42")
    print("=" * 65 + "\n")

    all_summaries = []

    for store in STORES:
        inputs = FORECAST_INPUTS[store]
        log.info("=== A otimizar: %s ===", store.upper())

        best_solution, best_score, all_scores, history = monte_carlo(
            store=store,
            forecast_customers=inputs["customers"],
            forecast_is_weekend=inputs["is_weekend"],
            n_samples=N_SAMPLES,
            seed=42,
        )

        lucro_real = -best_score
        lucros_todos = [-x for x in all_scores]
        plan = decode_solution(best_solution, inputs["customers"], inputs["is_weekend"])

        total_staff = sum(d["total_staff"] for d in plan)
        avg_discount = float(np.mean([d["pr"] for d in plan]))

        summary = {
            "store":          store,
            "algoritmo":      "Monte Carlo",
            "lucro_maximo":   round(lucro_real, 2),
            "lucro_medio":    round(float(np.mean(lucros_todos)), 2),
            "lucro_std":      round(float(np.std(lucros_todos)), 2),
            "total_staff":    total_staff,
            "avg_discount":   round(avg_discount * 100, 2),
            "n_samples":      N_SAMPLES,
        }
        all_summaries.append(summary)

        print(f"\n{'-' * 60}")
        print(f"  {store.upper()} — Lucro Maximo: ${lucro_real:,.2f}")
        print(f"  Lucro Medio: ${np.mean(lucros_todos):,.2f}  |  Std: ${np.std(lucros_todos):,.2f}")
        print(f"  Staff Total Semanal: {total_staff}  |  Desconto Medio: {avg_discount * 100:.1f}%")
        print(f"\n  Plano Semanal Otimo (Monte Carlo):")
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
        plan_df.to_csv(os.path.join(OUT_DIR, f"{store}_mc_best_plan.csv"), index=False)

        scores_df = pd.DataFrame({
            "sample": range(len(all_scores)),
            "lucro": lucros_todos,
        })
        scores_df.to_csv(os.path.join(OUT_DIR, f"{store}_mc_all_scores.csv"), index=False)

        conv_df = pd.DataFrame({
            "sample": range(len(history)),
            "best_lucro": [-x for x in history],
        })
        conv_df.to_csv(os.path.join(OUT_DIR, f"{store}_mc_convergence.csv"), index=False)

        # --- Graficos ---
        plot_convergence(store, history)
        plot_distribution(store, all_scores, best_score)
        plot_best_plan(store, plan, lucro_real)

    # --- Resumo geral ---
    summary_df = pd.DataFrame(all_summaries)
    summary_path = os.path.join(OUT_DIR, "monte_carlo_summary.csv")
    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 65)
    print("  RESUMO — O1: MONTE CARLO")
    print("=" * 65)
    print(summary_df[["store", "lucro_maximo", "lucro_medio", "lucro_std", "avg_discount"]].to_string(index=False))
    print(f"\n  Resultados guardados em: {os.path.abspath(OUT_DIR)}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
