"""
run_individual_optimization.py — Tarefa 1: Otimização Individual por Loja
                                  (Maximizar Lucro Base — Mono-Objetivo)

Algoritmo: Hill Climbing com Random Restarts
Objetivo único: Maximizar o lucro semanal (f1) para cada loja individualmente.

Resultados guardados em: results/03_Optimization/individual/
"""

import sys
import os
import logging
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.dont_write_bytecode = True
sys.path.insert(0, "src")

from utils.profit_logic import optimize_weekly_wrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("TIAPOSE.IndivOpt")

# ---------------------------------------------------------------------------
# Previsões — última semana observada (2014-06-08 Dom → 2014-06-14 Sáb)
# Dias: Dom, Seg, Ter, Qua, Qui, Sex, Sáb
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

DAY_LABELS = ["Dom", "Seg", "Ter", "Qua", "Qui", "Sex", "Sáb"]
STORES = list(FORECAST_INPUTS.keys())
OUT_DIR = "results/03_Optimization/individual"
os.makedirs(OUT_DIR, exist_ok=True)


# ===========================================================================
# Hill Climbing com Random Restarts
# ===========================================================================

def generate_initial_solution():
    """Gera uma solução inicial aleatória (21 variáveis de decisão)."""
    solution = []
    for _ in range(7):
        solution.extend([
            random.uniform(0.0, 0.30),   # desconto pr
            random.randint(0, 15),        # staff perito hr_x
            random.randint(0, 15),        # staff junior hr_j
        ])
    return np.array(solution, dtype=float)


def generate_neighbor(solution):
    """Gera uma solução vizinha com uma perturbação aleatória."""
    neighbor = solution.copy()
    index = random.randint(0, len(solution) - 1)

    if index % 3 == 0:
        # Variável contínua (desconto): perturbação suave
        neighbor[index] = max(0.0, min(0.30, neighbor[index] + random.uniform(-0.05, 0.05)))
    else:
        # Variável inteira (staff): ±1 ou ±2
        delta = random.choice([-2, -1, 1, 2])
        neighbor[index] = max(0, min(15, neighbor[index] + delta))

    return neighbor


def evaluate_profit(solution, store, forecast_customers, forecast_is_weekend):
    """Avalia o lucro (quanto mais negativo o f1, maior o lucro real)."""
    f1, _, _ = optimize_weekly_wrapper(solution, store, forecast_customers, forecast_is_weekend)
    return f1  # Negativo do lucro — minimizar == maximizar lucro


def hill_climbing_with_restarts(
    store, forecast_customers, forecast_is_weekend,
    iterations=2000, n_restarts=10, seed=42,
):
    """
    Hill Climbing com Random Restarts para maximizar lucro mono-objetivo.

    Executa n_restarts corridas independentes e devolve a melhor solução global.
    """
    random.seed(seed)
    np.random.seed(seed)

    global_best_solution = None
    global_best_score = float("inf")
    all_histories = []

    for restart in range(n_restarts):
        current_solution = generate_initial_solution()
        best_score = evaluate_profit(current_solution, store, forecast_customers, forecast_is_weekend)
        best_solution = current_solution.copy()

        history = [best_score]

        for _ in range(iterations):
            neighbor = generate_neighbor(best_solution)
            score = evaluate_profit(neighbor, store, forecast_customers, forecast_is_weekend)
            if score < best_score:
                best_solution = neighbor
                best_score = score
            history.append(best_score)

        all_histories.append(history)
        log.info(
            "  Restart %2d/%d | Lucro: EUR %.0f",
            restart + 1, n_restarts, -best_score,
        )

        if best_score < global_best_score:
            global_best_score = best_score
            global_best_solution = best_solution.copy()

    # Escolher o histórico do melhor restart para o gráfico de convergência
    best_restart_idx = np.argmin([h[-1] for h in all_histories])
    best_history = all_histories[best_restart_idx]

    return global_best_solution, global_best_score, best_history, all_histories


def decode_solution(solution, forecast_customers, forecast_is_weekend):
    """Descodifica o vetor de 21 variáveis num plano semanal legível."""
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
# Gráficos
# ===========================================================================

def plot_convergence(store, best_history, all_histories):
    """Gráfico de convergência do Hill Climbing (melhor restart + todos)."""
    fig, ax = plt.subplots(figsize=(10, 5))

    # Todas as corridas em cinza claro
    for i, h in enumerate(all_histories):
        lucro_h = [-x for x in h]
        ax.plot(lucro_h, color="lightgrey", alpha=0.5, linewidth=0.8)

    # Melhor corrida em destaque
    best_lucro = [-x for x in best_history]
    ax.plot(best_lucro, color="#2196F3", linewidth=2.0, label="Melhor Restart")

    ax.set_xlabel("Iterações", fontsize=11)
    ax.set_ylabel("Lucro (€)", fontsize=11)
    ax.set_title(
        f"Convergência Hill Climbing — {store.capitalize()} (Tarefa 1: Max Lucro)",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUT_DIR, f"{store}_convergence.png")
    plt.savefig(path, dpi=150)
    plt.close()
    log.info("Gráfico de convergência guardado → %s", path)


def plot_best_plan(store, plan, lucro):
    """Gráfico do plano semanal ótimo (composição staff + desconto)."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    days = [d["day_label"] for d in plan]
    hr_x = [d["hr_x"] for d in plan]
    hr_j = [d["hr_j"] for d in plan]
    pr = [d["pr"] for d in plan]
    total = [d["total_staff"] for d in plan]

    x = np.arange(len(days))
    w = 0.35

    # --- Staff composição ---
    ax = axes[0]
    ax.bar(x - w / 2, hr_x, w, label="Peritos", color="#2196F3", alpha=0.85)
    ax.bar(x + w / 2, hr_j, w, label="Juniores", color="#FF9800", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(days)
    ax.set_title("Composição de Staff")
    ax.set_ylabel("Funcionários")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # --- Desconto ---
    ax = axes[1]
    ax.bar(x, [p * 100 for p in pr], color="#4CAF50", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(days)
    ax.set_title("Desconto Diário (%)")
    ax.set_ylabel("Desconto (%)")
    ax.set_ylim(0, 35)
    ax.grid(axis="y", alpha=0.3)

    # --- Total staff ---
    ax = axes[2]
    colors = ["#F44336" if t > 8 else "#4CAF50" for t in total]
    ax.bar(x, total, color=colors, alpha=0.85)
    ax.axhline(8, color="black", ls="--", lw=1.2, label="Limite útil (8)")
    ax.set_xticks(x)
    ax.set_xticklabels(days)
    ax.set_title("Staff Total por Dia")
    ax.set_ylabel("Total")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"{store.capitalize()} — Tarefa 1: Lucro Máximo €{lucro:,.0f}",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"{store}_best_plan.png")
    plt.savefig(path, dpi=150)
    plt.close()
    log.info("Gráfico de plano ótimo guardado → %s", path)


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("\n" + "=" * 65)
    print("  TIAPOSE - TAREFA 1: Otimizacao Individual por Loja")
    print("  Objetivo: Maximizar Lucro Base (Mono-Objetivo)")
    print("  Algoritmo: Hill Climbing com Random Restarts")
    print("=" * 65 + "\n")

    all_summaries = []

    for store in STORES:
        inputs = FORECAST_INPUTS[store]
        log.info("=== A otimizar: %s ===", store.upper())

        best_solution, best_score, best_history, all_histories = hill_climbing_with_restarts(
            store=store,
            forecast_customers=inputs["customers"],
            forecast_is_weekend=inputs["is_weekend"],
            iterations=2000,
            n_restarts=10,
            seed=42,
        )

        lucro_real = -best_score
        plan = decode_solution(best_solution, inputs["customers"], inputs["is_weekend"])

        # --- Calcular métricas ---
        total_staff = sum(d["total_staff"] for d in plan)
        avg_discount = np.mean([d["pr"] for d in plan])

        summary = {
            "store":          store,
            "lucro_maximo":   round(lucro_real, 2),
            "total_staff":    total_staff,
            "avg_discount":   round(avg_discount * 100, 2),
            "restarts":       10,
            "iterations":     2000,
        }
        all_summaries.append(summary)

        # --- Imprimir tabela ---
        print(f"\n{'-' * 60}")
        print(f"  {store.upper()} - Lucro Maximo: EUR {lucro_real:,.2f}")
        print(f"{'-' * 60}")
        print(f"  Staff Total Semanal: {total_staff}")
        print(f"  Desconto Medio:      {avg_discount * 100:.1f}%")
        print(f"\n  Plano Semanal Otimo:")
        print(f"  {'Dia':<5} {'Desconto':>9} {'Peritos':>8} {'Juniores':>9} {'Total':>6} {'Clientes':>9} {'FdS':>5}")
        print(f"  {'-' * 56}")
        for d in plan:
            fds = "Sim" if d["is_weekend"] else "Nao"
            print(
                f"  {d['day_label']:<5} {d['pr'] * 100:>8.1f}%"
                f" {d['hr_x']:>8} {d['hr_j']:>9} {d['total_staff']:>6}"
                f" {d['customers']:>9} {fds:>5}"
            )

        # --- Guardar CSV detalhado por loja ---
        plan_df = pd.DataFrame(plan)
        plan_df.to_csv(os.path.join(OUT_DIR, f"{store}_best_plan.csv"), index=False)

        # --- Guardar histórico de convergência ---
        conv_df = pd.DataFrame({
            "iteration": range(len(best_history)),
            "lucro": [-x for x in best_history],
        })
        conv_df.to_csv(os.path.join(OUT_DIR, f"{store}_convergence.csv"), index=False)

        # --- Gráficos ---
        plot_convergence(store, best_history, all_histories)
        plot_best_plan(store, plan, lucro_real)

    # --- Resumo geral ---
    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(os.path.join(OUT_DIR, "optimization_summary.csv"), index=False)

    print("\n" + "=" * 65)
    print("  RESUMO - TAREFA 1: OTIMIZACAO INDIVIDUAL (MAX LUCRO)")
    print("=" * 65)
    print(summary_df.to_string(index=False))
    print(f"\n  Resultados guardados em: {os.path.abspath(OUT_DIR)}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
