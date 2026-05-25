"""
run_allocation_optimization.py - Tarefa 2 / Objetivo 2:
    Alocacao Conjunta de Lojas com Teto de 10.000 Unidades de Venda

Maximizar o lucro TOTAL das 4 lojas, sujeito a:
    sum(unidades_vendidas_todas_lojas_semana) <= 10.000

Algoritmo: Hill Climbing com Random Restarts + Penalty Function
Vetor de decisao: 84 variaveis (21 por loja x 4 lojas)

Resultados guardados em: results/03_Optimization/allocation/
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

from utils.profit_logic import (
    calculate_daily_metrics,
    calculate_weekly_profit,
    STORE_PARAMS,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("TIAPOSE.AllocOpt")

# ---------------------------------------------------------------------------
# Previsoes - ultima semana observada (2014-06-08 Dom -> 2014-06-14 Sab)
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
N_STORES = len(STORES)
N_DAYS = 7
N_VARS_PER_STORE = 21   # 7 dias x 3 variaveis (pr, hr_x, hr_j)
N_VARS_TOTAL = N_STORES * N_VARS_PER_STORE  # 84

UNIT_CAP = 10_000  # Teto de unidades de venda
PENALTY_COEFF = 50  # Coeficiente de penalizacao por unidade excedente

OUT_DIR = "results_v2/03_Optimization/allocation"
os.makedirs(OUT_DIR, exist_ok=True)


# ===========================================================================
# Funcoes de avaliacao
# ===========================================================================

def evaluate_store(store, decision_vars_21, forecast_customers, forecast_is_weekend):
    """
    Avalia uma loja individual: retorna lucro, total de unidades e plano detalhado.

    Args:
        store: Nome da loja
        decision_vars_21: Vetor de 21 variaveis de decisao para esta loja
        forecast_customers: Lista de 7 previsoes de clientes
        forecast_is_weekend: Lista de 7 booleanos

    Returns:
        (profit, total_units, weekly_plan_details)
    """
    weekly_plan = []
    total_units = 0
    daily_details = []

    for i in range(N_DAYS):
        pr_raw = decision_vars_21[i * 3]
        hr_x_raw = decision_vars_21[i * 3 + 1]
        hr_j_raw = decision_vars_21[i * 3 + 2]

        pr_clean = max(0.0, min(0.30, pr_raw))
        hr_x_clean = max(0, int(round(hr_x_raw)))
        hr_j_clean = max(0, int(round(hr_j_raw)))

        # Clientes previstos sem elasticidade — conforme enunciado
        effective_customers = int(round(forecast_customers[i]))

        # Calcular metricas diarias (unidades, vendas, custos)
        metrics = calculate_daily_metrics(
            store=store,
            is_weekend=forecast_is_weekend[i],
            customers=effective_customers,
            pr=pr_clean,
            hr_x=hr_x_clean,
            hr_j=hr_j_clean,
        )

        day_units = metrics["units_x"] + metrics["units_j"]
        total_units += day_units

        day_data = {
            "is_weekend": forecast_is_weekend[i],
            "customers": effective_customers,
            "pr": pr_clean,
            "hr_x": hr_x_clean,
            "hr_j": hr_j_clean,
        }
        weekly_plan.append(day_data)

        daily_details.append({
            "day": i,
            "day_label": DAY_LABELS[i],
            "pr": round(pr_clean, 4),
            "hr_x": hr_x_clean,
            "hr_j": hr_j_clean,
            "total_staff": hr_x_clean + hr_j_clean,
            "customers_forecast": forecast_customers[i],
            "customers_effective": effective_customers,
            "units": day_units,
            "is_weekend": forecast_is_weekend[i],
        })

    profit = calculate_weekly_profit(store, weekly_plan)
    return profit, total_units, daily_details


def evaluate_multi_store(full_solution):
    """
    Avalia a solucao conjunta (84 variaveis) para as 4 lojas.

    Returns:
        (total_profit, total_units, per_store_results)
        per_store_results: dict[store] -> {profit, units, details}
    """
    total_profit = 0.0
    total_units = 0
    per_store = {}

    for s_idx, store in enumerate(STORES):
        start = s_idx * N_VARS_PER_STORE
        end = start + N_VARS_PER_STORE
        store_vars = full_solution[start:end]

        inputs = FORECAST_INPUTS[store]
        profit, units, details = evaluate_store(
            store, store_vars,
            inputs["customers"], inputs["is_weekend"],
        )

        total_profit += profit
        total_units += units
        per_store[store] = {"profit": profit, "units": units, "details": details}

    return total_profit, total_units, per_store


def objective_with_penalty(full_solution):
    """
    Funcao objetivo com penalizacao por exceder 10.000 unidades.
    Retorna score negativo (para minimizar = maximizar lucro).
    Penaliza fortemente quando unidades > UNIT_CAP.
    """
    total_profit, total_units, per_store = evaluate_multi_store(full_solution)

    # Penalizacao: quanto mais se excede o teto, maior a penalizacao
    if total_units > UNIT_CAP:
        excess = total_units - UNIT_CAP
        penalty = excess * PENALTY_COEFF
    else:
        penalty = 0.0

    # Score: minimizar (-lucro + penalizacao)
    score = -total_profit + penalty
    return score, total_profit, total_units, per_store


# ===========================================================================
# Hill Climbing adaptado para multi-loja (84 variaveis)
# ===========================================================================

def generate_initial_solution():
    """Gera solucao inicial aleatoria para as 4 lojas (84 variaveis)."""
    solution = []
    for _ in range(N_STORES):
        for _ in range(N_DAYS):
            solution.extend([
                random.uniform(0.0, 0.30),   # desconto pr
                random.randint(0, 10),        # staff perito hr_x (max reduzido)
                random.randint(0, 10),        # staff junior hr_j (max reduzido)
            ])
    return np.array(solution, dtype=float)


def generate_neighbor(solution):
    """Gera vizinho com perturbacao aleatoria numa variavel."""
    neighbor = solution.copy()
    index = random.randint(0, len(solution) - 1)

    var_type = index % 3  # 0=desconto, 1=hr_x, 2=hr_j
    if var_type == 0:
        # Desconto: perturbacao suave
        neighbor[index] = max(0.0, min(0.30, neighbor[index] + random.uniform(-0.05, 0.05)))
    else:
        # Staff: perturbacao +/- 1 ou 2
        delta = random.choice([-2, -1, 1, 2])
        neighbor[index] = max(0, min(15, neighbor[index] + delta))

    return neighbor


def generate_smart_neighbor(solution):
    """
    Vizinho inteligente: se as unidades excedem o teto, tende a reduzir
    desconto ou staff. Caso contrario, explora normalmente.
    """
    _, total_units, _ = evaluate_multi_store(solution)
    neighbor = solution.copy()

    if total_units > UNIT_CAP and random.random() < 0.7:
        # Estrategia de reducao: escolher loja aleatoria e reduzir desconto ou staff
        store_idx = random.randint(0, N_STORES - 1)
        day_idx = random.randint(0, N_DAYS - 1)
        base = store_idx * N_VARS_PER_STORE + day_idx * 3

        action = random.choice(["reduce_discount", "reduce_staff"])
        if action == "reduce_discount":
            neighbor[base] = max(0.0, neighbor[base] - random.uniform(0.02, 0.08))
        else:
            staff_var = base + random.choice([1, 2])
            neighbor[staff_var] = max(0, neighbor[staff_var] - random.choice([1, 2]))
    else:
        # Exploracao normal
        neighbor = generate_neighbor(solution)

    return neighbor


def hill_climbing_allocation(iterations=3000, n_restarts=15, seed=42):
    """
    Hill Climbing com Random Restarts para otimizacao conjunta.
    Maximiza lucro total com restricao de 10.000 unidades.
    """
    random.seed(seed)
    np.random.seed(seed)

    global_best_solution = None
    global_best_score = float("inf")
    global_best_profit = 0.0
    global_best_units = 0
    global_best_per_store = None
    all_histories = []

    for restart in range(n_restarts):
        current_solution = generate_initial_solution()
        score, profit, units, per_store = objective_with_penalty(current_solution)
        best_score = score
        best_solution = current_solution.copy()
        best_profit = profit
        best_units = units
        best_per_store = per_store

        history = {"score": [score], "profit": [profit], "units": [units]}

        for it in range(iterations):
            neighbor = generate_smart_neighbor(best_solution)
            n_score, n_profit, n_units, n_per_store = objective_with_penalty(neighbor)

            if n_score < best_score:
                best_solution = neighbor
                best_score = n_score
                best_profit = n_profit
                best_units = n_units
                best_per_store = n_per_store

            history["score"].append(best_score)
            history["profit"].append(best_profit)
            history["units"].append(best_units)

        all_histories.append(history)

        feasible = "OK" if best_units <= UNIT_CAP else "EXCEDE"
        log.info(
            "  Restart %2d/%d | Lucro: EUR %.0f | Units: %d (%s)",
            restart + 1, n_restarts, best_profit, best_units, feasible,
        )

        if best_score < global_best_score:
            global_best_score = best_score
            global_best_solution = best_solution.copy()
            global_best_profit = best_profit
            global_best_units = best_units
            global_best_per_store = best_per_store

    # Melhor historico
    best_restart_idx = np.argmin([h["score"][-1] for h in all_histories])
    best_history = all_histories[best_restart_idx]

    return (
        global_best_solution,
        global_best_score,
        global_best_profit,
        global_best_units,
        global_best_per_store,
        best_history,
        all_histories,
    )


# ===========================================================================
# Graficos
# ===========================================================================

def plot_convergence(best_history, all_histories):
    """Grafico de convergencia: lucro e unidades ao longo das iteracoes."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # --- Lucro ---
    ax = axes[0]
    for h in all_histories:
        ax.plot(h["profit"], color="lightgrey", alpha=0.5, linewidth=0.8)
    ax.plot(best_history["profit"], color="#2196F3", linewidth=2.0, label="Melhor Restart")
    ax.set_xlabel("Iteracoes", fontsize=11)
    ax.set_ylabel("Lucro Total (EUR)", fontsize=11)
    ax.set_title("Convergencia - Lucro Total", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # --- Unidades ---
    ax = axes[1]
    for h in all_histories:
        ax.plot(h["units"], color="lightgrey", alpha=0.5, linewidth=0.8)
    ax.plot(best_history["units"], color="#4CAF50", linewidth=2.0, label="Melhor Restart")
    ax.axhline(UNIT_CAP, color="red", ls="--", lw=2.0, label=f"Teto ({UNIT_CAP:,})")
    ax.set_xlabel("Iteracoes", fontsize=11)
    ax.set_ylabel("Unidades Totais", fontsize=11)
    ax.set_title("Convergencia - Unidades Vendidas", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "Tarefa 2: Otimizacao Conjunta com Teto de 10.000 Unidades",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "convergence.png")
    plt.savefig(path, dpi=150)
    plt.close()
    log.info("Grafico de convergencia guardado -> %s", path)


def plot_allocation_summary(per_store):
    """Grafico resumo da alocacao: lucro e unidades por loja."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    stores = list(per_store.keys())
    profits = [per_store[s]["profit"] for s in stores]
    units = [per_store[s]["units"] for s in stores]
    store_labels = [s.capitalize() for s in stores]

    # --- Lucro por loja ---
    ax = axes[0]
    colors_profit = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]
    bars = ax.bar(store_labels, profits, color=colors_profit, alpha=0.85, edgecolor="k", linewidth=0.5)
    ax.set_title("Lucro por Loja (EUR)", fontsize=12, fontweight="bold")
    ax.set_ylabel("Lucro (EUR)")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, profits):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
                f"{val:,.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # --- Unidades por loja ---
    ax = axes[1]
    bars = ax.bar(store_labels, units, color=colors_profit, alpha=0.85, edgecolor="k", linewidth=0.5)
    ax.set_title("Unidades Vendidas por Loja", fontsize=12, fontweight="bold")
    ax.set_ylabel("Unidades")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, units):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{val:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # --- Proporcao de unidades (pie chart) ---
    ax = axes[2]
    ax.pie(units, labels=store_labels, colors=colors_profit, autopct="%1.1f%%",
           startangle=90, textprops={"fontsize": 10})
    ax.set_title(f"Distribuicao de Unidades\n(Total: {sum(units):,} / {UNIT_CAP:,})",
                 fontsize=12, fontweight="bold")

    fig.suptitle(
        "Tarefa 2: Alocacao Otima entre Lojas",
        fontsize=14, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "allocation_summary.png")
    plt.savefig(path, dpi=150)
    plt.close()
    log.info("Grafico de alocacao guardado -> %s", path)


def plot_store_plan(store, details, profit, units):
    """Grafico do plano semanal de cada loja."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    days = [d["day_label"] for d in details]
    hr_x = [d["hr_x"] for d in details]
    hr_j = [d["hr_j"] for d in details]
    pr = [d["pr"] for d in details]
    day_units = [d["units"] for d in details]

    x = np.arange(len(days))
    w = 0.35

    # --- Staff ---
    ax = axes[0]
    ax.bar(x - w / 2, hr_x, w, label="Peritos", color="#2196F3", alpha=0.85)
    ax.bar(x + w / 2, hr_j, w, label="Juniores", color="#FF9800", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(days)
    ax.set_title("Composicao de Staff")
    ax.set_ylabel("Funcionarios")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    # --- Desconto ---
    ax = axes[1]
    ax.bar(x, [p * 100 for p in pr], color="#4CAF50", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(days)
    ax.set_title("Desconto Diario (%)")
    ax.set_ylabel("Desconto (%)")
    ax.set_ylim(0, 35)
    ax.grid(axis="y", alpha=0.3)

    # --- Unidades por dia ---
    ax = axes[2]
    ax.bar(x, day_units, color="#9C27B0", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels(days)
    ax.set_title("Unidades Vendidas por Dia")
    ax.set_ylabel("Unidades")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"{store.capitalize()} - Lucro: EUR {profit:,.0f} | Unidades: {units:,}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"{store}_plan.png")
    plt.savefig(path, dpi=150)
    plt.close()
    log.info("Grafico de plano guardado -> %s", path)


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("\n" + "=" * 70)
    print("  TIAPOSE - TAREFA 2 / OBJETIVO 2")
    print("  Alocacao Conjunta: Maximizar Lucro com Teto de 10.000 Unidades")
    print("  Algoritmo: Hill Climbing com Random Restarts + Penalty Function")
    print("=" * 70 + "\n")

    log.info("=== A iniciar otimizacao conjunta (4 lojas, 84 variaveis) ===")

    (
        best_solution,
        best_score,
        best_profit,
        best_units,
        best_per_store,
        best_history,
        all_histories,
    ) = hill_climbing_allocation(
        iterations=3000,
        n_restarts=15,
        seed=42,
    )

    feasible = best_units <= UNIT_CAP

    # --- Imprimir resultados ---
    print("\n" + "=" * 70)
    print("  RESULTADOS - OBJETIVO 2: ALOCACAO CONJUNTA")
    print("=" * 70)
    print(f"\n  Lucro Total:     EUR {best_profit:>12,.2f}")
    print(f"  Unidades Totais: {best_units:>12,} / {UNIT_CAP:,}")
    print(f"  Restricao:       {'SATISFEITA' if feasible else 'VIOLADA'}")
    if not feasible:
        print(f"  Excesso:         {best_units - UNIT_CAP:,} unidades acima do teto")

    all_summaries = []

    for store in STORES:
        data = best_per_store[store]
        details = data["details"]
        profit = data["profit"]
        units = data["units"]

        pct = (units / best_units * 100) if best_units > 0 else 0

        summary = {
            "store": store,
            "lucro": round(profit, 2),
            "unidades": units,
            "pct_unidades": round(pct, 1),
            "total_staff": sum(d["total_staff"] for d in details),
            "avg_discount": round(np.mean([d["pr"] for d in details]) * 100, 2),
        }
        all_summaries.append(summary)

        print(f"\n  {'-' * 60}")
        print(f"  {store.upper()}")
        print(f"  {'-' * 60}")
        print(f"    Lucro:     EUR {profit:>10,.2f}")
        print(f"    Unidades:  {units:>10,} ({pct:.1f}% do total)")
        print(f"    Staff:     {sum(d['total_staff'] for d in details)}")
        print(f"\n    {'Dia':<5} {'Desc':>6} {'PrtX':>5} {'JnrJ':>5} {'Staff':>6} {'CltPrev':>8} {'CltEff':>7} {'Units':>7} {'FdS':>5}")
        print(f"    {'-' * 58}")
        for d in details:
            fds = "Sim" if d["is_weekend"] else "Nao"
            print(
                f"    {d['day_label']:<5} {d['pr']*100:>5.1f}%"
                f" {d['hr_x']:>5} {d['hr_j']:>5} {d['total_staff']:>6}"
                f" {d['customers_forecast']:>8} {d['customers_effective']:>7}"
                f" {d['units']:>7} {fds:>5}"
            )

        # --- CSV por loja ---
        plan_df = pd.DataFrame(details)
        plan_df.to_csv(os.path.join(OUT_DIR, f"{store}_plan.csv"), index=False)

        # --- Grafico por loja ---
        plot_store_plan(store, details, profit, units)

    # --- Resumo geral CSV ---
    summary_df = pd.DataFrame(all_summaries)

    # Adicionar linha de total
    total_row = {
        "store": "TOTAL",
        "lucro": round(best_profit, 2),
        "unidades": best_units,
        "pct_unidades": 100.0,
        "total_staff": summary_df["total_staff"].sum(),
        "avg_discount": round(summary_df["avg_discount"].mean(), 2),
    }
    summary_df = pd.concat([summary_df, pd.DataFrame([total_row])], ignore_index=True)
    summary_df.to_csv(os.path.join(OUT_DIR, "allocation_summary.csv"), index=False)

    # --- Convergencia CSV ---
    conv_df = pd.DataFrame({
        "iteration": range(len(best_history["score"])),
        "score": best_history["score"],
        "profit": best_history["profit"],
        "units": best_history["units"],
    })
    conv_df.to_csv(os.path.join(OUT_DIR, "convergence.csv"), index=False)

    # --- Graficos ---
    plot_convergence(best_history, all_histories)
    plot_allocation_summary(best_per_store)

    # --- Tabela final ---
    print("\n" + "=" * 70)
    print("  RESUMO - TAREFA 2: ALOCACAO CONJUNTA (TETO 10.000 UNIDADES)")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print(f"\n  Restricao de unidades: {'SATISFEITA' if feasible else 'VIOLADA'}")
    print(f"  Resultados guardados em: {os.path.abspath(OUT_DIR)}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
