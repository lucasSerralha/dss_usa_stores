"""
run_allocation_optimization_death_penalty.py - Tarefa 2 / Objetivo 2:
    Alocacao Conjunta de Lojas com Teto de 10.000 Unidades de Venda

Maximizar o lucro TOTAL das 4 lojas, sujeito a:
    sum(unidades_vendidas_todas_lojas_semana) <= 10.000

Algoritmo: Hill Climbing com Random Restarts + DEATH PENALTY
    - Soluções inviáveis (units > 10.000) são COMPLETAMENTE rejeitadas (score = +inf)
    - Diferente da Penalty Function, não existe coeficiente de penalização
    - O algoritmo nunca aceita soluções inviáveis

Vetor de decisao: 84 variaveis (21 por loja x 4 lojas)
Resultados guardados em: results/03_Optimization/allocation_death_penalty/
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
    ELASTICITY_K,
    PROFIT_SCALE,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("TIAPOSE.DeathPenalty")

# ---------------------------------------------------------------------------
# Previsoes - ultima semana observada
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

DAY_LABELS   = ["Dom", "Seg", "Ter", "Qua", "Qui", "Sex", "Sab"]
STORES       = list(FORECAST_INPUTS.keys())
N_STORES     = len(STORES)
N_DAYS       = 7
N_VARS_PER_STORE = 21       # 7 dias x 3 variaveis (pr, hr_x, hr_j)
N_VARS_TOTAL     = N_STORES * N_VARS_PER_STORE  # 84

UNIT_CAP = 10_000           # Teto de unidades de venda

OUT_DIR = "results/03_Optimization/allocation_death_penalty"
os.makedirs(OUT_DIR, exist_ok=True)


# ===========================================================================
# Funcoes de avaliacao
# ===========================================================================

def evaluate_store(store, decision_vars_21, forecast_customers, forecast_is_weekend):
    """
    Avalia uma loja: retorna lucro, total de unidades e plano detalhado.
    """
    weekly_plan  = []
    total_units  = 0
    daily_details = []

    for i in range(N_DAYS):
        pr_raw   = decision_vars_21[i * 3]
        hr_x_raw = decision_vars_21[i * 3 + 1]
        hr_j_raw = decision_vars_21[i * 3 + 2]

        pr_clean  = max(0.0, min(0.30, pr_raw))
        hr_x_clean = max(0, int(round(hr_x_raw)))
        hr_j_clean = max(0, int(round(hr_j_raw)))

        effective_customers = int(round(
            forecast_customers[i] * (1 + ELASTICITY_K * pr_clean)
        ))

        metrics = calculate_daily_metrics(
            store=store,
            is_weekend=forecast_is_weekend[i],
            customers=effective_customers,
            pr=pr_clean,
            hr_x=hr_x_clean,
            hr_j=hr_j_clean,
        )

        day_units    = metrics["units_x"] + metrics["units_j"]
        total_units += day_units

        day_data = {
            "is_weekend": forecast_is_weekend[i],
            "customers":  effective_customers,
            "pr":   pr_clean,
            "hr_x": hr_x_clean,
            "hr_j": hr_j_clean,
        }
        weekly_plan.append(day_data)

        daily_details.append({
            "day":                  i,
            "day_label":            DAY_LABELS[i],
            "pr":                   round(pr_clean, 4),
            "hr_x":                 hr_x_clean,
            "hr_j":                 hr_j_clean,
            "total_staff":          hr_x_clean + hr_j_clean,
            "customers_forecast":   forecast_customers[i],
            "customers_effective":  effective_customers,
            "units":                day_units,
            "is_weekend":           forecast_is_weekend[i],
        })

    profit = calculate_weekly_profit(store, weekly_plan)
    return profit, total_units, daily_details


def evaluate_multi_store(full_solution):
    """
    Avalia a solucao conjunta (84 variaveis) para as 4 lojas.
    Retorna (total_profit, total_units, per_store_results).
    """
    total_profit = 0.0
    total_units  = 0
    per_store    = {}

    for s_idx, store in enumerate(STORES):
        start      = s_idx * N_VARS_PER_STORE
        store_vars = full_solution[start: start + N_VARS_PER_STORE]
        inputs     = FORECAST_INPUTS[store]

        profit, units, details = evaluate_store(
            store, store_vars,
            inputs["customers"], inputs["is_weekend"],
        )
        total_profit += profit
        total_units  += units
        per_store[store] = {"profit": profit, "units": units, "details": details}

    return total_profit, total_units, per_store


def objective_death_penalty(full_solution):
    """
    Funcao objetivo com DEATH PENALTY.

    Se a solucao exceder 10.000 unidades -> REJEITADA (score = +inf).
    Caso contrario -> minimizar -lucro (ou seja, maximizar lucro).

    Nunca aceita solucoes inviáveis — sem coeficiente de penalizacao.
    """
    total_profit, total_units, per_store = evaluate_multi_store(full_solution)

    if total_units > UNIT_CAP:
        # Death Penalty: solucao morta — descartada completamente
        return float("inf"), total_profit, total_units, per_store

    return -total_profit, total_profit, total_units, per_store


# ===========================================================================
# Geracao de solucoes
# ===========================================================================

def generate_initial_solution():
    """
    Gera solucao inicial conservadora: staff baixo para garantir viabilidade.
    hr_x in [0, 2], hr_j in [0, 1] por dia-loja (evita exceder 10.000 units).
    """
    solution = []
    for _ in range(N_STORES):
        for _ in range(N_DAYS):
            solution.extend([
                random.uniform(0.0, 0.15),   # desconto conservador
                random.randint(0, 2),         # peritos: maximo 2
                random.randint(0, 1),         # juniores: maximo 1
            ])
    return np.array(solution, dtype=float)


def generate_feasible_initial_solution(max_attempts=200):
    """
    Gera solucao inicial garantidamente viavel (units <= UNIT_CAP).
    Tenta ate max_attempts vezes; usa solucao minima como fallback.
    """
    for _ in range(max_attempts):
        sol = generate_initial_solution()
        _, total_units, _ = evaluate_multi_store(sol)
        if total_units <= UNIT_CAP:
            return sol

    # Fallback: solucao minima (1 perito, 0 juniores, 5% desconto)
    log.warning("Fallback para solucao minima garantidamente viavel.")
    solution = []
    for _ in range(N_STORES):
        for _ in range(N_DAYS):
            solution.extend([0.05, 1, 0])
    return np.array(solution, dtype=float)


def generate_neighbor(solution):
    """Vizinho com perturbacao aleatoria numa variavel."""
    neighbor  = solution.copy()
    index     = random.randint(0, len(solution) - 1)
    var_type  = index % 3  # 0=desconto, 1=hr_x, 2=hr_j

    if var_type == 0:
        neighbor[index] = max(0.0, min(0.30,
            neighbor[index] + random.uniform(-0.05, 0.05)))
    else:
        delta = random.choice([-2, -1, 1, 2])
        neighbor[index] = max(0, min(15, neighbor[index] + delta))

    return neighbor


def generate_smart_neighbor(solution, total_units):
    """
    Vizinho inteligente:
    - Se perto do teto (>= 90%), tende a reduzir staff/desconto.
    - Caso contrario, explora aumentar staff/desconto para mais lucro.
    """
    neighbor = solution.copy()
    near_cap = total_units >= UNIT_CAP * 0.90

    if near_cap and random.random() < 0.75:
        # Estrategia de reducao
        store_idx = random.randint(0, N_STORES - 1)
        day_idx   = random.randint(0, N_DAYS - 1)
        base      = store_idx * N_VARS_PER_STORE + day_idx * 3

        if random.random() < 0.4:
            neighbor[base] = max(0.0, neighbor[base] - random.uniform(0.02, 0.08))
        else:
            staff_var = base + random.choice([1, 2])
            neighbor[staff_var] = max(0, neighbor[staff_var] - random.choice([1, 2]))
    elif not near_cap and random.random() < 0.5:
        # Estrategia de expansao: mais staff gera mais lucro
        store_idx = random.randint(0, N_STORES - 1)
        day_idx   = random.randint(0, N_DAYS - 1)
        base      = store_idx * N_VARS_PER_STORE + day_idx * 3
        staff_var = base + random.choice([1, 2])
        neighbor[staff_var] = min(15, neighbor[staff_var] + random.choice([1, 2]))
    else:
        neighbor = generate_neighbor(solution)

    return neighbor


# ===========================================================================
# Hill Climbing com Death Penalty
# ===========================================================================

def hill_climbing_death_penalty(iterations=4000, n_restarts=20, seed=42):
    """
    Hill Climbing com Random Restarts e Death Penalty.

    Regra de aceitacao:
      - Vizinho viavel e melhor que atual -> ACEITAR
      - Vizinho inviavel (units > 10.000)  -> REJEITAR (death penalty)
      - Vizinho viavel mas pior            -> REJEITAR (hill climbing puro)
    """
    random.seed(seed)
    np.random.seed(seed)

    global_best_solution = None
    global_best_score    = float("inf")
    global_best_profit   = 0.0
    global_best_units    = 0
    global_best_per_store = None
    all_histories        = []

    rejected_by_death = 0
    total_neighbors   = 0

    for restart in range(n_restarts):
        current_solution = generate_feasible_initial_solution()
        score, profit, units, per_store = objective_death_penalty(current_solution)

        # Seguranca: se mesmo o inicial for inviavel, forcar fallback
        if score == float("inf"):
            log.warning("Restart %d: solucao inicial inviavel, a gerar nova...", restart + 1)
            current_solution = generate_feasible_initial_solution(max_attempts=500)
            score, profit, units, per_store = objective_death_penalty(current_solution)

        best_score    = score
        best_solution = current_solution.copy()
        best_profit   = profit
        best_units    = units
        best_per_store = per_store

        history = {
            "score":  [score if score != float("inf") else 0],
            "profit": [profit],
            "units":  [units],
        }

        restart_rejected = 0

        for it in range(iterations):
            neighbor = generate_smart_neighbor(best_solution, best_units)
            n_score, n_profit, n_units, n_per_store = objective_death_penalty(neighbor)
            total_neighbors += 1

            if n_score == float("inf"):
                # Death Penalty: vizinho inviavel -> rejeitado
                restart_rejected += 1
                rejected_by_death += 1
            elif n_score < best_score:
                best_solution  = neighbor
                best_score     = n_score
                best_profit    = n_profit
                best_units     = n_units
                best_per_store = n_per_store

            history["score"].append(best_score if best_score != float("inf") else 0)
            history["profit"].append(best_profit)
            history["units"].append(best_units)

        all_histories.append(history)
        feasible = "OK" if best_units <= UNIT_CAP else "EXCEDE"
        log.info(
            "  Restart %2d/%d | Lucro: EUR %.0f | Units: %d (%s) | Rejeitados: %d",
            restart + 1, n_restarts, best_profit, best_units, feasible, restart_rejected,
        )

        if best_score < global_best_score:
            global_best_score     = best_score
            global_best_solution  = best_solution.copy()
            global_best_profit    = best_profit
            global_best_units     = best_units
            global_best_per_store = best_per_store

    log.info(
        "Death Penalty Stats: %d/%d vizinhos rejeitados (%.1f%%)",
        rejected_by_death, total_neighbors,
        100 * rejected_by_death / max(1, total_neighbors),
    )

    best_restart_idx = np.argmin([h["profit"][-1] * -1 for h in all_histories])
    best_history     = all_histories[best_restart_idx]

    return (
        global_best_solution,
        global_best_score,
        global_best_profit,
        global_best_units,
        global_best_per_store,
        best_history,
        all_histories,
        rejected_by_death,
        total_neighbors,
    )


# ===========================================================================
# Graficos
# ===========================================================================

def plot_convergence(best_history, all_histories):
    """Convergencia: lucro e unidades ao longo das iteracoes."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    ax = axes[0]
    for h in all_histories:
        ax.plot(h["profit"], color="lightgrey", alpha=0.5, linewidth=0.8)
    ax.plot(best_history["profit"], color="#2196F3", linewidth=2.0, label="Melhor Restart")
    ax.set_xlabel("Iteracoes")
    ax.set_ylabel("Lucro Total (EUR)")
    ax.set_title("Convergencia - Lucro Total", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    ax = axes[1]
    for h in all_histories:
        ax.plot(h["units"], color="lightgrey", alpha=0.5, linewidth=0.8)
    ax.plot(best_history["units"], color="#4CAF50", linewidth=2.0, label="Melhor Restart")
    ax.axhline(UNIT_CAP, color="red", ls="--", lw=2.0, label=f"Teto ({UNIT_CAP:,})")
    ax.set_xlabel("Iteracoes")
    ax.set_ylabel("Unidades Totais")
    ax.set_title("Convergencia - Unidades Vendidas", fontweight="bold")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        "O2 - Hill Climbing com Death Penalty (Teto 10.000 Unidades)",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "convergence.png")
    plt.savefig(path, dpi=150)
    plt.close()
    log.info("Grafico de convergencia -> %s", path)


def plot_allocation_summary(per_store):
    """Resumo: lucro e unidades por loja."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    stores       = list(per_store.keys())
    profits      = [per_store[s]["profit"] for s in stores]
    units        = [per_store[s]["units"]  for s in stores]
    store_labels = [s.capitalize() for s in stores]
    colors       = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]

    ax = axes[0]
    bars = ax.bar(store_labels, profits, color=colors, alpha=0.85, edgecolor="k")
    ax.set_title("Lucro por Loja (EUR)", fontweight="bold")
    ax.set_ylabel("Lucro (EUR)")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, profits):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
                f"{val:,.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax = axes[1]
    bars = ax.bar(store_labels, units, color=colors, alpha=0.85, edgecolor="k")
    ax.axhline(UNIT_CAP / N_STORES, color="red", ls="--", lw=1.5,
               label=f"Teto medio ({UNIT_CAP//N_STORES:,}/loja)")
    ax.set_title("Unidades Vendidas por Loja", fontweight="bold")
    ax.set_ylabel("Unidades")
    ax.legend(fontsize=8)
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, units):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 30,
                f"{val:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    ax = axes[2]
    ax.pie(units, labels=store_labels, colors=colors, autopct="%1.1f%%",
           startangle=90, textprops={"fontsize": 10})
    ax.set_title(
        f"Distribuicao de Unidades\n(Total: {sum(units):,} / {UNIT_CAP:,})",
        fontweight="bold",
    )

    fig.suptitle("O2 - Alocacao Otima com Death Penalty", fontsize=13, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "allocation_summary.png")
    plt.savefig(path, dpi=150)
    plt.close()
    log.info("Grafico de alocacao -> %s", path)


def plot_store_plan(store, details, profit, units):
    """Plano semanal de uma loja."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    days      = [d["day_label"] for d in details]
    hr_x      = [d["hr_x"]     for d in details]
    hr_j      = [d["hr_j"]     for d in details]
    pr        = [d["pr"]       for d in details]
    day_units = [d["units"]    for d in details]
    x = np.arange(len(days))
    w = 0.35

    ax = axes[0]
    ax.bar(x - w/2, hr_x, w, label="Peritos",   color="#2196F3", alpha=0.85)
    ax.bar(x + w/2, hr_j, w, label="Juniores",  color="#FF9800", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(days)
    ax.set_title("Composicao de Staff")
    ax.set_ylabel("Funcionarios")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(x, [p * 100 for p in pr], color="#4CAF50", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(days)
    ax.set_title("Desconto Diario (%)")
    ax.set_ylabel("Desconto (%)"); ax.set_ylim(0, 35)
    ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    ax.bar(x, day_units, color="#9C27B0", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(days)
    ax.set_title("Unidades Vendidas por Dia")
    ax.set_ylabel("Unidades"); ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"{store.capitalize()} - Lucro: EUR {profit:,.0f} | Unidades: {units:,}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"{store}_plan.png")
    plt.savefig(path, dpi=150)
    plt.close()
    log.info("Grafico de plano -> %s", path)


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("\n" + "=" * 70)
    print("  TIAPOSE - TAREFA 2 / OBJETIVO 2")
    print("  Alocacao Conjunta: Maximizar Lucro com Teto de 10.000 Unidades")
    print("  Algoritmo: Hill Climbing com Random Restarts + DEATH PENALTY")
    print("=" * 70 + "\n")

    log.info("=== A iniciar otimizacao conjunta com Death Penalty (4 lojas, 84 vars) ===")

    (
        best_solution,
        best_score,
        best_profit,
        best_units,
        best_per_store,
        best_history,
        all_histories,
        rejected_by_death,
        total_neighbors,
    ) = hill_climbing_death_penalty(
        iterations=4000,
        n_restarts=20,
        seed=42,
    )

    feasible = best_units <= UNIT_CAP

    print("\n" + "=" * 70)
    print("  RESULTADOS - OBJETIVO 2: ALOCACAO COM DEATH PENALTY")
    print("=" * 70)
    print(f"\n  Lucro Total:     EUR {best_profit:>12,.2f}")
    print(f"  Unidades Totais: {best_units:>12,} / {UNIT_CAP:,}")
    print(f"  Restricao:       {'SATISFEITA (OK)' if feasible else 'VIOLADA (X)'}")
    print(f"\n  [Death Penalty] Vizinhos rejeitados: {rejected_by_death:,} / {total_neighbors:,} "
          f"({100*rejected_by_death/max(1,total_neighbors):.1f}%)")

    all_summaries = []

    for store in STORES:
        data    = best_per_store[store]
        details = data["details"]
        profit  = data["profit"]
        units   = data["units"]
        pct     = (units / best_units * 100) if best_units > 0 else 0

        summary = {
            "store":        store,
            "lucro":        round(profit, 2),
            "unidades":     units,
            "pct_unidades": round(pct, 1),
            "total_staff":  sum(d["total_staff"] for d in details),
            "avg_discount": round(np.mean([d["pr"] for d in details]) * 100, 2),
        }
        all_summaries.append(summary)

        print(f"\n  {'-' * 60}")
        print(f"  {store.upper()}")
        print(f"  {'-' * 60}")
        print(f"    Lucro:     EUR {profit:>10,.2f}")
        print(f"    Unidades:  {units:>10,} ({pct:.1f}% do total)")
        print(f"    Staff:     {sum(d['total_staff'] for d in details)}")
        print(f"\n    {'Dia':<5} {'Desc':>6} {'PrtX':>5} {'JnrJ':>5} {'Staff':>6} "
              f"{'CltPrev':>8} {'CltEff':>7} {'Units':>7} {'FdS':>5}")
        print(f"    {'-' * 58}")
        for d in details:
            fds = "Sim" if d["is_weekend"] else "Nao"
            print(
                f"    {d['day_label']:<5} {d['pr']*100:>5.1f}%"
                f" {d['hr_x']:>5} {d['hr_j']:>5} {d['total_staff']:>6}"
                f" {d['customers_forecast']:>8} {d['customers_effective']:>7}"
                f" {d['units']:>7} {fds:>5}"
            )

        pd.DataFrame(details).to_csv(
            os.path.join(OUT_DIR, f"{store}_plan.csv"), index=False
        )
        plot_store_plan(store, details, profit, units)

    # CSV resumo
    summary_df = pd.DataFrame(all_summaries)
    total_row  = {
        "store":        "TOTAL",
        "lucro":        round(best_profit, 2),
        "unidades":     best_units,
        "pct_unidades": 100.0,
        "total_staff":  summary_df["total_staff"].sum(),
        "avg_discount": round(summary_df["avg_discount"].mean(), 2),
    }
    summary_df = pd.concat([summary_df, pd.DataFrame([total_row])], ignore_index=True)
    summary_df.to_csv(
        os.path.join(OUT_DIR, "allocation_summary_death_penalty.csv"), index=False
    )

    # CSV convergencia
    conv_df = pd.DataFrame({
        "iteration": range(len(best_history["score"])),
        "score":     best_history["score"],
        "profit":    best_history["profit"],
        "units":     best_history["units"],
    })
    conv_df.to_csv(os.path.join(OUT_DIR, "convergence.csv"), index=False)

    # Graficos gerais
    plot_convergence(best_history, all_histories)
    plot_allocation_summary(best_per_store)

    print("\n" + "=" * 70)
    print("  RESUMO FINAL - O2 DEATH PENALTY")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print(f"\n  Restricao de unidades: {'SATISFEITA (OK)' if feasible else 'VIOLADA (X)'}")
    print(f"  Resultados guardados em: {os.path.abspath(OUT_DIR)}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
