"""
run_allocation_optimization.py - Tarefa 2 / Objetivo 2:
    Alocação Conjunta de Lojas via Problema da Mochila (Knapsack)

Este algoritmo otimiza o conjunto das 4 lojas garantindo que o teto de 10.000 
unidades não seja ultrapassado. A abordagem utiliza:
1. NSGA-II para gerar a Fronteira de Pareto (Lucro vs Unidades) para cada loja.
2. Programação Dinâmica para resolver o Multiple Choice Knapsack Problem (MCKP).
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

from utils.profit_logic_knapsack import (
    optimize_allocation_wrapper,
    calculate_daily_metrics,
    STORE_PARAMS,
    ELASTICITY_K,
)
from optimization.nsga2_model_knapsack import run_optimization
from optimization.knapsack_solver import solve_mckp

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("TIAPOSE.AllocOpt")

# ---------------------------------------------------------------------------
# Previsões - última semana observada
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

STORES = list(FORECAST_INPUTS.keys())
UNIT_CAP = 10_000

OUT_DIR = "results/02_Allocation_Optimization_Knapsack"
os.makedirs(OUT_DIR, exist_ok=True)


def plot_allocation_summary(results, total_units):
    """Grafico resumo da alocacao: lucro e unidades por loja."""
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    stores = list(results.keys())
    profits = [results[s]["profit"] for s in stores]
    units = [results[s]["units"] for s in stores]
    store_labels = [s.capitalize() for s in stores]

    colors = ["#2196F3", "#FF9800", "#4CAF50", "#9C27B0"]

    # --- Lucro por loja ---
    ax = axes[0]
    bars = ax.bar(store_labels, profits, color=colors, alpha=0.85, edgecolor="k")
    ax.set_title("Lucro por Loja (EUR)", fontweight="bold")
    ax.set_ylabel("Lucro (€)")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, profits):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 500,
                f"{val:,.0f}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # --- Unidades por loja ---
    ax = axes[1]
    bars = ax.bar(store_labels, units, color=colors, alpha=0.85, edgecolor="k")
    ax.set_title("Unidades Vendidas por Loja", fontweight="bold")
    ax.set_ylabel("Unidades")
    ax.grid(axis="y", alpha=0.3)
    for bar, val in zip(bars, units):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 50,
                f"{val:,}", ha="center", va="bottom", fontsize=9, fontweight="bold")

    # --- Pie chart ---
    ax = axes[2]
    ax.pie(units, labels=store_labels, colors=colors, autopct="%1.1f%%", 
           startangle=90, textprops={'fontsize': 10})
    ax.set_title(f"Distribuicao de Unidades\n(Total: {total_units:,})", fontweight="bold")

    fig.suptitle("Otimizacao de Alocacao Conjunta - Teto 10.000 Unidades", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(OUT_DIR, "allocation_summary.png")
    plt.savefig(path, dpi=150)
    plt.close()
    log.info("Grafico resumo guardado -> %s", path)


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


def main():
    print("\n" + "=" * 70)
    print("  TIAPOSE - TAREFA 2 / OBJETIVO 2")
    print(f"  Alocacao Conjunta via Mochila (Capacidade: {UNIT_CAP:,})")
    print("=" * 70 + "\n")

    store_candidates = {}

    # 1. Gerar candidatos (Pareto Front) para cada loja
    for store in STORES:
        log.info("=== Gerando candidatos Pareto (Lucro vs Unidades) para: %s ===", store.upper())
        inputs = FORECAST_INPUTS[store]
        
        # NSGA-II para encontrar trade-offs entre Lucro e Unidades
        res = run_optimization(
            store=store,
            forecast_customers=inputs["customers"],
            forecast_is_weekend=inputs["is_weekend"],
            pop_size=100,
            n_max_gen=250,
            seed=42,
            verbose=False,
            profit_fn=optimize_allocation_wrapper,
            second_obj_name="units"
        )
        
        candidates = []
        for i in range(len(res["lucro"])):
            candidates.append({
                "profit": res["lucro"][i],
                "weight": int(res["second_obj"][i]),
                "plan": res["plans"][i],
                "store": store
            })
        
        store_candidates[store] = candidates
        log.info("  %d candidatos encontrados na fronteira de Pareto.", len(candidates))

    # 2. Resolver o Problema da Mochila de Escolha Multipla (MCKP)
    log.info("=== Resolvendo o Problema da Mochila (MCKP) para alocacao global ===")
    groups = [store_candidates[s] for s in STORES]
    best_profit, best_items, total_units = solve_mckp(groups, UNIT_CAP)

    if best_profit is None:
        log.error("ERRO: Nenhuma combinacao de planos respeita o teto de %d unidades.", UNIT_CAP)
        return

    # 3. Formatar e salvar resultados
    log.info("Solucao optima encontrada! Lucro Total: %.2f | Unidades Totais: %d", best_profit, total_units)
    
    final_results = {}
    all_summaries = []
    day_labels = ["Dom", "Seg", "Ter", "Qua", "Qui", "Sex", "Sab"]

    for item in best_items:
        store = item["store"]
        plan = item["plan"]
        profit = item["profit"]
        units = item["weight"]
        
        inputs = FORECAST_INPUTS[store]
        detailed_plan = []
        for d_idx, day_plan in enumerate(plan):
            d_p = day_plan.copy()
            d_p["day_label"] = day_labels[d_idx]
            d_p["is_weekend"] = inputs["is_weekend"][d_idx]
            d_p["customers_forecast"] = inputs["customers"][d_idx]
            
            # Recalcular clientes efetivos e unidades por dia para o grafico
            pr_clean = d_p["pr"]
            effective_customers = int(round(d_p["customers_forecast"] * (1 + ELASTICITY_K * pr_clean)))
            
            metrics = calculate_daily_metrics(
                store=store,
                is_weekend=d_p["is_weekend"],
                customers=effective_customers,
                pr=pr_clean,
                hr_x=d_p["hr_x"],
                hr_j=d_p["hr_j"]
            )
            d_p["units"] = metrics["units_x"] + metrics["units_j"]
            
            detailed_plan.append(d_p)
            
        final_results[store] = {
            "profit": profit,
            "units": units,
            "details": detailed_plan
        }
        
        summary = {
            "store": store,
            "lucro": round(profit, 2),
            "unidades": units,
            "pct_unidades": round(units / total_units * 100, 1),
            "total_staff": sum(d["total_staff"] for d in detailed_plan),
            "avg_discount": round(np.mean([d["pr"] for d in detailed_plan]) * 100, 2),
        }
        all_summaries.append(summary)
        
        # Salvar plano detalhado por loja
        pd.DataFrame(detailed_plan).to_csv(os.path.join(OUT_DIR, f"{store}_plan.csv"), index=False)
        
        # Gerar grafico detalhado por loja
        plot_store_plan(store, detailed_plan, profit, units)

    # Resumo Geral
    summary_df = pd.DataFrame(all_summaries)
    total_row = {
        "store": "TOTAL",
        "lucro": round(best_profit, 2),
        "unidades": total_units,
        "pct_unidades": 100.0,
        "total_staff": summary_df["total_staff"].sum(),
        "avg_discount": round(summary_df["avg_discount"].mean(), 2),
    }
    summary_df = pd.concat([summary_df, pd.DataFrame([total_row])], ignore_index=True)
    summary_df.to_csv(os.path.join(OUT_DIR, "allocation_summary_knapsack.csv"), index=False)

    print("\n" + "=" * 70)
    print("  RESUMO DA ALOCACAO CONJUNTA (MODELAGEM MOCHILA)")
    print("=" * 70)
    print(summary_df.to_string(index=False))
    print(f"\n  Teto de Unidades: {UNIT_CAP:,}")
    print(f"  Utilizacao:       {total_units:,} / {UNIT_CAP:,} ({(total_units/UNIT_CAP*100):.1f}%)")
    print(f"  Lucro Global:     EUR {best_profit:,.2f}")
    print(f"\n  Resultados guardados em: {os.path.abspath(OUT_DIR)}")
    print("=" * 70 + "\n")

    # Graficos
    plot_allocation_summary(final_results, total_units)


if __name__ == "__main__":
    main()
