"""
run_optimization.py — Full NSGA-II Optimization for all TIAPOSE stores.

Uses the last observed week (2014-06-08 to 2014-06-14, Sun→Sat) as the
forecast input horizon. Results are saved to results/03_Optimization_Report/.
"""

import sys
import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.dont_write_bytecode = True
sys.path.insert(0, "src")

from optimization.nsga2_model import run_optimization, N_DAYS

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("TIAPOSE.OptRun")

# ---------------------------------------------------------------------------
# Forecast inputs — last observed week (2014-06-08 Sun → 2014-06-14 Sat)
# Days: Sun, Mon, Tue, Wed, Thu, Fri, Sat
# ---------------------------------------------------------------------------
FORECAST_INPUTS = {
    "baltimore": {
        "customers":   [97, 61, 65, 71, 65, 89, 125],
        "is_weekend":  [True, False, False, False, False, False, True],
    },
    "lancaster": {
        "customers":   [116, 72, 77, 84, 77, 106, 149],
        "is_weekend":  [True, False, False, False, False, False, True],
    },
    "philadelphia": {
        "customers":   [230, 144, 154, 168, 154, 211, 298],
        "is_weekend":  [True, False, False, False, False, False, True],
    },
    "richmond": {
        "customers":   [64, 40, 42, 46, 42, 58, 82],
        "is_weekend":  [True, False, False, False, False, False, True],
    },
}

DAY_LABELS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
STORES = list(FORECAST_INPUTS.keys())
OUT_DIR = "results/03_Optimization_Report"
os.makedirs(OUT_DIR, exist_ok=True)


def plot_pareto_front(store: str, lucro: np.ndarray, staff: np.ndarray):
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(staff, lucro, c=lucro, cmap="plasma", s=60, alpha=0.85, edgecolors="k", linewidths=0.4)
    plt.colorbar(sc, ax=ax, label="Lucro (€)")
    ax.set_xlabel("Staff Total Semanal", fontsize=12)
    ax.set_ylabel("Lucro Semanal (€)", fontsize=12)
    ax.set_title(f"Fronteira de Pareto — {store.capitalize()}", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3)
    # Annotate best-profit and min-staff extremes
    idx_profit = np.argmax(lucro)
    idx_staff  = np.argmin(staff)
    ax.annotate("Max Lucro", xy=(staff[idx_profit], lucro[idx_profit]),
                xytext=(8, -20), textcoords="offset points",
                fontsize=8, color="darkgreen",
                arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1))
    ax.annotate("Min Staff", xy=(staff[idx_staff], lucro[idx_staff]),
                xytext=(8, 8), textcoords="offset points",
                fontsize=8, color="navy",
                arrowprops=dict(arrowstyle="->", color="navy", lw=1))
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"{store}_pareto_front.png")
    plt.savefig(path, dpi=150)
    plt.close()
    log.info("Pareto plot saved → %s", path)


def plot_best_weekly_plan(store: str, plan: list, lucro: float, staff: int):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    days = [DAY_LABELS[d["day"]] for d in plan]
    hr_x  = [d["hr_x"]  for d in plan]
    hr_j  = [d["hr_j"]  for d in plan]
    pr    = [d["pr"]    for d in plan]
    total = [d["total_staff"] for d in plan]

    x = np.arange(len(days))
    w = 0.35

    # --- Staff composition ---
    ax = axes[0]
    ax.bar(x - w/2, hr_x, w, label="Peritos",  color="#2196F3", alpha=0.85)
    ax.bar(x + w/2, hr_j, w, label="Juniores", color="#FF9800", alpha=0.85)
    ax.axhline(8, color="red", ls="--", lw=1.2, label="Limite útil (8)")
    ax.set_xticks(x); ax.set_xticklabels(days)
    ax.set_title("Composição de Staff"); ax.set_ylabel("Funcionários")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    # --- Discount ---
    ax = axes[1]
    ax.bar(x, [p * 100 for p in pr], color="#4CAF50", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(days)
    ax.set_title("Desconto Diário (%)"); ax.set_ylabel("Desconto (%)")
    ax.set_ylim(0, 35); ax.grid(axis="y", alpha=0.3)

    # --- Total staff ---
    ax = axes[2]
    colors = ["#F44336" if t > 8 else "#4CAF50" for t in total]
    ax.bar(x, total, color=colors, alpha=0.85)
    ax.axhline(8, color="black", ls="--", lw=1.2, label="Limite útil (8)")
    ax.set_xticks(x); ax.set_xticklabels(days)
    ax.set_title("Staff Total por Dia"); ax.set_ylabel("Total")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"{store.capitalize()} — Melhor Solução: Lucro €{lucro:,.0f} | Staff {staff}",
        fontsize=13, fontweight="bold"
    )
    plt.tight_layout()
    path = os.path.join(OUT_DIR, f"{store}_best_plan.png")
    plt.savefig(path, dpi=150)
    plt.close()
    log.info("Best plan plot saved → %s", path)


def main():
    print("\n" + "=" * 65)
    print("  TIAPOSE — NSGA-II FULL OPTIMIZATION (all stores)")
    print("=" * 65 + "\n")

    all_summaries = []

    for store in STORES:
        inputs = FORECAST_INPUTS[store]
        log.info("━━━ Optimizing: %s ━━━", store.upper())

        # Baltimore, Lancaster e Philadelphia não convergiram em 300 gerações.
        n_max_gen = 500 if store in ("baltimore", "lancaster", "philadelphia") else 300

        results = run_optimization(
            store=store,
            forecast_customers=inputs["customers"],
            forecast_is_weekend=inputs["is_weekend"],
            pop_size=100,
            n_max_gen=n_max_gen,
            seed=42,
            verbose=False,          # reduce noise; set True for debug
        )

        if len(results["lucro"]) == 0:
            log.warning("No feasible solutions found for %s — skipping.", store)
            continue

        lucro = results["lucro"]
        staff = results["staff"]
        plans = results["plans"]

        # --- Summary stats ---
        n_solutions = len(lucro)
        best_plan   = plans[0]   # highest profit (sorted in extract_pareto_solutions)
        min_staff_idx = int(np.argmin(staff))

        summary = {
            "store":            store,
            "n_pareto":         n_solutions,
            "max_profit":       round(float(lucro[0]), 2),
            "min_profit":       round(float(lucro[-1]), 2),
            "min_staff":        int(staff[min_staff_idx]),
            "max_staff":        int(staff[0]),
            "staff_at_max_profit": int(staff[0]),
            "profit_at_min_staff": round(float(lucro[min_staff_idx]), 2),
        }
        all_summaries.append(summary)

        # Print per-store table
        print(f"\n{'─'*60}")
        print(f"  {store.upper()} — {n_solutions} Pareto solutions")
        print(f"{'─'*60}")
        print(f"  Max Profit:  €{lucro[0]:>10,.2f}  (Staff: {int(staff[0])})")
        print(f"  Min Staff:    {int(staff[min_staff_idx]):>3}           (Profit: €{lucro[min_staff_idx]:>10,.2f})")
        print(f"\n  Best weekly plan (max profit):")
        print(f"  {'Day':<5} {'Desconto':>9} {'Peritos':>8} {'Juniores':>9} {'Total':>6}")
        print(f"  {'─'*43}")
        for d in best_plan:
            flag = " ⚠" if d["total_staff"] > 8 and not inputs["is_weekend"][d["day"]] else ""
            print(
                f"  {DAY_LABELS[d['day']]:<5} {d['pr']*100:>8.1f}%"
                f" {d['hr_x']:>8} {d['hr_j']:>9} {d['total_staff']:>6}{flag}"
            )

        # --- Save Pareto CSV ---
        pareto_df = pd.DataFrame({
            "lucro": lucro,
            "staff": staff,
        })
        for day_i in range(N_DAYS):
            pareto_df[f"pr_{DAY_LABELS[day_i]}"]    = results["pareto_X"][:, day_i * 3].round(4)
            pareto_df[f"hr_x_{DAY_LABELS[day_i]}"]  = np.round(results["pareto_X"][:, day_i * 3 + 1]).astype(int)
            pareto_df[f"hr_j_{DAY_LABELS[day_i]}"]  = np.round(results["pareto_X"][:, day_i * 3 + 2]).astype(int)
        pareto_df.to_csv(os.path.join(OUT_DIR, f"{store}_pareto.csv"), index=False)

        # --- Plots ---
        plot_pareto_front(store, lucro, staff)
        plot_best_weekly_plan(store, best_plan, lucro[0], int(staff[0]))

    # --- Master summary ---
    summary_df = pd.DataFrame(all_summaries)
    summary_df.to_csv(os.path.join(OUT_DIR, "optimization_summary.csv"), index=False)

    print("\n" + "=" * 65)
    print("  OPTIMIZATION SUMMARY")
    print("=" * 65)
    print(summary_df.to_string(index=False))
    print(f"\n  Results saved to: {os.path.abspath(OUT_DIR)}")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    main()
