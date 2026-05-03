"""
run_joint_optimization_o3.py — Joint O2 & O3 Optimization (MOEA/D + U-NSGA-III)

Resolve a otimização conjunta das 4 lojas em simultâneo:
  O3: minimizar (−LucroTotal, StaffTotal) sujeito a ≤10 000 unidades vendidas
  O2: solução de máximo lucro extraída da fronteira O3 (já respeita ≤10 000 unidades)

Input:  últimos 7 dias observados em data/processed/all_stores_processed.csv
Output: results/03_Optimization/joint_o3/
  joint_pareto_moead.csv        — fronteira de Pareto MOEA/D
  joint_pareto_unsga3.csv       — fronteira de Pareto U-NSGA-III
  joint_pareto_comparison.png   — comparação gráfica das duas fronteiras
  joint_best_plan_moead.png     — melhor plano conjunto MOEA/D
  joint_best_plan_unsga3.png    — melhor plano conjunto U-NSGA-III
  joint_o3_summary.csv          — resumo comparativo
"""

import os
import sys
import logging

# Force UTF-8 output so box-drawing and Portuguese characters print correctly
# on Windows terminals that default to cp1252.
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.dont_write_bytecode = True
sys.path.insert(0, "src")

from optimization.moead_model import run_joint_optimization as moead_joint
from optimization.unsga3_model import run_joint_optimization as unsga3_joint
from optimization.joint_problem import STORES, N_VARS, get_o2_solution

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("TIAPOSE.JointOpt")

DATA_PATH = "data/processed/all_stores_processed.csv"
OUT_DIR   = "results/03_Optimization/joint_o3"

# Número máximo de gerações — aumentar para mais qualidade (mais lento)
N_MAX_GEN = 300


# ===========================================================================
# Data loading
# ===========================================================================

def load_last_week(csv_path: str) -> tuple:
    """
    Lê os últimos 7 dias de Num_Customers e is_weekend de cada loja.

    Returns
    -------
    store_forecasts : list[list[int]]   — 4 listas de 7 inteiros
    store_weekends  : list[list[bool]]  — 4 listas de 7 booleanos
    day_labels      : list[str]         — nomes dos dias (ex: ['Sun','Mon',...])
    date_range      : str               — intervalo de datas para logging
    """
    df = pd.read_csv(csv_path, parse_dates=["Date"])

    store_forecasts = []
    store_weekends  = []
    day_labels      = None
    date_range      = None

    for store in STORES:
        sdf = df[df["store_id"] == store].sort_values("Date").tail(7)

        if len(sdf) < 7:
            raise ValueError(
                f"Loja '{store}' tem menos de 7 dias de dados em {csv_path}."
            )

        store_forecasts.append(sdf["Num_Customers"].astype(int).tolist())
        store_weekends.append(sdf["is_weekend"].astype(bool).tolist())

        if day_labels is None:
            day_labels  = [d.strftime("%a") for d in sdf["Date"]]
            date_from   = sdf["Date"].iloc[0].strftime("%Y-%m-%d")
            date_to     = sdf["Date"].iloc[-1].strftime("%Y-%m-%d")
            date_range  = f"{date_from} -> {date_to}"

    return store_forecasts, store_weekends, day_labels, date_range


# ===========================================================================
# Plotting
# ===========================================================================

def plot_pareto_comparison(moead_res: dict, unsga3_res: dict):
    """Fronteiras de Pareto de MOEA/D e U-NSGA-III no mesmo gráfico."""
    fig, ax = plt.subplots(figsize=(10, 7))

    if len(moead_res["lucro"]) > 0:
        ax.scatter(
            moead_res["staff"], moead_res["lucro"],
            c="#2196F3", s=55, alpha=0.75, label="MOEA/D",
            edgecolors="navy", linewidths=0.4,
        )
        # Estrela = solução O2 (máximo lucro)
        ax.scatter(
            moead_res["staff"][0], moead_res["lucro"][0],
            c="#2196F3", s=250, marker="*", zorder=5,
            edgecolors="black", linewidths=0.8, label="O2 MOEA/D",
        )

    if len(unsga3_res["lucro"]) > 0:
        ax.scatter(
            unsga3_res["staff"], unsga3_res["lucro"],
            c="#FF5722", s=55, alpha=0.75, label="U-NSGA-III",
            edgecolors="darkred", linewidths=0.4,
        )
        ax.scatter(
            unsga3_res["staff"][0], unsga3_res["lucro"][0],
            c="#FF5722", s=250, marker="*", zorder=5,
            edgecolors="black", linewidths=0.8, label="O2 U-NSGA-III",
        )

    ax.set_xlabel("Staff Total Semanal (4 lojas)", fontsize=12)
    ax.set_ylabel("Lucro Total Grupo ($)", fontsize=12)
    ax.set_title(
        "Fronteira de Pareto O3 — 4 Lojas Conjuntas\n"
        "Restrição: ≤ 10 000 unidades vendidas  |  ★ = Solução O2",
        fontsize=13, fontweight="bold",
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    path = os.path.join(OUT_DIR, "joint_pareto_comparison.png")
    plt.savefig(path, dpi=150)
    plt.close()
    log.info("Pareto comparison saved → %s", path)


def plot_best_plan(result: dict, algorithm_name: str,
                   store_weekends: list, day_labels: list):
    """
    Grelha 4×2: composição de staff e desconto diário para cada loja,
    usando a solução de maior lucro da fronteira O3.
    """
    if len(result["lucro"]) == 0:
        return

    best_plans = result["plans"][0]   # maior lucro (index 0 = sorted desc)

    fig, axes = plt.subplots(4, 2, figsize=(14, 18))
    fig.suptitle(
        f"{algorithm_name} — Melhor Plano Conjunto O3\n"
        f"Lucro Total: ${result['lucro'][0]:,.0f}  |  "
        f"Staff Total: {int(result['staff'][0])}",
        fontsize=13, fontweight="bold",
    )

    for row, store in enumerate(STORES):
        plan = best_plans[store]
        fw   = store_weekends[row]
        hr_x  = [d["hr_x"]         for d in plan]
        hr_j  = [d["hr_j"]         for d in plan]
        total = [d["total_staff"]   for d in plan]
        pr    = [d["pr"] * 100      for d in plan]
        x     = np.arange(7)
        w     = 0.35

        # ── Staff composition ──────────────────────────────────────────
        ax = axes[row, 0]
        ax.bar(x - w / 2, hr_x, w, label="Peritos",  color="#2196F3", alpha=0.85)
        ax.bar(x + w / 2, hr_j, w, label="Juniores", color="#FF9800", alpha=0.85)

        # Staff cap per day (red dashes)
        for xi, is_wk in enumerate(fw):
            cap = 12 if is_wk else 8
            ax.plot([xi - 0.45, xi + 0.45], [cap, cap],
                    "r--", lw=1.5, alpha=0.8)

        ax.set_xticks(x)
        ax.set_xticklabels(day_labels, fontsize=9)
        ax.set_title(f"{store.capitalize()} — Staff", fontsize=10)
        ax.set_ylabel("Funcionários")
        ax.grid(axis="y", alpha=0.3)
        if row == 0:
            ax.legend(fontsize=8)

        # ── Discount ───────────────────────────────────────────────────
        ax = axes[row, 1]
        colors = ["#81D4FA" if fw[i] else "#4CAF50" for i in range(7)]
        bars = ax.bar(x, pr, color=colors, alpha=0.85)

        # Label discount value on each bar
        for bar, val in zip(bars, pr):
            if val > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.3,
                    f"{val:.1f}%", ha="center", va="bottom", fontsize=7,
                )

        ax.set_xticks(x)
        ax.set_xticklabels(day_labels, fontsize=9)
        ax.set_title(f"{store.capitalize()} — Desconto (%)", fontsize=10)
        ax.set_ylabel("Desconto (%)")
        ax.set_ylim(0, 35)
        ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    fname = algorithm_name.lower().replace("/", "").replace(" ", "_")
    path  = os.path.join(OUT_DIR, f"joint_best_plan_{fname}.png")
    plt.savefig(path, dpi=150)
    plt.close()
    log.info("Best plan plot saved → %s", path)


# ===========================================================================
# CSV export
# ===========================================================================

def save_pareto_csv(result: dict, fname: str, day_labels: list):
    """Guarda a fronteira de Pareto completa num CSV."""
    if len(result["lucro"]) == 0:
        return

    rows = []
    for i in range(len(result["lucro"])):
        row = {
            "lucro_total": round(float(result["lucro"][i]), 2),
            "staff_total": int(result["staff"][i]),
        }
        # Variáveis de decisão por loja e por dia
        x = result["pareto_X"][i]
        for s, store in enumerate(STORES):
            x_s = x[s * N_VARS: (s + 1) * N_VARS]
            for d, lbl in enumerate(day_labels):
                row[f"{store}_pr_{lbl}"]    = round(float(x_s[d * 3]),     4)
                row[f"{store}_hr_x_{lbl}"]  = int(round(float(x_s[d * 3 + 1])))
                row[f"{store}_hr_j_{lbl}"]  = int(round(float(x_s[d * 3 + 2])))
        rows.append(row)

    path = os.path.join(OUT_DIR, fname)
    pd.DataFrame(rows).to_csv(path, index=False)
    log.info("Pareto CSV saved → %s  (%d solutions)", path, len(rows))


# ===========================================================================
# Main
# ===========================================================================

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("\n" + "=" * 68)
    print("  TIAPOSE — JOINT O2 & O3 OPTIMIZATION")
    print("  MOEA/D  +  U-NSGA-III  |  4 Lojas  |  84 Variáveis")
    print("=" * 68 + "\n")

    # ── 1. Carregar dados ─────────────────────────────────────────────
    store_forecasts, store_weekends, day_labels, date_range = load_last_week(DATA_PATH)

    print(f"  Horizonte de previsão: {date_range}  ({' '.join(day_labels)})")
    print(f"  {'Loja':<15}  {'Clientes previstos':>40}")
    print("  " + "─" * 58)
    for i, store in enumerate(STORES):
        customers_str = "  ".join(f"{c:>4}" for c in store_forecasts[i])
        print(f"  {store:<15}  {customers_str}")
    print()

    # ── 2. MOEA/D ─────────────────────────────────────────────────────
    print("─" * 68)
    print("  [1/2] MOEA/D — penalização para restrição 10 k unidades")
    print("─" * 68)

    moead_res = moead_joint(
        store_forecasts=store_forecasts,
        store_weekends=store_weekends,
        n_partitions=99,
        n_neighbors=20,
        prob_neighbor_mating=0.9,
        n_max_gen=N_MAX_GEN,
        seed=42,
        verbose=True,
    )
    moead_o2 = get_o2_solution(moead_res)

    # ── 3. U-NSGA-III ─────────────────────────────────────────────────
    print("\n" + "─" * 68)
    print("  [2/2] U-NSGA-III — restrição hard via constraint-dominance")
    print("─" * 68)

    unsga3_res = unsga3_joint(
        store_forecasts=store_forecasts,
        store_weekends=store_weekends,
        n_partitions=99,
        n_max_gen=N_MAX_GEN,
        seed=42,
        verbose=True,
    )
    unsga3_o2 = get_o2_solution(unsga3_res)

    # ── 4. Resultados ─────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  RESULTADOS")
    print("=" * 68)

    for algo, res, o2 in [
        ("MOEA/D",     moead_res,  moead_o2),
        ("U-NSGA-III", unsga3_res, unsga3_o2),
    ]:
        n = len(res["lucro"])
        print(f"\n  {algo} — {n} soluções Pareto")
        if n == 0:
            print("    (sem soluções viáveis)")
            continue

        print(f"    O3  Max Lucro : ${res['lucro'][0]:>12,.2f}  "
              f"(Staff: {int(res['staff'][0])})")
        print(f"    O3  Min Staff : {int(res['staff'][-1]):>4}  "
              f"(Lucro: ${res['lucro'][-1]:>12,.2f})")
        print(f"    O2  Solução   : ${o2['lucro']:>12,.2f}  "
              f"(Staff: {int(o2['staff'])})")

        # Plano detalhado O2 por loja
        print(f"\n    Plano semanal O2 ({algo}):")
        print(f"    {'Loja':<15} {'Dia':<5} {'Desc%':>6} {'Peritos':>8} {'Juniores':>9} {'Total':>6}")
        print("    " + "─" * 52)
        for store in STORES:
            for d_info in o2["plan"][store]:
                lbl  = day_labels[d_info["day"]]
                flag = " ⚠" if (
                    d_info["total_staff"] > 8 and
                    not store_weekends[STORES.index(store)][d_info["day"]]
                ) else ""
                print(
                    f"    {store:<15} {lbl:<5}"
                    f" {d_info['pr']*100:>5.1f}%"
                    f" {d_info['hr_x']:>8}"
                    f" {d_info['hr_j']:>9}"
                    f" {d_info['total_staff']:>6}{flag}"
                )

    # ── 5. Guardar outputs ────────────────────────────────────────────
    print("\n" + "─" * 68)
    print("  Saving results...")

    save_pareto_csv(moead_res,  "joint_pareto_moead.csv",   day_labels)
    save_pareto_csv(unsga3_res, "joint_pareto_unsga3.csv",  day_labels)

    plot_best_plan(moead_res,  "MOEA/D",     store_weekends, day_labels)
    plot_best_plan(unsga3_res, "U-NSGA-III", store_weekends, day_labels)
    plot_pareto_comparison(moead_res, unsga3_res)

    # ── 6. Summary CSV ────────────────────────────────────────────────
    summary_rows = []
    for algo, res, o2 in [
        ("MOEA/D",     moead_res,  moead_o2),
        ("U-NSGA-III", unsga3_res, unsga3_o2),
    ]:
        if len(res["lucro"]) == 0:
            continue
        summary_rows.append({
            "algorithm":            algo,
            "n_pareto_solutions":   len(res["lucro"]),
            "o3_max_profit":        round(float(res["lucro"][0]),  2),
            "o3_max_profit_staff":  int(res["staff"][0]),
            "o3_min_staff":         int(res["staff"][-1]),
            "o3_min_staff_profit":  round(float(res["lucro"][-1]), 2),
            "o2_profit":            round(o2["lucro"], 2),
            "o2_staff":             int(o2["staff"]),
        })

    if summary_rows:
        summary_df = pd.DataFrame(summary_rows)
        summary_path = os.path.join(OUT_DIR, "joint_o3_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        print(f"\n  Summary:\n{summary_df.to_string(index=False)}")
        print(f"\n  Saved → {os.path.abspath(summary_path)}")

    print(f"\n  All results → {os.path.abspath(OUT_DIR)}")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
