"""
run_moead_unsga3_o1.py — Otimização per-store O1 com MOEA/D e U-NSGA-III

Executa os dois algoritmos para as 4 lojas (objetivo O1: maximizar lucro
semanal por loja, sem restrição global de unidades) e guarda resultados
no mesmo formato que run_optimization.py (NSGA-II), para comparação.

Nota de implementação — MOEA/D e restrições:
  O pymoo rejeita problemas com n_ieq_constr>0 no MOEA/D (Tchebycheff).
  Para O1 usamos um problema sem restrições onde o excesso de staff (cap
  8 dias úteis / 12 fim-de-semana) é adicionado como penalização em F[0],
  à semelhança do que optimize_weekly_wrapper já calcula em f3.
  O U-NSGA-III não tem esta limitação e usa as restrições hard normalmente.

Output:
  results/03_Optimization/moead_o1/
    optimization_summary.csv
    {store}_pareto_front.png   {store}_best_plan.png   {store}_pareto.csv

  results/03_Optimization/unsga3_o1/
    (mesma estrutura)
"""

import os
import sys
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.stdout.reconfigure(encoding="utf-8", errors="replace")
sys.stderr.reconfigure(encoding="utf-8", errors="replace")

sys.dont_write_bytecode = True
sys.path.insert(0, "src")

# ---------------------------------------------------------------------------
# Imports dos módulos do projeto
# ---------------------------------------------------------------------------
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.util.ref_dirs import get_reference_directions

from optimization.nsga2_model import (
    IntegerRepair, extract_pareto_solutions,
    N_VARS, N_DAYS, XL, XU,
)
from optimization.unsga3_model import run_optimization as unsga3_run
from utils.profit_logic import optimize_weekly_wrapper

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("TIAPOSE.O1Run")

# ---------------------------------------------------------------------------
# Previsões — última semana observada (2014-06-08 Dom → 2014-06-14 Sáb)
# ---------------------------------------------------------------------------
FORECAST_INPUTS = {
    "baltimore":    {"customers": [97,  61,  65,  71,  65,  89, 125],
                     "is_weekend": [True, False, False, False, False, False, True]},
    "lancaster":    {"customers": [116,  72,  77,  84,  77, 106, 149],
                     "is_weekend": [True, False, False, False, False, False, True]},
    "philadelphia": {"customers": [230, 144, 154, 168, 154, 211, 298],
                     "is_weekend": [True, False, False, False, False, False, True]},
    "richmond":     {"customers": [64,  40,  42,  46,  42,  58,  82],
                     "is_weekend": [True, False, False, False, False, False, True]},
}

DAY_LABELS = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
STORES = list(FORECAST_INPUTS.keys())
N_MAX_GEN = 300


# ===========================================================================
# Problema penalty-based para MOEA/D O1
# (MOEA/D rejeita n_ieq_constr > 0, por isso absorvemos o excesso de staff
#  na penalização que optimize_weekly_wrapper já calcula como f3)
# ===========================================================================
class _MoeadO1Problem(ElementwiseProblem):
    """Versão sem restrições de TiaposeOptimization, para uso com MOEA/D."""

    def __init__(self, store: str, forecast_customers: list, forecast_is_weekend: list):
        self.store = store
        self.forecast_customers = list(forecast_customers)
        self.forecast_is_weekend = list(forecast_is_weekend)
        super().__init__(
            n_var=N_VARS, n_obj=2, n_ieq_constr=0,
            xl=XL, xu=XU, elementwise=True,
        )

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> None:
        neg_profit, total_staff, penalty = optimize_weekly_wrapper(
            decision_vars=x,
            store=self.store,
            forecast_customers=self.forecast_customers,
            forecast_is_weekend=self.forecast_is_weekend,
        )
        # Absorver penalização de excesso de staff em F[0]
        out["F"] = [neg_profit + penalty, total_staff]


def _run_moead_o1(store: str, forecast_customers: list, forecast_is_weekend: list,
                  n_partitions: int = 99, n_max_gen: int = 300,
                  seed: int = 42, verbose: bool = False) -> dict:
    """Corre MOEA/D per-store para O1 (sem restrições, penalty em F[0])."""
    ref_dirs = get_reference_directions("uniform", n_dim=2, n_partitions=n_partitions)
    pop_size = len(ref_dirs)

    log.info("MOEA/D O1 | loja=%-15s | subproblemas=%d | max_gen=%d | seed=%d",
             store, pop_size, n_max_gen, seed)

    problem = _MoeadO1Problem(store, forecast_customers, forecast_is_weekend)
    algorithm = MOEAD(
        ref_dirs=ref_dirs, n_neighbors=20, prob_neighbor_mating=0.9,
        crossover=SBX(eta=15, prob=0.9, prob_var=1.0 / N_VARS),
        mutation=PM(eta=20, prob=1.0 / N_VARS),
        sampling=FloatRandomSampling(),
        repair=IntegerRepair(),
    )
    termination = DefaultMultiObjectiveTermination(
        xtol=1e-6, cvtol=1e-6, ftol=0.0025, period=30, n_max_gen=n_max_gen,
    )
    res = minimize(problem, algorithm, termination, seed=seed,
                   verbose=verbose, save_history=False)
    log.info("MOEA/D O1 concluído | loja=%-15s | gerações=%d", store, res.algorithm.n_gen)
    return extract_pareto_solutions(res)


# ===========================================================================
# Plots
# ===========================================================================

def _plot_pareto(store: str, lucro: np.ndarray, staff: np.ndarray, out_dir: str, title: str):
    fig, ax = plt.subplots(figsize=(9, 6))
    sc = ax.scatter(staff, lucro, c=lucro, cmap="plasma", s=60, alpha=0.85,
                    edgecolors="k", linewidths=0.4)
    plt.colorbar(sc, ax=ax, label="Lucro ($)")
    ax.set_xlabel("Staff Total Semanal", fontsize=12)
    ax.set_ylabel("Lucro Semanal ($)", fontsize=12)
    ax.grid(True, alpha=0.3)
    idx_p = int(np.argmax(lucro)); idx_s = int(np.argmin(staff))
    ax.annotate("Max Lucro", xy=(staff[idx_p], lucro[idx_p]),
                xytext=(8, -20), textcoords="offset points", fontsize=8, color="darkgreen",
                arrowprops=dict(arrowstyle="->", color="darkgreen", lw=1))
    ax.annotate("Min Staff", xy=(staff[idx_s], lucro[idx_s]),
                xytext=(8, 8), textcoords="offset points", fontsize=8, color="navy",
                arrowprops=dict(arrowstyle="->", color="navy", lw=1))
    ax.set_title(f"Fronteira de Pareto — {store.capitalize()} ({title})", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{store}_pareto_front.png"), dpi=150)
    plt.close()


def _plot_best_plan(store: str, plan: list, lucro: float, staff: int,
                    is_weekend: list, out_dir: str, title: str):
    fig, axes = plt.subplots(1, 3, figsize=(14, 5))
    days  = [DAY_LABELS[d["day"]] for d in plan]
    hr_x  = [d["hr_x"]       for d in plan]
    hr_j  = [d["hr_j"]       for d in plan]
    pr    = [d["pr"]          for d in plan]
    total = [d["total_staff"] for d in plan]
    x = np.arange(len(days)); w = 0.35

    ax = axes[0]
    ax.bar(x - w/2, hr_x, w, label="Peritos",  color="#2196F3", alpha=0.85)
    ax.bar(x + w/2, hr_j, w, label="Juniores", color="#FF9800", alpha=0.85)
    for xi, is_wk in enumerate(is_weekend):
        cap = 12 if is_wk else 8
        ax.plot([xi - 0.45, xi + 0.45], [cap, cap], "r--", lw=1.5, alpha=0.8)
    ax.set_xticks(x); ax.set_xticklabels(days)
    ax.set_title("Composição de Staff"); ax.set_ylabel("Funcionários")
    ax.legend(fontsize=8); ax.grid(axis="y", alpha=0.3)

    ax = axes[1]
    ax.bar(x, [p * 100 for p in pr], color="#4CAF50", alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(days)
    ax.set_title("Desconto Diário (%)"); ax.set_ylabel("Desconto (%)")
    ax.set_ylim(0, 35); ax.grid(axis="y", alpha=0.3)

    ax = axes[2]
    colors = ["#F44336" if total[i] > (12 if is_weekend[i] else 8) else "#4CAF50"
              for i in range(len(total))]
    ax.bar(x, total, color=colors, alpha=0.85)
    ax.set_xticks(x); ax.set_xticklabels(days)
    ax.set_title("Staff Total por Dia"); ax.set_ylabel("Total")
    ax.grid(axis="y", alpha=0.3)

    fig.suptitle(
        f"{store.capitalize()} ({title}) — ${lucro:,.0f} | Staff {staff}",
        fontsize=13, fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f"{store}_best_plan.png"), dpi=150)
    plt.close()


# ===========================================================================
# Runner comum
# ===========================================================================

def _run_and_save(run_fn, algo_name: str, out_dir: str):
    """Corre run_fn para as 4 lojas, guarda CSVs e plots, devolve summary."""
    os.makedirs(out_dir, exist_ok=True)
    summaries = []

    for store in STORES:
        inp = FORECAST_INPUTS[store]
        res = run_fn(store, inp["customers"], inp["is_weekend"])

        if len(res["lucro"]) == 0:
            log.warning("[%s] sem soluções para %s", algo_name, store)
            continue

        lucro = res["lucro"]
        staff = res["staff"]
        idx_s = int(np.argmin(staff))

        summaries.append({
            "store":               store,
            "n_pareto":            len(lucro),
            "max_profit":          round(float(lucro[0]),          2),
            "min_profit":          round(float(lucro[-1]),         2),
            "min_staff":           int(staff[idx_s]),
            "max_staff":           int(staff[0]),
            "staff_at_max_profit": int(staff[0]),
            "profit_at_min_staff": round(float(lucro[idx_s]),      2),
        })

        # Pareto CSV
        X = res["pareto_X"]
        pdf = pd.DataFrame({"lucro": lucro, "staff": staff,
                             "desconto_medio": np.stack(
                                 [X[:, d * 3] for d in range(N_DAYS)], axis=1
                             ).mean(axis=1).round(4) * 100})
        for di in range(N_DAYS):
            pdf[f"pr_{DAY_LABELS[di]}"]   = X[:, di * 3].round(4)
            pdf[f"hr_x_{DAY_LABELS[di]}"] = np.round(X[:, di * 3 + 1]).astype(int)
            pdf[f"hr_j_{DAY_LABELS[di]}"] = np.round(X[:, di * 3 + 2]).astype(int)
        pdf.to_csv(os.path.join(out_dir, f"{store}_pareto.csv"), index=False)

        # Plots
        _plot_pareto(store, lucro, staff, out_dir, algo_name)
        _plot_best_plan(store, res["plans"][0], float(lucro[0]), int(staff[0]),
                        inp["is_weekend"], out_dir, algo_name)

        print(f"  {store:<15} | {len(lucro):>3} sols | "
              f"Max Lucro: ${lucro[0]:>10,.0f} | Min Staff: {int(staff[idx_s])}")

    df = pd.DataFrame(summaries)
    df.to_csv(os.path.join(out_dir, "optimization_summary.csv"), index=False)
    return df


# ===========================================================================
# Main
# ===========================================================================

def main():
    print("\n" + "=" * 68)
    print("  TIAPOSE — O1 per-store: MOEA/D  +  U-NSGA-III")
    print(f"  N_MAX_GEN={N_MAX_GEN}  |  n_partitions=99  |  seed=42")
    print("=" * 68)

    # ── MOEA/D ───────────────────────────────────────────────────────────────
    out_m = "results/03_Optimization/moead_o1"
    print(f"\n{'─'*68}")
    print(f"  [1/2] MOEA/D  →  {out_m}/")
    print(f"{'─'*68}")

    def _moead_runner(store, customers, weekends):
        return _run_moead_o1(store, customers, weekends,
                             n_partitions=99, n_max_gen=N_MAX_GEN, seed=42, verbose=False)

    df_m = _run_and_save(_moead_runner, "MOEA/D", out_m)

    # ── U-NSGA-III ───────────────────────────────────────────────────────────
    out_u = "results/03_Optimization/unsga3_o1"
    print(f"\n{'─'*68}")
    print(f"  [2/2] U-NSGA-III  →  {out_u}/")
    print(f"{'─'*68}")

    def _unsga3_runner(store, customers, weekends):
        return unsga3_run(store=store, forecast_customers=customers,
                          forecast_is_weekend=weekends,
                          n_partitions=99, n_max_gen=N_MAX_GEN, seed=42, verbose=False)

    df_u = _run_and_save(_unsga3_runner, "U-NSGA-III", out_u)

    # ── Comparação ────────────────────────────────────────────────────────────
    print("\n" + "=" * 68)
    print("  RESUMO — Max Lucro O1 por loja")
    print("=" * 68)
    print(f"  {'Loja':<15}  {'MOEA/D':>14}  {'U-NSGA-III':>14}")
    print(f"  {'─'*47}")
    for store in STORES:
        rm = df_m[df_m["store"] == store]
        ru = df_u[df_u["store"] == store]
        mp_m = f"${float(rm['max_profit'].iloc[0]):>12,.0f}" if len(rm) else "           n/a"
        mp_u = f"${float(ru['max_profit'].iloc[0]):>12,.0f}" if len(ru) else "           n/a"
        print(f"  {store:<15}  {mp_m}  {mp_u}")

    print(f"\n  Outputs em:")
    print(f"    {os.path.abspath(out_m)}")
    print(f"    {os.path.abspath(out_u)}")
    print("=" * 68 + "\n")


if __name__ == "__main__":
    main()
