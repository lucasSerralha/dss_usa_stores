"""
compare_optimization_nmaxgen.py — Sensibilidade N_MAX_GEN: MOEA/D vs U-NSGA-III

Varre N_MAX_GEN ∈ [50, 100, 150, 200, 300] com N_RUNS corridas independentes
e regista Hipervolume (HV) e nº soluções Pareto para cada configuração.

Nota de implementação:
  O MOEA/D (pymoo) NÃO suporta restrições explícitas (n_ieq_constr > 0).
  Para comparação justa, as restrições de staff diário são incorporadas como
  penalidade em F[0] no TiaposeNoCstr — abordagem idêntica à usada no problema
  conjunto O3 (joint_problem.py).

Output: results/04_Model_Testing/nmaxgen_comparison.csv
        results/04_Model_Testing/nmaxgen_convergence.png

Uso:
    python scripts/compare_optimization_nmaxgen.py
"""

import os
import sys
import time
import logging

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
sys.stdout.reconfigure(encoding="utf-8")

from pymoo.algorithms.moo.moead  import MOEAD
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.core.problem          import ElementwiseProblem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm   import PM
from pymoo.operators.sampling.rnd  import FloatRandomSampling
from pymoo.optimize                import minimize
from pymoo.termination.default     import DefaultMultiObjectiveTermination
from pymoo.util.ref_dirs           import get_reference_directions

from optimization.nsga2_model import (
    IntegerRepair, dummy_profit_function,
    N_VARS, INT_IDX, N_DAYS, XL, XU,
    extract_pareto_solutions,
)
from optimization.unsga3_model import run_optimization as run_unsga3

logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(message)s",
)

# ---------------------------------------------------------------------------
# Parâmetros
# ---------------------------------------------------------------------------
STORE      = "baltimore"
CLIENTES   = [97, 61, 65, 71, 65, 89, 125]   # Dom→Sáb (última semana observada)
FDS        = [True, False, False, False, False, False, True]

N_RUNS     = 3           # corridas independentes por (algoritmo × N_MAX_GEN)
N_MAXGENS  = [50, 100, 150, 200, 300]

HV_REF     = np.array([0.0, 210.0])   # pior ponto: [-lucro, staff]
OUT_DIR    = os.path.join(os.path.dirname(__file__), "..", "results", "04_Model_Testing")

PENALTY_STAFF = 5_000.0   # $ por unidade de staff excedente por dia (embedding restrição)


# ---------------------------------------------------------------------------
# Problema sem restrições explícitas — para MOEA/D
# Staff caps integrados como penalidade em F[0]
# ---------------------------------------------------------------------------
class TiaposeNoCstr(ElementwiseProblem):
    """
    Versão do problema de otimização sem n_ieq_constr — compatível com MOEA/D.
    As restrições de staff diário (cap 8 dias úteis, cap 12 FDS) são tratadas
    como penalidade aditiva em F[0] (negativo do lucro).
    """

    def __init__(self, store, forecast_customers, forecast_is_weekend,
                 profit_fn=None, penalty=PENALTY_STAFF):
        self.store               = store
        self.forecast_customers  = list(forecast_customers)
        self.forecast_is_weekend = list(forecast_is_weekend)
        self.profit_fn           = profit_fn if profit_fn is not None else dummy_profit_function
        self.penalty             = penalty
        self._weekday_idx = [d for d, w in enumerate(forecast_is_weekend) if not w]
        self._weekend_idx = [d for d, w in enumerate(forecast_is_weekend) if w]
        super().__init__(n_var=N_VARS, n_obj=2, n_ieq_constr=0,
                         xl=XL, xu=XU, elementwise=True)

    def _evaluate(self, x, out, *args, **kwargs):
        f1, f2, _ = self.profit_fn(
            x, self.store, self.forecast_customers, self.forecast_is_weekend
        )
        # Calcular violação de staff e adicionar penalidade a F[0]
        staff = np.round(x[INT_IDX]).reshape(N_DAYS, 2).sum(axis=1)
        pen = 0.0
        for d in self._weekday_idx:
            pen += max(0.0, staff[d] - 8) * self.penalty
        for d in self._weekend_idx:
            pen += max(0.0, staff[d] - 12) * self.penalty
        out["F"] = [f1 + pen, f2]


# ---------------------------------------------------------------------------
# Corrida MOEA/D com problema sem restrições
# ---------------------------------------------------------------------------
def run_moead_nocstr(store, forecast_customers, forecast_is_weekend,
                     n_max_gen=150, n_partitions=99, seed=42,
                     verbose=False, profit_fn=None):
    ref_dirs  = get_reference_directions("uniform", n_dim=2, n_partitions=n_partitions)
    problem   = TiaposeNoCstr(store, forecast_customers, forecast_is_weekend,
                              profit_fn=profit_fn)
    algorithm = MOEAD(
        ref_dirs=ref_dirs,
        n_neighbors=20,
        prob_neighbor_mating=0.9,
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
    return extract_pareto_solutions(res)


# ---------------------------------------------------------------------------
# Utilitário — Hipervolume
# ---------------------------------------------------------------------------
def _hv(pareto_F: np.ndarray) -> float:
    from pymoo.indicators.hv import Hypervolume
    if pareto_F is None or len(pareto_F) == 0:
        return 0.0
    mask = np.all(pareto_F < HV_REF, axis=1)
    if not np.any(mask):
        return 0.0
    return float(Hypervolume(ref_point=HV_REF).do(pareto_F[mask]))


# ---------------------------------------------------------------------------
# Ciclo principal
# ---------------------------------------------------------------------------
rows = []

CONFIGS = [
    ("MOEA/D",     run_moead_nocstr, {"n_partitions": 99}),
    ("U-NSGA-III", run_unsga3,       {"n_partitions": 99}),
]

for algo_name, run_fn, extra_kw in CONFIGS:
    print(f"\n{'='*60}")
    print(f"  Algoritmo: {algo_name}")
    print(f"{'='*60}")

    for n_gen in N_MAXGENS:
        hvs, nsols, times = [], [], []

        for run_i in range(N_RUNS):
            seed = 42 + run_i * 7
            t0   = time.perf_counter()

            res = run_fn(
                store=STORE,
                forecast_customers=CLIENTES,
                forecast_is_weekend=FDS,
                n_max_gen=n_gen,
                seed=seed,
                verbose=False,
                profit_fn=dummy_profit_function,
                **extra_kw,
            )

            elapsed = time.perf_counter() - t0
            hv_val  = _hv(res["pareto_F"])
            n_sol   = len(res["lucro"])

            hvs.append(hv_val)
            nsols.append(n_sol)
            times.append(elapsed)

            print(
                f"  N={n_gen:3d} | run {run_i+1}/{N_RUNS} | "
                f"HV={hv_val:.4f} | n_sol={n_sol:3d} | {elapsed:.2f}s"
            )

        rows.append({
            "algorithm":  algo_name,
            "n_max_gen":  n_gen,
            "hv_mean":    float(np.mean(hvs)),
            "hv_std":     float(np.std(hvs)),
            "nsol_mean":  float(np.mean(nsols)),
            "nsol_std":   float(np.std(nsols)),
            "time_mean":  float(np.mean(times)),
        })

# ---------------------------------------------------------------------------
# Guardar CSV
# ---------------------------------------------------------------------------
os.makedirs(OUT_DIR, exist_ok=True)
df = pd.DataFrame(rows)
csv_path = os.path.join(OUT_DIR, "nmaxgen_comparison.csv")
df.to_csv(csv_path, index=False)

print(f"\n{'='*60}")
print(f"  Resultados guardados em: {csv_path}")
print(f"{'='*60}")
print(df.to_string(index=False))

# ---------------------------------------------------------------------------
# Gráfico de convergência
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(12, 5))
fig.suptitle("Sensibilidade N_MAX_GEN — Baltimore", fontsize=13, fontweight="bold")

colors = {"MOEA/D": "#E67E22", "U-NSGA-III": "#4A90E2"}

for algo in df["algorithm"].unique():
    sub = df[df["algorithm"] == algo].sort_values("n_max_gen")
    c   = colors.get(algo, "grey")
    axes[0].errorbar(
        sub["n_max_gen"], sub["hv_mean"], yerr=sub["hv_std"],
        marker="o", label=algo, color=c, capsize=4, linewidth=2
    )
    axes[1].errorbar(
        sub["n_max_gen"], sub["nsol_mean"], yerr=sub["nsol_std"],
        marker="s", label=algo, color=c, capsize=4, linewidth=2
    )

for ax, ylabel, title in zip(
    axes,
    ["Hipervolume (média ± std)", "Nº Soluções Pareto (média ± std)"],
    ["Hipervolume vs N_MAX_GEN", "Soluções Pareto vs N_MAX_GEN"]
):
    ax.set_xlabel("N_MAX_GEN (gerações)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.axvline(150, color="grey", linestyle="--", alpha=0.7, label="N=150 usado")
    ax.legend()
    ax.grid(True, alpha=0.3)

plt.tight_layout()
png_path = os.path.join(OUT_DIR, "nmaxgen_convergence.png")
plt.savefig(png_path, dpi=150, bbox_inches="tight")
plt.close()
print(f"\n  Gráfico guardado em: {png_path}")
