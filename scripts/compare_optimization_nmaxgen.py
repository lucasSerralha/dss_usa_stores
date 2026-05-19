"""
compare_optimization_nmaxgen.py — Sensibilidade N_MAX_GEN: MOEA/D vs U-NSGA-III

Varre N_MAX_GEN ∈ [50, 100, 150, 200, 300] com N_RUNS corridas independentes
e regista Hipervolume (HV) e nº soluções Pareto para cada configuração.

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

from optimization.moead_model  import run_optimization as run_moead
from optimization.unsga3_model import run_optimization as run_unsga3
from optimization.nsga2_model  import dummy_profit_function

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
    ("MOEA/D",     run_moead,   {"n_partitions": 99}),
    ("U-NSGA-III", run_unsga3,  {"n_partitions": 99}),
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

    # HV
    axes[0].errorbar(
        sub["n_max_gen"], sub["hv_mean"], yerr=sub["hv_std"],
        marker="o", label=algo, color=c, capsize=4, linewidth=2
    )
    # N soluções
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
