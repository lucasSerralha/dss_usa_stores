"""
unsga3_diagnostics.py — Diagnóstico e Comparação do U-NSGA-III — TIAPOSE DSS

Compara o U-NSGA-III com o NSGA-II (baseline) no problema de otimização semanal.

Métricas avaliadas:
  - Hipervolume (HV): área da fronteira de Pareto dominada; maior é melhor.
  - Número de soluções não-dominadas: diversidade da fronteira.
  - Tempo de execução por corrida.
  - Estatísticas ao longo de N_RUNS corridas independentes (média ± desvio).

O diagnóstico usa a dummy_profit_function (sem profit_logic.py) para velocidade.
Para avaliação com dados reais, substituir profit_fn=None nas chamadas.
"""

import logging
import os
import sys
import time

import matplotlib
matplotlib.use("Agg")  # modo não-interativo para guardar figuras sem display
import matplotlib.pyplot as plt
import numpy as np

# ---------------------------------------------------------------------------
# Path bootstrapping — permite executar diretamente (python unsga3_diagnostics.py)
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, "..", ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from optimization.unsga3_model import run_optimization as run_unsga3
from optimization.nsga2_model import (
    dummy_profit_function,
    run_optimization as run_nsga2,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.WARNING,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("TIAPOSE.UNSGA3.Diagnostics")
logging.getLogger("TIAPOSE").setLevel(logging.WARNING)

# ---------------------------------------------------------------------------
# Parâmetros do diagnóstico
# ---------------------------------------------------------------------------
STORE              = "baltimore"
CLIENTES           = [80, 65, 70, 75, 60, 90, 110]   # Seg → Dom
FDS                = [False, False, False, False, False, True, True]
N_RUNS             = 5       # corridas independentes por algoritmo
N_MAX_GEN          = 150     # gerações máximas (reduzido para diagnóstico rápido)
POP_SIZE_NSGA2     = 100
N_PARTITIONS_U3    = 99      # 100 direções de referência

# Ponto de referência para hipervolume: pior ponto possível no espaço de objetivos
# F[0] = -Lucro → pior = valor mais alto (lucro muito negativo)
# F[1] = Staff  → pior = staff máximo possível (7 dias × 15+15 = 210)
HV_REF_POINT = np.array([0.0, 210.0])


# ===========================================================================
# Utilitários
# ===========================================================================

def compute_hypervolume(pareto_F: np.ndarray, ref_point: np.ndarray) -> float:
    """
    Calcula o hipervolume da fronteira de Pareto relativamente ao ponto de referência.
    Usa pymoo's Hypervolume indicator (exato para 2 objetivos).
    """
    from pymoo.indicators.hv import Hypervolume
    if pareto_F is None or len(pareto_F) == 0:
        return 0.0
    dominated = np.all(pareto_F < ref_point, axis=1)
    if not np.any(dominated):
        return 0.0
    hv = Hypervolume(ref_point=ref_point)
    return float(hv.do(pareto_F[dominated]))


def run_single(algorithm: str, seed: int) -> dict:
    """Executa um algoritmo com uma seed e devolve métricas."""
    t0 = time.perf_counter()
    if algorithm == "U-NSGA-III":
        res = run_unsga3(
            store=STORE,
            forecast_customers=CLIENTES,
            forecast_is_weekend=FDS,
            n_partitions=N_PARTITIONS_U3,
            n_max_gen=N_MAX_GEN,
            seed=seed,
            verbose=False,
            profit_fn=dummy_profit_function,
        )
    else:  # NSGA-II
        res = run_nsga2(
            store=STORE,
            forecast_customers=CLIENTES,
            forecast_is_weekend=FDS,
            pop_size=POP_SIZE_NSGA2,
            n_max_gen=N_MAX_GEN,
            seed=seed,
            verbose=False,
            profit_fn=dummy_profit_function,
        )
    elapsed = time.perf_counter() - t0

    n_sol = len(res["lucro"])
    hv    = compute_hypervolume(res["pareto_F"], HV_REF_POINT)

    return {
        "algorithm": algorithm,
        "seed":      seed,
        "n_sol":     n_sol,
        "hv":        hv,
        "elapsed":   elapsed,
        "pareto_F":  res["pareto_F"],
        "lucro":     res["lucro"],
        "staff":     res["staff"],
    }


# ===========================================================================
# Diagnóstico principal
# ===========================================================================

def run_diagnostics():
    print("\n" + "=" * 70)
    print("  TIAPOSE — Diagnóstico U-NSGA-III vs NSGA-II")
    print(f"  Loja: {STORE.upper()} | {N_RUNS} corridas | max_gen={N_MAX_GEN}")
    print("=" * 70)

    results = {"U-NSGA-III": [], "NSGA-II": []}

    for algo in ["U-NSGA-III", "NSGA-II"]:
        print(f"\n  Algoritmo: {algo}")
        for run_i in range(N_RUNS):
            seed = 42 + run_i * 7
            r = run_single(algo, seed)
            results[algo].append(r)
            print(
                f"    Run {run_i+1}/{N_RUNS} | seed={seed:3d} | "
                f"soluções={r['n_sol']:3d} | HV={r['hv']:.4f} | "
                f"tempo={r['elapsed']:.2f}s"
            )

    # --- Resumo estatístico ---
    print("\n" + "=" * 70)
    print(f"  {'Métrica':<28} | {'U-NSGA-III':>18} | {'NSGA-II':>18}")
    print("  " + "-" * 68)

    for metric, label in [("hv", "Hipervolume"), ("n_sol", "Nº Soluções Pareto"), ("elapsed", "Tempo (s)")]:
        for algo in ["U-NSGA-III", "NSGA-II"]:
            vals = [r[metric] for r in results[algo]]
            mu, sigma = np.mean(vals), np.std(vals)
            tag = f"{mu:.4f} ± {sigma:.4f}" if metric == "hv" else (
                f"{mu:.1f} ± {sigma:.1f}" if metric == "n_sol" else
                f"{mu:.2f} ± {sigma:.2f}"
            )
            results[algo + "_" + metric] = (mu, sigma, tag)

        tag_u3    = results["U-NSGA-III_" + metric][2]
        tag_nsga2 = results["NSGA-II_"    + metric][2]
        print(f"  {label:<28} | {tag_u3:>18} | {tag_nsga2:>18}")

    # --- Lucro e staff médios da melhor solução ---
    print("\n  Melhor solução (maior lucro), médias sobre corridas:")
    print(f"  {'Métrica':<28} | {'U-NSGA-III':>18} | {'NSGA-II':>18}")
    print("  " + "-" * 68)
    for algo in ["U-NSGA-III", "NSGA-II"]:
        best_lucros = [r["lucro"][0] if len(r["lucro"]) > 0 else 0.0 for r in results[algo]]
        best_staffs = [r["staff"][0] if len(r["staff"]) > 0 else 0.0 for r in results[algo]]
        results[algo + "_best_lucro"] = best_lucros
        results[algo + "_best_staff"] = best_staffs

    for metric, label in [("best_lucro", "Lucro máx. (€)"), ("best_staff", "Staff @ lucro máx.")]:
        vals_u3   = results["U-NSGA-III_" + metric]
        vals_n    = results["NSGA-II_"    + metric]
        tag_u3    = f"{np.mean(vals_u3):.2f} ± {np.std(vals_u3):.2f}"
        tag_n     = f"{np.mean(vals_n):.2f} ± {np.std(vals_n):.2f}"
        print(f"  {label:<28} | {tag_u3:>18} | {tag_n:>18}")

    print("=" * 70)

    # --- Guardar gráficos ---
    _plot_pareto_comparison(results)
    _plot_hypervolume_comparison(results)

    return results


def _plot_pareto_comparison(results: dict):
    """Sobrepõe as fronteiras de Pareto da última corrida de cada algoritmo."""
    fig, ax = plt.subplots(figsize=(8, 5))

    colors  = {"U-NSGA-III": "#8e24aa", "NSGA-II": "#2a7ae0"}
    markers = {"U-NSGA-III": "^",       "NSGA-II": "s"}

    for algo in ["U-NSGA-III", "NSGA-II"]:
        last = results[algo][-1]
        if len(last["lucro"]) == 0:
            continue
        ax.scatter(
            last["lucro"], last["staff"],
            c=colors[algo], marker=markers[algo],
            s=40, alpha=0.75, label=f"{algo} (seed={last['seed']})",
            edgecolors="none",
        )

    ax.set_xlabel("Lucro Semanal (€)", fontsize=11)
    ax.set_ylabel("Staff Total Semanal", fontsize=11)
    ax.set_title(f"Fronteira de Pareto — U-NSGA-III vs NSGA-II\nLoja: {STORE.upper()}", fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_dir = os.path.join(_SRC_DIR, "..", "results", "04_Model_Testing", "unsga3")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "pareto_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"\n  Gráfico guardado: {os.path.abspath(path)}")


def _plot_hypervolume_comparison(results: dict):
    """Boxplot do hipervolume por algoritmo ao longo das N_RUNS corridas."""
    fig, ax = plt.subplots(figsize=(6, 4))

    data   = [
        [r["hv"] for r in results["U-NSGA-III"]],
        [r["hv"] for r in results["NSGA-II"]],
    ]
    labels = ["U-NSGA-III", "NSGA-II"]
    colors = ["#8e24aa", "#2a7ae0"]

    bp = ax.boxplot(data, labels=labels, patch_artist=True, widths=0.4)
    for patch, color in zip(bp["boxes"], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.6)

    ax.set_ylabel("Hipervolume", fontsize=11)
    ax.set_title(
        f"Distribuição do Hipervolume ({N_RUNS} corridas)\nLoja: {STORE.upper()}", fontsize=12
    )
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()

    out_dir = os.path.join(_SRC_DIR, "..", "results", "04_Model_Testing", "unsga3")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "hypervolume_comparison.png")
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"  Gráfico guardado: {os.path.abspath(path)}")


# ===========================================================================
# Ponto de entrada
# ===========================================================================

if __name__ == "__main__":
    run_diagnostics()
