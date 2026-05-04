"""
unsga3_model.py — U-NSGA-III Multi-Objective Optimization — TIAPOSE DSS

U-NSGA-III (Unified NSGA-III) é uma extensão do NSGA-II/NSGA-III que combina:
  - Seleção por torneio (como NSGA-II) para melhor pressão seletiva em 2 objetivos
  - Direções de referência estruturadas Das-Dennis (como NSGA-III) para manter
    a diversidade ao longo da fronteira de Pareto

Vantagem sobre NSGA-III puro para 2 objetivos: o torneio binário preserva
melhor as soluções extremas e evita a perda de diversidade que NSGA-III pode
sofrer quando as direções de referência são mal alinhadas com a fronteira real.

Vantagem sobre NSGA-II: as direções de referência explícitas garantem cobertura
mais uniforme mesmo quando a fronteira é irregular ou não convexa.

Estrutura do problema (idêntica a nsga2_model.py):
  21 variáveis | 2 objetivos | n_constraints restrições G ≤ 0
"""

import logging
import os
import sys
from typing import Callable, Optional

import numpy as np
from pymoo.algorithms.moo.unsga3 import UNSGA3
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.util.ref_dirs import get_reference_directions

# ---------------------------------------------------------------------------
# Path bootstrapping
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Reutilizar componentes partilhados do nsga2_model (problema, repair, utils)
from optimization.nsga2_model import (
    IntegerRepair,
    TiaposeOptimization,
    dummy_profit_function,
    extract_pareto_solutions,
    N_VARS,
)

# Componentes para otimização conjunta O3 (84 variáveis, 4 lojas)
from optimization.joint_problem import (
    JointBiasedSampling,
    JointIntegerRepair,
    JointTiaposeOptimization,
    N_VARS_JOINT,
    extract_joint_pareto_solutions,
    get_o2_solution,
)

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("TIAPOSE.UNSGA3")

# ---------------------------------------------------------------------------
# Configuração das direções de referência (Das-Dennis para U-NSGA-III)
# ---------------------------------------------------------------------------
# Das-Dennis gera pontos estruturados no simplex unitário.
# Para 2 objetivos, n_partitions=p → p+1 direções igualmente espaçadas.
# pop_size = len(ref_dirs) para garantir um indivíduo por direção.
N_PARTITIONS: int = 99  # → 100 direções de referência / pop_size


def _build_ref_dirs(n_partitions: int = N_PARTITIONS) -> np.ndarray:
    return get_reference_directions("das-dennis", n_dim=2, n_partitions=n_partitions)


# ===========================================================================
# ORQUESTRADOR — ponto de entrada para outros módulos
# ===========================================================================

def run_optimization(
    store: str,
    forecast_customers: list,
    forecast_is_weekend: list,
    n_partitions: int = N_PARTITIONS,
    n_max_gen: int = 300,
    seed: int = 42,
    verbose: bool = True,
    profit_fn: Optional[Callable] = None,
) -> dict:
    """
    Configura e executa o U-NSGA-III, retornando as soluções da Fronteira de Pareto.

    Args:
        store: Nome da loja (ver STORE_PARAMS em profit_logic.py).
        forecast_customers: 7 inteiros com previsão de clientes (Seg → Dom).
        forecast_is_weekend: 7 booleanos de calendário.
        n_partitions: Número de partições Das-Dennis → n_partitions+1 direções.
                      Determina o tamanho da população e granularidade da fronteira.
        n_max_gen: Número máximo de gerações.
        seed: Semente para reprodutibilidade.
        verbose: Progresso a cada geração se True.
        profit_fn: Função de lucro plugável.
                   None  → usa optimize_weekly_wrapper (produção).
                   dummy_profit_function → para testes sem profit_logic.

    Returns:
        Dicionário de extract_pareto_solutions (mesma interface que NSGA-II).
    """
    ref_dirs = _build_ref_dirs(n_partitions)
    pop_size = len(ref_dirs)  # = n_partitions + 1

    log.info(
        "U-NSGA-III iniciado | loja=%-12s | pop=%d | direções=%d | max_gen=%d | seed=%d",
        store, pop_size, len(ref_dirs), n_max_gen, seed,
    )

    # --- Problema (idêntico para todos os algoritmos) ---
    problem = TiaposeOptimization(
        store=store,
        forecast_customers=forecast_customers,
        forecast_is_weekend=forecast_is_weekend,
        profit_fn=profit_fn,
    )

    # --- Algoritmo U-NSGA-III ---
    algorithm = UNSGA3(
        ref_dirs=ref_dirs,
        pop_size=pop_size,
        # SBX: mesmos parâmetros do NSGA-II para comparação justa
        crossover=SBX(eta=15, prob=0.9, prob_var=1.0 / N_VARS),
        # PM: perturbações suaves, idêntico ao NSGA-II
        mutation=PM(eta=20, prob=1.0 / N_VARS),
        sampling=FloatRandomSampling(),
        # Repair garante inteiros em hr_x, hr_j após operadores genéticos
        repair=IntegerRepair(),
        eliminate_duplicates=True,
    )

    # --- Critério de paragem adaptativo (idêntico ao NSGA-II) ---
    termination = DefaultMultiObjectiveTermination(
        xtol=1e-6,
        cvtol=1e-6,
        ftol=0.0025,
        period=30,
        n_max_gen=n_max_gen,
    )

    # --- Execução ---
    res = minimize(
        problem,
        algorithm,
        termination,
        seed=seed,
        verbose=verbose,
        save_history=False,
    )

    log.info("Concluído em %d gerações.", res.algorithm.n_gen)
    return extract_pareto_solutions(res)


# ===========================================================================
# ORQUESTRADOR CONJUNTO — O2/O3 nas 4 lojas (84 variáveis)
# ===========================================================================

def run_joint_optimization(
    store_forecasts: list,
    store_weekends: list,
    n_partitions: int = N_PARTITIONS,
    n_max_gen: int = 300,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    U-NSGA-III conjunto para O3 — 84 variáveis, 2 objetivos, restrição ≤10 000 unidades.

    Resolve O3: minimizar (−LucroTotal, StaffTotal) com a restrição global de
    10 000 unidades vendidas entre as 4 lojas exposta como n_ieq_constr=1.
    O U-NSGA-III trata-a via constraint-dominance: soluções inviáveis nunca
    dominam soluções viáveis, garantindo uma fronteira de Pareto 100% válida.

    A solução O2 (máximo lucro com ≤10 k unidades) obtém-se de seguida com
    get_o2_solution(resultado), que devolve o ponto de máximo lucro da fronteira.

    Args:
        store_forecasts: Lista de 4 listas com 7 previsões de clientes cada.
                         Ordem: baltimore, lancaster, philadelphia, richmond.
        store_weekends:  Lista de 4 listas com 7 booleanos (True = fim-de-semana).
        n_partitions:    Partições Das-Dennis → n_partitions+1 direções de referência.
        n_max_gen:       Máximo de gerações.
        seed:            Semente para reprodutibilidade.
        verbose:         Progresso por geração.

    Returns:
        Dicionário com chaves: pareto_F, pareto_X, lucro, staff, plans.
        Para extrair O2: get_o2_solution(resultado).
    """
    ref_dirs = _build_ref_dirs(n_partitions)
    pop_size = len(ref_dirs)

    log.info(
        "U-NSGA-III Joint (O3) | n_var=84 | pop=%d | direções=%d | max_gen=%d | seed=%d",
        pop_size, len(ref_dirs), n_max_gen, seed,
    )

    problem = JointTiaposeOptimization(
        store_forecasts=store_forecasts,
        store_weekends=store_weekends,
        use_penalty=False,  # U-NSGA-III: restrição 10 k via n_ieq_constr=1
    )

    algorithm = UNSGA3(
        ref_dirs=ref_dirs,
        pop_size=pop_size,
        crossover=SBX(eta=15, prob=0.9, prob_var=1.0 / N_VARS_JOINT),
        mutation=PM(eta=20, prob=1.0 / N_VARS_JOINT),
        sampling=JointBiasedSampling(),
        repair=JointIntegerRepair(),
        eliminate_duplicates=True,
    )

    termination = DefaultMultiObjectiveTermination(
        xtol=1e-6,
        cvtol=1e-6,
        ftol=0.0025,
        period=30,
        n_max_gen=n_max_gen,
    )

    res = minimize(
        problem,
        algorithm,
        termination,
        seed=seed,
        verbose=verbose,
        save_history=False,
    )

    log.info("U-NSGA-III Joint concluído em %d gerações.", res.algorithm.n_gen)
    return extract_joint_pareto_solutions(res)


# ===========================================================================
# DEMO — execução direta do ficheiro
# ===========================================================================

if __name__ == "__main__":
    import pprint

    # ------------------------------------------------------------------
    # Demo O1 — per-store (baltimore), função dummy para rapidez
    # ------------------------------------------------------------------
    STORE = "baltimore"
    CLIENTES_PREVISTOS = [80, 65, 70, 75, 60, 90, 110]
    CALENDARIO_FDS     = [False, False, False, False, False, True, True]

    print("=" * 62)
    print(f"  TIAPOSE — U-NSGA-III | O1 | Loja: {STORE.upper()}")
    print("=" * 62)

    res_o1 = run_optimization(
        store=STORE,
        forecast_customers=CLIENTES_PREVISTOS,
        forecast_is_weekend=CALENDARIO_FDS,
        n_partitions=N_PARTITIONS,
        n_max_gen=200,
        seed=42,
        verbose=True,
        profit_fn=dummy_profit_function,
    )

    n = len(res_o1["lucro"])
    print(f"\n  Pareto O1: {n} soluções\n")
    print(f"  {'Sol':>4} | {'Lucro':>12} | {'Staff':>8}")
    print("  " + "-" * 30)
    for i in range(min(n, 8)):
        print(f"  {i+1:>4} | {res_o1['lucro'][i]:>12.2f} | {res_o1['staff'][i]:>8.0f}")

    # ------------------------------------------------------------------
    # Demo O3 — conjunto 4 lojas, restrição hard ≤10 000 unidades
    # ------------------------------------------------------------------
    ALL_FORECASTS = [
        [80, 65, 70, 75, 60, 90, 110],   # baltimore
        [70, 55, 60, 65, 50, 80, 100],   # lancaster
        [90, 75, 80, 85, 70, 100, 120],  # philadelphia
        [85, 70, 75, 80, 65, 95, 115],   # richmond
    ]
    ALL_WEEKENDS = [[False, False, False, False, False, True, True]] * 4

    print("\n" + "=" * 62)
    print("  TIAPOSE — U-NSGA-III | O3 | 4 Lojas Conjuntas (84 variáveis)")
    print("=" * 62)

    res_o3 = run_joint_optimization(
        store_forecasts=ALL_FORECASTS,
        store_weekends=ALL_WEEKENDS,
        n_partitions=N_PARTITIONS,
        n_max_gen=100,
        seed=42,
        verbose=True,
    )

    n = len(res_o3["lucro"])
    print(f"\n  Fronteira Pareto O3: {n} soluções (restrição ≤10 000 unidades)\n")
    print(f"  {'Sol':>4} | {'Lucro Total':>14} | {'Staff Total':>12}")
    print("  " + "-" * 38)
    for i in range(min(n, 8)):
        print(
            f"  {i+1:>4} | {res_o3['lucro'][i]:>14.2f}"
            f" | {res_o3['staff'][i]:>12.0f}"
        )

    # Solução O2: ponto de máximo lucro da fronteira O3
    o2 = get_o2_solution(res_o3)
    print(f"\n  Solução O2 (máximo lucro): ${o2['lucro']:.2f} | Staff: {o2['staff']:.0f}")

    if res_o3["plans"]:
        print("\n  Plano conjunto O3 — melhor lucro (baltimore):")
        pprint.pprint(res_o3["plans"][0]["baltimore"], indent=4)
