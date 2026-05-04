"""
moead_model.py — MOEA/D Multi-Objective Optimization — TIAPOSE DSS

MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition) decompõe
o problema multi-objetivo num conjunto de subproblemas escalares usando vetores
de peso. Cada subproblema é otimizado cooperativamente com os seus vizinhos no
espaço de pesos (neighborhood de tamanho n_neighbors).

Decomposição: Tchebycheff (default pymoo)
  minimize: max_i { w_i * |f_i(x) - z*_i| }
  onde z* é o ponto de referência ideal (nadir estimate).

Vantagem sobre NSGA-II para 2 objetivos: distribui uniformemente os vetores de
peso, garantindo melhor cobertura da fronteira de Pareto; converge rapidamente
em fronteiras convexas/côncavas simples.

Estrutura do problema (idêntica a nsga2_model.py):
  21 variáveis | 2 objetivos | n_constraints restrições G ≤ 0
"""

import logging
import os
import sys
from typing import Callable, Optional

import numpy as np
from pymoo.algorithms.moo.moead import MOEAD
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
    UNITS_CAP,
    _evaluate_all_stores,
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
log = logging.getLogger("TIAPOSE.MOEAD")

# ---------------------------------------------------------------------------
# Configuração das direções de referência (vetores de peso para MOEA/D)
# ---------------------------------------------------------------------------
# Para 2 objetivos, n_partitions=p gera p+1 vetores igualmente espaçados no
# simplex unitário: (0,1), (1/p, (p-1)/p), ..., (1,0).
# n_partitions=99 → 100 subproblemas / tamanho de população.
N_PARTITIONS: int = 99  # → pop_size efetivo = 100


def _build_ref_dirs(n_partitions: int = N_PARTITIONS) -> np.ndarray:
    return get_reference_directions("uniform", n_dim=2, n_partitions=n_partitions)


# ===========================================================================
# ORQUESTRADOR — ponto de entrada para outros módulos
# ===========================================================================

def run_optimization(
    store: str,
    forecast_customers: list,
    forecast_is_weekend: list,
    n_partitions: int = N_PARTITIONS,
    n_neighbors: int = 20,
    prob_neighbor_mating: float = 0.9,
    n_max_gen: int = 300,
    seed: int = 42,
    verbose: bool = True,
    profit_fn: Optional[Callable] = None,
) -> dict:
    """
    Configura e executa o MOEA/D, retornando as soluções da Fronteira de Pareto.

    Args:
        store: Nome da loja (ver STORE_PARAMS em profit_logic.py).
        forecast_customers: 7 inteiros com previsão de clientes (Seg → Dom).
        forecast_is_weekend: 7 booleanos de calendário.
        n_partitions: Número de partições do simplex → n_partitions+1 subproblemas.
                      Controla o tamanho da população e granularidade da fronteira.
        n_neighbors: Tamanho da vizinhança (cooperação entre subproblemas vizinhos).
                     Valores típicos: 10-30. Maior → mais exploração global.
        prob_neighbor_mating: Probabilidade de cruzamento dentro da vizinhança.
                              Valores típicos: 0.7-0.9. Menor → mais diversidade.
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
        "MOEA/D iniciado | loja=%-12s | subproblemas=%d | vizinhos=%d | max_gen=%d | seed=%d",
        store, pop_size, n_neighbors, n_max_gen, seed,
    )

    # --- Problema (idêntico para todos os algoritmos) ---
    problem = TiaposeOptimization(
        store=store,
        forecast_customers=forecast_customers,
        forecast_is_weekend=forecast_is_weekend,
        profit_fn=profit_fn,
    )

    # --- Algoritmo MOEA/D ---
    algorithm = MOEAD(
        ref_dirs=ref_dirs,
        n_neighbors=n_neighbors,
        prob_neighbor_mating=prob_neighbor_mating,
        # SBX: mesmos parâmetros do NSGA-II para comparação justa
        crossover=SBX(eta=15, prob=0.9, prob_var=1.0 / N_VARS),
        # PM: perturbações suaves, idêntico ao NSGA-II
        mutation=PM(eta=20, prob=1.0 / N_VARS),
        sampling=FloatRandomSampling(),
        # Repair garante inteiros em hr_x, hr_j após operadores genéticos
        repair=IntegerRepair(),
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
# ORQUESTRADOR CONJUNTO — O3 nas 4 lojas (84 variáveis)
# ===========================================================================

def run_joint_optimization(
    store_forecasts: list,
    store_weekends: list,
    n_partitions: int = N_PARTITIONS,
    n_neighbors: int = 20,
    prob_neighbor_mating: float = 0.9,
    n_max_gen: int = 300,
    seed: int = 42,
    verbose: bool = True,
) -> dict:
    """
    MOEA/D conjunto para O3 — 84 variáveis, 2 objetivos, restrição ≤10 000 unidades.

    Resolve O3: minimizar (−LucroTotal, StaffTotal) com restrição global de
    10 000 unidades vendidas entre as 4 lojas. A restrição é tratada como
    penalização em F[0], abordagem correcta para MOEA/D (decomposição escalar
    sem constraint-dominance).

    A solução O2 (máximo lucro com ≤10 k unidades) obtém-se de seguida com
    get_o2_solution(resultado), que devolve o ponto de máximo lucro da fronteira.

    Args:
        store_forecasts: Lista de 4 listas com 7 previsões de clientes cada.
                         Ordem: baltimore, lancaster, philadelphia, richmond.
        store_weekends:  Lista de 4 listas com 7 booleanos (True = fim-de-semana).
        n_partitions:    Partições do simplex → n_partitions+1 subproblemas.
        n_neighbors:     Tamanho da vizinhança de cooperação (típico: 10–30).
        prob_neighbor_mating: Prob. de cruzamento dentro da vizinhança (típico: 0.7–0.9).
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
        "MOEA/D Joint (O3) | n_var=84 | subproblemas=%d | vizinhos=%d | max_gen=%d | seed=%d",
        pop_size, n_neighbors, n_max_gen, seed,
    )

    problem = JointTiaposeOptimization(
        store_forecasts=store_forecasts,
        store_weekends=store_weekends,
        use_penalty=True,  # MOEA/D: restrição 10 k como penalização em F[0]
    )

    algorithm = MOEAD(
        ref_dirs=ref_dirs,
        n_neighbors=n_neighbors,
        prob_neighbor_mating=prob_neighbor_mating,
        crossover=SBX(eta=15, prob=0.9, prob_var=1.0 / N_VARS_JOINT),
        mutation=PM(eta=20, prob=1.0 / N_VARS_JOINT),
        sampling=JointBiasedSampling(),
        repair=JointIntegerRepair(),
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

    log.info("MOEA/D Joint concluído em %d gerações.", res.algorithm.n_gen)
    result = extract_joint_pareto_solutions(res)
    return _filter_feasible_joint(result, store_forecasts, store_weekends)


def _filter_feasible_joint(result: dict, store_forecasts: list, store_weekends: list) -> dict:
    """
    Removes solutions where actual total units > 10 000 from a MOEA/D result.

    MOEA/D uses a penalty-augmented objective, so its Pareto front may include
    solutions that are near-infeasible (penalty too small to exclude them).
    Re-evaluating and filtering ensures the returned front is 100 % feasible.
    """
    n = len(result.get("lucro", []))
    if n == 0:
        return result

    mask = np.array([
        _evaluate_all_stores(result["pareto_X"][i], store_forecasts, store_weekends)[2] <= UNITS_CAP
        for i in range(n)
    ])

    n_removed = int((~mask).sum())
    if n_removed:
        log.info("MOEA/D: %d solução(ões) inviáveis removidas da fronteira (units > %d).", n_removed, UNITS_CAP)

    if not mask.any():
        log.warning("MOEA/D: nenhuma solução verdadeiramente viável após filtragem — a devolver resultado completo.")
        return result

    return {
        "pareto_F": result["pareto_F"][mask],
        "pareto_X": result["pareto_X"][mask],
        "lucro":    result["lucro"][mask],
        "staff":    result["staff"][mask],
        "plans":    [p for p, m in zip(result["plans"], mask) if m],
    }


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
    print(f"  TIAPOSE — MOEA/D | O1 | Loja: {STORE.upper()}")
    print("=" * 62)

    res_o1 = run_optimization(
        store=STORE,
        forecast_customers=CLIENTES_PREVISTOS,
        forecast_is_weekend=CALENDARIO_FDS,
        n_partitions=N_PARTITIONS,
        n_neighbors=20,
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
    # Demo O3 — conjunto 4 lojas, restrição ≤10 000 unidades (penalização)
    # ------------------------------------------------------------------
    ALL_FORECASTS = [
        [80, 65, 70, 75, 60, 90, 110],   # baltimore
        [70, 55, 60, 65, 50, 80, 100],   # lancaster
        [90, 75, 80, 85, 70, 100, 120],  # philadelphia
        [85, 70, 75, 80, 65, 95, 115],   # richmond
    ]
    ALL_WEEKENDS = [[False, False, False, False, False, True, True]] * 4

    print("\n" + "=" * 62)
    print("  TIAPOSE — MOEA/D | O3 | 4 Lojas Conjuntas (84 variáveis)")
    print("=" * 62)

    res_o3 = run_joint_optimization(
        store_forecasts=ALL_FORECASTS,
        store_weekends=ALL_WEEKENDS,
        n_partitions=N_PARTITIONS,
        n_neighbors=20,
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
