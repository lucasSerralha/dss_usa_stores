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
    decode_solution,
    dummy_profit_function,
    extract_pareto_solutions,
    N_DAYS,
    N_VARS,
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
# DEMO — execução direta do ficheiro
# ===========================================================================

if __name__ == "__main__":
    import pprint

    STORE = "baltimore"
    CLIENTES_PREVISTOS  = [80, 65, 70, 75, 60, 90, 110]
    CALENDARIO_FDS      = [False, False, False, False, False, True, True]

    print("=" * 62)
    print(f"  TIAPOSE — U-NSGA-III | Loja: {STORE.upper()}")
    print("=" * 62)

    resultados = run_optimization(
        store=STORE,
        forecast_customers=CLIENTES_PREVISTOS,
        forecast_is_weekend=CALENDARIO_FDS,
        n_partitions=N_PARTITIONS,
        n_max_gen=200,
        seed=42,
        verbose=True,
        profit_fn=dummy_profit_function,
    )

    n = len(resultados["lucro"])
    print(f"\n  Fronteira de Pareto: {n} soluções não-dominadas\n")
    print(f"  {'Sol':>4} | {'Lucro (€)':>12} | {'Staff Total':>11}")
    print("  " + "-" * 34)
    for i in range(min(n, 10)):
        print(
            f"  {i+1:>4} | {resultados['lucro'][i]:>12.2f}"
            f" | {resultados['staff'][i]:>11.0f}"
        )

    if resultados["plans"]:
        print("\n  Plano semanal — melhor lucro:")
        pprint.pprint(resultados["plans"][0], indent=4)
