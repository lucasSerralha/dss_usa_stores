"""
moead_model.py — MOEA/D Multi-Objective Optimization — TIAPOSE DSS

MOEA/D (Multi-Objective Evolutionary Algorithm based on Decomposition) decompõe
o problema multi-objetivo num conjunto de subproblemas escalares usando vetores
de peso. Cada subproblema é otimizado cooperativamente com os seus vizinhos no
espaço de pesos (neighborhood de tamanho n_neighbors).

Decomposição: Tchebycheff (default pymoo)
  minimize: max_i { w_i * |f_i(x) - z*_i| }
  onde z* é o ponto de referência ideal (nadir estimate).

Nota de design sobre restrições:
  A implementação MOEAD do pymoo não suporta n_ieq_constr. As restrições de
  staff são tratadas via penalização estática no objetivo F[0]:
    F[0] = -Lucro + PENALTY_WEIGHT × Σ_d max(0, staff_d − cap_d)
  Após a otimização, as soluções inviáveis (violação > 0) são removidas antes
  da extração do front de Pareto.

Estrutura do problema:
  21 variáveis | 2 objetivos | sem n_ieq_constr (penalty em F[0])
"""

import logging
import os
import sys
from typing import Callable, Optional

import numpy as np
from pymoo.algorithms.moo.moead import MOEAD
from pymoo.core.problem import ElementwiseProblem
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultMultiObjectiveTermination
from pymoo.util.nds.non_dominated_sorting import NonDominatedSorting
from pymoo.util.ref_dirs import get_reference_directions

# ---------------------------------------------------------------------------
# Path bootstrapping
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

# Reutilizar componentes partilhados do nsga2_model
from optimization.nsga2_model import (
    IntegerRepair,
    decode_solution,
    dummy_profit_function,
    INT_IDX,
    N_DAYS,
    N_VARS,
    XL,
    XU,
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
# Configuração das direções de referência
# ---------------------------------------------------------------------------
N_PARTITIONS: int = 99  # → 100 subproblemas / tamanho de população


def _build_ref_dirs(n_partitions: int = N_PARTITIONS) -> np.ndarray:
    return get_reference_directions("uniform", n_dim=2, n_partitions=n_partitions)


# ===========================================================================
# 1. PROBLEMA COM PENALIZAÇÃO (sem n_ieq_constr — compatível com MOEA/D)
# ===========================================================================

class TiaposeOptimizationPenalty(ElementwiseProblem):
    """
    Formulação do problema de otimização para MOEA/D.

    As restrições de staff são incorporadas como penalização estática em F[0]:
      F[0] = -Lucro + PENALTY_WEIGHT × violação_total
      F[1] =  Staff_Total_semanal

    Após a otimização, as soluções com violação > 0 são descartadas na extração
    do front de Pareto, garantindo que apenas soluções viáveis são reportadas.
    """

    PENALTY_WEIGHT: float = 100_000.0  # por unidade de staff em excesso

    def __init__(
        self,
        store: str,
        forecast_customers: list,
        forecast_is_weekend: list,
        profit_fn: Optional[Callable] = None,
    ):
        assert len(forecast_customers) == N_DAYS
        assert len(forecast_is_weekend) == N_DAYS

        self.store = store
        self.forecast_customers = list(forecast_customers)
        self.forecast_is_weekend = list(forecast_is_weekend)

        try:
            from utils.profit_logic import optimize_weekly_wrapper
            default_fn = optimize_weekly_wrapper
        except ImportError:
            default_fn = dummy_profit_function

        self.profit_fn = profit_fn if profit_fn is not None else default_fn

        self._weekday_idx = np.array(
            [d for d, w in enumerate(forecast_is_weekend) if not w]
        )
        self._weekend_idx = np.array(
            [d for d, w in enumerate(forecast_is_weekend) if w]
        )

        super().__init__(
            n_var=N_VARS,
            n_obj=2,
            n_ieq_constr=0,  # MOEA/D não suporta restrições explícitas
            xl=XL,
            xu=XU,
            elementwise=True,
        )

        log.info(
            "Problema (penalty) criado | loja=%-12s | dias_úteis=%d | fins_semana=%d",
            store, len(self._weekday_idx), len(self._weekend_idx),
        )

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> None:
        f1, f2, _ = self.profit_fn(
            decision_vars=x,
            store=self.store,
            forecast_customers=self.forecast_customers,
            forecast_is_weekend=self.forecast_is_weekend,
        )

        # Calcular violação das restrições de staff
        staff_per_day = np.round(x[INT_IDX]).reshape(N_DAYS, 2).sum(axis=1)
        viol_weekday = np.maximum(0.0, staff_per_day[self._weekday_idx] - 8.0)
        viol_weekend = np.maximum(0.0, staff_per_day[self._weekend_idx] - 12.0)
        total_violation = float(viol_weekday.sum() + viol_weekend.sum())

        out["F"] = [f1 + self.PENALTY_WEIGHT * total_violation, f2]

    def is_feasible(self, x: np.ndarray) -> bool:
        """Verifica se uma solução x satisfaz as restrições de staff."""
        staff_per_day = np.round(x[INT_IDX]).reshape(N_DAYS, 2).sum(axis=1)
        return (
            np.all(staff_per_day[self._weekday_idx] <= 8) and
            np.all(staff_per_day[self._weekend_idx] <= 12)
        )


# ===========================================================================
# 2. EXTRAÇÃO DO FRONT DE PARETO (pós-processamento específico MOEA/D)
# ===========================================================================

def extract_pareto_moead(res, problem: TiaposeOptimizationPenalty) -> dict:
    """
    Extrai as soluções viáveis não-dominadas do resultado MOEA/D.

    O MOEAD mantém um indivíduo por subproblema (sem archive de Pareto
    explícito). Esta função:
      1. Recolhe toda a população final (res.pop).
      2. Filtra soluções inviáveis (violação de staff).
      3. Aplica non-dominated sorting nas soluções viáveis.
      4. Devolve o front de Pareto no mesmo formato de extract_pareto_solutions.
    """
    # Recolher população final (MOEAD usa res.pop, não res.X como arquivo)
    if res.pop is not None and len(res.pop) > 0:
        all_X = res.pop.get("X")
        all_F = res.pop.get("F")
    elif res.X is not None:
        all_X = np.atleast_2d(res.X)
        all_F = np.atleast_2d(res.F)
    else:
        log.warning("Resultado MOEA/D sem soluções.")
        return {"pareto_F": np.array([]), "pareto_X": np.array([]),
                "lucro": np.array([]), "staff": np.array([]), "plans": []}

    # Filtrar soluções viáveis
    feas_mask = np.array([problem.is_feasible(all_X[i]) for i in range(len(all_X))])

    if not np.any(feas_mask):
        log.warning(
            "MOEA/D: nenhuma solução 100%% viável encontrada (%d inviáveis). "
            "Considerar aumentar PENALTY_WEIGHT ou n_max_gen.",
            len(all_X),
        )
        # Fallback: retornar todas as soluções (com menor violação)
        feas_mask = np.ones(len(all_X), dtype=bool)

    X_feas = all_X[feas_mask]
    F_feas = all_F[feas_mask]

    # Non-dominated sorting nas soluções viáveis
    nds = NonDominatedSorting()
    front_idx = nds.do(F_feas, only_non_dominated_front=True)

    X_pareto = X_feas[front_idx]
    F_pareto = F_feas[front_idx]

    lucro_col = -F_pareto[:, 0]  # reverter negação → valores positivos
    staff_col =  F_pareto[:, 1]
    order = np.argsort(-lucro_col)

    log.info(
        "MOEA/D Pareto: %d soluções | Lucro [%.0f, %.0f] | Staff [%.0f, %.0f]",
        len(F_pareto),
        lucro_col.min(), lucro_col.max(),
        staff_col.min(), staff_col.max(),
    )

    return {
        "pareto_F": F_pareto[order],
        "pareto_X": X_pareto[order],
        "lucro":    lucro_col[order],
        "staff":    staff_col[order],
        "plans":    [decode_solution(X_pareto[i]) for i in order],
    }


# ===========================================================================
# 3. ORQUESTRADOR — ponto de entrada para outros módulos
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
        n_neighbors: Tamanho da vizinhança MOEA/D. Valores típicos: 10-30.
        prob_neighbor_mating: Probabilidade de cruzamento na vizinhança (0.7-0.9).
        n_max_gen: Número máximo de gerações.
        seed: Semente para reprodutibilidade.
        verbose: Progresso a cada geração se True.
        profit_fn: Função de lucro plugável (None → produção, dummy → testes).

    Returns:
        Dicionário compatível com extract_pareto_solutions do nsga2_model.
    """
    ref_dirs = _build_ref_dirs(n_partitions)
    pop_size = len(ref_dirs)

    log.info(
        "MOEA/D iniciado | loja=%-12s | subproblemas=%d | vizinhos=%d | max_gen=%d | seed=%d",
        store, pop_size, n_neighbors, n_max_gen, seed,
    )

    problem = TiaposeOptimizationPenalty(
        store=store,
        forecast_customers=forecast_customers,
        forecast_is_weekend=forecast_is_weekend,
        profit_fn=profit_fn,
    )

    algorithm = MOEAD(
        ref_dirs=ref_dirs,
        n_neighbors=n_neighbors,
        prob_neighbor_mating=prob_neighbor_mating,
        crossover=SBX(eta=15, prob=0.9, prob_var=1.0 / N_VARS),
        mutation=PM(eta=20, prob=1.0 / N_VARS),
        sampling=FloatRandomSampling(),
        repair=IntegerRepair(),
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

    log.info("Concluído em %d gerações.", res.algorithm.n_gen)
    return extract_pareto_moead(res, problem)


# ===========================================================================
# 4. DEMO — execução direta do ficheiro
# ===========================================================================

if __name__ == "__main__":
    import pprint

    STORE = "baltimore"
    CLIENTES_PREVISTOS = [80, 65, 70, 75, 60, 90, 110]
    CALENDARIO_FDS     = [False, False, False, False, False, True, True]

    print("=" * 62)
    print(f"  TIAPOSE — MOEA/D | Loja: {STORE.upper()}")
    print("=" * 62)

    resultados = run_optimization(
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
