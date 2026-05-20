"""
joint_problem.py — Joint Multi-Objective Problem (O3) — TIAPOSE DSS

Decision vector X ∈ ℝ^84  (4 stores × 21 variables each):
  Block s (s=0..3): X[21·s : 21·s+21]
    X[3d+0] = pr_d   ∈ [0.00, 0.30]   daily discount
    X[3d+1] = hr_x_d ∈ {0, …, 15}     expert staff (int)
    X[3d+2] = hr_j_d ∈ {0, …, 15}     junior staff (int)
  Store order: baltimore (0), lancaster (1), philadelphia (2), richmond (3)

O3 — 2 objectives + 1 hard global constraint:
  minimize  F[0] = −Total_Profit_all_stores
  minimize  F[1] =  Total_Staff_all_stores
  subject to  G[0] = Total_Units_all_stores − 10 000 ≤ 0

O2 is derived from O3: the solution with maximum profit on the Pareto front
already satisfies the 10 k constraint, so it IS the O2 optimum.
Use get_o2_solution(pareto_result) to extract it.

Restrição global (única, conforme enunciado): Total_Units ≤ 10 000.

use_penalty=True  → MOEA/D:    10 k cap folded into F[0] as penalty.
use_penalty=False → U-NSGA-III: 10 k cap exposed as n_ieq_constr=1.
"""

import logging
import os
import sys

import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.repair import Repair
from pymoo.core.sampling import Sampling

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from utils.profit_logic import (
    STORE_PARAMS,
    calculate_daily_metrics,
)
from optimization.nsga2_model import INT_IDX, N_DAYS, N_VARS, XL, XU

log = logging.getLogger("TIAPOSE.Joint")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
STORES = ['baltimore', 'lancaster', 'philadelphia', 'richmond']
N_STORES = len(STORES)            # 4
N_VARS_JOINT = N_STORES * N_VARS  # 84
UNITS_CAP = 10_000                # hard global unit limit

# Per-unit penalty for the MOEA/D penalty approach.
# Must be large enough that infeasible solutions score worse than any feasible
# solution in the Tchebycheff scalarisation, yet not so large that the F[0]
# range collapses (which would break normalisation).  500 satisfies both:
# a 437-unit violation (3 experts/day) gives +218 500 penalty > 104 650
# (worst feasible = zero-staff fixed-cost loss), while keeping F[0] variation
# meaningful so the algorithm can distinguish good from bad solutions.
_PENALTY_UNITS_DEFAULT: float = 500.0

XL_JOINT: np.ndarray = np.tile(XL, N_STORES)  # shape (84,)
XU_JOINT: np.ndarray = np.tile(XU, N_STORES)  # shape (84,)

# hr_x and hr_j indices across the full 84-var vector
INT_IDX_JOINT: np.ndarray = np.concatenate(
    [INT_IDX + s * N_VARS for s in range(N_STORES)]
)


# ===========================================================================
# Repair operator — enforces integrality over all 84 variables
# ===========================================================================
class JointIntegerRepair(Repair):
    """Rounds hr_x/hr_j to integers and clips every variable to [xl, xu]."""

    def _do(self, problem, X: np.ndarray, **kwargs) -> np.ndarray:
        X[:, INT_IDX_JOINT] = np.round(X[:, INT_IDX_JOINT])
        np.clip(X, problem.xl, problem.xu, out=X)
        return X


# ===========================================================================
# Biased initialiser — keeps most initial solutions inside the 10 k limit
# ===========================================================================
# With uniform sampling in [0,15] for staff, ~100 % of random solutions
# violate the 10 000-unit cap (e.g. 15 experts × 7 customers × 17 units ×
# 4 stores × 7 days ≈ 250 000 units).  Capping initial staff at
# hr_x ≤ 3, hr_j ≤ 2 per store-day yields an average of ~7 800 units/week,
# so the majority of initial solutions are feasible.  This is critical for
# both MOEA/D (Tchebycheff normalisation) and U-NSGA-III (constraint-
# dominance) to converge reliably on this tight-constraint problem.
_XU_INIT_JOINT: np.ndarray = np.tile(
    [0.15, 3.0, 2.0], N_VARS_JOINT // 3   # [pr, hr_x, hr_j] × 28 store-days
)


class JointBiasedSampling(Sampling):
    """
    Biased random initialiser for the joint 84-variable O3 problem.

    Samples hr_x ∈ [0, 3] and hr_j ∈ [0, 2] (instead of [0, 15]) so that
    most individuals in generation 0 satisfy the ≤ 10 000-unit constraint.
    The optimisation upper bounds remain [0, 15] — only the *initial*
    sampling range is restricted.
    """

    def _do(self, problem, n_samples: int, **kwargs) -> np.ndarray:
        xu_init = np.minimum(problem.xu, _XU_INIT_JOINT)
        return np.random.uniform(problem.xl, xu_init, (n_samples, problem.n_var))


# ===========================================================================
# Single-pass metric evaluator across all 4 stores
# ===========================================================================
def _evaluate_all_stores(
    x: np.ndarray,
    store_forecasts: list,
    store_weekends: list,
) -> tuple:
    """
    Returns (total_profit, total_staff, total_units)
    from a flat 84-variable individual in a single pass.
    Formula exacta do enunciado: sem PROFIT_SCALE, sem elasticidade de clientes.
    """
    total_profit = 0.0
    total_staff = 0.0
    total_units = 0.0

    for s, store in enumerate(STORES):
        x_s = x[s * N_VARS: (s + 1) * N_VARS]
        fc = store_forecasts[s]
        fw = store_weekends[s]
        params = STORE_PARAMS[store]

        store_sales = 0.0
        store_hr_costs = 0.0

        for d in range(N_DAYS):
            pr = float(np.clip(x_s[d * 3], 0.00, 0.30))
            hr_x = int(round(float(np.clip(x_s[d * 3 + 1], 0, 15))))
            hr_j = int(round(float(np.clip(x_s[d * 3 + 2], 0, 15))))
            # Clientes previstos sem elasticidade artificial — conforme enunciado
            eff_c = int(round(fc[d]))

            m = calculate_daily_metrics(store, fw[d], eff_c, pr, hr_x, hr_j)
            store_sales += m['sales_x'] + m['sales_j']
            store_hr_costs += m['cost_x'] + m['cost_j']
            total_units += m['units_x'] + m['units_j']
            total_staff += hr_x + hr_j

        # Fórmula exata: R_s = Σ R_s,d - W_s (sem multiplicador de escala)
        store_profit = store_sales - store_hr_costs - params['W_s']
        total_profit += store_profit

    return total_profit, total_staff, total_units


# ===========================================================================
# Joint pymoo Problem — always O3 (2 objectives)
# ===========================================================================
class JointTiaposeOptimization(ElementwiseProblem):
    """
    84-variable joint O3 problem: minimize (−TotalProfit, TotalStaff)
    subject to TotalUnits ≤ 10 000.

    Parameters
    ----------
    store_forecasts : list[list[int]]
        4 inner lists, each with 7 customer-count forecasts.
        Order: baltimore, lancaster, philadelphia, richmond.
    store_weekends : list[list[bool]]
        Same structure; True for Saturday/Sunday.
    use_penalty : bool
        True  → 10 k cap absorbed into F[0] as penalty (MOEA/D).
        False → 10 k cap via n_ieq_constr=1 (U-NSGA-III).
    penalty_weight : float
        Multiplier per excess unit when use_penalty=True.
    """

    def __init__(
        self,
        store_forecasts: list,
        store_weekends: list,
        use_penalty: bool = False,
        penalty_weight: float = _PENALTY_UNITS_DEFAULT,
    ):
        assert len(store_forecasts) == N_STORES, f"Need {N_STORES} store forecasts"
        assert len(store_weekends) == N_STORES, f"Need {N_STORES} store weekend lists"

        self.store_forecasts = [list(f) for f in store_forecasts]
        self.store_weekends = [list(w) for w in store_weekends]
        self.use_penalty = use_penalty
        self.penalty_weight = penalty_weight

        # Always 2 objectives (O3); constraint only exposed when not using penalty
        super().__init__(
            n_var=N_VARS_JOINT,
            n_obj=2,
            n_ieq_constr=0 if use_penalty else 1,
            xl=XL_JOINT,
            xu=XU_JOINT,
            elementwise=True,
        )

        log.info(
            "JointProblem (O3) | n_var=%d | n_obj=2 | n_ieq=%d | use_penalty=%s",
            N_VARS_JOINT, 0 if use_penalty else 1, use_penalty,
        )

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> None:
        profit, staff, units = _evaluate_all_stores(
            x, self.store_forecasts, self.store_weekends
        )

        # F[0]: minimise negative profit
        f0 = -profit

        # Global 10 k units constraint (único hard-constraint do enunciado)
        violation = units - UNITS_CAP  # ≤ 0 → feasible

        if self.use_penalty and violation > 0:
            f0 += self.penalty_weight * violation

        out['F'] = [f0, staff]

        if not self.use_penalty:
            out['G'] = [violation]


# ===========================================================================
# Post-processing utilities
# ===========================================================================
def decode_joint_solution(x: np.ndarray) -> dict:
    """
    Splits an 84-var vector into 4 per-store weekly plans.

    Returns { store_name: [7 day-dicts with 'day','pr','hr_x','hr_j','total_staff'] }
    """
    from optimization.nsga2_model import decode_solution
    return {
        store: decode_solution(x[s * N_VARS: (s + 1) * N_VARS])
        for s, store in enumerate(STORES)
    }


def extract_joint_pareto_solutions(res) -> dict:
    """
    Organises the pymoo minimize() result for the joint O3 problem.

    Returns dict with keys:
      pareto_F  — raw objective matrix (n × 2)
      pareto_X  — decision variable matrix (n × 84)
      lucro     — total group profit (positive $), sorted descending
      staff     — total group staff for each Pareto solution
      plans     — list of decode_joint_solution dicts (one per solution)
    """
    if res.F is None or len(res.F) == 0:
        log.warning("Sem soluções viáveis encontradas no problema conjunto.")
        return {
            'pareto_F': np.array([]),
            'pareto_X': np.array([]),
            'lucro': np.array([]),
            'staff': np.array([]),
            'plans': [],
        }

    pareto_F = res.F   # (n, 2): [−profit, staff]
    pareto_X = res.X   # (n, 84)

    lucro_col = -pareto_F[:, 0]   # revert negation → positive profit
    staff_col = pareto_F[:, 1]

    order = np.argsort(-lucro_col)  # descending profit

    log.info(
        "Pareto conjunto (O3): %d soluções | Lucro [%.0f, %.0f] | Staff [%.0f, %.0f]",
        len(pareto_F),
        lucro_col.min(), lucro_col.max(),
        staff_col.min(), staff_col.max(),
    )

    return {
        'pareto_F': pareto_F[order],
        'pareto_X': pareto_X[order],
        'lucro': lucro_col[order],
        'staff': staff_col[order],
        'plans': [decode_joint_solution(pareto_X[i]) for i in order],
    }


def get_o2_solution(pareto_result: dict) -> dict:
    """
    Extracts the O2 solution from an O3 Pareto result.

    O2 = maximise total profit with ≤10 000 units constraint.
    Since every solution in the O3 Pareto front already satisfies the
    10 k constraint, the solution with maximum profit (index 0, because
    extract_joint_pareto_solutions sorts descending) IS the O2 optimum.

    Returns
    -------
    dict with keys:
      lucro  — best group profit ($)
      staff  — total HR staff for that solution
      plan   — decode_joint_solution dict (per-store weekly plans)
    """
    if len(pareto_result.get('lucro', [])) == 0:
        log.warning("Pareto vazio — não é possível extrair solução O2.")
        return {'lucro': None, 'staff': None, 'plan': None}

    return {
        'lucro': float(pareto_result['lucro'][0]),
        'staff': float(pareto_result['staff'][0]),
        'plan':  pareto_result['plans'][0],
    }
