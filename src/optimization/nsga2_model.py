"""
nsga2_model.py — NSGA-II Multi-Objective Optimization — TIAPOSE DSS

Problema: Otimizar a escala semanal de uma loja de retalho.
  21 variáveis de decisão (3 por dia × 7 dias):
    x[3d+0] = pr_d   ∈ [0.00, 0.30]   desconto diário (float)
    x[3d+1] = hr_x_d ∈ {0, …, 15}     staff perito (int)
    x[3d+2] = hr_j_d ∈ {0, …, 15}     staff junior (int)

  2 Objetivos conflituosos (minimização no pymoo):
    F[0] = -Lucro_semanal          maximizar lucro == minimizar negativo
    F[1] =  Staff_Total_semanal    minimizar headcount

  Restrições de negócio (G_d ≤ 0 → viável):
    G[d] = (hr_x_d + hr_j_d) − 8   para cada dia útil d   (cap weekday = 8)
    G[d] = (hr_x_d + hr_j_d) − 12  para cada fim-de-semana d (cap weekend = 12)

Nota de design:
  Em vez de codificar o excesso de staff como penalização num 3.º objetivo
  (abordagem de penalty function), usamos n_ieq_constr do pymoo. Desta forma
  o NSGA-II separa soluções viáveis de inviáveis via constraint-dominance,
  o que é matematicamente mais correto e converge mais rapidamente.
"""

import logging
import os
import sys
from typing import Callable, Optional

import numpy as np
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.core.problem import ElementwiseProblem
from pymoo.core.repair import Repair
from pymoo.operators.crossover.sbx import SBX
from pymoo.operators.mutation.pm import PM
from pymoo.operators.sampling.rnd import FloatRandomSampling
from pymoo.optimize import minimize
from pymoo.termination.default import DefaultMultiObjectiveTermination

# ---------------------------------------------------------------------------
# Path bootstrapping — permite executar este ficheiro diretamente
# ---------------------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_SRC_DIR = os.path.abspath(os.path.join(_SCRIPT_DIR, ".."))
if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from utils.profit_logic import optimize_weekly_wrapper  # função de avaliação real

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger("TIAPOSE.NSGA2")

# ---------------------------------------------------------------------------
# Constantes do espaço de decisão
# ---------------------------------------------------------------------------
N_DAYS: int = 7
N_VARS_PER_DAY: int = 3
N_VARS: int = N_DAYS * N_VARS_PER_DAY  # 21

# Índices das variáveis inteiras (hr_x, hr_j) no vetor plano de 21 posições.
# Reshape para (7, 3), selecionar colunas 1 e 2, voltar a aplanar.
# Resultado: [1, 2, 4, 5, 7, 8, 10, 11, 13, 14, 16, 17, 19, 20]
INT_IDX: np.ndarray = (
    np.arange(N_VARS).reshape(N_DAYS, N_VARS_PER_DAY)[:, 1:].ravel()
)

# Limites inferiores e superiores (repetir o padrão [pr, hr_x, hr_j] por 7 dias)
XL: np.ndarray = np.tile([0.00,  0,  0], N_DAYS).astype(float)  # shape (21,)
XU: np.ndarray = np.tile([0.30, 15, 15], N_DAYS).astype(float)  # shape (21,)


# ===========================================================================
# 1. REPAIR OPERATOR
#    Aplicado pelo pymoo após cada crossover e mutação para garantir que
#    hr_x e hr_j permanecem inteiros dentro dos bounds.
# ===========================================================================

class IntegerRepair(Repair):
    """
    Arredonda as variáveis de staff (hr_x, hr_j) para o inteiro mais próximo
    e faz clip de toda a população para [xl, xu].

    O pymoo chama _do com X de shape (pop_size, 21), pelo que toda a operação
    é vectorizada — sem loops Python sobre indivíduos ou variáveis.
    """

    def _do(self, problem, X: np.ndarray, **kwargs) -> np.ndarray:
        # Arredondar apenas as colunas de índice inteiro (hr_x, hr_j)
        X[:, INT_IDX] = np.round(X[:, INT_IDX])
        # Garantir que nenhum valor extravasa os bounds após mutação
        np.clip(X, problem.xl, problem.xu, out=X)
        return X


# ===========================================================================
# 2. FUNÇÃO DUMMY DE LUCRO
#    Placeholder para testes sem necessitar da lógica de negócio real.
#    Substituir profit_fn=dummy_profit_function por profit_fn=optimize_weekly_wrapper
#    quando a integração com profit_logic.py estiver pronta.
# ===========================================================================

def dummy_profit_function(
    decision_vars: np.ndarray,
    store: str,
    forecast_customers: list,
    forecast_is_weekend: list,
) -> tuple:
    """
    Simulação rápida de avaliação — apenas para validar o pipeline NSGA-II.

    Assinatura idêntica ao optimize_weekly_wrapper real:
      → (f1_lucro_invertido: float, f2_staff_total: float, f3_penalizacao: float)

    Lucro simulado: cresce com desconto e decresce com excesso de staff.
    """
    # Extrair staff de todos os dias — operação vectorizada sem loop
    staff_matrix = np.round(decision_vars[INT_IDX]).reshape(N_DAYS, 2)
    daily_staff = staff_matrix.sum(axis=1)         # shape (7,)
    total_staff = float(daily_staff.sum())

    # Desconto médio da semana
    discount_idx = np.arange(0, N_VARS, N_VARS_PER_DAY)  # [0,3,6,9,12,15,18]
    avg_discount = float(decision_vars[discount_idx].mean())

    # Receita fictícia: clientes × desconto × fator de conversão
    revenue = sum(
        forecast_customers[d] * (1 + avg_discount) * 10
        for d in range(N_DAYS)
    )
    hr_cost = total_staff * 70  # custo médio simulado por funcionário
    fake_profit = revenue - hr_cost - 700  # 700 = fixed cost

    # Penalização (informativa, ignorada na avaliação — tratada via G)
    weekday_mask = np.array([not w for w in forecast_is_weekend])
    excess = np.maximum(0, daily_staff[weekday_mask] - 8)
    penalty = float((excess * 1_000).sum())

    return (-fake_profit, total_staff, penalty)


# ===========================================================================
# 3. PROBLEMA PYMOO — TiaposeOptimization
# ===========================================================================

class TiaposeOptimization(ElementwiseProblem):
    """
    Define o problema de otimização de escala semanal para o pymoo.

    Parâmetros do construtor
    ------------------------
    store : str
        Nome da loja — deve existir em STORE_PARAMS (profit_logic.py).
    forecast_customers : list[int]
        7 valores de clientes previstos, um por dia (Seg → Dom).
    forecast_is_weekend : list[bool]
        7 booleanos — True se o dia for Sáb ou Dom.
    profit_fn : callable, opcional
        Função de avaliação com assinatura:
            (decision_vars, store, forecast_customers, forecast_is_weekend)
            → (f1: float, f2: float, f3: float)
        Por defeito usa optimize_weekly_wrapper (função real).
        Passar dummy_profit_function para testes isolados.
    """

    def __init__(
        self,
        store: str,
        forecast_customers: list,
        forecast_is_weekend: list,
        profit_fn: Optional[Callable] = None,
    ):
        assert len(forecast_customers) == N_DAYS, (
            f"forecast_customers deve ter {N_DAYS} elementos, recebeu {len(forecast_customers)}"
        )
        assert len(forecast_is_weekend) == N_DAYS, (
            f"forecast_is_weekend deve ter {N_DAYS} elementos, recebeu {len(forecast_is_weekend)}"
        )

        self.store = store
        self.forecast_customers = list(forecast_customers)
        self.forecast_is_weekend = list(forecast_is_weekend)

        # Injeção de dependência: real por defeito, dummy se explicitamente pedido
        self.profit_fn = profit_fn if profit_fn is not None else optimize_weekly_wrapper

        # Índices dos dias úteis e fins-de-semana para as restrições G
        self._weekday_idx: np.ndarray = np.array(
            [d for d, is_wk in enumerate(forecast_is_weekend) if not is_wk]
        )
        self._weekend_idx: np.ndarray = np.array(
            [d for d, is_wk in enumerate(forecast_is_weekend) if is_wk]
        )
        # Total de restrições = dias úteis (cap 8) + fins de semana (cap 12)
        n_constraints = len(self._weekday_idx) + len(self._weekend_idx)

        super().__init__(
            n_var=N_VARS,
            n_obj=2,                      # F[0]=-Lucro, F[1]=Staff Total
            n_ieq_constr=n_constraints,   # G weekday ≤ 8, G weekend ≤ 12
            xl=XL,
            xu=XU,
            elementwise=True,             # _evaluate recebe um vetor 1-D por chamada
        )

        log.info(
            "Problema criado | loja=%-12s | dias_úteis=%d | fins_semana=%d | restrições_G=%d",
            store, len(self._weekday_idx), len(self._weekend_idx), n_constraints,
        )

    def _evaluate(self, x: np.ndarray, out: dict, *args, **kwargs) -> None:
        """
        Avalia um único indivíduo x ∈ ℝ^21 e preenche out['F'] e out['G'].

        Chamado pelo pymoo para cada membro da população a cada geração.
        O paralelismo entre indivíduos é gerido externamente pelo runner
        (ver parâmetro runner= em minimize()).

        Estrutura de x (21 posições):
          [pr_0, hr_x_0, hr_j_0, pr_1, hr_x_1, hr_j_1, ..., pr_6, hr_x_6, hr_j_6]
        """
        # --- Objetivos F ---
        # profit_fn devolve (f1=-lucro, f2=staff, f3=penalização_ignorada)
        f1, f2, _ = self.profit_fn(
            decision_vars=x,
            store=self.store,
            forecast_customers=self.forecast_customers,
            forecast_is_weekend=self.forecast_is_weekend,
        )
        out["F"] = [f1, f2]

        # --- Restrições de desigualdade G ---
        # Extrair staff diário vectorizado (round() como salvaguarda pós-mutação).
        staff_per_day = np.round(x[INT_IDX]).reshape(N_DAYS, 2).sum(axis=1)  # (7,)
        # G[d] ≤ 0 → viável; G[d] > 0 → violação (pymoo usa constraint-dominance)
        g_weekday = (staff_per_day[self._weekday_idx] - 8).tolist()   # cap 8
        g_weekend  = (staff_per_day[self._weekend_idx] - 12).tolist() # cap 12
        out["G"] = g_weekday + g_weekend


# ===========================================================================
# 4. UTILITÁRIOS DE PÓS-PROCESSAMENTO
# ===========================================================================

def decode_solution(x: np.ndarray) -> list:
    """
    Converte o vetor plano de 21 valores num plano semanal legível.

    Retorna lista de 7 dicionários:
      { 'day': int, 'pr': float, 'hr_x': int, 'hr_j': int, 'total_staff': int }
    """
    # Reshape (7, 3) e extração vectorizada
    mat = x.reshape(N_DAYS, N_VARS_PER_DAY)
    return [
        {
            "day":         d,
            "pr":          round(float(mat[d, 0]), 4),
            "hr_x":        int(round(mat[d, 1])),
            "hr_j":        int(round(mat[d, 2])),
            "total_staff": int(round(mat[d, 1])) + int(round(mat[d, 2])),
        }
        for d in range(N_DAYS)
    ]


def extract_pareto_solutions(res) -> dict:
    """
    Organiza o resultado do minimize() num dicionário de trabalho.

    Chaves retornadas:
      'pareto_F'  — array (n, 2): [-Lucro, Staff] na ordem original do pymoo
      'pareto_X'  — array (n, 21): variáveis de decisão
      'lucro'     — array (n,): lucro positivo, ordenado decrescentemente
      'staff'     — array (n,): staff total correspondente
      'plans'     — lista de n planos semanais decodificados
    """
    if res.F is None or len(res.F) == 0:
        log.warning("Nenhuma solução viável encontrada. Verifica os bounds e restrições.")
        return {"pareto_F": np.array([]), "pareto_X": np.array([]),
                "lucro": np.array([]), "staff": np.array([]), "plans": []}

    pareto_F = res.F   # (n, 2)
    pareto_X = res.X   # (n, 21)

    lucro_col = -pareto_F[:, 0]   # reverter negação → valores positivos
    staff_col =  pareto_F[:, 1]

    # Ordenar por lucro decrescente para leitura intuitiva
    order = np.argsort(-lucro_col)

    log.info(
        "Pareto: %d soluções | Lucro [%.0f, %.0f] | Staff [%.0f, %.0f]",
        len(pareto_F),
        lucro_col.min(), lucro_col.max(),
        staff_col.min(), staff_col.max(),
    )

    return {
        "pareto_F": pareto_F[order],
        "pareto_X": pareto_X[order],
        "lucro":    lucro_col[order],
        "staff":    staff_col[order],
        "plans":    [decode_solution(pareto_X[i]) for i in order],
    }


# ===========================================================================
# 5. ORQUESTRADOR — ponto de entrada para outros módulos
# ===========================================================================

def run_optimization(
    store: str,
    forecast_customers: list,
    forecast_is_weekend: list,
    pop_size: int = 100,
    n_max_gen: int = 300,
    seed: int = 42,
    verbose: bool = True,
    profit_fn: Optional[Callable] = None,
) -> dict:
    """
    Configura e executa o NSGA-II, retornando as soluções da Fronteira de Pareto.

    Args:
        store: Nome da loja (ver STORE_PARAMS em profit_logic.py).
        forecast_customers: 7 inteiros com previsão de clientes (Seg → Dom).
        forecast_is_weekend: 7 booleanos de calendário.
        pop_size: Tamanho da população. Mínimo recomendado para 21 variáveis: 100.
        n_max_gen: Número máximo de gerações; a terminação adaptativa pode parar antes.
        seed: Semente para reprodutibilidade.
        verbose: Progresso a cada geração se True.
        profit_fn: Função de lucro plugável.
                   None  → usa optimize_weekly_wrapper (produção).
                   dummy_profit_function → para testes sem profit_logic.

    Returns:
        Dicionário de extract_pareto_solutions.
    """
    log.info(
        "NSGA-II iniciado | loja=%-12s | pop=%d | max_gen=%d | seed=%d",
        store, pop_size, n_max_gen, seed,
    )

    # --- Problema ---
    problem = TiaposeOptimization(
        store=store,
        forecast_customers=forecast_customers,
        forecast_is_weekend=forecast_is_weekend,
        profit_fn=profit_fn,
    )

    # --- Algoritmo ---
    algorithm = NSGA2(
        pop_size=pop_size,
        # SBX: η=15 equilibra exploração/exploitação; prob_var=1/n_var
        # aplica crossover em média a cada variável uma vez por geração.
        crossover=SBX(eta=15, prob=0.9, prob_var=1.0 / N_VARS),
        # PM: η=20 gera perturbações suaves, adequado para espaço contínuo/misto.
        mutation=PM(eta=20, prob=1.0 / N_VARS),
        # Amostragem uniforme dentro de [xl, xu] na geração inicial.
        sampling=FloatRandomSampling(),
        # Repair garante inteiros após crossover/mutação — essencial para hr_x, hr_j.
        repair=IntegerRepair(),
        eliminate_duplicates=True,
    )

    # --- Critério de paragem adaptativo ---
    # Para quando a métrica IGD (Inverted Generational Distance) estabiliza
    # por 30 gerações consecutivas, ou quando atinge n_max_gen.
    termination = DefaultMultiObjectiveTermination(
        xtol=1e-6,    # tolerância no espaço de decisão
        cvtol=1e-6,   # tolerância na violação de restrições
        ftol=0.0025,  # tolerância no espaço de objetivos
        period=30,    # janela de estabilidade (gerações)
        n_max_gen=n_max_gen,
    )

    # --- Execução ---
    res = minimize(
        problem,
        algorithm,
        termination,
        seed=seed,
        verbose=verbose,
        save_history=False,  # True só para análise de convergência (caro em memória)
    )

    log.info("Concluído em %d gerações.", res.algorithm.n_gen)
    return extract_pareto_solutions(res)


# ===========================================================================
# 6. DEMO — execução direta do ficheiro
# ===========================================================================

if __name__ == "__main__":
    import pprint

    STORE = "baltimore"
    CLIENTES_PREVISTOS = [80, 65, 70, 75, 60, 90, 110]   # Seg → Dom
    CALENDARIO_FDS     = [False, False, False, False, False, True, True]

    print("=" * 62)
    print(f"  TIAPOSE — NSGA-II | Loja: {STORE.upper()}")
    print("=" * 62)

    # Para usar a função dummy em vez da real, passar:
    #   profit_fn=dummy_profit_function
    resultados = run_optimization(
        store=STORE,
        forecast_customers=CLIENTES_PREVISTOS,
        forecast_is_weekend=CALENDARIO_FDS,
        pop_size=100,
        n_max_gen=200,
        seed=42,
        verbose=True,
    )

    # --- Tabela resumo da Fronteira de Pareto ---
    n = len(resultados["lucro"])
    print(f"\n  Fronteira de Pareto: {n} soluções não-dominadas\n")
    print(f"  {'Sol':>4} | {'Lucro (€)':>12} | {'Staff Total':>11}")
    print("  " + "-" * 34)
    for i in range(min(n, 10)):
        print(
            f"  {i+1:>4} | {resultados['lucro'][i]:>12.2f}"
            f" | {resultados['staff'][i]:>11.0f}"
        )

    # --- Detalhe da melhor solução (maior lucro) ---
    if resultados["plans"]:
        print("\n  Plano semanal — melhor lucro:")
        pprint.pprint(resultados["plans"][0], indent=4)
