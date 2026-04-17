import numpy as np
from pymoo.core.problem import ElementwiseProblem
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize
import sys
import os

# 1. Truque de Arquitetura: Ensinar ao Python onde está a pasta 'src'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__)) # Estamos em src/optimization
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, "..")) # Recuamos para src/
sys.path.append(SRC_DIR) # Injetamos no sistema

# 2. Agora já podemos importar da pasta utils normalmente!
from utils.profit_logic import optimize_weekly_wrapper

class TiaposeOptimization(ElementwiseProblem):
    def __init__(self, store, forecast_customers, forecast_is_weekend):
        self.store = store
        self.forecast_customers = forecast_customers
        self.forecast_is_weekend = forecast_is_weekend
        
        # Temos 21 variáveis de decisão (7 dias * 3 variáveis: pr, hr_x, hr_j)
        # Vamos definir os limites Mínimos (xl) e Máximos (xu) permitidos
        
        # Limite Inferior: Desconto 0%, 0 Peritos, 0 Juniores
        xl = np.array([0.0, 0, 0] * 7)
        
        # Limite Superior: Desconto 30%, Max 15 Peritos, Max 15 Juniores (ajustável)
        xu = np.array([0.30, 15, 15] * 7)
        
        # Inicializamos o problema com 3 Objetivos (n_obj=3 - Cenário 3)
        super().__init__(n_var=21, n_obj=3, n_ieq_constr=0, xl=xl, xu=xu)

    def _evaluate(self, x, out, *args, **kwargs):
        # Chamada ao wrapper que devolve os 3 objetivos calculados (Cenário 3)
        f1, f2, f3 = optimize_weekly_wrapper(
            decision_vars=x,
            store=self.store,
            forecast_customers=self.forecast_customers,
            forecast_is_weekend=self.forecast_is_weekend
        )
        
        # Alocação dos 3 objectivos no Pymoo
        out["F"] = [f1, f2, f3]

if __name__ == "__main__":
    print("--- A iniciar a Otimização Multi-Objetivo (NSGA-II) ---")
    
    # 1. Dados simulados da previsão
    previsao_clientes = [80, 65, 70, 75, 60, 90, 110]
    previsao_fds = [False, False, False, False, False, True, True]
    
    # 2. Instanciar o Problema
    problem = TiaposeOptimization('baltimore', previsao_clientes, previsao_fds)
    
    # 3. Configurar o Algoritmo (Tamanho da População de soluções)
    algorithm = NSGA2(pop_size=50)
    
    # 4. Executar a Evolução! (Vamos fazer apenas 20 gerações para testar)
    print("O algoritmo está a evoluir as soluções. Por favor aguarde...")
    res = minimize(problem,
                   algorithm,
                   ('n_gen', 20),
                   seed=1,
                   verbose=True)
    
    print("\n--- Evolução Concluída! ---")
    print(f"Foram encontradas {len(res.F)} soluções ótimas na Fronteira de Pareto.")