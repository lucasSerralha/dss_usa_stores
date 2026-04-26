import numpy as np
import random
import sys
import os
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(SRC_DIR)

from utils.profit_logic import optimize_weekly_wrapper

def generate_initial_solution():
    solution = []
    for _ in range(7):
        solution.extend([random.uniform(0.0, 0.30), random.randint(0, 15), random.randint(0, 15)])
    return np.array(solution)

def generate_neighbor(solution):
    neighbor = solution.copy()
    index = random.randint(0, len(solution) - 1)
    
    if index % 3 == 0:
        neighbor[index] = max(0.0, min(0.30, neighbor[index] + random.uniform(-0.05, 0.05)))
    else:
        neighbor[index] = max(0, neighbor[index] + random.choice([-1, 1]))
    return neighbor

def evaluate_solution(solution, store, forecast_customers, forecast_is_weekend):
    f1, f2, f3 = optimize_weekly_wrapper(solution, store, forecast_customers, forecast_is_weekend)
    # CENÁRIO 1: Queremos apenas maximizar o Lucro (f1) 
    return f1

def hill_climbing(store, forecast_customers, forecast_is_weekend, iterations=1000):
    current_solution = generate_initial_solution()
    best_score = evaluate_solution(current_solution, store, forecast_customers, forecast_is_weekend)
    best_solution = current_solution
    
    history = [best_score]

    for _ in range(iterations):
        neighbor = generate_neighbor(best_solution)
        score = evaluate_solution(neighbor, store, forecast_customers, forecast_is_weekend)
        if score < best_score:
            best_solution = neighbor
            best_score = score
        history.append(best_score)
        
    return best_solution, best_score, history

if __name__ == "__main__":
    previsao_clientes = [80, 65, 70, 75, 60, 90, 110]
    previsao_fds = [False, False, False, False, False, True, True]
    
    print("A executar Hill Climbing (Cenário 1: Maximização de Lucro)...")
    solution, score, history = hill_climbing('baltimore', previsao_clientes, previsao_fds, iterations=1000)

    # Inverter o f1 para voltar a ser o Lucro real positivo
    history_lucro = [-x for x in history]

    print(f"SUCESSO! Lucro Máximo Alcançado: ${history_lucro[-1]:.2f}")
    
    # -----------------------------------------------------
    # SALVAR A IMAGEM EM VEZ DE TENTAR ABRIR
    # -----------------------------------------------------
    out_dir = os.path.join(SRC_DIR, "..", "results")
    os.makedirs(out_dir, exist_ok=True)
    img_path = os.path.join(out_dir, "hill_climbing_convergence.png")
    
    plt.figure(figsize=(10, 5))
    plt.plot(history_lucro, color='green')
    plt.title("Convergência do Hill Climbing (Cenário 1: Maximização de Lucro)")
    plt.xlabel("Iterações")
    plt.ylabel("Lucro ($)")
    plt.grid(True)
    plt.savefig(img_path)
    plt.close()
    
    print(f"Gráfico guardado com sucesso em: {img_path}")