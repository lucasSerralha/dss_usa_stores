import numpy as np
import random
import sys
import os

# Importar ficheiros da pasta 'src'
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
SRC_DIR = os.path.abspath(os.path.join(SCRIPT_DIR, ".."))
sys.path.append(SRC_DIR)

# Importar função responsável por calcular os objetivos da otimização
from utils.profit_logic import optimize_weekly_wrapper


# Função para gerar uma solução inicial aleatória para os 7 dias da semana
def generate_initial_solution():

    solution = []

    for _ in range(7):

        # Percentagem de desconto entre
        pr = random.uniform(0.0, 0.30)

        # Nº de funcionários experientes
        hr_x = random.randint(0, 15)

        # Nº de funcionários juniores
        hr_j = random.randint(0, 15)

        # Guardar os valores
        solution.extend([pr, hr_x, hr_j])

    return np.array(solution)


# Função para gerar uma solução vizinha
# Uma solução vizinha é criada alterando ligeiramente um dos valores
def generate_neighbor(solution):

    neighbor = solution.copy()

    # Escolher aleatoriamente qual variável será alterada
    index = random.randint(0, len(solution) - 1)

    # Pequena alteração aleatória
    change = random.uniform(-0.05, 0.05)

    # Garantir que o valor não fica negativo
    neighbor[index] = max(0, neighbor[index] + change)

    return neighbor


# Função que avalia a qualidade de uma solução
def evaluate_solution(solution, store, forecast_customers, forecast_is_weekend):

    f1, f2, f3 = optimize_weekly_wrapper(
        decision_vars=solution,
        store=store,
        forecast_customers=forecast_customers,
        forecast_is_weekend=forecast_is_weekend
    )

    # Neste algoritmo utilizamos apenas o primeiro objetivo
    return f1


# Implementação do algoritmo Hill Climbing
def hill_climbing(store, forecast_customers, forecast_is_weekend, iterations=1000):

    # Gerar solução inicial
    current_solution = generate_initial_solution()

    # Avaliar solução inicial
    current_score = evaluate_solution(
        current_solution,
        store,
        forecast_customers,
        forecast_is_weekend
    )

    best_solution = current_solution
    best_score = current_score

    # Processo iterativo de melhoria
    for i in range(iterations):

        neighbor = generate_neighbor(best_solution)

        score = evaluate_solution(
            neighbor,
            store,
            forecast_customers,
            forecast_is_weekend
        )

        # Se a nova solução for melhor, substitui a atual
        if score < best_score:

            best_solution = neighbor
            best_score = score

    return best_solution, best_score


# Execução do algoritmo
if __name__ == "__main__":

    # Previsão de clientes por dia da semana
    previsao_clientes = [80, 65, 70, 75, 60, 90, 110]

    # Indicação de fim de semana
    previsao_fds = [False, False, False, False, False, True, True]

    # Executar Hill Climbing para a loja Baltimore
    solution, score = hill_climbing(
        'baltimore',
        previsao_clientes,
        previsao_fds
    )

    # Lista com os dias da semana
    days = ["Segunda", "Terça", "Quarta", "Quinta", "Sexta", "Sábado", "Domingo"]

    print("\nMelhor estratégia semanal encontrada:\n")

    # Mostrar estratégia por dia
    for i in range(7):

        pr = solution[i*3]
        hr_x = int(solution[i*3 + 1])
        hr_j = int(solution[i*3 + 2])

        print(f"{days[i]}:")
        print(f"  Desconto: {pr*100:.2f}%")
        print(f"  Funcionários Experientes: {hr_x}")
        print(f"  Funcionários Juniores: {hr_j}")
        print()

    # Mostrar score final
    print("Score da solução:", score)