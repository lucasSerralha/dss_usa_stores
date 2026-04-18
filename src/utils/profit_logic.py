import numpy as np

# Configuração dos parâmetros por loja (F_j, F_x, W_s) 
STORE_PARAMS = {
    'baltimore':    {'F_j': 1.00, 'F_x': 1.15, 'W_s': 700},
    'lancaster':    {'F_j': 1.05, 'F_x': 1.20, 'W_s': 730},
    'philadelphia': {'F_j': 1.10, 'F_x': 1.15, 'W_s': 760},
    'richmond':     {'F_j': 1.15, 'F_x': 1.25, 'W_s': 800}
}

def calculate_daily_metrics(store, is_weekend, customers, pr, hr_x, hr_j):
    """
    Calcula os clientes assistidos, unidades, vendas e custos diários.
    """
    params = STORE_PARAMS.get(store.lower())
    if not params:
        raise ValueError(f"Loja '{store}' não encontrada.")
        
    F_x, F_j = params['F_x'], params['F_j']

    # 1. Custos de RH [cite: 223, 225]
    cost_x_multiplier = 95 if is_weekend else 80
    cost_j_multiplier = 70 if is_weekend else 60
    
    cost_x = hr_x * cost_x_multiplier
    cost_j = hr_j * cost_j_multiplier

    # 2. Clientes Assistidos [cite: 228-233]
    # Os especialistas (X) atendem primeiro (máx 7 clientes cada)
    assisted_x = min(hr_x * 7, customers)
    
    # Os juniores (J) atendem o restante (máx 6 clientes cada)
    remaining_customers = customers - assisted_x
    assisted_j = min(hr_j * 6, remaining_customers)

    # 3. Unidades Vendidas por Cliente Assistido [cite: 257]
    # Apenas calcula se houver clientes assistidos
    u_per_x = round((F_x * 10) / np.log(2 - pr)) if assisted_x > 0 else 0
    u_per_j = round((F_j * 10) / np.log(2 - pr)) if assisted_j > 0 else 0

    # Total de unidades vendidas no dia
    units_x = assisted_x * u_per_x
    units_j = assisted_j * u_per_j

    # 4. Lucro/Vendas Diárias 
    # Arredondamento aplicado sobre o total diário
    sales_x = round(units_x * (1 - pr) * 1.07)
    sales_j = round(units_j * (1 - pr) * 1.07)

    return {
        'assisted_x': assisted_x, 'assisted_j': assisted_j,
        'units_x': units_x, 'units_j': units_j,
        'sales_x': sales_x, 'sales_j': sales_j,
        'cost_x': cost_x, 'cost_j': cost_j
    }

def calculate_weekly_profit(store, weekly_plan):
    """
    Recebe o nome da loja e uma lista de 7 dicionários (um por dia).
    Retorna o lucro semanal final.
    """
    total_sales = 0
    total_hr_costs = 0
    
    for day_data in weekly_plan:
        metrics = calculate_daily_metrics(
            store=store,
            is_weekend=day_data['is_weekend'],
            customers=day_data['customers'],
            pr=day_data['pr'],
            hr_x=day_data['hr_x'],
            hr_j=day_data['hr_j']
        )
        
        # Somatório das vendas e custos [cite: 260]
        total_sales += (metrics['sales_x'] + metrics['sales_j'])
        total_hr_costs += (metrics['cost_x'] + metrics['cost_j'])
        
    # Subtrai o custo fixo semanal da loja [cite: 262]
    fixed_cost = STORE_PARAMS[store.lower()]['W_s']
    final_profit = total_sales - total_hr_costs - fixed_cost
    
    return final_profit

def optimize_weekly_wrapper(decision_vars, store, forecast_customers, forecast_is_weekend):
    """
    Função Tradutora (Wrapper) para algoritmos de Otimização (Scipy, Pymoo, DEAP).
    
    decision_vars: Lista plana de 21 valores gerada pelo algoritmo.
                   Estrutura: [pr1, hr_x1, hr_j1, pr2, hr_x2, hr_j2, ... até ao dia 7]
    forecast_customers: Lista com os 7 valores previstos pelos modelos de Forecasting.
    forecast_is_weekend: Lista de 7 booleanos [True, False, ...] indicando o calendário.
    
    RETORNA: (Lucro_Invertido, Total_Staff, Penalizacao_Semana)
    """
    weekly_plan = []
    total_staff = 0
    penalizacao_semana = 0
    
    for i in range(7):
        # 1. Extrair as 3 variáveis do dia 'i' do vetor plano
        pr_raw = decision_vars[i * 3]
        hr_x_raw = decision_vars[(i * 3) + 1]
        hr_j_raw = decision_vars[(i * 3) + 2]
        
        # 2. Impor restrições de negócio (Clipping e Arredondamento)
        pr_clean = max(0.0, min(0.30, pr_raw))
        hr_x_clean = max(0, int(round(hr_x_raw)))
        hr_j_clean = max(0, int(round(hr_j_raw)))
        
        # Cálculo do Staff Diário para os novos Objetivos e Penalizações
        staff_dia = hr_x_clean + hr_j_clean
        total_staff += staff_dia
        
        # Lógica de Penalização (Cenário 3): Dias úteis com mais de 8 funcionários
        if not forecast_is_weekend[i]:
            if staff_dia > 8:
                extra_staff = staff_dia - 8
                penalizacao_semana += (extra_staff * 1000)
        
        # 3. Montar o dicionário do dia para o cálculo do lucro
        day_data = {
            'is_weekend': forecast_is_weekend[i],
            'customers': forecast_customers[i],
            'pr': pr_clean,
            'hr_x': hr_x_clean,
            'hr_j': hr_j_clean
        }
        weekly_plan.append(day_data)
        
    # 4. Calcular o lucro real usando a tua lógica original
    profit = calculate_weekly_profit(store, weekly_plan)
    
    # Retornamos a tripla de objetivos para a Otimização Multi-Objetivo
    # f1: Lucro Invertido (Minimizar -Profit == Maximizar Profit)
    # f2: Total Staff (Minimizar)
    # f3: Penalização por excesso de Staff em dias úteis (Minimizar)
    return (-profit, total_staff, penalizacao_semana)

# ==========================================
# TESTE DO WRAPPER DE OTIMIZAÇÃO (W5 - CENÁRIO 3)
# ==========================================
if __name__ == "__main__":
    import random

    print("--- A testar a Ponte de Otimização (3 Objetivos - Cenário 3) ---")

    # 1. Simular uma previsão (7 dias)
    dummy_forecast_customers = [80, 65, 70, 75, 60, 90, 110]
    dummy_forecast_weekend = [False, False, False, False, False, True, True] 
    
    # 2. Simular um vetor plano (21 valores) - Forçar excesso de staff para testar penalização
    # Vamos por 10 funcionários num dia de semana (índice 0)
    decision_vars_test = [0.10, 5, 5] + [0.10, 1, 1]*6

    # 3. Executar o tradutor
    f1, f2, f3 = optimize_weekly_wrapper(
        decision_vars=decision_vars_test,
        store='baltimore',
        forecast_customers=dummy_forecast_customers,
        forecast_is_weekend=dummy_forecast_weekend
    )

    print("\n--- Resultados Detalhados ---")
    print(f"Objetivo 1 (Lucro Invertido):  {f1}")
    print(f"Objetivo 2 (Total Staff):      {f2}")
    print(f"Objetivo 3 (Penalização):     {f3}")
    print(f"Lucro Real Estimado:          ${-f1}")
    
    print("\n✅ Teste de sanidade concluído. A tripla de objetivos está formatada corretamente.")