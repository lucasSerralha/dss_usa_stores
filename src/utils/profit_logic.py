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


# ==========================================
# TESTE DE VALIDAÇÃO Baltimore
# ==========================================
if __name__ == "__main__":
    # Dados de validação  [cite: 275-294]
    baltimore_test_week = [
        {'is_weekend': True,  'customers': 97,  'pr': 0.00, 'hr_x': 4,  'hr_j': 0},  # Sun
        {'is_weekend': False, 'customers': 61,  'pr': 0.05, 'hr_x': 0,  'hr_j': 10}, # Mon
        {'is_weekend': False, 'customers': 65,  'pr': 0.10, 'hr_x': 8,  'hr_j': 4},  # Tue
        {'is_weekend': False, 'customers': 71,  'pr': 0.15, 'hr_x': 20, 'hr_j': 0},  # Wed
        {'is_weekend': False, 'customers': 65,  'pr': 0.20, 'hr_x': 0,  'hr_j': 5},  # Thu
        {'is_weekend': False, 'customers': 89,  'pr': 0.25, 'hr_x': 4,  'hr_j': 5},  # Fri
        {'is_weekend': True,  'customers': 125, 'pr': 0.30, 'hr_x': 3,  'hr_j': 4}   # Sat
    ]

    profit = calculate_weekly_profit('baltimore', baltimore_test_week)
    print(f"Lucro Semanal Calculado: ${profit}")
    
    if profit == 146:
        print("Sucesso")
    else:
        print("Falha")
