import numpy as np
import logging

log = logging.getLogger("TIAPOSE.Knapsack")

def solve_mckp(groups, capacity):
    """
    Resolve o Multiple Choice Knapsack Problem (MCKP) via Programação Dinâmica.
    
    Args:
        groups: Lista de grupos, onde cada grupo é uma lista de dicionários:
                [{'profit': float, 'weight': int, 'id': any}, ...]
        capacity: Capacidade máxima da mochila (inteiro).
        
    Returns:
        best_profit: Lucro total máximo.
        best_items: Lista com um item de cada grupo que compõe o melhor lucro.
        total_weight: Peso total dos itens escolhidos.
    """
    n_groups = len(groups)
    # dp[group_idx][weight] = max_profit
    # Inicializar com -1 (inviável)
    dp = np.full((n_groups + 1, capacity + 1), -1.0)
    # parent[group_idx][weight] = index_of_item_in_group
    parent = np.full((n_groups + 1, capacity + 1), -1, dtype=int)
    
    dp[0][0] = 0.0
    
    for i in range(1, n_groups + 1):
        group = groups[i-1]
        for item_idx, item in enumerate(group):
            p = item['profit']
            w = int(item['weight'])
            
            if w > capacity:
                continue
                
            for prev_w in range(capacity - w + 1):
                if dp[i-1][prev_w] != -1.0:
                    new_p = dp[i-1][prev_w] + p
                    if new_p > dp[i][prev_w + w]:
                        dp[i][prev_w + w] = new_p
                        parent[i][prev_w + w] = item_idx
                        
    # Encontrar o melhor lucro na última linha
    best_profit = np.max(dp[n_groups])
    if best_profit == -1.0:
        log.error("Nenhuma solução viável encontrada para a capacidade %d", capacity)
        return None, None, None
        
    best_weight = np.argmax(dp[n_groups])
    
    # Backtracking para encontrar os itens
    best_items = []
    curr_w = int(best_weight)
    for i in range(n_groups, 0, -1):
        item_idx = parent[i][curr_w]
        item = groups[i-1][item_idx]
        best_items.append(item)
        curr_w -= int(item['weight'])
        
    best_items.reverse()
    return float(best_profit), best_items, int(best_weight)
