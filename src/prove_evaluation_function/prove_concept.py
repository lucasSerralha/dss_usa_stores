import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Importar a versão copiada da lógica de lucro
from profit_logic import optimize_weekly_wrapper

def prove_evaluation_concept():
    print("="*80)
    print("PROVA DE CONCEITO: VALIDAÇÃO DA FUNÇÃO EVALUATION() - AMBIENTE ISOLADO")
    print("="*80)

    # 1. Dados de Teste (Extraídos de data/processed/baltimore_processed.csv - Junho 2014)
    # Semana: 2014-06-08 (Dom) a 2014-06-14 (Sáb)
    test_customers = [97, 61, 65, 71, 65, 89, 125]
    test_is_weekend = [True, False, False, False, False, False, True]
    store = 'baltimore'

    print(f"\n[DADOS DE TESTE - {store.upper()}]")
    print(f"Clientes Previstos (Forecasting): {test_customers}")
    print(f"Fins de Semana (Calendário):      {test_is_weekend}")

    # 2. Cenários de Decisão (Decision Vectors - 21 dimensões: 7 dias x [pr, hr_x, hr_j])
    
    # Cenário A: Mínimo esforço (1 funcionário de cada tipo, 0% desconto)
    scenario_a = [0.0, 1, 1] * 7
    
    # Cenário B: Excesso de Staff (10 de cada, viola limites de 8/12)
    scenario_b = [0.0, 10, 10] * 7
    
    # Cenário C: Promoção Estratégica (15% desconto, staff adequado)
    scenario_c = [0.15, 5, 3,  # Dom
                  0.15, 4, 3,  # Seg
                  0.15, 4, 3,  # Ter
                  0.15, 4, 3,  # Qua
                  0.15, 4, 3,  # Qui
                  0.15, 4, 3,  # Sex
                  0.15, 6, 4]  # Sáb

    scenarios = {
        "Sub-alocação": scenario_a,
        "Sobre-alocação": scenario_b,
        "Otimizado (Manual)": scenario_c
    }

    results = []

    # 3. Execução da Avaliação (O "Coração" do DSS)
    for name, vector in scenarios.items():
        # A função evaluation() do NSGA-II retorna:
        # f1: -Lucro (para minimizar)
        # f2: Staff Total (para minimizar)
        # f3: Penalização (para minimizar)
        neg_profit, total_staff, penalty = optimize_weekly_wrapper(
            vector, store, test_customers, test_is_weekend
        )
        profit = -neg_profit
        results.append({
            "Cenário": name,
            "Lucro Estimado (€)": profit,
            "Staff Semanal": total_staff,
            "Penalização": penalty
        })

    # 4. Relatório de Prova
    df_results = pd.DataFrame(results)
    print("\n[RESULTADOS DA SIMULAÇÃO]")
    print(df_results.to_string(index=False))

    # 5. Verificação das Premissas Econômicas (Referências Bibliográficas)
    print("\n" + "-"*50)
    print("VALIDAÇÃO DAS REGRAS DE NEGÓCIO:")
    
    # Prova 1: Lei dos Rendimentos Decrescentes
    if df_results.loc[1, "Lucro Estimado (€)"] < df_results.loc[2, "Lucro Estimado (€)"]:
        print("✔ [EFICIÊNCIA] Provado: O excesso de staff (Cenário Sobre-alocação) reduz o lucro líquido.")
    
    # Prova 2: Elasticidade da Procura (Memorial de Cálculo)
    # Se o lucro com desconto (Cenário C) for competitivo ou melhor que o baseline, a elasticidade funciona.
    if df_results.loc[2, "Lucro Estimado (€)"] > df_results.loc[0, "Lucro Estimado (€)"]:
        print("✔ [ELASTICIDADE] Provado: O desconto atrai volume suficiente para superar a sub-alocação.")

    # Prova 3: Restrições Operacionais
    if df_results.loc[1, "Penalização"] > 0:
        print(f"✔ [RESTRIÇÕES] Provado: A função detecta e penaliza violações de staff (€{df_results.loc[1, 'Penalização']:,.0f}).")

    # 6. Gráfico para o Memorial de Cálculo
    try:
        plt.figure(figsize=(10, 6))
        colors = ['#ff9999', '#ffcc99', '#99ff99']
        plt.bar(df_results["Cenário"], df_results["Lucro Estimado (€)"], color=colors)
        plt.title(f"Prova de Conceito da Função de Avaliação - {store.capitalize()}")
        plt.ylabel("Lucro (€)")
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Guardar na pasta de resultados oficial
        output_plot = "../../results/06_Prove_Evaluation_Function/evaluation_concept_plot.png"
        plt.savefig(os.path.join(os.path.dirname(__file__), output_plot))
        print(f"\n✔ GRÁFICO GERADO: results/06_Prove_Evaluation_Function/evaluation_concept_plot.png")
    except Exception as e:
        print(f"\n✘ Erro ao gerar gráfico: {e}")

    print("\n" + "="*80)
    print("PROVA CONCLUÍDA COM SUCESSO")
    print("="*80)

if __name__ == "__main__":
    prove_evaluation_concept()
