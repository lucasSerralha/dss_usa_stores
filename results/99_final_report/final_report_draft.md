# 📊 Relatório Final: Sistema Inteligente de Apoio à Decisão (DSS)
## Otimização de Vendas e Escala — USA Stores 2026

**Data:** 08 de Maio de 2026  
**Status:** Esboço Consolidado (Versão 1.0)  
**Âmbito:** Baltimore, Lancaster, Philadelphia e Richmond

---

## 1. Introdução e Sumário Executivo
Este documento apresenta os resultados finais do desenvolvimento e implementação do DSS para a rede USA Stores. O sistema integra modelos avançados de *Forecasting* e otimização meta-heurística para maximizar o lucro operacional enquanto minimiza custos de pessoal.

**Principais Conquistas:**
*   Identificação de drivers de venda via **Análise de Contexto (Cenário C)**.
*   Previsões com erro médio (RMSE) reduzido em até 15% comparado ao baseline.
*   Geração de **Fronteiras de Pareto** para decisão executiva informada sobre alocação de staff.

---

## 2. Metodologia: O Motor de Inteligência

### 2.1 Previsão de Procura (Forecasting)
O sistema avaliou três arquiteturas de dados:
*   **Cenário A (Temporal):** Padrões cíclicos.
*   **Cenário B (Dynamics):** Persistência de curto prazo.
*   **Cenário C (Expert):** Inclusão de Promoções, Eventos e Feriados.
*   *Resultado:* O modelo **SARIMAX/Ensemble** no Cenário C provou ser o mais fidedigno para todas as localizações.

### 2.2 Otimização Multi-Objetivo (NSGA-II)
Utilizamos o algoritmo **NSGA-II** para resolver o conflito entre:
1.  **Maximizar Lucro:** Considerando elasticidade de preço ($k=2.5$) e escala financeira ($35\times$).
2.  **Minimizar Staff:** Respeitando limites operacionais (8/dia úteis, 12/fins-de-semana).

---

## 3. Resultados Consolidados por Localização

| Loja | Melhor Modelo | RMSE (Fidedignidade) | Lucro Máximo Est. (€/Semana) | Staff Sugerido (Ponto Ótimo) |
| :--- | :--- | :--- | :--- | :--- |
| **Baltimore** | SARIMAX (C) | 3,891 | ~64,120 | 50-60 |
| **Lancaster** | Ensemble (C) | *Consulte CSV* | ~77,525 | 45-55 |
| **Philadelphia** | Ensemble (C) | *Consulte CSV* | ~62,930 | 55-63 |
| **Richmond** | SARIMAX (C) | *Consulte CSV* | ~68,985 | 40-50 |

---

## 4. Validação da Prova de Conceito (PoC)
Realizamos um teste de sanidade isolado para garantir a integridade matemática da função de avaliação.
*   **Confirmação de Elasticidade:** O modelo reage positivamente a descontos estratégicos.
*   **Confirmação de Penalização:** Violações de staff em dias úteis são desencorajadas matematicamente (€1.000/pessoa extra).

---

## 5. Declarações de Uso de Inteligência Artificial (IA)

Em conformidade com as diretrizes de transparência de 2026:

1.  **IA Preditiva:** Utilização de modelos *Facebook Prophet* e *Ensembles* de Machine Learning para modelagem de incerteza.
2.  **IA Otimizadora:** Implementação de algoritmos genéticos (NSGA-II) via biblioteca *Pymoo*.
3.  **IA Generativa e Agêntica:** O desenvolvimento, depuração de lógica econômica (correção de elasticidade) e a orquestração deste pipeline foram auxiliados por **Agentes de IA (Gemini CLI)**, garantindo agilidade na prototipagem e validação de código.

---

## 6. Conclusão e Recomendações
O sistema está pronto para implantação via interface **Streamlit**. Recomendamos que os gerentes de loja foquem nas soluções da Fronteira de Pareto que equilibram um lucro de segurança com um staff não superior a 10% do limite operacional para evitar fadiga da equipe.

---
*Fim do Documento*
