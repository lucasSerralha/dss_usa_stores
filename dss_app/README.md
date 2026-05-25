# DSS USA Stores — Decision Support System

Plataforma integrada de apoio à decisão para previsão da procura, auditoria científica de modelos e otimização multiobjetivo de recursos em quatro lojas retalhistas nos EUA: **Baltimore, Lancaster, Philadelphia e Richmond**.

Esta aplicação fornece uma interface visual e interativa para explorar os resultados do pipeline de Ciência de Dados, permitindo o planeamento estratégico e tático ao nível da loja e da rede.

## 🚀 Como Executar

Certifique-se de que tem as dependências instaladas (ver `requirements.txt` na raiz do projeto) e execute a aplicação Streamlit:

```bash
streamlit run dss_app/app.py
```

## 🏗️ Estrutura da Aplicação

A aplicação está dividida em três pilares fundamentais, acessíveis através da navegação lateral:

### 1. Previsão de Vendas e Fluxo (`app.py` & `dev_dashboard.py`)
*   **Arquitetura Híbrida:** Integração de modelos **Prophet**, **LightGBM**, **SARIMAX** e **Ensemble**.
*   **Modelos Probabilísticos:** Estimativa de fluxo de clientes via **Poisson**, variabilidade de vendas via **Gaussiano** e conversão de descontos via **Logístico**.
*   **Precisão:** Foco no Cenário C (Context Expert), entregando ~90% de fidelidade preditiva.

### 2. Auditoria de Modelos (`pages/01_Auditoria_Predicoes.py`)
*   **Benchmarking Científico:** Avaliação de modelos contra a baseline *Seasonal Naive*.
*   **Métricas de Performance:** Análise detalhada de **RMSE**, **MAPE** e **NMAE** via *TimeSeriesSplit*.
*   **Leaderboard:** Ranking dinâmico dos melhores modelos por loja e por cenário de dados.

### 3. Otimização de Recursos
O motor de decisão opera em três camadas distintas:

*   **O1 — Otimização Monobjetivo Individual (`pages/02_Otimizacao_Monobjetivo.py`)**:
    *   Foco: Maximização de lucro por loja.
    *   Algoritmo: **Hill Climbing** com Random Restarts.
*   **O2 — Alocação Conjunta de Rede (`pages/02_Otimizacao_Monobjetivo.py`)**:
    *   Foco: Gestão de restrição logística (limite de 10.000 unidades/semana na rede).
    *   Algoritmo: Heurística com **Death Penalty** e suporte a **Knapsack (Mochila)** para alocação ótima.
*   **O3 — Estratégia Multiobjetivo (`pages/03_Otimizacao_Multiobjetivo.py`)**:
    *   Foco: Trade-off entre Lucro Total vs. Custo de Staffing.
    *   Algoritmo: **U-NSGA-III** e **MOEA/D** para geração da Fronteira de Pareto.
    *   **Laboratório Interativo:** Slider de escalarização ($w$) para seleção dinâmica de planos operacionais.

## 🛠️ Tecnologias Utilizadas

*   **Frontend/App:** [Streamlit](https://streamlit.io/)
*   **Visualizações:** [Plotly](https://plotly.com/python/)
*   **Previsão:** `prophet`, `scikit-learn`, `statsmodels`
*   **Otimização:** `pymoo` (NSGA-III, MOEA/D), implementações customizadas de Hill Climbing.
*   **Dados:** `pandas`, `numpy`

## 📊 Dashboard de Desenvolvimento (`dev_dashboard.py`)
Para uma visão técnica profunda, o `dev_dashboard.py` oferece ferramentas adicionais de diagnóstico:
*   Análise de Resíduos e Estabilidade.
*   Decomposição de Tendências e Sazonalidade.
*   Explicabilidade (XAI) com Importância de Variáveis.
*   Relatório Executivo consolidado para tomada de decisão.

---
**DSS USA Stores** | *Decision Support System* | 2026
