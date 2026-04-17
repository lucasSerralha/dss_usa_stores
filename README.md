# 🇺🇸 USA Stores Sales Forecasting & Optimization - DSS 2026

Este repositório contém um **Sistema Inteligente de Apoio à Decisão (DSS)** de alto desempenho para a previsão e otimização de vendas em quatro localizações estratégicas nos Estados Unidos: **Baltimore, Lancaster, Philadelphia e Richmond**.

---

## 🏗️ Arquitetura do Sistema 

O sistema segue uma arquitetura modular que separa a preparação de dados, o motor de inteligência e a interface de decisão:

### 📁 `src/` - Núcleo de Inteligência
*   **`data/preparation.py`**: Limpeza científica de dados, tratamento de outliers e engenharia de atributos avançada (Lags dinâmicos, médias móveis e contexto de feriados).
*   **`forecasting/trainer.py`**: Motor de treino multi-algoritmo capaz de avaliar modelos clássicos (ARIMAX, Holt-Winters) e Machine Learning (Random Forest, Prophet).
*   **`optimization/nsga2_model.py`**: Algoritmo Genético de vanguarda (NSGA-II) para a fase de decisão ótima (Staffing vs Profit).

### 📁 `results/` - Repositório de Evidências
*   Organizado por **Cenários de Experimentação**, permitindo comparar como diferentes conjuntos de variáveis afetam a fidedignidade da previsão.

---

## 🧪 Suíte de Experimentação Científica

Diferente de abordagens básicas, este sistema avalia automaticamente três cenários de variáveis para encontrar a máxima fidedignidade:
1.  **Cenário A (Temporal Base)**: Focado em padrões cíclicos puramente históricos.
2.  **Cenário B (Sales Dynamics)**: Integra a dinâmica de curto prazo (ontem) e persistência de dados.
3.  **Cenário C (Context Expert)**: Integra o contexto total de negócio (Promoções, Eventos Locais e Feriados).

---

## 📈 Modelos Integrados
*   **ARIMAX**: Modelo estatístico clássico que utiliza variáveis exógenas (Promoções) para ajustar a série temporal.
*   **Prophet (Meta)**: Abordagem Bayesiana robusta a anomalias e feriados complexos.
*   **Random Forest & Linear Regression**: Abordagem de Machine Learning para capturar correlações não lineares.
*   **Holt-Winters**: Suavização exponencial tripla para padrões puramente sazonais.

---

## 💎 Dashboard (Streamlit)
O sistema inclui uma interface de visualização interativa (Plotly) que oferece:
*   **Navegação por Separadores**: Previsão, Diagnóstico de Erros, Decomposição de Tendências e IA.
*   **Análise de Resíduos**: Visualização estatística para validar a honestidade dos modelos.
*   **XAI (Explainable AI)**: Gráficos de importância de variáveis para explicar os drivers do negócio.
*   **KPI Financeiro**: Estimativa de poupança financeira comparando a IA com o baseline.

---

## 🚀 Como Executar

1.  **Motor de Cálculo**: `python main_pipeline.py` (Processa e gera todos os relatórios).
2.  **Interface DSS**: `streamlit run dss_app/app.py` (Lança o dashboard interativo).

---

## 📅 Próximos Passos (Fase Atual)
Após a conclusão da fase de Previsão, o sistema está agora focado na **Otimização Multi-Objetivo**, utilizando os modelos treinados para determinar o número ideal de funcionários que minimiza custos e maximiza o lucro esperado por loja.
