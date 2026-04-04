# 🇺🇸 USA Stores Sales Forecasting - DSS Project 2026

Este repositório contém o sistema completo de previsão de vendas para quatro lojas de retalho nos Estados Unidos (Baltimore, Lancaster, Philadelphia e Richmond). O projeto foi desenvolvido como parte de um Sistema Inteligente de Apoio à Decisão (DSS), focando-se em técnicas avançadas de Ciência de Dados e Forecasting Multivariado.

---

## 🏗️ Arquitetura do Projeto ("Lean Architecture")
O projeto utiliza uma estrutura modular e profissional para facilitar o desenvolvimento e a manutenção:

- **`data/`**: Armazenamento de dados brutos (`raw`) e dados limpos e processados (`processed`).
- **`src/`**: Motor lógico centralizado:
    - `preparation.py`: Limpeza científica (tratamento de outliers via clipping 1%-99%), engenharia de variáveis (feriados dos EUA, lags semanais e médias móveis).
    - `trainer.py`: Módulo de treino comparativo com avaliação multi-modelo e geração automática de gráficos.
    - `utils/`: Funções utilitárias partilhadas.
- **`notebooks/`**: Relatórios de investigação técnica (`Research_&_Discovery.ipynb`) com justificações estatísticas (ACF, Boxplots).
- **`main_pipeline.py`**: Orquestrador único que executa o fluxo completo do sistema.

---

## 📊 Modelagem e Forecasting (W1 - W4)
O sistema avalia cinco abordagens distintas para garantir a maior precisão possível (H=7 dias):

1.  **Seasonal Naive (Baseline)**: Previsão baseada no mesmo dia da semana anterior.
2.  **Linear Regression**: Modelo estatístico multivariado base.
3.  **Random Forest**: Algoritmo de aprendizagem automática não-linear (Ensemble).
4.  **Holt-Winters (ETS)**: Suavização exponencial avançada com sazonalidade.
5.  **Prophet (Meta)**: Modelo bayesiano multivariado robusto a feriados e tendências.

**Métricas de Performance:** MAE (Mean Absolute Error), RMSE (Root Mean Squared Error) e MAPE.

---

## 🚀 Como Executar
Para limpar os dados, treinar todos os modelos, gerar o relatório master e os gráficos de previsão, basta correr:
```powershell
python main_pipeline.py
```

Os resultados serão gerados em:
- **Relatório:** `data/processed/final_model_report.csv`
- **Gráficos:** `data/processed/plots/`

---

## 📅 Próximos Passos
O projeto encontra-se atualmente na transição para a **Fase de Otimização (W5-W6)**, onde as previsões geradas serão utilizadas para maximizar os lucros através da alocação eficiente de recursos humanos e gestão de promoções.
