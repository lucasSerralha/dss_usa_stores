# USA Stores — Sistema de Apoio à Decisão (DSS) 2026

Sistema Inteligente de Apoio à Decisão para previsão de vendas e otimização de operações em quatro lojas nos EUA: **Baltimore, Lancaster, Philadelphia e Richmond**.

---

## Arquitetura

```
data/               Dados brutos e processados por loja
src/
  data/             Limpeza, feature engineering, análise estatística
  forecasting/      Motor de treino multi-algoritmo (SARIMAX, RF, Prophet, Ensemble)
  optimization/     Algoritmos de otimização (Hill Climbing, NSGA-II, MOEA/D, U-NSGA-III)
  model_testing/    Diagnósticos estatísticos (Gaussiano, Poisson, MOEA/D, U-NSGA-III)
  utils/            profit_logic.py — cálculo de lucro e wrapper de otimização
scripts/            Scripts de execução por fase
results/            Outputs organizados por fase (00–05)
docs/               Notebooks de análise e documentação
app.py              Dashboard Streamlit interativo
main_pipeline.py    Pipeline completo de forecasting
```

---

## Previsão de Vendas

Três cenários de experimentação com os mesmos modelos, avaliados por **MAE, RMSE, MAPE e NMAE**:

| Cenário | Variáveis incluídas |
|---|---|
| A — Temporal Base | Padrões cíclicos históricos (dia, semana, mês) |
| B — Sales Dynamics | Cenário A + lags de vendas (lag-1, lag-7, média móvel 7d) |
| C — Context Expert | Cenário B + promoções, eventos turísticos, feriados |

**Modelos avaliados**: SARIMAX, Prophet, Random Forest, Linear Regression, Holt-Winters, Ensemble Top-3 Experts (melhor RMSE na maioria das lojas), Seasonal Naive (baseline).

**Backtesting**: walk-forward com `TimeSeriesSplit(n_splits=10)` — resultados em `results/05_Backtesting/`.

---

## Otimização

### Objetivos e Algoritmos

| Objetivo | Problema | Algoritmo | Resultado |
|---|---|---|---|
| O1 | Maximizar lucro semanal por loja | Hill Climbing + random restarts | Solução única ótima por loja |
| O2 | Alocação conjunta ≤ 10 000 unidades (4 lojas) | Hill Climbing + penalização | Plano semanal com restrição de orçamento |
| O3 | Lucro vs Staff — fronteira de Pareto multi-loja | U-NSGA-III / MOEA/D | 64–96 soluções Pareto diversas |

### Modelos Multi-objetivo (O3)

- **U-NSGA-III** (`unsga3_model.py`): 64 soluções Pareto, lucro máx $37 135 com 85 staff — hipervolume superior.
- **MOEA/D** (`moead_model.py`): 96 soluções Pareto, lucro máx $17 185 com 86 staff.
- **Problema conjunto** (`joint_problem.py`): 84 variáveis de decisão (4 lojas × 7 dias × 3 variáveis: desconto, experts, júniors).

### Modelos Probabilísticos

- **Poisson (GLM)**: previsão da chegada diária de clientes.
- **Gaussiano (OLS)**: variabilidade das vendas com IC 95% (cenários Pessimista/Realista/Otimista).
- **Logístico**: probabilidade de conversão de clientes em função do desconto aplicado.

---

## Dashboard (Streamlit)

`streamlit run app.py`

| Tab | Conteúdo |
|---|---|
| Previsão de Vendas | Comparação interativa real vs previsto por modelo |
| Diagnóstico Técnico | Resíduos, histograma de erros, tabela NMAE completa |
| Decomposição Temporal | Tendência de longo prazo + sazonalidade semanal |
| Inteligência de IA | Importância de variáveis (XAI) |
| Modelos de Regras | Poisson / Gaussiano / Logístico em tempo real |
| Otimização | Resultados O1/O3, Pareto MOEA/D vs U-NSGA-III, sensibilidade N_MAX_GEN, **otimização O1 em tempo real** |

---

## Como Executar

### Pipeline completo
```bash
python main_pipeline.py
```

### Scripts de otimização
```bash
# O1 — Hill Climbing por loja
python scripts/run_individual_optimization.py

# O2 — Alocação conjunta com restrição de 10k unidades
python scripts/run_allocation_optimization.py

# O2 (Knapsack) — NSGA-II + Programação Dinâmica
python scripts/run_allocation_optimization_knapsack.py

# O2 — NSGA-II multi-objetivo
python scripts/run_optimization.py

# O3 — MOEA/D + U-NSGA-III (problema conjunto 4 lojas)
python scripts/run_joint_optimization_o3.py

# Backtesting walk-forward (10 folds, RF + LR + Holt-Winters)
python scripts/run_backtesting.py

# Análise de sensibilidade N_MAX_GEN (gera CSV para o dashboard)
python scripts/compare_optimization_nmaxgen.py
```

### Diagnósticos de modelos
```bash
python src/model_testing/moead/moead_diagnostics.py    # MOEA/D vs NSGA-II
python src/model_testing/unsga3/unsga3_diagnostics.py  # U-NSGA-III vs NSGA-II
python src/model_testing/gaussiano/gaussiano_diagnostics.py
python src/model_testing/poisson/poisson_diagnostics.py
```

---

## Resultados

```
results/
  00_Master_Summary/       fidelity_experimentation_report.csv (MAE/RMSE/MAPE/NMAE por modelo)
  01_EDA/                  Análise exploratória
  02_Forecasting/          Previsões por loja e cenário
  03_Optimization/
    individual/            O1 — Hill Climbing por loja
    allocation/            O2 — Alocação conjunta
    multiobjective/        NSGA-II Pareto fronts
    joint_o3/              MOEA/D + U-NSGA-III Pareto fronts
  04_Model_Testing/        Diagnósticos MOEA/D, U-NSGA-III, Gaussiano, Poisson
  05_Backtesting/          Walk-forward backtesting (MAE/RMSE/MAPE/NMAE por fold)
```

---

## Dependências principais

```
streamlit  plotly  pandas  numpy  scikit-learn  pymoo
statsmodels  prophet  xgboost  matplotlib
```

`pip install -r requirements.txt`
