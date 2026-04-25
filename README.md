# USA Stores DSS — Decision Support System

Sistema Inteligente de Apoio à Decisão para previsão de vendas e otimização de escalas em quatro lojas de retalho nos EUA: **Baltimore, Lancaster, Philadelphia e Richmond**.

---

## Estado do Projeto

| Fase | Módulo | Estado |
|------|--------|--------|
| W4 — Forecasting | `src/forecasting/trainer.py` | Concluído |
| W5 — Optimization | `src/optimization/nsga2_model.py` | Concluído |
| DSS App | `dss_app/app.py` | Operacional |

---

## Como Executar

```bash
# 1. Setup
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 2. Pipeline completo (dados → modelos → relatórios)
python3 main_pipeline.py

# 3. Otimização NSGA-II para todas as lojas
python3 run_optimization.py

# 4. Dashboard interativo
streamlit run dss_app/app.py
```

---

## Arquitetura

```
data/raw/{store}.csv
  → src/data/preparation.py        limpeza, feature engineering
  → data/processed/
  → src/forecasting/trainer.py     treino de 6 modelos por feature set
  → results/02_Forecasting_Report/ + data/processed/final_model_report.csv
  → src/optimization/nsga2_model.py  NSGA-II (2 objetivos, 1 restrição G)
  → results/03_Optimization_Report/
  → dss_app/app.py                 Streamlit UI
```

### Módulos principais

| Módulo | Função |
|--------|--------|
| `src/data/preparation.py` | Outlier clipping (1-99%), interpolação, lags 1-28d, feriados US |
| `src/forecasting/trainer.py` | Seasonal Naive, LR, RF, Holt-Winters, Prophet, SARIMAX; MAE/RMSE/MAPE |
| `src/optimization/nsga2_model.py` | `TiaposeOptimization(ElementwiseProblem)`, `IntegerRepair`, `run_optimization()` |
| `src/utils/profit_logic.py` | Custos RH, capacidade por staff type, lucro semanal |
| `run_optimization.py` | Orquestrador: corre NSGA-II para as 4 lojas, gera CSV + plots |

---

## Resultados — Forecasting (W4)

Melhor modelo por loja (ordenado por RMSE ascendente):

| Loja | Melhor Setup | Modelo | MAE | RMSE | MAPE |
|------|-------------|--------|-----|------|------|
| Baltimore | C_Context_Expert | Linear Regression | 3,450 | 4,672 | 12.9% |
| Lancaster | C_Context_Expert | Linear Regression | 5,112 | 6,318 | 15.2% |
| Philadelphia | C_Context_Expert | Linear Regression | 7,363 | 9,300 | 11.8% |
| Richmond | C_Context_Expert | Prophet | 1,872 | 2,429 | 10.3% |

**Feature set**: `C_Context_Expert` é consistentemente o mais eficaz em todas as lojas — inclui `Num_Customers`, `Pct_On_Sale`, `TouristEvent`, `is_holiday`, `day_of_week`, `sales_lag_1`, `sales_lag_7`, `sales_roll_mean_7`.

**Nota SARIMAX**: O modelo SARIMAX apresenta RMSE 8-25× superior a todos os outros modelos em todas as lojas (ex: Baltimore RMSE=40,590 vs. LR RMSE=4,672). Problema identificado — ver secção de análise abaixo.

---

## Resultados — Otimização NSGA-II (W5)

Semana de referência: 2014-06-08 (Dom) → 2014-06-14 (Sáb)  
Configuração: `pop_size=100`, `n_max_gen=300`, `seed=42`

| Loja | Soluções Pareto | Max Lucro (€) | Staff no Max Lucro | Min Staff | Lucro no Min Staff (€) |
|------|:-:|--:|--:|--:|--:|
| Baltimore | 100 | 2,086 | 72 | 5 | -464 |
| Lancaster | 100 | 2,523 | 72 | 16 | +115 |
| Philadelphia | 100 | 2,474 | 85 | 13 | -156 |
| Richmond | 100 | 1,859 | 56 | 6 | -455 |

**Convergência**: Lancaster (257 gen) e Richmond (281 gen) convergiram dentro do limite. Baltimore e Philadelphia atingiram o limite máximo (301 gen) — indicador de espaço de pesquisa mais complexo.

---

## Análise e Recomendações

### O que está a funcionar corretamente

**Fronteira de Pareto** — 100 soluções não-dominadas em todas as lojas confirma que o NSGA-II está a explorar o espaço de forma eficaz. A curva trade-off lucro/staff é suave e contínua.

**Restrição G (dias úteis ≤ 8 staff)** — a abordagem via `n_ieq_constr` funciona: as soluções de máximo lucro saturar sempre em exatamente 8 funcionários nos dias úteis, o que é o comportamento esperado — a restrição é ativa (binding) e corretamente respeitada.

**`IntegerRepair`** — o operador garante que `hr_x` e `hr_j` são sempre inteiros sem loops Python na população.

**Dominância expert vs. junior** — o algoritmo seleciona corretamente mais peritos (F_x > F_j, capacidade 7 > 6 clientes/dia). Esta é uma conclusão válida do modelo: peritos têm ROI superior por cabeça.

---

### Problemas identificados e o que mudar

#### 1. A variável desconto (`pr`) está degenerada

**Diagnóstico**: A fórmula de lucro atual faz com que a receita por cliente **decreça monotonamente** com o desconto, independentemente do volume:

```
pr=0.00 → revenue/customer = 17.75  
pr=0.10 → revenue/customer = 17.25  
pr=0.30 → revenue/customer = 16.23
```

O desconto aumenta as unidades vendidas (`1/log(2-pr)` cresce), mas o fator `(1-pr)` reduz mais o preço. O resultado é que `pr=0` é sempre ótimo — o algoritmo apenas usa descontos residuais por artefactos de arredondamento na fórmula `round(units)`.

**Impacto**: A variável de desconto tem 21 posições no vetor de decisão (7 dias) mas contribui zero para a diversidade da fronteira de Pareto. O espaço de pesquisa é inflado desnecessariamente.

**Correção recomendada** — introduzir elasticidade da procura:
```python
# Em optimize_weekly_wrapper, antes de montar day_data:
elasticity = 2.0  # calibrar com dados históricos
effective_customers = int(customers * (1 + elasticity * pr_clean))
```
Isto cria um trade-off real: mais desconto → mais clientes → mais volume, mas margem inferior. A fronteira de Pareto passa a ter 3 dimensões genuínas.

---

#### 2. Staff de fim de semana sem limite — soluções irrealistas

**Diagnóstico**: O máximo lucro usa 14-27 funcionários ao domingo. Sem restrição de fim de semana, o algoritmo maximiza a cobertura de clientes sem custo operacional extra (não há penalização de weekend).

**Correção recomendada** — adicionar restrição G para fins de semana:
```python
# Em TiaposeOptimization.__init__, mudar:
self._weekday_idx = np.array([d for d, is_wk in enumerate(forecast_is_weekend) if not is_wk])
# Para incluir todos os dias com um cap diferente:
WEEKEND_CAP = 12   # ajustável por loja
WEEKDAY_CAP = 8

# Em _evaluate, substituir out["G"]:
staff_per_day = np.round(x[INT_IDX]).reshape(N_DAYS, 2).sum(axis=1)
g_weekday = (staff_per_day[self._weekday_idx] - WEEKDAY_CAP).tolist()
g_weekend = (staff_per_day[self._weekend_idx] - WEEKEND_CAP).tolist()
out["G"] = g_weekday + g_weekend
```

---

#### 3. Baltimore e Philadelphia não convergiram

**Diagnóstico**: Ambas atingiram `n_max_gen=301`. Philadelphia tem os maiores volumes de clientes (144-298/dia) e Baltimore tem o maior staff máximo permitido nos fins de semana. O espaço de pesquisa é genuinamente mais complexo.

**Correção**: Aumentar para `n_max_gen=500` especificamente para estas duas lojas. Custo computacional: ~10-15 segundos extra por loja.

---

#### 4. SARIMAX com performance muito fraca

**Diagnóstico**: RMSE de SARIMAX é 8-25× superior ao Linear Regression em todas as lojas. A ordem `(1,1,1)(1,1,1,7)` com diferenciação dupla (regular + sazonal) é demasiado agressiva para séries com 300-400 observações e introduce instabilidade numérica (ConvergenceWarnings visíveis nos logs de Richmond, A e B).

**Correção recomendada**:
```python
# Reduzir para ordem mais conservadora:
order=(1,1,0), seasonal_order=(1,0,1,7)
# Ou usar auto-seleção com pmdarima:
from pmdarima import auto_arima
model = auto_arima(y_train, seasonal=True, m=7, stepwise=True, suppress_warnings=True)
```

---

#### 5. Escala do modelo de lucro não calibrada

**Diagnóstico**: O lucro semanal otimizado (€1,859-€2,523) é muito inferior às vendas históricas diárias observadas ($20,000-$80,000/dia). O modelo de lucro em `profit_logic.py` usa uma fórmula paramétrica que não foi calibrada contra os dados históricos reais — é um modelo de simulação relativa, não absoluta.

**Impacto prático**: Os valores absolutos de lucro não são comparáveis às vendas reais. As decisões de escala (mais/menos staff) são corretas na direção, mas a magnitude não é interpretável diretamente.

**Se a calibração for necessária**: Ajustar o preço base por unidade em `sales_x = round(units_x * (1-pr) * 1.07)` para que `1.07` reflita o preço médio real por unidade de produto vendida.

---

### Resumo das prioridades

| Prioridade | Ação | Impacto |
|:---:|------|---------|
| 1 | Adicionar elasticidade da procura ao desconto | Torna a variável `pr` genuinamente útil |
| 2 | Adicionar restrição G para staff de fim de semana | Elimina planos operacionalmente irrealistas |
| 3 | Corrigir SARIMAX (ordem ou auto_arima) | Remove outlier das métricas de forecasting |
| 4 | Aumentar `n_max_gen=500` para Baltimore e Philadelphia | Convergência completa |
| 5 | Calibrar escala do modelo de lucro | Resultados interpretáveis em € reais |

---

## Outputs gerados

```
results/
  00_Master_Summary/
    fidelity_experimentation_report.csv   métricas completas (6 modelos × 3 features × 4 lojas)
    model_comparison_summary.png
  02_Forecasting_Report/{Store}/{FeatureSet}/
    Comparison_All_Models.png
    Model_Details/SARIMAX.png
  03_Optimization_Report/
    {store}_pareto.csv        100 soluções não-dominadas com variáveis de decisão
    {store}_pareto_front.png  scatter Lucro vs. Staff
    {store}_best_plan.png     plano semanal detalhado da melhor solução
    optimization_summary.csv  resumo cross-store

data/processed/
  final_model_report.csv      consumido pelo dashboard Streamlit
  {store}_processed.csv × 4
  all_stores_processed.csv
```
