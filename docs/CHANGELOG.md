# Registo de Alterações — Assistência IA

> Documento gerado automaticamente a 2026-05-21.  
> Regista todas as alterações introduzidas por assistência IA (Claude Sonnet 4.6) ao projeto DSS USA Stores, com justificação para cada mudança.

---

## Índice

1. [Correção da Fórmula de Lucro](#1-correção-da-fórmula-de-lucro)
2. [Remoção de Parâmetros Não Documentados](#2-remoção-de-parâmetros-não-documentados)
3. [Correção da Variável Target (Forecasting)](#3-correção-da-variável-target-forecasting)
4. [Correção dos Feature Sets](#4-correção-dos-feature-sets)
5. [DSS — Homepage (app.py)](#5-dss--homepage-apppy)
6. [DSS — Página 01: Auditoria de Modelos](#6-dss--página-01-auditoria-de-modelos)
7. [DSS — Página 02: Otimização Monobjetivo](#7-dss--página-02-otimização-monobjetivo)
8. [DSS — Página 03: Otimização Multiobjetivo](#8-dss--página-03-otimização-multiobjetivo)
9. [Moeda nos Resultados](#9-moeda-nos-resultados)
10. [Análise da Fronteira de Pareto](#10-análise-da-fronteira-de-pareto)

---

## 1. Correção da Fórmula de Lucro

**Ficheiros:** `src/utils/profit_logic.py`, `src/utils/profit_logic_knapsack.py`

**Commits:** `8090607`, `77ac681`

### O que mudou

```python
# ANTES (incorreto) — arredondamento sobre o total diário
units_x = assisted_x * u_per_x
sales_x = round(units_x * (1 - pr) * 1.07)   # round do agregado

# DEPOIS (correto) — arredondamento por cliente, conforme enunciado
p_x = round(u_per_x * (1 - pr) * 1.07)       # P = round(U × (1−PR) × 1.07)
sales_x = assisted_x * p_x                    # sem round exterior
```

### Porquê

O enunciado define explicitamente:

```
U = round(F × 10 / ln(2 − PR))    ← unidades vendidas por cliente
P = round(U × (1 − PR) × 1.07)    ← PREÇO por cliente (arredondamento unitário)
R_{s,d} = Σ P − J·custo_J − X·custo_X
```

A formulação anterior aplicava `round()` sobre o total diário (`assisted_x × u_per_x × (1−PR) × 1.07`), o que difere matematicamente da especificação. O arredondamento deve ser aplicado ao preço unitário e depois multiplicado pelo número de clientes assistidos.

**Impacto:** Para 10 clientes, u_per_x=17, PR=0:
- Errado: `round(10 × 17 × 1.07) = round(181.9) = 182`
- Correto: `10 × round(17 × 1.07) = 10 × round(18.19) = 10 × 18 = 180`

### Também adicionado

- Campo `p_x` e `p_j` (preço por cliente) retornado pelo dicionário de `calculate_daily_metrics`, útil para debug e auditoria.
- Bloco `__main__` expandido com teste de sanidade e verificação manual.

---

## 2. Remoção de Parâmetros Não Documentados

**Ficheiros:** `src/utils/profit_logic.py`, `src/utils/profit_logic_knapsack.py`, `src/optimization/nsga2_model.py`, scripts de otimização

**Commit:** `8090607`

### O que mudou

Removidos dois parâmetros que **não existem no enunciado nem no guia do professor**:

| Parâmetro | Valor anterior | Justificação de remoção |
|-----------|---------------|------------------------|
| `PROFIT_SCALE = 35` | Multiplicador de escala financeira | Não mencionado no enunciado |
| `ELASTICITY_K = 2.5` | Coeficiente de elasticidade preço-procura | Não mencionado no enunciado |

Também removidos **limites de staff artificiais** que não constam do enunciado:
- Limite de 8 funcionários em dias úteis
- Limite de 12 funcionários em fins de semana

### Porquê

Estes parâmetros foram adicionados internamente pela equipa (documentados em `docs/00_memorial_calculo_otimizacao.md`) para tentar calibrar os lucros para uma escala percebida como mais realista (~€70k–100k/semana). Contudo:

1. O guia do professor foi lido integralmente — **nenhum destes parâmetros aparece**.
2. O enunciado especifica apenas: F_j, F_x (produtividade por loja), W_s (custo fixo), custos diários de J e X.
3. Os valores de referência do enunciado (Baltimore=146, Philadelphia=1728) são **consistentes com a fórmula sem escala**.
4. Usar parâmetros não documentados é um risco académico sério — o professor pode questionar a origem.

**Resultado:** Os lucros passaram para a escala correta (centenas de USD por semana, não dezenas de milhares), que é o que a fórmula do enunciado produz.

---

## 3. Correção da Variável Target (Forecasting)

**Ficheiro:** `src/data/preparation.py`

### O que mudou

```python
# ANTES
df['y'] = df['Sales']          # target = vendas em USD

# DEPOIS
df['y'] = df['Num_Customers']  # target = número diário de clientes
```

### Porquê

O enunciado especifica que o sistema deve prever `C_{s,d}` — o **número diário de clientes** — para alimentar a função de otimização:

```
A_{s,d} = min(7·X + 6·J, C_{s,d})
```

Prever `Sales` (USD) em vez de `Num_Customers` significa:
1. O output do forecasting não é compatível com o input da otimização.
2. Os modelos estariam a aprender a prever uma série com escala completamente diferente (~15k–80k USD/dia vs. ~40–300 clientes/dia).
3. Os resultados do backtesting (MAE, RMSE) estariam em USD em vez de clientes, tornando as métricas sem sentido no contexto do sistema.

**Consequência:** Todos os resultados de forecasting foram re-gerados para `results_v2/`.

---

## 4. Correção dos Feature Sets

**Ficheiro:** `main_pipeline.py`

### O que mudou

```python
# ANTES — Cenário C usava Num_Customers como feature (data leakage!)
experiments = {
    "A_Temporal_Base": ['day_of_week', 'IsWeekend', 'month', 'season_num',
                        'sales_lag_7', 'sales_lag_28'],
    "B_Sales_Dynamics": ['day_of_week', 'month', 'sales_lag_1', 'sales_lag_7',
                         'sales_roll_mean_7', 'sales_roll_std_7'],
    "C_Context_Expert": ['Num_Customers', 'Pct_On_Sale', 'TouristEvent',   # ← LEAKAGE
                         'is_holiday', 'days_to_next_holiday',
                         'day_of_week', 'sales_lag_1', 'sales_lag_7',
                         'sales_roll_mean_7']
}

# DEPOIS — Num_Customers é o TARGET; usar lags em vez
experiments = {
    "A_Temporal_Base": ['day_of_week', 'IsWeekend', 'month', 'season_num',
                        'customers_lag_7', 'customers_lag_28'],
    "B_Sales_Dynamics": ['day_of_week', 'month',
                         'customers_lag_1', 'customers_lag_7',
                         'sales_lag_7', 'sales_roll_mean_7'],
    "C_Context_Expert": ['customers_lag_7', 'Pct_On_Sale', 'TouristEvent',
                         'is_holiday', 'days_to_next_holiday', 'day_of_week',
                         'sales_lag_7', 'sales_roll_mean_7']
}
```

### Porquê

O Cenário C incluía `Num_Customers` (contemporâneo, i.e., do mesmo dia) como feature. Como `Num_Customers` é agora o **target**, isto constitui **data leakage circular**: o modelo teria acesso ao valor que está a tentar prever no momento de treino, produzindo métricas artificialmente perfeitas que não se mantêm em produção.

A solução é substituir `Num_Customers` contemporâneo por `customers_lag_7` (valor de 7 dias antes), que é conhecível no momento da previsão.

---

## 5. DSS — Homepage (app.py)

**Ficheiro:** `dss_app/app.py`

**Commit:** `77ac681`

### O que mudou

Reescrita completa da homepage. A versão anterior era estática/placeholder. A nova versão:

| Funcionalidade | Antes | Depois |
|---|---|---|
| Dados nos KPIs | Hardcoded ("—") | Carregados dos CSVs reais |
| Cards de loja | Estáticos | Lucro real do SA por loja |
| Navegação entre páginas | Texto simples | `st.page_link()` clicável |
| Moeda | — | USD ($) conforme enunciado |

**Dados carregados:**
- `results_v2/03_Optimization/individual/simulated_annealing/simulated_annealing_summary.csv` → KPI O1
- `results_v2/03_Optimization/allocation/allocation_summary.csv` → KPI O2
- `results_v2/03_Optimization/joint_o3/joint_o3_summary.csv` → KPI O3 (Pareto)
- `results_v2/00_Master_Summary/fidelity_experimentation_report.csv` → Precisão de previsão

### Porquê

A homepage deve funcionar como painel de controlo do sistema. Mostrar dados reais (em vez de placeholders) é necessário para a demonstração em vídeo e para o utilizador compreender o estado do sistema à partida.

---

## 6. DSS — Página 01: Auditoria de Modelos

**Ficheiro:** `dss_app/pages/01_Auditoria_Predicoes.py`

**Commit:** `77ac681`

### O que mudou

#### 6.1 Caminhos dos ficheiros

```python
# ANTES
path = os.path.join(root, "results", "00_Master_Summary", "fidelity_experimentation_report.csv")

# DEPOIS
path = os.path.join(root, "results_v2", "00_Master_Summary", "fidelity_experimentation_report.csv")
```

**Porquê:** A pasta `results/` foi descontinuada quando se corrigiu o target de `Sales` para `Num_Customers`. Todos os resultados novos estão em `results_v2/`.

#### 6.2 Unidades do RMSE e MAE

```python
# ANTES — mostrava símbolo de dólar (target era Sales em USD)
f'<span class="podium-metric-value">${row["RMSE"]:.2f}</span>'

# DEPOIS — mostra número com unidade "clientes" (target é Num_Customers)
f'<span class="podium-metric-value">{row["RMSE"]:.2f} clientes</span>'
```

**Porquê:** O RMSE e MAE medem erro em número de clientes (não em USD), portanto o símbolo `$` estava incorreto. A unidade correta é "clientes".

#### 6.3 Botão de exportação CSV

Adicionado `st.download_button` para exportar o leaderboard (guia §5.2 recomenda exportação).

---

## 7. DSS — Página 02: Otimização Monobjetivo

**Ficheiro:** `dss_app/pages/02_Otimizacao_Monobjetivo.py`

**Commit:** `77ac681`

### O que mudou

Reescrita completa. A versão anterior apenas mostrava resultados do Hill Climbing. A nova versão cobre:

#### 7.1 Todos os algoritmos O1

| Algoritmo | Antes | Depois |
|---|---|---|
| Monte Carlo | Ausente | Tab com métricas + convergência + plano |
| Hill Climbing | Presente (parcial) | Tab completa |
| Simulated Annealing | Ausente | Tab com métricas + convergência + plano |
| NSGA-II | Ausente | Tab com fronteira Pareto + plano |

Adicionada tabela comparativa cross-algorithm e gráfico de barras agrupado.

#### 7.2 Tabela enriquecida do plano semanal

```python
# Antes — mostrava apenas: Dia, FDS, Clientes, Desconto, Experts, Juniores, Staff
# Depois — calcula e mostra (guia §5.2):
{
    "Dia":           ...,
    "FDS":           ...,   # fim de semana?
    "Clientes":      ...,
    "Desconto":      ...,
    "Experts (X)":   ...,
    "Juniores (J)":  ...,
    "Staff Total":   ...,
    "Assist. X":     ...,   # clientes assistidos por experts
    "Assist. J":     ...,   # clientes assistidos por juniores
    "Unidades":      ...,   # unidades vendidas totais
    "Vendas ($)":    ...,   # receita em USD
    "Custo RH ($)":  ...,   # custo de pessoal
    "Lucro Dia ($)": ...,   # lucro diário líquido
}
```

**Porquê:** O guia §5.2 pede explicitamente: *"Tabela do plano semanal com day, PR, X, J, assisted, units, sales, cost, profit"*. A tabela anterior estava incompleta.

A função `_enrich_plan()` recalcula as métricas usando `profit_logic.py` directamente, garantindo que os valores são consistentes com a fórmula do enunciado.

#### 7.3 Seção O2 completa

Adicionados:
- Comparação lado a lado: HC + Penalty Function vs. HC + Death Penalty
- Indicador visual de violação da restrição (verde/vermelho)
- Gráficos de convergência para ambos os algoritmos
- Tabela por loja

#### 7.4 Botões de exportação CSV

Adicionado `st.download_button` em cada tab de algoritmo e na secção O2 (guia §5.2).

---

## 8. DSS — Página 03: Otimização Multiobjetivo

**Ficheiro:** `dss_app/pages/03_Otimizacao_Multiobjetivo.py`

**Commit:** `77ac681`

### O que mudou

#### 8.1 Caminho da fronteira Pareto por loja

```python
# ANTES (caminho errado — ficheiro não existia)
path = os.path.join(opt_dir, "multiobjective", f"{store.lower()}_pareto.csv")

# DEPOIS (caminho correto)
path = os.path.join(opt_dir, "multiobjective", store.capitalize(), "pareto_front.csv")
```

**Porquê:** O script `run_optimization.py` cria uma subpasta por loja com capitalização (`Baltimore/`, `Lancaster/`, etc.) e o ficheiro chama-se `pareto_front.csv`. O caminho anterior não correspondia à estrutura real de ficheiros, causando `None` em todos os carregamentos.

#### 8.2 Moeda nos gráficos

Todos os `$` em títulos de eixos, hovertemplates, anotações e callouts foram corrigidos para USD (`$`), consistente com o enunciado.

---

## 9. Moeda nos Resultados

**Todos os ficheiros DSS**

### Sequência de mudanças (e porquê)

1. **Sessão anterior:** A moeda estava como `$` (dólares americanos).
2. **Durante esta sessão:** Foi alterada para `€` (euros) por engano — assumiu-se que os parâmetros W_s e custos eram em euros por serem uma empresa europeia.
3. **Revertida para `$`:** O enunciado especifica claramente *"60 USD"* e *"80 USD"* para os custos de pessoal. As lojas são americanas, os dados históricos de `Sales` estão em USD, e os parâmetros de custo são USD. A moeda correcta é **USD ($)**.

**Lição:** O enunciado é a única fonte de verdade para parâmetros — não o contexto geográfico da empresa.

---

## 10. Análise da Fronteira de Pareto

*Não é uma alteração de código, mas uma análise documentada aqui para referência do relatório.*

### Observação

A fronteira de Pareto do NSGA-II (O1 por loja, objetivos: maximizar lucro + minimizar staff) é **quase perfeitamente monotóna**: correlação staff/lucro ≈ 0.995–0.998 em todas as lojas.

```
Baltimore: correlação = 0.995, lucro marginal ≈ $47/funcionário
Lancaster: correlação = 0.996
Philadelphia: correlação = 0.998
Richmond: correlação = 0.996
```

### Explicação matemática

O lucro marginal constante de ~$47 por funcionário é exactamente o que o modelo prediz:

```
Expert (Baltimore, dia útil, PR≈0%):
  u_x = round(1.15 × 10 / ln(2)) = 17 unidades
  p_x = round(17 × 1.07) = 18 USD por cliente
  Receita: 7 clientes × $18 = $126
  Custo:   $80/dia
  Lucro líquido: $46/expert-day ≈ $47 médio (com fim de semana)
```

Enquanto houver clientes não atendidos, cada funcionário adicional gera lucro positivo. A Pareto não mostra zona de over-staffing porque o NSGA-II parou antes de atingir a capacidade total:
- Staff máximo Pareto (Baltimore): 83
- Staff mínimo para servir TODOS os clientes: 85

### Problema de convergência identificado

O SA (O1 monobjetivo) encontrou melhores soluções que o NSGA-II:

| Algoritmo | Lucro | Staff | Domina Pareto? |
|---|---|---|---|
| Simulated Annealing | **$2,588** | 82 | Sim — 3 pontos da Pareto |
| NSGA-II máximo | $2,557 | 83 | — |

O SA domina 3 pontos da fronteira de Pareto, o que significa que a fronteira é **sub-ótima**. Para melhorar: aumentar `n_max_gen` de 500 para 1000+ gerações.

### Para o relatório

> A fronteira de Pareto revela um trade-off de capacidade com retorno marginal constante (~$47/funcionário) até ao ponto de saturação da procura. O NSGA-II não convergiu completamente (500 gerações insuficientes para Baltimore/Philadelphia), com o SA a encontrar soluções que dominam 3 pontos da fronteira. O "cotovelo" real ocorre quando staff ≥ 85 (Baltimore), ponto a partir do qual over-staffing reduz o lucro — zona não explorada pelo algoritmo.

---

## Resumo Executivo das Alterações

| # | Ficheiro(s) | Tipo | Impacto |
|---|---|---|---|
| 1 | `profit_logic.py`, `profit_logic_knapsack.py` | Correcção de bug | Fórmula agora fiel ao enunciado |
| 2 | `profit_logic*.py`, scripts de otimização | Remoção de parâmetros | Sem PROFIT_SCALE nem ELASTICITY_K |
| 3 | `src/data/preparation.py` | Correcção de target | Target = Num_Customers (não Sales) |
| 4 | `main_pipeline.py` | Correcção de features | Sem data leakage no Cenário C |
| 5 | `dss_app/app.py` | Reescrita | Dashboard com dados reais |
| 6 | `dss_app/pages/01_Auditoria_Predicoes.py` | Correcção | Paths + unidades RMSE/MAE + export |
| 7 | `dss_app/pages/02_Otimizacao_Monobjetivo.py` | Reescrita | Todos os algoritmos + tabela guia §5.2 |
| 8 | `dss_app/pages/03_Otimizacao_Multiobjetivo.py` | Correcção | Path Pareto + moeda |
| 9 | Todos os ficheiros DSS | Correcção de moeda | USD ($) conforme enunciado |

---

*Documento gerado com assistência Claude Sonnet 4.6 — 2026-05-21*
