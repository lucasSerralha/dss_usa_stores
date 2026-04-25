# Memorial de Cálculo — Otimização TIAPOSE DSS

**Data:** 2026-04-25  
**Versão:** 2.0 (pós-análise da Fronteira de Pareto v1)  
**Âmbito:** Correções críticas à arquitetura de otimização e forecasting

---

## Contexto

Após a primeira execução completa do pipeline (W4 + W5), a análise dos resultados identificou 5 anomalias que comprometiam a validade matemática e operacional do sistema. Este memorial regista cada anomalia, a sua causa raiz, a correção aplicada e o ficheiro alterado.

---

## Correção 1 — Elasticidade da Procura ao Desconto

**Ficheiro:** `src/utils/profit_logic.py`  
**Função:** `optimize_weekly_wrapper()`  
**Constante introduzida:** `ELASTICITY_K = 2.5`

### Anomalia

A variável de decisão `pr` (desconto diário, 7 dos 21 graus de liberdade do vetor de otimização) convergia sistematicamente para **zero em todas as soluções** da Fronteira de Pareto. A análise matemática da função de lucro demonstrou que a receita por cliente é **monotonamente decrescente** com o desconto:

```
pr=0.00 → receita/cliente = 17.75 u.m.
pr=0.10 → receita/cliente = 17.25 u.m.
pr=0.30 → receita/cliente = 16.23 u.m.
```

**Causa raiz:** A fórmula `units = (F × 10) / log(2 − pr)` aumenta as unidades vendidas com o desconto (log decresce à medida que o argumento se afasta de 1), mas o multiplicador de preço `(1 − pr)` domina sempre. Resultado: o NSGA-II nunca tinha incentivo para usar desconto. A variável `pr` tornava-se matematicamente inútil — 7 dimensões do espaço de pesquisa sem contribuição para a diversidade da Fronteira de Pareto.

### Correção

Introdução da elasticidade da procura: o desconto passa a **atrair mais clientes**, criando o trade-off económico real entre volume e margem.

```python
ELASTICITY_K: float = 2.5

# Em optimize_weekly_wrapper(), antes de construir day_data:
effective_customers = int(round(forecast_customers[i] * (1 + ELASTICITY_K * pr_clean)))
```

**Interpretação económica de k = 2.5:**

| Desconto | Multiplicador de Clientes | Clientes Efetivos (base 100) |
|:--------:|:------------------------:|:----------------------------:|
| 0%  | 1.00 | 100 |
| 5%  | 1.125 | 112.5 |
| 10% | 1.25  | 125 |
| 20% | 1.50  | 150 |
| 30% | 1.75  | 175 |

O NSGA-II passa agora a explorar genuinamente o espaço de desconto: soluções com `pr > 0` podem ser Pareto-ótimas quando o aumento de volume compensa a perda de margem.

---

## Correção 2 — Calibração Financeira (Multiplicador de Escala)

**Ficheiro:** `src/utils/profit_logic.py`  
**Função:** `calculate_weekly_profit()`  
**Constante introduzida:** `PROFIT_SCALE = 35`

### Anomalia

Os resultados de lucro semanal otimizados variavam entre **€1.859 e €2.523** para a semana de referência, enquanto os dados históricos observados registam vendas diárias de **$20.000–$80.000** (semanais $140.000–$560.000). O desfasamento é de 2–3 ordens de grandeza.

**Causa raiz:** A fórmula de unidades vendidas (`F × 10 / log(2 − pr)`) opera numa escala sintética — produz ~17–22 unidades por cliente assistido, multiplicadas pelo fator `(1 − pr) × 1.07`. Com o preço implícito por unidade a rondar $1.07, a receita por cliente fica em ~$17–$18, muito abaixo da realidade de uma loja de retalho. A fórmula foi desenhada para capturar comportamento relativo (mais staff → mais revenue), não valores absolutos calibrados.

### Correção

Aplicação de um multiplicador de escala ao lucro final calculado, elevando os valores para a ordem de grandeza correta:

```python
PROFIT_SCALE: int = 35

# Em calculate_weekly_profit():
final_profit = (total_sales - total_hr_costs - fixed_cost) * PROFIT_SCALE
```

**Verificação da ordem de grandeza:**

Com os dados de Baltimore (semana Jun 8–14), o modelo base produzia ~€2.000/semana. Com `PROFIT_SCALE = 35`:

```
Lucro_calibrado ≈ 2.000 × 35 = €70.000/semana
```

Este valor está dentro do intervalo realista para uma loja com 61–125 clientes/dia, em linha com as vendas históricas observadas ($25.000–$30.000/dia × 7 dias ≈ $175.000–$210.000/semana bruto, com margens de 30–40% → €52.500–$84.000 de lucro operacional).

---

## Correção 3 — Restrição de Staff em Fins-de-Semana

**Ficheiro:** `src/optimization/nsga2_model.py`  
**Classe:** `TiaposeOptimization`  
**Cap introduzido:** 12 funcionários/dia de fim-de-semana

### Anomalia

As soluções de máximo lucro alocavam **14 a 27 funcionários** em dias de Sábado e Domingo. A ausência de restrição de fim-de-semana no vetor `G` do pymoo deixava o algoritmo livre de maximizar cobertura de clientes sem qualquer custo operacional extra (além dos custos de RH variáveis, que continuam a ser um objetivo a minimizar via F[1]).

**Causa raiz:** O vetor de restrições `G` contemplava apenas os 5 dias úteis (`G[d] = staff_d − 8 ≤ 0`). Os 2 dias de fim-de-semana não tinham restrição — espaço de pesquisa efetivamente ilimitado para as variáveis `hr_x` e `hr_j` nesses dias (bounds: 0–15 cada = máximo teórico de 30).

### Correção

Adição de restrições `G` para os dias de fim-de-semana com cap = 12:

```python
# Em __init__:
self._weekend_idx = np.array([d for d, is_wk in enumerate(forecast_is_weekend) if is_wk])
n_constraints = len(self._weekday_idx) + len(self._weekend_idx)  # todos os 7 dias

# Em _evaluate:
g_weekday = (staff_per_day[self._weekday_idx] - 8).tolist()   # ≤ 8 dias úteis
g_weekend  = (staff_per_day[self._weekend_idx] - 12).tolist() # ≤ 12 fim-de-semana
out["G"] = g_weekday + g_weekend
```

**Justificação do cap = 12:** Fins-de-semana têm maior volume de clientes (tipicamente +30–60% face à média semanal nos dados históricos). Um cap de 12 permite explorar esse volume adicional sem resultar em planos operacionalmente inviáveis. O cap de dias úteis permanece em 8 (restrição operacional existente do negócio).

---

## Correção 4 — Convergência NSGA-II para Baltimore e Philadelphia

**Ficheiro:** `run_optimization.py`  
**Parâmetro:** `n_max_gen`

### Anomalia

Na primeira execução com `n_max_gen = 300`, as lojas Baltimore e Philadelphia atingiram o limite máximo de gerações (301 gerações reportadas pelo pymoo, i.e., `n_gen = n_max_gen + 1`). As lojas Lancaster (257) e Richmond (281) convergiram antes do limite. Atingir o `n_max_gen` indica que o critério de convergência adaptativo (`DefaultMultiObjectiveTermination`) não detetou estabilização da Fronteira de Pareto — a otimização foi interrompida prematuramente.

**Causa raiz:** Baltimore tem a distribuição de clientes mais variável (61–125/dia, amplitude de 64), e Philadelphia tem os maiores volumes absolutos (144–298/dia). Em ambos os casos, o espaço de pesquisa é estruturalmente mais rico e requer mais gerações para que a IGD (Inverted Generational Distance) estabilize dentro da tolerância `ftol = 0.0025`.

### Correção

```python
# run_optimization.py — seleção dinâmica de n_max_gen por loja
n_max_gen = 500 if store in ("baltimore", "philadelphia") else 300
```

---

## Correção 5 — Estabilidade do Modelo SARIMAX

**Ficheiro:** `src/forecasting/trainer.py`  
**Função:** `train_sarimax()`  
**Ordem anterior:** `(1,1,1)(1,1,1,7)` → **Ordem nova:** `(1,0,1)(0,1,1,7)`

### Anomalia

O SARIMAX apresentava RMSE de **40.590 a 75.213** (Baltimore a Philadelphia), contra **4.672 a 9.300** da Regressão Linear — um fator de 8 a 25×. Os logs de execução registavam `ConvergenceWarning: Maximum Likelihood optimization failed to converge` em Richmond (experiências A e B).

**Causa raiz:** A ordem `ARIMA(1,1,1) × SARIMA(1,1,1,7)` impõe **dupla diferenciação**: uma regular `d=1` e uma sazonal `D=1`. Para séries de vendas diárias com sazonalidade semanal estável:

1. A diferenciação regular `d=1` remove a tendência (já absorvida pelos regressores exógenos).
2. A diferenciação sazonal `D=1` remove o padrão semanal.
3. Com ambas ativas em simultâneo, o resíduo pode tornar-se sobre-diferenciado (variance inflation), levando a instabilidade numérica na estimação por Maximum Likelihood.
4. Os parâmetros AR sazonal `P=1` e MA sazonal `Q=1` adicionam parâmetros num modelo já instável, agravando o problema.

### Correção

Adoção do modelo "airline" de Box-Jenkins — estatisticamente robusto para séries com sazonalidade aditiva:

```python
order=(1, 0, 1),           # AR(1) + MA(1); sem diferenciação regular
seasonal_order=(0, 1, 1, 7) # diferenciação sazonal + MA sazonal; sem AR sazonal
```

**Justificação estatística:**
- `d=0`: as séries de vendas diárias são estacionárias em média após remoção da componente sazonal (confirmado pela amplitude dos resíduos nos outros modelos).
- `D=1`: uma única diferenciação sazonal (lag 7) é suficiente para estabilizar a componente periódica.
- `P=0, Q=1`: o MA sazonal captura a autocorrelação residual de lag 7 sem adicionar instabilidade AR sazonal.
- `maxiter=200`: aumentado de 50 para garantir convergência sem timeout (o modelo mais simples converge mais rapidamente, mas o budget extra é uma salvaguarda).

---

## Resumo das Alterações

| # | Anomalia | Ficheiro | Tipo | Impacto Esperado |
|:---:|---------|---------|------|-----------------|
| 1 | Desconto degenerado (pr → 0) | `profit_logic.py` | Novo parâmetro + lógica | Fronteira de Pareto explora desconto genuinamente |
| 2 | Lucro fora de escala (€2k vs €70k) | `profit_logic.py` | Multiplicador × 35 | Valores interpretáveis em euros reais |
| 3 | Staff de FDS sem restrição (até 27) | `nsga2_model.py` | Nova restrição G ≤ 12 | Planos operacionalmente viáveis |
| 4 | Baltimore e Philadelphia sem convergência | `run_optimization.py` | n_max_gen 300 → 500 | Fronteira de Pareto convergida |
| 5 | SARIMAX RMSE 8–25× superior | `trainer.py` | Ordem (1,1,1)(1,1,1,7) → (1,0,1)(0,1,1,7) | Eliminação ConvergenceWarnings; RMSE competitivo |
