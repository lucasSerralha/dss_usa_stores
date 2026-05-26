# CHANGELOG — Alterações Assistidas por IA

> Registo de todas as modificações geradas e aplicadas com assistência do modelo Claude Sonnet 4.6.
> Cada entrada documenta o que foi alterado, porquê, e a referência ao enunciado/guia.

---

## [2026-05-26] — Sessão de correção pré-apresentação

### 1. `src/utils/profit_logic.py` — Arredondamento por cliente (não por agregado)

**Ficheiro:** [src/utils/profit_logic.py](../src/utils/profit_logic.py)

**Problema:** A fórmula original arredondava o total de vendas do dia (`round(units_total * (1-pr) * 1.07)`), quando o enunciado especifica `P = round(U × (1−PR) × 1.07)` **por cliente** assistido.

**Antes:**
```python
sales_x = round(units_x * (1 - pr) * 1.07)
sales_j = round(units_j * (1 - pr) * 1.07)
```

**Depois:**
```python
p_x = round(u_per_x * (1 - pr) * 1.07) if assisted_x > 0 else 0
p_j = round(u_per_j * (1 - pr) * 1.07) if assisted_j > 0 else 0
sales_x = assisted_x * p_x
sales_j = assisted_j * p_j
```

**Justificação:** Enunciado §5.1 — fórmula de receita opera ao nível do cliente, não do dia.

---

### 2. `dss_app/pages/01_Auditoria_Predicoes.py` — Paths e unidades de RMSE

**Ficheiro:** [dss_app/pages/01_Auditoria_Predicoes.py](../dss_app/pages/01_Auditoria_Predicoes.py)

**Problemas corrigidos:**
- Path desatualizado: `results/` → `results_v2/00_Master_Summary/fidelity_experimentation_report.csv`
- RMSE exibia `$` (moeda) quando o target passou a ser `Num_Customers` (pessoas)
- Botão de exportação CSV em falta

**Antes (RMSE):**
```html
<span class="podium-metric-value">$123.45</span>
```

**Depois (RMSE):**
```html
<span class="podium-metric-value">123.45 clientes</span>
```

**Justificação:** O target da previsão é `Num_Customers` (escala ≈ 60–300), não `Sales` (escala ≈ $15k–$80k).

---

### 3. `dss_app/pages/03_Otimizacao_Monobjetivo.py` — Tabelas enriquecidas + moeda USD

**Ficheiro:** [dss_app/pages/03_Otimizacao_Monobjetivo.py](../dss_app/pages/03_Otimizacao_Monobjetivo.py)

**Problemas corrigidos:**
- Moeda alterada de `€` para `$` (guia especifica "60 USD", "80 USD" explicitamente)
- Adicionadas tabelas enriquecidas por dia (§5.2 do guia): PR, X, J, Assistidos, Unidades, Vendas, Custo RH, Lucro Dia
- Função `_enrich_plan()` integra `calculate_daily_metrics()` com os planos de otimização

**Função adicionada:**
```python
def _enrich_plan(df_plan, store):
    rows = []
    for _, r in df_plan.iterrows():
        m = calculate_daily_metrics(store, is_wk, cust, pr, hr_x, hr_j)
        rows.append({
            "Lucro Dia ($)": m["sales_x"] + m["sales_j"] - m["cost_x"] - m["cost_j"],
            ...
        })
    return pd.DataFrame(rows)
```

**Justificação:** Guia §5.2 define o plano tático com detalhe diário; enunciado §3 especifica USD.

---

### 4. `dss_app/pages/04_Otimizacao_Multiobjetivo.py` — Bug crítico: Seleção de plano negativo

**Ficheiro:** [dss_app/pages/04_Otimizacao_Multiobjetivo.py](../dss_app/pages/04_Otimizacao_Multiobjetivo.py)

**Problema:** A função `selecionar_por_w()` incluía soluções com lucro negativo no processo de escalarização. Para valores baixos de `w` (ex.: w=0.25, "Conservador"), a solução de staff mínimo (staff=21, lucro=−$23,310) obtinha score máximo porque `(1 − staff_norm) = 1.0` dominava.

**Exemplo do bug (U-NSGA-III, w=0.5):**
```
ANTES: lucro selecionado = −$23,310  ← ERRADO (plano com prejuízo)
DEPOIS: lucro selecionado = +$24,430 ← CORRETO (plano lucrativo)
```

**Antes:**
```python
def selecionar_por_w(df, w):
    lucro = df["lucro_total"].values.astype(float)  # inclui negativos
    staff = df["staff_total"].values.astype(float)
    lucro_n = (lucro - lucro.min()) / (lucro.max() - lucro.min() + 1e-9)
    staff_n = (staff - staff.min()) / (staff.max() - staff.min() + 1e-9)
    scores  = w * lucro_n + (1 - w) * (1 - staff_n)
    return int(np.argmax(scores))
```

**Depois:**
```python
def selecionar_por_w(df, w):
    # Filtrar apenas soluções viáveis (lucro > 0)
    df_viable = df[df["lucro_total"] > 0]
    if df_viable.empty:
        df_viable = df  # fallback
    lucro = df_viable["lucro_total"].values.astype(float)
    staff = df_viable["staff_total"].values.astype(float)
    lucro_n = (lucro - lucro.min()) / (lucro.max() - lucro.min() + 1e-9)
    staff_n = (staff - staff.min()) / (staff.max() - staff.min() + 1e-9)
    scores  = w * lucro_n + (1 - w) * (1 - staff_n)
    best_in_viable = int(np.argmax(scores))
    return int(df_viable.index[best_in_viable])
```

**Impacto após fix (U-NSGA-III, fronteira com 25/64 soluções lucrativas):**

| w    | Perfil      | Lucro (antes) | Lucro (depois) |
|------|-------------|---------------|----------------|
| 0.00 | Conservador | −$23,310      | **$875**       |
| 0.25 | Conservador | −$23,310      | **$875**       |
| 0.50 | Equilibrado | −$23,310      | **$24,430**    |
| 0.75 | Agressivo   | $37,135       | $37,135        |
| 1.00 | Agressivo   | $37,135       | $37,135        |

**Justificação:** Um plano de decisão com lucro negativo é inviável para apresentação a gestores. A restrição de viabilidade económica (lucro > 0) é implícita na formulação do problema.

---

### 5. `dss_app/app.py` — Homepage com KPIs reais

**Ficheiro:** [dss_app/app.py](../dss_app/app.py)

**Problemas corrigidos:**
- KPIs na página inicial carregavam valores fixos/mock
- Moeda alterada para `$` (USD)
- Paths atualizados para `results_v2/`

---

## Dados e Paths

| Path | Conteúdo | Status |
|------|----------|--------|
| `results/03_Optimization_Report/` | O3 com fórmula original (PROFIT_SCALE=35) | Usado pela página 04 |
| `results_v2/03_Optimization/` | O3 com fórmula corrigida (sem PROFIT_SCALE) | Disponível mas não usado em O3 |
| `results_v2/00_Master_Summary/` | Leaderboard de previsão (Num_Customers) | Usado pela página 01 |
| `results_v2/03_Optimization/individual/` | SA / HC por loja | Usado pela página 03 |

---

*Gerado automaticamente. Última atualização: 2026-05-26.*
