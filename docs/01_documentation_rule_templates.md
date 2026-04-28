# Relatório de Modelos Probabilísticos e de Regras

Este relatório descreve o comportamento dos modelos baseados em regras (Poisson, Gaussiano e Logístico) aplicados aos dados históricos das lojas.

## Loja: BALTIMORE

### 1. Modelo de Poisson (Chegada de Clientes)
O modelo de Poisson estima a taxa de chegada de clientes ($\lambda$) baseada no dia da semana e no mês. Comparado ao baseline de Lag-7 (mesmo dia da semana anterior), este modelo tende a suavizar variações bruscas.

| Dia | Lambda (Clientes/Dia) |
|---|---|
| Dom | 125.5 |
| Seg | 105.2 |
| Ter | 112.2 |
| Qua | 119.7 |
| Qui | 127.7 |
| Sex | 136.3 |
| Sab | 185.0 |

**Métricas de Desempenho (vs Lag-7):**
| modelo            | loja      |   MAE |   RMSE |   n_obs |
|:------------------|:----------|------:|-------:|--------:|
| Poisson (regras)  | baltimore | 89.24 |  111.3 |     549 |
| Lag-7 (historico) | baltimore | 36.49 |   78.7 |     549 |

### 2. Modelo Gaussiano (Variabilidade das Vendas)
Utiliza regressão linear para prever as vendas e modela o erro como uma distribuição normal (Gaussiana). Permite gerar cenários pessimistas, realistas e otimistas com base no desvio padrão ($\sigma$).

- **Média de Vendas ($\mu$):** EUR 49,345.23
- **Desvio Padrão ($\sigma$):** EUR 4,843.70
- **Coeficiente de Variação (CV):** 9.8%

| Dia | Pessimista | Realista | Otimista | IC 95% |
|---|---|---|---|---|
| Dom | 23,289 | 28,132 | 32,976 | [18,639 ; 37,626] |
| Seg | 30,343 | 35,186 | 40,030 | [25,693 ; 44,680] |
| Ter | 28,057 | 32,900 | 37,744 | [23,407 ; 42,394] |
| Qua | 25,309 | 30,153 | 34,997 | [20,660 ; 39,647] |
| Qui | 26,983 | 31,827 | 36,670 | [22,333 ; 41,320] |
| Sex | 30,849 | 35,692 | 40,536 | [26,199 ; 45,186] |
| Sab | 24,954 | 29,798 | 34,642 | [20,305 ; 39,291] |


### 3. Modelo Logístico (Conversão via Desconto)
Modela a probabilidade de um cliente realizar uma compra de alto valor (conversão) em função do desconto aplicado. Utiliza uma função sigmóide para mapear o impacto do desconto.

- **Qualidade do Modelo (AUC-ROC):** 0.810
- **Threshold de Venda/Cliente:** EUR 347.9

| Desconto (%) | Probabilidade de Conversão |
|---|---|
| 0.0% | 0.845 |
| 5.0% | 0.791 |
| 10.0% | 0.725 |
| 15.0% | 0.646 |
| 20.0% | 0.560 |
| 25.0% | 0.469 |
| 30.0% | 0.381 |

---

## Loja: LANCASTER

### 1. Modelo de Poisson (Chegada de Clientes)
O modelo de Poisson estima a taxa de chegada de clientes ($\lambda$) baseada no dia da semana e no mês. Comparado ao baseline de Lag-7 (mesmo dia da semana anterior), este modelo tende a suavizar variações bruscas.

| Dia | Lambda (Clientes/Dia) |
|---|---|
| Dom | 125.5 |
| Seg | 105.2 |
| Ter | 112.2 |
| Qua | 119.7 |
| Qui | 127.7 |
| Sex | 136.3 |
| Sab | 185.0 |

**Métricas de Desempenho (vs Lag-7):**
| modelo            | loja      |   MAE |   RMSE |   n_obs |
|:------------------|:----------|------:|-------:|--------:|
| Poisson (regras)  | lancaster | 89.24 |  111.3 |     549 |
| Lag-7 (historico) | lancaster | 36.49 |   78.7 |     549 |

### 2. Modelo Gaussiano (Variabilidade das Vendas)
Utiliza regressão linear para prever as vendas e modela o erro como uma distribuição normal (Gaussiana). Permite gerar cenários pessimistas, realistas e otimistas com base no desvio padrão ($\sigma$).

- **Média de Vendas ($\mu$):** EUR 49,345.23
- **Desvio Padrão ($\sigma$):** EUR 4,843.70
- **Coeficiente de Variação (CV):** 9.8%

| Dia | Pessimista | Realista | Otimista | IC 95% |
|---|---|---|---|---|
| Dom | 28,322 | 33,166 | 38,010 | [23,672 ; 42,659] |
| Seg | 31,605 | 36,449 | 41,293 | [26,955 ; 45,942] |
| Ter | 31,636 | 36,480 | 41,323 | [26,986 ; 45,973] |
| Qua | 31,395 | 36,238 | 41,082 | [26,745 ; 45,732] |
| Qui | 29,496 | 34,340 | 39,183 | [24,846 ; 43,833] |
| Sex | 33,698 | 38,541 | 43,385 | [29,048 ; 48,035] |
| Sab | 32,264 | 37,107 | 41,951 | [27,614 ; 46,601] |


### 3. Modelo Logístico (Conversão via Desconto)
Modela a probabilidade de um cliente realizar uma compra de alto valor (conversão) em função do desconto aplicado. Utiliza uma função sigmóide para mapear o impacto do desconto.

- **Qualidade do Modelo (AUC-ROC):** 0.810
- **Threshold de Venda/Cliente:** EUR 347.9

| Desconto (%) | Probabilidade de Conversão |
|---|---|
| 0.0% | 0.845 |
| 5.0% | 0.791 |
| 10.0% | 0.725 |
| 15.0% | 0.646 |
| 20.0% | 0.560 |
| 25.0% | 0.469 |
| 30.0% | 0.381 |

---

## Loja: PHILADELPHIA

### 1. Modelo de Poisson (Chegada de Clientes)
O modelo de Poisson estima a taxa de chegada de clientes ($\lambda$) baseada no dia da semana e no mês. Comparado ao baseline de Lag-7 (mesmo dia da semana anterior), este modelo tende a suavizar variações bruscas.

| Dia | Lambda (Clientes/Dia) |
|---|---|
| Dom | 125.5 |
| Seg | 105.2 |
| Ter | 112.2 |
| Qua | 119.7 |
| Qui | 127.7 |
| Sex | 136.3 |
| Sab | 185.0 |

**Métricas de Desempenho (vs Lag-7):**
| modelo            | loja         |   MAE |   RMSE |   n_obs |
|:------------------|:-------------|------:|-------:|--------:|
| Poisson (regras)  | philadelphia | 89.24 |  111.3 |     549 |
| Lag-7 (historico) | philadelphia | 36.49 |   78.7 |     549 |

### 2. Modelo Gaussiano (Variabilidade das Vendas)
Utiliza regressão linear para prever as vendas e modela o erro como uma distribuição normal (Gaussiana). Permite gerar cenários pessimistas, realistas e otimistas com base no desvio padrão ($\sigma$).

- **Média de Vendas ($\mu$):** EUR 49,345.23
- **Desvio Padrão ($\sigma$):** EUR 4,843.70
- **Coeficiente de Variação (CV):** 9.8%

| Dia | Pessimista | Realista | Otimista | IC 95% |
|---|---|---|---|---|
| Dom | 53,437 | 58,281 | 63,124 | [48,787 ; 67,774] |
| Seg | 53,890 | 58,733 | 63,577 | [49,240 ; 68,227] |
| Ter | 56,442 | 61,286 | 66,129 | [51,792 ; 70,779] |
| Qua | 52,436 | 57,279 | 62,123 | [47,786 ; 66,773] |
| Qui | 55,869 | 60,713 | 65,557 | [51,219 ; 70,206] |
| Sex | 61,048 | 65,892 | 70,735 | [56,398 ; 75,385] |
| Sab | 58,324 | 63,167 | 68,011 | [53,674 ; 72,661] |


### 3. Modelo Logístico (Conversão via Desconto)
Modela a probabilidade de um cliente realizar uma compra de alto valor (conversão) em função do desconto aplicado. Utiliza uma função sigmóide para mapear o impacto do desconto.

- **Qualidade do Modelo (AUC-ROC):** 0.810
- **Threshold de Venda/Cliente:** EUR 347.9

| Desconto (%) | Probabilidade de Conversão |
|---|---|
| 0.0% | 0.845 |
| 5.0% | 0.791 |
| 10.0% | 0.725 |
| 15.0% | 0.646 |
| 20.0% | 0.560 |
| 25.0% | 0.469 |
| 30.0% | 0.381 |

---

## Loja: RICHMOND

### 1. Modelo de Poisson (Chegada de Clientes)
O modelo de Poisson estima a taxa de chegada de clientes ($\lambda$) baseada no dia da semana e no mês. Comparado ao baseline de Lag-7 (mesmo dia da semana anterior), este modelo tende a suavizar variações bruscas.

| Dia | Lambda (Clientes/Dia) |
|---|---|
| Dom | 125.5 |
| Seg | 105.2 |
| Ter | 112.2 |
| Qua | 119.7 |
| Qui | 127.7 |
| Sex | 136.3 |
| Sab | 185.0 |

**Métricas de Desempenho (vs Lag-7):**
| modelo            | loja     |   MAE |   RMSE |   n_obs |
|:------------------|:---------|------:|-------:|--------:|
| Poisson (regras)  | richmond | 89.24 |  111.3 |     549 |
| Lag-7 (historico) | richmond | 36.49 |   78.7 |     549 |

### 2. Modelo Gaussiano (Variabilidade das Vendas)
Utiliza regressão linear para prever as vendas e modela o erro como uma distribuição normal (Gaussiana). Permite gerar cenários pessimistas, realistas e otimistas com base no desvio padrão ($\sigma$).

- **Média de Vendas ($\mu$):** EUR 49,345.23
- **Desvio Padrão ($\sigma$):** EUR 4,843.70
- **Coeficiente de Variação (CV):** 9.8%

| Dia | Pessimista | Realista | Otimista | IC 95% |
|---|---|---|---|---|
| Dom | 15,835 | 20,679 | 25,523 | [11,185 ; 30,172] |
| Seg | 20,924 | 25,768 | 30,612 | [16,275 ; 35,262] |
| Ter | 21,208 | 26,052 | 30,895 | [16,558 ; 35,545] |
| Qua | 20,995 | 25,839 | 30,683 | [16,346 ; 35,333] |
| Qui | 19,072 | 23,915 | 28,759 | [14,422 ; 33,409] |
| Sex | 23,233 | 28,077 | 32,921 | [18,584 ; 37,571] |
| Sab | 19,863 | 24,707 | 29,550 | [15,213 ; 34,200] |


### 3. Modelo Logístico (Conversão via Desconto)
Modela a probabilidade de um cliente realizar uma compra de alto valor (conversão) em função do desconto aplicado. Utiliza uma função sigmóide para mapear o impacto do desconto.

- **Qualidade do Modelo (AUC-ROC):** 0.810
- **Threshold de Venda/Cliente:** EUR 347.9

| Desconto (%) | Probabilidade de Conversão |
|---|---|
| 0.0% | 0.845 |
| 5.0% | 0.791 |
| 10.0% | 0.725 |
| 15.0% | 0.646 |
| 20.0% | 0.560 |
| 25.0% | 0.469 |
| 30.0% | 0.381 |

---
