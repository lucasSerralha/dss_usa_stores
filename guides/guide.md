# Guia Unificado de Desenvolvimento — Intelligent Decision Support System for USA Stores

> Documento unificado a partir de `project.pdf` (enunciado oficial) e `guia-projeto.pdf` (sugestões técnicas v1.0). Serve como **guia de desenvolvimento** e como **checklist de apresentação final**, garantindo que o projeto explore todas as opções relevantes e justifique as escolhas dos melhores modelos de previsão e otimização.

---

## 1. Visão Geral do Projeto

**Objetivo macro:** desenvolver um Sistema de Apoio à Decisão (DSS) inteligente para 4 lojas físicas dos EUA (Baltimore MD, Lancaster PA, Philadelphia PA, Richmond VA) combinando uma componente de **previsão (forecasting)** do número diário de clientes e uma componente de **otimização** do plano semanal de Recursos Humanos (J/X) e promoções (PR).

**Peso na nota:** 75% da Unidade Curricular.

**Datas-chave:**
- Apresentação MECD: 2025-05-25
- Apresentação MEGSI: 2026-05-27
- Resumos semanais obrigatórios desde 2026-03-16
- Submissão: ZIP único (relatório PDF obrigatório, slides opcional, código obrigatório)
- Vídeo demo no YouTube (≤ 5 min, com narração, link no relatório)

---

## 2. Dados

Quatro ficheiros CSV (`baltimore.csv`, `lancaster.csv`, `philadelphia.csv`, `richmond.csv`), cada um com colunas diárias:

| Coluna | Tipo | Descrição |
|---|---|---|
| `Date` | exógena | dia do ano |
| `Num_Employees` | exógena | número de empregados (= J + X) |
| `Num_Customers` | **alvo** | clientes diários (variável a prever) |
| `Pct_On_Sale` | exógena | % de produtos com desconto (= PR) |
| `TouristEvent` | exógena | No/Yes |
| `Sales` | endógena | nº diário de vendas |

**Tarefas iniciais obrigatórias:**
- [ ] Verificar `NA`s (existe pelo menos 1 em Baltimore `Pct_On_Sale`) e tratá-los justificadamente.
- [ ] Análise exploratória (sumários, distribuições, sazonalidade semanal, decomposições STL/seasonal_decompose).
- [ ] Identificar o último dia de cada série (a previsão será sobre as semanas finais).
- [ ] Documentar as estatísticas descritivas no relatório.

---

## 3. Componente A — Forecasting (Previsão)

**Meta:** prever `Num_Customers` para cada loja, horizonte H ∈ [1, 7] dias (multi-step ahead).

### 3.1 Requisitos formais
- [ ] Cada membro do grupo implementa **pelo menos um método de previsão diferente**.
- [ ] Treino em dados antigos / Teste em dados recentes.
- [ ] Métricas de qualidade definidas pelo grupo (recomendado: NMAE, RMSE, R²; opcional: MAPE, sMAPE).
- [ ] Justificação clara da metodologia para escolher o "melhor" modelo quando se usa mais de uma métrica.

### 3.2 Métodos a explorar (quantos mais, melhor)

#### Univariados (começar por aqui)
- [ ] **Naive / Seasonal Naive (S=7)** — *baseline obrigatório* (representa o que a empresa faria sem IA).
- [ ] **Holt-Winters / ETS** (pacote `forecast`).
- [ ] **ARIMA / auto.arima** (pacote `forecast`).
- [ ] **TBATS** (sazonalidades múltiplas).
- [ ] **Prophet** (opcional, bom em sazonalidades).
- [ ] **Machine Learning univariado** com `rminer::CasesSeries` + `lforecast()` (lags como atributos): MLP, SVM, Random Forest, XGBoost.

#### Multivariados (explorar se houver tempo — distingue um "bom" de um "excelente" projeto)
- [ ] **VAR** (Vector Autoregression).
- [ ] **ARIMAX / SARIMAX** com regressores exógenos (`Num_Employees`, `Pct_On_Sale`, `TouristEvent`, dia da semana).
- [ ] **Multivariate ML** (rminer multivariado, ou `caret`/`mlr3` em R; `sktime`/`darts` em Python).
- [ ] Testar **cenários preditivos** distintos: variáveis da mesma loja vs. variáveis de outras lojas (cross-store features).

### 3.3 Avaliação em duas fases

**Fase I — sanidade (prototipagem rápida):**
- [ ] Treinar com tudo exceto a última semana (7 dias).
- [ ] Gráfico predicted vs. desired.
- [ ] Calcular NMAE, RMSE, R² para cada modelo.
- [ ] Verificar que faz sentido (ordem de grandeza, captura de sazonalidade semanal).

**Fase II — avaliação robusta (obrigatória para boa nota):**
- [ ] Backtesting com **>10 iterações** (recomendado 15–20).
- [ ] Escolher entre **Growing Window** OU **Rolling Window** (justificar).
- [ ] Agregação dos erros via **mediana** (robusta) ou **média** (sensível a outliers) — justificar a escolha.
- [ ] Mesmas condições de teste para todos os métodos (períodos comparáveis).
- [ ] Gráfico único predicted vs. actual para toda a zona de teste, se não houver overlap.
- [ ] **Guardar previsões num CSV** para alimentar a Fase II da otimização (poupa retreino).

### 3.4 Análise final de qualidade
- [ ] Identificar o melhor método **univariado** por série e overall.
- [ ] Identificar o melhor método **multivariado**.
- [ ] Identificar o **melhor método global**.
- [ ] Comparar com **Seasonal Naive (S=7)** e reportar **% de melhoria**.
- [ ] Tabela-resumo com método × loja × métrica (mediana/média de N iterações).

### 3.5 Checklist de apresentação (Forecasting)
- [ ] Mostrei pelo menos 1 baseline (Seasonal Naive)?
- [ ] Mostrei pelo menos 4 famílias de métodos diferentes (estatístico, ML, multivariado, baseline)?
- [ ] Justifiquei a escolha do método final com métrica e teste robusto?
- [ ] Tenho gráficos predicted vs. actual?
- [ ] Tenho tabela comparativa final?
- [ ] Tenho o ganho % vs. Seasonal Naive reportado?

---

## 4. Componente B — Optimization (Otimização)

**Meta:** definir o plano semanal (próximos 7 dias) para todas as lojas. Por loja `s` e dia `d`, o plano contém: `J_{s,d}`, `X_{s,d}` (inteiros) e `PR_{s,d}` ∈ [0, 0.30].

**Dimensão da solução:** 4 lojas × 7 dias × 3 parâmetros = **84 variáveis**.

### 4.1 Modelo matemático (resumo)
- Custo diário Júnior: 60 USD (semana) / 70 USD (fim de semana).
- Custo diário eXpert: 80 USD (semana) / 95 USD (fim de semana).
- Clientes assistidos: `A_{s,d} = min(7·X + 6·J, C_{s,d})` — primeiro X, depois J.
- Unidades vendidas/cliente: `U = round(F · 10 / ln(2 - PR))`, F = F_J ou F_X.
- Lucro por cliente: `P = round(U · (1 - PR) · 1.07)`.
- Lucro diário: `R_{s,d} = sum(P) - J·custo_J - X·custo_X`.
- Lucro semanal: `R_s = sum(R_{s,d}) - W_s` (custo fixo semanal).

**Parâmetros por loja:**
| Loja | F_J | F_X | W_s |
|---|---|---|---|
| Baltimore | 1.00 | 1.15 | 700 |
| Lancaster | 1.05 | 1.20 | 730 |
| Philadelphia | 1.10 | 1.15 | 760 |
| Richmond | 1.15 | 1.25 | 800 |

### 4.2 Três objetivos (todos devem ser tratados)

- [ ] **O1 — Maximizar lucro total** (R_s) da semana planeada. Pode ser resolvido independentemente por loja.
- [ ] **O2 — Maximizar O1 com restrição rígida**: ≤ 10 000 unidades vendidas para as 4 lojas em conjunto. **Não separável** entre lojas.
- [ ] **O3 — Maximizar O2 e Minimizar HR total** (multi-objetivo). **Não separável**.

> **Nota crítica do enunciado:** usar o mesmo conjunto de parâmetros otimizáveis para os três objetivos poupa código.

### 4.3 Função de avaliação (passo crítico)
- [ ] Implementar `evaluation(s)` que recebe vetor de 84 valores e devolve lucro / vector de objetivos.
- [ ] Validar com os exemplos de Baltimore (slide 17, lucro = 146) e Philadelphia (slide 18, lucro = 1728).
- [ ] Aplicar `round()` dentro da função para J e X (otimizadores reais geram floats).
- [ ] Após otimização, **arredondar novamente a melhor solução** antes de apresentar.

**Tratamento de soluções inválidas (O2):**
- [ ] **Death penalty** — devolver `-Inf` (ou penalização enorme) — simples, mas pode paralisar a busca.
- [ ] **Repair function** — corrigir a solução; se for estocástica, guardar em `BEST <<- s` (R) ou variável global em Python para não perder a melhor reparada.

### 4.4 Representação e bounds
- [ ] Vetor numérico de 84 posições (compatível com a maior parte dos pacotes de otimização).
- [ ] `lower = rep(0, 84)`.
- [ ] `upper = calc_upper(s, n_clients)` ajustado às previsões da semana (não faz sentido J+X muito acima dos clientes previstos; PR ≤ 0.30).
- [ ] Se usar representação binária: calcular bits via `log2(n)` e usar `bin2int` (pacote `adana`).

### 4.5 Métodos de otimização a explorar
> **Cada membro do grupo deve implementar, configurar e experimentar pelo menos um método.**

#### Representação real
- [ ] **Monte Carlo (blind search)** — baseline simples.
- [ ] **Hill Climbing** — busca local, perturbação multiplicativa.
- [ ] **Simulated Annealing** (`optim(method="SANN")`) — preencher `gr=` para garantir bounds; estudar `temp` inicial.
- [ ] **Genetic Algorithm** (`GA` package em R, `DEAP` em Python).
- [ ] **Differential Evolution** (`DEoptim`).
- [ ] **Particle Swarm Optimization** (`pso`, `psoptim`).

#### Representação binária / inteira (opcional, valoriza)
- [ ] **Tabu Search** (`tabuSearch::tabuSearch`).
- [ ] **GA binário** (`genalg::rbga.bin`).

> **NÃO usar** full search nem grid search (computacionalmente proibitivo).

### 4.6 Análise de convergência (opcional, mas valoriza muito)
- [ ] Gráfico de fitness × iteração para cada método.
- [ ] Comparar diferentes parametrizações (ex.: `temp` no SANN, `popSize`/`generations` em GA/DE/PSO).
- [ ] Verificar se estagna cedo ou continua a melhorar.

### 4.7 Otimização multi-objetivo (O3)

**Abordagem 1 — escalarização ponderada:**
- [ ] `f = -w · f1_norm + K · (1-w) · f2_norm` com w=0.7 e K constante de normalização.
- [ ] Justificar pesos e normalização (escalas de profit e HR são muito diferentes).

**Abordagem 2 — multi-objetivo verdadeiro (preferível):**
- [ ] Função `eval2()` devolve vetor `(profit, -HR)` ou `(-profit, HR)`.
- [ ] **NSGA-II** (`mco::nsga2` em R, `pymoo` em Python).
- [ ] Apresentar **curva/frente de Pareto**.
- [ ] Discutir trade-offs (pontos extremos vs. pontos joelho).

### 4.8 Que semana otimizar — duas fases

**Fase I (validar implementação):**
- [ ] Otimizar a última semana com **valores reais** de clientes (ground truth).
- [ ] Testar todos os métodos isolados, garantir que correm e produzem soluções coerentes.

**Fase II (uso real do sistema):**
- [ ] Otimizar **com previsões** (as do melhor modelo da Fase II do forecasting).
- [ ] Correr a otimização para **cada uma das semanas testadas no backtesting** (≥10 runs).
- [ ] Agregar lucros via mediana/média.
- [ ] Tabela comparativa final: método × O1/O2/O3 × loja (e total).
- [ ] **Carregar previsões de CSV** em vez de retreinar — eficiência.

### 4.9 Checklist de apresentação (Optimization)
- [ ] Implementei pelo menos 4 métodos de otimização distintos (incluindo Monte Carlo como baseline)?
- [ ] Validei a `evaluation()` com os exemplos do enunciado (Baltimore=146, Philadelphia=1728)?
- [ ] Resolvi os 3 objetivos (O1, O2, O3)?
- [ ] Tenho análise de convergência?
- [ ] Tenho frente de Pareto para O3?
- [ ] Comparei otimização com **valores reais vs. previsões**?
- [ ] Justifiquei o método "melhor" para cada objetivo?
- [ ] Estudei o efeito de mudar parâmetros (popSize, temp, gerações)?

---

## 5. Componente C — Decision Support System (DSS)

> Construir **depois** de forecasting e otimização estarem maduros. Tipicamente 1–2 semanas.

### 5.1 Requisitos
- [ ] Interface — **gráfica (Shiny)** vale mais que **console**, mas qualquer das duas é aceite.
- [ ] Fluxo mínimo:
  1. Utilizador escolhe a semana para a qual quer plano.
  2. Sistema mostra previsões (e valores reais se existirem).
  3. Sistema mostra plano otimizado por dia: clientes, unidades vendidas, vendas, custos, lucro total.
- [ ] Permitir escolher loja e/ou ver as 4 lojas em conjunto.
- [ ] Permitir escolher objetivo (O1 / O2 / O3) e método de otimização.

### 5.2 Recomendações de UX (valoriza)
- [ ] Gráfico de previsão (linha) com banda de incerteza, se aplicável.
- [ ] Tabela do plano semanal com `day, PR, X, J, assisted, units, sales, cost, profit`.
- [ ] Resumo do total semanal (R_s) por loja e total agregado.
- [ ] Indicador visual de violação da restrição O2 (10 000 unidades).
- [ ] Botão para exportar plano em CSV.

### 5.3 Demo / vídeo
- [ ] Vídeo ≤ 5 min, com **narração de voz**.
- [ ] Demonstrar fluxo completo: escolher semana → previsão → otimização → plano final.
- [ ] Upload no YouTube; link no relatório.

---

## 6. Relatório (PDF) — Estrutura Obrigatória

> 20–40 páginas (corpo, sem capa/índice/bibliografia/anexos). Escrita direta e concisa.

- [ ] **Capa**
- [ ] **1. Introdução** — descrição do grupo, objetivos, estrutura do relatório.
- [ ] **2. Execução do projeto** — planeamento, organização do trupo, auto-avaliação **A** (0–20) justificada, proposta de auto-diferenciação individual (½ página por elemento descrevendo esforço, tarefas, código, testes).
- [ ] **3. Tarefa de Predição** — métodos, métricas, resultados, comparações, justificação do melhor.
- [ ] **4. Tarefa de Otimização** — modelação, métodos, configurações, resultados por O1/O2/O3, convergência, Pareto.
- [ ] **5. Demonstração do Sistema** — 1 frase com link YouTube + breve descrição da UI.
- [ ] **6. Conclusões** — apreciação, limitações, trabalho futuro.
- [ ] **Bibliografia** (opcional mas desejada — `forecast`/`fpp2`, `rminer`, etc.).
- [ ] **Anexos** (opcionais — gráficos extra, tabelas, prompts de IA usados e respetivos resultados).

### 6.1 Uso de IA (obrigatório declarar)
- [ ] Listar ferramentas usadas (ChatGPT, Gemini, Claude, etc.).
- [ ] Indicar queries representativas e resultados obtidos.
- [ ] Garantir que todo o código copiado é compreendido pelo grupo.

### 6.2 Auto-avaliação
- [ ] Propor nota A (0–20) com justificação ("executámos bem ... mas não conseguimos ...").
- [ ] Auto-diferenciação individual: soma ≤ M × P; nenhum aluno reprova por subida de outros; justificar.

---

## 7. Submissão

- [ ] **ZIP único** com:
  - Relatório PDF (obrigatório)
  - Slides PDF (opcional)
  - Código R/Python (obrigatório)
  - Ficheiros auxiliares (CSV de previsões guardadas, README do código)
- [ ] Link YouTube do vídeo dentro do relatório.

---

## 8. Plano de Trabalho Sugerido (cronograma)

| Fase | Duração | Marco |
|---|---|---|
| 1. Setup, EDA, definir métricas | 1 sem. | Datasets carregados, NA tratados, gráficos exploratórios. |
| 2. Forecasting Fase I (univariado) | 1–2 sem. | Cada membro com 1 método a correr; baseline Seasonal Naive. |
| 3. Forecasting Fase II (multivariado + backtesting) | 2 sem. | CSV final de previsões; tabela comparativa; método "vencedor". |
| 4. Otimização Fase I (real values, métodos isolados) | 2 sem. | `evaluation()` validada; 4+ métodos a correr. |
| 5. Otimização Fase II (com previsões + multi-obj) | 2 sem. | O1/O2/O3 resolvidos; Pareto; convergência. |
| 6. DSS (Shiny / console) | 1–2 sem. | UI funcional ligada a forecasting + otimização. |
| 7. Relatório + vídeo + ensaios | 1 sem. | ZIP submetido; vídeo no YouTube. |

> **Resumos semanais obrigatórios desde 2026-03-16** (Google Doc com link "anyone can view") com (a) sumário do trabalho do grupo, reuniões e duração, outputs, dúvidas; (b) por elemento: tarefas individuais e horas dedicadas.

---

## 9. Checklist Final de Apresentação (revisão de qualidade)

### 9.1 Cobertura completa
- [ ] **Os 3 objetivos de otimização foram resolvidos** (O1, O2, O3).
- [ ] **Forecasting** com baseline + univariado + multivariado.
- [ ] Cada membro do grupo implementou **1 método de previsão** e **1 método de otimização** distintos.
- [ ] DSS funcional ligando previsão e otimização.

### 9.2 Justificação das escolhas (essencial!)
- [ ] Para forecasting: explico **por que** o modelo X é o melhor (métrica, robustez, % melhoria vs. Seasonal Naive).
- [ ] Para otimização: explico **por que** o método Y é o melhor para cada objetivo (lucro mediano, convergência, robustez).
- [ ] Pesos/normalização de O3 justificados; ou Pareto apresentado.

### 9.3 Robustez
- [ ] Backtesting com ≥10 iterações.
- [ ] Otimização com ≥10 runs (uma por semana de teste).
- [ ] Métricas agregadas com mediana/média (escolha justificada).
- [ ] Mesmas condições de comparação para todos os métodos.

### 9.4 Realismo / extras valorizados
- [ ] Estudo do efeito de hiperparâmetros (temp do SANN, popSize do GA, etc.).
- [ ] Cenários adicionais (ex.: aumentar W_s, mudar F_J/F_X, simular rutura de stock).
- [ ] Visualizações claras e profissionais.
- [ ] Código limpo, modular, comentado, com README.
- [ ] Uso de IA documentado e transparente.

### 9.5 Forma e ética
- [ ] Sem plágio; referências citadas; uso de IA declarado.
- [ ] Relatório dentro de 20–40 páginas, escrita direta.
- [ ] Vídeo ≤ 5 min, com narração, link no relatório.
- [ ] ZIP final completo.

---

## 10. Pitfalls a Evitar

- **Não validar a `evaluation()`** antes de otimizar — gera horas de debug. Usar sempre os exemplos dos slides 17 e 18.
- **Esquecer de arredondar a solução final** após otimização real-valued.
- **Tratar O2 como separável** — não é, a restrição é global.
- **Comparar métodos em condições diferentes** (períodos de teste distintos, métricas distintas) — invalida conclusões.
- **Usar a média quando há outliers fortes** — preferir a mediana, ou justificar.
- **Otimizar com valores reais e apresentar como se fossem previsões** — Fase II tem de usar previsões.
- **Não documentar o uso de IA** — penalização ética.
- **Apresentar Pareto sem discutir trade-offs** — perde-se o ponto da multi-objetivo.
- **DSS deixado para o fim sem tempo** — começar a Shiny/console assim que o forecasting estiver maduro.

---

## 11. Referências Úteis

- `forecast` package: <https://otexts.com/fpp2/>
- `rminer` package: <https://dx.doi.org/10.21814/1822.36210>
- Demos de aula: `Roptim1.zip`, `demo-4-sann.R`, `opt-4-convergence-2demos.R`
- Multi-objetivo: `mco` (NSGA-II em R), `pymoo` (Python)
- Shiny: <https://shiny.posit.co/>

---

*Última verificação antes de submeter: ler este documento de uma ponta à outra e marcar mentalmente cada checkbox. Se ficar algum por preencher e não houver justificação no relatório, é nota perdida.*
