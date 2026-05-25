# DSS USA Stores: Forecasting & Multi-Objective Optimization
## Decision Support System (DSS) Project Presentation

---

## 1. Contexto e Objetivos Estratégicos

O projeto **DSS USA Stores** surge da necessidade de modernizar a gestão operacional de uma rede de retalho composta por quatro unidades geográficas distintas: **Baltimore, Lancaster, Philadelphia e Richmond**. O mercado retalhista atual caracteriza-se por uma volatilidade elevada, onde padrões históricos isolados já não garantem previsões fiáveis.

### Desafios de Negócio:
*   **Volatilidade da Procura:** Flutuações diárias influenciadas por promoções agressivas e eventos sazonais.
*   **Gestão de Recursos Humanos:** Equilíbrio entre a qualidade de serviço (Staff Experto) e a eficiência de custos (Staff Júnior).
*   **Gargalos Logísticos:** Restrições físicas de transporte que limitam o volume total de vendas da rede a 10.000 unidades por semana.

### Objetivos Técnicos:
1.  **Sistemas Preditivos de Alta Fidelidade:** Implementar uma arquitetura de *Ensemble* capaz de superar as baselines estatísticas em pelo menos 30%.
2.  **Otimização Hierárquica:** Resolver o problema de alocação de recursos em três níveis: tático por loja (O1), logístico de rede (O2) e estratégico multiobjetivo (O3).

---

## 2. Abordagem Metodológica Detalhada

A metodologia foi estruturada para garantir a robustez científica de ponta a ponta:

### A. Engenharia de Atributos (Feature Engineering)
*   **Decomposição Temporal:** Extração de componentes de tendência, sazonalidade semanal e anual através de modelos aditivos.
*   **Enriquecimento de Contexto:** Criação de variáveis binárias para feriados federais, eventos desportivos locais e campanhas de marketing.
*   **Dinâmicas de Vendas:** Introdução de *Lags* (t-1 a t-7) para capturar a autocorrelação serial e janelas deslizantes (*Rolling Means*) para suavização de ruído.

### B. Previsão Híbrida e Arquitetura de Cenários
*   **Framework de Experimentação:** Os modelos foram testados em três regimes de informação:
    *   **A (Base):** Apenas componentes temporais.
    *   **B (Dynamics):** Inclusão de comportamentos recentes de vendas.
    *   **C (Expert):** Contexto total de mercado (O cenário com melhor performance).
*   **Meta-Modelagem (Ensemble):** Combinação linear ponderada dos Top-3 modelos (ex: LightGBM + Prophet + SARIMAX), onde os pesos são inversamente proporcionais ao erro de validação.

### C. Modelação da Função de Avaliação (Profit Logic)
*   **Elasticidade-Preço:** Implementação de uma função de resposta de vendas baseada no desconto aplicado (0-30%).
*   **Modelo de Staffing:** Diferenciação de produtividade entre Expert (maior conversão) e Júnior (menor custo).
*   **Penalização de Restrições:** Uso de funções de penalidade (*Death Penalty*) para invalidar soluções que excedam a capacidade logística de rede.

---

## 3. Experiências Computacionais e Validação

Para assegurar a aplicabilidade real, as experiências foram desenhadas com rigor estatístico:

### Protocolo de Validação de Previsão:
*   **TimeSeriesSplit:** Uso de 5 *folds* temporais, garantindo que o modelo nunca treina com dados do futuro.
*   **Métricas Multidimensionais:** Avaliação via **RMSE** (para penalizar grandes erros), **MAPE** (para interpretabilidade percentual) e **NMAE** (para comparação entre lojas de diferentes escalas).

### Configuração dos Algoritmos de Otimização:
*   **Hill Climbing (O1/O2):** Implementação de um motor de busca local com perturbações estocásticas. A utilização de 10 *random restarts* permitiu mapear diferentes bacias de atração, garantindo a convergência para o ótimo global.
*   **U-NSGA-III (O3):** Utilização de uma população diversificada e mecanismos de preservação de elite baseados em nichos de referência. Esta abordagem permitiu gerar uma Fronteira de Pareto com densidade uniforme em todo o espaço de objetivos.

---

## 4. Resultados Obtidos: Análise Profunda

### Resultados de Previsão (Fidelity Analysis):
*   **Ganhos vs. Naive:** O sistema reduziu o erro RMSE em média **47%** em relação ao modelo *Seasonal Naive*.
*   **Impacto do Contexto:** A transição do Cenário A para o Cenário C resultou numa melhoria de precisão de **15-20%**, validando a hipótese de que o contexto promocional é o driver principal de vendas.

### Resultados de Otimização:
*   **Superioridade Algorítmica (O1):** O Hill Climbing superou o NSGA-II em termos de lucro máximo em **85%** nas instâncias de Baltimore e Richmond, devido ao seu foco exclusivo num único objetivo.
*   **Otimização de Rede (O2):** A abordagem de *Knapsack* provou ser superior na distribuição de "cotas de venda", aumentando o lucro total da rede em **$10,675/semana** comparado a heurísticas de alocação fixa.
*   **Trade-off Estratégico (O3):** A Fronteira de Pareto revelou que, para a unidade de Philadelphia, cada redução de 1 funcionário experto acarreta uma perda média de **$1,200** em vendas não realizadas por falta de conversão.

---

## 5. Demonstração e Funcionalidades do Sistema

A solução Streamlit foi desenhada para perfis de utilizador distintos (Analistas vs. Gestores):

### Auditoria Científica (Analyst View):
*   **Análise de Resíduos:** Gráficos de dispersão e histogramas para verificar a normalidade do erro e a ausência de viés sistemático.
*   **XAI (Explainable AI):** Uso de *Feature Importance* para identificar quais as variáveis (ex: Desconto de FDS, Proximidade de Feriado) que mais influenciam cada previsão.

### Centro de Operações Táticas (Manager View):
*   **Simulador "What-if":** Alteração dinâmica de parâmetros (ex: aumentar o limite de rede para 12.000) com visualização instantânea do impacto no lucro.
*   **Escalarização Dinâmica (w):** Slider que permite ao gestor definir o seu perfil de risco/custo. Ao ajustar o peso $w$, o sistema seleciona automaticamente o plano operacional na fronteira de Pareto que maximiza a utilidade para esse gestor específico.

---

## 6. Conclusões e Recomendações Futuras

O **DSS USA Stores** representa o estado da arte na aplicação de meta-heurísticas ao retalho:

1.  **Robustez Preditiva:** O uso de *Ensembles* adaptativos por loja mitiga o risco de falha de modelos individuais em mercados específicos.
2.  **Vantagem Competitiva:** A capacidade de otimizar preços e staff simultaneamente gera uma sinergia que maximiza a margem operacional além dos métodos tradicionais de gestão.
3.  **Transparência Decisória:** Ao expor a Fronteira de Pareto, o sistema deixa de ser uma "caixa-preta" e passa a ser uma ferramenta de suporte à negociação entre os departamentos de Vendas e Recursos Humanos.
4.  **Escalabilidade:** A arquitetura modular permite a inclusão de novas lojas ou novas restrições (ex: limites de staff por sindicato) com alterações mínimas no núcleo algorítmico.

---
**Projeto:** DSS USA Stores | **Data:** Maio 2026 | **Status:** Concluído para Deploy
