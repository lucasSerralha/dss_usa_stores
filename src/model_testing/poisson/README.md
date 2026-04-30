# Modelo Probabilístico de Poisson — TIAPOSE DSS

Este diretório contém a implementação e as ferramentas de diagnóstico para o **Modelo Probabilístico de Poisson**, utilizado para modelar a incerteza na chegada de clientes e calcular o impacto financeiro estocástico nas lojas.

## 1. O que é o Poisson Probabilístico?

Ao contrário de uma abordagem determinística, onde assumimos que a previsão (ex: 100 clientes) é um número fixo e exato, o **Modelo de Poisson** reconhece que a chegada de clientes é um processo estocástico. 

A distribuição de Poisson é ideal para modelar o número de eventos (clientes) que ocorrem num intervalo de tempo fixo, sendo definida por um único parâmetro $\lambda$ (lambda), que representa a taxa média de ocorrência.

### Vantagens para o DSS:
- **Gestão de Risco:** Permite calcular a probabilidade de a loja estar com falta de pessoal (*understaffing*).
- **Lucro Esperado:** Em vez de calcular o lucro para um cenário único, calculamos a média ponderada do lucro sobre todos os cenários possíveis de volume de clientes.
- **Robustez:** Ajuda a encontrar planos de alocação que funcionam bem mesmo quando a realidade diverge ligeiramente da previsão média.

---

## 2. Scripts Disponíveis

### 📊 `poisson_diagnostics.py`
Realiza testes estatísticos nos dados brutos (`data/raw`) para validar se a premissa de Poisson é adequada.
- **O que faz:** Calcula a razão Variância/Média (Índice de Dispersão), realiza o teste de Kolmogorov-Smirnov e gera gráficos de densidade.
- **Saída:** Relatórios no terminal e gráficos em `results/poisson_diagnostic_[loja].png`.

### 💰 `stochastic_profit.py`
Implementa a lógica de **Lucro Esperado (E[Profit])**.
- **O que faz:** Integra a função de lucro do projeto sobre a Função de Massa de Probabilidade (PMF) de Poisson. Trunca a soma infinita num ponto de confiança de 99.9%.
- **Recurso extra:** Calcula o **Risco de Understaffing**, que é a probabilidade de a demanda exceder a capacidade total de atendimento do staff alocado.

---

## 3. Como Utilizar

Certifique-se de que as dependências (`scipy`, `seaborn`, `matplotlib`) estão instaladas.

### Executar Diagnósticos:
```bash
python3 src/model_testing/poisson/poisson_diagnostics.py
```

### Executar Teste de Lucro Estocástico:
```bash
python3 -m src.model_testing.poisson.stochastic_profit
```

---

## 4. Exemplos de Saída e Interpretação

### Exemplo de Diagnóstico (Baltimore):
- **Média:** 137.99 | **Variância:** 13310.78
- **Razão Var/Média:** 96.46
- **Interpretação:** Uma razão > 1 indica **sobredispersão**. Isso sugere que, embora o Poisson seja a base, o modelo final (GLM) deve usar regressores (como feriados e descontos) para explicar essa variação extra.

### Exemplo de Lucro Estocástico:
Se a previsão média ($\lambda$) é de 100 clientes, mas a capacidade do staff é apenas para 66:
- **Lucro Determinístico:** $378.00 (calculado exatamente em 100 clientes).
- **Lucro Esperado (Estocástico):** $376.97.
- **Risco de Understaffing:** 99.98%.
- **Conclusão:** O modelo estocástico é mais "conservador" e realista, pois considera a perda de vendas nos cenários onde a demanda flutua acima da média.
