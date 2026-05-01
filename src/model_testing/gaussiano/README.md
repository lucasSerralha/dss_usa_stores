# Modelo Gaussiano — Teste e Diagnóstico

Este diretório contém a implementação dos testes de aderência para a distribuição **Gaussiana (Normal)** aplicada às vendas das lojas.

## Componentes

1.  **`gaussiano_diagnostics.py`**:
    *   Analisa a distribuição histórica da coluna `Sales` nos dados brutos.
    *   Realiza testes estatísticos de normalidade (Shapiro-Wilk e Normal Test de D'Agostino).
    *   Gera histogramas com curvas normais teóricas e Q-Q Plots para validação visual.
    *   **Resultados**: Salvos em `results/04_Model_Testing/Gaussiano/`.

2.  **`stochastic_profit_gaussiano.py`**:
    *   Calcula o lucro esperado integrando sobre a incerteza das vendas.
    *   Utiliza simulação de Monte Carlo para estimar o valor esperado do lucro operacional diário considerando que as vendas seguem uma distribuição Normal(mu, sigma).

## Por que Gaussiano para Vendas?

Diferente da chegada de clientes (contagem discreta → Poisson), o valor de vendas é uma variável contínua e de alta magnitude. Pelo Teorema Central do Limite, a agregação de múltiplas transações diárias tende a seguir uma distribuição Normal, o que justifica o uso deste modelo para capturar a variabilidade financeira do negócio.

## Como Executar

Para gerar os diagnósticos de todas as lojas:
```bash
python src/model_testing/gaussiano/gaussiano_diagnostics.py
```
