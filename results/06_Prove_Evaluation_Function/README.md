# 🧪 Prova de Conceito: Função de Avaliação (`evaluation()`)

Este diretório contém a evidência técnica da validação da função de avaliação utilizada no motor de otimização (NSGA-II) do projeto **DSS USA Stores**.

## 🎯 Objetivo
Provar que a função matemática que define o "sucesso" de uma escala semanal captura corretamente a dinâmica do negócio, incluindo:
1.  **Elasticidade da Procura**: O efeito dos descontos no volume de vendas.
2.  **Rendimentos Decrescentes**: O custo de oportunidade e excesso de pessoal.
3.  **Restrições Operacionais**: A aplicação de penalizações por violação de limites de staff.

## 🧠 Racional Técnico
A função `evaluation()` (implementada em `src/prove_evaluation_function/profit_logic.py`) traduz um vetor de decisão de 21 dimensões em três objetivos conflitantes. Esta prova de conceito utiliza dados históricos reais da loja de **Baltimore (Junho 2014)** para validar três cenários críticos:

*   **Sub-alocação**: Testa o limite inferior (falta de pessoal). Resulta em perda de vendas por incapacidade de atendimento.
*   **Sobre-alocação**: Testa o limite superior e a sensibilidade a custos. Demonstra que aumentar o staff indefinidamente reduz o lucro e ativa penalizações operacionais.
*   **Otimizado (Manual)**: Demonstra o "ponto doce" onde o equilíbrio entre desconto (elasticidade) e staff maximiza o lucro esperado.

## 🚀 Como Executar
Para reproduzir estes resultados utilizando o ambiente isolado e a versão específica do Python:

```bash
# Navegar para a raiz do projeto
cd /dss_usa_stores

# Executar o script de prova com Python 3.14
python src/prove_evaluation_function/prove_concept.py > results/06_Prove_Evaluation_Function/execution_log.txt
```

## 📊 Arquivos de Saída
*   `execution_log.txt`: Log detalhado da simulação, incluindo métricas diárias e validação de premissas.
*   `evaluation_concept_plot.png`: Visualização comparativa do lucro estimado por cenário.

## 📚 Referências Utilizadas
*   *Essentials of Metaheuristics*: Para o design do espaço de busca e convergência.
*   *Memorial de Cálculo (v2.0)*: Para calibração de escala (`PROFIT_SCALE`) e elasticidade (`ELASTICITY_K`).
