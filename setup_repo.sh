#!/bin/bash

# Nome do projeto
PROJECT_NAME="dss_usa_stores"

echo "🚀 Iniciando a estrutura do projeto: $PROJECT_NAME"

# 1. Criação das pastas principais (CRISP-DM)
mkdir -p data              # Data Understanding / CSVs
mkdir -p notebooks         # Data Preparation & EDA
mkdir -p src/models        # Modeling (Forecasting & Optimization)
mkdir -p src/utils         # Funções auxiliares (Profit Function)
mkdir -p dss_app           # Deployment (Interface Streamlit)
mkdir -p reports/summaries # Project Execution & Weekly Summaries
mkdir -p docs/assets       # Imagens e anexos para o relatório

# 2. Criação dos arquivos iniciais
touch requirements.txt
touch .gitignore
touch README.md
touch main.py

# 3. Criando templates Python para a equipe
# Template para a Função de Lucro (Essencial para o Aluno 1 validar)
cat <<EOT > src/utils/profit_logic.py
def calculate_profit(store, junior_count, expert_count, promo_pct, is_weekend):
    """
    Template para a função de avaliação (Slide 15).
    A ser preenchido pelo Aluno 1 e validado com os dados de Baltimore.
    """
    pass
EOT

# 4. Configuração básica do .gitignore para Python
cat <<EOT > .gitignore
__pycache__/
*.py[cod]
*$py.class
.venv/
env/
.env
data/*.csv
.vscode/
.idea/
EOT

# 5. Inicialização do Git
git init
git add .
git commit -m "Initial commit: Estrutura CRISP-DM criada pelo Aluno 1"

echo "✅ Repositório estruturado com sucesso!"
echo "Próximo passo: Colocar os arquivos CSV na pasta /data e validar a função de lucro."
