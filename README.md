# 🇺🇸 USA Stores Sales Forecasting - Projeto DSS 2026

Este repositório contém um Sistema Inteligente de Apoio à Decisão (DSS) para a previsão e otimização de vendas em quatro localizações de retalho nos Estados Unidos: **Baltimore, Lancaster, Philadelphia e Richmond**.

O projeto foi estruturado com uma arquitetura modular e profissional, focada em escalabilidade e reprodutibilidade científica (Target: 20/20).

---

## 🏗️ Arquitetura do Sistema (Estrutura Sénior)

O projeto está organizado em domínios lógicos para facilitar a manutenção e a integração entre equipas:

### 📁 `src/` - Motor de Lógica Central
*   **`data/preparation.py`**: Limpeza científica de dados, tratamento de outliers (Clipping 1%-99%), interpolação linear de valores em falta e engenharia de atributos (lags, médias móveis, feriados).
*   **`forecasting/trainer.py`**: Módulo de treino comparativo que avalia 5 modelos distintos simultaneamente e gera métricas de erro (MAE, RMSE, MAPE).
*   **`optimization/nsga2_model.py`**: Algoritmo Genético (NSGA-II) para a fase de otimização de staff e lucro (W5).
*   **`utils/profit_logic.py`**: Lógica de negócio, custos de RH e cálculo de margens de lucro.

### 📁 `results/` - Galeria de Resultados Profissionais
*   **`00_Master_Summary`**: Relatórios consolidados e gráficos comparativos globais.
*   **`01_EDA_Gallery`**: Análises estatísticas, correlações e decomposição sazonal.
*   **`02_Forecasting_Report`**: Gráficos detalhados de "Real vs Previsto" organizados por loja.

### 📁 `dss_app/` - Dashboard de Visualização
*   Aplicação interativa em **Streamlit** para visualização rápida da performance dos modelos e tendências de negócio.

---

## 📊 Modelos Integrados (Forecasting)

O sistema avalia e compara automaticamente as seguintes abordagens:
1.  **Seasonal Naive**: Referência base baseada em ciclos semanais.
2.  **Linear Regression**: Modelo estatístico multivariado de base.
3.  **Random Forest**: Aprendizagem automática não-linear (Ensemble).
4.  **Holt-Winters**: Suavização exponencial tripla com sazonalidade (ETS).
5.  **Prophet (Meta)**: Modelo Bayesiano de última geração para séries temporais.

---

## 🚀 Como Executar o Sistema

### 1. Correr o Pipeline Mestre
Para processar os dados e gerar todas as previsões e relatórios:
```powershell
python main_pipeline.py
```

### 2. Lançar o Dashboard
Para visualizar os resultados de forma interativa:
```powershell
streamlit run dss_app/app.py
```

---

## 📅 Estado do Projeto
Atualmente, o sistema concluiu a fase de **Forecasting (W4)** com sucesso, apresentando uma estrutura de dados higienizada e pronta para a **Fase de Otimização (W5)**, onde utilizaremos algoritmos evolutivos para maximizar a rentabilidade das lojas.
