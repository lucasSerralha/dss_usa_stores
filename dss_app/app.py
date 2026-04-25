import streamlit as st
import pandas as pd
import plotly.express as px
import os
import sys

# 0. Configurações de Pathing Seguro para importação do motor de otimização
src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '../src'))
if src_path not in sys.path:
    sys.path.append(src_path)

from optimization.nsga2_model import TiaposeOptimization
from pymoo.algorithms.moo.nsga2 import NSGA2
from pymoo.optimize import minimize

# Configuração da página
st.set_page_config(page_title="TIAPOSE - DSS", layout="wide")

st.title("Sistema Inteligente de Apoio à Decisão (DSS)")
st.markdown("*(Integração Fase 2: Forecasting & Otimização Multi-Objetivo 3D)*")

# Funções de carregamento de dados com cache para otimizar velocidade
def load_historical_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Apontar para o dataset consolidado (versão com features)
    data_path = os.path.abspath(os.path.join(script_dir, "../data/processed/all_stores_processed.csv"))
    
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    return None

def load_model_report():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Apontar para o relatório consolidado gerado pelo main_pipeline.py
    report_path = os.path.abspath(os.path.join(script_dir, "../data/processed/final_model_report.csv"))
    
    if os.path.exists(report_path):
        return pd.read_csv(report_path)
    return None

# Carregar os dados
df_history = load_historical_data()
df_models = load_model_report()

if df_history is not None:
    # --- BARRA LATERAL (MENU) ---
    st.sidebar.header("Parâmetros de Controlo")
    lojas = df_history['store_id'].unique().tolist()
    loja_selecionada = st.sidebar.selectbox("Selecione a Loja Alvo:", lojas)
    
    # --- SECÇÃO 1: CONTEXTO DE NEGÓCIO ---
    st.header(f"Análise de Negócio: {loja_selecionada}")
    df_loja = df_history[df_history['store_id'] == loja_selecionada]
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Dias em Registo", len(df_loja))
    col2.metric("Total de Clientes", f"{int(df_loja['Num_Customers'].sum()):,}")
    col3.metric("Média Vendas/Dia", f"${df_loja['Sales'].mean():,.0f}")
    col4.metric("Desconto Médio", f"{df_loja['Pct_On_Sale'].mean():.1f}%")
    
    st.markdown("---")
    
    # --- SECÇÃO 2: AVALIAÇÃO DE FORECASTING (W3 & W4) ---
    st.header("Desempenho dos Modelos Preditivos")
    
    if df_models is not None:
        # Filtro blindado contra maiúsculas, minúsculas e espaços perdidos
        df_models_loja = df_models[df_models['Store'].astype(str).str.strip().str.lower() == loja_selecionada.strip().lower()]
        
        if not df_models_loja.empty:
            st.markdown("Comparação do Erro Médio Absoluto (MAE) e da Raiz do Erro Quadrático Médio (RMSE) para um horizonte de 7 dias.")
            
            # Preparar dados para o gráfico de barras
            df_melted = df_models_loja.melt(id_vars=['Model'], value_vars=['MAE', 'RMSE'], 
                                            var_name='Métrica', value_name='Erro ($)')
            
            # Gerar o gráfico Plotly
            fig = px.bar(df_melted, x='Model', y='Erro ($)', color='Métrica', barmode='group',
                         title=f"Comparativo de Erros Preditivos - {loja_selecionada}",
                         color_discrete_sequence=['#1f77b4', '#ff7f0e'])
            
            # Desenhar o gráfico no ecrã
            st.plotly_chart(fig, use_container_width=True)
            
            # Lógica para destacar o modelo com menor erro (vencedor)
            melhor_modelo = df_models_loja.loc[df_models_loja['RMSE'].idxmin()]
            st.info(f"**Recomendação do Sistema:** Para a loja de {loja_selecionada}, o modelo mais preciso é o **{melhor_modelo['Model']}**, com um erro RMSE de apenas ${melhor_modelo['RMSE']:.2f}.")
            
        else:
            st.warning(f"Sem dados de avaliação para a loja {loja_selecionada} no relatório atual.")
    else:
        st.error("Ficheiro 'final_model_report.csv' não encontrado. Garante que o pipeline do António foi executado.")

    st.markdown("---")
    
    # --- SECÇÃO 3: OTIMIZAÇÃO DE ESCALAS (CENÁRIO 3) ---
    st.header("Simulador de Escalas (Fronteira de Pareto 3D)")
    st.markdown("Executa a otimização multi-objetivo para encontrar o equilíbrio entre **Lucro**, **Dimensão da Equipa** e **Restrições Operacionais**.")

    if st.button("Executar Simulação de Escalas (NSGA-II)"):
        with st.spinner("O algoritmo genético está a explorar as melhores combinações..."):
            # 1. Selecionar os últimos 7 dias para simular uma janela de planeamento
            df_slice = df_loja.sort_values('Date').tail(7)
            
            if len(df_slice) == 7:
                forecast_customers = df_slice['Num_Customers'].tolist()
                forecast_is_weekend = df_slice['IsWeekend'].tolist()

                # 2. Configurar o Problema de Otimização (Parâmetros leves para Web)
                problem = TiaposeOptimization(loja_selecionada, forecast_customers, forecast_is_weekend)
                algorithm = NSGA2(pop_size=20)
                
                # Execução do algoritmo
                res = minimize(problem, algorithm, ('n_gen', 15), seed=1, verbose=False)

                # 3. Preparação do DataFrame 3D para o Plotly
                # Pymoo Minimiza F1 (Lucro Neg), F2 (Staff), F3 (Penalização)
                df_pareto = pd.DataFrame(res.F, columns=['f1', 'f2', 'f3'])
                df_pareto['Lucro Projetado ($)'] = df_pareto['f1'] * -1
                df_pareto['Staff Total (Pessoas)'] = df_pareto['f2']
                df_pareto['Penalização (Score)'] = df_pareto['f3']

                # 4. Geração do Gráfico 3D Interativo
                fig_3d = px.scatter_3d(
                    df_pareto, 
                    x='Staff Total (Pessoas)', 
                    y='Penalização (Score)', 
                    z='Lucro Projetado ($)',
                    color='Lucro Projetado ($)',
                    title=f"Fronteira de Pareto 3D - Loja {loja_selecionada.capitalize()}",
                    labels={
                        'Staff Total (Pessoas)': 'Staff Total (Pessoas)', 
                        'Penalização (Score)': 'Penalização (Score)', 
                        'Lucro Projetado ($)': 'Lucro Projetado ($)'
                    },
                    template='plotly_white',
                    color_continuous_scale='Viridis'
                )
                
                st.plotly_chart(fig_3d, use_container_width=True)
                st.success(f"Otimização concluída! Foram identificadas {len(df_pareto)} soluções ótimas.")
                st.info("**Dica:** Roda o gráfico 3D com o rato para veres o 'compromisso' entre os três objetivos.")
            else:
                st.warning("Dados históricos insuficientes para simular uma janela de 7 dias.")

else:
    st.error("Ficheiro de dados históricos não encontrado na pasta 'data/processed/'.")
