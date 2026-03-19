import streamlit as st
import pandas as pd
import plotly.express as px
import os

# Configuração da página para ocupar o ecrã todo
st.set_page_config(page_title="TIAPOSE - DSS Prototype", layout="wide")

st.title("📊 Sistema Inteligente de Apoio à Decisão")
st.markdown("*(Protótipo - Fase de Business & Data Understanding)*")

# Função otimizada para carregar os dados unificados
@st.cache_data
def load_data():
    # Caminho dinâmico para encontrar a pasta processed
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.abspath(os.path.join(script_dir, "../data/processed/all_stores_merged.csv"))
    
    df = pd.read_csv(data_path)
    df['Date'] = pd.to_datetime(df['Date'])
    return df

try:
    df = load_data()
    
    # --- BARRA LATERAL (MENU) ---
    st.sidebar.header("⚙️ Parâmetros do DSS")
    lojas = df['Store'].unique().tolist()
    loja_selecionada = st.sidebar.selectbox("Selecione a Loja Alvo:", lojas)
    
    # Filtrar os dados apenas para a loja escolhida
    df_loja = df[df['Store'] == loja_selecionada]
    
    # --- PAINEL PRINCIPAL (MÉTRICAS) ---
    st.subheader(f"Análise Histórica: {loja_selecionada}")
    
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Dias Registados", len(df_loja))
    col2.metric("Total de Clientes", f"{df_loja['Num_Customers'].sum():,}")
    col3.metric("Média Vendas/Dia", f"${df_loja['Sales'].mean():,.2f}")
    col4.metric("Desconto Médio", f"{df_loja['Pct_On_Sale'].mean():.1f}%")
    
    # --- GRÁFICO INTERATIVO ---
    st.markdown("---")
    fig = px.line(
        df_loja, 
        x='Date', 
        y='Sales', 
        title=f"Evolução Temporal das Vendas",
        labels={'Sales': 'Vendas ($)', 'Date': 'Data'},
        template="plotly_white"
    )
    # Adicionar cor aos fins de semana ou eventos seria o próximo passo!
    st.plotly_chart(fig, use_container_width=True)
    
except FileNotFoundError:
    st.error("Erro: O ficheiro 'all_stores_merged.csv' não foi encontrado na pasta 'data/processed/'. Verifica se executaste o script de unificação.")