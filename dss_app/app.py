import streamlit as st
import sys
import pandas as pd
import os
import numpy as np

# Bloqueio de criação de pastas __pycache__
sys.dont_write_bytecode = True
import plotly.express as px
import plotly.graph_objects as go

# Configuração da página e estética (Tema Profissional)
st.set_page_config(page_title="USA Stores Forecasting DSS", layout="wide")

# Estilo CSS customizado para KPIs e estética corporativa
st.markdown("""
    <style>
    /* Estilo Base para as Métricas */
    .stMetric { 
        padding: 20px; 
        border-radius: 4px; 
        border: 1px solid #d3d3d3;
        box-shadow: none;
        background-color: #fafafa;
    }
    
    @media (prefers-color-scheme: dark) {
        .stMetric {
            border: 1px solid #333333;
            background-color: #1e1e1e;
        }
    }

    .metric-value-row {
        display: flex;
        align-items: baseline;
        gap: 10px;
    }
    
    .metric-delta {
        font-size: 0.9rem;
        color: #2c3e50;
        background: rgba(44, 62, 80, 0.1);
        padding: 2px 6px;
        border-radius: 4px;
        font-weight: 600;
    }

    @media (prefers-color-scheme: dark) {
        .metric-delta {
            color: #d1d5da;
            background: rgba(209, 213, 218, 0.1);
        }
    }
    
    h1, h2, h3 { font-family: 'Inter', sans-serif; font-weight: 600; }
    
    /* Posicionamento do Expander */
    .streamlit-expanderHeader {
        background-color: rgba(151, 166, 195, 0.1) !important;
        border-radius: 4px 4px 0 0 !important;
    }
    .streamlit-expanderContent {
        background-color: transparent !important;
        border-radius: 0 0 4px 4px !important;
        border: 1px solid rgba(151, 166, 195, 0.2) !important;
    }
    </style>
""", unsafe_allow_html=True)

# --- HELPERS PARA CARREGAR DADOS ---
@st.cache_data
def load_historical_data():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.abspath(os.path.join(script_dir, "../data/processed/all_stores_processed.csv"))
    if os.path.exists(data_path):
        df = pd.read_csv(data_path)
        df['Date'] = pd.to_datetime(df['Date'])
        return df
    return None

@st.cache_data
def load_master_report():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.abspath(os.path.join(script_dir, "../results/00_Master_Summary/fidelity_experimentation_report.csv"))
    return pd.read_csv(path) if os.path.exists(path) else None

@st.cache_data
def load_optimization_data(loja):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.abspath(os.path.join(script_dir, f"../results/03_Optimization_Report/{loja.capitalize()}/pareto_front.csv"))
    return pd.read_csv(path) if os.path.exists(path) else None

df_history = load_historical_data()
df_master = load_master_report()

if df_history is not None:
    # --- SIDEBAR PROFISSIONAL ---
    st.sidebar.title("DSS Control Panel")
    lojas_formatadas = [l.capitalize() for l in sorted(df_history['store_id'].unique().tolist())]
    loja_sel_raw = st.sidebar.selectbox("Selecione a Loja Alvo:", lojas_formatadas)
    loja_sel = loja_sel_raw.lower()
    
    if df_master is not None:
        cenarios = sorted(df_master['Experiment'].unique().tolist())
        cenario_sel = st.sidebar.radio("Cenário de Experimentação:", cenarios, index=len(cenarios)-1)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### Status do Sistema")
    st.sidebar.markdown("- **Modelos Preditivos:** Online")
    st.sidebar.markdown("- **Motor NSGA-II:** Convergido")
    st.sidebar.markdown(f"- **Loja Ativa:** {loja_sel.capitalize()}")

    # --- HEADER & KPIs ---
    st.title(f"USA Stores Analytics: {loja_sel.capitalize()}")
    st.markdown(f"**Framework de Apoio à Decisão** | Categoria: `{cenario_sel}`")
    
    df_loja = df_history[df_history['store_id'] == loja_sel]
    
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    col_kpi1.metric("Vendas Totais Históricas", f"${df_loja['Sales'].sum()/1e6:.2f}M")
    
    if df_master is not None:
        store_metrics = df_master[(df_master['Store'].str.lower() == loja_sel.lower()) & (df_master['Experiment'] == cenario_sel)]
        if not store_metrics.empty:
            rmse_min = store_metrics['RMSE'].min()
            melhor_nome = store_metrics.loc[store_metrics['RMSE'].idxmin(), 'Model']
            
            try:
                rmse_naive = store_metrics[store_metrics['Model'] == 'Seasonal Naive']['RMSE'].values[0]
                ganho = ((rmse_naive - rmse_min) / rmse_naive) * 100
            except IndexError:
                ganho = 0.0

            with col_kpi2:
                st.markdown(f"""
                    <div class="stMetric">
                        <p style="font-size: 14px; margin-bottom: 5px; opacity: 0.8;">Ganho de Precisão (vs Naive)</p>
                        <div class="metric-value-row">
                            <span style="font-size: 28px; font-weight: 700;">{ganho:.1f}%</span>
                            <span class="metric-delta">Modelo: {melhor_nome}</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            col_kpi3.metric("Melhor RMSE Diário", f"${rmse_min:,.0f}")
            fidelidade = 100 - store_metrics['MAPE'].min()
            col_kpi4.metric("Fidelidade (Acurácia)", f"{fidelidade:.1f}%")

    with st.expander("Sobre este Cenário de Previsão", expanded=False):
        descricoes = {
            "A_Temporal_Base": "Análise focada em padrões cíclicos semanais e mensais. Adequado para operações com baixa variabilidade externa.",
            "B_Sales_Dynamics": "Integração da dinâmica de vendas de curto prazo. Captura alterações recentes no comportamento do consumidor.",
            "C_Context_Expert": "Combinação de histórico, dinâmica de vendas e contexto de mercado (promoções e eventos)."
        }
        st.write(descricoes.get(cenario_sel, "Cenário não catalogado."))

    st.markdown("---")

    # --- TABS NAVIGATION ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Previsão de Vendas", 
        "Diagnóstico Técnico", 
        "Decomposição Temporal", 
        "Interpretabilidade Algorítmica",
        "Otimização de Escalas"
    ])

    # --- TAB 1: PREVISÃO ---
    with tab1:
        st.subheader("Previsão de Vendas")
        st.markdown('<p style="font-style: normal; margin-top: -10px; margin-bottom: -10px; opacity: 0.8;">Análise comparativa entre dados reais e projeções dos modelos preditivos.</p>', unsafe_allow_html=True)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path_raw = os.path.join(script_dir, f"../results/02_Forecasting_Report/{loja_sel.capitalize()}/{cenario_sel}/forecast_values.csv")
        
        if os.path.exists(path_raw):
            df_plot = pd.read_csv(path_raw)
            df_plot['Date'] = pd.to_datetime(df_plot['Date'])
            
            fig = go.Figure()
            # Cor sólida escura para dados reais
            fig.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot['Actual'], name='Dados Reais',
                                    line=dict(color='#2c3e50', width=3), mode='lines+markers'))
            
            models = [c for c in df_plot.columns if c not in ['Date', 'Actual']]
            # Paleta de tons neutros (cinzentos e azuis insaturados)
            colors = ['#7f8c8d', '#95a5a6', '#bdc3c7', '#5c6a79', '#34495e', '#e0e0e0']
            for i, m in enumerate(models):
                is_best = (m == melhor_nome)
                fig.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot[m], name=m,
                                        line=dict(dash='dash' if not is_best else 'solid', 
                                                  width=2.5 if is_best else 1.5, 
                                                  color=colors[i % len(colors)]),
                                        visible=True if is_best else 'legendonly'))
            
            fig.update_layout(hovermode="x unified", xaxis_title="Data", yaxis_title="Vendas ($)", margin=dict(t=80, b=10, l=10, r=10))
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
        else:
            st.warning("Dados brutos de previsão indisponíveis para este cenário.")

    # --- TAB 2: DIAGNÓSTICO ---
    with tab2:
        st.subheader("Análise de Erros e Resíduos")
        if os.path.exists(path_raw):
            df_err = df_plot.copy()
            df_err['Erro'] = df_err[melhor_nome] - df_err['Actual']
            col_err1, col_err2 = st.columns(2)
            with col_err1:
                fig_hist = px.histogram(df_err, x="Erro", nbins=20, title=f"Distribuição dos Erros ({melhor_nome})", color_discrete_sequence=['#5c6a79'])
                st.plotly_chart(fig_hist, use_container_width=True, theme="streamlit")
            with col_err2:
                fig_scat = px.scatter(df_err, x="Actual", y=melhor_nome, trendline="ols", title="Dispersão: Real vs Previsto", color_discrete_sequence=['#5c6a79'])
                st.plotly_chart(fig_scat, use_container_width=True, theme="streamlit")

    # --- TAB 3: DECOMPOSIÇÃO ---
    with tab3:
        st.subheader("Decomposição Estrutural")
        path_comp = os.path.join(script_dir, f"../results/02_Forecasting_Report/{loja_sel.capitalize()}/{cenario_sel}/prophet_components.csv")
        if os.path.exists(path_comp):
            df_comp = pd.read_csv(path_comp)
            df_comp['ds'] = pd.to_datetime(df_comp['ds'])
            col_c1, col_c2 = st.columns(2)
            with col_c1:
                fig_trend = px.line(df_comp, x='ds', y='trend', title="Tendência Estrutural", line_shape='spline', color_discrete_sequence=['#34495e'])
                st.plotly_chart(fig_trend, use_container_width=True, theme="streamlit")
            with col_c2:
                if 'weekly' in df_comp.columns:
                    df_weekly = df_comp.tail(7).copy()
                    dias_pt = {'Monday': 'Segunda', 'Tuesday': 'Terça', 'Wednesday': 'Quarta', 'Thursday': 'Quinta', 'Friday': 'Sexta', 'Saturday': 'Sábado', 'Sunday': 'Domingo'}
                    df_weekly['Dia'] = df_weekly['ds'].dt.day_name().map(dias_pt)
                    fig_week = px.bar(df_weekly, x='Dia', y='weekly', title="Sazonalidade Semanal", color='weekly', color_continuous_scale='Blues')
                    st.plotly_chart(fig_week, use_container_width=True, theme="streamlit")
                else:
                    st.info("Componente sazonal não aplicável a este conjunto de dados.")

    # --- TAB 4: INTELIGÊNCIA ---
    with tab4:
        st.subheader("Impacto das Variáveis Independentes")
        path_feat = os.path.join(script_dir, f"../results/02_Forecasting_Report/{loja_sel.capitalize()}/{cenario_sel}/feature_importance.csv")
        if os.path.exists(path_feat):
            df_feat = pd.read_csv(path_feat).head(10)
            # Utilização de uma cor sólida neutra em vez de um gradiente colorido
            fig_feat = px.bar(df_feat, x='Importance', y='Feature', orientation='h', title="Feature Importance", color_discrete_sequence=['#5c6a79'])
            fig_feat.update_layout(yaxis={'categoryorder':'total ascending'})
            st.plotly_chart(fig_feat, use_container_width=True, theme="streamlit")
        else:
            st.info("Informação de interpretabilidade algorítmica não disponível.")

    # --- TAB 5: OTIMIZAÇÃO E ESCALAS ---
    with tab5:
        st.subheader("Fronteira de Pareto: Gestão de Recursos vs. Retorno")
        st.markdown('<p style="font-style: normal; margin-top: -10px; margin-bottom: 20px; opacity: 0.8;">Análise multiobjetivo gerada pelo algoritmo NSGA-II.</p>', unsafe_allow_html=True)
        
        df_pareto = load_optimization_data(loja_sel)
        
        if df_pareto is not None:
            col_lucro = [c for c in df_pareto.columns if 'lucro' in c.lower() or 'profit' in c.lower()][0]
            col_staff = [c for c in df_pareto.columns if 'staff' in c.lower() or 'recursos' in c.lower()][0]
            col_desc = [c for c in df_pareto.columns if 'desc' in c.lower()][0]

            col_opt1, col_opt2 = st.columns([2, 1])
            
            with col_opt1:
                # Gráfico 3D com escala de cor 'Blues' (mais sóbria que Viridis)
                fig_3d = px.scatter_3d(
                    df_pareto, x=col_staff, y=col_desc, z=col_lucro,
                    color=col_lucro, color_continuous_scale='Blues',
                    title="Mapeamento do Espaço de Soluções",
                    labels={col_staff: "Equipa Semanal", col_desc: "Desconto Médio (%)", col_lucro: "Retorno Previsto ($)"}
                )
                fig_3d.update_layout(scene=dict(
                    xaxis_title='Volume de Recursos Humanos',
                    yaxis_title='Variável de Preço (%)',
                    zaxis_title='Projeção Financeira'
                ), margin=dict(l=0, r=0, b=0, t=40))
                st.plotly_chart(fig_3d, use_container_width=True, theme="streamlit")
                
            with col_opt2:
                best_profit = df_pareto.loc[df_pareto[col_lucro].idxmax()]
                st.markdown("### Parâmetros Ótimos")
                st.metric("Retorno Máximo Estimado", f"${best_profit[col_lucro]:,.2f}")
                st.metric("Alocação de Pessoal", f"{int(best_profit[col_staff])} elementos")
                st.metric("Ajuste de Preço Médio", f"{best_profit[col_desc]:.2f}%")
                
            st.markdown("### Resumo das Alternativas")
            df_top10 = df_pareto.sort_values(by=col_lucro, ascending=False).head(10).reset_index(drop=True)
            st.dataframe(df_top10, use_container_width=True)
            
        else:
            st.warning(f"Ficheiro de otimização ausente para a localização selecionada: {loja_sel.capitalize()}.")

else:
    st.error("Falha no carregamento do dataset principal. Verifique o diretório de dados processados.")