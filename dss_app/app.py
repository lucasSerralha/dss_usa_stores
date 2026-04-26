import streamlit as st
import sys
import os
import pandas as pd

# Bloqueio de criação de pastas __pycache__ (Manter repositório imaculado)
sys.dont_write_bytecode = True
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import numpy as np

# Configuração da página e estética (Tema Escuro/Executivo)
st.set_page_config(page_title="USA Stores Forecasting DSS", page_icon="📈", layout="wide")

# Estilo CSS customizado para KPIs de alto impacto e estética premium
st.markdown("""
    <style>
    /* Estilo Base para as Métricas */
    .stMetric { 
        padding: 20px; 
        border-radius: 12px; 
        border: 1px solid #30363d;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Estilo para Subtópicos: Quadrado Arredondado Cinzento sem Fundo */
    .sub-topic-box {
        display: inline-block;
        padding: 6px 14px;
        border-radius: 10px;
        border: 1px solid #30363d;
        font-weight: 600;
        font-size: 0.95rem;
        margin-bottom: 12px;
    }
    
    @media (prefers-color-scheme: light) {
        .sub-topic-box {
            border: 1px solid #d1d5da;
        }
    }

    /* Custom Metric Card para Ganho de Precisão (Lado a Lado) */
    .metric-value-row {
        display: flex;
        align-items: baseline;
        gap: 10px;
    }
    .metric-delta {
        font-size: 0.9rem;
        color: #238636;
        background: rgba(35, 134, 54, 0.1);
        padding: 2px 6px;
        border-radius: 8px;
        font-weight: bold;
    }
    
    h1, h2, h3 { font-family: 'Inter', sans-serif; font-weight: 700; }
    </style>
""", unsafe_allow_html=True)

# Helper para carregar dados
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

df_history = load_historical_data()
df_master = load_master_report()

if df_history is not None:
    # --- SIDEBAR PROFISSIONAL ---
    st.sidebar.title("💎 DSS Control Panel")
    lojas_formatadas = [l.capitalize() for l in sorted(df_history['store_id'].unique().tolist())]
    loja_sel_raw = st.sidebar.selectbox("🎯 Selecione a Loja Alvo:", lojas_formatadas)
    loja_sel = loja_sel_raw.lower()
    
    if df_master is not None:
        cenarios = sorted(df_master['Experiment'].unique().tolist())
        cenario_sel = st.sidebar.radio("🧪 Cenário de Experimentação:", cenarios, index=len(cenarios)-1)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📊 Status do Sistema")
    st.sidebar.success("Dados: Sincronizados")
    st.sidebar.info(f"Loja Ativa: {loja_sel.capitalize()}")

    # --- HEADER & KPIs ---
    st.title(f"🚀 USA Stores Ultra Analytics: {loja_sel.capitalize()}")
    st.markdown(f"🔬 **Framework de Decisão Inteligente** |  `{cenario_sel}`")
    
    df_loja = df_history[df_history['store_id'] == loja_sel]
    
    col_kpi1, col_kpi2, col_kpi3, col_kpi4 = st.columns(4)
    col_kpi1.metric("Vendas Totais", f"${df_loja['Sales'].sum()/1e6:.2f}M")
    
    # Cálculo de Ganho de Fidedignidade
    if df_master is not None:
        # Filtragem das métricas para a loja e cenário selecionados
        store_metrics = df_master[(df_master['Store'].str.lower() == loja_sel.lower()) & (df_master['Experiment'] == cenario_sel)]
        if not store_metrics.empty:
            rmse_min = store_metrics['RMSE'].min()
            # Cálculo comparativo com o Baseline (Seasonal Naive)
            rmse_naive = store_metrics[store_metrics['Model'] == 'Seasonal Naive']['RMSE'].values[0]
            ganho = ((rmse_naive - rmse_min) / rmse_naive) * 100
            
            # Bloco KPI: Ganho de Precisão (Lado a Lado)
            with col_kpi2:
                st.markdown(f"""
                    <div class="stMetric">
                        <p style="font-size: 14px; margin-bottom: 5px; opacity: 0.8;">Ganho de Precisão</p>
                        <div class="metric-value-row">
                            <span style="font-size: 28px; font-weight: 700;">{ganho:.1f}%</span>
                            <span class="metric-delta">↑ vs Baseline</span>
                        </div>
                    </div>
                """, unsafe_allow_html=True)
            
            col_kpi3.metric("Melhor RMSE Diário", f"${rmse_min:,.0f}")
            
            # Cálculo da Fidelidade (100% - Erro MAPE)
            fidelidade = 100 - store_metrics['MAPE'].min()
            col_kpi4.metric("Fidelidade (Acurácia)", f"{fidelidade:.1f}%")

    # --- NOVO POSICIONAMENTO DO EXPANDER (Com tom cinza sidebar) ---
    st.markdown("""
        <style>
        .streamlit-expanderHeader {
            background-color: rgba(151, 166, 195, 0.15) !important;
            border-radius: 8px 8px 0 0 !important;
        }
        .streamlit-expanderContent {
            background-color: rgba(151, 166, 195, 0.05) !important;
            border-radius: 0 0 8px 8px !important;
            border: 1px solid rgba(151, 166, 195, 0.1) !important;
        }
        </style>
    """, unsafe_allow_html=True)

    with st.expander("📝 Sobre este Cenário de Previsão", expanded=True):
        descricoes = {
            "A_Temporal_Base": "Análise focada em padrões cíclicos semanais e mensais. Ideal para operações estáveis sem grandes variações externas.",
            "B_Sales_Dynamics": "Integração da dinâmica de vendas de curto prazo (Lags de 1 dia). Captura mudanças rápidas no comportamento do consumidor.",
            "C_Context_Expert": "O 'Santo Graal' da previsão: Combina histórico, dinâmica de vendas e contexto externo (Promoções, Feriados e Eventos)."
        }
        st.write(descricoes.get(cenario_sel, "Cenário customizado carregado."))

    st.markdown("---")

    # --- ULTRA-TABS NAVIGATION ---
    tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Previsão de Vendas", "🔍 Diagnóstico Técnico", "🧬 Decomposição Temporal", "🧠 Inteligência de IA", "🎲 Modelos de Regras"])

    # --- TAB 1: PREVISÃO (CLEAN & INTERACTIVE) ---
    with tab1:
        st.subheader("📈 Previsão de Vendas Interativa")
        st.markdown('<p style="font-style: italic; margin-top: -10px; margin-bottom: -10px; opacity: 0.8;">Análise comparativa entre a procura real histórica e a capacidade de antecipação dos nossos modelos de Inteligência Artificial.</p>', unsafe_allow_html=True)
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        path_raw = os.path.join(script_dir, f"../results/02_Forecasting_Report/{loja_sel.capitalize()}/{cenario_sel}/forecast_values.csv")
        
        if os.path.exists(path_raw):
            df_plot = pd.read_csv(path_raw)
            df_plot['Date'] = pd.to_datetime(df_plot['Date'])
            melhor_nome = store_metrics.loc[store_metrics['RMSE'].idxmin(), 'Model']
            
            fig = go.Figure()
            # Actual Data - Cor Universal (Azul Vibrante) para contraste em Claro/Escuro
            fig.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot['Actual'], name='REALIDADE (Actual)',
                                    line=dict(color='#4A90E2', width=4), mode='lines+markers'))
            
            # Prediction Data
            models = [c for c in df_plot.columns if c not in ['Date', 'Actual']]
            # Paleta Executiva: Cores sóbrias e profissionais
            colors = ['#E67E22', '#1ABC9C', '#9B59B6', '#34495E', '#7F8C8D', '#D35400']
            for i, m in enumerate(models):
                is_best = (m == melhor_nome)
                fig.add_trace(go.Scatter(x=df_plot['Date'], y=df_plot[m], name=m,
                                        line=dict(dash='dash' if not is_best else 'solid', 
                                                  width=3 if is_best else 1.5, 
                                                  color=colors[i % len(colors)]),
                                        visible=True if is_best else 'legendonly'))
            
            fig.update_layout(hovermode="x unified", 
                              xaxis_title="Data", yaxis_title="Vendas ($)",
                              margin=dict(t=80, b=10, l=10, r=10),
                              legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, use_container_width=True, theme="streamlit")
            st.info(f"🏆 Padrão: Apenas o melhor modelo (**{melhor_nome}**) é mostrado inicialmente. Clique na legenda para comparar com outros.")
        else:
            st.warning("Dados brutos de previsão não encontrados.")

    # --- TAB 2: DIAGNÓSTICO (RESIDUAIS) ---
    with tab2:
        st.subheader("🔍 Análise de Erros e Resíduos")
        st.markdown('<p style="font-style: italic; margin-top: -10px; margin-bottom: 5px; opacity: 0.8;">Avaliação estatística da estabilidade do modelo e validação da ausência de viés sistemático nas projeções.</p>', unsafe_allow_html=True)
        
        if os.path.exists(path_raw):
            df_err = df_plot.copy()
            df_err['Erro'] = df_err[melhor_nome] - df_err['Actual']
            
            col_err1, col_err2 = st.columns(2)
            
            with col_err1:
                fig_hist = px.histogram(df_err, x="Erro", nbins=20, 
                                       title="Distribuição dos Erros de Previsão (Histograma)",
                                       labels={'Erro': 'Erro ($)', 'count': 'Frequência'},
                                       color_discrete_sequence=['#238636'])
                fig_hist.update_layout(margin=dict(t=80, b=20, l=10, r=10))
                st.plotly_chart(fig_hist, use_container_width=True, theme="streamlit")
                st.caption("Um histograma centrado no zero indica um modelo imparcial (sem viés).")
                
            with col_err2:
                fig_scat = px.scatter(df_err, x="Actual", y=melhor_nome, trendline="ols",
                                     title="Dispersão: Real vs Previsto", 
                                     labels={'Actual': 'Real ($)', melhor_nome: 'Previsto ($)'})
                fig_scat.update_layout(margin=dict(t=80, b=20, l=10, r=10))
                st.plotly_chart(fig_scat, use_container_width=True, theme="streamlit")
                st.caption("Quanto mais próximos os pontos da linha diagonal, melhor é a fidelidade do modelo.")

    # --- TAB 3: DECOMPOSIÇÃO (TREND/SEASONALITY) ---
    with tab3:
        st.subheader("🧬 Anatomia das Vendas")
        st.markdown('<p style="font-style: italic; margin-top: -10px; margin-bottom: 10px; opacity: 0.8;">Decomposição analítica dos componentes de tendência e padrões sazonais que regem o comportamento das vendas.</p>', unsafe_allow_html=True)
        
        path_comp = os.path.join(script_dir, f"../results/02_Forecasting_Report/{loja_sel.capitalize()}/{cenario_sel}/prophet_components.csv")
        
        if os.path.exists(path_comp):
            df_comp = pd.read_csv(path_comp)
            df_comp['ds'] = pd.to_datetime(df_comp['ds'])
            
            col_c1, col_c2 = st.columns(2)
            
            with col_c1:
                fig_trend = px.line(df_comp, x='ds', y='trend', title="Tendência de Longo Prazo",
                                   labels={'ds': 'Data', 'trend': 'Tendência ($)'},
                                   line_shape='spline', color_discrete_sequence=['#f1c40f'])
                fig_trend.update_layout(margin=dict(t=60, b=10, l=10, r=10))
                st.plotly_chart(fig_trend, use_container_width=True, theme="streamlit")
                
            with col_c2:
                if 'weekly' in df_comp.columns:
                    # Extrair um ciclo semanal (7 dias) e traduzir
                    df_weekly = df_comp.tail(7).copy()
                    dias_pt = {
                        'Monday': 'Segunda', 'Tuesday': 'Terça', 'Wednesday': 'Quarta',
                        'Thursday': 'Quinta', 'Friday': 'Sexta', 'Saturday': 'Sábado', 'Sunday': 'Domingo'
                    }
                    df_weekly['Dia'] = df_weekly['ds'].dt.day_name().map(dias_pt)
                    fig_week = px.bar(df_weekly, x='Dia', y='weekly', title="Padrão de Vendas: Ciclo Semanal",
                                     labels={'Dia': 'Dia da Semana', 'weekly': 'Desvio na Venda ($)'},
                                     color='weekly', color_continuous_scale='RdYlGn')
                    fig_week.update_layout(margin=dict(t=60, b=10, l=10, r=10))
                    st.plotly_chart(fig_week, use_container_width=True, theme="streamlit")
                else:
                    st.info("Sazonalidade não detetada para este período.")
        else:
            st.info("Gráficos de decomposição disponíveis na próxima execução do pipeline.")

    # --- TAB 4: INTELIGÊNCIA (FEATURES) ---
    with tab4:
        st.subheader("🧠 Porquê estas vendas? (Drivers de IA)")
        st.markdown('<p style="font-style: italic; margin-top: -10px; margin-bottom: 10px; opacity: 0.8;">Interpretabilidade Algorítmica (XAI): Identificação dos principais drivers que influenciam as projeções. Notar o impacto da variável <b>days_to_next_holiday</b> na antecipação de picos de consumo.</p>', unsafe_allow_html=True)
        
        path_feat = os.path.join(script_dir, f"../results/02_Forecasting_Report/{loja_sel.capitalize()}/{cenario_sel}/feature_importance.csv")
        
        if os.path.exists(path_feat):
            df_feat = pd.read_csv(path_feat).head(10)
            fig_feat = px.bar(df_feat, x='Importance', y='Feature', orientation='h',
                              title=f"Importância das Variáveis ({loja_sel.capitalize()})",
                              labels={'Feature': 'Variável', 'Importance': 'Importância'},
                              color='Importance', color_continuous_scale='Blues')
            fig_feat.update_layout(yaxis={'categoryorder':'total ascending'}, margin=dict(t=60, b=10, l=10, r=10))
            st.plotly_chart(fig_feat, use_container_width=True, theme="streamlit")
        else:
            st.warning("Análise de importância não disponível.")

    # --- TAB 5: MODELOS DE REGRAS PROBABILÍSTICOS ---
    with tab5:
        st.subheader("🎲 Modelos de Regras Probabilísticos")
        st.markdown(
            '<p style="font-style: italic; margin-top: -10px; margin-bottom: 10px; opacity: 0.8;">' +
            'Três modelos estatísticos baseados em regras de comportamento: Poisson (chegada de clientes), '
            'Gaussiano (variabilidade das vendas) e Logístico (conversão via desconto).</p>',
            unsafe_allow_html=True
        )

        # Bootstrap do path para importar os modelos
        _app_dir = os.path.dirname(os.path.abspath(__file__))
        _src_dir = os.path.abspath(os.path.join(_app_dir, "..", "src"))
        if _src_dir not in sys.path:
            sys.path.insert(0, _src_dir)

        # ── Filtros: mostrar/esconder cada modelo ─────────────────────────────
        st.markdown("""
            <style>
            div[data-testid="stCheckbox"] {
                border: 1px solid #30363d;
                border-radius: 10px;
                padding: 10px 16px;
                transition: border-color 0.2s, background 0.2s;
                cursor: pointer;
            }
            div[data-testid="stCheckbox"]:hover {
                border-color: #4A90E2;
                background: rgba(74, 144, 226, 0.06);
            }
            div[data-testid="stCheckbox"] label {
                font-weight: 600;
                font-size: 0.95rem;
                white-space: nowrap;
            }
            </style>
        """, unsafe_allow_html=True)

        col_cb1, col_cb2, col_cb3, _ = st.columns([2, 2, 2, 1])
        with col_cb1:
            show_poisson  = st.checkbox("📈 Modelo 1 — Poisson",   value=True)
        with col_cb2:
            show_gauss    = st.checkbox("📊 Modelo 2 — Gaussiano", value=True)
        with col_cb3:
            show_logistic = st.checkbox("🎯 Modelo 3 — Logístico", value=True)

        st.markdown("---")

        try:
            import numpy as np
            from optimization.probabilistic_models import (
                PoissonArrivalModel, GaussianSalesModel,
                LogisticConversionModel, compare_with_forecasting
            )

            # Carregar dados da loja seleccionada
            _data_path = os.path.abspath(
                os.path.join(_app_dir, "..", "data", "processed", f"{loja_sel}_processed.csv")
            )

            if not os.path.exists(_data_path):
                st.warning(f"Dados processados não encontrados para {loja_sel_raw}. Corre primeiro o pipeline.")
            else:
                df_loja_prob = pd.read_csv(_data_path, parse_dates=["Date"])

                # ── Treinar os 3 modelos ──────────────────────────────────────
                with st.spinner("A treinar os modelos probabilísticos..."):
                    pm = PoissonArrivalModel()
                    pm.fit(df_loja_prob, store=loja_sel)

                    gm = GaussianSalesModel()
                    gm.fit(df_loja_prob, store=loja_sel)

                    lm = LogisticConversionModel()
                    lm.fit(df_loja_prob, store=loja_sel)

                # Definir a semana de teste comum a todos os modelos
                split = int(len(df_loja_prob) * 0.8)
                test_week = df_loja_prob.iloc[split:split + 7].copy()
                _day_map = {0: "Seg", 1: "Ter", 2: "Qua", 3: "Qui", 4: "Sex", 5: "Sáb", 6: "Dom"}
                day_labels = [_day_map.get(int(r), f"Dia {i+1}") for i, r in enumerate(test_week["day_of_week"])]
                is_weekend_week = test_week["is_weekend"].astype(bool).tolist()

                # ── POISSON ───────────────────────────────────────────────────
                if show_poisson:
                    st.markdown('<div class="sub-topic-box">📈 Modelo 1 — Poisson: Previsão de Chegada de Clientes</div>', unsafe_allow_html=True)

                    # Prever para a semana de teste real
                    X_test_poisson = test_week[[f for f in pm._features if f in test_week.columns]]
                    lambdas = pm.predict_lambda(X_test_poisson)

                    col_p1, col_p2 = st.columns([2, 1])
                    with col_p1:
                        fig_poisson = go.Figure(go.Bar(
                            x=day_labels,
                            y=[round(l) for l in lambdas],
                            marker_color=["#FF9800" if w else "#4A90E2" for w in is_weekend_week],
                            text=[str(round(l)) for l in lambdas],
                            textposition="outside",
                        ))
                        fig_poisson.update_layout(
                            title="Clientes Previstos por Dia (próxima semana)",
                            xaxis_title="Dia", yaxis_title="Nº de Clientes (λ)",
                            margin=dict(t=60, b=10, l=10, r=10),
                            showlegend=False,
                        )
                        st.plotly_chart(fig_poisson, use_container_width=True, theme="streamlit")

                    with col_p2:
                        cmp_df = compare_with_forecasting(df_loja_prob, pm, store=loja_sel)
                        st.markdown("**Comparação com Baseline Lag-7**")
                        st.dataframe(cmp_df[["modelo", "MAE", "RMSE"]].rename(
                            columns={"modelo": "Método", "MAE": "MAE (clientes)", "RMSE": "RMSE"}
                        ), hide_index=True, use_container_width=True)
                        melhoria = round((1 - cmp_df.iloc[0]["MAE"] / cmp_df.iloc[1]["MAE"]) * 100, 1)
                        st.success(f"O Poisson é **{melhoria}% mais preciso** que o baseline histórico.")

                    if show_gauss or show_logistic:
                        st.markdown("---")

                # ── GAUSSIANO ─────────────────────────────────────────────────
                if show_gauss:
                    st.markdown('<div class="sub-topic-box">📊 Modelo 2 — Gaussiano: Variabilidade das Vendas</div>', unsafe_allow_html=True)

                    scenarios = gm.predict_scenarios(test_week)
                    lower_95, upper_95 = gm.confidence_interval(test_week, confidence=0.95)

                    fig_gauss = go.Figure()
                    # Usar nomes dos dias em vez de datas especificas
                    _day_map = {0: "Seg", 1: "Ter", 2: "Qua", 3: "Qui", 4: "Sex", 5: "Sáb", 6: "Dom"}
                    x_labels = [_day_map.get(int(r), f"Dia {i+1}") for i, r in
                                enumerate(test_week["day_of_week"])] if "day_of_week" in test_week.columns \
                                else [f"Dia {i+1}" for i in range(len(test_week))]
                    fig_gauss.add_trace(go.Scatter(x=x_labels, y=upper_95, mode="lines", line=dict(width=0), showlegend=False))
                    fig_gauss.add_trace(go.Scatter(
                        x=x_labels, y=lower_95, mode="lines", line=dict(width=0),
                        fill="tonexty", fillcolor="rgba(74,144,226,0.15)", name="IC 95%"
                    ))
                    fig_gauss.add_trace(go.Scatter(x=x_labels, y=scenarios["pessimistic"], mode="lines+markers",
                                                   line=dict(color="#E74C3C", dash="dot"), name="Pessimista"))
                    fig_gauss.add_trace(go.Scatter(x=x_labels, y=scenarios["realistic"], mode="lines+markers",
                                                   line=dict(color="#4A90E2", width=3), name="Realista"))
                    fig_gauss.add_trace(go.Scatter(x=x_labels, y=scenarios["optimistic"], mode="lines+markers",
                                                   line=dict(color="#2ECC71", dash="dot"), name="Otimista"))
                    
                    # Adicionar a linha do que "Realmente Aconteceu" (Vendas Reais)
                    target_col = "y" if "y" in test_week.columns else "Sales"
                    y_true = test_week[target_col].values
                    fig_gauss.add_trace(go.Scatter(x=x_labels, y=y_true, mode="lines+markers",
                                                   line=dict(color="#F1C40F", width=4), 
                                                   marker=dict(symbol="star", size=10, color="#2C3E50", line=dict(width=2, color="#F1C40F")),
                                                   name="Vendas Reais (Realidade)"))

                    fig_gauss.update_layout(
                        title=f"Cenários de Vendas — σ = ${gm.sigma:,.0f} | μ global = ${gm.mu_global:,.0f}",
                        xaxis_title="Dia da Semana", yaxis_title="Vendas ($)",
                        margin=dict(t=60, b=10, l=10, r=10),
                        hovermode="x unified",
                    )
                    st.plotly_chart(fig_gauss, use_container_width=True, theme="streamlit")

                    if show_logistic:
                        st.markdown("---")

                # ── LOGÍSTICO ─────────────────────────────────────────────────
                if show_logistic:
                    st.markdown('<div class="sub-topic-box">🎯 Modelo 3 — Logístico: Probabilidade de Conversão via Desconto</div>', unsafe_allow_html=True)

                    pr_range = np.linspace(0.0, 0.30, 60)
                    p_weekday = lm.conversion_curve(pr_range, is_weekend=0)
                    p_weekend = lm.conversion_curve(pr_range, is_weekend=1)
                    p_holiday = lm.conversion_curve(pr_range, is_weekend=0, is_holiday=1, days_to_next_holiday=0)

                    col_l1, col_l2 = st.columns([2, 1])
                    with col_l1:
                        fig_logit = go.Figure()
                        fig_logit.add_trace(go.Scatter(x=pr_range * 100, y=p_weekday, mode="lines",
                                                       name="Dia de semana", line=dict(color="#4A90E2", width=3)))
                        fig_logit.add_trace(go.Scatter(x=pr_range * 100, y=p_weekend, mode="lines",
                                                       name="Fim-de-semana", line=dict(color="#FF9800", width=3)))
                        fig_logit.add_trace(go.Scatter(x=pr_range * 100, y=p_holiday, mode="lines",
                                                       name="Feriado", line=dict(color="#E74C3C", width=2, dash="dash")))
                        fig_logit.add_hline(y=0.5, line_dash="dot", line_color="grey", annotation_text="50% conversão")
                        fig_logit.update_layout(
                            title=f"Curva de Conversão — AUC-ROC = {lm.auc:.3f}",
                            xaxis_title="Desconto Aplicado (%)", yaxis_title="P(Conversão)",
                            yaxis=dict(tickformat=".0%", range=[0, 1]),
                            margin=dict(t=60, b=10, l=10, r=10),
                            hovermode="x unified",
                        )
                        st.plotly_chart(fig_logit, use_container_width=True, theme="streamlit")

                    with col_l2:
                        st.metric("AUC-ROC", f"{lm.auc:.3f}", help="1.0 = perfeito | 0.5 = aleatório")
                        st.metric("Threshold", f"${lm.threshold:,.1f}/cliente")
                        st.info(
                            "**Desconto tem retornos decrescentes.**\n\n"
                            f"Com 0% de desconto, P(conv) = {lm.conversion_curve(np.array([0.0]))[0]*100:.1f}%.\n"
                            f"Com 30% de desconto, P(conv) = {lm.conversion_curve(np.array([0.30]))[0]*100:.1f}%."
                        )

                if not show_poisson and not show_gauss and not show_logistic:
                    st.info("Seleciona pelo menos um modelo para visualizar.")

        except ImportError as e:
            st.error(f"Erro ao importar os modelos probabilísticos: {e}")
        except Exception as e:
            st.error(f"Erro inesperado: {e}")
            st.exception(e)

else:
    st.error("⚠️ Erro Crítico: Dados não encontrados.")
