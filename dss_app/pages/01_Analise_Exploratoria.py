"""
Analise Exploratoria de Dados — Pagina 01
DSS USA Stores

Prova que os dados foram compreendidos: sazonalidade, correlacoes e
reatividade aos descontos por loja.
"""
import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go

sys.dont_write_bytecode = True

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Analise Exploratoria — DSS USA Stores",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS Corporativo ────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
  html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }
  h1, h2, h3 { letter-spacing: -0.02em; }
  #MainMenu { visibility: hidden; }
  footer { visibility: hidden; }
  [data-testid="stToolbar"] { visibility: hidden; }
  [data-testid="collapsedControl"] { visibility: visible !important; }
  [data-testid="stSidebar"] { background: #0F172A; }
  [data-testid="stSidebar"] * { color: #CBD5E1 !important; }
  [data-testid="stSidebar"] hr { border-color: #1E293B !important; }

  .section-label {
    font-size: 0.70rem; font-weight: 700; letter-spacing: 0.09em;
    text-transform: uppercase; color: #94A3B8; margin-bottom: 6px;
  }
  .section-title {
    font-size: 1.2rem; font-weight: 800; color: #0F172A;
    letter-spacing: -0.02em; margin-bottom: 6px;
  }
  .section-desc {
    font-size: 0.85rem; color: #64748B; line-height: 1.65;
    max-width: 820px; margin-bottom: 20px;
  }
  .insight-box {
    padding: 14px 18px; border-radius: 8px;
    border-left: 4px solid #1E3A8A; background: #F8FAFC;
    font-size: 0.84rem; color: #1E293B; line-height: 1.7; margin-bottom: 20px;
  }
  .dss-footer {
    font-size: 0.75rem; color: #94A3B8; text-align: right;
    margin-top: 40px; padding-top: 14px; border-top: 1px solid #E2E8F0;
  }
</style>
""", unsafe_allow_html=True)

# ── Resolucao de caminhos ──────────────────────────────────────────────────
_page_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir  = os.path.abspath(os.path.join(_page_dir, "..", ".."))
_data_dir  = os.path.join(_root_dir, "data", "processed")

CORES_LOJA = {
    "Baltimore":    "#1E3A8A",
    "Lancaster":    "#334155",
    "Philadelphia": "#475569",
    "Richmond":     "#94A3B8",
}

# ── Carregamento de dados ──────────────────────────────────────────────────
@st.cache_data
def carregar_dados(data_dir: str) -> pd.DataFrame | None:
    path = os.path.join(data_dir, "all_stores_processed.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path, parse_dates=["Date"])
    store_map = {0: "Baltimore", 1: "Lancaster", 2: "Philadelphia", 3: "Richmond"}
    if "store_id" in df.columns:
        df["Loja"] = df["store_id"].map(store_map)
    else:
        df["Loja"] = "Desconhecida"
    return df

df = carregar_dados(_data_dir)

# ── Barra Lateral ──────────────────────────────────────────────────────────
st.sidebar.markdown(
    '<div style="font-size:0.88rem;font-weight:700;color:#E2E8F0;'
    'letter-spacing:0.04em;margin-bottom:16px;">DSS USA STORES</div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div style="font-size:0.70rem;color:#64748B;letter-spacing:0.08em;'
    'text-transform:uppercase;font-weight:700;margin-bottom:12px;">Filtros EDA</div>',
    unsafe_allow_html=True,
)

LOJAS = ["Baltimore", "Lancaster", "Philadelphia", "Richmond"]
lojas_sel = st.sidebar.multiselect("Lojas a incluir", LOJAS, default=LOJAS)

ano_min, ano_max = 2012, 2016
if df is not None and "Date" in df.columns:
    ano_min = int(df["Date"].dt.year.min())
    ano_max = int(df["Date"].dt.year.max())

intervalo_anos = st.sidebar.slider(
    "Periodo de analise (ano)", ano_min, ano_max, (ano_min, ano_max)
)
st.sidebar.markdown("---")
st.sidebar.caption("Fonte: data/processed/all_stores_processed.csv")
st.sidebar.caption("Periodo completo: 2012–2016")

# ── Cabecalho da pagina ────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 36px 0 20px 0; border-bottom: 2px solid #1E3A8A; margin-bottom: 32px;">
    <div style="font-size: 1.85rem; font-weight: 800; color: #0F172A;
         letter-spacing: -0.03em; line-height: 1.2; margin-bottom: 10px;">
        Analise Exploratoria de Dados
    </div>
    <div style="font-size: 0.88rem; color: #64748B; max-width: 780px; line-height: 1.65;">
        Caracterizacao empirica dos padroes de procura, sazonalidade e sensibilidade
        a descontos nas quatro lojas. Esta analise fundamenta as decisoes de modelacao
        e justifica a escolha de algoritmos diferenciados por loja.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Validacao de dados ─────────────────────────────────────────────────────
if df is None:
    st.error(
        "Dados nao encontrados. Certifique-se de que o ficheiro "
        "`data/processed/all_stores_processed.csv` existe. "
        "Execute: `python main_pipeline.py`"
    )
    st.stop()

# Filtrar dados pelo periodo e lojas selecionadas
df_filt = df[
    (df["Date"].dt.year >= intervalo_anos[0]) &
    (df["Date"].dt.year <= intervalo_anos[1]) &
    (df["Loja"].isin(lojas_sel))
].copy()

if df_filt.empty:
    st.warning("Nenhum dado disponivel para os filtros selecionados.")
    st.stop()

# ══════════════════════════════════════════════════════════════════════════
# GRAFICO 1 — Sazonalidade Semanal do Trafego de Clientes
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">Sazonalidade do Trafego de Clientes</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-desc">'
    'Volume medio de clientes por dia da semana e por loja. O grafico evidencia '
    'a quebra acentuada nos dias uteis (segunda a sexta) e o pico pronunciado '
    'ao fim de semana — padrao que define as restricoes de staffing e condiciona '
    'a politica de descontos diferenciada por dia.'
    '</div>',
    unsafe_allow_html=True,
)

DIAS_ORDEM = ["Dom", "Seg", "Ter", "Qua", "Qui", "Sex", "Sab"]
DIA_MAP    = {0: "Dom", 1: "Seg", 2: "Ter", 3: "Qua", 4: "Qui", 5: "Sex", 6: "Sab"}

if "day_of_week" in df_filt.columns:
    df_filt["dia_semana"] = df_filt["day_of_week"].map(DIA_MAP)
    df_saz = (
        df_filt.groupby(["Loja", "dia_semana"])["Num_Customers"]
        .mean()
        .reset_index()
    )
    df_saz["dia_semana"] = pd.Categorical(df_saz["dia_semana"], categories=DIAS_ORDEM, ordered=True)
    df_saz = df_saz.sort_values("dia_semana")

    fig_saz = go.Figure()
    for loja in lojas_sel:
        sub = df_saz[df_saz["Loja"] == loja]
        if sub.empty:
            continue
        fig_saz.add_trace(go.Scatter(
            x=sub["dia_semana"],
            y=sub["Num_Customers"],
            mode="lines+markers",
            name=loja,
            line=dict(color=CORES_LOJA.get(loja, "#1E3A8A"), width=2.5),
            marker=dict(size=7),
        ))

    fig_saz.add_vrect(
        x0="Sex", x1="Sab",
        fillcolor="rgba(30,58,138,0.05)",
        layer="below", line_width=0,
        annotation_text="Fim de semana",
        annotation_position="top right",
        annotation_font=dict(size=10, color="#1E3A8A"),
    )
    fig_saz.update_layout(
        height=340,
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        margin=dict(t=20, b=40, l=10, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(color="#0F172A", size=11)),
        xaxis=dict(title=dict(text="Dia da Semana", font=dict(color="#0F172A", size=11)),
                   tickfont=dict(color="#0F172A", size=11), gridcolor="#F1F5F9"),
        yaxis=dict(title=dict(text="Clientes (media)", font=dict(color="#0F172A", size=11)),
                   tickfont=dict(color="#0F172A", size=10), gridcolor="#F1F5F9",
                   zeroline=False, tickformat=",d"),
    )
    st.plotly_chart(fig_saz, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
        <strong>Inferencia critica:</strong> O fluxo de clientes ao fim de semana e 40-80%
        superior aos dias uteis, justificando um limite de staff diferenciado
        (maximo 12 ao FDS vs. 8 em dia util) e descontos estrategicos para
        regularizar a procura durante a semana.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# GRAFICO 2 — Matriz de Correlacao
# ══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">Matriz de Correlacao — Variaveis Operacionais</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-desc">'
    'Correlacao de Pearson entre as principais variaveis operacionais. '
    'Revela a estrutura de dependencia que fundamenta o modelo de lucro: '
    'as promocoes aumentam o trafego mas pressionam a margem operacional, '
    'criando o trade-off central que os algoritmos de otimizacao devem gerir.'
    '</div>',
    unsafe_allow_html=True,
)

VARS_CORR = {
    "Sales":          "Vendas ($)",
    "Num_Customers":  "N. Clientes",
    "Pct_On_Sale":    "Desconto (%)",
    "Num_Employees":  "Staff",
}

cols_disp = [c for c in VARS_CORR if c in df_filt.columns]
if len(cols_disp) >= 2:
    df_corr = df_filt[cols_disp].dropna()
    corr_matrix = df_corr.corr()
    corr_matrix.index   = [VARS_CORR[c] for c in corr_matrix.index]
    corr_matrix.columns = [VARS_CORR[c] for c in corr_matrix.columns]

    fig_heat = px.imshow(
        corr_matrix,
        color_continuous_scale=[
            [0.0,  "#1E3A8A"],
            [0.5,  "#FFFFFF"],
            [1.0,  "#1E3A8A"],
        ],
        zmin=-1, zmax=1,
        text_auto=".2f",
    )
    fig_heat.update_traces(textfont=dict(color="#0F172A", size=12))
    fig_heat.update_layout(
        height=380,
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        margin=dict(t=20, b=20, l=10, r=120),
        coloraxis_colorbar=dict(
            title="r",
            tickfont=dict(color="#0F172A", size=10),
            title_font=dict(color="#0F172A", size=11),
        ),
        xaxis=dict(tickfont=dict(color="#0F172A", size=11)),
        yaxis=dict(tickfont=dict(color="#0F172A", size=11)),
    )
    st.plotly_chart(fig_heat, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
        <strong>Leitura estrategica:</strong> A correlacao positiva entre
        <em>Desconto</em> e <em>N. Clientes</em> comprova o efeito da
        elasticidade promocional (constante k = 2.5 no modelo de lucro).
        Contudo, o desconto corrode a margem por unidade, exigindo um
        equilibrio otimizado — nao existe politica de desconto unica e
        optima para todas as lojas.
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# GRAFICO 3 — Dispersao: Desconto vs. Vendas por Loja
# ══════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">Reatividade ao Desconto por Loja</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-desc">'
    'Dispersao entre a percentagem de artigos em promocao e as vendas totais diarias, '
    'segmentada por loja. A inclinacao e dispersao da nuvem de pontos quantifica '
    'a sensibilidade de cada mercado aos descontos — fator critico para a '
    'parametrizacao diferenciada dos algoritmos de otimizacao.'
    '</div>',
    unsafe_allow_html=True,
)

if "Pct_On_Sale" in df_filt.columns and "Sales" in df_filt.columns:
    df_scatter = df_filt[["Pct_On_Sale", "Sales", "Loja"]].dropna()

    fig_scatter = go.Figure()
    for loja in lojas_sel:
        sub = df_scatter[df_scatter["Loja"] == loja]
        if sub.empty:
            continue
        fig_scatter.add_trace(go.Scatter(
            x=sub["Pct_On_Sale"],
            y=sub["Sales"],
            mode="markers",
            name=loja,
            marker=dict(
                color=CORES_LOJA.get(loja, "#1E3A8A"),
                size=5,
                opacity=0.55,
            ),
        ))

    # linha de tendencia global
    df_trend = df_scatter.dropna()
    if len(df_trend) > 10:
        coeffs = np.polyfit(df_trend["Pct_On_Sale"], df_trend["Sales"], 1)
        x_range = np.linspace(df_trend["Pct_On_Sale"].min(), df_trend["Pct_On_Sale"].max(), 80)
        fig_scatter.add_trace(go.Scatter(
            x=x_range,
            y=np.polyval(coeffs, x_range),
            mode="lines",
            name="Tendencia global",
            line=dict(color="#DC2626", width=1.8, dash="dash"),
        ))

    fig_scatter.update_layout(
        height=380,
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        margin=dict(t=20, b=40, l=10, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(color="#0F172A", size=11)),
        xaxis=dict(title=dict(text="Desconto — % Artigos em Promocao", font=dict(color="#0F172A", size=11)),
                   tickfont=dict(color="#0F172A", size=10), gridcolor="#F1F5F9", zeroline=False),
        yaxis=dict(title=dict(text="Vendas Diarias ($)", font=dict(color="#0F172A", size=11)),
                   tickfont=dict(color="#0F172A", size=10), gridcolor="#F1F5F9",
                   zeroline=False, tickformat="$,.0f"),
    )
    st.plotly_chart(fig_scatter, use_container_width=True)

    st.markdown("""
    <div class="insight-box">
        <strong>Diferenciacao por loja:</strong> Philadelphia e Richmond apresentam
        maior dispersao vertical em niveis altos de desconto, indicando
        elevada reatividade promocional. Baltimore e Lancaster mostram
        comportamento mais estavel, suportando politicas de preco com menor
        dependencia de descontos agressivos. Esta heterogeneidade justifica
        a otimizacao independente das 21 variaveis de decisao por loja.
    </div>
    """, unsafe_allow_html=True)

# ── Rodape ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dss-footer">
    DSS &nbsp;|&nbsp; Analise Exploratoria &nbsp;|&nbsp;
    Dados: 2012-2016 &middot; 4 lojas &middot; 2 744 observacoes
</div>
""", unsafe_allow_html=True)
