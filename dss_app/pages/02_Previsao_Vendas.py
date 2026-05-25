"""
Previsao de Vendas — Pagina 02
DSS USA Stores

Ecra operacional do gestor: previsoes diarias com banda de incerteza
de Poisson, metricas de desempenho e tabela de planeamento semanal.
"""
import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go

sys.dont_write_bytecode = True

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Previsao de Vendas — DSS USA Stores",
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

  div[data-testid="metric-container"] {
    border: 1px solid #E2E8F0; border-radius: 8px; padding: 14px 18px;
  }
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
    max-width: 820px; margin-bottom: 18px;
  }
  .model-badge {
    display: inline-block; padding: 4px 12px; border-radius: 20px;
    font-size: 0.75rem; font-weight: 700;
    background: rgba(30,58,138,0.08); color: #1E3A8A;
    letter-spacing: 0.03em;
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
_fc_dir    = os.path.join(_root_dir, "results", "02_Forecasting_Report")
_data_dir  = os.path.join(_root_dir, "data", "processed")

LOJAS = ["Baltimore", "Lancaster", "Philadelphia", "Richmond"]

CENARIOS_PREF = ["C_Context_Expert", "B_Sales_Dynamics", "A_Temporal_Base"]

DIAS_PT = {
    0: "Domingo", 1: "Segunda", 2: "Terca", 3: "Quarta",
    4: "Quinta", 5: "Sexta", 6: "Sabado",
}

# ── Loaders ────────────────────────────────────────────────────────────────
@st.cache_data
def carregar_metricas(fc_dir: str, loja: str) -> pd.DataFrame | None:
    path = os.path.join(fc_dir, loja, "store_metrics.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


@st.cache_data
def carregar_previsoes(fc_dir: str, loja: str) -> tuple[pd.DataFrame | None, str]:
    """Carrega forecast_values.csv do melhor cenario disponivel."""
    for cenario in CENARIOS_PREF:
        path = os.path.join(fc_dir, loja, cenario, "forecast_values.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, parse_dates=["Date"])
            return df, cenario
    return None, ""


@st.cache_data
def carregar_clientes(data_dir: str, loja: str) -> pd.DataFrame | None:
    path = os.path.join(data_dir, f"{loja.lower()}_processed.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path, parse_dates=["Date"])


def melhor_modelo(df_metricas: pd.DataFrame | None) -> str:
    if df_metricas is None or df_metricas.empty:
        return "Ensemble (Top-3 Experts)"
    sub = df_metricas[~df_metricas["Model"].str.contains("Naive", na=False)]
    if sub.empty:
        return df_metricas.iloc[0]["Model"]
    return sub.loc[sub["MAPE"].idxmin(), "Model"]


# ── Barra Lateral ──────────────────────────────────────────────────────────
st.sidebar.markdown(
    '<div style="font-size:0.88rem;font-weight:700;color:#E2E8F0;'
    'letter-spacing:0.04em;margin-bottom:16px;">DSS USA STORES</div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div style="font-size:0.70rem;color:#64748B;letter-spacing:0.08em;'
    'text-transform:uppercase;font-weight:700;margin-bottom:12px;">Painel de Controlo</div>',
    unsafe_allow_html=True,
)

loja_sel = st.sidebar.selectbox("Loja", LOJAS, index=0)

# carregar dados para alimentar o selector de semanas
df_prev, cenario_ativo = carregar_previsoes(_fc_dir, loja_sel)

semana_idx = 0
semanas_disp: list[str] = []
if df_prev is not None and not df_prev.empty:
    df_prev = df_prev.sort_values("Date").reset_index(drop=True)
    # agrupar em janelas de 7 dias
    datas = df_prev["Date"].dt.date.unique()
    semanas = [datas[i : i + 7] for i in range(0, len(datas), 7) if len(datas[i : i + 7]) == 7]
    if semanas:
        semanas_disp = [
            f"Semana {i+1}: {s[0].strftime('%d/%m/%Y')} — {s[-1].strftime('%d/%m/%Y')}"
            for i, s in enumerate(semanas)
        ]
        semana_idx = st.sidebar.selectbox(
            "Semana de analise",
            range(len(semanas_disp)),
            format_func=lambda i: semanas_disp[i],
            index=max(0, len(semanas_disp) - 4),
        )

st.sidebar.markdown("---")
st.sidebar.caption(f"Loja: {loja_sel}")
st.sidebar.caption(f"Cenario: {cenario_ativo or 'N/A'}")
st.sidebar.caption("Banda de incerteza: Distribuicao de Poisson")
st.sidebar.caption("Intervalo: media +/- sqrt(procura)")

# ── Cabecalho da pagina ────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 36px 0 20px 0; border-bottom: 2px solid #1E3A8A; margin-bottom: 32px;">
    <div style="font-size: 1.85rem; font-weight: 800; color: #0F172A;
         letter-spacing: -0.03em; line-height: 1.2; margin-bottom: 10px;">
        Previsao de Vendas
    </div>
    <div style="font-size: 0.88rem; color: #64748B; max-width: 760px; line-height: 1.65;">
        Ecra operacional do gestor de loja. Confronto entre valores historicos e previstos
        para a semana selecionada, com banda de incerteza probabilistica (Poisson)
        para suporte ao planeamento de staff e stock.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Validacao ──────────────────────────────────────────────────────────────
if df_prev is None or df_prev.empty:
    st.error(
        f"Dados de previsao nao encontrados para {loja_sel}. "
        "Execute: `python main_pipeline.py`"
    )
    st.stop()

if not semanas_disp:
    st.warning("Sem semanas completas de previsao disponíveis.")
    st.stop()

# ── Filtrar semana selecionada ─────────────────────────────────────────────
datas_semana = semanas[semana_idx]
df_semana = df_prev[df_prev["Date"].dt.date.isin(datas_semana)].copy()
df_semana = df_semana.sort_values("Date").reset_index(drop=True)

# coluna de modelo vencedor: preferir Ensemble se existir, senao melhor MAPE
df_metricas = carregar_metricas(_fc_dir, loja_sel)
modelo_vencedor = melhor_modelo(df_metricas)

# coluna de previsao: usar modelo vencedor se disponivel, senao qualquer coluna numerica
if modelo_vencedor in df_semana.columns:
    col_prev = modelo_vencedor
elif "Ensemble (Top-3 Experts)" in df_semana.columns:
    col_prev = "Ensemble (Top-3 Experts)"
else:
    num_cols = [c for c in df_semana.columns if c not in ("Date", "Actual") and
                pd.api.types.is_numeric_dtype(df_semana[c])]
    col_prev = num_cols[0] if num_cols else None

# coluna de clientes: tentar cruzar com dados processados
df_clientes = carregar_clientes(_data_dir, loja_sel)
clientes_semana = None
if df_clientes is not None and "Num_Customers" in df_clientes.columns:
    df_clientes["Date"] = pd.to_datetime(df_clientes["Date"])
    mask = df_clientes["Date"].dt.date.isin(datas_semana)
    clientes_semana = df_clientes.loc[mask, "Num_Customers"].values

# ── Metricas no topo ───────────────────────────────────────────────────────
m1, m2, m3 = st.columns(3, gap="medium")

total_clientes = int(clientes_semana.sum()) if clientes_semana is not None and len(clientes_semana) > 0 else None
pico_clientes  = int(clientes_semana.max()) if clientes_semana is not None and len(clientes_semana) > 0 else None

with m1:
    st.metric(
        "Total de Clientes na Semana",
        f"{total_clientes:,}" if total_clientes is not None else "N/D",
        help="Soma do trafego real de clientes nos 7 dias",
    )

with m2:
    st.metric(
        "Pico Maximo Diario",
        f"{pico_clientes:,}" if pico_clientes is not None else "N/D",
        help="Dia com maior afluencia de clientes na semana",
    )

with m3:
    mape_str = ""
    if df_metricas is not None:
        row = df_metricas[df_metricas["Model"] == modelo_vencedor]
        if not row.empty:
            mape_str = f"MAPE: {row.iloc[0]['MAPE']:.1f}%"
    st.metric(
        "Modelo Vencedor Ativo",
        modelo_vencedor,
        delta=mape_str or None,
        help="Modelo com menor MAPE no conjunto de validacao",
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Grafico de Previsao com Banda de Poisson ───────────────────────────────
st.markdown('<div class="section-title">Previsao Semanal — Real vs. Previsto</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-desc">'
    'Comparacao entre os valores historicos de vendas (cinzento escuro) e a previsao '
    'do modelo vencedor (azul corporativo). A area sombreada representa o intervalo '
    'de incerteza baseado na distribuicao de Poisson: media &plusmn; &radic;(procura), '
    'quantificando o risco de sobre ou subabastecimento.'
    '</div>',
    unsafe_allow_html=True,
)

if col_prev is not None and "Actual" in df_semana.columns:
    datas_eixo = df_semana["Date"].dt.strftime("%a %d/%m").tolist()
    y_real     = df_semana["Actual"].values
    y_prev_val = df_semana[col_prev].values.astype(float)
    y_std      = np.sqrt(np.abs(y_prev_val))

    fig_fc = go.Figure()

    # Banda de incerteza (Poisson)
    fig_fc.add_trace(go.Scatter(
        x=datas_eixo + datas_eixo[::-1],
        y=list(y_prev_val + y_std) + list((y_prev_val - y_std)[::-1]),
        fill="toself",
        fillcolor="rgba(30,58,138,0.10)",
        line=dict(width=0),
        name="Incerteza Poisson",
        showlegend=True,
        hoverinfo="skip",
    ))

    # Linha de previsao
    fig_fc.add_trace(go.Scatter(
        x=datas_eixo,
        y=y_prev_val,
        mode="lines+markers",
        name=f"Previsto ({col_prev})",
        line=dict(color="#1E3A8A", width=2.8),
        marker=dict(size=8, color="#1E3A8A"),
    ))

    # Linha de valores reais
    fig_fc.add_trace(go.Scatter(
        x=datas_eixo,
        y=y_real,
        mode="lines+markers",
        name="Real (Historico)",
        line=dict(color="#334155", width=2, dash="dot"),
        marker=dict(size=7, color="#334155", symbol="square"),
    ))

    fig_fc.update_layout(
        height=380,
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        margin=dict(t=20, b=40, l=10, r=20),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1,
                    font=dict(color="#0F172A", size=11)),
        xaxis=dict(tickfont=dict(color="#0F172A", size=11), gridcolor="#F1F5F9"),
        yaxis=dict(title=dict(text="Vendas ($)", font=dict(color="#0F172A", size=11)),
                   tickfont=dict(color="#0F172A", size=10), gridcolor="#F1F5F9",
                   zeroline=False, tickformat="$,.0f"),
        hovermode="x unified",
    )
    st.plotly_chart(fig_fc, use_container_width=True)

else:
    st.info(
        f"Coluna de previsao '{col_prev}' ou 'Actual' nao encontrada "
        f"para {loja_sel} / {cenario_ativo}."
    )

# ── Tabela de Previsao Diaria ──────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-label">Escala de Previsao Diaria</div>', unsafe_allow_html=True)

if col_prev is not None and "Actual" in df_semana.columns:
    y_prev_arr = df_semana[col_prev].values.astype(float)
    y_real_arr = df_semana["Actual"].values.astype(float)
    desvio     = np.sqrt(np.abs(y_prev_arr))
    erro_abs   = np.abs(y_real_arr - y_prev_arr)

    df_tabela = pd.DataFrame({
        "Dia":             df_semana["Date"].dt.strftime("%A, %d/%m").tolist(),
        "Real ($)":        [f"${v:,.0f}" for v in y_real_arr],
        "Previsto ($)":    [f"${v:,.0f}" for v in y_prev_arr],
        "Incerteza (+/-)": [f"${v:,.0f}" for v in desvio],
        "Erro Absoluto":   [f"${v:,.0f}" for v in erro_abs],
        "FDS":             df_semana["Date"].dt.dayofweek.map(
                               lambda d: "S" if d >= 5 else "").tolist(),
    })

    st.dataframe(df_tabela, hide_index=True, use_container_width=True)

    # mape da semana
    mape_semana = float(np.mean(erro_abs / np.abs(y_real_arr + 1e-9)) * 100)
    st.markdown(f"""
    <div class="insight-box">
        MAPE desta semana: <strong>{mape_semana:.1f}%</strong> &nbsp;&mdash;&nbsp;
        Modelo: <strong>{col_prev}</strong> &nbsp;&mdash;&nbsp;
        Cenario: <strong>{cenario_ativo}</strong>
    </div>
    """, unsafe_allow_html=True)

# ── Rodape ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dss-footer">
    DSS &nbsp;|&nbsp; Previsao de Vendas &nbsp;|&nbsp;
    Banda de incerteza: Distribuicao de Poisson (&plusmn;&radic;procura)
</div>
""", unsafe_allow_html=True)
