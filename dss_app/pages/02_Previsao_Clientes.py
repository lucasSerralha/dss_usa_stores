"""
Previsao de Clientes — Pagina 02
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
import plotly.express as px

sys.dont_write_bytecode = True

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Previsao de Clientes — DSS USA Stores",
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
_fc_dir    = os.path.join(_root_dir, "results", "02_Forecasting")
_data_dir  = os.path.join(_root_dir, "data", "processed")

LOJAS = ["Baltimore", "Lancaster", "Philadelphia", "Richmond"]

# Ordem de preferencia automatica (C Expert ganha 3/4 lojas)
CENARIOS_PREF = ["C_Context_Expert", "D_Full_Context", "B_Sales_Dynamics", "A_Temporal_Base"]

# Numero de features por cenario (para a Auditoria de Cenarios)
CENARIO_FEATURES = {
    "A_Temporal_Base":  6,   # calendario + 2 lags clientes
    "B_Sales_Dynamics": 6,   # calendario + lags vendas + roll mean
    "C_Context_Expert": 8,   # contexto negocio (holidays, promos, eventos)
    "D_Full_Context":  18,   # todas as features sem leakage
}

# Nome legivel do cenario (para o gestor de loja)
CENARIO_LABELS = {
    "A_Temporal_Base":  "Base Temporal",
    "B_Sales_Dynamics": "Dinamica de Vendas",
    "C_Context_Expert": "Especialista de Contexto",
    "D_Full_Context":   "Contexto Completo",
}

# ── Loaders ────────────────────────────────────────────────────────────────
@st.cache_data
def carregar_metricas(fc_dir: str, loja: str, cenario: str = "") -> pd.DataFrame | None:
    # Tenta primeiro o ficheiro do cenario activo (tem SARIMAX + Ensemble)
    if cenario:
        path_cenario = os.path.join(fc_dir, loja, cenario, "store_metrics.csv")
        if os.path.exists(path_cenario):
            return pd.read_csv(path_cenario)
    # Fallback: ficheiro legado ao nivel da loja (pode nao ter todos os modelos)
    path_loja = os.path.join(fc_dir, loja, "store_metrics.csv")
    return pd.read_csv(path_loja) if os.path.exists(path_loja) else None


@st.cache_data
def carregar_previsoes_cenario(fc_dir: str, loja: str, cenario: str) -> pd.DataFrame | None:
    """Carrega forecast_values.csv de um cenario especifico."""
    path = os.path.join(fc_dir, loja, cenario, "forecast_values.csv")
    if os.path.exists(path):
        return pd.read_csv(path, parse_dates=["Date"])
    return None


@st.cache_data
def carregar_todos_cenarios(fc_dir: str, loja: str) -> pd.DataFrame | None:
    """Carrega e consolida metricas do melhor modelo de cada cenario."""
    rows = []
    for cenario in CENARIOS_PREF:
        path = os.path.join(fc_dir, loja, cenario, "store_metrics.csv")
        if not os.path.exists(path):
            continue
        df = pd.read_csv(path)
        # Melhor modelo excluindo Seasonal Naive (baseline trivial)
        sub = df[~df["Model"].str.contains("Naive", na=False)]
        if sub.empty:
            continue
        best = sub.loc[sub["MAPE"].idxmin()]
        rows.append({
            "Cenario":       cenario,
            "Nº Features":   CENARIO_FEATURES.get(cenario, "?"),
            "Melhor Modelo": best["Model"],
            "MAPE (%)":      round(float(best["MAPE"]), 1),
            "RMSE":          round(float(best["RMSE"]), 1),
            "MAE":           round(float(best["MAE"]), 1),
        })
    return pd.DataFrame(rows) if rows else None


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

# Selecao automatica do melhor cenario disponivel (transparente para o gestor)
cenario_ativo = next(
    (c for c in CENARIOS_PREF
     if os.path.exists(os.path.join(_fc_dir, loja_sel, c, "forecast_values.csv"))),
    CENARIOS_PREF[0],
)

# Carregar previsoes do melhor cenario
df_prev = carregar_previsoes_cenario(_fc_dir, loja_sel, cenario_ativo)

semana_idx = 0
semanas_disp: list[str] = []
semanas: list = []
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

# ── Cabecalho da pagina ────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 36px 0 20px 0; border-bottom: 2px solid #1E3A8A; margin-bottom: 32px;">
    <div style="font-size: 1.85rem; font-weight: 800; color: #0F172A;
         letter-spacing: -0.03em; line-height: 1.2; margin-bottom: 10px;">
        Previsao de Clientes
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
        f"Dados de previsao nao encontrados para {loja_sel} / {cenario_ativo}. "
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

# coluna de modelo vencedor: melhor MAPE excluindo Naive
df_metricas = carregar_metricas(_fc_dir, loja_sel, cenario_ativo)
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
    'Comparacao entre o numero real de clientes (cinzento escuro) e a previsao '
    'do modelo vencedor (azul corporativo). A area sombreada representa o intervalo '
    'de incerteza baseado na distribuicao de Poisson: media &plusmn; &radic;(procura), '
    'quantificando o risco de sub ou sobredimensionamento de staff.'
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
        yaxis=dict(title=dict(text="Nº de Clientes", font=dict(color="#0F172A", size=11)),
                   tickfont=dict(color="#0F172A", size=10), gridcolor="#F1F5F9",
                   zeroline=False, tickformat=",.0f"),
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
        "Dia":                df_semana["Date"].dt.strftime("%A, %d/%m").tolist(),
        "Real (clientes)":    [f"{v:,.0f}" for v in y_real_arr],
        "Previsto (clientes)":[f"{v:,.0f}" for v in y_prev_arr],
        "Incerteza (+/-)":    [f"{v:,.0f}" for v in desvio],
        "Erro Absoluto":      [f"{v:,.0f}" for v in erro_abs],
        "FDS":                df_semana["Date"].dt.dayofweek.map(
                                  lambda d: "S" if d >= 5 else "").tolist(),
    })

    st.dataframe(df_tabela, hide_index=True, use_container_width=True)

    # mape da semana
    mape_semana = float(np.mean(erro_abs / np.abs(y_real_arr + 1e-9)) * 100)
    st.markdown(f"""
    <div class="insight-box">
        MAPE desta semana: <strong>{mape_semana:.1f}%</strong> &nbsp;&mdash;&nbsp;
        Modelo: <strong>{col_prev}</strong> &nbsp;&mdash;&nbsp;
        Cenario: <strong>{CENARIO_LABELS.get(cenario_ativo, cenario_ativo)}</strong>
    </div>
    """, unsafe_allow_html=True)

# ── Auditoria de Modelos ───────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-title">Auditoria de Modelos</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-desc">'
    'Comparacao completa dos 6 modelos de forecasting e do Ensemble no cenario ativo. '
    'Permite auditar qual o modelo mais fidedigno para cada loja e validar a escolha do sistema. '
    'MAPE: erro percentual medio. RMSE: penaliza erros grandes. Menor e sempre melhor.'
    '</div>',
    unsafe_allow_html=True,
)

if df_metricas is not None and not df_metricas.empty:
    # Ordenar por MAPE crescente (melhor primeiro)
    df_audit = df_metricas.copy().sort_values("MAPE").reset_index(drop=True)

    # Formatar colunas numericas
    df_audit_fmt = pd.DataFrame({
        "Modelo":   df_audit["Model"],
        "MAPE (%)": df_audit["MAPE"].round(1).astype(str) + "%",
        "RMSE":     df_audit["RMSE"].round(1),
        "MAE":      df_audit["MAE"].round(1),
        "Cenario":  df_audit.get("Experiment", cenario_ativo),
    })

    # Destacar o vencedor
    def _highlight_winner(row):
        if row["Modelo"] == modelo_vencedor:
            return ["background-color: #EFF6FF; font-weight: 700;"] * len(row)
        return [""] * len(row)

    st.dataframe(
        df_audit_fmt.style.apply(_highlight_winner, axis=1),
        hide_index=True,
        use_container_width=True,
    )

    # Grafico de barras — MAPE por modelo
    fig_audit = px.bar(
        df_audit.sort_values("MAPE"),
        x="Model", y="MAPE",
        color="MAPE",
        color_continuous_scale=[[0, "#1E3A8A"], [0.5, "#3B82F6"], [1, "#CBD5E1"]],
        labels={"Model": "Modelo", "MAPE": "MAPE (%)"},
        text=df_audit.sort_values("MAPE")["MAPE"].round(1).astype(str) + "%",
    )
    fig_audit.update_traces(textposition="outside")
    fig_audit.update_layout(
        height=320,
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        margin=dict(t=20, b=10, l=10, r=10),
        coloraxis_showscale=False,
        xaxis=dict(tickfont=dict(color="#0F172A", size=10)),
        yaxis=dict(title="MAPE (%)", tickfont=dict(color="#0F172A", size=10),
                   gridcolor="#F1F5F9"),
        showlegend=False,
    )
    st.plotly_chart(fig_audit, use_container_width=True)

    winner_row = df_audit[df_audit['Model'] == modelo_vencedor]
    winner_mape_str = f"{winner_row.iloc[0]['MAPE']:.1f}%" if not winner_row.empty else "N/A"

    if "Ensemble" in modelo_vencedor:
        nota = "O Ensemble combina as previsoes dos Top-3 modelos por RMSE, sendo mais robusto que qualquer modelo isolado."
    else:
        nota = f"O {modelo_vencedor} supera o Ensemble nesta loja — as suas previsoes individuais sao suficientemente precisas."

    st.markdown(f"""
    <div class="insight-box">
        Modelo vencedor: <strong>{modelo_vencedor}</strong>
        &nbsp;&mdash;&nbsp; MAPE: <strong>{winner_mape_str}</strong><br>
        {nota}
    </div>
    """, unsafe_allow_html=True)
else:
    st.info("Metricas de auditoria nao disponiveis para esta loja.")

# ── Auditoria de Cenarios ──────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-title">Auditoria de Cenarios</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-desc">'
    'Comparacao dos 4 cenarios de features testados. Para cada cenario mostra-se o melhor '
    'modelo e as suas metricas de erro. Permite validar se adicionar mais features melhora '
    'a precisao ou se o overfitting penaliza cenarios mais complexos (D vs C).'
    '</div>',
    unsafe_allow_html=True,
)

df_cenarios = carregar_todos_cenarios(_fc_dir, loja_sel)

if df_cenarios is not None and not df_cenarios.empty:
    # Ordenar por MAPE (igual ao grafico — elimina inconsistencia tabela vs grafico)
    df_cenarios = df_cenarios.sort_values("MAPE (%)").reset_index(drop=True)
    melhor_cenario = df_cenarios.iloc[0]["Cenario"]

    # Usar nomes legiveis na tabela (sem underscores tecnicas)
    df_cenarios_disp = df_cenarios.copy()
    df_cenarios_disp["Cenario"] = df_cenarios_disp["Cenario"].map(
        lambda c: CENARIO_LABELS.get(c, c)
    )
    melhor_label = CENARIO_LABELS.get(melhor_cenario, melhor_cenario)
    ativo_label  = CENARIO_LABELS.get(cenario_ativo, cenario_ativo)

    def _highlight_cenario(row):
        if row["Cenario"] == melhor_label:
            return ["background-color: #EFF6FF; font-weight: 700;"] * len(row)
        if row["Cenario"] == ativo_label and ativo_label != melhor_label:
            return ["background-color: #F0FDF4;"] * len(row)
        return [""] * len(row)

    st.dataframe(
        df_cenarios_disp.style.apply(_highlight_cenario, axis=1),
        hide_index=True,
        use_container_width=True,
    )

    # Grafico de barras — MAPE por cenario (ja ordenado, nomes legiveis)
    fig_cenarios = px.bar(
        df_cenarios_disp,
        x="Cenario", y="MAPE (%)",
        color="MAPE (%)",
        color_continuous_scale=[[0, "#1E3A8A"], [0.5, "#3B82F6"], [1, "#CBD5E1"]],
        text=df_cenarios_disp["MAPE (%)"].astype(str) + "%",
        labels={"Cenario": "Cenario de Features", "MAPE (%)": "MAPE (%)"},
    )
    fig_cenarios.update_traces(textposition="outside")
    fig_cenarios.update_layout(
        height=300,
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        margin=dict(t=20, b=10, l=10, r=10),
        coloraxis_showscale=False,
        xaxis=dict(tickfont=dict(color="#0F172A", size=10)),
        yaxis=dict(title="MAPE (%)", tickfont=dict(color="#0F172A", size=10),
                   gridcolor="#F1F5F9"),
        showlegend=False,
    )
    st.plotly_chart(fig_cenarios, use_container_width=True)

    # Insight dinamico: adapta a mensagem consoante qual cenario venceu
    best_mape   = df_cenarios.iloc[0]["MAPE (%)"]
    best_model  = df_cenarios.iloc[0]["Melhor Modelo"]
    n_feat_best = df_cenarios.iloc[0]["Nº Features"]

    mape_c = df_cenarios[df_cenarios["Cenario"] == "C_Context_Expert"]["MAPE (%)"].values
    mape_d = df_cenarios[df_cenarios["Cenario"] == "D_Full_Context"]["MAPE (%)"].values
    n_feat_d = CENARIO_FEATURES.get("D_Full_Context", 18)

    if melhor_cenario == "D_Full_Context" and len(mape_c) > 0:
        diff = round(float(mape_c[0]) - float(best_mape), 1)
        insight = (
            f'Cenario preferencial: <strong>{melhor_label}</strong> ({n_feat_best} features)'
            f' &mdash; MAPE: <strong>{best_mape}%</strong> vs Especialista de Contexto: {mape_c[0]}%'
            f' (melhoria de {diff}pp).<br>'
            f'O contexto completo supera marginalmente o especialista nesta loja. '
            f'Ainda assim, {n_feat_d} features vs 8 e um risco de overfitting em lojas com menos dados.'
        )
    else:
        mape_d_val = f"{mape_d[0]}%" if len(mape_d) > 0 else "N/A"
        insight = (
            f'Cenario preferencial: <strong>{melhor_label}</strong> ({n_feat_best} features)'
            f' &mdash; Melhor modelo: <strong>{best_model}</strong>'
            f' &mdash; MAPE: <strong>{best_mape}%</strong>.<br>'
            f'O Contexto Completo ({n_feat_d} features) obtem {mape_d_val} &mdash; pior que o'
            f' Especialista apesar de usar o dobro das features. '
            f'Features selecionadas por conhecimento de dominio superam a quantidade bruta.'
        )

    st.markdown(f'<div class="insight-box">{insight}</div>', unsafe_allow_html=True)
else:
    st.info("Dados de cenarios nao disponiveis para esta loja.")

# ── Rodape ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dss-footer">
    DSS &nbsp;|&nbsp; Previsao de Clientes &nbsp;|&nbsp;
    Banda de incerteza: Distribuicao de Poisson (&plusmn;&radic;procura)
</div>
""", unsafe_allow_html=True)
