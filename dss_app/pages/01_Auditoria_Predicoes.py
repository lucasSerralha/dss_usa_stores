"""
Auditoria Científica de Modelos Preditivos
Página 01 — DSS USA Stores

Fontes de dados:
  - results/00_Master_Summary/fidelity_experimentation_report.csv  (7 modelos, MAE/RMSE/MAPE/NMAE)
  - results/04_Model_Testing/lightgbm_tuning_report.csv            (LightGBM, só RMSE/MAPE)
"""
import streamlit as st
import sys
import os
import pandas as pd
import plotly.graph_objects as go

sys.dont_write_bytecode = True

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Auditoria de Modelos — DSS USA Stores",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Corporate CSS (paleta consistente com app.py: azuis escuros + cinzentos) ───
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

  /* ── Podium cards ──────────────────────────────────────────────── */
  .podium-card {
    padding: 22px 18px 18px 18px;
    border-radius: 10px;
    border: 1px solid #E2E8F0;
    background: #FFFFFF;
    text-align: center;
  }
  .podium-rank {
    display: inline-flex;
    align-items: center; justify-content: center;
    width: 30px; height: 30px;
    border-radius: 6px;
    font-size: 0.82rem; font-weight: 700;
    margin-bottom: 10px;
  }
  .rank-1 { background: #1E3A8A; color: #FFFFFF; }
  .rank-2 { background: #334155; color: #FFFFFF; }
  .rank-3 { background: #CBD5E1; color: #0F172A; }

  .podium-place {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: #94A3B8; margin-bottom: 8px;
  }
  .podium-model {
    font-size: 0.96rem; font-weight: 700; color: #0F172A;
    margin-bottom: 14px; line-height: 1.3; min-height: 2.6em;
  }
  .podium-metric {
    display: flex; justify-content: space-between; align-items: center;
    font-size: 0.80rem; color: #64748B;
    border-top: 1px solid #F1F5F9; padding: 5px 0;
  }
  .podium-metric-value { font-weight: 700; color: #1E3A8A; }

  /* ── Section labels ────────────────────────────────────────────── */
  .section-label {
    font-size: 0.70rem; font-weight: 700; letter-spacing: 0.09em;
    text-transform: uppercase; color: #94A3B8; margin-bottom: 10px;
  }

  /* ── Footer ────────────────────────────────────────────────────── */
  .dss-footer {
    font-size: 0.75rem; color: #94A3B8; text-align: right;
    margin-top: 40px; padding-top: 14px; border-top: 1px solid #E2E8F0;
  }
</style>
""", unsafe_allow_html=True)

# ── Path resolution (pages/ → dss_app/ → project root) ───────────────────────
_page_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.abspath(os.path.join(_page_dir, "..", ".."))

# ── Data loading ───────────────────────────────────────────────────────────────
@st.cache_data
def load_master_report(root: str):
    path = os.path.join(root, "results_v2", "00_Master_Summary", "fidelity_experimentation_report.csv")
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_data
def load_lightgbm_report(root: str):
    """
    O relatório LightGBM tem MAPE em formato decimal (0.18 = 18%).
    Normaliza para percentagem para ser consistente com o relatório master.
    """
    path = os.path.join(root, "results_v2", "04_Model_Testing", "lightgbm_tuning_report.csv")
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    df = df.rename(columns={"Loja": "Store", "Cenario": "Experiment"})
    df["Store"] = df["Store"].str.strip().str.capitalize()
    df["Model"] = "LightGBM"
    df["MAPE"] = df["MAPE"] * 100   # 0.1867 → 18.67 (consistente com master report)
    df["MAE"] = None
    df["NMAE"] = None
    return df[["Store", "Experiment", "Model", "MAE", "RMSE", "MAPE", "NMAE"]]


df_master = load_master_report(_root_dir)
df_lgbm = load_lightgbm_report(_root_dir)

_frames = [f for f in [df_master, df_lgbm] if f is not None]
df_all = pd.concat(_frames, ignore_index=True) if _frames else None

# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.markdown(
    '<div style="font-size:0.88rem;font-weight:700;color:#E2E8F0;'
    'letter-spacing:0.04em;margin-bottom:16px;">DSS USA STORES</div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div style="font-size:0.70rem;color:#64748B;letter-spacing:0.08em;'
    'text-transform:uppercase;font-weight:700;margin-bottom:12px;">Filtros de Analise</div>',
    unsafe_allow_html=True,
)

LOJAS = ["Baltimore", "Lancaster", "Philadelphia", "Richmond"]
CENARIOS = ["A_Temporal_Base", "B_Sales_Dynamics", "C_Context_Expert"]
CENARIOS_LABEL = {
    "A_Temporal_Base":  "A — Temporal Base",
    "B_Sales_Dynamics": "B — Sales Dynamics",
    "C_Context_Expert": "C — Context Expert",
}

loja_sel = st.sidebar.selectbox("Loja", LOJAS, index=0)
cenario_sel = st.sidebar.selectbox(
    "Cenário de Dados",
    CENARIOS,
    format_func=lambda x: CENARIOS_LABEL[x],
    index=2,
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div style="font-size:0.70rem;color:#64748B;letter-spacing:0.08em;'
    'text-transform:uppercase;font-weight:700;margin-bottom:8px;">Legenda de Cenários</div>',
    unsafe_allow_html=True,
)
st.sidebar.caption("A — Padrões cíclicos: sazonalidade e calendário.")
st.sidebar.caption("B — Dinâmica de vendas: lags de curto prazo.")
st.sidebar.caption("C — Contexto completo: promoções, feriados e eventos.")

# ── Page header ────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style="padding: 36px 0 20px 0; border-bottom: 2px solid #1E3A8A; margin-bottom: 32px;">
    <div style="font-size: 1.85rem; font-weight: 800; color: #0F172A;
         letter-spacing: -0.03em; line-height: 1.2; margin-bottom: 10px;">
        Auditoria Científica de Modelos Preditivos
    </div>
    <div style="font-size: 0.88rem; color: #64748B; max-width: 700px; line-height: 1.65;">
        Avaliação empírica via TimeSeriesSplit. Comparação de algoritmos de Base,
        Estatísticos e Machine Learning.
    </div>
    <div style="margin-top: 14px; display: flex; gap: 10px; flex-wrap: wrap;">
        <span style="background: rgba(30,58,138,0.08); color: #1E3A8A;
              padding: 3px 10px; border-radius: 4px;
              font-size: 0.70rem; font-weight: 700; letter-spacing: 0.05em;
              text-transform: uppercase;">Loja: {loja_sel}</span>
        <span style="background: rgba(51,65,85,0.08); color: #334155;
              padding: 3px 10px; border-radius: 4px;
              font-size: 0.70rem; font-weight: 700; letter-spacing: 0.05em;
              text-transform: uppercase;">Cenário: {CENARIOS_LABEL[cenario_sel]}</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Data validation ────────────────────────────────────────────────────────────
if df_all is None:
    st.error(
        "Relatórios de modelos não encontrados. "
        "Execute o pipeline principal para gerar os dados."
    )
    st.stop()

df_filtered = df_all[
    (df_all["Store"].str.lower() == loja_sel.lower()) &
    (df_all["Experiment"] == cenario_sel)
].copy().reset_index(drop=True)

if df_filtered.empty:
    st.warning(
        f"Sem dados disponíveis para **{loja_sel}** / **{cenario_sel}**. "
        "Execute o pipeline principal."
    )
    st.stop()

df_sorted = df_filtered.sort_values("RMSE", ascending=True).reset_index(drop=True)
winner_model = df_sorted.iloc[0]["Model"]

# ── Podium: Top 3 ──────────────────────────────────────────────────────────────
st.markdown('<div class="section-label">Podio de Desempenho — Top 3 Modelos</div>', unsafe_allow_html=True)

top3 = df_sorted.head(3)
PLACE_LABELS = ["1. Lugar", "2. Lugar", "3. Lugar"]
RANK_CLASSES  = ["rank-1",  "rank-2",   "rank-3"]

cols_podium = st.columns(3, gap="medium")
for i, (_, row) in enumerate(top3.iterrows()):
    nmae_html = (
        f'<div class="podium-metric">'
        f'<span>NMAE</span>'
        f'<span class="podium-metric-value">{row["NMAE"]:.2f}%</span>'
        f'</div>'
        if pd.notna(row.get("NMAE")) else ""
    )
    with cols_podium[i]:
        st.markdown(f"""
        <div class="podium-card">
            <div><span class="podium-rank {RANK_CLASSES[i]}">{i + 1}</span></div>
            <div class="podium-place">{PLACE_LABELS[i]}</div>
            <div class="podium-model">{row["Model"]}</div>
            <div class="podium-metric">
                <span>RMSE</span>
                <span class="podium-metric-value">{row["RMSE"]:.2f} clientes</span>
            </div>
            <div class="podium-metric">
                <span>MAPE</span>
                <span class="podium-metric-value">{row["MAPE"]:.2f}%</span>
            </div>
            {nmae_html}
        </div>
        """, unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")

# ── Bar chart: RMSE comparison ─────────────────────────────────────────────────
st.markdown(
    '<div class="section-label">Comparação de RMSE — Todos os Modelos</div>',
    unsafe_allow_html=True,
)
st.caption(
    "Nota Técnica: RMSE e MAE em número de clientes. MAPE e NMAE em percentagem. "
    "Valores menores indicam maior precisão."
)

# Vencedor: #1E3A8A (Azul Escuro) — restantes: #CBD5E1 (Cinzento Claro)
bar_colors = [
    "#1E3A8A" if m == winner_model else "#CBD5E1"
    for m in df_sorted["Model"]
]

fig_bar = go.Figure(go.Bar(
    x=df_sorted["RMSE"],
    y=df_sorted["Model"],
    orientation="h",
    marker_color=bar_colors,
    text=df_sorted["RMSE"].map("{:.2f}".format),
    textposition="outside",
    textfont=dict(color="#0F172A", size=11),
))

fig_bar.update_layout(
    height=max(300, len(df_sorted) * 48),
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",
    margin=dict(t=10, b=40, l=10, r=110),
    showlegend=False,
    xaxis=dict(
        title=dict(text="RMSE (clientes)", font=dict(color="#0F172A", size=12)),
        tickfont=dict(color="#0F172A", size=11),
        gridcolor="#F1F5F9",
        zeroline=False,
    ),
    yaxis=dict(
        tickfont=dict(color="#0F172A", size=11),
        autorange="reversed",   # melhor modelo (menor RMSE) no topo
    ),
)

st.plotly_chart(fig_bar, use_container_width=True)

st.markdown("---")

# ── Leaderboard table ──────────────────────────────────────────────────────────
st.markdown(
    '<div class="section-label">Leaderboard Completo</div>',
    unsafe_allow_html=True,
)

_has_mae  = "MAE"  in df_sorted.columns and df_sorted["MAE"].notna().any()
_has_nmae = "NMAE" in df_sorted.columns and df_sorted["NMAE"].notna().any()

display_cols = ["Model"]
if _has_mae:
    display_cols.append("MAE")
display_cols += ["RMSE", "MAPE"]
if _has_nmae:
    display_cols.append("NMAE")

df_display = df_sorted[display_cols].copy()
df_display["RMSE"] = df_display["RMSE"].map("{:.2f}".format)
df_display["MAPE"] = df_display["MAPE"].map("{:.2f}%".format)
if _has_mae:
    df_display["MAE"] = df_display["MAE"].apply(
        lambda x: f"{x:.2f}" if pd.notna(x) else "N/A"
    )
if _has_nmae:
    df_display["NMAE"] = df_display["NMAE"].apply(
        lambda x: f"{x:.2f}%" if pd.notna(x) else "N/A"
    )

st.dataframe(df_display, hide_index=True, use_container_width=True)

# ── Export leaderboard ─────────────────────────────────────────────────────────
csv_leaderboard = df_sorted[display_cols].to_csv(index=False).encode("utf-8")
st.download_button(
    label="⬇ Exportar Leaderboard (CSV)",
    data=csv_leaderboard,
    file_name=f"leaderboard_{loja_sel.lower()}_{cenario_sel}.csv",
    mime="text/csv",
    use_container_width=False,
)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dss-footer">
    DSS &nbsp;|&nbsp; Auditoria de Modelos &nbsp;|&nbsp;
    TimeSeriesSplit cross-validation &nbsp;|&nbsp; RMSE/MAE em nº de clientes &nbsp;|&nbsp; MAPE/NMAE em %
</div>
""", unsafe_allow_html=True)
