"""
O3: Fronteira de Decisão Multiobjetivo — Laboratório de Decisão Interativo
Página 03 — DSS USA Stores

Laboratório interativo onde o slider w controla dinamicamente qual ponto da
fronteira de Pareto é selecionado como "Plano Sugerido", sem necessidade de
re-executar a otimização. O botão "Executar Simulação" actualiza a fronteira.
"""
import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go

sys.dont_write_bytecode = True

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Otimização Multiobjetivo — DSS USA Stores",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Corporate CSS ──────────────────────────────────────────────────────────────
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

  /* ── Section chrome ─────────────────────────────────────────── */
  .section-label {
    font-size: 0.70rem; font-weight: 700; letter-spacing: 0.09em;
    text-transform: uppercase; color: #94A3B8; margin-bottom: 10px;
  }
  .section-title {
    font-size: 1.25rem; font-weight: 800; color: #0F172A;
    letter-spacing: -0.02em; margin-bottom: 6px;
  }
  .section-desc {
    font-size: 0.85rem; color: #64748B; line-height: 1.65;
    max-width: 820px; margin-bottom: 20px;
  }

  /* ── Formula block ──────────────────────────────────────────── */
  .formula-block {
    padding: 20px 24px; border-radius: 8px;
    border-left: 4px solid #1E3A8A; background: #F8FAFC;
    margin-bottom: 20px;
  }
  .formula-title {
    font-size: 0.70rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: #1E3A8A; margin-bottom: 10px;
  }
  .formula-note {
    font-size: 0.80rem; color: #475569; margin-top: 10px; line-height: 1.6;
  }

  /* ── Status banners ─────────────────────────────────────────── */
  .banner-ok {
    padding: 11px 16px; border-radius: 8px;
    border: 1px solid #BBF7D0; background: #F0FDF4;
    font-size: 0.82rem; color: #065F46; font-weight: 600;
    margin-bottom: 16px;
  }
  .banner-warn {
    padding: 11px 16px; border-radius: 8px;
    border: 1px solid #FDE68A; background: #FFFBEB;
    font-size: 0.82rem; color: #92400E; font-weight: 600;
    margin-bottom: 16px;
  }
  .banner-info {
    padding: 11px 16px; border-radius: 8px;
    border: 1px solid #BAE6FD; background: #F0F9FF;
    font-size: 0.82rem; color: #0C4A6E; font-weight: 600;
    margin-bottom: 16px;
  }

  /* ── Stats strip ────────────────────────────────────────────── */
  .stats-strip {
    padding: 14px 20px; border-radius: 8px;
    border: 1px solid #E2E8F0; background: #F8FAFC;
    font-size: 0.83rem; color: #334155; line-height: 1.8;
  }
  .stats-strip strong { color: #1E3A8A; }

  /* ── Profile cards ──────────────────────────────────────────── */
  .profile-card {
    padding: 20px 18px 16px; border-radius: 10px;
    border: 1px solid #E2E8F0; background: #FFFFFF; text-align: center;
  }
  .profile-badge {
    display: inline-block; padding: 3px 10px; border-radius: 4px;
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.05em;
    text-transform: uppercase; margin-bottom: 10px;
  }
  .badge-blue  { background: rgba(30,58,138,0.10); color: #1E3A8A; }
  .badge-slate { background: rgba(51,65,85,0.10);  color: #334155; }
  .badge-green { background: rgba(5,150,105,0.10); color: #059669; }

  .profile-label {
    font-size: 0.68rem; font-weight: 600; letter-spacing: 0.06em;
    text-transform: uppercase; color: #94A3B8; margin-bottom: 6px;
  }
  .profile-value {
    font-size: 1.4rem; font-weight: 800; color: #0F172A; line-height: 1.15;
  }
  .profile-value-blue { color: #1E3A8A; }
  .profile-sub { font-size: 0.78rem; color: #64748B; line-height: 1.55; margin-top: 6px; }

  /* ── Optimal callout (inline under chart) ───────────────────── */
  .optimal-callout {
    padding: 14px 20px; border-radius: 8px; margin-top: 10px;
    border: 1.5px solid #DC2626; background: rgba(220,38,38,0.04);
    font-size: 0.83rem; color: #0F172A; line-height: 1.7;
  }
  .optimal-callout strong { color: #DC2626; }

  /* ── Footer ─────────────────────────────────────────────────── */
  .dss-footer {
    font-size: 0.75rem; color: #94A3B8; text-align: right;
    margin-top: 40px; padding-top: 14px; border-top: 1px solid #E2E8F0;
  }
</style>
""", unsafe_allow_html=True)

# ── Path + sys.path bootstrap ──────────────────────────────────────────────────
_page_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.abspath(os.path.join(_page_dir, "..", ".."))
_src_dir  = os.path.join(_root_dir, "src")
_opt_dir  = os.path.join(_root_dir, "results_v2", "03_Optimization")

if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

# ── Constants ──────────────────────────────────────────────────────────────────
STORES_ORDER = ["baltimore", "lancaster", "philadelphia", "richmond"]
LOJAS        = [s.capitalize() for s in STORES_ORDER]
LIMIT_UNITS  = 10_000
COLORSCALE_CORP = [[0.0, "#E2E8F0"], [0.5, "#3B82F6"], [1.0, "#1E3A8A"]]

DEFAULT_FORECASTS = {
    "baltimore":    [97,  61,  65,  71,  65,  89,  125],
    "lancaster":    [116, 72,  77,  84,  77,  106, 149],
    "philadelphia": [230, 144, 154, 168, 154, 211, 298],
    "richmond":     [64,  40,  42,  46,  42,  58,  82],
}
DEFAULT_WEEKENDS = [True, False, False, False, False, False, True]

# ── Data loaders ───────────────────────────────────────────────────────────────
@st.cache_data
def load_joint_pareto(opt_dir: str, algo: str):
    fname = "joint_pareto_unsga3.csv" if algo == "unsga3" else "joint_pareto_moead.csv"
    path  = os.path.join(opt_dir, "joint_o3", fname)
    if not os.path.exists(path):
        return None
    df = pd.read_csv(path)
    pr_cols = [c for c in df.columns if "_pr_" in c]
    df["avg_discount_pct"] = df[pr_cols].mean(axis=1) * 100
    return df[["lucro_total", "staff_total", "avg_discount_pct"]]


@st.cache_data
def load_joint_summary(opt_dir: str):
    path = os.path.join(opt_dir, "joint_o3", "joint_o3_summary.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


@st.cache_data
def load_store_pareto(opt_dir: str, store: str):
    path = os.path.join(opt_dir, "multiobjective", store.capitalize(), "pareto_front.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


# ── Pre-load pre-computed data ─────────────────────────────────────────────────
df_u3   = load_joint_pareto(_opt_dir, "unsga3")
df_summ = load_joint_summary(_opt_dir)

# ── Helper functions ───────────────────────────────────────────────────────────
def _safe_norm(series: pd.Series) -> pd.Series:
    """Normalização min-max com proteção contra divisor zero."""
    lo, hi = series.min(), series.max()
    denom = hi - lo
    if denom == 0:
        return pd.Series(0.5, index=series.index)
    return (series - lo) / denom


def get_optimal_point(df: pd.DataFrame, w: float) -> pd.Series:
    """Ponto da fronteira que maximiza Score = w·lucro_norm − (1−w)·staff_norm."""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    score = w * _safe_norm(df["lucro_total"]) - (1 - w) * _safe_norm(df["staff_total"])
    return df.loc[score.idxmax()]


def compute_profiles(df: pd.DataFrame, w: float) -> pd.DataFrame:
    """
    3 perfis de decisão derivados da fronteira Pareto.
    O Perfil Equilibrado atualiza dinamicamente com o valor de w.
    """
    if df is None or df.empty:
        return pd.DataFrame()

    p_agressivo   = df.loc[df["lucro_total"].idxmax()]
    p_equilibrado = get_optimal_point(df, w)

    df_pos = df[df["lucro_total"] > 0]
    p_conservador = (
        df_pos.loc[df_pos["staff_total"].idxmin()]
        if len(df_pos) else df.loc[df["staff_total"].idxmin()]
    )

    rows = []
    for label, sublabel, badge, row in [
        ("Perfil Agressivo",    "Maximo Lucro",           "badge-blue",  p_agressivo),
        ("Perfil Equilibrado",  f"Escalarizacao w={w:.2f}","badge-slate", p_equilibrado),
        ("Perfil Conservador",  "Minimo RH (lucro > 0)",  "badge-green", p_conservador),
    ]:
        rows.append({
            "Perfil":           label,
            "Criterio":         sublabel,
            "_badge":           badge,
            "Lucro Total (€)":  f"€{row['lucro_total']:,.0f}",
            "Staff Total":      int(row["staff_total"]),
            "Desc. Medio (%)":  f"{row['avg_discount_pct']:.2f}%",
            "_lucro_raw":       float(row["lucro_total"]),
            "_staff_raw":       int(row["staff_total"]),
            "_disc_raw":        float(row["avg_discount_pct"]),
        })
    return pd.DataFrame(rows)


# ── Session state ──────────────────────────────────────────────────────────────
if "rt_pareto_data" not in st.session_state:
    st.session_state["rt_pareto_data"] = None
if "rt_run_status" not in st.session_state:
    st.session_state["rt_run_status"] = None
if "rt_n_sol" not in st.session_state:
    st.session_state["rt_n_sol"] = 0
if "rt_rigor" not in st.session_state:
    st.session_state["rt_rigor"] = None

# ══════════════════════════════════════════════════════════════════════════════
# SIDEBAR
# ══════════════════════════════════════════════════════════════════════════════
st.sidebar.markdown(
    '<div style="font-size:0.88rem;font-weight:700;color:#E2E8F0;'
    'letter-spacing:0.04em;margin-bottom:16px;">DSS USA STORES</div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")

# ── Slider w (fora do form → reactivo) ────────────────────────────────────────
st.sidebar.markdown(
    '<div style="font-size:0.70rem;color:#64748B;letter-spacing:0.08em;'
    'text-transform:uppercase;font-weight:700;margin-bottom:10px;">'
    'Peso de Escalarizacao (w)</div>',
    unsafe_allow_html=True,
)
w_val = st.sidebar.slider(
    label="w", min_value=0.0, max_value=1.0, value=0.7, step=0.05,
    label_visibility="collapsed",
    key="w_slider",
)
st.sidebar.markdown(
    f'<div style="font-size:0.72rem;color:#94A3B8;margin-top:-6px;margin-bottom:10px;">'
    f'w = <strong style="color:#E2E8F0;">{w_val:.2f}</strong> &nbsp;|&nbsp; '
    f'{"Prioridade ao Lucro" if w_val >= 0.6 else "Prioridade ao Staff Minimo" if w_val <= 0.4 else "Equilibrio"}'
    f'</div>',
    unsafe_allow_html=True,
)

st.sidebar.markdown("---")

# ── Loja para Pareto individual ────────────────────────────────────────────────
loja_sel = st.sidebar.selectbox("Loja (Pareto individual)", LOJAS, index=0)

st.sidebar.markdown("---")

# ── Formulário de execução ─────────────────────────────────────────────────────
st.sidebar.markdown(
    '<div style="font-size:0.70rem;color:#64748B;letter-spacing:0.08em;'
    'text-transform:uppercase;font-weight:700;margin-bottom:10px;">'
    'Simulacao de Decisao</div>',
    unsafe_allow_html=True,
)
with st.sidebar.form("run_form", clear_on_submit=False):
    rigor = st.selectbox(
        "Rigor do U-NSGA-III",
        ["Rapido (Demo: 30 geracoes)", "Profundo (Producao: 150 geracoes)"],
        index=0,
    )
    submitted = st.form_submit_button(
        "Executar Simulacao de Decisao",
        type="primary",
        use_container_width=True,
    )

st.sidebar.markdown(
    '<div style="font-size:0.72rem;color:#64748B;margin-top:8px;line-height:1.5;">'
    "Demo: 10 direcoes × 30 ger.<br>"
    "Producao: 25 direcoes × 150 ger."
    "</div>",
    unsafe_allow_html=True,
)

# ── Executar se submetido ──────────────────────────────────────────────────────
if submitted:
    is_fast  = "Rapido" in rigor
    n_part   = 9  if is_fast else 24
    n_gen    = 30 if is_fast else 150

    with st.spinner("A calcular Fronteira de Pareto com U-NSGA-III..."):
        try:
            from optimization.unsga3_model import run_joint_optimization

            result = run_joint_optimization(
                store_forecasts=[DEFAULT_FORECASTS[s] for s in STORES_ORDER],
                store_weekends=[DEFAULT_WEEKENDS for _ in STORES_ORDER],
                n_partitions=n_part,
                n_max_gen=n_gen,
                seed=42,
                verbose=False,
            )
            if len(result.get("lucro", [])) > 0:
                # Índices das variáveis de desconto (pr) na matriz 84-var
                pr_idx = [s * 21 + d * 3 for s in range(4) for d in range(7)]
                df_rt = pd.DataFrame({
                    "lucro_total":      result["lucro"],
                    "staff_total":      result["staff"],
                    "avg_discount_pct": result["pareto_X"][:, pr_idx].mean(axis=1) * 100,
                })
                st.session_state["rt_pareto_data"] = df_rt
                st.session_state["rt_run_status"]  = "ok"
                st.session_state["rt_n_sol"]        = len(df_rt)
                st.session_state["rt_rigor"]        = rigor
            else:
                st.session_state["rt_run_status"] = "fallback"

        except Exception as exc:
            st.session_state["rt_run_status"] = "fallback"
            st.session_state["_rt_error"]     = str(exc)

# ── Dados activos (RT se disponível, senão pré-calculados) ────────────────────
_rt_ok = st.session_state["rt_run_status"] == "ok"
df_active = (
    st.session_state["rt_pareto_data"]
    if _rt_ok
    else df_u3
)

# ══════════════════════════════════════════════════════════════════════════════
# CABEÇALHO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div style="padding: 36px 0 20px 0; border-bottom: 2px solid #1E3A8A; margin-bottom: 28px;">
    <div style="font-size: 1.85rem; font-weight: 800; color: #0F172A;
         letter-spacing: -0.03em; line-height: 1.2; margin-bottom: 10px;">
        O3: Laboratorio de Decisao Multiobjetivo
    </div>
    <div style="font-size: 0.88rem; color: #64748B; max-width: 780px; line-height: 1.65;">
        Fronteira de Pareto interativa: mova o slider <strong>w</strong> para explorar o
        trade-off entre maximizacao de lucro e minimizacao de Recursos Humanos.
        O ponto otimo actualiza-se instantaneamente sem necessidade de re-executar.
    </div>
    <div style="margin-top: 14px; display: flex; gap: 10px; flex-wrap: wrap;">
        <span style="background:rgba(30,58,138,0.08);color:#1E3A8A;
              padding:3px 10px;border-radius:4px;
              font-size:0.70rem;font-weight:700;letter-spacing:0.05em;text-transform:uppercase;">
            84 Variaveis de Decisao</span>
        <span style="background:rgba(51,65,85,0.08);color:#334155;
              padding:3px 10px;border-radius:4px;
              font-size:0.70rem;font-weight:700;letter-spacing:0.05em;text-transform:uppercase;">
            2 Objetivos</span>
        <span style="background:rgba(5,150,105,0.08);color:#059669;
              padding:3px 10px;border-radius:4px;
              font-size:0.70rem;font-weight:700;letter-spacing:0.05em;text-transform:uppercase;">
            U-NSGA-III</span>
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECÇÃO 1 — ESCALARIZAÇÃO PONDERADA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">1. Escalarizacao Ponderada</div>', unsafe_allow_html=True)

col_formula, col_params = st.columns([3, 2], gap="large")

with col_formula:
    st.markdown('<div class="formula-block"><div class="formula-title">Funcao Escalar (Minimizacao)</div>', unsafe_allow_html=True)
    st.latex(r"\hat{F}(x) = -\,w \cdot \hat{f}_{\text{lucro}}(x) \;+\; (1-w) \cdot \hat{f}_{\text{RH}}(x)")
    st.latex(r"\hat{f} = \frac{f - f_{\min}}{f_{\max} - f_{\min}} \quad \text{(normaliz. min-max, denom. protegido)}")
    st.markdown(
        '<div class="formula-note">'
        "Mover o slider <strong>w</strong> na sidebar redefine instantaneamente qual solucao "
        "da fronteira de Pareto e seleccionada como plano operacional."
        "</div></div>",
        unsafe_allow_html=True,
    )

with col_params:
    if df_active is not None and not df_active.empty:
        opt = get_optimal_point(df_active, w_val)
        l_min_v = df_active["lucro_total"].min()
        l_max_v = df_active["lucro_total"].max()
        s_min_v = df_active["staff_total"].min()
        s_max_v = df_active["staff_total"].max()
        denom_l = (l_max_v - l_min_v) or 1.0
        denom_s = (s_max_v - s_min_v) or 1.0
        l_n_opt = (float(opt["lucro_total"]) - l_min_v) / denom_l
        s_n_opt = (float(opt["staff_total"]) - s_min_v) / denom_s

        df_params_tbl = pd.DataFrame({
            "Parametro": [
                "w (peso lucro)",
                "1 - w (peso RH)",
                "Lucro normalizado",
                "Staff normalizado",
                "Lucro Total (€)",
                "Staff Total",
            ],
            "Valor": [
                f"{w_val:.2f}",
                f"{1 - w_val:.2f}",
                f"{l_n_opt:.3f}",
                f"{s_n_opt:.3f}",
                f"€{opt['lucro_total']:,.0f}",
                f"{int(opt['staff_total'])}",
            ],
        })
        st.markdown(f'<div class="section-label">Solucao Optima para w = {w_val:.2f}</div>', unsafe_allow_html=True)
        st.dataframe(df_params_tbl, hide_index=True, use_container_width=True)

# Pareto stats strip
if df_summ is not None:
    row_u3 = df_summ[df_summ["algorithm"] == "U-NSGA-III"].iloc[0]
    row_md = df_summ[df_summ["algorithm"] == "MOEA/D"].iloc[0]
    src_label = "Real-Time Run" if _rt_ok else "Pre-calculado"
    n_sol = len(df_active) if df_active is not None else 0
    st.markdown(f"""
    <div class="stats-strip">
        Fonte activa: <strong>{src_label}</strong> ({n_sol} solucoes) &nbsp;|&nbsp;
        U-NSGA-III pre-calc: <strong>{int(row_u3['n_pareto_solutions'])} solucoes</strong>,
        max lucro <strong>€{row_u3['o3_max_profit']:,.0f}</strong> &nbsp;|&nbsp;
        MOEA/D: <strong>{int(row_md['n_pareto_solutions'])} solucoes</strong>,
        max lucro <strong>€{row_md['o3_max_profit']:,.0f}</strong>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# SECÇÃO 2 — SIMULAÇÃO E FRONTEIRA DE PARETO 2D
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">2. Fronteira de Pareto 2D — Laboratorio Interativo</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-desc">'
    'Cada ponto representa um plano operacional Pareto-otimo. '
    'A cor mapeia o desconto medio aplicado. '
    'A estrela vermelha indica o "Plano Sugerido" para o peso <strong>w</strong> '
    'seleccionado — mova o slider para explorar a fronteira em tempo real.'
    '</div>',
    unsafe_allow_html=True,
)

# ── Banner de status do run ────────────────────────────────────────────────────
if st.session_state["rt_run_status"] == "ok":
    st.markdown(
        f'<div class="banner-ok">'
        f'Simulacao concluida ({st.session_state["rt_rigor"]}) — '
        f'{st.session_state["rt_n_sol"]} solucoes Pareto geradas. '
        f'O grafico reflecte a nova fronteira.'
        f'</div>',
        unsafe_allow_html=True,
    )
elif st.session_state["rt_run_status"] == "fallback":
    err = st.session_state.get("_rt_error", "erro desconhecido")
    st.markdown(
        f'<div class="banner-warn">'
        f'Simulacao indisponivel ({err[:100]}...). A exibir dados pre-calculados.'
        f'</div>',
        unsafe_allow_html=True,
    )
elif st.session_state["rt_run_status"] is None:
    st.markdown(
        '<div class="banner-info">'
        'A exibir fronteira pre-calculada (U-NSGA-III, 150 geracoes, seed=42). '
        'Clique em "Executar Simulacao de Decisao" para gerar uma nova fronteira ao vivo.'
        '</div>',
        unsafe_allow_html=True,
    )

# ── Gráfico 2D Pareto ──────────────────────────────────────────────────────────
if df_active is not None and not df_active.empty:
    opt_point = get_optimal_point(df_active, w_val)
    opt_staff  = float(opt_point["staff_total"])
    opt_lucro  = float(opt_point["lucro_total"])
    opt_disc   = float(opt_point["avg_discount_pct"])

    fig = go.Figure()

    # Nuvem de Pareto — cor mapeada no desconto médio
    fig.add_trace(go.Scatter(
        x=df_active["staff_total"],
        y=df_active["lucro_total"],
        mode="markers",
        name="Solucoes Pareto",
        marker=dict(
            color=df_active["avg_discount_pct"],
            colorscale=COLORSCALE_CORP,
            size=9,
            opacity=0.82,
            line=dict(width=0.8, color="#334155"),
            colorbar=dict(
                title=dict(
                    text="Desc. Medio (%)",
                    font=dict(color="#0F172A", size=11),
                ),
                tickfont=dict(color="#0F172A", size=10),
                len=0.75,
                thickness=14,
                x=1.01,
            ),
        ),
        hovertemplate=(
            "<b>Solucao Pareto</b><br>"
            "Staff: %{x}<br>"
            "Lucro: €%{y:,.0f}<br>"
            "Desc. Medio: %{marker.color:.2f}%"
            "<extra></extra>"
        ),
    ))

    # Linha de break-even (Lucro = 0)
    x_range = [df_active["staff_total"].min() - 2, df_active["staff_total"].max() + 2]
    fig.add_shape(
        type="line",
        x0=x_range[0], x1=x_range[1], y0=0, y1=0,
        line=dict(color="#94A3B8", width=1.5, dash="dot"),
        layer="below",
    )
    fig.add_annotation(
        x=x_range[0] + 1, y=0,
        text="Break-even (€0)",
        showarrow=False,
        yshift=10,
        font=dict(color="#94A3B8", size=9),
        xanchor="left",
    )

    # Ponto ótimo — estrela vermelha vibrante
    fig.add_trace(go.Scatter(
        x=[opt_staff],
        y=[opt_lucro],
        mode="markers",
        name=f"Plano Sugerido (w={w_val:.2f})",
        marker=dict(
            symbol="star",
            size=22,
            color="#DC2626",
            line=dict(width=2, color="#0F172A"),
        ),
        hovertemplate=(
            f"<b>Plano Sugerido (w={w_val:.2f})</b><br>"
            "Staff: %{x}<br>"
            "Lucro: €%{y:,.0f}<extra></extra>"
        ),
    ))

    # Anotação com seta apontando para o ponto óptimo
    fig.add_annotation(
        x=opt_staff,
        y=opt_lucro,
        text=f"<b>Plano Sugerido</b><br>w = {w_val:.2f}<br>€{opt_lucro:,.0f}",
        showarrow=True,
        arrowhead=2,
        arrowwidth=2,
        arrowcolor="#DC2626",
        ax=55,
        ay=-45,
        font=dict(color="#0F172A", size=11),
        bgcolor="rgba(255,255,255,0.92)",
        bordercolor="#DC2626",
        borderwidth=1.5,
        borderpad=5,
    )

    fig.update_layout(
        height=480,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin=dict(t=20, b=50, l=10, r=100),
        legend=dict(
            orientation="h",
            yanchor="bottom", y=1.02,
            xanchor="left", x=0,
            font=dict(color="#0F172A", size=11),
            bgcolor="rgba(255,255,255,0.9)",
            bordercolor="#E2E8F0",
            borderwidth=1,
        ),
        xaxis=dict(
            title=dict(text="Staff Total — Minimizar", font=dict(color="#0F172A", size=12)),
            tickfont=dict(color="#0F172A", size=11),
            gridcolor="#F1F5F9",
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(text="Lucro Total (€) — Maximizar", font=dict(color="#0F172A", size=12)),
            tickfont=dict(color="#0F172A", size=11),
            gridcolor="#F1F5F9",
            tickformat="€,.0f",
            zeroline=False,
        ),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Callout informativo abaixo do gráfico
    st.markdown(f"""
    <div class="optimal-callout">
        <strong>Plano Sugerido (w = {w_val:.2f}):</strong>
        Lucro = <strong>€{opt_lucro:,.0f}</strong> &nbsp;|&nbsp;
        Staff = <strong>{int(opt_staff)} funcionarios</strong> &nbsp;|&nbsp;
        Desconto medio = <strong>{opt_disc:.2f}%</strong>
        <br>
        <span style="font-size:0.78rem;color:#64748B;">
        Este ponto minimiza F = −{w_val:.2f}·lucro_norm + {1-w_val:.2f}·staff_norm
        sobre a fronteira de Pareto activa.
        </span>
    </div>
    """, unsafe_allow_html=True)

else:
    st.info("Dados O3 nao disponiveis. Execute: python scripts/run_joint_optimization_o3.py")

# ── Pareto 2D por loja (NSGA-II individual) ────────────────────────────────────
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    f'<div class="section-label">Pareto Individual por Loja — NSGA-II ({loja_sel})</div>',
    unsafe_allow_html=True,
)

df_sp = load_store_pareto(_opt_dir, loja_sel)
if df_sp is not None:
    fig_sp = go.Figure()
    fig_sp.add_trace(go.Scatter(
        x=df_sp["staff"], y=df_sp["lucro"],
        mode="markers+lines",
        marker=dict(color="#1E3A8A", size=7, opacity=0.85),
        line=dict(color="#CBD5E1", width=1.5),
        hovertemplate="Staff: %{x}<br>Lucro: €%{y:,.0f}<extra></extra>",
    ))
    fig_sp.add_hline(
        y=0, line_dash="dot", line_color="#DC2626", line_width=1.5,
        annotation_text="Break-even",
        annotation_font=dict(color="#DC2626", size=10),
    )
    fig_sp.update_layout(
        height=270,
        plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
        margin=dict(t=10, b=40, l=10, r=20),
        showlegend=False,
        xaxis=dict(
            title=dict(text="Staff Total", font=dict(color="#0F172A", size=11)),
            tickfont=dict(color="#0F172A", size=10), gridcolor="#F1F5F9", zeroline=False,
        ),
        yaxis=dict(
            title=dict(text="Lucro (€)", font=dict(color="#0F172A", size=11)),
            tickfont=dict(color="#0F172A", size=10), gridcolor="#F1F5F9",
            tickformat="€,.0f", zeroline=False,
        ),
    )
    st.plotly_chart(fig_sp, use_container_width=True)
    max_l = df_sp["lucro"].max()
    min_l = df_sp["lucro"].min()
    max_s = df_sp["staff"].max()
    min_s = df_sp["staff"].min()
    delta_s = max_s - min_s or 1
    st.caption(
        f"{loja_sel} — {len(df_sp)} solucoes Pareto (NSGA-II). "
        f"Trade-off: ~€{(max_l - min_l) / delta_s:,.0f} de lucro por funcionario adicional. "
        f"Staff [{min_s:.0f} – {max_s:.0f}] | Lucro [€{min_l:,.0f} – €{max_l:,.0f}]."
    )

# ══════════════════════════════════════════════════════════════════════════════
# SECÇÃO 3 — PERFIS DE DECISÃO DINÂMICOS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">3. Decisao do Gestor — Perfis de Decisao</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-desc">'
    'Tres perfis derivados da fronteira activa. O <strong>Perfil Equilibrado</strong> '
    'actualiza-se dinamicamente com o valor do slider w, reflectindo sempre a solucao '
    'de escalarizacao para o peso corrente.'
    '</div>',
    unsafe_allow_html=True,
)

df_profiles = compute_profiles(df_active, w_val)

if not df_profiles.empty:
    profile_data = df_profiles.to_dict("records")
    profile_labels = ["Agressivo", "Equilibrado", "Conservador"]

    # Cards de resumo
    col_cards = st.columns(3, gap="medium")
    for i, (col, row) in enumerate(zip(col_cards, profile_data)):
        with col:
            is_equil = i == 1
            st.markdown(f"""
            <div class="profile-card">
                <span class="profile-badge {row['_badge']}">{profile_labels[i]}</span>
                <div class="profile-label">{row["Criterio"]}</div>
                <div class="profile-value {'profile-value-blue' if i == 0 else ''}">
                    {row["Lucro Total (€)"]}
                </div>
                <div class="profile-sub">
                    Staff: <strong>{row["Staff Total"]}</strong> &nbsp;|&nbsp;
                    Desc.: <strong>{row["Desc. Medio (%)"]}</strong>
                    {"<br><span style='font-size:0.72rem;color:#94A3B8;'>(actualiza com w)</span>" if is_equil else ""}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Tabela comparativa
    st.markdown('<div class="section-label">Tabela Comparativa</div>', unsafe_allow_html=True)
    df_tbl = df_profiles[["Perfil", "Criterio", "Lucro Total (€)", "Staff Total", "Desc. Medio (%)"]].copy()
    st.dataframe(df_tbl, hide_index=True, use_container_width=True)

    # Botões de exportação
    st.markdown("<br>", unsafe_allow_html=True)
    col_dl1, col_dl2, _ = st.columns([2, 2, 3])

    with col_dl1:
        csv_perfis = df_tbl.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Exportar Perfis (.csv)",
            data=csv_perfis,
            file_name=f"o3_perfis_w{w_val:.2f}.csv",
            mime="text/csv",
        )
    with col_dl2:
        if df_active is not None:
            csv_pareto = df_active[["lucro_total", "staff_total", "avg_discount_pct"]].rename(columns={
                "lucro_total":      "lucro_usd",
                "staff_total":      "staff_funcionarios",
                "avg_discount_pct": "desconto_medio_pct",
            }).to_csv(index=False).encode("utf-8")
            st.download_button(
                label="Exportar Pareto Completo (.csv)",
                data=csv_pareto,
                file_name="o3_pareto_completo.csv",
                mime="text/csv",
            )
else:
    st.info("Execute o pipeline O3 ou clique em 'Executar Simulacao de Decisao' para gerar perfis.")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dss-footer">
    DSS &nbsp;|&nbsp; O3 Multiobjetivo &nbsp;|&nbsp;
    U-NSGA-III &middot; 84 vars &middot; Pareto frontier &middot; Escalarizacao ponderada
</div>
""", unsafe_allow_html=True)
