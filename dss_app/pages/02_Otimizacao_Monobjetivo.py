"""
O1 + O2: Otimização Monobjetivo — Comparação de Algoritmos
Página 02 — DSS USA Stores

O1 — Maximização de lucro individual por loja
     Algoritmos: Monte Carlo (baseline), Hill Climbing, Simulated Annealing, NSGA-II
O2 — Alocação conjunta com restrição logística de 10 000 unidades/semana
     Algoritmos: Hill Climbing + Penalty Function, Hill Climbing + Death Penalty
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
    page_title="Otimização Monobjetivo — DSS USA Stores",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Corporate CSS ────────────────────────────────────────────────────────────
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

  /* ── Metric comparison cards ───────────────────────────────────── */
  .cmp-card {
    padding: 22px 20px 18px 20px;
    border-radius: 10px;
    border: 1px solid #E2E8F0;
    background: #FFFFFF;
    text-align: center;
  }
  .cmp-label {
    font-size: 0.70rem; font-weight: 700; letter-spacing: 0.08em;
    text-transform: uppercase; color: #94A3B8; margin-bottom: 10px;
  }
  .cmp-value {
    font-size: 1.75rem; font-weight: 800; color: #0F172A; line-height: 1.1;
    margin-bottom: 6px;
  }
  .cmp-value-blue { color: #1E3A8A; }
  .cmp-sub {
    font-size: 0.78rem; color: #64748B; line-height: 1.5;
  }
  .cmp-delta-pos {
    display: inline-block; margin-top: 10px;
    padding: 3px 10px; border-radius: 4px;
    font-size: 0.78rem; font-weight: 700;
    background: rgba(5,150,105,0.08); color: #059669;
  }
  .cmp-delta-neg {
    display: inline-block; margin-top: 10px;
    padding: 3px 10px; border-radius: 4px;
    font-size: 0.78rem; font-weight: 700;
    background: rgba(220,38,38,0.08); color: #DC2626;
  }

  /* ── Constraint indicator ──────────────────────────────────────── */
  .constraint-ok {
    padding: 18px 22px; border-radius: 10px;
    border: 1px solid #BBF7D0; background: #F0FDF4;
    text-align: center;
  }
  .constraint-ok .c-title { font-size: 0.70rem; font-weight: 700;
    letter-spacing: 0.08em; text-transform: uppercase;
    color: #059669; margin-bottom: 8px; }
  .constraint-ok .c-value { font-size: 1.6rem; font-weight: 800; color: #065F46; }
  .constraint-ok .c-badge { margin-top: 8px; display: inline-block;
    padding: 3px 12px; border-radius: 20px; font-size: 0.75rem;
    font-weight: 700; background: #059669; color: #FFFFFF; }

  .constraint-err {
    padding: 18px 22px; border-radius: 10px;
    border: 1px solid #FECACA; background: #FEF2F2;
    text-align: center;
  }
  .constraint-err .c-title { font-size: 0.70rem; font-weight: 700;
    letter-spacing: 0.08em; text-transform: uppercase;
    color: #DC2626; margin-bottom: 8px; }
  .constraint-err .c-value { font-size: 1.6rem; font-weight: 800; color: #7F1D1D; }
  .constraint-err .c-badge { margin-top: 8px; display: inline-block;
    padding: 3px 12px; border-radius: 20px; font-size: 0.75rem;
    font-weight: 700; background: #DC2626; color: #FFFFFF; }

  /* ── Section labels ────────────────────────────────────────────── */
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
    max-width: 800px; margin-bottom: 20px;
  }

  /* ── Footer ────────────────────────────────────────────────────── */
  .dss-footer {
    font-size: 0.75rem; color: #94A3B8; text-align: right;
    margin-top: 40px; padding-top: 14px; border-top: 1px solid #E2E8F0;
  }
</style>
""", unsafe_allow_html=True)

# ── Path resolution ────────────────────────────────────────────────────────────
_page_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.abspath(os.path.join(_page_dir, "..", ".."))
_opt_dir  = os.path.join(_root_dir, "results_v2", "03_Optimization")

# ── Colour palette per algorithm ───────────────────────────────────────────────
ALG_COLORS = {
    "Monte Carlo":         "#9C27B0",
    "Hill Climbing":       "#1E3A8A",
    "Simulated Annealing": "#FF6F00",
    "NSGA-II":             "#059669",
}

# ── Data loaders ───────────────────────────────────────────────────────────────
@st.cache_data
def load_hc_summary(opt_dir: str):
    p = os.path.join(opt_dir, "individual", "optimization_summary.csv")
    return pd.read_csv(p) if os.path.exists(p) else None

@st.cache_data
def load_hc_plan(opt_dir: str, store: str):
    p = os.path.join(opt_dir, "individual", f"{store.lower()}_best_plan.csv")
    return pd.read_csv(p) if os.path.exists(p) else None

@st.cache_data
def load_hc_convergence(opt_dir: str, store: str):
    p = os.path.join(opt_dir, "individual", f"{store.lower()}_convergence.csv")
    return pd.read_csv(p) if os.path.exists(p) else None

@st.cache_data
def load_sa_summary(opt_dir: str):
    p = os.path.join(opt_dir, "individual", "simulated_annealing", "simulated_annealing_summary.csv")
    return pd.read_csv(p) if os.path.exists(p) else None

@st.cache_data
def load_sa_plan(opt_dir: str, store: str):
    p = os.path.join(opt_dir, "individual", "simulated_annealing", f"{store.lower()}_sa_best_plan.csv")
    return pd.read_csv(p) if os.path.exists(p) else None

@st.cache_data
def load_sa_convergence(opt_dir: str, store: str):
    p = os.path.join(opt_dir, "individual", "simulated_annealing", f"{store.lower()}_sa_convergence.csv")
    return pd.read_csv(p) if os.path.exists(p) else None

@st.cache_data
def load_mc_summary(opt_dir: str):
    p = os.path.join(opt_dir, "individual", "monte_carlo", "monte_carlo_summary.csv")
    return pd.read_csv(p) if os.path.exists(p) else None

@st.cache_data
def load_mc_plan(opt_dir: str, store: str):
    p = os.path.join(opt_dir, "individual", "monte_carlo", f"{store.lower()}_mc_best_plan.csv")
    return pd.read_csv(p) if os.path.exists(p) else None

@st.cache_data
def load_mc_convergence(opt_dir: str, store: str):
    p = os.path.join(opt_dir, "individual", "monte_carlo", f"{store.lower()}_mc_convergence.csv")
    return pd.read_csv(p) if os.path.exists(p) else None

@st.cache_data
def load_nsga2_summary(opt_dir: str):
    p = os.path.join(opt_dir, "multiobjective", "optimization_summary.csv")
    return pd.read_csv(p) if os.path.exists(p) else None

@st.cache_data
def load_nsga2_pareto(opt_dir: str, store: str):
    p = os.path.join(opt_dir, "multiobjective", store.capitalize(), "pareto_front.csv")
    return pd.read_csv(p) if os.path.exists(p) else None

@st.cache_data
def load_alloc_penalty_summary(opt_dir: str):
    p = os.path.join(opt_dir, "allocation", "allocation_summary.csv")
    return pd.read_csv(p) if os.path.exists(p) else None

@st.cache_data
def load_alloc_penalty_conv(opt_dir: str):
    p = os.path.join(opt_dir, "allocation", "convergence.csv")
    return pd.read_csv(p) if os.path.exists(p) else None

@st.cache_data
def load_alloc_dp_summary(opt_dir: str):
    p = os.path.join(opt_dir, "allocation_death_penalty", "allocation_summary_death_penalty.csv")
    return pd.read_csv(p) if os.path.exists(p) else None

@st.cache_data
def load_alloc_dp_conv(opt_dir: str):
    p = os.path.join(opt_dir, "allocation_death_penalty", "convergence.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


# ── Pre-load all data ──────────────────────────────────────────────────────────
df_hc_sum   = load_hc_summary(_opt_dir)
df_sa_sum   = load_sa_summary(_opt_dir)
df_mc_sum   = load_mc_summary(_opt_dir)
df_n2_sum   = load_nsga2_summary(_opt_dir)
df_alloc_p  = load_alloc_penalty_summary(_opt_dir)
df_alloc_dp = load_alloc_dp_summary(_opt_dir)

# ── Build combined O1 comparison table ────────────────────────────────────────
STORES = ["Baltimore", "Lancaster", "Philadelphia", "Richmond"]

def _build_o1_comparison():
    rows = []
    for s in STORES:
        row = {"store": s}
        sl = s.lower()
        if df_mc_sum is not None:
            mc = df_mc_sum[df_mc_sum["store"] == sl]
            row["Monte Carlo"] = float(mc["lucro_maximo"].iloc[0]) if len(mc) else None
        if df_hc_sum is not None:
            hc = df_hc_sum[df_hc_sum["store"] == sl]
            row["Hill Climbing"] = float(hc["lucro_maximo"].iloc[0]) if len(hc) else None
        if df_sa_sum is not None:
            sa = df_sa_sum[df_sa_sum["store"] == sl]
            row["Simulated Annealing"] = float(sa["lucro_maximo"].iloc[0]) if len(sa) else None
        if df_n2_sum is not None:
            n2 = df_n2_sum[df_n2_sum["store"] == sl]
            row["NSGA-II"] = float(n2["max_profit"].iloc[0]) if len(n2) else None
        rows.append(row)
    return pd.DataFrame(rows)

df_o1_cmp = _build_o1_comparison()


# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.markdown(
    '<div style="font-size:0.88rem;font-weight:700;color:#E2E8F0;'
    'letter-spacing:0.04em;margin-bottom:16px;">DSS USA STORES</div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div style="font-size:0.70rem;color:#64748B;letter-spacing:0.08em;'
    'text-transform:uppercase;font-weight:700;margin-bottom:12px;">Filtros O1</div>',
    unsafe_allow_html=True,
)

loja_sel = st.sidebar.selectbox("Loja (detalhe O1)", STORES, index=0)

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div style="font-size:0.70rem;color:#64748B;letter-spacing:0.08em;'
    'text-transform:uppercase;font-weight:700;margin-bottom:8px;">Resumo de Algoritmos</div>',
    unsafe_allow_html=True,
)
st.sidebar.caption("O1 — 4 algoritmos comparados:")
st.sidebar.caption("   1. Monte Carlo (baseline estocástico)")
st.sidebar.caption("   2. Hill Climbing (10 restarts × 2000 iter)")
st.sidebar.caption("   3. Simulated Annealing (5 restarts)")
st.sidebar.caption("   4. NSGA-II (max da fronteira Pareto)")
st.sidebar.markdown("---")
st.sidebar.caption("O2 — 2 estratégias de restrição:")
st.sidebar.caption("   1. HC + Penalty Function")
st.sidebar.caption("   2. HC + Death Penalty")
st.sidebar.caption("Restrição: <= 10 000 unidades/semana")

# ── Page header ────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 36px 0 20px 0; border-bottom: 2px solid #1E3A8A; margin-bottom: 32px;">
    <div style="font-size: 1.85rem; font-weight: 800; color: #0F172A;
         letter-spacing: -0.03em; line-height: 1.2; margin-bottom: 10px;">
        Otimização Monobjetivo — O1 e O2
    </div>
    <div style="font-size: 0.88rem; color: #64748B; max-width: 760px; line-height: 1.65;">
        Comparação de 4 algoritmos para O1 (maximização de lucro por loja) e
        2 estratégias para O2 (alocação conjunta com restrição logística de 10 000 unidades).
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# O1 — COMPARAÇÃO DE ALGORITMOS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">O1: Maximização de Lucro por Loja — Comparação de Algoritmos</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-desc">'
    'Cada loja é otimizada de forma independente com 21 variáveis de decisão: '
    'desconto diário (0–30%), staff experto (hr_x) e staff junior (hr_j) para cada dia da semana. '
    'Os 4 algoritmos partem do mesmo espaço de pesquisa e são comparados pelo melhor lucro semanal encontrado.'
    '</div>',
    unsafe_allow_html=True,
)

# ── Tabela de comparação ───────────────────────────────────────────────────────
alg_cols = ["Monte Carlo", "Hill Climbing", "Simulated Annealing", "NSGA-II"]
available_alg = [c for c in alg_cols if c in df_o1_cmp.columns]

if available_alg:
    st.markdown('<div class="section-label">Lucro Máximo Semanal por Algoritmo e Loja ($)</div>', unsafe_allow_html=True)

    # Totals row
    totals = {"store": "TOTAL"}
    for col in available_alg:
        totals[col] = df_o1_cmp[col].sum() if df_o1_cmp[col].notna().all() else None

    df_display = pd.concat([df_o1_cmp, pd.DataFrame([totals])], ignore_index=True)
    df_fmt = df_display.copy()
    df_fmt["store"] = df_fmt["store"].str.capitalize()
    for col in available_alg:
        df_fmt[col] = df_fmt[col].apply(lambda x: f"${x:,.0f}" if pd.notna(x) else "N/A")
    df_fmt = df_fmt.rename(columns={"store": "Loja"})
    st.dataframe(df_fmt, hide_index=True, use_container_width=True)

    # ── Grouped bar chart ─────────────────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Comparativo Visual — Lucro por Algoritmo</div>', unsafe_allow_html=True)

    fig_cmp = go.Figure()
    for alg in available_alg:
        values = df_o1_cmp[alg].tolist()
        highlight = [alg == "Simulated Annealing"] * len(STORES)  # best overall
        fig_cmp.add_trace(go.Bar(
            name=alg,
            x=[s.capitalize() for s in df_o1_cmp["store"]],
            y=values,
            marker_color=ALG_COLORS.get(alg, "#94A3B8"),
            opacity=0.88,
            text=[f"${v:,.0f}" for v in values],
            textposition="outside",
            textfont=dict(size=9),
        ))

    fig_cmp.update_layout(
        barmode="group",
        height=350,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin=dict(t=30, b=30, l=10, r=10),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            font=dict(color="#0F172A", size=11),
        ),
        xaxis=dict(
            tickfont=dict(color="#0F172A", size=11),
            gridcolor="#F1F5F9",
        ),
        yaxis=dict(
            title=dict(text="Lucro Semanal ($)", font=dict(color="#0F172A", size=11)),
            tickfont=dict(color="#0F172A", size=10),
            gridcolor="#F1F5F9",
            zeroline=True,
            zerolinecolor="#E2E8F0",
            tickformat="$,.0f",
        ),
    )
    st.plotly_chart(fig_cmp, use_container_width=True)

# ── Per-store detail via tabs ──────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    f'<div class="section-label">Detalhe por Loja — {loja_sel}</div>',
    unsafe_allow_html=True,
)

tab_mc, tab_hc, tab_sa, tab_n2 = st.tabs([
    "Monte Carlo", "Hill Climbing", "Simulated Annealing", "NSGA-II"
])

def _plan_table(df_plan):
    """Render a weekly plan dataframe as a clean display table."""
    if df_plan is None:
        return None
    return pd.DataFrame({
        "Dia":         df_plan["day_label"],
        "FDS":         df_plan["is_weekend"].map(lambda x: "S" if x else ""),
        "Clientes":    df_plan["customers"],
        "% Desconto":  df_plan["pr"].apply(lambda x: f"{float(x)*100:.2f}%"),
        "Experts":     df_plan["hr_x"].astype(int),
        "Juniores":    df_plan["hr_j"].astype(int),
        "Staff Total": df_plan["total_staff"].astype(int),
    })


# Tab 1 — Monte Carlo ─────────────────────────────────────────────────────────
with tab_mc:
    df_mc_plan = load_mc_plan(_opt_dir, loja_sel)
    df_mc_conv = load_mc_convergence(_opt_dir, loja_sel)
    mc_lucro   = None

    if df_mc_sum is not None:
        mc_row = df_mc_sum[df_mc_sum["store"] == loja_sel.lower()]
        if len(mc_row):
            mc_lucro = float(mc_row["lucro_maximo"].iloc[0])
            mc_medio = float(mc_row["lucro_medio"].iloc[0])
            mc_std   = float(mc_row["lucro_std"].iloc[0])
            mc_n     = int(mc_row["n_samples"].iloc[0])

    if mc_lucro is not None:
        col1, col2, col3 = st.columns(3, gap="medium")
        col1.metric("Melhor Lucro", f"${mc_lucro:,.0f}")
        col2.metric("Lucro Médio", f"${mc_medio:,.0f}")
        col3.metric("Desvio Padrão", f"${mc_std:,.0f}")
        st.caption(f"Amostras avaliadas: {mc_n:,} | Seed: 42")

    if df_mc_plan is not None:
        st.markdown("**Plano Semanal Ótimo (Monte Carlo)**")
        st.dataframe(_plan_table(df_mc_plan), hide_index=True, use_container_width=True)

    if df_mc_conv is not None:
        step = max(1, len(df_mc_conv) // 400)
        df_cv = df_mc_conv.iloc[::step].copy()
        fig = go.Figure(go.Scatter(
            x=df_cv["sample"], y=df_cv["best_lucro"],
            mode="lines", line=dict(color="#9C27B0", width=1.8),
            fill="tozeroy", fillcolor="rgba(156,39,176,0.07)",
        ))
        fig.update_layout(
            height=230, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
            margin=dict(t=10, b=30, l=10, r=10), showlegend=False,
            title=dict(text="Convergência Monte Carlo — melhor lucro acumulado",
                       font=dict(size=11, color="#64748B")),
            xaxis=dict(title="Amostras", tickfont=dict(size=10), gridcolor="#F1F5F9", zeroline=False),
            yaxis=dict(title="Lucro ($)", tickfont=dict(size=10), gridcolor="#F1F5F9",
                       tickformat="$,.0f", zeroline=False),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"Dados Monte Carlo não encontrados para {loja_sel}. Execute: python scripts/run_monte_carlo_o1.py")


# Tab 2 — Hill Climbing ───────────────────────────────────────────────────────
with tab_hc:
    df_hc_plan = load_hc_plan(_opt_dir, loja_sel)
    df_hc_conv = load_hc_convergence(_opt_dir, loja_sel)
    hc_lucro   = None

    if df_hc_sum is not None:
        hc_row = df_hc_sum[df_hc_sum["store"] == loja_sel.lower()]
        if len(hc_row):
            hc_lucro    = float(hc_row["lucro_maximo"].iloc[0])
            hc_staff    = int(hc_row["total_staff"].iloc[0])
            hc_discount = float(hc_row["avg_discount"].iloc[0])

    if hc_lucro is not None:
        col1, col2, col3 = st.columns(3, gap="medium")
        col1.metric("Melhor Lucro", f"${hc_lucro:,.0f}")
        col2.metric("Staff Total", hc_staff)
        col3.metric("Desconto Médio", f"{hc_discount:.2f}%")
        st.caption("10 restarts × 2 000 iterações | Seed: 42")

    if df_hc_plan is not None:
        st.markdown("**Plano Semanal Ótimo (Hill Climbing)**")
        st.dataframe(_plan_table(df_hc_plan), hide_index=True, use_container_width=True)

    if df_hc_conv is not None:
        step = max(1, len(df_hc_conv) // 400)
        df_cv = df_hc_conv.iloc[::step].copy()
        lucro_max = float(df_hc_conv["lucro"].max())
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_cv["iteration"], y=df_cv["lucro"],
            mode="lines", line=dict(color="#1E3A8A", width=2),
            fill="tozeroy", fillcolor="rgba(30,58,138,0.07)",
        ))
        fig.add_trace(go.Scatter(
            x=[df_hc_conv["iteration"].max()], y=[lucro_max],
            mode="markers+text",
            marker=dict(color="#1E3A8A", size=9),
            text=[f"  ${lucro_max:,.0f}"],
            textposition="middle right",
            textfont=dict(color="#0F172A", size=10),
            showlegend=False,
        ))
        fig.update_layout(
            height=230, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
            margin=dict(t=10, b=30, l=10, r=80), showlegend=False,
            title=dict(text="Convergência Hill Climbing",
                       font=dict(size=11, color="#64748B")),
            xaxis=dict(title="Iteração", tickfont=dict(size=10), gridcolor="#F1F5F9", zeroline=False),
            yaxis=dict(title="Lucro ($)", tickfont=dict(size=10), gridcolor="#F1F5F9",
                       tickformat="$,.0f", zeroline=False),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"Dados HC não encontrados para {loja_sel}. Execute: python scripts/run_individual_optimization.py")


# Tab 3 — Simulated Annealing ─────────────────────────────────────────────────
with tab_sa:
    df_sa_plan = load_sa_plan(_opt_dir, loja_sel)
    df_sa_conv = load_sa_convergence(_opt_dir, loja_sel)
    sa_lucro   = None

    if df_sa_sum is not None:
        sa_row = df_sa_sum[df_sa_sum["store"] == loja_sel.lower()]
        if len(sa_row):
            sa_lucro    = float(sa_row["lucro_maximo"].iloc[0])
            sa_staff    = int(sa_row["total_staff"].iloc[0])
            sa_discount = float(sa_row["avg_discount"].iloc[0])

    if sa_lucro is not None:
        col1, col2, col3 = st.columns(3, gap="medium")
        col1.metric("Melhor Lucro", f"${sa_lucro:,.0f}")
        col2.metric("Staff Total", sa_staff)
        col3.metric("Desconto Médio", f"{sa_discount:.2f}%")
        st.caption("5 restarts | T_init=5000 | alpha=0.995 | ~1701 passos de temperatura")

    if df_sa_plan is not None:
        st.markdown("**Plano Semanal Ótimo (Simulated Annealing)**")
        st.dataframe(_plan_table(df_sa_plan), hide_index=True, use_container_width=True)

    if df_sa_conv is not None:
        step = max(1, len(df_sa_conv) // 400)
        df_cv = df_sa_conv.iloc[::step].copy()
        fig = go.Figure(go.Scatter(
            x=df_cv["temp_step"], y=df_cv["best_lucro"],
            mode="lines", line=dict(color="#FF6F00", width=2),
            fill="tozeroy", fillcolor="rgba(255,111,0,0.07)",
        ))
        fig.update_layout(
            height=230, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
            margin=dict(t=10, b=30, l=10, r=10), showlegend=False,
            title=dict(text="Convergência Simulated Annealing (melhor restart)",
                       font=dict(size=11, color="#64748B")),
            xaxis=dict(title="Passos de Temperatura", tickfont=dict(size=10), gridcolor="#F1F5F9", zeroline=False),
            yaxis=dict(title="Lucro ($)", tickfont=dict(size=10), gridcolor="#F1F5F9",
                       tickformat="$,.0f", zeroline=False),
        )
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.info(f"Dados SA não encontrados para {loja_sel}. Execute: python scripts/run_simulated_annealing_o1.py")


# Tab 4 — NSGA-II ─────────────────────────────────────────────────────────────
with tab_n2:
    df_n2_pareto = load_nsga2_pareto(_opt_dir, loja_sel)
    n2_lucro     = None

    if df_n2_sum is not None:
        n2_row = df_n2_sum[df_n2_sum["store"] == loja_sel.lower()]
        if len(n2_row):
            n2_lucro = float(n2_row["max_profit"].iloc[0])
            n2_n     = int(n2_row["n_pareto"].iloc[0])
            n2_staff = int(n2_row["staff_at_max_profit"].iloc[0])

    if n2_lucro is not None:
        col1, col2, col3 = st.columns(3, gap="medium")
        col1.metric("Melhor Lucro (max Pareto)", f"${n2_lucro:,.0f}")
        col2.metric("Soluções Pareto", n2_n)
        col3.metric("Staff @ Max Lucro", n2_staff)
        st.caption("NSGA-II | Pop=100 | Até 500 gerações | Seed: 42")

    if df_n2_pareto is not None:
        st.markdown("**Fronteira de Pareto — Lucro vs. Staff**")
        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(
            x=df_n2_pareto["staff"], y=df_n2_pareto["lucro"],
            mode="markers",
            marker=dict(
                color=df_n2_pareto["lucro"],
                colorscale=[[0, "#E2E8F0"], [0.5, "#34D399"], [1, "#059669"]],
                size=8, opacity=0.85,
                colorbar=dict(title=dict(text="Lucro ($)", font=dict(size=10))),
            ),
            hovertemplate="Staff: %{x}<br>Lucro: $%{y:,.0f}<extra></extra>",
        ))
        # Highlight max-profit point
        if len(df_n2_pareto):
            best = df_n2_pareto.loc[df_n2_pareto["lucro"].idxmax()]
            fig_p.add_trace(go.Scatter(
                x=[best["staff"]], y=[best["lucro"]],
                mode="markers", name="Max Lucro",
                marker=dict(symbol="star", size=18, color="#DC2626",
                            line=dict(width=1.5, color="#0F172A")),
            ))
        fig_p.add_hline(y=0, line_dash="dot", line_color="#94A3B8", line_width=1.2)
        fig_p.update_layout(
            height=260, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
            margin=dict(t=10, b=40, l=10, r=60), showlegend=False,
            xaxis=dict(title="Staff Total (semana)", tickfont=dict(size=10),
                       gridcolor="#F1F5F9", zeroline=False),
            yaxis=dict(title="Lucro ($)", tickfont=dict(size=10),
                       gridcolor="#F1F5F9", tickformat="$,.0f", zeroline=False),
        )
        st.plotly_chart(fig_p, use_container_width=True)

        # Best plan from Pareto
        best_row = df_n2_pareto.loc[df_n2_pareto["lucro"].idxmax()]
        day_cols = ["Sun", "Mon", "Tue", "Wed", "Thu", "Fri", "Sat"]
        day_labels_pt = ["Dom", "Seg", "Ter", "Qua", "Qui", "Sex", "Sab"]
        plan_rows = []
        for i, (dc, dl) in enumerate(zip(day_cols, day_labels_pt)):
            pr   = float(best_row.get(f"pr_{dc}", 0))
            hr_x = int(round(float(best_row.get(f"hr_x_{dc}", 0))))
            hr_j = int(round(float(best_row.get(f"hr_j_{dc}", 0))))
            plan_rows.append({
                "Dia": dl, "% Desconto": f"{pr*100:.2f}%",
                "Experts": hr_x, "Juniores": hr_j, "Staff Total": hr_x + hr_j,
            })
        st.markdown("**Plano Semanal da Solução com Máximo Lucro (Pareto)**")
        st.dataframe(pd.DataFrame(plan_rows), hide_index=True, use_container_width=True)
    else:
        st.info(f"Dados NSGA-II não encontrados para {loja_sel}. Execute: python scripts/run_optimization.py")


# ══════════════════════════════════════════════════════════════════════════════
# O2 — ALOCAÇÃO DE REDE COM RESTRIÇÃO LOGÍSTICA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">O2: Alocação de Rede com Restrição Logística</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-desc">'
    'Otimização conjunta das 4 lojas garantindo que o volume total de unidades vendidas '
    'não excede a capacidade da rede logística (10 000 unidades/semana). '
    'Dois algoritmos Hill Climbing com estratégias distintas de gestão de restrição são comparados.'
    '</div>',
    unsafe_allow_html=True,
)

LIMIT = 10_000

# ── Comparação: Penalty Function vs. Death Penalty ────────────────────────────
st.markdown('<div class="section-label">Comparativo — Penalty Function vs. Death Penalty</div>', unsafe_allow_html=True)

def _build_o2_comparison():
    """Build side-by-side comparison of both O2 algorithms."""
    methods = {
        "HC + Penalty Function": df_alloc_p,
        "HC + Death Penalty":    df_alloc_dp,
    }
    rows = []
    for name, df in methods.items():
        if df is None:
            continue
        total = df[df["store"] == "TOTAL"].iloc[0]
        rows.append({
            "Algoritmo":    name,
            "Lucro Total":  f"${float(total['lucro']):,.0f}",
            "Unidades":     f"{int(total['unidades']):,}",
            "Restrição OK": "SIM" if int(total['unidades']) <= LIMIT else "NAO",
            "Staff Total":  int(total["total_staff"]),
        })
    return pd.DataFrame(rows)

df_o2_cmp = _build_o2_comparison()
if not df_o2_cmp.empty:
    st.dataframe(df_o2_cmp, hide_index=True, use_container_width=True)

# ── Side-by-side cards ────────────────────────────────────────────────────────
col_pen, col_dp = st.columns(2, gap="medium")

def _constraint_card(label, df_summary):
    if df_summary is None:
        return
    total = df_summary[df_summary["store"] == "TOTAL"].iloc[0]
    units = int(total["unidades"])
    lucro = float(total["lucro"])
    ok    = units <= LIMIT
    css   = "constraint-ok" if ok else "constraint-err"
    badge = "Restrição Respeitada" if ok else "ALERTA: Violação"
    st.markdown(f"""
    <div class="{css}">
        <div class="c-title">{label}</div>
        <div class="c-value">{units:,} unidades</div>
        <div style="font-size:0.80rem;color:{'#065F46' if ok else '#7F1D1D'};margin-top:4px;">
            Lucro: ${lucro:,.0f}
        </div>
        <div><span class="c-badge">{badge}</span></div>
    </div>
    """, unsafe_allow_html=True)

with col_pen:
    _constraint_card("HC + Penalty Function", df_alloc_p)

with col_dp:
    _constraint_card("HC + Death Penalty", df_alloc_dp)

st.markdown("<br>", unsafe_allow_html=True)

# ── Resultado detalhado por loja (ambos algoritmos) ───────────────────────────
st.markdown('<div class="section-label">Resultado Detalhado por Loja</div>', unsafe_allow_html=True)

def _format_alloc_df(df):
    if df is None:
        return None
    stores_df = df[df["store"] != "TOTAL"].copy()
    stores_df["Loja"]        = stores_df["store"].str.capitalize()
    stores_df["Lucro ($)"]   = stores_df["lucro"].apply(lambda x: f"${x:,.0f}")
    stores_df["Unidades"]    = stores_df["unidades"].apply(lambda x: f"{int(x):,}")
    stores_df["Staff Total"] = stores_df["total_staff"].astype(int)
    stores_df["Desc. Médio"] = stores_df["avg_discount"].apply(lambda x: f"{x:.2f}%")
    return stores_df[["Loja", "Lucro ($)", "Unidades", "Staff Total", "Desc. Médio"]]

tab_o2_p, tab_o2_dp = st.tabs(["HC + Penalty Function", "HC + Death Penalty"])

with tab_o2_p:
    df_p_fmt = _format_alloc_df(df_alloc_p)
    if df_p_fmt is not None:
        st.dataframe(df_p_fmt, hide_index=True, use_container_width=True)

        # Convergence chart
        df_pconv = load_alloc_penalty_conv(_opt_dir)
        if df_pconv is not None:
            step = max(1, len(df_pconv) // 300)
            df_cv = df_pconv.iloc[::step].copy()
            fig2 = go.Figure()
            fig2.add_trace(go.Scatter(
                x=df_cv["iteration"], y=df_cv["units"],
                mode="lines", name="Unidades",
                line=dict(color="#1E3A8A", width=2),
            ))
            fig2.add_hline(
                y=LIMIT, line_dash="dot", line_color="#DC2626", line_width=1.5,
                annotation_text="Limite 10 000",
                annotation_font=dict(color="#DC2626", size=10),
            )
            fig2.update_layout(
                height=240, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                margin=dict(t=30, b=30, l=10, r=40), showlegend=False,
                title=dict(text="Convergência HC + Penalty — Unidades totais",
                           font=dict(size=11, color="#64748B")),
                xaxis=dict(title="Iteração", tickfont=dict(size=10), gridcolor="#F1F5F9"),
                yaxis=dict(title="Unidades", tickfont=dict(size=10), gridcolor="#F1F5F9", tickformat=",d"),
            )
            st.plotly_chart(fig2, use_container_width=True)
    else:
        st.info("Execute: python scripts/run_allocation_optimization.py")

with tab_o2_dp:
    df_dp_fmt = _format_alloc_df(df_alloc_dp)
    if df_dp_fmt is not None:
        st.dataframe(df_dp_fmt, hide_index=True, use_container_width=True)

        # Convergence chart
        df_dpconv = load_alloc_dp_conv(_opt_dir)
        if df_dpconv is not None:
            step = max(1, len(df_dpconv) // 300)
            df_cv = df_dpconv.iloc[::step].copy()
            fig3 = go.Figure()
            fig3.add_trace(go.Scatter(
                x=df_cv["iteration"], y=df_cv["units"],
                mode="lines", name="Unidades",
                line=dict(color="#FF6F00", width=2),
            ))
            fig3.add_hline(
                y=LIMIT, line_dash="dot", line_color="#DC2626", line_width=1.5,
                annotation_text="Limite 10 000",
                annotation_font=dict(color="#DC2626", size=10),
            )
            fig3.update_layout(
                height=240, plot_bgcolor="#FFFFFF", paper_bgcolor="#FFFFFF",
                margin=dict(t=30, b=30, l=10, r=40), showlegend=False,
                title=dict(text="Convergência HC + Death Penalty — Unidades totais",
                           font=dict(size=11, color="#64748B")),
                xaxis=dict(title="Iteração", tickfont=dict(size=10), gridcolor="#F1F5F9"),
                yaxis=dict(title="Unidades", tickfont=dict(size=10), gridcolor="#F1F5F9", tickformat=",d"),
            )
            st.plotly_chart(fig3, use_container_width=True)
    else:
        st.info("Execute: python scripts/run_allocation_optimization_death_penalty.py")

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dss-footer">
    DSS &nbsp;|&nbsp; Otimização Monobjetivo O1 + O2 &nbsp;|&nbsp;
    Monte Carlo &middot; Hill Climbing &middot; Simulated Annealing &middot; NSGA-II &middot; Penalty &middot; Death Penalty
</div>
""", unsafe_allow_html=True)
