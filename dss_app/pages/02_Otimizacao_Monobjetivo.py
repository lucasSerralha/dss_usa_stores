"""
O1 + O2: Otimização Monobjetivo
Página 02 — DSS USA Stores

O1 — Maximização de lucro individual por loja (Hill Climbing, 10 restarts × 2 000 iter)
O2 — Alocação conjunta com restrição logística de 10 000 unidades/semana
       Algoritmo único: Hill Climbing com Death Penalty (heurística monobjetivo)
"""
import streamlit as st
import sys
import os
import pandas as pd
import plotly.graph_objects as go

sys.dont_write_bytecode = True

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Otimização Monobjetivo — DSS USA Stores",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Corporate CSS (paleta azul escuro / cinzento — consistente com app.py) ────
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
  .constraint-ok .c-value { font-size: 1.6rem; font-weight: 800;
    color: #065F46; }
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
  .constraint-err .c-value { font-size: 1.6rem; font-weight: 800;
    color: #7F1D1D; }
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
  .arch-strip {
    margin-bottom: 24px; padding: 14px 18px;
    border-radius: 8px; border-left: 4px solid #1E3A8A;
    background: #F8FAFC; font-size: 0.85rem; color: #1E293B;
    line-height: 1.7;
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
_opt_dir  = os.path.join(_root_dir, "results", "03_Optimization_Report")

# ── Data loaders ───────────────────────────────────────────────────────────────
@st.cache_data
def load_o1_summary(opt_dir: str):
    path = os.path.join(opt_dir, "individual", "optimization_summary.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


@st.cache_data
def load_o1_plan(opt_dir: str, store: str):
    path = os.path.join(opt_dir, "individual", f"{store.lower()}_best_plan.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


@st.cache_data
def load_o1_convergence(opt_dir: str, store: str):
    path = os.path.join(opt_dir, "individual", f"{store.lower()}_convergence.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


@st.cache_data
def load_alloc_summary(opt_dir: str):
    path = os.path.join(opt_dir, "allocation", "allocation_summary.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


@st.cache_data
def load_alloc_convergence(opt_dir: str):
    path = os.path.join(opt_dir, "allocation", "convergence.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


# ── Pre-load data ──────────────────────────────────────────────────────────────
df_o1_summary = load_o1_summary(_opt_dir)
df_alloc_hc   = load_alloc_summary(_opt_dir)
df_alloc_conv = load_alloc_convergence(_opt_dir)

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

LOJAS = ["Baltimore", "Lancaster", "Philadelphia", "Richmond"]
loja_sel = st.sidebar.selectbox("Loja (detalhe O1)", LOJAS, index=0)

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div style="font-size:0.70rem;color:#64748B;letter-spacing:0.08em;'
    'text-transform:uppercase;font-weight:700;margin-bottom:8px;">Parametros Heuristica</div>',
    unsafe_allow_html=True,
)
st.sidebar.caption("Algoritmo O1 e O2: Hill Climbing")
st.sidebar.caption("Gestao de restricoes: Death Penalty")
st.sidebar.caption("Restarts: 10 | Iteracoes: 2 000")
st.sidebar.caption("Variaveis de decisao: 21 (pr, hr_x, hr_j × 7 dias)")
st.sidebar.caption("Restricao O2: <= 10 000 unidades/semana")

# ── Page header ────────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 36px 0 20px 0; border-bottom: 2px solid #1E3A8A; margin-bottom: 32px;">
    <div style="font-size: 1.85rem; font-weight: 800; color: #0F172A;
         letter-spacing: -0.03em; line-height: 1.2; margin-bottom: 10px;">
        Otimizacao Monobjetivo — O1 e O2
    </div>
    <div style="font-size: 0.88rem; color: #64748B; max-width: 760px; line-height: 1.65;">
        Plano operacional semanal otimizado por loja (O1) e alocacao conjunta com
        restricao logistica de rede (O2). Ambos os objetivos sao resolvidos
        estritamente por heuristicas monobjetivo: Hill Climbing com Death Penalty.
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# O1 — MAXIMIZACAO DE LUCRO POR LOJA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">O1: Maximizacao de Lucro por Loja</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-desc">'
    'Hill Climbing com random restarts otimiza 21 variaveis de decisao por loja: '
    'desconto diario (0–30%), staff experto (0–15) e staff junior (0–15) para cada '
    'dia da semana. O comparativo abaixo quantifica o ganho face a uma solucao inicial aleatoria.'
    '</div>',
    unsafe_allow_html=True,
)

if df_o1_summary is not None:
    # ── Summary bar chart: lucro_maximo por loja ──────────────────────────────
    st.markdown('<div class="section-label">Lucro Semanal Otimo — Todas as Lojas</div>', unsafe_allow_html=True)

    df_summ_plot = df_o1_summary.copy()
    df_summ_plot["store_label"] = df_summ_plot["store"].str.capitalize()

    fig_o1_bars = go.Figure(go.Bar(
        x=df_summ_plot["lucro_maximo"],
        y=df_summ_plot["store_label"],
        orientation="h",
        marker_color=[
            "#1E3A8A" if s.lower() == loja_sel.lower() else "#CBD5E1"
            for s in df_summ_plot["store_label"]
        ],
        text=df_summ_plot["lucro_maximo"].map("${:,.0f}".format),
        textposition="outside",
        textfont=dict(color="#0F172A", size=11),
    ))
    fig_o1_bars.update_layout(
        height=220,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin=dict(t=10, b=30, l=10, r=110),
        showlegend=False,
        xaxis=dict(
            title=dict(text="Lucro Semanal ($)", font=dict(color="#0F172A", size=11)),
            tickfont=dict(color="#0F172A", size=10),
            gridcolor="#F1F5F9",
            zeroline=False,
        ),
        yaxis=dict(
            tickfont=dict(color="#0F172A", size=11),
            autorange="reversed",
        ),
    )
    st.plotly_chart(fig_o1_bars, use_container_width=True)

# ── Per-store detail ───────────────────────────────────────────────────────────
df_o1_plan = load_o1_plan(_opt_dir, loja_sel)
df_o1_conv = load_o1_convergence(_opt_dir, loja_sel)

if df_o1_plan is not None and df_o1_conv is not None:
    lucro_inicial = float(df_o1_conv.iloc[0]["lucro"])
    lucro_otimo   = float(df_o1_conv["lucro"].max())
    ganho_abs     = lucro_otimo - lucro_inicial
    ganho_pct     = (ganho_abs / abs(lucro_inicial) * 100) if lucro_inicial != 0 else float("inf")

    # ── Comparativo: Solucao Inicial vs Hill Climbing ─────────────────────────
    st.markdown(
        f'<div class="section-label">Comparativo — Antes vs. Depois da Otimizacao ({loja_sel})</div>',
        unsafe_allow_html=True,
    )

    col_base, col_mid, col_opt = st.columns([5, 1, 5], gap="small")

    with col_base:
        st.markdown(f"""
        <div class="cmp-card">
            <div class="cmp-label">Solucao Inicial (Aleatoria)</div>
            <div class="cmp-value">${lucro_inicial:,.0f}</div>
            <div class="cmp-sub">Ponto de partida do Hill Climbing.<br>
            Solucao aleatoria sem qualquer optimizacao.</div>
        </div>
        """, unsafe_allow_html=True)

    with col_mid:
        st.markdown(
            '<div style="display:flex;align-items:center;justify-content:center;'
            'height:100%;font-size:1.4rem;color:#94A3B8;padding-top:20px;">&#8594;</div>',
            unsafe_allow_html=True,
        )

    with col_opt:
        delta_class = "cmp-delta-pos" if ganho_abs >= 0 else "cmp-delta-neg"
        delta_sign  = "+" if ganho_abs >= 0 else ""
        st.markdown(f"""
        <div class="cmp-card">
            <div class="cmp-label">Hill Climbing + Death Penalty (10 restarts × 2 000 iter)</div>
            <div class="cmp-value cmp-value-blue">${lucro_otimo:,.0f}</div>
            <div class="cmp-sub">Melhor solucao encontrada.<br>
            21 variaveis de decisao por loja.</div>
            <span class="{delta_class}">{delta_sign}${ganho_abs:,.0f} ({delta_sign}{ganho_pct:.1f}%)</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # ── Plano semanal otimo ────────────────────────────────────────────────────
    st.markdown('<div class="section-label">Plano Semanal Otimo</div>', unsafe_allow_html=True)

    df_plan_display = pd.DataFrame({
        "Dia":        df_o1_plan["day_label"],
        "FDS":        df_o1_plan["is_weekend"].map(lambda x: "S" if x else ""),
        "Clientes":   df_o1_plan["customers"],
        "% Promocao": df_o1_plan["pr"].map("{:.1f}%".format),
        "Experts":    df_o1_plan["hr_x"].astype(int),
        "Juniores":   df_o1_plan["hr_j"].astype(int),
        "Staff Total":df_o1_plan["total_staff"].astype(int),
    })
    st.dataframe(df_plan_display, hide_index=True, use_container_width=True)

    # ── Convergencia do Hill Climbing ──────────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Convergencia do Hill Climbing</div>', unsafe_allow_html=True)
    st.caption(
        "Evolucao do melhor lucro encontrado ao longo das iteracoes. "
        "A curva ascendente demonstra que o algoritmo escapa de otimos locais via random restarts."
    )

    df_conv_plot = df_o1_conv[df_o1_conv["iteration"] % 20 == 0].copy()

    fig_conv = go.Figure()
    fig_conv.add_trace(go.Scatter(
        x=df_conv_plot["iteration"],
        y=df_conv_plot["lucro"],
        mode="lines",
        line=dict(color="#1E3A8A", width=2.5),
        name="Lucro ($)",
        fill="tozeroy",
        fillcolor="rgba(30,58,138,0.07)",
    ))
    fig_conv.add_trace(go.Scatter(
        x=[df_o1_conv["iteration"].max()],
        y=[lucro_otimo],
        mode="markers+text",
        marker=dict(color="#1E3A8A", size=10, symbol="circle"),
        text=[f"  ${lucro_otimo:,.0f}"],
        textposition="middle right",
        textfont=dict(color="#0F172A", size=11),
        name="Solucao Otima",
        showlegend=False,
    ))
    fig_conv.update_layout(
        height=280,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin=dict(t=10, b=40, l=10, r=80),
        showlegend=False,
        xaxis=dict(
            title=dict(text="Iteracao", font=dict(color="#0F172A", size=11)),
            tickfont=dict(color="#0F172A", size=10),
            gridcolor="#F1F5F9",
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(text="Lucro ($)", font=dict(color="#0F172A", size=11)),
            tickfont=dict(color="#0F172A", size=10),
            gridcolor="#F1F5F9",
            zeroline=False,
            tickformat="$,.0f",
        ),
    )
    st.plotly_chart(fig_conv, use_container_width=True)

else:
    st.info(
        f"Dados de otimizacao individual nao encontrados para {loja_sel}. "
        "Execute: python scripts/run_individual_optimization.py"
    )

# ══════════════════════════════════════════════════════════════════════════════
# O2 — ALOCACAO DE REDE COM RESTRICAO LOGISTICA
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("---")
st.markdown('<div class="section-title">O2: Alocacao de Rede com Restricao Logistica</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-desc">'
    'Otimizacao conjunta das 4 lojas garantindo que o volume total de unidades vendidas '
    'nao excede a capacidade da rede logistica (10 000 unidades/semana). '
    'O problema e estritamente monobjetivo — maximizacao de lucro sujeita a restricao de '
    'capacidade — resolvido por Hill Climbing com Death Penalty: solucoes que violam a '
    'restricao recebem penalizacao proporcional ao excesso, tornando-as sempre inelegíveis.'
    '</div>',
    unsafe_allow_html=True,
)

# ── Indicadores de restricao: Antes vs. Depois ────────────────────────────────
LIMIT = 10_000

if df_alloc_conv is not None:
    units_inicial  = int(df_alloc_conv.iloc[0]["units"])
    profit_inicial = float(df_alloc_conv.iloc[0]["profit"])
else:
    units_inicial  = None
    profit_inicial = None

units_hc  = None
profit_hc = None

if df_alloc_hc is not None:
    total_hc  = df_alloc_hc[df_alloc_hc["store"] == "TOTAL"].iloc[0]
    units_hc  = int(total_hc["unidades"])
    profit_hc = float(total_hc["lucro"])

st.markdown(
    '<div class="section-label">Estado da Restricao Logistica — Antes vs. Depois da Otimizacao</div>',
    unsafe_allow_html=True,
)


def _constraint_card(label: str, units: int | None, extra: str = "") -> str:
    if units is None:
        return ""
    ok  = units <= LIMIT
    css = "constraint-ok" if ok else "constraint-err"
    badge = "Restricao Respeitada" if ok else "ALERTA: Quebra de Restricao — Death Penalty Aplicada"
    return f"""
    <div class="{css}">
        <div class="c-title">{label}</div>
        <div class="c-value">{units:,} unidades</div>
        <div style="font-size:0.80rem;color:{'#065F46' if ok else '#7F1D1D'};margin-top:4px;">{extra}</div>
        <div><span class="c-badge">{badge}</span></div>
    </div>
    """


col_r1, col_arrow, col_r2 = st.columns([5, 1, 5], gap="small")

with col_r1:
    st.markdown(
        _constraint_card(
            "Sem Otimizacao (Estado Inicial)",
            units_inicial,
            f"Lucro bruto: ${profit_inicial:,.0f}" if profit_inicial is not None else "",
        ),
        unsafe_allow_html=True,
    )

with col_arrow:
    st.markdown(
        '<div style="display:flex;align-items:center;justify-content:center;'
        'height:100%;font-size:1.4rem;color:#94A3B8;padding-top:20px;">&#8594;</div>',
        unsafe_allow_html=True,
    )

with col_r2:
    st.markdown(
        _constraint_card(
            "Hill Climbing + Death Penalty",
            units_hc,
            f"Lucro otimizado: ${profit_hc:,.0f}" if profit_hc is not None else "",
        ),
        unsafe_allow_html=True,
    )

st.markdown("<br>", unsafe_allow_html=True)

# ── Resultado detalhado por loja ───────────────────────────────────────────────
if df_alloc_hc is not None:
    st.markdown('<div class="section-label">Resultado por Loja — Hill Climbing + Death Penalty</div>', unsafe_allow_html=True)

    stores_hc = df_alloc_hc[df_alloc_hc["store"] != "TOTAL"].copy()
    stores_hc["Loja"]       = stores_hc["store"].str.capitalize()
    stores_hc["Lucro ($)"]  = stores_hc["lucro"].map("${:,.0f}".format)
    stores_hc["Unidades"]   = stores_hc["unidades"].map("{:,}".format)
    stores_hc["Staff Total"]= stores_hc["total_staff"].astype(int)

    col_tbl, col_summary = st.columns([7, 3], gap="medium")

    with col_tbl:
        st.dataframe(
            stores_hc[["Loja", "Lucro ($)", "Unidades", "Staff Total"]],
            hide_index=True,
            use_container_width=True,
        )

    with col_summary:
        if units_hc is not None and profit_hc is not None:
            units_gain  = (units_inicial or 0) - units_hc
            profit_gain = profit_hc - (profit_inicial or 0)
            st.markdown(f"""
            <div class="cmp-card" style="margin-top:0;">
                <div class="cmp-label">Resumo da Otimizacao</div>
                <div class="cmp-value cmp-value-blue" style="font-size:1.2rem;">${profit_hc:,.0f}</div>
                <div class="cmp-sub" style="margin-top:8px;">Lucro semanal total otimizado</div>
                <div class="cmp-sub" style="margin-top:8px;">
                    Reducao de unidades:<br>
                    <strong style="color:#059669;">{units_gain:,} un.</strong>
                </div>
                <div class="cmp-sub" style="margin-top:6px;">
                    Restricao de 10 000 unidades<br>
                    <strong style="color:#059669;">respeitada</strong>.
                </div>
            </div>
            """, unsafe_allow_html=True)

    # ── Grafico de barras: lucro por loja ──────────────────────────────────────
    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<div class="section-label">Lucro por Loja Apos Otimizacao</div>', unsafe_allow_html=True)

    fig_hc = go.Figure(go.Bar(
        x=stores_hc["Loja"],
        y=stores_hc["lucro"],
        marker_color=[
            "#1E3A8A" if s.lower() == loja_sel.lower() else "#CBD5E1"
            for s in stores_hc["Loja"]
        ],
        text=stores_hc["lucro"].map("${:,.0f}".format),
        textposition="outside",
        textfont=dict(color="#0F172A", size=11),
    ))
    fig_hc.update_layout(
        height=280,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin=dict(t=20, b=10, l=10, r=10),
        showlegend=False,
        xaxis=dict(
            tickfont=dict(color="#0F172A", size=11),
            gridcolor="#F1F5F9",
        ),
        yaxis=dict(
            title=dict(text="Lucro ($)", font=dict(color="#0F172A", size=11)),
            tickfont=dict(color="#0F172A", size=10),
            gridcolor="#F1F5F9",
            zeroline=True,
            zerolinecolor="#E2E8F0",
            tickformat="$,.0f",
        ),
    )
    st.plotly_chart(fig_hc, use_container_width=True)

else:
    st.info(
        "Dados de alocacao O2 nao encontrados. "
        "Execute: python scripts/run_allocation_optimization.py"
    )

# ── Convergencia O2: reducao de unidades via Death Penalty ────────────────────
if df_alloc_conv is not None:
    st.markdown("---")
    st.markdown(
        '<div class="section-label">Convergencia O2 — Reducao de Unidades via Death Penalty</div>',
        unsafe_allow_html=True,
    )
    st.caption(
        "A curva de unidades mostra como o algoritmo aprende progressivamente a "
        "respeitar a restricao logistica de 10 000 unidades. A linha tracejada marca o limite."
    )

    df_conv_o2 = df_alloc_conv[df_alloc_conv["iteration"] % 30 == 0].copy()

    fig_conv2 = go.Figure()

    fig_conv2.add_trace(go.Scatter(
        x=df_conv_o2["iteration"],
        y=df_conv_o2["units"],
        mode="lines",
        name="Unidades Totais",
        line=dict(color="#1E3A8A", width=2.5),
    ))

    fig_conv2.add_hline(
        y=LIMIT,
        line_dash="dot",
        line_color="#DC2626",
        line_width=1.5,
        annotation_text="Limite: 10 000 un.",
        annotation_position="top right",
        annotation_font=dict(color="#DC2626", size=10),
    )

    fig_conv2.add_trace(go.Scatter(
        x=df_conv_o2["iteration"],
        y=df_conv_o2["units"].clip(lower=LIMIT),
        fill="tonexty",
        fillcolor="rgba(220,38,38,0.06)",
        line=dict(width=0),
        showlegend=False,
        hoverinfo="skip",
    ))

    fig_conv2.update_layout(
        height=300,
        plot_bgcolor="#FFFFFF",
        paper_bgcolor="#FFFFFF",
        margin=dict(t=20, b=40, l=10, r=40),
        legend=dict(
            orientation="h", yanchor="bottom", y=1.02,
            xanchor="right", x=1,
            font=dict(color="#0F172A", size=11),
        ),
        xaxis=dict(
            title=dict(text="Iteracao", font=dict(color="#0F172A", size=11)),
            tickfont=dict(color="#0F172A", size=10),
            gridcolor="#F1F5F9",
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(text="Unidades Totais", font=dict(color="#0F172A", size=11)),
            tickfont=dict(color="#0F172A", size=10),
            gridcolor="#F1F5F9",
            zeroline=False,
            tickformat=",d",
        ),
    )
    st.plotly_chart(fig_conv2, use_container_width=True)

# ── Footer ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dss-footer">
    DSS &nbsp;|&nbsp; Otimizacao Monobjetivo O1 + O2 &nbsp;|&nbsp;
    Hill Climbing &middot; Death Penalty &middot; Heuristica Monobjetivo
</div>
""", unsafe_allow_html=True)
