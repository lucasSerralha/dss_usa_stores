"""
O3: Fronteira de Pareto Multiobjetivo — Laboratorio de Decisao
Pagina 04 — DSS USA Stores

Storytelling de negocio + Pareto 2D dinamico com estrela dourada no
ponto selecionado pelo slider de escalarizacao w.
"""
import streamlit as st
import sys
import os
import numpy as np
import pandas as pd
import plotly.graph_objects as go

sys.dont_write_bytecode = True

# ── Page config ────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Otimizacao Multiobjetivo — DSS USA Stores",
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
    max-width: 820px; margin-bottom: 20px;
  }
  .story-block {
    padding: 24px 28px; border-radius: 10px;
    border: 1px solid #E2E8F0; background: #F8FAFC;
    margin-bottom: 28px;
  }
  .story-title {
    font-size: 0.70rem; font-weight: 700; letter-spacing: 0.09em;
    text-transform: uppercase; color: #1E3A8A; margin-bottom: 14px;
  }
  .profile-row { display: flex; gap: 16px; flex-wrap: wrap; }
  .profile-card {
    flex: 1; min-width: 200px; padding: 16px 18px;
    border-radius: 8px; border: 1px solid #E2E8F0; background: #FFFFFF;
  }
  .profile-name {
    font-size: 0.78rem; font-weight: 700; text-transform: uppercase;
    letter-spacing: 0.06em; margin-bottom: 6px;
  }
  .profile-desc { font-size: 0.82rem; color: #475569; line-height: 1.6; }

  .formula-block {
    padding: 16px 20px; border-radius: 8px;
    border-left: 4px solid #1E3A8A; background: #F0F4FF;
    font-size: 0.86rem; color: #1E293B; line-height: 1.7;
    margin-bottom: 20px; font-family: 'Courier New', monospace;
  }
  .selected-plan {
    padding: 20px 24px; border-radius: 10px;
    border: 2px solid #1E3A8A; background: #EFF6FF;
  }
  .selected-plan-title {
    font-size: 0.70rem; font-weight: 700; letter-spacing: 0.09em;
    text-transform: uppercase; color: #1E3A8A; margin-bottom: 12px;
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
_opt_dir   = os.path.join(_root_dir, "results", "03_Optimization_Report")

# ── Loaders ────────────────────────────────────────────────────────────────
@st.cache_data
def carregar_pareto(opt_dir: str, algo: str) -> pd.DataFrame | None:
    fname = "joint_pareto_unsga3.csv" if algo == "U-NSGA-III" else "joint_pareto_moead.csv"
    path  = os.path.join(opt_dir, "joint_o3", fname)
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


@st.cache_data
def carregar_resumo_o3(opt_dir: str) -> pd.DataFrame | None:
    path = os.path.join(opt_dir, "joint_o3", "joint_o3_summary.csv")
    return pd.read_csv(path) if os.path.exists(path) else None


def calcular_avg_discount(df: pd.DataFrame) -> np.ndarray:
    """Media das colunas *_pr_* de cada solucao (escala 0-1 -> multiplica por 100)."""
    pr_cols = [c for c in df.columns if "_pr_" in c]
    if not pr_cols:
        return np.zeros(len(df))
    return df[pr_cols].mean(axis=1).values * 100


def selecionar_por_w(df: pd.DataFrame, w: float) -> int:
    """Indice posicional da solucao com melhor score escalar.

    Restringe a busca a solucoes lucrativas (lucro_total > 0) antes
    de aplicar a escalarizacao, evitando que perfis conservadores
    (w baixo) selecionem planos nao-viaveis apenas por terem staff minimo.
    Se nenhuma solucao for lucrativa, usa todas como fallback.
    """
    # 1. Filtrar apenas solucoes economicamente viaveis
    df_viable = df[df["lucro_total"] > 0]
    if df_viable.empty:
        df_viable = df  # fallback: sem solucoes lucrativas, usa tudo

    lucro = df_viable["lucro_total"].values.astype(float)
    staff = df_viable["staff_total"].values.astype(float)
    lucro_n = (lucro - lucro.min()) / (lucro.max() - lucro.min() + 1e-9)
    staff_n = (staff - staff.min()) / (staff.max() - staff.min() + 1e-9)
    scores  = w * lucro_n + (1 - w) * (1 - staff_n)

    # 2. Retornar o indice posicional no df original (index == posicao para CSV padrao)
    best_in_viable = int(np.argmax(scores))
    return int(df_viable.index[best_in_viable])


# ── Barra Lateral ──────────────────────────────────────────────────────────
st.sidebar.markdown(
    '<div style="font-size:0.88rem;font-weight:700;color:#E2E8F0;'
    'letter-spacing:0.04em;margin-bottom:16px;">DSS USA STORES</div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div style="font-size:0.70rem;color:#64748B;letter-spacing:0.08em;'
    'text-transform:uppercase;font-weight:700;margin-bottom:12px;">Laboratorio O3</div>',
    unsafe_allow_html=True,
)

algoritmo = st.sidebar.radio(
    "Algoritmo",
    ["U-NSGA-III", "MOEA/D"],
    index=0,
    help="U-NSGA-III: solucoes mais diversificadas; MOEA/D: convergencia mais rapida.",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div style="font-size:0.70rem;color:#64748B;letter-spacing:0.08em;'
    'text-transform:uppercase;font-weight:700;margin-bottom:8px;">Parametros de Execucao</div>',
    unsafe_allow_html=True,
)

rigor = st.sidebar.select_slider(
    "Rigor do Algoritmo",
    options=["Rapido (pop=20)", "Equilibrado (pop=50)", "Profundo (pop=100)"],
    value="Equilibrado (pop=50)",
    help="Controla o tamanho da populacao na execucao (apenas informativo — usa resultados pre-computados).",
)

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div style="font-size:0.70rem;color:#64748B;letter-spacing:0.08em;'
    'text-transform:uppercase;font-weight:700;margin-bottom:8px;">Escalarizacao</div>',
    unsafe_allow_html=True,
)

w = st.sidebar.slider(
    "Ponderacao w (Lucro vs. Staff)",
    min_value=0.0,
    max_value=1.0,
    value=0.5,
    step=0.05,
    help="w=0: minimizar staff; w=1: maximizar lucro; w=0.5: Knee Point equilibrado.",
)

perfil_label = "Agressivo (Lucro Maximo)" if w >= 0.75 else (
    "Conservador (Staff Minimo)" if w <= 0.25 else "Equilibrado (Knee Point)"
)

st.sidebar.markdown(f"""
<div style="margin-top:10px;padding:10px 14px;border-radius:6px;
     border:1px solid #1E293B;background:#1E293B;">
    <div style="font-size:0.70rem;font-weight:700;letter-spacing:0.08em;
         text-transform:uppercase;color:#94A3B8;margin-bottom:4px;">Perfil Ativo</div>
    <div style="font-size:0.88rem;font-weight:700;color:#E2E8F0;">{perfil_label}</div>
    <div style="font-size:0.75rem;color:#64748B;margin-top:4px;">w = {w:.2f}</div>
</div>
""", unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.caption(f"Algoritmo: {algoritmo}")
st.sidebar.caption(f"Rigor: {rigor}")
st.sidebar.caption("84 variaveis de decisao (4 lojas x 21)")

# ── Carregar dados ─────────────────────────────────────────────────────────
df_pareto  = carregar_pareto(_opt_dir, algoritmo)
df_resumo  = carregar_resumo_o3(_opt_dir)

# ── Cabecalho ──────────────────────────────────────────────────────────────
st.markdown("""
<div style="padding: 36px 0 20px 0; border-bottom: 2px solid #1E3A8A; margin-bottom: 32px;">
    <div style="font-size: 1.85rem; font-weight: 800; color: #0F172A;
         letter-spacing: -0.03em; line-height: 1.2; margin-bottom: 10px;">
        Otimizacao Multiobjetivo (O3) — Fronteira de Pareto
    </div>
    <div style="font-size: 0.88rem; color: #64748B; max-width: 780px; line-height: 1.65;">
        Laboratorio interativo de decisao estrategica. A fronteira de Pareto expoe o
        conjunto de solucoes nao dominadas que maximizam o lucro e minimizam o custo de
        staff simultaneamente. O gestor escolhe o perfil via ponderacao escalar.
    </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# STORYTELLING DE NEGOCIO — 3 Perfis
# ══════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="story-block">
    <div class="story-title">Semantica dos Perfis Estrategicos</div>
    <div class="profile-row">
        <div class="profile-card">
            <div class="profile-name" style="color:#DC2626;">Agressivo (w &ge; 0.75)</div>
            <div class="profile-desc">
                Prioriza a maximizacao do lucro absoluto, ignorando o custo fixo
                de staffing. Aceita escalas mais pesadas se isso gerar maior
                receita. Recomendado em periodos de pico ou expansao.
            </div>
        </div>
        <div class="profile-card">
            <div class="profile-name" style="color:#1E3A8A;">Conservador (w &le; 0.25)</div>
            <div class="profile-desc">
                Minimiza o staff total para eliminar o risco operacional fixo.
                Lucro pode ser subotimo, mas a estrutura de custos e previsivel.
                Recomendado em periodos de baixa procura ou incerteza.
            </div>
        </div>
        <div class="profile-card">
            <div class="profile-name" style="color:#059669;">Equilibrado — Knee Point (w &asymp; 0.5)</div>
            <div class="profile-desc">
                Usa escalarizacao para capturar o lucro otimo com custo controlado.
                O Knee Point e o ponto da fronteira de Pareto com maior retorno
                marginal: ganhos de lucro adicionais custam disproportionalmente
                mais staff. Recomendado como plano base operacional.
            </div>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# Formula de escalarizacao
st.markdown("""
<div class="formula-block">
    Score(s) = w &times; Lucro_norm(s)  +  (1 &minus; w) &times; (1 &minus; Staff_norm(s))<br>
    Lucro_norm = (Lucro &minus; min) / (max &minus; min) &nbsp;&nbsp;
    Staff_norm = (Staff &minus; min) / (max &minus; min)<br>
    Solucao selecionada = argmax Score(s)
</div>
""", unsafe_allow_html=True)

# ── Validacao ──────────────────────────────────────────────────────────────
if df_pareto is None or df_pareto.empty:
    st.error(
        f"Dados da fronteira de Pareto nao encontrados para {algoritmo}. "
        "Execute: `python scripts/run_joint_optimization_o3.py`"
    )
    st.stop()

# ── Pre-processar ──────────────────────────────────────────────────────────
df_pareto = df_pareto.copy()
df_pareto["avg_discount_pct"] = calcular_avg_discount(df_pareto)
idx_sel = selecionar_por_w(df_pareto, w)

lucros  = df_pareto["lucro_total"].values.astype(float)
staffs  = df_pareto["staff_total"].values.astype(float)
descs   = df_pareto["avg_discount_pct"].values

# Detectar convergencia prematura (MOEA/D pode colapsar para poucos pontos)
n_unique_points = df_pareto.groupby(["staff_total", "lucro_total"]).ngroups
convergencia_colapsada = n_unique_points < max(5, len(df_pareto) * 0.1)

# Deduplificar para visualizacao (sem duplicados sobrepostos)
df_plot = (
    df_pareto.groupby(["staff_total", "lucro_total"], as_index=False)
    .agg(avg_discount_pct=("avg_discount_pct", "mean"), count=("lucro_total", "size"))
)
df_plot_sorted = df_plot.sort_values("staff_total")

plot_staffs = df_plot_sorted["staff_total"].values.astype(float)
plot_lucros  = df_plot_sorted["lucro_total"].values.astype(float)
plot_descs   = df_plot_sorted["avg_discount_pct"].values
plot_counts  = df_plot_sorted["count"].values

# ── Metricas de contexto ───────────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4, gap="medium")

with m1:
    st.metric("Solucoes na Fronteira", len(df_pareto),
              help=f"Solucoes nao dominadas — {algoritmo} ({n_unique_points} pontos distintos)")
with m2:
    st.metric("Lucro Maximo", f"${lucros.max():,.0f}",
              help="Solucao com maior lucro total na fronteira")
with m3:
    n_lucrative = int((lucros > 0).sum())
    st.metric("Solucoes Lucrativas", f"{n_lucrative} / {len(df_pareto)}",
              help="Solucoes com lucro_total > 0. A escalarizacao restringe-se a estas.")
with m4:
    st.metric("Lucro Selecionado (w)", f"${lucros[idx_sel]:,.0f}",
              help=f"Ponto da fronteira com w={w:.2f} — restrito a solucoes lucrativas")

# Alerta de convergencia prematura
if convergencia_colapsada:
    st.warning(
        f"**{algoritmo} convergiu para {n_unique_points} pontos distintos** "
        f"(de {len(df_pareto)} solucoes). O algoritmo colocou {len(df_pareto) - n_unique_points} "
        f"solucoes no mesmo ponto — sinal de convergencia prematura na decomposicao de pesos. "
        f"A fronteira e exibida com os {n_unique_points} pontos unicos.",
        icon="⚠️",
    )

st.markdown("<br>", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════
# FRONTEIRA DE PARETO 2D DINAMICA
# ══════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-title">Fronteira de Pareto 2D — Trade-off Lucro vs. Staff</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="section-desc">'
    'Cada ponto e uma solucao nao dominada (pontos sobrepostos deduplificados). '
    'A <b>linha azul</b> conecta os pontos ordenados por staff — e a fronteira. '
    'A <b>linha tracejada vermelha</b> marca o break-even (lucro = $0). '
    'A <b>estrela dourada</b> e o plano recomendado para o valor de w selecionado.'
    '</div>',
    unsafe_allow_html=True,
)

fig_pareto = go.Figure()

# Linha da fronteira (conecta pontos ordenados por staff)
fig_pareto.add_trace(go.Scatter(
    x=plot_staffs,
    y=plot_lucros,
    mode="lines",
    name="Fronteira Pareto",
    line=dict(color="#93C5FD", width=2, dash="solid"),
    hoverinfo="skip",
))

# Pontos da fronteira (tamanho proporcional ao nr de solucoes sobrepostas)
marker_sizes = np.clip(8 + np.log1p(plot_counts) * 4, 8, 24).tolist()
fig_pareto.add_trace(go.Scatter(
    x=plot_staffs,
    y=plot_lucros,
    mode="markers",
    name="Solucoes Pareto",
    marker=dict(
        color=plot_descs,
        colorscale=[
            [0.0, "#EFF6FF"],
            [0.5, "#3B82F6"],
            [1.0, "#1E3A8A"],
        ],
        size=marker_sizes,
        colorbar=dict(
            title=dict(text="Desconto Medio (%)", font=dict(color="#0F172A", size=11)),
            tickfont=dict(color="#0F172A", size=10),
            x=1.02,
        ),
        line=dict(color="#FFFFFF", width=1),
        showscale=True,
    ),
    customdata=plot_counts,
    hovertemplate=(
        "Staff Total: %{x}<br>"
        "Lucro Total: $%{y:,.0f}<br>"
        "Desconto Medio: %{marker.color:.1f}%<br>"
        "Solucoes neste ponto: %{customdata}"
        "<extra></extra>"
    ),
))

# Linha de break-even (lucro = 0)
x_range = [float(plot_staffs.min()) - 1, float(plot_staffs.max()) + 1]
fig_pareto.add_trace(go.Scatter(
    x=x_range,
    y=[0, 0],
    mode="lines",
    name="Break-even ($0)",
    line=dict(color="#DC2626", width=1.5, dash="dash"),
    hoverinfo="skip",
))

# Estrela dourada no ponto selecionado
fig_pareto.add_trace(go.Scatter(
    x=[staffs[idx_sel]],
    y=[lucros[idx_sel]],
    mode="markers+text",
    name=f"Selecionado (w={w:.2f})",
    marker=dict(
        symbol="star",
        size=22,
        color="#F59E0B",
        line=dict(color="#92400E", width=1.5),
    ),
    text=[f"  {perfil_label}"],
    textposition="middle right",
    textfont=dict(color="#92400E", size=11, family="Inter"),
))

fig_pareto.update_layout(
    height=500,
    plot_bgcolor="#FFFFFF",
    paper_bgcolor="#FFFFFF",
    margin=dict(t=20, b=50, l=10, r=120),
    legend=dict(
        orientation="h", yanchor="bottom", y=1.02,
        xanchor="right", x=1,
        font=dict(color="#0F172A", size=11),
    ),
    xaxis=dict(
        title=dict(text="Staff Total a Minimizar", font=dict(color="#0F172A", size=12)),
        tickfont=dict(color="#0F172A", size=10),
        gridcolor="#F1F5F9",
        zeroline=False,
    ),
    yaxis=dict(
        title=dict(text="Lucro Total a Maximizar ($)", font=dict(color="#0F172A", size=12)),
        tickfont=dict(color="#0F172A", size=10),
        gridcolor="#F1F5F9",
        zeroline=False,
        tickformat="$,.0f",
    ),
)
st.plotly_chart(fig_pareto, use_container_width=True)

# ── Destaque da Decisao ────────────────────────────────────────────────────
st.markdown("---")
st.markdown('<div class="section-label">Plano Recomendado — Ponto Selecionado</div>', unsafe_allow_html=True)

col_plan, col_context = st.columns([6, 4], gap="medium")

with col_plan:
    sol = df_pareto.iloc[idx_sel]
    lucro_sel = float(sol["lucro_total"])
    staff_sel = int(sol["staff_total"])
    desc_sel  = float(sol["avg_discount_pct"])

    is_profitable = lucro_sel > 0
    plan_border   = "#059669" if is_profitable else "#DC2626"
    plan_bg       = "#F0FDF4" if is_profitable else "#FFF1F2"
    badge_color   = "#059669" if is_profitable else "#DC2626"
    badge_text    = "✓ Lucrativo" if is_profitable else "⚠ Deficit"
    lucro_color   = "#059669" if is_profitable else "#DC2626"

    st.markdown(f"""
    <div class="selected-plan" style="border-color:{plan_border};background:{plan_bg};">
        <div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:12px;">
            <div class="selected-plan-title" style="margin-bottom:0;">
                Plano Otimo para w = {w:.2f} — {perfil_label}
            </div>
            <span style="padding:3px 10px;border-radius:12px;font-size:0.72rem;
                  font-weight:700;background:{badge_color}20;color:{badge_color};">
                {badge_text}
            </span>
        </div>
        <table style="width:100%;font-size:0.88rem;border-collapse:collapse;">
            <tr>
                <td style="color:#64748B;padding:5px 0;">Lucro Total Semanal</td>
                <td style="font-weight:700;color:{lucro_color};text-align:right;">
                    ${lucro_sel:,.0f}
                </td>
            </tr>
            <tr>
                <td style="color:#64748B;padding:5px 0;">Staff Total</td>
                <td style="font-weight:700;color:#0F172A;text-align:right;">
                    {staff_sel} colaboradores
                </td>
            </tr>
            <tr>
                <td style="color:#64748B;padding:5px 0;">Desconto Medio</td>
                <td style="font-weight:700;color:#0F172A;text-align:right;">
                    {desc_sel:.1f}%
                </td>
            </tr>
            <tr>
                <td style="color:#64748B;padding:5px 0;">Indice na Fronteira</td>
                <td style="font-weight:700;color:#1E3A8A;text-align:right;">
                    #{idx_sel + 1} / {len(df_pareto)}
                </td>
            </tr>
        </table>
    </div>
    """, unsafe_allow_html=True)

with col_context:
    # Extremos da fronteira — restritos a solucoes lucrativas para comparacoes validas
    df_viable_ctx = df_pareto[df_pareto["lucro_total"] > 0]
    if df_viable_ctx.empty:
        df_viable_ctx = df_pareto  # fallback

    lucros_v  = df_viable_ctx["lucro_total"].values.astype(float)
    staffs_v  = df_viable_ctx["staff_total"].values.astype(float)

    idx_max_lucro = int(np.argmax(lucros_v))
    idx_min_staff = int(np.argmin(staffs_v))

    lucro_agressivo = lucros_v[idx_max_lucro]
    staff_agressivo = staffs_v[idx_max_lucro]
    lucro_conserv   = lucros_v[idx_min_staff]
    staff_conserv   = staffs_v[idx_min_staff]

    custo_extra_staff  = staff_sel - staff_conserv
    ganho_extra_lucro  = lucro_sel - lucro_conserv

    st.markdown(f"""
    <div style="padding:18px 20px;border-radius:10px;border:1px solid #E2E8F0;background:#FAFAFA;">
        <div style="font-size:0.70rem;font-weight:700;letter-spacing:0.08em;
             text-transform:uppercase;color:#94A3B8;margin-bottom:12px;">
             Analise de Trade-off</div>
        <div style="font-size:0.83rem;color:#475569;line-height:1.8;">
            Face ao perfil <b>Conservador</b> (staff={int(staff_conserv)}):<br>
            &nbsp;&bull; Staff adicional: <b style="color:#1E3A8A;">+{custo_extra_staff}</b><br>
            &nbsp;&bull; Lucro adicional: <b style="color:#059669;">+${ganho_extra_lucro:,.0f}</b><br><br>
            Face ao perfil <b>Agressivo</b> (${lucro_agressivo:,.0f}):<br>
            &nbsp;&bull; Lucro sacrificado: <b style="color:#DC2626;">
            -${lucro_agressivo - lucro_sel:,.0f}</b><br>
            &nbsp;&bull; Staff poupado: <b style="color:#059669;">
            -{int(staff_agressivo - staff_sel)}</b>
        </div>
    </div>
    """, unsafe_allow_html=True)

# ── Resumo de algoritmos ───────────────────────────────────────────────────
if df_resumo is not None and not df_resumo.empty:
    st.markdown("---")
    st.markdown('<div class="section-label">Comparativo de Algoritmos</div>', unsafe_allow_html=True)

    df_res_disp = df_resumo.copy()
    col_map = {
        "algorithm":           "Algoritmo",
        "n_pareto_solutions":  "Solucoes Pareto",
        "o3_max_profit":       "Lucro Maximo ($)",
        "o3_max_profit_staff": "Staff no Lucro Max",
        "o3_min_staff":        "Staff Minimo",
        "o3_min_staff_profit": "Lucro no Staff Min ($)",
    }
    df_res_disp = df_res_disp.rename(columns={k: v for k, v in col_map.items() if k in df_res_disp.columns})
    st.dataframe(df_res_disp, hide_index=True, use_container_width=True)

# ── Rodape ─────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dss-footer">
    DSS &nbsp;|&nbsp; Otimizacao Multiobjetivo O3 &nbsp;|&nbsp;
    U-NSGA-III &middot; MOEA/D &middot; Fronteira de Pareto &middot; Escalarizacao
</div>
""", unsafe_allow_html=True)
