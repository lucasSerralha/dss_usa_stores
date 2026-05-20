"""
DSS USA Stores — Homepage
Sistema de Apoio à Decisão para previsão de procura e otimização de recursos.
"""
import streamlit as st
import sys
import os
import pandas as pd
from datetime import date

sys.dont_write_bytecode = True

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DSS USA Stores",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Path resolution ────────────────────────────────────────────────────────────
_page_dir = os.path.dirname(os.path.abspath(__file__))
_root_dir = os.path.abspath(os.path.join(_page_dir, ".."))
_opt_dir  = os.path.join(_root_dir, "results_v2", "03_Optimization")

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800;900&display=swap');

  html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }
  h1, h2, h3 { letter-spacing: -0.02em; }
  #MainMenu, footer { visibility: hidden; }
  [data-testid="stToolbar"] { visibility: hidden; }
  [data-testid="collapsedControl"] { visibility: visible !important; }

  /* Sidebar */
  [data-testid="stSidebar"] { background: #0F172A; }
  [data-testid="stSidebar"] * { color: #CBD5E1 !important; }
  [data-testid="stSidebar"] hr { border-color: #1E293B !important; }

  /* KPI cards */
  .kpi-card {
    padding: 20px 22px 18px;
    border-radius: 10px;
    border: 1px solid #E2E8F0;
    background: #FFFFFF;
  }
  .kpi-label {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.09em;
    text-transform: uppercase; color: #94A3B8; margin-bottom: 8px;
  }
  .kpi-value {
    font-size: 1.85rem; font-weight: 900; color: #0F172A;
    line-height: 1.05; margin-bottom: 5px;
  }
  .kpi-value-blue  { color: #1E3A8A; }
  .kpi-value-green { color: #059669; }
  .kpi-value-orange{ color: #D97706; }
  .kpi-sub {
    font-size: 0.76rem; color: #64748B; line-height: 1.5;
  }

  /* Store cards */
  .store-card {
    padding: 22px 20px 18px;
    border-radius: 10px;
    border: 1px solid #E2E8F0;
    background: #FFFFFF;
  }
  .store-name {
    font-size: 1.0rem; font-weight: 800; color: #0F172A;
    margin-bottom: 14px; letter-spacing: -0.01em;
  }
  .store-metric {
    display: flex; justify-content: space-between; align-items: center;
    font-size: 0.79rem; color: #64748B;
    border-top: 1px solid #F1F5F9; padding: 6px 0;
  }
  .store-metric-value { font-weight: 700; color: #1E3A8A; }
  .store-profit {
    font-size: 1.35rem; font-weight: 800; color: #059669;
    margin-bottom: 14px;
  }

  /* Module nav cards */
  .mod-card {
    padding: 28px 24px 24px;
    border-radius: 10px;
    border: 1px solid #E2E8F0;
    background: #FFFFFF;
    height: 100%;
  }
  .mod-num {
    display: inline-flex; align-items: center; justify-content: center;
    width: 32px; height: 32px; border-radius: 7px;
    background: #1E3A8A; color: #FFFFFF;
    font-size: 0.80rem; font-weight: 800; margin-bottom: 14px;
  }
  .mod-title {
    font-size: 1.05rem; font-weight: 800; color: #0F172A;
    margin-bottom: 10px; letter-spacing: -0.01em;
  }
  .mod-desc {
    font-size: 0.83rem; color: #475569; line-height: 1.75;
    margin-bottom: 16px;
  }
  .mod-algo {
    font-size: 0.72rem; font-weight: 700; letter-spacing: 0.05em;
    text-transform: uppercase; color: #94A3B8; margin-bottom: 6px;
  }
  .tag {
    display: inline-block; margin: 2px 3px 2px 0;
    padding: 2px 8px; border-radius: 4px;
    font-size: 0.68rem; font-weight: 700;
    letter-spacing: 0.03em;
  }
  .tag-blue  { background: rgba(30,58,138,0.09); color: #1E3A8A; }
  .tag-slate { background: rgba(51,65,85,0.09);  color: #334155; }
  .tag-green { background: rgba(5,150,105,0.09); color: #059669; }
  .tag-orange{ background: rgba(217,119,6,0.09); color: #D97706; }
  .tag-purple{ background: rgba(124,58,237,0.09);color: #7C3AED; }

  /* Pipeline strip */
  .pipe-strip {
    margin-top: 8px; padding: 18px 24px;
    border-radius: 8px; border-left: 4px solid #1E3A8A;
    background: #F8FAFC; font-size: 0.84rem;
    color: #1E293B; line-height: 1.9;
  }
  .pipe-label {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.09em;
    text-transform: uppercase; color: #1E3A8A; margin-bottom: 8px;
  }

  /* Dividers */
  .section-label {
    font-size: 0.68rem; font-weight: 700; letter-spacing: 0.09em;
    text-transform: uppercase; color: #94A3B8; margin-bottom: 14px;
  }

  /* Footer */
  .dss-footer {
    font-size: 0.73rem; color: #94A3B8; text-align: right;
    margin-top: 48px; padding-top: 14px; border-top: 1px solid #E2E8F0;
  }
</style>
""", unsafe_allow_html=True)

TODAY = date.today().strftime("%d %b %Y")

# ── Load data (cached) ─────────────────────────────────────────────────────────
@st.cache_data
def _load_homepage_data():
    data = {}

    # O1 — Simulated Annealing (best single-store algorithm)
    p = os.path.join(_opt_dir, "individual", "simulated_annealing", "simulated_annealing_summary.csv")
    data["sa"] = pd.read_csv(p) if os.path.exists(p) else None

    # O2 — Penalty Function (better profit than Death Penalty)
    p = os.path.join(_opt_dir, "allocation", "allocation_summary.csv")
    data["o2"] = pd.read_csv(p) if os.path.exists(p) else None

    # O3 — Joint Pareto summary
    p = os.path.join(_opt_dir, "joint_o3", "joint_o3_summary.csv")
    data["o3"] = pd.read_csv(p) if os.path.exists(p) else None

    # Forecasting — master report
    p = os.path.join(_root_dir, "results_v2", "00_Master_Summary", "fidelity_experimentation_report.csv")
    data["forecast"] = pd.read_csv(p) if os.path.exists(p) else None

    return data

_d = _load_homepage_data()

# Derived KPIs
_sa_total   = float(_d["sa"]["lucro_maximo"].sum()) if _d["sa"] is not None else None
_o2_profit  = None
_o2_units   = None
if _d["o2"] is not None:
    tot = _d["o2"][_d["o2"]["store"] == "TOTAL"]
    if len(tot):
        _o2_profit = float(tot.iloc[0]["lucro"])
        _o2_units  = int(tot.iloc[0]["unidades"])

_o3_n_sol   = None
_o3_best    = None
if _d["o3"] is not None:
    u3 = _d["o3"][_d["o3"]["algorithm"] == "U-NSGA-III"]
    if len(u3):
        _o3_n_sol = int(u3.iloc[0]["n_pareto_solutions"])
        _o3_best  = float(u3.iloc[0]["o3_max_profit"])

_best_mape  = None
if _d["forecast"] is not None:
    sc = _d["forecast"][_d["forecast"]["Experiment"] == "C_Context_Expert"]
    if len(sc):
        _best_mape = float(sc.groupby("Store")["MAPE"].min().mean())


# ── Sidebar ────────────────────────────────────────────────────────────────────
st.sidebar.markdown(
    '<div style="font-size:0.88rem;font-weight:800;color:#E2E8F0;'
    'letter-spacing:0.04em;margin-bottom:4px;">DSS USA STORES</div>'
    '<div style="font-size:0.70rem;color:#64748B;margin-bottom:16px;">'
    'Decision Support System</div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div style="font-size:0.68rem;color:#64748B;letter-spacing:0.08em;'
    'text-transform:uppercase;font-weight:700;margin-bottom:10px;">Estado do Sistema</div>',
    unsafe_allow_html=True,
)
st.sidebar.caption(f"Relatório: {TODAY}")
st.sidebar.caption("Lojas: Baltimore · Lancaster · Philadelphia · Richmond")
st.sidebar.caption("Horizonte de previsão: 7 dias (semana seguinte)")
st.sidebar.caption("Restrição logística: 10 000 unidades/semana")

st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div style="font-size:0.68rem;color:#64748B;letter-spacing:0.08em;'
    'text-transform:uppercase;font-weight:700;margin-bottom:10px;">Algoritmos Utilizados</div>',
    unsafe_allow_html=True,
)
st.sidebar.caption("Previsão: Ensemble, RF, Prophet, SARIMAX, HW")
st.sidebar.caption("O1: Monte Carlo, Hill Climbing, SA, NSGA-II")
st.sidebar.caption("O2: HC + Penalty / Death Penalty")
st.sidebar.caption("O3: MOEA/D · U-NSGA-III (84 variáveis)")

# ══════════════════════════════════════════════════════════════════════════════
# HEADER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div style="padding: 40px 0 22px 0; border-bottom: 2px solid #1E3A8A; margin-bottom: 32px;">
  <div style="font-size: 0.68rem; font-weight: 700; letter-spacing: 0.12em;
       text-transform: uppercase; color: #1E3A8A; margin-bottom: 10px;">
    Sistema de Apoio à Decisão
  </div>
  <div style="font-size: 2.1rem; font-weight: 900; color: #0F172A;
       letter-spacing: -0.03em; line-height: 1.15; margin-bottom: 12px;">
    Previsão de Procura e Otimização<br>de Recursos — USA Stores
  </div>
  <div style="font-size: 0.88rem; color: #64748B; max-width: 700px; line-height: 1.7;">
    Pipeline integrado de previsão de clientes, auditoria científica de modelos preditivos
    e otimização multiobjetivo de staffing e descontos para quatro lojas retalhistas nos EUA.
  </div>
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# KPI STRIP
# ══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-label">Indicadores-Chave do Sistema</div>', unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4, gap="medium")

with k1:
    val  = f"${_sa_total:,.0f}" if _sa_total else "—"
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">Lucro O1 — Melhor Algoritmo (SA)</div>
      <div class="kpi-value kpi-value-green">{val}</div>
      <div class="kpi-sub">Semana prevista · Simulated Annealing<br>
        Soma das 4 lojas otimizadas individualmente</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    val  = f"${_o2_profit:,.0f}" if _o2_profit is not None else "—"
    un   = f"{_o2_units:,} un." if _o2_units else ""
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">Lucro O2 — Rede c/ Restrição 10k</div>
      <div class="kpi-value kpi-value-blue">{val}</div>
      <div class="kpi-sub">HC + Penalty Function · {un}<br>
        Restrição logística de 10 000 unidades/semana</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    val  = str(_o3_n_sol) if _o3_n_sol else "—"
    best = f"${_o3_best:,.0f}" if _o3_best else ""
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">Fronteira Pareto O3 (U-NSGA-III)</div>
      <div class="kpi-value kpi-value-orange">{val} soluções</div>
      <div class="kpi-sub">Max lucro: {best} · 84 variáveis de decisão<br>
        Trade-off lucro total vs. staff total</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    val  = f"{100 - _best_mape:.1f}%" if _best_mape else "—"
    st.markdown(f"""
    <div class="kpi-card">
      <div class="kpi-label">Precisão de Previsão (1 − MAPE)</div>
      <div class="kpi-value">{val}</div>
      <div class="kpi-sub">Média das 4 lojas · Cenário C (melhor modelo)<br>
        7 modelos avaliados · 3 cenários de features</div>
    </div>
    """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# STORE OVERVIEW GRID
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
st.markdown('<div class="section-label">Lojas — Melhor Resultado O1 (Simulated Annealing)</div>', unsafe_allow_html=True)

STORE_INFO = {
    "baltimore":    {"flag": "🇺🇸", "state": "Maryland"},
    "lancaster":    {"flag": "🇺🇸", "state": "Pennsylvania"},
    "philadelphia": {"flag": "🇺🇸", "state": "Pennsylvania"},
    "richmond":     {"flag": "🇺🇸", "state": "Virginia"},
}

cols_stores = st.columns(4, gap="medium")

if _d["sa"] is not None:
    for col, (_, row) in zip(cols_stores, _d["sa"].iterrows()):
        store  = row["store"]
        info   = STORE_INFO.get(store, {})
        lucro  = float(row["lucro_maximo"])
        staff  = int(row["total_staff"])
        disc   = float(row["avg_discount"])

        # Best forecast model for this store
        best_model = "—"
        if _d["forecast"] is not None:
            fc = _d["forecast"][
                (_d["forecast"]["Store"].str.lower() == store) &
                (_d["forecast"]["Experiment"] == "C_Context_Expert")
            ]
            if len(fc):
                best_model = fc.sort_values("RMSE").iloc[0]["Model"]
                # Shorten long names
                best_model = best_model.replace("Ensemble (Top-3 Experts)", "Ensemble Top-3")

        with col:
            st.markdown(f"""
            <div class="store-card">
              <div class="store-name">{store.capitalize()}</div>
              <div class="store-profit">${lucro:,.0f}<span style="font-size:0.75rem;
                font-weight:600;color:#64748B;">/semana</span></div>
              <div class="store-metric">
                <span>Staff total</span>
                <span class="store-metric-value">{staff}</span>
              </div>
              <div class="store-metric">
                <span>Desconto médio</span>
                <span class="store-metric-value">{disc:.2f}%</span>
              </div>
              <div class="store-metric">
                <span>Melhor modelo</span>
                <span class="store-metric-value" style="font-size:0.71rem;">{best_model}</span>
              </div>
            </div>
            """, unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# MODULE NAVIGATION CARDS
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("---")
st.markdown('<div class="section-label">Módulos do Sistema</div>', unsafe_allow_html=True)

m1, m2, m3 = st.columns(3, gap="medium")

with m1:
    st.markdown("""
    <div class="mod-card">
      <div class="mod-num">1</div>
      <div class="mod-title">Auditoria Científica de Modelos</div>
      <div class="mod-desc">
        Avaliação empírica de 7 modelos preditivos por loja e por cenário de features,
        via <strong>TimeSeriesSplit cross-validation</strong>.
        Leaderboard com ranking por RMSE, MAPE e NMAE. Podium visual dos 3 melhores modelos
        por combinação loja × cenário.
        <br><br>
        O modelo vencedor alimenta diretamente as previsões de clientes usadas na otimização.
      </div>
      <div class="mod-algo">Modelos avaliados</div>
      <span class="tag tag-blue">Seasonal Naive</span>
      <span class="tag tag-blue">Holt-Winters</span>
      <span class="tag tag-slate">Linear Regression</span>
      <span class="tag tag-slate">Random Forest</span>
      <span class="tag tag-green">Prophet</span>
      <span class="tag tag-green">SARIMAX</span>
      <span class="tag tag-orange">Ensemble Top-3</span>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/01_Auditoria_Predicoes.py", label="Abrir Auditoria de Modelos →", use_container_width=True)

with m2:
    st.markdown("""
    <div class="mod-card">
      <div class="mod-num">2</div>
      <div class="mod-title">Otimização Monobjetivo — O1 e O2</div>
      <div class="mod-desc">
        <strong>O1</strong> — Maximização de lucro semanal por loja de forma independente
        (21 variáveis: desconto diário, staff experto e júnior × 7 dias).
        Quatro algoritmos comparados em paralelo.<br><br>
        <strong>O2</strong> — Otimização conjunta das 4 lojas com restrição logística de
        <strong>≤ 10 000 unidades/semana</strong>. Duas estratégias de gestão de restrições
        comparadas.
      </div>
      <div class="mod-algo">Algoritmos O1</div>
      <span class="tag tag-purple">Monte Carlo</span>
      <span class="tag tag-blue">Hill Climbing</span>
      <span class="tag tag-orange">Simulated Annealing</span>
      <span class="tag tag-green">NSGA-II</span>
      <div class="mod-algo" style="margin-top:8px;">Algoritmos O2</div>
      <span class="tag tag-slate">HC + Penalty Function</span>
      <span class="tag tag-slate">HC + Death Penalty</span>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/02_Otimizacao_Monobjetivo.py", label="Abrir Otimização O1 + O2 →", use_container_width=True)

with m3:
    st.markdown("""
    <div class="mod-card">
      <div class="mod-num">3</div>
      <div class="mod-title">Otimização Multiobjetivo — O3</div>
      <div class="mod-desc">
        <strong>Fronteira de Pareto interativa</strong>: minimização simultânea de
        (−Lucro Total, Staff Total) para as 4 lojas em conjunto, sujeito a ≤ 10 000
        unidades/semana. <strong>84 variáveis de decisão</strong>.<br><br>
        Slider <strong>w</strong> na sidebar controla a escalarização ponderada —
        o ponto ótimo da fronteira atualiza instantaneamente. Três perfis de decisão
        derivados: Agressivo, Equilibrado, Conservador.
      </div>
      <div class="mod-algo">Algoritmos Multiobjetivo</div>
      <span class="tag tag-green">MOEA/D</span>
      <span class="tag tag-blue">U-NSGA-III</span>
      <div class="mod-algo" style="margin-top:8px;">Funcionalidades</div>
      <span class="tag tag-slate">Pareto 2D interativo</span>
      <span class="tag tag-slate">Perfis de decisão</span>
      <span class="tag tag-orange">Simulação em tempo real</span>
    </div>
    """, unsafe_allow_html=True)
    st.page_link("pages/03_Otimizacao_Multiobjetivo.py", label="Abrir Laboratório O3 →", use_container_width=True)

# ══════════════════════════════════════════════════════════════════════════════
# PIPELINE ARCHITECTURE
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("<br>", unsafe_allow_html=True)
st.markdown("""
<div class="pipe-strip">
  <div class="pipe-label">Pipeline End-to-End</div>
  <strong>Dados históricos</strong> (2013–2014 · 4 lojas)
  &nbsp;→&nbsp; <strong>Feature Engineering</strong> (lags, sazonalidade, feriados)
  &nbsp;→&nbsp; <strong>Previsão H=7</strong> (Ensemble/RF/Prophet/SARIMAX/HW)
  &nbsp;→&nbsp; <strong>O1: Planos táticos</strong> (Monte Carlo / Hill Climbing / Simulated Annealing / NSGA-II)
  &nbsp;→&nbsp; <strong>O2: Alocação de rede</strong> (HC + Penalty / Death Penalty · ≤10k un.)
  &nbsp;→&nbsp; <strong>O3: Fronteira estratégica</strong> (MOEA/D · U-NSGA-III · 84 vars · Pareto interativo)
</div>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════════════════
# FOOTER
# ══════════════════════════════════════════════════════════════════════════════
st.markdown(f"""
<div class="dss-footer">
  DSS USA Stores &nbsp;|&nbsp; Relatório gerado em {TODAY} &nbsp;|&nbsp;
  4 lojas · 21 variáveis/loja · Restrição 10 000 unidades/semana &nbsp;|&nbsp;
  Valores em USD
</div>
""", unsafe_allow_html=True)
