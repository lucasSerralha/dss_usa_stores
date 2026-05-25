import streamlit as st
import sys
from datetime import date

sys.dont_write_bytecode = True

# ── Page Configuration ─────────────────────────────────────────────────────
st.set_page_config(
    page_title="DSS USA Stores",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Design System — Paleta corporativa (azuis escuros + cinzentos) ──────────
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

  .hero {
    padding: 48px 0 24px 0; border-bottom: 2px solid #1E3A8A; margin-bottom: 36px;
  }
  .hero-title {
    font-size: 2.1rem; font-weight: 800; color: #0F172A;
    letter-spacing: -0.03em; line-height: 1.15; margin: 0 0 8px 0;
  }
  .hero-sub {
    font-size: 0.92rem; color: #64748B; line-height: 1.6; max-width: 720px;
  }

  .pillar-card {
    padding: 28px 24px; border-radius: 10px; border: 1px solid #E2E8F0;
    background: #FFFFFF; height: 100%;
    transition: box-shadow 0.2s ease, border-color 0.2s ease;
  }
  .pillar-card:hover { border-color: #1E3A8A; box-shadow: 0 4px 16px rgba(15,23,42,0.06); }
  .pillar-number {
    display: inline-block; width: 30px; height: 30px; line-height: 30px;
    text-align: center; border-radius: 6px; background: #1E3A8A; color: #FFFFFF;
    font-size: 0.78rem; font-weight: 700; margin-bottom: 14px;
  }
  .pillar-title { font-size: 1.05rem; font-weight: 700; color: #0F172A; margin-bottom: 8px; }
  .pillar-desc { font-size: 0.85rem; color: #475569; line-height: 1.7; }
  .pillar-tag {
    display: inline-block; margin-top: 14px; padding: 3px 10px; border-radius: 4px;
    font-size: 0.72rem; font-weight: 600; letter-spacing: 0.04em; text-transform: uppercase;
  }
  .tag-blue  { background: rgba(30,58,138,0.08); color: #1E3A8A; }
  .tag-slate { background: rgba(51,65,85,0.08);  color: #334155; }
  .tag-green { background: rgba(5,150,105,0.08); color: #059669; }

  .arch-strip {
    margin-top: 36px; padding: 20px 24px; border-radius: 8px;
    border-left: 4px solid #1E3A8A; background: #F8FAFC;
    font-size: 0.88rem; color: #1E293B; line-height: 1.8;
  }
  .arch-title { font-weight: 700; font-size: 0.95rem; color: #1E3A8A; margin-bottom: 6px; }

  .dss-footer {
    font-size: 0.75rem; color: #94A3B8; text-align: right;
    margin-top: 48px; padding-top: 16px; border-top: 1px solid #E2E8F0;
  }
  .nav-hint {
    margin-top: 32px; padding: 14px 20px; border-radius: 8px;
    border: 1px dashed #CBD5E1; background: #FAFAFA;
    font-size: 0.82rem; color: #64748B; text-align: center;
  }
</style>
""", unsafe_allow_html=True)

# ── Constantes ─────────────────────────────────────────────────────────────
TODAY = date.today().strftime("%d de %B de %Y")

# ── Barra Lateral ──────────────────────────────────────────────────────────
st.sidebar.markdown(
    '<div style="font-size:0.88rem;font-weight:700;color:#E2E8F0;'
    'letter-spacing:0.04em;margin-bottom:16px;">DSS USA STORES</div>',
    unsafe_allow_html=True,
)
st.sidebar.markdown(
    '<div style="font-size:0.78rem;color:#94A3B8;line-height:1.6;">'
    "Plataforma de apoio a decisao para previsao da procura, analise exploratoria de dados "
    "e otimizacao multiobjetivo de recursos em quatro lojas retalhistas nos EUA."
    "</div>",
    unsafe_allow_html=True,
)
st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div style="font-size:0.72rem;color:#64748B;letter-spacing:0.06em;'
    'text-transform:uppercase;font-weight:600;">Estado do Sistema</div>',
    unsafe_allow_html=True,
)
st.sidebar.caption(f"Data do relatorio: {TODAY}")
st.sidebar.caption("Lojas: Baltimore, Lancaster, Philadelphia, Richmond")
st.sidebar.caption("Restricao de rede: 10 000 unidades/semana")
st.sidebar.markdown("---")
st.sidebar.markdown(
    '<div style="font-size:0.72rem;color:#64748B;letter-spacing:0.06em;'
    'text-transform:uppercase;font-weight:600;margin-bottom:8px;">Navegacao</div>',
    unsafe_allow_html=True,
)
st.sidebar.caption("01 — Analise Exploratoria")
st.sidebar.caption("02 — Previsao de Vendas")
st.sidebar.caption("03 — Otimizacao Monobjetivo (O1/O2)")
st.sidebar.caption("04 — Otimizacao Multiobjetivo (O3)")

# ── Hero ───────────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero">
    <div class="hero-title">Sistema de Apoio a Decisao<br>para Lojas Retalhistas nos EUA</div>
    <div class="hero-sub">
        Plataforma integrada de inteligencia de negocios para previsao da procura,
        validacao cientifica de modelos e otimizacao de recursos em quatro lojas
        retalhistas norte-americanas. Construida para planeamento estrategico e
        tatico ao nivel da loja e da rede logistica.
    </div>
</div>
""", unsafe_allow_html=True)

# ── Tres Pilares ───────────────────────────────────────────────────────────
col1, col2, col3 = st.columns(3, gap="medium")

with col1:
    st.markdown("""
    <div class="pillar-card">
        <div class="pillar-number">1</div>
        <div class="pillar-title">Previsao de Vendas</div>
        <div class="pillar-desc">
            Arquitetura hibrida adaptada ao perfil de volatilidade de cada mercado.
            Tres cenarios de experimentacao (Base Temporal, Dinamica de Vendas,
            Contexto Especialista) alimentam um motor de ensemble que seleciona
            automaticamente os tres modelos de melhor desempenho por loja.<br><br>
            O pipeline atinge cerca de 90% de precisao preditiva (1 &minus; MAPE),
            complementado por modelos de chegada de Poisson para estimativa
            probabilistica do fluxo de clientes.
        </div>
        <span class="pillar-tag tag-blue">Ensemble &middot; Prophet &middot; SARIMAX &middot; Random Forest</span>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown("""
    <div class="pillar-card">
        <div class="pillar-number">2</div>
        <div class="pillar-title">Otimizacao Monobjetivo</div>
        <div class="pillar-desc">
            Dois objetivos mono-objetivo resolvidos por heuristicas locais:<br><br>
            <b>O1</b> &mdash; Maximizacao do lucro semanal por loja com Hill Climbing
            (10 restarts &times; 2 000 iteracoes, 21 variaveis de decisao).<br><br>
            <b>O2</b> &mdash; Alocacao sob limite logistico de rede de 10 000 unidades
            (Heuristica com Death Penalty): penaliza solucoes inviáveis
            proporcionalmente ao excesso, tornando-as sempre inelegiveis.
        </div>
        <span class="pillar-tag tag-slate">Hill Climbing &middot; Death Penalty &middot; Heuristica</span>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown("""
    <div class="pillar-card">
        <div class="pillar-number">3</div>
        <div class="pillar-title">Otimizacao Multiobjetivo</div>
        <div class="pillar-desc">
            Analise de trade-offs estrategicos via Fronteira de Pareto (O3).<br><br>
            O algoritmo U-NSGA-III gera simultaneamente solucoes que maximizam
            o lucro e minimizam o custo de staff. O gestor seleciona o perfil
            estrategico via ponderacao escalar <em>w</em>: Agressivo, Conservador
            ou Equilibrado (Knee Point).
        </div>
        <span class="pillar-tag tag-green">U-NSGA-III &middot; Pareto &middot; Escalarizacao</span>
    </div>
    """, unsafe_allow_html=True)

# ── Faixa de Arquitetura ───────────────────────────────────────────────────
st.markdown("""
<div class="arch-strip">
    <div class="arch-title">Pipeline de Ponta a Ponta</div>
    <b>EDA</b> (Analise Exploratoria dos Dados)
    &nbsp;&rarr;&nbsp; <b>Previsao</b> (Ensemble Cenario C)
    &nbsp;&rarr;&nbsp; <b>O1</b> Planos taticos por loja (Hill Climbing)
    &nbsp;&rarr;&nbsp; <b>O2</b> Alocacao de rede com restricao logistica de 10 000 unidades
    (Heuristica com Death Penalty)
    &nbsp;&rarr;&nbsp; <b>O3</b> Fronteira de Pareto estrategica (U-NSGA-III)
</div>
""", unsafe_allow_html=True)

# ── Indicacao de Navegacao ─────────────────────────────────────────────────
st.markdown("""
<div class="nav-hint">
    Utilize a <b>navegacao lateral</b> para aceder aos modulos:
    Analise Exploratoria, Previsao de Vendas, Otimizacao Monobjetivo (O1/O2) e
    Otimizacao Multiobjetivo (O3).
</div>
""", unsafe_allow_html=True)

# ── Rodape ─────────────────────────────────────────────────────────────────
st.markdown(f"""
<div class="dss-footer">
    DSS &nbsp;|&nbsp; Gerado em: {TODAY} &nbsp;|&nbsp; Todos os valores monetarios em USD
    &nbsp;|&nbsp; 4 lojas &middot; 21 variaveis de decisao por loja
</div>
""", unsafe_allow_html=True)
