import streamlit as st
import sys
import os
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import date

sys.dont_write_bytecode = True

st.set_page_config(
    page_title="TIAPOSE — Decision Support System",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Professional CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
  /* Typography */
  html, body, [class*="css"] { font-family: 'Inter', 'Segoe UI', sans-serif; }
  h1 { font-size: 1.6rem; font-weight: 700; letter-spacing: -0.02em; margin-bottom: 0; }
  h2 { font-size: 1.2rem; font-weight: 600; letter-spacing: -0.01em; }
  h3 { font-size: 1.0rem; font-weight: 600; }

  /* Section header with left accent bar */
  .section-header {
    border-left: 3px solid #1E3A8A;
    padding-left: 10px;
    margin: 20px 0 12px 0;
    font-size: 1.05rem;
    font-weight: 600;
    color: #0F172A;
  }

  /* KPI card */
  .kpi-card {
    padding: 18px 20px;
    border-radius: 8px;
    border: 1px solid #E2E8F0;
    background: #FFFFFF;
  }
  .kpi-label { font-size: 0.78rem; font-weight: 500; color: #64748B; text-transform: uppercase; letter-spacing: 0.06em; margin-bottom: 4px; }
  .kpi-value { font-size: 1.7rem; font-weight: 700; color: #0F172A; }
  .kpi-delta { font-size: 0.78rem; font-weight: 600; color: #059669; margin-top: 2px; }

  /* Algorithm comparison cards */
  .algo-card {
    padding: 16px;
    border-radius: 8px;
    border: 1px solid #E2E8F0;
    background: #FAFAFA;
    margin-bottom: 8px;
    height: 100%;
  }
  .algo-card-title { font-size: 0.95rem; font-weight: 700; color: #0F172A; margin-bottom: 2px; }
  .algo-card-sub   { font-size: 0.78rem; color: #64748B; margin-bottom: 10px; }
  .algo-card ul    { margin: 0; padding-left: 16px; font-size: 0.85rem; color: #334155; }
  .algo-card li    { margin-bottom: 4px; }

  /* Winner card */
  .winner-card {
    padding: 16px;
    border-radius: 8px;
    border: 1px solid #059669;
    background: rgba(5, 150, 105, 0.04);
    margin-bottom: 8px;
    height: 100%;
  }
  .winner-badge {
    display: inline-block;
    background: #059669;
    color: #FFFFFF;
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    padding: 2px 8px;
    border-radius: 4px;
    margin-bottom: 6px;
    text-transform: uppercase;
  }

  /* Verdict box */
  .verdict-box {
    padding: 20px 24px;
    border-radius: 8px;
    border-left: 4px solid #1E3A8A;
    background: #F8FAFC;
    margin-top: 20px;
    font-size: 0.9rem;
    color: #1E293B;
    line-height: 1.7;
  }
  .verdict-title { font-weight: 700; font-size: 1.0rem; color: #1E3A8A; margin-bottom: 8px; }

  /* Executive report */
  .report-header {
    border-bottom: 2px solid #1E3A8A;
    padding-bottom: 16px;
    margin-bottom: 24px;
  }
  .report-title  { font-size: 1.5rem; font-weight: 800; color: #0F172A; letter-spacing: -0.02em; }
  .report-sub    { font-size: 0.85rem; color: #64748B; margin-top: 4px; }
  .report-section {
    padding: 20px 24px;
    border-radius: 8px;
    border: 1px solid #E2E8F0;
    background: #FFFFFF;
    margin-bottom: 16px;
  }
  .report-section-title {
    font-size: 0.75rem;
    font-weight: 700;
    color: #1E3A8A;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin-bottom: 10px;
    padding-bottom: 8px;
    border-bottom: 1px solid #E2E8F0;
  }

  /* Metric comparison table row */
  .cmp-row { display: flex; justify-content: space-between; padding: 6px 0; border-bottom: 1px solid #F1F5F9; font-size: 0.88rem; }
  .cmp-row:last-child { border-bottom: none; }
  .cmp-label { color: #475569; }
  .cmp-val   { font-weight: 600; color: #0F172A; }

  /* Muted italic description */
  .desc { font-size: 0.85rem; color: #64748B; font-style: italic; margin: -8px 0 16px 0; }

  /* Streamlit native metric override */
  div[data-testid="metric-container"] {
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 14px 18px;
  }
</style>
""", unsafe_allow_html=True)

# ── Constants & paths ──────────────────────────────────────────────────────
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORES = ["baltimore", "lancaster", "philadelphia", "richmond"]
TODAY = date.today().strftime("%d %B %Y")

CHART_PALETTE = {
    "primary":   "#1E3A8A",
    "secondary": "#64748B",
    "accent":    "#0EA5E9",
    "green":     "#059669",
    "amber":     "#D97706",
    "red":       "#DC2626",
    "slate":     "#334155",
}

PLOTLY_TEMPLATE = dict(
    paper_bgcolor="white",
    plot_bgcolor="#F8FAFC",
    font=dict(family="Inter, Segoe UI, sans-serif", color="#334155", size=12),
    margin=dict(t=52, b=80, l=16, r=16),
    xaxis=dict(gridcolor="#E2E8F0", linecolor="#CBD5E1", zerolinecolor="#E2E8F0"),
    yaxis=dict(gridcolor="#E2E8F0", linecolor="#CBD5E1", zerolinecolor="#E2E8F0"),
    legend=dict(bgcolor="rgba(255,255,255,0.9)", bordercolor="#E2E8F0", borderwidth=1,
                orientation="h", yanchor="top", y=-0.15, xanchor="center", x=0.5),
)


# ── Data loaders ───────────────────────────────────────────────────────────
@st.cache_data
def load_historical():
    p = os.path.join(ROOT, "data/processed/all_stores_processed.csv")
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    df["Date"] = pd.to_datetime(df["Date"])
    return df


@st.cache_data
def load_master():
    p = os.path.join(ROOT, "results/00_Master_Summary/fidelity_experimentation_report.csv")
    return pd.read_csv(p) if os.path.exists(p) else None


@st.cache_data
def load_csv(rel_path):
    p = os.path.join(ROOT, rel_path)
    return pd.read_csv(p) if os.path.exists(p) else None


def styled_layout(**overrides):
    """Merge PLOTLY_TEMPLATE with overrides, deep-merging xaxis/yaxis dicts.

    Calling fig.update_layout(**styled_layout(yaxis={"tickformat": ".0%"}))
    avoids the duplicate-keyword TypeError that occurs when xaxis/yaxis
    overrides collide with the keys already present in PLOTLY_TEMPLATE.
    """
    layout = {k: (dict(v) if isinstance(v, dict) else v) for k, v in PLOTLY_TEMPLATE.items()}
    for axis_key in ("xaxis", "yaxis"):
        if axis_key in overrides:
            layout[axis_key] = {**layout.get(axis_key, {}), **overrides.pop(axis_key)}
    layout.update(overrides)
    return layout


df_history = load_historical()
df_master  = load_master()

if df_history is None:
    st.error("Critical error: processed data not found. Run the pipeline first.")
    st.stop()

# ── Pre-load all optimisation data (used in multiple tabs) ─────────────────
df_hc        = load_csv("results/03_Optimization_Report/individual/optimization_summary.csv")
df_nsga2     = load_csv("results/03_Optimization_Report/multiobjective/optimization_summary.csv")
df_alloc     = load_csv("results/03_Optimization_Report/allocation/allocation_summary.csv")
df_knap      = load_csv("results/03_Optimization_Report/allocation_knapsack/allocation_summary_knapsack.csv")
df_o3_sum    = load_csv("results/03_Optimization_Report/joint_o3/joint_o3_summary.csv")
df_p_moead   = load_csv("results/03_Optimization_Report/joint_o3/joint_pareto_moead.csv")
df_p_unsga3  = load_csv("results/03_Optimization_Report/joint_o3/joint_pareto_unsga3.csv")


# ── Sidebar ────────────────────────────────────────────────────────────────
st.sidebar.markdown("### DSS Control Panel")
st.sidebar.markdown("---")

stores_display = [s.capitalize() for s in sorted(df_history["store_id"].unique())]
store_raw  = st.sidebar.selectbox("Active Store", stores_display)
store      = store_raw.lower()

cenario_sel = "C_Context_Expert"
if df_master is not None:
    scenarios = sorted(df_master["Experiment"].unique())
    cenario_sel = st.sidebar.radio("Forecast Scenario", scenarios, index=len(scenarios) - 1)

st.sidebar.markdown("---")
st.sidebar.markdown("**System Status**")
st.sidebar.success("Data: Synchronised")
st.sidebar.info(f"Active store: {store.capitalize()}")

# ── Filtered master metrics for active store & scenario ────────────────────
store_metrics = pd.DataFrame()
if df_master is not None:
    store_metrics = df_master[
        (df_master["Store"].str.lower() == store) &
        (df_master["Experiment"] == cenario_sel)
    ]

# ── Page header ───────────────────────────────────────────────────────────
col_h1, col_h2 = st.columns([3, 1])
with col_h1:
    st.markdown(f"##USA Retail Stores Decision Support System")
    st.markdown(f"<p class='desc'>Store: {store.capitalize()} &nbsp;|&nbsp; Scenario: {cenario_sel} &nbsp;|&nbsp; {TODAY}</p>", unsafe_allow_html=True)
with col_h2:
    st.markdown("<br>", unsafe_allow_html=True)

# ── Top KPIs ───────────────────────────────────────────────────────────────
df_store = df_history[df_history["store_id"] == store]
kpi1, kpi2, kpi3, kpi4 = st.columns(4)

with kpi1:
    st.markdown(f"""
    <div class="kpi-card">
        <div class="kpi-label">Total Sales (Historical)</div>
        <div class="kpi-value">${df_store['Sales'].sum()/1e6:.2f}M</div>
    </div>""", unsafe_allow_html=True)

if not store_metrics.empty:
    rmse_min  = store_metrics["RMSE"].min()
    mape_min  = store_metrics["MAPE"].min()
    best_mod  = store_metrics.loc[store_metrics["RMSE"].idxmin(), "Model"]
    naive_row = store_metrics[store_metrics["Model"] == "Seasonal Naive"]["RMSE"]
    ganho_str = ""
    if not naive_row.empty:
        ganho = (naive_row.values[0] - rmse_min) / naive_row.values[0] * 100
        ganho_str = f"+{ganho:.1f}% vs Naive Baseline"

    with kpi2:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Best Model (Scenario)</div>
            <div class="kpi-value" style="font-size:1.1rem;">{best_mod}</div>
            <div class="kpi-delta">{ganho_str}</div>
        </div>""", unsafe_allow_html=True)

    with kpi3:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Best Daily RMSE</div>
            <div class="kpi-value">${rmse_min:,.0f}</div>
        </div>""", unsafe_allow_html=True)

    fidelity = 100 - mape_min
    with kpi4:
        st.markdown(f"""
        <div class="kpi-card">
            <div class="kpi-label">Forecast Fidelity (1 − MAPE)</div>
            <div class="kpi-value">{fidelity:.1f}%</div>
        </div>""", unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ── Navigation tabs ────────────────────────────────────────────────────────
(tab_forecast, tab_residual, tab_decomp, tab_xai, tab_prob,
 tab_audit, tab_o1, tab_o2, tab_o3, tab_exec) = st.tabs([
    "Sales Forecast",
    "Residual Analysis",
    "Trend Decomposition",
    "Feature Importance",
    "Probabilistic Models",
    "Model Audit",
    "O1 — Per-Store Optimisation",
    "O2 — Joint Allocation",
    "O3 — Multi-Objective Global",
    "Executive Report",
])


# ══════════════════════════════════════════════════════════════════════════
# TAB 1 — SALES FORECAST
# ══════════════════════════════════════════════════════════════════════════
with tab_forecast:
    st.markdown('<div class="section-header">Interactive Sales Forecast</div>', unsafe_allow_html=True)
    st.markdown('<p class="desc">Comparative view of actual historical demand vs model projections. The recommended champion model is highlighted by default.</p>', unsafe_allow_html=True)

    path_raw = os.path.join(ROOT, f"results/02_Forecasting_Report/{store.capitalize()}/{cenario_sel}/forecast_values.csv")
    best_model_name = None

    if os.path.exists(path_raw):
        df_plot = pd.read_csv(path_raw)
        df_plot["Date"] = pd.to_datetime(df_plot["Date"])

        if not store_metrics.empty:
            best_model_name = store_metrics.loc[store_metrics["RMSE"].idxmin(), "Model"]

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=df_plot["Date"], y=df_plot["Actual"],
            name="Actual", line=dict(color=CHART_PALETTE["primary"], width=3),
            mode="lines+markers", marker=dict(size=3)))

        model_cols = [c for c in df_plot.columns if c not in ["Date", "Actual"]]
        palette    = [CHART_PALETTE["accent"], CHART_PALETTE["amber"], CHART_PALETTE["secondary"],
                      "#7C3AED", "#0891B2", "#B45309"]

        for i, m in enumerate(model_cols):
            is_best = (m == best_model_name)
            fig.add_trace(go.Scatter(
                x=df_plot["Date"], y=df_plot[m], name=m,
                line=dict(color=palette[i % len(palette)],
                          width=2.5 if is_best else 1.2,
                          dash="solid" if is_best else "dash"),
                visible=True if is_best else "legendonly"))

        fig.update_layout(**PLOTLY_TEMPLATE,
                          title=f"Sales Forecast — {store.capitalize()} | {cenario_sel}",
                          xaxis_title="Date", yaxis_title="Sales ($)",
                          hovermode="x unified")
        st.plotly_chart(fig, use_container_width=True, theme=None)

        if best_model_name:
            st.info(f"Champion model for this store and scenario: **{best_model_name}** — shown by default. Click legend items to compare alternatives.")
    else:
        st.warning("Forecast data not found for this selection. Run the main pipeline first.")


# ══════════════════════════════════════════════════════════════════════════
# TAB 2 — RESIDUAL ANALYSIS
# ══════════════════════════════════════════════════════════════════════════
with tab_residual:
    st.markdown('<div class="section-header">Error and Residual Analysis</div>', unsafe_allow_html=True)
    st.markdown('<p class="desc">Statistical validation of model stability and absence of systematic bias. An unbiased model produces residuals centred at zero.</p>', unsafe_allow_html=True)

    if "df_plot" in dir() and best_model_name:
        df_err       = df_plot.copy()
        df_err["Residual"] = df_err[best_model_name] - df_err["Actual"]

        col_r1, col_r2 = st.columns(2)
        with col_r1:
            fig_hist = px.histogram(df_err, x="Residual", nbins=25,
                                    title=f"Residual Distribution — {best_model_name}",
                                    labels={"Residual": "Residual ($)", "count": "Frequency"},
                                    color_discrete_sequence=[CHART_PALETTE["primary"]])
            fig_hist.update_layout(**PLOTLY_TEMPLATE)
            st.plotly_chart(fig_hist, use_container_width=True, theme=None)
            st.caption("Histogram centred at zero confirms the model is free of systematic bias.")

        with col_r2:
            fig_scatter = px.scatter(df_err, x="Actual", y=best_model_name, trendline="ols",
                                     title="Actual vs Predicted Scatter",
                                     labels={"Actual": "Actual ($)", best_model_name: "Predicted ($)"},
                                     color_discrete_sequence=[CHART_PALETTE["accent"]])
            fig_scatter.update_layout(**PLOTLY_TEMPLATE)
            st.plotly_chart(fig_scatter, use_container_width=True, theme=None)
            st.caption("Points close to the diagonal indicate high model fidelity.")
    else:
        st.info("No forecast data loaded. Select a valid store and scenario in the Sales Forecast tab first.")


# ══════════════════════════════════════════════════════════════════════════
# TAB 3 — TREND DECOMPOSITION
# ══════════════════════════════════════════════════════════════════════════
with tab_decomp:
    st.markdown('<div class="section-header">Sales Anatomy — Trend & Seasonality</div>', unsafe_allow_html=True)
    st.markdown('<p class="desc">Decomposition of long-term trend and weekly seasonal patterns extracted via Facebook Prophet.</p>', unsafe_allow_html=True)

    path_comp = os.path.join(ROOT, f"results/02_Forecasting_Report/{store.capitalize()}/{cenario_sel}/prophet_components.csv")
    if os.path.exists(path_comp):
        df_comp = pd.read_csv(path_comp)
        df_comp["ds"] = pd.to_datetime(df_comp["ds"])

        col_c1, col_c2 = st.columns(2)
        with col_c1:
            fig_trend = px.line(df_comp, x="ds", y="trend",
                                title="Long-Term Trend",
                                labels={"ds": "Date", "trend": "Trend ($)"},
                                line_shape="spline",
                                color_discrete_sequence=[CHART_PALETTE["primary"]])
            fig_trend.update_layout(**PLOTLY_TEMPLATE)
            st.plotly_chart(fig_trend, use_container_width=True, theme=None)

        with col_c2:
            if "weekly" in df_comp.columns:
                df_wk = df_comp.tail(7).copy()
                day_map = {"Monday": "Monday", "Tuesday": "Tuesday", "Wednesday": "Wednesday",
                           "Thursday": "Thursday", "Friday": "Friday", "Saturday": "Saturday", "Sunday": "Sunday"}
                df_wk["Day"] = df_wk["ds"].dt.day_name().map(day_map)
                fig_wk = px.bar(df_wk, x="Day", y="weekly",
                                title="Weekly Seasonality Pattern",
                                labels={"Day": "Day of Week", "weekly": "Deviation ($)"},
                                color="weekly",
                                color_continuous_scale=[[0, "#DC2626"], [0.5, "#F8FAFC"], [1, "#059669"]])
                fig_wk.update_layout(**PLOTLY_TEMPLATE)
                st.plotly_chart(fig_wk, use_container_width=True, theme=None)
            else:
                st.info("Weekly seasonality component not available for this period.")
    else:
        st.info("Decomposition data available after running the main forecasting pipeline.")


# ══════════════════════════════════════════════════════════════════════════
# TAB 4 — FEATURE IMPORTANCE (XAI)
# ══════════════════════════════════════════════════════════════════════════
with tab_xai:
    st.markdown('<div class="section-header">Algorithmic Explainability — Feature Importance</div>', unsafe_allow_html=True)
    st.markdown('<p class="desc">Top drivers influencing model forecasts. Variable importance is extracted from the Random Forest component of the best ensemble.</p>', unsafe_allow_html=True)

    path_feat = os.path.join(ROOT, f"results/02_Forecasting_Report/{store.capitalize()}/{cenario_sel}/feature_importance.csv")
    if os.path.exists(path_feat):
        df_feat = pd.read_csv(path_feat).head(10)
        fig_feat = px.bar(df_feat, x="Importance", y="Feature", orientation="h",
                          title=f"Top 10 Predictive Features — {store.capitalize()}",
                          labels={"Feature": "Variable", "Importance": "Importance Score"},
                          color="Importance",
                          color_continuous_scale=[[0, "#CBD5E1"], [1, CHART_PALETTE["primary"]]])
        fig_feat.update_layout(**PLOTLY_TEMPLATE)
        fig_feat.update_yaxes(categoryorder="total ascending")
        st.plotly_chart(fig_feat, use_container_width=True, theme=None)
    else:
        st.warning("Feature importance data not available for this selection.")


# ══════════════════════════════════════════════════════════════════════════
# TAB 5 — PROBABILISTIC MODELS
# ══════════════════════════════════════════════════════════════════════════
with tab_prob:
    st.markdown('<div class="section-header">Rule-Based Probabilistic Models</div>', unsafe_allow_html=True)
    st.markdown('<p class="desc">Three statistical models operating on domain rules: Poisson (customer arrivals), Gaussian (sales variability), Logistic (discount conversion).</p>', unsafe_allow_html=True)

    _src = os.path.join(ROOT, "src")
    if _src not in sys.path:
        sys.path.insert(0, _src)

    col_cb1, col_cb2, col_cb3, _ = st.columns([2, 2, 2, 1])
    show_poisson  = col_cb1.checkbox("Poisson Model",   value=True)
    show_gauss    = col_cb2.checkbox("Gaussian Model",  value=True)
    show_logistic = col_cb3.checkbox("Logistic Model",  value=True)
    st.markdown("---")

    try:
        from optimization.probabilistic_models import (
            PoissonArrivalModel, GaussianSalesModel,
            LogisticConversionModel, compare_with_forecasting)

        data_path = os.path.join(ROOT, "data", "processed", f"{store}_processed.csv")
        if not os.path.exists(data_path):
            st.warning(f"Processed data not found for {store.capitalize()}. Run the main pipeline first.")
        else:
            df_prob = pd.read_csv(data_path, parse_dates=["Date"])
            # Processed CSVs use 'IsWeekend' (CamelCase) but probabilistic_models.py
            # expects 'is_weekend' (snake_case). Normalise so the feature is picked
            # up by the GLM and is available for the test-week display below.
            if "IsWeekend" in df_prob.columns and "is_weekend" not in df_prob.columns:
                df_prob["is_weekend"] = df_prob["IsWeekend"].astype(int)
            with st.spinner("Training probabilistic models..."):
                pm = PoissonArrivalModel(); pm.fit(df_prob, store=store)
                gm = GaussianSalesModel();  gm.fit(df_prob, store=store)
                lm = LogisticConversionModel(); lm.fit(df_prob, store=store)

            split     = int(len(df_prob) * 0.8)
            test_week = df_prob.iloc[split:split + 7].copy()
            day_ord   = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
            x_days    = [day_ord.get(int(r), f"D{i+1}") for i, r in enumerate(test_week["day_of_week"])]
            is_wknd   = test_week["is_weekend"].astype(bool).tolist()

            # ── Poisson ──────────────────────────────────────────────────
            if show_poisson:
                st.markdown('<div class="section-header" style="font-size:0.9rem;">Poisson Model — Customer Arrival Forecast</div>', unsafe_allow_html=True)
                X_pois  = test_week[[f for f in pm._features if f in test_week.columns]]
                lambdas = pm.predict_lambda(X_pois)

                col_p1, col_p2 = st.columns([2, 1])
                with col_p1:
                    bar_colors = [CHART_PALETTE["amber"] if w else CHART_PALETTE["primary"] for w in is_wknd]
                    fig_pois = go.Figure(go.Bar(
                        x=x_days, y=[round(l) for l in lambdas],
                        marker_color=bar_colors,
                        text=[str(round(l)) for l in lambdas],
                        textposition="outside"))
                    fig_pois.update_layout(**PLOTLY_TEMPLATE,
                                           title="Forecast Customer Arrivals (next week)",
                                           xaxis_title="Day", yaxis_title="Customers (λ)",
                                           showlegend=False)
                    st.plotly_chart(fig_pois, use_container_width=True, theme=None)
                with col_p2:
                    cmp = compare_with_forecasting(df_prob, pm, store=store)
                    st.markdown("**vs Lag-7 Baseline**")
                    st.dataframe(cmp[["modelo", "MAE", "RMSE"]].rename(
                        columns={"modelo": "Method", "MAE": "MAE (customers)", "RMSE": "RMSE"}),
                        hide_index=True, use_container_width=True)
                    imp = round((1 - cmp.iloc[0]["MAE"] / cmp.iloc[1]["MAE"]) * 100, 1)
                    st.success(f"Poisson is **{imp}% more accurate** than the lag-7 historical baseline.")

                if show_gauss or show_logistic:
                    st.markdown("---")

            # ── Gaussian ─────────────────────────────────────────────────
            if show_gauss:
                st.markdown('<div class="section-header" style="font-size:0.9rem;">Gaussian Model — Sales Variability & Confidence Intervals</div>', unsafe_allow_html=True)
                scenarios         = gm.predict_scenarios(test_week)
                lower_95, upper_95 = gm.confidence_interval(test_week, confidence=0.95)

                fig_g = go.Figure()
                fig_g.add_trace(go.Scatter(x=x_days, y=upper_95, mode="lines", line=dict(width=0), showlegend=False))
                fig_g.add_trace(go.Scatter(x=x_days, y=lower_95, mode="lines", line=dict(width=0),
                                           fill="tonexty", fillcolor="rgba(30,58,138,0.08)", name="95% CI"))
                fig_g.add_trace(go.Scatter(x=x_days, y=scenarios["pessimistic"], mode="lines+markers",
                                           line=dict(color=CHART_PALETTE["red"], dash="dot"), name="Pessimistic"))
                fig_g.add_trace(go.Scatter(x=x_days, y=scenarios["realistic"], mode="lines+markers",
                                           line=dict(color=CHART_PALETTE["primary"], width=2.5), name="Realistic"))
                fig_g.add_trace(go.Scatter(x=x_days, y=scenarios["optimistic"], mode="lines+markers",
                                           line=dict(color=CHART_PALETTE["green"], dash="dot"), name="Optimistic"))
                target = "y" if "y" in test_week.columns else "Sales"
                fig_g.add_trace(go.Scatter(x=x_days, y=test_week[target].values, mode="lines+markers",
                                           line=dict(color=CHART_PALETTE["amber"], width=3),
                                           marker=dict(symbol="star", size=9), name="Actual Sales"))
                fig_g.update_layout(**PLOTLY_TEMPLATE,
                                    title=f"Sales Scenarios — sigma=${gm.sigma:,.0f} | mu=${gm.mu_global:,.0f}",
                                    xaxis_title="Day of Week", yaxis_title="Sales ($)",
                                    hovermode="x unified")
                st.plotly_chart(fig_g, use_container_width=True, theme=None)
                if show_logistic:
                    st.markdown("---")

            # ── Logistic ─────────────────────────────────────────────────
            if show_logistic:
                st.markdown('<div class="section-header" style="font-size:0.9rem;">Logistic Model — Discount Conversion Probability</div>', unsafe_allow_html=True)
                pr_rng     = np.linspace(0.0, 0.30, 60)
                p_weekday  = lm.conversion_curve(pr_rng, is_weekend=0)
                p_weekend  = lm.conversion_curve(pr_rng, is_weekend=1)
                p_holiday  = lm.conversion_curve(pr_rng, is_weekend=0, is_holiday=1, days_to_next_holiday=0)

                col_l1, col_l2 = st.columns([2, 1])
                with col_l1:
                    fig_l = go.Figure()
                    fig_l.add_trace(go.Scatter(x=pr_rng*100, y=p_weekday, name="Weekday",
                                               line=dict(color=CHART_PALETTE["primary"], width=2.5)))
                    fig_l.add_trace(go.Scatter(x=pr_rng*100, y=p_weekend, name="Weekend",
                                               line=dict(color=CHART_PALETTE["amber"], width=2.5)))
                    fig_l.add_trace(go.Scatter(x=pr_rng*100, y=p_holiday, name="Holiday",
                                               line=dict(color=CHART_PALETTE["red"], width=2, dash="dash")))
                    fig_l.add_hline(y=0.5, line_dash="dot", line_color=CHART_PALETTE["secondary"],
                                    annotation_text="50% conversion threshold")
                    fig_l.update_layout(**PLOTLY_TEMPLATE,
                                        title=f"Conversion Curve — AUC-ROC = {lm.auc:.3f}",
                                        xaxis_title="Applied Discount (%)", yaxis_title="P(Conversion)",
                                        hovermode="x unified")
                    fig_l.update_yaxes(tickformat=".0%", range=[0, 1])
                    st.plotly_chart(fig_l, use_container_width=True, theme=None)
                with col_l2:
                    st.metric("AUC-ROC", f"{lm.auc:.3f}", help="1.0 = perfect | 0.5 = random")
                    st.metric("Threshold", f"${lm.threshold:,.1f}/customer")
                    p0  = lm.conversion_curve(np.array([0.0]))[0] * 100
                    p30 = lm.conversion_curve(np.array([0.30]))[0] * 100
                    st.info(f"At 0% discount: P(conv) = {p0:.1f}%\nAt 30% discount: P(conv) = {p30:.1f}%\n\nDiscount yields diminishing marginal returns.")

            if not show_poisson and not show_gauss and not show_logistic:
                st.info("Select at least one model to display.")

    except ImportError as e:
        st.error(f"Import error: {e}")
    except Exception as e:
        st.error(f"Unexpected error: {e}")
        st.exception(e)


# ══════════════════════════════════════════════════════════════════════════
# TAB 6 — MODEL AUDIT
# ══════════════════════════════════════════════════════════════════════════
with tab_audit:
    st.markdown('<div class="section-header">Forecast Model Audit — Leaderboard</div>', unsafe_allow_html=True)
    st.markdown('<p class="desc">Empirical benchmarking of all predictive models against the Seasonal Naive statistical baseline across stores and scenarios.</p>', unsafe_allow_html=True)

    df_base  = load_csv("results/04_Model_Testing/baseline_models_report.csv")
    df_lgb   = load_csv("results/04_Model_Testing/lightgbm_tuning_report.csv")
    df_naive = load_csv("results/02_Forecasting_Report/baseline/baseline_metrics.csv")

    if df_base is None or df_lgb is None or df_naive is None:
        st.warning("Some audit files are missing. Run the full pipeline to generate them.")
    else:
        df_base_c  = df_base[["Loja", "Cenario", "Modelo", "RMSE", "MAPE"]].copy()
        df_lgb_c   = df_lgb[["Loja", "Cenario", "RMSE", "MAPE"]].copy()
        df_lgb_c["Modelo"] = "LightGBM"
        df_naive_c = df_naive[["Store", "Model", "RMSE"]].copy()
        df_naive_c.rename(columns={"Store": "Loja", "Model": "Modelo"}, inplace=True)
        df_naive_c["Cenario"] = "Baseline_Naive"
        df_naive_c["MAPE"]    = np.nan

        df_all = pd.concat([df_base_c, df_lgb_c, df_naive_c], ignore_index=True)
        df_all["Loja"] = df_all["Loja"].str.lower()
        df_all.replace([np.inf, -np.inf], np.nan, inplace=True)

        st.markdown("#### Global Leaderboard")
        df_lb = df_all[["Loja", "Cenario", "Modelo", "MAPE", "RMSE"]].sort_values("RMSE").reset_index(drop=True)
        df_lb.rename(columns={"Loja": "Store", "Cenario": "Scenario", "Modelo": "Model"}, inplace=True)
        st.dataframe(df_lb, use_container_width=True, hide_index=True)

        st.markdown("---")

        df_naive_m = df_naive_c.copy()
        df_naive_m["Loja"]    = df_naive_m["Loja"].str.lower()
        df_naive_m["Cenario"] = cenario_sel

        df_chart = pd.concat([df_base_c, df_lgb_c, df_naive_m], ignore_index=True)
        df_chart["Loja"] = df_chart["Loja"].str.lower()
        df_store_chart = df_chart[
            (df_chart["Loja"] == store) & (df_chart["Cenario"] == cenario_sel)
        ].dropna(subset=["RMSE"])

        if not df_store_chart.empty:
            st.markdown(f"#### Performance Podium — {store.capitalize()} | {cenario_sel}")
            podium = df_store_chart.sort_values("RMSE").head(3)
            medals = ["1st Place", "2nd Place", "3rd Place"]
            cols   = st.columns(3)
            for idx, (col, (_, row)) in enumerate(zip(cols, podium.iterrows())):
                mape_str = f" | MAPE: {row['MAPE']*100:.1f}%" if pd.notna(row["MAPE"]) else ""
                border = "border: 1px solid #059669;" if idx == 0 else "border: 1px solid #E2E8F0;"
                col.markdown(f"""
                <div style="padding:16px;border-radius:8px;{border}">
                    <div style="font-size:0.75rem;font-weight:700;color:#64748B;text-transform:uppercase;letter-spacing:0.06em;">{medals[idx]}</div>
                    <div style="font-size:1.05rem;font-weight:700;color:#0F172A;margin:6px 0 4px 0;">{row['Modelo']}</div>
                    <div style="font-size:0.88rem;color:#475569;">RMSE: \${row['RMSE']:,.0f}{mape_str}</div>
                </div>""", unsafe_allow_html=True)

            st.markdown("<br>", unsafe_allow_html=True)
            best_m = podium.iloc[0]["Modelo"]
            df_s   = df_store_chart.sort_values("RMSE", ascending=False)
            fig_a  = go.Figure()
            fig_a.add_trace(go.Bar(
                x=df_s[df_s["Modelo"] != best_m]["RMSE"],
                y=df_s[df_s["Modelo"] != best_m]["Modelo"],
                orientation="h", name="Other models",
                marker_color=CHART_PALETTE["secondary"],
                text=df_s[df_s["Modelo"] != best_m]["RMSE"].round(0),
                textposition="auto"))
            fig_a.add_trace(go.Bar(
                x=df_s[df_s["Modelo"] == best_m]["RMSE"],
                y=df_s[df_s["Modelo"] == best_m]["Modelo"],
                orientation="h", name="Champion model",
                marker_color=CHART_PALETTE["green"],
                text=df_s[df_s["Modelo"] == best_m]["RMSE"].round(0),
                textposition="auto"))
            fig_a.update_layout(**PLOTLY_TEMPLATE,
                                xaxis_title="RMSE — lower is better",
                                yaxis_title="")
            st.plotly_chart(fig_a, use_container_width=True, theme=None)

        st.markdown("---")
        st.markdown("#### Strategic Conclusions")
        st.markdown("""
<div style="font-size:0.88rem;color:#334155;line-height:1.8;">
<b>1. Linear Regression and Prophet</b> outperform complex models in stable markets (Richmond, Baltimore),
confirming that parsimony is preferable when demand patterns are regular and exogenous context is rich.<br>
<b>2. LightGBM</b> adapts well to non-linear relationships but requires rigorous hyperparameter tuning
(max_depth, regularisation) to prevent overfitting on high-granularity daily series.<br>
<b>3. SARIMAX</b> collapses under Scenario B (short-term sales dynamics) due to MLE instability against
high variance and autoregressive dynamics — suitable only for near-stationary series with low-correlation exogenous variables.<br>
<b>4. Recommended architecture:</b> Ensemble (Top-3) as the primary forecasting layer,
feeding the optimisation engine with probabilistic demand inputs via the Poisson arrival model.
</div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 7 — O1: PER-STORE OPTIMISATION
# ══════════════════════════════════════════════════════════════════════════
with tab_o1:
    st.markdown('<div class="section-header">O1 — Per-Store Optimisation</div>', unsafe_allow_html=True)
    st.markdown('<p class="desc">Comparison of mono-objective (Hill Climbing) vs multi-objective (NSGA-II) algorithms for individual store profit and staffing optimisation. Decision space: 21 variables — daily discount, expert staff, and junior staff for 7 days.</p>', unsafe_allow_html=True)

    # Algorithm comparison cards
    col_a1, col_a2 = st.columns(2)
    with col_a1:
        st.markdown("""
        <div class="algo-card">
            <div class="algo-card-title">Hill Climbing with Random Restarts</div>
            <div class="algo-card-sub">Mono-objective | Local Search | 10 restarts x 2,000 iterations</div>
            <ul>
                <li>Converges directly to maximum profit</li>
                <li>Single, interpretable output plan</li>
                <li>Random restarts mitigate local minima</li>
                <li>Does not produce a trade-off frontier</li>
                <li>Ignores staffing cost as a secondary objective</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    with col_a2:
        st.markdown("""
        <div class="winner-card">
            <div class="winner-badge">Recommended for O1</div>
            <div class="algo-card-title">Hill Climbing with Random Restarts</div>
            <div class="algo-card-sub">Selected when the objective is maximum weekly profit per store</div>
            <ul>
                <li>Achieves 60–130% higher profit than NSGA-II's Pareto max</li>
                <li>Focused 100% on the single declared objective</li>
                <li>Low computational cost, fast convergence</li>
            </ul>
            <br>
            <div style="font-size:0.8rem;color:#475569;"><b>Use NSGA-II instead when</b> management needs to explore the profit-vs-staff trade-off frontier rather than a single optimal plan.</div>
        </div>""", unsafe_allow_html=True)

    # Comparative performance table
    if df_hc is not None and df_nsga2 is not None:
        st.markdown('<div class="section-header">Algorithm Performance Comparison</div>', unsafe_allow_html=True)

        df_cmp = df_hc[["store", "lucro_maximo", "total_staff", "avg_discount"]].copy()
        df_cmp.columns = ["Store", "HC Max Profit ($)", "HC Weekly Staff", "HC Avg Discount (%)"]
        nsga2_idx = df_nsga2.set_index("store")
        df_cmp["NSGA-II Max Profit ($)"]  = df_cmp["Store"].map(nsga2_idx["max_profit"])
        df_cmp["NSGA-II Staff at Max P"]  = df_cmp["Store"].map(nsga2_idx["staff_at_max_profit"])
        df_cmp["NSGA-II Pareto Solutions"] = df_cmp["Store"].map(nsga2_idx["n_pareto"])
        df_cmp["HC Advantage (%)"] = (
            (df_cmp["HC Max Profit ($)"] - df_cmp["NSGA-II Max Profit ($)"]) /
            df_cmp["NSGA-II Max Profit ($)"] * 100
        ).round(1)
        df_cmp["Store"] = df_cmp["Store"].str.capitalize()
        st.dataframe(df_cmp, use_container_width=True, hide_index=True)

        fig_o1_cmp = go.Figure()
        fig_o1_cmp.add_trace(go.Bar(
            name="Hill Climbing (Mono-obj)",
            x=df_cmp["Store"], y=df_cmp["HC Max Profit ($)"],
            marker_color=CHART_PALETTE["green"],
            text=df_cmp["HC Max Profit ($)"].apply(lambda v: f"${v:,.0f}"),
            textposition="outside"))
        fig_o1_cmp.add_trace(go.Bar(
            name="NSGA-II (Multi-obj)",
            x=df_cmp["Store"], y=df_cmp["NSGA-II Max Profit ($)"],
            marker_color=CHART_PALETTE["secondary"],
            text=df_cmp["NSGA-II Max Profit ($)"].apply(lambda v: f"${v:,.0f}"),
            textposition="outside"))
        fig_o1_cmp.update_layout(**PLOTLY_TEMPLATE, barmode="group",
                                  title="Maximum Weekly Profit by Algorithm and Store",
                                  xaxis_title="Store", yaxis_title="Weekly Profit ($)")
        st.plotly_chart(fig_o1_cmp, use_container_width=True, theme=None)

    # Per-store deep-dive
    st.markdown(f'<div class="section-header">Weekly Plan — {store.capitalize()}</div>', unsafe_allow_html=True)
    sub1, sub2 = st.tabs(["Hill Climbing — Optimal Plan", "NSGA-II — Pareto Frontier"])

    with sub1:
        df_plan = load_csv(f"results/03_Optimization_Report/individual/{store}_best_plan.csv")
        df_conv = load_csv(f"results/03_Optimization_Report/individual/{store}_convergence.csv")

        if df_plan is not None:
            col_pl1, col_pl2 = st.columns([3, 2])
            with col_pl1:
                disp = df_plan[["day_label", "pr", "hr_x", "hr_j", "total_staff", "customers"]].copy()
                disp.columns = ["Day", "Discount", "Expert Staff", "Junior Staff", "Total Staff", "Customers"]
                disp["Discount"] = (disp["Discount"] * 100).round(1).astype(str) + "%"
                st.dataframe(disp, use_container_width=True, hide_index=True)

                if df_hc is not None:
                    profit_hc = df_hc[df_hc["store"] == store]["lucro_maximo"].values[0]
                    st.metric("Optimal Weekly Profit (Hill Climbing)", f"${profit_hc:,.0f}")

            with col_pl2:
                if df_conv is not None:
                    fig_cv = px.line(df_conv, x="iteration", y="lucro",
                                     title="Hill Climbing Convergence",
                                     labels={"iteration": "Iteration", "lucro": "Profit ($)"},
                                     color_discrete_sequence=[CHART_PALETTE["green"]])
                    fig_cv.update_layout(**PLOTLY_TEMPLATE)
                    st.plotly_chart(fig_cv, use_container_width=True, theme=None)

            fig_plan = go.Figure()
            fig_plan.add_trace(go.Bar(name="Expert Staff", x=df_plan["day_label"], y=df_plan["hr_x"],
                                      marker_color=CHART_PALETTE["primary"]))
            fig_plan.add_trace(go.Bar(name="Junior Staff", x=df_plan["day_label"], y=df_plan["hr_j"],
                                      marker_color=CHART_PALETTE["accent"]))
            fig_plan.add_trace(go.Scatter(name="Discount (%)", x=df_plan["day_label"],
                                          y=df_plan["pr"] * 100, mode="lines+markers",
                                          yaxis="y2", line=dict(color=CHART_PALETTE["amber"], width=2.5)))
            fig_plan.update_layout(**PLOTLY_TEMPLATE, barmode="stack",
                                   title=f"Optimal Weekly Plan — {store.capitalize()}",
                                   xaxis_title="Day of Week", yaxis_title="Staff Count",
                                   yaxis2=dict(title="Discount (%)", overlaying="y", side="right", range=[0, 35]))
            st.plotly_chart(fig_plan, use_container_width=True, theme=None)

    with sub2:
        df_pf = load_csv(f"results/03_Optimization_Report/multiobjective/{store}_pareto.csv")
        if df_pf is not None:
            fig_pf = px.scatter(df_pf, x="staff", y="lucro",
                                title=f"Pareto Frontier — {store.capitalize()} (Profit vs Staff)",
                                labels={"staff": "Total Weekly Staff", "lucro": "Weekly Profit ($)"},
                                color="lucro",
                                color_continuous_scale=[[0, "#CBD5E1"], [1, CHART_PALETTE["primary"]]])
            fig_pf.update_layout(**PLOTLY_TEMPLATE)
            st.plotly_chart(fig_pf, use_container_width=True, theme=None)
            st.caption("Each point is a non-dominated solution. Management selects the preferred profit-vs-cost trade-off.")

    st.markdown("""
    <div class="verdict-box">
        <div class="verdict-title">O1 Recommendation — Per-Store Profit Maximisation</div>
        <b>Primary recommendation:</b> Hill Climbing with Random Restarts — delivers a single, interpretable optimal weekly plan with 60–130% higher profit than the NSGA-II Pareto maximum, because 100% of computational effort is directed at a single objective.<br><br>
        <b>Secondary recommendation:</b> NSGA-II — indicated when management requires explicit trade-off analysis between profit and staffing cost, generating 100 non-dominated solutions per store.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 8 — O2: JOINT ALLOCATION
# ══════════════════════════════════════════════════════════════════════════
with tab_o2:
    st.markdown('<div class="section-header">O2 — Joint Network Allocation (≤ 10,000 Units Constraint)</div>', unsafe_allow_html=True)
    st.markdown('<p class="desc">Simultaneous optimisation across all four stores subject to a hard network-wide sales ceiling of 10,000 units. Two algorithmic approaches are evaluated.</p>', unsafe_allow_html=True)

    col_b1, col_b2 = st.columns(2)
    with col_b1:
        st.markdown("""
        <div class="algo-card">
            <div class="algo-card-title">NSGA-II — Simple Joint Allocation</div>
            <div class="algo-card-sub">Multi-objective | Soft constraint via penalty function</div>
            <ul>
                <li>Optimises profit and staff simultaneously across 4 stores</li>
                <li>Unit cap enforced via penalty — not mathematically guaranteed</li>
                <li>Inter-store capacity allocation is not globally optimal</li>
                <li>Lancaster yields negative profit in this configuration</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    with col_b2:
        st.markdown("""
        <div class="winner-card">
            <div class="winner-badge">Recommended for O2</div>
            <div class="algo-card-title">NSGA-II + Dynamic Programming (Knapsack)</div>
            <div class="algo-card-sub">Hybrid | Evolutionary + Multiple Choice Knapsack Problem (MCKP)</div>
            <ul>
                <li>10,000-unit cap guaranteed as a hard constraint via DP</li>
                <li>Capacity allocation between stores is provably optimal</li>
                <li>31% higher total network profit vs simple NSGA-II</li>
                <li>All stores achieve positive profit</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    if df_alloc is not None and df_knap is not None:
        st.markdown('<div class="section-header">Network-Level Results Comparison</div>', unsafe_allow_html=True)

        tot_a = df_alloc[df_alloc["store"] == "TOTAL"].iloc[0]
        tot_k = df_knap[df_knap["store"]  == "TOTAL"].iloc[0]
        delta_p = tot_k["lucro"] - tot_a["lucro"]
        pct_p   = delta_p / abs(tot_a["lucro"]) * 100

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("NSGA-II Profit",  f"${tot_a['lucro']:,.0f}")
        c2.metric("Knapsack Profit", f"${tot_k['lucro']:,.0f}", delta=f"+${delta_p:,.0f} (+{pct_p:.0f}%)")
        c3.metric("NSGA-II Units",   f"{int(tot_a['unidades']):,}")
        c4.metric("Knapsack Units",  f"{int(tot_k['unidades']):,}",
                  delta="Hard constraint met" if tot_k["unidades"] <= 10000 else "Constraint violated")

        # Per-store profit comparison
        s_a = df_alloc[df_alloc["store"] != "TOTAL"].copy()
        s_k = df_knap[df_knap["store"]  != "TOTAL"].copy()

        fig_o2 = go.Figure()
        fig_o2.add_trace(go.Bar(
            name="NSGA-II Simple", x=s_a["store"].str.capitalize(), y=s_a["lucro"],
            marker_color=CHART_PALETTE["secondary"],
            text=s_a["lucro"].apply(lambda v: f"${v:,.0f}"), textposition="outside"))
        fig_o2.add_trace(go.Bar(
            name="Knapsack (Recommended)", x=s_k["store"].str.capitalize(), y=s_k["lucro"],
            marker_color=CHART_PALETTE["green"],
            text=s_k["lucro"].apply(lambda v: f"${v:,.0f}"), textposition="outside"))
        fig_o2.update_layout(**PLOTLY_TEMPLATE, barmode="group",
                             title="Weekly Profit per Store — NSGA-II vs Knapsack",
                             xaxis_title="Store", yaxis_title="Weekly Profit ($)")
        st.plotly_chart(fig_o2, use_container_width=True, theme=None)

        # Allocation detail table
        st.markdown('<div class="section-header">Knapsack Optimal Allocation Detail</div>', unsafe_allow_html=True)
        tbl = s_k.copy()
        tbl["store"]        = tbl["store"].str.capitalize()
        tbl["avg_discount"] = tbl["avg_discount"].round(2).astype(str) + "%"
        tbl.columns         = ["Store", "Profit ($)", "Units", "Network Share (%)", "Total Staff", "Avg Discount"]
        st.dataframe(tbl, use_container_width=True, hide_index=True)

        col_pie1, col_pie2 = st.columns(2)
        with col_pie1:
            fig_units = px.pie(s_k, values="unidades", names=s_k["store"].str.capitalize(),
                               title="Unit Allocation across Stores (Knapsack)",
                               color_discrete_sequence=[CHART_PALETTE["primary"], CHART_PALETTE["accent"],
                                                         CHART_PALETTE["green"], CHART_PALETTE["amber"]])
            fig_units.update_traces(textposition="inside", textinfo="percent+label")
            fig_units.update_layout(**PLOTLY_TEMPLATE)
            st.plotly_chart(fig_units, use_container_width=True, theme=None)

        with col_pie2:
            fig_profit_pie = px.pie(s_k, values="lucro", names=s_k["store"].str.capitalize(),
                                    title="Profit Distribution across Stores (Knapsack)",
                                    color_discrete_sequence=[CHART_PALETTE["primary"], CHART_PALETTE["accent"],
                                                              CHART_PALETTE["green"], CHART_PALETTE["amber"]])
            fig_profit_pie.update_traces(textposition="inside", textinfo="percent+label")
            fig_profit_pie.update_layout(**PLOTLY_TEMPLATE)
            st.plotly_chart(fig_profit_pie, use_container_width=True, theme=None)

    st.markdown("""
    <div class="verdict-box">
        <div class="verdict-title">O2 Recommendation — Joint Network Optimisation (≤ 10,000 Units)</div>
        <b>Recommended: NSGA-II + Dynamic Programming (Multiple Choice Knapsack Problem)</b><br><br>
        The Knapsack hybrid enforces the capacity constraint mathematically as a hard constraint via DP,
        and solves the inter-store capacity allocation problem provably optimally. This yields 31% higher
        total network profit (\$44,940 vs \$34,265) while guaranteeing all stores achieve positive returns.
        The simple NSGA-II approach treats the unit ceiling as a soft penalty, resulting in Lancaster
        producing a negative profit due to sub-optimal capacity sharing.
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 9 — O3: MULTI-OBJECTIVE GLOBAL
# ══════════════════════════════════════════════════════════════════════════
with tab_o3:
    st.markdown('<div class="section-header">O3 — Multi-Objective Global Optimisation (4 Stores, ≤ 10,000 Units)</div>', unsafe_allow_html=True)
    st.markdown('<p class="desc">Joint optimisation of two conflicting objectives across all four stores: maximise total weekly profit and minimise total staff. Constraint: ≤ 10,000 network units. MOEA/D vs U-NSGA-III.</p>', unsafe_allow_html=True)

    col_c1, col_c2 = st.columns(2)
    with col_c1:
        st.markdown("""
        <div class="algo-card">
            <div class="algo-card-title">MOEA/D — Multi-Objective Evolutionary Algorithm by Decomposition</div>
            <div class="algo-card-sub">Multi-objective | Weighted Scalarisation Decomposition</div>
            <ul>
                <li>96 Pareto solutions — good frontier density</li>
                <li>Efficient for large variable spaces (84 decision variables)</li>
                <li>Uniform weight distribution across objectives</li>
                <li>Maximum profit limited to \$17,185</li>
                <li>Minimum staff constrained to 65 — poor staff reduction</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    with col_c2:
        st.markdown("""
        <div class="winner-card">
            <div class="winner-badge">Recommended for O3</div>
            <div class="algo-card-title">U-NSGA-III — Unified Non-dominated Sorting GA III</div>
            <div class="algo-card-sub">Multi-objective | Reference-Point Based Elitism</div>
            <ul>
                <li>Maximum profit: \$37,135 — 116% higher than MOEA/D</li>
                <li>Minimum staff: 21 — superior resource reduction</li>
                <li>Reference-point mechanism improves objective space coverage</li>
                <li>64 Pareto solutions with higher decision utility</li>
            </ul>
        </div>""", unsafe_allow_html=True)

    # Quantitative metrics
    if df_o3_sum is not None:
        st.markdown('<div class="section-header">Algorithm Quality Metrics</div>', unsafe_allow_html=True)
        cols_m = st.columns(len(df_o3_sum))
        for i, (col, (_, row)) in enumerate(zip(cols_m, df_o3_sum.iterrows())):
            is_winner = "U-NSGA" in row["algorithm"]
            border = "border:1px solid #059669;" if is_winner else "border:1px solid #E2E8F0;"
            badge  = '<span style="display:inline-block;background:#059669;color:#FFF;font-size:0.65rem;font-weight:700;padding:2px 6px;border-radius:3px;margin-bottom:8px;letter-spacing:0.06em;">RECOMMENDED</span><br>' if is_winner else ""
            col.html(f"""
            <div style="padding:16px;border-radius:8px;{border}background:#FAFAFA;">
                {badge}
                <div style="font-size:0.9rem;font-weight:700;color:#0F172A;margin-bottom:12px;">{row['algorithm']}</div>
                <div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #F1F5F9;font-size:0.88rem;"><span style="color:#475569;">Pareto Solutions</span><span style="font-weight:600;color:#0F172A;">{row['n_pareto_solutions']}</span></div>
                <div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #F1F5F9;font-size:0.88rem;"><span style="color:#475569;">Max Profit (O2 point)</span><span style="font-weight:600;color:#0F172A;">${row['o2_profit']:,.0f}</span></div>
                <div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #F1F5F9;font-size:0.88rem;"><span style="color:#475569;">Staff at Max Profit</span><span style="font-weight:600;color:#0F172A;">{int(row['o2_staff'])}</span></div>
                <div style="display:flex;justify-content:space-between;padding:6px 0;border-bottom:1px solid #F1F5F9;font-size:0.88rem;"><span style="color:#475569;">Minimum Staff</span><span style="font-weight:600;color:#0F172A;">{int(row['o3_min_staff'])}</span></div>
                <div style="display:flex;justify-content:space-between;padding:6px 0;font-size:0.88rem;"><span style="color:#475569;">Profit at Min Staff</span><span style="font-weight:600;color:#0F172A;">${row['o3_min_staff_profit']:,.0f}</span></div>
            </div>""")

    # Pareto frontier comparison
    if df_p_moead is not None and df_p_unsga3 is not None:
        st.markdown('<div class="section-header">Pareto Frontier Comparison</div>', unsafe_allow_html=True)

        df_m = df_p_moead.drop_duplicates(["lucro_total", "staff_total"])
        df_u = df_p_unsga3.drop_duplicates(["lucro_total", "staff_total"])

        fig_o3 = go.Figure()
        fig_o3.add_trace(go.Scatter(
            x=df_m["staff_total"], y=df_m["lucro_total"],
            mode="markers", name="MOEA/D",
            marker=dict(color=CHART_PALETTE["secondary"], size=7, opacity=0.75, symbol="circle")))
        fig_o3.add_trace(go.Scatter(
            x=df_u["staff_total"], y=df_u["lucro_total"],
            mode="markers", name="U-NSGA-III",
            marker=dict(color=CHART_PALETTE["green"], size=7, opacity=0.75, symbol="diamond")))

        if df_o3_sum is not None:
            for _, row in df_o3_sum.iterrows():
                c = CHART_PALETTE["secondary"] if "MOEA" in row["algorithm"] else CHART_PALETTE["green"]
                fig_o3.add_trace(go.Scatter(
                    x=[row["o2_staff"]], y=[row["o2_profit"]],
                    mode="markers", name=f"O2 — {row['algorithm']}",
                    marker=dict(color=c, size=16, symbol="star",
                                line=dict(color="white", width=1.5))))

        fig_o3.update_layout(**PLOTLY_TEMPLATE,
                             title="O3 Pareto Frontier — 4 Stores Joint (Profit vs Total Staff)\nConstraint: ≤ 10,000 units | Star = O2 solution (max profit point)",
                             xaxis_title="Total Weekly Staff (all 4 stores)",
                             yaxis_title="Total Weekly Profit ($)",
                             hovermode="closest")
        st.plotly_chart(fig_o3, use_container_width=True, theme=None)

        # Diversity histograms
        col_h1, col_h2 = st.columns(2)
        with col_h1:
            fig_hm = px.histogram(df_m, x="lucro_total", nbins=20,
                                   title="MOEA/D — Profit Distribution across Pareto Solutions",
                                   labels={"lucro_total": "Total Profit ($)", "count": "Solutions"},
                                   color_discrete_sequence=[CHART_PALETTE["secondary"]])
            fig_hm.update_layout(**PLOTLY_TEMPLATE)
            st.plotly_chart(fig_hm, use_container_width=True, theme=None)
        with col_h2:
            fig_hu = px.histogram(df_u, x="lucro_total", nbins=20,
                                   title="U-NSGA-III — Profit Distribution across Pareto Solutions",
                                   labels={"lucro_total": "Total Profit ($)", "count": "Solutions"},
                                   color_discrete_sequence=[CHART_PALETTE["green"]])
            fig_hu.update_layout(**PLOTLY_TEMPLATE)
            st.plotly_chart(fig_hu, use_container_width=True, theme=None)

    # Best plan images
    img_m = os.path.join(ROOT, "results/03_Optimization_Report/joint_o3/joint_best_plan_moead.png")
    img_u = os.path.join(ROOT, "results/03_Optimization_Report/joint_o3/joint_best_plan_u-nsga-iii.png")
    if os.path.exists(img_m) or os.path.exists(img_u):
        st.markdown('<div class="section-header">Optimal Joint Plans — Visual Output</div>', unsafe_allow_html=True)
        ci1, ci2 = st.columns(2)
        if os.path.exists(img_m):
            ci1.image(img_m, caption="Best Joint Plan — MOEA/D", use_container_width=True)
        if os.path.exists(img_u):
            ci2.image(img_u, caption="Best Joint Plan — U-NSGA-III", use_container_width=True)

    st.markdown("""
    <div class="verdict-box">
        <div class="verdict-title">O3 Recommendation — Multi-Objective Joint Optimisation</div>
        <b>Recommended: U-NSGA-III</b><br><br>
        U-NSGA-III dominates MOEA/D on solution quality: 116% higher maximum profit (\$37,135 vs \$17,185),
        significantly wider staff reduction capability (minimum 21 vs 65), and broader coverage of the objective
        space. Although MOEA/D generates more Pareto solutions (96 vs 64), they are concentrated in a
        narrow low-profit region, offering limited decision utility.<br><br>
        <b>Final synthesis — Algorithm selection by objective:</b>
        O1 → Hill Climbing &nbsp;|&nbsp; O2 → NSGA-II + Knapsack &nbsp;|&nbsp; O3 → U-NSGA-III
    </div>""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════
# TAB 10 — EXECUTIVE REPORT
# ══════════════════════════════════════════════════════════════════════════
with tab_exec:
    st.markdown(f"""
    <div class="report-header">
        <div class="report-title">Executive Decision Report</div>
        <div class="report-sub">USA Retail Stores | Forecasting & Multi-Objective Optimisation | {TODAY}</div>
    </div>""", unsafe_allow_html=True)

    # ── Context ────────────────────────────────────────────────────────
    st.markdown("""
    <div class="report-section">
        <div class="report-section-title">System Context</div>
        <div style="font-size:0.88rem;color:#334155;line-height:1.8;">
        Decision Support System designed to maximise profitability and operational efficiency
        across four USA retail stores: Baltimore, Lancaster, Philadelphia, and Richmond. The system integrates
        a multi-model sales forecasting layer with a three-objective optimisation framework, enabling
        management to take data-driven staffing and pricing decisions at both individual store and network levels.
        <br><br>
        <b>Decision variables (per store, per week):</b> daily promotional discount (0–30%), expert staff allocation (0–15),
        junior staff allocation (0–15) — totalling 21 variables per store.
        </div>
    </div>""", unsafe_allow_html=True)

    # ── Forecasting summary ────────────────────────────────────────────
    st.markdown('<div class="section-header">Section 1 — Sales Forecasting</div>', unsafe_allow_html=True)

    if df_master is not None:
        best_by_store = (
            df_master[df_master["Experiment"] == "C_Context_Expert"]
            .sort_values("RMSE")
            .groupby("Store")
            .first()
            .reset_index()[["Store", "Model", "RMSE", "MAPE"]]
        )
        best_by_store["Fidelity (%)"] = (100 - best_by_store["MAPE"]).round(1)
        best_by_store["RMSE"]         = best_by_store["RMSE"].round(0)
        best_by_store["MAPE"]         = (best_by_store["MAPE"]).round(2)

        col_fs1, col_fs2 = st.columns([3, 2])
        with col_fs1:
            st.markdown("""
            <div class="report-section">
                <div class="report-section-title">Champion Model per Store — Scenario C (Context Expert)</div>""",
                unsafe_allow_html=True)
            st.dataframe(best_by_store, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

        with col_fs2:
            st.markdown("""
            <div class="report-section">
                <div class="report-section-title">Forecasting Architecture Decision</div>
                <div style="font-size:0.85rem;color:#334155;line-height:1.8;">
                <b>Recommended pipeline:</b><br>
                Scenario C (Context Expert) consistently produces the lowest error across all stores,
                integrating sales lags, promotional events, holidays, and customer context.<br><br>
                <b>Champion model:</b> Ensemble (Top-3 Experts) — automatically selects the three
                best-performing models per store, reducing variance and improving robustness.<br><br>
                <b>Baseline uplift:</b> Average +47% RMSE improvement vs Seasonal Naive across all stores.
                </div>
            </div>""", unsafe_allow_html=True)

        # RMSE by store — bar chart
        fig_fc_bar = px.bar(best_by_store, x="Store", y="RMSE",
                            title="Best Model RMSE by Store — Scenario C",
                            color="Model",
                            color_discrete_sequence=[CHART_PALETTE["primary"], CHART_PALETTE["accent"],
                                                      CHART_PALETTE["green"], CHART_PALETTE["amber"]],
                            text="RMSE")
        fig_fc_bar.update_traces(texttemplate="$%{text:,.0f}", textposition="outside")
        fig_fc_bar.update_layout(**PLOTLY_TEMPLATE,
                                  yaxis_title="RMSE ($)", xaxis_title="Store")
        st.plotly_chart(fig_fc_bar, use_container_width=True, theme=None)

    # ── Objective 1 ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Section 2 — Objective O1: Per-Store Profit Maximisation</div>', unsafe_allow_html=True)

    if df_hc is not None and df_nsga2 is not None:
        col_o1a, col_o1b = st.columns([2, 3])
        with col_o1a:
            st.markdown("""
            <div class="report-section">
                <div class="report-section-title">Selected Algorithm</div>
                <div style="font-size:0.88rem;color:#334155;line-height:1.8;">
                <b>Hill Climbing with Random Restarts</b><br>
                Mono-objective local search | 10 restarts × 2,000 iterations<br><br>
                <b>Rationale:</b> Concentrates 100% of search effort on maximising weekly profit.
                Outperforms NSGA-II by 60–130% per store because the multi-objective algorithm
                allocates effort across the profit-staff trade-off space rather than converging
                on a single profit maximum.<br><br>
                <b>When to use NSGA-II instead:</b> When management needs explicit trade-off
                analysis between profit and staffing cost (Pareto frontier with 100 solutions per store).
                </div>
            </div>""", unsafe_allow_html=True)

        with col_o1b:
            df_o1_tbl = df_hc.copy()
            df_o1_tbl["store"] = df_o1_tbl["store"].str.capitalize()
            df_o1_tbl["HC Profit ($)"]       = df_o1_tbl["lucro_maximo"]
            df_o1_tbl["NSGA-II Profit ($)"]  = df_o1_tbl["store"].str.lower().map(
                df_nsga2.set_index("store")["max_profit"])
            df_o1_tbl["Uplift vs NSGA-II (%)"] = (
                (df_o1_tbl["lucro_maximo"] - df_o1_tbl["NSGA-II Profit ($)"]) /
                df_o1_tbl["NSGA-II Profit ($)"] * 100).round(1)
            df_o1_tbl["Avg Discount (%)"]    = df_o1_tbl["avg_discount"]
            df_o1_tbl["Weekly Staff"]        = df_o1_tbl["total_staff"]
            display_cols = ["store", "HC Profit ($)", "NSGA-II Profit ($)", "Uplift vs NSGA-II (%)", "Avg Discount (%)", "Weekly Staff"]
            df_o1_tbl = df_o1_tbl[display_cols].rename(columns={"store": "Store"})
            st.markdown('<div class="report-section"><div class="report-section-title">O1 Results — Hill Climbing vs NSGA-II</div>', unsafe_allow_html=True)
            st.dataframe(df_o1_tbl, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

        fig_o1_exec = go.Figure()
        fig_o1_exec.add_trace(go.Bar(
            name="Hill Climbing (Selected)", x=df_hc["store"].str.capitalize(), y=df_hc["lucro_maximo"],
            marker_color=CHART_PALETTE["green"],
            text=df_hc["lucro_maximo"].apply(lambda v: f"${v:,.0f}"), textposition="outside"))
        fig_o1_exec.add_trace(go.Bar(
            name="NSGA-II", x=df_nsga2["store"].str.capitalize(), y=df_nsga2["max_profit"],
            marker_color=CHART_PALETTE["secondary"],
            text=df_nsga2["max_profit"].apply(lambda v: f"${v:,.0f}"), textposition="outside"))
        fig_o1_exec.update_layout(**PLOTLY_TEMPLATE, barmode="group",
                                   title="O1 — Maximum Weekly Profit per Store",
                                   xaxis_title="Store", yaxis_title="Weekly Profit ($)")
        st.plotly_chart(fig_o1_exec, use_container_width=True, theme=None)

    # ── Objective 2 ────────────────────────────────────────────────────
    st.markdown('<div class="section-header">Section 3 — Objective O2: Joint Network Allocation (≤ 10,000 Units)</div>', unsafe_allow_html=True)

    if df_alloc is not None and df_knap is not None:
        tot_a = df_alloc[df_alloc["store"] == "TOTAL"].iloc[0]
        tot_k = df_knap[df_knap["store"]  == "TOTAL"].iloc[0]

        col_o2a, col_o2b = st.columns([2, 3])
        with col_o2a:
            st.markdown(f"""
            <div class="report-section">
                <div class="report-section-title">Selected Algorithm</div>
                <div style="font-size:0.88rem;color:#334155;line-height:1.8;">
                <b>NSGA-II + Dynamic Programming (MCKP)</b><br>
                Hybrid: evolutionary search + exact DP solver<br><br>
                <b>Rationale:</b> The 10,000-unit ceiling is enforced as a mathematically guaranteed
                hard constraint through the Multiple Choice Knapsack formulation. Dynamic Programming
                solves the inter-store capacity distribution problem exactly, achieving a globally
                optimal allocation.<br><br>
                <b>Result:</b> +31% network profit (\${tot_k['lucro']:,.0f} vs \${tot_a['lucro']:,.0f})
                while keeping all stores profitable and honouring the unit cap.
                </div>
            </div>""", unsafe_allow_html=True)

        with col_o2b:
            s_k = df_knap[df_knap["store"] != "TOTAL"].copy()
            s_a = df_alloc[df_alloc["store"] != "TOTAL"].copy()
            merged = s_k[["store", "lucro", "unidades", "total_staff"]].copy()
            merged.columns = ["Store", "Knapsack Profit ($)", "Units", "Staff"]
            merged["NSGA-II Profit ($)"] = merged["Store"].map(s_a.set_index("store")["lucro"])
            merged["Store"] = merged["Store"].str.capitalize()
            st.markdown('<div class="report-section"><div class="report-section-title">O2 Results — Knapsack vs NSGA-II Simple</div>', unsafe_allow_html=True)
            st.dataframe(merged, use_container_width=True, hide_index=True)
            st.metric("Knapsack Network Total", f"${tot_k['lucro']:,.0f}",
                      delta=f"+${tot_k['lucro']-tot_a['lucro']:,.0f} vs NSGA-II")
            st.markdown("</div>", unsafe_allow_html=True)

        fig_o2_exec = go.Figure()
        fig_o2_exec.add_trace(go.Bar(
            name="Knapsack (Selected)", x=s_k["store"].str.capitalize(), y=s_k["lucro"],
            marker_color=CHART_PALETTE["green"],
            text=s_k["lucro"].apply(lambda v: f"${v:,.0f}"), textposition="outside"))
        fig_o2_exec.add_trace(go.Bar(
            name="NSGA-II Simple", x=s_a["store"].str.capitalize(), y=s_a["lucro"],
            marker_color=CHART_PALETTE["secondary"],
            text=s_a["lucro"].apply(lambda v: f"${v:,.0f}"), textposition="outside"))
        fig_o2_exec.update_layout(**PLOTLY_TEMPLATE, barmode="group",
                                   title="O2 — Network Weekly Profit by Store",
                                   xaxis_title="Store", yaxis_title="Weekly Profit ($)")
        st.plotly_chart(fig_o2_exec, use_container_width=True, theme=None)

    # ── Objective 3 ────────────────────────────────────════════════════
    st.markdown('<div class="section-header">Section 4 — Objective O3: Multi-Objective Global (Profit vs Staff)</div>', unsafe_allow_html=True)

    if df_o3_sum is not None and df_p_moead is not None and df_p_unsga3 is not None:
        row_m = df_o3_sum[df_o3_sum["algorithm"].str.contains("MOEA")].iloc[0]
        row_u = df_o3_sum[df_o3_sum["algorithm"].str.contains("NSGA")].iloc[0]

        col_o3a, col_o3b = st.columns([2, 3])
        with col_o3a:
            st.markdown(f"""
            <div class="report-section">
                <div class="report-section-title">Selected Algorithm</div>
                <div style="font-size:0.88rem;color:#334155;line-height:1.8;">
                <b>U-NSGA-III</b><br>
                Reference-point based elitism | Unified selection<br><br>
                <b>Rationale:</b> Delivers 116% higher maximum profit than MOEA/D (\$37,135 vs \$17,185)
                and achieves a minimum weekly staff of 21 versus MOEA/D's 65,
                demonstrating superior objective space exploration.<br><br>
                MOEA/D generates 96 Pareto solutions but they cluster in a narrow
                low-profit band (\${row_m['o3_min_staff_profit']:,.0f} to \${row_m['o2_profit']:,.0f}),
                offering limited management utility.<br><br>
                U-NSGA-III's reference-point mechanism provides broader coverage
                and higher-quality solutions across the full profit-staff trade-off range.
                </div>
            </div>""", unsafe_allow_html=True)

        with col_o3b:
            cmp_tbl = pd.DataFrame({
                "Metric": ["Algorithm type", "Pareto solutions", "Max profit (O2 point)",
                           "Staff at max profit", "Min staff achievable",
                           "Profit at min staff", "Recommendation"],
                "MOEA/D": [
                    "Decomposition", str(int(row_m["n_pareto_solutions"])),
                    f"${row_m['o2_profit']:,.0f}", str(int(row_m["o2_staff"])),
                    str(int(row_m["o3_min_staff"])), f"${row_m['o3_min_staff_profit']:,.0f}", "No"],
                "U-NSGA-III": [
                    "Reference-point", str(int(row_u["n_pareto_solutions"])),
                    f"${row_u['o2_profit']:,.0f}", str(int(row_u["o2_staff"])),
                    str(int(row_u["o3_min_staff"])), f"${row_u['o3_min_staff_profit']:,.0f}", "Yes"],
            })
            st.markdown('<div class="report-section"><div class="report-section-title">O3 Algorithm Comparison</div>', unsafe_allow_html=True)
            st.dataframe(cmp_tbl, use_container_width=True, hide_index=True)
            st.markdown("</div>", unsafe_allow_html=True)

        # Pareto frontier executive view
        df_m_e = df_p_moead.drop_duplicates(["lucro_total", "staff_total"])
        df_u_e = df_p_unsga3.drop_duplicates(["lucro_total", "staff_total"])

        fig_o3_exec = go.Figure()
        fig_o3_exec.add_trace(go.Scatter(
            x=df_m_e["staff_total"], y=df_m_e["lucro_total"],
            mode="markers", name="MOEA/D",
            marker=dict(color=CHART_PALETTE["secondary"], size=6, opacity=0.65)))
        fig_o3_exec.add_trace(go.Scatter(
            x=df_u_e["staff_total"], y=df_u_e["lucro_total"],
            mode="markers", name="U-NSGA-III (Selected)",
            marker=dict(color=CHART_PALETTE["green"], size=6, opacity=0.75)))
        fig_o3_exec.add_trace(go.Scatter(
            x=[row_u["o2_staff"]], y=[row_u["o2_profit"]],
            mode="markers", name="Selected operating point (U-NSGA-III O2)",
            marker=dict(color=CHART_PALETTE["green"], size=16, symbol="star",
                        line=dict(color="white", width=1.5))))
        fig_o3_exec.update_layout(**PLOTLY_TEMPLATE,
                                   title="O3 Pareto Frontier — Profit vs Staff (4 Stores Combined)",
                                   xaxis_title="Total Weekly Staff", yaxis_title="Total Weekly Profit ($)",
                                   hovermode="closest")
        st.plotly_chart(fig_o3_exec, use_container_width=True, theme=None)

    # ── Final synthesis ────────────────────────────────────────────────
    st.markdown('<div class="section-header">Section 5 — Strategic Synthesis and Recommendations</div>', unsafe_allow_html=True)

    # Summary table
    if df_hc is not None and df_knap is not None and df_o3_sum is not None:
        tot_hc     = df_hc["lucro_maximo"].sum()
        tot_k_val  = df_knap[df_knap["store"] == "TOTAL"]["lucro"].values[0]
        row_u_fin  = df_o3_sum[df_o3_sum["algorithm"].str.contains("NSGA")].iloc[0]

        summary = pd.DataFrame({
            "Objective": ["O1 — Per-Store Max Profit", "O2 — Joint Network (≤10k units)", "O3 — Multi-Objective Joint"],
            "Selected Algorithm": ["Hill Climbing (Random Restarts)", "NSGA-II + Dynamic Programming (Knapsack)", "U-NSGA-III"],
            "Type": ["Mono-objective | Local Search", "Hybrid | Evolutionary + Exact DP", "Multi-objective | Reference-Point EA"],
            "Key Result": [
                f"${tot_hc:,.0f} total weekly profit across 4 stores",
                f"${tot_k_val:,.0f} network profit | All stores positive | Hard constraint met",
                f"${row_u_fin['o2_profit']:,.0f} max profit | {int(row_u_fin['o3_min_staff'])} min staff | 64 Pareto solutions",
            ],
            "Decision Rationale": [
                "100% search focus on profit maximisation; 60–130% better than NSGA-II",
                "+31% profit vs simple NSGA-II; mathematically guaranteed capacity constraint",
                "+116% max profit vs MOEA/D; wider objective space coverage",
            ]
        })
        st.dataframe(summary, use_container_width=True, hide_index=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown(f"""
    <div class="report-section">
        <div class="report-section-title">Executive Conclusions</div>
        <div style="font-size:0.88rem;color:#334155;line-height:1.9;">

<b>Forecasting layer:</b> Scenario C (Context Expert) with Ensemble (Top-3) delivers the highest
predictive fidelity across all stores. The ensemble architecture automatically adapts the model
composition per store, making it robust to local demand characteristics. Average fidelity: ~90%
(1 − MAPE). The Poisson arrival model supplements the ensemble with probabilistic customer
flow estimates for operational planning.<br><br>

<b>O1 — Per-store staffing and pricing:</b> Hill Climbing with Random Restarts is the correct tool
for tactical weekly planning. It produces a single, immediately actionable plan per store
(discount rate and staff mix by day) that maximises weekly profit without requiring management
to interpret a Pareto frontier. Philadelphia has the highest profit potential (\$257,775/week)
driven by customer volume; Richmond has the highest optimal discount rate (21.3%),
suggesting a price-elastic customer base.<br><br>

<b>O2 — Network capacity allocation:</b> The Knapsack hybrid ensures that the 10,000-unit
network ceiling is respected mathematically while maximising aggregate profit. Richmond captures
36.5% of the network capacity due to its superior profit-per-unit efficiency. Lancaster's
negative result under simple NSGA-II (Lancaster loses \$4,445) is corrected to positive
(\$8,330) through optimal capacity reallocation in the Knapsack solution.<br><br>

<b>O3 — Strategic trade-off analysis:</b> U-NSGA-III provides management with a decision frontier
ranging from maximum profit (\$37,135 with 85 total staff) to minimum staff (21 total, accepting
negative profit). The recommended operating point is the O2 solution on this frontier —
\$37,135 weekly profit at 85 total staff. The Pareto frontier enables scenario planning:
a 10-unit reduction in weekly staff costs approximately \$5,000–8,000 in weekly profit,
quantifying the operational efficiency trade-off.<br><br>

<b>System integration recommendation:</b> Deploy the forecasting ensemble (Scenario C) to
generate weekly demand inputs → feed into Hill Climbing for per-store tactical plans (O1) →
run Knapsack allocation to enforce the network capacity constraint (O2) → use the U-NSGA-III
Pareto frontier quarterly for strategic staffing level reviews (O3).

        </div>
    </div>
    <div style="font-size:0.78rem;color:#94A3B8;text-align:right;margin-top:12px;">
        DSS &nbsp;|&nbsp; Generated: {TODAY} &nbsp;|&nbsp; All monetary values in USD
    </div>
    """, unsafe_allow_html=True)
