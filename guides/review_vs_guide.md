# TIAPOSE — Full Project Review vs. guide.md

> Maps the current codebase state against every requirement in guide.md.  
> Goal: identify what is done, what is partially done, and what is missing before the 2026-05-27 deadline.

---

## COMPONENT A — FORECASTING

### What we HAVE

| Item | Status | Where |
|---|---|---|
| Seasonal Naive baseline (S=7) | ✅ Done | `src/forecasting/baseline.py` |
| Holt-Winters / ETS | ✅ Done | `src/forecasting/trainer.py` |
| SARIMAX (with exogenous) | ✅ Done | `src/forecasting/trainer.py` |
| Prophet (yearly + weekly) | ✅ Done | `src/forecasting/trainer.py` |
| Random Forest (ML) | ✅ Done | `src/forecasting/trainer.py` |
| Linear Regression (ML baseline) | ✅ Done | `src/forecasting/trainer.py` |
| Ensemble Top-3 (auto-select) | ✅ Done | `src/forecasting/trainer.py` |
| LightGBM (experimental) | ✅ Done | `src/model_testing/test_lightgbm.py` (not in main pipeline) |
| XGBoost (experimental) | ✅ Done | `src/model_testing/xgboost/xgboost_model.py` (isolated) |
| Three feature scenarios (A, B, C) | ✅ Done | `main_pipeline.py` |
| Phase I evaluation (last week test) | ✅ Done | Per-store `store_metrics.csv` |
| Summary table (model × store × metric) | ✅ Done | `results/00_Master_Summary/` |
| % improvement vs Seasonal Naive | ✅ Done | Dev dashboard Tab 6 |
| Forecasts saved to CSV | ✅ Done | `results/02_Forecasting_Report/*/forecast_values.csv` |
| EDA (ACF/PACF, decomposition, scatter) | ✅ Done | `results/01_EDA_Gallery/` |
| Correlation analysis | ✅ Done | `src/data/statistical_analysis.py` |

### What is MISSING or INCOMPLETE

| Item | Status | Notes |
|---|---|---|
| **Backtesting ≥10 iterations** | ❌ MISSING | Notebooks exist (`03_backtesting.ipynb`) but NOT integrated into the pipeline. Explicit requirement for "boa nota" |
| Growing vs Rolling Window (justified) | ❌ MISSING | Required alongside backtesting |
| Median/mean aggregation of backtest errors | ❌ MISSING | Required methodology |
| **TBATS** (multi-seasonal) | ❌ MISSING | Listed as desirable univariate method |
| **VAR** (Vector Autoregression) | ❌ MISSING | Listed as desirable multivariate method |
| Cross-store features (Scenario D) | ❌ MISSING | Variables from other stores as predictors — not implemented |
| Best univariate vs best multivariate split | ⚠️ Partial | Dashboard mentions best model but doesn't explicitly split by family |
| LightGBM integrated into main pipeline | ⚠️ Partial | Exists in model_testing but not in `main_pipeline.py` |
| Backtesting CSV of forecasted values (≥10 weeks) | ❌ MISSING | Required to feed Optimization Phase II |

---

## COMPONENT B — OPTIMIZATION

### What we HAVE

| Item | Status | Where |
|---|---|---|
| O1 — Hill Climbing per store | ✅ Done | `src/optimization/hill_climbing.py` |
| O1 — NSGA-II Pareto per store | ✅ Done | `src/optimization/nsga2_model.py` |
| O2 — NSGA-II joint (84 vars, soft constraint) | ✅ Done | `src/optimization/joint_problem.py` + `scripts/run_allocation_optimization.py` |
| O2 — Knapsack DP (hard constraint, recommended) | ✅ Done | `src/optimization/knapsack_solver.py` + `scripts/run_allocation_optimization_knapsack.py` |
| O3 — MOEA/D joint multi-objective | ✅ Done | `src/optimization/moead_model.py` + `scripts/run_joint_optimization_o3.py` |
| O3 — U-NSGA-III joint multi-objective | ✅ Done | `src/optimization/unsga3_model.py` + `scripts/run_joint_optimization_o3.py` |
| Pareto frontier plots for O3 | ✅ Done | `results/03_Optimization_Report/joint_o3/` |
| Convergence plots for O1 | ✅ Done | `results/03_Optimization_Report/individual/` |
| Weekly plan output (day, PR, X, J, profit) | ✅ Done | Per-store `*_best_plan.csv` |
| Probabilistic demand models (Poisson, Gaussian, Logistic) | ✅ Done | `src/optimization/probabilistic_models.py` |

### What is MISSING or INCOMPLETE

| Item | Status | Notes |
|---|---|---|
| **Monte Carlo (blind search) baseline** | ❌ MISSING | guide.md §4.9 checklist: "≥4 methods including Monte Carlo as baseline" |
| **Simulated Annealing** | ❌ MISSING | Explicitly listed in §4.5 as a required method |
| Differential Evolution | ❌ MISSING | Listed in §4.5 as a method to explore |
| Particle Swarm Optimization (PSO) | ❌ MISSING | Listed in §4.5 as a method to explore |
| **Validation with guide examples** (Baltimore=146, Philadelphia=1728) | ❓ UNKNOWN | §4.3 says this is critical; not confirmed in code or test scripts |
| **Optimization Phase II** (≥10 runs using forecasted values) | ❌ MISSING | Currently only Phase I (real values) is run |
| Comparison: real values vs. forecasts (same table) | ❌ MISSING | Guide requires a direct side-by-side comparison |
| **Hyperparameter sensitivity study** | ❌ MISSING | §4.6 and §9.4: vary popSize, temp, generations; convergence graphs per config |
| Weighted scalarization for O3 | ❌ MISSING | §4.7 asks for at least one scalarization alternative alongside Pareto |
| Death penalty / repair strategy documented | ⚠️ Partial | Knapsack enforces hard constraint via DP; penalty approach not explicit |
| `round()` applied to final solution post-optimization | ⚠️ Partial | Repair operator rounds in NSGA-II; not confirmed in all runners |

---

## COMPONENT C — DSS INTERFACE

### What we HAVE

| Item | Status | Where |
|---|---|---|
| Full dev dashboard (all tabs, all models) | ✅ Done | `dss_app/dev_dashboard.py` (1,392 lines) |
| Stakeholder demo shell (layout/branding) | ⚠️ Partial | `dss_app/app.py` (283 lines, landing page only) |
| Tab: Sales Forecast (actual vs models) | ✅ Done | dev_dashboard Tab 1 |
| Tab: Residual analysis | ✅ Done | dev_dashboard Tab 2 |
| Tab: Trend Decomposition | ✅ Done | dev_dashboard Tab 3 |
| Tab: Feature Importance (XAI) | ✅ Done | dev_dashboard Tab 4 |
| Tab: Probabilistic models | ✅ Done | dev_dashboard Tab 5 |
| Tab: Model Leaderboard | ✅ Done | dev_dashboard Tab 6 |
| Tab: O1 optimization (HC vs NSGA-II) | ✅ Done | dev_dashboard Tab 7 |
| Tab: O2 joint allocation (NSGA-II vs Knapsack) | ✅ Done | dev_dashboard Tab 8 |
| Tab: O3 multi-objective (MOEA/D vs U-NSGA-III) | ✅ Done | dev_dashboard Tab 9 |
| Tab: Executive Report | ✅ Done | dev_dashboard Tab 10 |
| Store selector (all 4 stores) | ✅ Done | Sidebar |
| Scenario selector (A / B / C) | ✅ Done | Sidebar |

### What is MISSING or INCOMPLETE

| Item | Status | Notes |
|---|---|---|
| **Week selection** by user in the DSS | ❌ MISSING | Core DSS requirement: user picks a week, system shows forecast + plan |
| **Functional stakeholder demo (`app.py`)** | ❌ MISSING | Currently a landing page; all actual DSS features are only in dev_dashboard |
| O2 constraint violation indicator (10,000 units) | ❌ MISSING | Visual red/green badge for the network capacity constraint |
| **Export plan to CSV** button | ❌ MISSING | Required by guide §5.2 |
| Uncertainty band on forecast chart | ❌ MISSING | Gaussian model provides CI but not shown on main forecast chart |
| Objective selector (O1 / O2 / O3) in UI | ⚠️ Partial | Fixed tabs; no single dropdown for objective selection |
| **Video demo (≤5 min, voice narration)** | ❌ MISSING | Must be uploaded to YouTube, link in report |

---

## REPORT & SUBMISSION

| Item | Status | Notes |
|---|---|---|
| Report PDF (20–40 pages) | ❌ MISSING | Not found in repo |
| Project self-assessment (nota A, justification) | ❌ MISSING | Part of report |
| Individual self-differentiation (½ page/member) | ❌ MISSING | Part of report |
| AI usage declaration | ❌ MISSING | Required in annexes |
| ZIP submission package | ❌ MISSING | Not yet assembled |

---

## PRIORITY GAPS — RANKED BY GRADE IMPACT

### CRITICAL (significant points at risk)

1. **Backtesting ≥10 iterations** — explicitly required for "boa nota" (guide §3.2 Phase II).
   - Create `scripts/run_backtesting.py` from notebook logic.
   - Output: CSV of backtested forecasts per store, ≥10 weeks, aggregated by median/mean.

2. **Monte Carlo baseline for optimization** — guide §4.9 checklist item.
   - Create `src/optimization/monte_carlo.py` + runner script.

3. **Optimization Phase II with forecasted values** — currently Phase I only (real values).
   - Run O1/O2/O3 for each of the ≥10 backtested forecast weeks; report median profit.
   - Create `scripts/run_optimization_phaseII.py`.

4. **Evaluation function validation** — validate `evaluation()` against Baltimore=146, Philadelphia=1728 (guide §4.3).
   - Add a short validation script or notebook cell before the presentation.

5. **Simulated Annealing** — each group member must implement a distinct method (guide §4.5).
   - Create `src/optimization/simulated_annealing.py`.

6. **Functional stakeholder DSS (`app.py`)** — current version is a branding page only.
   - Add week selection, forecast display, optimized plan, export CSV button.

7. **Video demo** — mandatory deliverable, cannot be code-generated. Must be recorded and uploaded.

### HIGH PRIORITY (loses points, feasible to fix)

8. **Hyperparameter sensitivity study** — vary popSize, temp, generations; convergence graphs per config for ≥2 methods.

9. **Comparison table: real values vs. forecasts** — run optimization in both Phase I and Phase II, display side-by-side.

10. **Weighted scalarization for O3** — quick weighted sum alternative (w=0.7) to complement the Pareto approach.

### NICE-TO-HAVE (valorizados but not blocking)

11. TBATS model (multi-seasonal univariate)
12. VAR model (multivariate time series)
13. Cross-store feature scenario (Scenario D)
14. LightGBM integrated into `main_pipeline.py`
15. Export to CSV button in dashboard
16. O2 constraint badge (visual indicator in DSS)

---

## VERIFICATION CHECKLIST (pre-presentation)

Cross-check against guide.md §9 checklists before submitting:

- [ ] §9.1 — Os 3 objetivos de otimização foram resolvidos (O1, O2, O3)?
- [ ] §9.1 — Forecasting com baseline + univariado + multivariado?
- [ ] §9.1 — Cada membro implementou 1 método de previsão e 1 de otimização distintos?
- [ ] §9.1 — DSS funcional ligando previsão e otimização?
- [ ] §9.2 — Justificação do melhor modelo de forecasting (métrica, robustez, % melhoria)?
- [ ] §9.2 — Justificação do melhor método de otimização por objetivo?
- [ ] §9.2 — Pesos/normalização de O3 justificados ou Pareto apresentado?
- [ ] §9.3 — Backtesting com ≥10 iterações?
- [ ] §9.3 — Otimização com ≥10 runs (uma por semana de teste)?
- [ ] §9.3 — Métricas agregadas com mediana/média (escolha justificada)?
- [ ] §9.3 — Mesmas condições de comparação para todos os métodos?
- [ ] §9.4 — Estudo de hiperparâmetros (temp SANN, popSize GA)?
- [ ] §9.5 — Vídeo ≤5 min, com narração, link no relatório?
- [ ] §9.5 — ZIP final completo (PDF + código + CSVs)?
