"""
probabilistic_models.py — Modelos de Regras Probabilisticos — TIAPOSE DSS
Implementa tres modelos estatisticos baseados em regras de comportamento:
  1. PoissonArrivalModel  — previsao de chegada de clientes (GLM-Poisson)
  2. GaussianSalesModel   — variabilidade das vendas (OLS + cenarios)
  3. LogisticConversionModel — probabilidade de conversao via desconto
"""

import logging
import os
import sys
import warnings

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
log = logging.getLogger("TIAPOSE.ProbModels")


# ===========================================================================
# 1. MODELO DE POISSON — Previsao de Chegada de Clientes
# ===========================================================================

class PoissonArrivalModel:

    # Features usadas (colunas do dataset processado)
    FEATURES = [
        "day_of_week", "is_weekend", "is_holiday",
        "days_to_next_holiday",
        "month", "TouristEvent", "Pct_On_Sale",
    ]

    def __init__(self):
        self._result = None
        self._features = []
        self._fitted = False

    def fit(self, df, store=""):
        # Ajusta o GLM-Poisson aos dados historicos
        self._features = [f for f in self.FEATURES if f in df.columns]
        X = sm.add_constant(df[self._features].astype(float), has_constant="add")
        y = df["Num_Customers"].astype(float)
        glm = sm.GLM(y, X, family=sm.families.Poisson(link=sm.families.links.log()))
        self._result = glm.fit(disp=False)
        self._fitted = True

        y_pred = self._result.predict(X)
        mae  = mean_absolute_error(y, y_pred)
        rmse = np.sqrt(mean_squared_error(y, y_pred))
        log.info("[Poisson][%s] AIC=%.1f | MAE=%.1f clientes | RMSE=%.1f",
                 store, self._result.aic, mae, rmse)
        return self

    def predict_lambda(self, X_future):
        # Prevê lambda (nr esperado de clientes) para cada linha de X_future
        if not self._fitted:
            raise RuntimeError("Modelo nao treinado. Chama .fit() primeiro.")
        X = sm.add_constant(X_future[self._features].astype(float), has_constant="add")
        return self._result.predict(X).values

    def predict_week(self, is_weekend, month=5, tourist_event=0,
                     pct_on_sale=10.0, is_holiday=None, days_to_next_holiday=None):
        # Atalho: preve lambda para uma semana de 7 dias
        # days_to_next_holiday: None -> [7,6,5,4,3,2,1] (valor neutro descendente)
        if is_holiday is None:
            is_holiday = [0] * 7
        if days_to_next_holiday is None:
            days_to_next_holiday = [7, 6, 5, 4, 3, 2, 1]
        rows = []
        for d in range(7):
            rows.append({
                "day_of_week":          d,
                "is_weekend":           int(is_weekend[d]),
                "is_holiday":           int(is_holiday[d]),
                "days_to_next_holiday": int(days_to_next_holiday[d]),
                "month":                month,
                "TouristEvent":         tourist_event,
                "Pct_On_Sale":          pct_on_sale,
            })
        return self.predict_lambda(pd.DataFrame(rows))

    def summary(self):
        return str(self._result.summary()) if self._fitted else "Modelo nao treinado."


# ===========================================================================
# 2. MODELO GAUSSIANO — Variabilidade das Vendas
# ===========================================================================

class GaussianSalesModel:

    # Features usadas (a coluna alvo e 'Sales' ou 'y')
    FEATURES = [
        "Num_Customers", "Pct_On_Sale", "TouristEvent",
        "is_holiday", "is_weekend", "day_of_week", "month",
        "sales_lag_7",
    ]

    def __init__(self):
        self._model  = LinearRegression()
        self._scaler = StandardScaler()
        self._features = []
        self._sigma = 0.0
        self._mu_global = 0.0
        self._fitted = False

    def fit(self, df, store=""):
        # Treina OLS nos primeiros 80% e estima sigma pelos residuos de teste (20%)
        target = "y" if "y" in df.columns else "Sales"
        self._features = [f for f in self.FEATURES if f in df.columns]
        X = df[self._features].astype(float).fillna(0)
        y = df[target].astype(float)
        self._mu_global = float(y.mean())

        # Split temporal 80/20 (sem aleatoriedade — respeitar ordem temporal)
        split = int(len(df) * 0.8)
        X_train, X_test = X.iloc[:split], X.iloc[split:]
        y_train, y_test = y.iloc[:split], y.iloc[split:]

        X_train_sc = self._scaler.fit_transform(X_train)
        X_test_sc  = self._scaler.transform(X_test)
        self._model.fit(X_train_sc, y_train)

        y_pred_test  = self._model.predict(X_test_sc)
        residuals    = y_test.values - y_pred_test
        self._sigma  = float(np.std(residuals))   # incerteza preditiva real
        self._fitted = True

        mae  = mean_absolute_error(y_test, y_pred_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
        cv   = self._sigma / max(self._mu_global, 1) * 100
        log.info("[Gaussiano][%s] mu=%.0f | sigma=%.0f (CV=%.1f%%) | MAE=%.0f | RMSE=%.0f",
                 store, self._mu_global, self._sigma, cv, mae, rmse)
        return self

    def predict_scenarios(self, X_future, sigma_override=None):
        # Gera 3 cenarios: pessimista (mu-sigma), realista (mu), otimista (mu+sigma)
        if not self._fitted:
            raise RuntimeError("Modelo nao treinado. Chama .fit() primeiro.")
        feats = [f for f in self._features if f in X_future.columns]
        X_sc  = self._scaler.transform(X_future[feats].astype(float).fillna(0))
        mu    = self._model.predict(X_sc)
        sigma = sigma_override if sigma_override is not None else self._sigma
        return {
            "pessimistic": np.maximum(0, mu - sigma),
            "realistic":   mu,
            "optimistic":  mu + sigma,
        }

    def confidence_interval(self, X_future, confidence=0.95):
        # Intervalo de confiança simetrico para as previsoes
        if not self._fitted:
            raise RuntimeError("Modelo nao treinado. Chama .fit() primeiro.")
        z    = stats.norm.ppf(1 - (1 - confidence) / 2)
        feats = [f for f in self._features if f in X_future.columns]
        X_sc  = self._scaler.transform(X_future[feats].astype(float).fillna(0))
        mu    = self._model.predict(X_sc)
        return (np.maximum(0, mu - z * self._sigma), mu + z * self._sigma)

    @property
    def sigma(self):
        return self._sigma

    @property
    def mu_global(self):
        return self._mu_global


# ===========================================================================
# 3. MODELO LOGISTICO — Probabilidade de Conversao via Desconto
# ===========================================================================

class LogisticConversionModel:

    # Features usadas (Pct_On_Sale e a variavel principal de interesse)
    FEATURES = [
        "Pct_On_Sale", "is_weekend", "is_holiday",
        "days_to_next_holiday",
        "TouristEvent", "month",
    ]

    def __init__(self, random_state=42):
        self._model  = LogisticRegression(
            max_iter=1000, random_state=random_state,
            class_weight="balanced", C=1.0,
        )
        self._scaler   = StandardScaler()
        self._features = []
        self._threshold = 0.0
        self._auc = 0.0
        self._fitted = False

    def fit(self, df, store=""):
        # Variavel alvo: vendas/cliente >= mediana historica? (1=converteu, 0=nao)
        sales_per_customer = df["Sales"] / df["Num_Customers"].replace(0, np.nan)
        self._threshold = float(sales_per_customer.median())
        y = (sales_per_customer >= self._threshold).astype(int)

        self._features = [f for f in self.FEATURES if f in df.columns]
        X = df[self._features].astype(float).fillna(0)

        # Split estratificado 80/20
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train_sc = self._scaler.fit_transform(X_train)
        X_test_sc  = self._scaler.transform(X_test)
        self._model.fit(X_train_sc, y_train)
        self._fitted = True

        y_prob    = self._model.predict_proba(X_test_sc)[:, 1]
        self._auc = roc_auc_score(y_test, y_prob)

        coef_str = " | ".join(f"{f}={c:+.3f}" for f, c in
                               zip(self._features, self._model.coef_[0]))
        log.info("[Logistico][%s] AUC=%.3f | threshold=%.1f vendas/cliente",
                 store, self._auc, self._threshold)
        log.info("[Logistico][%s] Coeficientes: %s", store, coef_str)
        return self

    def predict_conversion_prob(self, X_future):
        # Estima P(conversao=1) para cada linha de X_future
        if not self._fitted:
            raise RuntimeError("Modelo nao treinado. Chama .fit() primeiro.")
        X = X_future[self._features].astype(float).fillna(0)
        return self._model.predict_proba(self._scaler.transform(X))[:, 1]

    def conversion_curve(self, pr_values, is_weekend=0, is_holiday=0,
                         days_to_next_holiday=7, tourist_event=0, month=6):
        # Curva P(conversao) vs desconto, mantendo as outras variaveis fixas
        rows = [
            {
                "Pct_On_Sale":          pr * 100,
                "is_weekend":           is_weekend,
                "is_holiday":           is_holiday,
                "days_to_next_holiday": days_to_next_holiday,
                "TouristEvent":         tourist_event,
                "month":                month,
            }
            for pr in pr_values
        ]
        return self.predict_conversion_prob(pd.DataFrame(rows))

    @property
    def auc(self):
        return self._auc

    @property
    def threshold(self):
        return self._threshold


# ===========================================================================
# COMPARACAO — Poisson vs. Lag-7 historico
# ===========================================================================

def compare_with_forecasting(df, poisson_model, store=""):
    # Compara o Poisson com o baseline lag-7 nos ultimos 20% dos dados
    split   = int(len(df) * 0.8)
    test_df = df.iloc[split:].copy()
    y_true  = test_df["Num_Customers"].values

    X_test   = test_df[[f for f in PoissonArrivalModel.FEATURES if f in test_df.columns]]
    y_poisson = poisson_model.predict_lambda(X_test)

    lag_col = "customers_lag_7"
    y_lag7  = test_df[lag_col].values if lag_col in test_df.columns else np.full(len(test_df), np.nan)

    rows = []
    for name, y_pred in [("Poisson (regras)", y_poisson), ("Lag-7 (historico)", y_lag7)]:
        mask = ~np.isnan(y_pred)
        if mask.sum() == 0:
            continue
        rows.append({
            "modelo": name, "loja": store,
            "MAE":    round(mean_absolute_error(y_true[mask], y_pred[mask]), 2),
            "RMSE":   round(np.sqrt(mean_squared_error(y_true[mask], y_pred[mask])), 2),
            "n_obs":  int(mask.sum()),
        })

    result_df = pd.DataFrame(rows)
    log.info("[Comparacao][%s]\n%s", store, result_df.to_string(index=False))
    return result_df


# ===========================================================================
# SELF-TEST — executar directamente: python src/optimization/probabilistic_models.py
# ===========================================================================

if __name__ == "__main__":
    _ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    os.chdir(_ROOT)

    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(name)s — %(message)s",
                        datefmt="%H:%M:%S")

    DATA_PATH = "data/processed/baltimore_processed.csv"
    if not os.path.exists(DATA_PATH):
        print(f"[ERRO] Ficheiro nao encontrado: {DATA_PATH}")
        print("  -> Corre primeiro: python main_pipeline.py")
        sys.exit(1)

    df    = pd.read_csv(DATA_PATH, parse_dates=["Date"])
    STORE = "baltimore"
    IS_WEEKEND  = [True, False, False, False, False, False, True]
    DAY_LABELS  = ["Dom", "Seg", "Ter", "Qua", "Qui", "Sex", "Sab"]
    SEP = "=" * 60

    print(f"\nDataset: {len(df)} registos | {df.shape[1]} colunas")

    # ── 1. POISSON ────────────────────────────────────────────────────────────
    print(f"\n{SEP}\nMODELO 1: POISSON — Chegada de Clientes\n{SEP}")
    pm = PoissonArrivalModel()
    pm.fit(df, store=STORE)
    lambdas = pm.predict_week(is_weekend=IS_WEEKEND, month=6)
    print("\n  Previsao lambda (clientes/dia):")
    for label, lam in zip(DAY_LABELS, lambdas):
        print(f"    {label}: {lam:6.1f}  -> int: {int(round(lam))}")
    print("\n  Comparacao Poisson vs. Lag-7:")
    print(compare_with_forecasting(df, pm, store=STORE).to_string(index=False))

    # ── 2. GAUSSIANO ──────────────────────────────────────────────────────────
    print(f"\n{SEP}\nMODELO 2: GAUSSIANO — Variabilidade das Vendas\n{SEP}")
    gm = GaussianSalesModel()
    gm.fit(df, store=STORE)
    test_df   = df.iloc[int(len(df) * 0.8):].head(7).copy()
    scenarios = gm.predict_scenarios(test_df)
    lower, upper = gm.confidence_interval(test_df)
    print(f"\n  sigma={gm.sigma:.0f} | mu_global={gm.mu_global:.0f}")
    print(f"\n  {'Dia':<4} {'Pessimista':>12} {'Realista':>12} {'Otimista':>12} {'IC 95%':>20}")
    print(f"  {'-'*64}")
    for i, label in enumerate(DAY_LABELS):
        print(f"  {label:<4} ${scenarios['pessimistic'][i]:>11,.0f} "
              f"${scenarios['realistic'][i]:>11,.0f} "
              f"${scenarios['optimistic'][i]:>11,.0f} "
              f"  [${lower[i]:,.0f} ; ${upper[i]:,.0f}]")

    # ── 3. LOGISTICO ──────────────────────────────────────────────────────────
    print(f"\n{SEP}\nMODELO 3: LOGISTICO — Conversao via Desconto\n{SEP}")
    lm = LogisticConversionModel()
    lm.fit(df, store=STORE)
    print(f"\n  AUC-ROC={lm.auc:.3f} | threshold={lm.threshold:.1f} vendas/cliente")
    print("\n  Curva de conversao (dia de semana):")
    for pr in np.arange(0.0, 0.31, 0.05):
        p = lm.conversion_curve(np.array([pr]))[0]
        print(f"    PR={pr*100:5.1f}%  P(conv)={p:.3f}  {'#' * int(p * 40)}")

    print(f"\n{SEP}\nTODOS OS MODELOS TESTADOS COM SUCESSO\n{SEP}")
