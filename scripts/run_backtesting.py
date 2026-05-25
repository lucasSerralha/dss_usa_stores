"""
run_backtesting.py — Walk-forward backtesting com TimeSeriesSplit (N_SPLITS > 10)

Para cada loja e cada fold:
  - Treina Random Forest, Regressão Linear e Holt-Winters
  - Avalia MAE, RMSE, MAPE, NMAE
  - TARGET: Num_Customers (v2 — enunciado obriga prever nº de clientes diários)
  - Features: lags de clientes + variáveis exógenas (sem Num_Customers contemporâneo)

Output: results_v2/05_Backtesting/
  backtesting_results.csv   — métricas por fold e por loja
  backtesting_summary.csv   — média das métricas agregada por loja e modelo
  {store}_backtesting.png   — evolução dos erros por fold
"""

import os
import sys
import logging
import glob
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.holtwinters import ExponentialSmoothing

sys.dont_write_bytecode = True
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("TIAPOSE.Backtesting")

N_SPLITS = 11  # enunciado exige >10 iterações de backtesting
# Features alinhadas com C_Context_Expert v2 (target = Num_Customers)
# Num_Customers foi removido: é agora o TARGET, não uma feature (evitar leakage circular)
FEATURES = [
    'customers_lag_7', 'Pct_On_Sale', 'TouristEvent', 'is_holiday',
    'days_to_next_holiday', 'day_of_week', 'sales_lag_7',
    'sales_roll_mean_7'
]
DATA_DIR = "data/processed"
OUT_DIR  = "results_v2/05_Backtesting"


# ---------------------------------------------------------------------------
# Métricas
# ---------------------------------------------------------------------------

def _mape(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return float(np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100)

def _nmae(y_true, y_pred):
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mean_actual = float(np.mean(y_true))
    if mean_actual == 0:
        return 0.0
    return float(mean_absolute_error(y_true, y_pred) / mean_actual * 100)

def _metrics(y_true, y_pred, prefix):
    return {
        f"{prefix}_MAE":  round(float(mean_absolute_error(y_true, y_pred)), 4),
        f"{prefix}_RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 4),
        f"{prefix}_MAPE": round(_mape(y_true, y_pred), 4),
        f"{prefix}_NMAE": round(_nmae(y_true, y_pred), 4),
    }


# ---------------------------------------------------------------------------
# Backtesting por loja
# ---------------------------------------------------------------------------

def backtest_store(file_path: str, n_splits: int = N_SPLITS) -> pd.DataFrame:
    store_name = os.path.basename(file_path).replace("_processed.csv", "").capitalize()
    log.info(f"A processar: {store_name}")

    df = pd.read_csv(file_path, parse_dates=["ds"])
    available_features = [f for f in FEATURES if f in df.columns]

    if not available_features:
        log.error(f"  {store_name}: nenhuma feature disponível — skipping")
        return pd.DataFrame()

    X = df[available_features].fillna(0).values
    y = df["y"].values

    tscv = TimeSeriesSplit(n_splits=n_splits)
    fold_records = []

    for fold_idx, (train_idx, test_idx) in enumerate(tscv.split(X), start=1):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]

        row = {
            "Store":      store_name,
            "Fold":       fold_idx,
            "Train_Size": len(train_idx),
            "Test_Size":  len(test_idx),
        }

        # Random Forest
        rf = RandomForestRegressor(n_estimators=100, random_state=42)
        rf.fit(X_train, y_train)
        row.update(_metrics(y_test, rf.predict(X_test), "RF"))

        # Regressão Linear
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        row.update(_metrics(y_test, lr.predict(X_test), "LR"))

        # Holt-Winters (apenas série temporal, sem exógenas)
        try:
            hw = ExponentialSmoothing(
                y_train, seasonal_periods=7, trend="add", seasonal="add"
            ).fit(optimized=True)
            row.update(_metrics(y_test, hw.forecast(len(y_test)), "HW"))
        except Exception as e:
            log.warning(f"  {store_name} Fold {fold_idx}: Holt-Winters falhou — {e}")
            for m in ("MAE", "RMSE", "MAPE", "NMAE"):
                row[f"HW_{m}"] = None

        fold_records.append(row)
        log.info(
            f"  Fold {fold_idx:2d}/{n_splits} "
            f"| RF_NMAE={row['RF_NMAE']:.2f}% "
            f"| LR_NMAE={row['LR_NMAE']:.2f}%"
        )

    return pd.DataFrame(fold_records)


# ---------------------------------------------------------------------------
# Visualização
# ---------------------------------------------------------------------------

def plot_store_results(results_df: pd.DataFrame, store_name: str, out_dir: str):
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle(f"Backtesting Walk-Forward — {store_name}", fontsize=14, fontweight="bold")

    model_styles = [
        ("Random Forest",     "RF",  "#4A90E2"),
        ("Linear Regression", "LR",  "#50C878"),
        ("Holt-Winters",      "HW",  "#E67E22"),
    ]

    for ax, metric in zip(axes, ["NMAE", "RMSE"]):
        for label, prefix, color in model_styles:
            col = f"{prefix}_{metric}"
            if col in results_df.columns:
                ax.plot(
                    results_df["Fold"],
                    results_df[col],
                    marker="o", label=label, color=color, linewidth=2
                )
        ax.set_title(f"{metric} por Fold")
        ax.set_xlabel("Fold")
        ax.set_ylabel(metric + (" (%)" if metric in ("NMAE", "MAPE") else " ($)"))
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(
        os.path.join(out_dir, f"{store_name.lower()}_backtesting.png"),
        dpi=150, bbox_inches="tight"
    )
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    processed_files = sorted(glob.glob(os.path.join(DATA_DIR, "*_processed.csv")))
    processed_files = [f for f in processed_files if "all_stores" not in f]

    if not processed_files:
        log.error("Nenhum ficheiro processado encontrado em data/processed/")
        return

    all_fold_results = []

    for file_path in processed_files:
        fold_df = backtest_store(file_path, n_splits=N_SPLITS)
        if fold_df.empty:
            continue
        all_fold_results.append(fold_df)
        store_name = fold_df["Store"].iloc[0]
        plot_store_results(fold_df, store_name, OUT_DIR)

    if not all_fold_results:
        log.error("Nenhum resultado de backtesting gerado.")
        return

    # Guardar resultados detalhados por fold
    combined = pd.concat(all_fold_results, ignore_index=True)
    combined.to_csv(os.path.join(OUT_DIR, "backtesting_results.csv"), index=False)

    # Resumo: média por loja e modelo
    model_styles = [
        ("Random Forest",     "RF"),
        ("Linear Regression", "LR"),
        ("Holt-Winters",      "HW"),
    ]
    summary_rows = []
    for store in combined["Store"].unique():
        sdf = combined[combined["Store"] == store]
        for model_label, prefix in model_styles:
            row = {"Store": store, "Model": model_label}
            for metric in ("MAE", "RMSE", "MAPE", "NMAE"):
                col = f"{prefix}_{metric}"
                if col in sdf.columns:
                    row[metric] = round(sdf[col].dropna().mean(), 4)
            summary_rows.append(row)

    summary_df = pd.DataFrame(summary_rows)
    summary_df.to_csv(os.path.join(OUT_DIR, "backtesting_summary.csv"), index=False)

    print("\n" + "=" * 65)
    print(f"BACKTESTING SUMMARY  ({N_SPLITS} folds, walk-forward)")
    print("=" * 65)
    print(summary_df.to_string(index=False))
    print(f"\nResultados exportados para: {os.path.abspath(OUT_DIR)}")


if __name__ == "__main__":
    main()
