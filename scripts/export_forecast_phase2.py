"""
export_forecast_phase2.py — Consolida as previsões do experimento C_Context_Expert
e exporta-as como inputs da Fase II (otimização).

Lê: results/02_Forecasting/{Store}/C_Context_Expert/forecast_values.csv
     (gerados pelo main_pipeline.py / trainer.py)

Exporta: data/processed/phase2_forecast_inputs.csv
  Store  | Date | Actual_Sales | Forecast_Sales (Ensemble Top-3)

Este ficheiro é o ponto de ligação entre a Fase I (Previsão) e a Fase II (Otimização).
"""

import os
import sys
import logging
import pandas as pd

sys.dont_write_bytecode = True

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("TIAPOSE.ForecastExport")

STORES          = ["Baltimore", "Lancaster", "Philadelphia", "Richmond"]
BEST_EXPERIMENT = "C_Context_Expert"
BEST_MODEL      = "Ensemble (Top-3 Experts)"
RESULTS_DIR     = "results/02_Forecasting"
OUT_PATH        = "data/processed/phase2_forecast_inputs.csv"


def load_store_forecast(store: str) -> pd.DataFrame | None:
    csv_path = os.path.join(RESULTS_DIR, store, BEST_EXPERIMENT, "forecast_values.csv")

    if not os.path.exists(csv_path):
        log.warning(f"{store}: {csv_path} não encontrado — skipping")
        return None

    df = pd.read_csv(csv_path, parse_dates=["Date"])

    if BEST_MODEL in df.columns:
        forecast_col = BEST_MODEL
    else:
        candidates = [c for c in df.columns if c not in ("Date", "Actual")]
        if not candidates:
            log.error(f"{store}: nenhuma coluna de previsão disponível")
            return None
        forecast_col = candidates[0]
        log.warning(f"{store}: '{BEST_MODEL}' não encontrado — usando '{forecast_col}'")

    result = pd.DataFrame({
        "Store":          store,
        "Date":           df["Date"],
        "Actual_Sales":   df["Actual"],
        "Forecast_Sales": df[forecast_col],
    })

    nmae = (
        (result["Actual_Sales"] - result["Forecast_Sales"]).abs().mean()
        / result["Actual_Sales"].mean() * 100
    )
    log.info(f"{store}: {len(df)} dias | modelo='{forecast_col}' | NMAE={nmae:.2f}%")
    return result


def main():
    records = [load_store_forecast(s) for s in STORES]
    records = [r for r in records if r is not None]

    if not records:
        log.error("Nenhum dado de previsão encontrado. Execute main_pipeline.py primeiro.")
        return

    combined = pd.concat(records, ignore_index=True)
    combined["Date"] = pd.to_datetime(combined["Date"])
    combined = combined.sort_values(["Store", "Date"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    combined.to_csv(OUT_PATH, index=False)

    print("\n" + "=" * 60)
    print("PHASE II FORECAST INPUTS — EXPORTADO")
    print("=" * 60)

    summary = combined.groupby("Store").agg(
        N_Dias=("Date", "count"),
        Data_Inicio=("Date", "min"),
        Data_Fim=("Date", "max"),
        NMAE_pct=("Actual_Sales", lambda x: round(
            (combined.loc[x.index, "Actual_Sales"] - combined.loc[x.index, "Forecast_Sales"])
            .abs().mean() / x.mean() * 100, 2
        )),
    )
    print(summary.to_string())
    print(f"\nFicheiro: {os.path.abspath(OUT_PATH)}")
    print(f"Total de registos: {len(combined)}")


if __name__ == "__main__":
    main()
