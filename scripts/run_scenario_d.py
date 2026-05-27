"""
Treino dedicado do Cenário D — D_Full_Context
==============================================
Treina apenas o novo cenário D para as 4 lojas e guarda os resultados
directamente em results/02_Forecasting/ (directório lido pelo DSS App).

Não re-treina nem sobrescreve os cenários A, B ou C existentes.

Execução (a partir da raiz do projecto):
    python scripts/run_scenario_d.py
"""
import os
import sys
import glob
import logging

# Bloqueio de __pycache__
sys.dont_write_bytecode = True

# Adicionar raiz do projecto ao path para importar src.*
_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, _root)

from src.forecasting.trainer import train_and_evaluate_all

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s — %(levelname)s — %(message)s",
)
logger = logging.getLogger(__name__)

# ── Definição do Cenário D ─────────────────────────────────────────────────
# Todas as features disponíveis sem leakage (18 features):
#   Calendário completo + contexto externo + todos os lags + estatísticas móveis
D_FULL_CONTEXT = [
    # Calendário completo
    "day_of_week", "IsWeekend", "month", "season_num",
    # Contexto externo
    "is_holiday", "days_to_next_holiday", "TouristEvent", "Pct_On_Sale",
    # Lags de clientes (curto, médio e longo prazo)
    "customers_lag_7", "customers_lag_14", "customers_lag_21", "customers_lag_28",
    # Lags de vendas (curto, médio e longo prazo)
    "sales_lag_7", "sales_lag_14", "sales_lag_21", "sales_lag_28",
    # Estatísticas móveis de vendas
    "sales_roll_mean_7", "sales_roll_std_7",
]

# ── Directório de saída — mesmo que o DSS App lê ──────────────────────────
OUTPUT_DIR = os.path.join(_root, "results")

# ── Descoberta dos ficheiros processados ──────────────────────────────────
processed_files = sorted(glob.glob(
    os.path.join(_root, "data", "processed", "*_processed.csv")
))
processed_files = [f for f in processed_files if "all_stores" not in f]

if not processed_files:
    logger.error(
        "Nenhum ficheiro *_processed.csv encontrado em data/processed/. "
        "Execute primeiro: python main_pipeline.py"
    )
    sys.exit(1)

logger.info(
    "Cenário D — D_Full_Context (%d features)", len(D_FULL_CONTEXT)
)
logger.info("Lojas detectadas: %s", [os.path.basename(f) for f in processed_files])
logger.info("Directório de saída: %s", OUTPUT_DIR)
print()

# ── Treino por loja ────────────────────────────────────────────────────────
for f in processed_files:
    store = os.path.basename(f).replace("_processed.csv", "").capitalize()
    logger.info("A treinar loja: %s", store)
    try:
        train_and_evaluate_all(
            file_path=f,
            output_dir=OUTPUT_DIR,
            custom_features=D_FULL_CONTEXT,
            experiment_name="D_Full_Context",
        )
        logger.info("  %s — concluído.", store)
    except Exception as exc:
        logger.error("  %s — ERRO: %s", store, exc)

print()
logger.info(
    "Cenário D concluído. Resultados em: results/02_Forecasting/*/D_Full_Context/"
)
logger.info(
    "O DSS App vai seleccionar automaticamente D_Full_Context "
    "como cenário preferencial na próxima sessão."
)
