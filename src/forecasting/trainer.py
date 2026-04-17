import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet
import joblib
import logging

# Configuração do Logger para monitorização de processos
logger = logging.getLogger(__name__)

def calculate_mape(y_true, y_pred):
    """
    Calcula o Erro Médio Percentual Absoluto (MAPE).
    Métrica crucial para entender o erro em termos de percentagem face ao valor real.
    """
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100

def plot_forecast_results(store_name, y_test, results_dict, output_dir):
    """
    Gera e exporta uma galeria visual profissional com as previsões.
    Inclui um gráfico comparativo global e detalhes individuais por modelo.
    """
    # 1. GERAÇÃO DO GRÁFICO COMPARATIVO COMBINADO
    plt.figure(figsize=(14, 7))
    
    # Visualização focada nos últimos 30 pontos para garantir legibilidade
    last_30_indices = y_test.index[-30:]
    y_true_last_30 = y_test.loc[last_30_indices].values
    
    plt.plot(range(30), y_true_last_30, label='Vendas Reais (Ground Truth)', color='black', linewidth=3, marker='o', zorder=5)
    
    # Paleta de cores profissional para distinção dos modelos
    colors = ['#4A90E2', '#50C878', '#E67E22', '#F1C40F', '#9B59B6']
    for i, (model_name, y_pred) in enumerate(results_dict.items()):
        plt.plot(range(30), y_pred[-30:], label=f'Previsto: {model_name}', linestyle='--', alpha=0.8, color=colors[i % len(colors)], linewidth=2)
    
    plt.title(f'Comparação de Previsão de Vendas: {store_name.capitalize()}', fontsize=16, fontweight='bold', pad=20)
    plt.xlabel('Dias (Últimos 30 dias do conjunto de teste)', fontsize=12)
    plt.ylabel('Vendas ($)', fontsize=12)
    plt.legend(loc='upper left', frameon=True, shadow=True)
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Guardar gráfico combinado na pasta de Forecasting Report
    os.makedirs(output_dir, exist_ok=True)
    comparison_path = os.path.join(output_dir, 'Comparison_All_Models.png')
    plt.savefig(comparison_path, dpi=300, bbox_inches='tight')
    plt.close()

    # 2. GERAÇÃO DE GRÁFICOS DETALHADOS POR MODELO
    details_dir = os.path.join(output_dir, 'Model_Details')
    os.makedirs(details_dir, exist_ok=True)

    for i, (model_name, y_pred) in enumerate(results_dict.items()):
        plt.figure(figsize=(12, 6))
        plt.plot(range(30), y_true_last_30, label='Vendas Reais', color='black', linewidth=2, marker='o')
        plt.plot(range(30), y_pred[-30:], label=f'Previsão: {model_name}', linestyle='--', color=colors[i % len(colors)], linewidth=2.5)
        
        plt.title(f'Detalhe do Modelo: {model_name} ({store_name.capitalize()})', fontsize=14)
        plt.xlabel('Dias')
        plt.ylabel('Vendas')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        file_name = model_name.replace(' ', '_') + '.png'
        plt.savefig(os.path.join(details_dir, file_name), dpi=150, bbox_inches='tight')
        plt.close()

    logger.info(f"  [{store_name}] Galeria de resultados profissionais gerada em: {output_dir}")

def train_sarimax(train_df, test_df):
    """
    Treina o modelo SARIMAX com variáveis exógenas e parametrização de segurança.
    Garante que os dados são numéricos e preenchidos, evitando falhas de convergência.
    """
    # 1. Seleção e Limpeza de Variáveis Exógenas (Critério de Especialista)
    exo_cols = ['Pct_On_Sale', 'is_weekend', 'TouristEvent']
    
    def prepare_exog(df):
        exog = df[exo_cols].copy()
        
        # Tratamento de strings (ex: 'Yes'/'No' -> 1/0) para TouristEvent
        if 'TouristEvent' in exog.columns and exog['TouristEvent'].dtype == object:
            exog['TouristEvent'] = exog['TouristEvent'].map({'Yes': 1, 'No': 0, '0': 0, '1': 1})
        
        # Conversão explícita para float e preenchimento de NaNs com 0 para estabilidade
        return exog.astype(float).fillna(0)

    exog_train = prepare_exog(train_df)
    exog_test = prepare_exog(test_df)
    
    y_train = train_df['y'].astype(float)
    y_test = test_df['y'].astype(float)
    
    # 2. Configuração do Modelo com Fallback Robusto (1,1,1)(1,1,1)7
    # Desativamos enforce_stationarity e enforce_invertibility para dados reais ruidosos
    model = SARIMAX(
        y_train, 
        exog=exog_train, 
        order=(1,1,1), 
        seasonal_order=(1,1,1,7),
        enforce_stationarity=False,
        enforce_invertibility=False
    )
    
    # Ajuste silencioso (disp=False) para o pipeline principal
    results = model.fit(disp=False)
    
    # 3. Geração de Previsões e Cálculo de Métricas Consistentes
    y_pred = results.forecast(steps=len(y_test), exog=exog_test)
    y_pred_vals = y_pred.values
    
    metrics = {
        'MAE': mean_absolute_error(y_test, y_pred_vals),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_vals)),
        'MAPE': calculate_mape(y_test, y_pred_vals)
    }
    
    return metrics, y_pred_vals

def train_and_evaluate_all(file_path, output_dir='data/processed/', custom_features=None, experiment_name='Default'):
    """
    Core do Pipeline: Treina e compara múltiplos modelos de forecasting.
    Permite a passagem de 'custom_features' para experimentação de variáveis.
    """
    store_name = os.path.basename(file_path).replace('_processed.csv', '')
    logger.info(f"[{experiment_name}] Iniciando Treino para: {store_name}")
    
    # Carregamento do dataset pré-processado
    df = pd.read_csv(file_path, parse_dates=['ds'])
    
    # Seleção de variáveis explicativas (Features) - EXCLUÍMOS 'Num_Employees' (Causalidade)
    # Definimos conjuntos de features para experimentação
    features_all = [
        'Num_Customers', 'Pct_On_Sale', 'TouristEvent',
        'is_holiday', 'day_of_week', 'is_weekend', 'month', 'season_num',
        'sales_lag_1', 'sales_lag_7', 'sales_lag_14', 'sales_lag_21', 'sales_lag_28',
        'customers_lag_1', 'customers_lag_7', 'customers_lag_14', 'customers_lag_21', 'customers_lag_28',
        'sales_roll_mean_7', 'sales_roll_std_7'
    ]
    
    # Se existirem custom_features, usamos essas. Caso contrário, usamos o set completo.
    features = custom_features if custom_features is not None else [f for f in features_all if f in df.columns]
    
    logger.info(f"  [{store_name}] Variáveis utilizadas: {len(features)}")
    
    # Divisão temporal (80% treino para histórico, 20% teste para validação futura)
    split_idx = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
    
    X_train, X_test = train_df[features], test_df[features]
    y_train, y_test = train_df['y'], test_df['y']
    
    store_metrics = []
    plot_data = {}

    # 1. Cenário Base: Seasonal Naive (Tarefa do Lucas)
    # Estratégia: Utiliza o valor de há exatamente 7 dias como previsão para hoje
    y_pred_naive = test_df['sales_lag_7'].values
    store_metrics.append({
        'Model': 'Seasonal Naive',
        'MAE': mean_absolute_error(y_test, y_pred_naive),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_naive)),
        'MAPE': calculate_mape(y_test, y_pred_naive)
    })
    plot_data['Seasonal Naive'] = y_pred_naive

    # 2. Regressão Linear (Baseline de Machine Learning)
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    y_pred_lr = lr.predict(X_test)
    store_metrics.append({
        'Model': 'Linear Regression',
        'MAE': mean_absolute_error(y_test, y_pred_lr),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_lr)),
        'MAPE': calculate_mape(y_test, y_pred_lr)
    })
    plot_data['Linear Regression'] = y_pred_lr

    # 3. Random Forest (Modelo Robusto Anti-Overfitting - Tarefa do Rafael)
    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    store_metrics.append({
        'Model': 'Random Forest',
        'MAE': mean_absolute_error(y_test, y_pred_rf),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
        'MAPE': calculate_mape(y_test, y_pred_rf)
    })
    plot_data['Random Forest'] = y_pred_rf
    
    # Persistência do modelo para futura utilização na fase de Otimização (W5)
    if not os.path.exists('models'): os.makedirs('models')
    joblib.dump(rf, f'models/{store_name}_rf_model.pkl')

    # 4. Holt-Winters (Suavização Exponencial Tripla - Tarefa do Pedro O)
    try:
        hw = ExponentialSmoothing(y_train, seasonal_periods=7, trend='add', seasonal='add').fit()
        y_pred_hw = hw.forecast(len(y_test))
        y_pred_hw_vals = y_pred_hw.values
        store_metrics.append({
            'Model': 'Holt-Winters',
            'MAE': mean_absolute_error(y_test, y_pred_hw_vals),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_hw_vals)),
            'MAPE': calculate_mape(y_test, y_pred_hw_vals)
        })
        plot_data['Holt-Winters'] = y_pred_hw_vals
    except Exception as e:
        logger.error(f"  [{store_name}] Falha no Holt-Winters: {e}")

    # 5. Facebook Prophet (Módulos Baseados em Decomposição - Tarefa do António)
    try:
        m = Prophet(yearly_seasonality=True, weekly_seasonality=True, daily_seasonality=False)
        for col in features:
            if col not in ['ds', 'y']: m.add_regressor(col)
        m.fit(train_df[['ds', 'y'] + features])
        forecast = m.predict(test_df[['ds'] + features])
        y_pred_prophet = forecast['yhat'].values
        store_metrics.append({
            'Model': 'Prophet',
            'MAE': mean_absolute_error(y_test, y_pred_prophet),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_prophet)),
            'MAPE': calculate_mape(y_test, y_pred_prophet)
        })
        plot_data['Prophet'] = y_pred_prophet

        # NOVO: EXPORTAÇÃO DOS COMPONENTES DO PROPHET (Anatomia da Série)
        # Capturamos a tendência e as sazonalidades para o dashboard ultra
        # Note: 'weekly' e 'yearly' podem não existir se não houver dados suficientes
        comp_cols = ['ds', 'trend']
        if 'weekly' in forecast.columns: comp_cols.append('weekly')
        if 'yearly' in forecast.columns: comp_cols.append('yearly')
        
        results_subpath = os.path.join(output_dir, '02_Forecasting_Report', store_name.capitalize(), experiment_name)
        forecast[comp_cols].to_csv(os.path.join(results_subpath, "prophet_components.csv"), index=False)

    except Exception as e:
        logger.error(f"  [{store_name}] Falha no Prophet: {e}")

    # 6. SARIMAX (Nova Implementação Robusta - Requisito Prof.)
    try:
        metrics_sarimax, y_pred_sarimax = train_sarimax(train_df, test_df)
        store_metrics.append({
            'Model': 'SARIMAX',
            **metrics_sarimax
        })
        plot_data['SARIMAX'] = y_pred_sarimax
    except Exception as e:
        logger.error(f"  [{store_name}] Falha no SARIMAX: {e}")

    # EXPORTAÇÃO DA GALERIA VISUAL E MÉTRICAS POR LOJA
    results_subpath = os.path.join(output_dir, '02_Forecasting_Report', store_name.capitalize(), experiment_name)
    plot_forecast_results(store_name, y_test, plot_data, results_subpath)

    # NOVO: EXPORTAÇÃO DE DADOS BRUTOS PARA PLOTLY
    # Criamos um DataFrame consolidado com Datas, Real e Previsões
    forecast_df = pd.DataFrame({'Date': test_df['ds'].values, 'Actual': y_test.values})
    for model_name, y_pred in plot_data.items():
        forecast_df[model_name] = y_pred
    
    forecast_df.to_csv(os.path.join(results_subpath, "forecast_values.csv"), index=False)

    # NOVO: EXPORTAÇÃO DE IMPORTÂNCIA DE VARIÁVEIS (BASEADO NO RANDOM FOREST)
    importance_df = pd.DataFrame({
        'Feature': features,
        'Importance': rf.feature_importances_
    }).sort_values(by='Importance', ascending=False)
    importance_df.to_csv(os.path.join(results_subpath, "feature_importance.csv"), index=False)

    # Consolidação das métricas com identificador da experiência
    metrics_df = pd.DataFrame(store_metrics)
    metrics_df['Experiment'] = experiment_name
    metrics_df.to_csv(os.path.join(results_subpath, "store_metrics.csv"), index=False)
    
    return {store_name: store_metrics}
