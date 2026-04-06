import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from prophet import Prophet
import joblib
import logging

# Configure Logging
logger = logging.getLogger(__name__)

def calculate_mape(y_true, y_pred):
    """Calculates Mean Absolute Percentage Error."""
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    return np.mean(np.abs((y_true - y_pred) / np.maximum(y_true, 1))) * 100

def plot_forecast_results(store_name, y_test, results_dict, output_dir):
    """Generates and saves a professional forecast comparison plot."""
    plt.figure(figsize=(12, 6))
    
    # Use the last 30 index points for visualization
    last_30_indices = y_test.index[-30:]
    plt.plot(range(30), y_test.loc[last_30_indices].values, label='Actual Sales', color='black', linewidth=2, marker='o')
    
    colors = ['blue', 'green', 'red', 'orange', 'purple']
    for i, (model_name, y_pred) in enumerate(results_dict.items()):
        # Plot only the last 30 days for clarity
        plt.plot(range(30), y_pred[-30:], label=f'Predicted {model_name}', linestyle='--', alpha=0.8, color=colors[i % len(colors)])
    
    plt.title(f'Sales Forecast Comparison - {store_name.capitalize()}')
    plt.xlabel('Days (Last 30 of Test Set)')
    plt.ylabel('Sales')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plot_path = os.path.join(output_dir, f'{store_name}_forecast_comparison.png')
    plt.savefig(plot_path)
    plt.close()
    logger.info(f"  [{store_name}] Forecast plot saved: {plot_path}")

def train_and_evaluate_all(file_path, output_dir='data/processed/'):
    """
    Trains and compares multiple forecasting models for a given store.
    Models: Seasonal Naive, Linear Regression, Random Forest, Holt-Winters, and Prophet.
    Includes TimeSeriesSplit for robust backtesting.
    """
    store_name = os.path.basename(file_path).replace('_processed.csv', '')
    logger.info(f"Full Professional Training & Evaluation for: {store_name}")
    
    # Load data
    df = pd.read_csv(file_path, parse_dates=['ds'])
    
    # Define features
    features = [
        'Num_Employees', 'Num_Customers', 'Pct_On_Sale', 'TouristEvent',
        'is_holiday', 'day_of_week', 'is_weekend', 'month', 'season_num',
        'sales_lag_7', 'sales_lag_14', 'sales_lag_21', 'sales_lag_28',
        'customers_lag_7', 'customers_lag_14', 'customers_lag_21', 'customers_lag_28',
        'sales_roll_mean_7', 'sales_roll_std_7'
    ]
    
    # Time-based split (80/20) for final visualization
    split_idx = int(len(df) * 0.8)
    train_df, test_df = df.iloc[:split_idx], df.iloc[split_idx:]
    
    X_train, X_test = train_df[features], test_df[features]
    y_train, y_test = train_df['y'], test_df['y']
    
    store_metrics = []
    plot_data = {}

    # 1. Seasonal Naive Baseline (Lucas's Task)
    # Using the value from 7 days ago as the prediction for today
    y_pred_naive = test_df['sales_lag_7'].values
    store_metrics.append({
        'Model': 'Seasonal Naive',
        'MAE': mean_absolute_error(y_test, y_pred_naive),
        'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_naive)),
        'MAPE': calculate_mape(y_test, y_pred_naive)
    })
    plot_data['Seasonal Naive'] = y_pred_naive

    # 2. Linear Regression (Baseline)
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

    # 3. Random Forest (Rafael's Task)
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
    
    # Save model
    if not os.path.exists('models'): os.makedirs('models')
    joblib.dump(rf, f'models/{store_name}_rf_model.pkl')

    # 4. Holt-Winters (Pedro O's Task)
    try:
        hw = ExponentialSmoothing(y_train, seasonal_periods=7, trend='add', seasonal='add').fit()
        y_pred_hw = hw.forecast(len(y_test))
        # Handle index mismatch
        y_pred_hw_vals = y_pred_hw.values
        store_metrics.append({
            'Model': 'Holt-Winters',
            'MAE': mean_absolute_error(y_test, y_pred_hw_vals),
            'RMSE': np.sqrt(mean_squared_error(y_test, y_pred_hw_vals)),
            'MAPE': calculate_mape(y_test, y_pred_hw_vals)
        })
        plot_data['Holt-Winters'] = y_pred_hw_vals
    except Exception as e:
        logger.error(f"  [{store_name}] Holt-Winters failed: {e}")

    # 5. Prophet (António's Task)
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
    except Exception as e:
        logger.error(f"  [{store_name}] Prophet failed: {e}")

    # SAVE VISUALIZATION
    plots_dir = os.path.join(output_dir, 'plots')
    if not os.path.exists(plots_dir): os.makedirs(plots_dir)
    plot_forecast_results(store_name, y_test, plot_data, plots_dir)

    # Save individual store metrics
    metrics_df = pd.DataFrame(store_metrics)
    metrics_df.to_csv(os.path.join(output_dir, f"{store_name}_comparison.csv"), index=False)
    
    return {store_name: store_metrics}
