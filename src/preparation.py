import pandas as pd
import numpy as np
import holidays
import os
import glob
import logging

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize US Holidays
us_holidays = holidays.US()

def get_season(date):
    """Returns the season for a given date."""
    month = date.month
    if month in [12, 1, 2]: return 'Winter'
    elif month in [3, 4, 5]: return 'Spring'
    elif month in [6, 7, 8]: return 'Summer'
    else: return 'Autumn'

def prepare_store_data(file_path, output_dir='data/processed/'):
    """
    Performs complete data preparation for a single store file.
    Includes cleaning, anomaly handling, and feature engineering.
    """
    store_name = os.path.basename(file_path).replace('.csv', '')
    logger.info(f"Preparing data for store: {store_name}")
    
    # 1. Load and Sort
    df = pd.read_csv(file_path, parse_dates=['Date'])
    df = df.sort_values('Date').reset_index(drop=True)
    
    # 2. Missing Values Handling (Linear Interpolation)
    # Pct_On_Sale: Important for promotion tracking
    df['Pct_On_Sale'] = df['Pct_On_Sale'].interpolate(method='linear').fillna(method='bfill')
    
    # 3. Anomaly Handling (Dirty Data)
    # Case: Customers = 0 but Sales > 0 (Data error)
    mask_dirty = (df['Num_Customers'] == 0) & (df['Sales'] > 0)
    if mask_dirty.any():
        logger.warning(f"  [{store_name}] Fixed {mask_dirty.sum()} dirty data points (Zero-Customer Sales)")
        df.loc[mask_dirty, 'Num_Customers'] = np.nan
        df['Num_Customers'] = df['Num_Customers'].interpolate(method='linear')
    
    # 4. Outlier Handling (Clipping 1% - 99%)
    # This ensures extreme values don't distort the model's learning
    for col in ['Num_Customers', 'Num_Employees', 'Sales']:
        if col in df.columns:
            q_low = df[col].quantile(0.01)
            q_high = df[col].quantile(0.99)
            df[col] = df[col].clip(q_low, q_high)
            
    # 5. Encoding
    if df['TouristEvent'].dtype == 'object':
        df['TouristEvent'] = df['TouristEvent'].map({'No': 0, 'Yes': 1}).fillna(0)
    
    # 5. Feature Engineering - Time Features
    df['is_holiday'] = df['Date'].apply(lambda x: 1 if x in us_holidays else 0)
    df['day_of_week'] = df['Date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['month'] = df['Date'].dt.month
    df['year'] = df['Date'].dt.year
    df['season_num'] = df['Date'].apply(get_season).map({'Winter': 0, 'Spring': 1, 'Summer': 2, 'Autumn': 3})
    
    # 6. Forecasting Features - Lags (7, 14, 21, 28 days)
    for lag in [7, 14, 21, 28]:
        df[f'sales_lag_{lag}'] = df['Sales'].shift(lag)
        df[f'customers_lag_{lag}'] = df['Num_Customers'].shift(lag)
        
    # 7. Forecasting Features - Rolling Means (7 days)
    df['sales_roll_mean_7'] = df['Sales'].shift(1).rolling(window=7).mean()
    df['sales_roll_std_7'] = df['Sales'].shift(1).rolling(window=7).std()
    
    # 8. Prophet Compatibility
    df['ds'] = df['Date'] # Prophet requires 'ds' for dates
    df['y'] = df['Sales'] # Prophet requires 'y' for target
    
    # 9. Final Cleanup (Drop rows with NaN due to lags)
    df = df.dropna().reset_index(drop=True)
    
    # 10. Export
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    output_path = os.path.join(output_dir, f"{store_name}_processed.csv")
    df.to_csv(output_path, index=False)
    logger.info(f"  [{store_name}] Successfully saved to: {output_path}")
    
    return df

def run_full_preparation(input_dir='data/raw/', output_dir='data/processed/'):
    """Processes all raw files in the input directory."""
    csv_files = glob.glob(os.path.join(input_dir, "*.csv"))
    if not csv_files:
        logger.error(f"No CSV files found in {input_dir}")
        return
        
    for f in csv_files:
        prepare_store_data(f, output_dir)
    
    # NEW: Create a professional merged dataset for global analysis
    create_merged_dataset(output_dir)
    
    logger.info("Data preparation completed for all stores.")

def create_merged_dataset(processed_dir='data/processed/'):
    """Combines all processed store CSVs into a single global dataset."""
    logger.info("Creating a professional merged dataset (all_stores_processed.csv)")
    processed_files = glob.glob(os.path.join(processed_dir, "*_processed.csv"))
    
    # Filter out any existing merged file if it happens to match the pattern
    processed_files = [f for f in processed_files if 'all_stores' not in f]
    
    all_data = []
    for f in processed_files:
        temp_df = pd.read_csv(f)
        store_name = os.path.basename(f).replace('_processed.csv', '')
        temp_df['store_id'] = store_name # Tag each row with its store
        all_data.append(temp_df)
    
    if all_data:
        merged_df = pd.concat(all_data, ignore_index=True)
        output_path = os.path.join(processed_dir, "all_stores_processed.csv")
        merged_df.to_csv(output_path, index=False)
        logger.info(f"  Merged dataset saved: {output_path}")

if __name__ == "__main__":
    run_full_preparation()
