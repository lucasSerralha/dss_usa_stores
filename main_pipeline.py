import os
import sys
import logging
import pandas as pd
import glob

# Add src to python path to allow imports from our dynamic structure
sys.path.append(os.path.join(os.getcwd(), 'src'))

# Import centralized modules (W4/W5 Professional Structure)
from preparation import run_full_preparation
from trainer import train_and_evaluate_all

# Configure Logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def clean_old_reports(processed_dir='data/processed/'):
    """Cleaning up legacy and individual comparison files for minimalism."""
    old_files = glob.glob(os.path.join(processed_dir, "*_comparison.csv"))
    for f in old_files:
        try:
            os.remove(f)
        except:
            pass
    logger.info("Removed individual store reports. Master report is the source of truth.")

def main():
    print("\n" + "="*60)
    print("USA STORES FORECASTING - MASTER PIPELINE (PRO 2026)")
    print("="*60)
    
    # 1. CLEANING AND FEATURE ENGINEERING
    logger.info("Step 1: Data Preparation & Professional Feature Engineering")
    run_full_preparation(input_dir='data/raw/', output_dir='data/processed/')
    
    # 2. EVALUATION OF ALL MANDATORY MODELS (W4 - Forecasting II)
    logger.info("Step 2: Multi-Model Performance Comparison (5 Models Integrated)")
    processed_files = glob.glob('data/processed/*_processed.csv')
    
    # Filter out the merged file
    processed_files = [f for f in processed_files if 'all_stores' not in f]
    
    if not processed_files:
        logger.error("No processed files found! Pipeline aborted.")
        return

    # To store all results in a single professional list
    master_results_list = []
    
    for f in processed_files:
        # Train and evaluate models for this store
        res_dict = train_and_evaluate_all(f, output_dir='data/processed/')
        
        # Flatten and tag with Store Name
        store_name = list(res_dict.keys())[0]
        for row in res_dict[store_name]:
            row['Store'] = store_name.capitalize()
            master_results_list.append(row)
    
    # 3. GENERATE SINGLE MASTER REPORT
    logger.info("Step 3: Generating Master Comparison Report (MAE, RMSE, MAPE)")
    master_report_df = pd.DataFrame(master_results_list)
    
    # Reorder columns for professional layout
    cols = ['Store', 'Model', 'MAE', 'RMSE', 'MAPE']
    master_report_df = master_report_df[cols].sort_values(['Store', 'MAE'])
    
    report_path = os.path.join('data/processed', 'final_model_report.csv')
    master_report_df.to_csv(report_path, index=False)
    logger.info(f"Master performance report successfully saved: {report_path}")
    
    # CLEANUP: Remove individual comparison files
    clean_old_reports()

    # 4. FINAL VISUAL SUMMARY
    print("\n" + "="*60)
    print("FINAL PERFORMANCE REPORT (ORDERED BY BEST MAE)")
    print("="*60)
    print(master_report_df.to_string(index=False))
    
    # Best Model Recommendation
    print("\n" + "="*60)
    print("RECOMMENDED MODELS BY STORE")
    print("="*60)
    best_models = master_report_df.groupby('Store').first().reset_index()
    print(best_models.to_string(index=False))
    print("="*60 + "\n")
    
    logger.info("Master Pipeline finished.")
    logger.info("Forecast Plots available in: data/processed/plots/")
    logger.info("Master Report available in: data/processed/final_model_report.csv")

if __name__ == "__main__":
    main()
