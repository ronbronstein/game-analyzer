#!/usr/bin/env python3
"""
Utility functions for game data analysis
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging
from datetime import datetime
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('gameanalytics.log')
    ]
)

logger = logging.getLogger('gameanalytics')

# Plot styling
def set_plot_style():
    """Set consistent plot style for all visualizations"""
    plt.style.use('ggplot')
    sns.set_style("whitegrid")
    sns.set_palette("deep")
    plt.rcParams.update({'font.size': 12})

# Data loading
def load_csv_data(file_path, parse_dates=True):
    """Load data from CSV with error handling"""
    try:
        logger.info(f"Loading data from {file_path}")
        df = pd.read_csv(file_path)
        
        # Parse dates if requested
        if parse_dates and 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601', errors='coerce')
            # Remove rows with invalid timestamps
            invalid_timestamps = df['timestamp'].isna().sum()
            if invalid_timestamps > 0:
                logger.warning(f"Removed {invalid_timestamps} rows with invalid timestamps")
                df = df.dropna(subset=['timestamp'])
            
            # Add date column if not present
            if 'date' not in df.columns:
                df['date'] = df['timestamp'].dt.date
        
        logger.info(f"Loaded {len(df)} rows from {file_path}")
        return df
    except Exception as e:
        logger.error(f"Error loading data from {file_path}: {e}")
        return pd.DataFrame()

def load_json_data(file_path):
    """Load data from JSON with error handling"""
    try:
        logger.info(f"Loading JSON data from {file_path}")
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        logger.info(f"Successfully loaded JSON data from {file_path}")
        return data
    except Exception as e:
        logger.error(f"Error loading JSON data from {file_path}: {e}")
        return None

# File system operations
def ensure_directory(directory):
    """Create directory if it doesn't exist"""
    if not os.path.exists(directory):
        logger.info(f"Creating directory: {directory}")
        os.makedirs(directory, exist_ok=True)
    return directory

# Data validation
def validate_dataframe(df, required_columns=None):
    """Validate that DataFrame has required columns and is not empty"""
    if df is None or df.empty:
        logger.error("DataFrame is empty or None")
        return False
    
    if required_columns:
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            logger.error(f"DataFrame is missing required columns: {', '.join(missing_columns)}")
            return False
    
    return True

# Statistics and metrics
def calculate_basic_metrics(df):
    """Calculate basic metrics from transactions DataFrame"""
    metrics = {}
    
    if df.empty:
        logger.warning("Cannot calculate metrics: DataFrame is empty")
        return metrics
    
    try:
        metrics['total_transactions'] = len(df)
        metrics['unique_users'] = df['user'].nunique()
        
        if 'date' in df.columns:
            metrics['date_range'] = f"{df['date'].min()} to {df['date'].max()}"
            metrics['days_covered'] = (df['date'].max() - df['date'].min()).days + 1
        
        if 'cash_with_sign' in df.columns:
            metrics['total_cash_volume'] = abs(df['cash_with_sign']).sum()
            metrics['net_cash_change'] = df['cash_with_sign'].sum()
            metrics['avg_transaction_value'] = df['cash_with_sign'].mean()
        
        if 'category' in df.columns:
            top_categories = df['category'].value_counts().head(5).to_dict()
            metrics['top_categories'] = top_categories
        
        logger.info(f"Calculated basic metrics: {len(metrics)} metrics computed")
        return metrics
    except Exception as e:
        logger.error(f"Error calculating basic metrics: {e}")
        return metrics

# Progress tracking
def print_progress(current, total, message="Processing", length=50):
    """Print a progress bar to show processing status"""
    progress = current / total
    block = int(length * progress)
    progress_bar = "[" + "#" * block + "-" * (length - block) + "]"
    percentage = progress * 100
    print(f"\r{message}: {progress_bar} {current}/{total} ({percentage:.1f}%)", end='', flush=True)
    if current == total:
        print()  # Add newline when complete

# Time measurement
class Timer:
    """Simple timer class to measure execution time"""
    def __init__(self, name="Operation"):
        self.name = name
        self.start_time = None
    
    def __enter__(self):
        self.start_time = datetime.now()
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        end_time = datetime.now()
        duration = (end_time - self.start_time).total_seconds()
        logger.info(f"{self.name} completed in {duration:.2f} seconds") 