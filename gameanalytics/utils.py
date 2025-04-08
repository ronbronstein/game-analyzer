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
import sys
import time
import numpy as np
import matplotlib.dates as mdates
import subprocess

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger('gameanalytics')

# Plot styling
def set_plot_style(theme='dark'):
    """Set matplotlib style for consistent visuals"""
    plt.style.use('ggplot')
    
    if theme == 'dark':
        plt.rcParams.update({
            'figure.facecolor': '#1e1e1e',
            'axes.facecolor': '#2d2d2d',
            'axes.edgecolor': '#757575',
            'axes.labelcolor': 'white',
            'axes.titlecolor': 'white',
            'xtick.color': 'white',
            'ytick.color': 'white',
            'text.color': 'white',
            'grid.color': '#3a3a3a',
        })
    
    # Set higher DPI and figure size for better quality
    plt.rcParams['figure.dpi'] = 100
    plt.rcParams['figure.figsize'] = (12, 8)
    
    # Improve font rendering
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['Arial', 'DejaVu Sans', 'Liberation Sans', 'Bitstream Vera Sans', 'sans-serif']
    
    # Configure date formatting
    plt.rcParams['date.autoformatter.day'] = '%Y-%m-%d'
    plt.rcParams['date.autoformatter.hour'] = '%m-%d %H:%M'
    
    # Increase font sizes for better readability
    plt.rcParams['font.size'] = 12
    plt.rcParams['axes.titlesize'] = 16
    plt.rcParams['axes.labelsize'] = 14

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
        os.makedirs(directory, exist_ok=True)
        logger.info(f"Created directory: {directory}")
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
    """Simple timer for performance tracking"""
    def __init__(self, description):
        self.description = description
        
    def __enter__(self):
        self.start = time.time()
        return self
        
    def __exit__(self, *args):
        elapsed = time.time() - self.start
        logger.info(f"{self.description} completed in {elapsed:.2f}s")

def format_date_axis(ax, date_column, rotation=45):
    """Format the date axis properly for better readability"""
    if not len(date_column):
        return
    
    # Determine the date range and choose appropriate formatter
    date_range = (pd.to_datetime(date_column.max()) - pd.to_datetime(date_column.min())).days
    
    if date_range > 365:  # More than a year
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    elif date_range > 180:  # 6+ months
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %Y'))
        ax.xaxis.set_major_locator(mdates.MonthLocator())
    elif date_range > 30:  # 1+ month
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.xaxis.set_major_locator(mdates.WeekdayLocator(interval=1))
    elif date_range > 7:  # 1+ week
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d'))
        ax.xaxis.set_major_locator(mdates.DayLocator(interval=1))
    else:  # Less than a week
        ax.xaxis.set_major_formatter(mdates.DateFormatter('%b %d %H:%M'))
        ax.xaxis.set_major_locator(mdates.HourLocator(interval=6))
    
    # Rotate date labels for better readability
    plt.setp(ax.get_xticklabels(), rotation=rotation, ha='right')
    
    # Add some padding at the bottom for rotated labels
    plt.subplots_adjust(bottom=0.2)

def is_docker_running():
    """Check if Docker daemon is running
    
    Returns:
        bool: True if Docker is running, False otherwise
    """
    try:
        result = subprocess.run(
            ['docker', 'info'],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=3
        )
        return result.returncode == 0
    except (subprocess.SubprocessError, FileNotFoundError):
        return False 