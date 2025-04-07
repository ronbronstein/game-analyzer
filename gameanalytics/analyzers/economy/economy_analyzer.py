#!/usr/bin/env python3
"""
Economy Health Analyzer
Analyzes economic health and transaction patterns
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
import json
from collections import defaultdict
from datetime import datetime, timedelta
import traceback

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from gameanalytics.utils import load_csv_data, ensure_directory, set_plot_style, logger, Timer
from gameanalytics.config import ANALYSIS_CONFIG, VISUALIZATION_CONFIG

class EconomyAnalyzer:
    """Analyze economic health metrics and transaction patterns"""
    
    def __init__(self, data_dir='balance_data', output_dir='economy_analysis'):
        """Initialize the economy analyzer
        
        Args:
            data_dir (str): Directory with balance data
            output_dir (str): Directory to save analysis results
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.data = None
        self.analysis_results = {}
        
        # Set plot style
        set_plot_style()
    
    def load_data(self):
        """Load the processed balance data"""
        with Timer("Loading economy data"):
            try:
                # Load full dataset
                transactions_path = os.path.join(self.data_dir, 'balance_updates.csv')
                df = load_csv_data(transactions_path)
                
                if df.empty:
                    logger.error(f"Failed to load transaction data from {transactions_path}")
                    return False
                
                # Load summary data
                try:
                    user_summary_path = os.path.join(self.data_dir, 'user_summary.csv')
                    category_summary_path = os.path.join(self.data_dir, 'category_summary.csv')
                    daily_summary_path = os.path.join(self.data_dir, 'daily_summary.csv')
                    
                    user_summary = load_csv_data(user_summary_path, parse_dates=False)
                    category_summary = load_csv_data(category_summary_path, parse_dates=False)
                    daily_summary = load_csv_data(daily_summary_path)
                    
                    if daily_summary.empty:
                        logger.warning("No daily summary data found. Some visualizations will be skipped.")
                    
                    # Load stats
                    stats_path = os.path.join(self.data_dir, 'stats_summary.json')
                    if os.path.exists(stats_path):
                        with open(stats_path, 'r') as f:
                            stats = json.load(f)
                    else:
                        logger.warning(f"Stats summary not found at {stats_path}. Will calculate from data.")
                        # Calculate basic stats
                        stats = {
                            'total_transactions': len(df),
                            'unique_users': df['user'].nunique(),
                            'date_range': f"{df['date'].min()} to {df['date'].max()}",
                            'total_cash_volume': float(abs(df['cash_with_sign']).sum()),
                            'net_cash_change': float(df['cash_with_sign'].sum())
                        }
                except Exception as e:
                    logger.warning(f"Error loading summary data: {e}. Will continue with transaction data only.")
                    user_summary = pd.DataFrame()
                    category_summary = pd.DataFrame()
                    daily_summary = pd.DataFrame()
                    
                    # Calculate basic stats
                    stats = {
                        'total_transactions': len(df),
                        'unique_users': df['user'].nunique(),
                        'date_range': f"{df['date'].min()} to {df['date'].max()}",
                        'total_cash_volume': float(abs(df['cash_with_sign']).sum()),
                        'net_cash_change': float(df['cash_with_sign'].sum())
                    }
                
                self.data = {
                    'transactions': df,
                    'user_summary': user_summary,
                    'category_summary': category_summary,
                    'daily_summary': daily_summary,
                    'stats': stats
                }
                
                logger.info(f"Loaded economy data: {len(df)} transactions, {df['user'].nunique()} users")
                return True
            except Exception as e:
                logger.error(f"Error loading economy data: {e}")
                return False
    
    def analyze_user_activity(self):
        """Analyze user activity patterns"""
        with Timer("Analyzing user activity"):
            df = self.data['transactions']
            user_summary = self.data['user_summary']
            
            if user_summary.empty and not df.empty:
                # Calculate user summary if not provided
                logger.info("Calculating user summary from transaction data")
                user_summary = df.groupby('user').agg({
                    'timestamp': 'count',
                    'cash_with_sign': 'sum'
                }).rename(columns={'timestamp': 'transaction_count'})
                user_summary = user_summary.sort_values('transaction_count', ascending=False)
            
            # Top users by transaction count
            top_users = user_summary.head(ANALYSIS_CONFIG['top_n_users']).copy()
            
            # Create visualization directory
            ensure_directory(self.output_dir)
            
            # Plot top users by transaction count
            plt.figure(figsize=(14, 10))
            ax = sns.barplot(
                x='transaction_count',
                y=top_users.index,
                data=top_users
            )
            
            # Add value labels
            for i, v in enumerate(top_users['transaction_count']):
                ax.text(v + 5, i, f"{v:,}", va='center')
            
            plt.title('Top Users by Transaction Count', fontsize=16)
            plt.xlabel('Number of Transactions', fontsize=14)
            plt.ylabel('User', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'top_users_transactions.png'), dpi=300)
            plt.close()
            
            # User activity by hour
            if 'hour' in df.columns:
                hour_activity = df.groupby('hour')['user'].nunique().reset_index()
                hour_activity.columns = ['hour', 'unique_users']
                
                plt.figure(figsize=(12, 8))
                ax = sns.barplot(x='hour', y='unique_users', data=hour_activity)
                
                plt.title('User Activity by Hour of Day', fontsize=16)
                plt.xlabel('Hour (24h format)', fontsize=14)
                plt.ylabel('Unique Active Users', fontsize=14)
                plt.xticks(range(0, 24))
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'user_activity_by_hour.png'), dpi=300)
                plt.close()
            
            # User activity by day of week
            if 'day_of_week' in df.columns:
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                day_activity = df.groupby('day_of_week')['user'].nunique()
                
                # Reorder days if all days are present
                if set(day_order).issubset(set(day_activity.index)):
                    day_activity = day_activity.reindex(day_order)
                
                day_activity = day_activity.reset_index()
                day_activity.columns = ['day_of_week', 'unique_users']
                
                plt.figure(figsize=(12, 8))
                ax = sns.barplot(x='day_of_week', y='unique_users', data=day_activity)
                
                plt.title('User Activity by Day of Week', fontsize=16)
                plt.xlabel('Day', fontsize=14)
                plt.ylabel('Unique Active Users', fontsize=14)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'user_activity_by_day.png'), dpi=300)
                plt.close()
            
            # User growth over time
            user_growth = df.groupby('date')['user'].nunique().cumsum().reset_index()
            user_growth.columns = ['date', 'cumulative_users']
            
            plt.figure(figsize=(14, 8))
            plt.plot(user_growth['date'], user_growth['cumulative_users'], marker='o', linestyle='-', linewidth=2)
            plt.title('Cumulative Unique Users Over Time', fontsize=16)
            plt.xlabel('Date', fontsize=14)
            plt.ylabel('Cumulative Unique Users', fontsize=14)
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'user_growth.png'), dpi=300)
            plt.close()
            
            results = {
                'top_users_by_transactions': top_users,
                'hourly_activity': hour_activity if 'hour_activity' in locals() else None,
                'daily_activity': day_activity if 'day_activity' in locals() else None,
                'user_growth': user_growth
            }
            
            self.analysis_results['user_activity'] = results
            return results
    
    def analyze_income_sources(self):
        """Analyze income sources and transaction categories"""
        with Timer("Analyzing income sources"):
            df = self.data['transactions']
            category_summary = self.data['category_summary']
            
            # Create visualization directory
            ensure_directory(self.output_dir)
            
            # If category_summary is empty or doesn't have required columns, recalculate
            if category_summary.empty or 'transaction_count' not in category_summary.columns:
                category_summary = df.groupby('category').agg({
                    'timestamp': 'count',
                    'cash_with_sign': 'sum',
                    'user': 'nunique'
                }).rename(columns={'timestamp': 'transaction_count', 'user': 'unique_users'})
            
            # Ensure we have a cash_with_sign column with valid data
            cash_column = None
            for col in ['cash_with_sign', 'net_cash', 'net_cash_flow', 'cash']:
                if col in category_summary.columns and not category_summary[col].isna().all():
                    cash_column = col
                    break
            
            # If we still don't have a valid cash column, calculate it from the dataframe
            if cash_column is None:
                logger.warning("No valid cash column found, calculating from transaction data")
                if 'cash_with_sign' in df.columns:
                    cash_by_category = df.groupby('category')['cash_with_sign'].sum()
                    category_summary['cash_with_sign'] = category_summary.index.map(lambda x: cash_by_category.get(x, 0))
                    cash_column = 'cash_with_sign'
                else:
                    logger.error("No cash column available in data")
                    cash_column = None
                    
            # Use the identified cash column for calculations
            if cash_column:
                if cash_column != 'cash_with_sign':
                    # Rename the column for consistency
                    category_summary['cash_with_sign'] = category_summary[cash_column]
            else:
                # Create a placeholder column if nothing is available
                logger.warning("Creating placeholder cash_with_sign column")
                category_summary['cash_with_sign'] = 0
            
            # Calculate average cash per transaction
            if 'transaction_count' in category_summary.columns:
                category_summary['avg_cash_per_transaction'] = category_summary['cash_with_sign'] / category_summary['transaction_count']
            else:
                logger.warning("Cannot calculate avg_cash_per_transaction: missing transaction_count column")
                category_summary['avg_cash_per_transaction'] = 0
                
            # Transaction counts by category
            cat_counts = category_summary.sort_values('transaction_count', ascending=False).copy()
            
            plt.figure(figsize=(14, 10))
            ax = sns.barplot(
                x='transaction_count',
                y=cat_counts.index,
                data=cat_counts
            )
            
            # Add value labels
            for i, v in enumerate(cat_counts['transaction_count']):
                ax.text(v + (cat_counts['transaction_count'].max() * 0.01), i, str(v), va='center')
            
            plt.title('Transaction Counts by Category', fontsize=16)
            plt.xlabel('Number of Transactions', fontsize=14)
            plt.ylabel('Category', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'transactions_by_category.png'), dpi=300)
            plt.close()
            
            # Cash flow by category - improved version focusing on game economy insights
            cat_cash = category_summary.sort_values('cash_with_sign', ascending=True).copy()  # Note: ascending=True to show most negative (largest expense) at top
            
            # Add game developer insights - separate income/expense and add color coding
            cat_cash['transaction_type'] = cat_cash['cash_with_sign'].apply(lambda x: 'Income' if x >= 0 else 'Expense')
            cat_cash['label_position'] = cat_cash['cash_with_sign'].apply(lambda x: x + (abs(cat_cash['cash_with_sign']).max() * 0.01) if x >= 0 else x - (abs(cat_cash['cash_with_sign']).max() * 0.05))
            
            plt.figure(figsize=(14, 10))
            # Use palette to distinguish between income/expense
            ax = sns.barplot(
                x='cash_with_sign',
                y=cat_cash.index,
                hue='transaction_type',
                palette={'Income': '#66BB6A', 'Expense': '#EF5350'},
                data=cat_cash
            )
            
            # Add value labels with different positioning for positive/negative
            for i, (v, t) in enumerate(zip(cat_cash['cash_with_sign'], cat_cash['transaction_type'])):
                if t == 'Income':
                    ax.text(v + (abs(cat_cash['cash_with_sign']).max() * 0.01), i, f"{v:,.0f}", va='center')
                else:
                    ax.text(v - (abs(cat_cash['cash_with_sign']).max() * 0.05), i, f"{v:,.0f}", va='center', ha='right', color='white')
            
            plt.title('Economy Balance by Category (Game Developer View)', fontsize=16)
            plt.xlabel('Net Cash Flow (- = Money Sink, + = Money Source)', fontsize=14)
            plt.ylabel('Category', fontsize=14)
            
            # Add vertical line at x=0 to clearly show income vs expense
            plt.axvline(x=0, color='black', linestyle='-', alpha=0.3)
            
            # Add annotations with game balance insights
            cash_sum = cat_cash['cash_with_sign'].sum()
            inflation_status = "INFLATION" if cash_sum > 0 else "DEFLATION" if cash_sum < 0 else "STABLE"
            plt.figtext(0.5, 0.01, f"Economy Status: {inflation_status} (Net: {cash_sum:,.0f})", 
                     ha="center", fontsize=12, 
                     bbox={"facecolor":"orange" if inflation_status != "STABLE" else "green", "alpha":0.2, "pad":5})
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.97])  # Make room for the annotation
            plt.savefig(os.path.join(self.output_dir, 'cash_flow_by_category.png'), dpi=300)
            plt.close()
            
            # Average transaction value by category - modify to be more useful
            # Skip this chart as it's less useful unless specifically requested
            
            # Transaction types (credit vs debit)
            if 'cash_type' in df.columns:
                type_counts = df['cash_type'].value_counts()
                
                plt.figure(figsize=(10, 7))
                
                # Create colors list
                colors = ['#66BB6A', '#EF5350']  # Green for credit, red for debit
                
                # Create dynamic explode parameter
                explode = [0.05] * len(type_counts)
                
                plt.pie(
                    type_counts, 
                    labels=type_counts.index, 
                    autopct='%1.1f%%', 
                    colors=colors[:len(type_counts)],
                    startangle=90, 
                    explode=explode, 
                    shadow=True, 
                    textprops={'fontsize': 14}
                )
                
                plt.axis('equal')
                plt.title('Transaction Types: Credits vs Debits', fontsize=16)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'transaction_types.png'), dpi=300)
                plt.close()
            
            # Top income sources over time
            top_categories = category_summary.sort_values('transaction_count', ascending=False).head(5).index.tolist()
            
            # Filter for top categories
            cat_time = df[df['category'].isin(top_categories)].copy()
            
            if not cat_time.empty:
                cat_time_grouped = cat_time.groupby(['date', 'category'])['cash_with_sign'].sum().reset_index()
                
                # Pivot for plotting
                cat_time_pivot = cat_time_grouped.pivot(index='date', columns='category', values='cash_with_sign').fillna(0)
                
                plt.figure(figsize=(14, 8))
                for category in cat_time_pivot.columns:
                    plt.plot(cat_time_pivot.index, cat_time_pivot[category], label=category, linewidth=2)
                
                plt.title('Cash Flow by Top Categories Over Time', fontsize=16)
                plt.xlabel('Date', fontsize=14)
                plt.ylabel('Daily Cash Flow', fontsize=14)
                plt.legend(loc='best')
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'category_flow_over_time.png'), dpi=300)
                plt.close()
            
            results = {
                'category_counts': cat_counts,
                'category_cash_flow': cat_cash,
                'category_avg_value': category_summary.sort_values('avg_cash_per_transaction', ascending=False).copy(),
                'type_distribution': type_counts if 'type_counts' in locals() else None,
                'category_time_series': cat_time_pivot if 'cat_time_pivot' in locals() else None
            }
            
            self.analysis_results['income_sources'] = results
            return results
    
    def analyze_economy_health(self):
        """Analyze overall economy health metrics"""
        with Timer("Analyzing economy health"):
            df = self.data['transactions']
            daily_summary = self.data['daily_summary']
            
            # Create visualization directory
            ensure_directory(self.output_dir)
            
            # Calculate daily metrics if not available
            if daily_summary.empty and not df.empty:
                logger.info("Calculating daily summary from transaction data")
                daily_summary = df.groupby('date').agg({
                    'timestamp': 'count',
                    'cash_with_sign': 'sum',
                    'user': 'nunique'
                }).rename(columns={'timestamp': 'transaction_count', 'user': 'active_users'})
                daily_summary = daily_summary.reset_index()
            
            # Calculate economy health metrics
            if not daily_summary.empty:
                daily_metrics = daily_summary.copy()
                daily_metrics['cumulative_cash'] = daily_metrics['cash_with_sign'].cumsum()
                
                # Calculate running metrics
                daily_metrics['avg_per_transaction'] = daily_metrics['cash_with_sign'] / daily_metrics['transaction_count']
                
                # Plot metrics
                fig, axes = plt.subplots(2, 2, figsize=(16, 12))
                
                # Plot 1: Cumulative Net Change
                axes[0, 0].plot(daily_metrics['date'], daily_metrics['cumulative_cash'], 
                              marker='o', linestyle='-', color='green', linewidth=2)
                axes[0, 0].set_title('Cumulative Net Change in Economy', fontsize=14)
                axes[0, 0].set_xlabel('Date', fontsize=12)
                axes[0, 0].set_ylabel('Cumulative Change', fontsize=12)
                axes[0, 0].grid(True, alpha=0.3)
                
                # Plot 2: Daily Transaction Volume
                axes[0, 1].plot(daily_metrics['date'], daily_metrics['transaction_count'], 
                              marker='o', linestyle='-', color='blue', linewidth=2)
                axes[0, 1].set_title('Daily Transaction Count', fontsize=14)
                axes[0, 1].set_xlabel('Date', fontsize=12)
                axes[0, 1].set_ylabel('Number of Transactions', fontsize=12)
                axes[0, 1].grid(True, alpha=0.3)
                
                # Plot 3: Daily Cash Flow
                axes[1, 0].plot(daily_metrics['date'], daily_metrics['cash_with_sign'], 
                              marker='o', linestyle='-', color='purple', linewidth=2)
                axes[1, 0].set_title('Daily Net Cash Flow', fontsize=14)
                axes[1, 0].set_xlabel('Date', fontsize=12)
                axes[1, 0].set_ylabel('Net Cash Flow', fontsize=12)
                axes[1, 0].grid(True, alpha=0.3)
                
                # Plot 4: Active Users
                axes[1, 1].plot(daily_metrics['date'], daily_metrics['active_users'], 
                              marker='o', linestyle='-', color='orange', linewidth=2)
                axes[1, 1].set_title('Daily Active Users', fontsize=14)
                axes[1, 1].set_xlabel('Date', fontsize=12)
                axes[1, 1].set_ylabel('Unique Active Users', fontsize=12)
                axes[1, 1].grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'economy_health_metrics.png'), dpi=300)
                plt.close()
            
            # Calculate wealth distribution
            user_wealth = df.groupby('user')['cash_with_sign'].sum().sort_values(ascending=False)
            
            # Create wealth quintiles
            quintiles = []
            user_count = len(user_wealth)
            
            if user_count > 0:
                quintile_size = max(1, user_count // 5)
                
                top_20 = user_wealth.iloc[:quintile_size].sum()
                next_20 = user_wealth.iloc[quintile_size:quintile_size*2].sum()
                middle_20 = user_wealth.iloc[quintile_size*2:quintile_size*3].sum()
                fourth_20 = user_wealth.iloc[quintile_size*3:quintile_size*4].sum()
                bottom_20 = user_wealth.iloc[quintile_size*4:].sum()
                
                quintiles = [
                    {'group': 'Top 20%', 'wealth': top_20},
                    {'group': 'Next 20%', 'wealth': next_20},
                    {'group': 'Middle 20%', 'wealth': middle_20},
                    {'group': 'Fourth 20%', 'wealth': fourth_20},
                    {'group': 'Bottom 20%', 'wealth': bottom_20}
                ]
                
                quintiles_df = pd.DataFrame(quintiles)
                
                plt.figure(figsize=(12, 8))
                ax = sns.barplot(
                    x='group',
                    y='wealth',
                    data=quintiles_df,
                    hue='group',
                    legend=False
                )
                
                # Add value labels
                for i, v in enumerate(quintiles_df['wealth']):
                    ax.text(i, v + (quintiles_df['wealth'].max() * 0.01) if v > 0 else v - (quintiles_df['wealth'].abs().max() * 0.05), 
                           f"{v:,.0f}", ha='center')
                
                plt.title('Wealth Distribution by Quintile', fontsize=16)
                plt.xlabel('Wealth Group', fontsize=14)
                plt.ylabel('Total Wealth', fontsize=14)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'wealth_distribution.png'), dpi=300)
                plt.close()
            
            # Calculate Gini coefficient
            gini = 0
            if len(user_wealth) > 1:
                # Sort wealth values
                wealth_values = user_wealth.values
                wealth_values = np.sort(wealth_values)
                
                # Calculate Gini coefficient
                n = len(wealth_values)
                index = np.arange(1, n + 1)
                gini = (np.sum((2 * index - n - 1) * wealth_values)) / (n * np.sum(wealth_values))
            
            # Lorenz curve
            if len(user_wealth) > 1:
                wealth_values = user_wealth.values
                wealth_sorted = np.sort(wealth_values)
                wealth_cumsum = np.cumsum(wealth_sorted)
                wealth_total = wealth_cumsum[-1]
                
                if wealth_total > 0:  # Avoid division by zero
                    lorenz_curve = wealth_cumsum / wealth_total
                    
                    plt.figure(figsize=(10, 8))
                    
                    # Plot perfect equality line
                    plt.plot([0, 1], [0, 1], '--', color='gray', label='Perfect Equality')
                    
                    # Plot Lorenz curve
                    plt.plot(np.arange(len(lorenz_curve)) / len(lorenz_curve), lorenz_curve, color='red', label=f'Lorenz Curve (Gini={gini:.3f})')
                    
                    plt.title('Lorenz Curve of Wealth Distribution', fontsize=16)
                    plt.xlabel('Cumulative Share of Players', fontsize=14)
                    plt.ylabel('Cumulative Share of Wealth', fontsize=14)
                    plt.legend()
                    plt.grid(True, alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.output_dir, 'lorenz_curve.png'), dpi=300)
                    plt.close()
            
            results = {
                'daily_metrics': daily_metrics if 'daily_metrics' in locals() else None,
                'wealth_quintiles': quintiles if 'quintiles' in locals() else [],
                'gini_coefficient': gini
            }
            
            self.analysis_results['economy_health'] = results
            return results
    
    def analyze_user_progression(self):
        """Analyze user progression and balance histories"""
        with Timer("Analyzing user progression"):
            df = self.data['transactions']
            
            # Create visualization directory
            ensure_directory(self.output_dir)
            
            # Get top 10 most active users
            top_users = df['user'].value_counts().head(10).index.tolist()
            
            # Calculate cumulative balances for each user
            user_balances = []
            
            for user in top_users:
                user_df = df[df['user'] == user].sort_values('timestamp').copy()
                if not user_df.empty:
                    user_df['cumulative_balance'] = user_df['cash_with_sign'].cumsum()
                    user_df['user_name'] = user  # Add user name for plotting
                    user_balances.append(user_df)
            
            if user_balances:
                # Combine data for plotting
                all_balances = pd.concat(user_balances)
                
                # Plot balance progression
                plt.figure(figsize=(15, 10))
                
                for user in top_users:
                    user_data = all_balances[all_balances['user_name'] == user]
                    if not user_data.empty:
                        plt.plot(user_data['timestamp'], user_data['cumulative_balance'], 
                               marker='o', markersize=3, linewidth=2, label=user)
                
                plt.title('Balance Progression for Top 10 Most Active Users', fontsize=16)
                plt.xlabel('Date', fontsize=14)
                plt.ylabel('Cumulative Balance', fontsize=14)
                plt.grid(True, alpha=0.3)
                plt.legend(loc='best')
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'top_users_progression.png'), dpi=300)
                plt.close()
            
            # Create activity heatmap
            if 'hour' in df.columns and 'day_of_week' in df.columns:
                day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                activity_pivot = pd.pivot_table(
                    df, 
                    values='timestamp', 
                    index='hour',
                    columns='day_of_week',
                    aggfunc='count',
                    fill_value=0
                )
                
                # Reorder columns to standard week days
                if set(day_order).issubset(set(activity_pivot.columns)):
                    activity_pivot = activity_pivot.reindex(columns=day_order)
                
                plt.figure(figsize=(14, 10))
                sns.heatmap(activity_pivot, cmap='YlGnBu', annot=True, fmt='g')
                plt.title('Activity Heatmap by Day and Hour', fontsize=16)
                plt.xlabel('Day of Week', fontsize=14)
                plt.ylabel('Hour of Day', fontsize=14)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'activity_heatmap.png'), dpi=300)
                plt.close()
            
            # Average earnings by time
            if 'hour' in df.columns:
                hourly_earnings = df.groupby('hour')['cash_with_sign'].mean().reset_index()
                
                plt.figure(figsize=(14, 8))
                ax = sns.barplot(
                    x='hour',
                    y='cash_with_sign',
                    data=hourly_earnings,
                    hue='hour',
                    legend=False
                )
                
                plt.title('Average Cash Flow by Hour of Day', fontsize=16)
                plt.xlabel('Hour (24h format)', fontsize=14)
                plt.ylabel('Average Cash Flow', fontsize=14)
                plt.xticks(range(0, 24))
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'hourly_earnings.png'), dpi=300)
                plt.close()
            
            results = {
                'top_users_progression': all_balances if 'all_balances' in locals() else None,
                'activity_heatmap': activity_pivot if 'activity_pivot' in locals() else None,
                'hourly_earnings': hourly_earnings if 'hourly_earnings' in locals() else None
            }
            
            self.analysis_results['user_progression'] = results
            return results
            
    def analyze_player_retention(self):
        """Analyze player retention metrics for game developers"""
        with Timer("Analyzing player retention"):
            df = self.data['transactions']
            
            # Create visualization directory
            ensure_directory(self.output_dir)
            
            # Need timestamps for retention analysis
            if 'timestamp' not in df.columns or df.empty:
                logger.error("Cannot analyze player retention: Missing timestamp data")
                return None
            
            # Calculate first and last activity dates for each user
            user_activity = df.groupby('user').agg({
                'timestamp': ['min', 'max'],
                'cash_with_sign': ['count', 'sum']
            })
            
            user_activity.columns = ['first_activity', 'last_activity', 'transaction_count', 'net_cash_flow']
            
            # Calculate days active (span between first and last activity)
            user_activity['days_active'] = (user_activity['last_activity'] - user_activity['first_activity']).dt.days
            
            # For users with only one transaction, set days_active to 0
            user_activity.loc[user_activity['days_active'].isna(), 'days_active'] = 0
            
            # Calculate retention status - active in last 7 days?
            max_date = df['timestamp'].max()
            user_activity['active_last_7_days'] = (max_date - user_activity['last_activity']).dt.days <= 7
            user_activity['active_last_30_days'] = (max_date - user_activity['last_activity']).dt.days <= 30
            
            # Calculate churn categories based on last activity
            user_activity['churn_status'] = pd.cut(
                (max_date - user_activity['last_activity']).dt.days,
                bins=[-1, 7, 30, 60, 90, float('inf')],
                labels=['Active', 'Inactive (7-30d)', 'Inactive (30-60d)', 'Inactive (60-90d)', 'Churned']
            )
            
            # Calculate user lifetime values
            user_activity['lifetime_value'] = user_activity['net_cash_flow']
            
            # Calculate retention rates by date cohort
            # Group users by month of first activity
            user_activity['first_month'] = user_activity['first_activity'].dt.to_period('M')
            
            # Calculate cohort retention rates
            cohort_data = user_activity.groupby('first_month').agg({
                'user': 'size',  # Count of users in cohort
                'active_last_30_days': 'sum'  # Count of retained users
            }).rename(columns={'user': 'cohort_size', 'active_last_30_days': 'retained_users'})
            
            cohort_data['retention_rate'] = cohort_data['retained_users'] / cohort_data['cohort_size'] * 100
            
            # Plot retention by cohort
            plt.figure(figsize=(14, 8))
            ax = sns.barplot(
                x=cohort_data.index.astype(str),
                y='retention_rate',
                data=cohort_data.reset_index()
            )
            
            plt.title('30-Day Retention Rate by User Cohort', fontsize=16)
            plt.xlabel('First Activity Month', fontsize=14)
            plt.ylabel('Retention Rate (%)', fontsize=14)
            plt.xticks(rotation=45)
            plt.axhline(y=cohort_data['retention_rate'].mean(), color='r', linestyle='--', label=f'Avg: {cohort_data["retention_rate"].mean():.1f}%')
            plt.legend()
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'retention_by_cohort.png'), dpi=300)
            plt.close()
            
            # Plot user churn status
            churn_counts = user_activity['churn_status'].value_counts().sort_index()
            
            plt.figure(figsize=(12, 8))
            ax = sns.barplot(
                x=churn_counts.index,
                y=churn_counts.values
            )
            
            # Add percentage labels
            total = churn_counts.sum()
            for i, v in enumerate(churn_counts):
                ax.text(i, v + 5, f"{v} ({v/total*100:.1f}%)", ha='center')
            
            plt.title('Player Activity Status', fontsize=16)
            plt.xlabel('Status', fontsize=14)
            plt.ylabel('Number of Players', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'player_churn_status.png'), dpi=300)
            plt.close()
            
            # Plot player longevity - days active distribution
            plt.figure(figsize=(14, 8))
            # Use distplot with KDE
            sns.histplot(user_activity['days_active'], kde=True, bins=30)
            plt.title('Player Longevity Distribution', fontsize=16)
            plt.xlabel('Days Active (time between first and last activity)', fontsize=14)
            plt.ylabel('Number of Players', fontsize=14)
            
            # Add median and mean lines
            median_days = user_activity['days_active'].median()
            mean_days = user_activity['days_active'].mean()
            plt.axvline(median_days, color='r', linestyle='--', label=f'Median: {median_days:.1f} days')
            plt.axvline(mean_days, color='g', linestyle='--', label=f'Mean: {mean_days:.1f} days')
            plt.legend()
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'player_longevity.png'), dpi=300)
            plt.close()
            
            # Calculate game developer insights
            results = {
                'active_players_7d': user_activity['active_last_7_days'].sum(),
                'active_players_30d': user_activity['active_last_30_days'].sum(),
                'total_players': len(user_activity),
                'avg_player_lifespan': mean_days,
                'median_player_lifespan': median_days,
                'retention_by_cohort': cohort_data,
                'avg_retention_rate': cohort_data['retention_rate'].mean(),
                'churn_rates': (churn_counts / total * 100).to_dict(),
                'player_data': user_activity
            }
            
            self.analysis_results['player_retention'] = results
            return results
    
    def generate_html_report(self):
        """Generate a comprehensive HTML report"""
        with Timer("Generating HTML report"):
            # Only generate report if we have analysis results
            if not self.analysis_results:
                logger.warning("No analysis results available for report generation")
                return
            
            # Create output directory
            ensure_directory(self.output_dir)
            
            # HTML report path
            report_path = os.path.join(self.output_dir, 'economy_analysis_report.html')
            
            # Prepare report template
            html_template = """
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Economy Analysis Report</title>
                <style>
                    body {
                        font-family: Arial, sans-serif;
                        line-height: 1.6;
                        margin: 0;
                        padding: 0;
                        color: #333;
                    }
                    .container {
                        width: 90%;
                        max-width: 1200px;
                        margin: 0 auto;
                        padding: 20px;
                    }
                    h1, h2, h3, h4 {
                        color: #2c3e50;
                    }
                    h1 {
                        border-bottom: 2px solid #2c3e50;
                        padding-bottom: 10px;
                    }
                    h2 {
                        margin-top: 30px;
                        border-bottom: 1px solid #ddd;
                        padding-bottom: 5px;
                    }
                    .section {
                        margin-bottom: 40px;
                    }
                    .metrics-grid {
                        display: grid;
                        grid-template-columns: repeat(auto-fill, minmax(300px, 1fr));
                        gap: 20px;
                        margin: 20px 0;
                    }
                    .metric-card {
                        background-color: #f8f9fa;
                        border-radius: 5px;
                        padding: 15px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                    }
                    .metric-title {
                        font-size: 16px;
                        font-weight: bold;
                        margin-bottom: 10px;
                        color: #2c3e50;
                    }
                    .metric-value {
                        font-size: 24px;
                        font-weight: bold;
                        color: #3498db;
                    }
                    .metric-description {
                        font-size: 14px;
                        color: #7f8c8d;
                        margin-top: 5px;
                    }
                    img {
                        max-width: 100%;
                        height: auto;
                        margin: 20px 0;
                        border: 1px solid #ddd;
                        border-radius: 5px;
                    }
                    .image-pair {
                        display: grid;
                        grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                        gap: 20px;
                        margin: 20px 0;
                    }
                    table {
                        width: 100%;
                        border-collapse: collapse;
                        margin: 20px 0;
                    }
                    th, td {
                        padding: 12px 15px;
                        text-align: left;
                        border-bottom: 1px solid #ddd;
                    }
                    th {
                        background-color: #f2f2f2;
                        font-weight: bold;
                    }
                    tr:hover {
                        background-color: #f5f5f5;
                    }
                    .insights {
                        background-color: #e8f4f8;
                        border-left: 4px solid #3498db;
                        padding: 15px;
                        margin: 20px 0;
                        border-radius: 0 5px 5px 0;
                    }
                    .insights h3 {
                        margin-top: 0;
                        color: #3498db;
                    }
                    .warning {
                        background-color: #fff3e0;
                        border-left: 4px solid #ff9800;
                        padding: 15px;
                        margin: 20px 0;
                        border-radius: 0 5px 5px 0;
                    }
                    .footer {
                        margin-top: 50px;
                        padding-top: 20px;
                        border-top: 1px solid #ddd;
                        font-size: 14px;
                        color: #7f8c8d;
                        text-align: center;
                    }
                    .developer-card {
                        background-color: #e8f6e8;
                        border-radius: 5px;
                        padding: 15px;
                        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
                        border-left: 4px solid #27ae60;
                        margin: 20px 0;
                    }
                    .developer-title {
                        font-size: 18px;
                        font-weight: bold;
                        color: #27ae60;
                        margin-bottom: 10px;
                    }
                </style>
            </head>
            <body>
                <div class="container">
                    <h1>Discord Game Economy Analysis Report</h1>
                    <p>Analysis generated on {timestamp}</p>

                    <div class="section">
                        <h2>Economy Health Overview</h2>
                        <div class="metrics-grid">
                            <div class="metric-card">
                                <div class="metric-title">Total Transactions</div>
                                <div class="metric-value">{total_transactions:,}</div>
                                <div class="metric-description">Number of economy transactions analyzed</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-title">Unique Users</div>
                                <div class="metric-value">{unique_users:,}</div>
                                <div class="metric-description">Distinct players in the economy</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-title">Total Cash Volume</div>
                                <div class="metric-value">{total_cash_volume:,.0f}</div>
                                <div class="metric-description">Total currency flowing through economy</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-title">Net Cash Change</div>
                                <div class="metric-value">{net_cash_change:,.0f}</div>
                                <div class="metric-description">Net currency added/removed (+ means inflation)</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-title">Gini Coefficient</div>
                                <div class="metric-value">{gini_coefficient:.3f}</div>
                                <div class="metric-description">Measure of wealth inequality (0=equal, 1=unequal)</div>
                            </div>
                            <div class="metric-card">
                                <div class="metric-title">Active Players (7d)</div>
                                <div class="metric-value">{active_players_7d}</div>
                                <div class="metric-description">Players active in the last 7 days</div>
                            </div>
                        </div>

                        {developer_insights}

                        <h3>Economy Health Metrics</h3>
                        <div class="image-pair">
                            <img src="economy_health_over_time.png" alt="Economy Health Over Time">
                            <img src="wealth_distribution.png" alt="Wealth Distribution">
                        </div>
                    </div>

                    <div class="section">
                        <h2>User Activity Analysis</h2>
                        <div class="image-pair">
                            <img src="top_users_transactions.png" alt="Top Users by Transaction Count">
                            <img src="user_growth.png" alt="User Growth Over Time">
                        </div>

                        <h3>Player Activity Patterns</h3>
                        <div class="image-pair">
                            <img src="user_activity_by_hour.png" alt="User Activity by Hour" onerror="this.style.display='none'">
                            <img src="user_activity_by_day.png" alt="User Activity by Day" onerror="this.style.display='none'">
                        </div>
                    </div>

                    <div class="section">
                        <h2>Transaction Categories Analysis</h2>
                        <div class="image-pair">
                            <img src="transactions_by_category.png" alt="Transactions by Category">
                            <img src="cash_flow_by_category.png" alt="Cash Flow by Category">
                        </div>
                    </div>

                    <div class="section">
                        <h2>Player Progression Analysis</h2>
                        <div class="image-pair">
                            <img src="wealth_progression.png" alt="Wealth Progression" onerror="this.style.display='none'">
                            <img src="user_progression_tiers.png" alt="User Progression Tiers" onerror="this.style.display='none'">
                        </div>
                    </div>

                    {player_retention_section}

                    <div class="footer">
                        <p>Report generated using Discord Game Economy Analysis Tool</p>
                        <p>Analysis Date: {timestamp}</p>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # Get statistics
            stats = self.data.get('stats', {})
            
            # Default values for metrics
            metrics = {
                'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'total_transactions': stats.get('total_transactions', 0),
                'unique_users': stats.get('unique_users', 0),
                'total_cash_volume': stats.get('total_cash_volume', 0),
                'net_cash_change': stats.get('net_cash_change', 0),
                'gini_coefficient': 0.5,  # Default if not calculated
                'active_players_7d': 0
            }
            
            # Get economy health metrics if available
            if 'economy_health' in self.analysis_results:
                econ_health = self.analysis_results['economy_health']
                if 'gini_coefficient' in econ_health:
                    metrics['gini_coefficient'] = econ_health['gini_coefficient']
            
            # Get player retention metrics if available
            if 'player_retention' in self.analysis_results:
                retention = self.analysis_results['player_retention']
                metrics['active_players_7d'] = retention.get('active_players_7d', 0)
                
                # Create retention section for HTML
                player_retention_html = f"""
                <div class="section">
                    <h2>Player Retention Analysis</h2>
                    <div class="metrics-grid">
                        <div class="metric-card">
                            <div class="metric-title">Active Players (7d)</div>
                            <div class="metric-value">{retention.get('active_players_7d', 0)}</div>
                            <div class="metric-description">Players active in the last 7 days</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-title">Active Players (30d)</div>
                            <div class="metric-value">{retention.get('active_players_30d', 0)}</div>
                            <div class="metric-description">Players active in the last 30 days</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-title">Total Players</div>
                            <div class="metric-value">{retention.get('total_players', 0)}</div>
                            <div class="metric-description">Total players in the dataset</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-title">Average Player Lifespan</div>
                            <div class="metric-value">{retention.get('avg_player_lifespan', 0):.1f} days</div>
                            <div class="metric-description">Average time between first and last activity</div>
                        </div>
                        <div class="metric-card">
                            <div class="metric-title">Average Retention Rate</div>
                            <div class="metric-value">{retention.get('avg_retention_rate', 0):.1f}%</div>
                            <div class="metric-description">Average 30-day retention rate across cohorts</div>
                        </div>
                    </div>
                    
                    <div class="image-pair">
                        <img src="retention_by_cohort.png" alt="Retention by Cohort" onerror="this.style.display='none'">
                        <img src="player_churn_status.png" alt="Player Churn Status" onerror="this.style.display='none'">
                    </div>
                    
                    <img src="player_longevity.png" alt="Player Longevity Distribution" onerror="this.style.display='none'">
                </div>
                """
            else:
                player_retention_html = ""
            
            # Create game developer insights section
            developer_insights_html = ""
            if 'income_sources' in self.analysis_results and 'player_retention' in self.analysis_results:
                income_sources = self.analysis_results['income_sources']
                retention = self.analysis_results['player_retention']
                
                # Calculate average currency per transaction
                avg_transaction_value = metrics['total_cash_volume'] / metrics['total_transactions'] if metrics['total_transactions'] > 0 else 0
                
                # Calculate economy insights
                is_inflation = metrics['net_cash_change'] > 0
                inflation_rate = (metrics['net_cash_change'] / metrics['total_cash_volume']) * 100 if metrics['total_cash_volume'] > 0 else 0
                retention_rate = retention.get('avg_retention_rate', 0)
                
                developer_insights_html = f"""
                <div class="developer-card">
                    <div class="developer-title">Game Developer Insights</div>
                    
                    <h3>Economy Balance</h3>
                    <p>The economy is currently experiencing <strong>{"inflation" if is_inflation else "deflation"}</strong> 
                       at a rate of <strong>{abs(inflation_rate):.1f}%</strong>. 
                       This means players are {"earning" if is_inflation else "spending"} more currency than they are {"spending" if is_inflation else "earning"}.</p>
                    
                    <h3>Player Engagement</h3>
                    <p>Your game has a <strong>{retention_rate:.1f}%</strong> 30-day retention rate. 
                       Players remain active for an average of <strong>{retention.get('avg_player_lifespan', 0):.1f} days</strong>.</p>
                    
                    <h3>Transaction Patterns</h3>
                    <p>The average transaction value is <strong>{avg_transaction_value:.1f}</strong> currency units.</p>
                    
                    <h3>Recommendations</h3>
                    <ul>
                        {"<li>Consider adding more money sinks to balance the economy and reduce inflation.</li>" if is_inflation else 
                         "<li>Consider adding more money sources or reducing costs to prevent economic stagnation.</li>"}
                        {"<li>Review player retention strategies as your retention rate is below typical benchmark of 15-25%.</li>" if retention_rate < 15 else ""}
                        {"<li>Economy appears relatively balanced with good player retention.</li>" if not is_inflation and retention_rate >= 15 else ""}
                    </ul>
                </div>
                """
            
            # Format HTML template with metrics
            html_content = html_template.format(
                **metrics,
                developer_insights=developer_insights_html,
                player_retention_section=player_retention_html
            )
            
            # Write report
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"HTML report generated: {report_path}")
            return report_path
    
    def run_analysis(self):
        """Run the complete economy analysis pipeline"""
        logger.info("Starting economy analysis...")
        
        # Load data
        if not self.load_data():
            logger.error("Failed to load economy data")
            return False
        
        # Run analyses
        self.analyze_user_activity()
        self.analyze_income_sources()
        self.analyze_economy_health()
        self.analyze_user_progression()
        self.analyze_player_retention()
        
        # Generate HTML report
        report_path = self.generate_html_report()
        
        # Try to open the report
        try:
            import webbrowser
            webbrowser.open('file://' + os.path.abspath(report_path))
            logger.info(f"Opened report in browser: {report_path}")
        except Exception as e:
            logger.error(f"Could not open report automatically: {e}")
        
        logger.info("Economy analysis complete!")
        return True

def run_economy_analysis(data_dir='balance_data', output_dir='economy_analysis', show_all_players=False, developer_view=False):
    """Run the complete economy analysis pipeline
    
    Args:
        data_dir (str): Directory with extracted data
        output_dir (str): Directory to save analysis results
        show_all_players (bool): If True, show all players instead of just top players
        developer_view (bool): If True, focus on game developer metrics
        
    Returns:
        bool: True if analysis completed successfully, False otherwise
    """
    try:
        # Create analyzer instance
        analyzer = EconomyAnalyzer(data_dir, output_dir)
        
        # Update config if showing all players
        if show_all_players:
            # Override the top_n_users setting to include all users
            from gameanalytics.config import ANALYSIS_CONFIG
            ANALYSIS_CONFIG['top_n_users'] = 1000  # Set to a large number
        
        # Load data
        if not analyzer.load_data():
            logger.error("Failed to load economy data")
            return False
        
        # Modify the run_analysis method to prioritize developer-focused metrics
        if developer_view:
            # Run developer-focused analyses
            logger.info("Running economy analysis with focus on game developer metrics")
            analyzer.analyze_income_sources()  # Enhanced with economy balance view
            analyzer.analyze_player_retention()  # Added player retention metrics
            analyzer.analyze_economy_health()  # Core economy health metrics
            analyzer.analyze_user_activity()  # User engagement patterns
            analyzer.analyze_user_progression()  # Player progression metrics
        else:
            # Run standard analyses
            analyzer.run_analysis()
        
        # Generate report
        analyzer.generate_html_report()
        
        return True
    except Exception as e:
        logger.error(f"Error in economy analysis: {e}")
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    # Get data directory from command line if provided
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = 'balance_data'
    
    run_economy_analysis(data_dir) 