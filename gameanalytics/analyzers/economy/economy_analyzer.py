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
                
                # Check if cash_with_sign exists, if not use net_cash if available
                if 'cash_with_sign' not in category_summary.columns and 'net_cash' in category_summary.columns:
                    category_summary['cash_with_sign'] = category_summary['net_cash']
                elif 'cash_with_sign' not in category_summary.columns and 'net_cash_flow' in category_summary.columns:
                    category_summary['cash_with_sign'] = category_summary['net_cash_flow']
                # As a last resort, calculate from the dataframe
                elif 'cash_with_sign' not in category_summary.columns:
                    logger.warning("cash_with_sign column missing, calculating from transaction data")
                    cash_by_category = df.groupby('category')['cash_with_sign'].sum()
                    category_summary['cash_with_sign'] = category_summary.index.map(lambda x: cash_by_category.get(x, 0))
            
            # Ensure we have the cash_with_sign column
            if 'cash_with_sign' not in category_summary.columns:
                logger.warning("cash_with_sign column not found in category_summary, using placeholder values")
                category_summary['cash_with_sign'] = 0
            
            # Calculate average cash per transaction
            if 'transaction_count' in category_summary.columns and 'cash_with_sign' in category_summary.columns:
                category_summary['avg_cash_per_transaction'] = category_summary['cash_with_sign'] / category_summary['transaction_count']
            else:
                logger.warning("Cannot calculate avg_cash_per_transaction: missing required columns")
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
            
            # Cash flow by category - only create this chart if cash_with_sign column exists
            if 'cash_with_sign' in category_summary.columns:
                cat_cash = category_summary.sort_values('cash_with_sign', ascending=False).copy()
                
                plt.figure(figsize=(14, 10))
                ax = sns.barplot(
                    x='cash_with_sign',
                    y=cat_cash.index,
                    data=cat_cash
                )
                
                # Add value labels
                for i, v in enumerate(cat_cash['cash_with_sign']):
                    ax.text(v + (abs(cat_cash['cash_with_sign']).max() * 0.01), i, f"{v:,.0f}", va='center')
                
                plt.title('Net Cash Flow by Category', fontsize=16)
                plt.xlabel('Net Cash Flow', fontsize=14)
                plt.ylabel('Category', fontsize=14)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'cash_flow_by_category.png'), dpi=300)
                plt.close()
            else:
                logger.error("Cannot create cash flow chart: 'cash_with_sign' column missing from category summary")
            
            # Average transaction value by category
            cat_avg = category_summary.sort_values('avg_cash_per_transaction', ascending=False).copy()
            
            plt.figure(figsize=(14, 10))
            ax = sns.barplot(
                x='avg_cash_per_transaction',
                y=cat_avg.index,
                data=cat_avg
            )
            
            # Add value labels
            for i, v in enumerate(cat_avg['avg_cash_per_transaction']):
                ax.text(v + (cat_avg['avg_cash_per_transaction'].max() * 0.01), i, f"{v:,.1f}", va='center')
            
            plt.title('Average Transaction Value by Category', fontsize=16)
            plt.xlabel('Average Value', fontsize=14)
            plt.ylabel('Category', fontsize=14)
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'avg_value_by_category.png'), dpi=300)
            plt.close()
            
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
                'category_avg_value': cat_avg,
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
            
    def generate_html_report(self):
        """Generate a comprehensive HTML report"""
        with Timer("Generating HTML report"):
            # Extract stats and analysis results
            stats = self.data['stats']
            total_transactions = stats.get('total_transactions', 0)
            unique_users = stats.get('unique_users', 0)
            date_range = stats.get('date_range', 'Unknown')
            total_cash_volume = stats.get('total_cash_volume', 0)
            net_cash_change = stats.get('net_cash_change', 0)
            
            # Extract analysis results
            user_activity = self.analysis_results.get('user_activity', {})
            income_sources = self.analysis_results.get('income_sources', {})
            economy_health = self.analysis_results.get('economy_health', {})
            user_progression = self.analysis_results.get('user_progression', {})
            
            # Gini coefficient
            gini = economy_health.get('gini_coefficient', 0)
            
            # Create HTML content
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Discord Economy Analysis Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                    h1 {{ color: #2c3e50; }}
                    h2 {{ color: #3498db; margin-top: 40px; }}
                    h3 {{ color: #7f8c8d; margin-top: 30px; }}
                    .stats-container {{ display: flex; flex-wrap: wrap; }}
                    .stat-box {{ 
                        background-color: #f8f9fa; 
                        border-radius: 8px; 
                        padding: 15px; 
                        margin: 10px; 
                        min-width: 200px;
                        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
                    }}
                    .stat-title {{ font-weight: bold; color: #7f8c8d; }}
                    .stat-value {{ font-size: 24px; color: #2c3e50; margin: 10px 0; }}
                    .good {{ color: #27ae60; }}
                    .bad {{ color: #e74c3c; }}
                    .neutral {{ color: #f39c12; }}
                    .gallery {{ 
                        display: grid; 
                        grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
                        gap: 20px;
                        margin-top: 30px;
                    }}
                    .chart {{ 
                        width: 100%; 
                        border-radius: 8px; 
                        box-shadow: 0 4px 8px rgba(0,0,0,0.1); 
                    }}
                    .insight {{ 
                        background-color: #e8f4f8; 
                        border-left: 4px solid #3498db; 
                        padding: 15px;
                        margin: 20px 0;
                        border-radius: 0 8px 8px 0;
                    }}
                    .recommendation {{ 
                        background-color: #eafaf1; 
                        border-left: 4px solid #27ae60; 
                        padding: 15px;
                        margin: 20px 0;
                        border-radius: 0 8px 8px 0;
                    }}
                    .warning {{ 
                        background-color: #fef9e7; 
                        border-left: 4px solid #f39c12; 
                        padding: 15px;
                        margin: 20px 0;
                        border-radius: 0 8px 8px 0;
                    }}
                </style>
            </head>
            <body>
                <h1>Discord Economy Analysis Report</h1>
                <p>This report provides a comprehensive analysis of the Discord economy game data.</p>
                
                <h2>Executive Summary</h2>
                <div class="stats-container">
                    <div class="stat-box">
                        <div class="stat-title">Total Transactions</div>
                        <div class="stat-value">{total_transactions:,}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-title">Unique Users</div>
                        <div class="stat-value">{unique_users:,}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-title">Total Economy Volume</div>
                        <div class="stat-value">{total_cash_volume:,.2f}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-title">Net Economy Change</div>
                        <div class="stat-value {'good' if net_cash_change >= 0 else 'bad'}">{net_cash_change:,.2f}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-title">Wealth Inequality (Gini)</div>
                        <div class="stat-value {'good' if gini < 0.4 else 'bad' if gini > 0.6 else 'neutral'}">{gini:.3f}</div>
                    </div>
                    <div class="stat-box">
                        <div class="stat-title">Date Range</div>
                        <div class="stat-value" style="font-size: 18px;">{date_range}</div>
                    </div>
                </div>
                
                <div class="insight">
                    <p><strong>Key Insight:</strong> The economy has a {'positive' if net_cash_change >= 0 else 'negative'} net change of {abs(net_cash_change):,.2f}, 
                    indicating {'inflation' if net_cash_change > 0 else 'deflation'} in the system. 
                    The Gini coefficient of {gini:.3f} suggests {'high' if gini > 0.6 else 'moderate' if gini > 0.4 else 'low'} wealth inequality.</p>
                </div>
                
                <h2>User Activity Analysis</h2>
                <div class="gallery">
                    <div>
                        <h3>Top Users by Transaction Count</h3>
                        <img src="top_users_transactions.png" class="chart" alt="Top Users by Transaction Count">
                    </div>
                    <div>
                        <h3>User Activity by Hour</h3>
                        <img src="user_activity_by_hour.png" class="chart" alt="User Activity by Hour">
                    </div>
                    <div>
                        <h3>User Activity by Day</h3>
                        <img src="user_activity_by_day.png" class="chart" alt="User Activity by Day">
                    </div>
                    <div>
                        <h3>User Growth Over Time</h3>
                        <img src="user_growth.png" class="chart" alt="User Growth Over Time">
                    </div>
                    <div>
                        <h3>Activity Heatmap</h3>
                        <img src="activity_heatmap.png" class="chart" alt="Activity Heatmap">
                    </div>
                </div>
                
                <h2>Income Sources Analysis</h2>
                <div class="gallery">
                    <div>
                        <h3>Transaction Count by Category</h3>
                        <img src="transactions_by_category.png" class="chart" alt="Transaction Count by Category">
                    </div>
                    <div>
                        <h3>Net Cash Flow by Category</h3>
                        <img src="cash_flow_by_category.png" class="chart" alt="Net Cash Flow by Category">
                    </div>
                    <div>
                        <h3>Average Transaction Value</h3>
                        <img src="avg_value_by_category.png" class="chart" alt="Average Transaction Value">
                    </div>
                    <div>
                        <h3>Transaction Types</h3>
                        <img src="transaction_types.png" class="chart" alt="Transaction Types">
                    </div>
                    <div>
                        <h3>Category Flow Over Time</h3>
                        <img src="category_flow_over_time.png" class="chart" alt="Category Flow Over Time">
                    </div>
                </div>
                
                <h2>Economy Health Analysis</h2>
                <div class="gallery">
                    <div>
                        <h3>Economy Health Metrics</h3>
                        <img src="economy_health_metrics.png" class="chart" alt="Economy Health Metrics">
                    </div>
                    <div>
                        <h3>Wealth Distribution</h3>
                        <img src="wealth_distribution.png" class="chart" alt="Wealth Distribution">
                    </div>
                    <div>
                        <h3>Lorenz Curve</h3>
                        <img src="lorenz_curve.png" class="chart" alt="Lorenz Curve">
                    </div>
                    <div>
                        <h3>User Balance Progression</h3>
                        <img src="top_users_progression.png" class="chart" alt="User Balance Progression">
                    </div>
                    <div>
                        <h3>Average Earnings by Hour</h3>
                        <img src="hourly_earnings.png" class="chart" alt="Average Earnings by Hour">
                    </div>
                </div>
                
                <div class="warning">
                    <p><strong>Health Assessment:</strong> 
            """
            
            # Add economy assessment
            if net_cash_change > 10000:
                html_content += """
                    The economy shows significant <strong>inflation</strong>, with currency being created faster than it's removed.
                    Consider adding more money sinks or reducing income sources to maintain game balance.
                """
            elif net_cash_change < -5000:
                html_content += """
                    The economy is experiencing <strong>deflation</strong>, with more currency being removed than created.
                    Consider increasing rewards or adding new income sources to stimulate activity.
                """
            else:
                html_content += """
                    The economy appears to be well-balanced in terms of inflation/deflation, with currency creation and removal
                    in good proportion to each other.
                """
            
            html_content += """
                    </p>
                </div>
                
                <h2>Recommendations for Game Balance</h2>
                <ul>
            """
            
            # Add recommendations based on analysis
            if 'gini_coefficient' in economy_health and economy_health['gini_coefficient'] > 0.6:
                html_content += """
                    <li>The wealth inequality is high (Gini coefficient > 0.6). Consider implementing mechanics that help new players catch up, such as
                    progressive taxation, wealth redistribution events, or diminishing returns for top earners.</li>
                """
            
            if 'daily_metrics' in economy_health and economy_health['daily_metrics'] is not None:
                daily_metrics = economy_health['daily_metrics']
                if 'active_users' in daily_metrics.columns and daily_metrics['active_users'].mean() < 10:
                    html_content += """
                        <li>The daily active user count is low. Consider implementing events or incentives to boost regular participation.</li>
                    """
            
            # Add general recommendations
            html_content += """
                    <li>Regularly monitor the economy for inflation or deflation trends and adjust income sources and sinks accordingly.</li>
                    <li>Consider implementing seasonal events to break patterns and introduce variety to the game economy.</li>
                    <li>Review the most profitable activities to ensure they're not being exploited or dominating gameplay.</li>
                    <li>Balance the risk-reward ratio across different income sources to encourage diverse gameplay.</li>
                    <li>Track user retention and correlation with economic progress to ensure the economy is keeping players engaged.</li>
                </ul>
                
                <footer>
                    <p>Analysis report generated on {}</p>
                </footer>
            </body>
            </html>
            """.format(datetime.now().strftime('%Y-%m-%d %H:%M'))
            
            # Create output directory
            ensure_directory(self.output_dir)
            
            # Write HTML file
            report_path = os.path.join(self.output_dir, 'economy_analysis_report.html')
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

def run_economy_analysis(data_dir='balance_data', output_dir='economy_analysis'):
    """Run economy analysis from outside the class"""
    analyzer = EconomyAnalyzer(data_dir, output_dir)
    return analyzer.run_analysis()

if __name__ == "__main__":
    # Get data directory from command line if provided
    if len(sys.argv) > 1:
        data_dir = sys.argv[1]
    else:
        data_dir = 'balance_data'
    
    run_economy_analysis(data_dir) 