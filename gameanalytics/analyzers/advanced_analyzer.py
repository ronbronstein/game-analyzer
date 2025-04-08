#!/usr/bin/env python3
"""
Advanced Economy Analysis Module
Provides advanced economic analysis and metrics including time series analysis,
economic forecasting, and player network analysis.
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
import networkx as nx
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from gameanalytics.utils import load_csv_data, ensure_directory, set_plot_style, logger, Timer

class AdvancedEconomyAnalyzer:
    """Advanced economy analysis class with time series and network analysis"""
    
    def __init__(self, data_file='balance_data/balance_updates.csv', output_dir='advanced_analysis'):
        """Initialize the analyzer with data sources and output directory"""
        self.data_file = data_file
        self.output_dir = output_dir
        self.transactions = None
        self.summary_data = None
        self.time_series_data = None
        self.network_data = None
        self.forecast_data = None
        self.analysis_results = {}
        
        # Ensure the output directory exists
        ensure_directory(self.output_dir)
        
        # Set up plot style
        set_plot_style()
    
    def load_data(self):
        """Load transaction data and prepare datasets for analysis"""
        with Timer("Loading advanced analysis data"):
            # Load transaction data
            self.transactions = load_csv_data(self.data_file)
            if self.transactions is None or self.transactions.empty:
                logger.error(f"Failed to load data from {self.data_file}")
                return False
            
            # Convert timestamp to datetime if needed
            if 'timestamp' in self.transactions.columns and not pd.api.types.is_datetime64_any_dtype(self.transactions['timestamp']):
                self.transactions['timestamp'] = pd.to_datetime(self.transactions['timestamp'])
            
            # Ensure date column exists
            if 'date' not in self.transactions.columns and 'timestamp' in self.transactions.columns:
                self.transactions['date'] = self.transactions['timestamp'].dt.date
            
            # Prepare time series data - daily metrics
            if 'date' in self.transactions.columns and 'cash_with_sign' in self.transactions.columns:
                # Aggregate by date
                self.time_series_data = self.transactions.groupby('date').agg({
                    'cash_with_sign': ['sum', 'count', 'mean'],
                    'user': 'nunique'
                })
                
                # Flatten multi-index columns
                self.time_series_data.columns = ['_'.join(col).strip() for col in self.time_series_data.columns.values]
                self.time_series_data.rename(columns={
                    'cash_with_sign_sum': 'daily_cash_flow',
                    'cash_with_sign_count': 'transaction_count',
                    'cash_with_sign_mean': 'avg_transaction_value',
                    'user_nunique': 'active_users'
                }, inplace=True)
                
                # Calculate running totals and additional metrics
                self.time_series_data['cumulative_cash_flow'] = self.time_series_data['daily_cash_flow'].cumsum()
                self.time_series_data['transaction_per_user'] = self.time_series_data['transaction_count'] / self.time_series_data['active_users']
                
                # Convert index to DateTimeIndex for time series analysis
                self.time_series_data.index = pd.to_datetime(self.time_series_data.index)
                
                # Sort by date
                self.time_series_data = self.time_series_data.sort_index()
            
            # Prepare network data (player interactions)
            if 'user' in self.transactions.columns and 'reason' in self.transactions.columns:
                # Filter transfer transactions
                transfers = self.transactions[self.transactions['reason'].str.contains('transfer|sent to|give-money', case=False, na=False)].copy()
                
                if not transfers.empty and 'transfer_target' in transfers.columns:
                    self.network_data = transfers[['user', 'transfer_target', 'cash_with_sign']].copy()
                else:
                    # Try to extract transfer targets from reason field if not already extracted
                    def extract_target(reason):
                        if not isinstance(reason, str):
                            return None
                        import re
                        match = re.search(r'(?:sent|transfer(?:red)?)\s+to\s+@?([^\s]+)', reason, re.IGNORECASE)
                        if match:
                            return match.group(1)
                        return None
                    
                    transfers['transfer_target'] = transfers['reason'].apply(extract_target)
                    self.network_data = transfers[['user', 'transfer_target', 'cash_with_sign']].dropna().copy()
            
            logger.info(f"Loaded {len(self.transactions)} transactions for advanced analysis")
            return True
    
    def analyze_time_series(self):
        """Perform time series analysis on economic data"""
        with Timer("Time series analysis"):
            if self.time_series_data is None or self.time_series_data.empty:
                logger.error("No time series data available for analysis")
                return False
            
            results = {}
            
            # Check if we have enough data points for seasonal decomposition (at least 2 cycles)
            min_periods_for_seasonal = 14  # At least 2 weeks for weekly seasonality
            
            if len(self.time_series_data) >= min_periods_for_seasonal:
                # 1. Seasonal Decomposition
                try:
                    # Run seasonal decomposition on daily cash flow
                    seasonal_result = seasonal_decompose(
                        self.time_series_data['daily_cash_flow'], 
                        model='additive', 
                        period=7  # Assuming weekly patterns
                    )
                    
                    # Plot the decomposition
                    fig, axes = plt.subplots(4, 1, figsize=(12, 10), sharex=True)
                    seasonal_result.observed.plot(ax=axes[0], title='Observed')
                    seasonal_result.trend.plot(ax=axes[1], title='Trend')
                    seasonal_result.seasonal.plot(ax=axes[2], title='Seasonal')
                    seasonal_result.resid.plot(ax=axes[3], title='Residual')
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.output_dir, 'seasonal_decomposition.png'), dpi=300)
                    plt.close()
                    
                    results['seasonal_decomposition'] = {
                        'trend': seasonal_result.trend.to_dict(),
                        'seasonal': seasonal_result.seasonal.to_dict(),
                        'has_seasonality': seasonal_result.seasonal.abs().mean() > 0.1 * seasonal_result.observed.abs().mean()
                    }
                except Exception as e:
                    logger.error(f"Error in seasonal decomposition: {e}")
            
            # 2. ARIMA Forecasting for economic indicators
            try:
                # Prepare data for forecasting
                data = self.time_series_data['daily_cash_flow'].copy()
                
                # Fit ARIMA model (p,d,q) = (1,1,1) for simplicity
                model = ARIMA(data, order=(1, 1, 1))
                model_fit = model.fit()
                
                # Forecast next 7 days
                forecast_steps = 7
                forecast = model_fit.forecast(steps=forecast_steps)
                forecast_index = pd.date_range(
                    start=data.index[-1] + timedelta(days=1), 
                    periods=forecast_steps, 
                    freq='D'
                )
                
                # Create forecast DataFrame
                forecast_df = pd.DataFrame({
                    'date': forecast_index,
                    'forecasted_cash_flow': forecast,
                    'lower_ci': model_fit.get_forecast(forecast_steps).conf_int().iloc[:, 0],
                    'upper_ci': model_fit.get_forecast(forecast_steps).conf_int().iloc[:, 1]
                })
                forecast_df.set_index('date', inplace=True)
                
                # Store forecast data
                self.forecast_data = forecast_df
                
                # Visualize forecast
                plt.figure(figsize=(12, 6))
                
                # Plot historical data
                plt.plot(data.index, data, label='Historical Daily Cash Flow')
                
                # Plot forecast with confidence intervals
                plt.plot(forecast_df.index, forecast_df['forecasted_cash_flow'], 'r--', label='Forecast')
                plt.fill_between(
                    forecast_df.index, 
                    forecast_df['lower_ci'], 
                    forecast_df['upper_ci'], 
                    color='pink', 
                    alpha=0.3, 
                    label='95% Confidence Interval'
                )
                
                plt.title('Daily Cash Flow Forecast (Next 7 Days)', fontsize=16)
                plt.xlabel('Date', fontsize=12)
                plt.ylabel('Cash Flow', fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'cash_flow_forecast.png'), dpi=300)
                plt.close()
                
                # Calculate forecast statistics
                forecast_stats = {
                    'forecast_mean': forecast.mean(),
                    'forecast_std': forecast.std(),
                    'forecast_trend': 'up' if forecast[-1] > forecast[0] else 'down',
                    'forecast_change_pct': ((forecast[-1] - forecast[0]) / abs(forecast[0])) * 100 if forecast[0] != 0 else 0
                }
                
                results['forecast'] = forecast_stats
                
            except Exception as e:
                logger.error(f"Error in ARIMA forecasting: {e}")
                
            # 3. User activity patterns
            try:
                # Visualize active users over time
                plt.figure(figsize=(12, 6))
                plt.plot(self.time_series_data.index, self.time_series_data['active_users'], marker='o', linewidth=2)
                plt.title('Daily Active Users Over Time', fontsize=16)
                plt.xlabel('Date', fontsize=12)
                plt.ylabel('Number of Active Users', fontsize=12)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'active_users_trend.png'), dpi=300)
                plt.close()
                
                # Weekly patterns of activity
                if len(self.time_series_data) >= 7:
                    self.time_series_data['day_of_week'] = self.time_series_data.index.day_name()
                    
                    # Ensure we have all days of the week in correct order
                    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                    
                    # Activity by day of week
                    day_activity = self.time_series_data.groupby('day_of_week').agg({
                        'transaction_count': 'mean',
                        'active_users': 'mean',
                        'daily_cash_flow': 'mean'
                    }).reindex(day_order)
                    
                    # Plot activity by day of week
                    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
                    
                    sns.barplot(x=day_activity.index, y='transaction_count', data=day_activity, ax=axes[0])
                    axes[0].set_title('Average Transactions by Day of Week', fontsize=14)
                    axes[0].set_ylabel('Avg. Transactions', fontsize=12)
                    
                    sns.barplot(x=day_activity.index, y='active_users', data=day_activity, ax=axes[1])
                    axes[1].set_title('Average Active Users by Day of Week', fontsize=14)
                    axes[1].set_ylabel('Avg. Active Users', fontsize=12)
                    
                    sns.barplot(x=day_activity.index, y='daily_cash_flow', data=day_activity, ax=axes[2])
                    axes[2].set_title('Average Cash Flow by Day of Week', fontsize=14)
                    axes[2].set_ylabel('Avg. Cash Flow', fontsize=12)
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.output_dir, 'day_of_week_patterns.png'), dpi=300)
                    plt.close()
                    
                    results['day_patterns'] = day_activity.to_dict()
                    
                    # Find the most and least active days
                    most_active_day = day_activity['transaction_count'].idxmax()
                    least_active_day = day_activity['transaction_count'].idxmin()
                    
                    results['peak_activity'] = {
                        'most_active_day': most_active_day,
                        'least_active_day': least_active_day,
                        'activity_ratio': day_activity.loc[most_active_day, 'transaction_count'] / 
                                         day_activity.loc[least_active_day, 'transaction_count']
                    }
            except Exception as e:
                logger.error(f"Error in user activity analysis: {e}")
                
            self.analysis_results['time_series'] = results
            return True
    
    def analyze_player_network(self):
        """Analyze player-to-player interactions using network analysis"""
        with Timer("Player network analysis"):
            if self.network_data is None or self.network_data.empty:
                logger.warning("No network data available for analysis")
                return False
            
            results = {}
            
            try:
                # Clean network data - drop rows with missing targets
                network_data = self.network_data.dropna(subset=['user', 'transfer_target']).copy()
                
                if network_data.empty:
                    logger.warning("No valid network data after cleaning")
                    return False
                
                # Create directed graph
                G = nx.DiGraph()
                
                # Add edges with weights
                for _, row in network_data.iterrows():
                    source = row['user']
                    target = row['transfer_target']
                    value = abs(row['cash_with_sign'])
                    
                    if G.has_edge(source, target):
                        G[source][target]['weight'] += value
                        G[source][target]['count'] += 1
                    else:
                        G.add_edge(source, target, weight=value, count=1)
                
                # Calculate network metrics
                results['node_count'] = G.number_of_nodes()
                results['edge_count'] = G.number_of_edges()
                
                # Degree centrality (who sends/receives the most)
                in_degree = dict(G.in_degree(weight='weight'))
                out_degree = dict(G.out_degree(weight='weight'))
                
                results['top_receivers'] = dict(sorted(in_degree.items(), key=lambda x: x[1], reverse=True)[:5])
                results['top_senders'] = dict(sorted(out_degree.items(), key=lambda x: x[1], reverse=True)[:5])
                
                # Calculate betweenness centrality (who connects different communities)
                if G.number_of_edges() > 0:
                    betweenness = nx.betweenness_centrality(G, weight='weight')
                    results['key_connectors'] = dict(sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:5])
                
                # Visualize the network if it's not too large
                max_nodes_for_viz = 50
                if G.number_of_nodes() <= max_nodes_for_viz:
                    plt.figure(figsize=(14, 14))
                    
                    # Position nodes using spring layout
                    pos = nx.spring_layout(G, k=0.5, iterations=50)
                    
                    # Get edge weights for line thickness
                    edge_weights = [G[u][v]['weight'] / 500 for u, v in G.edges()]
                    
                    # Get node sizes based on total transactions (in + out)
                    node_sizes = []
                    for node in G.nodes():
                        in_val = in_degree.get(node, 0)
                        out_val = out_degree.get(node, 0)
                        node_sizes.append(100 + (in_val + out_val) / 100)
                    
                    # Draw the graph
                    nx.draw_networkx_nodes(G, pos, node_size=node_sizes, node_color='lightblue', alpha=0.8)
                    nx.draw_networkx_edges(G, pos, width=edge_weights, edge_color='gray', alpha=0.6, arrows=True, arrowstyle='->', arrowsize=15)
                    nx.draw_networkx_labels(G, pos, font_size=10)
                    
                    plt.title('Player Transaction Network', fontsize=16)
                    plt.axis('off')
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.output_dir, 'player_network.png'), dpi=300)
                    plt.close()
                else:
                    # For larger networks, create a simplified visualization with top nodes
                    top_nodes = list(dict(sorted(betweenness.items(), key=lambda x: x[1], reverse=True)[:max_nodes_for_viz]).keys())
                    H = G.subgraph(top_nodes)
                    
                    plt.figure(figsize=(14, 14))
                    pos = nx.spring_layout(H, k=0.5, iterations=50)
                    
                    # Get edge weights and node sizes for the subgraph
                    sub_edge_weights = [H[u][v]['weight'] / 500 for u, v in H.edges()]
                    
                    sub_node_sizes = []
                    for node in H.nodes():
                        in_val = in_degree.get(node, 0)
                        out_val = out_degree.get(node, 0)
                        sub_node_sizes.append(100 + (in_val + out_val) / 100)
                    
                    # Draw the simplified graph
                    nx.draw_networkx_nodes(H, pos, node_size=sub_node_sizes, node_color='lightblue', alpha=0.8)
                    nx.draw_networkx_edges(H, pos, width=sub_edge_weights, edge_color='gray', alpha=0.6, arrows=True, arrowstyle='->', arrowsize=15)
                    nx.draw_networkx_labels(H, pos, font_size=10)
                    
                    plt.title(f'Top {max_nodes_for_viz} Players Transaction Network', fontsize=16)
        plt.axis('off')
        plt.tight_layout()
                    plt.savefig(os.path.join(self.output_dir, 'top_players_network.png'), dpi=300)
        plt.close()
        
                # Detect communities if we have enough nodes
                if G.number_of_nodes() >= 10:
                    try:
                        # Use the Louvain method for community detection
                        import community as community_louvain
                        partition = community_louvain.best_partition(G.to_undirected())
                        
                        # Count nodes in each community
                        community_counts = {}
                        for node, community_id in partition.items():
                            if community_id not in community_counts:
                                community_counts[community_id] = 0
                            community_counts[community_id] += 1
                        
                        results['communities'] = {
                            'count': len(community_counts),
                            'sizes': community_counts
                        }
                    except ImportError:
                        logger.warning("python-louvain package not installed, skipping community detection")
                    except Exception as e:
                        logger.error(f"Error in community detection: {e}")
                
                self.analysis_results['player_network'] = results
                return True
                
            except Exception as e:
                logger.error(f"Error in player network analysis: {e}")
                return False
    
    def generate_html_report(self):
        """Generate a comprehensive HTML report with all advanced analyses"""
        with Timer("Generating advanced HTML report"):
            # Extract key results
            time_series_results = self.analysis_results.get('time_series', {})
            network_results = self.analysis_results.get('player_network', {})
            
            # Prepare forecast insights
            forecast_insights = ""
            if 'forecast' in time_series_results:
                forecast = time_series_results['forecast']
                trend_direction = forecast.get('forecast_trend', 'stable')
                trend_pct = forecast.get('forecast_change_pct', 0)
                
                forecast_insights = f"""
                <div class="insight-card">
                    <h3>Cash Flow Forecast</h3>
                    <p>The economy is trending <strong>{trend_direction}</strong> with a projected change of <strong>{trend_pct:.1f}%</strong> over the next 7 days.</p>
                    <img src="cash_flow_forecast.png" alt="Cash Flow Forecast" class="analysis-image">
                </div>
                """
            
            # Prepare seasonality insights
            seasonality_insights = ""
            if 'seasonal_decomposition' in time_series_results:
                has_seasonality = time_series_results['seasonal_decomposition'].get('has_seasonality', False)
                
                if has_seasonality:
                    seasonality_insights = f"""
                    <div class="insight-card">
                        <h3>Seasonal Patterns Detected</h3>
                        <p>The economy shows significant weekly patterns in activity and cash flow.</p>
                        <img src="seasonal_decomposition.png" alt="Seasonal Decomposition" class="analysis-image">
                    </div>
                    """
                else:
                    seasonality_insights = f"""
                    <div class="insight-card">
                        <h3>No Strong Seasonal Patterns</h3>
                        <p>The economy does not show significant weekly patterns in activity and cash flow.</p>
                        <img src="seasonal_decomposition.png" alt="Seasonal Decomposition" class="analysis-image">
                    </div>
                    """
            
            # Prepare peak activity insights
            activity_insights = ""
            if 'peak_activity' in time_series_results:
                peak_data = time_series_results['peak_activity']
                most_active = peak_data.get('most_active_day', 'Unknown')
                least_active = peak_data.get('least_active_day', 'Unknown')
                ratio = peak_data.get('activity_ratio', 1)
                
                activity_insights = f"""
                <div class="insight-card">
                    <h3>Activity Patterns</h3>
                    <p>The most active day is <strong>{most_active}</strong>, while the least active is <strong>{least_active}</strong>.</p>
                    <p>Activity on {most_active} is <strong>{ratio:.1f}x</strong> higher than on {least_active}.</p>
                    <img src="day_of_week_patterns.png" alt="Day of Week Patterns" class="analysis-image">
                </div>
                """
            
            # Prepare player network insights
            network_insights = ""
            if network_results:
                node_count = network_results.get('node_count', 0)
                edge_count = network_results.get('edge_count', 0)
                
                # Top senders and receivers
                top_senders = network_results.get('top_senders', {})
                top_receivers = network_results.get('top_receivers', {})
                
                senders_list = "".join([f"<li>{name}: {value:.0f}</li>" for name, value in top_senders.items()])
                receivers_list = "".join([f"<li>{name}: {value:.0f}</li>" for name, value in top_receivers.items()])
                
                network_file = "player_network.png"
                if node_count > 50:
                    network_file = "top_players_network.png"
                
                network_insights = f"""
                <div class="insight-card">
                    <h3>Player Transaction Network</h3>
                    <p>The economy has <strong>{node_count}</strong> active players exchanging currency through <strong>{edge_count}</strong> connections.</p>
                    
                    <div class="network-stats">
                        <div class="stats-column">
                            <h4>Top Senders:</h4>
                            <ul>{senders_list}</ul>
                        </div>
                        <div class="stats-column">
                            <h4>Top Receivers:</h4>
                            <ul>{receivers_list}</ul>
                        </div>
                    </div>
                    
                    <img src="{network_file}" alt="Player Network" class="analysis-image">
                </div>
                """
            
            # Create complete HTML report
        html_content = f"""
        <!DOCTYPE html>
        <html>
        <head>
                <title>Advanced Economy Analysis</title>
            <style>
                    body {{ font-family: Arial, sans-serif; margin: 0; padding: 0; background-color: #f5f5f5; color: #333; }}
                    .container {{ max-width: 1200px; margin: 0 auto; padding: 20px; }}
                    h1, h2, h3, h4 {{ color: #2c3e50; }}
                    h1 {{ text-align: center; margin-bottom: 30px; padding-bottom: 15px; border-bottom: 2px solid #3498db; }}
                    h2 {{ margin-top: 40px; padding-bottom: 10px; border-bottom: 1px solid #ddd; }}
                    
                    .insights-container {{ display: flex; flex-wrap: wrap; gap: 20px; justify-content: space-between; }}
                    .insight-card {{ background: white; border-radius: 8px; padding: 20px; box-shadow: 0 3px 10px rgba(0,0,0,0.1); margin-bottom: 20px; flex: 1 1 45%; }}
                    .insight-card h3 {{ color: #3498db; margin-top: 0; }}
                    .analysis-image {{ max-width: 100%; height: auto; margin-top: 15px; border-radius: 5px; }}
                    
                    .network-stats {{ display: flex; gap: 20px; margin-top: 15px; }}
                    .stats-column {{ flex: 1; }}
                    .stats-column ul {{ padding-left: 20px; }}
                    
                    table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
                    th, td {{ padding: 12px 15px; text-align: left; border-bottom: 1px solid #ddd; }}
                    th {{ background-color: #f0f0f0; }}
                    
                    footer {{ margin-top: 50px; text-align: center; color: #7f8c8d; font-size: 0.9em; }}
            </style>
        </head>
        <body>
                <div class="container">
            <h1>Advanced Economy Analysis</h1>
                    
                    <h2>Economic Forecasting & Time Series Analysis</h2>
                    <div class="insights-container">
                        {forecast_insights}
                        {seasonality_insights}
                        {activity_insights}
                    </div>
                    
                    <h2>Player Interaction Network</h2>
                    <div class="insights-container">
                        {network_insights}
            </div>
                    
            <footer>
                <p>Analysis generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}</p>
            </footer>
                </div>
        </body>
        </html>
        """
        
        # Write HTML file
            report_path = os.path.join(self.output_dir, 'advanced_economy_report.html')
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(html_content)
            
            logger.info(f"Advanced analysis report generated: {report_path}")
            return True
    
    def run_analysis(self):
        """Run the complete advanced analysis pipeline"""
        # Load and prepare data
        if not self.load_data():
            logger.error("Failed to load data for advanced analysis")
            return False
        
        # Run time series analysis
        self.analyze_time_series()
        
        # Run player network analysis
        self.analyze_player_network()
        
        # Generate HTML report
        self.generate_html_report()
        
        logger.info("Advanced analysis completed successfully!")
        return True


def run_advanced_analysis(data_file='balance_data/balance_updates.csv', output_dir='advanced_analysis'):
    """Run advanced analysis from outside the class"""
    logger.info(f"Running advanced analysis on {data_file}, saving results to {output_dir}...")
    
    try:
        # Create analyzer instance
        analyzer = AdvancedEconomyAnalyzer(data_file, output_dir)
        
        # Run analysis
        success = analyzer.run_analysis()
        
        if success:
            logger.info(f"Advanced analysis completed successfully. Results saved to {output_dir}/")
            return True
        else:
            logger.error("Advanced analysis failed")
            return False
            
    except Exception as e:
        logger.error(f"Error during advanced analysis: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    run_advanced_analysis() 