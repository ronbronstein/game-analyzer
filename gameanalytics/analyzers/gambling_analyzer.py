#!/usr/bin/env python3
"""
Gambling Activity Analyzer
Provides detailed analysis of gambling outcomes and game fairness
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
import sys
from statsmodels.stats.proportion import proportions_ztest
from statsmodels.tsa.seasonal import seasonal_decompose

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from gameanalytics.utils import load_csv_data, ensure_directory, set_plot_style, logger, Timer

class GamblingAnalyzer:
    """Analyze gambling activities and outcomes"""
    
    def __init__(self, data_file='balance_data/balance_updates.csv', output_dir='gambling_analysis'):
        """Initialize the analyzer
        
        Args:
            data_file (str): Path to balance updates CSV
            output_dir (str): Directory to save analysis results
        """
        self.data_file = data_file
        self.output_dir = output_dir
        self.df = None
        self.gambling_df = None
        self.outcomes_df = None
        self.game_stats = None
        
        # Set plot style
        set_plot_style()
    
    def load_data(self):
        """Load the balance data and prepare for analysis"""
        with Timer("Loading and preparing gambling data"):
            # Load data
            self.df = load_csv_data(self.data_file)
            
            if self.df.empty:
                logger.error(f"Failed to load data from {self.data_file}")
                return False
            
            # Filter for gambling transactions
            gambling_categories = [
                'Animal Race Bet', 'Animal Race Win', 
                'Blackjack Bet', 'Blackjack Win',
                'Roulette Bet', 'Roulette Win',
                'Slots & Other Gambling'
            ]
            
            # First check if we have the improved category column
            if 'improved_category' in self.df.columns:
                self.gambling_df = self.df[self.df['improved_category'].isin(gambling_categories)].copy()
            # Fallback to regular category
            elif 'category' in self.df.columns:
                legacy_categories = ['Betting', 'Gambling', 'Animal Race']
                self.gambling_df = self.df[self.df['category'].isin(legacy_categories)].copy()
            else:
                # Create manual identification based on reason text
                gambling_keywords = ['bet', 'won', 'gamble', 'slot', 'blackjack', 'roulette', 'animal race']
                self.gambling_df = self.df[
                    self.df['reason'].str.lower().apply(
                        lambda x: any(keyword in str(x).lower() for keyword in gambling_keywords)
                    )
                ].copy()
            
            if self.gambling_df.empty:
                logger.warning("No gambling transactions found")
                return False
                
            logger.info(f"Found {len(self.gambling_df)} gambling transactions")
            return True
    
    def analyze_user_outcomes(self):
        """Analyze outcomes (wins/losses) by user"""
        with Timer("Analyzing gambling outcomes by user"):
            # Identify winning and losing transactions
            if 'improved_category' in self.gambling_df.columns:
                self.gambling_df['is_win'] = self.gambling_df['improved_category'].str.contains('Win')
                self.gambling_df['is_bet'] = self.gambling_df['improved_category'].str.contains('Bet')
            else:
                # Use keywords in reason text to identify wins
                winning_keywords = ['won', 'win', 'ended']
                bet_keywords = ['bet', 'wager']
                
                self.gambling_df['is_win'] = self.gambling_df['reason'].apply(
                    lambda x: any(keyword in str(x).lower() for keyword in winning_keywords)
                )
                self.gambling_df['is_bet'] = self.gambling_df['reason'].apply(
                    lambda x: any(keyword in str(x).lower() for keyword in bet_keywords)
                )
            
            # Separate wins and bets to avoid double-counting
            wins = self.gambling_df[self.gambling_df['is_win']].copy()
            bets = self.gambling_df[self.gambling_df['is_bet']].copy()
            
            # Handle cases where some transactions might be both or neither
            other_gambling = self.gambling_df[~(self.gambling_df['is_win'] | self.gambling_df['is_bet'])].copy()
            
            # Group by user
            user_wins = wins.groupby('user').agg({
                'cash_with_sign': ['sum', 'count'],
                'timestamp': 'min'  # First win timestamp
            })
            
            user_wins.columns = ['win_amount', 'win_count', 'first_win']
            
            user_bets = bets.groupby('user').agg({
                'cash_with_sign': ['sum', 'count'],
                'timestamp': 'min'  # First bet timestamp
            })
            
            user_bets.columns = ['bet_amount', 'bet_count', 'first_bet']
            
            # Merge the results
            user_outcomes = pd.merge(
                user_bets, user_wins, 
                left_index=True, right_index=True, 
                how='outer'
            ).fillna(0)
            
            # Calculate derived metrics
            user_outcomes['net_profit'] = user_outcomes['win_amount'] + user_outcomes['bet_amount']
            user_outcomes['total_bets'] = user_outcomes['bet_count']
            user_outcomes['total_wins'] = user_outcomes['win_count']
            user_outcomes['win_rate'] = np.where(
                user_outcomes['total_bets'] > 0,
                (user_outcomes['total_wins'] / user_outcomes['total_bets']) * 100,
                0
            )
            
            # Calculate ROI (Return on Investment)
            user_outcomes['roi'] = np.where(
                user_outcomes['bet_amount'] != 0,
                (user_outcomes['net_profit'] / abs(user_outcomes['bet_amount'])) * 100,
                0
            )
            
            # Calculate average win and bet amounts
            user_outcomes['avg_win'] = np.where(
                user_outcomes['win_count'] > 0,
                user_outcomes['win_amount'] / user_outcomes['win_count'],
                0
            )
            
            user_outcomes['avg_bet'] = np.where(
                user_outcomes['bet_count'] > 0,
                abs(user_outcomes['bet_amount']) / user_outcomes['bet_count'],
                0
            )
            
            # Sort by net profit
            user_outcomes = user_outcomes.sort_values('net_profit', ascending=False)
            
            self.outcomes_df = user_outcomes
            
            logger.info(f"Analyzed outcomes for {len(user_outcomes)} users")
            
            # Print some basic statistics
            total_bets = bets['cash_with_sign'].sum()
            total_wins = wins['cash_with_sign'].sum()
            overall_roi = (total_wins + total_bets) / abs(total_bets) if total_bets != 0 else 0
            
            logger.info(f"Overall gambling statistics:")
            logger.info(f"  Total bet amount: {abs(total_bets):,.2f}")
            logger.info(f"  Total win amount: {total_wins:,.2f}")
            logger.info(f"  Net change: {total_wins + total_bets:,.2f}")
            logger.info(f"  Overall ROI: {overall_roi * 100:.2f}%")
            
            return user_outcomes
    
    def analyze_game_performance(self):
        """Analyze performance metrics by game type"""
        with Timer("Analyzing game performance"):
            # Identify game types
            if 'improved_category' in self.gambling_df.columns:
                game_map = {
                    'Blackjack': ['Blackjack Bet', 'Blackjack Win'],
                    'Roulette': ['Roulette Bet', 'Roulette Win'],
                    'Animal Race': ['Animal Race Bet', 'Animal Race Win'],
                    'Slots': ['Slots & Other Gambling']
                }
                
                # Map each transaction to a game type
                def map_to_game(category):
                    for game, categories in game_map.items():
                        if category in categories:
                            return game
                    return 'Other Gambling'
                
                self.gambling_df['game_type'] = self.gambling_df['improved_category'].apply(map_to_game)
            else:
                # Use keywords in reason text to identify game types
                def identify_game(reason):
                    reason = str(reason).lower()
                    if 'blackjack' in reason:
                        return 'Blackjack'
                    elif 'roulette' in reason:
                        return 'Roulette'
                    elif 'animal' in reason and 'race' in reason:
                        return 'Animal Race'
                    elif any(term in reason for term in ['slot', 'slots', 'slot-machine']):
                        return 'Slots'
                    else:
                        return 'Other Gambling'
                
                self.gambling_df['game_type'] = self.gambling_df['reason'].apply(identify_game)
            
            # Analyze by game type
            results = []
            game_types = self.gambling_df['game_type'].unique()
            
            for game in game_types:
                game_df = self.gambling_df[self.gambling_df['game_type'] == game].copy()
                
                wins = game_df[game_df['is_win']].copy()
                bets = game_df[game_df['is_bet']].copy()
                
                # Calculate metrics
                win_count = len(wins)
                bet_count = len(bets)
                win_amount = wins['cash_with_sign'].sum() if not wins.empty else 0
                bet_amount = abs(bets['cash_with_sign'].sum()) if not bets.empty else 0
                
                # Calculate advanced metrics
                if bet_amount > 0:
                    payout_ratio = win_amount / bet_amount
                    house_edge = 1 - payout_ratio
                else:
                    payout_ratio = 0
                    house_edge = 1
                
                if bet_count > 0:
                    win_probability = win_count / bet_count
                else:
                    win_probability = 0
                
                # Calculate EV (Expected Value) per bet
                ev_per_bet = 0
                if bet_count > 0 and bet_amount > 0:
                    avg_bet = bet_amount / bet_count
                    avg_win = win_amount / win_count if win_count > 0 else 0
                    ev_per_bet = (win_probability * avg_win) - ((1 - win_probability) * avg_bet)
                
                # Perform statistical hypothesis test for fair odds
                # Test if win probability is significantly different from 0.5 (fair odds)
                p_value = None
                if bet_count >= 10:  # Need enough samples for valid test
                    count = np.array([win_count])
                    nobs = np.array([bet_count])
                    stat, p_value = proportions_ztest(count, nobs, value=0.5)
                
                # Calculate user stats for this game
                unique_players = game_df['user'].nunique()
                winning_players = wins['user'].nunique() if not wins.empty else 0
                losing_players = bets['user'].nunique() if not bets.empty else 0
                
                results.append({
                    'game': game,
                    'win_count': win_count,
                    'bet_count': bet_count,
                    'win_amount': win_amount,
                    'bet_amount': bet_amount,
                    'payout_ratio': payout_ratio,
                    'house_edge': house_edge,
                    'win_probability': win_probability,
                    'ev_per_bet': ev_per_bet,
                    'p_value': p_value,
                    'unique_players': unique_players,
                    'winning_players': winning_players,
                    'losing_players': losing_players
                })
            
            self.game_stats = pd.DataFrame(results)
            
            if not self.game_stats.empty:
                logger.info("\nGambling Analysis by Game Type:")
                for _, row in self.game_stats.iterrows():
                    logger.info(f"\n{row['game']}:")
                    logger.info(f"  Win Count: {row['win_count']}, Bet Count: {row['bet_count']}")
                    logger.info(f"  Win Amount: {row['win_amount']:,.2f}, Bet Amount: {row['bet_amount']:,.2f}")
                    logger.info(f"  Payout Ratio: {row['payout_ratio']:.2f} (House Edge: {row['house_edge']:.2f})")
                    logger.info(f"  Win Probability: {row['win_probability']:.2f}")
                    logger.info(f"  Expected Value per Bet: {row['ev_per_bet']:.2f}")
                    logger.info(f"  Fair Odds P-Value: {row['p_value'] if row['p_value'] is not None else 'Insufficient data'}")
                    
                    # Interpret p-value
                    if row['p_value'] is not None:
                        if row['p_value'] < 0.05:
                            logger.info(f"  Odds Analysis: Not fair odds (statistically significant advantage/disadvantage)")
                        else:
                            logger.info(f"  Odds Analysis: Could be fair odds (no statistical significance)")
            
            return self.game_stats
    
    def analyze_transaction_patterns(self):
        """Analyze temporal patterns in gambling activity"""
        with Timer("Analyzing gambling patterns"):
            # Make sure timestamps are available
            if 'timestamp' not in self.gambling_df.columns:
                logger.error("Timestamp column not found for pattern analysis")
                return None
            
            # Ensure timestamps are in datetime format
            self.gambling_df['timestamp'] = pd.to_datetime(self.gambling_df['timestamp'])
            
            # Analyze daily gambling volume
            daily_gambling = self.gambling_df.groupby(self.gambling_df['timestamp'].dt.date).agg({
                'cash_with_sign': ['sum', lambda x: abs(x).sum()],
                'user': 'nunique',
                'timestamp': 'count'
            })
            
            daily_gambling.columns = ['net_amount', 'volume', 'unique_players', 'transaction_count']
            
            # Check if we have enough days for time series analysis
            if len(daily_gambling) >= 14:  # Need at least 2 weeks for seasonal analysis
                try:
                    # Decompose the time series to identify trends and seasonal patterns
                    gambling_volume = daily_gambling['volume'].fillna(0)
                    
                    # Using weekly seasonality (7 days)
                    decomposition = seasonal_decompose(gambling_volume, model='additive', period=7)
                    
                    # Create DataFrame for seasonal components
                    seasonal_df = pd.DataFrame({
                        'volume': gambling_volume,
                        'trend': decomposition.trend,
                        'seasonal': decomposition.seasonal,
                        'residual': decomposition.resid
                    })
                    
                    logger.info("\nTemporal pattern analysis complete:")
                    
                    # Calculate day of week patterns
                    if 'day_of_week' in self.gambling_df.columns:
                        day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
                        dow_volume = self.gambling_df.groupby('day_of_week')['cash_with_sign'].apply(lambda x: abs(x).sum())
                        
                        # Sort by day of week
                        if set(day_order).issubset(set(dow_volume.index)):
                            dow_volume = dow_volume.reindex(day_order)
                        
                        most_active_day = dow_volume.idxmax()
                        least_active_day = dow_volume.idxmin()
                        
                        logger.info(f"  Most active gambling day: {most_active_day}")
                        logger.info(f"  Least active gambling day: {least_active_day}")
                    
                    return {
                        'daily_gambling': daily_gambling,
                        'seasonal_analysis': seasonal_df,
                        'has_weekly_pattern': abs(decomposition.seasonal).mean() > (gambling_volume.mean() * 0.1)
                    }
                
                except Exception as e:
                    logger.error(f"Error in time series decomposition: {e}")
                    return {'daily_gambling': daily_gambling}
            else:
                logger.warning("Not enough data for time series decomposition")
                return {'daily_gambling': daily_gambling}
    
    def generate_visualizations(self):
        """Generate visualizations of gambling analysis"""
        with Timer("Generating gambling visualizations"):
            # Create output directory
            ensure_directory(self.output_dir)
            
            # 1. User outcomes visualizations
            if self.outcomes_df is not None and not self.outcomes_df.empty:
                # Top winners/losers by net profit
                plt.figure(figsize=(14, 10))
                # Get top 10 winners and losers
                top_winners = self.outcomes_df.head(10)
                top_losers = self.outcomes_df.sort_values('net_profit').head(10)
                
                # Combine winners and losers
                top_combined = pd.concat([top_winners, top_losers])
                
                # Set color based on profit/loss
                colors = ['green' if x >= 0 else 'red' for x in top_combined['net_profit']]
                
                # Create bar plot
                ax = sns.barplot(
                    x='net_profit', 
                    y=top_combined.index, 
                    data=top_combined,
                    palette=colors
                )
                
                plt.title('Top Winners and Losers by Net Profit', fontsize=16)
                plt.xlabel('Net Profit', fontsize=14)
                plt.ylabel('User', fontsize=14)
                plt.axvline(x=0, color='black', linestyle='--')
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'top_gamblers_by_profit.png'), dpi=300)
                plt.close()
                
                # Win rates vs. ROI scatter plot
                plt.figure(figsize=(12, 10))
                
                # Only include users with at least 5 bets
                active_gamblers = self.outcomes_df[self.outcomes_df['total_bets'] >= 5].copy()
                
                if not active_gamblers.empty:
                    # Calculate point size based on number of bets
                    point_sizes = active_gamblers['total_bets'] / active_gamblers['total_bets'].max() * 100 + 20
                    
                    # Color based on profit/loss
                    colors = ['green' if x >= 0 else 'red' for x in active_gamblers['net_profit']]
                    
                    plt.scatter(
                        active_gamblers['win_rate'],
                        active_gamblers['roi'],
                        s=point_sizes,
                        c=colors,
                        alpha=0.6
                    )
                    
                    plt.title('Win Rate vs. Return on Investment (ROI)', fontsize=16)
                    plt.xlabel('Win Rate (%)', fontsize=14)
                    plt.ylabel('ROI (%)', fontsize=14)
                    plt.axhline(y=0, color='black', linestyle='--')
                    plt.axvline(x=50, color='black', linestyle='--')  # Fair odds line
                    plt.grid(True, alpha=0.3)
                    
                    # Add annotations for top 5 winners and losers
                    top_5_winners = active_gamblers.nlargest(5, 'net_profit')
                    top_5_losers = active_gamblers.nsmallest(5, 'net_profit')
                    
                    for idx, row in pd.concat([top_5_winners, top_5_losers]).iterrows():
                        plt.annotate(
                            idx,
                            (row['win_rate'], row['roi']),
                            fontsize=9,
                            xytext=(5, 5),
                            textcoords='offset points'
                        )
                    
                    plt.tight_layout()
                    plt.savefig(os.path.join(self.output_dir, 'win_rate_vs_roi.png'), dpi=300)
                plt.close()
            
            # 2. Game statistics visualizations
            if self.game_stats is not None and not self.game_stats.empty:
                # Payout ratio by game type
                plt.figure(figsize=(12, 8))
                ax = sns.barplot(
                    x='game',
                    y='payout_ratio',
                    data=self.game_stats,
                    palette='YlGnBu'
                )
                
                # Add value labels
                for i, row in enumerate(self.game_stats.itertuples()):
                    ax.text(
                        i, 
                        row.payout_ratio + 0.02, 
                        f"{row.payout_ratio:.2f}", 
                        ha='center'
                    )
                
                # Add reference line for fair odds (1.0)
                plt.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Fair Odds')
                
                plt.title('Payout Ratio by Game Type', fontsize=16)
                plt.xlabel('Game', fontsize=14)
                plt.ylabel('Payout Ratio', fontsize=14)
                plt.legend()
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'payout_ratios.png'), dpi=300)
                plt.close()
                
                # Game participation stats
                plt.figure(figsize=(12, 8))
                
                # Get win probability and player count
                game_stats_subset = self.game_stats[['game', 'win_probability', 'unique_players']].copy()
                game_stats_subset = game_stats_subset.sort_values('unique_players', ascending=False)
                
                # Create bar chart with two y-axes
                fig, ax1 = plt.subplots(figsize=(12, 8))
                
                # Plot unique players on primary axis
                sns.barplot(
                    x='game', 
                    y='unique_players', 
                    data=game_stats_subset,
                    color='skyblue',
                    ax=ax1
                )
                
                ax1.set_xlabel('Game', fontsize=14)
                ax1.set_ylabel('Number of Players', fontsize=14)
                
                # Create secondary y-axis for win probability
                ax2 = ax1.twinx()
                ax2.plot(
                    game_stats_subset['game'].values,
                    game_stats_subset['win_probability'].values,
                    'ro-',
                    linewidth=2,
                    markersize=8
                )
                
                ax2.set_ylabel('Win Probability', fontsize=14, color='r')
                ax2.tick_params(axis='y', colors='r')
                
                # Add reference line for fair odds (0.5)
                ax2.axhline(y=0.5, color='r', linestyle='--', alpha=0.3)
                
                plt.title('Game Popularity and Win Probability', fontsize=16)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'game_participation.png'), dpi=300)
                plt.close()
                
                # Expected value comparison
                plt.figure(figsize=(12, 8))
                
                # Sort by EV
                ev_data = self.game_stats.sort_values('ev_per_bet', ascending=False)
                
                # Set bar colors based on EV (positive or negative)
                colors = ['green' if x >= 0 else 'red' for x in ev_data['ev_per_bet']]
                
                ax = sns.barplot(
                    x='game',
                    y='ev_per_bet',
                    data=ev_data,
                    palette=colors
                )
                
                # Add value labels
                for i, row in enumerate(ev_data.itertuples()):
                    ax.text(
                        i, 
                        row.ev_per_bet + (0.01 if row.ev_per_bet >= 0 else -0.01), 
                        f"{row.ev_per_bet:.2f}", 
                        ha='center'
                    )
                
                # Add reference line for break-even (0.0)
                plt.axhline(y=0.0, color='black', linestyle='--', alpha=0.7)
                
                plt.title('Expected Value (EV) per Bet by Game Type', fontsize=16)
                plt.xlabel('Game', fontsize=14)
                plt.ylabel('Expected Value per Bet', fontsize=14)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'expected_values.png'), dpi=300)
                plt.close()
            
            logger.info(f"Visualizations saved to {self.output_dir}/")
    
    def generate_html_report(self):
        """Generate an HTML report with the analysis results"""
        with Timer("Generating gambling HTML report"):
            # Prepare HTML content
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Gambling Analysis Report</title>
                <style>
                    body {{ font-family: Arial, sans-serif; margin: 40px; line-height: 1.6; }}
                    h1 {{ color: #2c3e50; }}
                    h2 {{ color: #3498db; margin-top: 40px; }}
                    h3 {{ color: #7f8c8d; margin-top: 30px; }}
                    table {{
                        width: 100%;
                        border-collapse: collapse;
                        margin: 20px 0;
                    }}
                    table, th, td {{
                        border: 1px solid #ddd;
                    }}
                    th, td {{
                        padding: 12px;
                        text-align: left;
                    }}
                    th {{
                        background-color: #f2f2f2;
                    }}
                    tr:nth-child(even) {{
                        background-color: #f9f9f9;
                    }}
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
                    .good {{ color: #27ae60; }}
                    .bad {{ color: #e74c3c; }}
                    .neutral {{ color: #f39c12; }}
                </style>
            </head>
            <body>
                <h1>Gambling Analysis Report</h1>
                
                <div class="insight">
                    <p>This report analyzes gambling activities in the UnbelievaBoat economy, focusing on game fairness, 
                    player outcomes, and gambling patterns.</p>
                </div>
            """
            
            # Add game statistics section
            if self.game_stats is not None and not self.game_stats.empty:
                html_content += """
                <h2>Game Statistics</h2>
                <div class="gallery">
                    <div>
                        <img src="payout_ratios.png" class="chart" alt="Payout Ratios">
                    </div>
                    <div>
                        <img src="game_participation.png" class="chart" alt="Game Participation">
                    </div>
                    <div>
                        <img src="expected_values.png" class="chart" alt="Expected Values">
                    </div>
                </div>
                
                <h3>Detailed Game Statistics</h3>
                <table>
                    <tr>
                        <th>Game</th>
                        <th>Win Probability</th>
                        <th>Payout Ratio</th>
                        <th>House Edge</th>
                        <th>Expected Value</th>
                        <th>Fair Odds?</th>
                    </tr>
                """
                
                for _, row in self.game_stats.iterrows():
                    # Determine if the game has fair odds
                    fair_odds = "Unknown"
                    fair_odds_class = "neutral"
                    
                    if row['p_value'] is not None:
                        if row['p_value'] < 0.05:
                            fair_odds = "No"
                            fair_odds_class = "bad"
                        else:
                            fair_odds = "Possibly"
                            fair_odds_class = "good"
                    
                    # Determine EV class
                    ev_class = "good" if row['ev_per_bet'] >= 0 else "bad"
                    
                    html_content += f"""
                    <tr>
                        <td>{row['game']}</td>
                        <td>{row['win_probability']:.2f}</td>
                        <td>{row['payout_ratio']:.2f}</td>
                        <td>{row['house_edge']:.2f}</td>
                        <td class="{ev_class}">{row['ev_per_bet']:.2f}</td>
                        <td class="{fair_odds_class}">{fair_odds}</td>
                    </tr>
                    """
                
                html_content += """
                </table>
                """
                
                # Add game insights
                html_content += """
                <div class="insight">
                """
                
                # Find best and worst games
                if len(self.game_stats) > 1:
                    best_game = self.game_stats.loc[self.game_stats['ev_per_bet'].idxmax()]
                    worst_game = self.game_stats.loc[self.game_stats['ev_per_bet'].idxmin()]
                    
                    html_content += f"""
                    <p><strong>Game Insights:</strong></p>
                    <ul>
                        <li>Best game: <strong class="good">{best_game['game']}</strong> with EV of {best_game['ev_per_bet']:.2f}</li>
                        <li>Worst game: <strong class="bad">{worst_game['game']}</strong> with EV of {worst_game['ev_per_bet']:.2f}</li>
                    """
                    
                    # Add insights about house edge
                    avg_house_edge = self.game_stats['house_edge'].mean()
                    html_content += f"""
                        <li>Average house edge: {avg_house_edge:.2f} ({(avg_house_edge * 100):.1f}%)</li>
                    </ul>
                    """
                
                html_content += """
                </div>
                """
            
            # Add player outcomes section
            if self.outcomes_df is not None and not self.outcomes_df.empty:
                html_content += """
                <h2>Player Outcomes</h2>
                <div class="gallery">
                    <div>
                        <img src="top_gamblers_by_profit.png" class="chart" alt="Top Gamblers by Profit">
                    </div>
                    <div>
                        <img src="win_rate_vs_roi.png" class="chart" alt="Win Rate vs ROI">
                    </div>
                </div>
                
                <h3>Top Winners</h3>
                <table>
                    <tr>
                        <th>Player</th>
                        <th>Net Profit</th>
                        <th>Win Rate</th>
                        <th>ROI</th>
                        <th>Total Bets</th>
                        <th>Total Wins</th>
                    </tr>
                """
                
                for user, row in self.outcomes_df.head(10).iterrows():
                    html_content += f"""
                    <tr>
                        <td>{user}</td>
                        <td class="{'good' if row['net_profit'] >= 0 else 'bad'}">{row['net_profit']:,.2f}</td>
                        <td>{row['win_rate']:.1f}%</td>
                        <td class="{'good' if row['roi'] >= 0 else 'bad'}">{row['roi']:.1f}%</td>
                        <td>{row['total_bets']:.0f}</td>
                        <td>{row['total_wins']:.0f}</td>
                    </tr>
                    """
                
                html_content += """
                </table>
                
                <h3>Top Losers</h3>
                <table>
                    <tr>
                        <th>Player</th>
                        <th>Net Profit</th>
                        <th>Win Rate</th>
                        <th>ROI</th>
                        <th>Total Bets</th>
                        <th>Total Wins</th>
                    </tr>
                """
                
                for user, row in self.outcomes_df.sort_values('net_profit').head(10).iterrows():
                    html_content += f"""
                    <tr>
                        <td>{user}</td>
                        <td class="{'good' if row['net_profit'] >= 0 else 'bad'}">{row['net_profit']:,.2f}</td>
                        <td>{row['win_rate']:.1f}%</td>
                        <td class="{'good' if row['roi'] >= 0 else 'bad'}">{row['roi']:.1f}%</td>
                        <td>{row['total_bets']:.0f}</td>
                        <td>{row['total_wins']:.0f}</td>
                    </tr>
                    """
                
                html_content += """
                </table>
                """
                
                # Add player insights
                html_content += """
                <div class="insight">
                """
                
                # Calculate win rate statistics
                avg_win_rate = self.outcomes_df['win_rate'].mean()
                avg_roi = self.outcomes_df['roi'].mean()
                profitable_players = (self.outcomes_df['net_profit'] > 0).sum()
                total_players = len(self.outcomes_df)
                profitable_pct = profitable_players / total_players * 100 if total_players > 0 else 0
                
                html_content += f"""
                <p><strong>Player Insights:</strong></p>
                <ul>
                    <li>Average win rate: {avg_win_rate:.1f}%</li>
                    <li>Average ROI: {avg_roi:.1f}%</li>
                    <li>Profitable players: {profitable_players} out of {total_players} ({profitable_pct:.1f}%)</li>
                </ul>
                """
                
                html_content += """
                </div>
                """
            
            # Close HTML
            html_content += f"""
                <footer>
                    <p>Analysis generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}</p>
                </footer>
            </body>
            </html>
            """
            
            # Write HTML file
            ensure_directory(self.output_dir)
            report_path = os.path.join(self.output_dir, 'gambling_analysis_report.html')
            
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"HTML report generated: {report_path}")
            return report_path
    
    def run_analysis(self):
        """Run the complete gambling analysis pipeline"""
        logger.info("Starting gambling analysis...")
        
        # Load data
        if not self.load_data():
            logger.error("Failed to load gambling data")
            return False
        
        # Analyze user outcomes
        self.analyze_user_outcomes()
        
        # Analyze game performance
        self.analyze_game_performance()
        
        # Analyze transaction patterns
        self.analyze_transaction_patterns()
        
        # Generate visualizations
        self.generate_visualizations()
        
        # Generate HTML report
        report_path = self.generate_html_report()
        
        # Try to open the report
        try:
            import webbrowser
            webbrowser.open('file://' + os.path.abspath(report_path))
            logger.info(f"Opened report in browser: {report_path}")
        except Exception as e:
            logger.error(f"Could not open report automatically: {e}")
        
        logger.info("Gambling analysis complete!")
        return True

def run_gambling_analysis(data_file='balance_data/balance_updates.csv', output_dir='gambling_analysis'):
    """Run gambling analysis from outside the class"""
    analyzer = GamblingAnalyzer(data_file, output_dir)
    return analyzer.run_analysis()

if __name__ == "__main__":
    # Get data file from command line if provided
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        data_file = 'balance_data/balance_updates.csv'
    
    run_gambling_analysis(data_file) 