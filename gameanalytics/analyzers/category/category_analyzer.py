#!/usr/bin/env python3
"""
Category Analysis Module
Analyzes and improves transaction categories with focus on the "Other" category
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import re
import os
import sys
from collections import Counter

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
from gameanalytics.utils import load_csv_data, ensure_directory, set_plot_style, logger, Timer
from gameanalytics.config import ANALYSIS_CONFIG, CATEGORIZATION_CONFIG

class CategoryAnalyzer:
    """Analyze and improve transaction categories"""
    
    def __init__(self, data_file='balance_data/balance_updates.csv', output_dir='category_analysis'):
        """Initialize the category analyzer
        
        Args:
            data_file (str): Path to balance updates CSV
            output_dir (str): Directory to save analysis results
        """
        self.data_file = data_file
        self.output_dir = output_dir
        self.df = None
        self.reason_counts = None
        self.outcomes_df = None
        self.game_stats = None
        
        # Set plot style
        set_plot_style()
    
    def load_data(self):
        """Load the balance data CSV"""
        with Timer("Loading category data"):
            try:
                # Load data
                self.df = load_csv_data(self.data_file)
                
                if self.df.empty:
                    logger.error(f"Failed to load data from {self.data_file}")
                    return False
                
                logger.info(f"Loaded {len(self.df)} transactions from {self.data_file}")
                return True
            except Exception as e:
                logger.error(f"Error loading data: {e}")
                return False
    
    def analyze_other_category(self):
        """Analyze transactions in the 'Other' category"""
        with Timer("Analyzing 'Other' category"):
            # Filter for "Other" category
            other_df = self.df[self.df['category'] == 'Other'].copy()
            
            if other_df.empty:
                logger.warning("No transactions found in the 'Other' category")
                return None
            
            # Count occurrences of each reason
            self.reason_counts = other_df['reason'].value_counts()
            
            logger.info(f"Found {len(other_df)} transactions in the 'Other' category")
            logger.info(f"Top reasons in 'Other' category:")
            for reason, count in self.reason_counts.head(10).items():
                logger.info(f"  {reason}: {count} transactions")
            
            return self.reason_counts
    
    def analyze_transaction_outcomes(self):
        """Analyze betting outcomes (win/loss ratio)"""
        with Timer("Analyzing gambling outcomes"):
            # Look for betting categories
            betting_df = self.df[self.df['category'].isin(['Betting', 'Gambling', 'Animal Race'])].copy()
            
            if betting_df.empty:
                logger.warning("No betting transactions found")
                return None
            
            # Identify winning transactions
            winning_keywords = ['won', 'ended', 'win']
            betting_df['is_win'] = betting_df['reason'].apply(
                lambda x: any(keyword in str(x).lower() for keyword in winning_keywords)
            )
            
            # Group by user
            user_outcomes = betting_df.groupby(['user', 'is_win']).agg({
                'cash_with_sign': ['sum', 'count']
            }).reset_index()
            
            user_outcomes.columns = ['user', 'is_win', 'net_amount', 'transaction_count']
            
            # Reshaping for easier comparison
            user_outcomes_pivot = user_outcomes.pivot(index='user', columns='is_win', 
                                                    values=['transaction_count', 'net_amount'])
            
            # Fill NaNs with 0
            user_outcomes_pivot = user_outcomes_pivot.fillna(0)
            
            # Calculate win rates and net profit
            results = []
            
            for user in user_outcomes_pivot.index:
                try:
                    win_count = user_outcomes_pivot.loc[user, ('transaction_count', True)]
                    loss_count = user_outcomes_pivot.loc[user, ('transaction_count', False)]
                    win_amount = user_outcomes_pivot.loc[user, ('net_amount', True)]
                    loss_amount = user_outcomes_pivot.loc[user, ('net_amount', False)]
                    
                    total_count = win_count + loss_count
                    win_rate = (win_count / total_count) * 100 if total_count > 0 else 0
                    net_profit = win_amount + loss_amount
                    
                    results.append({
                        'user': user,
                        'win_count': win_count,
                        'loss_count': loss_count, 
                        'win_rate': win_rate,
                        'win_amount': win_amount,
                        'loss_amount': loss_amount,
                        'net_profit': net_profit
                    })
                except:
                    # Skip users with missing data
                    continue
            
            self.outcomes_df = pd.DataFrame(results)
            if not self.outcomes_df.empty:
                self.outcomes_df = self.outcomes_df.sort_values('net_profit', ascending=False)
                
                logger.info(f"Analyzed gambling outcomes for {len(self.outcomes_df)} users")
                logger.info(f"Top winners:")
                for _, row in self.outcomes_df.head(5).iterrows():
                    logger.info(f"  {row['user']}: Win rate {row['win_rate']:.1f}%, Net profit {row['net_profit']:.2f}")
            
            return self.outcomes_df
    
    def improved_categorize_transaction(self, reason):
        """Enhanced transaction categorization function"""
        if not reason or not isinstance(reason, str):
            return "Other"
            
        reason = reason.lower()
        
        # Use category keywords from config
        for category, keywords in CATEGORIZATION_CONFIG['category_keywords'].items():
            for keyword in keywords:
                if re.search(keyword, reason):
                    return category
        
        return "Other"
    
    def apply_improved_categories(self):
        """Apply improved categorization to the data"""
        with Timer("Applying improved categorization"):
            # Apply new categorization
            self.df['category'] = self.df['reason'].apply(self.improved_categorize_transaction)
            
            # Calculate category distribution
            category_counts = self.df['category'].value_counts()
            
            logger.info(f"Category distribution:")
            for category, count in category_counts.head(10).items():
                logger.info(f"  {category}: {count} transactions")
            
            return self.df
    
    def analyze_win_lose_ratio(self):
        """Analyze the ratio of wins to losses for gambling activities"""
        with Timer("Analyzing gambling win/loss ratios"):
            # Identify gambling categories
            gambling_cats = [cat for cat in self.df['category'].unique() 
                            if any(term in cat.lower() for term in ['bet', 'win', 'gambling', 'slots', 'blackjack', 'roulette', 'animal race'])]
            
            if not gambling_cats:
                logger.warning("No gambling categories found")
                return None
            
            # Filter for gambling transactions
            gambling_df = self.df[self.df['category'].isin(gambling_cats)].copy()
            
            # Separate wins and bets
            wins = gambling_df[gambling_df['category'].str.contains('Win')]
            bets = gambling_df[gambling_df['category'].str.contains('Bet')]
            
            # Analyze by game type
            game_types = ['Blackjack', 'Roulette', 'Animal Race']
            
            results = []
            
            for game in game_types:
                game_wins = wins[wins['category'].str.contains(game)]
                game_bets = bets[bets['category'].str.contains(game)]
                
                total_win_amount = game_wins['cash_with_sign'].sum()
                total_bet_amount = abs(game_bets['cash_with_sign'].sum())
                
                win_count = len(game_wins)
                bet_count = len(game_bets)
                
                # Calculate payout ratio and house edge
                payout_ratio = total_win_amount / total_bet_amount if total_bet_amount > 0 else 0
                house_edge = 1 - payout_ratio
                
                # Calculate win probability
                win_probability = win_count / bet_count if bet_count > 0 else 0
                
                results.append({
                    'game': game,
                    'win_count': win_count,
                    'bet_count': bet_count,
                    'total_win_amount': total_win_amount,
                    'total_bet_amount': total_bet_amount,
                    'payout_ratio': payout_ratio,
                    'house_edge': house_edge,
                    'win_probability': win_probability
                })
            
            self.game_stats = pd.DataFrame(results)
            
            if not self.game_stats.empty:
                logger.info(f"\nGambling Analysis by Game Type:")
                for _, row in self.game_stats.iterrows():
                    logger.info(f"\n{row['game']}:")
                    logger.info(f"  Win Count: {row['win_count']}, Bet Count: {row['bet_count']}")
                    logger.info(f"  Win Amount: {row['total_win_amount']:,.2f}, Bet Amount: {row['total_bet_amount']:,.2f}")
                    logger.info(f"  Payout Ratio: {row['payout_ratio']:.2f} (House Edge: {row['house_edge']:.2f})")
                    logger.info(f"  Win Probability: {row['win_probability']:.2f}")
            
            return self.game_stats
    
    def generate_visualizations(self):
        """Generate visualizations for the analysis"""
        with Timer("Generating category visualizations"):
            # Create output directory
            ensure_directory(self.output_dir)
            
            # 1. Plot category distribution
            fig, axes = plt.subplots(1, 2, figsize=(18, 12))
            
            # Category counts
            category_counts = self.df['category'].value_counts()
            sns.barplot(x=category_counts.values, y=category_counts.index, ax=axes[0])
            axes[0].set_title('Category Distribution', fontsize=14)
            axes[0].set_xlabel('Transaction Count', fontsize=12)
            
            # Top categories
            top_categories = category_counts.head(15)
            sns.barplot(x=top_categories.values, y=top_categories.index, ax=axes[1])
            axes[1].set_title('Top 15 Categories', fontsize=14)
            axes[1].set_xlabel('Transaction Count', fontsize=12)
            
            plt.tight_layout()
            plt.savefig(os.path.join(self.output_dir, 'category_distribution.png'), dpi=300)
            plt.close()
            
            # 2. Plot 'Other' category reasons
            if self.reason_counts is not None and not self.reason_counts.empty:
                plt.figure(figsize=(14, 10))
                top_reasons = self.reason_counts.head(15)
                sns.barplot(x=top_reasons.values, y=top_reasons.index)
                plt.title('Top 15 Reasons in "Other" Category', fontsize=16)
                plt.xlabel('Transaction Count', fontsize=14)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'other_category_reasons.png'), dpi=300)
                plt.close()
            
            # 3. Plot gambling outcomes by user
            if self.outcomes_df is not None and not self.outcomes_df.empty:
                # Top users by net profit
                plt.figure(figsize=(14, 10))
                top_users = self.outcomes_df.head(15)
                
                sns.barplot(x='net_profit', y='user', data=top_users, hue='user', legend=False)
                plt.title('Top 15 Users by Gambling Net Profit', fontsize=16)
                plt.xlabel('Net Profit', fontsize=14)
                plt.ylabel('User', fontsize=14)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'gambling_net_profit.png'), dpi=300)
                plt.close()
                
                # Win rates
                plt.figure(figsize=(14, 10))
                top_win_rates = self.outcomes_df.sort_values('win_rate', ascending=False).head(15)
                
                sns.barplot(x='win_rate', y='user', data=top_win_rates, hue='user', legend=False)
                plt.title('Top 15 Users by Gambling Win Rate', fontsize=16)
                plt.xlabel('Win Rate (%)', fontsize=14)
                plt.ylabel('User', fontsize=14)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'gambling_win_rates.png'), dpi=300)
                plt.close()
            
            # 4. Plot game statistics
            if self.game_stats is not None and not self.game_stats.empty:
                # Payout ratio by game type
                plt.figure(figsize=(12, 8))
                sns.barplot(x='game', y='payout_ratio', data=self.game_stats)
                plt.title('Payout Ratio by Game Type', fontsize=16)
                plt.xlabel('Game', fontsize=14)
                plt.ylabel('Payout Ratio', fontsize=14)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'payout_ratios.png'), dpi=300)
                plt.close()
                
                # Win probability by game type
                plt.figure(figsize=(12, 8))
                sns.barplot(x='game', y='win_probability', data=self.game_stats)
                plt.title('Win Probability by Game Type', fontsize=16)
                plt.xlabel('Game', fontsize=14)
                plt.ylabel('Win Probability', fontsize=14)
                plt.tight_layout()
                plt.savefig(os.path.join(self.output_dir, 'win_probabilities.png'), dpi=300)
                plt.close()
            
            logger.info(f"Visualizations saved to {self.output_dir}/")
    
    def generate_html_report(self):
        """Generate an HTML report with the analysis results"""
        with Timer("Generating category HTML report"):
            # Count transactions by category
            category_counts = self.df['category'].value_counts()
            
            # Calculate transaction volume by category
            volume_by_category = self.df.groupby('category')['cash_amount'].sum().sort_values(ascending=False)
            
            # Create HTML content
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Transaction Category Analysis</title>
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
                <h1>Transaction Category Analysis</h1>
                
                <div class="insight">
                    <p>This analysis improves the categorization of transactions in the UnbelievaBoat economy game data, 
                    with special focus on the "Other" category and gambling outcomes.</p>
                </div>
                
                <h2>Category Distribution</h2>
                <div class="gallery">
                    <div>
                        <img src="category_distribution.png" class="chart" alt="Category Distribution">
                    </div>
                </div>
                
                <h3>Improved Categories</h3>
                <table>
                    <tr>
                        <th>Category</th>
                        <th>Count</th>
                    </tr>
            """
            
            for category, count in category_counts.head(20).items():
                html_content += f"""
                    <tr>
                        <td>{category}</td>
                        <td>{count}</td>
                    </tr>
                """
            
            html_content += """
                </table>
                
                <h3>Transaction Volume by Category</h3>
                <table>
                    <tr>
                        <th>Category</th>
                        <th>Total Amount</th>
                    </tr>
            """
            
            for category, amount in volume_by_category.head(20).items():
                html_content += f"""
                    <tr>
                        <td>{category}</td>
                        <td>{amount:,.2f}</td>
                    </tr>
                """
            
            html_content += """
                </table>
                
                <h2>"Other" Category Analysis</h2>
            """
            
            if self.reason_counts is not None and not self.reason_counts.empty:
                html_content += """
                <div class="gallery">
                    <div>
                        <img src="other_category_reasons.png" class="chart" alt="Other Category Reasons">
                    </div>
                </div>
                
                <table>
                    <tr>
                        <th>Reason</th>
                        <th>Count</th>
                    </tr>
                """
                
                for reason, count in self.reason_counts.head(20).items():
                    html_content += f"""
                        <tr>
                            <td>{reason}</td>
                            <td>{count}</td>
                        </tr>
                    """
                
                html_content += "</table>"
            
            html_content += """
                <h2>Gambling Analysis</h2>
            """
            
            if self.game_stats is not None and not self.game_stats.empty:
                html_content += """
                <div class="gallery">
                    <div>
                        <img src="payout_ratios.png" class="chart" alt="Payout Ratios">
                    </div>
                    <div>
                        <img src="win_probabilities.png" class="chart" alt="Win Probabilities">
                    </div>
                </div>
                
                <h3>Game Statistics</h3>
                <table>
                    <tr>
                        <th>Game</th>
                        <th>Win Count</th>
                        <th>Bet Count</th>
                        <th>Payout Ratio</th>
                        <th>House Edge</th>
                        <th>Win Probability</th>
                    </tr>
                """
                
                for _, row in self.game_stats.iterrows():
                    html_content += f"""
                        <tr>
                            <td>{row['game']}</td>
                            <td>{row['win_count']}</td>
                            <td>{row['bet_count']}</td>
                            <td class="{'good' if row['payout_ratio'] >= 0.9 else 'bad' if row['payout_ratio'] < 0.7 else 'neutral'}">{row['payout_ratio']:.2f}</td>
                            <td class="{'good' if row['house_edge'] <= 0.1 else 'bad' if row['house_edge'] > 0.3 else 'neutral'}">{row['house_edge']:.2f}</td>
                            <td class="{'good' if row['win_probability'] >= 0.45 else 'bad' if row['win_probability'] < 0.3 else 'neutral'}">{row['win_probability']:.2f}</td>
                        </tr>
                    """
                
                html_content += "</table>"
            
            if self.outcomes_df is not None and not self.outcomes_df.empty:
                html_content += """
                <div class="gallery">
                    <div>
                        <img src="gambling_net_profit.png" class="chart" alt="Gambling Net Profit">
                    </div>
                    <div>
                        <img src="gambling_win_rates.png" class="chart" alt="Gambling Win Rates">
                    </div>
                </div>
                
                <h3>Top Gamblers</h3>
                <table>
                    <tr>
                        <th>User</th>
                        <th>Win Count</th>
                        <th>Loss Count</th>
                        <th>Win Rate</th>
                        <th>Net Profit</th>
                    </tr>
                """
                
                for _, row in self.outcomes_df.head(20).iterrows():
                    html_content += f"""
                        <tr>
                            <td>{row['user']}</td>
                            <td>{row['win_count']:.0f}</td>
                            <td>{row['loss_count']:.0f}</td>
                            <td class="{'good' if row['win_rate'] > 50 else 'bad' if row['win_rate'] < 30 else 'neutral'}">{row['win_rate']:.1f}%</td>
                            <td class="{'good' if row['net_profit'] > 0 else 'bad'}">{row['net_profit']:,.2f}</td>
                        </tr>
                    """
                
                html_content += "</table>"
            
            # Add insights section
            html_content += """
                <h2>Key Insights</h2>
                <div class="insight">
                    <h3>Category Distribution</h3>
                    <ul>
            """
            
            # Calculate percentage of Other category
            other_pct = (category_counts.get('Other', 0) / len(self.df)) * 100
            
            html_content += f"""
                        <li>Original categorization had {category_counts.get('Other', 0)} transactions ({other_pct:.1f}%) in the "Other" category</li>
            """
            
            # Add gambling insights if available
            if self.game_stats is not None and not self.game_stats.empty:
                # Find best and worst games
                best_game = self.game_stats.loc[self.game_stats['payout_ratio'].idxmax()]
                worst_game = self.game_stats.loc[self.game_stats['payout_ratio'].idxmin()]
                
                html_content += f"""
                        <li>Best payout game: {best_game['game']} with {best_game['payout_ratio']:.2f} ratio ({(best_game['payout_ratio'] * 100):.1f}%)</li>
                        <li>Worst payout game: {worst_game['game']} with {worst_game['payout_ratio']:.2f} ratio ({(worst_game['payout_ratio'] * 100):.1f}%)</li>
                """
            
            html_content += """
                    </ul>
                </div>
                
                <footer>
                    <p>Analysis generated on {}</p>
                </footer>
            </body>
            </html>
            """.format(pd.Timestamp.now().strftime('%Y-%m-%d %H:%M'))
            
            # Write HTML file
            report_path = os.path.join(self.output_dir, 'category_analysis_report.html')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"HTML report generated: {report_path}")
            return report_path
    
    def run_analysis(self):
        """Run the complete category analysis pipeline"""
        logger.info("Starting category analysis...")
        
        # Load data
        if not self.load_data():
            logger.error("Failed to load category data")
            return False
        
        # Analyze "Other" category
        self.analyze_other_category()
        
        # Apply improved categorization
        self.apply_improved_categories()
        
        # Analyze gambling outcomes
        self.analyze_transaction_outcomes()
        
        # Analyze win/lose ratio
        self.analyze_win_lose_ratio()
        
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
        
        logger.info("Category analysis complete!")
        return True

def run_category_analysis(data_file='balance_data/balance_updates.csv', output_dir='category_analysis'):
    """Run category analysis from outside the class"""
    analyzer = CategoryAnalyzer(data_file, output_dir)
    return analyzer.run_analysis()

if __name__ == "__main__":
    # Get data file from command line if provided
    if len(sys.argv) > 1:
        data_file = sys.argv[1]
    else:
        data_file = 'balance_data/balance_updates.csv'
    
    run_category_analysis(data_file) 