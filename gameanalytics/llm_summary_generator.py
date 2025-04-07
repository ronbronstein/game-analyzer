#!/usr/bin/env python3
"""
LLM-Based Economy Analysis Summary Generator
Generates executive summary and insights using LLM across all analysis components
"""

import os
import sys
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
import logging
import requests
from pathlib import Path
import re

# Add project root to path for imports
sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from gameanalytics.utils import ensure_directory, logger, Timer

class LLMSummaryGenerator:
    """Generate comprehensive summaries and insights using LLM"""
    
    def __init__(self, input_dirs=None, output_dir='summary_analysis', api_key=None, model="gpt-3.5-turbo"):
        """Initialize the summary generator with data sources and output directory"""
        # Default input directories
        if input_dirs is None:
            self.input_dirs = {
                'balance_data': 'balance_data',
                'economy': 'economy_analysis',
                'gambling': 'gambling_analysis',
                'category': 'category_analysis',
                'advanced': 'advanced_analysis'
            }
        else:
            self.input_dirs = input_dirs
            
        self.output_dir = output_dir
        self.api_key = api_key  # OpenAI API key
        self.model = model  # Default model
        self.data = {}  # Container for loaded data
        self.html_reports = {}  # Container for HTML report paths
        self.insights = {}  # Container for generated insights
        
        # Flag to determine if we're using OpenAI API
        self.use_openai = self.api_key is not None and self.api_key.startswith("sk-")
        
        # Ensure the output directory exists
        ensure_directory(self.output_dir)
    
    def load_analysis_data(self):
        """Load data from all analysis components"""
        with Timer("Loading analysis data for summary"):
            # Load balance data stats
            try:
                stats_path = os.path.join(self.input_dirs['balance_data'], 'stats_summary.json')
                if os.path.exists(stats_path):
                    with open(stats_path, 'r') as f:
                        self.data['stats'] = json.load(f)
                    logger.info(f"Loaded basic stats from {stats_path}")
            except Exception as e:
                logger.error(f"Error loading stats summary: {e}")
            
            # Find and store HTML report paths
            self.find_html_reports()
            
            # Load CSV summaries if available
            csv_files = {
                'user_summary': os.path.join(self.input_dirs['balance_data'], 'user_summary.csv'),
                'category_summary': os.path.join(self.input_dirs['balance_data'], 'category_summary.csv'),
                'daily_summary': os.path.join(self.input_dirs['balance_data'], 'daily_summary.csv')
            }
            
            for name, path in csv_files.items():
                try:
                    if os.path.exists(path):
                        df = pd.read_csv(path)
                        # Convert to dict for easier serialization
                        if len(df) <= 50:  # Only store if not too large
                            self.data[name] = df.to_dict(orient='records')
                        else:
                            # Store only top and bottom entries
                            self.data[name] = {
                                'top': df.head(20).to_dict(orient='records'),
                                'bottom': df.tail(5).to_dict(orient='records'),
                                'shape': df.shape
                            }
                        logger.info(f"Loaded {name} from {path}")
                except Exception as e:
                    logger.error(f"Error loading {name}: {e}")
            
            # Extract key metrics from HTML reports
            self.extract_metrics_from_reports()
            
            return len(self.data) > 0
    
    def find_html_reports(self):
        """Find and store paths to all HTML reports"""
        report_patterns = {
            'economy': '*economy*report*.html',
            'gambling': '*gambling*report*.html',
            'category': '*category*report*.html',
            'advanced': '*advanced*report*.html'
        }
        
        for analysis_type, pattern in report_patterns.items():
            if analysis_type in self.input_dirs:
                dir_path = self.input_dirs[analysis_type]
                if os.path.exists(dir_path):
                    # Find matching HTML files
                    matches = list(Path(dir_path).glob(pattern))
                    if matches:
                        self.html_reports[analysis_type] = str(matches[0])
                        logger.info(f"Found {analysis_type} report: {matches[0]}")
    
    def extract_metrics_from_reports(self):
        """Extract key metrics from HTML reports using regex patterns"""
        self.data['metrics'] = {}
        
        # Different patterns to extract from different report types
        extraction_patterns = {
            'economy': {
                'gini_coefficient': r'Gini Coefficient: ([\d\.]+)',
                'top_users': r'<h3>Top Users</h3>.*?<table>(.*?)</table>',
                'wealth_distribution': r'<h3>Wealth Distribution</h3>.*?<table>(.*?)</table>'
            },
            'gambling': {
                'win_rate': r'Overall Win Rate: ([\d\.]+)%',
                'house_edge': r'House Edge: ([\d\.]+)%',
                'most_profitable_game': r'Most Profitable Game: ([^<]+)',
                'top_winners': r'<h3>Top Winners</h3>.*?<table>(.*?)</table>'
            },
            'category': {
                'top_categories': r'<h3>Top Transaction Categories</h3>.*?<table>(.*?)</table>',
                'category_distribution': r'<h3>Category Distribution</h3>.*?<table>(.*?)</table>'
            },
            'advanced': {
                'forecast_trend': r'trending <strong>([^<]+)</strong>',
                'seasonality': r'<h3>([^<]+Seasonal Patterns[^<]+)</h3>'
            }
        }
        
        for report_type, patterns in extraction_patterns.items():
            if report_type in self.html_reports:
                try:
                    # Read the HTML file
                    with open(self.html_reports[report_type], 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Extract metrics
                    report_metrics = {}
                    for metric_name, pattern in patterns.items():
                        match = re.search(pattern, content, re.DOTALL)
                        if match:
                            # Simple metrics have one capture group
                            if len(match.groups()) == 1 and '<table>' not in pattern:
                                report_metrics[metric_name] = match.group(1)
                            # Tables need special handling
                            elif '<table>' in pattern:
                                # Extract table HTML and convert to structured data (simplified)
                                table_html = match.group(1)
                                report_metrics[metric_name] = table_html
                    
                    self.data['metrics'][report_type] = report_metrics
                    logger.info(f"Extracted metrics from {report_type} report")
                except Exception as e:
                    logger.error(f"Error extracting metrics from {report_type} report: {e}")
    
    def prepare_prompt(self):
        """Prepare the LLM prompt with all analysis data"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
        
        # Basic stats
        stats_text = "No basic statistics available."
        if 'stats' in self.data:
            stats = self.data['stats']
            stats_text = f"""
Basic Statistics:
- Total transactions: {stats.get('total_transactions', 'N/A')}
- Unique users: {stats.get('unique_users', 'N/A')}
- Date range: {stats.get('date_range', 'N/A')}
- Total cash volume: {stats.get('total_cash_volume', 'N/A')}
- Net cash change: {stats.get('net_cash_change', 'N/A')}
            """
        
        # User summary
        user_text = "No user summary available."
        if 'user_summary' in self.data:
            if isinstance(self.data['user_summary'], dict) and 'top' in self.data['user_summary']:
                # Only display a few top users
                top_users = self.data['user_summary']['top'][:5]
                user_text = "Top Users by Transaction Count:\n"
                for i, user in enumerate(top_users, 1):
                    user_text += f"- {i}. {user.get('user', 'Unknown')}: {user.get('transaction_count', 0)} transactions, Net cash: {user.get('net_cash', 0)}\n"
            else:
                # Full user summary available
                users = self.data['user_summary'][:5]  # Only first 5
                user_text = "Top Users by Transaction Count:\n"
                for i, user in enumerate(users, 1):
                    user_text += f"- {i}. {user.get('user', 'Unknown')}: {user.get('transaction_count', 0)} transactions, Net cash: {user.get('net_cash', 0)}\n"
        
        # Category summary
        category_text = "No category summary available."
        if 'category_summary' in self.data:
            if isinstance(self.data['category_summary'], dict) and 'top' in self.data['category_summary']:
                top_categories = self.data['category_summary']['top'][:5]
                category_text = "Top Transaction Categories:\n"
                for i, category in enumerate(top_categories, 1):
                    category_text += f"- {i}. {category.get('category', 'Unknown')}: {category.get('transaction_count', 0)} transactions\n"
            else:
                categories = self.data['category_summary'][:5]  # Only first 5
                category_text = "Top Transaction Categories:\n"
                for i, category in enumerate(categories, 1):
                    category_text += f"- {i}. {category.get('category', 'Unknown')}: {category.get('transaction_count', 0)} transactions\n"
        
        # Metrics from HTML reports
        metrics_text = ""
        if 'metrics' in self.data:
            metrics = self.data['metrics']
            
            # Economy metrics
            if 'economy' in metrics:
                economy_metrics = metrics['economy']
                metrics_text += "\nEconomy Analysis:\n"
                if 'gini_coefficient' in economy_metrics:
                    metrics_text += f"- Gini Coefficient: {economy_metrics['gini_coefficient']}\n"
            
            # Gambling metrics
            if 'gambling' in metrics:
                gambling_metrics = metrics['gambling']
                metrics_text += "\nGambling Analysis:\n"
                if 'win_rate' in gambling_metrics:
                    metrics_text += f"- Overall Win Rate: {gambling_metrics['win_rate']}%\n"
                if 'house_edge' in gambling_metrics:
                    metrics_text += f"- House Edge: {gambling_metrics['house_edge']}%\n"
                if 'most_profitable_game' in gambling_metrics:
                    metrics_text += f"- Most Profitable Game: {gambling_metrics['most_profitable_game']}\n"
            
            # Advanced metrics
            if 'advanced' in metrics:
                advanced_metrics = metrics['advanced']
                metrics_text += "\nAdvanced Analysis:\n"
                if 'forecast_trend' in advanced_metrics:
                    metrics_text += f"- Forecast Trend: {advanced_metrics['forecast_trend']}\n"
                if 'seasonality' in advanced_metrics:
                    metrics_text += f"- Seasonality: {advanced_metrics['seasonality']}\n"
        
        # Build the final prompt
        prompt = f"""
You are an expert economic analyst examining data from a Discord economy game.

Based on the following analysis data, create a concise executive summary with key insights, patterns, and recommendations.

DATA SUMMARY (as of {timestamp}):

{stats_text}

{user_text}

{category_text}

{metrics_text}

Please provide:
1. An executive summary (3-5 sentences)
2. Key insights and patterns (3-5 bullet points)
3. Recommendations for economy health and balance (2-3 suggestions)

Focus on economy patterns, user behavior, gambling activity, and wealth distribution. Be concise but insightful.
"""
        return prompt
    
    def call_openai_api(self, prompt):
        """Call OpenAI API to generate insights"""
        try:
            # API endpoint and headers
            url = "https://api.openai.com/v1/chat/completions"
            headers = {
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}"
            }
            
            # Request payload
            payload = {
                "model": self.model,
                "messages": [
                    {"role": "system", "content": "You are an expert economic analyst specializing in virtual economies in games."},
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.7,
                "max_tokens": 1000
            }
            
            # Make API call
            response = requests.post(url, headers=headers, json=payload)
            
            # Handle response
            if response.status_code == 200:
                result = response.json()
                if 'choices' in result and len(result['choices']) > 0:
                    return result['choices'][0]['message']['content']
                else:
                    logger.error("No content returned from OpenAI API")
                    return None
            else:
                logger.error(f"API error: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error calling OpenAI API: {e}")
            return None
    
    def generate_local_insights(self, prompt):
        """Generate insights locally as a fallback without using an API"""
        # Create a simple rule-based summary when API is not available
        
        # Extract key statistics
        total_transactions = 0
        unique_users = 0
        date_range = "unknown period"
        net_cash_change = 0
        
        if 'stats' in self.data:
            stats = self.data['stats']
            total_transactions = stats.get('total_transactions', 0)
            unique_users = stats.get('unique_users', 0)
            date_range = stats.get('date_range', 'unknown period')
            net_cash_change = stats.get('net_cash_change', 0)
        
        # Extract top categories
        top_categories = []
        if 'category_summary' in self.data:
            if isinstance(self.data['category_summary'], dict) and 'top' in self.data['category_summary']:
                categories = self.data['category_summary']['top'][:3]
                top_categories = [cat.get('category', 'Unknown') for cat in categories]
            else:
                categories = self.data['category_summary'][:3]
                top_categories = [cat.get('category', 'Unknown') for cat in categories]
        
        # Generate a basic summary
        economy_state = "growing" if net_cash_change > 0 else "contracting"
        
        executive_summary = f"""
The Discord economy analyzed {total_transactions} transactions from {unique_users} users over {date_range}. 
The economy is {economy_state} with a net change of {net_cash_change} currency units. 
The most active transaction categories were {', '.join(top_categories) if top_categories else 'not identified'}.
"""
        
        # Key insights - generic but informative
        insights = f"""
Key Insights:
- The economy shows {economy_state} tendencies based on the net cash flow.
- The community has {unique_users} active participants in the economy.
- Transaction activity is concentrated in {', '.join(top_categories) if top_categories else 'various'} categories.
- Regular monitoring of transaction patterns is recommended to maintain economy health.
- Consider balancing income and spending mechanisms to manage inflation/deflation.
"""
        
        # Basic recommendations
        recommendations = f"""
Recommendations:
1. {'Implement currency sinks to reduce inflation' if net_cash_change > 0 else 'Introduce additional currency sources to stimulate the economy'}
2. Monitor user engagement across different game activities for balanced participation
3. Regularly review the gambling mechanics to ensure they maintain appropriate house edge
"""
        
        return executive_summary + insights + recommendations
    
    def generate_insights(self):
        """Generate insights using LLM (either OpenAI or local)"""
        with Timer("Generating insights"):
            # Prepare the prompt with all data
            prompt = self.prepare_prompt()
            
            # Try to call OpenAI if API key is available
            if self.use_openai:
                logger.info(f"Calling OpenAI API with model: {self.model}")
                insights = self.call_openai_api(prompt)
                if insights:
                    self.insights = {'text': insights, 'source': 'openai'}
                    return True
                else:
                    logger.warning("OpenAI API call failed, falling back to local insights generation")
            
            # Fallback to local insights
            logger.info("Generating insights locally")
            insights = self.generate_local_insights(prompt)
            self.insights = {'text': insights, 'source': 'local'}
            
            return True
    
    def generate_html_report(self):
        """Generate an HTML report with the insights"""
        with Timer("Generating HTML summary report"):
            # Get insights text
            insights_text = self.insights.get('text', 'No insights available.')
            insights_source = self.insights.get('source', 'unknown')
            
            # Format the insights for HTML (convert newlines, etc.)
            formatted_insights = insights_text.replace('\n', '<br>')
            
            # Create HTML content
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M")
            html_content = f"""
            <!DOCTYPE html>
            <html>
            <head>
                <title>Economy Analysis Executive Summary</title>
                <style>
                    body {{ font-family: Arial, sans-serif; line-height: 1.6; margin: 0; padding: 0; background-color: #f5f5f5; color: #333; }}
                    .container {{ max-width: 800px; margin: 0 auto; padding: 20px; }}
                    .header {{ background-color: #2c3e50; color: white; padding: 20px; text-align: center; border-radius: 5px 5px 0 0; }}
                    .content {{ background-color: white; padding: 30px; border-radius: 0 0 5px 5px; box-shadow: 0 2px 10px rgba(0,0,0,0.1); }}
                    h1 {{ margin-top: 0; }}
                    h2 {{ color: #3498db; border-bottom: 1px solid #eee; padding-bottom: 10px; margin-top: 30px; }}
                    .insights {{ background-color: #f9f9f9; padding: 20px; border-left: 4px solid #3498db; margin: 20px 0; }}
                    .footer {{ margin-top: 30px; font-size: 0.8em; text-align: center; color: #7f8c8d; }}
                    .source-info {{ font-style: italic; color: #7f8c8d; margin-top: 20px; }}
                </style>
            </head>
            <body>
                <div class="container">
                    <div class="header">
                        <h1>Discord Economy Analysis</h1>
                        <p>Executive Summary & Insights</p>
                    </div>
                    
                    <div class="content">
                        <h2>Economic Analysis Summary</h2>
                        
                        <div class="insights">
                            {formatted_insights}
                        </div>
                        
                        <h2>Individual Reports</h2>
                        <p>For detailed analysis, please refer to the following reports:</p>
                        <ul>
                            {'<li><a href="../' + self.html_reports.get('economy', '#') + '">Economy Analysis Report</a></li>' if 'economy' in self.html_reports else ''}
                            {'<li><a href="../' + self.html_reports.get('gambling', '#') + '">Gambling Analysis Report</a></li>' if 'gambling' in self.html_reports else ''}
                            {'<li><a href="../' + self.html_reports.get('category', '#') + '">Category Analysis Report</a></li>' if 'category' in self.html_reports else ''}
                            {'<li><a href="../' + self.html_reports.get('advanced', '#') + '">Advanced Economy Analysis Report</a></li>' if 'advanced' in self.html_reports else ''}
                        </ul>
                        
                        <p class="source-info">Analysis generated using {insights_source.upper()} on {timestamp}</p>
                        
                        <div class="footer">
                            <p>Discord Economy Analysis Tool &copy; {datetime.now().year}</p>
                        </div>
                    </div>
                </div>
            </body>
            </html>
            """
            
            # Write HTML file
            report_path = os.path.join(self.output_dir, 'executive_summary.html')
            with open(report_path, 'w', encoding='utf-8') as f:
                f.write(html_content)
            
            logger.info(f"Summary report generated: {report_path}")
            return True
    
    def run(self):
        """Run the complete summary generation pipeline"""
        # Load data from all analysis components
        if not self.load_analysis_data():
            logger.error("Failed to load analysis data")
            return False
        
        # Generate insights
        if not self.generate_insights():
            logger.error("Failed to generate insights")
            return False
        
        # Generate HTML report
        if not self.generate_html_report():
            logger.error("Failed to generate HTML report")
            return False
        
        logger.info("Summary generation completed successfully!")
        return True


def generate_summary(api_key=None, model="gpt-3.5-turbo"):
    """Generate summary from outside the class"""
    logger.info("Generating economic analysis summary...")
    
    try:
        # Create summary generator
        generator = LLMSummaryGenerator(api_key=api_key, model=model)
        
        # Run the generator
        success = generator.run()
        
        if success:
            logger.info("Summary generation completed successfully.")
            
            # Try to open the report
            report_path = os.path.join(generator.output_dir, 'executive_summary.html')
            try:
                import webbrowser
                webbrowser.open('file://' + os.path.abspath(report_path))
                logger.info(f"Opened summary report: {report_path}")
            except:
                logger.info(f"Summary report available at: {report_path}")
            
            return True
        else:
            logger.error("Summary generation failed")
            return False
    except Exception as e:
        logger.error(f"Error generating summary: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

if __name__ == "__main__":
    # Check if API key is available in environment variables
    import os
    api_key = os.environ.get('OPENAI_API_KEY', None)
    
    if len(sys.argv) > 1:
        api_key = sys.argv[1]
    
    generate_summary(api_key=api_key) 