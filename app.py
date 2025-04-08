#!/usr/bin/env python3
"""
Discord Game Economy Analysis Tool - Web Interface
A modern web interface for running analysis and viewing results with AI-powered recommendations
"""

import os
import sys
import logging
from flask import Flask
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import from gameanalytics package
from gameanalytics.utils import logger, ensure_directory
from gameanalytics.routes import init_routes
from gameanalytics.database import init_database
from gameanalytics.errors import handle_error

# Load environment variables
load_dotenv()

# Default Claude API prompt
DEFAULT_CLAUDE_PROMPT = '''
You are an expert game economy analyst reviewing Discord game economy data.
Please analyze the following data and provide clear, insightful recommendations for the game developers.

Important recommendations to consider:
1. Player retention strategies based on user activity patterns
2. Economy balance adjustments if inflation/deflation is detected
3. Game feature optimization based on player preferences
4. Ways to improve player engagement based on transaction patterns
5. Suggestions to address any concerning trends in the data

Please structure your response in the following format:
- Executive Summary (2-3 sentences)
- Key Findings (bullet points, with supporting data)
- Specific Recommendations (prioritized, actionable items)
- Trends to Monitor (what should we keep an eye on)

Don't use technical jargon and focus on clear, actionable insights that will help improve the game economy.

DATA:
{analysis_data}
'''

@handle_error
def create_app():
    """Create and configure the Flask application"""
    # Configure Flask app
    app = Flask(__name__, static_folder='static', template_folder='templates')
    
    # Load configuration from environment variables
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev_key_change_in_production')
    app.config['PORT'] = int(os.getenv('PORT', '5001'))
    app.config['DEBUG'] = os.getenv('DEBUG', 'false').lower() == 'true'
    app.config['THEME'] = os.getenv('THEME', 'dark')
    app.config['IMAGE_SIZE_FACTOR'] = float(os.getenv('IMAGE_SIZE_FACTOR', '1.5'))
    app.config['CLAUDE_API_KEY'] = os.getenv('CLAUDE_API_KEY', '')
    app.config['CLAUDE_MODEL'] = os.getenv('CLAUDE_MODEL', 'claude-3-opus-20240229')
    app.config['CLAUDE_PROMPT'] = os.getenv('CLAUDE_PROMPT', DEFAULT_CLAUDE_PROMPT)
    
    # Initialize the database
    try:
        init_database()
    except Exception as e:
        logger.error(f"Error initializing database: {e}")
        logger.warning("Continuing without database support")
    
    # Initialize routes
    init_routes(app)
    
    return app

@handle_error
def create_required_directories():
    """Create required directories if they don't exist"""
    dirs = [
        'static',
        'templates',
        'balance_data',
        'economy_analysis',
        'gambling_analysis',
        'category_analysis',
        'advanced_analysis',
        'ai_recommendations',
        'database'
    ]
    
    for directory in dirs:
        ensure_directory(directory)

@handle_error
def main():
    """Run the Flask application"""
    # Create required directories
    create_required_directories()
    
    # Create the Flask app
    app = create_app()
    
    # Run the Flask app
    app.run(
        host='0.0.0.0',
        port=app.config['PORT'],
        debug=app.config['DEBUG']
    )

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nApplication interrupted by user.")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1) 