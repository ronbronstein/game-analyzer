"""
Configuration settings for the game analytics package
"""

# Default directories
DEFAULT_DIRS = {
    'extraction': 'balance_data',
    'economy': 'economy_analysis',
    'gambling': 'gambling_analysis',
    'category': 'category_analysis',
    'advanced': 'advanced_analysis'
}

# Extraction settings
EXTRACTION_CONFIG = {
    'chunk_size': 1000,  # Number of messages to process in each parallel chunk
    'default_max_workers': None,  # None = auto-determine based on CPU cores
    'regex_timeout': 1.0  # Timeout for regex pattern matching (seconds)
}

# Analysis settings
ANALYSIS_CONFIG = {
    'min_transactions_for_analysis': 10,  # Minimum transactions needed for statistical analysis
    'min_days_for_trends': 7,  # Minimum days needed for trend analysis
    'min_days_for_seasonality': 14,  # Minimum days needed for seasonal decomposition
    'significance_threshold': 0.05,  # p-value threshold for statistical significance
    'top_n_users': 20,  # Number of top users to include in reports
    'top_n_categories': 15  # Number of top categories to include in reports
}

# Visualization settings
VISUALIZATION_CONFIG = {
    'dpi': 300,  # DPI for saved images
    'figure_width': 12,  # Default figure width in inches
    'figure_height': 8,  # Default figure height in inches
    'color_palette': 'deep',  # Seaborn color palette
    'style': 'whitegrid',  # Seaborn style
    'font_size': 12,  # Default font size
    'title_font_size': 16,  # Title font size
    'label_font_size': 14,  # Axis label font size
}

# Categorization settings
CATEGORIZATION_CONFIG = {
    # Mapping of category names to keywords
    'category_keywords': {
        'Work': ['work'],
        'Daily Bonus': ['daily'],
        'Role Income': ['role income'],
        'Passive Income': ['income'],
        'Chat Reward': ['chat money'],
        'Robbery': ['rob'],
        'Robbed': ['robbed'],
        'Animal Race Bet': ['animal.*race.*bet'],
        'Animal Race Win': ['animal.*race.*won'],
        'Blackjack Bet': ['blackjack.*bet'],
        'Blackjack Win': ['blackjack.*ended'],
        'Roulette Bet': ['roulette.*bet'],
        'Roulette Win': ['roulette.*won'],
        'Slots & Other Gambling': ['slot-machine', 'gamble', 'slot', 'coinflip', 'dice'],
        'Player Transfer': ['give-money', 'transfer', 'sent to'],
        'Shopping': ['shop', 'buy', 'purchase', 'store'],
        'Crime': ['crime'],
        'Slut Command': ['slut'],
        'Begging': ['beg'],
        'Refund': ['refund'],
        'Admin Commands': ['reset', 'remove-money']
    },
    
    # List of categories for gambling analysis
    'gambling_categories': [
        'Animal Race Bet', 'Animal Race Win',
        'Blackjack Bet', 'Blackjack Win',
        'Roulette Bet', 'Roulette Win',
        'Slots & Other Gambling'
    ],
    
    # Legacy categories (for backward compatibility)
    'legacy_gambling_categories': ['Betting', 'Gambling', 'Animal Race']
}

# Report settings
REPORT_CONFIG = {
    'open_browser_automatically': True,
    'include_css': True,
    'timestamp_format': '%Y-%m-%d %H:%M'
}

# Debug settings
DEBUG_CONFIG = {
    'verbose_output': False,
    'log_level': 'INFO',  # Options: DEBUG, INFO, WARNING, ERROR, CRITICAL
    'log_file': 'gameanalytics.log'
}

# Custom configuration based on run modes
RUN_MODES = {
    'quick': {
        'chunk_size': 2000,
        'top_n_users': 10,
        'top_n_categories': 10
    },
    'thorough': {
        'chunk_size': 500,
        'significance_threshold': 0.01,
        'top_n_users': 30,
        'top_n_categories': 20
    },
    'debug': {
        'chunk_size': 100,
        'verbose_output': True,
        'log_level': 'DEBUG'
    }
} 