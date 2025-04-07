# Project Guide

## Repository Structure

The Discord Economy Analysis Tool is organized as follows:

```
analyze-game-data/
├── gameanalytics/             # Main package
│   ├── analyzers/             # Analysis modules
│   │   ├── economy/           # Economy analysis
│   │   ├── gambling/          # Gambling analysis
│   │   ├── category/          # Category analysis
│   │   └── advanced_analyzer.py # Advanced analysis (in advanced branch)
│   ├── extractors/            # Data extraction modules
│   └── utils.py               # Utility functions
├── run_analysis.py            # Basic analysis script
├── run_complete_analysis.py   # Advanced analysis script (in advanced branch) 
├── requirements.txt           # Dependencies
├── LICENSE                    # MIT License
├── README.md                  # Quick overview
├── SETUP.md                   # Setup instructions
└── PROJECT_GUIDE.md           # This file
```

## Branches

The repository has two main branches:

1. **main**: Contains the core economy analysis functionality
   - Extraction of transaction data
   - Basic economy, gambling, and category analysis
   - HTML report generation

2. **advanced-features**: Extends the main branch with advanced features
   - Time series analysis with forecasting
   - Player network analysis
   - LLM-based summary generation
   - Comprehensive analysis pipeline

## Development Workflow

When making changes to the codebase:

1. For basic features and improvements:
   ```bash
   git checkout main
   # Make changes
   git add .
   git commit -m "Description of changes"
   ```

2. For advanced features:
   ```bash
   git checkout advanced-features
   # Make changes
   git add .
   git commit -m "Description of advanced changes"
   ```

3. To merge advanced features into main (after testing):
   ```bash
   git checkout main
   git merge advanced-features
   # Resolve any conflicts
   git commit -m "Merge advanced features into main"
   ```

## Adding New Components

### Adding a New Analyzer

1. Create a new file in the appropriate directory:
   ```bash
   touch gameanalytics/analyzers/new_analyzer.py
   ```

2. Follow the existing patterns, including:
   - Proper class structure
   - Loading data method
   - Analysis methods
   - Report generation
   - Timer decorators for performance monitoring

3. Update run_analysis.py to include your new analyzer

### Adding New Data Sources

To add support for new data sources:

1. Create a new extractor in gameanalytics/extractors/
2. Implement required methods for data extraction
3. Update the main extraction function in run_analysis.py 