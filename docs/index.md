# Discord Game Economy Analysis Tool Documentation

Welcome to the documentation for the Discord Game Economy Analysis Tool. This tool helps game developers analyze their game economy to make data-driven decisions.

## Table of Contents

1. [Getting Started](SETUP.md)
   - Installation instructions and basic usage

2. [Contributing](CONTRIBUTING.md)
   - Guidelines for contributing to the project
   - Repository structure and development workflow

3. [Game Rules Reference](game_guide.md)
   - Reference for UnbelievaBoat Discord economy game rules
   - Used as context for analysis and recommendations

## Key Features

- **Data Extraction**: Extract transaction data from Discord exports
- **Economy Analysis**: Analyze wealth distribution, inflation/deflation, and transaction patterns
- **Gambling Analysis**: Evaluate gambling game fairness and player behavior
- **Category Analysis**: Identify and analyze transaction categories
- **Web Interface**: Modern UI for viewing results and fetching data
- **Game Developer Insights**: Recommendations for balancing your game economy

## Advanced Features

- **Player Retention Analysis**: Track player engagement and retention metrics
- **Time Series Analysis**: Identify trends and patterns in economic data
- **Network Analysis**: Visualize player-to-player transactions
- **Executive Summary**: Generate concise summaries of key findings

## Quickstart Guide

```bash
# Start the web interface (recommended)
python app.py

# Or run from command line with all options
python run_analysis.py export.json

# Run with developer-focused metrics
python run_analysis.py export.json --developer-view
``` 