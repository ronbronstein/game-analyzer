# Advanced Features for Discord Economy Analysis

This branch adds two major advanced features to the Discord Economy Analysis Tool:

## 1. Advanced Economy Analysis

The Advanced Economy Analysis module provides deeper insights into the economy's health and behavior using advanced statistical methods:

### Time Series Analysis
- **Seasonal Decomposition**: Identifies weekly patterns in economic activity
- **ARIMA Forecasting**: Predicts future economic trends based on historical data
- **Temporal Patterns**: Analyzes day-of-week effects on economy activity

### Player Network Analysis
- **Transaction Network**: Visualizes player-to-player transactions as a network graph
- **Centrality Metrics**: Identifies key players based on transaction patterns
- **Community Detection**: Finds groups of players who frequently transact together

## 2. LLM-Based Summary Generator

The LLM Summary Generator provides executive summaries of all analysis results using natural language models:

### Features
- **Cross-Analysis Insights**: Combines data from all analysis modules
- **Executive Summary**: Concise overview of key economic findings
- **Recommendations**: Suggested actions to improve economy health
- **Flexible Integration**: Works with OpenAI API or uses local generation as fallback

### Usage

To use these advanced features, run the complete analysis pipeline:

```bash
# With OpenAI API key for LLM summaries
python run_complete_analysis.py export.json --api-key YOUR_OPENAI_API_KEY

# Without OpenAI (uses local summary generation)
python run_complete_analysis.py export.json

# Skip extraction if already processed
python run_complete_analysis.py --skip-extraction
```

## Technical Implementation

- The Advanced Economy Analysis uses `statsmodels` for time series analysis and forecasting
- Network analysis is implemented using `networkx` with optional community detection via `python-louvain`
- The LLM Summary Generator uses the OpenAI API if an API key is provided, with a rule-based local fallback

## Outputs

- Advanced time series visualizations in `advanced_analysis/`
- Player network graphs showing transaction relationships
- Executive summary report in `summary_analysis/executive_summary.html` 