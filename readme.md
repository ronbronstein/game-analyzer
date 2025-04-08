# Discord Economy Game Data Analyzer

A comprehensive tool for analyzing economy data from Discord games, especially those using UnbelievaBoat.

## Features

- **Data Extraction**: Extract economy transactions from Discord export files
- **Economy Analysis**: Analyze wealth distribution, Gini coefficient, and economy health
- **Gambling Analysis**: Track gambling statistics and ROI for different games
- **Category Analysis**: Analyze transaction patterns by category
- **Web Interface**: Clean, modern UI for running analysis and viewing results
- **Discord Integration**: Direct fetching of Discord data through bot API
- **Advanced Features** (in `advanced-features` branch):
  - Time Series Analysis: Detect trends and patterns in economic activity
  - Player Network Analysis: Visualize and analyze player interactions
  - LLM-Based Summary: Generate executive summaries with key insights

## Getting Started

See [SETUP.md](SETUP.md) for detailed installation and usage instructions.

## Quick Start

```bash
# Run the web interface
python app.py

# Direct Discord data fetching
python fetch_discord_data.py --token YOUR_BOT_TOKEN

# Basic analysis (command line)
python run_analysis.py path/to/export.json

# Use the simple shell script
./run_analysis.sh --file export.json

# Advanced analysis (after switching to advanced-features branch)
python run_complete_analysis.py path/to/export.json
```

## Web Interface

The tool includes a modern web interface that allows you to:

- Fetch Discord data directly using your bot token
- Run different types of analysis with a single click
- View analysis results with larger, interactive visualizations
- Access full HTML reports
- Customize the interface with light/dark themes

To start the web interface:

```bash
python app.py
```

Then open your browser to http://localhost:5000

## Requirements

- Python 3.8+
- Required packages in requirements.txt
- Docker (optional, for Discord data fetching)

## Game Developer Perspective

This tool is designed with game developers in mind, providing:

- Economy health metrics to balance your game
- Player behavior insights to improve engagement
- Gambling analysis to ensure fair mechanics
- Transaction pattern analysis to identify potential exploits

## License

MIT

## Acknowledgements

- Thanks to the Discord UnbelievaBoat economy bot
- Visualization libraries: Matplotlib, Seaborn
- Advanced analysis: NetworkX, StatsModels
- Discord Chat Exporter by Tyrrrz