# Discord Economy Game Analysis Tool

An advanced data analysis toolkit for UnbelievaBoat's Discord economy game, providing comprehensive insights into player behavior, economy health, and gambling mechanics.

## Features

- **Comprehensive Analysis**: Extract and analyze Discord game economy data from UnbelievaBoat
- **Economy Health Tracking**: Monitor inflation, wealth distribution, and transaction patterns
- **Gambling Analysis**: Evaluate win rates, expected values, and fairness metrics for gambling games
- **Transaction Categorization**: Automatically categorize transactions with improved accuracy
- **Statistical Analysis**: Perform statistical tests on game mechanics and player behavior
- **Data Visualization**: Generate rich visualizations for all aspects of the economy
- **HTML Reports**: Create browser-viewable reports with detailed insights and recommendations

## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/analyze-game-data.git
cd analyze-game-data
```

2. Install dependencies:
```bash
pip install pandas matplotlib seaborn statsmodels networkx
```

3. Optional dependencies for enhanced features:
```bash
pip install plotly
```

## Usage

### Basic Analysis

Run the complete analysis pipeline on a Discord export file:
```bash
python run_analysis.py path/to/export.json
```

### Advanced Options

Skip data extraction if you've already processed the data:
```bash
python run_analysis.py --skip-extraction
```

Run specific analysis components:
```bash
python run_analysis.py --economy-only
python run_analysis.py --gambling-only
python run_analysis.py --category-only
python run_analysis.py --advanced-only
```

### Output

Results are saved to the following directories:
- `balance_data/`: Extracted transaction data
- `economy_analysis/`: Economy health metrics and visualizations
- `gambling_analysis/`: Gambling activity analysis
- `category_analysis/`: Transaction category analysis
- `advanced_analysis/`: Advanced metrics and network analysis

HTML reports are automatically opened in your default browser.

## Project Structure

```
analyze-game-data/
├── gameanalytics/             # Main package
│   ├── extractors/            # Data extraction modules
│   ├── processors/            # Data processing modules
│   ├── analyzers/             # Analysis modules
│   │   ├── economy/           # Economy analysis
│   │   ├── gambling/          # Gambling analysis
│   │   └── category/          # Category analysis
│   ├── visualizers/           # Visualization modules
│   ├── config.py              # Configuration settings
│   └── utils.py               # Shared utility functions
├── run_analysis.py            # Main execution script
├── archive/                   # Archive of old scripts
├── IMPROVEMENTS.md            # Documentation of improvements
└── REORGANIZATION_SUMMARY.md  # Summary of code reorganization
```

## Key Metrics

The tool analyzes and generates the following key metrics:

- **Economy Health**
  - Total currency in circulation
  - Inflation/deflation rates
  - Gini coefficient for wealth inequality
  - Top earners and spenders

- **Gambling Analysis**
  - House edge for each game type
  - Win probabilities and payout ratios
  - Expected values for each game
  - Statistical fairness tests

- **Transaction Patterns**
  - Daily/weekly activity cycles
  - Popular transaction categories
  - User interaction networks
  - User behavior patterns

## Documentation

For more detailed information:
- `IMPROVEMENTS.md`: Details on improvements made to the codebase
- `REORGANIZATION_SUMMARY.md`: Overview of the code reorganization

## Troubleshooting

**Export file too large**  
If the Discord export.json is too large, try splitting it into smaller time periods or run with `--skip-extraction` if you've already extracted the data.

**Missing dependencies**  
Run `pip install -r requirements.txt` to ensure all dependencies are installed.

**Report display issues**  
Make sure you have a modern web browser installed. Reports use modern HTML/CSS features.

## License

This project is available under the MIT License.

## Acknowledgements

- UnbelievaBoat Discord bot creators for their excellent economy game
- The pandas, matplotlib, and seaborn teams for their data analysis tools