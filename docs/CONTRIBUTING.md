# Contributing to Discord Game Economy Analysis Tool

This document provides guidelines for contributing to the Discord Game Economy Analysis Tool.

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
├── fetch_discord_data.py      # Script to fetch Discord data
├── app.py                     # Web interface
├── requirements.txt           # Dependencies
├── LICENSE                    # MIT License
├── README.md                  # Project overview
└── docs/                      # Documentation
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

### Setting up the Environment

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/analyze-game-data.git
   cd analyze-game-data
   ```

2. Create a virtual environment:
   ```bash
   python -m venv venv
   
   # On Windows:
   venv\Scripts\activate
   
   # On macOS/Linux:
   source venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Install the package in development mode:
   ```bash
   pip install -e .
   ```

### Making Changes

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

## GitHub Setup

### Connecting to GitHub

1. Create a new GitHub repository:
   - Name: `analyze-game-data` (or choose your own name)
   - Description: "A comprehensive Discord economy game data analyzer"
   - Visibility: Public or Private as preferred
   - Don't initialize with README/license (we already have these)

2. Add the remote repository:
   ```bash
   git remote add origin https://github.com/yourusername/analyze-game-data.git
   ```

3. Push your code:
   ```bash
   # Push the main branch
   git push -u origin main
   
   # Push the advanced-features branch
   git checkout advanced-features
   git push -u origin advanced-features
   ```

### GitHub Pages (Optional)

To set up GitHub Pages for documentation:

1. Configure GitHub Pages in your repo settings:
   - Go to "Settings" > "Pages"
   - Set source to "Deploy from a branch"
   - Select "main" branch and "/docs" folder

2. Add documentation files to the docs directory:
   ```bash
   cp README.md docs/index.md
   ```

### GitHub Actions (Optional)

You can set up GitHub Actions for automated testing:

1. Create a workflows directory:
   ```bash
   mkdir -p .github/workflows
   ```

2. Create a basic workflow file:
   ```bash
   touch .github/workflows/python-tests.yml
   ```

3. Add a workflow configuration to run tests automatically.

## Testing

Before submitting changes, please run tests:

```bash
# Run unit tests
pytest

# Run the application to verify it works
python run_analysis.py --skip-extraction
```

## Code Style

Please follow these style guidelines:

- Use PEP 8 guidelines for Python code
- Add docstrings for all functions and classes
- Comment complex code sections
- Use meaningful variable and function names

## Submitting Changes

1. Create a new branch for your feature:
   ```bash
   git checkout -b feature/your-feature-name
   ```

2. Make your changes and commit them:
   ```bash
   git add .
   git commit -m "Description of your changes"
   ```

3. Push your branch to GitHub:
   ```bash
   git push -u origin feature/your-feature-name
   ```

4. Create a pull request on GitHub to merge your changes.

## Questions?

If you have questions about contributing, please open an issue on GitHub. 