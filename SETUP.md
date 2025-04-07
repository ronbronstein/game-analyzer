# Setup and Usage Instructions

This document provides instructions for setting up and running the Discord Economy Analysis Tool.

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/yourusername/analyze-game-data.git
cd analyze-game-data
```

### 2. Create a Virtual Environment

```bash
# Python 3.8+ recommended
python -m venv venv

# Activate the environment:
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate
```

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

## Basic Usage

### 1. Analyze Discord Export Data

To analyze your Discord economy data, export the conversation history from your Discord server and run:

```bash
python run_analysis.py path/to/export.json
```

This will:
- Extract economy data from the Discord export
- Analyze economy health metrics
- Analyze gambling activities
- Analyze transaction categories
- Generate HTML reports in their respective directories

### 2. View Reports

After running the analysis, HTML reports will open automatically in your default browser:
- Economy Health Report: `economy_analysis/economy_analysis_report.html`
- Gambling Analysis Report: `gambling_analysis/gambling_analysis_report.html`
- Category Analysis Report: `category_analysis/category_analysis_report.html`

### 3. Advanced Features (Optional)

For advanced features, switch to the advanced-features branch:

```bash
git checkout advanced-features
```

Then run the complete analysis pipeline:

```bash
python run_complete_analysis.py path/to/export.json
```

This adds:
- Advanced Time Series Analysis
- Player Network Analysis
- LLM-Based Summary Generation

## Command-Line Options

### Basic Analysis

```bash
# Skip extraction if you've already processed the data
python run_analysis.py --skip-extraction

# Run only specific components
python run_analysis.py --economy-only
python run_analysis.py --gambling-only
python run_analysis.py --category-only
```

### Advanced Analysis 

```bash
# With OpenAI API for LLM summaries (on advanced-features branch)
python run_complete_analysis.py path/to/export.json --api-key YOUR_OPENAI_API_KEY

# Skip LLM summary generation
python run_complete_analysis.py path/to/export.json --skip-llm
```

## Troubleshooting

### Data Extraction Issues

If you encounter issues with data extraction:

1. Check that your export.json file is valid JSON
2. Try running the fix script: `python tools/fixes/fix_json_better.py export.json`
3. Ensure your Discord export contains UnbelievaBoat messages

### Missing Dependencies

If you encounter missing dependencies errors, try:

```bash
pip install -r requirements.txt --upgrade
```

For network analysis features, you may need to install additional packages:

```bash
pip install python-louvain
```

### OpenAI API Issues

If using the LLM summary feature with OpenAI API:

1. Ensure your API key is valid
2. Set the API key as an environment variable:
   ```bash
   # Windows
   set OPENAI_API_KEY=your_api_key_here
   
   # macOS/Linux
   export OPENAI_API_KEY=your_api_key_here
   ```
3. If API access fails, the system will fall back to local summary generation 