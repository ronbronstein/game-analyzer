#!/bin/bash
# Discord Game Economy Analysis Tool - Run Script
# This script provides an easy way to run the analysis tool with common options

# Set default variables
EXPORT_FILE="export.json"
SKIP_EXTRACTION=false
SPECIFIC_ANALYSIS=""

# Print header
echo "============================================================"
echo "              Discord Game Economy Analysis Tool             "
echo "============================================================"

# Parse command line arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    --file)
      EXPORT_FILE="$2"
      shift 2
      ;;
    --skip-extraction)
      SKIP_EXTRACTION=true
      shift
      ;;
    --economy-only)
      SPECIFIC_ANALYSIS="--economy-only"
      shift
      ;;
    --gambling-only)
      SPECIFIC_ANALYSIS="--gambling-only"
      shift
      ;;
    --category-only)
      SPECIFIC_ANALYSIS="--category-only"
      shift
      ;;
    --advanced-only)
      SPECIFIC_ANALYSIS="--advanced-only"
      shift
      ;;
    --help)
      echo "Usage: ./run_analysis.sh [options]"
      echo ""
      echo "Options:"
      echo "  --file <path>         Path to Discord export JSON file (default: export.json)"
      echo "  --skip-extraction     Skip data extraction phase (use existing data)"
      echo "  --economy-only        Run only economy analysis"
      echo "  --gambling-only       Run only gambling analysis"
      echo "  --category-only       Run only category analysis"
      echo "  --advanced-only       Run only advanced analysis"
      echo "  --help                Show this help message"
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      echo "Use --help for usage information"
      exit 1
      ;;
  esac
done

# Build command
CMD="python run_analysis.py"

if [ "$SKIP_EXTRACTION" = true ]; then
  CMD="$CMD --skip-extraction"
else
  CMD="$CMD $EXPORT_FILE"
fi

if [ ! -z "$SPECIFIC_ANALYSIS" ]; then
  CMD="$CMD $SPECIFIC_ANALYSIS"
fi

# Run the analysis
echo "Running: $CMD"
echo "============================================================"
eval $CMD 