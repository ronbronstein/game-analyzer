#!/usr/bin/env python3
"""
Complete Discord Economy Analysis Pipeline
Runs the complete analysis pipeline with advanced features
"""

import os
import sys
import argparse
from datetime import datetime
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('gameanalytics.log')
    ]
)

logger = logging.getLogger(__name__)

def print_header(message):
    """Print a formatted header"""
    logger.info("\n" + "=" * 80)
    logger.info(f" {message} ".center(80, "*"))
    logger.info("=" * 80 + "\n")

def run_extraction(json_file):
    """Run data extraction phase"""
    print_header("Data Extraction Phase")
    
    try:
        # Import the run_analysis module
        from run_analysis import run_extraction as extract_data
        
        # Run extraction
        success = extract_data(json_file)
        
        if not success:
            logger.error("Extraction phase failed")
            return False
        
        logger.info("Extraction phase completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error during extraction: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def run_economy_analysis():
    """Run economy analysis phase"""
    print_header("Economy Analysis Phase")
    
    try:
        # Import the run_analysis module
        from run_analysis import run_economy_analysis as analyze_economy
        
        # Run analysis
        success = analyze_economy()
        
        if not success:
            logger.error("Economy analysis phase failed")
            return False
        
        logger.info("Economy analysis phase completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error during economy analysis: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def run_gambling_analysis():
    """Run gambling analysis phase"""
    print_header("Gambling Analysis Phase")
    
    try:
        # Import the run_analysis module
        from run_analysis import run_gambling_analysis as analyze_gambling
        
        # Run analysis
        success = analyze_gambling()
        
        if not success:
            logger.error("Gambling analysis phase failed")
            return False
        
        logger.info("Gambling analysis phase completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error during gambling analysis: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def run_category_analysis():
    """Run category analysis phase"""
    print_header("Category Analysis Phase")
    
    try:
        # Import the run_analysis module
        from run_analysis import run_category_analysis as analyze_category
        
        # Run analysis
        success = analyze_category()
        
        if not success:
            logger.error("Category analysis phase failed")
            return False
        
        logger.info("Category analysis phase completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error during category analysis: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def run_advanced_analysis():
    """Run advanced economy analysis phase"""
    print_header("Advanced Economy Analysis Phase")
    
    try:
        # Import the advanced analyzer
        from gameanalytics.analyzers.advanced_analyzer import run_advanced_analysis
        
        # Run analysis
        success = run_advanced_analysis()
        
        if not success:
            logger.error("Advanced analysis phase failed")
            return False
        
        logger.info("Advanced analysis phase completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error during advanced analysis: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def run_llm_summary(api_key=None):
    """Run LLM-based summary generation phase"""
    print_header("LLM Summary Generation Phase")
    
    try:
        # Import the LLM summary generator
        from gameanalytics.llm_summary_generator import generate_summary
        
        # Run summary generation
        success = generate_summary(api_key=api_key)
        
        if not success:
            logger.error("LLM summary generation phase failed")
            return False
        
        logger.info("LLM summary generation phase completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error during LLM summary generation: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return False

def run_complete_pipeline(json_file=None, skip_extraction=False, skip_llm=False, api_key=None):
    """Run the complete analysis pipeline"""
    start_time = datetime.now()
    
    print_header("Discord Economy Analysis Pipeline")
    
    # Track completed phases
    completed_phases = []
    
    # 1. Data Extraction
    if not skip_extraction and json_file:
        if run_extraction(json_file):
            completed_phases.append("extraction")
    elif skip_extraction:
        logger.info("Skipping extraction phase as requested")
    else:
        logger.error("Cannot run extraction - no JSON file provided")
    
    # 2. Economy Analysis
    if run_economy_analysis():
        completed_phases.append("economy_analysis")
    
    # 3. Gambling Analysis
    if run_gambling_analysis():
        completed_phases.append("gambling_analysis")
    
    # 4. Category Analysis
    if run_category_analysis():
        completed_phases.append("category_analysis")
    
    # 5. Advanced Economy Analysis
    if run_advanced_analysis():
        completed_phases.append("advanced_analysis")
    
    # 6. LLM Summary Generation (if not skipped)
    if not skip_llm:
        if run_llm_summary(api_key):
            completed_phases.append("llm_summary")
    else:
        logger.info("Skipping LLM summary generation as requested")
    
    # Calculate execution time
    end_time = datetime.now()
    execution_time = (end_time - start_time).total_seconds()
    
    # Print summary
    print_header("Analysis Complete")
    
    logger.info(f"\nCompleted phases: {', '.join(completed_phases)}")
    logger.info(f"Total execution time: {execution_time:.2f} seconds")
    
    # Output directories
    output_dirs = []
    for phase in completed_phases:
        if phase == "extraction":
            output_dirs.append("balance_data")
        elif phase == "economy_analysis":
            output_dirs.append("economy_analysis")
        elif phase == "gambling_analysis":
            output_dirs.append("gambling_analysis")
        elif phase == "category_analysis":
            output_dirs.append("category_analysis")
        elif phase == "advanced_analysis":
            output_dirs.append("advanced_analysis")
        elif phase == "llm_summary":
            output_dirs.append("summary_analysis")
    
    logger.info("\nResults available in:")
    for directory in output_dirs:
        logger.info(f"  - {directory}/")
    
    logger.info("\nThank you for using the Discord Economy Analysis Tool!")
    return True

def main():
    """Main function"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Complete Discord Economy Analysis Pipeline")
    parser.add_argument("json_file", nargs="?", help="Path to Discord export JSON file")
    parser.add_argument("--skip-extraction", action="store_true", help="Skip data extraction phase")
    parser.add_argument("--skip-llm", action="store_true", help="Skip LLM summary generation")
    parser.add_argument("--api-key", help="OpenAI API key for LLM summary generation")
    
    args = parser.parse_args()
    
    # If no JSON file is provided and extraction is not skipped, prompt for file
    json_file = args.json_file
    if not json_file and not args.skip_extraction:
        json_file = input("Enter the path to Discord export JSON file: ")
    
    # Check if API key is provided or in environment variable
    api_key = args.api_key
    if not api_key and not args.skip_llm:
        # Try to get from environment
        api_key = os.environ.get('OPENAI_API_KEY')
        
        if not api_key:
            logger.warning("No OpenAI API key provided. LLM summary will use local generation instead.")
    
    # Run the pipeline
    run_complete_pipeline(
        json_file=json_file, 
        skip_extraction=args.skip_extraction,
        skip_llm=args.skip_llm,
        api_key=api_key
    )

if __name__ == "__main__":
    main() 