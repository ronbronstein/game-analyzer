#!/usr/bin/env python3
"""
Discord Economy Analysis Pipeline
Modular analysis pipeline for Discord game economy data
"""

import os
import sys
import argparse
from datetime import datetime
import traceback
import importlib.util

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import utils
from gameanalytics.utils import logger, Timer, ensure_directory
from gameanalytics.config import DEFAULT_DIRS, EXTRACTION_CONFIG, REPORT_CONFIG

def check_dependencies():
    """Check if all required dependencies are installed"""
    required_packages = [
        ('pandas', 'Data processing'),
        ('matplotlib', 'Visualization'),
        ('seaborn', 'Enhanced visualization'),
        ('numpy', 'Numerical computing'),
        ('statsmodels', 'Statistical analysis')
    ]
    
    optional_packages = [
        ('webbrowser', 'Opening reports automatically'),
        ('plotly', 'Interactive visualizations')
    ]
    
    missing_required = []
    missing_optional = []
    
    logger.info("Checking required dependencies...")
    for package, description in required_packages:
        try:
            importlib.import_module(package)
            logger.info(f"✓ {package} - {description}")
        except ImportError:
            missing_required.append(package)
            logger.error(f"✗ {package} - {description}")
    
    logger.info("\nChecking optional dependencies...")
    for package, description in optional_packages:
        try:
            importlib.import_module(package)
            logger.info(f"✓ {package} - {description}")
        except ImportError:
            missing_optional.append(package)
            logger.warning(f"! {package} - {description}")
    
    if missing_required:
        logger.error("\nMissing required dependencies. Please install them with:")
        logger.error(f"    pip install {' '.join(missing_required)}")
        return False
    
    if missing_optional:
        logger.warning("\nMissing optional dependencies. For full functionality, install with:")
        logger.warning(f"    pip install {' '.join(missing_optional)}")
    
    return True

def validate_json_file(file_path):
    """Validate that the file exists and has a valid JSON structure"""
    if not os.path.exists(file_path):
        logger.error(f"File not found: {file_path}")
        return False
        
    if not file_path.lower().endswith('.json'):
        logger.warning(f"File does not have .json extension: {file_path}")
        # Continue anyway, as it might still be valid JSON
    
    try:
        import json
        with open(file_path, 'r', encoding='utf-8') as file:
            # Just try to parse the first chunk to verify it's valid JSON
            chunk = file.read(4096)
            json.loads(chunk)
        return True
    except json.JSONDecodeError as e:
        logger.error(f"Invalid JSON file: {e}")
        return False
    except Exception as e:
        logger.error(f"Error reading file: {e}")
        return False

def run_extraction(json_file, output_dir=DEFAULT_DIRS['extraction'], max_workers=None, debug=False):
    """Run data extraction phase"""
    with Timer("Data extraction phase"):
        try:
            # Import the extractor module
            from gameanalytics.extractors.balance_extractor import extract_and_process
            
            # Run extraction
            logger.info(f"Extracting data from {json_file} to {output_dir}...")
            df = extract_and_process(json_file, output_dir, max_workers)
            
            if df is None or isinstance(df, bool) and not df:
                logger.error("Extraction failed")
                return False
            
            logger.info(f"Extraction completed successfully. Processed data saved to {output_dir}/")
            return True
        except Exception as e:
            logger.error(f"Error during extraction: {e}")
            if debug:
                logger.error(traceback.format_exc())
            return False

def run_economy_analysis(data_dir=DEFAULT_DIRS['extraction'], output_dir=DEFAULT_DIRS['economy']):
    """Run economy analysis phase"""
    with Timer("Economy analysis phase"):
        try:
            # Import the analysis module
            from gameanalytics.analyzers.economy.economy_analyzer import run_economy_analysis
            
            # Run analysis
            logger.info(f"Analyzing economy data from {data_dir}...")
            success = run_economy_analysis(data_dir, output_dir)
            
            if not success:
                logger.error("Economy analysis failed")
                return False
            
            logger.info(f"Economy analysis completed successfully. Results saved to {output_dir}/")
            return True
        except Exception as e:
            logger.error(f"Error during economy analysis: {e}")
            logger.error(traceback.format_exc())
            return False

def run_gambling_analysis(data_file=f"{DEFAULT_DIRS['extraction']}/balance_updates.csv", output_dir=DEFAULT_DIRS['gambling']):
    """Run gambling analysis phase"""
    with Timer("Gambling analysis phase"):
        try:
            # Import the gambling analyzer
            from gameanalytics.analyzers.gambling_analyzer import run_gambling_analysis
            
            # Run analysis
            logger.info(f"Analyzing gambling data from {data_file}...")
            success = run_gambling_analysis(data_file, output_dir)
            
            if not success:
                logger.error("Gambling analysis failed")
                return False
            
            logger.info(f"Gambling analysis completed successfully. Results saved to {output_dir}/")
            return True
        except Exception as e:
            logger.error(f"Error during gambling analysis: {e}")
            logger.error(traceback.format_exc())
            return False

def run_category_analysis(data_file=f"{DEFAULT_DIRS['extraction']}/balance_updates.csv", output_dir=DEFAULT_DIRS['category']):
    """Run category analysis phase"""
    with Timer("Category analysis phase"):
        try:
            # Import the category analyzer
            from gameanalytics.analyzers.category.category_analyzer import run_category_analysis
            
            # Run analysis
            logger.info(f"Analyzing transaction categories from {data_file}...")
            success = run_category_analysis(data_file, output_dir)
            
            if not success:
                logger.error("Category analysis failed")
                return False
            
            logger.info(f"Category analysis completed successfully. Results saved to {output_dir}/")
            return True
        except Exception as e:
            logger.error(f"Error during category analysis: {e}")
            logger.error(traceback.format_exc())
            return False
        
def run_advanced_analysis(data_file=f"{DEFAULT_DIRS['extraction']}/balance_updates.csv", output_dir=DEFAULT_DIRS['advanced']):
    """Run advanced economy analysis"""
    with Timer("Advanced analysis phase"):
        try:
            # Import the advanced analyzer
            from gameanalytics.analyzers.advanced_analyzer import run_advanced_analysis
        
        # Run analysis
            logger.info(f"Running advanced economy analysis on {data_file}...")
            success = run_advanced_analysis(data_file, output_dir)
            
            if not success:
                logger.error("Advanced analysis failed")
                return False
            
            logger.info(f"Advanced analysis completed successfully. Results saved to {output_dir}/")
            return True
        except Exception as e:
            logger.error(f"Error during advanced analysis: {e}")
            logger.error(traceback.format_exc())
            return False

def open_reports(base_dir='.'):
    """Try to open the generated HTML reports in the browser"""
    if not REPORT_CONFIG.get('open_browser_automatically', True):
        return
        
    try:
        import webbrowser
        
        # List of possible report locations
        reports = [
            os.path.join(base_dir, DEFAULT_DIRS['economy'], 'economy_analysis_report.html'),
            os.path.join(base_dir, DEFAULT_DIRS['gambling'], 'gambling_analysis_report.html'),
            os.path.join(base_dir, DEFAULT_DIRS['category'], 'category_analysis_report.html'),
            os.path.join(base_dir, DEFAULT_DIRS['advanced'], 'advanced_economy_report.html')
        ]
        
        opened = 0
        for report in reports:
            if os.path.exists(report):
                webbrowser.open('file://' + os.path.abspath(report))
                logger.info(f"Opened report: {report}")
                opened += 1
                # Small delay to prevent browser from opening too many tabs at once
                import time
                time.sleep(1)
        
        if opened > 0:
            logger.info(f"Opened {opened} report(s) in your browser")
        else:
            logger.warning("No reports found to open")
    except Exception as e:
        logger.error(f"Error opening reports: {e}")

def main():
    """Main entry point for the analysis pipeline"""
    start_time = datetime.now()
    
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Discord Economy Game Analysis")
    parser.add_argument("json_file", nargs="?", help="Path to Discord export JSON file")
    parser.add_argument("--skip-extraction", action="store_true", help="Skip data extraction phase")
    parser.add_argument("--economy-only", action="store_true", help="Run only economy analysis")
    parser.add_argument("--gambling-only", action="store_true", help="Run only gambling analysis")
    parser.add_argument("--category-only", action="store_true", help="Run only category analysis")
    parser.add_argument("--advanced-only", action="store_true", help="Run only advanced economy analysis")
    parser.add_argument("--debug", action="store_true", help="Show debug information")
    parser.add_argument("--workers", type=int, help="Number of worker processes for extraction")
    
    args = parser.parse_args()
    
    # Print header
    logger.info("=" * 80)
    logger.info(" Discord Economy Analysis Pipeline ".center(80, "*"))
    logger.info("=" * 80)
    
    # Check dependencies
    if not check_dependencies():
        return False
    
    # Determine JSON file path
    json_file = args.json_file
    if not json_file and not args.skip_extraction:
        json_file = input("Enter the path to your Discord export JSON file: ")
    
    # Validate JSON file if we're not skipping extraction
    if not args.skip_extraction:
        if not validate_json_file(json_file):
            logger.error("JSON file validation failed. Cannot proceed with extraction.")
            if not args.skip_extraction:
                return False
    
    # Run the appropriate analysis phases
    completed_phases = []
    
    # If specific analysis flags are set, only run those
    run_specific = args.economy_only or args.gambling_only or args.category_only or args.advanced_only
    
    # Extraction phase
    if not args.skip_extraction and (not run_specific or args.economy_only or args.gambling_only or args.category_only or args.advanced_only):
        logger.info("\n" + "=" * 80)
        logger.info(" Data Extraction Phase ".center(80, "*"))
        logger.info("=" * 80 + "\n")
        
        if run_extraction(json_file, max_workers=args.workers, debug=args.debug):
            completed_phases.append("extraction")
    else:
        if not os.path.exists(os.path.join(DEFAULT_DIRS['extraction'], 'balance_updates.csv')):
            logger.warning("No existing extracted data found. You may need to run extraction first.")
            logger.info("Checking if we need to generate sample data for testing...")
            
            # If skipping extraction but no data exists, check if this is for testing
            try:
                import generate_data
                logger.warning("Using generated sample data for testing purposes only.")
                logger.warning("For real analysis, run without --skip-extraction and provide a Discord export.json")
                generate_data.main()
            except ImportError:
                logger.error("Cannot find extraction data and generate_data module not available.")
                logger.error("Please run without --skip-extraction to process real Discord data.")
                return False
    
    # Economy analysis phase
    if not run_specific or args.economy_only:
        logger.info("\n" + "=" * 80)
        logger.info(" Economy Analysis Phase ".center(80, "*"))
        logger.info("=" * 80 + "\n")
        
        if run_economy_analysis():
            completed_phases.append("economy_analysis")
    
    # Gambling analysis phase
    if not run_specific or args.gambling_only:
        logger.info("\n" + "=" * 80)
        logger.info(" Gambling Analysis Phase ".center(80, "*"))
        logger.info("=" * 80 + "\n")
        
        if run_gambling_analysis():
            completed_phases.append("gambling_analysis")
    
    # Category analysis phase
    if not run_specific or args.category_only:
        logger.info("\n" + "=" * 80)
        logger.info(" Category Analysis Phase ".center(80, "*"))
        logger.info("=" * 80 + "\n")
        
        if run_category_analysis():
            completed_phases.append("category_analysis")
    
    # Advanced analysis phase
    if not run_specific or args.advanced_only:
        logger.info("\n" + "=" * 80)
        logger.info(" Advanced Economy Analysis Phase ".center(80, "*"))
        logger.info("=" * 80 + "\n")
        
        if run_advanced_analysis():
            completed_phases.append("advanced_analysis")
    
    # Calculate execution time
    end_time = datetime.now()
    execution_time = (end_time - start_time).total_seconds()
    
    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info(" Analysis Complete ".center(80, "*"))
    logger.info("=" * 80)
    
    logger.info(f"\nCompleted phases: {', '.join(completed_phases)}")
    logger.info(f"Total execution time: {execution_time:.2f} seconds")
    
    # Output directories
    output_dirs = []
    for phase in completed_phases:
        if phase == "extraction":
            output_dirs.append(DEFAULT_DIRS['extraction'])
        elif phase == "economy_analysis":
            output_dirs.append(DEFAULT_DIRS['economy'])
        elif phase == "gambling_analysis":
            output_dirs.append(DEFAULT_DIRS['gambling'])
        elif phase == "category_analysis":
            output_dirs.append(DEFAULT_DIRS['category'])
        elif phase == "advanced_analysis":
            output_dirs.append(DEFAULT_DIRS['advanced'])
    
    logger.info("\nResults available in:")
    for directory in output_dirs:
        logger.info(f"  - {directory}")
    
    # Try to open reports
    if not args.debug:
        open_reports()
    
    logger.info("\nThank you for using the Discord Economy Analysis Tool!")
    return True

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.info("\nAnalysis interrupted by user.")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)