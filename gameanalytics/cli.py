#!/usr/bin/env python3
"""
Command-line interface for the Discord Game Economy Analysis Tool
"""

import os
import sys
import argparse
import importlib.util
from datetime import datetime
import subprocess


def main():
    """Main entry point for the CLI"""
    parser = argparse.ArgumentParser(
        description="Discord Game Economy Analysis Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  discord-analysis analyze export.json      Run full analysis on export.json
  discord-analysis web                      Start the web interface
  discord-analysis fetch --token=BOT_TOKEN  Fetch Discord data
  discord-analysis -h                       Show this help message
"""
    )
    
    # Create subparsers for different commands
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Analyze command
    analyze_parser = subparsers.add_parser("analyze", help="Analyze Discord game data")
    analyze_parser.add_argument("json_file", nargs="?", help="Path to Discord export JSON file")
    analyze_parser.add_argument("--skip-extraction", action="store_true", help="Skip data extraction phase")
    analyze_parser.add_argument("--economy-only", action="store_true", help="Run only economy analysis")
    analyze_parser.add_argument("--gambling-only", action="store_true", help="Run only gambling analysis")
    analyze_parser.add_argument("--category-only", action="store_true", help="Run only category analysis")
    analyze_parser.add_argument("--advanced-only", action="store_true", help="Run only advanced economy analysis")
    analyze_parser.add_argument("--debug", action="store_true", help="Show debug information")
    analyze_parser.add_argument("--workers", type=int, help="Number of worker processes for extraction")
    
    # Web interface command
    web_parser = subparsers.add_parser("web", help="Start the web interface")
    web_parser.add_argument("--port", type=int, default=5000, help="Port to run the web server on")
    web_parser.add_argument("--debug", action="store_true", help="Run web server in debug mode")
    
    # Fetch command
    fetch_parser = subparsers.add_parser("fetch", help="Fetch Discord data")
    fetch_parser.add_argument("--token", help="Discord bot token")
    fetch_parser.add_argument("--channel", help="Discord channel ID")
    fetch_parser.add_argument("--output", default="export.json", help="Output file path")
    fetch_parser.add_argument("--no-docker", action="store_true", help="Don't use Docker")
    fetch_parser.add_argument("--no-backup", action="store_true", help="Don't backup existing export file")
    
    # Parse arguments
    args = parser.parse_args()
    
    # If no command is provided, show help
    if not args.command:
        parser.print_help()
        return 1
    
    # Execute the appropriate command
    if args.command == "analyze":
        return run_analyze(args)
    elif args.command == "web":
        return run_web(args)
    elif args.command == "fetch":
        return run_fetch(args)
    
    return 0

def run_analyze(args):
    """Run analysis command"""
    # Build command arguments
    cmd = [sys.executable, "run_analysis.py"]
    
    if args.skip_extraction:
        cmd.append("--skip-extraction")
    elif args.json_file:
        cmd.append(args.json_file)
    
    if args.economy_only:
        cmd.append("--economy-only")
    if args.gambling_only:
        cmd.append("--gambling-only")
    if args.category_only:
        cmd.append("--category-only")
    if args.advanced_only:
        cmd.append("--advanced-only")
    if args.debug:
        cmd.append("--debug")
    if args.workers:
        cmd.extend(["--workers", str(args.workers)])
    
    # Run the analysis script
    try:
        return subprocess.call(cmd)
    except Exception as e:
        print(f"Error running analysis: {e}")
        return 1

def run_web(args):
    """Run web interface command"""
    # Import app module if available
    try:
        import app
        # Call the main function directly
        app.app.config['PORT'] = args.port
        app.app.config['DEBUG'] = args.debug
        app.main()
        return 0
    except ImportError:
        # If app.py is not in the path, try to run it as a subprocess
        try:
            cmd = [sys.executable, "app.py"]
            if args.port != 5000:
                os.environ['PORT'] = str(args.port)
            if args.debug:
                os.environ['DEBUG_MODE'] = 'true'
            return subprocess.call(cmd)
        except Exception as e:
            print(f"Error starting web interface: {e}")
            return 1

def run_fetch(args):
    """Run fetch command"""
    # Build command arguments
    cmd = [sys.executable, "fetch_discord_data.py"]
    
    if args.token:
        cmd.extend(["--token", args.token])
    if args.channel:
        cmd.extend(["--channel", args.channel])
    if args.output != "export.json":
        cmd.extend(["--output", args.output])
    if args.no_docker:
        cmd.append("--no-docker")
    if args.no_backup:
        cmd.append("--no-backup")
    
    # Run the fetch script
    try:
        return subprocess.call(cmd)
    except Exception as e:
        print(f"Error fetching Discord data: {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 