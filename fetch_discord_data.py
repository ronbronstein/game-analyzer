#!/usr/bin/env python3
"""
Discord Data Fetcher
Fetches Discord channel data using Discord Chat Exporter
"""

import os
import sys
import argparse
import subprocess
import time
from datetime import datetime
import shutil
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import utility functions
try:
    from gameanalytics.utils import logger, ensure_directory
except ImportError:
    import logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger("fetch_discord_data")
    
    def ensure_directory(directory):
        if not os.path.exists(directory):
            os.makedirs(directory, exist_ok=True)
        return directory

def load_env_config():
    """Load configuration from .env file"""
    # Load .env file if it exists
    if os.path.exists('.env'):
        load_dotenv()
        logger.info("Loaded configuration from .env file")
    else:
        logger.warning(".env file not found, using default values or command line arguments")
    
    # Get required values from environment or provide defaults
    config = {
        'bot_token': os.getenv('DISCORD_BOT_TOKEN', None),
        'channel_id': os.getenv('DISCORD_CHANNEL_ID', '1344274217969123381'),
        'backup_exports': os.getenv('BACKUP_EXPORTS', 'true').lower() == 'true',
        'max_backup_files': int(os.getenv('MAX_BACKUP_FILES', '5'))
    }
    
    return config

def backup_existing_file(file_path, max_backups=5):
    """Create a backup of an existing file with timestamp"""
    if not os.path.exists(file_path):
        return None
    
    # Create backup filename with timestamp
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    backup_path = f"{file_path}.backup_{timestamp}"
    
    # Copy the file
    logger.info(f"Creating backup of {file_path} to {backup_path}")
    shutil.copy2(file_path, backup_path)
    
    # Clean up old backups if we exceed max_backups
    backup_files = [f for f in os.listdir(os.path.dirname(file_path)) 
                    if f.startswith(os.path.basename(file_path) + '.backup_')]
    backup_files.sort(reverse=True)  # Newest first
    
    # Remove excess backups
    if len(backup_files) > max_backups:
        for old_backup in backup_files[max_backups:]:
            old_path = os.path.join(os.path.dirname(file_path), old_backup)
            logger.info(f"Removing old backup: {old_backup}")
            os.remove(old_path)
    
    return backup_path

def fetch_discord_data(bot_token, channel_id, output_file="export.json", use_docker=True):
    """Fetch Discord channel data using Discord Chat Exporter"""
    if not bot_token:
        logger.error("Bot token is required. Set DISCORD_BOT_TOKEN in .env file or use --token argument.")
        return False
    
    logger.info(f"Fetching Discord data for channel ID: {channel_id}")
    
    try:
        if use_docker:
            # Use Docker container
            cmd = [
                "docker", "run", "--rm",
                "-v", f"{os.getcwd()}:/out",
                "tyrrrz/discordchatexporter", "export",
                "-t", bot_token,
                "-c", channel_id,
                "-f", "Json",
                "--bot",
                "-o", f"/out/{output_file}"
            ]
        else:
            # If not using Docker, assume DiscordChatExporter is installed locally
            cmd = [
                "DiscordChatExporter.Cli", "export",
                "--token", bot_token,
                "--channel", channel_id,
                "--format", "Json",
                "--bot",
                "--output", output_file
            ]
        
        logger.info(f"Running command: {' '.join(cmd).replace(bot_token, '***TOKEN***')}")
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        stdout, stderr = process.communicate()
        
        if process.returncode != 0:
            logger.error(f"Error fetching Discord data: {stderr.decode('utf-8')}")
            return False
        
        logger.info("Discord data fetched successfully!")
        logger.info(f"Output file: {output_file}")
        
        # Verify file exists and has content
        if not os.path.exists(output_file):
            logger.error(f"Output file {output_file} not found!")
            return False
        
        file_size = os.path.getsize(output_file)
        logger.info(f"Output file size: {file_size / (1024*1024):.2f} MB")
        
        return True
    except Exception as e:
        logger.error(f"Error fetching Discord data: {e}")
        return False

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Fetch Discord Channel Data")
    parser.add_argument("--token", help="Discord bot token")
    parser.add_argument("--channel", help="Discord channel ID")
    parser.add_argument("--output", default="export.json", help="Output file path")
    parser.add_argument("--no-docker", action="store_true", help="Don't use Docker (use local DiscordChatExporter)")
    parser.add_argument("--no-backup", action="store_true", help="Don't backup existing export file")
    
    args = parser.parse_args()
    
    # Load configuration
    config = load_env_config()
    
    # Override with command line arguments if provided
    bot_token = args.token or config['bot_token']
    channel_id = args.channel or config['channel_id']
    output_file = args.output
    
    # Backup existing file if it exists and backup is not disabled
    if os.path.exists(output_file) and config['backup_exports'] and not args.no_backup:
        backup_existing_file(output_file, config['max_backup_files'])
    
    # Fetch Discord data
    success = fetch_discord_data(
        bot_token, 
        channel_id, 
        output_file, 
        use_docker=not args.no_docker
    )
    
    return 0 if success else 1

if __name__ == "__main__":
    sys.exit(main()) 