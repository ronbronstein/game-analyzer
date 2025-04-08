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
import json
from datetime import datetime
import shutil
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# Import utility functions and database
try:
    from gameanalytics.utils import logger, ensure_directory
    from gameanalytics.database import store_messages, update_extraction_status, init_database
    from gameanalytics.errors import DataExtractionError, handle_error
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

@handle_error
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
    dir_path = os.path.dirname(file_path)
    # Use current directory if dirname returns empty string
    dir_path = dir_path if dir_path else '.'
    
    base_filename = os.path.basename(file_path)
    backup_files = [f for f in os.listdir(dir_path) 
                    if f.startswith(base_filename + '.backup_')]
    backup_files.sort(reverse=True)  # Newest first
    
    # Remove excess backups
    if len(backup_files) > max_backups:
        for old_backup in backup_files[max_backups:]:
            old_path = os.path.join(dir_path, old_backup)
            logger.info(f"Removing old backup: {old_backup}")
            os.remove(old_path)
    
    return backup_path

@handle_error
def fetch_discord_data(bot_token, channel_id, output_file="export.json", use_docker=True):
    """Fetch Discord channel data using DiscordChatExporter"""
    if not bot_token:
        logger.error("Bot token is required. Set DISCORD_BOT_TOKEN in .env file or use --token argument.")
        raise DataExtractionError("Bot token is required")
    
    logger.info(f"Fetching Discord data for channel ID: {channel_id}")
    
    try:
        # Initialize database if not already initialized
        init_database()
        
        # Use a temporary output file when fetching data to avoid locks
        temp_output = f"{output_file}.temp"
        
        # Update extraction status to in_progress
        status_id = update_extraction_status({
            'status': 'in_progress',
            'start_time': datetime.now().isoformat()
        })
        
        if use_docker:
            # Use Docker to run DiscordChatExporter
            absolute_path = os.path.abspath(os.path.dirname(output_file))
            filename = os.path.basename(temp_output)
            command = f"docker run --rm -v {absolute_path}:/out tyrrrz/discordchatexporter export -t {bot_token} -c {channel_id} -f Json -o /out/{filename}"
            logger.info(f"Fetching Discord data for channel ID: {channel_id}")
            logger.info(f"Running command: {command.replace(bot_token, '***TOKEN***')}")
            
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                shell=True,
                text=True
            )
            
            for line in process.stdout:
                logger.info(f"Task output: {line.strip()}")
                
            return_code = process.wait()
            
            if return_code != 0:
                logger.error(f"Task failed with return code {return_code}")
                update_extraction_status({
                    'id': status_id,
                    'status': 'failed',
                    'end_time': datetime.now().isoformat()
                })
                raise DataExtractionError(f"Discord data export failed with code {return_code}")
                
            # If successful, move temp file to final destination
            if os.path.exists(temp_output):
                # Make sure final destination is not being accessed
                try:
                    if os.path.exists(output_file):
                        os.remove(output_file)
                except:
                    logger.warning(f"Could not remove existing {output_file}, it may be in use")
                    update_extraction_status({
                        'id': status_id,
                        'status': 'failed',
                        'end_time': datetime.now().isoformat()
                    })
                    raise DataExtractionError("Could not update output file, it may be in use")
                    
                try:
                    os.rename(temp_output, output_file)
                except:
                    logger.warning(f"Could not rename {temp_output} to {output_file}")
                    update_extraction_status({
                        'id': status_id,
                        'status': 'failed',
                        'end_time': datetime.now().isoformat()
                    })
                    raise DataExtractionError("Could not finalize output file")
                
                # Process the data and store in database
                process_and_store_data(output_file, status_id)
                
                return True
        else:
            # For future implementation: use local DiscordChatExporter.CLI installation
            logger.error("Local DiscordChatExporter not implemented. Please use Docker.")
            update_extraction_status({
                'id': status_id,
                'status': 'failed',
                'end_time': datetime.now().isoformat(),
                'error': "Local DiscordChatExporter not implemented"
            })
            raise DataExtractionError("Local DiscordChatExporter not implemented")
    except Exception as e:
        logger.error(f"Error fetching Discord data: {str(e)}")
        update_extraction_status({
            'status': 'failed',
            'end_time': datetime.now().isoformat(),
            'error': str(e)
        })
        raise DataExtractionError(f"Error fetching Discord data: {str(e)}", original_exception=e)

@handle_error
def process_and_store_data(json_file, status_id):
    """Process JSON data and store in database"""
    logger.info(f"Processing and storing Discord data from {json_file}")
    
    try:
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        messages = []
        for msg in data.get('messages', []):
            message = {
                'message_id': msg.get('id'),
                'channel_id': data.get('channel', {}).get('id'),
                'timestamp': msg.get('timestamp'),
                'author_id': msg.get('author', {}).get('id'),
                'content': msg.get('content'),
                'message_type': 'message',
                'raw_data': msg
            }
            messages.append(message)
        
        # Store in database
        count = store_messages(messages)
        
        # Update extraction status
        update_extraction_status({
            'id': status_id,
            'status': 'completed',
            'end_time': datetime.now().isoformat(),
            'total_messages': count,
            'last_processed_message_id': messages[-1]['message_id'] if messages else None,
            'last_processed_timestamp': messages[-1]['timestamp'] if messages else None
        })
        
        logger.info(f"Successfully processed and stored {count} messages")
        
        # Also create the traditional CSV output for backward compatibility
        ensure_directory('balance_data')
        
        logger.info("Data processing and storage completed successfully")
        return True
    except Exception as e:
        logger.error(f"Error processing and storing data: {str(e)}")
        update_extraction_status({
            'id': status_id,
            'status': 'failed',
            'end_time': datetime.now().isoformat(),
            'error': f"Error processing data: {str(e)}"
        })
        raise DataExtractionError(f"Error processing Discord data: {str(e)}", original_exception=e)

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
    
    try:
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
    except Exception as e:
        logger.error(f"Error in main function: {str(e)}")
        return 1

if __name__ == "__main__":
    sys.exit(main()) 