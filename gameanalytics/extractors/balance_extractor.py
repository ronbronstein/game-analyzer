#!/usr/bin/env python3
"""
Discord Balance Data Extractor
Extracts balance update data from Discord export JSON efficiently
"""

import re
import json
import pandas as pd
from datetime import datetime
import sys
import os
import logging
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
from gameanalytics.utils import load_json_data, ensure_directory, Timer, print_progress, logger

class BalanceExtractor:
    """Extract balance updates from Discord JSON export efficiently"""
    
    def __init__(self, json_file, output_dir='balance_data', chunk_size=1000, max_workers=None):
        """Initialize the extractor
        
        Args:
            json_file (str): Path to Discord export JSON
            output_dir (str): Directory to save extracted data
            chunk_size (int): Number of messages to process in each chunk
            max_workers (int): Maximum number of worker processes (None = auto)
        """
        self.json_file = json_file
        self.output_dir = output_dir
        self.chunk_size = chunk_size
        self.max_workers = max_workers or max(1, multiprocessing.cpu_count() - 1)
        self.balance_updates = []
        
        # Regex patterns for extracting data
        self.patterns = {
            'user': [
                r'\*\*User:\*\*\s+\*\*@?([^*]+)\*\*',
                r'\*\*User:\*\*\s+@([^\n]+)',
                r'User:\s+@?([^\n]+)'
            ],
            'amount': [
                r'\*\*Amount:\*\*\s+Cash:\s+`([^`]+)`\s+\|\s+Bank:\s+`([^`]+)`',
                r'Amount:\s+Cash:\s+`([^`]+)`\s+\|\s+Bank:\s+`([^`]+)`'
            ],
            'reason': [
                r'\*\*Reason:\*\*\s+([^\n]+)',
                r'Reason:\s+([^\n]+)'
            ]
        }
    
    def _extract_with_patterns(self, text, pattern_list):
        """Try multiple regex patterns and return first match"""
        for pattern in pattern_list:
            match = re.search(pattern, text)
            if match:
                return match
        return None
    
    def _process_message_chunk(self, messages, chunk_id):
        """Process a chunk of messages and extract balance updates
        
        Used for parallel processing.
        """
        balance_updates = []
        bot_messages = 0
        
        for msg in messages:
            # Check for UnbelievaBoat messages
            author_name = msg.get('author', {}).get('name', '')
            
            if author_name != 'UnbelievaBoat':
                continue
                
            bot_messages += 1
            
            # Extract embeds
            embeds = msg.get('embeds', [])
            
            for embed in embeds:
                embed_author = embed.get('author', {}).get('name', '')
                
                if embed_author != 'Balance updated':
                    continue
                
                try:
                    # Get the description text
                    description = embed.get('description', '')
                    if not description:
                        continue
                    
                    # Extract data using pattern matching
                    user_match = self._extract_with_patterns(description, self.patterns['user'])
                    amount_match = self._extract_with_patterns(description, self.patterns['amount'])
                    reason_match = self._extract_with_patterns(description, self.patterns['reason'])
                    
                    # Skip if required data is missing
                    if not user_match or not amount_match:
                        continue
                    
                    # Process the extracted data
                    user = user_match.group(1).strip() if user_match else "Unknown"
                    cash_str = amount_match.group(1).strip() if amount_match else "0"
                    bank_str = amount_match.group(2).strip() if amount_match else "0"
                    reason = reason_match.group(1).strip() if reason_match else "Unknown"
                    
                    # Clean the amounts
                    cash_str = cash_str.replace(',', '')
                    bank_str = bank_str.replace(',', '')
                    
                    # Handle percentage values
                    if '%' in cash_str or '%' in bank_str:
                        continue
                    
                    # Parse amounts
                    try:
                        cash_amount = float(cash_str.replace('+', '').replace('-', ''))
                        bank_amount = float(bank_str.replace('+', '').replace('-', ''))
                    except ValueError:
                        continue
                    
                    # Determine transaction types
                    cash_transaction_type = "credit" if '+' in cash_str else "debit"
                    bank_transaction_type = "credit" if '+' in bank_str else "debit"
                    
                    # Create update record
                    balance_updates.append({
                        'timestamp': msg.get('timestamp', ''),
                        'user': user,
                        'cash_amount': cash_amount,
                        'cash_type': cash_transaction_type,
                        'bank_amount': bank_amount,
                        'bank_type': bank_transaction_type,
                        'reason': reason,
                        'color': embed.get('color', '')
                    })
                    
                except Exception as e:
                    # Log the error but continue processing
                    error_msg = f"Error in chunk {chunk_id}: {str(e)}"
                    logger.error(error_msg)
        
        return {
            'chunk_id': chunk_id,
            'updates': balance_updates,
            'bot_messages': bot_messages
        }
    
    def extract(self):
        """Extract all balance updates using parallel processing"""
        with Timer("Data extraction"):
            # Load the JSON data
            data = load_json_data(self.json_file)
            if not data:
                logger.error(f"Failed to load data from {self.json_file}")
                return False
            
            if 'messages' not in data or not data['messages']:
                logger.error("No messages found in the data")
                return False
            
            total_messages = len(data['messages'])
            logger.info(f"Processing {total_messages} messages from {self.json_file}")
            
            # Split messages into chunks for parallel processing
            message_chunks = []
            for i in range(0, total_messages, self.chunk_size):
                end = min(i + self.chunk_size, total_messages)
                message_chunks.append((data['messages'][i:end], i // self.chunk_size))
            
            # Process chunks in parallel
            balance_updates = []
            bot_messages_total = 0
            
            with ProcessPoolExecutor(max_workers=self.max_workers) as executor:
                futures = {executor.submit(self._process_message_chunk, chunk, chunk_id): chunk_id 
                          for chunk, chunk_id in message_chunks}
                
                # Process results as they complete
                for i, future in enumerate(as_completed(futures)):
                    result = future.result()
                    balance_updates.extend(result['updates'])
                    bot_messages_total += result['bot_messages']
                    
                    # Update progress
                    print_progress(i + 1, len(futures), "Processing message chunks")
            
            self.balance_updates = balance_updates
            
            logger.info(f"Extraction complete: Found {len(balance_updates)} balance updates from {bot_messages_total} bot messages")
            return len(balance_updates) > 0
    
    def _categorize_transaction(self, reason):
        """Categorize transactions based on reason text"""
        if not reason or not isinstance(reason, str):
            return "Other"
            
        reason = reason.lower()
        
        # Work and income
        if 'work' in reason:
            return 'Work'
        elif 'daily' in reason:
            return 'Daily Bonus'
        elif 'role income' in reason:
            return 'Role Income'
        elif 'income' in reason:
            return 'Passive Income'
        elif 'chat money' in reason:
            return 'Chat Reward'
        
        # Gambling activities
        elif 'rob' in reason:
            return 'Robbery'
        elif 'robbed' in reason:
            return 'Robbed'
        elif 'animal' in reason and 'race' in reason and 'bet' in reason:
            return 'Animal Race Bet'
        elif 'animal' in reason and 'race' in reason and 'won' in reason:
            return 'Animal Race Win'
        elif 'blackjack' in reason and 'bet' in reason:
            return 'Blackjack Bet'
        elif 'blackjack' in reason and 'ended' in reason:
            return 'Blackjack Win'
        elif 'roulette' in reason and 'bet' in reason:
            return 'Roulette Bet'
        elif 'roulette' in reason and 'won' in reason:
            return 'Roulette Win'
        elif any(term in reason for term in ['slot-machine', 'gamble', 'slot', 'coinflip', 'dice']):
            return 'Slots & Other Gambling'
        
        # Transfers and shopping
        elif 'give-money' in reason:
            return 'Player Transfer'
        elif any(term in reason for term in ['shop', 'buy', 'purchase', 'store']):
            return 'Shopping'
        
        # Crime
        elif 'crime' in reason:
            return 'Crime'
        elif 'slut' in reason:
            return 'Slut Command'
        elif 'beg' in reason:
            return 'Begging'
        elif 'refund' in reason:
            return 'Refund'
        
        # Admin commands
        elif 'reset' in reason or 'remove-money' in reason:
            return 'Admin Commands'
            
        return 'Other'
    
    def _extract_transfer_target(self, reason):
        """Extract the target user from transfer reasons"""
        if not isinstance(reason, str):
            return None
            
        # Look for patterns like "sent to @username" or "transfer to @username"
        match = re.search(r'(?:sent|transfer(?:red)?)\s+to\s+@?([^\s]+)', reason, re.IGNORECASE)
        if match:
            return match.group(1)
        return None
    
    def create_dataframe(self):
        """Convert balance updates to a pandas DataFrame with additional columns"""
        if not self.balance_updates:
            logger.error("No balance updates to convert to DataFrame")
            return pd.DataFrame()
        
        # Create DataFrame
        df = pd.DataFrame(self.balance_updates)
        
        # Convert timestamp to datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], format='ISO8601', errors='coerce')
        
        # Remove rows with invalid timestamps
        invalid_timestamps = df['timestamp'].isna().sum()
        if invalid_timestamps > 0:
            logger.warning(f"Removed {invalid_timestamps} rows with invalid timestamps")
            df = df.dropna(subset=['timestamp'])
        
        # Add derived columns
        df['date'] = df['timestamp'].dt.date
        df['time'] = df['timestamp'].dt.time
        df['hour'] = df['timestamp'].dt.hour
        df['day_of_week'] = df['timestamp'].dt.day_name()
        df['week'] = df['timestamp'].dt.isocalendar().week
        df['month'] = df['timestamp'].dt.month
        
        # Add signs based on transaction type
        df['cash_with_sign'] = df.apply(
            lambda row: -row['cash_amount'] if row['cash_type'] == 'debit' else row['cash_amount'], 
            axis=1
        )
        
        df['bank_with_sign'] = df.apply(
            lambda row: -row['bank_amount'] if row['bank_type'] == 'debit' else row['bank_amount'], 
            axis=1
        )
        
        # Add transaction category
        df['category'] = df['reason'].apply(self._categorize_transaction)
        
        # Add source and target for transfers
        df['is_transfer'] = df['category'] == 'Player Transfer'
        df['transfer_target'] = df.apply(
            lambda row: self._extract_transfer_target(row['reason']) if row['is_transfer'] else None, 
            axis=1
        )
        
        logger.info(f"Created DataFrame with {len(df)} rows and {len(df.columns)} columns")
        return df
    
    def save_data(self, df):
        """Save the processed data to CSV and generate summary files"""
        if df.empty:
            logger.error("No data to save")
            return False
        
        # Create output directory
        ensure_directory(self.output_dir)
        
        # Save full dataset
        csv_path = os.path.join(self.output_dir, 'balance_updates.csv')
        df.to_csv(csv_path, index=False)
        logger.info(f"Saved complete dataset to {csv_path}")
        
        # Save summary by user
        user_summary = df.groupby('user').agg({
            'timestamp': 'count',
            'cash_with_sign': 'sum',
            'bank_with_sign': 'sum',
            'category': lambda x: x.value_counts().index[0]  # Most common category
        }).rename(columns={'timestamp': 'transaction_count'})
        
        user_summary['total_balance'] = user_summary['cash_with_sign'] + user_summary['bank_with_sign']
        user_summary = user_summary.sort_values('transaction_count', ascending=False)
        
        user_summary_path = os.path.join(self.output_dir, 'user_summary.csv')
        user_summary.to_csv(user_summary_path)
        logger.info(f"Saved user summary to {user_summary_path}")
        
        # Save summary by category
        category_summary = df.groupby('category').agg({
            'timestamp': 'count',
            'cash_with_sign': 'sum',
            'bank_with_sign': 'sum',
            'user': 'nunique'
        }).rename(columns={'timestamp': 'transaction_count', 'user': 'unique_users'})
        
        category_summary['avg_cash_per_transaction'] = category_summary['cash_with_sign'] / category_summary['transaction_count']
        category_summary = category_summary.sort_values('transaction_count', ascending=False)
        
        category_path = os.path.join(self.output_dir, 'category_summary.csv')
        category_summary.to_csv(category_path)
        logger.info(f"Saved category summary to {category_path}")
        
        # Save daily activity
        daily_summary = df.groupby('date').agg({
            'timestamp': 'count',
            'cash_with_sign': 'sum',
            'user': 'nunique',
            'cash_amount': lambda x: abs(x).sum()
        }).rename(columns={
            'timestamp': 'transaction_count', 
            'user': 'active_users',
            'cash_amount': 'transaction_volume'
        })
        
        daily_summary['avg_per_user'] = daily_summary['cash_with_sign'] / daily_summary['active_users']
        daily_summary = daily_summary.sort_index()
        
        daily_path = os.path.join(self.output_dir, 'daily_summary.csv')
        daily_summary.to_csv(daily_path)
        logger.info(f"Saved daily summary to {daily_path}")
        
        # Generate stats summary
        total_transactions = len(df)
        unique_users = df['user'].nunique()
        date_range = f"{df['date'].min()} to {df['date'].max()}"
        total_cash_volume = abs(df['cash_with_sign']).sum()
        net_cash_change = df['cash_with_sign'].sum()
        
        # Save stats
        stats = {
            'total_transactions': total_transactions,
            'unique_users': unique_users,
            'date_range': date_range,
            'total_cash_volume': float(total_cash_volume),
            'net_cash_change': float(net_cash_change),
            'top_categories': category_summary['transaction_count'].head(10).to_dict(),
            'top_users': user_summary['transaction_count'].head(10).to_dict()
        }
        
        stats_path = os.path.join(self.output_dir, 'stats_summary.json')
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2, default=str)
        
        logger.info(f"Saved stats summary to {stats_path}")
        
        logger.info("\nBasic Statistics:")
        logger.info(f"Total transactions: {total_transactions:,}")
        logger.info(f"Unique users: {unique_users:,}")
        logger.info(f"Date range: {date_range}")
        logger.info(f"Total cash volume: {total_cash_volume:,.2f}")
        logger.info(f"Net cash change: {net_cash_change:,.2f}")
        
        return True

def extract_and_process(json_file_path, output_dir='balance_data', max_workers=None):
    """Extract and process data from Discord JSON export"""
    # Initialize extractor
    extractor = BalanceExtractor(json_file_path, output_dir, max_workers=max_workers)
    
    # Extract data
    if not extractor.extract():
        logger.error("Failed to extract balance updates")
        return False
    
    # Create DataFrame
    df = extractor.create_dataframe()
    
    if df.empty:
        logger.error("Failed to create DataFrame")
        return False
    
    # Save data
    if not extractor.save_data(df):
        logger.error("Failed to save data")
        return False
    
    logger.info(f"Extraction and processing complete. Data saved to {output_dir}/")
    return df

if __name__ == "__main__":
    if len(sys.argv) > 1:
        json_file_path = sys.argv[1]
    else:
        json_file_path = input("Enter the path to your Discord export JSON file: ")
    
    # Extract and process data
    extract_and_process(json_file_path) 