#!/usr/bin/env python3
"""
Database module for the Discord Game Analysis application.
Provides SQLite database functionality for caching and persistence.
"""

import os
import sqlite3
from datetime import datetime
import json
import pandas as pd
from contextlib import contextmanager

from gameanalytics.utils import logger, ensure_directory

# Database constants
DB_DIR = "database"
DB_FILE = os.path.join(DB_DIR, "game_analytics.db")


@contextmanager
def get_db_connection():
    """Context manager for database connections"""
    ensure_directory(DB_DIR)
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()


def get_connection():
    """Get a database connection
    
    Returns:
        SQLite connection object
    """
    ensure_directory(DB_DIR)
    conn = sqlite3.connect(DB_FILE)
    conn.row_factory = sqlite3.Row
    return conn


def init_tables(conn):
    """Initialize database tables using an existing connection
    
    Args:
        conn: SQLite connection object
    """
    cursor = conn.cursor()
    
    # Create messages table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS messages (
        message_id TEXT PRIMARY KEY,
        channel_id TEXT NOT NULL,
        timestamp DATETIME NOT NULL,
        author_id TEXT NOT NULL,
        content TEXT,
        message_type TEXT,
        raw_data TEXT
    )
    ''')
    
    # Create balance_updates table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS balance_updates (
        update_id INTEGER PRIMARY KEY AUTOINCREMENT,
        message_id TEXT REFERENCES messages(message_id),
        user_id TEXT NOT NULL,
        timestamp DATETIME NOT NULL,
        amount REAL NOT NULL,
        balance_after REAL NOT NULL,
        transaction_type TEXT NOT NULL,
        category TEXT,
        raw_data TEXT
    )
    ''')
    
    # Create user_stats table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS user_stats (
        user_id TEXT PRIMARY KEY,
        username TEXT,
        current_balance REAL,
        last_activity DATETIME,
        transactions_count INTEGER,
        total_gambling_wins REAL,
        total_gambling_losses REAL,
        last_updated DATETIME
    )
    ''')
    
    # Create analysis_results table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS analysis_results (
        analysis_id INTEGER PRIMARY KEY AUTOINCREMENT,
        analysis_type TEXT NOT NULL,
        timestamp DATETIME NOT NULL,
        parameters TEXT,
        result_data TEXT,
        file_path TEXT
    )
    ''')
    
    # Create extraction_status table
    cursor.execute('''
    CREATE TABLE IF NOT EXISTS extraction_status (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        last_processed_message_id TEXT,
        last_processed_timestamp DATETIME,
        total_messages INTEGER,
        total_balance_updates INTEGER,
        start_time DATETIME,
        end_time DATETIME,
        status TEXT
    )
    ''')
    
    # Create indexes for performance
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_timestamp ON messages(timestamp)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_messages_author ON messages(author_id)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_balance_user_timestamp ON balance_updates(user_id, timestamp)')
    cursor.execute('CREATE INDEX IF NOT EXISTS idx_balance_transaction ON balance_updates(transaction_type)')
    
    conn.commit()
    logger.info("Database tables initialized successfully")


def init_database():
    """Initialize the database with required tables"""
    conn = get_connection()
    try:
        init_tables(conn)
    finally:
        conn.close()


def store_messages(messages_data):
    """Store message data in the database
    
    Args:
        messages_data: List of dictionaries containing message data
            Each dict should have: message_id, channel_id, timestamp, author_id, content, message_type
    
    Returns:
        int: Number of messages inserted
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        count = 0
        
        for message in messages_data:
            raw_data = json.dumps(message.get('raw_data', {})) if 'raw_data' in message else None
            
            # Use INSERT OR REPLACE to handle duplicates
            cursor.execute('''
            INSERT OR REPLACE INTO messages 
            (message_id, channel_id, timestamp, author_id, content, message_type, raw_data)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                message['message_id'],
                message['channel_id'],
                message['timestamp'],
                message['author_id'],
                message.get('content'),
                message.get('message_type'),
                raw_data
            ))
            count += 1
        
        conn.commit()
        return count


def store_balance_updates(balance_updates):
    """Store balance update data in the database
    
    Args:
        balance_updates: List of dictionaries containing balance update data
            Each dict should have: message_id, user_id, timestamp, amount, balance_after, transaction_type, category
    
    Returns:
        int: Number of balance updates inserted
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        count = 0
        
        for update in balance_updates:
            raw_data = json.dumps(update.get('raw_data', {})) if 'raw_data' in update else None
            
            cursor.execute('''
            INSERT OR REPLACE INTO balance_updates
            (message_id, user_id, timestamp, amount, balance_after, transaction_type, category, raw_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                update.get('message_id'),
                update['user_id'],
                update['timestamp'],
                update['amount'],
                update['balance_after'],
                update['transaction_type'],
                update.get('category'),
                raw_data
            ))
            count += 1
        
        conn.commit()
        return count


def update_user_stats(user_stats):
    """Update user statistics in the database
    
    Args:
        user_stats: List of dictionaries containing user statistics
            Each dict should have: user_id, username, current_balance, etc.
    
    Returns:
        int: Number of user stats updated
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        count = 0
        
        for user in user_stats:
            cursor.execute('''
            INSERT OR REPLACE INTO user_stats
            (user_id, username, current_balance, last_activity, transactions_count, 
             total_gambling_wins, total_gambling_losses, last_updated)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            ''', (
                user['user_id'],
                user.get('username'),
                user.get('current_balance', 0),
                user.get('last_activity'),
                user.get('transactions_count', 0),
                user.get('total_gambling_wins', 0),
                user.get('total_gambling_losses', 0),
                datetime.now().isoformat()
            ))
            count += 1
        
        conn.commit()
        return count


def get_balance_updates_df(start_date=None, end_date=None, user_id=None, transaction_type=None):
    """Get balance updates as a pandas DataFrame with optional filtering
    
    Args:
        start_date: Optional datetime to filter updates after this date
        end_date: Optional datetime to filter updates before this date
        user_id: Optional user ID to filter updates for a specific user
        transaction_type: Optional transaction type filter
        
    Returns:
        DataFrame: Balance updates data
    """
    query = "SELECT * FROM balance_updates"
    conditions = []
    params = []
    
    if start_date:
        conditions.append("timestamp >= ?")
        params.append(start_date.isoformat() if hasattr(start_date, 'isoformat') else start_date)
    
    if end_date:
        conditions.append("timestamp <= ?")
        params.append(end_date.isoformat() if hasattr(end_date, 'isoformat') else end_date)
    
    if user_id:
        conditions.append("user_id = ?")
        params.append(user_id)
    
    if transaction_type:
        conditions.append("transaction_type = ?")
        params.append(transaction_type)
    
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    
    with get_db_connection() as conn:
        return pd.read_sql_query(query, conn, params=params)


def get_extraction_status():
    """Get the latest extraction status
    
    Returns:
        dict: Latest extraction status or None if no extraction has been done
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM extraction_status ORDER BY id DESC LIMIT 1")
        row = cursor.fetchone()
        
        if row:
            return dict(row)
        return None


def update_extraction_status(status_data):
    """Update extraction status in the database
    
    Args:
        status_data: Dictionary containing extraction status information
    
    Returns:
        int: ID of the inserted status
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        cursor.execute('''
        INSERT INTO extraction_status
        (last_processed_message_id, last_processed_timestamp, total_messages, 
         total_balance_updates, start_time, end_time, status)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            status_data.get('last_processed_message_id'),
            status_data.get('last_processed_timestamp'),
            status_data.get('total_messages', 0),
            status_data.get('total_balance_updates', 0),
            status_data.get('start_time', datetime.now().isoformat()),
            status_data.get('end_time'),
            status_data.get('status', 'in_progress')
        ))
        
        conn.commit()
        return cursor.lastrowid


def store_analysis_result(analysis_type, parameters, result_data, file_path=None):
    """Store analysis results in the database
    
    Args:
        analysis_type: Type of analysis (economy, gambling, category, advanced)
        parameters: Dictionary of parameters used for the analysis
        result_data: Analysis results data
        file_path: Optional path to the results file
        
    Returns:
        int: ID of the inserted analysis result
    """
    with get_db_connection() as conn:
        cursor = conn.cursor()
        
        params_json = json.dumps(parameters) if parameters else None
        result_json = json.dumps(result_data) if result_data else None
        
        cursor.execute('''
        INSERT INTO analysis_results
        (analysis_type, timestamp, parameters, result_data, file_path)
        VALUES (?, ?, ?, ?, ?)
        ''', (
            analysis_type,
            datetime.now().isoformat(),
            params_json,
            result_json,
            file_path
        ))
        
        conn.commit()
        return cursor.lastrowid


def export_to_csv(output_dir='balance_data'):
    """Export database data to CSV files for backward compatibility
    
    Args:
        output_dir: Directory to save CSV files
        
    Returns:
        bool: Success status
    """
    try:
        ensure_directory(output_dir)
        
        with get_db_connection() as conn:
            # Export balance updates
            df_balance = pd.read_sql_query("SELECT * FROM balance_updates", conn)
            df_balance.to_csv(os.path.join(output_dir, 'balance_updates.csv'), index=False)
            
            # Export user stats
            df_users = pd.read_sql_query("SELECT * FROM user_stats", conn)
            df_users.to_csv(os.path.join(output_dir, 'user_stats.csv'), index=False)
            
            # Export messages (optional, may be large)
            # df_messages = pd.read_sql_query("SELECT * FROM messages", conn)
            # df_messages.to_csv(os.path.join(output_dir, 'messages.csv'), index=False)
            
            logger.info(f"Database data exported to CSV files in {output_dir}")
            return True
    except Exception as e:
        logger.error(f"Error exporting database to CSV: {e}")
        return False 