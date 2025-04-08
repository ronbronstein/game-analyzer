#!/usr/bin/env python3
"""
Error handling module for the Discord Game Analysis application.
Defines custom exceptions and error handling utilities.
"""

from functools import wraps
import traceback
import sys

from gameanalytics.utils import logger


class GameAnalyticsError(Exception):
    """Base exception class for all application-specific errors"""
    def __init__(self, message, original_exception=None):
        self.message = message
        self.original_exception = original_exception
        super().__init__(self.message)


class DataExtractionError(GameAnalyticsError):
    """Exception raised for errors during data extraction"""
    pass


class DatabaseError(GameAnalyticsError):
    """Exception raised for errors related to database operations"""
    pass


class AnalysisError(GameAnalyticsError):
    """Exception raised for errors during analysis"""
    pass


class ConfigurationError(GameAnalyticsError):
    """Exception raised for configuration errors"""
    pass


class APIError(Exception):
    """API Error Exception class.
    
    A base exception class for all API-related errors. This allows us to return
    JSON error responses with appropriate HTTP status codes.
    """
    
    def __init__(self, message, status_code=400, payload=None):
        super().__init__(self)
        self.message = message
        self.status_code = status_code
        self.payload = payload

    def to_dict(self):
        error_dict = dict(self.payload or ())
        error_dict['message'] = self.message
        error_dict['status'] = 'error'
        return error_dict


def handle_error(func):
    """Decorator to standardize error handling
    
    This decorator catches exceptions, logs them, and returns appropriate 
    response values (False for functions that return boolean, None for others)
    
    Args:
        func: The function to decorate
        
    Returns:
        The decorated function
    """
    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            return func(*args, **kwargs)
        except GameAnalyticsError as e:
            # Log application-specific errors with custom message
            logger.error(f"{type(e).__name__}: {e.message}")
            if e.original_exception:
                logger.debug(f"Original exception: {str(e.original_exception)}")
            return False if func.__name__.startswith(('run_', 'is_', 'has_', 'can_')) else None
        except Exception as e:
            # Log unexpected errors with traceback
            logger.error(f"Unexpected error in {func.__name__}: {str(e)}")
            logger.error(traceback.format_exc())
            return False if func.__name__.startswith(('run_', 'is_', 'has_', 'can_')) else None
    return wrapper


def safe_execute(func, error_message, *args, **kwargs):
    """Execute a function safely with standardized error handling
    
    Args:
        func: Function to execute
        error_message: Error message to log if the function fails
        *args: Positional arguments to pass to the function
        **kwargs: Keyword arguments to pass to the function
        
    Returns:
        The return value of the function or None if an exception occurs
    """
    try:
        return func(*args, **kwargs)
    except Exception as e:
        logger.error(f"{error_message}: {str(e)}")
        logger.debug(traceback.format_exc())
        return None


def log_error(error, level='error'):
    """Log an error with appropriate level and format
    
    Args:
        error: The exception or error message to log
        level: Log level (error, warning, info, debug)
        
    Returns:
        None
    """
    error_str = str(error)
    
    if level == 'error':
        logger.error(error_str)
    elif level == 'warning':
        logger.warning(error_str)
    elif level == 'info':
        logger.info(error_str)
    elif level == 'debug':
        logger.debug(error_str)
    
    # If it's an exception, log the traceback at debug level
    if isinstance(error, Exception):
        logger.debug(traceback.format_exc())


class ErrorReporter:
    """Utility class for reporting and tracking errors"""
    
    def __init__(self):
        self.errors = []
    
    def report(self, error, context=None):
        """Report an error
        
        Args:
            error: Exception or error message
            context: Additional context information
            
        Returns:
            None
        """
        error_info = {
            'error': str(error),
            'type': type(error).__name__ if isinstance(error, Exception) else 'string',
            'context': context or {},
            'traceback': traceback.format_exc() if isinstance(error, Exception) else None
        }
        
        self.errors.append(error_info)
        
        # Log the error
        msg = f"{error_info['type']}: {error_info['error']}"
        if context:
            context_str = ', '.join(f"{k}={v}" for k, v in context.items())
            msg += f" [{context_str}]"
        
        logger.error(msg)
    
    def get_errors(self):
        """Get all reported errors
        
        Returns:
            list: List of error information dictionaries
        """
        return self.errors
    
    def has_errors(self):
        """Check if any errors have been reported
        
        Returns:
            bool: True if errors exist, False otherwise
        """
        return len(self.errors) > 0
    
    def clear(self):
        """Clear all reported errors
        
        Returns:
            None
        """
        self.errors = [] 