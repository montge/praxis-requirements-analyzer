# src/logger/logger.py

"""
Logging Configuration and Utilities Module

This module provides a customized logging setup with colored output and exception handling.
It includes:
- ColoredFormatter for adding ANSI colors to log output
- Setup function for creating configured loggers
- Exception handling decorators and utilities
"""

import os
import logging
import time
from functools import wraps
from typing import Optional, Callable, Any
from datetime import datetime
import asyncio

# ANSI color codes for terminal output
COLORS = {
    'DEBUG': '\033[36m',     # Cyan
    'INFO': '\033[32m',      # Green
    'WARNING': '\033[33m',   # Yellow
    'ERROR': '\033[31m',     # Red
    'CRITICAL': '\033[41m',  # Red background
    'RESET': '\033[0m'       # Reset color
}

class ColoredFormatter(logging.Formatter):
    """
    Custom formatter that adds colors to log levels and module names.
    
    This formatter adds ANSI color codes to make log output more readable:
    - Level names are colored based on severity
    - Module names are colored in purple
    - Messages maintain their original format
    """
    
    def format(self, record: logging.LogRecord) -> str:
        """
        Format the log record with colors.
        
        Args:
            record: The log record to format
            
        Returns:
            str: The formatted log message with color codes
        """
        # Save original levelname
        orig_levelname = record.levelname
        
        # Add color to levelname
        record.levelname = f"{COLORS.get(record.levelname, '')}{record.levelname}{COLORS['RESET']}"
        
        # Add color to module/logger name
        record.name = f"\033[35m{record.name}{COLORS['RESET']}"  # Purple
        
        # Format the message
        result = super().format(record)
        
        # Restore original levelname
        record.levelname = orig_levelname
        return result

def setup_logger(logger_name: str, default_level: int = logging.INFO) -> logging.Logger:
    """
    Set up a logger with colored output and environment-based configuration.
    
    Args:
        logger_name: Name of the logger to create
        default_level: Default logging level if not specified in environment
        
    Returns:
        logging.Logger: Configured logger instance
        
    Environment Variables:
        LOGGING_LEVEL: Override the default logging level
    """
    # Get logging level from environment variable
    log_level = os.getenv('LOGGING_LEVEL', 'INFO').upper()
    level = getattr(logging, log_level, default_level)
    
    # Create logger
    logger = logging.getLogger(logger_name)
    logger.setLevel(level)
    
    # Create console handler if none exists
    if not logger.handlers:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        
        # Create colored formatter
        formatter = ColoredFormatter(
            '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S'
        )
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        
        logger.debug(f"Logger {logger_name} initialized with level {log_level}")
    
    return logger

def handle_exception(message: str = None) -> Callable:
    """
    Decorator to handle and log exceptions with colored output.
    
    This decorator wraps both async and sync functions to provide consistent
    exception handling and logging.
    
    Args:
        message: Optional custom error message
        
    Returns:
        Callable: Decorated function that includes exception handling
        
    Example:
        @handle_exception("Failed to process data")
        async def process_data():
            # Function implementation
    """
    def decorator(func: Callable) -> Callable:
        # Check if the function is async
        is_async = asyncio.iscoroutinefunction(func)
        
        @wraps(func)
        async def async_wrapper(*args, **kwargs) -> Any:
            logger = logging.getLogger(func.__module__)
            try:
                return await func(*args, **kwargs)
            except Exception as e:
                error_msg = f"{COLORS['ERROR']}Error in {func.__name__}: {str(e)}{COLORS['RESET']}"
                if message:
                    error_msg = f"{COLORS['ERROR']}{message}: {str(e)}{COLORS['RESET']}"
                logger.error(error_msg)
                logger.debug(f"Function arguments: args={args}, kwargs={kwargs}")
                logger.exception(f"{COLORS['ERROR']}Detailed error trace:{COLORS['RESET']}")
                raise
        
        @wraps(func)
        def sync_wrapper(*args, **kwargs) -> Any:
            logger = logging.getLogger(func.__module__)
            try:
                return func(*args, **kwargs)
            except Exception as e:
                error_msg = f"{COLORS['ERROR']}Error in {func.__name__}: {str(e)}{COLORS['RESET']}"
                if message:
                    error_msg = f"{COLORS['ERROR']}{message}: {str(e)}{COLORS['RESET']}"
                logger.error(error_msg)
                logger.debug(f"Function arguments: args={args}, kwargs={kwargs}")
                logger.exception(f"{COLORS['ERROR']}Detailed error trace:{COLORS['RESET']}")
                raise
        
        return async_wrapper if is_async else sync_wrapper
    return decorator

def log_error(logger: logging.Logger, message: str, exception: Exception) -> None:
    """
    Log an error with colored output and detailed information.
    
    Args:
        logger: Logger instance to use
        message: Error message to log
        exception: Exception that was caught
        
    Example:
        try:
            # Some code
        except Exception as e:
            log_error(logger, "Operation failed", e)
    """
    error_msg = f"{COLORS['ERROR']}{message}: {str(exception)}{COLORS['RESET']}"
    logger.error(error_msg)
    logger.debug(f"Exception details: {str(exception)}")
    logger.exception(f"{COLORS['ERROR']}Detailed error trace:{COLORS['RESET']}")