"""
Logger Module for CryptoTradeAnalyzer

This module sets up logging for the CryptoTradeAnalyzer,
providing consistent log format and output across the application.

Author: CryptoTradeAnalyzer Team
"""

import logging
import sys
import os
from datetime import datetime

def setup_logger(log_level=logging.DEBUG, log_file=None):
    """
    Set up logger with specified log level and optional file output
    
    Parameters:
    -----------
    log_level : int
        Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
    log_file : str
        Path to log file (if None, log to console only)
        
    Returns:
    --------
    logging.Logger
        Configured logger
    """
    # Create logs directory if it doesn't exist and log_file is specified
    if log_file:
        os.makedirs(os.path.dirname(log_file), exist_ok=True)
    
    # Configure the root logger
    logger = logging.getLogger()
    logger.setLevel(log_level)
    
    # Remove existing handlers to avoid duplicate logs
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Create console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # Create file handler if log_file specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    # Log initial message
    logger.info(f"Logger initialized at level {logging.getLevelName(log_level)}")
    if log_file:
        logger.info(f"Logging to file: {log_file}")
    
    return logger

def get_logger(name):
    """
    Get a named logger
    
    Parameters:
    -----------
    name : str
        Name for the logger
        
    Returns:
    --------
    logging.Logger
        Named logger
    """
    return logging.getLogger(name)

def setup_file_logger(name, log_dir='logs'):
    """
    Set up a logger with file output
    
    Parameters:
    -----------
    name : str
        Name for the logger and log file
    log_dir : str
        Directory for log files
        
    Returns:
    --------
    logging.Logger
        Configured logger
    """
    # Create logs directory
    os.makedirs(log_dir, exist_ok=True)
    
    # Create log file path with timestamp
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    log_file = os.path.join(log_dir, f'{name}_{timestamp}.log')
    
    # Set up logger
    return setup_logger(log_level=logging.DEBUG, log_file=log_file)
