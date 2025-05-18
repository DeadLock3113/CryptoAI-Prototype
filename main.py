"""
CryptoTradeAnalyzer - Entry point for the application

This is the main entry point for the CryptoTradeAnalyzer system, offering
a web interface for cryptocurrency analysis.

Author: CryptoTradeAnalyzer Team
Version: 1.0
"""

import logging
import os
import sys

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Import the app from simple_app
from simple_app import app

# The application will be run by gunicorn in the workflow.
