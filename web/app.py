"""
Web Application Module for CryptoTradeAnalyzer

This module provides a Flask web interface for the CryptoTradeAnalyzer,
allowing users to upload data, run backtests, visualize results,
and train/evaluate ML models through a browser interface.

Author: CryptoTradeAnalyzer Team
"""

import os
import logging
from flask import Flask, render_template, request, redirect, url_for, flash, session
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from io import BytesIO
import base64

# Import routes
from web.routes import register_routes

# Set up logger
logger = logging.getLogger(__name__)

def create_app():
    """
    Create and configure the Flask application
    
    Returns:
    --------
    Flask
        Configured Flask application
    """
    # Create Flask app
    app = Flask(__name__)
    
    # Configure app
    app.config.update(
        SECRET_KEY=os.environ.get("SESSION_SECRET", "crypto_trade_analyzer_secret"),
        MAX_CONTENT_LENGTH=50 * 1024 * 1024,  # 50MB max upload size
        UPLOAD_FOLDER=os.path.join(os.getcwd(), 'uploads'),
        PLOTS_FOLDER=os.path.join(os.getcwd(), 'web', 'static', 'plots'),
        TEMPLATES_AUTO_RELOAD=True,
        DEBUG=True,
    )
    
    # Ensure upload and plots folders exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PLOTS_FOLDER'], exist_ok=True)
    
    # Register routes
    register_routes(app)
    
    # Register error handlers
    @app.errorhandler(404)
    def page_not_found(e):
        return render_template('error.html', 
                               error_code=404, 
                               error_message="Page not found"), 404
    
    @app.errorhandler(500)
    def server_error(e):
        return render_template('error.html', 
                               error_code=500, 
                               error_message="Internal server error"), 500
    
    # Register template context processors
    @app.context_processor
    def inject_globals():
        return {
            'app_name': 'CryptoTradeAnalyzer',
            'version': '1.0',
        }
    
    return app

def run_app():
    """
    Run the Flask application
    """
    app = create_app()
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == '__main__':
    run_app()
