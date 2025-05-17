"""
CryptoTradeAnalyzer - Web Application

This is a simplified entry point for the CryptoTradeAnalyzer web application.
It provides a Flask-based interface for cryptocurrency trading analysis.

Author: CryptoTradeAnalyzer Team
Version: 1.0
"""

import os
import logging
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy.orm import DeclarativeBase

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize database
class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

def create_app():
    """
    Create and configure the Flask application
    
    Returns:
    --------
    Flask
        Configured Flask application
    """
    # Create Flask app
    app = Flask(__name__, 
                template_folder='web/templates',
                static_folder='web/static')
    
    # Configure app
    app.config.update(
        SECRET_KEY=os.environ.get("SESSION_SECRET", "crypto_trade_analyzer_secret"),
        MAX_CONTENT_LENGTH=50 * 1024 * 1024,  # 50MB max upload size
        UPLOAD_FOLDER=os.path.join(os.getcwd(), 'uploads'),
        PLOTS_FOLDER=os.path.join(os.getcwd(), 'web', 'static', 'plots'),
        TEMPLATES_AUTO_RELOAD=True,
        DEBUG=True,
        # Database configuration
        SQLALCHEMY_DATABASE_URI=os.environ.get("DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/crypto_analyzer"),
        SQLALCHEMY_TRACK_MODIFICATIONS=False,
        SQLALCHEMY_ENGINE_OPTIONS={
            "pool_recycle": 300,
            "pool_pre_ping": True,
        },
    )
    
    # Ensure upload and plots folders exist
    os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
    os.makedirs(app.config['PLOTS_FOLDER'], exist_ok=True)
    
    # Initialize the database with the app
    db.init_app(app)
    
    # Create all database tables
    with app.app_context():
        import models  # Import models to register them with SQLAlchemy
        db.create_all()
    
    # Define routes
    @app.route('/')
    def index():
        """Home page"""
        return render_template('index.html')
    
    @app.route('/upload')
    def upload():
        """Data upload page"""
        return render_template('upload.html')
    
    @app.route('/analysis')
    def analysis():
        """Data analysis page"""
        return render_template('analysis.html', 
                             summary={
                                 'filename': 'Sample data',
                                 'rows': 1000,
                                 'columns': ['timestamp', 'open', 'high', 'low', 'close', 'volume'],
                                 'start': '2023-01-01',
                                 'end': '2023-12-31',
                             },
                             stats={
                                 'min': 15000,
                                 'max': 68000,
                                 'mean': 32000,
                                 'std': 8000,
                                 'median': 30000,
                                 'last': 42000,
                                 'change': 180.0,
                             },
                             chart_data=None)
    
    @app.route('/indicators')
    def indicators():
        """Technical indicators page"""
        indicator_options = [
            {'id': 'sma', 'name': 'Simple Moving Average (SMA)'},
            {'id': 'ema', 'name': 'Exponential Moving Average (EMA)'},
            {'id': 'rsi', 'name': 'Relative Strength Index (RSI)'},
            {'id': 'macd', 'name': 'Moving Average Convergence Divergence (MACD)'},
            {'id': 'bbands', 'name': 'Bollinger Bands'},
            {'id': 'stoch', 'name': 'Stochastic Oscillator'},
        ]
        return render_template('indicators.html', indicator_options=indicator_options)

    @app.route('/backtest')
    def backtest():
        """Backtesting page"""
        strategy_options = [
            {'id': 'crossover', 'name': 'Moving Average Crossover'},
            {'id': 'macd_rsi', 'name': 'MACD + RSI Strategy'},
            {'id': 'support_resistance', 'name': 'Support & Resistance Levels'},
            {'id': 'bollinger_bands', 'name': 'Bollinger Bands Strategy'},
            {'id': 'ichimoku', 'name': 'Ichimoku Cloud Strategy'},
            {'id': 'ml_enhanced', 'name': 'ML-Enhanced Strategy'},
        ]
        return render_template('backtest.html', strategy_options=strategy_options)
    
    @app.route('/models')
    def models():
        """ML Models page"""
        model_options = [
            {'id': 'lstm', 'name': 'LSTM (Long Short-Term Memory)'},
            {'id': 'rnn', 'name': 'RNN (Recurrent Neural Network / GRU)'},
            {'id': 'ensemble', 'name': 'Ensemble Model'},
        ]
        target_options = ['close', 'high', 'low', 'open']
        return render_template('models.html', 
                             model_options=model_options,
                             target_options=target_options,
                             has_indicators=False)
                             
    @app.route('/clear_data')
    def clear_data():
        """Clear session data and return to home page"""
        # Clear session data
        session.clear()
        flash('All data has been cleared', 'success')
        return redirect(url_for('index'))

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

# Create and run the app
app = create_app()

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)