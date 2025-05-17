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
from flask_login import LoginManager, login_user, logout_user, login_required, current_user
from werkzeug.security import generate_password_hash, check_password_hash

# Set up logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize database
class Base(DeclarativeBase):
    pass

db = SQLAlchemy(model_class=Base)

# Initialize login manager
login_manager = LoginManager()

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
    
    # Initialize login manager with the app
    login_manager.init_app(app)
    login_manager.login_view = 'login'
    login_manager.login_message = 'Accedi per accedere a questa pagina'
    login_manager.login_message_category = 'warning'
    
    # Create all database tables
    with app.app_context():
        import models  # Import models to register them with SQLAlchemy
        db.create_all()
    
    # User loader function for Flask-Login
    @login_manager.user_loader
    def load_user(user_id):
        # Import here to avoid circular imports
        from models import User
        return User.query.get(int(user_id))
    
    # Define routes
    @app.route('/')
    def index():
        """Home page"""
        return render_template('index.html')
        
    @app.route('/login', methods=['GET', 'POST'])
    def login():
        """Login page"""
        # If user is already logged in, redirect to index
        if current_user.is_authenticated:
            return redirect(url_for('index'))
            
        if request.method == 'POST':
            email = request.form['email']
            password = request.form['password']
            remember = 'remember' in request.form
            
            # Import here to avoid circular imports
            from models import User
            
            # Find user by email
            user = User.query.filter_by(email=email).first()
            
            # Check if user exists and password is correct
            if user and check_password_hash(user.password_hash, password):
                login_user(user, remember=remember)
                flash('Login effettuato con successo!', 'success')
                
                # Redirect to next page or index
                next_page = request.args.get('next')
                return redirect(next_page or url_for('index'))
            else:
                flash('Login fallito. Controlla email e password.', 'danger')
                
        return render_template('login.html')
        
    @app.route('/logout')
    @login_required
    def logout():
        """Logout user"""
        logout_user()
        flash('Logout effettuato con successo!', 'success')
        return redirect(url_for('index'))
        
    @app.route('/register', methods=['GET', 'POST'])
    def register():
        """Register page"""
        # If user is already logged in, redirect to index
        if current_user.is_authenticated:
            return redirect(url_for('index'))
            
        if request.method == 'POST':
            username = request.form['username']
            email = request.form['email']
            password = request.form['password']
            confirm_password = request.form['confirm_password']
            
            # Import here to avoid circular imports
            from models import User
            
            # Check if passwords match
            if password != confirm_password:
                flash('Le password non corrispondono.', 'danger')
                return redirect(url_for('register'))
                
            # Check if username or email already exists
            if User.query.filter_by(username=username).first():
                flash('Username già in uso. Scegli un altro username.', 'danger')
                return redirect(url_for('register'))
                
            if User.query.filter_by(email=email).first():
                flash('Email già registrata. Utilizza un\'altra email.', 'danger')
                return redirect(url_for('register'))
                
            # Create new user
            new_user = User(
                username=username,
                email=email,
                password_hash=generate_password_hash(password)
            )
            
            # Add user to database
            db.session.add(new_user)
            db.session.commit()
            
            flash('Registrazione completata con successo! Ora puoi accedere.', 'success')
            return redirect(url_for('login'))
            
        return render_template('register.html')
        
    @app.route('/profile', methods=['GET', 'POST'])
    @login_required
    def profile():
        """User profile page"""
        # Import models
        from models import Dataset, Backtest, MLModel
        
        if request.method == 'POST':
            # Update user information
            username = request.form['username']
            email = request.form['email']
            
            # Check if username or email is already taken (by another user)
            if username != current_user.username and User.query.filter_by(username=username).first():
                flash('Username già in uso. Scegli un altro username.', 'danger')
                return redirect(url_for('profile'))
                
            if email != current_user.email and User.query.filter_by(email=email).first():
                flash('Email già registrata. Utilizza un\'altra email.', 'danger')
                return redirect(url_for('profile'))
                
            # Update user
            current_user.username = username
            current_user.email = email
            
            # Save changes
            db.session.commit()
            
            flash('Profilo aggiornato con successo!', 'success')
            return redirect(url_for('profile'))
        
        # Get user's datasets, backtests, and models
        datasets = Dataset.query.filter_by(user_id=current_user.id).order_by(Dataset.created_at.desc()).all()
        backtests = Backtest.query.filter_by(user_id=current_user.id).order_by(Backtest.created_at.desc()).all()
        models = MLModel.query.filter_by(user_id=current_user.id).order_by(MLModel.created_at.desc()).all()
        
        return render_template('profile.html', datasets=datasets, backtests=backtests, models=models)
        
    @app.route('/change_password', methods=['POST'])
    @login_required
    def change_password():
        """Change user password"""
        current_password = request.form['current_password']
        new_password = request.form['new_password']
        confirm_password = request.form['confirm_password']
        
        # Check if current password is correct
        if not check_password_hash(current_user.password_hash, current_password):
            flash('Password attuale non corretta.', 'danger')
            return redirect(url_for('profile'))
            
        # Check if new passwords match
        if new_password != confirm_password:
            flash('Le nuove password non corrispondono.', 'danger')
            return redirect(url_for('profile'))
            
        # Update password
        current_user.password_hash = generate_password_hash(new_password)
        
        # Save changes
        db.session.commit()
        
        flash('Password aggiornata con successo!', 'success')
        return redirect(url_for('profile'))
    
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