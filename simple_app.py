"""
CryptoTradeAnalyzer - Versione semplificata

Questa è una versione semplificata dell'applicazione CryptoTradeAnalyzer 
che utilizza SQLite e include tutte le funzionalità principali.
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
import uuid
from datetime import datetime
from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify
from flask_sqlalchemy import SQLAlchemy
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__, 
            template_folder='web/templates',
            static_folder='web/static')

# Configuration
app.config.update(
    SECRET_KEY="crypto_trade_analyzer_secret_key",
    MAX_CONTENT_LENGTH=50 * 1024 * 1024,  # 50MB max upload size
    UPLOAD_FOLDER=os.path.join(os.getcwd(), 'uploads'),
    TEMPLATES_AUTO_RELOAD=True,
    DEBUG=True,
    SQLALCHEMY_DATABASE_URI="sqlite:///crypto_analyzer.db",
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
)

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize database
db = SQLAlchemy(app)

# Models
class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    datasets = db.relationship('Dataset', backref='user', lazy='dynamic')
    
    def is_authenticated(self):
        return True
        
    def is_active(self):
        return True
        
    def is_anonymous(self):
        return False
        
    def get_id(self):
        return str(self.id)

class Dataset(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), nullable=False)
    symbol = db.Column(db.String(20), nullable=False)
    description = db.Column(db.Text)
    file_path = db.Column(db.String(256), nullable=False)
    rows_count = db.Column(db.Integer)
    start_date = db.Column(db.DateTime)
    end_date = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)

# Create database tables
with app.app_context():
    db.create_all()

# Helper functions
def get_current_user():
    """Get current user from session"""
    if 'user_id' in session:
        return User.query.get(session['user_id'])
    return None

def load_dataset(dataset_id):
    """Load dataset from file"""
    dataset = Dataset.query.get(dataset_id)
    if dataset and os.path.exists(dataset.file_path):
        return pd.read_csv(dataset.file_path, index_col=0, parse_dates=True)
    return None

def generate_price_chart(data, symbol):
    """Generate price chart for dataset"""
    plt.figure(figsize=(10, 6))
    
    # Plot closing price
    if 'close' in data.columns:
        plt.plot(data.index, data['close'], label='Prezzo di chiusura')
    
    # Plot volume as bar chart on secondary y-axis if present
    if 'volume' in data.columns:
        ax1 = plt.gca()
        ax2 = ax1.twinx()
        ax2.bar(data.index, data['volume'], alpha=0.3, color='gray', label='Volume')
        ax2.set_ylabel('Volume')
    
    # Add labels and legend
    plt.title(f"{symbol} - Andamento Prezzi")
    plt.xlabel('Data')
    plt.ylabel('Prezzo')
    plt.grid(True, alpha=0.3)
    plt.legend(loc='upper left')
    
    # Save plot to buffer
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    buffer.seek(0)
    
    # Convert plot to base64 for embedding in HTML
    chart_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
    plt.close()
    
    return chart_data

# Routes
@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page"""
    if 'user_id' in session:
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']
        remember = 'remember' in request.form
        
        # Find user by email
        user = User.query.filter_by(email=email).first()
        
        # Check if user exists and password is correct
        if user and check_password_hash(user.password_hash, password):
            session['user_id'] = user.id
            session['username'] = user.username
            flash('Login effettuato con successo!', 'success')
            
            # Redirect to next page or index
            next_page = request.args.get('next')
            return redirect(next_page or url_for('index'))
        else:
            flash('Login fallito. Controlla email e password.', 'danger')
            
    return render_template('login.html')

@app.route('/logout')
def logout():
    """Logout user"""
    session.pop('user_id', None)
    session.pop('username', None)
    flash('Logout effettuato con successo!', 'success')
    return redirect(url_for('index'))

@app.route('/register', methods=['GET', 'POST'])
def register():
    """Register page"""
    if 'user_id' in session:
        return redirect(url_for('index'))
        
    if request.method == 'POST':
        username = request.form['username']
        email = request.form['email']
        password = request.form['password']
        confirm_password = request.form['confirm_password']
        
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
def profile():
    """User profile page"""
    # Check if user is logged in
    user = get_current_user()
    if not user:
        flash('Devi effettuare il login per accedere a questa pagina.', 'warning')
        return redirect(url_for('login', next=request.url))
    
    if request.method == 'POST':
        # Update user information
        username = request.form['username']
        email = request.form['email']
        
        # Check if username or email is already taken (by another user)
        if username != user.username and User.query.filter_by(username=username).first():
            flash('Username già in uso. Scegli un altro username.', 'danger')
            return redirect(url_for('profile'))
            
        if email != user.email and User.query.filter_by(email=email).first():
            flash('Email già registrata. Utilizza un\'altra email.', 'danger')
            return redirect(url_for('profile'))
            
        # Update user
        user.username = username
        user.email = email
        
        # Save changes
        db.session.commit()
        
        # Update session
        session['username'] = username
        
        flash('Profilo aggiornato con successo!', 'success')
        return redirect(url_for('profile'))
    
    # Get user's datasets
    datasets = Dataset.query.filter_by(user_id=user.id).order_by(Dataset.created_at.desc()).all()
    
    return render_template('profile.html', datasets=datasets, backtests=[], models=[])

@app.route('/change_password', methods=['POST'])
def change_password():
    """Change user password"""
    # Check if user is logged in
    user = get_current_user()
    if not user:
        flash('Devi effettuare il login per accedere a questa pagina.', 'warning')
        return redirect(url_for('login', next=request.url))
    
    current_password = request.form['current_password']
    new_password = request.form['new_password']
    confirm_password = request.form['confirm_password']
    
    # Check if current password is correct
    if not check_password_hash(user.password_hash, current_password):
        flash('Password attuale non corretta.', 'danger')
        return redirect(url_for('profile'))
        
    # Check if new passwords match
    if new_password != confirm_password:
        flash('Le nuove password non corrispondono.', 'danger')
        return redirect(url_for('profile'))
        
    # Update password
    user.password_hash = generate_password_hash(new_password)
    
    # Save changes
    db.session.commit()
    
    flash('Password aggiornata con successo!', 'success')
    return redirect(url_for('profile'))

@app.route('/upload', methods=['GET', 'POST'])
def upload():
    """Data upload page"""
    # Check if user is logged in
    user = get_current_user()
    if not user:
        flash('Devi effettuare il login per accedere a questa pagina.', 'warning')
        return redirect(url_for('login', next=request.url))
    
    if request.method == 'POST':
        # Check if a file was uploaded
        if 'file' not in request.files:
            flash('Nessun file selezionato', 'danger')
            return redirect(request.url)
            
        file = request.files['file']
        
        # Check if file is empty
        if file.filename == '':
            flash('Nessun file selezionato', 'danger')
            return redirect(request.url)
            
        # Check file extension
        if not file.filename.endswith('.csv'):
            flash('Solo file CSV sono supportati', 'danger')
            return redirect(request.url)
            
        # Check if this is a delete request
        if 'delete_dataset' in request.form:
            dataset_id = request.form.get('delete_dataset')
            dataset = Dataset.query.filter_by(id=dataset_id, user_id=user.id).first()
            
            if dataset:
                try:
                    # Delete associated file if it exists
                    if dataset.file_path and os.path.exists(dataset.file_path):
                        os.remove(dataset.file_path)
                    
                    # Delete from database
                    db.session.delete(dataset)
                    db.session.commit()
                    
                    flash(f'Dataset "{dataset.name}" eliminato con successo.', 'success')
                except Exception as e:
                    logger.error(f"Error deleting dataset: {str(e)}")
                    db.session.rollback()
                    flash(f'Errore durante l\'eliminazione del dataset: {str(e)}', 'danger')
            else:
                flash('Dataset non trovato o non hai i permessi per eliminarlo.', 'warning')
            
            return redirect(url_for('upload'))
            
        # Get form data for file upload
        name = request.form['name']
        symbol = request.form['symbol']
        description = request.form.get('description', '')
        date_format = request.form.get('date_format', 'auto')
        delimiter = request.form.get('delimiter', ',')
        
        # Check for duplicate dataset name
        existing_dataset = Dataset.query.filter_by(name=name, user_id=user.id).first()
        if existing_dataset:
            flash(f'Esiste già un dataset con il nome "{name}". Scegli un nome diverso.', 'warning')
            return redirect(request.url)
        
        # Save the file with a unique name
        try:
            # Make sure upload folder exists
            os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
            
            # Generate unique filename
            filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            
            # Log the filepath for debugging
            logger.debug(f"Saving file to: {filepath}")
            
            # Save the file
            file.save(filepath)
            
            # Check if file was saved successfully
            if not os.path.exists(filepath):
                raise Exception("File could not be saved to disk")
        except Exception as e:
            logger.error(f"Error saving file: {str(e)}")
            flash(f'Errore nel salvataggio del file: {str(e)}', 'danger')
            return redirect(request.url)
        
        try:
            # Process the CSV file
            if delimiter == '\\t':
                delimiter = '\t'  # Replace escaped tab with actual tab
            
            # Load data with specified parameters
            if date_format == 'auto':
                # Try automatic parsing
                data = pd.read_csv(filepath, delimiter=delimiter)
                # Find timestamp/date column
                date_col = next((col for col in data.columns if 'time' in col.lower() or 'date' in col.lower()), None)
                if date_col:
                    data[date_col] = pd.to_datetime(data[date_col])
                    data.set_index(date_col, inplace=True)
            elif date_format == 'unix':
                # Assume unix timestamp
                data = pd.read_csv(filepath, delimiter=delimiter)
                date_col = next((col for col in data.columns if 'time' in col.lower() or 'date' in col.lower()), None)
                if date_col:
                    data[date_col] = pd.to_datetime(data[date_col], unit='s')
                    data.set_index(date_col, inplace=True)
            else:
                # Parse with specific format
                data = pd.read_csv(filepath, delimiter=delimiter)
                date_col = next((col for col in data.columns if 'time' in col.lower() or 'date' in col.lower()), None)
                if date_col:
                    data[date_col] = pd.to_datetime(data[date_col], format=date_format)
                    data.set_index(date_col, inplace=True)
            
            # Ensure required columns are present
            required_columns = ['open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in [c.lower() for c in data.columns]]
            
            if missing_columns:
                flash(f'Colonne mancanti nel file CSV: {", ".join(missing_columns)}', 'danger')
                return redirect(request.url)
            
            # Standardize column names (case insensitive)
            # Create a mapping from existing columns to standard names
            column_mapping = {}
            for col in data.columns:
                if col.lower() == 'open':
                    column_mapping[col] = 'open'
                elif col.lower() == 'high':
                    column_mapping[col] = 'high'
                elif col.lower() == 'low':
                    column_mapping[col] = 'low'
                elif col.lower() == 'close':
                    column_mapping[col] = 'close'
                elif col.lower() == 'volume':
                    column_mapping[col] = 'volume'
            
            # Rename columns
            data.rename(columns=column_mapping, inplace=True)
            
            # Sort data by index (timestamp)
            data.sort_index(inplace=True)
            
            # Save processed data
            processed_filepath = os.path.join(app.config['UPLOAD_FOLDER'], f"processed_{filename}")
            data.to_csv(processed_filepath)
            
            # Create database entry
            new_dataset = Dataset(
                name=name,
                symbol=symbol,
                description=description,
                file_path=processed_filepath,
                rows_count=len(data),
                start_date=data.index[0] if len(data) > 0 else None,
                end_date=data.index[-1] if len(data) > 0 else None,
                user_id=user.id
            )
            
            db.session.add(new_dataset)
            db.session.commit()
            
            flash(f'Dataset "{name}" caricato con successo! {len(data)} righe caricate.', 'success')
            return redirect(url_for('analysis', dataset_id=new_dataset.id))
            
        except Exception as e:
            # Ensure we remove the uploaded file if processing fails
            if os.path.exists(filepath):
                os.remove(filepath)
            
            logger.error(f"Errore durante l'elaborazione del file CSV: {str(e)}")
            flash(f'Errore durante l\'elaborazione del file: {str(e)}', 'danger')
            return redirect(request.url)
    
    return render_template('upload.html')

@app.route('/analysis')
@app.route('/analysis/<int:dataset_id>')
def analysis(dataset_id=None):
    """Data analysis page"""
    # Check if user is logged in
    user = get_current_user()
    if not user:
        flash('Devi effettuare il login per accedere a questa pagina.', 'warning')
        return redirect(url_for('login', next=request.url))
    
    # Get all user's datasets for the dropdown selector
    user_datasets = Dataset.query.filter_by(user_id=user.id).order_by(Dataset.created_at.desc()).all()
    
    if not user_datasets:
        flash('Non hai ancora caricato nessun dataset. Carica prima un file CSV.', 'info')
        return redirect(url_for('upload'))
    
    # If dataset_id is not provided, use the most recent one
    if dataset_id is None and user_datasets:
        dataset_id = user_datasets[0].id
    
    # Get the selected dataset
    selected_dataset = None
    if dataset_id:
        selected_dataset = Dataset.query.filter_by(id=dataset_id, user_id=user.id).first()
    
    if not selected_dataset:
        flash('Dataset non trovato.', 'danger')
        return redirect(url_for('upload'))
    
    try:
        # Load the data
        data = load_dataset(selected_dataset.id)
        
        if data is None:
            flash('Impossibile caricare il dataset. File non trovato.', 'danger')
            return redirect(url_for('upload'))
        
        # Prepare summary information
        summary = {
            'filename': selected_dataset.name,
            'symbol': selected_dataset.symbol,
            'rows': selected_dataset.rows_count,
            'columns': list(data.columns),
            'start': selected_dataset.start_date.strftime('%Y-%m-%d') if selected_dataset.start_date else 'N/A',
            'end': selected_dataset.end_date.strftime('%Y-%m-%d') if selected_dataset.end_date else 'N/A',
            'description': selected_dataset.description,
        }
        
        # Calculate statistics
        stats = {}
        if 'close' in data.columns:
            stats = {
                'min': data['close'].min(),
                'max': data['close'].max(),
                'mean': data['close'].mean(),
                'std': data['close'].std(),
                'median': data['close'].median(),
                'last': data['close'].iloc[-1] if not data.empty else None,
                'change': ((data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100) if not data.empty and data['close'].iloc[0] > 0 else 0,
            }
        
        # Generate price chart
        chart_data = generate_price_chart(data, selected_dataset.symbol)
        
        return render_template('analysis.html',
                            user_datasets=user_datasets,
                            selected_dataset=selected_dataset,
                            summary=summary,
                            stats=stats,
                            chart_data=chart_data)
                                
    except Exception as e:
        logger.error(f"Errore durante l'analisi del dataset: {str(e)}")
        flash(f'Errore durante l\'analisi del dataset: {str(e)}', 'danger')
        return redirect(url_for('upload'))

@app.route('/indicators', methods=['GET', 'POST'])
def indicators():
    """Technical indicators page"""
    # Check if user is logged in
    user = get_current_user()
    if not user:
        flash('Devi effettuare il login per accedere a questa pagina.', 'warning')
        return redirect(url_for('login', next=request.url))
    
    # Get user's datasets
    user_datasets = Dataset.query.filter_by(user_id=user.id).order_by(Dataset.created_at.desc()).all()
    
    # Get selected dataset if provided
    dataset_id = request.args.get('dataset_id', type=int) or request.form.get('dataset_id', type=int)
    selected_dataset = None
    
    if dataset_id:
        selected_dataset = Dataset.query.filter_by(id=dataset_id, user_id=user.id).first()
        
    # Handle POST request to calculate indicators
    if request.method == 'POST' and selected_dataset:
        try:
            # Get the indicators to calculate
            selected_indicators = request.form.getlist('indicators')
            
            if not selected_indicators:
                flash('Seleziona almeno un indicatore da calcolare.', 'warning')
                return redirect(url_for('indicators', dataset_id=dataset_id))
            
            # Load dataset
            data = load_dataset(selected_dataset.id)
            
            if data is None:
                flash('Impossibile caricare il dataset. File non trovato.', 'danger')
                return redirect(url_for('indicators'))
            
            # Calculate selected indicators
            calculated = []
            
            # SMA - Simple Moving Average
            if 'sma' in selected_indicators:
                period = int(request.form.get('sma_period', 20))
                data[f'SMA_{period}'] = data['close'].rolling(window=period).mean()
                calculated.append(f'SMA ({period})')
            
            # EMA - Exponential Moving Average
            if 'ema' in selected_indicators:
                period = int(request.form.get('ema_period', 20))
                data[f'EMA_{period}'] = data['close'].ewm(span=period, adjust=False).mean()
                calculated.append(f'EMA ({period})')
            
            # MACD - Moving Average Convergence Divergence
            if 'macd' in selected_indicators:
                fast = int(request.form.get('macd_fast', 12))
                slow = int(request.form.get('macd_slow', 26))
                signal = int(request.form.get('macd_signal', 9))
                
                data[f'EMA_{fast}'] = data['close'].ewm(span=fast, adjust=False).mean()
                data[f'EMA_{slow}'] = data['close'].ewm(span=slow, adjust=False).mean()
                
                # MACD Line
                data['MACD'] = data[f'EMA_{fast}'] - data[f'EMA_{slow}']
                
                # Signal Line
                data['MACD_Signal'] = data['MACD'].ewm(span=signal, adjust=False).mean()
                
                # Histogram
                data['MACD_Hist'] = data['MACD'] - data['MACD_Signal']
                
                calculated.append(f'MACD ({fast},{slow},{signal})')
            
            # RSI - Relative Strength Index
            if 'rsi' in selected_indicators:
                period = int(request.form.get('rsi_period', 14))
                
                # Calculate price changes
                delta = data['close'].diff()
                
                # Separate gains and losses
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                
                # Calculate average gain and loss
                avg_gain = gain.rolling(window=period).mean()
                avg_loss = loss.rolling(window=period).mean()
                
                # Calculate RS and RSI
                rs = avg_gain / avg_loss
                data['RSI'] = 100 - (100 / (1 + rs))
                
                calculated.append(f'RSI ({period})')
            
            # Bollinger Bands
            if 'bb' in selected_indicators:
                period = int(request.form.get('bb_period', 20))
                stddev = float(request.form.get('bb_stddev', 2))
                
                # Calculate middle band (SMA)
                data['BB_Middle'] = data['close'].rolling(window=period).mean()
                
                # Calculate standard deviation
                data['BB_StdDev'] = data['close'].rolling(window=period).std()
                
                # Calculate upper and lower bands
                data['BB_Upper'] = data['BB_Middle'] + (data['BB_StdDev'] * stddev)
                data['BB_Lower'] = data['BB_Middle'] - (data['BB_StdDev'] * stddev)
                
                calculated.append(f'Bollinger Bands ({period}, {stddev})')
            
            # Stochastic Oscillator
            if 'stoch' in selected_indicators:
                k_period = int(request.form.get('stoch_k', 14))
                d_period = int(request.form.get('stoch_d', 3))
                smooth = int(request.form.get('stoch_smooth', 3))
                
                # Calculate %K
                low_min = data['low'].rolling(window=k_period).min()
                high_max = data['high'].rolling(window=k_period).max()
                
                data['%K'] = 100 * ((data['close'] - low_min) / (high_max - low_min))
                
                # Apply smoothing to %K if needed
                if smooth > 1:
                    data['%K'] = data['%K'].rolling(window=smooth).mean()
                
                # Calculate %D (SMA of %K)
                data['%D'] = data['%K'].rolling(window=d_period).mean()
                
                calculated.append(f'Stochastic ({k_period}, {d_period}, {smooth})')
            
            # Generate price chart with indicators
            plt.figure(figsize=(12, 8))
            
            # Plot price
            plt.subplot(2, 1, 1)
            plt.plot(data.index, data['close'], label='Prezzo')
            
            # Plot trend indicators on price chart
            if 'sma' in selected_indicators:
                period = int(request.form.get('sma_period', 20))
                plt.plot(data.index, data[f'SMA_{period}'], label=f'SMA {period}')
            
            if 'ema' in selected_indicators:
                period = int(request.form.get('ema_period', 20))
                plt.plot(data.index, data[f'EMA_{period}'], label=f'EMA {period}')
            
            if 'bb' in selected_indicators:
                plt.plot(data.index, data['BB_Upper'], 'r--', label='BB Upper')
                plt.plot(data.index, data['BB_Middle'], 'g--', label='BB Middle')
                plt.plot(data.index, data['BB_Lower'], 'r--', label='BB Lower')
            
            plt.title(f"{selected_dataset.symbol} - Prezzo con Indicatori")
            plt.ylabel('Prezzo')
            plt.grid(True, alpha=0.3)
            plt.legend(loc='upper left')
            
            # Plot oscillators in separate subplot
            has_oscillators = 'rsi' in selected_indicators or 'macd' in selected_indicators or 'stoch' in selected_indicators
            
            if has_oscillators:
                plt.subplot(2, 1, 2)
                
                if 'rsi' in selected_indicators:
                    plt.plot(data.index, data['RSI'], label='RSI')
                    plt.axhline(y=70, color='r', linestyle='--', alpha=0.3)
                    plt.axhline(y=30, color='g', linestyle='--', alpha=0.3)
                
                if 'macd' in selected_indicators:
                    plt.plot(data.index, data['MACD'], label='MACD')
                    plt.plot(data.index, data['MACD_Signal'], label='Signal')
                    
                    # Plot histogram as bars
                    plt.bar(data.index, data['MACD_Hist'], width=1, label='Histogram', alpha=0.3)
                
                if 'stoch' in selected_indicators:
                    plt.plot(data.index, data['%K'], label='%K')
                    plt.plot(data.index, data['%D'], label='%D')
                    plt.axhline(y=80, color='r', linestyle='--', alpha=0.3)
                    plt.axhline(y=20, color='g', linestyle='--', alpha=0.3)
                
                plt.title("Oscillatori")
                plt.ylabel('Valore')
                plt.grid(True, alpha=0.3)
                plt.legend(loc='upper left')
            
            # Save plot to buffer
            buffer = io.BytesIO()
            plt.tight_layout()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            
            # Convert plot to base64 for embedding in HTML
            chart_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
            flash(f'Indicatori calcolati con successo: {", ".join(calculated)}', 'success')
            return render_template('indicators_result.html', 
                                  user_datasets=user_datasets,
                                  selected_dataset=selected_dataset,
                                  chart_data=chart_data,
                                  calculated=calculated)
            
        except Exception as e:
            logger.error(f"Errore durante il calcolo degli indicatori: {str(e)}")
            flash(f'Errore durante il calcolo degli indicatori: {str(e)}', 'danger')
            return redirect(url_for('indicators', dataset_id=dataset_id))
    
    return render_template('indicators.html', 
                          user_datasets=user_datasets,
                          selected_dataset=selected_dataset)

@app.route('/backtest', methods=['GET', 'POST'])
def backtest():
    """Backtesting page"""
    # Check if user is logged in
    user = get_current_user()
    if not user:
        flash('Devi effettuare il login per accedere a questa pagina.', 'warning')
        return redirect(url_for('login', next=request.url))
    
    # Get user's datasets
    user_datasets = Dataset.query.filter_by(user_id=user.id).order_by(Dataset.created_at.desc()).all()
    
    # Get selected dataset if provided
    dataset_id = request.args.get('dataset_id', type=int) or request.form.get('dataset_id', type=int)
    selected_dataset = None
    
    if dataset_id:
        selected_dataset = Dataset.query.filter_by(id=dataset_id, user_id=user.id).first()
    
    # Handle POST request to run backtest
    if request.method == 'POST' and selected_dataset:
        try:
            # Get strategy parameters
            strategy_type = request.form.get('strategy_type')
            initial_capital = float(request.form.get('initial_capital', 10000))
            commission_rate = float(request.form.get('commission_rate', 0.001))
            
            if not strategy_type:
                flash('Seleziona una strategia da testare.', 'warning')
                return redirect(url_for('backtest', dataset_id=dataset_id))
            
            # Load dataset
            data = load_dataset(selected_dataset.id)
            
            if data is None:
                flash('Impossibile caricare il dataset. File non trovato.', 'danger')
                return redirect(url_for('backtest'))
            
            # Make sure all required columns are present
            required_columns = ['open', 'high', 'low', 'close']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                flash(f'Colonne mancanti nel dataset: {", ".join(missing_columns)}', 'danger')
                return redirect(url_for('backtest', dataset_id=dataset_id))
            
            # Initialize results
            results = {
                'initial_capital': initial_capital,
                'final_capital': 0,
                'total_return': 0,
                'annualized_return': 0,
                'max_drawdown': 0,
                'win_rate': 0,
                'profit_factor': 0,
                'trades': [],
                'equity_curve': []
            }
            
            # Set up the dataframe for the backtest
            backtest_data = data.copy()
            
            # Add columns for signals, positions, and equity
            backtest_data['signal'] = 0  # 1 for buy, -1 for sell, 0 for hold
            backtest_data['position'] = 0  # Current position
            backtest_data['equity'] = initial_capital  # Equity curve
            
            # Implement strategies
            if strategy_type == 'sma_crossover':
                # Moving Average Crossover
                fast_period = int(request.form.get('sma_fast_period', 50))
                slow_period = int(request.form.get('sma_slow_period', 200))
                
                # Calculate moving averages
                backtest_data[f'SMA_{fast_period}'] = backtest_data['close'].rolling(window=fast_period).mean()
                backtest_data[f'SMA_{slow_period}'] = backtest_data['close'].rolling(window=slow_period).mean()
                
                # Generate signals: 1 when fast crosses above slow, -1 when fast crosses below slow
                backtest_data['signal'] = 0
                
                # Golden Cross (Buy signal)
                golden_cross = (backtest_data[f'SMA_{fast_period}'] > backtest_data[f'SMA_{slow_period}']) & \
                               (backtest_data[f'SMA_{fast_period}'].shift(1) <= backtest_data[f'SMA_{slow_period}'].shift(1))
                
                # Death Cross (Sell signal)
                death_cross = (backtest_data[f'SMA_{fast_period}'] < backtest_data[f'SMA_{slow_period}']) & \
                              (backtest_data[f'SMA_{fast_period}'].shift(1) >= backtest_data[f'SMA_{slow_period}'].shift(1))
                
                backtest_data.loc[golden_cross, 'signal'] = 1
                backtest_data.loc[death_cross, 'signal'] = -1
                
                strategy_name = f"SMA Crossover ({fast_period}/{slow_period})"
                
            elif strategy_type == 'rsi_strategy':
                # RSI Strategy
                rsi_period = int(request.form.get('rsi_period', 14))
                rsi_overbought = int(request.form.get('rsi_overbought', 70))
                rsi_oversold = int(request.form.get('rsi_oversold', 30))
                
                # Calculate RSI
                delta = backtest_data['close'].diff()
                gain = delta.where(delta > 0, 0)
                loss = -delta.where(delta < 0, 0)
                
                avg_gain = gain.rolling(window=rsi_period).mean()
                avg_loss = loss.rolling(window=rsi_period).mean()
                
                rs = avg_gain / avg_loss
                backtest_data['RSI'] = 100 - (100 / (1 + rs))
                
                # Generate signals
                backtest_data['signal'] = 0
                
                # Buy signal: RSI crosses above oversold
                buy_signal = (backtest_data['RSI'] > rsi_oversold) & (backtest_data['RSI'].shift(1) <= rsi_oversold)
                
                # Sell signal: RSI crosses below overbought
                sell_signal = (backtest_data['RSI'] < rsi_overbought) & (backtest_data['RSI'].shift(1) >= rsi_overbought)
                
                backtest_data.loc[buy_signal, 'signal'] = 1
                backtest_data.loc[sell_signal, 'signal'] = -1
                
                strategy_name = f"RSI Strategy (Period: {rsi_period}, Overbought: {rsi_overbought}, Oversold: {rsi_oversold})"
                
            elif strategy_type == 'macd_strategy':
                # MACD Strategy
                macd_fast = int(request.form.get('macd_fast', 12))
                macd_slow = int(request.form.get('macd_slow', 26))
                macd_signal = int(request.form.get('macd_signal', 9))
                
                # Calculate MACD
                backtest_data[f'EMA_{macd_fast}'] = backtest_data['close'].ewm(span=macd_fast, adjust=False).mean()
                backtest_data[f'EMA_{macd_slow}'] = backtest_data['close'].ewm(span=macd_slow, adjust=False).mean()
                
                # MACD Line
                backtest_data['MACD'] = backtest_data[f'EMA_{macd_fast}'] - backtest_data[f'EMA_{macd_slow}']
                
                # Signal Line
                backtest_data['MACD_Signal'] = backtest_data['MACD'].ewm(span=macd_signal, adjust=False).mean()
                
                # Histogram
                backtest_data['MACD_Hist'] = backtest_data['MACD'] - backtest_data['MACD_Signal']
                
                # Generate signals
                backtest_data['signal'] = 0
                
                # Buy signal: MACD crosses above signal line
                buy_signal = (backtest_data['MACD'] > backtest_data['MACD_Signal']) & \
                            (backtest_data['MACD'].shift(1) <= backtest_data['MACD_Signal'].shift(1))
                
                # Sell signal: MACD crosses below signal line
                sell_signal = (backtest_data['MACD'] < backtest_data['MACD_Signal']) & \
                             (backtest_data['MACD'].shift(1) >= backtest_data['MACD_Signal'].shift(1))
                
                backtest_data.loc[buy_signal, 'signal'] = 1
                backtest_data.loc[sell_signal, 'signal'] = -1
                
                strategy_name = f"MACD Strategy (Fast: {macd_fast}, Slow: {macd_slow}, Signal: {macd_signal})"
                
            elif strategy_type == 'triple_sma':
                # Triple SMA Strategy (Custom strategy)
                sma_short = int(request.form.get('sma_short', 5))
                sma_medium = int(request.form.get('sma_medium', 20))
                sma_long = int(request.form.get('sma_long', 50))
                
                # Calculate SMAs
                backtest_data[f'SMA_{sma_short}'] = backtest_data['close'].rolling(window=sma_short).mean()
                backtest_data[f'SMA_{sma_medium}'] = backtest_data['close'].rolling(window=sma_medium).mean()
                backtest_data[f'SMA_{sma_long}'] = backtest_data['close'].rolling(window=sma_long).mean()
                
                # Generate signals
                backtest_data['signal'] = 0
                
                # Buy signal: Short > Medium > Long and previous day was not in this configuration
                buy_condition = (backtest_data[f'SMA_{sma_short}'] > backtest_data[f'SMA_{sma_medium}']) & \
                               (backtest_data[f'SMA_{sma_medium}'] > backtest_data[f'SMA_{sma_long}'])
                
                buy_signal = buy_condition & (~buy_condition.shift(1).fillna(False))
                
                # Sell signal: Short < Medium and previous day Short was > Medium
                sell_condition = (backtest_data[f'SMA_{sma_short}'] < backtest_data[f'SMA_{sma_medium}'])
                sell_signal = sell_condition & (~sell_condition.shift(1).fillna(False)) & \
                             (backtest_data[f'SMA_{sma_short}'].shift(1) > backtest_data[f'SMA_{sma_medium}'].shift(1))
                
                backtest_data.loc[buy_signal, 'signal'] = 1
                backtest_data.loc[sell_signal, 'signal'] = -1
                
                strategy_name = f"Triple SMA Strategy (Short: {sma_short}, Medium: {sma_medium}, Long: {sma_long})"
                
            elif strategy_type == 'qqe_strategy':
                # QQE Trading Strategy (TradingView implementation)
                qqe_period = int(request.form.get('qqe_period', 18))
                stop_loss_pct = float(request.form.get('stop_loss_pct', 2.0)) / 100  # Convert percentage to decimal
                
                # Calculate highest high and lowest low for QQE period
                backtest_data['high_period'] = backtest_data['high'].rolling(window=qqe_period).max()
                backtest_data['low_period'] = backtest_data['low'].rolling(window=qqe_period).min()
                
                # Calculate QQE value based on the formula from TradingView code
                backtest_data['qqe'] = (backtest_data['close'] - 0.5 * (backtest_data['high_period'] + backtest_data['low_period'])) / \
                                      (0.5 * (backtest_data['high_period'] - backtest_data['low_period']))
                
                # Generate signals: 1 when QQE crosses above zero, -1 when QQE crosses below zero
                backtest_data['signal'] = 0
                
                # Calculate crossovers and crossunders
                long_signal = (backtest_data['qqe'] > 0) & (backtest_data['qqe'].shift(1) <= 0)
                short_signal = (backtest_data['qqe'] < 0) & (backtest_data['qqe'].shift(1) >= 0)
                
                backtest_data.loc[long_signal, 'signal'] = 1
                backtest_data.loc[short_signal, 'signal'] = -1
                
                # Store stop loss levels for trades
                backtest_data['stop_loss_long'] = backtest_data['low_period'] - stop_loss_pct * backtest_data['close']
                backtest_data['stop_loss_short'] = backtest_data['high_period'] + stop_loss_pct * backtest_data['close']
                
                strategy_name = f"QQE Strategy (Period: {qqe_period}, Stop Loss: {stop_loss_pct*100:.1f}%)"
                
            else:
                flash('Strategia non riconosciuta.', 'danger')
                return redirect(url_for('backtest', dataset_id=dataset_id))
            
            # Remove NaN values from the beginning of the dataset due to moving averages
            backtest_data = backtest_data.dropna()
            
            # Run the backtest
            current_position = 0
            current_capital = initial_capital
            trades = []
            entry_price = 0
            entry_date = None
            
            # Debug
            logger.debug(f"Starting backtest with capital: {current_capital}")
            signals_count = backtest_data[backtest_data['signal'] != 0].shape[0]
            logger.debug(f"Total signals in dataset: {signals_count}")
            
            for idx, row in backtest_data.iterrows():
                # Check for signals
                if row['signal'] == 1 and current_position == 0:  # Buy signal and no position
                    # Calculate shares to buy (all in)
                    price = row['close']
                    
                    # Avoid division by zero or very small prices
                    if price <= 0.0001:
                        continue
                        
                    shares = current_capital / price  # All-in approach
                    cost = shares * price * (1 + commission_rate)  # Include commission
                    
                    if cost <= current_capital:
                        current_position = shares
                        entry_price = price
                        entry_date = idx
                        
                        # Record trade entry
                        trades.append({
                            'type': 'buy',
                            'date': str(entry_date),
                            'price': float(entry_price),
                            'shares': float(shares),
                            'value': float(shares * entry_price),
                            'commission': float(shares * entry_price * commission_rate)
                        })
                        
                        # Debug
                        logger.debug(f"BUY at {idx}: price={price}, shares={shares}, capital={current_capital}")
                
                elif (row['signal'] == -1 or idx == backtest_data.index[-1]) and current_position > 0:  # Sell signal or end of data
                    # Sell position
                    price = row['close']
                    shares = current_position
                    value = shares * price
                    commission = value * commission_rate
                    buy_commission = shares * entry_price * commission_rate
                    profit = value - (shares * entry_price) - commission - buy_commission
                    
                    # Update capital
                    current_capital = current_capital + profit
                    current_position = 0
                    
                    # Record trade exit
                    exit_date = idx
                    trades.append({
                        'type': 'sell',
                        'date': str(exit_date),
                        'price': float(price),
                        'shares': float(shares),
                        'value': float(value),
                        'commission': float(commission),
                        'profit': float(profit),
                        'profit_pct': float((profit / (shares * entry_price)) * 100)
                    })
                    
                    # Debug
                    logger.debug(f"SELL at {idx}: price={price}, shares={shares}, profit={profit}, new capital={current_capital}")
                
                # Update equity for this row
                if current_position > 0:
                    # Current value of position minus commissions
                    position_value = current_position * row['close']
                    equity = (current_capital - (current_position * entry_price)) + position_value
                else:
                    equity = current_capital
                
                backtest_data.at[idx, 'position'] = current_position
                backtest_data.at[idx, 'equity'] = equity
            
            # Calculate final results
            final_capital = current_capital if current_position == 0 else current_capital + (current_position * backtest_data['close'].iloc[-1]) - (current_position * backtest_data['close'].iloc[-1] * commission_rate)
            
            # Create equity curve
            equity_curve = backtest_data['equity'].tolist()
            
            # Calculate stats
            total_return = ((final_capital / initial_capital) - 1) * 100
            
            # Annualized return
            start_date = backtest_data.index[0]
            end_date = backtest_data.index[-1]
            years = (end_date - start_date).days / 365.25
            annualized_return = (pow((final_capital / initial_capital), (1 / years)) - 1) * 100 if years > 0 else 0
            
            # Maximum drawdown
            rolling_max = backtest_data['equity'].cummax()
            drawdowns = (backtest_data['equity'] / rolling_max - 1) * 100
            max_drawdown = abs(drawdowns.min())
            
            # Win rate and profit factor
            winning_trades = [t for t in trades if t.get('type') == 'sell' and t.get('profit', 0) > 0]
            losing_trades = [t for t in trades if t.get('type') == 'sell' and t.get('profit', 0) <= 0]
            
            total_trades = len(winning_trades) + len(losing_trades)
            win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0
            
            gross_profit = sum(t.get('profit', 0) for t in winning_trades)
            gross_loss = abs(sum(t.get('profit', 0) for t in losing_trades))
            profit_factor = gross_profit / gross_loss if gross_loss > 0 else float('inf')
            
            # Prepare results
            results = {
                'strategy_name': strategy_name,
                'initial_capital': initial_capital,
                'final_capital': final_capital,
                'total_return': total_return,
                'annualized_return': annualized_return,
                'max_drawdown': max_drawdown,
                'win_rate': win_rate,
                'profit_factor': profit_factor,
                'total_trades': total_trades,
                'trades': trades,
                'equity_curve': equity_curve,
                'dates': [d.strftime('%Y-%m-%d') for d in backtest_data.index.tolist()]
            }
            
            # Generate performance chart
            plt.figure(figsize=(12, 8))
            
            # Plot equity curve
            plt.subplot(2, 1, 1)
            plt.plot(backtest_data.index, backtest_data['equity'], label='Equity')
            plt.title(f"Performance: {strategy_name}")
            plt.ylabel('Equity ($)')
            plt.grid(True, alpha=0.3)
            plt.legend(loc='upper left')
            
            # Plot buy/sell signals
            plt.subplot(2, 1, 2)
            plt.plot(backtest_data.index, backtest_data['close'], label='Close Price')
            
            # Plot buy signals
            buy_signals = backtest_data[backtest_data['signal'] == 1]
            plt.scatter(buy_signals.index, buy_signals['close'], marker='^', color='green', label='Buy Signal', alpha=0.7, s=100)
            
            # Plot sell signals
            sell_signals = backtest_data[backtest_data['signal'] == -1]
            plt.scatter(sell_signals.index, sell_signals['close'], marker='v', color='red', label='Sell Signal', alpha=0.7, s=100)
            
            plt.title(f"{selected_dataset.symbol} - Segnali di Trading")
            plt.ylabel('Prezzo')
            plt.grid(True, alpha=0.3)
            plt.legend(loc='upper left')
            
            plt.tight_layout()
            
            # Save plot to buffer
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            
            # Convert plot to base64 for embedding in HTML
            chart_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
            # Save backtest results to session
            session['backtest_results'] = results
            session['backtest_chart'] = chart_data
            
            # Return results template
            return render_template('backtest_result.html',
                                  user_datasets=user_datasets,
                                  selected_dataset=selected_dataset,
                                  results=results,
                                  chart_data=chart_data)
            
        except Exception as e:
            logger.error(f"Errore durante il backtest: {str(e)}")
            flash(f'Errore durante il backtest: {str(e)}', 'danger')
            return redirect(url_for('backtest', dataset_id=dataset_id))
    
    return render_template('backtest.html',
                          user_datasets=user_datasets,
                          selected_dataset=selected_dataset)

@app.route('/models', methods=['GET', 'POST'])
def models():
    """ML Models page"""
    # Check if user is logged in
    user = get_current_user()
    if not user:
        flash('Devi effettuare il login per accedere a questa pagina.', 'warning')
        return redirect(url_for('login', next=request.url))
    
    # Get user's datasets
    user_datasets = Dataset.query.filter_by(user_id=user.id).order_by(Dataset.created_at.desc()).all()
    
    # Get selected dataset if provided
    dataset_id = request.args.get('dataset_id', type=int) or request.form.get('dataset_id', type=int)
    selected_dataset = None
    
    if dataset_id:
        selected_dataset = Dataset.query.filter_by(id=dataset_id, user_id=user.id).first()
    
    # Check for GPU availability
    try:
        import tensorflow as tf
        gpu_available = len(tf.config.list_physical_devices('GPU')) > 0
        if gpu_available:
            # Enable memory growth to avoid allocating all GPU memory
            for gpu in tf.config.list_physical_devices('GPU'):
                tf.config.experimental.set_memory_growth(gpu, True)
            gpu_info = tf.config.list_physical_devices('GPU')[0].device_type
            logger.debug(f"GPU detected: {gpu_info}")
        else:
            logger.debug("No GPU available, using CPU")
    except Exception as e:
        gpu_available = False
        logger.debug(f"Error checking GPU availability: {str(e)}")
    
    # Handle POST request to train model
    if request.method == 'POST' and selected_dataset:
        try:
            # Get model parameters
            model_type = request.form.get('model_type')
            lookback = int(request.form.get('lookback', 30))
            epochs = int(request.form.get('epochs', 50))
            batch_size = int(request.form.get('batch_size', 32))
            test_size = float(request.form.get('test_size', 0.2)) / 100.0  # Convert from percentage
            
            if not model_type:
                flash('Seleziona un tipo di modello da addestrare.', 'warning')
                return redirect(url_for('models', dataset_id=dataset_id))
            
            # Load dataset
            data = load_dataset(selected_dataset.id)
            
            if data is None:
                flash('Impossibile caricare il dataset. File non trovato.', 'danger')
                return redirect(url_for('models'))
            
            # Check if 'close' column exists
            if 'close' not in data.columns:
                flash('Colonna "close" mancante nel dataset. Questo campo è richiesto per i modelli ML.', 'danger')
                return redirect(url_for('models', dataset_id=dataset_id))
            
            # Use only 'close' price for prediction
            import numpy as np
            data_price = np.array(data['close']).reshape(-1, 1)
            
            # Normalize data
            from sklearn.preprocessing import MinMaxScaler
            scaler = MinMaxScaler(feature_range=(0, 1))
            data_normalized = scaler.fit_transform(data_price)
            
            # Create sequences for training
            X = []
            y = []
            for i in range(len(data_normalized) - lookback):
                X.append(data_normalized[i:i+lookback])
                y.append(data_normalized[i+lookback])
            
            X = np.array(X)
            y = np.array(y)
            
            # Split data into training and testing sets
            from sklearn.model_selection import train_test_split
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, shuffle=False)
            
            # Define model builder function first
            def build_model(model_type, lookback):
                import tensorflow as tf
                from tensorflow.keras.models import Sequential
                from tensorflow.keras.layers import Dense, LSTM, Dropout, SimpleRNN, GRU
                from tensorflow.keras.optimizers import Adam
                
                model = Sequential()
                model_name = ""
                
                if model_type == 'lstm':
                    model.add(LSTM(units=50, return_sequences=True, input_shape=(lookback, 1)))
                    model.add(Dropout(0.2))
                    model.add(LSTM(units=50, return_sequences=False))
                    model.add(Dropout(0.2))
                    model.add(Dense(units=1))
                    model_name = "LSTM"
                    
                elif model_type == 'rnn':
                    model.add(SimpleRNN(units=50, return_sequences=True, input_shape=(lookback, 1)))
                    model.add(Dropout(0.2))
                    model.add(SimpleRNN(units=50, return_sequences=False))
                    model.add(Dropout(0.2))
                    model.add(Dense(units=1))
                    model_name = "RNN semplice"
                    
                elif model_type == 'gru':
                    model.add(GRU(units=50, return_sequences=True, input_shape=(lookback, 1)))
                    model.add(Dropout(0.2))
                    model.add(GRU(units=50, return_sequences=False))
                    model.add(Dropout(0.2))
                    model.add(Dense(units=1))
                    model_name = "GRU"
                
                else:
                    return None, None
                
                model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')
                return model, model_name
            
            # Initialize model
            import tensorflow as tf
            from tensorflow.keras.callbacks import EarlyStopping
            
            # Set device placement (CPU/GPU)
            if gpu_available:
                with tf.device('/GPU:0'):
                    logger.debug("Building model on GPU")
                    model, model_name = build_model(model_type, lookback)
            else:
                with tf.device('/CPU:0'):
                    logger.debug("Building model on CPU")
                    model, model_name = build_model(model_type, lookback)
            
            if model is None:
                flash('Tipo di modello non riconosciuto.', 'danger')
                return redirect(url_for('models', dataset_id=dataset_id))
            
            # Train model with early stopping
            early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
            
            # Training message
            logger.debug(f"Starting training of {model_name} model with {len(X_train)} samples")
            
            history = model.fit(
                X_train, y_train,
                epochs=epochs,
                batch_size=batch_size,
                validation_data=(X_test, y_test),
                callbacks=[early_stopping],
                verbose=1
            )
            
            # Make predictions
            y_pred = model.predict(X_test)
            
            # Inverse transform the predictions and actual values
            y_test_actual = scaler.inverse_transform(y_test)
            y_pred_actual = scaler.inverse_transform(y_pred)
            
            # Calculate metrics
            from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
            mse = mean_squared_error(y_test_actual, y_pred_actual)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(y_test_actual, y_pred_actual)
            r2 = r2_score(y_test_actual, y_pred_actual)
            
            # Generate forecast plot
            plt.figure(figsize=(12, 6))
            
            # Plot actual vs predicted on test set
            test_dates = data.index[-len(y_test):]
            plt.plot(test_dates, y_test_actual, label='Valori Reali')
            plt.plot(test_dates, y_pred_actual, label='Previsioni')
            
            plt.title(f"Previsione Prezzo - Modello {model_name}")
            plt.xlabel('Data')
            plt.ylabel('Prezzo')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save plot to buffer
            buffer = io.BytesIO()
            plt.savefig(buffer, format='png', dpi=100)
            buffer.seek(0)
            
            # Convert plot to base64 for embedding in HTML
            chart_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
            plt.close()
            
            # Generate loss plot
            plt.figure(figsize=(10, 6))
            plt.plot(history.history['loss'], label='Training Loss')
            plt.plot(history.history['val_loss'], label='Validation Loss')
            plt.title('Training and Validation Loss')
            plt.xlabel('Epochs')
            plt.ylabel('Loss (MSE)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Save loss plot to buffer
            loss_buffer = io.BytesIO()
            plt.savefig(loss_buffer, format='png', dpi=100)
            loss_buffer.seek(0)
            
            # Convert plot to base64 for embedding in HTML
            loss_chart_data = base64.b64encode(loss_buffer.getvalue()).decode('utf-8')
            plt.close()
            
            # Save model
            import os
            model_dir = os.path.join(os.getcwd(), 'models/saved')
            os.makedirs(model_dir, exist_ok=True)
            model_path = os.path.join(model_dir, f"{selected_dataset.symbol}_{model_type}_{lookback}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.h5")
            model.save(model_path)
            
            # Prepare results
            results = {
                'model_type': model_type,
                'model_name': model_name,
                'lookback': lookback,
                'epochs': epochs,
                'batch_size': batch_size,
                'test_size': test_size * 100,  # Convert back to percentage
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'r2': r2,
                'trained_epochs': len(history.history['loss']),
                'model_path': model_path,
                'gpu_used': gpu_available
            }
            
            # Store in session for potential reuse
            session['model_results'] = results
            
            # Return results template
            return render_template('models_result.html',
                                 user_datasets=user_datasets,
                                 selected_dataset=selected_dataset,
                                 results=results,
                                 chart_data=chart_data,
                                 loss_chart_data=loss_chart_data)
            
        except Exception as e:
            logger.error(f"Errore durante il training del modello: {str(e)}")
            flash(f'Errore durante il training del modello: {str(e)}', 'danger')
            return redirect(url_for('models', dataset_id=dataset_id))
    
    return render_template('models.html', 
                          user_datasets=user_datasets,
                          selected_dataset=selected_dataset,
                          gpu_available=gpu_available)

@app.route('/clear_data')
def clear_data():
    """Clear session data and return to home page"""
    session.clear()
    flash('Dati di sessione eliminati con successo.', 'success')
    return redirect(url_for('index'))

@app.errorhandler(404)
def page_not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    logger.error(f"Server error: {str(e)}")
    return render_template('500.html'), 500

@app.context_processor
def inject_globals():
    """Inject global variables into templates"""
    return {
        'app_name': 'CryptoTradeAnalyzer',
        'current_year': datetime.now().year,
        'current_user': get_current_user(),
    }

# Run the app
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)