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
            
        # Get form data
        name = request.form['name']
        symbol = request.form['symbol']
        description = request.form.get('description', '')
        date_format = request.form.get('date_format', 'auto')
        delimiter = request.form.get('delimiter', ',')
        
        # Save the file with a unique name
        filename = f"{uuid.uuid4()}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
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

@app.route('/indicators')
def indicators():
    """Technical indicators page"""
    # Check if user is logged in
    user = get_current_user()
    if not user:
        flash('Devi effettuare il login per accedere a questa pagina.', 'warning')
        return redirect(url_for('login', next=request.url))
    
    # Get user's datasets
    user_datasets = Dataset.query.filter_by(user_id=user.id).order_by(Dataset.created_at.desc()).all()
    
    return render_template('indicators.html', user_datasets=user_datasets)

@app.route('/backtest')
def backtest():
    """Backtesting page"""
    # Check if user is logged in
    user = get_current_user()
    if not user:
        flash('Devi effettuare il login per accedere a questa pagina.', 'warning')
        return redirect(url_for('login', next=request.url))
    
    # Get user's datasets
    user_datasets = Dataset.query.filter_by(user_id=user.id).order_by(Dataset.created_at.desc()).all()
    
    return render_template('backtest.html', user_datasets=user_datasets)

@app.route('/models')
def models():
    """ML Models page"""
    # Check if user is logged in
    user = get_current_user()
    if not user:
        flash('Devi effettuare il login per accedere a questa pagina.', 'warning')
        return redirect(url_for('login', next=request.url))
    
    # Get user's datasets
    user_datasets = Dataset.query.filter_by(user_id=user.id).order_by(Dataset.created_at.desc()).all()
    
    return render_template('models.html', user_datasets=user_datasets)

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