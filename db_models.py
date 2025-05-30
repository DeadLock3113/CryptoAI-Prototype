"""
Database Models for CryptoTradeAnalyzer

This module defines the database models used for storing cryptocurrency data,
user information, and analysis results.

Author: CryptoTradeAnalyzer Team
"""

from flask_login import UserMixin
from datetime import datetime
from database import db

class User(UserMixin, db.Model):
    """User model for authentication and data association"""
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(64), unique=True, nullable=False)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(256), nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # API keys
    binance_api_key = db.Column(db.String(256))
    binance_api_secret = db.Column(db.String(256))
    kraken_api_key = db.Column(db.String(256))
    kraken_api_secret = db.Column(db.String(256))
    
    # Telegram settings
    telegram_bot_token = db.Column(db.String(256))
    telegram_chat_id = db.Column(db.String(64))
    
    # Relationships
    datasets = db.relationship('Dataset', backref='user', lazy='dynamic')
    strategies = db.relationship('Strategy', backref='user', lazy='dynamic')
    backtests = db.relationship('Backtest', backref='user', lazy='dynamic')
    api_profiles = db.relationship('ApiProfile', backref='user', lazy='dynamic',
                                  cascade='all, delete-orphan')
    
class NotificationSettings(db.Model):
    """Impostazioni di notifica per l'utente"""
    id = db.Column(db.Integer, primary_key=True)
    timeframe = db.Column(db.String(10), default='1h')  # '1m', '5m', '15m', '30m', '1h', '4h', '1d'
    enabled = db.Column(db.Boolean, default=False)
    last_notification = db.Column(db.DateTime)
    price_change_threshold = db.Column(db.Float, default=1.0)  # Percentuale di cambio per notifiche di movimento significativo
    volume_change_threshold = db.Column(db.Float, default=20.0)  # Percentuale di cambio del volume
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False, unique=True)
    
    # Relazione con l'utente
    user = db.relationship('User', backref=db.backref('notification_settings', uselist=False))
    
    def __repr__(self):
        return f'<NotificationSettings id={self.id} user_id={self.user_id}>'


class ApiProfile(db.Model):
    """API profile model for storing exchange API keys"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(64), nullable=False)
    data = db.Column(db.JSON, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Foreign keys
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    def __repr__(self):
        return f'<ApiProfile {self.name}>'

class Dataset(db.Model):
    """Cryptocurrency dataset model"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), nullable=False)
    symbol = db.Column(db.String(20), nullable=False)
    description = db.Column(db.Text)
    file_path = db.Column(db.String(256), nullable=False)
    rows_count = db.Column(db.Integer)
    start_date = db.Column(db.DateTime)
    end_date = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Foreign keys
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # Relationships
    price_data = db.relationship('PriceData', backref='dataset', lazy='dynamic',
                               cascade='all, delete-orphan')
    backtests = db.relationship('Backtest', backref='dataset', lazy='dynamic')
    indicators = db.relationship('Indicator', backref='dataset', lazy='dynamic',
                               cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Dataset {self.name} ({self.symbol})>'

class PriceData(db.Model):
    """Cryptocurrency price data model"""
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False, index=True)
    open = db.Column(db.Float, nullable=False)
    high = db.Column(db.Float, nullable=False)
    low = db.Column(db.Float, nullable=False)
    close = db.Column(db.Float, nullable=False)
    volume = db.Column(db.Float)
    
    # Foreign keys
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'), nullable=False)
    
    # Add a unique constraint to ensure no duplicate timestamps for a dataset
    __table_args__ = (db.UniqueConstraint('dataset_id', 'timestamp', name='_dataset_timestamp_uc'),)
    
    def __repr__(self):
        return f'<PriceData {self.timestamp} {self.close}>'

class Indicator(db.Model):
    """Technical indicator model"""
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False, index=True)
    indicator_type = db.Column(db.String(50), nullable=False, index=True)
    value = db.Column(db.Float)
    parameters = db.Column(db.JSON)  # Store indicator parameters as JSON
    
    # Foreign keys
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'), nullable=False)
    
    # Add a unique constraint to ensure no duplicate indicators for a timestamp
    __table_args__ = (db.UniqueConstraint('dataset_id', 'timestamp', 'indicator_type', name='_dataset_timestamp_indicator_uc'),)
    
    def __repr__(self):
        return f'<Indicator {self.indicator_type} {self.timestamp}>'

class Strategy(db.Model):
    """Trading strategy model"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), nullable=False)
    description = db.Column(db.Text)
    strategy_type = db.Column(db.String(50), nullable=False)
    parameters = db.Column(db.JSON)  # Store strategy parameters as JSON
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Foreign keys
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    
    # Relationships
    backtests = db.relationship('Backtest', backref='strategy', lazy='dynamic')
    
    def __repr__(self):
        return f'<Strategy {self.name}>'

class Backtest(db.Model):
    """Backtest results model"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), nullable=False)
    description = db.Column(db.Text)
    initial_capital = db.Column(db.Float, nullable=False)
    commission_rate = db.Column(db.Float, default=0.001)
    start_date = db.Column(db.DateTime)
    end_date = db.Column(db.DateTime)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Performance metrics
    total_return = db.Column(db.Float)
    annualized_return = db.Column(db.Float)
    sharpe_ratio = db.Column(db.Float)
    max_drawdown = db.Column(db.Float)
    win_rate = db.Column(db.Float)
    profit_factor = db.Column(db.Float)
    
    # Store data as JSON
    equity_curve = db.Column(db.JSON)  # Store equity curve as JSON
    trades = db.Column(db.JSON)  # Store trades as JSON
    
    # Foreign keys
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'), nullable=False)
    strategy_id = db.Column(db.Integer, db.ForeignKey('strategy.id'), nullable=False)
    
    def __repr__(self):
        return f'<Backtest {self.name}>'

class MLModel(db.Model):
    """Machine learning model metadata"""
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(128), nullable=False)
    model_type = db.Column(db.String(50), nullable=False)
    parameters = db.Column(db.JSON)  # Store model parameters as JSON
    metrics = db.Column(db.JSON)  # Store training/validation metrics as JSON
    model_path = db.Column(db.String(256))  # Path to saved model file
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Foreign keys
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'), nullable=False)
    
    # Relationships
    predictions = db.relationship('Prediction', backref='ml_model', lazy='dynamic',
                                cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<MLModel {self.name} ({self.model_type})>'

class Prediction(db.Model):
    """Model predictions"""
    id = db.Column(db.Integer, primary_key=True)
    timestamp = db.Column(db.DateTime, nullable=False, index=True)
    target_type = db.Column(db.String(20), nullable=False)  # 'price', 'direction', etc.
    value = db.Column(db.Float, nullable=False)
    
    # Foreign keys
    ml_model_id = db.Column(db.Integer, db.ForeignKey('ml_model.id'), nullable=False)
    
    def __repr__(self):
        return f'<Prediction {self.timestamp} {self.value}>'

class SignalConfig(db.Model):
    """Configurazione per la generazione di segnali di trading basati su AI"""
    id = db.Column(db.Integer, primary_key=True)
    config_id = db.Column(db.String(64), unique=True, nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    model_ids = db.Column(db.String(256), nullable=False)  # JSON lista di ID dei modelli
    dataset_id = db.Column(db.Integer, db.ForeignKey('dataset.id'), nullable=False)
    timeframe = db.Column(db.String(10), default='1h')
    risk_level = db.Column(db.Integer, default=2)
    auto_tp_sl = db.Column(db.Boolean, default=True)
    telegram_enabled = db.Column(db.Boolean, default=True)
    is_active = db.Column(db.Boolean, default=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, onupdate=datetime.utcnow)
    
    # Relationships
    user = db.relationship('User', backref=db.backref('signal_configs', lazy='dynamic'))
    dataset = db.relationship('Dataset', backref=db.backref('signal_configs', lazy='dynamic'))
    
    def __repr__(self):
        return f'<SignalConfig {self.config_id}>'