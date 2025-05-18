'''
CryptoTradeAnalyzer - Applicazione unificata

Questa versione utilizza una singola istanza SQLAlchemy per tutto il progetto
e garantisce la corretta interazione con il database.
'''

import os
from flask import Flask

# Creazione app Flask
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
    SQLALCHEMY_DATABASE_URI=os.environ.get("DATABASE_URL"),
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
)

# Ensure upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Importa e inizializza l'istanza db
from database import db
db.init_app(app)

# Create all tables
with app.app_context():
    from db_models import User, Dataset, PriceData, Indicator, Strategy, Backtest, MLModel, Prediction, ApiProfile, SignalConfig
    db.create_all()
    
# Importa le restanti funzionalità da simple_app
from simple_app import (
    update_database_schema, get_current_user, login, logout, register, 
    profile, index, analysis, indicators, backtest, models,
    upload, custom_strategy, training_visualizer
)

if __name__ == "__main__":
    app.run(debug=True)
