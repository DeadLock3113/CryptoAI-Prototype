"""
Script per risolvere il problema delle definizioni duplicate dei modelli SQLAlchemy.
Questo script modifica db_models.py per garantire che tutte le parti dell'applicazione
utilizzino la stessa istanza di SQLAlchemy.
"""

import os
import re
import shutil
import logging

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def backup_file(file_path):
    """Crea un backup del file specificato"""
    backup_path = f"{file_path}.bak"
    shutil.copy2(file_path, backup_path)
    logger.info(f"Backup creato: {backup_path}")
    return backup_path

def fix_db_models():
    """Modifica db_models.py per garantire che l'esportazione dell'istanza db sia corretta"""
    file_path = "db_models.py"
    backup_file(file_path)
    
    # Leggi il contenuto del file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Garantisci che l'istanza db sia importata da database.py e che non ci siano definizioni duplicate
    if "from database import db" not in content:
        content = re.sub(
            r'from flask_sqlalchemy import SQLAlchemy[\r\n]+from sqlalchemy\.orm import UserMixin',
            'from database import db\nfrom sqlalchemy.orm import UserMixin',
            content
        )
    
    # Scrivi il contenuto aggiornato
    with open(file_path, 'w') as f:
        f.write(content)
    
    logger.info(f"File {file_path} aggiornato con successo")

def fix_main_app():
    """Modifica main.py per garantire che l'import di app funzioni correttamente"""
    file_path = "main.py"
    backup_file(file_path)
    
    # Leggi il contenuto del file
    with open(file_path, 'r') as f:
        content = f.read()
    
    # Assicura che main.py importi app da simple_app correttamente
    if 'import app' in content or 'from app import' in content:
        content = content.replace('import app', 'from simple_app import app')
        content = content.replace('from app import', 'from simple_app import')
    
    # Scrivi il contenuto aggiornato
    with open(file_path, 'w') as f:
        f.write(content)
    
    logger.info(f"File {file_path} aggiornato con successo")

def create_unified_app():
    """Crea una nuova versione dell'app con un'architettura più pulita"""
    # Crea un nuovo app.py che inizializza correttamente l'applicazione
    file_path = "unified_app.py"
    with open(file_path, 'w') as f:
        f.write("""'''
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
    SQLALCHEMY_DATABASE_URI=os.environ.get("DATABASE_URL", "sqlite:///crypto_analyzer.db"),
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
""")
    
    logger.info(f"File {file_path} creato con successo")

def main():
    """Funzione principale per la correzione dei modelli"""
    try:
        logger.info("Inizio correzione architettura database...")
        fix_db_models()
        fix_main_app()
        create_unified_app()
        logger.info("Correzione completata con successo!")
        logger.info("Per utilizzare la nuova architettura, esegui: python unified_app.py")
    except Exception as e:
        logger.error(f"Errore durante la correzione: {str(e)}")

if __name__ == "__main__":
    main()