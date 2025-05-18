"""
Script per la creazione della tabella signal_config nel database.
Eseguire questo script una sola volta per aggiungere la tabella mancante.
"""

import logging
import os
from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import text
import sys

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def create_signal_config_table():
    """Crea la tabella signal_config nel database se non esiste"""
    # Crea un'applicazione Flask temporanea
    app = Flask(__name__)
    
    # Configura il database
    app.config['SQLALCHEMY_DATABASE_URI'] = os.environ.get("DATABASE_URL")
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
    
    # Inizializza SQLAlchemy
    db = SQLAlchemy(app)
    
    # Definisci la tabella
    with app.app_context():
        # Esegui la query per creare la tabella
        query = text("""
        CREATE TABLE IF NOT EXISTS signal_config (
            id SERIAL PRIMARY KEY,
            config_id VARCHAR(64) UNIQUE NOT NULL,
            user_id INTEGER NOT NULL REFERENCES "user" (id),
            model_ids VARCHAR(256) NOT NULL,
            dataset_id INTEGER NOT NULL REFERENCES dataset (id),
            timeframe VARCHAR(10) DEFAULT '1h',
            risk_level INTEGER DEFAULT 2,
            auto_tp_sl BOOLEAN DEFAULT TRUE,
            telegram_enabled BOOLEAN DEFAULT TRUE,
            is_active BOOLEAN DEFAULT TRUE,
            created_at TIMESTAMP DEFAULT NOW(),
            updated_at TIMESTAMP
        );
        """)
        
        with db.engine.connect() as conn:
            conn.execute(query)
            conn.commit()
        
        logger.info("Tabella signal_config creata con successo.")

if __name__ == "__main__":
    create_signal_config_table()