"""
Script per creare e inizializzare correttamente la tabella ml_model.
Questo script funziona direttamente con l'applicazione Flask.
"""

import os
import sys
from sqlalchemy import create_engine, text
import logging

# Configurazione logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fix_ml_model")

def fix_ml_model_table():
    """Crea la tabella ml_model nel database dell'applicazione"""
    # Configurazione del database
    db_path = 'instance/cryptotradeanalyzer.db'
    db_url = f'sqlite:///{db_path}'
    
    if not os.path.exists(db_path):
        logger.error(f"Database non trovato: {db_path}")
        return False
    
    try:
        # Connessione al database
        engine = create_engine(db_url)
        
        # Verifica se la tabella ml_model esiste
        with engine.connect() as conn:
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='ml_model'"))
            if result.fetchone():
                logger.info("La tabella ml_model esiste già")
                return True
            
            # Crea la tabella se non esiste
            logger.info("Creazione della tabella ml_model...")
            conn.execute(text("""
            CREATE TABLE ml_model (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name VARCHAR(128) NOT NULL,
                model_type VARCHAR(50) NOT NULL,
                parameters TEXT,
                metrics TEXT,
                model_path VARCHAR(256),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                user_id INTEGER NOT NULL,
                dataset_id INTEGER,
                FOREIGN KEY (user_id) REFERENCES user(id),
                FOREIGN KEY (dataset_id) REFERENCES dataset(id)
            )
            """))
            
            # Verifica se la tabella è stata creata
            result = conn.execute(text("SELECT name FROM sqlite_master WHERE type='table' AND name='ml_model'"))
            if result.fetchone():
                logger.info("Tabella ml_model creata con successo")
                return True
            else:
                logger.error("Errore: la tabella ml_model non è stata creata")
                return False
                
    except Exception as e:
        logger.error(f"Errore durante la creazione della tabella ml_model: {str(e)}")
        return False

if __name__ == "__main__":
    if fix_ml_model_table():
        print("La tabella ml_model è stata creata con successo")
    else:
        print("Si è verificato un errore durante la creazione della tabella ml_model")
        sys.exit(1)