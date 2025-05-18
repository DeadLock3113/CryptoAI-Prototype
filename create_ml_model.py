"""
Script per la creazione della tabella ml_model nel database.
Eseguire questo script una sola volta per aggiungere la tabella mancante.
"""

import os
import sqlite3
import logging

# Configurazione del logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("create_ml_model")

def create_ml_model_table():
    """Crea la tabella ml_model nel database se non esiste"""
    # Percorso del database
    db_path = 'instance/cryptotradeanalyzer.db'
    
    # Verifica se il database esiste
    if not os.path.exists(db_path):
        logger.error(f"Database non trovato: {db_path}")
        return False
    
    try:
        # Connessione al database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # Verifica se la tabella esiste
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='ml_model'")
        if cursor.fetchone():
            logger.info("La tabella ml_model esiste già")
            conn.close()
            return True
        
        # Creazione della tabella ml_model
        logger.info("Creazione della tabella ml_model...")
        create_table_sql = """
        CREATE TABLE ml_model (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name VARCHAR(128) NOT NULL,
            model_type VARCHAR(50) NOT NULL,
            parameters JSON,
            metrics JSON,
            model_path VARCHAR(256),
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            user_id INTEGER NOT NULL,
            dataset_id INTEGER,
            FOREIGN KEY (user_id) REFERENCES user(id),
            FOREIGN KEY (dataset_id) REFERENCES dataset(id)
        )
        """
        cursor.execute(create_table_sql)
        conn.commit()
        logger.info("Tabella ml_model creata con successo")
        
        # Chiusura della connessione
        conn.close()
        return True
    except Exception as e:
        logger.error(f"Errore durante la creazione della tabella ml_model: {str(e)}")
        return False

if __name__ == "__main__":
    if create_ml_model_table():
        print("La tabella ml_model è stata creata con successo")
    else:
        print("Si è verificato un errore durante la creazione della tabella ml_model")