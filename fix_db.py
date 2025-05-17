"""
Script per aggiungere le colonne per le API key e credenziali Telegram alla tabella User.
"""
import os
from sqlalchemy import text
from main import app
from database import db

def aggiungi_colonne_missing():
    with app.app_context():
        # Questi comandi sono compatibili con PostgreSQL
        colonne_da_aggiungere = [
            "ALTER TABLE \"user\" ADD COLUMN IF NOT EXISTS binance_api_key VARCHAR(256)",
            "ALTER TABLE \"user\" ADD COLUMN IF NOT EXISTS binance_api_secret VARCHAR(256)",
            "ALTER TABLE \"user\" ADD COLUMN IF NOT EXISTS kraken_api_key VARCHAR(256)",
            "ALTER TABLE \"user\" ADD COLUMN IF NOT EXISTS kraken_api_secret VARCHAR(256)",
            "ALTER TABLE \"user\" ADD COLUMN IF NOT EXISTS telegram_bot_token VARCHAR(256)",
            "ALTER TABLE \"user\" ADD COLUMN IF NOT EXISTS telegram_chat_id VARCHAR(64)"
        ]
        
        conn = db.engine.connect()
        
        for cmd in colonne_da_aggiungere:
            try:
                conn.execute(text(cmd))
                print(f"Eseguito: {cmd}")
            except Exception as e:
                # Ignora errori perché la colonna potrebbe già esistere
                print(f"Nota: {str(e)}")
                
        conn.commit()
        conn.close()
        
        print("Migrazione completata!")

if __name__ == "__main__":
    aggiungi_colonne_missing()