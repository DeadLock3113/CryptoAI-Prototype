"""
Script temporaneo per la migrazione del database.
"""
from main import app
from database import db
import models  # Importa i modelli

if __name__ == "__main__":
    with app.app_context():
        # Migra il database aggiungendo le nuove colonne
        db.create_all()
        print("Migrazione completata: sono state aggiunte le colonne per API keys e credenziali Telegram.")