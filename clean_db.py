"""
Script per ripulire il database dei dati sensibili (API keys, account, etc.)
Questo script eliminerà tutti i dati sensibili dal database per un test delle funzionalità.
"""

import os
import sqlite3
import sys
from datetime import datetime

def connect_to_db():
    """Connessione al database"""
    try:
        # Ottieni l'URL del database dall'ambiente
        db_url = os.environ.get("DATABASE_URL")
        
        if not db_url:
            print("Errore: DATABASE_URL non trovato nelle variabili di ambiente.")
            sys.exit(1)
        
        # Rimuovi il prefisso "sqlite:///" se presente
        if db_url.startswith("sqlite:///"):
            db_path = db_url[10:]
        else:
            db_path = db_url
        
        # Connessione al database
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        
        return conn
    except Exception as e:
        print(f"Errore durante la connessione al database: {str(e)}")
        sys.exit(1)

def clean_api_keys(conn):
    """Rimuovi le API keys e i token Telegram"""
    try:
        cursor = conn.cursor()
        
        # Controlla se la tabella user esiste
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user'")
        if not cursor.fetchone():
            print("Tabella 'user' non trovata nel database.")
            return
        
        # Pulisci le API keys e token Telegram
        cursor.execute("""
            UPDATE user SET 
            binance_api_key = NULL,
            binance_api_secret = NULL,
            kraken_api_key = NULL,
            kraken_api_secret = NULL,
            telegram_bot_token = NULL,
            telegram_chat_id = NULL
        """)
        
        conn.commit()
        print("API keys e token Telegram eliminati con successo.")
    except Exception as e:
        conn.rollback()
        print(f"Errore durante la pulizia delle API keys: {str(e)}")

def clean_api_profiles(conn):
    """Elimina i profili API"""
    try:
        cursor = conn.cursor()
        
        # Controlla se la tabella api_profile esiste
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='api_profile'")
        if not cursor.fetchone():
            print("Tabella 'api_profile' non trovata nel database.")
            return
        
        # Elimina tutti i profili API
        cursor.execute("DELETE FROM api_profile")
        
        conn.commit()
        print("Profili API eliminati con successo.")
    except Exception as e:
        conn.rollback()
        print(f"Errore durante l'eliminazione dei profili API: {str(e)}")

def reset_user_password(conn):
    """Resetta la password dell'utente di test"""
    try:
        cursor = conn.cursor()
        
        # Controlla se la tabella user esiste
        cursor.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='user'")
        if not cursor.fetchone():
            print("Tabella 'user' non trovata nel database.")
            return
            
        # Genera una password di test (in questo caso 'password')
        # Non usare werkzeug direttamente qui per evitare dipendenze
        test_password = 'pbkdf2:sha256:600000$oAOKztbGaQ1W3emg$bbe1fcf175c4d53e3f46cc1af58f6c10f92e58ddf71b1e3d6c07a179f6b17cc7'
        
        # Aggiorna la password dell'utente di test (assumendo che esista)
        cursor.execute("UPDATE user SET password_hash = ? WHERE id = 1", (test_password,))
        
        conn.commit()
        print("Password dell'utente resettata con successo.")
    except Exception as e:
        conn.rollback()
        print(f"Errore durante il reset della password: {str(e)}")

def main():
    """Funzione principale"""
    print("Inizializzazione pulizia database...")
    conn = connect_to_db()
    
    try:
        # Pulisci i dati
        clean_api_keys(conn)
        clean_api_profiles(conn)
        reset_user_password(conn)
        
        print("\nPulizia completata con successo!")
        print("Ora puoi testare tutte le funzionalità con un database pulito.")
        print("Le credenziali dell'utente di test sono:")
        print("  Username: admin (o il nome utente esistente)")
        print("  Password: password")
    except Exception as e:
        print(f"Errore durante l'esecuzione: {str(e)}")
    finally:
        conn.close()

if __name__ == "__main__":
    main()