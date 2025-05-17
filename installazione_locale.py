"""
Script di configurazione per l'installazione locale di CryptoTradeAnalyzer.
Questo script verifica i requisiti e configura l'ambiente per l'esecuzione locale.
"""
import os
import sys
import platform
import subprocess
import logging
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("setup")

def controllo_sistema():
    """Verifica i requisiti di sistema"""
    logger.info("Verifica del sistema in corso...")
    
    # Verifica versione Python
    python_version = sys.version_info
    if python_version.major < 3 or (python_version.major == 3 and python_version.minor < 9):
        logger.error("È richiesto Python 3.9 o superiore. Versione rilevata: %s.%s", 
                    python_version.major, python_version.minor)
        return False

    logger.info("✓ Versione Python OK: %s.%s", python_version.major, python_version.minor)

    # Verifica RAM
    try:
        import psutil
        ram_gb = psutil.virtual_memory().total / (1024 * 1024 * 1024)
        if ram_gb < 4:
            logger.warning("Rilevati solo %.1f GB di RAM. Consigliati almeno 4GB.", ram_gb)
        else:
            logger.info("✓ RAM OK: %.1f GB", ram_gb)
    except ImportError:
        logger.warning("Impossibile verificare la RAM (modulo psutil non installato)")

    # Verifica GPU
    check_gpu()
    
    return True

def check_gpu():
    """Verifica la presenza di GPU supportate"""
    try:
        # Verifica CUDA per NVIDIA
        try:
            import torch
            if torch.cuda.is_available():
                logger.info("✓ GPU NVIDIA rilevata con supporto CUDA")
                logger.info("  - Dispositivo: %s", torch.cuda.get_device_name(0))
                logger.info("  - CUDA: %s", torch.version.cuda)
                return "nvidia"
        except (ImportError, AttributeError):
            pass

        # Verifica ROCm per AMD
        try:
            # Il supporto per AMD ROCm esiste in PyTorch, ma richiede un'installazione specifica
            if hasattr(torch, 'hip') and torch.hip.is_available():
                logger.info("✓ GPU AMD rilevata con supporto ROCm")
                return "amd"
        except (ImportError, AttributeError):
            pass

        # Controllo sistema operativo
        if platform.system() == "Linux":
            try:
                gpu_info = subprocess.check_output("lspci | grep -i 'vga\\|3d\\|display'", shell=True).decode()
                if "nvidia" in gpu_info.lower():
                    logger.info("✓ GPU NVIDIA rilevata, ma CUDA non è configurato")
                    logger.info("  Installa CUDA Toolkit per abilitare l'accelerazione GPU")
                elif "amd" in gpu_info.lower() or "radeon" in gpu_info.lower():
                    logger.info("✓ GPU AMD rilevata, ma ROCm non è configurato")
                    logger.info("  Installa ROCm per abilitare l'accelerazione GPU")
                else:
                    logger.info("Nessuna GPU NVIDIA o AMD rilevata")
            except:
                logger.warning("Impossibile verificare la presenza di GPU via lspci")
        else:
            logger.warning("Controllo GPU avanzato disponibile solo su Linux")
    
    except Exception as e:
        logger.warning(f"Errore durante il controllo della GPU: {str(e)}")
    
    logger.info("! L'applicazione verrà eseguita in modalità CPU")
    return "cpu"

def crea_file_configurazione():
    """Crea i file di configurazione necessari"""
    logger.info("Creazione file di configurazione...")
    
    # Crea la configurazione per SQLite (più facile in locale)
    config_content = """# Configurazione locale per CryptoTradeAnalyzer
# Generato automaticamente da installazione_locale.py

DATABASE_URI = "sqlite:///instance/cryptotradeanalyzer.db"
SECRET_KEY = "applicazione-locale-secure-key-12345"
OFFLINE_MODE = False
DEBUG = True
"""
    
    # Crea directory di instance se non esiste
    Path("instance").mkdir(exist_ok=True)
    
    # Scrivi il file di configurazione
    with open("instance/config.py", "w") as f:
        f.write(config_content)
    
    logger.info("✓ File di configurazione creato: instance/config.py")
    return True

def crea_script_avvio():
    """Crea script di avvio appropriati per diversi sistemi operativi"""
    logger.info("Creazione script di avvio...")
    
    # Per Windows (batch file)
    if platform.system() == "Windows":
        with open("avvia_app.bat", "w") as f:
            f.write('@echo off\n')
            f.write('echo Avvio CryptoTradeAnalyzer...\n')
            f.write('python main.py\n')
            f.write('pause\n')
        logger.info("✓ Script di avvio creato: avvia_app.bat")
    
    # Per Linux/Mac (shell script)
    else:
        with open("avvia_app.sh", "w") as f:
            f.write('#!/bin/bash\n')
            f.write('echo "Avvio CryptoTradeAnalyzer..."\n')
            f.write('python main.py\n')
        os.chmod("avvia_app.sh", 0o755)  # Rendi eseguibile
        logger.info("✓ Script di avvio creato: avvia_app.sh")
    
    return True

def inizializza_database():
    """Inizializza il database SQLite locale"""
    logger.info("Inizializzazione database...")
    
    # Crea la directory instance se non esiste
    Path("instance").mkdir(exist_ok=True)
    
    # Esegui la creazione del database
    try:
        from main import app
        from database import db
        with app.app_context():
            db.create_all()
        logger.info("✓ Database inizializzato con successo")
        return True
    except Exception as e:
        logger.error(f"Errore durante l'inizializzazione del database: {str(e)}")
        return False

def main():
    """Funzione principale per la configurazione dell'ambiente locale"""
    logger.info("Benvenuto nel processo di configurazione di CryptoTradeAnalyzer!")
    logger.info("Verifica che i requisiti di sistema siano soddisfatti...")

    # Controlla i requisiti di sistema
    if not controllo_sistema():
        logger.error("Il sistema non soddisfa i requisiti minimi.")
        logger.info("Per favore aggiorna Python o installa le dipendenze mancanti.")
        return False
    
    # Crea file di configurazione
    if not crea_file_configurazione():
        logger.error("Errore durante la creazione dei file di configurazione.")
        return False
    
    # Crea script di avvio
    if not crea_script_avvio():
        logger.error("Errore durante la creazione degli script di avvio.")
        return False
    
    # Inizializza il database
    if not inizializza_database():
        logger.error("Errore durante l'inizializzazione del database.")
        return False
    
    # Tutto completato con successo
    logger.info("\n✓ Configurazione completata con successo!")
    logger.info("Per avviare l'applicazione:")
    if platform.system() == "Windows":
        logger.info("  Esegui avvia_app.bat oppure 'python main.py'")
    else:
        logger.info("  Esegui ./avvia_app.sh oppure 'python main.py'")
    logger.info("L'applicazione sarà disponibile su http://localhost:5000\n")
    
    return True

if __name__ == "__main__":
    main()