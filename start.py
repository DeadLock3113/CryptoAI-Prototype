import os
import sys
import logging
from simple_app import app

# Configura il logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

# Verifica se siamo in un ambiente PyInstaller
if getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS'):
    # Siamo nell'eseguibile frozen
    application_path = os.path.dirname(sys.executable)
    os.chdir(application_path)
    logging.info(f"Avvio in modalità eseguibile. Directory: {application_path}")
    # In modalità eseguibile è meglio disabilitare il debug
    debug_mode = False
else:
    # Siamo in modalità sviluppo
    debug_mode = True
    logging.info("Avvio in modalità sviluppo")

if __name__ == '__main__':
    logging.info(f"Avvio del server su 0.0.0.0:5000 (debug: {debug_mode})")
    app.run(host='0.0.0.0', port=5000, debug=debug_mode, use_reloader=debug_mode)