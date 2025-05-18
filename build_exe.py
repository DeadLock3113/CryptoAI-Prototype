"""
CryptoTradeAnalyzer - Script per la creazione dell'eseguibile Windows

Questo script configura e utilizza PyInstaller per creare un eseguibile Windows
dell'applicazione CryptoTradeAnalyzer. L'eseguibile avrà tutte le dipendenze
incluse per funzionare su qualsiasi sistema Windows senza necessità di installare
Python o altre librerie.

Utilizzo:
    python build_exe.py

Autore: CryptoTradeAnalyzer Team
"""

import os
import sys
import shutil
import subprocess
from pathlib import Path


def main():
    """Funzione principale per la creazione dell'eseguibile"""
    print("=== CryptoTradeAnalyzer - Creazione Eseguibile Windows ===")
    
    # Verifica se PyInstaller è installato
    try:
        import PyInstaller
        print("PyInstaller trovato:", PyInstaller.__version__)
    except ImportError:
        print("PyInstaller non trovato. Installazione in corso...")
        subprocess.check_call([sys.executable, "-m", "pip", "install", "pyinstaller"])
        print("PyInstaller installato correttamente.")
    
    # Directory del progetto
    project_dir = Path(__file__).parent.absolute()
    print(f"Directory del progetto: {project_dir}")
    
    # File principale dell'applicazione
    main_file = project_dir / "start.py"
    if not main_file.exists():
        print(f"ERRORE: Il file {main_file} non esiste.")
        return False
    
    # Directory per i file temporanei e l'output
    build_dir = project_dir / "build"
    dist_dir = project_dir / "dist"
    
    # Pulisci le directory precedenti se esistono
    for directory in [build_dir, dist_dir]:
        if directory.exists():
            print(f"Pulizia della directory {directory}...")
            shutil.rmtree(directory)
    
    # Crea il file di specifiche per PyInstaller
    spec_file = """# -*- mode: python ; coding: utf-8 -*-

block_cipher = None

added_files = [
    ('web/templates', 'web/templates'),
    ('web/static', 'web/static'),
    ('models', 'models'),
    ('strategies', 'strategies'),
    ('indicators', 'indicators'),
    ('utils', 'utils'),
    ('data', 'data'),
    ('backtesting', 'backtesting')
]

a = Analysis(
    ['start.py'],
    pathex=[],
    binaries=[],
    datas=added_files,
    hiddenimports=[
        'matplotlib.backends.backend_tkagg',
        'sklearn.utils._cython_blas',
        'sklearn.neighbors.typedefs',
        'sklearn.neighbors.quad_tree',
        'sklearn.tree._utils',
        'sqlalchemy.sql.default_comparator',
        'pandas._libs.tslibs.timedeltas',
        'pandas._libs.tslibs.nattype',
        'pandas._libs.tslibs.np_datetime',
        'nltk',
        'gensim',
        'torch',
        'torchvision',
        'tensorflow',
        'flask_sqlalchemy',
        'flask_login',
        'flask_wtf',
        'email_validator',
        'gunicorn.workers',
        'trading_calendar',
        'backtrader',
        'trafilatura',
        'textblob',
        'vadersentiment',
        'vadersentiment.vaderSentiment',
        'werkzeug.security',
        'training_handler',
        'sqlalchemy.dialects.postgresql'
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name='CryptoTradeAnalyzer',
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=True,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon='web/static/images/logo.ico',
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name='CryptoTradeAnalyzer',
)
"""
    
    # Scrivi il file delle specifiche
    spec_path = project_dir / "CryptoTradeAnalyzer.spec"
    with open(spec_path, "w") as f:
        f.write(spec_file)
    print(f"File di specifiche creato: {spec_path}")
    
    # Esegui PyInstaller
    print("\nAvvio di PyInstaller...")
    cmd = [
        sys.executable, 
        "-m", 
        "PyInstaller", 
        "--clean",
        "CryptoTradeAnalyzer.spec"
    ]
    
    try:
        subprocess.check_call(cmd)
        print("\nCompilazione completata con successo!")
        
        # Percorso dell'eseguibile generato
        exe_path = dist_dir / "CryptoTradeAnalyzer" / "CryptoTradeAnalyzer.exe"
        print(f"\nEseguibile creato: {exe_path}")
        
        # Crea file README con istruzioni
        readme = """
CryptoTradeAnalyzer - Versione Eseguibile Windows
=================================================

Questa è la versione compilata di CryptoTradeAnalyzer per Windows.
Non è necessario installare Python o altre librerie.

ISTRUZIONI:
1. Fare doppio clic su 'CryptoTradeAnalyzer.exe' per avviare l'applicazione
2. Al primo avvio, verrà creato automaticamente un database SQLite
3. Utilizzare l'applicazione normalmente attraverso l'interfaccia web
4. Per chiudere l'applicazione, chiudere la finestra del terminale

NOTA IMPORTANTE:
* È possibile che all'avvio Windows mostri un avviso di sicurezza.
  Questo è normale per applicazioni non firmate. Selezionare "Esegui comunque".
* Se si utilizza un database PostgreSQL esterno, inserire i dettagli di connessione 
  nelle impostazioni dell'applicazione.
* Per motivi di sicurezza, è consigliabile utilizzare l'applicazione solo in 
  ambiente locale o in una rete protetta.

Copyright © CryptoTradeAnalyzer Team
"""
        
        readme_path = dist_dir / "CryptoTradeAnalyzer" / "README.txt"
        with open(readme_path, "w") as f:
            f.write(readme)
        
        print(f"File README creato: {readme_path}")
        print("\nOperazione completata! L'eseguibile è pronto per la distribuzione.")
        
        return True
    
    except subprocess.CalledProcessError as e:
        print(f"ERRORE durante l'esecuzione di PyInstaller: {e}")
        return False


if __name__ == "__main__":
    main()