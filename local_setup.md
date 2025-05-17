# Guida all'Installazione Locale di CryptoTradeAnalyzer

Questa guida ti mostrerà come installare e configurare CryptoTradeAnalyzer sul tuo computer personale.

## Requisiti di Sistema

- Python 3.9 o superiore
- PostgreSQL (opzionale, puoi usare SQLite per iniziare)
- Spazio su disco: almeno 500MB per il software e le dipendenze
- RAM: almeno 4GB (consigliato 8GB o più per analisi su dataset più grandi)
- GPU: opzionale, ma consigliata per modelli di machine learning (supporto sia per NVIDIA che AMD)

## Procedura di Installazione

### 1. Clona il repository o scarica i file

```bash
git clone [URL_DEL_REPOSITORY] cryptotradeanalyzer
cd cryptotradeanalyzer
```

Oppure scarica i file direttamente da Replit e salva in una nuova cartella.

### 2. Crea un ambiente virtuale (consigliato)

```bash
python -m venv venv
```

Attiva l'ambiente virtuale:

- **Windows**:
  ```bash
  venv\Scripts\activate
  ```

- **Linux/Mac**:
  ```bash
  source venv/bin/activate
  ```

### 3. Installa le dipendenze

```bash
pip install -r requirements.txt
```

### 4. Configura il database

#### Opzione 1: Usa SQLite (più semplice)

Modifica il file `main.py` e cambia la configurazione del database:

```python
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///cryptotradeanalyzer.db"
```

#### Opzione 2: Usa PostgreSQL (prestazioni migliori)

```python
app.config["SQLALCHEMY_DATABASE_URI"] = "postgresql://username:password@localhost/cryptotradedb"
```

### 5. Inizializza il database

```bash
python -c "from main import app; from database import db; with app.app_context(): db.create_all()"
```

### 6. Avvia l'applicazione

```bash
python main.py
```

Per avviare con gunicorn (raccomandato per produzione):

```bash
gunicorn --bind 0.0.0.0:5000 main:app
```

L'applicazione sarà accessibile all'indirizzo: `http://localhost:5000`

## Modalità Offline

CryptoTradeAnalyzer può funzionare completamente offline. In questo caso:

1. I prezzi delle criptovalute saranno quelli predefiniti invece di quelli in tempo reale
2. Le API degli exchange (Binance, Kraken) non saranno disponibili
3. Le notifiche Telegram non funzioneranno

Puoi comunque:
- Caricare e analizzare i tuoi dataset esistenti
- Usare tutti gli indicatori tecnici
- Addestrare e utilizzare i modelli di machine learning
- Eseguire backtest sulle strategie

## Requisiti Hardware per Calcolo GPU

### AMD
Per utilizzare la tua GPU AMD:
- Driver AMD aggiornati
- ROCm installato (per PyTorch)

### NVIDIA
Per utilizzare la tua GPU NVIDIA:
- Driver NVIDIA aggiornati
- CUDA Toolkit 11.3 o superiore

Il software rileverà automaticamente la GPU disponibile.

## Note Aggiuntive

- Per il supporto completo delle funzionalità, è consigliata una connessione internet
- I dati sensibili (API keys, token) vengono salvati nel database locale
- Per backup regolari, consigliamo di esportare periodicamente il database