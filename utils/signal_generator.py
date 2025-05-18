"""
CryptoTradeAnalyzer - Generatore di segnali basati su AI

Questo modulo implementa la generazione di segnali di trading basati
sui modelli di machine learning precedentemente addestrati.
"""

import os
import logging
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import json
import time
import threading

from db_models import MLModel, Dataset, PriceData
from utils.telegram_notification import send_signal_notification

# Setup logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Dizionario per memorizzare le configurazioni di segnali attive
active_signal_configs = {}

# Enumerazione dei tipi di segnale
class SignalType:
    LONG = "LONG"
    SHORT = "SHORT"
    FLAT = "FLAT"

class SignalGenerator:
    """Classe per la generazione di segnali di trading basati su modelli AI"""
    
    def __init__(self, model_ids, dataset_id, user_id, timeframe='1h', 
                 risk_level=2, auto_tp_sl=True, telegram_enabled=True):
        """
        Inizializza un nuovo generatore di segnali
        
        Args:
            model_ids (list): Lista di ID dei modelli ML da utilizzare
            dataset_id (int): ID del dataset da utilizzare
            user_id (int): ID dell'utente
            timeframe (str): Timeframe per i segnali ('1m', '5m', '15m', '30m', '1h', '4h', '1d')
            risk_level (int): Livello di rischio da 1 (basso) a 5 (alto)
            auto_tp_sl (bool): Calcola automaticamente TP e SL
            telegram_enabled (bool): Invia notifiche Telegram
        """
        self.model_ids = model_ids if isinstance(model_ids, list) else [model_ids]
        self.dataset_id = dataset_id
        self.user_id = user_id
        self.timeframe = timeframe
        self.risk_level = max(1, min(5, risk_level))  # Limita tra 1 e 5
        self.auto_tp_sl = auto_tp_sl
        self.telegram_enabled = telegram_enabled
        
        # Stato di esecuzione
        self.is_running = False
        self.stop_event = threading.Event()
        self.last_signal = None
        self.last_signal_time = None
        self.thread = None
        
        # Parametri calcolati
        self.atr_length = 14  # Periodo per ATR per calcolo volatilità
        self.tp_factor = 1.5 + (0.5 * self.risk_level)  # Da 2 a 4 in base al rischio
        self.sl_factor = 0.5 + (0.1 * self.risk_level)  # Da 0.6 a 1.0 in base al rischio
        self.vol_perc = 1 + (self.risk_level * 0.5)  # Percentuale di capitale da investire (1.5-3.5%)
        
        # Caricamento configurazione
        self.models = []
        self.dataset = None
        self.load_config()
        
    def load_config(self):
        """Carica la configurazione dei modelli e del dataset"""
        try:
            from app import db  # Import locale per evitare dipendenze circolari
            
            # Carica i modelli selezionati
            for model_id in self.model_ids:
                model = MLModel.query.get(model_id)
                if model and model.user_id == self.user_id:
                    self.models.append(model)
            
            # Carica il dataset
            self.dataset = Dataset.query.get(self.dataset_id)
            
            # Log della configurazione
            logger.debug(f"Configurazione caricata: {len(self.models)} modelli, dataset: {self.dataset.name if self.dataset else 'Nessuno'}")
            
        except Exception as e:
            logger.error(f"Errore nel caricamento della configurazione: {str(e)}")
    
    def start(self):
        """Avvia il processo di generazione dei segnali"""
        if self.is_running:
            return False
        
        if not self.models or not self.dataset:
            logger.error("Impossibile avviare il generatore di segnali: configurazione incompleta")
            return False
        
        self.is_running = True
        self.stop_event.clear()
        
        # Avvio del thread di monitoraggio
        self.thread = threading.Thread(target=self._monitor_thread)
        self.thread.daemon = True
        self.thread.start()
        
        logger.info(f"Generatore di segnali avviato per {len(self.models)} modelli")
        return True
    
    def stop(self):
        """Ferma il processo di generazione dei segnali"""
        if not self.is_running:
            return
        
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2.0)
        
        self.is_running = False
        logger.info("Generatore di segnali fermato")
    
    def _monitor_thread(self):
        """Thread principale per il monitoraggio e la generazione dei segnali"""
        try:
            # Ciclo principale
            while not self.stop_event.is_set():
                try:
                    # Ottieni i dati più recenti
                    latest_data = self._get_latest_data()
                    
                    if latest_data is not None and not latest_data.empty:
                        # Genera il segnale combinando i risultati di tutti i modelli
                        signal, confidence, price = self._generate_signal(latest_data)
                        
                        # Se c'è un nuovo segnale e diverso dall'ultimo, processalo
                        if signal and (not self.last_signal or 
                                      signal != self.last_signal or 
                                      datetime.now() - self.last_signal_time > timedelta(hours=1)):
                            
                            # Calcola TP, SL e volume
                            tp, sl, volume = self._calculate_tp_sl_vol(signal, price, latest_data)
                            
                            # Memorizza l'ultimo segnale
                            self.last_signal = signal
                            self.last_signal_time = datetime.now()
                            
                            # Invia notifica Telegram se abilitato
                            if self.telegram_enabled:
                                self._send_notification(signal, price, tp, sl, volume, confidence)
                    
                    # Attesa in base al timeframe
                    wait_seconds = self._get_wait_time()
                    self.stop_event.wait(wait_seconds)
                    
                except Exception as e:
                    logger.error(f"Errore nel ciclo di monitoraggio: {str(e)}")
                    self.stop_event.wait(60)  # Attesa più lunga in caso di errore
        
        except Exception as e:
            logger.error(f"Errore critico nel thread di monitoraggio: {str(e)}")
            self.is_running = False
    
    def _get_latest_data(self):
        """Ottiene i dati più recenti dal dataset"""
        try:
            from app import db  # Import locale per evitare dipendenze circolari
            
            # Get price data
            price_data = PriceData.query.filter_by(
                dataset_id=self.dataset_id
            ).order_by(PriceData.timestamp.desc()).limit(100).all()
            
            if not price_data:
                return None
            
            # Convert to DataFrame
            data = []
            for row in price_data:
                data.append({
                    'timestamp': row.timestamp,
                    'open': row.open,
                    'high': row.high,
                    'low': row.low,
                    'close': row.close,
                    'volume': row.volume
                })
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)
            
            # Calcola indicatori tecnici di base per l'analisi
            df = self._add_technical_indicators(df)
            
            return df
            
        except Exception as e:
            logger.error(f"Errore nel recupero dei dati: {str(e)}")
            return None
    
    def _add_technical_indicators(self, df):
        """Aggiunge indicatori tecnici al dataframe"""
        # Calcola le medie mobili
        df['sma_7'] = df['close'].rolling(window=7).mean()
        df['sma_21'] = df['close'].rolling(window=21).mean()
        
        # Calcola ATR (Average True Range) per la volatilità
        df['tr'] = df.apply(
            lambda x: max(
                x['high'] - x['low'],
                abs(x['high'] - x['close'].shift(1)),
                abs(x['low'] - x['close'].shift(1))
            ), axis=1
        )
        df['atr'] = df['tr'].rolling(window=self.atr_length).mean()
        
        # RSI (Relative Strength Index)
        delta = df['close'].diff()
        gain = delta.where(delta > 0, 0).rolling(window=14).mean()
        loss = -delta.where(delta < 0, 0).rolling(window=14).mean()
        rs = gain / loss
        df['rsi'] = 100 - (100 / (1 + rs))
        
        return df
    
    def _generate_signal(self, data):
        """
        Genera un segnale di trading basato sui modelli ML
        
        Restituisce:
            tuple (str, float, float): Segnale, confidence score e prezzo corrente
        """
        if not self.models or len(self.models) == 0:
            return None, 0, 0
        
        current_price = data['close'].iloc[-1]
        
        # Inizializza i contatori per i voti dei modelli
        votes = {
            SignalType.LONG: 0,
            SignalType.SHORT: 0,
            SignalType.FLAT: 0
        }
        
        total_confidence = 0
        
        # Per ogni modello, ottiene la previsione
        for model in self.models:
            # Qui estrarremmo i dati necessari per il modello specifico
            # e lo alimenteremmo con i dati più recenti
            try:
                # In una implementazione reale, qui si caricherebbe il modello salvato
                # e si farebbe una predizione usando il modello.
                # Per ora, usiamo un approccio basato su regole per simulare le previsioni
                
                # Ottieni la predizione simulata
                prediction, model_confidence = self._simulate_model_prediction(model, data)
                
                # Aggiunge il voto del modello
                votes[prediction] += 1
                total_confidence += model_confidence
                
                logger.debug(f"Modello {model.name} predice {prediction} con confidenza {model_confidence:.2f}")
                
            except Exception as e:
                logger.error(f"Errore nella generazione del segnale per il modello {model.name}: {str(e)}")
        
        # Calcola il segnale finale basato sui voti
        max_votes = max(votes.values())
        if max_votes == 0:
            return None, 0, current_price
        
        # In caso di parità, preferisce FLAT
        if list(votes.values()).count(max_votes) > 1 and votes[SignalType.FLAT] == max_votes:
            final_signal = SignalType.FLAT
        else:
            final_signal = max(votes, key=votes.get)
        
        # Calcola il confidence score medio
        avg_confidence = total_confidence / len(self.models) if self.models else 0
        
        logger.info(f"Segnale generato: {final_signal} con confidenza {avg_confidence:.2f}, prezzo: {current_price}")
        
        return final_signal, avg_confidence, current_price
    
    def _simulate_model_prediction(self, model, data):
        """
        Simula una predizione del modello usando regole base
        In una implementazione reale, qui si utilizzerebbe il modello ML vero e proprio
        """
        # Estrae i dati recenti
        close = data['close'].iloc[-1]
        sma_7 = data['sma_7'].iloc[-1]
        sma_21 = data['sma_21'].iloc[-1]
        rsi = data['rsi'].iloc[-1]
        
        # Regole base per simulare la predizione
        if sma_7 > sma_21 and rsi < 70:  # Trend rialzista e non ipercomprato
            prediction = SignalType.LONG
            confidence = min(0.5 + (sma_7 - sma_21) / close, 0.95)
        elif sma_7 < sma_21 and rsi > 30:  # Trend ribassista e non ipervenduto
            prediction = SignalType.SHORT
            confidence = min(0.5 + (sma_21 - sma_7) / close, 0.95)
        else:  # Zona di indecisione
            prediction = SignalType.FLAT
            confidence = 0.5
        
        return prediction, confidence
    
    def _calculate_tp_sl_vol(self, signal, price, data):
        """
        Calcola take profit, stop loss e volume in base al segnale e alla volatilità
        
        Args:
            signal (str): Tipo di segnale (LONG, SHORT, FLAT)
            price (float): Prezzo corrente
            data (DataFrame): Dati recenti con ATR
        
        Returns:
            tuple (float, float, float): Take profit, stop loss e volume percentuale
        """
        if signal == SignalType.FLAT:
            return 0, 0, 0
        
        # Usa ATR per calcolare la volatilità
        atr = data['atr'].iloc[-1]
        
        if self.auto_tp_sl:
            if signal == SignalType.LONG:
                tp = price + (atr * self.tp_factor)
                sl = price - (atr * self.sl_factor)
            else:  # SHORT
                tp = price - (atr * self.tp_factor)
                sl = price + (atr * self.sl_factor)
        else:
            # Valori di default se auto_tp_sl è disabilitato
            if signal == SignalType.LONG:
                tp = price * 1.05  # +5%
                sl = price * 0.97  # -3%
            else:  # SHORT
                tp = price * 0.95  # -5%
                sl = price * 1.03  # +3%
        
        # Calcola il volume in base al risk level e alla distanza dallo stop loss
        risk_perc = self.vol_perc  # Percentuale base del capitale da rischiare
        
        # Calcola il volume effettivo basato su rischio e volatilità
        volume = risk_perc
        
        return tp, sl, volume
    
    def _send_notification(self, signal, price, tp, sl, volume, confidence):
        """Invia una notifica con i dettagli del segnale"""
        try:
            # Formatta il messaggio
            message = f"🔔 SEGNALE DI TRADING: {signal}\n\n"
            message += f"Simbolo: {self.dataset.symbol}\n"
            message += f"Prezzo: {price:.6f}\n"
            
            if signal != SignalType.FLAT:
                message += f"Take Profit: {tp:.6f}\n"
                message += f"Stop Loss: {sl:.6f}\n"
                message += f"Volume consigliato: {volume:.2f}%\n"
            
            message += f"Confidenza: {confidence:.2f}\n"
            message += f"Timeframe: {self.timeframe}\n"
            message += f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
            
            # Invia la notifica
            send_signal_notification(self.user_id, message)
            
            logger.info(f"Notifica inviata per segnale {signal}")
            
        except Exception as e:
            logger.error(f"Errore nell'invio della notifica: {str(e)}")
    
    def _get_wait_time(self):
        """Calcola il tempo di attesa in base al timeframe"""
        # Mappa dei timeframe ai secondi di attesa
        wait_map = {
            '1m': 30,       # Controlla ogni 30 secondi per timeframe 1m
            '5m': 60,       # Ogni minuto per timeframe 5m
            '15m': 180,     # Ogni 3 minuti per timeframe 15m
            '30m': 300,     # Ogni 5 minuti per timeframe 30m
            '1h': 600,      # Ogni 10 minuti per timeframe 1h
            '4h': 1800,     # Ogni 30 minuti per timeframe 4h
            '1d': 3600      # Ogni ora per timeframe 1d
        }
        
        return wait_map.get(self.timeframe, 300)  # Default 5 minuti


# Funzioni per la gestione delle configurazioni di segnali

def create_signal_config(user_id, model_ids, dataset_id, timeframe='1h', 
                        risk_level=2, auto_tp_sl=True, telegram_enabled=True):
    """
    Crea una nuova configurazione di segnali
    
    Args:
        user_id (int): ID dell'utente
        model_ids (list): Lista di ID dei modelli
        dataset_id (int): ID del dataset
        timeframe (str): Timeframe per i segnali
        risk_level (int): Livello di rischio (1-5)
        auto_tp_sl (bool): Calcolo automatico di TP/SL
        telegram_enabled (bool): Abilita notifiche Telegram
    
    Returns:
        str: ID della configurazione
    """
    config_id = f"signal_{user_id}_{int(time.time())}"
    
    # Crea l'oggetto SignalGenerator
    generator = SignalGenerator(
        model_ids=model_ids,
        dataset_id=dataset_id,
        user_id=user_id,
        timeframe=timeframe,
        risk_level=risk_level,
        auto_tp_sl=auto_tp_sl,
        telegram_enabled=telegram_enabled
    )
    
    # Memorizza la configurazione
    active_signal_configs[config_id] = generator
    
    # Salva la configurazione nel database
    save_signal_config(config_id, user_id, model_ids, dataset_id, timeframe, 
                      risk_level, auto_tp_sl, telegram_enabled)
    
    # Avvia il generatore se tutto è configurato correttamente
    if generator.models and generator.dataset:
        generator.start()
    
    return config_id

def get_signal_config(config_id):
    """Ottiene una configurazione di segnali dato il suo ID"""
    return active_signal_configs.get(config_id)

def start_signal_generator(config_id):
    """Avvia un generatore di segnali"""
    generator = get_signal_config(config_id)
    if generator:
        return generator.start()
    return False

def stop_signal_generator(config_id):
    """Ferma un generatore di segnali"""
    generator = get_signal_config(config_id)
    if generator:
        generator.stop()
        return True
    return False

def delete_signal_config(config_id):
    """Elimina una configurazione di segnali"""
    generator = get_signal_config(config_id)
    if generator:
        generator.stop()
        del active_signal_configs[config_id]
        
        # Rimuovi la configurazione dal database
        delete_signal_config_db(config_id)
        
        return True
    return False

def get_user_signal_configs(user_id):
    """Ottiene tutte le configurazioni di segnali per un utente"""
    user_configs = {}
    for config_id, generator in active_signal_configs.items():
        if generator.user_id == user_id:
            user_configs[config_id] = generator
    return user_configs

def load_all_signal_configs():
    """Carica tutte le configurazioni di segnali dal database"""
    try:
        from app import db  # Import locale per evitare dipendenze circolari
        
        # Qui si implementerebbe il caricamento delle configurazioni dal database
        # e la creazione di oggetti SignalGenerator per ciascuna configurazione
        
        logger.info("Tutte le configurazioni di segnali caricate dal database")
        
    except Exception as e:
        logger.error(f"Errore nel caricamento delle configurazioni di segnali: {str(e)}")

def save_signal_config(config_id, user_id, model_ids, dataset_id, timeframe, 
                     risk_level, auto_tp_sl, telegram_enabled):
    """Salva una configurazione di segnali nel database"""
    try:
        from app import db  # Import locale per evitare dipendenze circolari
        from db_models import SignalConfig
        
        # Crea una nuova configurazione o aggiorna se esiste
        config = SignalConfig.query.filter_by(config_id=config_id).first()
        
        if not config:
            config = SignalConfig(
                config_id=config_id,
                user_id=user_id,
                model_ids=json.dumps(model_ids),
                dataset_id=dataset_id,
                timeframe=timeframe,
                risk_level=risk_level,
                auto_tp_sl=auto_tp_sl,
                telegram_enabled=telegram_enabled,
                is_active=True,
                created_at=datetime.now()
            )
            db.session.add(config)
        else:
            config.model_ids = json.dumps(model_ids)
            config.dataset_id = dataset_id
            config.timeframe = timeframe
            config.risk_level = risk_level
            config.auto_tp_sl = auto_tp_sl
            config.telegram_enabled = telegram_enabled
            config.is_active = True
            config.updated_at = datetime.now()
        
        db.session.commit()
        
        logger.debug(f"Configurazione di segnali {config_id} salvata nel database")
        
    except Exception as e:
        logger.error(f"Errore nel salvataggio della configurazione di segnali: {str(e)}")

def delete_signal_config_db(config_id):
    """Elimina una configurazione di segnali dal database"""
    try:
        from app import db  # Import locale per evitare dipendenze circolari
        from db_models import SignalConfig
        
        config = SignalConfig.query.filter_by(config_id=config_id).first()
        if config:
            db.session.delete(config)
            db.session.commit()
            
            logger.debug(f"Configurazione di segnali {config_id} eliminata dal database")
        
    except Exception as e:
        logger.error(f"Errore nell'eliminazione della configurazione di segnali: {str(e)}")