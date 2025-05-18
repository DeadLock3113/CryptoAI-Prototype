"""
CryptoTradeAnalyzer - Generatore di segnali basati su AI

Questo modulo implementa la generazione di segnali di trading basati
sui modelli di machine learning precedentemente addestrati.
"""

import json
import logging
import random
import threading
import time
import uuid
from datetime import datetime, timedelta
import os

import numpy as np
import pandas as pd
from sqlalchemy import create_engine, text
from flask import current_app
from sqlalchemy.orm import Session
from sqlalchemy.ext.declarative import declarative_base

# Import dei moduli interni
from database import db
from db_models import User, MLModel, Dataset, PriceData, SignalConfig
from utils.technical_indicators import add_indicators
from utils.telegram_notification import send_telegram_message, send_signal_notification

# Configurazione logging
logger = logging.getLogger(__name__)

# Dizionario per tenere traccia dei generatori di segnali attivi
active_generators = {}

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
        self.model_ids = model_ids if isinstance(model_ids, list) else json.loads(model_ids)
        self.dataset_id = dataset_id
        self.user_id = user_id
        self.timeframe = timeframe
        self.risk_level = min(max(risk_level, 1), 5)  # Assicura che sia tra 1 e 5
        self.auto_tp_sl = auto_tp_sl
        self.telegram_enabled = telegram_enabled
        
        # Variabili interne
        self.models = []
        self.dataset = None
        self.user = None
        self.running = False
        self.monitor_thread = None
        self.last_signal = None
        self.last_signal_time = None
        
        # Carica la configurazione
        self.load_config()

    def load_config(self):
        """Carica la configurazione dei modelli e del dataset"""
        try:
            # Ottieni i modelli ML
            self.models = MLModel.query.filter(MLModel.id.in_(self.model_ids)).all()
            if not self.models:
                logger.error(f"Nessun modello trovato per gli ID: {self.model_ids}")
                return False
                
            # Ottieni il dataset
            self.dataset = Dataset.query.get(self.dataset_id)
            if not self.dataset:
                logger.error(f"Dataset non trovato per ID: {self.dataset_id}")
                return False
                
            # Ottieni l'utente
            self.user = User.query.get(self.user_id)
            if not self.user:
                logger.error(f"Utente non trovato per ID: {self.user_id}")
                return False
            
            logger.info(f"Configurazione caricata: {len(self.models)} modelli, dataset {self.dataset.name}")
            return True
            
        except Exception as e:
            logger.error(f"Errore nel caricamento della configurazione: {str(e)}")
            return False

    def start(self):
        """Avvia il processo di generazione dei segnali"""
        if self.running:
            logger.info("Generatore di segnali già in esecuzione")
            return False
            
        if not self.user or not self.dataset or not self.models:
            logger.error("Configurazione incompleta, impossibile avviare")
            return False
            
        try:
            self.running = True
            self.monitor_thread = threading.Thread(target=self._monitor_thread)
            self.monitor_thread.daemon = True
            self.monitor_thread.start()
            
            logger.info(f"Generatore di segnali avviato per dataset {self.dataset.name}")
            return True
            
        except Exception as e:
            logger.error(f"Errore nell'avvio del generatore di segnali: {str(e)}")
            self.running = False
            return False

    def stop(self):
        """Ferma il processo di generazione dei segnali"""
        self.running = False
        logger.info("Generatore di segnali fermato")
        return True

    def _monitor_thread(self):
        """Thread principale per il monitoraggio e la generazione dei segnali"""
        logger.info(f"Thread di monitoraggio avviato per dataset {self.dataset.name}")
        
        wait_time = self._get_wait_time()
        
        while self.running:
            try:
                # Ottieni gli ultimi dati
                data = self._get_latest_data()
                if data is not None and not data.empty:
                    # Aggiungi indicatori tecnici
                    data = self._add_technical_indicators(data)
                    
                    # Genera un segnale
                    signal, confidence, price = self._generate_signal(data)
                    
                    # Se c'è un segnale e non è uguale all'ultimo inviato
                    if signal and (self.last_signal != signal or 
                                  (self.last_signal_time and 
                                   datetime.now() - self.last_signal_time > timedelta(hours=4))):
                        
                        # Calcola TP, SL e volume
                        if signal != SignalType.FLAT and self.auto_tp_sl:
                            tp, sl, volume = self._calculate_tp_sl_vol(signal, price, data)
                        else:
                            tp, sl, volume = None, None, None
                        
                        # Invia la notifica
                        if self.telegram_enabled:
                            self._send_notification(signal, price, tp, sl, volume, confidence)
                        
                        # Aggiorna l'ultimo segnale
                        self.last_signal = signal
                        self.last_signal_time = datetime.now()
                        
                        logger.info(f"Segnale generato: {signal} a {price} con confidenza {confidence:.2f}")
            
            except Exception as e:
                logger.error(f"Errore nel thread di monitoraggio: {str(e)}")
            
            # Attendi prima del prossimo controllo
            time.sleep(wait_time)
        
        logger.info("Thread di monitoraggio terminato")

    def _get_latest_data(self):
        """Ottiene i dati più recenti dal dataset"""
        try:
            # Ottieni gli ultimi 100 record dal dataset
            price_data = PriceData.query.filter_by(dataset_id=self.dataset_id) \
                                     .order_by(PriceData.timestamp.desc()) \
                                     .limit(100).all()
            
            if not price_data:
                logger.warning(f"Nessun dato trovato nel dataset {self.dataset.name}")
                return None
                
            # Converti in DataFrame
            data = []
            for record in price_data:
                data.append({
                    'timestamp': record.timestamp,
                    'open': record.open,
                    'high': record.high,
                    'low': record.low,
                    'close': record.close,
                    'volume': record.volume
                })
            
            df = pd.DataFrame(data)
            df.set_index('timestamp', inplace=True)
            df.sort_index(inplace=True)  # Assicuriamoci che i dati siano in ordine cronologico
            
            return df
            
        except Exception as e:
            logger.error(f"Errore nell'ottenere i dati più recenti: {str(e)}")
            return None

    def _add_technical_indicators(self, df):
        """Aggiunge indicatori tecnici al dataframe"""
        try:
            # Indicatori comuni per la generazione di segnali
            # Utilizziamo la funzione di utilità per aggiungere gli indicatori
            df = add_indicators(df, {
                'sma': [20, 50, 200],
                'ema': [9, 21],
                'rsi': [14],
                'macd': {'fast': 12, 'slow': 26, 'signal': 9},
                'bollinger': {'window': 20, 'std': 2},
                'atr': [14]
            })
            
            return df
            
        except Exception as e:
            logger.error(f"Errore nell'aggiunta degli indicatori tecnici: {str(e)}")
            return df

    def _generate_signal(self, data):
        """
        Genera un segnale di trading basato sui modelli ML
        
        Restituisce:
            tuple (str, float, float): Segnale, confidence score e prezzo corrente
        """
        try:
            # Prendi l'ultimo prezzo
            price = data['close'].iloc[-1]
            
            # Pesi per la confidenza in base al rischio
            risk_weights = {
                1: 0.9,  # Conservativo, alta confidenza richiesta
                2: 0.8,
                3: 0.7,  # Moderato
                4: 0.6,
                5: 0.5   # Aggressivo, bassa confidenza sufficiente
            }
            
            # Threshold base per la confidenza in base al livello di rischio
            confidence_threshold = risk_weights.get(self.risk_level, 0.7)
            
            # Accumula predizioni dai modelli
            signals = []
            confidences = []
            
            for model in self.models:
                # Nella versione reale, qui caricheremmo e utilizzeremmo il modello salvato
                # Per ora simuliamo la predizione basata sul tipo di modello
                signal, conf = self._simulate_model_prediction(model, data)
                signals.append(signal)
                confidences.append(conf)
            
            if not signals:
                return SignalType.FLAT, 0.0, price
            
            # Calcola il segnale prevalente
            signal_counts = {
                SignalType.LONG: signals.count(SignalType.LONG),
                SignalType.SHORT: signals.count(SignalType.SHORT),
                SignalType.FLAT: signals.count(SignalType.FLAT)
            }
            
            # Trova il segnale con più occorrenze
            final_signal = max(signal_counts.items(), key=lambda x: x[1])[0]
            
            # Calcola la confidenza media per il segnale finale
            signal_indices = [i for i, s in enumerate(signals) if s == final_signal]
            signal_confidences = [confidences[i] for i in signal_indices]
            avg_confidence = sum(signal_confidences) / len(signal_confidences) if signal_confidences else 0
            
            # Applica una penalità se i modelli sono in disaccordo
            max_count = max(signal_counts.values())
            agreement_ratio = max_count / len(signals)
            
            # Penalizza la confidenza in base al disaccordo
            final_confidence = avg_confidence * agreement_ratio
            
            # Se la confidenza è sotto la soglia, rimani neutrale (FLAT)
            if final_signal != SignalType.FLAT and final_confidence < confidence_threshold:
                logger.info(f"Confidenza {final_confidence:.2f} sotto la soglia {confidence_threshold}, segnale FLAT")
                return SignalType.FLAT, final_confidence, price
            
            return final_signal, final_confidence, price
            
        except Exception as e:
            logger.error(f"Errore nella generazione del segnale: {str(e)}")
            return SignalType.FLAT, 0.0, price

    def _simulate_model_prediction(self, model, data):
        """
        Simula una predizione del modello usando regole base
        In una implementazione reale, qui si utilizzerebbe il modello ML vero e proprio
        """
        try:
            # Ottieni l'ultimo record
            last_row = data.iloc[-1]
            
            # Variabili per la decisione
            sma_20 = last_row.get('sma_20', 0)
            sma_50 = last_row.get('sma_50', 0)
            sma_200 = last_row.get('sma_200', 0)
            rsi_14 = last_row.get('rsi_14', 50)
            macd = last_row.get('macd', 0)
            macd_signal = last_row.get('macd_signal', 0)
            bb_upper = last_row.get('bollinger_upper', float('inf'))
            bb_lower = last_row.get('bollinger_lower', 0)
            close = last_row['close']
            
            # Inizializza conteggi e confidenza
            long_points = 0
            short_points = 0
            
            # Regole di base per simulare la predizione
            
            # SMA Crossover
            if sma_20 > sma_50:
                long_points += 1
            elif sma_20 < sma_50:
                short_points += 1
                
            # Trend a lungo termine (SMA 200)
            if close > sma_200:
                long_points += 0.5
            elif close < sma_200:
                short_points += 0.5
                
            # RSI - Condizioni di ipercomprato/ipervenduto
            if rsi_14 < 30:  # Ipervenduto
                long_points += 1
            elif rsi_14 > 70:  # Ipercomprato
                short_points += 1
                
            # MACD Crossover
            if macd > macd_signal:
                long_points += 1
            elif macd < macd_signal:
                short_points += 1
                
            # Bande di Bollinger
            if close > bb_upper:
                short_points += 0.5
            elif close < bb_lower:
                long_points += 0.5
                
            # Calcola il segnale in base ai punti
            diff = long_points - short_points
            
            # Determina il segnale
            if diff > 1.5:
                signal = SignalType.LONG
                confidence = min(0.5 + (diff / 8), 0.95)
            elif diff < -1.5:
                signal = SignalType.SHORT
                confidence = min(0.5 + (abs(diff) / 8), 0.95)
            else:
                signal = SignalType.FLAT
                confidence = 0.5
                
            # Aggiunge una piccola casualità per simulare la variabilità tra modelli
            confidence += random.uniform(-0.05, 0.05)
            confidence = max(0.1, min(confidence, 0.95))
            
            return signal, confidence
            
        except Exception as e:
            logger.error(f"Errore nella simulazione della predizione: {str(e)}")
            return SignalType.FLAT, 0.0

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
        try:
            # Usa l'ATR per calcolare i livelli di TP e SL basati sulla volatilità
            atr = data['atr_14'].iloc[-1] if 'atr_14' in data.columns else price * 0.01
            
            # Moltiplica l'ATR per un fattore basato sul livello di rischio
            risk_multipliers = {
                1: {'tp': 3.0, 'sl': 1.5},   # Conservativo
                2: {'tp': 3.5, 'sl': 2.0},
                3: {'tp': 4.0, 'sl': 2.5},   # Moderato
                4: {'tp': 4.5, 'sl': 3.0},
                5: {'tp': 5.0, 'sl': 3.5}    # Aggressivo
            }
            
            # Ottieni i moltiplicatori in base al rischio
            multipliers = risk_multipliers.get(self.risk_level, {'tp': 4.0, 'sl': 2.5})
            
            # Calcola TP e SL in base alla direzione del segnale
            if signal == SignalType.LONG:
                tp = price + (atr * multipliers['tp'])
                sl = price - (atr * multipliers['sl'])
            elif signal == SignalType.SHORT:
                tp = price - (atr * multipliers['tp'])
                sl = price + (atr * multipliers['sl'])
            else:
                return None, None, None
                
            # Calcola il volume consigliato in percentuale del portafoglio
            # Basato sul risk management e sul livello di rischio
            risk_percentage = {
                1: 1.0,  # Conservativo
                2: 2.0,
                3: 3.0,  # Moderato
                4: 5.0,
                5: 7.0   # Aggressivo
            }
            
            volume_percentage = risk_percentage.get(self.risk_level, 3.0)
            
            # Arrotonda i valori per maggiore leggibilità
            # Mantieni 5 cifre significative per il prezzo
            decimals = max(0, 5 - len(str(int(price))))
            tp = round(tp, decimals)
            sl = round(sl, decimals)
            
            return tp, sl, volume_percentage
            
        except Exception as e:
            logger.error(f"Errore nel calcolo di TP/SL/Volume: {str(e)}")
            # Valori fallback
            if signal == SignalType.LONG:
                return price * 1.05, price * 0.95, 2.0
            elif signal == SignalType.SHORT:
                return price * 0.95, price * 1.05, 2.0
            else:
                return None, None, None

    def _send_notification(self, signal, price, tp, sl, volume, confidence):
        """Invia una notifica con i dettagli del segnale"""
        try:
            # Verifica che l'utente abbia configurato Telegram
            if not self.user.telegram_bot_token or not self.user.telegram_chat_id:
                logger.warning("Impossibile inviare notifica, configurazione Telegram mancante")
                return False
                
            # Prepara il messaggio
            message = f"🔔 SEGNALE DI TRADING\n\n"
            message += f"Simbolo: {self.dataset.symbol}\n"
            message += f"Prezzo: {price:.5g}\n"
            message += f"Segnale: {signal}\n"
            
            if signal != SignalType.FLAT and tp is not None and sl is not None:
                message += f"Take Profit: {tp:.5g}\n"
                message += f"Stop Loss: {sl:.5g}\n"
                message += f"Volume consigliato: {volume}%\n"
                
            message += f"Confidenza: {confidence:.2f}\n"
            message += f"Timeframe: {self.timeframe}\n"
            message += f"Data: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
            
            # Invia il messaggio
            success = send_signal_notification(self.user_id, message)
            
            return success
            
        except Exception as e:
            logger.error(f"Errore nell'invio della notifica: {str(e)}")
            return False

    def _get_wait_time(self):
        """Calcola il tempo di attesa in base al timeframe"""
        # Converti il timeframe in secondi
        timeframe_seconds = {
            '1m': 60,
            '5m': 300,
            '15m': 900,
            '30m': 1800,
            '1h': 3600,
            '4h': 14400,
            '1d': 86400
        }
        
        # Ottieni il tempo di attesa in base al timeframe
        wait_time = timeframe_seconds.get(self.timeframe, 3600)
        
        # Per evitare richieste troppo frequenti, imposta un minimo di 30 secondi
        return max(wait_time // 4, 30)

# Funzioni di utilità per gestire le configurazioni di segnali

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
    try:
        # Genera un ID univoco per questa configurazione
        config_id = str(uuid.uuid4())
        
        # Converti model_ids in stringa JSON
        model_ids_json = json.dumps(model_ids)
        
        # Crea la configurazione
        config = SignalConfig(
            config_id=config_id,
            user_id=user_id,
            model_ids=model_ids_json,
            dataset_id=dataset_id,
            timeframe=timeframe,
            risk_level=risk_level,
            auto_tp_sl=auto_tp_sl,
            telegram_enabled=telegram_enabled,
            is_active=False
        )
        
        # Salva nel database
        db.session.add(config)
        db.session.commit()
        
        logger.info(f"Configurazione di segnali creata: {config_id}")
        return config_id
        
    except Exception as e:
        logger.error(f"Errore nella creazione della configurazione di segnali: {str(e)}")
        db.session.rollback()
        return None

def get_signal_config(config_id):
    """Ottiene una configurazione di segnali dato il suo ID"""
    try:
        config = SignalConfig.query.filter_by(config_id=config_id).first()
        return config
    except Exception as e:
        logger.error(f"Errore nel recupero della configurazione {config_id}: {str(e)}")
        return None

def start_signal_generator(config_id):
    """Avvia un generatore di segnali"""
    try:
        # Ottieni la configurazione
        config = get_signal_config(config_id)
        if not config:
            logger.error(f"Configurazione {config_id} non trovata")
            return False
            
        # Verifica se il generatore è già attivo
        if config_id in active_generators:
            logger.info(f"Generatore {config_id} già attivo")
            return True
            
        # Crea un nuovo generatore
        generator = SignalGenerator(
            model_ids=config.model_ids,
            dataset_id=config.dataset_id,
            user_id=config.user_id,
            timeframe=config.timeframe,
            risk_level=config.risk_level,
            auto_tp_sl=config.auto_tp_sl,
            telegram_enabled=config.telegram_enabled
        )
        
        # Avvia il generatore
        success = generator.start()
        
        if success:
            # Salva il generatore nella lista dei generatori attivi
            active_generators[config_id] = generator
            
            # Aggiorna lo stato nel database
            config.is_active = True
            db.session.commit()
            
            logger.info(f"Generatore di segnali {config_id} avviato")
            return True
        else:
            logger.error(f"Impossibile avviare il generatore {config_id}")
            return False
            
    except Exception as e:
        logger.error(f"Errore nell'avvio del generatore {config_id}: {str(e)}")
        return False

def stop_signal_generator(config_id):
    """Ferma un generatore di segnali"""
    try:
        # Verifica se il generatore è attivo
        if config_id not in active_generators:
            logger.info(f"Generatore {config_id} non attivo")
            return True
            
        # Ottieni il generatore
        generator = active_generators[config_id]
        
        # Ferma il generatore
        generator.stop()
        
        # Rimuovi il generatore dalla lista dei generatori attivi
        del active_generators[config_id]
        
        # Aggiorna lo stato nel database
        config = get_signal_config(config_id)
        if config:
            config.is_active = False
            db.session.commit()
        
        logger.info(f"Generatore di segnali {config_id} fermato")
        return True
        
    except Exception as e:
        logger.error(f"Errore nella fermata del generatore {config_id}: {str(e)}")
        return False

def delete_signal_config(config_id):
    """Elimina una configurazione di segnali"""
    try:
        # Prima ferma il generatore se è attivo
        if config_id in active_generators:
            stop_signal_generator(config_id)
        
        logger.info(f"Configurazione di segnali {config_id} eliminata")
        return True
        
    except Exception as e:
        logger.error(f"Errore nell'eliminazione della configurazione {config_id}: {str(e)}")
        return False

def get_user_signal_configs(user_id):
    """Ottiene tutte le configurazioni di segnali per un utente"""
    try:
        configs = SignalConfig.query.filter_by(user_id=user_id).all()
        return configs
    except Exception as e:
        logger.error(f"Errore nel recupero delle configurazioni dell'utente {user_id}: {str(e)}")
        return []

def load_all_signal_configs():
    """Carica tutte le configurazioni di segnali dal database"""
    try:
        configs = SignalConfig.query.filter_by(is_active=True).all()
        for config in configs:
            if config.config_id not in active_generators:
                start_signal_generator(config.config_id)
                
        logger.info(f"Caricati {len(configs)} generatori di segnali")
        return True
        
    except Exception as e:
        logger.error(f"Errore nel caricamento delle configurazioni di segnali: {str(e)}")
        return False