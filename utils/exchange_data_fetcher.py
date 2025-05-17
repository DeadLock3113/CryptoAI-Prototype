"""
Modulo per ottenere dati storici dagli exchange

Questo modulo fornisce funzioni per recuperare dati storici da vari exchange 
di criptovalute come Binance e Kraken per l'addestramento dei modelli.
"""

import os
import logging
import time
import json
import hmac
import hashlib
import base64
import pandas as pd
import requests
from datetime import datetime, timedelta
from urllib.parse import urlencode
from typing import List, Dict, Any, Optional, Tuple

# Setup logging
logger = logging.getLogger(__name__)

class ExchangeAPIError(Exception):
    """Eccezione per errori nelle API degli exchange"""
    pass

def fetch_binance_historical_data(
    symbol: str,
    interval: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    limit: int = 1000,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None
) -> pd.DataFrame:
    """
    Recupera dati storici da Binance.
    
    Args:
        symbol: Coppia di trading (es. 'BTCUSDT')
        interval: Intervallo temporale (es. '1d', '4h', '1h', '15m', '5m', '1m')
        start_time: Data di inizio (opzionale)
        end_time: Data di fine (opzionale)
        limit: Numero massimo di candele da recuperare per richiesta
        api_key: Chiave API (opzionale, per limiti più alti)
        api_secret: Secret API (opzionale, per limiti più alti)
        
    Returns:
        DataFrame con i dati storici
    """
    endpoint = 'https://api.binance.com/api/v3/klines'
    
    # Converte datetime in millisecondi timestamp per Binance
    params = {
        'symbol': symbol,
        'interval': interval,
        'limit': limit
    }
    
    if start_time:
        params['startTime'] = int(start_time.timestamp() * 1000)
    if end_time:
        params['endTime'] = int(end_time.timestamp() * 1000)
    
    # Aggiungi autenticazione se fornita
    headers = {}
    if api_key:
        headers['X-MBX-APIKEY'] = api_key
    
    logger.info(f"Recupero dati storici da Binance per {symbol} con intervallo {interval}")
    
    try:
        response = requests.get(endpoint, params=params, headers=headers)
        response.raise_for_status()
        
        # Convertire la risposta in DataFrame
        data = response.json()
        
        if not data:
            logger.warning(f"Nessun dato restituito da Binance per {symbol}")
            return pd.DataFrame()
        
        # Creare il DataFrame
        df = pd.DataFrame(data, columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        # Conversione colonne numeriche
        numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'quote_asset_volume', 
                          'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume']
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
        
        # Conversione timestamp in datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        # Imposta timestamp come indice
        df.set_index('timestamp', inplace=True)
        
        logger.info(f"Recuperati {len(df)} record da Binance per {symbol}")
        return df
    
    except Exception as e:
        error_msg = f"Errore nel recupero dati da Binance: {str(e)}"
        logger.error(error_msg)
        raise ExchangeAPIError(error_msg)

def fetch_kraken_historical_data(
    pair: str,
    interval: int,  # in minuti (1, 5, 15, 30, 60, 240, 1440, 10080, 21600)
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None
) -> pd.DataFrame:
    """
    Recupera dati storici da Kraken.
    
    Args:
        pair: Coppia di trading (es. 'XBTUSD')
        interval: Intervallo in minuti (1, 5, 15, 30, 60, 240, 1440, 10080, 21600)
        start_time: Data di inizio (opzionale)
        end_time: Data di fine (opzionale)
        api_key: Chiave API (opzionale, per limiti più alti)
        api_secret: Secret API (opzionale, per limiti più alti)
        
    Returns:
        DataFrame con i dati storici
    """
    endpoint = 'https://api.kraken.com/0/public/OHLC'
    
    # Converti intervallo in minuti a secondi per Kraken
    interval_seconds = interval * 60
    
    params = {
        'pair': pair,
        'interval': interval_seconds // 60  # Kraken accetta intervalli in minuti
    }
    
    if start_time:
        params['since'] = int(start_time.timestamp())
    
    logger.info(f"Recupero dati storici da Kraken per {pair} con intervallo {interval}m")
    
    try:
        response = requests.get(endpoint, params=params)
        response.raise_for_status()
        
        result = response.json()
        
        if 'error' in result and result['error']:
            raise ExchangeAPIError(f"Errore API Kraken: {result['error']}")
        
        if 'result' not in result:
            logger.warning(f"Risposta Kraken non valida per {pair}")
            return pd.DataFrame()
        
        # Kraken ritorna un dizionario con la chiave del nome della coppia
        pair_data = result['result']
        
        # Trova la chiave corretta (la prima che non è 'last')
        data_key = next((k for k in pair_data.keys() if k != 'last'), None)
        
        if not data_key or not pair_data[data_key]:
            logger.warning(f"Nessun dato restituito da Kraken per {pair}")
            return pd.DataFrame()
        
        # Creare il DataFrame
        df = pd.DataFrame(pair_data[data_key], columns=[
            'timestamp', 'open', 'high', 'low', 'close', 'vwap', 
            'volume', 'count'
        ])
        
        # Conversione colonne numeriche
        numeric_columns = ['open', 'high', 'low', 'close', 'vwap', 'volume', 'count']
        df[numeric_columns] = df[numeric_columns].apply(pd.to_numeric)
        
        # Conversione timestamp in datetime
        df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
        
        # Imposta timestamp come indice
        df.set_index('timestamp', inplace=True)
        
        # Filtra per end_time se specificato
        if end_time:
            df = df[df.index <= end_time]
        
        logger.info(f"Recuperati {len(df)} record da Kraken per {pair}")
        return df
    
    except Exception as e:
        error_msg = f"Errore nel recupero dati da Kraken: {str(e)}"
        logger.error(error_msg)
        raise ExchangeAPIError(error_msg)

def get_historical_data(
    exchange: str,
    symbol: str,
    interval: str,
    start_time: Optional[datetime] = None,
    end_time: Optional[datetime] = None,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None
) -> pd.DataFrame:
    """
    Funzione unificata per recuperare dati storici da vari exchange.
    
    Args:
        exchange: Nome dell'exchange ('binance' o 'kraken')
        symbol: Coppia di trading
        interval: Intervallo temporale
        start_time: Data di inizio (opzionale)
        end_time: Data di fine (opzionale)
        api_key: Chiave API (opzionale)
        api_secret: Secret API (opzionale)
        
    Returns:
        DataFrame con i dati storici
    """
    exchange = exchange.lower()
    
    # Imposta date predefinite se non specificate
    if not end_time:
        end_time = datetime.now()
    if not start_time:
        # Predefinito: un anno di dati
        start_time = end_time - timedelta(days=365)
    
    # Normalizza l'intervallo per ogni exchange
    if exchange == 'binance':
        # Binance accetta formati come '1d', '4h', '1h', '15m', '5m', '1m'
        # Converti se necessario
        interval_map = {
            '1m': '1m', '5m': '5m', '15m': '15m', '30m': '30m',
            '1h': '1h', '2h': '2h', '4h': '4h', '6h': '6h', '8h': '8h', '12h': '12h',
            '1d': '1d', '3d': '3d', '1w': '1w', '1M': '1M'
        }
        
        # Converti format numerico (es. '60' per 60 minuti) a format Binance
        if interval.isdigit():
            minutes = int(interval)
            if minutes == 1:
                interval = '1m'
            elif minutes == 5:
                interval = '5m'
            elif minutes == 15:
                interval = '15m'
            elif minutes == 30:
                interval = '30m'
            elif minutes == 60:
                interval = '1h'
            elif minutes == 240:
                interval = '4h'
            elif minutes == 1440:
                interval = '1d'
            else:
                interval = '1h'  # default
        
        binance_interval = interval_map.get(interval, '1h')
        return fetch_binance_historical_data(
            symbol=symbol, 
            interval=binance_interval,
            start_time=start_time,
            end_time=end_time,
            api_key=api_key,
            api_secret=api_secret
        )
        
    elif exchange == 'kraken':
        # Kraken accetta intervalli in minuti: 1, 5, 15, 30, 60, 240, 1440, 10080, 21600
        # Converti se necessario
        kraken_interval_map = {
            '1m': 1, '5m': 5, '15m': 15, '30m': 30,
            '1h': 60, '4h': 240,
            '1d': 1440, '1w': 10080
        }
        
        # Se l'intervallo è fornito come stringa (es. '1h'), converti in minuti
        if isinstance(interval, str) and not interval.isdigit():
            interval = kraken_interval_map.get(interval, 60)  # default a 1h
        else:
            interval = int(interval)
            
        return fetch_kraken_historical_data(
            pair=symbol,
            interval=interval,
            start_time=start_time,
            end_time=end_time,
            api_key=api_key,
            api_secret=api_secret
        )
    
    else:
        raise ValueError(f"Exchange non supportato: {exchange}. Supportati: binance, kraken")

def save_historical_data(df: pd.DataFrame, file_path: str) -> str:
    """
    Salva i dati storici in un file CSV.
    
    Args:
        df: DataFrame con i dati
        file_path: Percorso dove salvare il file
        
    Returns:
        Percorso completo del file salvato
    """
    try:
        # Crea la directory se non esiste
        os.makedirs(os.path.dirname(os.path.abspath(file_path)), exist_ok=True)
        
        # Salva il file
        df.to_csv(file_path)
        logger.info(f"Dati salvati con successo in: {file_path}")
        return file_path
    
    except Exception as e:
        error_msg = f"Errore nel salvataggio dei dati: {str(e)}"
        logger.error(error_msg)
        raise IOError(error_msg)

def get_available_timeframes(exchange: str) -> List[str]:
    """
    Restituisce i timeframe disponibili per un dato exchange.
    
    Args:
        exchange: Nome dell'exchange ('binance' o 'kraken')
        
    Returns:
        Lista di timeframe disponibili
    """
    exchange = exchange.lower()
    
    if exchange == 'binance':
        return ['1m', '3m', '5m', '15m', '30m', '1h', '2h', '4h', '6h', '8h', '12h', '1d', '3d', '1w', '1M']
    
    elif exchange == 'kraken':
        return ['1m', '5m', '15m', '30m', '1h', '4h', '1d', '1w']
    
    else:
        raise ValueError(f"Exchange non supportato: {exchange}")

def get_available_symbols(
    exchange: str, 
    base_only: bool = False,
    api_key: Optional[str] = None,
    api_secret: Optional[str] = None
) -> List[str]:
    """
    Recupera i simboli disponibili su un exchange.
    
    Args:
        exchange: Nome dell'exchange ('binance' o 'kraken')
        base_only: Se True, restituisce solo i simboli base (BTC, ETH, ecc.)
        api_key: Chiave API (opzionale)
        api_secret: Secret API (opzionale)
        
    Returns:
        Lista di simboli disponibili
    """
    exchange = exchange.lower()
    
    if exchange == 'binance':
        try:
            response = requests.get('https://api.binance.com/api/v3/exchangeInfo')
            response.raise_for_status()
            
            data = response.json()
            symbols = [s['symbol'] for s in data['symbols'] if s['status'] == 'TRADING']
            
            if base_only:
                # Estrai le valute base (prima parte del simbolo)
                base_currencies = set()
                for s in data['symbols']:
                    if s['status'] == 'TRADING':
                        base_currencies.add(s['baseAsset'])
                return sorted(list(base_currencies))
            
            return symbols
        
        except Exception as e:
            logger.error(f"Errore nel recupero simboli Binance: {str(e)}")
            # Restituisci alcuni simboli comuni come fallback
            return ['BTCUSDT', 'ETHUSDT', 'BNBUSDT', 'ADAUSDT', 'DOGEUSDT']
    
    elif exchange == 'kraken':
        try:
            response = requests.get('https://api.kraken.com/0/public/AssetPairs')
            response.raise_for_status()
            
            data = response.json()
            
            if 'error' in data and data['error']:
                raise ExchangeAPIError(f"Errore API Kraken: {data['error']}")
            
            symbols = list(data['result'].keys())
            
            if base_only:
                # Estrai le valute base (prima parte del simbolo)
                base_currencies = set()
                for s, info in data['result'].items():
                    base = info.get('base', '')
                    if base:
                        # Rimuovi eventuali prefissi X o Z che Kraken aggiunge
                        if base.startswith('X') or base.startswith('Z'):
                            base = base[1:]
                        base_currencies.add(base)
                return sorted(list(base_currencies))
            
            return symbols
        
        except Exception as e:
            logger.error(f"Errore nel recupero simboli Kraken: {str(e)}")
            # Restituisci alcuni simboli comuni come fallback
            return ['XXBTZUSD', 'XETHZUSD', 'ADAUSD', 'DOTUSD', 'XLTCZUSD']
    
    else:
        raise ValueError(f"Exchange non supportato: {exchange}")