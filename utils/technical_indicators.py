"""
CryptoTradeAnalyzer - Indicatori Tecnici

Questo modulo implementa vari indicatori tecnici utilizzati per l'analisi e la generazione di segnali.
"""

import numpy as np
import pandas as pd


def add_sma(df, window):
    """Aggiunge una media mobile semplice (SMA) al DataFrame"""
    df[f'sma_{window}'] = df['close'].rolling(window=window).mean()
    return df


def add_ema(df, window):
    """Aggiunge una media mobile esponenziale (EMA) al DataFrame"""
    df[f'ema_{window}'] = df['close'].ewm(span=window, adjust=False).mean()
    return df


def add_rsi(df, window):
    """Calcola l'indice di forza relativa (RSI)"""
    delta = df['close'].diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    avg_gain = gain.rolling(window=window).mean()
    avg_loss = loss.rolling(window=window).mean()
    
    # Calcola RS e RSI
    rs = avg_gain / avg_loss
    df[f'rsi_{window}'] = 100 - (100 / (1 + rs))
    
    return df


def add_macd(df, fast=12, slow=26, signal=9):
    """Calcola MACD (Moving Average Convergence Divergence)"""
    # Calcola le EMA
    ema_fast = df['close'].ewm(span=fast, adjust=False).mean()
    ema_slow = df['close'].ewm(span=slow, adjust=False).mean()
    
    # Calcola MACD e linea di segnale
    df['macd'] = ema_fast - ema_slow
    df['macd_signal'] = df['macd'].ewm(span=signal, adjust=False).mean()
    df['macd_histogram'] = df['macd'] - df['macd_signal']
    
    return df


def add_bollinger_bands(df, window=20, std=2):
    """Calcola le bande di Bollinger"""
    # Calcola la media mobile
    df['bollinger_middle'] = df['close'].rolling(window=window).mean()
    
    # Calcola la deviazione standard
    rolling_std = df['close'].rolling(window=window).std()
    
    # Calcola le bande superiori e inferiori
    df['bollinger_upper'] = df['bollinger_middle'] + (rolling_std * std)
    df['bollinger_lower'] = df['bollinger_middle'] - (rolling_std * std)
    
    return df


def add_atr(df, window=14):
    """Calcola l'Average True Range (ATR)"""
    high_low = df['high'] - df['low']
    high_close = (df['high'] - df['close'].shift()).abs()
    low_close = (df['low'] - df['close'].shift()).abs()
    
    # Prendi il massimo tra i tre valori
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Calcola l'ATR come media mobile del True Range
    df[f'atr_{window}'] = tr.rolling(window=window).mean()
    
    return df


def add_stochastic(df, k_window=14, d_window=3):
    """Calcola l'oscillatore stocastico"""
    # Calcola il %K
    low_min = df['low'].rolling(window=k_window).min()
    high_max = df['high'].rolling(window=k_window).max()
    
    df['stoch_k'] = 100 * ((df['close'] - low_min) / (high_max - low_min))
    
    # Calcola il %D (media mobile di %K)
    df['stoch_d'] = df['stoch_k'].rolling(window=d_window).mean()
    
    return df


def add_volume_indicators(df):
    """Aggiunge indicatori basati sul volume"""
    # Volumi medi
    df['volume_sma_20'] = df['volume'].rolling(window=20).mean()
    
    # OBV (On-Balance Volume)
    df['obv'] = np.where(df['close'] > df['close'].shift(1), 
                        df['volume'], 
                        np.where(df['close'] < df['close'].shift(1), 
                                -df['volume'], 0)).cumsum()
    
    return df


def add_indicators(df, indicators_config):
    """
    Aggiunge indicatori tecnici a un DataFrame in base alla configurazione
    
    Args:
        df (DataFrame): DataFrame con dati OHLCV
        indicators_config (dict): Dizionario con la configurazione degli indicatori
            Es: {
                'sma': [20, 50, 200],
                'ema': [9, 21],
                'rsi': [14],
                'macd': {'fast': 12, 'slow': 26, 'signal': 9},
                'bollinger': {'window': 20, 'std': 2},
                'atr': [14]
            }
    
    Returns:
        DataFrame: DataFrame con gli indicatori aggiunti
    """
    # Copia il DataFrame per non modificare l'originale
    result = df.copy()
    
    # Aggiungi gli indicatori richiesti
    if 'sma' in indicators_config:
        for window in indicators_config['sma']:
            result = add_sma(result, window)
    
    if 'ema' in indicators_config:
        for window in indicators_config['ema']:
            result = add_ema(result, window)
    
    if 'rsi' in indicators_config:
        for window in indicators_config['rsi']:
            result = add_rsi(result, window)
    
    if 'macd' in indicators_config:
        config = indicators_config['macd']
        fast = config.get('fast', 12)
        slow = config.get('slow', 26)
        signal = config.get('signal', 9)
        result = add_macd(result, fast, slow, signal)
    
    if 'bollinger' in indicators_config:
        config = indicators_config['bollinger']
        window = config.get('window', 20)
        std = config.get('std', 2)
        result = add_bollinger_bands(result, window, std)
    
    if 'atr' in indicators_config:
        for window in indicators_config['atr']:
            result = add_atr(result, window)
    
    if 'stochastic' in indicators_config:
        config = indicators_config['stochastic']
        k_window = config.get('k_window', 14)
        d_window = config.get('d_window', 3)
        result = add_stochastic(result, k_window, d_window)
    
    if 'volume' in indicators_config and indicators_config['volume']:
        result = add_volume_indicators(result)
    
    return result