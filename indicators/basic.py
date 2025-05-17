"""
Basic Technical Indicators for CryptoTradeAnalyzer

This module contains implementations of standard technical indicators used
for cryptocurrency market analysis.

Indicators implemented:
- Moving Averages (SMA, EMA)
- RSI (Relative Strength Index)
- MACD (Moving Average Convergence Divergence)
- Bollinger Bands
- Stochastic Oscillator

Author: CryptoTradeAnalyzer Team
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate_indicators(df, include=None):
    """
    Calculate all basic technical indicators for the given dataframe
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with OHLCV data
    include : list
        List of indicators to include (None for all)
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with added indicator columns
    """
    if df is None or df.empty:
        logger.error("Empty dataframe provided for indicator calculation")
        return df
    
    required_columns = ['open', 'high', 'low', 'close', 'volume']
    if not all(col in df.columns for col in required_columns):
        logger.error(f"DataFrame missing required columns. Required: {required_columns}, Found: {df.columns}")
        return df
    
    # Create a copy to avoid modifying the original
    result = df.copy()
    
    # Define which indicators to include
    all_indicators = ['sma', 'ema', 'rsi', 'macd', 'bbands', 'stoch']
    indicators_to_calculate = include if include else all_indicators
    
    # Calculate selected indicators
    for indicator in indicators_to_calculate:
        if indicator == 'sma':
            for period in [5, 10, 20, 50, 200]:
                result[f'sma_{period}'] = calculate_sma(result['close'], period)
        
        elif indicator == 'ema':
            for period in [5, 10, 20, 50, 200]:
                result[f'ema_{period}'] = calculate_ema(result['close'], period)
        
        elif indicator == 'rsi':
            result['rsi'] = calculate_rsi(result['close'])
        
        elif indicator == 'macd':
            macd_df = calculate_macd(result['close'])
            result['macd'] = macd_df['macd']
            result['macd_signal'] = macd_df['signal']
            result['macd_hist'] = macd_df['histogram']
        
        elif indicator == 'bbands':
            bb_df = calculate_bollinger_bands(result['close'])
            result['bb_upper'] = bb_df['upper']
            result['bb_middle'] = bb_df['middle']
            result['bb_lower'] = bb_df['lower']
            result['bb_width'] = bb_df['width']
        
        elif indicator == 'stoch':
            stoch_df = calculate_stochastic(result)
            result['stoch_k'] = stoch_df['k']
            result['stoch_d'] = stoch_df['d']
    
    return result

def calculate_sma(series, window):
    """
    Calculate Simple Moving Average
    
    Parameters:
    -----------
    series : pandas.Series
        Price series
    window : int
        Moving average window
        
    Returns:
    --------
    pandas.Series
        SMA values
    """
    return series.rolling(window=window).mean()

def calculate_ema(series, window):
    """
    Calculate Exponential Moving Average
    
    Parameters:
    -----------
    series : pandas.Series
        Price series
    window : int
        Moving average window
        
    Returns:
    --------
    pandas.Series
        EMA values
    """
    return series.ewm(span=window, adjust=False).mean()

def calculate_rsi(series, window=14):
    """
    Calculate Relative Strength Index
    
    Parameters:
    -----------
    series : pandas.Series
        Price series
    window : int
        RSI window
        
    Returns:
    --------
    pandas.Series
        RSI values
    """
    # Calculate price changes
    delta = series.diff()
    
    # Separate gains and losses
    gains = delta.where(delta > 0, 0)
    losses = -delta.where(delta < 0, 0)
    
    # Calculate average gains and losses
    avg_gain = gains.rolling(window=window).mean()
    avg_loss = losses.rolling(window=window).mean()
    
    # Calculate RS
    rs = avg_gain / avg_loss
    
    # Calculate RSI
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def calculate_macd(series, fast=12, slow=26, signal=9):
    """
    Calculate MACD (Moving Average Convergence Divergence)
    
    Parameters:
    -----------
    series : pandas.Series
        Price series
    fast : int
        Fast EMA window
    slow : int
        Slow EMA window
    signal : int
        Signal line window
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with MACD, signal line, and histogram
    """
    # Calculate EMAs
    fast_ema = calculate_ema(series, fast)
    slow_ema = calculate_ema(series, slow)
    
    # Calculate MACD line
    macd_line = fast_ema - slow_ema
    
    # Calculate signal line
    signal_line = macd_line.ewm(span=signal, adjust=False).mean()
    
    # Calculate histogram
    histogram = macd_line - signal_line
    
    # Return as DataFrame
    return pd.DataFrame({
        'macd': macd_line,
        'signal': signal_line,
        'histogram': histogram
    })

def calculate_bollinger_bands(series, window=20, num_std=2):
    """
    Calculate Bollinger Bands
    
    Parameters:
    -----------
    series : pandas.Series
        Price series
    window : int
        Moving average window
    num_std : int or float
        Number of standard deviations
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with upper, middle, and lower bands
    """
    # Calculate middle band (SMA)
    middle_band = calculate_sma(series, window)
    
    # Calculate standard deviation
    std = series.rolling(window=window).std()
    
    # Calculate upper and lower bands
    upper_band = middle_band + (std * num_std)
    lower_band = middle_band - (std * num_std)
    
    # Calculate bandwidth
    bandwidth = (upper_band - lower_band) / middle_band
    
    # Return as DataFrame
    return pd.DataFrame({
        'upper': upper_band,
        'middle': middle_band,
        'lower': lower_band,
        'width': bandwidth
    })

def calculate_stochastic(df, k_window=14, d_window=3):
    """
    Calculate Stochastic Oscillator
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with OHLC data
    k_window : int
        %K window
    d_window : int
        %D window
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with %K and %D values
    """
    # Get high and low prices
    high = df['high']
    low = df['low']
    close = df['close']
    
    # Calculate %K
    lowest_low = low.rolling(window=k_window).min()
    highest_high = high.rolling(window=k_window).max()
    k = 100 * ((close - lowest_low) / (highest_high - lowest_low))
    
    # Calculate %D (SMA of %K)
    d = k.rolling(window=d_window).mean()
    
    # Return as DataFrame
    return pd.DataFrame({
        'k': k,
        'd': d
    })
