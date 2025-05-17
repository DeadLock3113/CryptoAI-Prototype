"""
Advanced Technical Indicators for CryptoTradeAnalyzer

This module contains implementations of more advanced technical indicators used
for cryptocurrency market analysis.

Indicators implemented:
- Fibonacci Retracement Levels
- Average True Range (ATR)
- Ichimoku Cloud
- Volume Weighted Average Price (VWAP)
- On-Balance Volume (OBV)
- Money Flow Index (MFI)
- Parabolic SAR

Author: CryptoTradeAnalyzer Team
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate_advanced_indicators(df):
    """
    Calculate all advanced technical indicators for the given dataframe
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with OHLCV data
        
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
    
    # Calculate ATR
    result = calculate_atr(result)
    
    # Calculate OBV
    result['obv'] = calculate_obv(result)
    
    # Calculate MFI
    result['mfi'] = calculate_mfi(result)
    
    # Calculate VWAP
    result['vwap'] = calculate_vwap(result)
    
    # Calculate Parabolic SAR
    result['psar'] = calculate_parabolic_sar(result)
    
    # Calculate Ichimoku Cloud
    ichimoku = calculate_ichimoku(result)
    result = pd.concat([result, ichimoku], axis=1)
    
    # Calculate Fibonacci Levels - Note: Not a time-series indicator, calculated as needed
    
    return result

def calculate_atr(df, window=14):
    """
    Calculate Average True Range (ATR)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with OHLC data
    window : int
        ATR window
        
    Returns:
    --------
    pandas.DataFrame
        Original DataFrame with ATR column added
    """
    high = df['high']
    low = df['low']
    close = df['close'].shift(1)
    
    # Calculate True Range
    tr1 = high - low
    tr2 = abs(high - close)
    tr3 = abs(low - close)
    
    tr = pd.DataFrame({
        'tr1': tr1,
        'tr2': tr2,
        'tr3': tr3
    }).max(axis=1)
    
    # Calculate ATR (Average True Range)
    atr = tr.rolling(window=window).mean()
    
    # Add ATR to DataFrame
    df_copy = df.copy()
    df_copy['atr'] = atr
    
    return df_copy

def calculate_obv(df):
    """
    Calculate On-Balance Volume (OBV)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with OHLCV data
        
    Returns:
    --------
    pandas.Series
        OBV values
    """
    close = df['close']
    volume = df['volume']
    
    # Calculate daily price change
    price_change = close.diff()
    
    # Initialize OBV
    obv = pd.Series(0, index=close.index)
    
    # Calculate OBV
    for i in range(1, len(close)):
        if price_change.iloc[i] > 0:
            obv.iloc[i] = obv.iloc[i-1] + volume.iloc[i]
        elif price_change.iloc[i] < 0:
            obv.iloc[i] = obv.iloc[i-1] - volume.iloc[i]
        else:
            obv.iloc[i] = obv.iloc[i-1]
    
    return obv

def calculate_mfi(df, window=14):
    """
    Calculate Money Flow Index (MFI)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with OHLCV data
    window : int
        MFI window
        
    Returns:
    --------
    pandas.Series
        MFI values
    """
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    money_flow = typical_price * df['volume']
    
    # Calculate positive and negative money flow
    price_change = typical_price.diff()
    positive_flow = pd.Series(0, index=typical_price.index)
    negative_flow = pd.Series(0, index=typical_price.index)
    
    # Populate positive and negative flow series
    positive_flow[price_change > 0] = money_flow[price_change > 0]
    negative_flow[price_change < 0] = money_flow[price_change < 0]
    
    # Calculate positive and negative money flow sums
    positive_mf_sum = positive_flow.rolling(window=window).sum()
    negative_mf_sum = negative_flow.rolling(window=window).sum()
    
    # Calculate money flow ratio
    mf_ratio = positive_mf_sum / negative_mf_sum
    
    # Calculate MFI
    mfi = 100 - (100 / (1 + mf_ratio))
    
    return mfi

def calculate_vwap(df):
    """
    Calculate Volume Weighted Average Price (VWAP)
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with OHLCV data
        
    Returns:
    --------
    pandas.Series
        VWAP values
    """
    # Calculate typical price
    typical_price = (df['high'] + df['low'] + df['close']) / 3
    
    # Calculate VWAP
    vwap = (typical_price * df['volume']).cumsum() / df['volume'].cumsum()
    
    return vwap

def calculate_parabolic_sar(df, af_start=0.02, af_increment=0.02, af_max=0.2):
    """
    Calculate Parabolic SAR
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with OHLC data
    af_start : float
        Starting acceleration factor
    af_increment : float
        Acceleration factor increment
    af_max : float
        Maximum acceleration factor
        
    Returns:
    --------
    pandas.Series
        Parabolic SAR values
    """
    high = df['high']
    low = df['low']
    
    # Initialize SAR, EP, AF, and trend direction
    psar = pd.Series(0.0, index=df.index)
    ep = pd.Series(0.0, index=df.index)  # Extreme point
    af = pd.Series(0.0, index=df.index)  # Acceleration factor
    trend = pd.Series(1, index=df.index)  # 1 for uptrend, -1 for downtrend
    
    # Initialize first values
    trend.iloc[0] = 1  # Start with uptrend
    ep.iloc[0] = high.iloc[0]  # Initial EP is first high
    psar.iloc[0] = low.iloc[0]  # Initial SAR is first low
    af.iloc[0] = af_start
    
    # Calculate PSAR
    for i in range(1, len(df)):
        # Update SAR
        psar.iloc[i] = psar.iloc[i-1] + af.iloc[i-1] * (ep.iloc[i-1] - psar.iloc[i-1])
        
        # Ensure SAR doesn't go beyond yesterday's range
        if trend.iloc[i-1] == 1:
            psar.iloc[i] = min(psar.iloc[i], low.iloc[i-1], low.iloc[i-2] if i > 1 else low.iloc[i-1])
        else:
            psar.iloc[i] = max(psar.iloc[i], high.iloc[i-1], high.iloc[i-2] if i > 1 else high.iloc[i-1])
        
        # Check for trend reversal
        if (trend.iloc[i-1] == 1 and psar.iloc[i] > low.iloc[i]) or \
           (trend.iloc[i-1] == -1 and psar.iloc[i] < high.iloc[i]):
            # Reverse trend
            trend.iloc[i] = -trend.iloc[i-1]
            # Reset AF
            af.iloc[i] = af_start
            # Reset EP
            ep.iloc[i] = high.iloc[i] if trend.iloc[i] == 1 else low.iloc[i]
            # Reverse SAR value
            psar.iloc[i] = ep.iloc[i-1]
        else:
            # Continue trend
            trend.iloc[i] = trend.iloc[i-1]
            # Update EP if needed
            if trend.iloc[i] == 1 and high.iloc[i] > ep.iloc[i-1]:
                ep.iloc[i] = high.iloc[i]
                # Increase AF
                af.iloc[i] = min(af.iloc[i-1] + af_increment, af_max)
            elif trend.iloc[i] == -1 and low.iloc[i] < ep.iloc[i-1]:
                ep.iloc[i] = low.iloc[i]
                # Increase AF
                af.iloc[i] = min(af.iloc[i-1] + af_increment, af_max)
            else:
                ep.iloc[i] = ep.iloc[i-1]
                af.iloc[i] = af.iloc[i-1]
    
    return psar

def calculate_ichimoku(df):
    """
    Calculate Ichimoku Cloud
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with OHLC data
        
    Returns:
    --------
    pandas.DataFrame
        DataFrame with Ichimoku components
    """
    # Calculate Tenkan-sen (Conversion Line): (highest high + lowest low) / 2 for the past 9 periods
    high_9 = df['high'].rolling(window=9).max()
    low_9 = df['low'].rolling(window=9).min()
    tenkan_sen = (high_9 + low_9) / 2
    
    # Calculate Kijun-sen (Base Line): (highest high + lowest low) / 2 for the past 26 periods
    high_26 = df['high'].rolling(window=26).max()
    low_26 = df['low'].rolling(window=26).min()
    kijun_sen = (high_26 + low_26) / 2
    
    # Calculate Senkou Span A (Leading Span A): (Conversion Line + Base Line) / 2, plotted 26 periods ahead
    senkou_span_a = ((tenkan_sen + kijun_sen) / 2).shift(26)
    
    # Calculate Senkou Span B (Leading Span B): (highest high + lowest low) / 2 for the past 52 periods, plotted 26 periods ahead
    high_52 = df['high'].rolling(window=52).max()
    low_52 = df['low'].rolling(window=52).min()
    senkou_span_b = ((high_52 + low_52) / 2).shift(26)
    
    # Calculate Chikou Span (Lagging Span): Current closing price, plotted 26 periods behind
    chikou_span = df['close'].shift(-26)
    
    # Return as DataFrame
    return pd.DataFrame({
        'ichimoku_tenkan': tenkan_sen,
        'ichimoku_kijun': kijun_sen,
        'ichimoku_senkou_a': senkou_span_a,
        'ichimoku_senkou_b': senkou_span_b,
        'ichimoku_chikou': chikou_span
    })

def calculate_fibonacci_levels(df, swing_high, swing_low):
    """
    Calculate Fibonacci Retracement Levels
    
    Parameters:
    -----------
    df : pandas.DataFrame
        DataFrame with OHLC data
    swing_high : float
        The swing high price
    swing_low : float
        The swing low price
        
    Returns:
    --------
    dict
        Dictionary with Fibonacci levels
    """
    diff = swing_high - swing_low
    
    # Calculate Fibonacci levels
    level_0 = swing_high
    level_23_6 = swing_high - 0.236 * diff
    level_38_2 = swing_high - 0.382 * diff
    level_50_0 = swing_high - 0.5 * diff
    level_61_8 = swing_high - 0.618 * diff
    level_78_6 = swing_high - 0.786 * diff
    level_100 = swing_low
    
    # Extension levels
    level_138_2 = swing_low - 0.382 * diff
    level_161_8 = swing_low - 0.618 * diff
    
    # Return as dictionary
    return {
        'fib_0': level_0,
        'fib_23.6': level_23_6,
        'fib_38.2': level_38_2,
        'fib_50': level_50_0,
        'fib_61.8': level_61_8,
        'fib_78.6': level_78_6,
        'fib_100': level_100,
        'fib_138.2': level_138_2,
        'fib_161.8': level_161_8
    }
