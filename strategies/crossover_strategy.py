"""
Moving Average Crossover Strategy for CryptoTradeAnalyzer

This module implements a classic moving average crossover strategy using configurable
fast and slow moving averages.

Author: CryptoTradeAnalyzer Team
"""

import pandas as pd
import numpy as np
import logging
from strategies.base import Strategy
from indicators.basic import calculate_sma, calculate_ema

logger = logging.getLogger(__name__)

class CrossoverStrategy(Strategy):
    """
    Moving Average Crossover Strategy class
    
    This strategy generates buy signals when a fast moving average crosses above
    a slow moving average, and sell signals when it crosses below.
    """
    
    def __init__(self, fast_ma=20, slow_ma=50, ma_type='sma'):
        """
        Initialize the Moving Average Crossover Strategy
        
        Parameters:
        -----------
        fast_ma : int
            Fast moving average period
        slow_ma : int
            Slow moving average period
        ma_type : str
            Type of moving average ('sma' or 'ema')
        """
        super().__init__(name="MA Crossover Strategy")
        
        # Set default parameters
        self.parameters = {
            'fast_ma': fast_ma,
            'slow_ma': slow_ma,
            'ma_type': ma_type
        }
        
        logger.info(f"Initialized MA Crossover Strategy with fast_ma={fast_ma}, slow_ma={slow_ma}, ma_type={ma_type}")
    
    def calculate_moving_averages(self, data):
        """
        Calculate moving averages based on parameters
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame with price data
            
        Returns:
        --------
        tuple
            Tuple of (fast_ma, slow_ma) Series
        """
        # Extract parameters
        fast_ma_period = self.parameters['fast_ma']
        slow_ma_period = self.parameters['slow_ma']
        ma_type = self.parameters['ma_type']
        
        # Calculate moving averages
        if ma_type.lower() == 'sma':
            fast_ma = calculate_sma(data['close'], fast_ma_period)
            slow_ma = calculate_sma(data['close'], slow_ma_period)
        elif ma_type.lower() == 'ema':
            fast_ma = calculate_ema(data['close'], fast_ma_period)
            slow_ma = calculate_ema(data['close'], slow_ma_period)
        else:
            logger.warning(f"Unknown MA type: {ma_type}. Using SMA.")
            fast_ma = calculate_sma(data['close'], fast_ma_period)
            slow_ma = calculate_sma(data['close'], slow_ma_period)
        
        return fast_ma, slow_ma
    
    def generate_signals(self, data):
        """
        Generate trading signals based on moving average crossovers
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame with price data
            
        Returns:
        --------
        pandas.Series
            Series with signal values (1 for buy, -1 for sell, 0 for hold)
        """
        logger.info("Generating signals for MA Crossover Strategy")
        
        # Calculate moving averages
        fast_ma, slow_ma = self.calculate_moving_averages(data)
        
        # Add moving averages to data for reference
        data_with_ma = data.copy()
        data_with_ma[f"ma_fast_{self.parameters['fast_ma']}"] = fast_ma
        data_with_ma[f"ma_slow_{self.parameters['slow_ma']}"] = slow_ma
        
        # Initialize signal Series
        signals = pd.Series(0, index=data.index)
        
        # Previous day's fast MA > slow MA
        previous_diff = fast_ma.shift(1) - slow_ma.shift(1)
        
        # Current day's fast MA > slow MA
        current_diff = fast_ma - slow_ma
        
        # Buy signal: Fast MA crosses above Slow MA
        buy_signal = (previous_diff <= 0) & (current_diff > 0)
        signals[buy_signal] = 1
        
        # Sell signal: Fast MA crosses below Slow MA
        sell_signal = (previous_diff >= 0) & (current_diff < 0)
        signals[sell_signal] = -1
        
        # Save signals for reference
        self.signals = signals
        
        # Calculate number of buy and sell signals
        num_buy = buy_signal.sum()
        num_sell = sell_signal.sum()
        logger.info(f"Generated {num_buy} buy signals and {num_sell} sell signals")
        
        return signals
    
    def generate_positions(self, data):
        """
        Generate position series (1 for long, -1 for short, 0 for flat)
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame with price data
            
        Returns:
        --------
        pandas.Series
            Series with position values
        """
        # Generate signals if not already generated
        if self.signals is None or len(self.signals) != len(data):
            self.generate_signals(data)
        
        # Convert signals to positions
        positions = self.signals.copy()
        
        # Fill forward positions (maintain position until new signal)
        for i in range(1, len(positions)):
            if positions.iloc[i] == 0:
                positions.iloc[i] = positions.iloc[i-1]
        
        return positions
