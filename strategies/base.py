"""
Base Strategy Module for CryptoTradeAnalyzer

This module defines the base Strategy class that all trading strategies must inherit from.
It provides common functionality and interfaces for all strategies.

Author: CryptoTradeAnalyzer Team
"""

from abc import ABC, abstractmethod
import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

class Strategy(ABC):
    """
    Base Strategy class for all trading strategies
    
    This abstract class defines the interface that all trading strategies must implement.
    """
    
    def __init__(self, name="BaseStrategy"):
        """
        Initialize the strategy
        
        Parameters:
        -----------
        name : str
            Name of the strategy
        """
        self.name = name
        self.signals = None
        self.parameters = {}
        logger.info(f"Initialized strategy: {name}")
    
    @abstractmethod
    def generate_signals(self, data):
        """
        Generate trading signals based on the strategy logic
        
        This method must be implemented by all concrete strategy classes.
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame with price data and indicators
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added signal column
        """
        pass
    
    def set_parameters(self, **kwargs):
        """
        Set strategy parameters
        
        Parameters:
        -----------
        **kwargs : dict
            Dictionary of parameter name-value pairs
        """
        self.parameters.update(kwargs)
        logger.info(f"Updated parameters for {self.name}: {kwargs}")
        return self
    
    def get_parameters(self):
        """
        Get current strategy parameters
        
        Returns:
        --------
        dict
            Dictionary of current parameters
        """
        return self.parameters
    
    def optimize_parameters(self, data, parameter_grid, metric='profit'):
        """
        Optimize strategy parameters using grid search
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame with price data and indicators
        parameter_grid : dict
            Dictionary of parameter names and lists of values to try
        metric : str
            Performance metric to optimize ('profit', 'sharpe', etc.)
            
        Returns:
        --------
        dict
            Dictionary of optimized parameters
        """
        logger.info(f"Optimizing parameters for {self.name}")
        
        # Initialize variables to track best performance
        best_performance = -float('inf') if metric != 'drawdown' else float('inf')
        best_params = None
        
        # Generate all parameter combinations
        import itertools
        param_keys = parameter_grid.keys()
        param_values = parameter_grid.values()
        param_combinations = list(itertools.product(*param_values))
        
        # Test each parameter combination
        for i, combination in enumerate(param_combinations):
            # Set parameters
            params = dict(zip(param_keys, combination))
            self.set_parameters(**params)
            
            # Generate signals
            signals = self.generate_signals(data)
            
            # Calculate performance
            performance = self._calculate_performance(data, signals, metric)
            
            # Check if this is the best performance so far
            if ((metric != 'drawdown' and performance > best_performance) or 
                (metric == 'drawdown' and performance < best_performance)):
                best_performance = performance
                best_params = params
                
            # Log progress periodically
            if (i + 1) % max(1, len(param_combinations) // 10) == 0:
                logger.info(f"Optimization progress: {i+1}/{len(param_combinations)} combinations tested")
        
        # Set the strategy to use the best parameters
        if best_params:
            logger.info(f"Optimization complete. Best parameters: {best_params}, {metric}: {best_performance}")
            self.set_parameters(**best_params)
        else:
            logger.warning("Optimization failed to find better parameters")
        
        return best_params
    
    def _calculate_performance(self, data, signals, metric='profit'):
        """
        Calculate performance metrics for a given set of signals
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame with price data
        signals : pandas.Series
            Series with signal values
        metric : str
            Performance metric to calculate
            
        Returns:
        --------
        float
            Performance metric value
        """
        # Convert signals to position
        position = signals.copy()
        
        # Calculate returns based on position
        returns = data['close'].pct_change() * position.shift(1)
        cumulative_returns = (1 + returns).cumprod() - 1
        
        # Calculate selected metric
        if metric == 'profit':
            # Final cumulative return
            performance = cumulative_returns.iloc[-1] if len(cumulative_returns) > 0 else 0
        
        elif metric == 'sharpe':
            # Sharpe ratio (assuming 0 risk-free rate)
            if returns.std() == 0:
                performance = 0
            else:
                performance = returns.mean() / returns.std() * (252 ** 0.5)  # Annualized
        
        elif metric == 'sortino':
            # Sortino ratio (considering only downside deviation)
            downside_returns = returns[returns < 0]
            if len(downside_returns) == 0 or downside_returns.std() == 0:
                performance = 0
            else:
                performance = returns.mean() / downside_returns.std() * (252 ** 0.5)  # Annualized
        
        elif metric == 'drawdown':
            # Maximum drawdown
            peak = cumulative_returns.cummax()
            drawdown = (cumulative_returns - peak) / (1 + peak)
            performance = drawdown.min()
        
        elif metric == 'win_rate':
            # Win rate
            trades = returns[returns != 0]
            if len(trades) == 0:
                performance = 0
            else:
                performance = (trades > 0).sum() / len(trades)
        
        else:
            logger.warning(f"Unknown performance metric: {metric}. Using profit.")
            performance = cumulative_returns.iloc[-1] if len(cumulative_returns) > 0 else 0
        
        return performance
