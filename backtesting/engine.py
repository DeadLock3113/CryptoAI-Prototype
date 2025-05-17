"""
Backtesting Engine Module for CryptoTradeAnalyzer

This module implements a comprehensive backtesting engine to evaluate
trading strategies on historical data, with realistic modeling of
slippage, fees, and market conditions.

Author: CryptoTradeAnalyzer Team
"""

import pandas as pd
import numpy as np
import logging
from copy import deepcopy
import matplotlib.pyplot as plt

from backtesting.metrics import calculate_performance_metrics

logger = logging.getLogger(__name__)

class BacktestEngine:
    """
    Backtesting Engine for cryptocurrency trading strategies
    
    This class handles the execution of trading strategies on historical data,
    tracks trades, calculates performance metrics, and generates reports.
    """
    
    def __init__(self, data, strategy, initial_capital=10000, commission=0.001, 
                 slippage=0.0005, position_size=1.0, enable_fractional=True):
        """
        Initialize the Backtest Engine
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame with OHLCV data and indicators
        strategy : Strategy
            Strategy object that implements generate_signals method
        initial_capital : float
            Initial capital for the backtest
        commission : float
            Commission fee per trade (percentage, e.g., 0.001 = 0.1%)
        slippage : float
            Slippage model (percentage, e.g., 0.0005 = 0.05%)
        position_size : float
            Position size as a fraction of capital (0 to 1)
        enable_fractional : bool
            Whether to allow fractional positions
        """
        # Make a copy of data to avoid modifying the original
        self.data = data.copy()
        self.strategy = strategy
        self.initial_capital = initial_capital
        self.commission = commission
        self.slippage = slippage
        self.position_size = position_size
        self.enable_fractional = enable_fractional
        
        # Initialize state variables
        self.current_capital = initial_capital
        self.positions = pd.Series(0, index=data.index)
        self.holdings = pd.Series(0.0, index=data.index)
        self.cash = pd.Series(initial_capital, index=data.index)
        self.equity = pd.Series(initial_capital, index=data.index)
        self.trades = []
        self.metrics = {}
        
        logger.info(f"Initialized Backtest Engine with {len(data)} data points")
        logger.info(f"Initial capital: ${initial_capital:.2f}, Commission: {commission*100:.3f}%, Slippage: {slippage*100:.3f}%")
    
    def run(self):
        """
        Run the backtest
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with backtest results
        """
        logger.info(f"Running backtest with {self.strategy.name} strategy")
        
        # Generate signals from the strategy
        signals = self.strategy.generate_signals(self.data)
        
        # Process signals to generate portfolio changes
        self._process_signals(signals)
        
        # Calculate equity curve and returns
        self._calculate_equity_curve()
        
        # Calculate performance metrics
        self._calculate_metrics()
        
        # Prepare results dataframe
        results = pd.DataFrame({
            'price': self.data['close'],
            'signal': signals,
            'position': self.positions,
            'holdings': self.holdings,
            'cash': self.cash,
            'equity': self.equity
        })
        
        logger.info(f"Backtest complete: Final equity: ${self.equity.iloc[-1]:.2f}")
        return results
    
    def _process_signals(self, signals):
        """
        Process trading signals to update positions and cash
        
        Parameters:
        -----------
        signals : pandas.Series
            Series with signal values (1 for buy, -1 for sell, 0 for hold)
        """
        last_position = 0
        
        # Process each signal to update positions
        for i, timestamp in enumerate(self.data.index):
            current_price = self.data.loc[timestamp, 'close']
            current_signal = signals.loc[timestamp]
            
            # Determine position size
            available_capital = self.cash.iloc[i-1] if i > 0 else self.initial_capital
            max_position_size = (available_capital * self.position_size) / current_price
            
            # Update position based on signal
            new_position = self._determine_position(current_signal, last_position)
            
            # Calculate position change
            position_change = new_position - last_position
            
            # If there's a change in position, record a trade
            if position_change != 0:
                self._execute_trade(timestamp, current_price, position_change, max_position_size)
            
            # Update position
            self.positions.loc[timestamp] = new_position
            
            # Calculate holdings value
            self.holdings.loc[timestamp] = new_position * current_price
            
            # Update cash (in first iteration, use initial capital)
            if i == 0:
                self.cash.loc[timestamp] = self.initial_capital - (position_change * current_price)
            else:
                self.cash.loc[timestamp] = self.cash.iloc[i-1] - (position_change * current_price)
            
            # Update last position
            last_position = new_position
    
    def _determine_position(self, signal, last_position):
        """
        Determine the new position based on signal and current position
        
        Parameters:
        -----------
        signal : int
            Signal value (1 for buy, -1 for sell, 0 for hold)
        last_position : float
            Current position
            
        Returns:
        --------
        float
            New position
        """
        if signal == 1:  # Buy signal
            return 1.0 if self.enable_fractional else 1
        elif signal == -1:  # Sell signal
            return -1.0 if self.enable_fractional else -1
        else:  # Hold signal
            return last_position
    
    def _execute_trade(self, timestamp, price, position_change, max_position_size):
        """
        Execute a trade and record the details
        
        Parameters:
        -----------
        timestamp : datetime
            Time of the trade
        price : float
            Execution price
        position_change : float
            Change in position
        max_position_size : float
            Maximum position size based on available capital
        """
        # Apply position size limit
        actual_position_change = np.sign(position_change) * min(abs(position_change), max_position_size)
        
        # Apply slippage to the execution price
        execution_price = price * (1 + np.sign(position_change) * self.slippage)
        
        # Calculate commission
        trade_value = abs(actual_position_change * execution_price)
        commission_amount = trade_value * self.commission
        
        # Record the trade
        trade = {
            'timestamp': timestamp,
            'price': price,
            'execution_price': execution_price,
            'position_change': actual_position_change,
            'trade_value': trade_value,
            'commission': commission_amount,
            'type': 'buy' if position_change > 0 else 'sell'
        }
        
        self.trades.append(trade)
        
        # Log trade
        logger.debug(f"Trade executed: {trade['type'].upper()} at {timestamp} - "
                    f"Price: ${execution_price:.2f}, Size: {abs(actual_position_change):.4f}, "
                    f"Value: ${trade_value:.2f}, Commission: ${commission_amount:.2f}")
    
    def _calculate_equity_curve(self):
        """
        Calculate the equity curve
        """
        for i, timestamp in enumerate(self.data.index):
            # Equity = Cash + Holdings
            self.equity.loc[timestamp] = self.cash.loc[timestamp] + self.holdings.loc[timestamp]
    
    def _calculate_metrics(self):
        """
        Calculate performance metrics
        """
        # Calculate metrics using the helper function
        self.metrics = calculate_performance_metrics(
            self.data['close'], self.positions, self.equity, self.trades)
    
    def print_results(self):
        """
        Print backtest results
        """
        logger.info("\n===== BACKTEST RESULTS =====")
        logger.info(f"Strategy: {self.strategy.name}")
        logger.info(f"Period: {self.data.index[0]} to {self.data.index[-1]}")
        logger.info(f"Initial Capital: ${self.initial_capital:.2f}")
        logger.info(f"Final Equity: ${self.equity.iloc[-1]:.2f}")
        
        profit = self.equity.iloc[-1] - self.initial_capital
        roi = profit / self.initial_capital * 100
        
        logger.info(f"Absolute Return: ${profit:.2f}")
        logger.info(f"Return on Investment: {roi:.2f}%")
        
        # Print trades summary
        num_trades = len(self.trades)
        winning_trades = sum(1 for trade in self.trades if trade['type'] == 'sell' and trade['price'] > trade['execution_price'])
        win_rate = winning_trades / num_trades if num_trades > 0 else 0
        
        logger.info(f"Number of Trades: {num_trades}")
        logger.info(f"Win Rate: {win_rate:.2%}")
        
        # Print other metrics
        logger.info(f"Sharpe Ratio: {self.metrics['sharpe_ratio']:.4f}")
        logger.info(f"Sortino Ratio: {self.metrics['sortino_ratio']:.4f}")
        logger.info(f"Max Drawdown: {self.metrics['max_drawdown']:.2%}")
        logger.info(f"Avg Trade Profit: ${self.metrics['avg_trade_profit']:.2f}")
        logger.info(f"Profit Factor: {self.metrics['profit_factor']:.4f}")
        logger.info(f"Calmar Ratio: {self.metrics['calmar_ratio']:.4f}")
        logger.info("============================\n")
    
    def plot_results(self, figsize=(15, 12)):
        """
        Plot backtest results
        
        Parameters:
        -----------
        figsize : tuple
            Figure size (width, height)
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object with plots
        """
        # Create figure with subplots
        fig, axes = plt.subplots(4, 1, figsize=figsize, sharex=True)
        
        # Plot price and positions
        axes[0].plot(self.data.index, self.data['close'], label='Price')
        axes[0].set_title('Price Chart')
        axes[0].set_ylabel('Price')
        axes[0].legend()
        
        # Plot buy/sell points
        buy_signals = self.positions.diff() > 0
        sell_signals = self.positions.diff() < 0
        
        axes[0].plot(self.data.index[buy_signals], self.data.loc[buy_signals, 'close'], '^', 
                    markersize=8, color='g', label='Buy')
        axes[0].plot(self.data.index[sell_signals], self.data.loc[sell_signals, 'close'], 'v', 
                    markersize=8, color='r', label='Sell')
        
        # Plot positions
        axes[1].plot(self.data.index, self.positions, label='Position')
        axes[1].set_title('Positions')
        axes[1].set_ylabel('Position')
        axes[1].legend()
        
        # Plot equity curve
        axes[2].plot(self.data.index, self.equity, label='Equity')
        axes[2].set_title('Equity Curve')
        axes[2].set_ylabel('Equity')
        axes[2].legend()
        
        # Plot drawdown
        peak = self.equity.cummax()
        drawdown = (self.equity - peak) / peak
        axes[3].fill_between(self.data.index, drawdown, 0, color='r', alpha=0.3, label='Drawdown')
        axes[3].set_title('Drawdown')
        axes[3].set_ylabel('Drawdown')
        axes[3].set_xlabel('Date')
        axes[3].legend()
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def get_trade_history(self):
        """
        Get trade history as a DataFrame
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with trade details
        """
        if not self.trades:
            logger.warning("No trades recorded")
            return pd.DataFrame()
        
        return pd.DataFrame(self.trades)
    
    def get_performance_summary(self):
        """
        Get performance summary as a DataFrame
        
        Returns:
        --------
        pandas.DataFrame
            DataFrame with performance metrics
        """
        metrics_df = pd.DataFrame({
            'Metric': [
                'Initial Capital',
                'Final Equity',
                'Absolute Return',
                'Return on Investment',
                'Annual Return',
                'Sharpe Ratio',
                'Sortino Ratio',
                'Max Drawdown',
                'Number of Trades',
                'Win Rate',
                'Avg Trade Profit',
                'Profit Factor',
                'Calmar Ratio'
            ],
            'Value': [
                f"${self.initial_capital:.2f}",
                f"${self.equity.iloc[-1]:.2f}",
                f"${self.equity.iloc[-1] - self.initial_capital:.2f}",
                f"{(self.equity.iloc[-1] / self.initial_capital - 1) * 100:.2f}%",
                f"{self.metrics['annual_return'] * 100:.2f}%",
                f"{self.metrics['sharpe_ratio']:.4f}",
                f"{self.metrics['sortino_ratio']:.4f}",
                f"{self.metrics['max_drawdown']:.2%}",
                f"{len(self.trades)}",
                f"{self.metrics['win_rate']:.2%}",
                f"${self.metrics['avg_trade_profit']:.2f}",
                f"{self.metrics['profit_factor']:.4f}",
                f"{self.metrics['calmar_ratio']:.4f}"
            ]
        })
        
        return metrics_df
    
    def compare_to_baseline(self, baseline_data=None):
        """
        Compare strategy performance to a baseline (e.g., buy and hold)
        
        Parameters:
        -----------
        baseline_data : pandas.Series, optional
            Series with baseline equity curve. If None, buy and hold is used.
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with comparison metrics
        """
        # If no baseline provided, use buy and hold
        if baseline_data is None:
            # Buy and hold strategy
            buy_hold_positions = pd.Series(1, index=self.data.index)
            buy_hold_equity = pd.Series(index=self.data.index)
            
            # Calculate buy and hold equity
            for i, timestamp in enumerate(self.data.index):
                if i == 0:
                    buy_price = self.data.loc[timestamp, 'close']
                    num_units = self.initial_capital / buy_price
                    buy_hold_equity.loc[timestamp] = self.initial_capital
                else:
                    current_price = self.data.loc[timestamp, 'close']
                    buy_hold_equity.loc[timestamp] = num_units * current_price
            
            baseline_data = buy_hold_equity
        
        # Calculate baseline metrics
        baseline_return = baseline_data.iloc[-1] / baseline_data.iloc[0] - 1
        
        # Calculate annualized returns
        days = (self.data.index[-1] - self.data.index[0]).days
        annualized_strategy = ((1 + (self.equity.iloc[-1] / self.initial_capital - 1)) ** (365 / days)) - 1
        annualized_baseline = ((1 + baseline_return) ** (365 / days)) - 1
        
        # Calculate outperformance
        outperformance = annualized_strategy - annualized_baseline
        
        # Calculate correlation
        strategy_returns = self.equity.pct_change().dropna()
        baseline_returns = baseline_data.pct_change().dropna()
        
        # Align the two series
        common_index = strategy_returns.index.intersection(baseline_returns.index)
        if len(common_index) > 0:
            strategy_returns = strategy_returns.loc[common_index]
            baseline_returns = baseline_returns.loc[common_index]
            correlation = strategy_returns.corr(baseline_returns)
        else:
            correlation = np.nan
        
        # Create comparison DataFrame
        comparison = pd.DataFrame({
            'Metric': [
                'Strategy Return',
                'Baseline Return',
                'Strategy Annualized Return',
                'Baseline Annualized Return',
                'Outperformance',
                'Correlation',
                'Strategy Sharpe Ratio',
                'Strategy Max Drawdown',
                'Baseline Max Drawdown'
            ],
            'Value': [
                f"{(self.equity.iloc[-1] / self.initial_capital - 1) * 100:.2f}%",
                f"{baseline_return * 100:.2f}%",
                f"{annualized_strategy * 100:.2f}%",
                f"{annualized_baseline * 100:.2f}%",
                f"{outperformance * 100:.2f}%",
                f"{correlation:.4f}",
                f"{self.metrics['sharpe_ratio']:.4f}",
                f"{self.metrics['max_drawdown']:.2%}",
                f"{(baseline_data.cummax() - baseline_data).max() / baseline_data.cummax().max():.2%}"
            ]
        })
        
        return comparison
