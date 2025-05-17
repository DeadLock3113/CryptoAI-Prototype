"""
Performance Metrics Module for CryptoTradeAnalyzer

This module calculates various performance metrics for evaluating
trading strategies, including risk-adjusted returns, drawdowns,
win rates, and other statistical measures.

Author: CryptoTradeAnalyzer Team
"""

import pandas as pd
import numpy as np
import logging

logger = logging.getLogger(__name__)

def calculate_returns(equity):
    """
    Calculate return series from equity curve
    
    Parameters:
    -----------
    equity : pandas.Series
        Equity curve
        
    Returns:
    --------
    pandas.Series
        Return series
    """
    return equity.pct_change().fillna(0)

def calculate_drawdown(equity):
    """
    Calculate drawdown series from equity curve
    
    Parameters:
    -----------
    equity : pandas.Series
        Equity curve
        
    Returns:
    --------
    pandas.Series
        Drawdown series
    """
    # Calculate running maximum
    running_max = equity.cummax()
    
    # Calculate drawdown
    drawdown = (equity - running_max) / running_max
    
    return drawdown

def calculate_sharpe_ratio(returns, risk_free_rate=0, periods_per_year=252):
    """
    Calculate Sharpe ratio
    
    Parameters:
    -----------
    returns : pandas.Series
        Return series
    risk_free_rate : float
        Annualized risk-free rate
    periods_per_year : int
        Number of periods in a year (252 for trading days)
        
    Returns:
    --------
    float
        Sharpe ratio
    """
    # Convert risk-free rate to per-period
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    
    # Calculate excess returns
    excess_returns = returns - rf_per_period
    
    # Calculate Sharpe ratio
    if len(excess_returns) < 2:
        return 0
    
    sharpe = np.sqrt(periods_per_year) * excess_returns.mean() / excess_returns.std()
    
    return sharpe

def calculate_sortino_ratio(returns, risk_free_rate=0, periods_per_year=252):
    """
    Calculate Sortino ratio
    
    Parameters:
    -----------
    returns : pandas.Series
        Return series
    risk_free_rate : float
        Annualized risk-free rate
    periods_per_year : int
        Number of periods in a year (252 for trading days)
        
    Returns:
    --------
    float
        Sortino ratio
    """
    # Convert risk-free rate to per-period
    rf_per_period = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
    
    # Calculate excess returns
    excess_returns = returns - rf_per_period
    
    # Calculate downside deviation (only negative returns)
    downside_returns = excess_returns[excess_returns < 0]
    
    if len(downside_returns) < 2:
        return 0
    
    downside_deviation = np.sqrt(np.sum(downside_returns ** 2) / len(downside_returns))
    
    # Handle case of no negative returns
    if downside_deviation == 0:
        return np.inf
    
    # Calculate Sortino ratio
    sortino = np.sqrt(periods_per_year) * excess_returns.mean() / downside_deviation
    
    return sortino

def calculate_calmar_ratio(returns, max_drawdown, periods_per_year=252):
    """
    Calculate Calmar ratio
    
    Parameters:
    -----------
    returns : pandas.Series
        Return series
    max_drawdown : float
        Maximum drawdown (positive value)
    periods_per_year : int
        Number of periods in a year (252 for trading days)
        
    Returns:
    --------
    float
        Calmar ratio
    """
    # Calculate annualized return
    annualized_return = (1 + returns.mean()) ** periods_per_year - 1
    
    # Avoid division by zero
    if max_drawdown == 0:
        return np.inf
    
    # Calculate Calmar ratio
    calmar = annualized_return / abs(max_drawdown)
    
    return calmar

def calculate_win_rate(trades):
    """
    Calculate win rate from trades
    
    Parameters:
    -----------
    trades : list
        List of trade dictionaries
        
    Returns:
    --------
    float
        Win rate
    """
    if not trades:
        return 0
    
    # Count winning trades
    winning_trades = sum(1 for trade in trades 
                         if (trade['type'] == 'sell' and trade['price'] > trade['execution_price']) or
                            (trade['type'] == 'buy' and trade['price'] < trade['execution_price']))
    
    # Calculate win rate
    win_rate = winning_trades / len(trades)
    
    return win_rate

def calculate_profit_factor(trades):
    """
    Calculate profit factor from trades
    
    Parameters:
    -----------
    trades : list
        List of trade dictionaries
        
    Returns:
    --------
    float
        Profit factor
    """
    if not trades:
        return 0
    
    # Sum profits and losses
    gross_profit = sum(abs(trade['position_change'] * trade['price']) 
                      for trade in trades 
                      if (trade['type'] == 'sell' and trade['price'] > trade['execution_price']) or
                         (trade['type'] == 'buy' and trade['price'] < trade['execution_price']))
    
    gross_loss = sum(abs(trade['position_change'] * trade['price']) 
                    for trade in trades 
                    if (trade['type'] == 'sell' and trade['price'] < trade['execution_price']) or
                       (trade['type'] == 'buy' and trade['price'] > trade['execution_price']))
    
    # Calculate profit factor
    if gross_loss == 0:
        return np.inf
    
    profit_factor = gross_profit / (gross_loss if gross_loss > 0 else 1)
    
    return profit_factor

def calculate_avg_trade_profit(trades, equity):
    """
    Calculate average profit per trade
    
    Parameters:
    -----------
    trades : list
        List of trade dictionaries
    equity : pandas.Series
        Equity curve
        
    Returns:
    --------
    float
        Average profit per trade
    """
    if not trades:
        return 0
    
    # Calculate total profit
    total_profit = equity.iloc[-1] - equity.iloc[0]
    
    # Calculate average profit per trade
    avg_profit = total_profit / len(trades)
    
    return avg_profit

def calculate_performance_metrics(prices, positions, equity, trades, risk_free_rate=0, periods_per_year=252):
    """
    Calculate all performance metrics
    
    Parameters:
    -----------
    prices : pandas.Series
        Price series
    positions : pandas.Series
        Position series
    equity : pandas.Series
        Equity curve
    trades : list
        List of trade dictionaries
    risk_free_rate : float
        Annualized risk-free rate
    periods_per_year : int
        Number of periods in a year (252 for trading days)
        
    Returns:
    --------
    dict
        Dictionary with all metrics
    """
    # Calculate returns
    returns = calculate_returns(equity)
    
    # Calculate drawdown
    drawdown = calculate_drawdown(equity)
    max_drawdown = abs(drawdown.min())
    
    # Calculate Sharpe ratio
    sharpe_ratio = calculate_sharpe_ratio(returns, risk_free_rate, periods_per_year)
    
    # Calculate Sortino ratio
    sortino_ratio = calculate_sortino_ratio(returns, risk_free_rate, periods_per_year)
    
    # Calculate win rate
    win_rate = calculate_win_rate(trades)
    
    # Calculate profit factor
    profit_factor = calculate_profit_factor(trades)
    
    # Calculate average trade profit
    avg_trade_profit = calculate_avg_trade_profit(trades, equity)
    
    # Calculate annualized return
    total_return = equity.iloc[-1] / equity.iloc[0] - 1
    days = (equity.index[-1] - equity.index[0]).days
    annual_return = ((1 + total_return) ** (365 / max(days, 1))) - 1
    
    # Calculate Calmar ratio
    calmar_ratio = calculate_calmar_ratio(returns, max_drawdown, periods_per_year)
    
    # Return all metrics
    metrics = {
        'total_return': total_return,
        'annual_return': annual_return,
        'sharpe_ratio': sharpe_ratio,
        'sortino_ratio': sortino_ratio,
        'max_drawdown': max_drawdown,
        'win_rate': win_rate,
        'profit_factor': profit_factor,
        'avg_trade_profit': avg_trade_profit,
        'calmar_ratio': calmar_ratio,
        'num_trades': len(trades)
    }
    
    return metrics
