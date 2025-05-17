"""
Plotting Utilities Module for CryptoTradeAnalyzer

This module provides functions for generating visualizations of trading
strategy results, performance metrics, and market data.

Author: CryptoTradeAnalyzer Team
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import FuncFormatter
import logging
import os

logger = logging.getLogger(__name__)

def plot_price_with_signals(data, signals=None, figsize=(14, 7), 
                           buy_marker='^', sell_marker='v',
                           buy_color='green', sell_color='red',
                           title=None, save_path=None):
    """
    Plot price chart with buy/sell signals
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with OHLCV data
    signals : pandas.Series
        Series with signals (1 for buy, -1 for sell, 0 for hold)
    figsize : tuple
        Figure size (width, height)
    buy_marker : str
        Marker for buy signals
    sell_marker : str
        Marker for sell signals
    buy_color : str
        Color for buy signals
    sell_color : str
        Color for sell signals
    title : str
        Title for the plot
    save_path : str
        Path to save the figure
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object with the plot
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot price
    ax.plot(data.index, data['close'], label='Price', color='blue')
    
    # Plot signals if provided
    if signals is not None:
        # Find buy and sell signals
        buy_signals = signals == 1
        sell_signals = signals == -1
        
        # Plot buy signals
        if buy_signals.any():
            ax.scatter(data.index[buy_signals], data.loc[buy_signals, 'close'], 
                      marker=buy_marker, color=buy_color, s=100, label='Buy Signal')
        
        # Plot sell signals
        if sell_signals.any():
            ax.scatter(data.index[sell_signals], data.loc[sell_signals, 'close'], 
                      marker=sell_marker, color=sell_color, s=100, label='Sell Signal')
    
    # Format plot
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title('Price Chart with Signals', fontsize=14)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Format dates on x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    # Rotate date labels
    plt.xticks(rotation=45)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save figure: {str(e)}")
    
    return fig

def plot_equity_curve(equity, benchmark=None, figsize=(14, 7), 
                     title=None, save_path=None):
    """
    Plot equity curve
    
    Parameters:
    -----------
    equity : pandas.Series
        Equity curve
    benchmark : pandas.Series
        Benchmark equity curve for comparison
    figsize : tuple
        Figure size (width, height)
    title : str
        Title for the plot
    save_path : str
        Path to save the figure
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object with the plot
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot equity curve
    ax.plot(equity.index, equity, label='Strategy', color='blue')
    
    # Plot benchmark if provided
    if benchmark is not None:
        # Reindex benchmark to match equity index
        if not benchmark.index.equals(equity.index):
            benchmark = benchmark.reindex(equity.index, method='ffill')
        
        # Scale benchmark to start at the same value as equity
        benchmark = benchmark * (equity.iloc[0] / benchmark.iloc[0])
        
        # Plot benchmark
        ax.plot(benchmark.index, benchmark, label='Benchmark', color='gray', linestyle='--')
    
    # Format plot
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title('Equity Curve', fontsize=14)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Equity', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Format y-axis as currency
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'${x:,.0f}'))
    
    # Format dates on x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    # Rotate date labels
    plt.xticks(rotation=45)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save figure: {str(e)}")
    
    return fig

def plot_drawdown(equity, figsize=(14, 7), title=None, save_path=None):
    """
    Plot drawdown chart
    
    Parameters:
    -----------
    equity : pandas.Series
        Equity curve
    figsize : tuple
        Figure size (width, height)
    title : str
        Title for the plot
    save_path : str
        Path to save the figure
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object with the plot
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate drawdown
    peak = equity.cummax()
    drawdown = (equity - peak) / peak
    
    # Plot drawdown
    ax.fill_between(drawdown.index, drawdown, 0, color='red', alpha=0.3)
    ax.plot(drawdown.index, drawdown, color='red', label='Drawdown')
    
    # Format plot
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title('Drawdown', fontsize=14)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Drawdown', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Format y-axis as percentage
    ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1%}'))
    
    # Format dates on x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    # Rotate date labels
    plt.xticks(rotation=45)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save figure: {str(e)}")
    
    return fig

def plot_returns_distribution(returns, figsize=(14, 7), bins=50, 
                             title=None, save_path=None):
    """
    Plot returns distribution
    
    Parameters:
    -----------
    returns : pandas.Series
        Return series
    figsize : tuple
        Figure size (width, height)
    bins : int
        Number of bins for histogram
    title : str
        Title for the plot
    save_path : str
        Path to save the figure
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object with the plot
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot histogram
    ax.hist(returns, bins=bins, alpha=0.6, color='blue', edgecolor='black')
    
    # Plot normal distribution fit
    import scipy.stats as stats
    x = np.linspace(returns.min(), returns.max(), 100)
    mu, sigma = returns.mean(), returns.std()
    y = stats.norm.pdf(x, mu, sigma) * len(returns) * (returns.max() - returns.min()) / bins
    ax.plot(x, y, 'r--', linewidth=2, label=f'Normal: $\mu={mu:.4f}$, $\sigma={sigma:.4f}$')
    
    # Add mean and std lines
    ax.axvline(returns.mean(), color='red', linestyle='--', alpha=0.3)
    ax.axvline(returns.mean() + returns.std(), color='green', linestyle='--', alpha=0.3)
    ax.axvline(returns.mean() - returns.std(), color='green', linestyle='--', alpha=0.3)
    
    # Format plot
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title('Returns Distribution', fontsize=14)
    
    ax.set_xlabel('Return', fontsize=12)
    ax.set_ylabel('Frequency', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Format x-axis as percentage
    ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f'{x:.1%}'))
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save figure: {str(e)}")
    
    return fig

def plot_rolling_sharpe(returns, window=252, figsize=(14, 7), 
                       title=None, save_path=None):
    """
    Plot rolling Sharpe ratio
    
    Parameters:
    -----------
    returns : pandas.Series
        Return series
    window : int
        Rolling window size
    figsize : tuple
        Figure size (width, height)
    title : str
        Title for the plot
    save_path : str
        Path to save the figure
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object with the plot
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Calculate rolling Sharpe ratio
    rolling_mean = returns.rolling(window=window).mean()
    rolling_std = returns.rolling(window=window).std()
    rolling_sharpe = np.sqrt(252) * rolling_mean / rolling_std
    
    # Plot rolling Sharpe ratio
    ax.plot(rolling_sharpe.index, rolling_sharpe, label=f'{window}-day Rolling Sharpe')
    
    # Add zero line
    ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
    
    # Format plot
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f'{window}-day Rolling Sharpe Ratio', fontsize=14)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Sharpe Ratio', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Format dates on x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    # Rotate date labels
    plt.xticks(rotation=45)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save figure: {str(e)}")
    
    return fig

def plot_indicators(data, indicators=None, figsize=(14, 10), 
                   title=None, save_path=None):
    """
    Plot price chart with technical indicators
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with OHLCV data and indicators
    indicators : dict
        Dictionary with indicator names and configurations
        Format: {indicator_name: {'columns': [], 'panel': int}}
    figsize : tuple
        Figure size (width, height)
    title : str
        Title for the plot
    save_path : str
        Path to save the figure
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object with the plot
    """
    # Default indicators if none provided
    if indicators is None:
        indicators = {
            'Price': {
                'columns': ['close'],
                'panel': 0
            },
            'SMA': {
                'columns': ['sma_20', 'sma_50'],
                'panel': 0
            },
            'RSI': {
                'columns': ['rsi'],
                'panel': 1
            },
            'MACD': {
                'columns': ['macd', 'macd_signal', 'macd_hist'],
                'panel': 2
            }
        }
    
    # Check which indicators are available in the data
    available_indicators = {}
    for name, config in indicators.items():
        available_columns = [col for col in config['columns'] if col in data.columns]
        if available_columns:
            available_indicators[name] = {
                'columns': available_columns,
                'panel': config['panel']
            }
    
    if not available_indicators:
        logger.warning("No indicators available in the data")
        return None
    
    # Calculate number of panels needed
    num_panels = max([config['panel'] for config in available_indicators.values()]) + 1
    
    # Create figure with subplots
    fig, axes = plt.subplots(num_panels, 1, figsize=figsize, sharex=True)
    
    # Handle single panel case
    if num_panels == 1:
        axes = [axes]
    
    # Plot indicators on respective panels
    for name, config in available_indicators.items():
        panel = config['panel']
        ax = axes[panel]
        
        for column in config['columns']:
            # Skip if column not in data
            if column not in data.columns:
                continue
            
            # Plot column
            if column == 'macd_hist':
                # Plot histogram as bar chart
                ax.bar(data.index, data[column], alpha=0.5, label=column)
            else:
                ax.plot(data.index, data[column], label=column)
        
        # Add overbought/oversold lines for RSI
        if name == 'RSI':
            ax.axhline(70, color='red', linestyle='--', alpha=0.5)
            ax.axhline(30, color='green', linestyle='--', alpha=0.5)
        
        # Add zero line for MACD
        if name == 'MACD':
            ax.axhline(0, color='gray', linestyle='--', alpha=0.5)
        
        # Set panel title and legend
        ax.set_title(name)
        ax.legend(loc='upper left')
        ax.grid(True, alpha=0.3)
    
    # Set main title if provided
    if title:
        fig.suptitle(title, fontsize=14)
    
    # Format dates on x-axis
    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    axes[-1].set_xlabel('Date')
    
    # Rotate date labels
    plt.xticks(rotation=45)
    
    # Tight layout
    plt.tight_layout()
    
    # Adjust for suptitle
    if title:
        plt.subplots_adjust(top=0.95)
    
    # Save figure if path provided
    if save_path:
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save figure: {str(e)}")
    
    return fig

def plot_model_predictions(actual, predictions, model_name='Model', 
                          figsize=(14, 7), title=None, save_path=None):
    """
    Plot model predictions against actual values
    
    Parameters:
    -----------
    actual : pandas.Series
        Series with actual values
    predictions : pandas.Series
        Series with predicted values
    model_name : str
        Name of the model
    figsize : tuple
        Figure size (width, height)
    title : str
        Title for the plot
    save_path : str
        Path to save the figure
        
    Returns:
    --------
    matplotlib.figure.Figure
        Figure object with the plot
    """
    # Create figure
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot actual values
    ax.plot(actual.index, actual, label='Actual', color='blue')
    
    # Plot predictions
    ax.plot(predictions.index, predictions, label=f'{model_name} Predictions', 
           color='red', linestyle='--')
    
    # Format plot
    if title:
        ax.set_title(title, fontsize=14)
    else:
        ax.set_title(f'{model_name} Predictions vs Actual', fontsize=14)
    
    ax.set_xlabel('Date', fontsize=12)
    ax.set_ylabel('Price', fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend()
    
    # Format dates on x-axis
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
    
    # Rotate date labels
    plt.xticks(rotation=45)
    
    # Tight layout
    plt.tight_layout()
    
    # Save figure if path provided
    if save_path:
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            logger.info(f"Figure saved to {save_path}")
        except Exception as e:
            logger.error(f"Failed to save figure: {str(e)}")
    
    return fig

def plot_results(data, results=None, figsize=(15, 12), save_dir='plots'):
    """
    Generate and save all plots for a backtest
    
    Parameters:
    -----------
    data : pandas.DataFrame
        DataFrame with OHLCV data and indicators
    results : pandas.DataFrame
        DataFrame with backtest results
    figsize : tuple
        Figure size (width, height)
    save_dir : str
        Directory to save the figures
        
    Returns:
    --------
    list
        List of figure objects
    """
    logger.info("Generating plots")
    figures = []
    
    # Create save directory if it doesn't exist
    os.makedirs(save_dir, exist_ok=True)
    
    # Plot price chart with signals
    if results is not None and 'signal' in results.columns:
        fig_price = plot_price_with_signals(
            data, 
            signals=results['signal'],
            figsize=figsize,
            title='Price Chart with Signals',
            save_path=os.path.join(save_dir, 'price_signals.png')
        )
        figures.append(fig_price)
    
    # Plot equity curve
    if results is not None and 'equity' in results.columns:
        fig_equity = plot_equity_curve(
            results['equity'],
            figsize=figsize,
            title='Equity Curve',
            save_path=os.path.join(save_dir, 'equity_curve.png')
        )
        figures.append(fig_equity)
        
        # Plot drawdown
        fig_drawdown = plot_drawdown(
            results['equity'],
            figsize=figsize,
            title='Drawdown',
            save_path=os.path.join(save_dir, 'drawdown.png')
        )
        figures.append(fig_drawdown)
        
        # Plot returns distribution
        returns = results['equity'].pct_change().dropna()
        fig_returns = plot_returns_distribution(
            returns,
            figsize=figsize,
            title='Returns Distribution',
            save_path=os.path.join(save_dir, 'returns_distribution.png')
        )
        figures.append(fig_returns)
        
        # Plot rolling Sharpe ratio
        if len(returns) >= 30:  # Need at least 30 data points
            fig_sharpe = plot_rolling_sharpe(
                returns,
                window=min(252, len(returns) // 2),  # Adjust window size for small datasets
                figsize=figsize,
                title='Rolling Sharpe Ratio',
                save_path=os.path.join(save_dir, 'rolling_sharpe.png')
            )
            figures.append(fig_sharpe)
    
    # Plot indicators
    indicators = {
        'Price': {
            'columns': ['close'],
            'panel': 0
        }
    }
    
    # Check for common indicators in the data
    if 'sma_20' in data.columns and 'sma_50' in data.columns:
        indicators['SMA'] = {
            'columns': ['sma_20', 'sma_50'],
            'panel': 0
        }
    
    if 'rsi' in data.columns:
        indicators['RSI'] = {
            'columns': ['rsi'],
            'panel': 1
        }
    
    if 'macd' in data.columns and 'macd_signal' in data.columns:
        macd_cols = ['macd', 'macd_signal']
        if 'macd_hist' in data.columns:
            macd_cols.append('macd_hist')
        
        indicators['MACD'] = {
            'columns': macd_cols,
            'panel': 2
        }
    
    fig_indicators = plot_indicators(
        data,
        indicators=indicators,
        figsize=figsize,
        title='Technical Indicators',
        save_path=os.path.join(save_dir, 'indicators.png')
    )
    
    if fig_indicators:
        figures.append(fig_indicators)
    
    logger.info(f"Generated {len(figures)} plots in {save_dir}")
    
    return figures
