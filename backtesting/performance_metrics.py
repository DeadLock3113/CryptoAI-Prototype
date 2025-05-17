"""
Modulo per il calcolo delle metriche di performance nel backtesting.

Questo modulo contiene classi e funzioni per calcolare varie metriche
di performance sulle strategie di trading durante il backtesting.

Author: CryptoTradeAnalyzer Team
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Any, Optional, Union

class PerformanceMetrics:
    """
    Classe per il calcolo delle metriche di performance nel backtesting.
    """
    
    @staticmethod
    def calculate_return(entry_price: float, exit_price: float, position_type: str) -> float:
        """
        Calcola il ritorno percentuale di un trade.
        
        Args:
            entry_price: Prezzo di entrata
            exit_price: Prezzo di uscita
            position_type: Tipo di posizione ('long' o 'short')
            
        Returns:
            Ritorno percentuale
        """
        if position_type == 'long':
            return (exit_price - entry_price) / entry_price * 100
        elif position_type == 'short':
            return (entry_price - exit_price) / entry_price * 100
        else:
            raise ValueError(f"Tipo di posizione non valido: {position_type}")
    
    @staticmethod
    def calculate_drawdown(equity_curve: pd.Series) -> Tuple[float, float]:
        """
        Calcola il drawdown massimo.
        
        Args:
            equity_curve: Serie con la curva dell'equity
            
        Returns:
            Tuple (drawdown massimo percentuale, durata massima del drawdown)
        """
        # Calcolo del drawdown
        rolling_max = equity_curve.cummax()
        drawdown = (equity_curve / rolling_max - 1) * 100
        max_drawdown = drawdown.min()
        
        # Calcolo della durata del drawdown
        is_drawdown = drawdown < 0
        drawdown_starts = is_drawdown & ~is_drawdown.shift(1).fillna(False)
        drawdown_ends = ~is_drawdown & is_drawdown.shift(1).fillna(False)
        
        if not drawdown_starts.any() or not drawdown_ends.any():
            return max_drawdown, 0
        
        start_dates = drawdown_starts[drawdown_starts].index
        end_dates = drawdown_ends[drawdown_ends].index
        
        # Calcola la durata massima del drawdown
        max_duration = 0
        for start in start_dates:
            # Trova la fine corrispondente
            ends_after_start = end_dates[end_dates > start]
            if len(ends_after_start) > 0:
                end = ends_after_start[0]
                duration = (end - start).days
                max_duration = max(max_duration, duration)
        
        return max_drawdown, max_duration
    
    @staticmethod
    def calculate_sharpe_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
        """
        Calcola il Sharpe ratio.
        
        Args:
            returns: Serie con i rendimenti percentuali
            risk_free_rate: Tasso privo di rischio annualizzato
            periods_per_year: Numero di periodi in un anno
            
        Returns:
            Sharpe ratio
        """
        # Calcola i rendimenti in eccesso rispetto al tasso privo di rischio
        excess_returns = returns - risk_free_rate / periods_per_year
        
        # Calcola la media e la deviazione standard annualizzate
        mean_return = excess_returns.mean() * periods_per_year
        std_return = excess_returns.std() * np.sqrt(periods_per_year)
        
        # Calcola il Sharpe ratio
        if std_return == 0:
            return 0.0
        
        return mean_return / std_return
    
    @staticmethod
    def calculate_sortino_ratio(returns: pd.Series, risk_free_rate: float = 0.0, periods_per_year: int = 252) -> float:
        """
        Calcola il Sortino ratio.
        
        Args:
            returns: Serie con i rendimenti percentuali
            risk_free_rate: Tasso privo di rischio annualizzato
            periods_per_year: Numero di periodi in un anno
            
        Returns:
            Sortino ratio
        """
        # Calcola i rendimenti in eccesso rispetto al tasso privo di rischio
        excess_returns = returns - risk_free_rate / periods_per_year
        
        # Calcola la media annualizzata
        mean_return = excess_returns.mean() * periods_per_year
        
        # Calcola la deviazione standard dei rendimenti negativi
        negative_returns = excess_returns[excess_returns < 0]
        if len(negative_returns) == 0:
            return np.inf if mean_return > 0 else 0.0
        
        downside_std = negative_returns.std() * np.sqrt(periods_per_year)
        
        # Calcola il Sortino ratio
        if downside_std == 0:
            return 0.0
        
        return mean_return / downside_std
    
    @staticmethod
    def calculate_calmar_ratio(returns: pd.Series, equity_curve: pd.Series, periods_per_year: int = 252) -> float:
        """
        Calcola il Calmar ratio.
        
        Args:
            returns: Serie con i rendimenti percentuali
            equity_curve: Serie con la curva dell'equity
            periods_per_year: Numero di periodi in un anno
            
        Returns:
            Calmar ratio
        """
        # Calcola il rendimento annualizzato
        annual_return = returns.mean() * periods_per_year
        
        # Calcola il massimo drawdown (in percentuale positiva)
        max_drawdown, _ = PerformanceMetrics.calculate_drawdown(equity_curve)
        max_drawdown_positive = abs(max_drawdown)
        
        # Calcola il Calmar ratio
        if max_drawdown_positive == 0:
            return 0.0
        
        return annual_return / max_drawdown_positive
    
    @staticmethod
    def calculate_profit_factor(trades: pd.DataFrame) -> float:
        """
        Calcola il profit factor.
        
        Args:
            trades: DataFrame con i trade, deve contenere una colonna 'profit'
            
        Returns:
            Profit factor
        """
        if 'profit' not in trades.columns:
            raise ValueError("Il DataFrame dei trade deve contenere una colonna 'profit'")
        
        winning_trades = trades[trades['profit'] > 0]
        losing_trades = trades[trades['profit'] < 0]
        
        total_profit = winning_trades['profit'].sum() if len(winning_trades) > 0 else 0
        total_loss = abs(losing_trades['profit'].sum()) if len(losing_trades) > 0 else 0
        
        if total_loss == 0:
            return float('inf') if total_profit > 0 else 0.0
        
        return total_profit / total_loss
    
    @staticmethod
    def calculate_win_rate(trades: pd.DataFrame) -> float:
        """
        Calcola il win rate.
        
        Args:
            trades: DataFrame con i trade, deve contenere una colonna 'profit'
            
        Returns:
            Win rate
        """
        if 'profit' not in trades.columns:
            raise ValueError("Il DataFrame dei trade deve contenere una colonna 'profit'")
        
        total_trades = len(trades)
        if total_trades == 0:
            return 0.0
        
        winning_trades = len(trades[trades['profit'] > 0])
        
        return winning_trades / total_trades
    
    @staticmethod
    def calculate_average_profit(trades: pd.DataFrame) -> float:
        """
        Calcola il profitto medio per trade.
        
        Args:
            trades: DataFrame con i trade, deve contenere una colonna 'profit'
            
        Returns:
            Profitto medio per trade
        """
        if 'profit' not in trades.columns:
            raise ValueError("Il DataFrame dei trade deve contenere una colonna 'profit'")
        
        if len(trades) == 0:
            return 0.0
        
        return trades['profit'].mean()
    
    @staticmethod
    def calculate_average_win(trades: pd.DataFrame) -> float:
        """
        Calcola il profitto medio dei trade vincenti.
        
        Args:
            trades: DataFrame con i trade, deve contenere una colonna 'profit'
            
        Returns:
            Profitto medio dei trade vincenti
        """
        if 'profit' not in trades.columns:
            raise ValueError("Il DataFrame dei trade deve contenere una colonna 'profit'")
        
        winning_trades = trades[trades['profit'] > 0]
        
        if len(winning_trades) == 0:
            return 0.0
        
        return winning_trades['profit'].mean()
    
    @staticmethod
    def calculate_average_loss(trades: pd.DataFrame) -> float:
        """
        Calcola la perdita media dei trade perdenti.
        
        Args:
            trades: DataFrame con i trade, deve contenere una colonna 'profit'
            
        Returns:
            Perdita media dei trade perdenti
        """
        if 'profit' not in trades.columns:
            raise ValueError("Il DataFrame dei trade deve contenere una colonna 'profit'")
        
        losing_trades = trades[trades['profit'] < 0]
        
        if len(losing_trades) == 0:
            return 0.0
        
        return losing_trades['profit'].mean()
    
    @staticmethod
    def calculate_expectancy(trades: pd.DataFrame) -> float:
        """
        Calcola l'aspettativa matematica della strategia.
        
        Args:
            trades: DataFrame con i trade, deve contenere una colonna 'profit'
            
        Returns:
            Aspettativa matematica
        """
        if 'profit' not in trades.columns:
            raise ValueError("Il DataFrame dei trade deve contenere una colonna 'profit'")
        
        win_rate = PerformanceMetrics.calculate_win_rate(trades)
        avg_win = PerformanceMetrics.calculate_average_win(trades)
        avg_loss = PerformanceMetrics.calculate_average_loss(trades)
        
        return (win_rate * avg_win) + ((1 - win_rate) * avg_loss)
    
    @staticmethod
    def calculate_all_metrics(trades: pd.DataFrame, returns: pd.Series, equity_curve: pd.Series) -> Dict[str, float]:
        """
        Calcola tutte le metriche di performance.
        
        Args:
            trades: DataFrame con i trade
            returns: Serie con i rendimenti percentuali
            equity_curve: Serie con la curva dell'equity
            
        Returns:
            Dizionario con tutte le metriche
        """
        metrics = {}
        
        # Metriche basate sui trade
        metrics['profit_factor'] = PerformanceMetrics.calculate_profit_factor(trades)
        metrics['win_rate'] = PerformanceMetrics.calculate_win_rate(trades)
        metrics['avg_profit'] = PerformanceMetrics.calculate_average_profit(trades)
        metrics['avg_win'] = PerformanceMetrics.calculate_average_win(trades)
        metrics['avg_loss'] = PerformanceMetrics.calculate_average_loss(trades)
        metrics['expectancy'] = PerformanceMetrics.calculate_expectancy(trades)
        
        # Metriche basate sui rendimenti
        metrics['sharpe_ratio'] = PerformanceMetrics.calculate_sharpe_ratio(returns)
        metrics['sortino_ratio'] = PerformanceMetrics.calculate_sortino_ratio(returns)
        
        # Metriche basate sull'equity curve
        max_drawdown, max_drawdown_duration = PerformanceMetrics.calculate_drawdown(equity_curve)
        metrics['max_drawdown'] = max_drawdown
        metrics['max_drawdown_duration'] = max_drawdown_duration
        metrics['calmar_ratio'] = PerformanceMetrics.calculate_calmar_ratio(returns, equity_curve)
        
        return metrics