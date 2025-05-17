"""
Modulo per l'ottimizzazione delle strategie di trading in CryptoTradeAnalyzer.

Questo modulo contiene le classi e le funzioni per ottimizzare le 
strategie di trading utilizzando tecniche di machine learning.
L'ottimizzazione viene applicata solo quando può migliorare
l'accuratezza della strategia.

Author: CryptoTradeAnalyzer Team
"""

import pandas as pd
import numpy as np
import random
import math
import copy
import logging
from typing import Dict, List, Tuple, Any, Optional, Union
from datetime import datetime

from strategies.custom_strategy import CustomStrategy, TradingSignal, CustomStrategyBuilder
# Configurazione del logger
logger = logging.getLogger(__name__)

# Classe interna per le metriche di performance
class PerformanceMetrics:
    """
    Classe per il calcolo delle metriche di performance di una strategia.
    """
    
    @staticmethod
    def calculate_accuracy(signals, price_changes):
        """Calcola l'accuratezza dei segnali"""
        if len(signals) == 0:
            return 0.0
            
        correct_signals = ((signals == TradingSignal.BUY) & (price_changes > 0)) | \
                         ((signals == TradingSignal.SELL) & (price_changes < 0)) | \
                         ((signals == TradingSignal.HOLD) & (abs(price_changes) < 0.005))
        
        return float(correct_signals.mean() if len(correct_signals) > 0 else 0.0)
    
    @staticmethod
    def calculate_profit_factor(profits):
        """Calcola il profit factor dai profitti/perdite"""
        if not profits:
            return 0.0
            
        winning_trades = [p for p in profits if p > 0]
        losing_trades = [p for p in profits if p < 0]
        
        total_profit = sum(winning_trades) if winning_trades else 0
        total_loss = abs(sum(losing_trades)) if losing_trades else 0
        
        return float(total_profit / total_loss if total_loss > 0 else (1.0 if total_profit > 0 else 0.0))
    
    @staticmethod
    def calculate_sharpe_ratio(returns, risk_free_rate=0.0):
        """Calcola il Sharpe ratio dai rendimenti"""
        if not returns or len(returns) < 2:
            return 0.0
            
        mean_return = float(np.mean(returns))
        std_return = float(np.std(returns))
        
        if std_return == 0:
            return 0.0
            
        return float((mean_return - risk_free_rate) / std_return)


class StrategyOptimizer:
    """
    Classe per l'ottimizzazione delle strategie di trading.
    Utilizza algoritmi genetici e altre tecniche per migliorare
    le strategie esistenti, aggiornandole solo quando c'è un miglioramento
    misurabile dell'accuratezza e delle prestazioni.
    """
    
    def __init__(self, 
                 data: pd.DataFrame, 
                 strategy: CustomStrategy, 
                 train_test_split: float = 0.7,
                 population_size: int = 20,
                 generations: int = 10,
                 mutation_rate: float = 0.2,
                 crossover_rate: float = 0.7):
        """
        Inizializza l'ottimizzatore di strategie.
        
        Args:
            data: DataFrame con i dati di prezzo
            strategy: Strategia da ottimizzare
            train_test_split: Proporzione dei dati per training
            population_size: Dimensione della popolazione per l'algoritmo genetico
            generations: Numero di generazioni
            mutation_rate: Probabilità di mutazione
            crossover_rate: Probabilità di crossover
        """
        self.data = data
        self.original_strategy = strategy
        self.train_test_split = train_test_split
        self.population_size = population_size
        self.generations = generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        
        # Dividi i dati in training e testing
        split_idx = int(len(data) * train_test_split)
        self.train_data = data.iloc[:split_idx].copy()
        self.test_data = data.iloc[split_idx:].copy()
        
        # Prepara gli oggetti per tracciare le performance
        self.best_strategy = None
        self.best_fitness = -float('inf')
        self.improvement = None
        
        # Calcola le performance della strategia originale come benchmark
        self.original_performance = self._evaluate_strategy(strategy, self.test_data)
        logger.info(f"Performance originale: {self.original_performance}")
    
    def optimize(self) -> Tuple[CustomStrategy, Dict[str, float], bool]:
        """
        Ottimizza la strategia utilizzando un algoritmo genetico.
        L'aggiornamento viene applicato solo se c'è un miglioramento
        misurabile delle performance.
        
        Returns:
            Tupla (strategia ottimizzata, metriche di miglioramento, flag se migliorata)
        """
        # Genera la popolazione iniziale
        population = self._initialize_population()
        
        # Evolvi la popolazione per il numero specificato di generazioni
        for generation in range(self.generations):
            # Valuta il fitness di ciascuna strategia nella popolazione
            fitness_scores = [self._fitness_function(strategy) for strategy in population]
            
            # Traccia la strategia migliore di questa generazione
            best_idx = np.argmax(fitness_scores)
            current_best_strategy = population[best_idx]
            current_best_fitness = fitness_scores[best_idx]
            
            # Aggiorna la migliore strategia complessiva se necessario
            if current_best_fitness > self.best_fitness:
                self.best_strategy = copy.deepcopy(current_best_strategy)
                self.best_fitness = current_best_fitness
                logger.info(f"Nuova migliore strategia trovata nella generazione {generation + 1} con fitness {self.best_fitness}")
            
            # Seleziona i genitori per la riproduzione
            parents = self._select_parents(population, fitness_scores)
            
            # Crea una nuova popolazione attraverso crossover e mutazione
            new_population = []
            
            while len(new_population) < self.population_size:
                # Seleziona due genitori casuali
                parent1, parent2 = random.sample(parents, 2)
                
                # Applica crossover con una certa probabilità
                if random.random() < self.crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    # Se non c'è crossover, usa uno dei genitori
                    child = copy.deepcopy(random.choice([parent1, parent2]))
                
                # Applica mutazione con una certa probabilità
                if random.random() < self.mutation_rate:
                    child = self._mutate(child)
                
                new_population.append(child)
            
            # Sostituisci la vecchia popolazione con la nuova
            population = new_population
            
            logger.info(f"Generazione {generation + 1} completata. Miglior fitness: {self.best_fitness}")
        
        # Calcola le performance della migliore strategia trovata
        if self.best_strategy:
            best_performance = self._evaluate_strategy(self.best_strategy, self.test_data)
            
            # Calcola il miglioramento percentuale delle metriche chiave
            improvement = {}
            
            for metric in ["accuracy", "profit_factor", "sharpe_ratio"]:
                if metric in self.original_performance and metric in best_performance:
                    original_value = self.original_performance[metric]
                    new_value = best_performance[metric]
                    
                    # Evita divisione per zero
                    if original_value != 0:
                        pct_improvement = ((new_value - original_value) / abs(original_value)) * 100
                    else:
                        pct_improvement = float('inf') if new_value > 0 else 0
                    
                    improvement[metric] = pct_improvement
            
            # Determina se c'è stato un miglioramento significativo
            is_improved = (
                improvement.get("accuracy", 0) > 5 or  # +5% accuratezza
                improvement.get("profit_factor", 0) > 10 or  # +10% profit factor
                improvement.get("sharpe_ratio", 0) > 15  # +15% Sharpe ratio
            )
            
            if is_improved:
                logger.info(f"Strategia migliorata! Miglioramenti: {improvement}")
                self.improvement = improvement
                return self.best_strategy, improvement, True
            else:
                logger.info("Nessun miglioramento significativo trovato.")
                return self.original_strategy, improvement, False
        
        # Se non è stata trovata una strategia migliore, restituisci quella originale
        return self.original_strategy, {}, False
    
    def _initialize_population(self) -> List[CustomStrategy]:
        """
        Inizializza una popolazione di strategie con variazioni
        casuali dei parametri.
        
        Returns:
            Lista di strategie per la popolazione iniziale
        """
        population = []
        
        # Ottieni la descrizione dei parametri della strategia originale
        param_desc = self.original_strategy.get_parameters_description()
        current_params = self.original_strategy.parameters
        
        for _ in range(self.population_size):
            # Crea una copia della strategia originale
            strategy_type = type(self.original_strategy).__name__
            new_params = {}
            
            # Varia casualmente i parametri
            for param_name, param_info in param_desc.items():
                current_value = current_params.get(param_name, param_info.get("default"))
                
                if param_info.get("type") == "int":
                    min_val = param_info.get("min", current_value // 2)
                    max_val = param_info.get("max", current_value * 2)
                    new_value = random.randint(min_val, max_val)
                
                elif param_info.get("type") == "float":
                    min_val = param_info.get("min", current_value / 2)
                    max_val = param_info.get("max", current_value * 2)
                    new_value = random.uniform(min_val, max_val)
                
                elif param_info.get("type") == "select" and param_info.get("options"):
                    new_value = random.choice(param_info.get("options"))
                
                else:
                    new_value = current_value
                
                new_params[param_name] = new_value
            
            # Crea una nuova strategia con i parametri variati
            new_strategy = CustomStrategyBuilder.create_strategy(
                strategy_type=strategy_type,
                name=self.original_strategy.name,
                description=self.original_strategy.description,
                parameters=new_params
            )
            
            population.append(new_strategy)
        
        return population
    
    def _fitness_function(self, strategy: CustomStrategy) -> float:
        """
        Calcola il valore di fitness di una strategia sui dati di training.
        
        Args:
            strategy: Strategia da valutare
            
        Returns:
            Valore di fitness
        """
        performance = self._evaluate_strategy(strategy, self.train_data)
        
        # Calcola il fitness combinando diverse metriche
        fitness = (
            2.0 * performance.get("accuracy", 0) +  # Accuratezza ha peso 2
            1.5 * performance.get("profit_factor", 0) +  # Profit factor ha peso 1.5
            1.0 * performance.get("sharpe_ratio", 0) +  # Sharpe ratio ha peso 1
            0.5 * performance.get("win_rate", 0)  # Win rate ha peso 0.5
        )
        
        return fitness
    
    def _evaluate_strategy(self, strategy: CustomStrategy, data: pd.DataFrame) -> Dict[str, float]:
        """
        Valuta le performance di una strategia su un set di dati.
        
        Args:
            strategy: Strategia da valutare
            data: DataFrame con i dati su cui valutare la strategia
            
        Returns:
            Dizionario con le metriche di performance
        """
        # Genera i segnali per i dati forniti
        signals = strategy.generate_signals(data)
        
        # Calcola le performance utilizzando le funzioni dalla classe PerformanceMetrics
        metrics = {}
        
        # Accuratezza (rispetto a un trend semplice)
        price_changes = data['close'].pct_change().shift(-1)  # Cambio percentuale del prezzo futuro
        correct_signals = ((signals == TradingSignal.BUY) & (price_changes > 0)) | \
                         ((signals == TradingSignal.SELL) & (price_changes < 0)) | \
                         ((signals == TradingSignal.HOLD) & (abs(price_changes) < 0.005))
        
        metrics["accuracy"] = correct_signals.mean()
        
        # Simula i risultati di trading
        position = 0  # 1 = long, -1 = short, 0 = no position
        trades = []
        
        for i, signal in enumerate(signals):
            if i >= len(data) - 1:
                break
                
            price = data.iloc[i]['close']
            next_price = data.iloc[i+1]['close']
            
            if signal == TradingSignal.BUY and position <= 0:
                # Chiudi posizione short se presente
                if position == -1:
                    trades.append(price - entry_price)  # profit/loss
                
                # Apri long
                position = 1
                entry_price = price
                
            elif signal == TradingSignal.SELL and position >= 0:
                # Chiudi posizione long se presente
                if position == 1:
                    trades.append(price - entry_price)  # profit/loss
                
                # Apri short
                position = -1
                entry_price = price
        
        # Calcola metriche dai trade
        if trades:
            winning_trades = [t for t in trades if t > 0]
            losing_trades = [t for t in trades if t < 0]
            
            metrics["win_rate"] = len(winning_trades) / len(trades) if trades else 0
            
            total_profit = sum(winning_trades) if winning_trades else 0
            total_loss = abs(sum(losing_trades)) if losing_trades else 0
            
            metrics["profit_factor"] = total_profit / total_loss if total_loss > 0 else total_profit if total_profit > 0 else 0
            
            # Calcola Sharpe ratio (semplificato)
            returns = [t / entry_price for t in trades]
            avg_return = sum(returns) / len(returns)
            std_return = np.std(returns) if len(returns) > 1 else 1
            
            metrics["sharpe_ratio"] = avg_return / std_return if std_return > 0 else 0
        else:
            metrics["win_rate"] = 0
            metrics["profit_factor"] = 0
            metrics["sharpe_ratio"] = 0
        
        return metrics
    
    def _select_parents(self, population: List[CustomStrategy], fitness_scores: List[float]) -> List[CustomStrategy]:
        """
        Seleziona i genitori per la riproduzione in base al loro fitness.
        Utilizza una selezione a torneo.
        
        Args:
            population: Lista di strategie nella popolazione
            fitness_scores: Lista dei valori di fitness corrispondenti
            
        Returns:
            Lista di strategie selezionate come genitori
        """
        parents = []
        tournament_size = max(2, self.population_size // 5)
        
        # Normalizza i fitness per evitare valori negativi
        min_fitness = min(fitness_scores)
        normalized_fitness = [f - min_fitness + 1e-6 for f in fitness_scores]
        
        # Esegui la selezione a torneo
        for _ in range(self.population_size):
            # Seleziona un sottogruppo casuale
            candidates_idx = random.sample(range(len(population)), tournament_size)
            candidates_fitness = [normalized_fitness[i] for i in candidates_idx]
            
            # Seleziona il migliore del torneo
            best_candidate_idx = candidates_idx[np.argmax(candidates_fitness)]
            parents.append(population[best_candidate_idx])
        
        return parents
    
    def _crossover(self, parent1: CustomStrategy, parent2: CustomStrategy) -> CustomStrategy:
        """
        Esegue un crossover tra due strategie genitori per creare un figlio.
        
        Args:
            parent1: Prima strategia genitore
            parent2: Seconda strategia genitore
            
        Returns:
            Nuova strategia creata dal crossover
        """
        # Ottieni i parametri da entrambi i genitori
        params1 = parent1.parameters
        params2 = parent2.parameters
        
        # Crea un nuovo set di parametri combinando quelli dei genitori
        new_params = {}
        
        for param_name in params1.keys():
            # Scegli casualmente il parametro da uno dei genitori
            # oppure crea un valore intermedio per parametri numerici
            if random.random() < 0.5:
                new_params[param_name] = params1[param_name]
            else:
                new_params[param_name] = params2[param_name]
            
            # Per valori numerici, a volte crea una media pesata
            if isinstance(params1[param_name], (int, float)) and isinstance(params2[param_name], (int, float)):
                if random.random() < 0.3:  # 30% di probabilità di usare un valore intermedio
                    weight = random.random()  # Peso casuale
                    value = weight * params1[param_name] + (1 - weight) * params2[param_name]
                    
                    # Arrotonda a intero se necessario
                    if isinstance(params1[param_name], int):
                        value = int(round(value))
                    
                    new_params[param_name] = value
        
        # Crea una nuova strategia con i parametri combinati
        strategy_type = type(parent1).__name__
        return CustomStrategyBuilder.create_strategy(
            strategy_type=strategy_type,
            name=parent1.name,
            description=parent1.description,
            parameters=new_params
        )
    
    def _mutate(self, strategy: CustomStrategy) -> CustomStrategy:
        """
        Applica una mutazione alla strategia, modificando casualmente alcuni parametri.
        
        Args:
            strategy: Strategia da mutare
            
        Returns:
            Strategia mutata
        """
        # Ottieni la descrizione dei parametri e i valori attuali
        param_desc = strategy.get_parameters_description()
        current_params = strategy.parameters
        new_params = copy.deepcopy(current_params)
        
        # Seleziona casualmente alcuni parametri da mutare
        params_to_mutate = random.sample(
            list(current_params.keys()),
            k=max(1, int(len(current_params) * self.mutation_rate))
        )
        
        for param_name in params_to_mutate:
            param_info = param_desc.get(param_name, {})
            current_value = current_params[param_name]
            
            if param_info.get("type") == "int":
                # Muta parametri interi
                min_val = param_info.get("min", max(1, current_value // 2))
                max_val = param_info.get("max", current_value * 2)
                
                # Aggiungi o sottrai un valore casuale
                delta = random.randint(1, max(1, (max_val - min_val) // 5))
                if random.random() < 0.5:
                    delta = -delta
                
                new_value = max(min_val, min(max_val, current_value + delta))
                new_params[param_name] = new_value
                
            elif param_info.get("type") == "float":
                # Muta parametri float
                min_val = param_info.get("min", current_value / 2)
                max_val = param_info.get("max", current_value * 2)
                
                # Aggiungi o sottrai una percentuale casuale
                delta = random.uniform(0.05, 0.25) * current_value
                if random.random() < 0.5:
                    delta = -delta
                
                new_value = max(min_val, min(max_val, current_value + delta))
                new_params[param_name] = new_value
                
            elif param_info.get("type") == "select" and param_info.get("options"):
                # Seleziona casualmente un nuovo valore dalle opzioni
                options = [opt for opt in param_info.get("options") if opt != current_value]
                if options:
                    new_params[param_name] = random.choice(options)
        
        # Crea una nuova strategia con i parametri mutati
        strategy_type = type(strategy).__name__
        return CustomStrategyBuilder.create_strategy(
            strategy_type=strategy_type,
            name=strategy.name,
            description=strategy.description,
            parameters=new_params
        )


class BacktestPerformanceMetrics:
    """
    Classe helper per calcolare metriche di performance nel backtest.
    """
    
    @staticmethod
    def calculate_accuracy(signals: pd.Series, data: pd.DataFrame) -> float:
        """
        Calcola l'accuratezza dei segnali generati.
        
        Args:
            signals: Serie di segnali generati
            data: DataFrame con i dati di prezzo
            
        Returns:
            Accuratezza come percentuale
        """
        price_changes = data['close'].pct_change().shift(-1)
        correct_signals = ((signals == TradingSignal.BUY) & (price_changes > 0)) | \
                         ((signals == TradingSignal.SELL) & (price_changes < 0)) | \
                         ((signals == TradingSignal.HOLD) & (abs(price_changes) < 0.005))
        
        return correct_signals.mean() if len(correct_signals) > 0 else 0.0
    
    @staticmethod
    def calculate_profit_factor(trades: List[float]) -> float:
        """
        Calcola il profit factor dai trade.
        
        Args:
            trades: Lista di profitti/perdite dei trade
            
        Returns:
            Profit factor
        """
        winning_trades = [t for t in trades if t > 0]
        losing_trades = [t for t in trades if t < 0]
        
        total_profit = sum(winning_trades) if winning_trades else 0
        total_loss = abs(sum(losing_trades)) if losing_trades else 0
        
        return total_profit / total_loss if total_loss > 0 else float('inf') if total_profit > 0 else 0
    
    @staticmethod
    def calculate_sharpe_ratio(trades: List[float], risk_free_rate: float = 0.0) -> float:
        """
        Calcola il Sharpe ratio dai trade.
        
        Args:
            trades: Lista di profitti/perdite dei trade
            risk_free_rate: Tasso privo di rischio annualizzato
            
        Returns:
            Sharpe ratio
        """
        if not trades:
            return 0.0
        
        returns = trades
        avg_return = sum(returns) / len(returns)
        std_return = np.std(returns) if len(returns) > 1 else 1
        
        # Assumiamo che i rendimenti siano annualizzati
        excess_return = avg_return - risk_free_rate
        
        return excess_return / std_return if std_return > 0 else 0