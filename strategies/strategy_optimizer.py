"""
Modulo per l'ottimizzazione delle strategie di trading in CryptoTradeAnalyzer.

Questo modulo contiene le classi e le funzioni per ottimizzare le 
strategie di trading utilizzando tecniche di machine learning.
L'ottimizzazione viene applicata solo quando può migliorare
l'accuratezza della strategia.

Author: CryptoTradeAnalyzer Team
"""

import logging
import random
import numpy as np
import pandas as pd
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
        self.strategy = strategy
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
        logger.info(f"Avvio ottimizzazione della strategia {self.strategy.name}")
        
        # Inizializza la popolazione
        population = self._initialize_population()
        
        # Evoluzione della popolazione per n generazioni
        for gen in range(self.generations):
            logger.info(f"Generazione {gen + 1}/{self.generations}")
            
            # Calcola il fitness di ciascuna strategia nella popolazione
            fitness_scores = [self._fitness_function(strategy) for strategy in population]
            
            # Trova la migliore strategia di questa generazione
            best_idx = np.argmax(fitness_scores)
            current_best = population[best_idx]
            current_best_fitness = fitness_scores[best_idx]
            
            # Aggiorna la migliore strategia globale se necessario
            if current_best_fitness > self.best_fitness:
                self.best_fitness = current_best_fitness
                self.best_strategy = current_best
                logger.info(f"Nuova migliore strategia trovata (fitness: {self.best_fitness:.4f})")
            
            # Seleziona i genitori per la prossima generazione
            parents = self._select_parents(population, fitness_scores)
            
            # Crea la nuova popolazione
            new_population = []
            
            # Elitismo: mantieni la migliore strategia
            new_population.append(current_best)
            
            # Genera il resto della popolazione
            while len(new_population) < self.population_size:
                # Seleziona due genitori
                parent1 = random.choice(parents)
                parent2 = random.choice(parents)
                
                # Applica crossover con probabilità crossover_rate
                if random.random() < self.crossover_rate:
                    child = self._crossover(parent1, parent2)
                else:
                    child = parent1  # Se non viene fatto crossover, usa il primo genitore
                
                # Applica mutazione con probabilità mutation_rate
                if random.random() < self.mutation_rate:
                    child = self._mutate(child)
                
                new_population.append(child)
            
            # Sostituisci la vecchia popolazione con la nuova
            population = new_population
        
        # Se non è stata trovata una strategia migliore, usa quella originale
        if self.best_strategy is None:
            self.best_strategy = self.strategy
            logger.info("Nessuna strategia migliore trovata, utilizzo l'originale")
            return self.strategy, {}, False
        
        # Valuta le performance della miglior strategia sul test set
        best_performance = self._evaluate_strategy(self.best_strategy, self.test_data)
        
        # Calcola il miglioramento percentuale rispetto alla strategia originale
        improvements = {}
        is_improved = False
        
        for metric, value in best_performance.items():
            if metric in self.original_performance and self.original_performance[metric] > 0:
                improvement_pct = ((value - self.original_performance[metric]) / 
                                  self.original_performance[metric]) * 100
                improvements[metric] = improvement_pct
                
                # Se c'è un miglioramento significativo, aggiorna la strategia
                if improvement_pct > 5:  # 5% di miglioramento è considerato significativo
                    is_improved = True
        
        # Se non c'è un miglioramento significativo, ritorna la strategia originale
        if not is_improved:
            logger.info("Miglioramento non significativo, strategia originale mantenuta")
            return self.strategy, improvements, False
        
        logger.info(f"Strategia ottimizzata con successo: {improvements}")
        return self.best_strategy, improvements, True
    
    def _initialize_population(self) -> List[CustomStrategy]:
        """
        Inizializza una popolazione di strategie con variazioni
        casuali dei parametri.
        
        Returns:
            Lista di strategie per la popolazione iniziale
        """
        population = []
        
        # La prima strategia della popolazione è quella originale
        population.append(self.strategy)
        
        # Ottieni informazioni sui parametri
        param_info = self.strategy.get_parameter_info() if hasattr(self.strategy, 'get_parameter_info') else {}
        
        # Genera il resto della popolazione
        for _ in range(self.population_size - 1):
            # Crea una copia dei parametri originali
            new_params = self.strategy.parameters.copy()
            
            # Modifica casualmente alcuni parametri
            for param_name, info in param_info.items():
                # Probabilità di mutare ciascun parametro
                if random.random() < 0.5:
                    continue
                
                # Gestisci diversi tipi di parametri
                if info['type'] == 'int':
                    min_val = info.get('min', 1)
                    max_val = info.get('max', 100)
                    new_params[param_name] = random.randint(min_val, max_val)
                
                elif info['type'] == 'float':
                    min_val = info.get('min', 0.0)
                    max_val = info.get('max', 1.0)
                    step = info.get('step', 0.1)
                    steps = int((max_val - min_val) / step)
                    new_params[param_name] = min_val + random.randint(0, steps) * step
                
                elif info['type'] == 'select' and 'options' in info:
                    new_params[param_name] = random.choice(info['options'])
            
            # Crea una nuova strategia con i parametri modificati
            new_strategy = CustomStrategyBuilder.create_strategy(
                strategy_type=self.strategy.__class__.__name__,
                name=self.strategy.name,
                description=self.strategy.description,
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
        # Valuta la strategia sui dati di training
        performance = self._evaluate_strategy(strategy, self.train_data)
        
        # Calcola il valore di fitness come combinazione delle metriche
        fitness = (
            performance.get('accuracy', 0) * 0.3 +
            performance.get('profit_factor', 0) * 0.3 +
            performance.get('sharpe_ratio', 0) * 0.2 +
            performance.get('win_rate', 0) * 0.2
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
        try:
            # Genera i segnali
            signals = strategy.generate_signals(data)
            
            # Calcola le variazioni di prezzo
            price_changes = data['close'].pct_change().shift(-1)
            
            # Calcola l'accuratezza dei segnali
            accuracy = PerformanceMetrics.calculate_accuracy(signals, price_changes)
            
            # Simula il trading
            position = 0  # 0 = no position, 1 = long, -1 = short
            trades = []
            entry_price = 0
            
            for i, signal in enumerate(signals):
                if i >= len(data) - 1:
                    break
                    
                current_price = data['close'].iloc[i]
                next_price = data['close'].iloc[i+1]
                
                if signal == TradingSignal.BUY and position <= 0:
                    # Chiudi short se presente
                    if position == -1:
                        trades.append(entry_price - current_price)
                    
                    # Apri long
                    position = 1
                    entry_price = current_price
                    
                elif signal == TradingSignal.SELL and position >= 0:
                    # Chiudi long se presente
                    if position == 1:
                        trades.append(current_price - entry_price)
                    
                    # Apri short
                    position = -1
                    entry_price = current_price
            
            # Chiudi l'ultima posizione
            if position == 1:
                trades.append(data['close'].iloc[-1] - entry_price)
            elif position == -1:
                trades.append(entry_price - data['close'].iloc[-1])
            
            # Calcola le metriche di performance
            # Win rate
            winning_trades = [t for t in trades if t > 0]
            win_rate = len(winning_trades) / len(trades) if trades else 0
            
            # Profit factor
            profit_factor = PerformanceMetrics.calculate_profit_factor(trades)
            
            # Sharpe ratio
            returns = [t / entry_price for t in trades] if entry_price > 0 else []
            sharpe_ratio = PerformanceMetrics.calculate_sharpe_ratio(returns)
            
            return {
                'accuracy': float(accuracy),
                'win_rate': float(win_rate),
                'profit_factor': float(profit_factor),
                'sharpe_ratio': float(sharpe_ratio),
                'trades_count': len(trades)
            }
            
        except Exception as e:
            logger.error(f"Errore durante la valutazione della strategia: {e}")
            return {
                'accuracy': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'sharpe_ratio': 0.0,
                'trades_count': 0
            }
    
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
        
        # Seleziona i genitori con tornei
        for _ in range(self.population_size):
            # Seleziona casualmente due candidati per il torneo
            idx1, idx2 = random.sample(range(len(population)), 2)
            
            # Il candidato con il fitness più alto vince
            if fitness_scores[idx1] > fitness_scores[idx2]:
                parents.append(population[idx1])
            else:
                parents.append(population[idx2])
        
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
        # Crea una copia dei parametri
        child_params = {}
        
        # Ottieni le informazioni sui parametri
        param_info = parent1.get_parameter_info()
        
        # Per ogni parametro, scegli casualmente da quale genitore prenderlo
        for param_name in param_info.keys():
            if random.random() < 0.5:
                child_params[param_name] = parent1.parameters.get(param_name)
            else:
                child_params[param_name] = parent2.parameters.get(param_name)
        
        # Crea una nuova strategia con i parametri del figlio
        child = CustomStrategyBuilder.create_strategy(
            strategy_type=parent1.__class__.__name__,
            name=parent1.name,
            description=parent1.description,
            parameters=child_params
        )
        
        return child
    
    def _mutate(self, strategy: CustomStrategy) -> CustomStrategy:
        """
        Applica una mutazione alla strategia, modificando casualmente alcuni parametri.
        
        Args:
            strategy: Strategia da mutare
            
        Returns:
            Strategia mutata
        """
        # Crea una copia dei parametri
        new_params = strategy.parameters.copy()
        
        # Ottieni informazioni sui parametri
        param_info = strategy.get_parameter_info()
        
        # Seleziona casualmente un parametro da mutare
        param_to_mutate = random.choice(list(param_info.keys()))
        info = param_info[param_to_mutate]
        
        # Muta il parametro in base al suo tipo
        if info['type'] == 'int':
            min_val = info.get('min', 1)
            max_val = info.get('max', 100)
            
            # Genera un nuovo valore che è diverso dal precedente
            current_val = new_params[param_to_mutate]
            new_val = current_val
            
            while new_val == current_val:
                new_val = random.randint(min_val, max_val)
            
            new_params[param_to_mutate] = new_val
        
        elif info['type'] == 'float':
            min_val = info.get('min', 0.0)
            max_val = info.get('max', 1.0)
            step = info.get('step', 0.1)
            
            # Genera un nuovo valore che è diverso dal precedente
            current_val = new_params[param_to_mutate]
            new_val = current_val
            
            while new_val == current_val:
                steps = int((max_val - min_val) / step)
                new_val = min_val + random.randint(0, steps) * step
            
            new_params[param_to_mutate] = new_val
        
        elif info['type'] == 'select' and 'options' in info:
            # Scegli un'opzione diversa da quella corrente
            options = info['options'].copy()
            current_val = new_params[param_to_mutate]
            
            if len(options) > 1:
                options.remove(current_val)
                new_params[param_to_mutate] = random.choice(options)
        
        # Crea una nuova strategia con i parametri mutati
        mutated = CustomStrategyBuilder.create_strategy(
            strategy_type=strategy.__class__.__name__,
            name=strategy.name,
            description=strategy.description,
            parameters=new_params
        )
        
        return mutated