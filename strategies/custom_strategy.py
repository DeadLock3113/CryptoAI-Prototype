"""
Modulo per le strategie di trading personalizzate in CryptoTradeAnalyzer.

Questo modulo consente agli utenti di creare, salvare e testare strategie
di trading personalizzate con parametri configurabili.

Author: CryptoTradeAnalyzer Team
"""
import os
import json
import logging
import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any, Callable

# Setup logging
logger = logging.getLogger(__name__)

class TradingSignal:
    """Classe per rappresentare un segnale di trading."""
    BUY = 1
    SELL = -1
    HOLD = 0
    
    @staticmethod
    def to_string(signal: int) -> str:
        """Converte un segnale numerico in stringa."""
        if signal == TradingSignal.BUY:
            return "ACQUISTO"
        elif signal == TradingSignal.SELL:
            return "VENDITA"
        else:
            return "MANTIENI"
    
    @staticmethod
    def get_color(signal: int) -> str:
        """Restituisce il colore associato al segnale."""
        if signal == TradingSignal.BUY:
            return "success"
        elif signal == TradingSignal.SELL:
            return "danger"
        else:
            return "secondary"

class CustomStrategy:
    """
    Classe base per le strategie di trading personalizzate.
    """
    
    def __init__(self, name: str, description: str = "", parameters: Dict[str, Any] = None):
        """
        Inizializza una nuova strategia personalizzata.
        
        Args:
            name: Nome della strategia
            description: Descrizione della strategia
            parameters: Parametri configurabili della strategia
        """
        self.name = name
        self.description = description
        self.parameters = parameters or {}
        self.signals = []
        self.last_update = None
        
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Genera i segnali di trading in base alla strategia.
        Da sovrascrivere nelle classi derivate.
        
        Args:
            data: DataFrame con i dati di prezzo
            
        Returns:
            Serie con i segnali di trading
        """
        # Implementazione di base che non genera segnali
        signals = pd.Series(index=data.index, data=TradingSignal.HOLD)
        self.signals = signals
        self.last_update = datetime.datetime.now()
        return signals
    
    def get_parameters_description(self) -> Dict[str, Any]:
        """
        Restituisce la descrizione dei parametri della strategia.
        Da sovrascrivere nelle classi derivate.
        
        Returns:
            Dizionario con le descrizioni dei parametri
        """
        return {}
    
    def get_strategy_info(self) -> Dict[str, Any]:
        """
        Restituisce le informazioni sulla strategia.
        
        Returns:
            Dizionario con le informazioni
        """
        return {
            'name': self.name,
            'description': self.description,
            'parameters': self.parameters,
            'parameters_description': self.get_parameters_description(),
            'last_update': self.last_update.isoformat() if self.last_update else None
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """
        Converte la strategia in un dizionario.
        
        Returns:
            Dizionario con i dati della strategia
        """
        return {
            'name': self.name,
            'description': self.description,
            'parameters': self.parameters,
            'type': self.__class__.__name__
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CustomStrategy':
        """
        Crea una strategia da un dizionario.
        
        Args:
            data: Dizionario con i dati della strategia
            
        Returns:
            Istanza della strategia
        """
        strategy = cls(
            name=data.get('name', 'Strategia personalizzata'),
            description=data.get('description', ''),
            parameters=data.get('parameters', {})
        )
        return strategy

class MovingAverageCrossStrategy(CustomStrategy):
    """
    Strategia basata sull'incrocio di medie mobili.
    Genera segnali di acquisto quando la media breve supera la media lunga,
    e segnali di vendita quando la media breve scende sotto la media lunga.
    """
    
    def __init__(self, name: str = "Incrocio Medie Mobili", 
               description: str = "Strategia basata sull'incrocio di medie mobili", 
               parameters: Dict[str, Any] = None):
        """
        Inizializza la strategia con parametri predefiniti se non specificati.
        
        Args:
            name: Nome della strategia
            description: Descrizione della strategia
            parameters: Parametri della strategia, inclusi i periodi delle medie mobili
        """
        if parameters is None:
            parameters = {
                'short_window': 20,
                'long_window': 50,
                'price_column': 'close'
            }
        super().__init__(name, description, parameters)
    
    def get_parameters_description(self) -> Dict[str, Any]:
        """
        Restituisce la descrizione dei parametri della strategia.
        
        Returns:
            Dizionario con le descrizioni dei parametri
        """
        return {
            'short_window': {
                'name': 'Periodo Media Mobile Breve',
                'description': 'Numero di periodi per la media mobile a breve termine',
                'type': 'int',
                'min': 5,
                'max': 100,
                'default': 20
            },
            'long_window': {
                'name': 'Periodo Media Mobile Lunga',
                'description': 'Numero di periodi per la media mobile a lungo termine',
                'type': 'int',
                'min': 20,
                'max': 200,
                'default': 50
            },
            'price_column': {
                'name': 'Colonna Prezzo',
                'description': 'Colonna da utilizzare per il calcolo delle medie mobili',
                'type': 'select',
                'options': ['open', 'high', 'low', 'close'],
                'default': 'close'
            }
        }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Genera segnali di trading basati sull'incrocio di medie mobili.
        
        Args:
            data: DataFrame con i dati di prezzo
            
        Returns:
            Serie con i segnali di trading
        """
        if data.empty:
            return pd.Series(index=data.index, data=TradingSignal.HOLD)
        
        # Estrai i parametri
        short_window = self.parameters.get('short_window', 20)
        long_window = self.parameters.get('long_window', 50)
        price_column = self.parameters.get('price_column', 'close')
        
        # Verifica che la colonna del prezzo esista
        if price_column not in data.columns:
            logger.warning(f"Colonna {price_column} non trovata nei dati. Utilizzando 'close'.")
            price_column = 'close'
        
        # Assicurati che ci siano abbastanza dati
        if len(data) < max(short_window, long_window):
            logger.warning("Non ci sono abbastanza dati per calcolare le medie mobili.")
            return pd.Series(index=data.index, data=TradingSignal.HOLD)
        
        # Calcola le medie mobili
        short_ma = data[price_column].rolling(window=short_window).mean()
        long_ma = data[price_column].rolling(window=long_window).mean()
        
        # Inizializza i segnali a HOLD
        signals = pd.Series(index=data.index, data=TradingSignal.HOLD)
        
        # Genera segnali di acquisto/vendita
        signals[short_ma > long_ma] = TradingSignal.BUY  # Segnale di acquisto
        signals[short_ma < long_ma] = TradingSignal.SELL  # Segnale di vendita
        
        # Salva i segnali e aggiorna il timestamp
        self.signals = signals
        self.last_update = datetime.datetime.now()
        
        return signals

class BollingerBandsStrategy(CustomStrategy):
    """
    Strategia basata sulle bande di Bollinger.
    Genera segnali di acquisto quando il prezzo tocca la banda inferiore,
    e segnali di vendita quando il prezzo tocca la banda superiore.
    """
    
    def __init__(self, name: str = "Strategia Bande di Bollinger", 
               description: str = "Strategia basata sulle bande di Bollinger", 
               parameters: Dict[str, Any] = None):
        """
        Inizializza la strategia con parametri predefiniti se non specificati.
        
        Args:
            name: Nome della strategia
            description: Descrizione della strategia
            parameters: Parametri della strategia
        """
        if parameters is None:
            parameters = {
                'window': 20,
                'num_std': 2,
                'price_column': 'close'
            }
        super().__init__(name, description, parameters)
    
    def get_parameters_description(self) -> Dict[str, Any]:
        """
        Restituisce la descrizione dei parametri della strategia.
        
        Returns:
            Dizionario con le descrizioni dei parametri
        """
        return {
            'window': {
                'name': 'Periodo',
                'description': 'Numero di periodi per il calcolo della media mobile',
                'type': 'int',
                'min': 5,
                'max': 100,
                'default': 20
            },
            'num_std': {
                'name': 'Deviazioni Standard',
                'description': 'Numero di deviazioni standard per le bande',
                'type': 'float',
                'min': 0.5,
                'max': 4.0,
                'step': 0.1,
                'default': 2.0
            },
            'price_column': {
                'name': 'Colonna Prezzo',
                'description': 'Colonna da utilizzare per il calcolo delle bande',
                'type': 'select',
                'options': ['open', 'high', 'low', 'close'],
                'default': 'close'
            }
        }
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Genera segnali di trading basati sulle bande di Bollinger.
        
        Args:
            data: DataFrame con i dati di prezzo
            
        Returns:
            Serie con i segnali di trading
        """
        if data.empty:
            return pd.Series(index=data.index, data=TradingSignal.HOLD)
        
        # Estrai i parametri
        window = self.parameters.get('window', 20)
        num_std = self.parameters.get('num_std', 2.0)
        price_column = self.parameters.get('price_column', 'close')
        
        # Verifica che la colonna del prezzo esista
        if price_column not in data.columns:
            logger.warning(f"Colonna {price_column} non trovata nei dati. Utilizzando 'close'.")
            price_column = 'close'
        
        # Assicurati che ci siano abbastanza dati
        if len(data) < window:
            logger.warning("Non ci sono abbastanza dati per calcolare le bande di Bollinger.")
            return pd.Series(index=data.index, data=TradingSignal.HOLD)
        
        # Calcola le bande di Bollinger
        rolling_mean = data[price_column].rolling(window=window).mean()
        rolling_std = data[price_column].rolling(window=window).std()
        
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        
        # Inizializza i segnali a HOLD
        signals = pd.Series(index=data.index, data=TradingSignal.HOLD)
        
        # Genera segnali di acquisto/vendita
        signals[data[price_column] <= lower_band] = TradingSignal.BUY  # Segnale di acquisto
        signals[data[price_column] >= upper_band] = TradingSignal.SELL  # Segnale di vendita
        
        # Salva i segnali e aggiorna il timestamp
        self.signals = signals
        self.last_update = datetime.datetime.now()
        
        return signals

class RSIStrategy(CustomStrategy):
    """
    Strategia basata sull'indicatore RSI (Relative Strength Index).
    Genera segnali di acquisto quando l'RSI è sotto la soglia di ipervenduto,
    e segnali di vendita quando l'RSI è sopra la soglia di ipercomprato.
    """
    
    def __init__(self, name: str = "Strategia RSI", 
               description: str = "Strategia basata sull'indicatore RSI", 
               parameters: Dict[str, Any] = None):
        """
        Inizializza la strategia con parametri predefiniti se non specificati.
        
        Args:
            name: Nome della strategia
            description: Descrizione della strategia
            parameters: Parametri della strategia
        """
        if parameters is None:
            parameters = {
                'window': 14,
                'overbought': 70,
                'oversold': 30,
                'price_column': 'close'
            }
        super().__init__(name, description, parameters)
    
    def get_parameters_description(self) -> Dict[str, Any]:
        """
        Restituisce la descrizione dei parametri della strategia.
        
        Returns:
            Dizionario con le descrizioni dei parametri
        """
        return {
            'window': {
                'name': 'Periodo RSI',
                'description': 'Numero di periodi per il calcolo del RSI',
                'type': 'int',
                'min': 2,
                'max': 50,
                'default': 14
            },
            'overbought': {
                'name': 'Soglia Ipercomprato',
                'description': 'Livello RSI sopra il quale si considera il mercato ipercomprato',
                'type': 'int',
                'min': 50,
                'max': 90,
                'default': 70
            },
            'oversold': {
                'name': 'Soglia Ipervenduto',
                'description': 'Livello RSI sotto il quale si considera il mercato ipervenduto',
                'type': 'int',
                'min': 10,
                'max': 50,
                'default': 30
            },
            'price_column': {
                'name': 'Colonna Prezzo',
                'description': 'Colonna da utilizzare per il calcolo del RSI',
                'type': 'select',
                'options': ['open', 'high', 'low', 'close'],
                'default': 'close'
            }
        }
    
    def _calculate_rsi(self, data: pd.Series, window: int) -> pd.Series:
        """
        Calcola l'RSI per una serie di dati.
        
        Args:
            data: Serie con i dati di prezzo
            window: Periodo per il calcolo dell'RSI
            
        Returns:
            Serie con i valori dell'RSI
        """
        # Calcola le variazioni giornaliere
        delta = data.diff()
        
        # Separa guadagni e perdite
        gain = delta.where(delta > 0, 0)
        loss = -delta.where(delta < 0, 0)
        
        # Calcola le medie dei guadagni e delle perdite
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        # Calcola l'RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Genera segnali di trading basati sull'RSI.
        
        Args:
            data: DataFrame con i dati di prezzo
            
        Returns:
            Serie con i segnali di trading
        """
        if data.empty:
            return pd.Series(index=data.index, data=TradingSignal.HOLD)
        
        # Estrai i parametri
        window = self.parameters.get('window', 14)
        overbought = self.parameters.get('overbought', 70)
        oversold = self.parameters.get('oversold', 30)
        price_column = self.parameters.get('price_column', 'close')
        
        # Verifica che la colonna del prezzo esista
        if price_column not in data.columns:
            logger.warning(f"Colonna {price_column} non trovata nei dati. Utilizzando 'close'.")
            price_column = 'close'
        
        # Assicurati che ci siano abbastanza dati
        if len(data) < window:
            logger.warning("Non ci sono abbastanza dati per calcolare il RSI.")
            return pd.Series(index=data.index, data=TradingSignal.HOLD)
        
        # Calcola l'RSI
        rsi = self._calculate_rsi(data[price_column], window)
        
        # Inizializza i segnali a HOLD
        signals = pd.Series(index=data.index, data=TradingSignal.HOLD)
        
        # Genera segnali di acquisto/vendita
        signals[rsi <= oversold] = TradingSignal.BUY  # Segnale di acquisto
        signals[rsi >= overbought] = TradingSignal.SELL  # Segnale di vendita
        
        # Salva i segnali e aggiorna il timestamp
        self.signals = signals
        self.last_update = datetime.datetime.now()
        
        return signals

class SentimentBasedStrategy(CustomStrategy):
    """
    Strategia basata sull'analisi del sentiment.
    Genera segnali di acquisto quando il sentiment è molto positivo,
    e segnali di vendita quando il sentiment è molto negativo.
    """
    
    def __init__(self, name: str = "Strategia Basata sul Sentiment", 
               description: str = "Strategia che genera segnali basati sull'analisi del sentiment", 
               parameters: Dict[str, Any] = None):
        """
        Inizializza la strategia con parametri predefiniti se non specificati.
        
        Args:
            name: Nome della strategia
            description: Descrizione della strategia
            parameters: Parametri della strategia
        """
        if parameters is None:
            parameters = {
                'positive_threshold': 0.4,
                'negative_threshold': -0.4,
                'price_influence': 0.5
            }
        super().__init__(name, description, parameters)
    
    def get_parameters_description(self) -> Dict[str, Any]:
        """
        Restituisce la descrizione dei parametri della strategia.
        
        Returns:
            Dizionario con le descrizioni dei parametri
        """
        return {
            'positive_threshold': {
                'name': 'Soglia Sentiment Positivo',
                'description': 'Soglia sopra la quale il sentiment è considerato molto positivo',
                'type': 'float',
                'min': 0.1,
                'max': 0.9,
                'step': 0.1,
                'default': 0.4
            },
            'negative_threshold': {
                'name': 'Soglia Sentiment Negativo',
                'description': 'Soglia sotto la quale il sentiment è considerato molto negativo',
                'type': 'float',
                'min': -0.9,
                'max': -0.1,
                'step': 0.1,
                'default': -0.4
            },
            'price_influence': {
                'name': 'Influenza Trend Prezzo',
                'description': 'Peso del trend di prezzo nella decisione (0 = solo sentiment, 1 = solo prezzo)',
                'type': 'float',
                'min': 0.0,
                'max': 1.0,
                'step': 0.1,
                'default': 0.5
            }
        }
    
    def generate_signals(self, data: pd.DataFrame, sentiment_data: Optional[pd.Series] = None) -> pd.Series:
        """
        Genera segnali di trading basati sul sentiment e sul trend del prezzo.
        
        Args:
            data: DataFrame con i dati di prezzo
            sentiment_data: Serie con i dati del sentiment per ogni data
            
        Returns:
            Serie con i segnali di trading
        """
        if data.empty:
            return pd.Series(index=data.index, data=TradingSignal.HOLD)
        
        # Se non ci sono dati di sentiment, usa solo il trend del prezzo
        if sentiment_data is None or sentiment_data.empty:
            # Calcola una media mobile semplice per determinare il trend
            window = 20
            sma = data['close'].rolling(window=window).mean()
            
            # Inizializza i segnali a HOLD
            signals = pd.Series(index=data.index, data=TradingSignal.HOLD)
            
            # Genera segnali basati sul trend del prezzo
            signals[data['close'] > sma] = TradingSignal.BUY  # Trend rialzista
            signals[data['close'] < sma] = TradingSignal.SELL  # Trend ribassista
            
            # Salva i segnali e aggiorna il timestamp
            self.signals = signals
            self.last_update = datetime.datetime.now()
            
            return signals
        
        # Estrai i parametri
        positive_threshold = self.parameters.get('positive_threshold', 0.4)
        negative_threshold = self.parameters.get('negative_threshold', -0.4)
        price_influence = self.parameters.get('price_influence', 0.5)
        
        # Calcola il trend del prezzo
        window = 20
        sma = data['close'].rolling(window=window).mean()
        price_trend = pd.Series(index=data.index, data=TradingSignal.HOLD)
        price_trend[data['close'] > sma] = TradingSignal.BUY
        price_trend[data['close'] < sma] = TradingSignal.SELL
        
        # Genera segnali basati sul sentiment
        sentiment_signals = pd.Series(index=data.index, data=TradingSignal.HOLD)
        
        # Allinea i dati di sentiment con i dati di prezzo
        aligned_sentiment = pd.Series(index=data.index)
        for date in data.index:
            date_str = date.strftime('%Y-%m-%d')
            if date_str in sentiment_data.index:
                aligned_sentiment[date] = sentiment_data[date_str]
        
        # Genera segnali basati sul sentiment
        sentiment_signals[aligned_sentiment > positive_threshold] = TradingSignal.BUY
        sentiment_signals[aligned_sentiment < negative_threshold] = TradingSignal.SELL
        
        # Combina i segnali del sentiment e del trend del prezzo
        combined_signals = pd.Series(index=data.index, data=TradingSignal.HOLD)
        for date in data.index:
            if pd.notna(aligned_sentiment[date]):
                # Se c'è un dato di sentiment, combina i segnali
                if sentiment_signals[date] == price_trend[date]:
                    # Se entrambi danno lo stesso segnale, usa quello
                    combined_signals[date] = sentiment_signals[date]
                elif sentiment_signals[date] == TradingSignal.HOLD:
                    # Se il sentiment è neutro, usa il trend del prezzo
                    combined_signals[date] = price_trend[date]
                elif price_trend[date] == TradingSignal.HOLD:
                    # Se il trend del prezzo è neutro, usa il sentiment
                    combined_signals[date] = sentiment_signals[date]
                else:
                    # Se i segnali sono contrastanti, usa la ponderazione
                    sentiment_weight = 1 - price_influence
                    combined_value = (sentiment_signals[date] * sentiment_weight + 
                                    price_trend[date] * price_influence)
                    
                    if combined_value > 0.3:
                        combined_signals[date] = TradingSignal.BUY
                    elif combined_value < -0.3:
                        combined_signals[date] = TradingSignal.SELL
                    else:
                        combined_signals[date] = TradingSignal.HOLD
            else:
                # Se non c'è un dato di sentiment, usa solo il trend del prezzo
                combined_signals[date] = price_trend[date]
        
        # Salva i segnali e aggiorna il timestamp
        self.signals = combined_signals
        self.last_update = datetime.datetime.now()
        
        return combined_signals

class CustomStrategyBuilder:
    """
    Classe per la creazione e gestione di strategie personalizzate.
    """
    
    STRATEGY_TYPES = {
        'MovingAverageCrossStrategy': MovingAverageCrossStrategy,
        'BollingerBandsStrategy': BollingerBandsStrategy,
        'RSIStrategy': RSIStrategy,
        'SentimentBasedStrategy': SentimentBasedStrategy
    }
    
    @staticmethod
    def create_strategy(strategy_type: str, name: str, description: str = "", 
                       parameters: Dict[str, Any] = None) -> CustomStrategy:
        """
        Crea una nuova strategia del tipo specificato.
        
        Args:
            strategy_type: Tipo di strategia da creare
            name: Nome della strategia
            description: Descrizione della strategia
            parameters: Parametri della strategia
            
        Returns:
            Istanza della strategia creata
        """
        if strategy_type not in CustomStrategyBuilder.STRATEGY_TYPES:
            raise ValueError(f"Tipo di strategia non supportato: {strategy_type}")
        
        strategy_class = CustomStrategyBuilder.STRATEGY_TYPES[strategy_type]
        return strategy_class(name=name, description=description, parameters=parameters)
    
    @staticmethod
    def get_available_strategies() -> List[Dict[str, Any]]:
        """
        Restituisce l'elenco delle strategie disponibili.
        
        Returns:
            Lista con le informazioni sulle strategie disponibili
        """
        strategies = []
        
        for strategy_type, strategy_class in CustomStrategyBuilder.STRATEGY_TYPES.items():
            # Crea un'istanza temporanea per ottenere la descrizione dei parametri
            temp_instance = strategy_class()
            
            strategies.append({
                'type': strategy_type,
                'name': temp_instance.name,
                'description': temp_instance.description,
                'parameters': temp_instance.get_parameters_description()
            })
        
        return strategies
    
    @staticmethod
    def load_strategy_from_dict(data: Dict[str, Any]) -> CustomStrategy:
        """
        Carica una strategia da un dizionario.
        
        Args:
            data: Dizionario con i dati della strategia
            
        Returns:
            Istanza della strategia caricata
        """
        strategy_type = data.get('type')
        
        if strategy_type not in CustomStrategyBuilder.STRATEGY_TYPES:
            raise ValueError(f"Tipo di strategia non supportato: {strategy_type}")
        
        strategy_class = CustomStrategyBuilder.STRATEGY_TYPES[strategy_type]
        return strategy_class.from_dict(data)
    
    @staticmethod
    def combine_strategies(strategies: List[CustomStrategy], 
                         weights: Optional[List[float]] = None, 
                         name: str = "Strategia Combinata") -> CustomStrategy:
        """
        Combina più strategie in una singola strategia.
        
        Args:
            strategies: Lista di strategie da combinare
            weights: Pesi da assegnare a ciascuna strategia (opzionale)
            name: Nome della strategia combinata
            
        Returns:
            Strategia combinata
        """
        if not strategies:
            raise ValueError("La lista di strategie è vuota")
        
        # Se i pesi non sono specificati, usa pesi uguali
        if weights is None:
            weights = [1.0 / len(strategies)] * len(strategies)
        elif len(weights) != len(strategies):
            raise ValueError("Il numero di pesi deve essere uguale al numero di strategie")
        
        # Normalizza i pesi
        total_weight = sum(weights)
        normalized_weights = [w / total_weight for w in weights]
        
        # Crea una nuova strategia combinata
        description = "Strategia combinata basata su: " + ", ".join(s.name for s in strategies)
        combined_strategy = CustomStrategy(name=name, description=description)
        
        # Definisci la funzione per generare i segnali combinati
        def generate_combined_signals(data: pd.DataFrame) -> pd.Series:
            if data.empty:
                return pd.Series(index=data.index, data=TradingSignal.HOLD)
            
            # Genera i segnali per ciascuna strategia
            all_signals = []
            for i, strategy in enumerate(strategies):
                signals = strategy.generate_signals(data)
                all_signals.append(signals)
            
            # Combina i segnali usando i pesi
            combined = pd.Series(index=data.index, data=0.0)
            for i, signals in enumerate(all_signals):
                combined += signals * normalized_weights[i]
            
            # Converti in segnali discreti
            result = pd.Series(index=data.index, data=TradingSignal.HOLD)
            result[combined > 0.3] = TradingSignal.BUY
            result[combined < -0.3] = TradingSignal.SELL
            
            return result
        
        # Sostituisci il metodo generate_signals con quello combinato
        combined_strategy.generate_signals = generate_combined_signals
        
        return combined_strategy