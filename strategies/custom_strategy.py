"""
Modulo per le strategie di trading personalizzate in CryptoTradeAnalyzer.

Questo modulo contiene le classi e le funzioni per implementare
strategie di trading personalizzate. Le strategie possono essere
create dall'utente e ottimizzate con tecniche di IA.

Author: CryptoTradeAnalyzer Team
"""

import logging
import pandas as pd
import numpy as np
from enum import Enum
from typing import Dict, Any, List, Optional, Tuple, Union, Callable

# Configurazione del logger
logger = logging.getLogger(__name__)

class TradingSignal(Enum):
    """Enumerazione dei possibili segnali di trading."""
    BUY = 1   # Segnale di acquisto
    SELL = -1  # Segnale di vendita
    HOLD = 0   # Segnale di attesa

class CustomStrategy:
    """
    Classe base per le strategie di trading personalizzate.
    """
    
    def __init__(self, name: str, description: str = "", parameters: Dict[str, Any] = None):
        """
        Inizializza una strategia di trading personalizzata.
        
        Args:
            name: Nome della strategia
            description: Descrizione della strategia
            parameters: Parametri della strategia in formato dizionario
        """
        self.name = name
        self.description = description
        self.parameters = parameters or {}
        
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Genera segnali di trading basati sui dati forniti.
        Da sovrascrivere nelle classi figlie.
        
        Args:
            data: DataFrame con i dati di prezzo
            
        Returns:
            Serie di segnali di trading
        """
        raise NotImplementedError("Questo metodo deve essere implementato nelle sottoclassi")
    
    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Restituisce informazioni sui parametri della strategia.
        
        Returns:
            Dizionario con informazioni sui parametri
        """
        raise NotImplementedError("Questo metodo deve essere implementato nelle sottoclassi")
    
    def validate_parameters(self) -> bool:
        """
        Verifica che i parametri della strategia siano validi.
        
        Returns:
            True se i parametri sono validi, False altrimenti
        """
        # Verifica che ci siano tutti i parametri necessari
        required_params = self.get_parameter_info().keys()
        for param in required_params:
            if param not in self.parameters:
                logger.error(f"Parametro mancante: {param}")
                return False
        
        return True

class MovingAverageCrossStrategy(CustomStrategy):
    """
    Strategia basata sull'incrocio di medie mobili.
    Genera segnali di acquisto quando la media breve incrocia al rialzo la media lunga,
    e segnali di vendita quando la media breve incrocia al ribasso la media lunga.
    """
    
    def __init__(self, name: str = "Moving Average Cross Strategy", 
                 description: str = "Strategia basata sull'incrocio di medie mobili",
                 parameters: Dict[str, Any] = None):
        """
        Inizializza la strategia di incrocio delle medie mobili.
        
        Args:
            name: Nome della strategia
            description: Descrizione della strategia
            parameters: Parametri della strategia (short_window, long_window, ma_type)
        """
        default_params = {
            "short_window": 20,
            "long_window": 50,
            "ma_type": "SMA"  # SMA, EMA, WMA
        }
        
        # Usa i parametri forniti o quelli di default
        parameters = parameters or default_params
        
        super().__init__(name, description, parameters)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Genera segnali di trading basati sull'incrocio di medie mobili.
        
        Args:
            data: DataFrame con i dati di prezzo
            
        Returns:
            Serie di segnali di trading
        """
        # Verifica che i parametri siano validi
        if not self.validate_parameters():
            logger.error("Parametri non validi per la strategia")
            # Ritorna serie vuota
            return pd.Series(index=data.index)
        
        # Estrai parametri
        short_window = self.parameters["short_window"]
        long_window = self.parameters["long_window"]
        ma_type = self.parameters["ma_type"]
        
        # Verifica che il DataFrame contenga i dati necessari
        if "close" not in data.columns:
            logger.error("Dati mancanti: colonna 'close' non trovata")
            return pd.Series(index=data.index)
        
        # Calcola le medie mobili
        if ma_type == "SMA":
            short_ma = data["close"].rolling(window=short_window).mean()
            long_ma = data["close"].rolling(window=long_window).mean()
        elif ma_type == "EMA":
            short_ma = data["close"].ewm(span=short_window, adjust=False).mean()
            long_ma = data["close"].ewm(span=long_window, adjust=False).mean()
        elif ma_type == "WMA":
            # Weighted Moving Average
            weights_short = np.arange(1, short_window + 1)
            weights_long = np.arange(1, long_window + 1)
            
            short_ma = data["close"].rolling(window=short_window).apply(
                lambda x: np.sum(weights_short * x) / weights_short.sum(), raw=True)
            
            long_ma = data["close"].rolling(window=long_window).apply(
                lambda x: np.sum(weights_long * x) / weights_long.sum(), raw=True)
        else:
            logger.error(f"Tipo di media mobile non supportato: {ma_type}")
            return pd.Series(index=data.index)
        
        # Inizializza la serie di segnali con HOLD
        signals = pd.Series(TradingSignal.HOLD, index=data.index)
        
        # Genera segnali
        for i in range(1, len(data)):
            # Se la media breve incrocia al rialzo la media lunga, BUY
            if short_ma.iloc[i-1] < long_ma.iloc[i-1] and short_ma.iloc[i] >= long_ma.iloc[i]:
                signals.iloc[i] = TradingSignal.BUY
            # Se la media breve incrocia al ribasso la media lunga, SELL
            elif short_ma.iloc[i-1] > long_ma.iloc[i-1] and short_ma.iloc[i] <= long_ma.iloc[i]:
                signals.iloc[i] = TradingSignal.SELL
        
        return signals
    
    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Restituisce informazioni sui parametri della strategia.
        
        Returns:
            Dizionario con informazioni sui parametri
        """
        return {
            "short_window": {
                "name": "Periodo Media Breve",
                "description": "Periodo per la media mobile breve",
                "type": "int",
                "default": 20,
                "min": 5,
                "max": 50
            },
            "long_window": {
                "name": "Periodo Media Lunga",
                "description": "Periodo per la media mobile lunga",
                "type": "int",
                "default": 50,
                "min": 20,
                "max": 200
            },
            "ma_type": {
                "name": "Tipo Media Mobile",
                "description": "Tipo di media mobile da utilizzare",
                "type": "select",
                "options": ["SMA", "EMA", "WMA"],
                "default": "SMA"
            }
        }

class BollingerBandsStrategy(CustomStrategy):
    """
    Strategia basata sulle bande di Bollinger.
    Genera segnali di acquisto quando il prezzo tocca la banda inferiore,
    e segnali di vendita quando il prezzo tocca la banda superiore.
    """
    
    def __init__(self, name: str = "Bollinger Bands Strategy", 
                 description: str = "Strategia basata sulle bande di Bollinger",
                 parameters: Dict[str, Any] = None):
        """
        Inizializza la strategia basata sulle bande di Bollinger.
        
        Args:
            name: Nome della strategia
            description: Descrizione della strategia
            parameters: Parametri della strategia (window, num_std)
        """
        default_params = {
            "window": 20,
            "num_std": 2.0
        }
        
        # Usa i parametri forniti o quelli di default
        parameters = parameters or default_params
        
        super().__init__(name, description, parameters)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Genera segnali di trading basati sulle bande di Bollinger.
        
        Args:
            data: DataFrame con i dati di prezzo
            
        Returns:
            Serie di segnali di trading
        """
        # Verifica che i parametri siano validi
        if not self.validate_parameters():
            logger.error("Parametri non validi per la strategia")
            return pd.Series(index=data.index)
        
        # Estrai parametri
        window = self.parameters["window"]
        num_std = self.parameters["num_std"]
        
        # Verifica che il DataFrame contenga i dati necessari
        if "close" not in data.columns:
            logger.error("Dati mancanti: colonna 'close' non trovata")
            return pd.Series(index=data.index)
        
        # Calcola la media mobile
        rolling_mean = data["close"].rolling(window=window).mean()
        
        # Calcola la deviazione standard
        rolling_std = data["close"].rolling(window=window).std()
        
        # Calcola le bande di Bollinger
        upper_band = rolling_mean + (rolling_std * num_std)
        lower_band = rolling_mean - (rolling_std * num_std)
        
        # Inizializza la serie di segnali con HOLD
        signals = pd.Series(TradingSignal.HOLD, index=data.index)
        
        # Genera segnali
        for i in range(1, len(data)):
            # Se il prezzo tocca o scende sotto la banda inferiore, BUY
            if data["close"].iloc[i-1] > lower_band.iloc[i-1] and data["close"].iloc[i] <= lower_band.iloc[i]:
                signals.iloc[i] = TradingSignal.BUY
            # Se il prezzo tocca o sale sopra la banda superiore, SELL
            elif data["close"].iloc[i-1] < upper_band.iloc[i-1] and data["close"].iloc[i] >= upper_band.iloc[i]:
                signals.iloc[i] = TradingSignal.SELL
        
        return signals
    
    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Restituisce informazioni sui parametri della strategia.
        
        Returns:
            Dizionario con informazioni sui parametri
        """
        return {
            "window": {
                "name": "Periodo",
                "description": "Periodo per il calcolo delle bande",
                "type": "int",
                "default": 20,
                "min": 10,
                "max": 50
            },
            "num_std": {
                "name": "Deviazioni Standard",
                "description": "Numero di deviazioni standard per le bande",
                "type": "float",
                "default": 2.0,
                "min": 1.0,
                "max": 3.0,
                "step": 0.1
            }
        }

class RSIStrategy(CustomStrategy):
    """
    Strategia basata sull'indicatore RSI (Relative Strength Index).
    Genera segnali di acquisto quando l'RSI è in zona di ipervenduto,
    e segnali di vendita quando l'RSI è in zona di ipercomprato.
    """
    
    def __init__(self, name: str = "RSI Strategy", 
                 description: str = "Strategia basata sull'indicatore RSI",
                 parameters: Dict[str, Any] = None):
        """
        Inizializza la strategia basata sull'RSI.
        
        Args:
            name: Nome della strategia
            description: Descrizione della strategia
            parameters: Parametri della strategia (window, overbought, oversold)
        """
        default_params = {
            "window": 14,
            "overbought": 70,
            "oversold": 30
        }
        
        # Usa i parametri forniti o quelli di default
        parameters = parameters or default_params
        
        super().__init__(name, description, parameters)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Genera segnali di trading basati sull'RSI.
        
        Args:
            data: DataFrame con i dati di prezzo
            
        Returns:
            Serie di segnali di trading
        """
        # Verifica che i parametri siano validi
        if not self.validate_parameters():
            logger.error("Parametri non validi per la strategia")
            return pd.Series(index=data.index)
        
        # Estrai parametri
        window = self.parameters["window"]
        overbought = self.parameters["overbought"]
        oversold = self.parameters["oversold"]
        
        # Verifica che il DataFrame contenga i dati necessari
        if "close" not in data.columns:
            logger.error("Dati mancanti: colonna 'close' non trovata")
            return pd.Series(index=data.index)
        
        # Calcola l'RSI
        rsi = self._calculate_rsi(data["close"], window)
        
        # Inizializza la serie di segnali con HOLD
        signals = pd.Series(TradingSignal.HOLD, index=data.index)
        
        # Genera segnali
        for i in range(1, len(data)):
            # Se l'RSI esce dalla zona di ipervenduto, BUY
            if rsi.iloc[i-1] <= oversold and rsi.iloc[i] > oversold:
                signals.iloc[i] = TradingSignal.BUY
            # Se l'RSI entra nella zona di ipercomprato, SELL
            elif rsi.iloc[i-1] >= overbought and rsi.iloc[i] < overbought:
                signals.iloc[i] = TradingSignal.SELL
        
        return signals
    
    def _calculate_rsi(self, data: pd.Series, window: int) -> pd.Series:
        """
        Calcola l'indicatore RSI.
        
        Args:
            data: Serie di prezzi
            window: Periodo per il calcolo dell'RSI
            
        Returns:
            Serie con i valori dell'RSI
        """
        # Calcola le variazioni di prezzo
        delta = data.diff()
        
        # Separa variazioni positive e negative
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        
        # Calcola la media mobile delle variazioni
        avg_gain = gain.rolling(window=window).mean()
        avg_loss = loss.rolling(window=window).mean()
        
        # Calcola l'RSI
        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        
        return rsi
    
    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Restituisce informazioni sui parametri della strategia.
        
        Returns:
            Dizionario con informazioni sui parametri
        """
        return {
            "window": {
                "name": "Periodo RSI",
                "description": "Periodo per il calcolo dell'RSI",
                "type": "int",
                "default": 14,
                "min": 7,
                "max": 30
            },
            "overbought": {
                "name": "Soglia Ipercomprato",
                "description": "Livello RSI considerato ipercomprato",
                "type": "int",
                "default": 70,
                "min": 60,
                "max": 90
            },
            "oversold": {
                "name": "Soglia Ipervenduto",
                "description": "Livello RSI considerato ipervenduto",
                "type": "int",
                "default": 30,
                "min": 10,
                "max": 40
            }
        }

class SentimentBasedStrategy(CustomStrategy):
    """
    Strategia basata sull'analisi del sentiment.
    Genera segnali di acquisto quando il sentiment è positivo,
    e segnali di vendita quando il sentiment è negativo.
    """
    
    def __init__(self, name: str = "Sentiment-Based Strategy", 
                 description: str = "Strategia che utilizza l'analisi del sentiment",
                 parameters: Dict[str, Any] = None):
        """
        Inizializza la strategia basata sul sentiment.
        
        Args:
            name: Nome della strategia
            description: Descrizione della strategia
            parameters: Parametri della strategia (positive_threshold, negative_threshold, trend_weight)
        """
        default_params = {
            "positive_threshold": 0.3,
            "negative_threshold": -0.3,
            "trend_weight": 0.5  # Peso del trend rispetto al sentiment (0-1)
        }
        
        # Usa i parametri forniti o quelli di default
        parameters = parameters or default_params
        
        super().__init__(name, description, parameters)
    
    def generate_signals(self, data: pd.DataFrame) -> pd.Series:
        """
        Genera segnali di trading basati sul sentiment.
        
        Args:
            data: DataFrame con i dati di prezzo
            
        Returns:
            Serie di segnali di trading
        """
        # Verifica che i parametri siano validi
        if not self.validate_parameters():
            logger.error("Parametri non validi per la strategia")
            return pd.Series(index=data.index)
        
        # Estrai parametri
        positive_threshold = self.parameters["positive_threshold"]
        negative_threshold = self.parameters["negative_threshold"]
        trend_weight = self.parameters["trend_weight"]
        
        # Verifica che il DataFrame contenga i dati necessari
        required_columns = ["close", "sentiment"]
        for col in required_columns:
            if col not in data.columns:
                logger.error(f"Dati mancanti: colonna '{col}' non trovata")
                return pd.Series(index=data.index)
        
        # Calcola il trend di prezzo (media mobile a 5 giorni)
        price_ma5 = data["close"].rolling(window=5).mean()
        price_trend = (data["close"] - price_ma5) / price_ma5
        
        # Combina sentiment e trend
        combined_signal = (data["sentiment"] * (1 - trend_weight) + 
                          price_trend * trend_weight)
        
        # Inizializza la serie di segnali con HOLD
        signals = pd.Series(TradingSignal.HOLD, index=data.index)
        
        # Genera segnali
        for i in range(1, len(data)):
            # Se il segnale combinato supera la soglia positiva, BUY
            if combined_signal.iloc[i] >= positive_threshold:
                signals.iloc[i] = TradingSignal.BUY
            # Se il segnale combinato scende sotto la soglia negativa, SELL
            elif combined_signal.iloc[i] <= negative_threshold:
                signals.iloc[i] = TradingSignal.SELL
        
        return signals
    
    def get_parameter_info(self) -> Dict[str, Dict[str, Any]]:
        """
        Restituisce informazioni sui parametri della strategia.
        
        Returns:
            Dizionario con informazioni sui parametri
        """
        return {
            "positive_threshold": {
                "name": "Soglia Positiva",
                "description": "Soglia del sentiment per segnali di acquisto",
                "type": "float",
                "default": 0.3,
                "min": 0.1,
                "max": 0.7,
                "step": 0.05
            },
            "negative_threshold": {
                "name": "Soglia Negativa",
                "description": "Soglia del sentiment per segnali di vendita",
                "type": "float",
                "default": -0.3,
                "min": -0.7,
                "max": -0.1,
                "step": 0.05
            },
            "trend_weight": {
                "name": "Peso del Trend",
                "description": "Peso da assegnare al trend di prezzo (vs sentiment)",
                "type": "float",
                "default": 0.5,
                "min": 0.0,
                "max": 1.0,
                "step": 0.1
            }
        }

class CustomStrategyBuilder:
    """
    Builder per creare strategie personalizzate.
    """
    
    @staticmethod
    def create_strategy(strategy_type: str, name: str = None, description: str = None, 
                       parameters: Dict[str, Any] = None) -> CustomStrategy:
        """
        Crea una strategia del tipo specificato.
        
        Args:
            strategy_type: Tipo di strategia da creare
            name: Nome della strategia (opzionale)
            description: Descrizione della strategia (opzionale)
            parameters: Parametri della strategia (opzionale)
            
        Returns:
            Istanza della strategia creata
        """
        # Dizionario di classi di strategia disponibili
        strategy_classes = {
            "MovingAverageCrossStrategy": MovingAverageCrossStrategy,
            "BollingerBandsStrategy": BollingerBandsStrategy,
            "RSIStrategy": RSIStrategy,
            "SentimentBasedStrategy": SentimentBasedStrategy
        }
        
        # Verifica che il tipo di strategia sia valido
        if strategy_type not in strategy_classes:
            logger.error(f"Tipo di strategia non supportato: {strategy_type}")
            raise ValueError(f"Tipo di strategia non supportato: {strategy_type}")
        
        # Crea un'istanza della strategia
        strategy_class = strategy_classes[strategy_type]
        
        # Se nome e descrizione non sono specificati, usa quelli di default
        if name is None:
            name = strategy_class().name
        if description is None:
            description = strategy_class().description
        
        return strategy_class(name=name, description=description, parameters=parameters)
    
    @staticmethod
    def get_available_strategy_types() -> List[str]:
        """
        Restituisce i tipi di strategia disponibili.
        
        Returns:
            Lista di tipi di strategia disponibili
        """
        return [
            "MovingAverageCrossStrategy",
            "BollingerBandsStrategy",
            "RSIStrategy",
            "SentimentBasedStrategy"
        ]
    
    @staticmethod
    def get_strategy_info(strategy_type: str) -> Dict[str, Any]:
        """
        Restituisce informazioni su un tipo di strategia.
        
        Args:
            strategy_type: Tipo di strategia
            
        Returns:
            Dizionario con informazioni sulla strategia
        """
        # Crea un'istanza della strategia per ottenere le informazioni
        strategy = CustomStrategyBuilder.create_strategy(strategy_type)
        
        return {
            "name": strategy.name,
            "description": strategy.description,
            "parameters": strategy.get_parameter_info()
        }