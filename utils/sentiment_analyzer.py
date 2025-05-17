"""
Modulo per l'analisi del sentiment in CryptoTradeAnalyzer.

Questo modulo utilizza diverse librerie di NLP (NLTK, TextBlob, VADER) 
per analizzare il sentiment dei dati di mercato e delle notizie 
sulle criptovalute.

Author: CryptoTradeAnalyzer Team
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import base64
import io
import json
import random
from datetime import datetime, timedelta
from textblob import TextBlob
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import nltk
from typing import Dict, List, Tuple, Any, Optional, Union

# Assicurati che NLTK abbia i dati necessari
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')


class SentimentAnalyzer:
    """
    Classe per l'analisi del sentiment dei dati di mercato e delle notizie.
    Utilizza diverse tecniche di NLP e combina i risultati.
    """

    def __init__(self, use_vader: bool = True, use_textblob: bool = True):
        """
        Inizializza l'analizzatore di sentiment.
        
        Args:
            use_vader: Indica se utilizzare VADER per l'analisi
            use_textblob: Indica se utilizzare TextBlob per l'analisi
        """
        self.use_vader = use_vader
        self.use_textblob = use_textblob
        
        if use_vader:
            self.vader = SentimentIntensityAnalyzer()
        
        # Dizionario di termini specifici per il dominio crypto
        self.crypto_lexicon = {
            'hodl': 3.0,
            'moon': 2.5,
            'dump': -2.0,
            'fud': -2.5,
            'shill': -1.5,
            'bullish': 3.0,
            'bearish': -3.0,
            'rally': 2.0,
            'correction': -1.5,
            'profit': 2.0,
            'loss': -2.0,
            'gain': 2.0,
            'crash': -3.0,
            'surge': 2.5,
            'plummet': -2.5,
            'to the moon': 3.0,
            'diamond hands': 2.0,
            'paper hands': -1.5,
            'rekt': -2.5,
            'bagholder': -2.0,
            'buy the dip': 1.5,
            'sell the news': -1.0,
            'whale': 0.0,  # neutral by itself
            'fomo': -1.0,
            'leverage': 0.0,  # neutral by itself
            'liquidation': -2.5,
            'long': 1.0,
            'short': -1.0,
            'position': 0.0,  # neutral by itself
            'squeeze': 0.0,  # can be positive or negative depending on context
            'institutional': 1.5,
            'adoption': 2.5,
            'regulation': -1.0,
            'ban': -3.0,
            'legal': 1.0,
            'illegal': -2.5,
            'whitepaper': 0.5,
            'roadmap': 1.0,
            'rugpull': -3.5,
            'scam': -3.5,
            'hack': -3.0,
            'security': 1.5,
            'vulnerability': -2.0,
            'upgrade': 2.0,
            'downtime': -2.0,
            'mainnet': 2.0,
            'testnet': 0.5,
            'fork': 0.0,  # neutral by itself
            'halving': 1.5,
            'mining': 0.5,
            'proof of work': 0.0,
            'proof of stake': 0.5,
            'smart contract': 1.0,
            'defi': 1.5,
            'nft': 0.5,
            'meme coin': -0.5,
            'altcoin': 0.0,
            'shitcoin': -3.0,
            'ico': -0.5,
            'airdrop': 1.0,
            'staking': 1.5,
            'yield farming': 1.0,
            'liquidity': 1.0,
            'volume': 0.5,
            'market cap': 0.0,
            'momentum': 0.5,
            'resistance': -0.5,
            'support': 0.5,
            'breakout': 2.0,
            'breakdown': -2.0,
            'overbought': -1.0,
            'oversold': 1.0,
            'golden cross': 2.5,
            'death cross': -2.5,
            'fib': 0.0,
            'macd': 0.0,
            'rsi': 0.0
        }
    
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analizza il sentiment di un testo.
        
        Args:
            text: Testo da analizzare
            
        Returns:
            Dizionario con i risultati dell'analisi
        """
        if not text or text.strip() == "":
            return {"compound": 0.0, "positive": 0.0, "negative": 0.0, "neutral": 1.0}
        
        scores = {"compound": 0.0, "positive": 0.0, "negative": 0.0, "neutral": 0.0}
        methods_count = 0
        
        # Analisi con VADER
        if self.use_vader:
            vader_scores = self.vader.polarity_scores(text)
            
            # Aggiungi contributo dei termini specifici per crypto
            compound_adjustment = 0.0
            text_lower = text.lower()
            
            for term, score in self.crypto_lexicon.items():
                if term in text_lower:
                    # Il contributo è pesato in base alla lunghezza del testo
                    weight = min(1.0, 5.0 / (len(text.split()) + 1))
                    compound_adjustment += score * weight * 0.1
            
            # Limita l'aggiustamento per evitare valori fuori scala
            compound_adjustment = max(-0.5, min(0.5, compound_adjustment))
            vader_scores['compound'] = max(-1.0, min(1.0, vader_scores['compound'] + compound_adjustment))
            
            for key in scores:
                scores[key] += vader_scores[key]
            
            methods_count += 1
        
        # Analisi con TextBlob
        if self.use_textblob:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Convertiamo la polarità di TextBlob nel formato di VADER
            textblob_scores = {
                "compound": polarity,
                "positive": max(0, polarity) * subjectivity,
                "negative": max(0, -polarity) * subjectivity,
                "neutral": (1 - subjectivity)
            }
            
            for key in scores:
                scores[key] += textblob_scores[key]
            
            methods_count += 1
        
        # Media dei risultati
        if methods_count > 0:
            for key in scores:
                scores[key] /= methods_count
        
        return scores
    
    def analyze_dataframe(self, df: pd.DataFrame, text_column: str, date_column: Optional[str] = None) -> pd.DataFrame:
        """
        Analizza il sentiment di una colonna di testo in un DataFrame.
        
        Args:
            df: DataFrame con i dati
            text_column: Nome della colonna contenente il testo
            date_column: Nome della colonna contenente la data (opzionale)
            
        Returns:
            DataFrame con i risultati dell'analisi
        """
        results = []
        
        for _, row in df.iterrows():
            text = row[text_column]
            sentiment = self.analyze_text(text)
            
            result = {
                'text': text,
                'compound': sentiment['compound'],
                'positive': sentiment['positive'],
                'negative': sentiment['negative'],
                'neutral': sentiment['neutral']
            }
            
            if date_column and date_column in row:
                result['date'] = row[date_column]
            
            results.append(result)
        
        return pd.DataFrame(results)
    
    def get_sentiment_colors(self, compound_score: float) -> Tuple[str, str]:
        """
        Restituisce il colore corrispondente a un punteggio di sentiment.
        
        Args:
            compound_score: Punteggio compound del sentiment
            
        Returns:
            Tupla (codice colore, nome colore)
        """
        if compound_score >= 0.5:
            return '#28a745', 'success'  # verde forte
        elif compound_score >= 0.05:
            return '#5cb85c', 'success'  # verde
        elif compound_score > -0.05:
            return '#6c757d', 'secondary'  # grigio
        elif compound_score > -0.5:
            return '#dc3545', 'danger'  # rosso
        else:
            return '#bd2130', 'danger'  # rosso forte
    
    def get_sentiment_label(self, compound_score: float) -> str:
        """
        Restituisce l'etichetta corrispondente a un punteggio di sentiment.
        
        Args:
            compound_score: Punteggio compound del sentiment
            
        Returns:
            Etichetta del sentiment
        """
        if compound_score >= 0.5:
            return 'Molto Positivo'
        elif compound_score >= 0.05:
            return 'Positivo'
        elif compound_score > -0.05:
            return 'Neutrale'
        elif compound_score > -0.5:
            return 'Negativo'
        else:
            return 'Molto Negativo'
    
    def get_simplified_sentiment(self, compound_score: float) -> str:
        """
        Restituisce un'etichetta semplificata per il sentiment.
        
        Args:
            compound_score: Punteggio compound del sentiment
            
        Returns:
            Etichetta semplificata
        """
        if compound_score >= 0.25:
            return 'Positivo'
        elif compound_score <= -0.25:
            return 'Negativo'
        else:
            return 'Neutrale'
    
    def generate_sentiment_chart(self, sentiment_data: pd.DataFrame, title: str = "Analisi del Sentiment") -> str:
        """
        Genera un grafico per visualizzare l'andamento del sentiment.
        
        Args:
            sentiment_data: DataFrame con i dati del sentiment
            title: Titolo del grafico
            
        Returns:
            Immagine del grafico codificata in base64
        """
        plt.figure(figsize=(12, 6))
        
        # Ordina per data se presente
        if 'date' in sentiment_data.columns:
            sentiment_data = sentiment_data.sort_values('date')
            x = sentiment_data['date']
        else:
            x = range(len(sentiment_data))
        
        # Plot principale per il sentiment composto
        plt.plot(x, sentiment_data['compound'], color='blue', linewidth=2)
        
        # Aree colorate per positivo e negativo
        plt.fill_between(x, sentiment_data['compound'], 0, 
                         where=(sentiment_data['compound'] > 0), 
                         color='green', alpha=0.3)
        plt.fill_between(x, sentiment_data['compound'], 0, 
                         where=(sentiment_data['compound'] < 0), 
                         color='red', alpha=0.3)
        
        # Linea dello zero
        plt.axhline(y=0, color='black', linestyle='-', alpha=0.3)
        
        # Linee tratteggiate per le soglie di sentiment
        plt.axhline(y=0.05, color='green', linestyle='--', alpha=0.5)
        plt.axhline(y=-0.05, color='red', linestyle='--', alpha=0.5)
        
        # Formattazione
        plt.title(title, fontsize=16)
        plt.ylabel('Sentiment Score', fontsize=12)
        
        if 'date' in sentiment_data.columns:
            plt.xlabel('Data', fontsize=12)
            plt.gcf().autofmt_xdate()  # Ruota le etichette della data
        else:
            plt.xlabel('Elemento', fontsize=12)
        
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # Converti il grafico in immagine base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        image_png = buffer.getvalue()
        buffer.close()
        
        # Pulisci la figura di matplotlib
        plt.close()
        
        return base64.b64encode(image_png).decode('utf-8')
    
    def summarize_sentiment(self, sentiment_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Genera un riepilogo del sentiment.
        
        Args:
            sentiment_data: DataFrame con i dati del sentiment
            
        Returns:
            Dizionario con il riepilogo
        """
        if len(sentiment_data) == 0:
            return {
                "score": 0.0,
                "label": "Neutrale",
                "simplified": "Neutrale",
                "color": "secondary",
                "interpretation": "Nessun dato disponibile per l'analisi"
            }
        
        # Calcola il sentiment medio
        avg_compound = sentiment_data['compound'].mean()
        
        # Determina l'etichetta e il colore
        _, color = self.get_sentiment_colors(avg_compound)
        label = self.get_sentiment_label(avg_compound)
        simplified = self.get_simplified_sentiment(avg_compound)
        
        # Calcola la tendenza recente se ci sono abbastanza dati
        trend_direction = None
        trend_strength = None
        
        if len(sentiment_data) >= 5 and 'date' in sentiment_data.columns:
            recent_data = sentiment_data.sort_values('date').tail(5)
            oldest_avg = recent_data.head(2)['compound'].mean()
            newest_avg = recent_data.tail(2)['compound'].mean()
            
            trend_change = newest_avg - oldest_avg
            
            if abs(trend_change) < 0.05:
                trend_direction = "stabile"
                trend_strength = "molto bassa"
            else:
                trend_direction = "in crescita" if trend_change > 0 else "in calo"
                
                if abs(trend_change) < 0.15:
                    trend_strength = "bassa"
                elif abs(trend_change) < 0.3:
                    trend_strength = "moderata"
                else:
                    trend_strength = "alta"
        
        # Genera una interpretazione del sentiment
        if avg_compound >= 0.5:
            interpretation = "estremamente positivo, suggerendo un forte ottimismo nel mercato"
        elif avg_compound >= 0.25:
            interpretation = "generalmente positivo, con un buon livello di ottimismo"
        elif avg_compound >= 0.05:
            interpretation = "leggermente positivo, con cauto ottimismo"
        elif avg_compound > -0.05:
            interpretation = "neutrale, senza una chiara direzione"
        elif avg_compound > -0.25:
            interpretation = "leggermente negativo, con una certa cautela"
        elif avg_compound > -0.5:
            interpretation = "generalmente negativo, con un livello significativo di pessimismo"
        else:
            interpretation = "estremamente negativo, suggerendo un forte pessimismo nel mercato"
        
        return {
            "score": avg_compound,
            "label": label,
            "simplified": simplified,
            "color": color,
            "interpretation": interpretation,
            "trend_direction": trend_direction,
            "trend_strength": trend_strength
        }


def generate_sample_news_data(symbol: str, start_date: datetime, end_date: datetime, count: int = 20) -> pd.DataFrame:
    """
    Genera dati di esempio per notizie e sentiment.
    Utile per test e demo quando non sono disponibili dati reali.
    
    Args:
        symbol: Simbolo della criptovaluta
        start_date: Data di inizio
        end_date: Data di fine
        count: Numero di notizie da generare
        
    Returns:
        DataFrame con le notizie generate
    """
    # Template per titoli di notizie
    bullish_titles = [
        f"{symbol} potrebbe raggiungere nuovi massimi storici secondo gli analisti",
        f"Crescente adozione istituzionale per {symbol} con nuovi investimenti",
        f"Aggiornamento tecnologico importante in arrivo per {symbol}",
        f"{symbol} registra un aumento significativo delle transazioni",
        f"Nuova partnership strategica annunciata per {symbol}",
        f"Gli sviluppatori di {symbol} annunciano importanti miglioramenti alla rete",
        f"Grande exchange aggiunge il supporto per {symbol}",
        f"L'analisi on-chain mostra forti segnali rialzisti per {symbol}",
        f"Il rapporto accumulo/distribuzione di {symbol} suggerisce un trend rialzista",
        f"Celebre investitore si esprime positivamente su {symbol}"
    ]
    
    bearish_titles = [
        f"Preoccupazioni normative crescenti per {symbol}",
        f"Volume di vendita in aumento per {symbol} mentre il mercato vacilla",
        f"Analisti avvertono di una possibile correzione per {symbol}",
        f"Vulnerabilità di sicurezza scoperta nell'ecosistema {symbol}",
        f"Principali holder stanno riducendo le posizioni in {symbol}",
        f"Concorrenza crescente potrebbe minacciare la posizione di {symbol}",
        f"Indicatori tecnici mostrano segnali di sovraccarico per {symbol}",
        f"Report negativo pubblicato sulle prospettive a lungo termine di {symbol}",
        f"Exchange importante rimuove le coppie di trading per {symbol}",
        f"Diminuzione dell'attività degli sviluppatori per {symbol}"
    ]
    
    neutral_titles = [
        f"{symbol} si stabilizza dopo la recente volatilità",
        f"Nuovo paper accademico analizza l'ecosistema di {symbol}",
        f"Confronto tra {symbol} e altre criptovalute principali",
        f"Revisione dei fondamentali di {symbol} nel contesto attuale",
        f"Intervista al team di sviluppo di {symbol}",
        f"Come funziona la tecnologia dietro {symbol}",
        f"Prospettive di mercato miste per {symbol} secondo gli esperti",
        f"Volume di scambio stabile per {symbol} questa settimana",
        f"Analisi della distribuzione dei wallet di {symbol}",
        f"Confronto tra gli attuali e precedenti cicli di mercato per {symbol}"
    ]
    
    # Calcola il range di date
    date_range = (end_date - start_date).days
    
    # Crea le notizie
    news_data = []
    for _ in range(count):
        days_offset = random.randint(0, date_range)
        news_date = start_date + timedelta(days=days_offset)
        
        # Scegli casualmente un sentiment e un titolo
        sentiment_type = random.choices(['bullish', 'bearish', 'neutral'], [0.4, 0.3, 0.3])[0]
        
        if sentiment_type == 'bullish':
            title = random.choice(bullish_titles)
            sentiment_base = random.uniform(0.1, 0.9)
        elif sentiment_type == 'bearish':
            title = random.choice(bearish_titles)
            sentiment_base = random.uniform(-0.9, -0.1)
        else:
            title = random.choice(neutral_titles)
            sentiment_base = random.uniform(-0.1, 0.1)
        
        # Aggiungi un po' di rumore al sentiment
        sentiment = max(-1.0, min(1.0, sentiment_base + random.uniform(-0.1, 0.1)))
        
        news_data.append({
            'date': news_date,
            'title': title,
            'sentiment': sentiment
        })
    
    # Ordina per data
    news_data.sort(key=lambda x: x['date'])
    
    return pd.DataFrame(news_data)