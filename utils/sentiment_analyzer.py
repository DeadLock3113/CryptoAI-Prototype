"""
Modulo per l'analisi del sentiment in CryptoTradeAnalyzer.

Questo modulo utilizza diverse librerie di NLP (NLTK, TextBlob, VADER) 
per analizzare il sentiment dei dati di mercato e delle notizie 
sulle criptovalute.

Author: CryptoTradeAnalyzer Team
"""

import io
import base64
import random
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List, Tuple

# NLTK
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

# Ensure the VADER lexicon is downloaded
try:
    nltk.data.find('sentiment/vader_lexicon.zip')
except LookupError:
    nltk.download('vader_lexicon')

# TextBlob
from textblob import TextBlob

# Configurazione del logger
logger = logging.getLogger(__name__)

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
        
        # Inizializza gli analizzatori di sentiment
        if self.use_vader:
            self.vader = SentimentIntensityAnalyzer()
        
    def analyze_text(self, text: str) -> Dict[str, Any]:
        """
        Analizza il sentiment di un testo.
        
        Args:
            text: Testo da analizzare
            
        Returns:
            Dizionario con i risultati dell'analisi
        """
        results = {
            'text': text,
            'compound': 0.0,
            'positive': 0.0,
            'negative': 0.0,
            'neutral': 0.0
        }
        
        # Analisi VADER
        if self.use_vader:
            vader_scores = self.vader.polarity_scores(text)
            
            # Aggiorna i risultati con i punteggi VADER
            for key, value in vader_scores.items():
                if key != 'compound':
                    # Normalizza i punteggi
                    results[key] = value
                else:
                    # Il punteggio compound è già normalizzato
                    results[key] = value
        
        # Analisi TextBlob
        if self.use_textblob:
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            
            # TextBlob fornisce un punteggio di polarità tra -1 e 1
            # Convertiamo in un formato simile a VADER per coerenza
            if polarity > 0:
                textblob_positive = polarity
                textblob_negative = 0.0
            else:
                textblob_positive = 0.0
                textblob_negative = abs(polarity)
                
            textblob_neutral = 1.0 - (textblob_positive + textblob_negative)
            
            # Se utilizziamo entrambi gli analizzatori, facciamo una media
            if self.use_vader:
                results['compound'] = (results['compound'] + polarity) / 2
                results['positive'] = (results['positive'] + textblob_positive) / 2
                results['negative'] = (results['negative'] + textblob_negative) / 2
                results['neutral'] = (results['neutral'] + textblob_neutral) / 2
            else:
                results['compound'] = polarity
                results['positive'] = textblob_positive
                results['negative'] = textblob_negative
                results['neutral'] = textblob_neutral
        
        return results
    
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
        # Crea una copia del DataFrame
        result_df = df.copy()
        
        # Analizza il sentiment per ogni riga
        sentiment_results = []
        for text in df[text_column]:
            sentiment_results.append(self.analyze_text(text))
        
        # Aggiungi i risultati al DataFrame
        result_df['sentiment'] = [result['compound'] for result in sentiment_results]
        result_df['sentiment_positive'] = [result['positive'] for result in sentiment_results]
        result_df['sentiment_negative'] = [result['negative'] for result in sentiment_results]
        result_df['sentiment_neutral'] = [result['neutral'] for result in sentiment_results]
        
        # Ordina per data se specificata
        if date_column and date_column in df.columns:
            result_df = result_df.sort_values(by=date_column)
        
        return result_df
    
    def get_sentiment_colors(self, compound_score: float) -> Tuple[str, str]:
        """
        Restituisce il colore corrispondente a un punteggio di sentiment.
        
        Args:
            compound_score: Punteggio compound del sentiment
            
        Returns:
            Tupla (codice colore, nome colore)
        """
        if compound_score >= 0.5:
            return '#28a745', 'success'  # Verde intenso
        elif compound_score >= 0.05:
            return '#5cb85c', 'success-light'  # Verde chiaro
        elif compound_score <= -0.5:
            return '#dc3545', 'danger'  # Rosso intenso
        elif compound_score <= -0.05:
            return '#f8d7da', 'danger-light'  # Rosso chiaro
        else:
            return '#6c757d', 'neutral'  # Grigio
    
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
        elif compound_score <= -0.5:
            return 'Molto Negativo'
        elif compound_score <= -0.05:
            return 'Negativo'
        else:
            return 'Neutro'
    
    def get_simplified_sentiment(self, compound_score: float) -> str:
        """
        Restituisce un'etichetta semplificata per il sentiment.
        
        Args:
            compound_score: Punteggio compound del sentiment
            
        Returns:
            Etichetta semplificata
        """
        if compound_score >= 0.05:
            return 'Positivo'
        elif compound_score <= -0.05:
            return 'Negativo'
        else:
            return 'Neutro'
    
    def generate_sentiment_chart(self, sentiment_data: pd.DataFrame, title: str = "Analisi del Sentiment") -> str:
        """
        Genera un grafico per visualizzare l'andamento del sentiment.
        
        Args:
            sentiment_data: DataFrame con i dati del sentiment
            title: Titolo del grafico
            
        Returns:
            Immagine del grafico codificata in base64
        """
        # Configura il grafico
        plt.figure(figsize=(12, 6))
        
        # Se c'è una colonna 'date', usa come indice
        if 'date' in sentiment_data.columns:
            x = sentiment_data['date']
        else:
            x = range(len(sentiment_data))
        
        # Grafico del sentiment
        plt.plot(x, sentiment_data['sentiment'], marker='o', linestyle='-', color='blue', label='Sentiment')
        
        # Aggiungi linee orizzontali per le soglie
        plt.axhline(y=0.05, color='green', linestyle='--', alpha=0.5, label='Soglia Positiva')
        plt.axhline(y=-0.05, color='red', linestyle='--', alpha=0.5, label='Soglia Negativa')
        
        # Colorazione delle aree
        plt.fill_between(range(len(sentiment_data)), 0.05, 1, alpha=0.1, color='green')
        plt.fill_between(range(len(sentiment_data)), -0.05, 0.05, alpha=0.1, color='gray')
        plt.fill_between(range(len(sentiment_data)), -1, -0.05, alpha=0.1, color='red')
        
        # Aggiungi titolo e etichette
        plt.title(title, fontsize=16)
        plt.xlabel('Data')
        plt.ylabel('Sentiment Score')
        plt.ylim(-1, 1)
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Ruota le etichette delle date
        if 'date' in sentiment_data.columns:
            plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Converti il grafico in immagine base64
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', bbox_inches='tight')
        buffer.seek(0)
        
        image_png = buffer.getvalue()
        buffer.close()
        
        graphic = base64.b64encode(image_png).decode('utf-8')
        
        plt.close()
        
        return graphic
    
    def summarize_sentiment(self, sentiment_data: pd.DataFrame) -> Dict[str, Any]:
        """
        Genera un riepilogo del sentiment.
        
        Args:
            sentiment_data: DataFrame con i dati del sentiment
            
        Returns:
            Dizionario con il riepilogo
        """
        # Calcola statistiche di base
        mean_sentiment = sentiment_data['sentiment'].mean()
        max_sentiment = sentiment_data['sentiment'].max()
        min_sentiment = sentiment_data['sentiment'].min()
        
        # Calcola la percentuale di sentiment positivo, negativo e neutro
        positive_count = (sentiment_data['sentiment'] >= 0.05).sum()
        negative_count = (sentiment_data['sentiment'] <= -0.05).sum()
        neutral_count = ((sentiment_data['sentiment'] > -0.05) & (sentiment_data['sentiment'] < 0.05)).sum()
        total_count = len(sentiment_data)
        
        positive_pct = (positive_count / total_count) * 100 if total_count > 0 else 0
        negative_pct = (negative_count / total_count) * 100 if total_count > 0 else 0
        neutral_pct = (neutral_count / total_count) * 100 if total_count > 0 else 0
        
        # Calcola la tendenza (ultimi 5 record vs precedenti)
        if len(sentiment_data) >= 10:
            recent_sentiment = sentiment_data['sentiment'].iloc[-5:].mean()
            previous_sentiment = sentiment_data['sentiment'].iloc[:-5].mean()
            trend = recent_sentiment - previous_sentiment
            trend_label = 'in miglioramento' if trend > 0.1 else 'in peggioramento' if trend < -0.1 else 'stabile'
        else:
            trend = 0
            trend_label = 'non disponibile'
        
        # Colori per ciascuna categoria
        positive_color = self.get_sentiment_colors(0.5)[0]
        negative_color = self.get_sentiment_colors(-0.5)[0]
        neutral_color = self.get_sentiment_colors(0)[0]
        
        # Crea il riepilogo
        summary = {
            'mean_sentiment': mean_sentiment,
            'max_sentiment': max_sentiment,
            'min_sentiment': min_sentiment,
            'positive_count': int(positive_count),
            'negative_count': int(negative_count),
            'neutral_count': int(neutral_count),
            'total_count': total_count,
            'positive_pct': positive_pct,
            'negative_pct': negative_pct,
            'neutral_pct': neutral_pct,
            'positive_color': positive_color,
            'negative_color': negative_color,
            'neutral_color': neutral_color,
            'trend': trend,
            'trend_label': trend_label,
            'overall_sentiment': self.get_sentiment_label(mean_sentiment),
            'overall_color': self.get_sentiment_colors(mean_sentiment)[0]
        }
        
        return summary


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
    # Titoli positivi
    positive_titles = [
        f"{symbol} segna un nuovo massimo storico con aumenti del volume",
        f"Gli analisti prevedono un futuro brillante per {symbol}",
        f"{symbol} ottiene nuove partnership importanti nel settore",
        f"Adozione massiccia di {symbol} da parte di istituzioni finanziarie",
        f"I fondamentali di {symbol} mostrano segnali positivi",
        f"{symbol} vede un aumento dell'interesse da parte degli investitori",
        f"Nuove funzionalità promettenti in arrivo per {symbol}",
        f"{symbol} supera le aspettative in termini di adozione",
        f"Analisi tecnica suggerisce tendenza rialzista per {symbol}",
        f"Grandi investitori accumulano {symbol} a questi livelli"
    ]
    
    # Titoli negativi
    negative_titles = [
        f"Preoccupazioni normative pesano sul prezzo di {symbol}",
        f"{symbol} affronta un forte calo dopo il recente rally",
        f"Gli analisti avvertono di possibili correzioni per {symbol}",
        f"Vulnerabilità di sicurezza scoperta nell'ecosistema {symbol}",
        f"{symbol} perde terreno rispetto ai concorrenti",
        f"Vendite massicce colpiscono {symbol} nel mercato cripto",
        f"Problemi tecnici compromettono la fiducia in {symbol}",
        f"Critiche alla governance del progetto {symbol}",
        f"Sentiment di mercato negativo pesa su {symbol}",
        f"Analisi tecnica indica tendenza ribassista per {symbol}"
    ]
    
    # Titoli neutri
    neutral_titles = [
        f"{symbol} si stabilizza dopo recenti movimenti di mercato",
        f"Rapporto di analisi approfondita su {symbol} pubblicato oggi",
        f"Cosa aspettarsi da {symbol} nei prossimi mesi",
        f"Studio confronta {symbol} con altri asset simili",
        f"Il team di {symbol} annuncia aggiornamenti trimestrali",
        f"Analisi delle tendenze di volume per {symbol}",
        f"{symbol} introduce nuove caratteristiche tecniche",
        f"Intervista con gli sviluppatori principali di {symbol}",
        f"Confronto tra {symbol} e asset tradizionali",
        f"Previsioni di mercato includono {symbol} tra i progetti da monitorare"
    ]
    
    # Genera date casuali tra start_date e end_date
    date_range = (end_date - start_date).days
    if date_range <= 0:
        date_range = 30  # Fallback a 30 giorni
    
    dates = [start_date + timedelta(days=random.randint(0, date_range)) for _ in range(count)]
    dates.sort()
    
    # Genera titoli casuali
    titles = []
    sources = ['CryptoNews', 'TokenInsider', 'BlockchainToday', 'CoinAnalyst', 'DigitalAssetTimes']
    
    for _ in range(count):
        sentiment_type = random.choices(['positive', 'negative', 'neutral'], weights=[0.4, 0.3, 0.3])[0]
        
        if sentiment_type == 'positive':
            title = random.choice(positive_titles)
        elif sentiment_type == 'negative':
            title = random.choice(negative_titles)
        else:
            title = random.choice(neutral_titles)
        
        titles.append(title)
    
    # Crea il DataFrame
    news_data = pd.DataFrame({
        'date': dates,
        'title': titles,
        'source': [random.choice(sources) for _ in range(count)]
    })
    
    return news_data