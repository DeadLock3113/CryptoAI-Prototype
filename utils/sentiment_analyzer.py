"""
Modulo per l'analisi del sentimento di mercato per CryptoTradeAnalyzer.

Questo modulo fornisce funzionalità per analizzare il sentiment di mercato
utilizzando vari approcci (VADER, TextBlob, ecc.) e integrando dati da
diverse fonti (notizie, social media, ecc.).

Author: CryptoTradeAnalyzer Team
"""
import os
import json
import logging
import datetime
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Union, Any

# Import dei moduli per l'analisi del sentimento
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from textblob import TextBlob
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer as VaderSentimentAnalyzer

# Import dei moduli per la visualizzazione
import matplotlib.pyplot as plt

# Setup logging
logger = logging.getLogger(__name__)

# Download necesario per NLTK (solo la prima volta)
try:
    nltk.data.find('vader_lexicon')
except LookupError:
    nltk.download('vader_lexicon')

class SentimentAnalyzer:
    """
    Classe per l'analisi del sentimento di mercato.
    Supporta diverse fonti di dati e diversi metodi di analisi.
    """
    
    def __init__(self):
        """Inizializza l'analizzatore di sentimento."""
        self.vader = VaderSentimentAnalyzer()
        self.nltk_vader = SentimentIntensityAnalyzer()
        
        # Cache per i risultati delle analisi
        self.cache = {}
        
    def analyze_text(self, text: str, method: str = 'vader') -> Dict[str, float]:
        """
        Analizza il sentimento di un testo usando il metodo specificato.
        
        Args:
            text: Il testo da analizzare
            method: Il metodo da usare ('vader', 'textblob', o 'combined')
            
        Returns:
            Un dizionario con i punteggi di sentimento
        """
        if not text:
            return {'compound': 0, 'positive': 0, 'negative': 0, 'neutral': 0}
        
        if method == 'vader':
            return self.vader.polarity_scores(text)
        
        elif method == 'nltk_vader':
            return self.nltk_vader.polarity_scores(text)
            
        elif method == 'textblob':
            blob = TextBlob(text)
            polarity = blob.sentiment.polarity
            subjectivity = blob.sentiment.subjectivity
            
            # Converti il punteggio TextBlob in un formato simile a VADER
            # per coerenza nelle interfacce
            positive = max(0, polarity)
            negative = max(0, -polarity)
            neutral = 1.0 - (positive + negative)
            
            return {
                'compound': polarity,
                'positive': positive,
                'negative': negative,
                'neutral': neutral,
                'subjectivity': subjectivity
            }
            
        elif method == 'combined':
            # Combina i risultati di più metodi
            vader_scores = self.vader.polarity_scores(text)
            textblob_scores = self.analyze_text(text, method='textblob')
            
            # Media ponderata dei punteggi
            compound = (vader_scores['compound'] + textblob_scores['compound']) / 2
            positive = (vader_scores['positive'] + textblob_scores['positive']) / 2
            negative = (vader_scores['negative'] + textblob_scores['negative']) / 2
            neutral = (vader_scores['neutral'] + textblob_scores['neutral']) / 2
            
            return {
                'compound': compound,
                'positive': positive,
                'negative': negative,
                'neutral': neutral
            }
        
        else:
            raise ValueError(f"Metodo non supportato: {method}")
    
    def analyze_news(self, news_data: List[Dict[str, Any]], method: str = 'vader') -> Dict[str, Any]:
        """
        Analizza il sentiment di un insieme di notizie.
        
        Args:
            news_data: Lista di dizionari contenenti notizie (devono avere 'title', 'description', e 'date')
            method: Il metodo di analisi da utilizzare
            
        Returns:
            Un dizionario con i risultati dell'analisi
        """
        if not news_data:
            return {
                'overall_sentiment': 0,
                'daily_sentiment': {},
                'sentiment_trend': []
            }
        
        # Ordina le notizie per data
        sorted_news = sorted(news_data, key=lambda x: x.get('date', datetime.datetime.now()))
        
        # Analizza ogni notizia
        sentiments = []
        for news in sorted_news:
            # Analizza titolo e descrizione insieme
            text = f"{news.get('title', '')} {news.get('description', '')}"
            sentiment = self.analyze_text(text, method=method)
            
            # Aggiungi al risultato
            sentiments.append({
                'date': news.get('date'),
                'title': news.get('title'),
                'sentiment': sentiment
            })
        
        # Calcola il sentiment giornaliero
        daily_sentiment = {}
        for item in sentiments:
            date_str = item['date'].strftime('%Y-%m-%d') if item['date'] else 'unknown'
            if date_str not in daily_sentiment:
                daily_sentiment[date_str] = []
            daily_sentiment[date_str].append(item['sentiment']['compound'])
        
        # Calcola la media per ogni giorno
        for date_str in daily_sentiment:
            daily_sentiment[date_str] = sum(daily_sentiment[date_str]) / len(daily_sentiment[date_str])
        
        # Converti in una serie temporale ordinata
        dates = sorted(daily_sentiment.keys())
        sentiment_trend = [daily_sentiment[date] for date in dates]
        
        # Calcola il sentiment complessivo come media
        overall_sentiment = sum(s['sentiment']['compound'] for s in sentiments) / len(sentiments)
        
        return {
            'overall_sentiment': overall_sentiment,
            'daily_sentiment': daily_sentiment,
            'sentiment_trend': list(zip(dates, sentiment_trend)),
            'detailed_sentiment': sentiments
        }
    
    def generate_sentiment_chart(self, 
                               sentiment_data: Dict[str, Any], 
                               price_data: Optional[pd.DataFrame] = None,
                               title: str = 'Analisi del Sentiment') -> bytes:
        """
        Genera un grafico dell'andamento del sentiment nel tempo.
        
        Args:
            sentiment_data: Dati del sentiment ottenuti da analyze_news
            price_data: Opzionale, DataFrame con i dati dei prezzi per confronto
            title: Titolo del grafico
            
        Returns:
            Immagine del grafico in formato binario
        """
        # Estrai i dati del sentiment
        sentiment_trend = sentiment_data.get('sentiment_trend', [])
        if not sentiment_trend:
            # Crea un'immagine vuota con un messaggio
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.text(0.5, 0.5, 'Nessun dato disponibile', 
                    horizontalalignment='center', verticalalignment='center', transform=ax.transAxes)
            ax.axis('off')
            
            # Salva l'immagine
            from io import BytesIO
            buf = BytesIO()
            plt.savefig(buf, format='png')
            plt.close(fig)
            return buf.getvalue()
        
        # Prepara i dati per il grafico
        dates = [datetime.datetime.strptime(d, '%Y-%m-%d') for d, _ in sentiment_trend]
        sentiment_values = [v for _, v in sentiment_trend]
        
        # Crea il grafico
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # Grafico del sentiment
        color = 'tab:blue'
        ax1.set_xlabel('Data')
        ax1.set_ylabel('Sentiment', color=color)
        ax1.plot(dates, sentiment_values, color=color, marker='o', linestyle='-', label='Sentiment')
        
        # Colora l'area positiva/negativa
        ax1.axhline(y=0, color='gray', linestyle='-', alpha=0.3)
        ax1.fill_between(dates, sentiment_values, 0, where=[v > 0 for v in sentiment_values], 
                         color='green', alpha=0.3, interpolate=True)
        ax1.fill_between(dates, sentiment_values, 0, where=[v < 0 for v in sentiment_values], 
                         color='red', alpha=0.3, interpolate=True)
        
        ax1.tick_params(axis='y', labelcolor=color)
        
        # Se ci sono dati di prezzo, aggiungili su un asse secondario
        if price_data is not None and not price_data.empty:
            ax2 = ax1.twinx()
            color = 'tab:red'
            ax2.set_ylabel('Prezzo', color=color)
            
            # Filtra i dati di prezzo per le date nel grafico del sentiment
            min_date, max_date = min(dates), max(dates)
            filtered_price = price_data[(price_data.index >= min_date) & (price_data.index <= max_date)]
            
            if not filtered_price.empty:
                ax2.plot(filtered_price.index, filtered_price['close'], color=color, 
                         linestyle='--', label='Prezzo')
                ax2.tick_params(axis='y', labelcolor=color)
        
        # Formattazione
        fig.tight_layout()
        plt.grid(True, alpha=0.3)
        plt.title(title)
        
        # Aggiungi legenda
        lines, labels = ax1.get_legend_handles_labels()
        if price_data is not None and not price_data.empty and 'ax2' in locals():
            lines2, labels2 = ax2.get_legend_handles_labels()
            lines += lines2
            labels += labels2
        
        fig.legend(lines, labels, loc="upper right", bbox_to_anchor=(0.95, 0.95))
        
        # Salva l'immagine
        from io import BytesIO
        buf = BytesIO()
        plt.savefig(buf, format='png')
        plt.close(fig)
        return buf.getvalue()
    
    def interpret_sentiment(self, sentiment_score: float) -> str:
        """
        Interpreta un punteggio di sentiment in termini qualitativi.
        
        Args:
            sentiment_score: Punteggio compound del sentiment (tra -1 e 1)
            
        Returns:
            Descrizione qualitativa del sentiment
        """
        if sentiment_score >= 0.5:
            return "Molto Positivo"
        elif sentiment_score >= 0.1:
            return "Positivo"
        elif sentiment_score > -0.1:
            return "Neutrale"
        elif sentiment_score > -0.5:
            return "Negativo"
        else:
            return "Molto Negativo"
    
    def get_simplified_score(self, sentiment_score: float) -> Tuple[str, str]:
        """
        Restituisce una versione semplificata del punteggio di sentiment.
        
        Args:
            sentiment_score: Punteggio compound del sentiment (tra -1 e 1)
            
        Returns:
            Tuple con (categoria, colore)
        """
        if sentiment_score >= 0.5:
            return ("Rialzista", "success")
        elif sentiment_score >= 0.1:
            return ("Moderatamente Rialzista", "info")
        elif sentiment_score > -0.1:
            return ("Neutrale", "secondary")
        elif sentiment_score > -0.5:
            return ("Moderatamente Ribassista", "warning")
        else:
            return ("Ribassista", "danger")
    
    def generate_sentiment_summary(self, sentiment_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Genera un riepilogo dell'analisi del sentiment.
        
        Args:
            sentiment_data: Dati del sentiment ottenuti da analyze_news
            
        Returns:
            Dizionario con il riepilogo dell'analisi
        """
        overall_sentiment = sentiment_data.get('overall_sentiment', 0)
        sentiment_trend = sentiment_data.get('sentiment_trend', [])
        
        # Interpretazione generale
        interpretation = self.interpret_sentiment(overall_sentiment)
        simplified, color = self.get_simplified_score(overall_sentiment)
        
        # Calcola la tendenza del sentiment (ultimi N valori)
        trend_direction = None
        trend_strength = None
        
        if len(sentiment_trend) >= 3:
            # Prendi gli ultimi 3 valori
            last_values = [v for _, v in sentiment_trend[-3:]]
            
            # Calcola la pendenza della retta di regressione
            x = np.arange(len(last_values))
            slope, _ = np.polyfit(x, last_values, 1)
            
            # Determina la direzione e la forza del trend
            if abs(slope) < 0.01:
                trend_direction = "stabile"
                trend_strength = "debole"
            else:
                trend_direction = "in crescita" if slope > 0 else "in diminuzione"
                trend_strength = "forte" if abs(slope) > 0.05 else "moderato"
        
        return {
            'score': overall_sentiment,
            'interpretation': interpretation,
            'simplified': simplified,
            'color': color,
            'trend_direction': trend_direction,
            'trend_strength': trend_strength
        }

# Funzione di utilità per simulare i dati di notizie (quando non disponibili da API esterne)
def generate_sample_news_data(symbol: str, 
                           start_date: datetime.datetime, 
                           end_date: datetime.datetime, 
                           num_news: int = 30) -> List[Dict[str, Any]]:
    """
    Genera dati di esempio per le notizie quando non sono disponibili dati reali.
    
    Args:
        symbol: Simbolo della criptovaluta
        start_date: Data di inizio
        end_date: Data di fine
        num_news: Numero di notizie da generare
        
    Returns:
        Lista di notizie generate
    """
    # Titoli e descrizioni di esempio per vari scenari di mercato
    positive_templates = [
        {
            'title': f"{symbol} raggiunge nuovi massimi mentre il mercato cresce",
            'description': f"Il prezzo di {symbol} è salito significativamente oggi, raggiungendo nuovi massimi storici nel contesto di un mercato rialzista."
        },
        {
            'title': f"Gli analisti prevedono ulteriori guadagni per {symbol}",
            'description': f"Secondo diversi esperti di mercato, {symbol} potrebbe continuare la sua tendenza al rialzo nelle prossime settimane grazie a fondamentali solidi."
        },
        {
            'title': f"Crescente adozione istituzionale spinge in alto {symbol}",
            'description': f"Un numero crescente di istituzioni finanziarie sta investendo in {symbol}, contribuendo all'aumento del prezzo e della fiducia nel mercato."
        },
        {
            'title': f"Aggiornamento tecnologico promettente per {symbol}",
            'description': f"La rete di {symbol} ha implementato con successo un importante aggiornamento tecnologico che promette di migliorare la sicurezza e la scalabilità."
        }
    ]
    
    negative_templates = [
        {
            'title': f"{symbol} in forte calo dopo le notizie normative",
            'description': f"Il prezzo di {symbol} è sceso bruscamente oggi in seguito all'annuncio di possibili nuove regolamentazioni nel settore delle criptovalute."
        },
        {
            'title': f"Gli investitori vendono {symbol} nel mezzo dell'incertezza del mercato",
            'description': f"L'incertezza economica globale ha spinto molti investitori a liquidare le loro posizioni in {symbol}, causando una significativa pressione di vendita."
        },
        {
            'title': f"Problemi tecnici colpiscono la rete di {symbol}",
            'description': f"La rete di {symbol} ha riscontrato diversi problemi tecnici nelle ultime 24 ore, sollevando preoccupazioni sulla sua stabilità e affidabilità."
        },
        {
            'title': f"Gli analisti avvertono di una possibile correzione per {symbol}",
            'description': f"Diversi analisti di mercato ritengono che {symbol} sia attualmente sopravvalutato e prevedono una correzione del prezzo nel breve termine."
        }
    ]
    
    neutral_templates = [
        {
            'title': f"{symbol} stabilizza dopo la recente volatilità",
            'description': f"Dopo un periodo di significativa volatilità, il prezzo di {symbol} si è stabilizzato in un intervallo relativamente ristretto nelle ultime sessioni di trading."
        },
        {
            'title': f"Il volume di trading di {symbol} diminuisce durante il weekend",
            'description': f"Come previsto, il volume di trading di {symbol} è diminuito durante il fine settimana, con il prezzo che ha mostrato movimenti limitati."
        },
        {
            'title': f"Nuovo exchange aggiunge il supporto per {symbol}",
            'description': f"Un exchange di criptovalute ha annunciato l'aggiunta di {symbol} alla sua piattaforma, ampliando le opzioni di trading per questa criptovaluta."
        },
        {
            'title': f"Rapporto trimestrale pubblicato dalla fondazione {symbol}",
            'description': f"La fondazione {symbol} ha pubblicato il suo rapporto trimestrale, delineando i progressi del progetto e i piani futuri senza particolari sorprese."
        }
    ]
    
    # Genera date casuali nell'intervallo
    delta = (end_date - start_date).total_seconds()
    random_dates = [start_date + datetime.timedelta(seconds=np.random.randint(0, delta)) 
                  for _ in range(num_news)]
    random_dates.sort()
    
    # Genera notizie casuali
    news_data = []
    for date in random_dates:
        # Scegli casualmente il "sentimento" della notizia
        sentiment_type = np.random.choice(['positive', 'negative', 'neutral'], 
                                        p=[0.4, 0.3, 0.3])
        
        if sentiment_type == 'positive':
            template = np.random.choice(positive_templates)
        elif sentiment_type == 'negative':
            template = np.random.choice(negative_templates)
        else:
            template = np.random.choice(neutral_templates)
        
        news_data.append({
            'title': template['title'],
            'description': template['description'],
            'date': date,
            'source': np.random.choice(['CryptoNews', 'BlockchainReport', 'CoinDesk', 'CryptoMarketUpdate']),
            'url': f"https://example.com/news/{np.random.randint(1000, 9999)}",
            'sentiment_type': sentiment_type  # Solo per riferimento, non usato nell'analisi
        })
    
    return news_data