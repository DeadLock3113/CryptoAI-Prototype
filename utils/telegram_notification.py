"""
Utility per le notifiche Telegram

Questo modulo fornisce funzioni per inviare notifiche Telegram
agli utenti in base ai movimenti di prezzo delle criptovalute.
"""

import logging
import requests
from datetime import datetime

from database import db
from db_models import User

# Configurazione logging
logger = logging.getLogger(__name__)

def send_telegram_message(bot_token, chat_id, message):
    """
    Invia un messaggio Telegram utilizzando il bot API.
    
    Args:
        bot_token (str): Token del bot Telegram
        chat_id (str): ID della chat Telegram
        message (str): Messaggio da inviare
        
    Returns:
        bool: True se l'invio è avvenuto con successo, False altrimenti
    """
    try:
        # Controllo che token e chat_id siano validi
        if not bot_token or not chat_id:
            logger.error("Token o chat_id mancanti")
            return False
        
        # Prepara l'URL per l'API di Telegram
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        
        # Prepara i dati per la richiesta
        data = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML"
        }
        
        # Invia la richiesta
        response = requests.post(url, data=data, timeout=10)
        
        # Verifica la risposta
        if response.status_code == 200:
            logger.info(f"Messaggio Telegram inviato con successo a {chat_id}")
            return True
        else:
            logger.error(f"Errore nell'invio del messaggio Telegram: {response.text}")
            return False
    
    except Exception as e:
        logger.error(f"Errore nell'invio del messaggio Telegram: {str(e)}")
        return False


def send_price_alert(user, symbol, price, alert_type, change_percent=None):
    """
    Invia un'allerta di prezzo via Telegram.
    
    Args:
        user (User): Oggetto utente con le credenziali Telegram
        symbol (str): Simbolo della criptovaluta
        price (float): Prezzo attuale
        alert_type (str): Tipo di allerta (es. 'new_high', 'new_low', 'threshold', 'update')
        change_percent (float, optional): Percentuale di cambiamento, se applicabile
        
    Returns:
        bool: True se l'invio è avvenuto con successo, False altrimenti
    """
    try:
        # Controllo che l'utente abbia le credenziali Telegram
        if not user or not user.telegram_bot_token or not user.telegram_chat_id:
            logger.error(f"Credenziali Telegram mancanti per l'utente {user.id if user else 'None'}")
            return False
        
        # Prepara il messaggio in base al tipo di allerta
        if alert_type == 'new_high':
            emoji = "🔺"
            title = f"{emoji} NUOVO MASSIMO {emoji}"
            message = f"{title}\n\n"
            message += f"Il prezzo di {symbol} ha raggiunto un nuovo massimo!\n"
            message += f"Prezzo attuale: {price:.5g}"
            
        elif alert_type == 'new_low':
            emoji = "🔻"
            title = f"{emoji} NUOVO MINIMO {emoji}"
            message = f"{title}\n\n"
            message += f"Il prezzo di {symbol} ha raggiunto un nuovo minimo!\n"
            message += f"Prezzo attuale: {price:.5g}"
            
        elif alert_type == 'threshold':
            # Determina se è un movimento positivo o negativo
            if change_percent and change_percent > 0:
                emoji = "📈"
                title = f"{emoji} MOVIMENTO IMPORTANTE {emoji}"
                movement = "aumentato"
            elif change_percent and change_percent < 0:
                emoji = "📉"
                title = f"{emoji} MOVIMENTO IMPORTANTE {emoji}"
                movement = "diminuito"
                change_percent = abs(change_percent)  # Rendiamo positivo per la visualizzazione
            else:
                emoji = "⚠️"
                title = f"{emoji} ALLERTA PREZZO {emoji}"
                movement = "cambiato"
                change_percent = 0
            
            message = f"{title}\n\n"
            message += f"Il prezzo di {symbol} è {movement} del {change_percent:.2f}%!\n"
            message += f"Prezzo attuale: {price:.5g}"
            
        else:  # 'update' o default
            emoji = "ℹ️"
            title = f"{emoji} AGGIORNAMENTO PREZZO {emoji}"
            message = f"{title}\n\n"
            message += f"Aggiornamento prezzo per {symbol}\n"
            message += f"Prezzo attuale: {price:.5g}"
            
            if change_percent is not None:
                # Aggiungi freccia in base al movimento
                arrow = "↗️" if change_percent > 0 else "↘️" if change_percent < 0 else "↔️"
                message += f"\nVariazione: {arrow} {change_percent:.2f}%"
        
        # Aggiungi data e ora
        message += f"\nData: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
        
        # Invia il messaggio
        return send_telegram_message(user.telegram_bot_token, user.telegram_chat_id, message)
    
    except Exception as e:
        logger.error(f"Errore nell'invio dell'allerta di prezzo: {str(e)}")
        return False


def send_balance_update(user, balance_info):
    """
    Invia un aggiornamento sul saldo dell'account via Telegram.
    
    Args:
        user (User): Oggetto utente con le credenziali Telegram
        balance_info (dict): Informazioni sul saldo
        
    Returns:
        bool: True se l'invio è avvenuto con successo, False altrimenti
    """
    try:
        # Controllo che l'utente abbia le credenziali Telegram
        if not user or not user.telegram_bot_token or not user.telegram_chat_id:
            logger.error(f"Credenziali Telegram mancanti per l'utente {user.id if user else 'None'}")
            return False
        
        # Prepara il messaggio
        emoji = "💰"
        title = f"{emoji} AGGIORNAMENTO SALDO {emoji}"
        message = f"{title}\n\n"
        
        # Aggiungi il saldo totale
        total_balance_eur = balance_info.get('total_balance_eur', 0)
        total_balance_usdt = balance_info.get('total_balance_usdt', 0)
        
        message += f"Saldo Totale: {total_balance_eur:.2f} EUR ({total_balance_usdt:.2f} USDT)\n\n"
        
        # Aggiungi le valute
        message += "Dettaglio saldo:\n"
        for currency, amount in balance_info.get('currencies', {}).items():
            # Ignora valute con saldo 0
            if amount > 0.00001:
                eur_value = balance_info.get('currency_values', {}).get(currency, 0)
                message += f"• {currency}: {amount:.6g}"
                if eur_value > 0:
                    message += f" ({eur_value:.2f} EUR)"
                message += "\n"
        
        # Aggiungi data e ora
        message += f"\nData: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}"
        
        # Invia il messaggio
        return send_telegram_message(user.telegram_bot_token, user.telegram_chat_id, message)
    
    except Exception as e:
        logger.error(f"Errore nell'invio dell'aggiornamento del saldo: {str(e)}")
        return False


def send_signal_notification(user_id, message_text):
    """
    Invia una notifica di segnale di trading via Telegram.
    
    Args:
        user_id (int): ID dell'utente
        message_text (str): Testo del messaggio con dettagli del segnale
        
    Returns:
        bool: True se l'invio è avvenuto con successo, False altrimenti
    """
    try:
        # Ottieni l'utente
        user = User.query.get(user_id)
        if not user:
            logger.error(f"Utente non trovato con ID {user_id}")
            return False
        
        # Controllo che l'utente abbia le credenziali Telegram
        if not user.telegram_bot_token or not user.telegram_chat_id:
            logger.error(f"Credenziali Telegram mancanti per l'utente {user_id}")
            return False
        
        # Invia il messaggio
        return send_telegram_message(user.telegram_bot_token, user.telegram_chat_id, message_text)
    
    except Exception as e:
        logger.error(f"Errore nell'invio della notifica di segnale: {str(e)}")
        return False