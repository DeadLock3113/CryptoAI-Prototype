"""
Utility per le notifiche Telegram

Questo modulo fornisce funzioni per inviare notifiche Telegram
agli utenti in base ai movimenti di prezzo delle criptovalute.
"""

import requests
import logging
from datetime import datetime

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
    if not bot_token or not chat_id:
        logger.warning("Bot token o chat ID mancanti. Impossibile inviare notifica Telegram.")
        return False
    
    try:
        url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
        data = {
            "chat_id": chat_id,
            "text": message,
            "parse_mode": "HTML"
        }
        
        response = requests.post(url, data=data, timeout=5)
        
        if response.status_code == 200:
            logger.info(f"Messaggio Telegram inviato con successo a {chat_id}")
            return True
        else:
            logger.error(f"Errore nell'invio del messaggio Telegram: {response.status_code}, {response.text}")
            return False
            
    except Exception as e:
        logger.error(f"Errore durante l'invio del messaggio Telegram: {str(e)}")
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
    if not user or not user.telegram_bot_token or not user.telegram_chat_id:
        return False
    
    # Timestamp corrente
    now = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    
    # Costruisci il messaggio in base al tipo di allerta
    if alert_type == 'new_high':
        message = f"🔺 <b>NUOVO MASSIMO per {symbol}</b> 🔺\n\n"
        message += f"Prezzo: <b>{price:.8f}</b>\n"
        message += f"Variazione: <b>+{change_percent:.2f}%</b>\n"
        message += f"Data: {now}\n"
    
    elif alert_type == 'new_low':
        message = f"🔻 <b>NUOVO MINIMO per {symbol}</b> 🔻\n\n"
        message += f"Prezzo: <b>{price:.8f}</b>\n"
        message += f"Variazione: <b>{change_percent:.2f}%</b>\n"
        message += f"Data: {now}\n"
    
    elif alert_type == 'threshold':
        direction = "🔺" if change_percent > 0 else "🔻"
        message = f"{direction} <b>MOVIMENTO SIGNIFICATIVO per {symbol}</b> {direction}\n\n"
        message += f"Prezzo: <b>{price:.8f}</b>\n"
        message += f"Variazione: <b>{change_percent:+.2f}%</b>\n"
        message += f"Data: {now}\n"
    
    elif alert_type == 'update':
        # Notifica di aggiornamento prezzo normale
        emoji = "🟢" if change_percent >= 0 else "🔴"
        message = f"{emoji} <b>Aggiornamento {symbol}</b>\n\n"
        message += f"Prezzo: <b>{price:.8f}</b>\n"
        message += f"Variazione: <b>{change_percent:+.2f}%</b>\n"
        message += f"Data: {now}\n"
    
    else:
        # Tipo di messaggio generico
        message = f"<b>Aggiornamento {symbol}</b>\n\n"
        message += f"Prezzo: <b>{price:.8f}</b>\n"
        message += f"Data: {now}\n"
    
    # Invia il messaggio Telegram
    return send_telegram_message(user.telegram_bot_token, user.telegram_chat_id, message)

def send_balance_update(user, balance_info):
    """
    Invia un aggiornamento sul saldo dell'account via Telegram.
    
    Args:
        user (User): Oggetto utente con le credenziali Telegram
        balance_info (dict): Informazioni sul saldo
        
    Returns:
        bool: True se l'invio è avvenuto con successo, False altrimenti
    """
    if not user or not user.telegram_bot_token or not user.telegram_chat_id:
        return False
    
    # Timestamp corrente
    now = datetime.now().strftime("%d-%m-%Y %H:%M:%S")
    
    # Costruisci il messaggio con le informazioni di saldo
    message = f"💰 <b>AGGIORNAMENTO SALDO</b> 💰\n\n"
    
    # Aggiungi saldo totale
    message += f"Saldo totale: <b>{balance_info['total_balance']:.2f} USDT</b>\n\n"
    
    # Aggiungi dettagli per ogni valuta
    message += "<b>Dettaglio:</b>\n"
    for currency, amount in balance_info['currencies'].items():
        message += f"- {currency}: <b>{amount:.8f}</b>\n"
    
    message += f"\nData: {now}\n"
    
    # Invia il messaggio Telegram
    return send_telegram_message(user.telegram_bot_token, user.telegram_chat_id, message)

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
        # Importa i modelli necessari
        from db_models import User
        from app import db
        
        # Ottieni l'utente dal database
        user = User.query.get(user_id)
        
        if not user or not user.telegram_bot_token or not user.telegram_chat_id:
            logger.warning(f"Utente {user_id} non ha configurato Telegram")
            return False
        
        # Invia il messaggio già formattato
        return send_telegram_message(user.telegram_bot_token, user.telegram_chat_id, message_text)
        
    except Exception as e:
        logger.error(f"Errore nell'invio della notifica di segnale: {str(e)}")
        return False