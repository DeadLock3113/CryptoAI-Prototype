"""
Interazione con le API degli exchange di criptovalute

Questo modulo fornisce funzioni per interagire con le API degli exchange
come Binance e Kraken per ottenere informazioni sul saldo, prezzi, ecc.
"""

import logging
import hmac
import hashlib
import time
import json
import base64
import requests
from urllib.parse import urlencode

logger = logging.getLogger(__name__)

class ExchangeAPIError(Exception):
    """Eccezione per errori nelle API degli exchange"""
    pass

def get_binance_account_balance(api_key, api_secret):
    """
    Ottiene il saldo dell'account Binance.
    
    Args:
        api_key (str): Chiave API di Binance
        api_secret (str): Secret API di Binance
        
    Returns:
        dict: Informazioni sul saldo
        
    Raises:
        ExchangeAPIError: Se si verifica un errore durante la richiesta
    """
    if not api_key or not api_secret:
        logger.warning("API key o API secret mancanti per Binance")
        raise ExchangeAPIError("API key o API secret mancanti per Binance")
    
    try:
        # Endpoint e parametri per il saldo dell'account
        endpoint = 'https://api.binance.com/api/v3/account'
        
        # Aggiungi timestamp per la firma
        timestamp = int(time.time() * 1000)
        params = {
            'timestamp': timestamp
        }
        
        # Genera la firma HMAC SHA256
        query_string = urlencode(params)
        signature = hmac.new(
            api_secret.encode('utf-8'),
            query_string.encode('utf-8'),
            hashlib.sha256
        ).hexdigest()
        
        params['signature'] = signature
        
        # Intestazioni con l'API key
        headers = {
            'X-MBX-APIKEY': api_key
        }
        
        # Effettua la richiesta
        response = requests.get(endpoint, params=params, headers=headers, timeout=10)
        
        # Verifica la risposta
        if response.status_code == 200:
            data = response.json()
            
            # Estrai i saldi
            balances = []
            for asset in data['balances']:
                free = float(asset['free'])
                locked = float(asset['locked'])
                total = free + locked
                
                if total > 0:  # Ignora saldi zero
                    balances.append({
                        'currency': asset['asset'],
                        'free': free,
                        'locked': locked,
                        'total': total
                    })
            
            # Converti i saldi in formato unificato
            result = {
                'exchange': 'Binance',
                'balances': balances
            }
            
            return result
        else:
            error_msg = f"Errore Binance API: {response.status_code} - {response.text}"
            logger.error(error_msg)
            raise ExchangeAPIError(error_msg)
            
    except Exception as e:
        error_msg = f"Errore durante la richiesta all'API Binance: {str(e)}"
        logger.error(error_msg)
        raise ExchangeAPIError(error_msg)

def get_kraken_account_balance(api_key, api_secret):
    """
    Ottiene il saldo dell'account Kraken.
    
    Args:
        api_key (str): Chiave API di Kraken
        api_secret (str): Secret API di Kraken
        
    Returns:
        dict: Informazioni sul saldo
        
    Raises:
        ExchangeAPIError: Se si verifica un errore durante la richiesta
    """
    if not api_key or not api_secret:
        logger.warning("API key o API secret mancanti per Kraken")
        raise ExchangeAPIError("API key o API secret mancanti per Kraken")
    
    try:
        # Endpoint e parametri per il saldo dell'account
        api_url = "https://api.kraken.com"
        endpoint = "/0/private/Balance"
        
        # Nonce per la sicurezza (timestamp)
        nonce = str(int(time.time() * 1000))
        
        # Dati della richiesta
        data = {
            "nonce": nonce,
        }
        
        # Genera la firma
        post_data = urlencode(data)
        encoded = (str(data['nonce']) + post_data).encode()
        message = endpoint.encode() + hashlib.sha256(encoded).digest()
        
        # Firma con HMAC-SHA512
        signature = hmac.new(
            base64.b64decode(api_secret),
            message,
            hashlib.sha512
        ).digest()
        
        # Intestazioni con la firma
        headers = {
            'API-Key': api_key,
            'API-Sign': base64.b64encode(signature).decode()
        }
        
        # Effettua la richiesta
        response = requests.post(
            api_url + endpoint,
            data=data,
            headers=headers,
            timeout=10
        )
        
        # Verifica la risposta
        if response.status_code == 200:
            data = response.json()
            
            if data['error']:
                error_msg = f"Errore Kraken API: {data['error']}"
                logger.error(error_msg)
                raise ExchangeAPIError(error_msg)
            
            # Estrai i saldi
            balances = []
            for currency, amount in data['result'].items():
                # Kraken aggiunge prefissi X e Z alle valute, rimuoviamoli
                clean_currency = currency
                if currency.startswith('X') or currency.startswith('Z'):
                    clean_currency = currency[1:]
                
                balances.append({
                    'currency': clean_currency,
                    'free': float(amount),  # Kraken non distingue tra free e locked
                    'locked': 0.0,
                    'total': float(amount)
                })
            
            # Converti i saldi in formato unificato
            result = {
                'exchange': 'Kraken',
                'balances': balances
            }
            
            return result
        else:
            error_msg = f"Errore Kraken API: {response.status_code} - {response.text}"
            logger.error(error_msg)
            raise ExchangeAPIError(error_msg)
            
    except Exception as e:
        error_msg = f"Errore durante la richiesta all'API Kraken: {str(e)}"
        logger.error(error_msg)
        raise ExchangeAPIError(error_msg)

def get_account_balances(user):
    """
    Ottiene il saldo degli account dell'utente da tutti gli exchange configurati.
    
    Args:
        user (User): Oggetto utente con le credenziali API
        
    Returns:
        dict: Informazioni sui saldi degli exchange
    """
    # Se l'utente non ha configurato nessuna API, mostriamo dei dati di esempio
    if not (user.binance_api_key or user.kraken_api_key):
        # Crea un saldo di esempio per la demo
        eur_usd_rate = get_eur_usd_exchange_rate()
        
        return {
            'exchanges': [{
                'exchange': 'Demo',
                'balances': [
                    {'currency': 'BTC', 'free': 0.5, 'locked': 0.0, 'total': 0.5, 'usdt_value': 20000.0, 'eur_value': 18000.0},
                    {'currency': 'ETH', 'free': 2.5, 'locked': 0.0, 'total': 2.5, 'usdt_value': 6250.0, 'eur_value': 5625.0},
                    {'currency': 'USDT', 'free': 1000.0, 'locked': 0.0, 'total': 1000.0, 'usdt_value': 1000.0, 'eur_value': 900.0},
                    {'currency': 'EUR', 'free': 3000.0, 'locked': 0.0, 'total': 3000.0, 'usdt_value': 3350.0, 'eur_value': 3000.0}
                ]
            }],
            'currencies': {'BTC': 0.5, 'ETH': 2.5, 'USDT': 1000.0, 'EUR': 3000.0},
            'currency_values': {'BTC': 20000.0, 'ETH': 6250.0, 'USDT': 1000.0, 'EUR': 3350.0},
            'total_balance_usdt': 30600.0,
            'total_balance_eur': 27540.0,
            'eur_usd_rate': eur_usd_rate
        }

    results = {
        'exchanges': [],
        'currencies': {},
        'currency_values': {},  # Valori in USDT per ogni valuta
        'total_balance_usdt': 0.0,
        'total_balance_eur': 0.0,
        'eur_usd_rate': 0.0
    }
    
    # Ottieni prezzi attuali in USDT per la conversione
    prices = get_usdt_prices()
    
    # Ottieni tasso di cambio EUR/USD per la conversione
    eur_usd_rate = get_eur_usd_exchange_rate()
    results['eur_usd_rate'] = eur_usd_rate
    
    # Aggiungiamo valori fissi per EUR e USD
    eur_to_usdt_rate = 1.0 / eur_usd_rate  # 1 EUR = X USDT
    usd_to_usdt_rate = 1.0  # 1 USD = 1 USDT (approssimativamente)
    
    # Aggiorniamo anche i prezzi con valori hardcoded per le valute non trovate
    if 'BTC' not in prices:
        prices['BTC'] = 40000.0
    if 'ETH' not in prices:
        prices['ETH'] = 2500.0
    if 'XBT' not in prices:  # Kraken usa XBT al posto di BTC
        prices['XBT'] = prices.get('BTC', 40000.0)
    
    # Controlla se l'utente ha configurato Binance
    if user.binance_api_key and user.binance_api_secret:
        try:
            binance_balance = get_binance_account_balance(
                user.binance_api_key,
                user.binance_api_secret
            )
            
            # Calcola il valore totale in USDT
            for balance in binance_balance['balances']:
                currency = balance['currency']
                amount = balance['total']
                
                # Ignora saldi zero
                if amount <= 0:
                    continue
                
                # Aggiungi alla lista valute aggregate
                if currency not in results['currencies']:
                    results['currencies'][currency] = 0.0
                    results['currency_values'][currency] = 0.0
                
                results['currencies'][currency] += amount
                
                # Converti in USDT 
                usdt_value = 0.0
                
                if currency in prices:
                    usdt_value = amount * prices[currency]
                elif currency == 'USDT':
                    usdt_value = amount
                elif currency == 'EUR':
                    usdt_value = amount * eur_to_usdt_rate
                elif currency == 'USD':
                    usdt_value = amount * usd_to_usdt_rate
                
                # Aggiungi il valore anche se è zero per mostrare tutte le valute
                balance['usdt_value'] = usdt_value
                balance['eur_value'] = usdt_value * eur_usd_rate
                results['total_balance_usdt'] += usdt_value
                results['currency_values'][currency] += usdt_value
            
            results['exchanges'].append(binance_balance)
        except ExchangeAPIError as e:
            logger.error(f"Errore nell'ottenere il saldo Binance: {str(e)}")
    
    # Controlla se l'utente ha configurato Kraken
    if user.kraken_api_key and user.kraken_api_secret:
        try:
            kraken_balance = get_kraken_account_balance(
                user.kraken_api_key,
                user.kraken_api_secret
            )
            
            # Calcola il valore totale in USDT
            for balance in kraken_balance['balances']:
                currency = balance['currency']
                amount = balance['total']
                
                # Ignora saldi zero
                if amount <= 0:
                    continue
                
                # Aggiungi alla lista valute aggregate
                if currency not in results['currencies']:
                    results['currencies'][currency] = 0.0
                    results['currency_values'][currency] = 0.0
                
                results['currencies'][currency] += amount
                
                # Converti in USDT con supporto specifico per valute Kraken
                usdt_value = 0.0
                
                if currency in prices:
                    usdt_value = amount * prices[currency]
                elif currency == 'USDT':
                    usdt_value = amount
                elif currency == 'EUR':
                    usdt_value = amount * eur_to_usdt_rate
                elif currency == 'USD':
                    usdt_value = amount * usd_to_usdt_rate
                elif currency == 'XBT':  # Supporto per Bitcoin in Kraken (XBT)
                    usdt_value = amount * prices.get('XBT', 40000.0)
                
                # Aggiungi il valore anche se è zero per mostrare tutte le valute
                balance['usdt_value'] = usdt_value
                balance['eur_value'] = usdt_value * eur_usd_rate
                results['total_balance_usdt'] += usdt_value
                results['currency_values'][currency] += usdt_value
            
            results['exchanges'].append(kraken_balance)
        except ExchangeAPIError as e:
            logger.error(f"Errore nell'ottenere il saldo Kraken: {str(e)}")
    
    # Calcola il saldo totale in EUR
    results['total_balance_eur'] = results['total_balance_usdt'] * eur_usd_rate
    
    # Aggiungiamo un log per debug
    logger.debug(f"Account balance: {results}")
    
    return results

def get_usdt_prices():
    """
    Ottiene i prezzi correnti delle principali criptovalute in USDT.
    
    Returns:
        dict: Prezzi delle criptovalute in USDT
    """
    try:
        # Usa l'API pubblica di Binance per ottenere i prezzi
        response = requests.get('https://api.binance.com/api/v3/ticker/price', timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            prices = {}
            
            for item in data:
                symbol = item['symbol']
                price = float(item['price'])
                
                # Considera solo le coppie con USDT
                if symbol.endswith('USDT'):
                    base_currency = symbol[:-4]  # Rimuovi 'USDT' dalla fine
                    prices[base_currency] = price
            
            return prices
        else:
            logger.error(f"Errore nell'ottenere i prezzi: {response.status_code} - {response.text}")
            return {}
    except Exception as e:
        logger.error(f"Errore durante la richiesta dei prezzi: {str(e)}")
        return {}
        
def get_eur_usd_exchange_rate():
    """
    Ottiene il tasso di cambio EUR/USD corrente.
    
    Returns:
        float: Tasso di cambio EUR/USD (1 USD = X EUR)
    """
    try:
        # Utilizziamo l'API di ExchangeRate-API per ottenere il tasso di cambio
        response = requests.get('https://open.er-api.com/v6/latest/USD', timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            # Ottieni il tasso di cambio EUR
            eur_rate = data['rates']['EUR']
            return eur_rate
        else:
            logger.error(f"Errore nell'ottenere il tasso di cambio: {response.status_code} - {response.text}")
            # Valore di fallback se non riusciamo a ottenere il tasso aggiornato
            return 0.85  # Valore approssimativo EUR/USD
    except Exception as e:
        logger.error(f"Errore durante la richiesta del tasso di cambio: {str(e)}")
        return 0.85  # Valore approssimativo EUR/USD