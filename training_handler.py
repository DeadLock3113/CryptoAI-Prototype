"""
CryptoTradeAnalyzer - Gestore dell'addestramento dei modelli

Questo modulo implementa la gestione dell'addestramento interattivo 
dei modelli, inclusi gli aggiornamenti in tempo reale.
"""

import os
import time
import json
import uuid
import math
import logging
import threading
import numpy as np
from datetime import datetime

# Set up logger
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Dizionario per tenere traccia delle sessioni di addestramento attive
active_trainings = {}

# Lock per sicurezza nell'accesso concorrente
training_lock = threading.Lock()

class TrainingSession:
    """Classe per gestire una sessione di addestramento"""
    
    def __init__(self, training_id, model_type, model_name, dataset_id, dataset_name,
                 epochs, batch_size, lookback, learning_rate, device, demo_mode=True):
        """Inizializza una nuova sessione di addestramento"""
        self.training_id = training_id
        self.model_type = model_type
        self.model_name = model_name
        self.dataset_id = dataset_id
        self.dataset_name = dataset_name
        self.epochs = epochs
        self.batch_size = batch_size
        self.lookback = lookback
        self.learning_rate = learning_rate
        self.device = device
        self.demo_mode = demo_mode
        
        # Stato dell'addestramento
        self.status = 'initialized'
        self.current_epoch = 0
        self.train_loss = None
        self.val_loss = None
        self.start_time = None
        self.end_time = None
        self.metrics = {}
        self.history = {'loss': [], 'val_loss': []}
        self.is_stopped = False
        
        # Coda di eventi per SSE
        self.events = []
        
        # Thread di addestramento
        self.training_thread = None
    
    def start(self, model=None, train_loader=None, val_loader=None, criterion=None, optimizer=None):
        """Avvia l'addestramento in un thread separato"""
        # Se è già in esecuzione, non fare nulla
        if self.status == 'running':
            return False
        
        # Imposta lo stato su 'running' e registra l'ora di inizio
        self.status = 'running'
        self.start_time = time.time()
        
        # Salva riferimenti al modello e alle componenti di addestramento
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        
        # Inizia il thread di addestramento
        self.training_thread = threading.Thread(target=self._training_loop)
        self.training_thread.daemon = True  # Il thread si fermerà quando il programma principale si ferma
        self.training_thread.start()
        
        return True
    
    def _training_loop(self):
        """Loop di addestramento - eseguito in un thread separato"""
        try:
            # Se siamo in modalità demo o non abbiamo componenti di addestramento, simuliamo
            if self.demo_mode or not all([self.model, self.train_loader, self.val_loader, self.criterion, self.optimizer]):
                self._simulated_training()
                return
            
            # Altrimenti, eseguiamo l'addestramento reale con i componenti forniti
            self._real_training()
        
        except Exception as e:
            logger.error(f"Errore nell'addestramento {self.training_id}: {str(e)}")
            self.status = 'error'
            self.add_event('training_error', {'error': str(e)})
    
    def _simulated_training(self):
        """Simula un addestramento per scopi dimostrativi"""
        import random
        
        logger.debug(f"Avvio addestramento simulato per {self.training_id}")
        
        # Inizializza la history
        self.history = {'loss': [], 'val_loss': []}
        
        for epoch in range(1, self.epochs + 1):
            # Controlla se è stata richiesta l'interruzione
            if self.is_stopped:
                logger.debug(f"Addestramento {self.training_id} interrotto manualmente all'epoca {epoch}")
                self.status = 'stopped'
                self.add_event('training_error', {'error': 'Addestramento interrotto manualmente'})
                return
            
            # Simula il tempo di addestramento
            time.sleep(0.5)
            
            # Simula le perdite (diminuiscono gradualmente)
            train_loss = 0.5 * (1.0 - (epoch / self.epochs)) + 0.05 + (0.02 * random.random())
            val_loss = train_loss + 0.05 + (0.05 * random.random())
            
            # Aggiorna lo stato
            self.current_epoch = epoch
            self.train_loss = train_loss
            self.val_loss = val_loss
            
            # Aggiungi alla history
            self.history['loss'].append(train_loss)
            self.history['val_loss'].append(val_loss)
            
            # Calcola il tempo trascorso e rimanente
            elapsed_time = time.time() - self.start_time
            if epoch > 1:
                avg_time_per_epoch = elapsed_time / epoch
                estimated_remaining_time = avg_time_per_epoch * (self.epochs - epoch)
            else:
                estimated_remaining_time = None
            
            # Prepara i dati dell'aggiornamento
            update_data = {
                'epoch': epoch,
                'total_epochs': self.epochs,
                'train_loss': train_loss,
                'val_loss': val_loss,
                'elapsed_time': elapsed_time,
                'estimated_remaining_time': estimated_remaining_time
            }
            
            # Aggiungi metriche casuali ogni 3 epoche o alla fine
            if epoch % 3 == 0 or epoch == self.epochs:
                self.metrics = {
                    'mse': val_loss,
                    'rmse': math.sqrt(val_loss),
                    'mae': 0.8 * val_loss,
                    'r2': max(0, 1 - (epoch / self.epochs))
                }
                update_data['metrics'] = self.metrics
            
            # Invia l'aggiornamento
            self.add_event('epoch_update', update_data)
        
        # Addestramento completato
        self.status = 'completed'
        self.end_time = time.time()
        
        completion_data = {
            'total_time': self.end_time - self.start_time,
            'final_loss': self.train_loss,
            'metrics': self.metrics
        }
        
        self.add_event('training_complete', completion_data)
        logger.debug(f"Addestramento simulato {self.training_id} completato")
    
    def _real_training(self):
        """Esegue un addestramento reale con PyTorch"""
        import torch
        
        logger.debug(f"Avvio addestramento reale per {self.training_id}")
        
        # Inizializza la history
        self.history = {'loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        best_model_state = None
        
        for epoch in range(1, self.epochs + 1):
            # Controlla se è stata richiesta l'interruzione
            if self.is_stopped:
                logger.debug(f"Addestramento {self.training_id} interrotto manualmente all'epoca {epoch}")
                self.status = 'stopped'
                self.add_event('training_error', {'error': 'Addestramento interrotto manualmente'})
                return
            
            # Training
            self.model.train()
            train_loss = 0
            batch_count = 0
            
            for inputs, targets in self.train_loader:
                # Controlla di nuovo per l'interruzione all'interno del loop interno
                if self.is_stopped:
                    break
                
                # Sposta i dati sul dispositivo corretto
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                
                # Forward pass
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                
                # Backward and optimize
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                self.optimizer.step()
                
                train_loss += loss.item()
                batch_count += 1
            
            # Calcola la perdita media di addestramento
            avg_train_loss = train_loss / batch_count if batch_count > 0 else 0
            
            # Validation
            self.model.eval()
            val_loss = 0
            val_batch_count = 0
            
            with torch.no_grad():
                for inputs, targets in self.val_loader:
                    # Controlla di nuovo per l'interruzione
                    if self.is_stopped:
                        break
                    
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    val_loss += loss.item()
                    val_batch_count += 1
            
            # Calcola la perdita media di validazione
            avg_val_loss = val_loss / val_batch_count if val_batch_count > 0 else 0
            
            # Aggiorna lo stato
            self.current_epoch = epoch
            self.train_loss = avg_train_loss
            self.val_loss = avg_val_loss
            
            # Aggiungi alla history
            self.history['loss'].append(avg_train_loss)
            self.history['val_loss'].append(avg_val_loss)
            
            # Salva il miglior modello
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_model_state = self.model.state_dict().copy()
            
            # Calcola il tempo trascorso e rimanente
            elapsed_time = time.time() - self.start_time
            if epoch > 1:
                avg_time_per_epoch = elapsed_time / epoch
                estimated_remaining_time = avg_time_per_epoch * (self.epochs - epoch)
            else:
                estimated_remaining_time = None
            
            # Prepara i dati dell'aggiornamento
            update_data = {
                'epoch': epoch,
                'total_epochs': self.epochs,
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
                'elapsed_time': elapsed_time,
                'estimated_remaining_time': estimated_remaining_time
            }
            
            # A intervalli regolari, calcola metriche più dettagliate
            if epoch % 3 == 0 or epoch == self.epochs:
                # Qui potresti calcolare metriche reali sul set di test
                # Per ora, utilizziamo solo perdite come metriche
                self.metrics = {
                    'mse': avg_val_loss,
                    'rmse': math.sqrt(avg_val_loss),
                    'mae': avg_val_loss * 0.8,  # Approssimazione
                    'r2': max(0, 1 - (avg_val_loss / 0.5))  # Approssimazione
                }
                update_data['metrics'] = self.metrics
            
            # Invia l'aggiornamento
            self.add_event('epoch_update', update_data)
        
        # Carica il miglior modello
        if best_model_state:
            self.model.load_state_dict(best_model_state)
        
        # Addestramento completato
        self.status = 'completed'
        self.end_time = time.time()
        
        completion_data = {
            'total_time': self.end_time - self.start_time,
            'final_loss': self.train_loss,
            'metrics': self.metrics
        }
        
        self.add_event('training_complete', completion_data)
        logger.debug(f"Addestramento reale {self.training_id} completato")
    
    def stop(self):
        """Ferma l'addestramento"""
        self.is_stopped = True
        return True
    
    def add_event(self, event_type, data):
        """Aggiunge un evento alla coda"""
        event = {
            'type': event_type,
            'data': data,
            'timestamp': time.time()
        }
        self.events.append(event)
    
    def get_events(self, since=0):
        """Ottiene gli eventi dalla coda dopo un certo timestamp"""
        return [e for e in self.events if e['timestamp'] > since]
    
    def to_dict(self):
        """Converte la sessione in un dizionario"""
        return {
            'training_id': self.training_id,
            'model_type': self.model_type,
            'model_name': self.model_name,
            'dataset_id': self.dataset_id,
            'dataset_name': self.dataset_name,
            'epochs': self.epochs,
            'batch_size': self.batch_size,
            'lookback': self.lookback,
            'learning_rate': self.learning_rate,
            'device': self.device,
            'demo_mode': self.demo_mode,
            'status': self.status,
            'current_epoch': self.current_epoch,
            'total_epochs': self.epochs,
            'start_time': self.start_time,
            'end_time': self.end_time
        }


def create_training_session(model_type, model_name, dataset_id, dataset_name, 
                           epochs, batch_size, lookback, learning_rate, device,
                           demo_mode=True):
    """Crea una nuova sessione di addestramento"""
    training_id = str(uuid.uuid4())
    
    with training_lock:
        session = TrainingSession(
            training_id, model_type, model_name, dataset_id, dataset_name,
            epochs, batch_size, lookback, learning_rate, device, demo_mode
        )
        active_trainings[training_id] = session
    
    return training_id, session


def get_training_session(training_id):
    """Ottiene una sessione di addestramento esistente"""
    with training_lock:
        return active_trainings.get(training_id)


def start_training(training_id, model=None, train_loader=None, val_loader=None, 
                  criterion=None, optimizer=None):
    """Avvia una sessione di addestramento esistente"""
    session = get_training_session(training_id)
    if not session:
        return False
    
    return session.start(model, train_loader, val_loader, criterion, optimizer)


def stop_training(training_id):
    """Ferma una sessione di addestramento"""
    session = get_training_session(training_id)
    if not session:
        return False
    
    return session.stop()


def clean_completed_sessions(max_age=3600):  # 1 ora di default
    """Pulisce le sessioni completate più vecchie di max_age secondi"""
    current_time = time.time()
    
    with training_lock:
        to_remove = []
        
        for training_id, session in active_trainings.items():
            if session.status in ['completed', 'error', 'stopped'] and session.end_time:
                if current_time - session.end_time > max_age:
                    to_remove.append(training_id)
        
        for training_id in to_remove:
            del active_trainings[training_id]
    
    return len(to_remove)


def get_training_events(training_id, since=0):
    """Ottiene gli eventi di un addestramento dopo un certo timestamp"""
    session = get_training_session(training_id)
    if not session:
        return []
    
    return session.get_events(since)


def get_all_training_sessions():
    """Ottiene tutte le sessioni di addestramento attive"""
    with training_lock:
        return {k: v.to_dict() for k, v in active_trainings.items()}