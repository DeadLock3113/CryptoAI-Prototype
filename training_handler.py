"""
CryptoTradeAnalyzer - Gestore dell'addestramento dei modelli

Questo modulo implementa la gestione dell'addestramento interattivo 
dei modelli, inclusi gli aggiornamenti in tempo reale.
"""

import uuid
import time
import threading
import logging
import random
import math
import json
from datetime import datetime
from threading import Thread
from queue import Queue
from collections import deque

# Configurazione logging
logger = logging.getLogger(__name__)

# Dizionario per memorizzare le sessioni di addestramento attive
active_sessions = {}

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
        
        # Statistiche di addestramento
        self.status = 'initialized'  # initialized, running, completed, error, stopped
        self.start_time = None
        self.end_time = None
        self.current_epoch = 0
        self.train_loss = None
        self.val_loss = None
        self.best_val_loss = float('inf')
        self.early_stopping_counter = 0
        self.early_stopping_patience = 5
        self.metrics = {}
        
        # Per il thread e la comunicazione
        self.thread = None
        self.stop_event = threading.Event()
        self.events = deque()
        self.latest_timestamp = time.time()
        
        # Componenti per l'addestramento reale
        self.model = None
        self.train_loader = None
        self.val_loader = None
        self.criterion = None
        self.optimizer = None
        
        logger.debug(f"Nuova sessione di addestramento creata: {training_id}")
    
    def start(self, model=None, train_loader=None, val_loader=None, criterion=None, optimizer=None):
        """Avvia l'addestramento in un thread separato"""
        if self.status != 'initialized':
            logger.warning(f"Impossibile avviare la sessione {self.training_id} con stato {self.status}")
            return False
        
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.criterion = criterion
        self.optimizer = optimizer
        
        self.status = 'running'
        self.start_time = time.time()
        self.add_event('training_started', {'message': 'Addestramento iniziato'})
        
        # Avvia il thread di addestramento
        self.thread = Thread(target=self._training_loop)
        self.thread.daemon = True
        self.thread.start()
        
        logger.debug(f"Addestramento avviato per sessione: {self.training_id}")
        return True
    
    def _training_loop(self):
        """Loop di addestramento - eseguito in un thread separato"""
        try:
            # Scegli tra addestramento reale o simulato
            if self.demo_mode or not all([self.model, self.train_loader, self.val_loader, self.criterion, self.optimizer]):
                logger.debug(f"Modalità demo attivata per sessione {self.training_id}")
                self._simulated_training()
            else:
                logger.debug(f"Addestramento reale avviato per sessione {self.training_id}")
                self._real_training()
                
            # Completamento con successo
            if not self.stop_event.is_set():
                self.status = 'completed'
                self.end_time = time.time()
                self.add_event('training_complete', {
                    'message': 'Addestramento completato con successo',
                    'total_time': self.end_time - self.start_time,
                    'final_loss': self.train_loss,
                    'metrics': self.metrics
                })
                logger.debug(f"Addestramento completato per sessione {self.training_id}")
        
        except Exception as e:
            # Gestione degli errori
            logger.error(f"Errore nell'addestramento per sessione {self.training_id}: {str(e)}")
            self.status = 'error'
            self.end_time = time.time()
            self.add_event('training_error', {
                'error': str(e),
                'message': 'Si è verificato un errore durante l\'addestramento'
            })
    
    def _simulated_training(self):
        """Simula un addestramento per scopi dimostrativi"""
        # Parametri di simulazione
        max_epochs = min(self.epochs, 10) if self.demo_mode else self.epochs
        batches_per_epoch = int(1000 / self.batch_size)  # Simula 1000 campioni
        
        # Valori iniziali per loss
        base_train_loss = 0.5
        base_val_loss = 0.6
        decay_rate = 0.7
        
        for epoch in range(1, max_epochs + 1):
            if self.stop_event.is_set():
                logger.debug(f"Addestramento interrotto durante l'epoca {epoch}")
                break
            
            epoch_start_time = time.time()
            self.current_epoch = epoch
            
            # Simulazione batch per batch
            batch_train_loss = base_train_loss * (1.0 + 0.3 * random.random())
            for batch in range(1, batches_per_epoch + 1):
                if self.stop_event.is_set():
                    break
                
                # Simula il progresso dei batch e una varianza nella loss
                progress = batch / batches_per_epoch
                noise = 0.1 * random.random() - 0.05  # Rumore ±5%
                current_loss = batch_train_loss * (1.0 - 0.3 * progress) + noise
                
                # Invia evento per il batch
                if batch % max(1, int(batches_per_epoch / 10)) == 0:  # Ogni 10% dei batch
                    self.add_event('batch_complete', {
                        'batch': batch,
                        'total_batches': batches_per_epoch,
                        'loss': current_loss,
                        'epoch': epoch
                    })
                
                # Simula il tempo di calcolo
                time.sleep(0.05 if self.demo_mode else 0.2)
            
            # Calcola loss di epoca con decadimento e rumore
            epoch_factor = 1.0 / (1.0 + epoch * decay_rate / max_epochs)
            random_factor = 1.0 + (random.random() - 0.5) * 0.1  # ±5%
            
            # Loss di training
            self.train_loss = base_train_loss * epoch_factor * random_factor
            
            # Loss di validazione (leggermente più alta con una possibilità di aumentare)
            random_val_factor = 1.0 + (random.random() - 0.4) * 0.2  # Tendenza al miglioramento
            self.val_loss = base_val_loss * epoch_factor * random_val_factor
            
            # Occasionalmente, simula un peggioramento per early stopping
            if epoch > max_epochs // 2 and random.random() < 0.3:
                self.val_loss *= 1.1
            
            # Traccia il miglior modello e early stopping
            if self.val_loss < self.best_val_loss:
                self.best_val_loss = self.val_loss
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
            
            # Calcola tempo trascorso per l'epoca
            epoch_time = time.time() - epoch_start_time
            
            # Invia evento di completamento epoca
            self.add_event('epoch_complete', {
                'epoch': epoch,
                'total_epochs': max_epochs,
                'loss': self.train_loss,
                'val_loss': self.val_loss,
                'elapsed_time': time.time() - self.start_time
            })
            
            # Early stopping
            if self.early_stopping_counter >= self.early_stopping_patience:
                logger.debug(f"Early stopping attivato all'epoca {epoch}")
                self.add_event('early_stopping', {
                    'message': f'Early stopping attivato dopo {epoch} epoche',
                    'best_val_loss': self.best_val_loss
                })
                break
            
            # Pausa tra le epoche
            time.sleep(0.2)
    
    def _real_training(self):
        """Esegue un addestramento reale con PyTorch"""
        import torch
        
        # Verifica dei componenti necessari
        if not all([self.model, self.train_loader, self.val_loader, self.criterion, self.optimizer]):
            raise ValueError("Componenti mancanti per l'addestramento reale")
        
        # Adattamento per demo mode
        max_epochs = min(self.epochs, 5) if self.demo_mode else self.epochs
        
        # Inizializzazione
        self.model.to(self.device)
        best_model_state = self.model.state_dict().copy()
        
        # Training loop
        for epoch in range(1, max_epochs + 1):
            if self.stop_event.is_set():
                logger.debug(f"Addestramento interrotto durante l'epoca {epoch}")
                break
            
            epoch_start_time = time.time()
            self.current_epoch = epoch
            
            # Training
            self.model.train()
            train_loss = 0
            batch_count = 0
            
            for batch_idx, (inputs, targets) in enumerate(self.train_loader):
                if self.stop_event.is_set():
                    break
                
                # Limita i batch in demo mode
                if self.demo_mode and batch_idx >= min(len(self.train_loader), 10):
                    break
                
                # Forward e backward pass
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                
                # Update weights
                for param in self.model.parameters():
                    param.grad.data.clamp_(-1, 1)  # Gradient clipping
                self.optimizer.step()
                
                train_loss += loss.item()
                batch_count += 1
                
                # Invia evento per il batch
                if batch_idx % max(1, int(len(self.train_loader) / 10)) == 0:  # Ogni 10% dei batch
                    self.add_event('batch_complete', {
                        'batch': batch_idx + 1,
                        'total_batches': len(self.train_loader),
                        'loss': loss.item(),
                        'epoch': epoch
                    })
            
            # Calcola loss media
            train_loss = train_loss / batch_count if batch_count > 0 else float('inf')
            self.train_loss = train_loss
            
            # Validation
            self.model.eval()
            val_loss = 0
            val_batch_count = 0
            
            with torch.no_grad():
                for batch_idx, (inputs, targets) in enumerate(self.val_loader):
                    # Limita i batch in demo mode
                    if self.demo_mode and batch_idx >= min(len(self.val_loader), 5):
                        break
                    
                    inputs, targets = inputs.to(self.device), targets.to(self.device)
                    outputs = self.model(inputs)
                    loss = self.criterion(outputs, targets)
                    val_loss += loss.item()
                    val_batch_count += 1
            
            # Calcola loss media
            val_loss = val_loss / val_batch_count if val_batch_count > 0 else float('inf')
            self.val_loss = val_loss
            
            # Traccia il miglior modello
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                best_model_state = self.model.state_dict().copy()
                self.early_stopping_counter = 0
            else:
                self.early_stopping_counter += 1
            
            # Invia evento di completamento epoca
            self.add_event('epoch_complete', {
                'epoch': epoch,
                'total_epochs': max_epochs,
                'loss': train_loss,
                'val_loss': val_loss,
                'elapsed_time': time.time() - self.start_time
            })
            
            # Early stopping
            if self.early_stopping_counter >= self.early_stopping_patience:
                logger.debug(f"Early stopping attivato all'epoca {epoch}")
                self.add_event('early_stopping', {
                    'message': f'Early stopping attivato dopo {epoch} epoche',
                    'best_val_loss': self.best_val_loss
                })
                break
        
        # Carica il miglior modello
        self.model.load_state_dict(best_model_state)
        
        # Salva risultati finali nelle metriche
        self.metrics = {
            'final_train_loss': self.train_loss,
            'best_val_loss': self.best_val_loss,
            'epochs_completed': self.current_epoch,
            'total_time': time.time() - self.start_time
        }
    
    def stop(self):
        """Ferma l'addestramento"""
        if self.status == 'running':
            self.stop_event.set()
            self.status = 'stopped'
            self.end_time = time.time()
            self.add_event('training_stopped', {
                'message': 'Addestramento interrotto manualmente',
                'elapsed_time': self.end_time - self.start_time
            })
            logger.debug(f"Addestramento interrotto per sessione {self.training_id}")
            return True
        return False
    
    def add_event(self, event_type, data):
        """Aggiunge un evento alla coda"""
        timestamp = time.time()
        self.latest_timestamp = timestamp
        
        event = {
            'timestamp': timestamp,
            'type': event_type,
            'data': data
        }
        
        self.events.append(event)
        
        # Limita la dimensione della coda
        while len(self.events) > 1000:
            self.events.popleft()
    
    def get_events(self, since=0):
        """Ottiene gli eventi dalla coda dopo un certo timestamp"""
        return [event for event in self.events if event['timestamp'] > since]
    
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
            'train_loss': self.train_loss,
            'val_loss': self.val_loss,
            'start_time': self.start_time,
            'end_time': self.end_time
        }


def create_training_session(model_type, model_name, dataset_id, dataset_name, 
                           epochs, batch_size, lookback, learning_rate, device,
                           demo_mode=True):
    """Crea una nuova sessione di addestramento"""
    training_id = str(uuid.uuid4())
    
    # Crea una nuova sessione
    session = TrainingSession(
        training_id=training_id,
        model_type=model_type,
        model_name=model_name,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        epochs=epochs,
        batch_size=batch_size,
        lookback=lookback,
        learning_rate=learning_rate,
        device=device,
        demo_mode=demo_mode
    )
    
    # Salva la sessione nel dizionario globale
    active_sessions[training_id] = session
    
    logger.debug(f"Creata nuova sessione di addestramento: {training_id}")
    return training_id, session


def get_training_session(training_id):
    """Ottiene una sessione di addestramento esistente"""
    return active_sessions.get(training_id)


def start_training(training_id, model=None, train_loader=None, val_loader=None, 
                  criterion=None, optimizer=None):
    """Avvia una sessione di addestramento esistente"""
    session = get_training_session(training_id)
    if not session:
        logger.warning(f"Sessione di addestramento non trovata: {training_id}")
        return False
    
    return session.start(model, train_loader, val_loader, criterion, optimizer)


def stop_training(training_id):
    """Ferma una sessione di addestramento"""
    session = get_training_session(training_id)
    if not session:
        logger.warning(f"Sessione di addestramento non trovata: {training_id}")
        return False
    
    return session.stop()


def clean_completed_sessions(max_age=3600):  # 1 ora di default
    """Pulisce le sessioni completate più vecchie di max_age secondi"""
    now = time.time()
    sessions_to_remove = []
    
    for training_id, session in active_sessions.items():
        if session.status in ['completed', 'error', 'stopped']:
            if session.end_time and now - session.end_time > max_age:
                sessions_to_remove.append(training_id)
    
    for training_id in sessions_to_remove:
        del active_sessions[training_id]
    
    logger.debug(f"Pulite {len(sessions_to_remove)} sessioni completate")
    return len(sessions_to_remove)


def get_training_events(training_id, since=0):
    """Ottiene gli eventi di un addestramento dopo un certo timestamp"""
    session = get_training_session(training_id)
    if not session:
        return []
    
    return session.get_events(since)


def get_all_training_sessions():
    """Ottiene tutte le sessioni di addestramento attive"""
    return {training_id: session.to_dict() for training_id, session in active_sessions.items()}