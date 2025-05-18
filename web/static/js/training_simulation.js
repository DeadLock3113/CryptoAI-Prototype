/**
 * Simulazione locale dell'addestramento del modello
 * Questo script fornisce un'alternativa alla connessione EventSource
 * quando ci sono problemi di connessione
 */

class TrainingSimulator {
    constructor(config) {
        // Configurazione
        this.modelType = config.modelType || 'lstm';
        this.totalEpochs = config.epochs || 10;
        this.currentEpoch = 0;
        this.startTime = Date.now();
        
        // Handlers degli eventi
        this.onEpochComplete = config.onEpochComplete || function() {};
        this.onTrainingComplete = config.onTrainingComplete || function() {};
        this.onLogUpdate = config.onLogUpdate || function() {};
        
        // Parametri di simulazione
        this.baseLoss = 0.5;
        this.baseValLoss = 0.6;
        this.decayRate = 0.7;
        this.metrics = {
            mse: 0.015,
            rmse: 0.12,
            mae: 0.09,
            r2: 0.75
        };
        
        // Stato
        this.isRunning = false;
        this.simulationInterval = null;
    }
    
    start() {
        if (this.isRunning) return;
        
        this.isRunning = true;
        this.onLogUpdate("Simulazione locale avviata - i dati sono simulati per scopi dimostrativi");
        this.onLogUpdate(`Addestramento modello ${this.modelType} per ${this.totalEpochs} epoche`);
        
        // Simula la prima epoca immediatamente
        this.simulateEpoch();
        
        // Imposta un intervallo per le epoche successive
        this.simulationInterval = setInterval(() => {
            this.simulateEpoch();
            
            // Se abbiamo completato tutte le epoche, termina
            if (this.currentEpoch >= this.totalEpochs) {
                this.stop();
                this.onTrainingComplete({
                    message: "Addestramento completato con successo",
                    metrics: this.metrics,
                    totalTime: (Date.now() - this.startTime) / 1000
                });
            }
        }, 3000); // 3 secondi per epoca
    }
    
    stop() {
        if (!this.isRunning) return;
        
        clearInterval(this.simulationInterval);
        this.isRunning = false;
        this.onLogUpdate("Simulazione terminata");
    }
    
    simulateEpoch() {
        this.currentEpoch++;
        
        // Calcola loss simulata con decadimento esponenziale
        const epochFactor = 1.0 / (1.0 + this.currentEpoch * this.decayRate / this.totalEpochs);
        const randomFactor = 1.0 + (Math.random() - 0.5) * 0.1; // ±5%
        
        // Loss di training
        const trainLoss = this.baseLoss * epochFactor * randomFactor;
        
        // Loss di validazione (leggermente più alta con una possibilità di aumentare)
        const randomValFactor = 1.0 + (Math.random() - 0.4) * 0.2; // Tendenza al miglioramento
        const valLoss = this.baseValLoss * epochFactor * randomValFactor;
        
        // Aggiorna metriche
        this.updateMetrics();
        
        // Log
        this.onLogUpdate(`[Epoca ${this.currentEpoch}/${this.totalEpochs}] Loss: ${trainLoss.toFixed(4)}, Val Loss: ${valLoss.toFixed(4)}`);
        
        // Notifica completamento epoca
        this.onEpochComplete({
            epoch: this.currentEpoch,
            total_epochs: this.totalEpochs,
            loss: trainLoss,
            val_loss: valLoss,
            elapsed_time: (Date.now() - this.startTime) / 1000,
            metrics: this.metrics
        });
    }
    
    updateMetrics() {
        // Applica miglioramenti graduali alle metriche
        const progress = this.currentEpoch / this.totalEpochs;
        const improvementFactor = Math.min(0.05, 0.02 + (progress * 0.03));
        
        // MSE, RMSE, MAE migliorano (diminuiscono)
        this.metrics.mse = Math.max(0.001, this.metrics.mse * (1 - improvementFactor * (0.8 + Math.random() * 0.4)));
        this.metrics.rmse = Math.max(0.005, this.metrics.rmse * (1 - improvementFactor * (0.8 + Math.random() * 0.4)));
        this.metrics.mae = Math.max(0.005, this.metrics.mae * (1 - improvementFactor * (0.8 + Math.random() * 0.4)));
        
        // R2 migliora (aumenta)
        this.metrics.r2 = Math.min(0.99, this.metrics.r2 + (improvementFactor * (0.8 + Math.random() * 0.4) * (1 - this.metrics.r2)));
        
        // Arrotonda per presentazione
        this.metrics.mse = parseFloat(this.metrics.mse.toFixed(4));
        this.metrics.rmse = parseFloat(this.metrics.rmse.toFixed(4));
        this.metrics.mae = parseFloat(this.metrics.mae.toFixed(4));
        this.metrics.r2 = parseFloat(this.metrics.r2.toFixed(4));
        
        // Aggiungi timestamp
        this.metrics.calculated_at = new Date().toLocaleTimeString();
    }
}

// Funzione per avviare la simulazione
function startLocalSimulation() {
    // Ottieni i parametri dal DOM
    const modelType = document.querySelector('[data-param="model_type"]')?.textContent || 'lstm';
    const epochs = parseInt(document.querySelector('[data-param="epochs"]')?.textContent || '10');
    
    // Riferimenti per funzioni del grafico e log
    const lossChart = window.trainingChart;
    const logContainer = document.getElementById('training-log');
    const progressBar = document.getElementById('training-progress-bar');
    
    // Crea il simulatore
    const simulator = new TrainingSimulator({
        modelType: modelType,
        epochs: epochs,
        onEpochComplete: function(data) {
            // Aggiorna l'interfaccia con i dati dell'epoca
            if (window.handleEpochComplete) {
                window.handleEpochComplete(data);
            } else {
                // Aggiorna elementi base
                document.getElementById('current-epoch').textContent = data.epoch;
                document.getElementById('current-loss').textContent = data.loss.toFixed(4);
                
                // Aggiorna barra di progresso
                const progress = (data.epoch / data.total_epochs) * 100;
                progressBar.style.width = progress + '%';
                progressBar.textContent = progress.toFixed(0) + '%';
                
                // Aggiorna metriche nella tabella
                if (data.metrics) {
                    document.getElementById('metric-mse').textContent = data.metrics.mse?.toFixed(4) || '-';
                    document.getElementById('metric-rmse').textContent = data.metrics.rmse?.toFixed(4) || '-';
                    document.getElementById('metric-mae').textContent = data.metrics.mae?.toFixed(4) || '-';
                    document.getElementById('metric-r2').textContent = data.metrics.r2?.toFixed(4) || '-';
                }
                
                // Aggiorna grafico se disponibile
                if (lossChart) {
                    lossChart.data.labels.push('Epoca ' + data.epoch);
                    lossChart.data.datasets[0].data.push(data.loss);
                    lossChart.data.datasets[1].data.push(data.val_loss);
                    lossChart.update();
                }
            }
        },
        onTrainingComplete: function(data) {
            // Aggiorna l'interfaccia con i dati finali
            document.getElementById('training-status').textContent = 'Addestramento completato';
            progressBar.style.width = '100%';
            progressBar.textContent = '100%';
            progressBar.classList.remove('progress-bar-animated');
            
            // Log
            const logEntry = document.createElement('div');
            logEntry.className = 'log-entry text-success';
            logEntry.textContent = `[${new Date().toLocaleTimeString()}] ${data.message}`;
            logContainer.appendChild(logEntry);
            logContainer.scrollTop = logContainer.scrollHeight;
        },
        onLogUpdate: function(message) {
            // Aggiunta messaggio al log
            const logEntry = document.createElement('div');
            logEntry.className = 'log-entry';
            logEntry.textContent = `[${new Date().toLocaleTimeString()}] ${message}`;
            logContainer.appendChild(logEntry);
            logContainer.scrollTop = logContainer.scrollHeight;
        }
    });
    
    // Avvia simulazione
    simulator.start();
    
    // Salva il simulatore come oggetto globale
    window.trainingSimulator = simulator;
}

// Esporta funzioni
window.startLocalSimulation = startLocalSimulation;
window.TrainingSimulator = TrainingSimulator;