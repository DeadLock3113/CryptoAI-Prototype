/**
 * Training Visualizer - Script per la visualizzazione interattiva del progresso di addestramento
 * CryptoTradeAnalyzer
 */

document.addEventListener('DOMContentLoaded', function() {
    // Elementi DOM
    const trainingProgressContainer = document.getElementById('training-progress-container');
    const lossChartCanvas = document.getElementById('loss-chart');
    const trainingStatusElement = document.getElementById('training-status');
    const trainingProgressBar = document.getElementById('training-progress-bar');
    const stopTrainingBtn = document.getElementById('stop-training-btn');
    
    // Se non ci sono gli elementi necessari, esci
    if (!trainingProgressContainer || !lossChartCanvas) {
        return;
    }
    
    // Inizializziamo Chart.js per il grafico delle perdite
    let lossChart = null;
    
    function initLossChart() {
        const ctx = lossChartCanvas.getContext('2d');
        lossChart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Training Loss',
                        data: [],
                        borderColor: 'rgba(75, 192, 192, 1)',
                        backgroundColor: 'rgba(75, 192, 192, 0.2)',
                        borderWidth: 2,
                        tension: 0.1
                    },
                    {
                        label: 'Validation Loss',
                        data: [],
                        borderColor: 'rgba(255, 99, 132, 1)',
                        backgroundColor: 'rgba(255, 99, 132, 0.2)',
                        borderWidth: 2,
                        tension: 0.1
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: {
                    x: {
                        title: {
                            display: true,
                            text: 'Epoca'
                        }
                    },
                    y: {
                        title: {
                            display: true,
                            text: 'Loss'
                        },
                        beginAtZero: true
                    }
                },
                plugins: {
                    tooltip: {
                        mode: 'index',
                        intersect: false,
                    },
                    legend: {
                        position: 'top',
                    },
                    title: {
                        display: true,
                        text: 'Andamento Loss Durante Training'
                    }
                },
                animation: {
                    duration: 500
                }
            }
        });
    }
    
    // Funzione per aggiornare lo stato del training con animazione
    function updateTrainingStatus(status, progress, message) {
        if (trainingStatusElement) {
            trainingStatusElement.textContent = message || status;
        }
        
        if (trainingProgressBar) {
            // Applica transizione fluida per l'animazione
            trainingProgressBar.style.transition = 'width 0.8s ease-in-out';
            trainingProgressBar.style.width = `${progress}%`;
            trainingProgressBar.setAttribute('aria-valuenow', progress);
            trainingProgressBar.textContent = `${Math.round(progress)}%`;
            
            // Aggiorna il colore della barra in base al progresso
            if (progress < 25) {
                trainingProgressBar.className = 'progress-bar progress-bar-striped progress-bar-animated bg-info';
            } else if (progress < 50) {
                trainingProgressBar.className = 'progress-bar progress-bar-striped progress-bar-animated bg-primary';
            } else if (progress < 75) {
                trainingProgressBar.className = 'progress-bar progress-bar-striped progress-bar-animated bg-warning';
            } else {
                trainingProgressBar.className = 'progress-bar progress-bar-striped progress-bar-animated bg-success';
            }
        }
    }
    
    // Funzione per aggiornare il grafico con nuovi dati
    function updateLossChart(epoch, trainLoss, valLoss) {
        if (!lossChart) {
            initLossChart();
        }
        
        lossChart.data.labels.push(epoch);
        lossChart.data.datasets[0].data.push(trainLoss);
        lossChart.data.datasets[1].data.push(valLoss);
        lossChart.update();
    }
    
    // Funzione per resettare il grafico
    function resetChart() {
        if (lossChart) {
            lossChart.data.labels = [];
            lossChart.data.datasets[0].data = [];
            lossChart.data.datasets[1].data = [];
            lossChart.update();
        }
    }
    
    // Connessione SSE per aggiornamenti in tempo reale
    let eventSource = null;
    
    function startTrainingMonitor(trainingId) {
        // Reset UI
        resetChart();
        updateTrainingStatus('Inizializzazione...', 0, 'Preparazione addestramento...');
        
        // Chiudi eventuale connessione esistente
        if (eventSource) {
            eventSource.close();
        }
        
        // Crea nuova connessione SSE
        eventSource = new EventSource(`/training-progress/${trainingId}`);
        
        // Gestisci eventi
        eventSource.onopen = function() {
            console.log('Connessione al monitor di addestramento stabilita');
            updateTrainingStatus('Connesso', 5, 'Connessione stabilita, in attesa di dati...');
        };
        
        eventSource.addEventListener('epoch_update', function(e) {
            const data = JSON.parse(e.data);
            updateLossChart(data.epoch, data.train_loss, data.val_loss);
            
            // Calcola il progresso dell'addestramento
            const progress = (data.epoch / data.total_epochs) * 100;
            
            // Aggiorna lo stato con animazione
            updateTrainingStatus('In addestramento', 
                                progress, 
                                `Epoca ${data.epoch}/${data.total_epochs}: Train Loss=${data.train_loss.toFixed(6)}, Val Loss=${data.val_loss.toFixed(6)}`);
            
            // Aggiornamento delle metriche RT
            updateRealTimeMetrics(data);
        });
        
        eventSource.addEventListener('training_complete', function(e) {
            const data = JSON.parse(e.data);
            updateTrainingStatus('Completato', 100, `Addestramento completato in ${data.total_time}s. Errore finale: ${data.final_loss.toFixed(6)}`);
            
            if (eventSource) {
                eventSource.close();
                eventSource = null;
            }
        });
        
        eventSource.addEventListener('training_error', function(e) {
            const data = JSON.parse(e.data);
            updateTrainingStatus('Errore', 0, `Errore: ${data.error}`);
            
            if (eventSource) {
                eventSource.close();
                eventSource = null;
            }
        });
        
        eventSource.onerror = function() {
            console.error('Errore nella connessione al monitor di addestramento');
            updateTrainingStatus('Errore di connessione', 0, 'Impossibile connettersi al server per il monitoraggio.');
            
            if (eventSource) {
                eventSource.close();
                eventSource = null;
            }
        };
    }
    
    // Gestisci pulsante stop
    if (stopTrainingBtn) {
        stopTrainingBtn.addEventListener('click', function() {
            if (confirm('Sei sicuro di voler interrompere l\'addestramento?')) {
                fetch('/stop-training', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({training_id: getTrainingId()})
                })
                .then(response => response.json())
                .then(data => {
                    if (data.success) {
                        updateTrainingStatus('Interrotto', 0, 'Addestramento interrotto manualmente.');
                    } else {
                        alert('Impossibile interrompere l\'addestramento: ' + data.error);
                    }
                })
                .catch(error => {
                    console.error('Errore durante l\'invio della richiesta di stop:', error);
                });
            }
        });
    }
    
    // Funzione per ottenere l'ID di training dalla pagina
    function getTrainingId() {
        return trainingProgressContainer.dataset.trainingId;
    }
    
    // Avvia il monitoraggio se c'è un ID di training nella pagina
    const trainingId = getTrainingId();
    if (trainingId) {
        startTrainingMonitor(trainingId);
    }
    
    // Funzione per animare il cambiamento dei valori
    function animateValueChange(element, newValue, duration = 800) {
        if (!element) return;
        
        const startValue = parseFloat(element.textContent) || 0;
        const diff = newValue - startValue;
        const startTime = performance.now();
        
        function updateValue(currentTime) {
            const elapsedTime = currentTime - startTime;
            
            if (elapsedTime < duration) {
                const progress = elapsedTime / duration;
                // Funzione di easing per un'animazione più piacevole
                const easeOutProgress = 1 - Math.pow(1 - progress, 2);
                const currentValue = startValue + (diff * easeOutProgress);
                
                // Arrotonda a 6 decimali per metriche precise
                if (Math.abs(currentValue) < 1) {
                    element.textContent = currentValue.toFixed(6);
                } else {
                    element.textContent = currentValue.toFixed(2);
                }
                
                requestAnimationFrame(updateValue);
            } else {
                // Valore finale
                if (Math.abs(newValue) < 1) {
                    element.textContent = newValue.toFixed(6);
                } else {
                    element.textContent = newValue.toFixed(2);
                }
            }
        }
        
        requestAnimationFrame(updateValue);
    }
    
    // Funzione per calcolare le metriche in tempo reale
    function updateRealTimeMetrics(data) {
        // Aggiornamento con animazione dell'epoca corrente
        const currentEpochElement = document.getElementById('current-epoch');
        if (currentEpochElement) {
            animateValueChange(currentEpochElement, data.epoch, 400);
        }
        
        // Aggiornamento con animazione della loss corrente
        const currentLossElement = document.getElementById('current-loss');
        if (currentLossElement) {
            animateValueChange(currentLossElement, data.train_loss, 600);
        }
        
        // Aggiornamento del tempo trascorso
        const timeElapsedElement = document.getElementById('time-elapsed');
        if (timeElapsedElement && data.elapsed_time) {
            timeElapsedElement.textContent = formatTime(data.elapsed_time);
        }
        
        // Calcolo e aggiornamento del tempo rimanente stimato
        const timeRemainingElement = document.getElementById('time-remaining');
        if (timeRemainingElement && data.epoch > 0 && data.total_epochs > 0) {
            // Calcola tempo medio per epoca
            const avgTimePerEpoch = data.elapsed_time / data.epoch;
            // Calcola quante epoche rimangono
            const remainingEpochs = data.total_epochs - data.epoch;
            // Stima il tempo rimanente
            const estimatedRemainingTime = avgTimePerEpoch * remainingEpochs;
            
            timeRemainingElement.textContent = formatTime(estimatedRemainingTime);
        }
        
        // Aggiorna metriche di prestazione se disponibili
        if (data.metrics) {
            // MSE (Mean Squared Error)
            const mseMeasure = data.metrics.mse || data.train_loss;
            updateMetricWithAnimation('metric-mse', mseMeasure);
            
            // RMSE (Root Mean Squared Error)
            const rmseMeasure = data.metrics.rmse || Math.sqrt(data.train_loss);
            updateMetricWithAnimation('metric-rmse', rmseMeasure);
            
            // MAE (Mean Absolute Error)
            if (data.metrics.mae) {
                updateMetricWithAnimation('metric-mae', data.metrics.mae);
            }
            
            // R² (Coefficient of determination)
            if (data.metrics.r2) {
                updateMetricWithAnimation('metric-r2', data.metrics.r2);
            }
        }
    }
    
    // Aggiorna una metrica con animazione
    function updateMetricWithAnimation(elementId, value) {
        const element = document.getElementById(elementId);
        const updateTimeElement = document.getElementById(`update-${elementId.replace('metric-', '')}`);
        
        if (element) {
            animateValueChange(element, value, 800);
            
            if (updateTimeElement) {
                updateTimeElement.textContent = new Date().toLocaleTimeString();
            }
        }
    }
    
    // Formatta i tempi per la visualizzazione
    function formatTime(seconds) {
        if (seconds === undefined) return "-";
        
        if (seconds < 60) {
            return `${Math.round(seconds)}s`;
        } else if (seconds < 3600) {
            const minutes = Math.floor(seconds / 60);
            const remainingSeconds = Math.round(seconds % 60);
            return `${minutes}m ${remainingSeconds}s`;
        } else {
            const hours = Math.floor(seconds / 3600);
            const minutes = Math.floor((seconds % 3600) / 60);
            return `${hours}h ${minutes}m`;
        }
    }
    
    // Esponi funzioni pubbliche
    window.TrainingVisualizer = {
        startMonitoring: startTrainingMonitor,
        updateStatus: updateTrainingStatus,
        resetChart: resetChart,
        updateRealTimeMetrics: updateRealTimeMetrics
    };
});