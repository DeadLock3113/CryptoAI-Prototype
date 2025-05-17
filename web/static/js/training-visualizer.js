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
    
    // Funzione per aggiornare lo stato del training
    function updateTrainingStatus(status, progress, message) {
        if (trainingStatusElement) {
            trainingStatusElement.textContent = message || status;
        }
        
        if (trainingProgressBar) {
            trainingProgressBar.style.width = `${progress}%`;
            trainingProgressBar.setAttribute('aria-valuenow', progress);
            trainingProgressBar.textContent = `${Math.round(progress)}%`;
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
            updateTrainingStatus('In addestramento', 
                                (data.epoch / data.total_epochs) * 100, 
                                `Epoca ${data.epoch}/${data.total_epochs}: Train Loss=${data.train_loss.toFixed(6)}, Val Loss=${data.val_loss.toFixed(6)}`);
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
    
    // Esponi funzioni pubbliche
    window.TrainingVisualizer = {
        startMonitoring: startTrainingMonitor,
        updateStatus: updateTrainingStatus,
        resetChart: resetChart
    };
});