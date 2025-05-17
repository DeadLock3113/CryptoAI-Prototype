"""
Web Routes Module for CryptoTradeAnalyzer

This module defines all the routes for the CryptoTradeAnalyzer web interface,
including data upload, analysis, backtesting, and machine learning.

Author: CryptoTradeAnalyzer Team
"""

import os
import logging
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import io
import base64
from flask import render_template, request, redirect, url_for, flash, session, jsonify, send_file
from datetime import datetime
import uuid

# Import components
from data.loader import DataLoader
from indicators.basic import calculate_indicators
from indicators.advanced import calculate_advanced_indicators
from strategies.crossover_strategy import CrossoverStrategy
from backtesting.engine import BacktestEngine
from models.lstm import LSTMModel
from models.rnn import RNNModel
from models.ensemble import EnsembleModel
from utils.plot_results import plot_results

logger = logging.getLogger(__name__)

def register_routes(app):
    """
    Register all routes with the Flask application
    
    Parameters:
    -----------
    app : Flask
        Flask application
    """
    
    @app.route('/')
    def index():
        """Home page"""
        return render_template('index.html')
    
    @app.route('/upload', methods=['GET', 'POST'])
    def upload():
        """Data upload page"""
        if request.method == 'POST':
            # Check if a file was uploaded
            if 'file' not in request.files:
                flash('No file part', 'danger')
                return redirect(request.url)
            
            file = request.files['file']
            
            # Check if file is empty
            if file.filename == '':
                flash('No file selected', 'danger')
                return redirect(request.url)
            
            # Check file extension
            if not file.filename.endswith('.csv'):
                flash('Only CSV files are supported', 'danger')
                return redirect(request.url)
            
            # Save the file with a unique name
            filename = f"{uuid.uuid4()}_{file.filename}"
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            try:
                # Load the data
                loader = DataLoader()
                data = loader.load_from_csv(filepath)
                
                # Store basic info in session
                session['data_path'] = filepath
                session['data_filename'] = file.filename
                session['data_rows'] = len(data)
                session['data_columns'] = list(data.columns)
                session['data_start'] = str(data.index[0])
                session['data_end'] = str(data.index[-1])
                
                flash(f'Successfully loaded {len(data)} rows from {file.filename}', 'success')
                return redirect(url_for('analysis'))
                
            except Exception as e:
                flash(f'Error loading data: {str(e)}', 'danger')
                return redirect(request.url)
        
        return render_template('upload.html')
    
    @app.route('/analysis')
    def analysis():
        """Data analysis page"""
        # Check if data is loaded
        if 'data_path' not in session:
            flash('Please upload data first', 'warning')
            return redirect(url_for('upload'))
        
        # Load data
        loader = DataLoader()
        data = loader.load_from_csv(session['data_path'])
        
        # Prepare data summary
        summary = {
            'filename': session['data_filename'],
            'rows': session['data_rows'],
            'columns': session['data_columns'],
            'start': session['data_start'],
            'end': session['data_end'],
        }
        
        # Generate price chart
        plt.figure(figsize=(10, 6))
        plt.plot(data.index, data['close'], label='Close Price')
        
        # Add volume as bar chart on secondary y-axis
        if 'volume' in data.columns:
            ax1 = plt.gca()
            ax2 = ax1.twinx()
            ax2.bar(data.index, data['volume'], alpha=0.3, color='gray', label='Volume')
            ax2.set_ylabel('Volume')
        
        plt.title(f'{session["data_filename"]} - Price Chart')
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.grid(True, alpha=0.3)
        plt.legend()
        
        # Save to buffer
        buffer = io.BytesIO()
        plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
        buffer.seek(0)
        
        # Encode as base64 for embedding in HTML
        chart_data = base64.b64encode(buffer.getvalue()).decode('utf-8')
        plt.close()
        
        # Calculate basic statistics
        stats = {
            'min': data['close'].min(),
            'max': data['close'].max(),
            'mean': data['close'].mean(),
            'std': data['close'].std(),
            'median': data['close'].median(),
            'last': data['close'].iloc[-1],
            'change': (data['close'].iloc[-1] / data['close'].iloc[0] - 1) * 100,
        }
        
        return render_template('analysis.html', 
                             summary=summary,
                             stats=stats, 
                             chart_data=chart_data)
    
    @app.route('/indicators', methods=['GET', 'POST'])
    def indicators():
        """Technical indicators page"""
        # Check if data is loaded
        if 'data_path' not in session:
            flash('Please upload data first', 'warning')
            return redirect(url_for('upload'))
        
        # Load data
        loader = DataLoader()
        data = loader.load_from_csv(session['data_path'])
        
        if request.method == 'POST':
            # Get selected indicators
            selected_indicators = request.form.getlist('indicators')
            
            if not selected_indicators:
                flash('Please select at least one indicator', 'warning')
                return redirect(request.url)
            
            # Calculate indicators
            try:
                data_with_indicators = calculate_indicators(data, selected_indicators)
                
                # Save data with indicators
                indicators_path = os.path.join(app.config['UPLOAD_FOLDER'], 'data_with_indicators.csv')
                data_with_indicators.to_csv(indicators_path)
                session['indicators_path'] = indicators_path
                
                # Generate plot for each indicator
                indicator_plots = []
                
                # Plot SMA
                if 'sma' in selected_indicators:
                    plt.figure(figsize=(10, 6))
                    plt.plot(data_with_indicators.index, data_with_indicators['close'], label='Close Price')
                    for period in [20, 50, 200]:
                        if f'sma_{period}' in data_with_indicators.columns:
                            plt.plot(data_with_indicators.index, data_with_indicators[f'sma_{period}'], 
                                    label=f'SMA {period}')
                    
                    plt.title('Simple Moving Averages')
                    plt.xlabel('Date')
                    plt.ylabel('Price')
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    
                    # Save to buffer
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                    buffer.seek(0)
                    
                    # Encode as base64 for embedding in HTML
                    indicator_plots.append({
                        'name': 'Simple Moving Averages (SMA)',
                        'description': 'Moving averages smooth price data to identify trends.',
                        'img': base64.b64encode(buffer.getvalue()).decode('utf-8')
                    })
                    plt.close()
                
                # Plot RSI
                if 'rsi' in selected_indicators and 'rsi' in data_with_indicators.columns:
                    plt.figure(figsize=(10, 6))
                    plt.plot(data_with_indicators.index, data_with_indicators['rsi'], label='RSI')
                    plt.axhline(70, color='red', linestyle='--', alpha=0.5, label='Overbought')
                    plt.axhline(30, color='green', linestyle='--', alpha=0.5, label='Oversold')
                    
                    plt.title('Relative Strength Index (RSI)')
                    plt.xlabel('Date')
                    plt.ylabel('RSI')
                    plt.ylim(0, 100)
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    
                    # Save to buffer
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                    buffer.seek(0)
                    
                    # Encode as base64 for embedding in HTML
                    indicator_plots.append({
                        'name': 'Relative Strength Index (RSI)',
                        'description': 'RSI measures the speed and change of price movements.',
                        'img': base64.b64encode(buffer.getvalue()).decode('utf-8')
                    })
                    plt.close()
                
                # Plot MACD
                if 'macd' in selected_indicators and 'macd' in data_with_indicators.columns:
                    plt.figure(figsize=(10, 6))
                    
                    # Plot MACD and signal line
                    plt.plot(data_with_indicators.index, data_with_indicators['macd'], label='MACD')
                    plt.plot(data_with_indicators.index, data_with_indicators['macd_signal'], label='Signal Line')
                    
                    # Plot histogram
                    plt.bar(data_with_indicators.index, data_with_indicators['macd_hist'], 
                           alpha=0.5, label='Histogram')
                    
                    plt.title('Moving Average Convergence Divergence (MACD)')
                    plt.xlabel('Date')
                    plt.ylabel('MACD')
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    
                    # Save to buffer
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                    buffer.seek(0)
                    
                    # Encode as base64 for embedding in HTML
                    indicator_plots.append({
                        'name': 'MACD',
                        'description': 'MACD is a trend-following momentum indicator.',
                        'img': base64.b64encode(buffer.getvalue()).decode('utf-8')
                    })
                    plt.close()
                
                # Plot Bollinger Bands
                if 'bbands' in selected_indicators and 'bb_upper' in data_with_indicators.columns:
                    plt.figure(figsize=(10, 6))
                    plt.plot(data_with_indicators.index, data_with_indicators['close'], label='Close Price')
                    plt.plot(data_with_indicators.index, data_with_indicators['bb_upper'], 
                            label='Upper Band', linestyle='--')
                    plt.plot(data_with_indicators.index, data_with_indicators['bb_middle'], 
                            label='Middle Band')
                    plt.plot(data_with_indicators.index, data_with_indicators['bb_lower'], 
                            label='Lower Band', linestyle='--')
                    
                    plt.title('Bollinger Bands')
                    plt.xlabel('Date')
                    plt.ylabel('Price')
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    
                    # Save to buffer
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                    buffer.seek(0)
                    
                    # Encode as base64 for embedding in HTML
                    indicator_plots.append({
                        'name': 'Bollinger Bands',
                        'description': 'Bollinger Bands measure price volatility.',
                        'img': base64.b64encode(buffer.getvalue()).decode('utf-8')
                    })
                    plt.close()
                
                # Plot Stochastic
                if 'stoch' in selected_indicators and 'stoch_k' in data_with_indicators.columns:
                    plt.figure(figsize=(10, 6))
                    plt.plot(data_with_indicators.index, data_with_indicators['stoch_k'], label='%K')
                    plt.plot(data_with_indicators.index, data_with_indicators['stoch_d'], label='%D')
                    plt.axhline(80, color='red', linestyle='--', alpha=0.5, label='Overbought')
                    plt.axhline(20, color='green', linestyle='--', alpha=0.5, label='Oversold')
                    
                    plt.title('Stochastic Oscillator')
                    plt.xlabel('Date')
                    plt.ylabel('Stochastic')
                    plt.ylim(0, 100)
                    plt.grid(True, alpha=0.3)
                    plt.legend()
                    
                    # Save to buffer
                    buffer = io.BytesIO()
                    plt.savefig(buffer, format='png', dpi=300, bbox_inches='tight')
                    buffer.seek(0)
                    
                    # Encode as base64 for embedding in HTML
                    indicator_plots.append({
                        'name': 'Stochastic Oscillator',
                        'description': 'Stochastic oscillator compares closing price to price range.',
                        'img': base64.b64encode(buffer.getvalue()).decode('utf-8')
                    })
                    plt.close()
                
                # Store selected indicators in session
                session['selected_indicators'] = selected_indicators
                
                return render_template('indicators_result.html', 
                                     indicator_plots=indicator_plots,
                                     selected_indicators=selected_indicators)
                
            except Exception as e:
                flash(f'Error calculating indicators: {str(e)}', 'danger')
                return redirect(request.url)
        
        # Indicator options
        indicator_options = [
            {'id': 'sma', 'name': 'Simple Moving Average (SMA)'},
            {'id': 'ema', 'name': 'Exponential Moving Average (EMA)'},
            {'id': 'rsi', 'name': 'Relative Strength Index (RSI)'},
            {'id': 'macd', 'name': 'Moving Average Convergence Divergence (MACD)'},
            {'id': 'bbands', 'name': 'Bollinger Bands'},
            {'id': 'stoch', 'name': 'Stochastic Oscillator'},
        ]
        
        return render_template('indicators.html', indicator_options=indicator_options)
    
    @app.route('/backtest', methods=['GET', 'POST'])
    def backtest():
        """Backtesting page"""
        # Check if data is loaded
        if 'data_path' not in session:
            flash('Please upload data first', 'warning')
            return redirect(url_for('upload'))
        
        # Load data
        loader = DataLoader()
        
        # Use data with indicators if available
        if 'indicators_path' in session and os.path.exists(session['indicators_path']):
            data = loader.load_from_csv(session['indicators_path'])
            has_indicators = True
        else:
            data = loader.load_from_csv(session['data_path'])
            # Calculate basic indicators
            data = calculate_indicators(data)
            has_indicators = False
        
        if request.method == 'POST':
            # Get strategy parameters
            strategy_type = request.form.get('strategy_type', 'crossover')
            initial_capital = float(request.form.get('initial_capital', 10000))
            commission = float(request.form.get('commission', 0.001))
            
            # Strategy-specific parameters
            if strategy_type == 'crossover':
                fast_ma = int(request.form.get('fast_ma', 20))
                slow_ma = int(request.form.get('slow_ma', 50))
                ma_type = request.form.get('ma_type', 'sma')
                
                # Initialize strategy
                strategy = CrossoverStrategy(fast_ma=fast_ma, slow_ma=slow_ma, ma_type=ma_type)
            else:
                flash('Selected strategy not implemented', 'danger')
                return redirect(request.url)
            
            # Run backtest
            try:
                # Initialize backtest engine
                engine = BacktestEngine(
                    data=data,
                    strategy=strategy,
                    initial_capital=initial_capital,
                    commission=commission
                )
                
                # Run backtest
                results = engine.run()
                
                # Generate plots
                plots_dir = os.path.join(app.config['PLOTS_FOLDER'], str(uuid.uuid4()))
                os.makedirs(plots_dir, exist_ok=True)
                
                figures = plot_results(data, results, save_dir=plots_dir)
                
                # Create performance summary
                performance = engine.get_performance_summary()
                
                # Store relative paths to plots
                plot_files = []
                for filename in os.listdir(plots_dir):
                    if filename.endswith('.png'):
                        plot_files.append(os.path.join('plots', os.path.basename(plots_dir), filename))
                
                # Store backtest results in session
                session['backtest_results'] = {
                    'strategy_type': strategy_type,
                    'initial_capital': initial_capital,
                    'commission': commission,
                    'final_equity': float(results['equity'].iloc[-1]),
                    'return_pct': float((results['equity'].iloc[-1] / initial_capital - 1) * 100),
                    'num_trades': len(engine.trades),
                    'plots_dir': plots_dir,
                    'plot_files': plot_files,
                }
                
                return render_template('backtest_result.html',
                                     strategy=strategy.name,
                                     results=session['backtest_results'],
                                     performance=performance.to_dict('records'),
                                     plot_files=plot_files)
                                     
            except Exception as e:
                flash(f'Error running backtest: {str(e)}', 'danger')
                return redirect(request.url)
        
        # Strategy options
        strategy_options = [
            {'id': 'crossover', 'name': 'Moving Average Crossover'},
            # Add more strategies here as they are implemented
        ]
        
        return render_template('backtest.html', 
                             strategy_options=strategy_options,
                             has_indicators=has_indicators)
    
    @app.route('/models', methods=['GET', 'POST'])
    def models():
        """Machine learning models page"""
        # Check if data is loaded
        if 'data_path' not in session:
            flash('Please upload data first', 'warning')
            return redirect(url_for('upload'))
        
        # Load data
        loader = DataLoader()
        
        # Use data with indicators if available
        if 'indicators_path' in session and os.path.exists(session['indicators_path']):
            data = loader.load_from_csv(session['indicators_path'])
            has_indicators = True
        else:
            data = loader.load_from_csv(session['data_path'])
            # Calculate basic indicators
            data = calculate_indicators(data)
            has_indicators = False
        
        if request.method == 'POST':
            # Get model parameters
            model_type = request.form.get('model_type', 'lstm')
            sequence_length = int(request.form.get('sequence_length', 60))
            epochs = int(request.form.get('epochs', 50))
            target_column = request.form.get('target_column', 'close')
            
            # Create model based on type
            try:
                if model_type == 'lstm':
                    model = LSTMModel(sequence_length=sequence_length, epochs=epochs)
                    model_name = 'LSTM'
                elif model_type == 'rnn':
                    model = RNNModel(sequence_length=sequence_length, epochs=epochs)
                    model_name = 'RNN'
                elif model_type == 'ensemble':
                    model = EnsembleModel(sequence_length=sequence_length)
                    model_name = 'Ensemble'
                else:
                    flash('Selected model not implemented', 'danger')
                    return redirect(request.url)
                
                # Train model
                history = model.train(data, target_column=target_column, epochs=epochs)
                
                # Make predictions
                predictions = model.predict(data, target_column=target_column)
                
                # Create plot of actual vs predicted values
                plt.figure(figsize=(12, 6))
                
                # Plot actual values
                actual = data[target_column].iloc[-len(predictions):]
                plt.plot(actual.index, actual, label='Actual', linewidth=2)
                
                # Plot predictions
                plt.plot(predictions.index, predictions, label='Predicted', linewidth=2, linestyle='--')
                
                plt.title(f'{model_name} Model Predictions vs Actual')
                plt.xlabel('Date')
                plt.ylabel('Price')
                plt.grid(True, alpha=0.3)
                plt.legend()
                
                # Save plot
                plots_dir = os.path.join(app.config['PLOTS_FOLDER'], str(uuid.uuid4()))
                os.makedirs(plots_dir, exist_ok=True)
                
                predictions_plot = os.path.join(plots_dir, 'predictions.png')
                plt.savefig(predictions_plot, dpi=300, bbox_inches='tight')
                plt.close()
                
                # Get relative path for template
                plot_file = os.path.join('plots', os.path.basename(plots_dir), 'predictions.png')
                
                # If the model has evaluation metrics, include them
                metrics = None
                if hasattr(model, 'evaluation_metrics') and model.evaluation_metrics:
                    metrics = model.evaluation_metrics
                
                return render_template('model_result.html',
                                     model_type=model_type,
                                     model_name=model_name,
                                     sequence_length=sequence_length,
                                     epochs=epochs,
                                     target_column=target_column,
                                     plot_file=plot_file,
                                     metrics=metrics)
                
            except Exception as e:
                flash(f'Error training model: {str(e)}', 'danger')
                return redirect(request.url)
        
        # Model options
        model_options = [
            {'id': 'lstm', 'name': 'LSTM (Long Short-Term Memory)'},
            {'id': 'rnn', 'name': 'RNN (Recurrent Neural Network / GRU)'},
            {'id': 'ensemble', 'name': 'Ensemble (Combined Models)'},
        ]
        
        # Target column options
        target_options = ['close', 'high', 'low', 'open']
        
        return render_template('models.html',
                             model_options=model_options,
                             target_options=target_options,
                             has_indicators=has_indicators)
    
    @app.route('/download_results')
    def download_results():
        """Download backtest results"""
        # Check if backtest results are available
        if 'backtest_results' not in session:
            flash('No backtest results to download', 'warning')
            return redirect(url_for('backtest'))
        
        # Create a text report
        report = []
        report.append("CryptoTradeAnalyzer - Backtest Results")
        report.append("=" * 50)
        report.append(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Strategy: {session['backtest_results']['strategy_type']}")
        report.append(f"Initial Capital: ${session['backtest_results']['initial_capital']:.2f}")
        report.append(f"Commission: {session['backtest_results']['commission'] * 100:.3f}%")
        report.append(f"Final Equity: ${session['backtest_results']['final_equity']:.2f}")
        report.append(f"Return: {session['backtest_results']['return_pct']:.2f}%")
        report.append(f"Number of Trades: {session['backtest_results']['num_trades']}")
        report.append("=" * 50)
        
        # Create a file in memory
        output = io.StringIO()
        output.write('\n'.join(report))
        output.seek(0)
        
        # Return the file for download
        return send_file(
            io.BytesIO(output.getvalue().encode('utf-8')),
            mimetype='text/plain',
            as_attachment=True,
            download_name='backtest_results.txt'
        )
    
    @app.route('/clear_data')
    def clear_data():
        """Clear all data and session"""
        # Clear session
        session.clear()
        
        flash('All data cleared', 'success')
        return redirect(url_for('index'))
