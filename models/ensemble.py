"""
Ensemble Model Module for CryptoTradeAnalyzer

This module implements an ensemble model that combines predictions from multiple
machine learning models to improve accuracy and robustness.

Author: CryptoTradeAnalyzer Team
"""

import os
import numpy as np
import pandas as pd
import logging
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error

from models.lstm import LSTMModel
from models.rnn import RNNModel

logger = logging.getLogger(__name__)

class EnsembleModel:
    """
    Ensemble Model for time series prediction
    
    This class combines predictions from multiple models (LSTM, RNN, etc.)
    to create a more robust prediction.
    """
    
    def __init__(self, weights=None):
        """
        Initialize the Ensemble Model
        
        Parameters:
        -----------
        weights : dict, optional
            Model weights in the ensemble, e.g., {'lstm': 0.6, 'rnn': 0.4}
        """
        self.models = {
            'lstm': LSTMModel(),
            'rnn': RNNModel(),
        }
        
        # Default to equal weights if not provided
        if weights is None:
            self.weights = {model_name: 1.0 / len(self.models) for model_name in self.models}
        else:
            # Normalize weights to sum to 1
            total = sum(weights.values())
            self.weights = {k: v / total for k, v in weights.items()}
            
        self.metrics = {}
        self.predictions = {}
        self.target_column = 'close'
        
    def train(self, data, sequence_length=60, target_column='close', epochs=50, batch_size=32):
        """
        Train all models in the ensemble
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame with price and indicator data
        sequence_length : int
            Number of timesteps to use for each input sequence
        target_column : str
            Target column to predict
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
            
        Returns:
        --------
        dict
            Training metrics for all models
        """
        self.target_column = target_column
        training_metrics = {}
        
        for model_name, model in self.models.items():
            logger.info(f"Training {model_name} model...")
            history = model.train(
                data,
                sequence_length=sequence_length,
                target_column=target_column,
                epochs=epochs,
                batch_size=batch_size
            )
            training_metrics[model_name] = history
            
        # Store training metrics
        self.metrics['training'] = training_metrics
        
        logger.info(f"Ensemble model training completed for all {len(self.models)} models")
        
        return training_metrics
    
    def predict(self, data, sequence_length=None, target_column=None):
        """
        Generate ensemble predictions by combining predictions from all models
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame with price and indicator data
        sequence_length : int, optional
            Number of timesteps to use for each input sequence
        target_column : str, optional
            Target column to predict
            
        Returns:
        --------
        pandas.Series
            Ensemble prediction
        """
        if target_column is not None:
            self.target_column = target_column
            
        # Generate predictions from each model
        model_predictions = {}
        for model_name, model in self.models.items():
            logger.info(f"Generating predictions from {model_name} model...")
            pred = model.predict(data, sequence_length, target_column)
            if pred is not None:
                model_predictions[model_name] = pred
            
        if not model_predictions:
            logger.error("No predictions could be generated. Ensure models are trained.")
            return None
            
        # Store individual model predictions
        self.predictions = model_predictions
        
        # Create a combined prediction using the weighted average
        ensemble_pred = None
        total_weight = 0
        
        for model_name, pred in model_predictions.items():
            weight = self.weights.get(model_name, 0)
            if weight > 0:
                if ensemble_pred is None:
                    ensemble_pred = pred * weight
                else:
                    # Align indices if needed
                    common_index = ensemble_pred.index.intersection(pred.index)
                    ensemble_pred = ensemble_pred.loc[common_index] + pred.loc[common_index] * weight
                total_weight += weight
                
        # Normalize if not all models contributed
        if total_weight < 1.0 and total_weight > 0 and ensemble_pred is not None:
            ensemble_pred = ensemble_pred / total_weight
            
        if ensemble_pred is not None:
            ensemble_pred.name = f'ensemble_predicted_{self.target_column}'
            logger.info(f"Generated ensemble prediction with {len(ensemble_pred)} points")
        
        return ensemble_pred
    
    def evaluate(self, actual_data):
        """
        Evaluate ensemble and individual model performance
        
        Parameters:
        -----------
        actual_data : pandas.Series
            Actual values to compare against predictions
            
        Returns:
        --------
        dict
            Evaluation metrics
        """
        if not self.predictions:
            logger.error("No predictions available. Run predict() first.")
            return None
            
        metrics = {}
        
        # Evaluate each model
        for model_name, predictions in self.predictions.items():
            # Align indices
            common_index = actual_data.index.intersection(predictions.index)
            
            if len(common_index) == 0:
                logger.warning(f"No common indices between actual data and {model_name} predictions.")
                continue
                
            actual = actual_data.loc[common_index]
            preds = predictions.loc[common_index]
            
            # Calculate metrics
            mse = mean_squared_error(actual, preds)
            rmse = np.sqrt(mse)
            mae = mean_absolute_error(actual, preds)
            
            # Calculate directional accuracy
            actual_diff = actual.diff().dropna()
            pred_diff = preds.diff().dropna()
            
            # Align indices again after diff()
            common_diff_index = actual_diff.index.intersection(pred_diff.index)
            
            if len(common_diff_index) > 0:
                actual_direction = actual_diff.loc[common_diff_index] > 0
                pred_direction = pred_diff.loc[common_diff_index] > 0
                directional_accuracy = np.mean(actual_direction == pred_direction)
            else:
                directional_accuracy = np.nan
                
            metrics[model_name] = {
                'mse': mse,
                'rmse': rmse,
                'mae': mae,
                'directional_accuracy': directional_accuracy
            }
            
        # Calculate ensemble metrics
        ensemble_pred = self.predict(None)  # This will use stored predictions
        
        if ensemble_pred is not None:
            common_index = actual_data.index.intersection(ensemble_pred.index)
            
            if len(common_index) > 0:
                actual = actual_data.loc[common_index]
                preds = ensemble_pred.loc[common_index]
                
                # Calculate metrics
                mse = mean_squared_error(actual, preds)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(actual, preds)
                
                # Calculate directional accuracy
                actual_diff = actual.diff().dropna()
                pred_diff = preds.diff().dropna()
                
                # Align indices again after diff()
                common_diff_index = actual_diff.index.intersection(pred_diff.index)
                
                if len(common_diff_index) > 0:
                    actual_direction = actual_diff.loc[common_diff_index] > 0
                    pred_direction = pred_diff.loc[common_diff_index] > 0
                    directional_accuracy = np.mean(actual_direction == pred_direction)
                else:
                    directional_accuracy = np.nan
                    
                metrics['ensemble'] = {
                    'mse': mse,
                    'rmse': rmse,
                    'mae': mae,
                    'directional_accuracy': directional_accuracy
                }
                
        # Store evaluation metrics
        self.metrics['evaluation'] = metrics
        
        return metrics
    
    def plot_predictions(self, actual_data, title="Ensemble Model Predictions"):
        """
        Plot actual vs predicted values for all models and the ensemble
        
        Parameters:
        -----------
        actual_data : pandas.Series
            Series with actual values
        title : str
            Plot title
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure with the plot
        """
        plt.figure(figsize=(14, 7))
        
        # Plot actual data
        plt.plot(actual_data, label='Actual Prices', color='black', linewidth=2)
        
        # Plot individual model predictions
        colors = ['blue', 'green', 'purple', 'orange']
        color_idx = 0
        
        for model_name, predictions in self.predictions.items():
            plt.plot(predictions, label=f'{model_name.upper()} Predictions', 
                    color=colors[color_idx % len(colors)], alpha=0.6, linestyle='--')
            color_idx += 1
            
        # Plot ensemble prediction
        ensemble_pred = self.predict(None)  # This will use stored predictions
        if ensemble_pred is not None:
            plt.plot(ensemble_pred, label='Ensemble Predictions', 
                    color='red', linewidth=2)
        
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return plt.gcf()
    
    def get_optimal_weights(self, actual_data, method='grid_search', metric='rmse'):
        """
        Find optimal weights for the ensemble
        
        Parameters:
        -----------
        actual_data : pandas.Series
            Actual values to compare against predictions
        method : str
            Method to find optimal weights ('grid_search' or 'gradient_descent')
        metric : str
            Metric to optimize ('rmse', 'mae', 'directional_accuracy')
            
        Returns:
        --------
        dict
            Optimal weights for each model
        """
        if not self.predictions:
            logger.error("No predictions available. Run predict() first.")
            return None
        
        if method == 'grid_search' and len(self.predictions) == 2:
            # Simple grid search for two models
            model_names = list(self.predictions.keys())
            best_metric_value = float('inf') if metric in ['rmse', 'mae'] else 0
            best_weights = {model_name: 0.5 for model_name in model_names}
            
            # Create a common index for all predictions
            common_indices = None
            for preds in self.predictions.values():
                if common_indices is None:
                    common_indices = preds.index
                else:
                    common_indices = common_indices.intersection(preds.index)
            
            common_indices = common_indices.intersection(actual_data.index)
            
            if len(common_indices) == 0:
                logger.warning("No common indices between actual data and predictions.")
                return best_weights
                
            # Grid search
            for weight1 in np.linspace(0, 1, 21):  # 0, 0.05, 0.1, ..., 0.95, 1.0
                weight2 = 1 - weight1
                weights = {model_names[0]: weight1, model_names[1]: weight2}
                
                # Calculate weighted prediction
                weighted_pred = self.predictions[model_names[0]].loc[common_indices] * weight1 + \
                              self.predictions[model_names[1]].loc[common_indices] * weight2
                
                # Calculate metric
                if metric == 'rmse':
                    value = np.sqrt(mean_squared_error(
                        actual_data.loc[common_indices], weighted_pred))
                    is_better = value < best_metric_value
                elif metric == 'mae':
                    value = mean_absolute_error(
                        actual_data.loc[common_indices], weighted_pred)
                    is_better = value < best_metric_value
                elif metric == 'directional_accuracy':
                    actual_diff = actual_data.loc[common_indices].diff().dropna()
                    pred_diff = weighted_pred.diff().dropna()
                    common_diff_index = actual_diff.index.intersection(pred_diff.index)
                    
                    if len(common_diff_index) > 0:
                        actual_direction = actual_diff.loc[common_diff_index] > 0
                        pred_direction = pred_diff.loc[common_diff_index] > 0
                        value = np.mean(actual_direction == pred_direction)
                        is_better = value > best_metric_value
                    else:
                        continue
                        
                if is_better:
                    best_metric_value = value
                    best_weights = weights.copy()
            
            logger.info(f"Found optimal weights via grid search: {best_weights}")
            logger.info(f"Best {metric}: {best_metric_value}")
            
            # Update weights
            self.weights = best_weights
            
            return best_weights
        else:
            logger.warning("Optimal weight finding only implemented for grid search with 2 models.")
            return self.weights