"""
Ensemble Model Module for CryptoTradeAnalyzer

This module implements an ensemble approach combining multiple ML models
(LSTM, RNN, and classical ML algorithms) to improve prediction accuracy.
The ensemble can use various combination methods such as:
- Simple averaging
- Weighted averaging
- Stacking (meta-learner)
- Voting

Author: CryptoTradeAnalyzer Team
"""

import numpy as np
import pandas as pd
import os
import logging
import time
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.svm import SVR
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

# Import internal models
from models.lstm import LSTMModel
from models.rnn import RNNModel
from models.preprocessing import TimeSeriesPreprocessor

logger = logging.getLogger(__name__)

class EnsembleModel:
    """
    Ensemble Model for time series forecasting
    
    This class implements an ensemble approach that combines multiple models
    to improve prediction accuracy for cryptocurrency price forecasting.
    """
    
    def __init__(self, models=None, weights=None, ensemble_method='weighted_average', 
                 sequence_length=60, meta_model='ridge'):
        """
        Initialize the Ensemble model
        
        Parameters:
        -----------
        models : list
            List of model names to include in the ensemble
            Options: 'lstm', 'rnn', 'rf' (Random Forest), 'gbm' (Gradient Boosting), 
                    'svr' (Support Vector Regression), 'linear' (Linear Regression)
        weights : list
            List of weights for each model (if using weighted averaging)
        ensemble_method : str
            Method for combining model predictions:
            - 'simple_average': Simple average of all model predictions
            - 'weighted_average': Weighted average of model predictions
            - 'stacking': Train a meta-model on the predictions of base models
        sequence_length : int
            Length of input sequences for deep learning models
        meta_model : str
            Type of meta-model to use for stacking ('ridge', 'linear', 'rf', 'gbm')
        """
        # Default model list if none provided
        if models is None:
            self.models_list = ['lstm', 'rnn', 'rf', 'gbm']
        else:
            self.models_list = models
        
        # Initialize weights (equal by default)
        if weights is None:
            self.weights = [1.0 / len(self.models_list) for _ in self.models_list]
        else:
            # Normalize weights if they don't sum to 1
            total = sum(weights)
            self.weights = [w / total for w in weights]
        
        self.ensemble_method = ensemble_method
        self.sequence_length = sequence_length
        self.meta_model_type = meta_model
        
        # Initialize model instances
        self.model_instances = {}
        self.meta_model = None
        
        # Initialize preprocessor
        self.preprocessor = TimeSeriesPreprocessor()
        
        # Training history and metrics
        self.history = {}
        self.evaluation_metrics = {}
        
        logger.info(f"Initialized Ensemble model with methods: {ensemble_method}")
        logger.info(f"Base models: {self.models_list}")
    
    def _init_models(self):
        """
        Initialize all model instances
        """
        for model_name in self.models_list:
            if model_name.lower() == 'lstm':
                self.model_instances[model_name] = LSTMModel(sequence_length=self.sequence_length)
            
            elif model_name.lower() == 'rnn':
                self.model_instances[model_name] = RNNModel(sequence_length=self.sequence_length)
            
            elif model_name.lower() == 'rf':
                self.model_instances[model_name] = RandomForestRegressor(
                    n_estimators=100, 
                    max_depth=20,
                    random_state=42
                )
            
            elif model_name.lower() == 'gbm':
                self.model_instances[model_name] = GradientBoostingRegressor(
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=3,
                    random_state=42
                )
            
            elif model_name.lower() == 'svr':
                self.model_instances[model_name] = SVR(
                    kernel='rbf',
                    C=1.0,
                    epsilon=0.1
                )
            
            elif model_name.lower() == 'linear':
                self.model_instances[model_name] = LinearRegression()
            
            else:
                logger.warning(f"Unknown model type: {model_name}. Skipping.")
        
        # Initialize meta-model for stacking if needed
        if self.ensemble_method == 'stacking':
            if self.meta_model_type == 'ridge':
                self.meta_model = Ridge(alpha=1.0)
            elif self.meta_model_type == 'linear':
                self.meta_model = LinearRegression()
            elif self.meta_model_type == 'rf':
                self.meta_model = RandomForestRegressor(n_estimators=50, random_state=42)
            elif self.meta_model_type == 'gbm':
                self.meta_model = GradientBoostingRegressor(n_estimators=50, random_state=42)
            else:
                logger.warning(f"Unknown meta-model type: {self.meta_model_type}. Using Ridge regression.")
                self.meta_model = Ridge(alpha=1.0)
    
    def prepare_data(self, df, target_column='close'):
        """
        Prepare data for all models in the ensemble
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with OHLCV data and indicators
        target_column : str
            Name of target column to predict
            
        Returns:
        --------
        dict
            Dictionary with prepared data for each model type
        """
        logger.info("Preparing data for Ensemble model")
        
        # Make a copy of the dataframe
        data = df.copy()
        
        # Add technical indicators
        data = self.preprocessor.add_technical_indicators(data)
        
        # Add lag features for close price
        data = self.preprocessor.add_lag_features(data, [target_column])
        
        # Add rolling features for close price
        data = self.preprocessor.add_rolling_features(data, [target_column])
        
        # Add return features
        data = self.preprocessor.add_return_features(data, [target_column])
        
        # Add time features if possible
        if isinstance(data.index, pd.DatetimeIndex):
            data = self.preprocessor.add_time_features(data)
        
        # Drop rows with NaN values
        data.dropna(inplace=True)
        
        # Define feature columns (all except target)
        feature_columns = [col for col in data.columns if col != target_column]
        
        # Scale features and target
        data = self.preprocessor.scale_features(data, feature_columns)
        data = self.preprocessor.scale_target(data, target_column)
        
        # Split data into train, validation, and test sets
        train_data, val_data, test_data = self.preprocessor.train_val_test_split(data)
        
        # Prepare data for different model types
        prepared_data = {
            'train_data': train_data,
            'val_data': val_data,
            'test_data': test_data,
            'feature_columns': feature_columns,
            'target_column': target_column,
            'dl_models': {},  # For deep learning models (LSTM, RNN)
            'ml_models': {}   # For classical ML models
        }
        
        # Create sequences for deep learning models
        if any(model in self.models_list for model in ['lstm', 'rnn']):
            X_train, y_train = self.preprocessor.create_sequences(
                train_data, feature_columns, target_column, self.sequence_length)
            
            X_val, y_val = self.preprocessor.create_sequences(
                val_data, feature_columns, target_column, self.sequence_length)
            
            X_test, y_test = self.preprocessor.create_sequences(
                test_data, feature_columns, target_column, self.sequence_length)
            
            prepared_data['dl_models'] = {
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test
            }
        
        # Prepare data for classical ML models
        if any(model in self.models_list for model in ['rf', 'gbm', 'svr', 'linear']):
            # ML models don't need sequences, just feature vectors
            X_train = train_data[feature_columns].values
            y_train = train_data[target_column].values
            
            X_val = val_data[feature_columns].values
            y_val = val_data[target_column].values
            
            X_test = test_data[feature_columns].values
            y_test = test_data[target_column].values
            
            prepared_data['ml_models'] = {
                'X_train': X_train,
                'y_train': y_train,
                'X_val': X_val,
                'y_val': y_val,
                'X_test': X_test,
                'y_test': y_test
            }
        
        logger.info(f"Data prepared for {len(self.models_list)} models")
        return prepared_data
    
    def train(self, df, target_column='close', epochs=50):
        """
        Train all models in the ensemble
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with OHLCV data and indicators
        target_column : str
            Name of target column to predict
        epochs : int
            Number of epochs for neural network models
            
        Returns:
        --------
        dict
            Dictionary with training metrics for each model
        """
        logger.info("Training Ensemble model")
        
        # Initialize models if not already done
        if not self.model_instances:
            self._init_models()
        
        # Prepare data for all models
        prepared_data = self.prepare_data(df, target_column)
        
        # Train each model
        for model_name, model in self.model_instances.items():
            logger.info(f"Training {model_name} model")
            
            if model_name in ['lstm', 'rnn']:
                # Deep learning models
                if not hasattr(model, 'model') or model.model is None:
                    # Build model if not already built
                    input_shape = (prepared_data['dl_models']['X_train'].shape[1],
                                  prepared_data['dl_models']['X_train'].shape[2])
                    model.build_model(input_shape)
                
                # Train model
                history = model.model.fit(
                    prepared_data['dl_models']['X_train'],
                    prepared_data['dl_models']['y_train'],
                    validation_data=(
                        prepared_data['dl_models']['X_val'],
                        prepared_data['dl_models']['y_val']
                    ),
                    epochs=epochs,
                    batch_size=32,
                    verbose=1
                )
                
                # Store history
                self.history[model_name] = history.history
                
            else:
                # Classical ML models
                model.fit(
                    prepared_data['ml_models']['X_train'],
                    prepared_data['ml_models']['y_train']
                )
                
                # Evaluate on validation set to store metrics
                y_pred = model.predict(prepared_data['ml_models']['X_val'])
                mse = mean_squared_error(prepared_data['ml_models']['y_val'], y_pred)
                mae = mean_absolute_error(prepared_data['ml_models']['y_val'], y_pred)
                
                # Store "history" (just final metrics)
                self.history[model_name] = {
                    'val_loss': [mse],
                    'val_mae': [mae]
                }
        
        # If using stacking, train the meta-model
        if self.ensemble_method == 'stacking':
            self._train_meta_model(prepared_data)
        
        # Evaluate the ensemble
        self.evaluate(prepared_data)
        
        return self.history
    
    def _train_meta_model(self, prepared_data):
        """
        Train the meta-model for stacking ensemble
        
        Parameters:
        -----------
        prepared_data : dict
            Dictionary with prepared data
        """
        logger.info("Training meta-model for stacking ensemble")
        
        # Get validation data for all models
        base_predictions = []
        
        for model_name, model in self.model_instances.items():
            if model_name in ['lstm', 'rnn']:
                # Deep learning models
                preds = model.model.predict(prepared_data['dl_models']['X_val'])
                base_predictions.append(preds.flatten())
            else:
                # Classical ML models
                preds = model.predict(prepared_data['ml_models']['X_val'])
                base_predictions.append(preds)
        
        # Stack predictions into a feature matrix
        meta_features = np.column_stack(base_predictions)
        
        # Get validation targets
        if 'lstm' in self.models_list or 'rnn' in self.models_list:
            meta_targets = prepared_data['dl_models']['y_val']
        else:
            meta_targets = prepared_data['ml_models']['y_val']
        
        # Train meta-model
        self.meta_model.fit(meta_features, meta_targets)
        logger.info("Meta-model training completed")
    
    def evaluate(self, prepared_data=None, df=None, target_column='close'):
        """
        Evaluate the ensemble model
        
        Parameters:
        -----------
        prepared_data : dict
            Dictionary with prepared data (if already prepared)
        df : pandas.DataFrame
            DataFrame with OHLCV data (if data not prepared yet)
        target_column : str
            Name of target column to predict
            
        Returns:
        --------
        dict
            Dictionary with evaluation metrics
        """
        # If no prepared data and no dataframe, we can't evaluate
        if prepared_data is None and df is None:
            logger.error("No data provided for evaluation")
            return None
        
        # If no prepared data but dataframe provided, prepare the data
        if prepared_data is None:
            prepared_data = self.prepare_data(df, target_column)
        
        logger.info("Evaluating Ensemble model")
        
        # Evaluate each base model first
        base_predictions = {}
        base_metrics = {}
        
        # Get test data
        if 'lstm' in self.models_list or 'rnn' in self.models_list:
            X_test_dl = prepared_data['dl_models']['X_test']
            y_test_dl = prepared_data['dl_models']['y_test']
        
        if 'rf' in self.models_list or 'gbm' in self.models_list or 'svr' in self.models_list or 'linear' in self.models_list:
            X_test_ml = prepared_data['ml_models']['X_test']
            y_test_ml = prepared_data['ml_models']['y_test']
        
        # Get predictions from each base model
        for model_name, model in self.model_instances.items():
            if model_name in ['lstm', 'rnn']:
                preds = model.model.predict(X_test_dl)
                base_predictions[model_name] = preds.flatten()
                
                # Calculate metrics
                mse = mean_squared_error(y_test_dl, preds)
                mae = mean_absolute_error(y_test_dl, preds)
                rmse = np.sqrt(mse)
                
                base_metrics[model_name] = {
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse
                }
                
            else:
                preds = model.predict(X_test_ml)
                base_predictions[model_name] = preds
                
                # Calculate metrics
                mse = mean_squared_error(y_test_ml, preds)
                mae = mean_absolute_error(y_test_ml, preds)
                rmse = np.sqrt(mse)
                
                base_metrics[model_name] = {
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse
                }
        
        # Get ensemble predictions based on the chosen method
        if self.ensemble_method == 'simple_average':
            # Simple average of all predictions
            ensemble_predictions = np.mean(list(base_predictions.values()), axis=0)
            
        elif self.ensemble_method == 'weighted_average':
            # Weighted average of predictions
            weighted_preds = []
            for i, (model_name, preds) in enumerate(base_predictions.items()):
                weighted_preds.append(preds * self.weights[i])
            
            ensemble_predictions = np.sum(weighted_preds, axis=0)
            
        elif self.ensemble_method == 'stacking':
            # Stacking: use meta-model to combine predictions
            meta_features = np.column_stack(list(base_predictions.values()))
            ensemble_predictions = self.meta_model.predict(meta_features)
        
        # Calculate ensemble metrics
        if 'lstm' in self.models_list or 'rnn' in self.models_list:
            y_test = y_test_dl
        else:
            y_test = y_test_ml
            
        ensemble_mse = mean_squared_error(y_test, ensemble_predictions)
        ensemble_mae = mean_absolute_error(y_test, ensemble_predictions)
        ensemble_rmse = np.sqrt(ensemble_mse)
        
        # Calculate directional accuracy
        ensemble_direction = np.sign(np.diff(ensemble_predictions))
        actual_direction = np.sign(np.diff(y_test))
        directional_accuracy = np.mean(ensemble_direction == actual_direction)
        
        # Store metrics
        ensemble_metrics = {
            'mse': ensemble_mse,
            'mae': ensemble_mae,
            'rmse': ensemble_rmse,
            'directional_accuracy': directional_accuracy,
            'base_models': base_metrics
        }
        
        self.evaluation_metrics = ensemble_metrics
        
        # Log results
        logger.info(f"Ensemble evaluation: MSE={ensemble_mse:.4f}, MAE={ensemble_mae:.4f}, RMSE={ensemble_rmse:.4f}")
        logger.info(f"Directional accuracy: {directional_accuracy:.4f}")
        
        return ensemble_metrics
    
    def predict(self, df, target_column='close'):
        """
        Make predictions with the ensemble model
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with OHLCV data and indicators
        target_column : str
            Name of target column to predict
            
        Returns:
        --------
        pandas.Series
            Series with ensemble predictions
        """
        logger.info("Making predictions with Ensemble model")
        
        # Initialize models if not already done
        if not self.model_instances:
            self._init_models()
            self.load()
        
        # Prepare data
        prepared_data = self.prepare_data(df, target_column)
        
        # Get test data
        if 'lstm' in self.models_list or 'rnn' in self.models_list:
            X_test_dl = prepared_data['dl_models']['X_test']
        
        if 'rf' in self.models_list or 'gbm' in self.models_list or 'svr' in self.models_list or 'linear' in self.models_list:
            X_test_ml = prepared_data['ml_models']['X_test']
        
        # Get predictions from each base model
        base_predictions = {}
        
        for model_name, model in self.model_instances.items():
            if model_name in ['lstm', 'rnn']:
                if hasattr(model, 'model') and model.model is not None:
                    preds = model.model.predict(X_test_dl)
                    base_predictions[model_name] = preds.flatten()
                else:
                    logger.warning(f"{model_name} model not trained/loaded. Skipping.")
            else:
                # Classical ML models
                try:
                    preds = model.predict(X_test_ml)
                    base_predictions[model_name] = preds
                except:
                    logger.warning(f"{model_name} model not trained/loaded. Skipping.")
        
        # If no predictions, return None
        if not base_predictions:
            logger.error("No predictions generated. Models may not be trained.")
            return None
        
        # Get ensemble predictions based on the chosen method
        if self.ensemble_method == 'simple_average':
            # Simple average of all predictions
            ensemble_predictions = np.mean(list(base_predictions.values()), axis=0)
            
        elif self.ensemble_method == 'weighted_average':
            # Weighted average of predictions
            weighted_preds = []
            for i, (model_name, preds) in enumerate(base_predictions.items()):
                if i < len(self.weights):  # In case not all models made predictions
                    weighted_preds.append(preds * self.weights[i])
            
            ensemble_predictions = np.sum(weighted_preds, axis=0)
            
        elif self.ensemble_method == 'stacking':
            # Stacking: use meta-model to combine predictions
            if self.meta_model is not None:
                meta_features = np.column_stack(list(base_predictions.values()))
                ensemble_predictions = self.meta_model.predict(meta_features)
            else:
                logger.warning("Meta-model not trained. Falling back to simple average.")
                ensemble_predictions = np.mean(list(base_predictions.values()), axis=0)
        
        # Inverse scale predictions
        if hasattr(self.preprocessor, 'scalers') and 'target' in self.preprocessor.scalers:
            ensemble_predictions = self.preprocessor.inverse_scale_target(
                ensemble_predictions.reshape(-1, 1)).flatten()
        
        # Create Series with predictions
        test_data = prepared_data['test_data']
        pred_index = test_data.index
        
        # Adjust for sequence_length
        if 'lstm' in self.models_list or 'rnn' in self.models_list:
            pred_index = pred_index[self.sequence_length:]
        
        # Ensure pred_index and predictions have same length
        if len(pred_index) > len(ensemble_predictions):
            pred_index = pred_index[:len(ensemble_predictions)]
        
        predictions_series = pd.Series(ensemble_predictions, index=pred_index)
        
        logger.info(f"Generated {len(predictions_series)} ensemble predictions")
        
        return predictions_series
    
    def save(self, base_dir='models/saved'):
        """
        Save all models in the ensemble
        
        Parameters:
        -----------
        base_dir : str
            Base directory to save models
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        logger.info("Saving Ensemble model")
        
        # Create directory if it doesn't exist
        os.makedirs(base_dir, exist_ok=True)
        
        try:
            # Save each model
            for model_name, model in self.model_instances.items():
                if model_name in ['lstm', 'rnn']:
                    # Deep learning models have their own save method
                    model.save(os.path.join(base_dir, f'{model_name}_model'))
                else:
                    # Save scikit-learn models
                    joblib.dump(model, os.path.join(base_dir, f'{model_name}_model.pkl'))
            
            # Save meta-model if exists
            if self.meta_model is not None:
                joblib.dump(self.meta_model, os.path.join(base_dir, 'meta_model.pkl'))
            
            # Save configuration
            config = {
                'models_list': self.models_list,
                'weights': self.weights,
                'ensemble_method': self.ensemble_method,
                'sequence_length': self.sequence_length,
                'meta_model_type': self.meta_model_type
            }
            joblib.dump(config, os.path.join(base_dir, 'ensemble_config.pkl'))
            
            logger.info(f"Ensemble model saved to {base_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save ensemble model: {str(e)}")
            return False
    
    def load(self, base_dir='models/saved'):
        """
        Load all models in the ensemble
        
        Parameters:
        -----------
        base_dir : str
            Base directory to load models from
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        logger.info("Loading Ensemble model")
        
        try:
            # Load configuration if exists
            config_path = os.path.join(base_dir, 'ensemble_config.pkl')
            if os.path.exists(config_path):
                config = joblib.load(config_path)
                self.models_list = config['models_list']
                self.weights = config['weights']
                self.ensemble_method = config['ensemble_method']
                self.sequence_length = config['sequence_length']
                self.meta_model_type = config['meta_model_type']
            
            # Initialize models
            self._init_models()
            
            # Load each model
            for model_name, model in self.model_instances.items():
                if model_name in ['lstm', 'rnn']:
                    # Deep learning models have their own load method
                    model.load(os.path.join(base_dir, f'{model_name}_model'))
                else:
                    # Load scikit-learn models
                    model_path = os.path.join(base_dir, f'{model_name}_model.pkl')
                    if os.path.exists(model_path):
                        self.model_instances[model_name] = joblib.load(model_path)
                    else:
                        logger.warning(f"Model file not found: {model_path}")
            
            # Load meta-model if exists
            meta_model_path = os.path.join(base_dir, 'meta_model.pkl')
            if os.path.exists(meta_model_path):
                self.meta_model = joblib.load(meta_model_path)
            
            logger.info(f"Ensemble model loaded from {base_dir}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load ensemble model: {str(e)}")
            return False
    
    def plot_model_comparison(self):
        """
        Plot comparison of base model performances
        
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object for the plot
        """
        if not self.evaluation_metrics or 'base_models' not in self.evaluation_metrics:
            logger.error("No evaluation metrics available. Run evaluate() first.")
            return None
        
        # Extract metrics for each model
        models = list(self.evaluation_metrics['base_models'].keys()) + ['Ensemble']
        rmse_values = [self.evaluation_metrics['base_models'][model]['rmse'] for model in models[:-1]]
        rmse_values.append(self.evaluation_metrics['rmse'])
        
        mae_values = [self.evaluation_metrics['base_models'][model]['mae'] for model in models[:-1]]
        mae_values.append(self.evaluation_metrics['mae'])
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot RMSE
        bar_width = 0.35
        x = np.arange(len(models))
        axes[0].bar(x, rmse_values, bar_width, label='RMSE')
        axes[0].set_ylabel('RMSE')
        axes[0].set_title('Model Comparison - RMSE')
        axes[0].set_xticks(x)
        axes[0].set_xticklabels(models)
        
        # Plot MAE
        axes[1].bar(x, mae_values, bar_width, label='MAE')
        axes[1].set_ylabel('MAE')
        axes[1].set_title('Model Comparison - MAE')
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(models)
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
