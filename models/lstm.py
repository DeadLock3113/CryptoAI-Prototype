"""
LSTM Model Module for CryptoTradeAnalyzer

This module implements a Long Short-Term Memory (LSTM) neural network model
for cryptocurrency price prediction.

Author: CryptoTradeAnalyzer Team
"""

import os
import numpy as np
import pandas as pd
import logging
from datetime import datetime
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

logger = logging.getLogger(__name__)

class LSTMModel:
    """
    LSTM Model for time series prediction
    
    This class implements a Long Short-Term Memory (LSTM) neural network for
    cryptocurrency price prediction.
    """
    
    def __init__(self):
        """
        Initialize the LSTM Model
        """
        self.model = None
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.sequence_length = 60  # Default sequence length
        self.features = []
        self.target_column = 'close'
        self.model_path = 'models/saved/lstm_model'
        self.metrics = {}
        
        # Create directory for saved models if it doesn't exist
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        
    def _prepare_data(self, data, sequence_length=None, target_column=None, test_size=0.2):
        """
        Prepare data for LSTM model training
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame with price and indicator data
        sequence_length : int, optional
            Number of timesteps to use for each input sequence
        target_column : str, optional
            Target column to predict
        test_size : float
            Proportion of data to use for testing
            
        Returns:
        --------
        tuple
            (X_train, y_train, X_test, y_test, scaler)
        """
        if sequence_length is not None:
            self.sequence_length = sequence_length
        
        if target_column is not None:
            self.target_column = target_column
        
        # Identify feature columns (exclude date/time columns)
        self.features = [col for col in data.columns if col != 'timestamp' 
                         and not pd.api.types.is_datetime64_any_dtype(data[col])]
        
        logger.info(f"Using features: {self.features}")
        
        # Extract feature data
        data_for_training = data[self.features].values
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(data_for_training)
        
        # Create sequences
        X, y = [], []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i])
            # Find the index of the target column
            target_idx = self.features.index(self.target_column)
            y.append(scaled_data[i, target_idx])
            
        X, y = np.array(X), np.array(y)
        
        # Split into train and test sets
        split_idx = int(len(X) * (1 - test_size))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        logger.info(f"Training data shape: {X_train.shape}, Test data shape: {X_test.shape}")
        
        return X_train, y_train, X_test, y_test
    
    def build_model(self, input_shape):
        """
        Build the LSTM model architecture
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of input data (sequence_length, features)
            
        Returns:
        --------
        tensorflow.keras.models.Sequential
            Compiled LSTM model
        """
        model = Sequential()
        
        # First LSTM layer with return sequences for stacking
        model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
        model.add(Dropout(0.2))
        
        # Second LSTM layer
        model.add(LSTM(units=50, return_sequences=False))
        model.add(Dropout(0.2))
        
        # Output layer
        model.add(Dense(units=1))
        
        # Compile the model
        model.compile(optimizer='adam', loss='mean_squared_error')
        
        logger.info(f"Model built with input shape: {input_shape}")
        
        return model
    
    def train(self, data, sequence_length=None, target_column=None, epochs=50, batch_size=32):
        """
        Train the LSTM model
        
        Parameters:
        -----------
        data : pandas.DataFrame
            DataFrame with price and indicator data
        sequence_length : int, optional
            Number of timesteps to use for each input sequence
        target_column : str, optional
            Target column to predict
        epochs : int
            Number of training epochs
        batch_size : int
            Batch size for training
            
        Returns:
        --------
        dict
            Training history
        """
        # Prepare data
        X_train, y_train, X_test, y_test = self._prepare_data(
            data, sequence_length, target_column
        )
        
        # Build model
        self.model = self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        # Setup callbacks
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            ModelCheckpoint(filepath=self.model_path, save_best_only=True)
        ]
        
        # Train the model
        history = self.model.fit(
            X_train, y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(X_test, y_test),
            callbacks=callbacks,
            verbose=1
        )
        
        # Evaluate the model
        self._evaluate_model(X_test, y_test, data)
        
        # Save training history
        self.metrics['history'] = {
            'loss': history.history['loss'],
            'val_loss': history.history['val_loss']
        }
        
        # Save model parameters
        self.metrics['parameters'] = {
            'sequence_length': self.sequence_length,
            'target_column': self.target_column,
            'features': self.features,
            'epochs': epochs,
            'batch_size': batch_size
        }
        
        logger.info(f"Model trained for {len(history.history['loss'])} epochs")
        
        return history.history
    
    def _evaluate_model(self, X_test, y_test, original_data):
        """
        Evaluate model performance
        
        Parameters:
        -----------
        X_test : numpy.ndarray
            Test input data
        y_test : numpy.ndarray
            Test target data
        original_data : pandas.DataFrame
            Original dataframe with unscaled data
            
        Returns:
        --------
        dict
            Performance metrics
        """
        # Get predictions
        predictions = self.model.predict(X_test)
        
        # Create a dummy array to inverse transform the scaled predictions
        dummy = np.zeros((len(predictions), len(self.features)))
        target_idx = self.features.index(self.target_column)
        dummy[:, target_idx] = predictions.flatten()
        
        # Inverse transform
        predictions_unscaled = self.scaler.inverse_transform(dummy)[:, target_idx]
        
        # Create a dummy array for actual values
        dummy = np.zeros((len(y_test), len(self.features)))
        dummy[:, target_idx] = y_test
        actual_unscaled = self.scaler.inverse_transform(dummy)[:, target_idx]
        
        # Calculate metrics
        mse = mean_squared_error(y_test, predictions)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, predictions)
        mse_unscaled = mean_squared_error(actual_unscaled, predictions_unscaled)
        rmse_unscaled = np.sqrt(mse_unscaled)
        
        # Calculate directional accuracy
        actual_diff = np.diff(actual_unscaled)
        pred_diff = np.diff(predictions_unscaled)
        directional_accuracy = np.mean((actual_diff > 0) == (pred_diff > 0))
        
        # Store metrics
        self.metrics['performance'] = {
            'mse': mse,
            'rmse': rmse,
            'mae': mae,
            'mse_original': mse_unscaled,
            'rmse_original': rmse_unscaled,
            'directional_accuracy': directional_accuracy
        }
        
        logger.info(f"Model evaluation: RMSE={rmse:.6f}, RMSE (original)={rmse_unscaled:.2f}, " +
                   f"Directional Accuracy={directional_accuracy:.2%}")
        
        return self.metrics['performance']
    
    def predict(self, data, sequence_length=None, target_column=None):
        """
        Make predictions with the trained model
        
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
        numpy.ndarray
            Predicted values
        """
        if self.model is None:
            try:
                self.model = tf.keras.models.load_model(self.model_path)
                logger.info(f"Loaded model from {self.model_path}")
            except:
                logger.error("No trained model found. Please train the model first.")
                return None
        
        if sequence_length is not None:
            self.sequence_length = sequence_length
        
        if target_column is not None:
            self.target_column = target_column
            
        # Update features if needed
        all_columns = [col for col in data.columns if col != 'timestamp' 
                     and not pd.api.types.is_datetime64_any_dtype(data[col])]
        
        if not self.features or set(self.features) != set(all_columns):
            self.features = all_columns
            
        # Extract feature data
        data_for_prediction = data[self.features].values
        
        # Scale the data
        scaled_data = self.scaler.fit_transform(data_for_prediction)
        
        # Create sequences
        X = []
        for i in range(self.sequence_length, len(scaled_data)):
            X.append(scaled_data[i-self.sequence_length:i])
            
        X = np.array(X)
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Create a dummy array to inverse transform the scaled predictions
        dummy = np.zeros((len(predictions), len(self.features)))
        target_idx = self.features.index(self.target_column)
        dummy[:, target_idx] = predictions.flatten()
        
        # Inverse transform
        predictions_unscaled = self.scaler.inverse_transform(dummy)[:, target_idx]
        
        # Create a series with the same index as the input data
        result = pd.Series(
            index=data.index[self.sequence_length:],
            data=predictions_unscaled,
            name=f'predicted_{self.target_column}'
        )
        
        logger.info(f"Generated {len(result)} predictions")
        
        return result
    
    def plot_predictions(self, actual, predictions, title="LSTM Model Predictions"):
        """
        Plot actual vs predicted values
        
        Parameters:
        -----------
        actual : pandas.Series
            Series with actual values
        predictions : pandas.Series
            Series with predicted values
        title : str
            Plot title
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure with the plot
        """
        plt.figure(figsize=(12, 6))
        plt.plot(actual, label='Actual Prices', color='blue')
        plt.plot(predictions, label='Predicted Prices', color='red', linestyle='--')
        plt.title(title)
        plt.xlabel('Date')
        plt.ylabel('Price')
        plt.legend()
        plt.grid(True, alpha=0.3)
        
        return plt.gcf()