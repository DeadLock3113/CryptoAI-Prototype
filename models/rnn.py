"""
RNN Model Module for CryptoTradeAnalyzer

This module implements a Recurrent Neural Network (RNN) with GRU cells for time series
forecasting of cryptocurrency prices. GRU (Gated Recurrent Unit) cells are used as a
more efficient alternative to LSTM cells while still capturing temporal dependencies.

Author: CryptoTradeAnalyzer Team
"""

import numpy as np
import pandas as pd
import os
import logging
import time
import matplotlib.pyplot as plt
from datetime import datetime

# TensorFlow imports
import tensorflow as tf
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import GRU, Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau, TensorBoard
from tensorflow.keras.optimizers import Adam

# Preprocessing imports
from models.preprocessing import TimeSeriesPreprocessor

logger = logging.getLogger(__name__)

class RNNModel:
    """
    RNN Model for time series forecasting
    
    This class implements a Recurrent Neural Network (RNN) with GRU cells for
    forecasting cryptocurrency prices based on historical data and technical indicators.
    """
    
    def __init__(self, sequence_length=60, units=100, dropout_rate=0.2, 
                 learning_rate=0.001, batch_size=32, epochs=50, bidirectional=True):
        """
        Initialize the RNN model
        
        Parameters:
        -----------
        sequence_length : int
            Length of input sequences (lookback window)
        units : int
            Number of GRU units in the first layer
        dropout_rate : float
            Dropout rate for regularization
        learning_rate : float
            Learning rate for optimizer
        batch_size : int
            Batch size for training
        epochs : int
            Maximum number of epochs for training
        bidirectional : bool
            Whether to use bidirectional GRU layers
        """
        self.sequence_length = sequence_length
        self.units = units
        self.dropout_rate = dropout_rate
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.epochs = epochs
        self.bidirectional = bidirectional
        
        # Initialize preprocessor
        self.preprocessor = TimeSeriesPreprocessor()
        
        # Initialize model
        self.model = None
        
        # Training history
        self.history = None
        
        # Store model evaluation metrics
        self.evaluation_metrics = {}
        
        logger.info(f"Initialized RNN model with sequence_length={sequence_length}, units={units}, bidirectional={bidirectional}")
    
    def build_model(self, input_shape):
        """
        Build the RNN model architecture
        
        Parameters:
        -----------
        input_shape : tuple
            Shape of input data (sequence_length, n_features)
            
        Returns:
        --------
        tensorflow.keras.models.Sequential
            Compiled RNN model
        """
        model = Sequential()
        
        # First GRU layer with return sequences for stacking
        if self.bidirectional:
            from tensorflow.keras.layers import Bidirectional
            model.add(Bidirectional(GRU(units=self.units, 
                                       return_sequences=True), 
                                  input_shape=input_shape))
        else:
            model.add(GRU(units=self.units, 
                         return_sequences=True, 
                         input_shape=input_shape))
        
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        # Second GRU layer
        if self.bidirectional:
            model.add(Bidirectional(GRU(units=self.units // 2, 
                                       return_sequences=False)))
        else:
            model.add(GRU(units=self.units // 2, 
                         return_sequences=False))
        
        model.add(BatchNormalization())
        model.add(Dropout(self.dropout_rate))
        
        # Dense output layer
        model.add(Dense(1))
        
        # Compile model
        optimizer = Adam(learning_rate=self.learning_rate)
        model.compile(optimizer=optimizer, loss='mse', metrics=['mae'])
        
        # Store model
        self.model = model
        
        logger.info(f"Built RNN model with input shape {input_shape}")
        return model
    
    def prepare_data(self, df, target_column='close'):
        """
        Prepare data for RNN model
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with OHLCV data and indicators
        target_column : str
            Name of target column to predict
            
        Returns:
        --------
        tuple
            Tuple of (X_train, y_train, X_val, y_val, X_test, y_test)
        """
        logger.info("Preparing data for RNN model")
        
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
        
        # Create sequences
        X_train, y_train = self.preprocessor.create_sequences(
            train_data, feature_columns, target_column, self.sequence_length)
        
        X_val, y_val = self.preprocessor.create_sequences(
            val_data, feature_columns, target_column, self.sequence_length)
        
        X_test, y_test = self.preprocessor.create_sequences(
            test_data, feature_columns, target_column, self.sequence_length)
        
        logger.info(f"Prepared data: X_train shape={X_train.shape}, y_train shape={y_train.shape}")
        logger.info(f"Validation data: X_val shape={X_val.shape}, y_val shape={y_val.shape}")
        logger.info(f"Test data: X_test shape={X_test.shape}, y_test shape={y_test.shape}")
        
        return X_train, y_train, X_val, y_val, X_test, y_test
    
    def train(self, df, target_column='close', epochs=None):
        """
        Train the RNN model
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with OHLCV data and indicators
        target_column : str
            Name of target column to predict
        epochs : int
            Number of training epochs (overrides the one set in __init__)
            
        Returns:
        --------
        tensorflow.keras.callbacks.History
            Training history
        """
        logger.info("Training RNN model")
        
        # Use provided epochs if given, otherwise use the one set in __init__
        if epochs is not None:
            self.epochs = epochs
        
        # Prepare data
        X_train, y_train, X_val, y_val, X_test, y_test = self.prepare_data(df, target_column)
        
        # Build model if not already built
        if self.model is None:
            self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        
        # Create model directory if it doesn't exist
        model_dir = 'models/saved'
        os.makedirs(model_dir, exist_ok=True)
        
        # Define callbacks
        callbacks = [
            # Early stopping to prevent overfitting
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True),
            
            # Model checkpoint to save best model
            ModelCheckpoint(
                filepath=os.path.join(model_dir, 'rnn_model'),
                monitor='val_loss',
                save_best_only=True
            ),
            
            # Reduce learning rate when plateau is reached
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-5),
            
            # TensorBoard logging
            TensorBoard(log_dir=os.path.join('logs', f'rnn_{datetime.now().strftime("%Y%m%d-%H%M%S")}'))
        ]
        
        # Train model
        start_time = time.time()
        
        history = self.model.fit(
            X_train, y_train,
            validation_data=(X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            verbose=1
        )
        
        train_time = time.time() - start_time
        logger.info(f"Model training completed in {train_time:.2f} seconds")
        
        # Store training history
        self.history = history.history
        
        # Evaluate model
        self.evaluate(X_test, y_test)
        
        return history
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate the RNN model on test data
        
        Parameters:
        -----------
        X_test : numpy.ndarray
            Test features
        y_test : numpy.ndarray
            Test target
            
        Returns:
        --------
        dict
            Evaluation metrics
        """
        if self.model is None:
            logger.error("Model not trained yet. Call train() first.")
            return None
        
        # Evaluate model
        logger.info("Evaluating RNN model on test data")
        evaluation = self.model.evaluate(X_test, y_test, verbose=1)
        
        # Store metrics
        metrics = {
            'mse': evaluation[0],
            'mae': evaluation[1],
            'rmse': np.sqrt(evaluation[0])
        }
        
        # Make predictions
        y_pred = self.model.predict(X_test)
        
        # Inverse scale predictions and actual values
        if hasattr(self.preprocessor, 'scalers') and 'target' in self.preprocessor.scalers:
            y_test_original = self.preprocessor.inverse_scale_target(y_test.reshape(-1, 1)).flatten()
            y_pred_original = self.preprocessor.inverse_scale_target(y_pred).flatten()
            
            # Calculate additional metrics on original scale
            metrics['mse_original'] = np.mean((y_test_original - y_pred_original) ** 2)
            metrics['mae_original'] = np.mean(np.abs(y_test_original - y_pred_original))
            metrics['rmse_original'] = np.sqrt(metrics['mse_original'])
            
            # Calculate directional accuracy (% of correct direction predictions)
            y_test_direction = np.sign(np.diff(y_test_original))
            y_pred_direction = np.sign(np.diff(y_pred_original))
            directional_accuracy = np.mean(y_test_direction == y_pred_direction)
            metrics['directional_accuracy'] = directional_accuracy
        
        # Store evaluation metrics
        self.evaluation_metrics = metrics
        
        # Log metrics
        logger.info(f"RNN model evaluation: MSE={metrics['mse']:.4f}, MAE={metrics['mae']:.4f}, RMSE={metrics['rmse']:.4f}")
        if 'directional_accuracy' in metrics:
            logger.info(f"Directional accuracy: {metrics['directional_accuracy']:.4f}")
        
        return metrics
    
    def predict(self, df, target_column='close'):
        """
        Make predictions with the RNN model
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with OHLCV data and indicators
        target_column : str
            Name of target column to predict
            
        Returns:
        --------
        numpy.ndarray
            Array of predictions
        """
        logger.info("Making predictions with RNN model")
        
        # Load model if not loaded
        if self.model is None:
            self.load()
            
            if self.model is None:
                logger.error("Failed to load model. Train or load a model first.")
                return None
        
        # Prepare data similar to training
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
        
        # Create sequences
        X, y = self.preprocessor.create_sequences(
            data, feature_columns, target_column, self.sequence_length)
        
        # Make predictions
        predictions = self.model.predict(X)
        
        # Inverse scale predictions
        if hasattr(self.preprocessor, 'scalers') and 'target' in self.preprocessor.scalers:
            predictions = self.preprocessor.inverse_scale_target(predictions).flatten()
        
        # Create DataFrame with predictions
        # The predictions are for sequence_length points after the start of each sequence
        pred_index = data.index[self.sequence_length:]
        if len(pred_index) > len(predictions):
            pred_index = pred_index[:len(predictions)]
        
        predictions_df = pd.DataFrame({
            'predictions': predictions
        }, index=pred_index)
        
        logger.info(f"Generated {len(predictions)} predictions")
        
        return predictions_df['predictions']
    
    def save(self, filepath='models/saved/rnn_model'):
        """
        Save the RNN model to disk
        
        Parameters:
        -----------
        filepath : str
            Path to save the model
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        if self.model is None:
            logger.error("No model to save. Train a model first.")
            return False
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)
            
            # Save model
            self.model.save(filepath)
            logger.info(f"Model saved to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to save model: {str(e)}")
            return False
    
    def load(self, filepath='models/saved/rnn_model'):
        """
        Load the RNN model from disk
        
        Parameters:
        -----------
        filepath : str
            Path to the saved model
            
        Returns:
        --------
        bool
            True if successful, False otherwise
        """
        try:
            if not os.path.exists(filepath):
                logger.error(f"Model file not found: {filepath}")
                return False
            
            # Load model
            self.model = load_model(filepath)
            logger.info(f"Model loaded from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            return False
    
    def plot_training_history(self):
        """
        Plot training history
        
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object for the plot
        """
        if self.history is None:
            logger.error("No training history. Train a model first.")
            return None
        
        # Create figure
        fig, axes = plt.subplots(2, 1, figsize=(12, 10))
        
        # Plot loss
        axes[0].plot(self.history['loss'], label='Training Loss')
        axes[0].plot(self.history['val_loss'], label='Validation Loss')
        axes[0].set_title('Model Loss')
        axes[0].set_ylabel('Loss')
        axes[0].set_xlabel('Epoch')
        axes[0].legend()
        
        # Plot MAE
        axes[1].plot(self.history['mae'], label='Training MAE')
        axes[1].plot(self.history['val_mae'], label='Validation MAE')
        axes[1].set_title('Model MAE')
        axes[1].set_ylabel('MAE')
        axes[1].set_xlabel('Epoch')
        axes[1].legend()
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
    
    def plot_predictions(self, y_true, y_pred, title="RNN Model Predictions"):
        """
        Plot predictions vs actual values
        
        Parameters:
        -----------
        y_true : numpy.ndarray
            True values
        y_pred : numpy.ndarray
            Predicted values
        title : str
            Plot title
            
        Returns:
        --------
        matplotlib.figure.Figure
            Figure object for the plot
        """
        # Create figure
        fig, ax = plt.subplots(figsize=(12, 6))
        
        # Plot true values
        ax.plot(y_true, label='Actual')
        
        # Plot predictions
        ax.plot(y_pred, label='Predicted')
        
        # Add labels and title
        ax.set_title(title)
        ax.set_ylabel('Price')
        ax.set_xlabel('Time')
        ax.legend()
        
        # Adjust layout
        plt.tight_layout()
        
        return fig
