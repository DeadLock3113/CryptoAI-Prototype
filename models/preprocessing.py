"""
Data Preprocessing Module for CryptoTradeAnalyzer

This module handles data preprocessing tasks needed for machine learning models, including:
- Feature engineering 
- Scaling/normalization
- Train-test splitting
- Time series windowing
- Feature selection

Author: CryptoTradeAnalyzer Team
"""

import pandas as pd
import numpy as np
import logging
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.feature_selection import SelectKBest, f_regression

logger = logging.getLogger(__name__)

class TimeSeriesPreprocessor:
    """
    Time Series Preprocessor for machine learning models
    
    This class handles data preprocessing specific to time series data,
    particularly for financial market data.
    """
    
    def __init__(self):
        """Initialize the preprocessor with default parameters"""
        self.scalers = {}
        self.feature_columns = None
        self.target_column = None
        self.sequence_length = None
        self.train_indices = None
        self.val_indices = None
        self.test_indices = None
    
    def add_time_features(self, df):
        """
        Add time-based features to the DataFrame
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with datetime index
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added time features
        """
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Make sure the index is datetime
        if not isinstance(df.index, pd.DatetimeIndex):
            logger.warning("DataFrame index is not DatetimeIndex. Attempting to convert.")
            try:
                result.index = pd.to_datetime(result.index)
            except:
                logger.error("Failed to convert index to datetime. Time features not added.")
                return df
        
        # Add day of week
        result['dayofweek'] = result.index.dayofweek
        
        # Add day of month
        result['dayofmonth'] = result.index.day
        
        # Add day of year
        result['dayofyear'] = result.index.dayofyear
        
        # Add month
        result['month'] = result.index.month
        
        # Add quarter
        result['quarter'] = result.index.quarter
        
        # Add year
        result['year'] = result.index.year
        
        # Add hour (if available)
        if result.index.hour.any():
            result['hour'] = result.index.hour
        
        # Convert categorical time features to cyclical features
        # This helps models understand the cyclical nature of time
        
        # Day of week (0-6) -> sin and cos
        result['dayofweek_sin'] = np.sin(2 * np.pi * result['dayofweek'] / 7)
        result['dayofweek_cos'] = np.cos(2 * np.pi * result['dayofweek'] / 7)
        
        # Month (1-12) -> sin and cos
        result['month_sin'] = np.sin(2 * np.pi * (result['month'] - 1) / 12)
        result['month_cos'] = np.cos(2 * np.pi * (result['month'] - 1) / 12)
        
        # Drop original categorical features
        result.drop(['dayofweek', 'month'], axis=1, inplace=True)
        
        logger.info("Added time features to DataFrame")
        return result
    
    def add_lag_features(self, df, columns, lag_periods=[1, 3, 7, 14, 30]):
        """
        Add lagged values as features
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame to add lag features to
        columns : list
            List of column names to create lags for
        lag_periods : list
            List of lag periods
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added lag features
        """
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Add lag features for each column and lag period
        for column in columns:
            if column not in df.columns:
                logger.warning(f"Column {column} not found in DataFrame")
                continue
                
            for lag in lag_periods:
                result[f'{column}_lag_{lag}'] = df[column].shift(lag)
        
        logger.info(f"Added lag features for columns {columns} with periods {lag_periods}")
        return result
    
    def add_rolling_features(self, df, columns, windows=[5, 10, 20, 50]):
        """
        Add rolling statistics as features
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame to add rolling features to
        columns : list
            List of column names to create rolling stats for
        windows : list
            List of rolling window sizes
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added rolling features
        """
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Add rolling features for each column and window
        for column in columns:
            if column not in df.columns:
                logger.warning(f"Column {column} not found in DataFrame")
                continue
                
            for window in windows:
                # Rolling mean
                result[f'{column}_rolling_mean_{window}'] = df[column].rolling(window=window).mean()
                
                # Rolling standard deviation
                result[f'{column}_rolling_std_{window}'] = df[column].rolling(window=window).std()
                
                # Rolling min and max
                result[f'{column}_rolling_min_{window}'] = df[column].rolling(window=window).min()
                result[f'{column}_rolling_max_{window}'] = df[column].rolling(window=window).max()
        
        logger.info(f"Added rolling features for columns {columns} with windows {windows}")
        return result
    
    def add_return_features(self, df, columns, periods=[1, 3, 7, 14, 30]):
        """
        Add return features (percent change)
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame to add return features to
        columns : list
            List of column names to create returns for
        periods : list
            List of periods for returns
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added return features
        """
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Add return features for each column and period
        for column in columns:
            if column not in df.columns:
                logger.warning(f"Column {column} not found in DataFrame")
                continue
                
            for period in periods:
                result[f'{column}_return_{period}'] = df[column].pct_change(periods=period)
        
        logger.info(f"Added return features for columns {columns} with periods {periods}")
        return result
    
    def add_technical_indicators(self, df):
        """
        Add technical indicators as features
        
        This is a wrapper that calls the indicators module to add technical indicators
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with OHLCV data
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with added technical indicators
        """
        # Import here to avoid circular imports
        from indicators.basic import calculate_indicators
        
        # Calculate indicators
        result = calculate_indicators(df)
        
        logger.info("Added technical indicators as features")
        return result
    
    def scale_features(self, df, feature_columns, scaler_type='minmax'):
        """
        Scale features using specified scaler
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with features to scale
        feature_columns : list
            List of column names to scale
        scaler_type : str
            Type of scaler to use ('minmax' or 'standard')
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with scaled features
        """
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Initialize scaler
        if scaler_type.lower() == 'minmax':
            scaler = MinMaxScaler()
        elif scaler_type.lower() == 'standard':
            scaler = StandardScaler()
        else:
            logger.warning(f"Unknown scaler type: {scaler_type}. Using MinMaxScaler.")
            scaler = MinMaxScaler()
        
        # Filter columns that exist in the DataFrame
        valid_columns = [col for col in feature_columns if col in df.columns]
        if len(valid_columns) != len(feature_columns):
            missing_columns = set(feature_columns) - set(valid_columns)
            logger.warning(f"Some columns not found in DataFrame: {missing_columns}")
        
        if not valid_columns:
            logger.error("No valid columns to scale")
            return df
        
        # Store feature columns for later reference
        self.feature_columns = valid_columns
        
        # Fit and transform the data
        scaled_values = scaler.fit_transform(result[valid_columns])
        
        # Replace original values with scaled values
        for i, col in enumerate(valid_columns):
            result[col] = scaled_values[:, i]
            
        # Store scaler for inverse transformation later
        self.scalers['features'] = scaler
        
        logger.info(f"Scaled {len(valid_columns)} feature columns using {scaler_type}")
        return result
    
    def scale_target(self, df, target_column, scaler_type='minmax'):
        """
        Scale target variable using specified scaler
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with target to scale
        target_column : str
            Name of target column
        scaler_type : str
            Type of scaler to use ('minmax' or 'standard')
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame with scaled target
        """
        # Create a copy to avoid modifying the original
        result = df.copy()
        
        # Check if target column exists
        if target_column not in df.columns:
            logger.error(f"Target column {target_column} not found in DataFrame")
            return df
        
        # Store target column for later reference
        self.target_column = target_column
        
        # Initialize scaler
        if scaler_type.lower() == 'minmax':
            scaler = MinMaxScaler()
        elif scaler_type.lower() == 'standard':
            scaler = StandardScaler()
        else:
            logger.warning(f"Unknown scaler type: {scaler_type}. Using MinMaxScaler.")
            scaler = MinMaxScaler()
        
        # Fit and transform the data (reshape for scikit-learn)
        scaled_values = scaler.fit_transform(result[[target_column]])
        
        # Replace original values with scaled values
        result[target_column] = scaled_values
        
        # Store scaler for inverse transformation later
        self.scalers['target'] = scaler
        
        logger.info(f"Scaled target column {target_column} using {scaler_type}")
        return result
    
    def inverse_scale_target(self, scaled_values):
        """
        Inverse transform scaled target values
        
        Parameters:
        -----------
        scaled_values : numpy.ndarray
            Array of scaled target values
            
        Returns:
        --------
        numpy.ndarray
            Array of original-scale target values
        """
        if 'target' not in self.scalers:
            logger.error("No target scaler found. Cannot inverse transform.")
            return scaled_values
        
        # Reshape for inverse transform if needed
        if len(scaled_values.shape) == 1:
            scaled_values = scaled_values.reshape(-1, 1)
        
        # Inverse transform
        original_values = self.scalers['target'].inverse_transform(scaled_values)
        
        return original_values
    
    def train_val_test_split(self, df, train_size=0.7, val_size=0.15, test_size=0.15, shuffle=False):
        """
        Split data into training, validation, and test sets
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame to split
        train_size : float
            Proportion of data for training
        val_size : float
            Proportion of data for validation
        test_size : float
            Proportion of data for testing
        shuffle : bool
            Whether to shuffle data before splitting
            
        Returns:
        --------
        tuple
            Tuple of (train_df, val_df, test_df)
        """
        # Check that proportions sum to 1
        if not np.isclose(train_size + val_size + test_size, 1.0):
            logger.warning("Split proportions do not sum to 1. Normalizing.")
            total = train_size + val_size + test_size
            train_size /= total
            val_size /= total
            test_size /= total
        
        # For time series data, we usually want to keep the temporal order
        if shuffle:
            logger.warning("Shuffling time series data may impact performance")
            indices = np.random.permutation(len(df))
        else:
            indices = np.arange(len(df))
        
        # Calculate split points
        train_end = int(train_size * len(indices))
        val_end = train_end + int(val_size * len(indices))
        
        # Split indices
        train_indices = indices[:train_end]
        val_indices = indices[train_end:val_end]
        test_indices = indices[val_end:]
        
        # Store indices for reference
        self.train_indices = train_indices
        self.val_indices = val_indices
        self.test_indices = test_indices
        
        # Split DataFrame
        train_df = df.iloc[train_indices].copy()
        val_df = df.iloc[val_indices].copy()
        test_df = df.iloc[test_indices].copy()
        
        logger.info(f"Split data into train ({len(train_df)}), validation ({len(val_df)}), and test ({len(test_df)}) sets")
        return train_df, val_df, test_df
    
    def create_sequences(self, df, feature_columns, target_column, sequence_length):
        """
        Create sequences for time series modeling
        
        Parameters:
        -----------
        df : pandas.DataFrame
            DataFrame with features and target
        feature_columns : list
            List of feature column names
        target_column : str
            Name of target column
        sequence_length : int
            Length of sequences (lookback window)
            
        Returns:
        --------
        tuple
            Tuple of (X, y) where X is 3D array of shape (n_samples, sequence_length, n_features)
            and y is array of target values
        """
        # Store parameters for reference
        self.feature_columns = feature_columns
        self.target_column = target_column
        self.sequence_length = sequence_length
        
        # Filter columns that exist in the DataFrame
        valid_columns = [col for col in feature_columns if col in df.columns]
        if len(valid_columns) != len(feature_columns):
            missing_columns = set(feature_columns) - set(valid_columns)
            logger.warning(f"Some feature columns not found in DataFrame: {missing_columns}")
        
        if target_column not in df.columns:
            logger.error(f"Target column {target_column} not found in DataFrame")
            raise ValueError(f"Target column {target_column} not found in DataFrame")
        
        # Extract feature and target arrays
        features = df[valid_columns].values
        target = df[target_column].values
        
        # Create sequences
        X, y = [], []
        for i in range(len(df) - sequence_length):
            X.append(features[i:i+sequence_length])
            y.append(target[i+sequence_length])
        
        logger.info(f"Created {len(X)} sequences of length {sequence_length} with {len(valid_columns)} features")
        return np.array(X), np.array(y)
    
    def select_features(self, X, y, k=10):
        """
        Select the k best features using f_regression
        
        Parameters:
        -----------
        X : numpy.ndarray
            Feature matrix
        y : numpy.ndarray
            Target vector
        k : int
            Number of features to select
            
        Returns:
        --------
        numpy.ndarray
            Selected feature matrix
        """
        if k >= X.shape[1]:
            logger.warning(f"k ({k}) >= number of features ({X.shape[1]}). Returning all features.")
            return X
        
        # Create feature selector
        selector = SelectKBest(score_func=f_regression, k=k)
        
        # Fit and transform
        X_selected = selector.fit_transform(X, y)
        
        # Get selected feature indices
        selected_indices = selector.get_support(indices=True)
        
        # If we have feature names, log them
        if self.feature_columns is not None:
            selected_features = [self.feature_columns[i] for i in selected_indices]
            logger.info(f"Selected {k} best features: {selected_features}")
        else:
            logger.info(f"Selected {k} best features")
        
        return X_selected
