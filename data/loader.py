"""
Data Loader module for CryptoTradeAnalyzer

This module provides functionality to load data from various sources:
- CSV files
- APIs (like Binance, CoinGecko)
- Database connections

Author: CryptoTradeAnalyzer Team
"""

import pandas as pd
import numpy as np
import logging
import os
import sqlite3
from datetime import datetime

logger = logging.getLogger(__name__)

class DataLoader:
    """Data loading class that supports multiple data sources"""
    
    def __init__(self):
        """Initialize the data loader"""
        self.data = None
        self.source_type = None
        self.source_path = None
    
    def load_from_csv(self, file_path):
        """
        Load data from a CSV file
        
        Parameters:
        -----------
        file_path : str
            Path to the CSV file
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the loaded data
        """
        try:
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"CSV file not found: {file_path}")
                
            # Load the CSV file
            df = pd.read_csv(file_path)
            
            # Basic data validation and preprocessing
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            
            # Check if all required columns exist (case-insensitive)
            df_cols_lower = [col.lower() for col in df.columns]
            for req_col in required_columns:
                if req_col not in df_cols_lower and req_col.upper() not in df.columns:
                    # Try to infer columns or use defaults
                    if req_col == 'timestamp' and 'date' in df_cols_lower:
                        df = df.rename(columns={'date': 'timestamp'})
                    else:
                        raise ValueError(f"Required column '{req_col}' not found in CSV")
            
            # Standardize column names (convert to lowercase)
            df.columns = [col.lower() for col in df.columns]
            
            # Ensure timestamp is in datetime format
            if 'timestamp' in df.columns:
                if pd.api.types.is_numeric_dtype(df['timestamp']):
                    # Convert Unix timestamp to datetime
                    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
                else:
                    # Try to parse as datetime string
                    df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # Set timestamp as index
            df = df.set_index('timestamp')
            
            # Store source info
            self.data = df
            self.source_type = 'csv'
            self.source_path = file_path
            
            logger.info(f"Successfully loaded data from CSV: {file_path}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading CSV data: {str(e)}")
            raise
    
    def load_from_api(self, api_name, symbol, timeframe='1d', start_date=None, end_date=None, api_key=None):
        """
        Load data from a cryptocurrency API
        
        Parameters:
        -----------
        api_name : str
            Name of the API to use ('binance', 'coinbase', etc.)
        symbol : str
            Trading pair symbol (e.g., 'BTC/USDT')
        timeframe : str
            Candle timeframe ('1m', '5m', '1h', '1d', etc.)
        start_date : str
            Start date in 'YYYY-MM-DD' format
        end_date : str
            End date in 'YYYY-MM-DD' format
        api_key : str
            API key if required
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the loaded data
        """
        # This is a placeholder for API loading functionality
        # In a real implementation, we would connect to various APIs
        logger.warning(f"API data loading from {api_name} not fully implemented yet")
        
        # Mock response for demonstration
        raise NotImplementedError("API data loading is not implemented yet")
    
    def load_from_database(self, connection_string, query):
        """
        Load data from a database
        
        Parameters:
        -----------
        connection_string : str
            Database connection string or path
        query : str
            SQL query to execute
            
        Returns:
        --------
        pandas.DataFrame
            DataFrame containing the loaded data
        """
        try:
            # For SQLite
            if connection_string.endswith('.db'):
                conn = sqlite3.connect(connection_string)
                df = pd.read_sql_query(query, conn)
                conn.close()
            else:
                # Could be extended for other DB types
                raise ValueError("Only SQLite databases are currently supported")
            
            # Process and validate data similar to CSV loading
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            df_cols_lower = [col.lower() for col in df.columns]
            
            for req_col in required_columns:
                if req_col not in df_cols_lower:
                    raise ValueError(f"Required column '{req_col}' not found in database result")
            
            # Standardize column names
            df.columns = [col.lower() for col in df.columns]
            
            # Ensure timestamp is in datetime format
            if pd.api.types.is_numeric_dtype(df['timestamp']):
                df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            else:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
            
            # Sort by timestamp
            df = df.sort_values('timestamp')
            
            # Set timestamp as index
            df = df.set_index('timestamp')
            
            # Store source info
            self.data = df
            self.source_type = 'database'
            self.source_path = connection_string
            
            logger.info(f"Successfully loaded data from database: {connection_string}")
            return df
            
        except Exception as e:
            logger.error(f"Error loading database data: {str(e)}")
            raise
    
    def get_data(self):
        """
        Return the currently loaded data
        
        Returns:
        --------
        pandas.DataFrame
            The currently loaded data
        """
        if self.data is None:
            logger.warning("No data has been loaded yet")
        return self.data
    
    def save_to_csv(self, file_path):
        """
        Save the currently loaded data to a CSV file
        
        Parameters:
        -----------
        file_path : str
            Path to save the CSV file
        """
        if self.data is None:
            logger.error("No data to save")
            return False
        
        try:
            # Reset index to make timestamp a column
            self.data.reset_index().to_csv(file_path, index=False)
            logger.info(f"Data saved to CSV: {file_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving data to CSV: {str(e)}")
            return False
