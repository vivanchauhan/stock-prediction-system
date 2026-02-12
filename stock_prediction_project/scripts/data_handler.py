"""
Stock Data Collection and Preprocessing
This script fetches historical stock data and prepares it for ML models
"""

import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from sklearn.preprocessing import MinMaxScaler
import pickle
import os

class StockDataHandler:
    def __init__(self, ticker='AAPL', period='2y'):
        """
        Initialize stock data handler
        
        Args:
            ticker: Stock symbol (e.g., 'AAPL', 'GOOGL', 'TSLA')
            period: Data period ('1y', '2y', '5y', 'max')
        """
        self.ticker = ticker
        self.period = period
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        self.data = None
        self.scaled_data = None
        
    def fetch_data(self):
        """Fetch stock data from Yahoo Finance"""
        print(f"Fetching data for {self.ticker}...")
        stock = yf.Ticker(self.ticker)
        self.data = stock.history(period=self.period)
        
        if self.data.empty:
            raise ValueError(f"No data found for ticker {self.ticker}")
        
        print(f"Fetched {len(self.data)} days of data")
        return self.data
    
    def add_technical_indicators(self):
        """Add technical indicators as features"""
        # Moving Averages
        self.data['MA_5'] = self.data['Close'].rolling(window=5).mean()
        self.data['MA_20'] = self.data['Close'].rolling(window=20).mean()
        self.data['MA_50'] = self.data['Close'].rolling(window=50).mean()
        
        # Relative Strength Index (RSI)
        delta = self.data['Close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        self.data['RSI'] = 100 - (100 / (1 + rs))
        
        # Volatility
        self.data['Volatility'] = self.data['Close'].rolling(window=20).std()
        
        # Drop NaN values created by rolling calculations
        self.data = self.data.dropna()
        
        return self.data
    
    def prepare_data_for_lstm(self, feature_col='Close', lookback=60):
        """
        Prepare data for LSTM model
        
        Args:
            feature_col: Column to use for prediction
            lookback: Number of previous days to use for prediction
        """
        # Scale the data
        data_values = self.data[feature_col].values.reshape(-1, 1)
        self.scaled_data = self.scaler.fit_transform(data_values)
        
        # Create sequences
        X, y = [], []
        for i in range(lookback, len(self.scaled_data)):
            X.append(self.scaled_data[i-lookback:i, 0])
            y.append(self.scaled_data[i, 0])
        
        X, y = np.array(X), np.array(y)
        
        # Reshape for LSTM [samples, time steps, features]
        X = np.reshape(X, (X.shape[0], X.shape[1], 1))
        
        return X, y
    
    def save_scaler(self, filepath='models/scaler.pkl'):
        """Save the scaler for later use"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'wb') as f:
            pickle.dump(self.scaler, f)
        print(f"Scaler saved to {filepath}")
    
    def load_scaler(self, filepath='models/scaler.pkl'):
        """Load a saved scaler"""
        with open(filepath, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f"Scaler loaded from {filepath}")

if __name__ == "__main__":
    # Example usage
    handler = StockDataHandler(ticker='AAPL', period='2y')
    data = handler.fetch_data()
    data = handler.add_technical_indicators()
    
    print("\nData Summary:")
    print(data.tail())
    print(f"\nShape: {data.shape}")
    print(f"\nColumns: {data.columns.tolist()}")
    
    # Prepare data for LSTM
    X, y = handler.prepare_data_for_lstm(lookback=60)
    print(f"\nLSTM Input shape: {X.shape}")
    print(f"LSTM Output shape: {y.shape}")
    
    # Save scaler
    handler.save_scaler()