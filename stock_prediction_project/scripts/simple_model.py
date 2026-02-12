"""
Simple Linear Regression Model (Alternative/Backup)
Faster training, good for quick testing and demonstrations
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt
import pickle
import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(__file__)))
from scripts.data_handler import StockDataHandler

class SimpleStockModel:
    def __init__(self):
        self.model = LinearRegression()
        self.scaler = MinMaxScaler()
        
    def prepare_features(self, data, lookback=5):
        """
        Create features using simple moving averages
        
        Args:
            data: Stock data DataFrame
            lookback: Number of days to look back
        """
        # Create lagged features
        features = pd.DataFrame()
        
        for i in range(1, lookback + 1):
            features[f'lag_{i}'] = data['Close'].shift(i)
        
        # Add moving averages
        features['MA_5'] = data['Close'].rolling(window=5).mean()
        features['MA_10'] = data['Close'].rolling(window=10).mean()
        features['MA_20'] = data['Close'].rolling(window=20).mean()
        
        # Add volume
        features['Volume'] = data['Volume']
        
        # Target
        features['Target'] = data['Close']
        
        # Drop NaN
        features = features.dropna()
        
        return features
    
    def train(self, ticker='AAPL', period='1y', lookback=5):
        """Train simple model"""
        print(f"Training simple model for {ticker}...")
        
        # Fetch data
        handler = StockDataHandler(ticker=ticker, period=period)
        data = handler.fetch_data()
        
        # Prepare features
        features = self.prepare_features(data, lookback=lookback)
        
        # Split features and target
        X = features.drop('Target', axis=1)
        y = features['Target']
        
        # Split train/test
        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Scale
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train
        self.model.fit(X_train_scaled, y_train)
        
        # Evaluate
        y_pred = self.model.predict(X_test_scaled)
        
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)
        
        print(f"\nModel Performance:")
        print(f"RMSE: ${rmse:.2f}")
        print(f"MAE: ${mae:.2f}")
        
        # Plot
        plt.figure(figsize=(12, 6))
        plt.plot(y_test.values, label='Actual', color='blue')
        plt.plot(y_pred, label='Predicted', color='red', linestyle='--')
        plt.title(f'{ticker} - Simple Model Predictions')
        plt.xlabel('Days')
        plt.ylabel('Price ($)')
        plt.legend()
        plt.grid(True)
        plt.savefig('models/simple_model_results.png')
        print("Plot saved to models/simple_model_results.png")
        
        return self.model, rmse, mae
    
    def predict_next_days(self, data, days=7):
        """Predict next N days"""
        predictions = []
        
        # Use last available data point
        last_features = self.prepare_features(data, lookback=5).iloc[-1:]
        last_features = last_features.drop('Target', axis=1)
        
        for _ in range(days):
            # Scale and predict
            X_scaled = self.scaler.transform(last_features)
            pred = self.model.predict(X_scaled)[0]
            predictions.append(pred)
            
            # Update features (simplified)
            # In real scenario, you'd update all lagged features properly
        
        return np.array(predictions)

if __name__ == "__main__":
    model = SimpleStockModel()
    model.train(ticker='AAPL', period='2y', lookback=10)