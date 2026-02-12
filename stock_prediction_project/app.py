"""
Stock Price Prediction Web App (Simplified - No TensorFlow Required)
Uses scikit-learn models for faster installation and compatibility
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, mean_absolute_error
import warnings
warnings.filterwarnings('ignore')

# Page configuration
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

# Title and description
st.title("📈 Stock Market Price Prediction")
st.markdown("### AI-Powered Stock Price Forecasting using Machine Learning")

# Sidebar for user inputs
st.sidebar.header("Configuration")

# Stock selection
ticker = st.sidebar.text_input("Enter Stock Ticker", value="AAPL", help="e.g., AAPL, GOOGL, TSLA, MSFT")
ticker = ticker.upper()

# Period selection
period_options = {
    "1 Year": "1y",
    "2 Years": "2y",
    "5 Years": "5y",
    "Max": "max"
}
period_label = st.sidebar.selectbox("Select Data Period", list(period_options.keys()), index=1)
period = period_options[period_label]

# Model selection
model_options = {
    "Random Forest": "rf",
    "Gradient Boosting": "gb",
    "Ridge Regression": "ridge"
}
model_label = st.sidebar.selectbox("Select Model", list(model_options.keys()), index=0)
model_type = model_options[model_label]

# Prediction days
prediction_days = st.sidebar.slider("Days to Predict", min_value=1, max_value=30, value=7)

# Lookback window
lookback = st.sidebar.slider("Lookback Window (days)", min_value=30, max_value=120, value=60)

# Train button
train_button = st.sidebar.button("🚀 Train Model", type="primary")

# Predict button
predict_button = st.sidebar.button("🔮 Make Predictions")

# Session state for model
if 'model' not in st.session_state:
    st.session_state.model = None
if 'scaler' not in st.session_state:
    st.session_state.scaler = None
if 'data' not in st.session_state:
    st.session_state.data = None
if 'trained_ticker' not in st.session_state:
    st.session_state.trained_ticker = None

def fetch_stock_data(ticker, period):
    """Fetch stock data from Yahoo Finance"""
    stock = yf.Ticker(ticker)
    data = stock.history(period=period)
    return data

def add_technical_indicators(data):
    """Add technical indicators as features"""
    df = data.copy()
    
    # Moving Averages
    df['MA_5'] = df['Close'].rolling(window=5).mean()
    df['MA_20'] = df['Close'].rolling(window=20).mean()
    df['MA_50'] = df['Close'].rolling(window=50).mean()
    
    # RSI
    delta = df['Close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # Volatility
    df['Volatility'] = df['Close'].rolling(window=20).std()
    
    # Price change
    df['Price_Change'] = df['Close'].pct_change()
    
    # Volume change
    df['Volume_Change'] = df['Volume'].pct_change()
    
    df = df.dropna()
    return df

def prepare_features(data, lookback=60):
    """Prepare features for ML model"""
    features = []
    targets = []
    
    close_prices = data['Close'].values
    
    for i in range(lookback, len(close_prices)):
        # Use last N days as features
        features.append(close_prices[i-lookback:i])
        targets.append(close_prices[i])
    
    return np.array(features), np.array(targets)

def get_model(model_type):
    """Get the selected model"""
    if model_type == "rf":
        return RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    elif model_type == "gb":
        return GradientBoostingRegressor(n_estimators=100, random_state=42)
    else:
        return Ridge(alpha=1.0)

# Training section
if train_button:
    with st.spinner(f"Training model for {ticker}... Please wait."):
        try:
            # Fetch data
            data = fetch_stock_data(ticker, period)
            if data.empty:
                st.error(f"❌ No data found for ticker {ticker}")
            else:
                st.success(f"✅ Fetched {len(data)} days of historical data")
                
                # Add technical indicators
                data = add_technical_indicators(data)
                
                # Prepare features
                X, y = prepare_features(data, lookback=lookback)
                
                # Split data
                split_idx = int(len(X) * 0.8)
                X_train, X_test = X[:split_idx], X[split_idx:]
                y_train, y_test = y[:split_idx], y[split_idx:]
                
                # Train model
                progress_bar = st.progress(0)
                model = get_model(model_type)
                
                progress_bar.progress(30)
                model.fit(X_train, y_train)
                progress_bar.progress(100)
                
                # Evaluate
                y_pred = model.predict(X_test)
                
                mse = mean_squared_error(y_test, y_pred)
                rmse = np.sqrt(mse)
                mae = mean_absolute_error(y_test, y_pred)
                
                # Store in session state
                st.session_state.model = model
                st.session_state.data = data
                st.session_state.trained_ticker = ticker
                
                # Display results
                st.success("✅ Model training completed!")
                
                # Show metrics
                metric_col1, metric_col2, metric_col3 = st.columns(3)
                with metric_col1:
                    st.metric("RMSE", f"${rmse:.2f}")
                with metric_col2:
                    st.metric("MAE", f"${mae:.2f}")
                with metric_col3:
                    accuracy = 100 - (mae / y_test.mean() * 100)
                    st.metric("Accuracy", f"{accuracy:.1f}%")
                
                # Plot predictions vs actual
                fig = go.Figure()
                fig.add_trace(go.Scatter(
                    y=y_test,
                    mode='lines',
                    name='Actual Price',
                    line=dict(color='blue', width=2)
                ))
                fig.add_trace(go.Scatter(
                    y=y_pred,
                    mode='lines',
                    name='Predicted Price',
                    line=dict(color='red', width=2, dash='dash')
                ))
                fig.update_layout(
                    title=f"{ticker} - Model Validation Results",
                    xaxis_title="Days",
                    yaxis_title="Price ($)",
                    height=500,
                    hovermode='x unified'
                )
                st.plotly_chart(fig, use_container_width=True)
                
        except Exception as e:
            st.error(f"❌ Error during training: {str(e)}")
            import traceback
            st.code(traceback.format_exc())

# Prediction section
if predict_button:
    if st.session_state.model is None or st.session_state.trained_ticker != ticker:
        st.warning("⚠️ Please train the model first!")
    else:
        with st.spinner("Generating predictions..."):
            try:
                model = st.session_state.model
                data = st.session_state.data
                
                # Get latest data
                latest_prices = data['Close'].values[-lookback:]
                
                # Predict future prices
                predictions = []
                current_sequence = latest_prices.copy()
                
                for _ in range(prediction_days):
                    # Prepare input
                    X_pred = current_sequence[-lookback:].reshape(1, -1)
                    
                    # Predict next day
                    next_pred = model.predict(X_pred)[0]
                    predictions.append(next_pred)
                    
                    # Update sequence
                    current_sequence = np.append(current_sequence, next_pred)
                
                # Create future dates
                last_date = data.index[-1]
                future_dates = pd.date_range(
                    start=last_date + timedelta(days=1),
                    periods=prediction_days,
                    freq='D'
                )
                
                # Display predictions
                st.success(f"✅ Generated {prediction_days}-day forecast")
                
                # Show prediction table
                pred_df = pd.DataFrame({
                    'Date': future_dates,
                    'Predicted Price': [f"${p:.2f}" for p in predictions]
                })
                pred_df['Date'] = pred_df['Date'].dt.strftime('%Y-%m-%d')
                
                st.subheader("📊 Prediction Results")
                st.dataframe(pred_df, use_container_width=True)
                
                # Calculate price change
                current_price = data['Close'].iloc[-1]
                final_price = predictions[-1]
                price_change = final_price - current_price
                percent_change = (price_change / current_price) * 100
                
                # Show summary metrics
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Current Price", f"${current_price:.2f}")
                with col2:
                    st.metric(f"Price in {prediction_days} Days", f"${final_price:.2f}")
                with col3:
                    st.metric("Expected Change", f"${price_change:.2f}", f"{percent_change:+.2f}%")
                with col4:
                    trend = "📈 Bullish" if price_change > 0 else "📉 Bearish"
                    st.metric("Trend", trend)
                
                # Plot historical + predictions
                fig = go.Figure()
                
                # Historical data (last 90 days)
                historical_dates = data.index[-90:]
                historical_prices = data['Close'].values[-90:]
                
                fig.add_trace(go.Scatter(
                    x=historical_dates,
                    y=historical_prices,
                    mode='lines',
                    name='Historical Price',
                    line=dict(color='blue', width=2)
                ))
                
                # Predictions
                fig.add_trace(go.Scatter(
                    x=future_dates,
                    y=predictions,
                    mode='lines+markers',
                    name='Predicted Price',
                    line=dict(color='red', width=2, dash='dash'),
                    marker=dict(size=8)
                ))
                
                fig.update_layout(
                    title=f"{ticker} - Price Forecast ({prediction_days} Days)",
                    xaxis_title="Date",
                    yaxis_title="Price ($)",
                    height=600,
                    hovermode='x unified'
                )
                
                st.plotly_chart(fig, use_container_width=True)
                
            except Exception as e:
                st.error(f"❌ Error during prediction: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

# Information section
with st.expander("ℹ️ About This Application"):
    st.markdown("""
    ### How It Works
    
    This application uses **Machine Learning** models to predict stock prices.
    
    **Available Models:**
    - **Random Forest**: Ensemble of decision trees (recommended)
    - **Gradient Boosting**: Sequential ensemble method
    - **Ridge Regression**: Linear model with regularization
    
    **Features:**
    - Real-time data from Yahoo Finance
    - Technical indicators (Moving Averages, RSI, Volatility)
    - Multiple ML algorithms
    - Interactive visualizations
    
    **How to Use:**
    1. Enter a stock ticker (e.g., AAPL, GOOGL, TSLA)
    2. Select model and parameters
    3. Click "Train Model" to train on historical data
    4. Click "Make Predictions" to forecast future prices
    
    **Note:** This is for educational purposes. Not financial advice!
    """)

# Footer
st.sidebar.markdown("---")
st.sidebar.markdown("### 📚 Project Info")
st.sidebar.info(f"Stock Market Prediction System\nBuilt with Python & Scikit-learn\nModel: {model_label}")