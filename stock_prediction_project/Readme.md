# 📈 Stock Market Price Prediction System

A complete Machine Learning project for predicting stock prices using LSTM (Long Short-Term Memory) neural networks.

## 🎯 Project Overview

This project uses deep learning to predict stock market prices based on historical data. It includes:

- Data collection from Yahoo Finance
- Feature engineering with technical indicators
- LSTM neural network model
- Interactive web interface
- Prediction visualization

## 🚀 Quick Start Guide

### Step 1: Install Python

Make sure you have Python 3.8+ installed. Check with:

```bash
python --version
```

### Step 2: Install Dependencies

Open terminal/command prompt and navigate to the project folder, then run:

```bash
pip install -r requirements.txt
```

This will install all required libraries:

- numpy, pandas (data manipulation)
- matplotlib, seaborn, plotly (visualization)
- scikit-learn (ML utilities)
- tensorflow (deep learning)
- yfinance (stock data)
- streamlit (web app)

### Step 3: Run the Web App

```bash
streamlit run app.py
```

This will open the application in your browser at `http://localhost:8501`

## 📱 Using the Application

### Training a Model:

1. Enter a stock ticker (e.g., AAPL for Apple, GOOGL for Google, TSLA for Tesla)
2. Select the data period (1 year, 2 years, etc.)
3. Set the lookback window (how many past days to consider)
4. Click "🚀 Train Model"
5. Wait for training to complete (2-5 minutes)

### Making Predictions:

1. After training, select how many days to predict (1-30)
2. Click "🔮 Make Predictions"
3. View the forecast table and chart

## 🛠️ Advanced Usage

### Training Model via Script

You can train the model directly without the web interface:

```bash
cd scripts
python train_model.py
```

This will:

- Fetch AAPL stock data for 2 years
- Train an LSTM model
- Save the model to `models/lstm_model.h5`
- Generate performance plots

### Customizing the Model

Edit `scripts/train_model.py` to change:

- Stock ticker
- Training period
- Number of epochs
- Lookback window

```python
model, handler, metrics = train_complete_model(
    ticker='TSLA',      # Change stock
    period='5y',        # Change period
    lookback=90,        # Change lookback
    epochs=100          # More epochs = better accuracy (but slower)
)
```

## 📊 Project Structure

```
stock_prediction_project/
│
├── app.py                          # Streamlit web application
├── requirements.txt                # Python dependencies
│
├── scripts/
│   ├── data_handler.py            # Data fetching and preprocessing
│   └── train_model.py             # Model training script
│
├── models/                         # Saved models (created after training)
│   ├── lstm_model.h5              # Trained LSTM model
│   ├── scaler.pkl                 # Data scaler
│   ├── metadata.pkl               # Training metadata
│   ├── training_history.png       # Training loss plot
│   └── prediction_results.png     # Prediction vs actual plot
│
└── data/                          # Data files (optional)
```

## 🧠 How It Works

### 1. Data Collection

- Fetches historical stock data from Yahoo Finance
- Includes: Open, High, Low, Close, Volume

### 2. Feature Engineering

- **Moving Averages (MA)**: 5-day, 20-day, 50-day trends
- **RSI (Relative Strength Index)**: Momentum indicator
- **Volatility**: Price variation measure

### 3. Data Preprocessing

- Normalizes prices to 0-1 range using MinMaxScaler
- Creates sequences (e.g., use 60 days to predict day 61)

### 4. LSTM Model Architecture

```
Layer 1: LSTM (50 units) + Dropout (0.2)
Layer 2: LSTM (50 units) + Dropout (0.2)
Layer 3: LSTM (50 units) + Dropout (0.2)
Layer 4: Dense (1 unit) - Output
```

### 5. Training

- Splits data: 80% training, 20% testing
- Uses Adam optimizer
- Monitors validation loss with early stopping

### 6. Prediction

- Uses last N days to predict next day
- Iteratively predicts multiple days ahead
- Converts predictions back to actual price range

## 📈 Performance Metrics

The model is evaluated using:

- **RMSE** (Root Mean Squared Error): Average prediction error in dollars
- **MAE** (Mean Absolute Error): Average absolute error
- **MAPE** (Mean Absolute Percentage Error): Error as percentage

Lower values = better accuracy

## ⚙️ Troubleshooting

### Issue: "No module named 'tensorflow'"

**Solution:** Install TensorFlow

```bash
pip install tensorflow
```

### Issue: "No data found for ticker"

**Solution:** Check if ticker symbol is valid on Yahoo Finance

### Issue: Training is very slow

**Solution:** Reduce epochs or use GPU if available

```python
# In train_model.py, change:
epochs=20  # Instead of 50
```

### Issue: Port 8501 already in use

**Solution:** Use different port

```bash
streamlit run app.py --server.port 8502
```

## 🎓 Understanding the Code

### Key Files Explained:

**data_handler.py**

- `fetch_data()`: Downloads stock data
- `add_technical_indicators()`: Calculates MA, RSI, etc.
- `prepare_data_for_lstm()`: Creates training sequences

**train_model.py**

- `build_model()`: Creates LSTM architecture
- `train()`: Trains the model
- `predict()`: Makes predictions
- `evaluate()`: Calculates accuracy metrics

**app.py**

- Streamlit web interface
- Interactive controls
- Real-time predictions
- Visualization

## 🔄 Improving Accuracy

To get better predictions:

1. **Use more data**: Change period to '5y' or 'max'
2. **Increase lookback**: Try 90 or 120 days instead of 60
3. **Train longer**: Increase epochs to 100+
4. **Add more features**: Include news sentiment, market indices
5. **Try different stocks**: Some stocks are more predictable than others

## ⚠️ Important Notes

1. **Educational Purpose**: This is a learning project, not financial advice
2. **Market Unpredictability**: Stock markets are influenced by many factors beyond historical prices
3. **No Guarantees**: Past performance doesn't guarantee future results
4. **Use Responsibly**: Always do your own research before making investment decisions

## 🎯 Project Presentation Tips

For your college presentation, highlight:

1. **Problem Statement**: Predicting stock prices is challenging but valuable
2. **Solution**: Deep learning (LSTM) can capture temporal patterns
3. **Implementation**: Show the web app, train a model live
4. **Results**: Display accuracy metrics and prediction charts
5. **Learnings**: Discuss challenges (data quality, overfitting, etc.)
6. **Future Scope**: Sentiment analysis, multiple stocks, trading strategy

## 📚 Learning Resources

- **LSTM Tutorial**: https://colah.github.io/posts/2015-08-Understanding-LSTMs/
- **Stock Market Basics**: https://www.investopedia.com/
- **TensorFlow Docs**: https://www.tensorflow.org/
- **Streamlit Docs**: https://docs.streamlit.io/

## 🤝 Common Stock Tickers to Try

- **Tech**: AAPL (Apple), GOOGL (Google), MSFT (Microsoft), TSLA (Tesla)
- **Finance**: JPM (JPMorgan), GS (Goldman Sachs), V (Visa)
- **Retail**: AMZN (Amazon), WMT (Walmart)
- **Crypto**: BTC-USD (Bitcoin), ETH-USD (Ethereum)

## 💡 Tips for Success

1. Start with 2-year data and 60-day lookback
2. Train on reliable stocks like AAPL first
3. Don't predict too far ahead (7-14 days max)
4. Compare predictions with actual prices next week
5. Document everything for your project report

## 🎉 You're Ready!

Run `streamlit run app.py` and start predicting! Good luck with your project! 🚀
