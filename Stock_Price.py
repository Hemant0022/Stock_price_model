# These are the libraries and frameworks that to be installed at the intialization
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import matplotlib.pyplot as plt
from datetime import datetime
from statsmodels.tsa.arima.model import ARIMA

# Function to fetch stock data using yfinance
def fetch_stock_data(symbol):
    stock = yf.Ticker(symbol)
    hist = stock.history(period="13mo", auto_adjust=True)
    return stock, hist

# Function to preprocess stock data
def preprocess_stock_data(data):
    df = data[['Open', 'High', 'Low', 'Close', 'Volume']]
    df.index = pd.to_datetime(df.index)
    return df

# Function to build and train the LSTM model
def build_lstm_model(input_shape):
    model = Sequential()
    model.add(LSTM(units=50, return_sequences=True, input_shape=(input_shape[1], 1)))
    model.add(LSTM(units=50))
    model.add(Dense(units=1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

# Function to train the ARIMA model
def train_arima_model(train_data):
    model = ARIMA(train_data, order=(5, 1, 0))
    model_fit = model.fit()
    return model_fit

# Streamlit App
st.set_page_config(page_title="Indian Stock Price Prediction", layout="wide")
st.title("Indian Stock Price Prediction")

# Sidebar for Indian Stock Symbols
st.sidebar.title("Indian Stock Symbols")
indian_symbols = {
    'Reliance Industries': 'RELIANCE.NS',
    'Tata Consultancy Services': 'TCS.NS',
    'HDFC Bank': 'HDFCBANK.NS',
    'ICICI Bank': 'ICICIBANK.NS',
    'Infosys': 'INFY.NS',
    'Bharti Airtel': 'AIRTEL.NS',
    'Kotak Mahindra Bank': 'KOTAKBANK.NS',
    'Larsen & Toubro': 'LT.NS',
    'State Bank of India': 'SBIN.NS',
    'Maruti Suzuki': 'MARUTI.NS',
    'Hindustan Unilever': 'HINDUNILVR.NS',
    'Axis Bank': 'AXISBANK.NS',
    'Wipro': 'WIPRO.NS',
    'Mahindra & Mahindra': 'M&M.NS',
    'Bajaj Finance': 'BAJFINANCE.NS',
    'Dr. Reddy’s Laboratories': 'DRREDDY.NS',
    'Sun Pharmaceutical': 'SUNPHARMA.NS',
    'Titan Company': 'TITAN.NS',
    'Nestle India': 'NESTLEIND.NS',
}

# Display the stock symbols in the sidebar
for company, symbol in indian_symbols.items():
    st.sidebar.write(f"{company}: {symbol}")

# User input for stock symbol
symbol_input = st.text_input("Enter Stock Symbol (or select from the sidebar):", "")

# Button to show data
if st.button("Show Stock Data"):
    if symbol_input:
        stock, data = fetch_stock_data(symbol_input)
        info = stock.info

        if not data.empty:
            df = preprocess_stock_data(data)

            # Display historical data
            st.subheader(f"Stock Data for {symbol_input}")
            st.write(df.head(10))

            # Display key financial info
            st.subheader(f"Key Info for {symbol_input}")
            st.markdown(f"""
            - 📈 **Current Price**: ₹{info.get("currentPrice", "N/A")}
            - 🕘 **Open**: ₹{info.get("open", "N/A")}
            - 🔼 **Day High**: ₹{info.get("dayHigh", "N/A")}
            - 🔽 **Day Low**: ₹{info.get("dayLow", "N/A")}
            - 💹 **52-Week High**: ₹{info.get("fiftyTwoWeekHigh", "N/A")}
            - 📉 **52-Week Low**: ₹{info.get("fiftyTwoWeekLow", "N/A")}
            - 💼 **Market Cap**: ₹{info.get("marketCap", "N/A"):,}
            - 📊 **P/E Ratio**: {info.get("trailingPE", "N/A")}
            - 🧾 **EPS**: ₹{info.get("trailingEps", "N/A")}
            - 🏢 **Sector**: {info.get("sector", "N/A")}
            - 🏭 **Industry**: {info.get("industry", "N/A")}
            """)

            # Display current closing price
            latest_price = df['Close'].iloc[-1]
            st.metric(label="Current Closing Price", value=f"₹{latest_price:.2f}")

            # Visualize OHLC data
            st.subheader(f"Stock Price (OHLC) for {symbol_input}")
            fig, ax = plt.subplots(figsize=(12, 6))
            df[['Open', 'High', 'Low', 'Close']].plot(ax=ax)
            ax.set_title(f"{symbol_input} - OHLC Price")
            ax.set_xlabel('Date')
            ax.set_ylabel('Price (INR)')
            plt.xticks(rotation=45)
            plt.grid(True)
            st.pyplot(fig)

            # Monthly breakdown
            df['Month'] = df.index.month
            df['Year'] = df.index.year
            current_year = datetime.now().year
            for month in range(1, 13):
                month_data = df[(df['Month'] == month) & (df['Year'] == current_year)]
                if not month_data.empty:
                    month_name = datetime(2000, month, 1).strftime('%B')
                    col1, col2 = st.columns([1, 2])
                    with col1:
                        st.subheader(f"{month_name} {current_year} Stock Prices")
                        st.write(month_data[['Open', 'High', 'Low', 'Close']])
                    with col2:
                        fig, ax = plt.subplots(figsize=(10, 5))
                        month_data[['Open', 'High', 'Low', 'Close']].plot(ax=ax)
                        ax.set_title(f"{symbol_input} - {month_name} Price Action")
                        ax.set_xlabel('Date')
                        ax.set_ylabel('Price (INR)')
                        plt.xticks(rotation=45)
                        plt.grid(True)
                        st.pyplot(fig)

            # LSTM data preparation
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaled_data = scaler.fit_transform(df[['Close']].values)
            X_train, y_train = [], []
            for i in range(60, len(scaled_data)):
                X_train.append(scaled_data[i - 60:i, 0])
                y_train.append(scaled_data[i, 0])
            X_train, y_train = np.array(X_train), np.array(y_train)
            X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))

            # Train LSTM
            lstm_model = build_lstm_model(X_train.shape)
            lstm_model.fit(X_train, y_train, epochs=50, batch_size=32, verbose=0)

            # Train ARIMA
            arima_model = train_arima_model(df['Close'])

            # Prediction
            inputs = scaled_data[-60:]
            inputs = np.reshape(inputs, (1, inputs.shape[0], 1))
            predicted_price_lstm = lstm_model.predict(inputs)
            predicted_price_lstm = scaler.inverse_transform(predicted_price_lstm)[0][0]
            arima_forecast = arima_model.forecast(steps=1)
            arima_predicted_price = arima_forecast.iloc[0]
            ensemble_prediction = 0.6 * predicted_price_lstm + 0.4 * arima_predicted_price

            # Show predictions
            st.subheader(f"Predicted Stock Price for {symbol_input}")
            st.write(f"LSTM Predicted Price: ₹{predicted_price_lstm:.2f}")
            st.write(f"ARIMA Predicted Price: ₹{arima_predicted_price:.2f}")
            st.write(f"Ensemble Predicted Price: ₹{ensemble_prediction:.2f}")

        else:
            st.error("Error fetching data. Please check the stock symbol.")
    else:
        st.warning("Please enter a stock symbol.")
