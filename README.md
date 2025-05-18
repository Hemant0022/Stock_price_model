# 📊 Indian Stock Price Prediction Web App

This project is a Streamlit-based web application that allows users to view historical and current stock data for Indian companies, along with predicted future prices using **LSTM**, **ARIMA**, and **ensemble models**.

# 🚀 Features

- 📈 Displays key financial indicators: Open, High, Low, Close, Volume, Market Cap, P/E Ratio, EPS, 52-week High/Low
- 📊 Interactive charts with monthly breakdowns of OHLC prices
- 🔮 Predicts future stock prices using:
  - LSTM (Long Short-Term Memory Neural Network)
  - ARIMA (AutoRegressive Integrated Moving Average)
  - Ensemble forecast combining both models
- 🏦 Supports 20+ major Indian stock symbols
- 🧠 Fully integrated deep learning (Keras) and statistical (statsmodels) forecasting

# 🏗️ Tech Stack

- [Streamlit](https://streamlit.io/)
- [YFinance](https://github.com/ranaroussi/yfinance)
- [Pandas](https://pandas.pydata.org/)
- [NumPy](https://numpy.org/)
- [Matplotlib](https://matplotlib.org/)
- [scikit-learn](https://scikit-learn.org/)
- [Keras](https://keras.io/)
- [Statsmodels](https://www.statsmodels.org/)

---

--Requirements.txt:
streamlit
yfinance
pandas
numpy
scikit-learn
tensorflow
matplotlib
statsmodels


# ⚙️ Installation

1. Clone the repository:

   ```terminal
   git clone https://github.com/Hemant0022/Stock_price_model.git
   cd Stock_price_model
   
2. Installing the dependencies
   ```terminal
   pip install -r requirements.txt

4. Run with:
   ```terminal
   streamlit run app.py

6. Maintainance (Upgrading the dependencies result in better enhancements of the code):
   ```terminal
    pip install --upgrade -r requirements.txt
   
## Note: System must be connected to the internet for fetching stock data from yfinance.
