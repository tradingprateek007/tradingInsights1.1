import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
import plotly.graph_objects as go


# Fetch price data
def fetch_history(ticker, period="6mo"):
    return yf.Ticker(ticker).history(period=period)["Close"]


# ARIMA forecast
def forecast_arima(prices, periods=20):
    model = ARIMA(prices[:-periods], order=(5, 1, 0))
    fit = model.fit()
    forecast = fit.forecast(steps=periods)
    forecast_index = pd.bdate_range(prices.index[-periods], periods=periods)
    return pd.Series(data=forecast.values, index=forecast_index)


# LSTM forecast (patched)
def forecast_lstm(prices, periods=20, look_back=10):
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(prices[:-periods].values.reshape(-1, 1))

    X, y = [], []
    for i in range(len(scaled) - look_back):
        X.append(scaled[i:i + look_back])
        y.append(scaled[i + look_back])

    X, y = np.array(X), np.array(y)

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, verbose=0)

    input_seq = scaled[-look_back:].reshape(1, look_back, 1)
    forecasts = []

    for _ in range(periods):
        pred = model.predict(input_seq, verbose=0)
        forecasts.append(pred[0][0])
        pred_reshaped = pred.reshape(1, 1, 1)
        input_seq = np.concatenate((input_seq[:, 1:, :], pred_reshaped), axis=1)

    forecast = scaler.inverse_transform(np.array(forecasts).reshape(-1, 1)).flatten()
    forecast_index = pd.bdate_range(prices.index[-periods], periods=periods)
    return pd.Series(data=forecast, index=forecast_index)


# Streamlit Backtest UI
def past_forecast(ticker):
    st.title("Backtest: ARIMA vs LSTM vs Actual")

    prices = fetch_history(ticker)
    if prices.empty or len(prices) < 40:
        st.warning("Not enough price data.")
        return

    actual = prices[-60:]
    arima_f = forecast_arima(prices, periods=60)
    lstm_f = forecast_lstm(prices, periods=60)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=actual.index, y=actual.values, name="Actual Price", line=dict(color='green')))
    fig.add_trace(go.Scatter(x=arima_f.index, y=arima_f.values, name="ARIMA_Prediction", line=dict(dash='dot', color='blue')))
    fig.add_trace(go.Scatter(x=lstm_f.index, y=lstm_f.values, name="LSTM_Prediction", line=dict(dash='dot', color='orange')))
    fig.update_layout(title=f"{ticker} Forecast vs Actual (Last 4 Weeks)",
                      xaxis_title="Date", yaxis_title="Price")

    st.plotly_chart(fig, use_container_width=True)

    # Metrics
    arima_rmse = np.sqrt(mean_squared_error(actual.values, arima_f.values))
    lstm_rmse = np.sqrt(mean_squared_error(actual.values, lstm_f.values))
    arima_pct_error = np.mean(np.abs((actual.values - arima_f.values) / actual.values)) * 100
    lstm_pct_error = np.mean(np.abs((actual.values - lstm_f.values) / actual.values)) * 100

    st.subheader("Model Accuracy Metrics (Last 4 Weeks)")
    st.markdown(f"**ARIMA RMSE**: {arima_rmse:.2f}")
    st.markdown(f"**LSTM RMSE**: {lstm_rmse:.2f}")
    st.markdown(f"**ARIMA Avg % Error**: {arima_pct_error:.2f}%")
    st.markdown(f"**LSTM Avg % Error**: {lstm_pct_error:.2f}%")