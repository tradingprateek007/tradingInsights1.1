import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import plotly.graph_objects as go


# Fetch price data
def fetch_history(ticker, period="5y"):
    return yf.Ticker(ticker).history(period=period)["Close"]


# ARIMA forecast
def forecast_arima(prices, periods=20):
    model = ARIMA(prices, order=(5, 1, 0))
    fit = model.fit()
    forecast = fit.forecast(steps=periods)
    forecast_index = pd.bdate_range(prices.index[-1] + pd.Timedelta(days=1), periods=periods)
    return pd.Series(data=forecast.values, index=forecast_index)


def forecast_lstm(prices, periods=20, look_back=10):
    from keras.models import Sequential
    from keras.layers import LSTM, Dense
    from sklearn.preprocessing import MinMaxScaler
    import numpy as np
    import pandas as pd

    if len(prices) < (look_back + periods + 1):
        raise ValueError("Not enough data for LSTM forecast.")

    df = pd.DataFrame({'Close': prices})
    df.dropna(inplace=True)

    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df)

    X, y = [], []
    for i in range(len(scaled) - look_back - periods):
        X.append(scaled[i:i + look_back])
        y.append(scaled[i + look_back])

    X = np.array(X)
    y = np.array(y)

    if X.shape[0] == 0:
        raise ValueError("No valid training sequences.")

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(look_back, 1)))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, verbose=0)

    input_seq = scaled[-look_back:].reshape(1, look_back, 1)
    preds = []

    for _ in range(periods):
        next_pred = model.predict(input_seq, verbose=0)
        next_pred_reshaped = next_pred.reshape(1, 1, 1)
        input_seq = np.concatenate((input_seq[:, 1:, :], next_pred_reshaped), axis=1)
        preds.append(next_pred[0][0])

    forecast = scaler.inverse_transform(np.array(preds).reshape(-1, 1)).flatten()
    forecast_index = pd.bdate_range(prices.index[-1] + pd.Timedelta(days=1), periods=periods)
    return pd.Series(data=forecast, index=forecast_index)

# Options-implied target range (expected move)
def iv_based_target(current_price, iv=0.3, days=20):
    daily_vol = iv / np.sqrt(252)
    expected_move = current_price * daily_vol * np.sqrt(days)
    return current_price - expected_move, current_price + expected_move


# MACD calculation
def compute_macd(prices):
    ema12 = prices.ewm(span=12, adjust=False).mean()
    ema26 = prices.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    return macd, signal


# Main Forecast Tab
def future_forecast(ticker):
    st.title("Forecast Models: ARIMA, LSTM, IV-Based")

    prices = yf.Ticker(ticker).history(period="6mo")["Close"]
    if prices.empty:
        st.warning("No price data found.")
        return

    macd, signal = compute_macd(prices)
    current_price = prices.iloc[-1]
    arima_f = forecast_arima(prices)
    lstm_f = forecast_lstm(prices)
    iv_lower, iv_upper = iv_based_target(current_price)

    # Plot
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices.index, y=prices.values, name="Historical", line=dict(color='white')))
    fig.add_trace(go.Scatter(x=arima_f.index, y=arima_f.values, name="ARIMA Forecast", line=dict(color='blue')))
    fig.add_trace(go.Scatter(x=lstm_f.index, y=lstm_f.values, name="LSTM Forecast", line=dict(color='green')))
    fig.add_trace(go.Scatter(x=macd.index, y=macd.values, name="MACD", line=dict(color='orange', dash='dot')))
    fig.add_trace(go.Scatter(x=signal.index, y=signal.values, name="Signal Line", line=dict(color='red', dash='dot')))

    fig.add_hline(y=iv_upper, line=dict(dash='dash', color='red'), name="IV Upper")
    fig.add_hline(y=iv_lower, line=dict(dash='dash', color='green'), name="IV Lower")
    fig.update_layout(
        title=f"{ticker} Forecast (Next {len(arima_f)} Days)",
        xaxis_title="Date",
        yaxis_title="Price",
        legend_title="Models"
    )

    st.plotly_chart(fig, use_container_width=True)