from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.holtwinters import ExponentialSmoothing
from keras.models import Sequential
from keras.layers import LSTM, Dense
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
import plotly.graph_objects as go


def forecast_arima(prices, periods=20):
    model = ARIMA(prices[:-periods], order=(5, 1, 0))
    fit = model.fit()
    forecast = fit.forecast(steps=periods)
    idx = pd.bdate_range(prices.index[-1] + pd.Timedelta(days=1), periods=periods)
    return pd.Series(forecast.values, index=idx)


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
        next_input = pred.reshape(1, 1, 1)
        input_seq = np.concatenate((input_seq[:, 1:, :], next_input), axis=1)

    forecast = scaler.inverse_transform(np.array(forecasts).reshape(-1, 1)).flatten()
    idx = pd.bdate_range(prices.index[-1] + pd.Timedelta(days=1), periods=periods)
    return pd.Series(forecast, index=idx)


def plot_forecasts(ticker, prices, st):
    forecast_arima_vals = forecast_arima(prices)
    forecast_lstm_vals = forecast_lstm(prices)

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices.index, y=prices, mode='lines', name='Actual Price'))
    fig.add_trace(go.Scatter(x=forecast_arima_vals.index, y=forecast_arima_vals, mode='lines', name='ARIMA Forecast'))
    fig.add_trace(go.Scatter(x=forecast_lstm_vals.index, y=forecast_lstm_vals, mode='lines', name='LSTM Forecast'))

    fig.update_layout(title=f"{ticker} Forecasts", xaxis_title="Date", yaxis_title="Price")
    st.plotly_chart(fig)