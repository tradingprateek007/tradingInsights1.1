import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from prophet import Prophet
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from keras.models import Sequential
from keras.layers import LSTM, Dense
import plotly.graph_objects as go
import ta

def fetch_history(ticker, period="1y"):
    df = yf.Ticker(ticker).history(period=period)
    return df

def add_features(df):
    df = df.copy()
    df["rsi"] = ta.momentum.RSIIndicator(df["Close"]).rsi()
    macd = ta.trend.MACD(df["Close"])
    df["macd"] = macd.macd()
    df["volume"] = df["Volume"]
    df.dropna(inplace=True)
    return df

def forecast_arima(prices, periods):
    model = ARIMA(prices, order=(5,1,0))
    fit = model.fit()
    forecast = fit.forecast(steps=periods)
    forecast_index = pd.bdate_range(prices.index[-1] + pd.Timedelta(days=1), periods=periods)
    return pd.Series(forecast.values, index=forecast_index)

def forecast_prophet(prices, periods):
    # Prophet does not support timezone-aware datetimes
    ds = prices.index.tz_localize(None)  # remove tz info
    df = pd.DataFrame({"ds": ds, "y": prices.values})
    m = Prophet()
    m.fit(df)
    future = m.make_future_dataframe(periods=periods)
    forecast = m.predict(future)
    pred = forecast.tail(periods).set_index("ds")["yhat"]
    return pred

def forecast_lstm(df, periods, look_back=10):
    features = ["Close", "rsi", "macd", "volume"]
    scaler = MinMaxScaler()
    scaled = scaler.fit_transform(df[features].values)

    if scaled.shape[0] < look_back + periods:
        raise ValueError(f"Not enough data: have {scaled.shape[0]} rows, need at least {look_back+periods}")

    X, y = [], []
    for i in range(len(scaled) - look_back - periods):
        X.append(scaled[i:i+look_back])
        y.append(scaled[i+look_back][0])  # Close price

    X, y = np.array(X), np.array(y)

    model = Sequential()
    model.add(LSTM(50, activation='relu', input_shape=(look_back, X.shape[2])))
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mse')
    model.fit(X, y, epochs=20, verbose=0)

    last_sequence = scaled[-look_back:]
    input_seq = last_sequence.reshape(1, look_back, scaled.shape[1])

    preds = []
    for _ in range(periods):
        pred = model.predict(input_seq, verbose=0)
        next_step = np.zeros((1, 1, scaled.shape[1]))
        next_step[0, 0, 0] = pred  # predicted Close
        input_seq = np.append(input_seq[:,1:,:], next_step, axis=1)
        preds.append(pred[0][0])

    forecast = scaler.inverse_transform(
        np.hstack([np.array(preds).reshape(-1,1), np.zeros((periods,scaled.shape[1]-1))])
    )[:,0]

    forecast_index = pd.bdate_range(df.index[-1] + pd.Timedelta(days=1), periods=periods)
    return pd.Series(forecast, index=forecast_index)

def compute_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mape = mean_absolute_percentage_error(actual, pred) * 100
    return rmse, mape

def future_forecast(ticker):
    st.title("📈 Future Forecast (Enhanced)")

    forecast_days = st.slider("Forecast horizon (days)", min_value=1, max_value=30, value=7)

    df_raw = fetch_history(ticker, period="1y")
    if df_raw.empty or len(df_raw) < 60:
        st.warning("Not enough data to forecast.")
        return

    df = add_features(df_raw)

    prices = df["Close"]
    actual = prices[-forecast_days:]

    arima_f = forecast_arima(prices, forecast_days)
    prophet_f = forecast_prophet(prices, forecast_days)
    lstm_f = forecast_lstm(df, forecast_days)

    # Metrics
    actual_vs = prices[-forecast_days:]
    arima_rmse, arima_mape = compute_metrics(actual_vs, arima_f.head(len(actual_vs)))
    lstm_rmse, lstm_mape = compute_metrics(actual_vs, lstm_f.head(len(actual_vs)))
    prophet_rmse, prophet_mape = compute_metrics(actual_vs, prophet_f.head(len(actual_vs)))

    st.subheader("Forecast Metrics")
    metrics = pd.DataFrame({
        "Model": ["ARIMA", "LSTM", "Prophet"],
        "RMSE": [arima_rmse, lstm_rmse, prophet_rmse],
        "MAPE (%)": [arima_mape, lstm_mape, prophet_mape]
    })
    st.dataframe(metrics)

    st.subheader("Forecast Visualization")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=prices.index, y=prices.values, name="Historical", line=dict(color="blue")))
    fig.add_trace(go.Scatter(x=arima_f.index, y=arima_f.values, name="ARIMA Forecast", line=dict(dash="dot", color="orange")))
    fig.add_trace(go.Scatter(x=lstm_f.index, y=lstm_f.values, name="LSTM Forecast", line=dict(dash="dot", color="green")))
    fig.add_trace(go.Scatter(x=prophet_f.index, y=prophet_f.values, name="Prophet Forecast", line=dict(dash="dot", color="red")))
    fig.update_layout(
        title=f"{ticker} Forecast ({forecast_days} days ahead)",
        xaxis_title="Date",
        yaxis_title="Price"
    )

    st.plotly_chart(fig, use_container_width=True)