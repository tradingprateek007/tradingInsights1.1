import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from scipy.stats import zscore
import plotly.graph_objects as go

from core import past_forecast, sentiment_news
from core import future_forecast
from core.momentum import momentum_burst_tab
from core.sentiment_news import render_sentiment_tab


# def fetch_history(ticker, period="5y"):
#     return yf.Ticker(ticker).history(period=period)["Close"]


def compute_zscore(series, window=20):
    return zscore(series[-window:])[-1]


def fetch_option_data(ticker):
    stock = yf.Ticker(ticker)
    expirations = stock.options
    if not expirations:
        raise ValueError("No options data available.")
    options = stock.option_chain(expirations[0])
    calls = options.calls.copy()
    puts = options.puts.copy()
    calls["type"] = "call"
    puts["type"] = "put"
    all_options = pd.concat([calls, puts])
    all_options["expirationDate"] = expirations[0]
    return all_options


def calculate_indicators(options_data, current_price):
    results = {}

    options_data = options_data.copy()
    options_data = options_data[options_data["impliedVolatility"].notna()]
    options_data = options_data[
        (options_data["strike"] >= current_price * 0.9)
        & (options_data["strike"] <= current_price * 1.1)
    ]

    if options_data.empty:
        return {
            "avg_iv": np.nan,
            "put_call_ratio": np.nan,
            "put_call_oi_ratio": np.nan,
            "vol_skew": np.nan,
        }

    results["avg_iv"] = options_data["impliedVolatility"].mean()

    # Volume-based Put/Call Ratio
    put_vol = options_data[options_data["type"] == "put"]["volume"].sum()
    call_vol = options_data[options_data["type"] == "call"]["volume"].sum()
    results["put_call_ratio"] = put_vol / call_vol if call_vol > 0 else np.nan

    # Open Interest-based Put/Call Ratio
    put_oi = options_data[options_data["type"] == "put"]["openInterest"].sum()
    call_oi = options_data[options_data["type"] == "call"]["openInterest"].sum()
    results["put_call_oi_ratio"] = put_oi / call_oi if call_oi > 0 else np.nan

    # Volatility Skew
    otm_calls = options_data[
        (options_data["type"] == "call") & (options_data["strike"] > current_price)
    ]
    otm_puts = options_data[
        (options_data["type"] == "put") & (options_data["strike"] < current_price)
    ]
    if otm_calls.empty or otm_puts.empty:
        results["vol_skew"] = np.nan
    else:
        call_iv = otm_calls["impliedVolatility"].dropna().mean()
        put_iv = otm_puts["impliedVolatility"].dropna().mean()
        results["vol_skew"] = put_iv - call_iv if pd.notna(put_iv) and pd.notna(call_iv) else np.nan

    return results


def generate_signal_with_indicators(indicators):
    signals = []

    # IV Level
    if indicators['avg_iv'] < 0.3:
        signals.append("Buy (IV is low)")
    else:
        signals.append("Sell (IV is high)")

    # Volume Sentiment
    if indicators['put_call_ratio'] > 1:
        signals.append("Contrarian Buy (High put/call)")
    else:
        signals.append("Bullish Flow (Call volume dominates)")

    # Open Interest Sentiment
    if indicators.get('oi_put_call_ratio', 0) > 1:
        signals.append("Bearish Positioning (More puts open)")
    else:
        signals.append("Bullish Positioning (More calls open)")

    # Volatility Skew (Handle more realistically)
    skew = indicators.get('vol_skew', 0)
    if abs(skew) < 0.08:
        signals.append("Neutral Skew (Typical hedging behavior)")
    elif skew > 0.08:
        signals.append("Bearish Skew (Put IV much higher)")
    else:
        signals.append("Bullish Skew (Call IV much higher)")

    return signals

def explain_signal(indicators):
    iv = indicators.get("avg_iv")
    pcr = indicators.get("put_call_ratio")
    oi_ratio = indicators.get("put_call_oi_ratio")
    skew = indicators.get("vol_skew")
    explanation = []

    if pd.isna(iv):
        explanation.append("IV unavailable.")
    elif iv > 0.4:
        explanation.append(f"IV is high ({iv:.2f}) → Options are expensive.")
    elif iv < 0.2:
        explanation.append(f"IV is low ({iv:.2f}) → Options are cheap, consider buying.")
    else:
        explanation.append(f"IV is moderate ({iv:.2f}) → Balanced pricing.")

    if pd.isna(pcr):
        explanation.append("Put/Call volume ratio unavailable.")
    elif pcr > 1:
        explanation.append(f"Today's Put/Call volume ratio is {pcr:.2f} → Bearish sentiment intraday.")
    else:
        explanation.append(f"Today's Put/Call volume ratio is {pcr:.2f} → More call activity.")

    if pd.isna(oi_ratio):
        explanation.append("Put/Call open interest ratio unavailable.")
    elif oi_ratio > 1:
        explanation.append(f"Open Interest ratio is {oi_ratio:.2f} → Market positioning is bearish.")
    else:
        explanation.append(f"Open Interest ratio is {oi_ratio:.2f} → Market is positioned bullishly.")

    if pd.isna(skew):
        explanation.append("Skew unavailable.")
    elif skew > 0:
        explanation.append(f"Skew = {skew:.3f} → Put IV higher → Bearish bias.")
    else:
        explanation.append(f"Skew = {skew:.3f} → Call IV higher → Bullish bias.")

    return explanation


def generate_signal(ticker):
    st.subheader(f"📊 Option Signal Generator for {ticker}")
    try:
        options_data = fetch_option_data(ticker)
        current_price = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
        indicators = calculate_indicators(options_data, current_price)

        st.subheader("🔎 Option Sentiment")
        for signal in generate_signal_with_indicators(indicators):
            st.markdown(f"- **{signal}**")

        st.subheader("💡 Explanation")
        for line in explain_signal(indicators):
            st.markdown(f"- {line}")

        st.subheader("🧪 Raw Options Data")
        st.dataframe(options_data[[
            'contractSymbol', 'type', 'strike', 'lastPrice',
            'impliedVolatility', 'volume', 'openInterest'
        ]])

    except Exception as e:
        st.error(f"Error generating signal: {e}")


def main():
    st.set_page_config(layout="wide")
    st.title("🧠 Options Strategy Dashboard")

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Live Signals",
        "Backtest Strategy",
        "Signal Generation",
        "Momentum Bursts",
        "Market Sentiment"
    ])
    with tab1:
        ticker = st.text_input("Enter Stock Ticker", "AAPL").strip().upper()
        if st.button("Generate Option Signal"):
            generate_signal(ticker)

    with tab2:
        ticker2 = st.text_input("Ticker for Past Forecast", "AAPL")
        if st.button("Run Past Forecast"):
            past_forecast.past_forecast(ticker2.strip().upper())

    with tab3:
        ticker3 = st.text_input("Ticker for Future Forecast", "AAPL")
        if st.button("Run Future Forecast"):
            future_forecast.future_forecast(ticker3.strip().upper())

    with tab4:  # Momentum
        momentum_burst_tab()

    with tab5:
        render_sentiment_tab()
        # st.header("Live Market News & Sentiment")
        # ticker = st.text_input("Enter stock ticker for sentiment analysis", "AAPL").strip().upper()
        #
        # if st.button("Fetch Sentiment"):
        #     try:
        #         st.info(f"Fetching news for {ticker}...")
        #         df = sentiment_news.fetch_finviz_news(ticker)
        #         df = sentiment_news.analyze_sentiment(df)
        #         st.dataframe(df[["datetime", "headline", "sentiment"]])
        #     except Exception as e:
        #         st.error(f"Failed to fetch sentiment: {e}")

if __name__ == "__main__":
    main()