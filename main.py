import yfinance as yf
import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
from scipy.stats import zscore
import plotly.graph_objects as go
import streamlit as st
import plotly.express as px
from core import past_forecast, sentiment_news

from core import past_forecast, sentiment_news, alpaca_trading
from core import future_forecast
from core.momentum import momentum_burst_tab
from core.sentiment_news import render_sentiment_tab


# def fetch_history(ticker, period="5y"):
#     return yf.Ticker(ticker).history(period=period)["Close"]

def render_trading_tab():
    st.title("📈 Make Trades with Alpaca (Paper)")

    acc = alpaca_trading.get_account()
    st.metric("Account Equity", f"${acc.equity}")
    st.metric("Buying Power", f"${acc.buying_power}")

    st.subheader("Your Current Positions")
    positions = alpaca_trading.get_positions()
    if positions:
        pos_data = []
        for p in positions:
            pos_data.append({
                "Symbol": p.symbol,
                "Qty": p.qty,
                "Side": p.side,
                "Market Value": p.market_value
            })
        st.table(pos_data)
    else:
        st.info("No current positions.")

    st.subheader("Place a New Trade")
    symbol = st.text_input("Ticker Symbol", value="AAPL").upper()
    qty = st.number_input("Quantity", min_value=1, value=1)
    side = st.selectbox("Side", ["buy", "sell"])
    if st.button("Submit Order"):
        try:
            order = alpaca_trading.place_order(symbol, qty, side)
            st.success(f"Order submitted: {order.id}")
        except Exception as e:
            st.error(f"Order failed: {e}")

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





def generate_signal(ticker):
    st.subheader(f"📊 Option Signal Generator for {ticker}")

    try:
        # 📝 Fetch option chain
        options_data = fetch_option_data(ticker)
        current_price = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
        indicators = calculate_indicators(options_data, current_price)

        # ✅ Display Sentiment Signals
        display_signals(indicators)

        # 🧠 Display Detailed Explanations
        display_explanations(indicators)

        # 📈 Show Skew Curve
        display_skew_curve(options_data)

        # 🧪 Show Raw Options Data
        display_raw_options(options_data)

    except Exception as e:
        st.error(f"Error generating signal: {e}")


def display_signals(indicators):
    st.subheader("🔎 Option Sentiment")
    signals = generate_signal_with_indicators(indicators)
    for signal in signals:
        # Example confidence based on IV (dummy logic here)
        conf = 75 if indicators['avg_iv'] > 0.5 else 60
        st.markdown(f"- **{signal}** (confidence: {conf}%)")


def display_explanations(indicators):
    st.subheader("💡 Explanation")
    lines = explain_signal(indicators)
    for line in lines:
        st.markdown(f"- {line}")


def display_skew_curve(options_data):
    st.subheader("📈 Implied Volatility Skew")
    fig = px.line(
        options_data,
        x='strike',
        y='impliedVolatility',
        color='type',
        title='Implied Volatility by Strike (Skew)'
    )
    st.plotly_chart(fig, use_container_width=True)


def display_raw_options(options_data):
    st.subheader("🧪 Raw Options Data")
    st.dataframe(options_data[[
        'contractSymbol', 'type', 'strike', 'lastPrice',
        'impliedVolatility', 'volume', 'openInterest'
    ]])


def explain_signal(indicators):
    lines = []
    iv = indicators['avg_iv']
    skew = indicators['vol_skew']
    put_call = indicators['put_call_ratio']

    if iv > 0.5:
        lines.append(f"Implied volatility is elevated (IV={iv:.2f}) — options are expensive, consider selling premium.")
    elif iv < 0.2:
        lines.append(f"Implied volatility is low (IV={iv:.2f}) — options are cheap, consider buying premium.")

    if skew > 0:
        lines.append(f"Put IV > Call IV by {skew:.3f} → bearish skew → bearish sentiment priced in.")
    elif skew < 0:
        lines.append(f"Call IV > Put IV by {abs(skew):.3f} → bullish skew → bullish sentiment priced in.")

    if put_call > 1:
        lines.append(f"Put/Call ratio is {put_call:.2f} → more puts traded → bearish bias.")
    elif put_call < 0.8:
        lines.append(f"Put/Call ratio is {put_call:.2f} → more calls traded → bullish bias.")
    else:
        lines.append(f"Put/Call ratio is {put_call:.2f} → neutral sentiment.")

    return lines

def main():
    st.set_page_config(layout="wide")
    st.title("🧠 Options Strategy Dashboard")

    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "Live Signals",
        "Backtest Strategy",
        "Signal Generation",
        "Momentum Bursts",
        "Market Sentiment",
        "Trading Tab"
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
    with tab6:
        render_trading_tab()
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