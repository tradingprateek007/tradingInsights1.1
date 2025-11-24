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
from core.gpt_strategy import render_gpt_strategy_tab
from core.gpt_strategy_backtest import render_gpt_strategy_backtest_tab
from core.insider_sentiment import render_insider_sentiment_tab
from core.momentum import momentum_burst_tab
from core.revised_stratergy import render_revised_strategy_signal_tab
from core.sentiment_news import render_sentiment_tab
import streamlit as st
import plotly.express as px
import yfinance as yf

from core.trading_tab import render_trading_tab
from core.backtest_revised_strategy import render_revised_strategy_backtest_tab  # 👈 NEW


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
        results["vol_skew"] = (
            put_iv - call_iv if pd.notna(put_iv) and pd.notna(call_iv) else np.nan
        )

    return results


def generate_signal_with_indicators(indicators):
    signals = []

    # IV Level
    if indicators["avg_iv"] < 0.3:
        signals.append("Buy (IV is low)")
    else:
        signals.append("Sell (IV is high)")

    # Volume Sentiment
    if indicators["put_call_ratio"] > 1:
        signals.append("Contrarian Buy (High put/call)")
    else:
        signals.append("Bullish Flow (Call volume dominates)")

    # Open Interest Sentiment
    if indicators.get("oi_put_call_ratio", 0) > 1:
        signals.append("Bearish Positioning (More puts open)")
    else:
        signals.append("Bullish Positioning (More calls open)")

    # Volatility Skew (Handle more realistically)
    skew = indicators.get("vol_skew", 0)
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
        options_data = fetch_option_data(ticker)
        current_price = yf.Ticker(ticker).history(period="1d")["Close"].iloc[-1]
        indicators = calculate_indicators(options_data, current_price)

        st.subheader("🔎 Option Sentiment")
        signals = generate_signal_with_indicators(indicators)
        for signal in signals:
            conf = 75 if indicators["avg_iv"] > 0.5 else 60
            st.markdown(f"- **{signal}** (confidence: {conf}%)")

        st.subheader("📌 What to Do?")

        actionable = []
        iv = indicators["avg_iv"]
        put_call = indicators["put_call_ratio"]
        oi_ratio = indicators["put_call_oi_ratio"]
        skew = indicators["vol_skew"]

        if iv < 0.2:
            actionable.append(
                "✅ Implied Volatility is low — Consider **Buying Call or Put**."
            )
        elif iv > 0.5:
            actionable.append(
                "⚠️ Implied Volatility is high — Consider **Selling Premium** or **Closing existing option positions**."
            )

        if put_call > 1.2:
            actionable.append(
                "📉 High Put Volume — Bearish flow. Consider **Buying Puts** or **Closing Call positions**."
            )
        elif put_call < 0.8:
            actionable.append(
                "📈 High Call Volume — Bullish flow. Consider **Buying Calls** or **Closing Put positions**."
            )

        if oi_ratio > 1.2:
            actionable.append(
                "💼 More Puts open — Bearish Positioning. Consider defensive trades."
            )
        elif oi_ratio < 0.8:
            actionable.append(
                "💼 More Calls open — Bullish Positioning. Momentum may follow through."
            )

        if skew > 0.1:
            actionable.append(
                "⚠️ Bearish skew detected — Put IV is much higher than Call IV. Sentiment **already priced in**."
            )
        elif skew < -0.1:
            actionable.append(
                "📈 Bullish skew detected — Call IV dominates. **Potential bullish breakout expected**."
            )
        else:
            actionable.append("🟰 Skew is neutral — No strong divergence.")

        for a in actionable:
            st.markdown(f"- {a}")

        st.subheader("💡 Explanation")
        lines = explain_signal(indicators)
        for line in lines:
            st.markdown(f"- {line}")

        st.subheader("📈 Implied Volatility Skew")
        display_skew_curve(options_data)

        st.subheader("🧪 Raw Options Data")
        display_raw_options(options_data)

    except Exception as e:
        st.error(f"Error generating signal: {e}")


def display_signals(indicators):
    st.subheader("🔎 Option Sentiment")
    signals = generate_signal_with_indicators(indicators)
    for signal in signals:
        conf = 75 if indicators["avg_iv"] > 0.5 else 60
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
        x="strike",
        y="impliedVolatility",
        color="type",
        title="Implied Volatility by Strike (Skew)",
    )
    st.plotly_chart(fig, use_container_width=True)


def display_raw_options(options_data):
    st.subheader("🧪 Raw Options Data")
    st.dataframe(
        options_data[
            [
                "contractSymbol",
                "type",
                "strike",
                "lastPrice",
                "impliedVolatility",
                "volume",
                "openInterest",
            ]
        ]
    )


def explain_signal(indicators):
    lines = []
    iv = indicators.get("avg_iv", np.nan)
    skew = indicators.get("vol_skew", np.nan)
    put_call = indicators.get("put_call_ratio", np.nan)

    # Implied Volatility
    if pd.notna(iv):
        lines.append(f"**Implied Volatility (IV)**: The average IV is {iv:.2f}.")
        if iv > 0.5:
            lines.append(
                "- _Formal_: Elevated IV suggests that options are currently priced at a premium due to heightened market uncertainty or expected volatility."
            )
            lines.append(
                "- _Simple_: Options are expensive right now — traders expect big price moves."
            )
            lines.append(
                "👉 **Strategy Tip**: Consider selling options (e.g., credit spreads or covered calls) to benefit from high premiums."
            )
        elif iv < 0.2:
            lines.append(
                "- _Formal_: Low IV implies options are relatively cheap, reflecting stable market expectations."
            )
            lines.append(
                "- _Simple_: Options are cheap — market isn’t expecting major moves."
            )
            lines.append(
                "👉 **Strategy Tip**: Consider buying calls or puts if you anticipate upcoming movement."
            )

    # Skew
    if pd.notna(skew):
        lines.append(f"**Volatility Skew**: The skew is {skew:.3f}.")
        if skew > 0:
            lines.append(
                "- _Formal_: Put IV exceeds Call IV, indicating demand for downside protection — a bearish indicator."
            )
            lines.append(
                "- _Simple_: Traders are buying puts — they might be worried the stock will drop."
            )
            lines.append(
                "👉 **Strategy Tip**: Market sentiment is cautious. Look at protective puts or bear spreads."
            )
        elif skew < 0:
            lines.append(
                "- _Formal_: Call IV exceeds Put IV, suggesting optimistic sentiment and demand for upside exposure."
            )
            lines.append(
                "- _Simple_: Traders are buying calls — they might expect the stock to rise."
            )
            lines.append(
                "👉 **Strategy Tip**: Bullish momentum. Explore long calls or bull call spreads."
            )
        else:
            lines.append(
                "- _Formal_: Balanced IV skew suggests neutrality or equal hedging demand."
            )
            lines.append("- _Simple_: No strong lean — traders are hedging both directions.")
            lines.append(
                "👉 **Strategy Tip**: Consider neutral strategies like iron condors."
            )

    # Put/Call Volume Ratio
    if pd.notna(put_call):
        lines.append(f"**Put/Call Volume Ratio**: The ratio is {put_call:.2f}.")
        if put_call > 1:
            lines.append(
                "- _Formal_: A high ratio indicates more puts are being traded than calls, suggesting bearish sentiment."
            )
            lines.append(
                "- _Simple_: More puts than calls — traders may be betting against the stock."
            )
            lines.append(
                "👉 **Strategy Tip**: Caution warranted. Evaluate closing bullish positions or deploying bearish ones."
            )
        elif put_call < 0.8:
            lines.append(
                "- _Formal_: A low ratio reflects heavier call trading, indicative of bullish sentiment."
            )
            lines.append(
                "- _Simple_: More calls than puts — traders are betting the stock goes up."
            )
            lines.append(
                "👉 **Strategy Tip**: Reinforce bullish strategies or initiate new long call positions."
            )
        else:
            lines.append(
                "- _Formal_: Balanced put/call activity implies neutral sentiment."
            )
            lines.append("- _Simple_: Market looks indecisive.")
            lines.append(
                "👉 **Strategy Tip**: Use range-bound strategies or wait for clearer signals."
            )

    # Summary Suggestion
    lines.append(
        "🧠 **Interpretation Summary**: Use the above indicators holistically. For instance, a low IV + bullish skew + low put/call ratio strengthens a call-buying thesis."
    )
    lines.append(
        "🧭 **Action Decision**: Match strategy with conditions — high IV + bearish skew = sell calls or buy puts; low IV + bullish skew = buy calls."
    )

    return lines


def main():
    st.set_page_config(layout="wide")
    st.title("🧠 Options Strategy Dashboard")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs(
        [
            "Live Signals",
            "Backtest Strategy",
            "Signal Generation",
            "Momentum Bursts",
            "Market Sentiment",
            "Insider Sentiment",
            "Revised Option Strategy!",
            "Revised Strategy Backtest",
            "GPT Strategy",  # 👈 NEW
            "GPT Strategy Backtest",  # 👈 NEW
            "Trading Tab",
        ]
    )

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
        render_insider_sentiment_tab()

    with tab7:
        render_revised_strategy_signal_tab()

    with tab8:
        render_revised_strategy_backtest_tab()  # 👈 backtest UI

    with tab9:
        render_gpt_strategy_tab()

    with tab10:
        render_gpt_strategy_backtest_tab()

    with tab11:
        render_trading_tab()


if __name__ == "__main__":
    main()